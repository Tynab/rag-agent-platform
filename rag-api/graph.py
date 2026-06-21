"""
graph.py — Lớp tích hợp Neo4j GraphRAG cho rag-agent-platform
==========================================================

Mô tả
-----
Module quản lý toàn bộ tương tác với Neo4j: tạo/cập nhật cấu trúc đồ thị
khi ingest tài liệu, trích xuất thực thể bằng LLM, xây dựng quan hệ
co-occurrence, và traversal đồ thị để làm giàu kết quả tìm kiếm vector.

Kiến trúc đồ thị
----------------
Cấu trúc node và relationship trong Neo4j:

    (:Project {name})
        └─[:HAS_DOCUMENT]─►
    (:Document {id, source_file, project, module})
        └─[:HAS_CHUNK]─►
    (:Chunk {id, text, chunk_index, source_file, module, project})
        ├─[:NEXT_CHUNK]─► (:Chunk)          — chuỗi chunk liên tiếp trong file
        ├─[:MENTIONS]─► (:Entity)           — thực thể được đề cập (khi extraction bật)
        └─[:CO_OCCURS_WITH]─► (:Entity)     — thực thể xuất hiện cùng nhau

Chú ý quan trọng về Chunk.id:
    Chunk.id = Qdrant point ID (UUID5 tất định từ path + content hash).
    Điều này cho phép tham chiếu chéo chính xác giữa Qdrant và Neo4j
    mà không cần join table hay mapping riêng.

Tích hợp Hybrid RAG
--------------------
Qdrant đảm nhận tìm kiếm vector (cosine similarity) để lấy top-k chunk
liên quan nhất về mặt ngữ nghĩa. Neo4j làm giàu kết quả bằng cách:
    1. Với mỗi chunk tìm được từ Qdrant, tìm thêm các chunk khác đề cập
       cùng thực thể (qua quan hệ MENTIONS + CO_OCCURS_WITH).
    2. Trả về danh sách chunk mở rộng, đã loại trùng, sắp xếp theo score.
Hai nguồn kết quả được gộp lại trước khi đưa vào LLM.

Trích xuất thực thể (tùy chọn)
--------------------------------
Khi GRAPH_ENTITY_EXTRACTION=true, sau khi lưu chunk vào Neo4j, module gọi
LLM chat model để trích xuất danh sách thực thể từ nội dung chunk:
    - Tên thực thể (entity name)
    - Loại thực thể (PERSON, ORG, PRODUCT, CONCEPT, LOCATION, OTHER)
Thực thể được lưu thành node (:Entity) với quan hệ MENTIONS từ Chunk.
Các thực thể xuất hiện trong cùng chunk sẽ có quan hệ CO_OCCURS_WITH.

Chú ý hiệu suất
---------------
- GRAPH_ENTITY_EXTRACTION=true làm chậm quá trình ingest đáng kể vì gọi LLM
  cho mỗi chunk. Nên tắt khi ingest lần đầu số lượng lớn tài liệu.
- GRAPH_ENABLED=false sẽ bỏ qua hoàn toàn Neo4j — RAG chỉ dùng Qdrant.
- Kết nối Neo4j được tái sử dụng qua module-level driver singleton.
"""

import logging
import os

from neo4j import Driver, GraphDatabase

logger = logging.getLogger("rag-graph")


def _require_env(name: str) -> str:
    """Trả về giá trị biến môi trường *name*.

    Raise RuntimeError nếu biến không tồn tại — dùng cho các biến cấu hình bắt buộc.
    """
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set")
    return value


# ── Cấu hình — đọc từ biến môi trường khi import ──────────────────────────────────────
GRAPH_ENABLED: bool = os.environ.get("GRAPH_ENABLED", "true").lower() == "true"
GRAPH_ENTITY_EXTRACTION: bool = os.environ.get(
    "GRAPH_ENTITY_EXTRACTION", "false").lower() == "true"
NEO4J_DATABASE: str = os.environ.get("NEO4J_DATABASE", "neo4j")

if GRAPH_ENABLED:
    NEO4J_URI: str = _require_env("NEO4J_URI")
    NEO4J_USERNAME: str = _require_env("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = _require_env("NEO4J_PASSWORD")
else:
    NEO4J_URI = NEO4J_USERNAME = NEO4J_PASSWORD = ""

# ── Driver singleton — một kết nối Neo4j duy nhất cho toàn service ───────────────
_driver: Driver | None = None


def get_driver() -> Driver:
    """Trả về singleton Neo4j Driver. Khởi tạo lần đầu, tái sử dụng cho mọi request sau.

    Driver quản lý connection pool nội bộ — không cần tạo nhiều driver.
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


def ping() -> bool:
    """Kiểm tra kết nối Neo4j bằng cách gọi verify_connectivity().

    Trả về True nếu kết nối thành công, False nếu có ngoại lệ.
    Không raise exception — được dùng trong /health endpoint.
    """
    try:
        get_driver().verify_connectivity()
        return True
    except Exception as exc:
        logger.warning("Neo4j ping failed: %s", exc)
        return False


# ── Schema ────────────────────────────────────────────────────────────────────
def setup_schema() -> None:
    """Tạo uniqueness constraints (idempotent — an toàn khi gọi mỗi lần ingest)."""
    driver = get_driver()
    constraints = [
        "CREATE CONSTRAINT project_name  IF NOT EXISTS FOR (p:Project)  REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT document_id   IF NOT EXISTS FOR (d:Document) REQUIRE d.id   IS UNIQUE",
        "CREATE CONSTRAINT chunk_id      IF NOT EXISTS FOR (c:Chunk)    REQUIRE c.id   IS UNIQUE",
    ]
    with driver.session(database=NEO4J_DATABASE) as session:
        for stmt in constraints:
            session.run(stmt)
    logger.info("Neo4j schema ready")


# ── Thao tác ghi ─────────────────────────────────────────────────────────────
def save_project_graph(
    project: str,
    chunk_point_pairs: list[tuple],
) -> int:
    """
    Lưu đồ thị Project → Document → Chunk vào Neo4j.

    Args:
        project: tên project (khớp với data/raw/<project>)
        chunk_point_pairs: danh sách (LangChain Document, qdrant_point_id str)

    Returns:
        Số Chunk node đã lưu/cập nhật.
    """
    if not chunk_point_pairs:
        return 0

    # Nhóm chunk theo source_file → Document
    doc_chunks: dict[str, list[tuple]] = {}
    for chunk, point_id in chunk_point_pairs:
        key = chunk.metadata.get("source_file", "unknown")
        doc_chunks.setdefault(key, []).append((chunk, point_id))

    saved = 0
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        # Đảm bảo tồn tại Project node
        session.run("MERGE (:Project {name: $p})", p=project)

        for source_file, pairs in doc_chunks.items():
            first_chunk = pairs[0][0]
            doc_id = f"{project}::{source_file}"

            # Gộp Document + liên kết với Project
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.name      = $name,
                    d.path      = $path,
                    d.project   = $project,
                    d.file_type = $file_type
                WITH d
                MATCH (proj:Project {name: $project})
                MERGE (proj)-[:HAS_DOCUMENT]->(d)
                """,
                id=doc_id,
                name=source_file,
                path=first_chunk.metadata.get("source_path", ""),
                project=project,
                file_type=first_chunk.metadata.get("file_type", ""),
            )

            # Sắp xếp theo chunk_index cho liên kết NEXT_CHUNK
            ordered = sorted(
                pairs, key=lambda x: x[0].metadata.get("chunk_index", 0))
            prev_id: str | None = None

            for chunk, point_id in ordered:
                session.run(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.text        = $text,
                        c.chunk_index = $idx,
                        c.project     = $project,
                        c.source_file = $sf,
                        c.qdrant_id   = $id
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    id=point_id,
                    text=chunk.page_content,
                    idx=chunk.metadata.get("chunk_index", 0),
                    project=project,
                    sf=source_file,
                    doc_id=doc_id,
                )

                if prev_id:
                    session.run(
                        """
                        MATCH (a:Chunk {id: $a}), (b:Chunk {id: $b})
                        MERGE (a)-[:NEXT_CHUNK]->(b)
                        """,
                        a=prev_id,
                        b=point_id,
                    )

                prev_id = point_id
                saved += 1

    logger.info("Saved %d chunks to Neo4j for project '%s'", saved, project)
    return saved


def save_entities(chunk_id: str, entities: list[dict], project: str) -> None:
    """
    Lưu các thực thể đã trích xuất cho một chunk.
    Tạo quan hệ MENTIONS từ Chunk → Entity,
    và CO_OCCURS_WITH giữa các thực thể cùng xuất hiện trong chunk.
    """
    if not entities:
        return

    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        for ent in entities:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            session.run(
                """
                MERGE (e:Entity {name: $name, project: $project})
                SET e.type = $type
                WITH e
                MATCH (c:Chunk {id: $cid})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=name,
                project=project,
                type=ent.get("type", "Other"),
                cid=chunk_id,
            )

        # Co-occurrence giữa tất cả thực thể cùng xuất hiện trong chunk này
        if len(entities) > 1:
            session.run(
                """
                MATCH (c:Chunk {id: $cid})-[:MENTIONS]->(e1:Entity)
                MATCH (c)-[:MENTIONS]->(e2:Entity)
                WHERE e1.name < e2.name
                MERGE (e1)-[:CO_OCCURS_WITH]->(e2)
                """,
                cid=chunk_id,
            )


def delete_project_graph(project: str) -> None:
    """
    Xóa tất cả Neo4j node của một project.
    Được gọi trước reset-ingest để đồng bộ đồ thị với Qdrant.
    """
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(
            "MATCH (d:Document {project: $p})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c",
            p=project,
        )
        session.run(
            "MATCH (d:Document {project: $p}) DETACH DELETE d", p=project)
        # Xóa entity mồ côi (không còn được đề cập bởi bất kỳ chunk nào trong project)
        session.run(
            """
            MATCH (e:Entity {project: $p})
            WHERE NOT ()-[:MENTIONS]->(e)
            DETACH DELETE e
            """,
            p=project,
        )
    logger.info("Deleted Neo4j graph for project '%s'", project)


# ── Thao tác đọc / làm giàu kết quả ─────────────────────────────────────────────
def get_graph_enrichment(qdrant_ids: list[str], limit: int = 5) -> list[dict]:
    """
    Từ các Qdrant point ID đã lấy bởi vector search, tìm thêm các Chunk liên quan
    thông qua co-occurrence thực thể trong đồ thị.

    Trả về danh sách dict: {chunk_id, text, project, source_file, matched_entities}.
    Trả về [] khi có lỗi (không gây dừng /ask).
    """
    if not qdrant_ids:
        return []

    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $ids
                WITH collect(DISTINCT e.name) AS entity_names
                MATCH (c2:Chunk)-[:MENTIONS]->(e2:Entity)
                WHERE e2.name IN entity_names
                  AND NOT c2.id IN $ids
                WITH c2, collect(DISTINCT e2.name) AS matched
                RETURN c2.id          AS chunk_id,
                       c2.text        AS text,
                       c2.project     AS project,
                       c2.source_file AS source_file,
                       matched        AS matched_entities
                ORDER BY size(matched) DESC
                LIMIT $lim
                """,
                ids=qdrant_ids,
                lim=limit,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.warning("Graph enrichment query failed: %s", exc)
        return []


def get_entities_for_project(project: str) -> list[dict]:
    """Trả về tất cả thực thể của một project, sắp xếp theo số lần đề cập (giảm dần)."""
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (c:Chunk {project: $p})-[:MENTIONS]->(e:Entity)
                RETURN e.name AS name, e.type AS type, count(c) AS mentions
                ORDER BY mentions DESC
                """,
                p=project,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.warning("get_entities_for_project failed: %s", exc)
        return []


def get_graph_stats() -> dict:
    """Trả về số lượng node theo từng label cho tất cả project."""
    try:
        driver = get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            counts = {}
            for label in ("Project", "Document", "Chunk", "Entity"):
                res = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                counts[label.lower() + "s"] = res.single()["c"]
            return counts
    except Exception as exc:
        logger.warning("Graph stats failed: %s", exc)
        return {}
