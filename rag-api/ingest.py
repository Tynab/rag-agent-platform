"""
ingest.py — Pipeline nạp tài liệu vào Qdrant và Neo4j cho rag-agent-platform
=========================================================================

Mô tả
-----
Module xử lý toàn bộ quy trình ingest: tải tài liệu từ disk, chia thành
các chunk nhỏ, embed bằng Ollama, upsert vào Qdrant và tùy chọn lưu đồ thị
quan hệ vào Neo4j.

Quy trình xử lý chi tiết
-------------------------
    load_documents()
        Tải tất cả file trong thư mục project song song bằng ThreadPoolExecutor.
        Hỗ trợ các định dạng: .md .txt .pdf .docx .csv .json .jsonl .yaml .yml
        .xml .html .py .js .ts .go .rs .java .cs .sh .sql .log

    split_documents()
        Chia tài liệu thành chunk bằng RecursiveCharacterTextSplitter.
        Kích thước chunk và overlap lấy từ biến môi trường CHUNK_SIZE và
        CHUNK_OVERLAP. Mỗi chunk giữ lại metadata nguồn.

    _run_async_pipeline()
        Pipeline bất đồng bộ sử dụng asyncio + httpx gọi trực tiếp đến
        Ollama endpoint /api/embed (không qua LangChain để tối đa hiệu suất).
        - asyncio.gather(): toàn bộ batch embed được submit đồng thời
        - asyncio.Semaphore(INGEST_EMBED_WORKERS): giới hạn số request embed
          đồng thời, tránh làm quá tải GPU/CPU
        - AsyncQdrantClient: upsert vector không blocking
        Sau mỗi batch embed+upsert, kết quả được ghi vào Qdrant theo batch
        kích thước UPSERT_BATCH_SIZE.

    save_graph() (tùy chọn, khi GRAPH_ENABLED=true)
        Ánh xạ mỗi chunk vào Neo4j theo mô hình:
        (:Project)-[:HAS_DOCUMENT]->(:Document)-[:HAS_CHUNK]->(:Chunk)
        Nếu GRAPH_ENTITY_EXTRACTION=true, LLM trích xuất thực thể từ nội dung
        chunk và lưu quan hệ (:Chunk)-[:MENTIONS]->(:Entity) và
        (:Entity)-[:CO_OCCURS_WITH]->(:Entity).

Idempotency
-----------
make_point_id() sinh UUID5 tất định từ tổ hợp (đường dẫn file + hash nội dung).
Chạy /ingest nhiều lần trên cùng một file sẽ upsert đúng point_id đã tồn tại
— Qdrant tự cập nhật payload, không tạo bản ghi trùng lặp.

Metadata của mỗi chunk trong Qdrant
------------------------------------
Mỗi PointStruct có payload đầy đủ:
    source_file     — tên file gốc
    relative_path   — đường dẫn tương đối trong data/raw/
    file_type       — phần mở rộng (md, pdf, py ...)
    chunk_index     — số thứ tự chunk trong file
    content_hash    — SHA-256 hash nội dung chunk (dùng cho idempotency)
    embedding_model — tên model embedding đã dùng
    module          — tên thư mục con (dùng để filter theo module trong /ask)
    doc_type        — loại tài liệu: prd, schema, api, architecture, spec, other
    chunk_type      — loại nội dung: paragraph, table, code, heading
    language        — ngôn ngữ lập trình nếu là code chunk
    status          — trạng thái: active
    created_at      — timestamp ISO 8601 lúc ingest
    page_content    — nội dung văn bản đầy đủ của chunk
    project         — tên project

Suy luận module từ cấu trúc thư mục
-------------------------------------
    data/raw/{project}/{module}/file.md  →  chunk.module = tên thư mục con
    data/raw/{project}/file.md           →  chunk.module = project
Module được dùng bởi POST /ask với tham số module= để lọc kết quả search.

Các biến môi trường sử dụng
----------------------------
    EMBEDDING_MODEL       — tên model Ollama để embed
    OLLAMA_BASE_URL       — URL Ollama server
    QDRANT_URL            — URL Qdrant server
    COLLECTION_NAME       — tiền tố tên collection
    CHUNK_SIZE            — kích thước chunk (ký tự)
    CHUNK_OVERLAP         — overlap giữa chunk kề nhau (ký tự)
    UPSERT_BATCH_SIZE     — số vector upsert vào Qdrant mỗi batch
    INGEST_EMBED_WORKERS  — số request embed asyncio đồng thời
    GRAPH_ENABLED         — true/false bật Neo4j
    GRAPH_ENTITY_EXTRACTION — true/false bật LLM entity extraction
    NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set")
    return value


LOG_LEVEL = _require_env("LOG_LEVEL")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rag-ingest")

RAW_DATA_DIR = _require_env("RAW_DATA_DIR")
OLLAMA_BASE_URL = _require_env("OLLAMA_BASE_URL")
QDRANT_URL = _require_env("QDRANT_URL")
COLLECTION_NAME = _require_env("COLLECTION_NAME")
EMBEDDING_MODEL = _require_env("EMBEDDING_MODEL")

CHUNK_SIZE = int(_require_env("CHUNK_SIZE"))
CHUNK_OVERLAP = int(_require_env("CHUNK_OVERLAP"))
UPSERT_BATCH_SIZE = int(_require_env("UPSERT_BATCH_SIZE"))
# Số batch embed/upsert chạy đồng thời trong asyncio pipeline.
# Giá trị này cân bằng giữa hiệu suất và ổn định GPU — tăng khi có nhiều
# GPU/CPU sẵn có và OLLAMA_NUM_PARALLEL lớn.
INGEST_EMBED_WORKERS: int = max(1, int(os.environ.get("INGEST_EMBED_WORKERS", "1")))

GRAPH_ENABLED: bool = os.environ.get("GRAPH_ENABLED", "true").lower() == "true"
GRAPH_ENTITY_EXTRACTION: bool = os.environ.get(
    "GRAPH_ENTITY_EXTRACTION", "false").lower() == "true"
CHAT_MODEL: str | None = os.environ.get("CHAT_MODEL")

_graph_module = None
if GRAPH_ENABLED:
    try:
        import graph as _graph_module  # type: ignore[import-not-found]
    except Exception as _graph_import_err:
        logger.warning(
            "Neo4j graph module unavailable, running Qdrant-only: %s", _graph_import_err)
        GRAPH_ENABLED = False


def get_collection_name(project: str) -> str:
    """Tạo tên Qdrant collection theo quy ước {COLLECTION_NAME}__{project}.

    Dùng hai dấu gạch dưới __ làm separator để tránh nhầm lẫn với dấu gạch
    trong tên project (ví dụ: "yan_raw_docs__my-project").
    """
    return f"{COLLECTION_NAME}__{project}"


def get_projects(raw_data_dir: str = RAW_DATA_DIR) -> list[str]:
    """Trả về danh sách tên project đã sắp xếp — là các thư mục con cấp 1 trong raw_data_dir.

    Mỗi thư mục con tương ứng một project độc lập với collection Qdrant riêng.
    Trả về [] nếu raw_data_dir không tồn tại.
    """
    root = Path(raw_data_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".json",
    ".jsonl",
    ".yml",
    ".yaml",
    ".xml",
    ".html",
    ".htm",
    ".sql",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".java",
    ".go",
    ".rs",
    ".sh",
}


def _safe_load_text(path: Path) -> list[Document]:
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return TextLoader(str(path), encoding=encoding).load()
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Cannot load text file {path}: {last_error}")


def _load_single_file(path: Path, root: Path) -> list[Document]:
    """Tải một file và gắn metadata. Trả về [] nếu không hỗ trợ hoặc lỗi."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif suffix == ".docx":
            loaded = Docx2txtLoader(str(path)).load()
        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            loaded = _safe_load_text(path)
        else:
            logger.info("Skipping unsupported file: %s", path)
            return []
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load file: %s", path)
        return []

    relative_path = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    for index, doc in enumerate(loaded):
        doc.metadata.update(
            {
                "source_file": path.name,
                "source_path": str(path),
                "relative_path": relative_path,
                "file_type": suffix.replace(".", ""),
                "loader_index": index,
            }
        )
    return loaded


def load_documents(raw_data_dir: str = RAW_DATA_DIR) -> list[Document]:
    """
    Tải tất cả file được hỗ trợ từ *raw_data_dir* song song (ThreadPoolExecutor).

    Phương pháp tải theo đuôi file:
      .pdf   → PyPDFLoader (mỗi trang = 1 Document)
      .docx  → Docx2txtLoader (toàn bộ file = 1 Document)
      others → TextLoader (thử nhiều encoding: utf-8, utf-8-sig, latin-1)

    Mỗi Document được đính kèm metadata: source_file, source_path,
    relative_path (tương đối với raw_data_dir), file_type, loader_index.
    File không hỗ trợ sẽ bị bỏ qua; lỗi tải sẽ được log và bỏ qua.
    """
    docs: list[Document] = []
    root = Path(raw_data_dir)

    logger.info("Loading raw documents from: %s", root)

    if not root.exists():
        logger.warning("Raw data directory does not exist: %s", root)
        return docs

    file_paths = sorted(p for p in root.rglob("*") if p.is_file())
    if not file_paths:
        logger.info("Loaded document units: 0")
        return docs

    max_workers = min(8, len(file_paths))
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="loader") as pool:
        futures = [pool.submit(_load_single_file, p, root) for p in file_paths]
        for f in futures:
            docs.extend(f.result())

    logger.info("Loaded document units: %s", len(docs))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Cắt nhỏ tài liệu thành chunk dùng RecursiveCharacterTextSplitter.

    Separator ưu tiên heading Markdown („\n# “, „\n## “ …) để giữ
    cấu trúc tài liệu kỹ thuật vốn có heading phân cấp rõ ràng.
    Mỗi chunk được đính thêm: chunk_index, content_hash (SHA-256),
    chunk_size, chunk_overlap, embedding_model để hỗ trợ tra cứu và debug.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(docs)

    # A7: chunk_index đánh số THEO TỪNG FILE (0..n trong mỗi file) thay vì global
    # toàn project. split_documents giữ nguyên thứ tự nên chunk cùng file nằm liền
    # nhau. Lợi ích: thêm/xóa một file không làm dịch chunk_index của file khác →
    # make_point_id của các chunk không đổi vẫn ổn định (idempotency thực sự).
    # LƯU Ý: đổi cách đánh số ⇒ point_id thay đổi ⇒ cần /reset-ingest một lần.
    per_file_index: dict[tuple, int] = {}
    for chunk in chunks:
        key = (
            chunk.metadata.get("relative_path", ""),
            chunk.metadata.get("loader_index", ""),
        )
        idx = per_file_index.get(key, 0)
        per_file_index[key] = idx + 1
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")).hexdigest()
        chunk.metadata.update(
            {
                "chunk_index": idx,
                "content_hash": content_hash,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": EMBEDDING_MODEL,
            }
        )

    logger.info("Created chunks: %s", len(chunks))
    return chunks


def make_point_id(chunk: Document) -> str:
    """
    UUID xác định (deterministic) để /ingest trở thành idempotent với các chunk không thay đổi.
    Chạy lại /ingest trên cùng file sẽ cập nhật cùng point ID thay vì tạo điểm trùng lặp.
    """
    raw = "|".join(
        [
            str(chunk.metadata.get("relative_path", "")),
            str(chunk.metadata.get("loader_index", "")),
            str(chunk.metadata.get("chunk_index", "")),
            str(chunk.metadata.get("content_hash", "")),
            str(chunk.metadata.get("embedding_model", "")),
        ]
    )

    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def batched(items: list[Document], batch_size: int) -> Iterable[list[Document]]:
    """Phân trang danh sách *items* thành các lô kích thước *batch_size*."""
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def _extract_entities_for_document(doc_text: str, llm: ChatOllama) -> list[dict]:
    """
    Gọi LLM để trích xuất thực thể có tên từ văn bản tổng hợp của tài liệu.
    Trả về danh sách dict {name, type}.  Lỗi trả về [] (không gây dừng ingest).
    Mỗi Document gọi một lần (không phải mỗi Chunk) để giảm thời gian ingest.
    """
    prompt = (
        "Extract named entities (features, components, APIs, services, concepts) "
        "from this technical text.\n"
        'Return ONLY a JSON array: [{"name": "...", "type": "Feature|Component|API|Service|Concept|Other"}]\n'
        "Return [] if none found. No explanation — just the JSON array.\n\n"
        f"Text:\n{doc_text[:2000]}"
    )
    try:
        content = llm.invoke(prompt).content.strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if 0 <= start < end:
            return json.loads(content[start:end])
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
    return []


def get_embeddings() -> OllamaEmbeddings:
    """Trả về instance OllamaEmbeddings dùng *EMBEDDING_MODEL* hiện tại."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# A1: timeout cho lần embed đồng bộ (query path). Lấy theo RAG_TIMEOUT nếu có.
_EMBED_HTTP_TIMEOUT: float = float(os.environ.get("RAG_TIMEOUT", "600"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """A1 — đường embed DUY NHẤT, dùng chung cho ingest (đồng bộ) và /ask.

    Gọi thẳng Ollama /api/embed với CÙNG contract như batch ingest
    ({"model": EMBEDDING_MODEL, "input": [...]}) → vector câu hỏi và vector tài liệu
    luôn được sinh theo đúng một cách, tránh lệch recall do trước đây /ask dùng
    LangChain OllamaEmbeddings.embed_query còn ingest dùng raw httpx /api/embed.
    """
    with httpx.Client(timeout=_EMBED_HTTP_TIMEOUT) as client:
        resp = client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# R2: số lần retry khi embed một batch lúc ingest thất bại (OOM/transient HTTP).
_INGEST_EMBED_RETRIES: int = int(os.environ.get("INGEST_EMBED_RETRIES", "3"))


async def _embed_and_upsert_batch_async(
    http_client: httpx.AsyncClient,
    qdrant: AsyncQdrantClient,
    sem: asyncio.Semaphore,
    batch_no: int,
    total_batches: int,
    chunk_batch: list[Document],
    collection_name: str,
    project: str,
) -> list[tuple]:
    """
    Embed một batch qua Ollama /api/embed rồi upsert vào Qdrant.
    Semaphore giới hạn số batch chạy đồng thời = INGEST_EMBED_WORKERS.
    """
    async with sem:
        texts = [c.page_content for c in chunk_batch]
        logger.info("Embed %s/%s: %s chunks...", batch_no, total_batches, len(texts))
        # R2: retry embed lúc ingest (trước đây chỉ query path của /ask mới retry OOM).
        vectors = None
        for _attempt in range(1, _INGEST_EMBED_RETRIES + 1):
            try:
                resp = await http_client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": EMBEDDING_MODEL, "input": texts},
                )
                resp.raise_for_status()
                vectors = resp.json()["embeddings"]
                break
            except Exception as exc:
                if _attempt >= _INGEST_EMBED_RETRIES:
                    raise
                _wait = 2.0 * _attempt
                logger.warning(
                    "Embed batch %s/%s thất bại (lần %s/%s) — retry sau %.1fs: %s",
                    batch_no, total_batches, _attempt, _INGEST_EMBED_RETRIES, _wait, exc,
                )
                await asyncio.sleep(_wait)

        _now = datetime.now(timezone.utc).isoformat()
        points = [
            PointStruct(
                id=make_point_id(chunk),
                vector=vector,
                payload={
                    **chunk.metadata,
                    "page_content": chunk.page_content,
                    "project": project,
                    "module": _infer_module(
                        chunk.metadata.get("relative_path", ""), project
                    ),
                    "doc_type": _infer_doc_type(
                        chunk.metadata.get("source_file", ""),
                        chunk.metadata.get("relative_path", ""),
                    ),
                    "chunk_type": _infer_chunk_type(chunk.page_content),
                    # A6: trước đây hardcode "vi" cho mọi chunk (gây hiểu nhầm và mâu
                    # thuẫn docstring). Dùng file_type thực tế làm nhãn thay thế.
                    "language": chunk.metadata.get("file_type", "text"),
                    "status": "active",
                    "created_at": _now,
                },
            )
            for chunk, vector in zip(chunk_batch, vectors)
        ]

        await qdrant.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info("Upsert done %s/%s | %s chunks", batch_no, total_batches, len(points))
        return [(c, str(p.id)) for c, p in zip(chunk_batch, points)]


async def _run_async_pipeline(
    chunks: list[Document],
    collection_name: str,
    project: str,
    reset: bool,
) -> tuple[list[tuple], int]:
    """
    Async embed + upsert pipeline: INGEST_EMBED_WORKERS batch chạy đồng thời.

    Cách hoạt động:
      • asyncio.gather gửi tất cả batch cùng lúc.
      • Semaphore giới hạn Ollama request đồng thời = INGEST_EMBED_WORKERS.
      • Embed và upsert của mỗi batch chạy nối tiếp trong cùng coroutine,
        nhưng nhiều batch chạy song song → GPU và Qdrant luôn bận.
      • Tăng INGEST_EMBED_WORKERS = OLLAMA_NUM_PARALLEL để khai thác tối đa.
    """
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        # Phát hiện kích thước vector từ model embedding
        sample_resp = await http_client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": ["health check"]},
        )
        sample_resp.raise_for_status()
        sample_vector = sample_resp.json()["embeddings"][0]

        sync_client = QdrantClient(url=QDRANT_URL)
        if reset and sync_client.collection_exists(collection_name):
            logger.warning("Deleting collection before reset: %s", collection_name)
            sync_client.delete_collection(collection_name)
        _ensure_collection(sync_client, collection_name, vector_size=len(sample_vector))

        qdrant = AsyncQdrantClient(url=QDRANT_URL)
        sem = asyncio.Semaphore(INGEST_EMBED_WORKERS)
        chunk_batches = list(batched(chunks, UPSERT_BATCH_SIZE))
        total_batches = len(chunk_batches)

        logger.info(
            "Async pipeline: %s batches, batch_size=%s, concurrency=%s",
            total_batches, UPSERT_BATCH_SIZE, INGEST_EMBED_WORKERS,
        )

        tasks = [
            _embed_and_upsert_batch_async(
                http_client, qdrant, sem,
                i + 1, total_batches, batch,
                collection_name, project,
            )
            for i, batch in enumerate(chunk_batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await qdrant.close()

    # R1: một batch lỗi KHÔNG làm hỏng cả project ingest. Bỏ qua batch lỗi và ghi
    # log số batch thất bại; các batch thành công vẫn được upsert. count phản ánh
    # đúng số point thực tế đã vào Qdrant (partial, thay vì mất sạch như asyncio.gather mặc định).
    failed = [r for r in results if isinstance(r, Exception)]
    if failed:
        logger.error(
            "Ingest project=%s: %d/%d batch thất bại (đã bỏ qua, phần còn lại vẫn upsert). Lỗi đầu tiên: %s",
            project, len(failed), len(tasks), failed[0],
        )
    all_pairs: list[tuple] = [
        pair for r in results if not isinstance(r, Exception) for pair in r
    ]
    count = sync_client.count(collection_name=collection_name, exact=True)
    return all_pairs, count.count


# ── Helpers suy luận metadata chunk ──────────────────────────────────────────

# R5: pattern suy luận doc_type, khớp trên chuỗi đã chuẩn hóa ('_' và '-' → khoảng
# trắng) nên bắt được cả tên file underscore/prose như "08_API_Specification.md",
# "07_Data_Model_ERD.md", "05_RBAC_Permissions_Matrix.md" — trước đây các pattern
# yêu cầu hyphen nên gần như luôn rơi về 'document'. Pattern đặt từ cụ thể → tổng quát.
_DOC_TYPE_REGEXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bprd\b|product requirement"),                        "prd"),
    (re.compile(r"\bbrd\b|business requirement"),                       "brd"),
    (re.compile(r"\bsrs\b|software requirement"),                       "srs"),
    (re.compile(r"\bapi\b|api specification|api spec"),                 "api"),
    (re.compile(r"\berd\b|data model|entity relationship"),            "data-model"),
    (re.compile(r"\brbac\b|permission|role matrix|access control"),     "rbac"),
    (re.compile(r"\bschema\b|\bddl\b"),                                 "schema"),
    (re.compile(r"architecture|\bc4\b|tech stack|technical design"),    "architecture"),
    (re.compile(r"user stor|acceptance criteria|backlog"),             "user-stories"),
    (re.compile(r"\bqa\b|test strateg|test case|test plan|\buat\b"),    "test"),
    (re.compile(r"security|privacy|compliance|hardening|\bgdpr\b|owasp"), "security"),
    (re.compile(r"devops|operations|\bci/cd\b|ci cd|deployment"),       "devops"),
    (re.compile(r"roadmap|release|milestone"),                         "roadmap"),
    (re.compile(r"analytic|reporting|\bkpi\b|dashboard"),               "analytics"),
    (re.compile(r"glossary|terminology"),                              "glossary"),
    (re.compile(r"\bux\b|sitemap|wireframe|user flow|design system"),   "ux"),
    (re.compile(r"product brief|one page|\boverview\b|\bvision\b"),     "brief"),
    (re.compile(r"\bauth\b|authentication|\bjwt\b"),                    "auth"),
    (re.compile(r"billing|payment|invoice|subscription"),              "billing"),
    (re.compile(r"marketplace"),                                       "marketplace"),
    (re.compile(r"partner"),                                           "partner"),
    (re.compile(r"meeting"),                                           "meeting-notes"),
]


def _infer_module(relative_path: str, project: str) -> str:
    """
    Trích xuất tên module từ cấu trúc thư mục tương đối với project dir.
    module/file.md            → trả về module (thư mục cấp một).
    module/subdir/file.md     → trả về module (thư mục cấp một).
    file.md (phẳng)           → trả về project.
    """
    parts = Path(relative_path).parts
    if len(parts) >= 2 and not Path(parts[0]).suffix:
        return parts[0]
    return project


def _infer_doc_type(source_file: str, relative_path: str) -> str:
    """Suy luận loại tài liệu (prd, srs, api, data-model, rbac, security...) từ tên
    file và đường dẫn.

    R5: chuẩn hóa '_' và '-' thành khoảng trắng rồi collapse whitespace trước khi khớp
    word-boundary regex — bắt được tên file underscore/prose mà phiên bản cũ bỏ sót.
    """
    combined = re.sub(r"[_\-]+", " ", (source_file + " " + relative_path).lower())
    combined = re.sub(r"\s+", " ", combined)
    for pat, doc_type in _DOC_TYPE_REGEXES:
        if pat.search(combined):
            return doc_type
    return "document"


def _infer_chunk_type(content: str) -> str:
    """
    Heuristic phân loại chunk theo nội dung:
      table     — nhiều dòng chứa ký tự |
      code      — dòng đầu bắt đầu bằng ký hiệu code
      paragraph — mặc định
    """
    lines = [ln for ln in content.strip().splitlines() if ln.strip()]
    if not lines:
        return "paragraph"
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    if pipe_lines >= 2 and pipe_lines / len(lines) > 0.4:
        return "table"
    code_starters = (
        "```", "    ", "\t",
        "def ", "class ", "function ", "import ", "from ",
        "const ", "var ", "let ", "public ", "private ",
        "SELECT ", "INSERT ", "UPDATE ", "CREATE ",
    )
    if any(lines[0].startswith(s) for s in code_starters):
        return "code"
    indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
    if len(lines) >= 3 and indented / len(lines) > 0.5:
        return "code"
    return "paragraph"


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """
    Tạo Qdrant collection nếu chưa tồn tại.
    Sử dụng COSINE similarity với kích thước vector được phát hiện tự động từ model embedding.
    Idempotent: gọi nhiều lần không gây lỗi.
    """
    if client.collection_exists(collection_name):
        return

    logger.info(
        "Creating Qdrant collection '%s' with vector size %s",
        collection_name,
        vector_size,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def _ingest_project(project: str, reset: bool = False) -> dict:
    collection_name = get_collection_name(project)
    project_dir = str(Path(RAW_DATA_DIR) / project)

    logger.info("Ingesting project '%s' → collection '%s'",
                project, collection_name)
    logger.info("Reset mode: %s | dir: %s", reset, project_dir)

    docs = load_documents(project_dir)

    if not docs:
        return {
            "project": project,
            "collection": collection_name,
            "status": "empty",
            "message": f"No supported files found in {project_dir}",
            "documents": 0,
            "chunks": 0,
        }

    chunks = split_documents(docs)

    if reset and GRAPH_ENABLED and _graph_module is not None:
        try:
            _graph_module.delete_project_graph(project)
        except Exception as exc:
            logger.warning("Graph delete failed (non-fatal): %s", exc)

    # Async pipeline: embed + upsert INGEST_EMBED_WORKERS batch đồng thời
    all_pairs, points_count = asyncio.run(
        _run_async_pipeline(chunks, collection_name, project, reset)
    )
    logger.info("Project '%s' done. Points: %s", project, points_count)

    # ── Tích hợp Neo4j graph ──────────────────────────────────────────────────
    graph_chunks_saved = 0
    if GRAPH_ENABLED and _graph_module is not None:
        try:
            _graph_module.setup_schema()
            graph_chunks_saved = _graph_module.save_project_graph(
                project, all_pairs)

            if GRAPH_ENTITY_EXTRACTION and CHAT_MODEL:
                logger.info(
                    "Entity extraction enabled for project '%s'", project)
                llm = ChatOllama(model=CHAT_MODEL,
                                 base_url=OLLAMA_BASE_URL, temperature=0)
                # Mỗi Document gọi LLM một lần (không phải mỗi Chunk) — nhanh hơn nhiều
                doc_pairs: dict[str, list[tuple]] = {}
                for chunk, pid in all_pairs:
                    sf = chunk.metadata.get("source_file", "unknown")
                    doc_pairs.setdefault(sf, []).append((chunk, pid))
                for sf, pairs in doc_pairs.items():
                    # Lấy tối đa 5 chunk đầu làm đại diện văn bản tài liệu
                    doc_text = " ".join(c.page_content for c, _ in pairs[:5])
                    entities = _extract_entities_for_document(doc_text, llm)
                    for chunk, pid in pairs:
                        chunk_entities = [
                            e for e in entities
                            if (e.get("name") or "").lower() in chunk.page_content.lower()
                        ]
                        if chunk_entities:
                            _graph_module.save_entities(
                                pid, chunk_entities, project)
                logger.info(
                    "Entity extraction complete for project '%s'", project)
        except Exception as exc:
            logger.warning("Graph integration failed (non-fatal): %s", exc)

    return {
        "project": project,
        "collection": collection_name,
        "status": "ok",
        "documents": len(docs),
        "chunks": len(chunks),
        "upserted": len(all_pairs),
        "points_count": points_count,
        "graph_chunks_saved": graph_chunks_saved,
        "idempotent": True,
        "reset": reset,
    }


def ingest(project: str | None = None, reset: bool = False) -> dict:
    """
    Ingest tài liệu vào Qdrant.
    - project=None  → ingest TẤT CẢ thư mục project con trong RAW_DATA_DIR.
    - project="foo" → chỉ ingest data/raw/foo/ vào collection yan_raw_docs__foo.
    """
    if project is not None:
        return _ingest_project(project, reset=reset)

    projects = get_projects()
    if not projects:
        return {
            "status": "empty",
            "message": (
                f"No project subdirectories found in {RAW_DATA_DIR}. "
                "Create subdirectories like data/raw/auth/, data/raw/marketplace/, etc."
            ),
            "projects": {},
        }

    logger.info("Ingesting %s project(s): %s", len(projects), projects)

    results: dict = {}
    for proj in projects:
        results[proj] = _ingest_project(proj, reset=reset)

    return {
        "status": "ok",
        "projects_ingested": list(results.keys()),
        "total_documents": sum(r.get("documents", 0) for r in results.values()),
        "total_chunks": sum(r.get("chunks", 0) for r in results.values()),
        "reset": reset,
        "details": results,
    }
