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
        - Ingest tăng dần (không-reset): chỉ embed + upsert chunk có point_id CHƯA
          có trong collection (chunk mới/đã sửa); chunk không đổi được bỏ qua
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
        Ingest không-reset chỉ ghi (và trích xuất thực thể cho) chunk chưa có node
        trong Neo4j — xem _plan_graph_writes().

Idempotency & ingest tăng dần
-----------------------------
make_point_id() sinh UUID5 tất định từ tổ hợp (relative_path, loader_index,
chunk_index, content_hash, embedding_model). Ingest không-reset tra trước các ID
đã có trong collection (retrieve theo trang, không lấy payload/vector) và chỉ
embed + upsert chunk có ID mới → chạy lại /ingest khi không có gì đổi KHÔNG gọi
Ollama lần nào (`skipped_unchanged` trong response).
F11: ingest không-reset còn xóa các point (và Chunk Neo4j) KHÔNG thuộc lần ingest
hiện tại — chunk của file đã sửa/xóa/đổi tên — để index không giữ nội dung cũ.
Tập giữ lại là MỌI chunk hiện tại (mới + không đổi), không chỉ chunk vừa upsert.
Lưu ý: payload của chunk không đổi KHÔNG được ghi lại (doc_type/module suy luận,
source_path, created_at giữ giá trị lần embed đầu) — sau khi đổi luật suy luận
metadata hoặc RAW_DATA_DIR, chạy /reset-ingest để làm mới.
reset=True embed lại toàn bộ; collection cũ chỉ bị xóa khi batch embed THẬT đầu tiên
thành công → Ollama down / EMBEDDING_MODEL mới chưa pull thì index cũ được giữ nguyên
(response có failed_batches + batch_error) thay vì bị xóa trắng.

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
    INGEST_EMBED_RETRIES  — số lần thử embed một batch lúc ingest (mặc định 3)
    RAG_TIMEOUT           — timeout httpx (giây) mỗi lần gọi Ollama /api/embed, cả
                            batch ingest lẫn embed câu hỏi (mặc định 600)
    OLLAMA_REQUEST_TIMEOUT — timeout httpx (giây) mỗi lần gọi CHAT_MODEL: entity
                            extraction lúc ingest và /ask của app.py (mặc định 600;
                            <= 0 = không giới hạn)
    GRAPH_ENABLED         — true/false bật Neo4j
    GRAPH_ENTITY_EXTRACTION — true/false bật LLM entity extraction
    NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_DATABASE
"""

import asyncio
import hashlib
import importlib
import json
import logging
import os
import re
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, ExtendedPointId, PointIdsList, PointStruct, VectorParams


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
# Timeout (giây) httpx cho mỗi lời gọi CHAT_MODEL (entity extraction lúc ingest, /ask ở
# app.py). Mặc định khớp `${OLLAMA_REQUEST_TIMEOUT:-600}` của rag-api trong
# docker-compose.yml; <= 0 = không giới hạn (cùng quy ước với agent-api/llm.py).
# ChatOllama gọi Ollama ở chế độ stream nên đây là giới hạn thời gian tới token đầu
# tiên (gồm load/swap model) và khoảng lặng giữa hai chunk — không phải tổng thời gian.
OLLAMA_REQUEST_TIMEOUT: float = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT", "600"))


def make_chat_model(model: str, *, temperature: float) -> ChatOllama:
    """Tạo ChatOllama với timeout httpx THẬT cho rag-api (dùng chung ingest + app.py).

    langchain-ollama 0.3.x KHÔNG có field `request_timeout`: kwarg đó bị pydantic bỏ qua
    im lặng → httpx client chạy với timeout=None (vô hạn), một lời gọi treo sẽ chiếm
    slot Ollama duy nhất mãi mãi. Timeout phải đi qua `client_kwargs={"timeout": ...}`
    (chuyển thẳng vào httpx.Client bên trong ollama.Client). Dict `client_kwargs` tạo MỚI
    mỗi lần vì ChatOllama ghi thêm header vào dict này.
    """
    timeout = OLLAMA_REQUEST_TIMEOUT if OLLAMA_REQUEST_TIMEOUT > 0 else None
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        client_kwargs={"timeout": timeout},
    )

_graph_module: ModuleType | None = None
if GRAPH_ENABLED:
    try:
        # import_module (thay vì `import graph as ...`) để biến giữ kiểu ModuleType | None
        # nhất quán cho mypy; hành vi y hệt — lỗi import (thiếu neo4j...) vẫn rơi vào except.
        _graph_module = importlib.import_module("graph")
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


class InvalidProjectError(ValueError):
    """F6: tên project không hợp lệ — API trả HTTP 400 thay vì chạm vào filesystem/Qdrant."""


# F6: tên project đi thẳng vào đường dẫn filesystem (RAW_DATA_DIR/{project}) và tên
# Qdrant collection ({COLLECTION_NAME}__{project}) → chỉ chấp nhận MỘT thành phần tên
# đơn giản: ký tự đầu là chữ/số, sau đó chữ/số/'_'/'.'/'-', tối đa 128 ký tự.
# Chặn "", ".", "..", "/", "~", đường dẫn tuyệt đối, dấu phân cách, khoảng trắng và
# thư mục ẩn (".git"). \w khớp cả chữ Unicode để không làm vỡ project tên tiếng Việt.
_PROJECT_NAME_RE = re.compile(r"[^\W_][\w.-]{0,127}")


def validate_project_name(project: str) -> str:
    """F6 — trả về *project* nếu hợp lệ, ngược lại raise InvalidProjectError."""
    if not isinstance(project, str) or not _PROJECT_NAME_RE.fullmatch(project):
        raise InvalidProjectError(
            f"Tên project không hợp lệ: {project!r}. Chỉ chấp nhận chữ, số, '_', '.', '-' "
            "(ký tự đầu là chữ hoặc số, tối đa 128 ký tự), không chứa dấu phân cách đường dẫn."
        )
    return project


def project_dir_for(project: str, raw_data_dir: str | None = None) -> Path:
    """F6 — kiểm tra *project* rồi trả về thư mục RAW_DATA_DIR/{project}.

    Hai lớp chặn: (1) regex tên đơn thành phần; (2) resolve() (theo cả symlink) và
    bắt buộc kết quả nằm TRONG RAW_DATA_DIR, khác chính RAW_DATA_DIR — chặn symlink
    trỏ ra ngoài thư mục dữ liệu. Raise InvalidProjectError nếu không hợp lệ.
    """
    validate_project_name(project)
    root = Path(raw_data_dir or RAW_DATA_DIR)
    project_dir = root / project
    real_root = root.resolve()
    real_dir = project_dir.resolve()
    if real_dir == real_root or not real_dir.is_relative_to(real_root):
        raise InvalidProjectError(f"Project {project!r} trỏ ra ngoài RAW_DATA_DIR.")
    return project_dir


def get_projects(raw_data_dir: str = RAW_DATA_DIR) -> list[str]:
    """Trả về danh sách tên project đã sắp xếp — là các thư mục con cấp 1 trong raw_data_dir.

    Mỗi thư mục con tương ứng một project độc lập với collection Qdrant riêng.
    Trả về [] nếu raw_data_dir không tồn tại.
    F6: bỏ qua thư mục có tên không hợp lệ (thư mục ẩn, có khoảng trắng...) hoặc là
    symlink trỏ ra ngoài raw_data_dir — cùng quy tắc với project_dir_for().
    """
    root = Path(raw_data_dir)
    if not root.exists():
        return []
    names: list[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            project_dir_for(p.name, raw_data_dir)
        except InvalidProjectError as exc:
            if not p.name.startswith("."):
                logger.warning("Bỏ qua thư mục project: %s", exc)
            continue
        names.append(p.name)
    return sorted(names)


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
        content = str(llm.invoke(prompt).content).strip()
        start = content.find("[")
        end = content.rfind("]") + 1
        if 0 <= start < end:
            return json.loads(content[start:end])
    except Exception as exc:
        logger.warning("Entity extraction failed: %s", exc)
    return []


# Timeout (giây) httpx cho MỌI lần gọi Ollama /api/embed của rag-api: embed câu hỏi
# (embed_texts, dùng bởi /ask và /retrieve) lẫn từng batch lúc ingest
# (_run_async_pipeline). Mặc định khớp `${RAG_TIMEOUT:-600}` trong docker-compose.yml.
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


# R2: số lần THỬ (gồm lần đầu) khi embed một batch lúc ingest (OOM/transient HTTP).
# Mặc định khớp `${INGEST_EMBED_RETRIES:-3}` trong docker-compose.yml; kẹp tối thiểu 1
# — giá trị 0 trước đây làm vòng lặp không chạy lần nào và batch lỗi vì vectors=None.
_INGEST_EMBED_RETRIES: int = max(1, int(os.environ.get("INGEST_EMBED_RETRIES", "3")))


def collection_vector_size(client: QdrantClient, collection_name: str) -> int | None:
    """A2 — trả về số chiều vector của collection (None nếu không đọc được).

    Dùng chung cho app.py (phát hiện sớm việc đổi EMBEDDING_MODEL lệch dimension → báo
    lỗi rõ ràng yêu cầu /reset-ingest, thay vì để Qdrant ném lỗi khó hiểu giữa chừng truy
    vấn) và cho ingest (biết vector size của collection có sẵn mà KHÔNG phải gọi embed).
    """
    try:
        params: Any = client.get_collection(collection_name).config.params.vectors
        if hasattr(params, "size"):
            return int(params.size)
        if isinstance(params, dict) and params:
            first = next(iter(params.values()))
            size = getattr(first, "size", None)
            return int(size) if size else None
    except Exception as exc:
        logger.debug("A2: không đọc được vector size của %s: %s", collection_name, exc)
    return None


class _CollectionGate:
    """Tạo collection Qdrant đúng một lần với số chiều của batch embed THẬT đầu tiên.

    Thay cho lời gọi embed "health check" trước đây (mỗi project ingest embed thêm một
    chuỗi giả chỉ để biết vector size): collection đã tồn tại → size đọc từ config
    (collection_vector_size, không embed); chưa tồn tại → batch embed xong ĐẦU TIÊN tạo
    collection ngay trước khi upsert. ensure() không có await bên trong nên các coroutine
    batch chạy đồng thời không thể chen ngang giữa bước kiểm tra và bước tạo; batch đầu
    lỗi thì batch thành công kế tiếp tạo collection.

    *drop_existing* (reset=True): collection cũ CHỈ bị xóa ở batch embed thành công đầu
    tiên, ngay trước khi tạo lại theo số chiều mới. Trước đây embed "health check" lỗi
    (Ollama down, EMBEDDING_MODEL mới chưa pull) làm /reset-ingest dừng TRƯỚC bước xóa;
    nếu xóa ngay từ đầu thì mọi batch lỗi sẽ để lại project không còn index nào.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int | None,
        *,
        drop_existing: bool = False,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._drop_pending = drop_existing

    @property
    def drop_pending(self) -> bool:
        """True nếu reset chưa xóa collection cũ (chưa batch embed nào thành công)."""
        return self._drop_pending

    def drop_if_pending(self) -> None:
        """reset: xóa collection cũ (một lần) — số chiều cũ không còn ràng buộc batch sau."""
        if not self._drop_pending:
            return
        self._drop_pending = False
        self._vector_size = None
        if self._client.collection_exists(self._collection_name):
            logger.warning("Deleting collection before reset: %s", self._collection_name)
            self._client.delete_collection(self._collection_name)

    def ensure(self, dim: int) -> None:
        self.drop_if_pending()
        if self._vector_size is None:
            _ensure_collection(self._client, self._collection_name, vector_size=dim)
            self._vector_size = dim
        elif dim != self._vector_size:
            # Lệch dimension → upsert chắc chắn hỏng; báo rõ nguyên nhân thay vì lỗi Qdrant mơ hồ.
            raise RuntimeError(
                f"Collection '{self._collection_name}' có vector dim={self._vector_size} nhưng "
                f"EMBEDDING_MODEL hiện tại sinh dim={dim} → EMBEDDING_MODEL đã thay đổi. "
                "Chạy POST /reset-ingest để dựng lại index."
            )


async def _embed_and_upsert_batch_async(
    http_client: httpx.AsyncClient,
    qdrant: AsyncQdrantClient,
    sem: asyncio.Semaphore,
    batch_no: int,
    total_batches: int,
    chunk_batch: list[Document],
    collection_name: str,
    project: str,
    gate: _CollectionGate,
) -> list[tuple]:
    """
    Embed một batch qua Ollama /api/embed rồi upsert vào Qdrant.
    Semaphore giới hạn số batch chạy đồng thời = INGEST_EMBED_WORKERS.
    *gate* bảo đảm collection tồn tại (tạo theo số chiều vector thật) trước khi upsert.
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
                embeddings = resp.json()["embeddings"]
                # Số vector phải khớp số chunk: zip() bên dưới cắt im lặng → chunk thiếu
                # vector bị coi như đã upsert, bước dọn F11 xóa bản cũ của nó → index thủng.
                # Coi là batch lỗi (retry như lỗi HTTP) — không bao giờ chạm tới gate/dọn.
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Ollama /api/embed trả {len(embeddings)} vector cho {len(texts)} chunk"
                    )
                vectors = embeddings
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
        assert vectors is not None  # vòng lặp trên hoặc gán vectors hoặc raise
        if vectors:
            gate.ensure(len(vectors[0]))

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


# F11: số point mỗi trang khi scroll/xóa point cũ.
_STALE_PAGE_SIZE: int = 1000


def _delete_stale_points(client: QdrantClient, collection_name: str, keep_ids: set[str]) -> list[str]:
    """F11 — xóa các point KHÔNG thuộc lần ingest hiện tại (dùng cho ingest không-reset).

    Point ID = UUID5(relative_path, loader_index, chunk_index, content_hash, embedding_model)
    nên chunk đã sửa, file đã xóa/đổi tên hay vector của model cũ đều có ID nằm ngoài
    *keep_ids* → bị xóa; chunk không đổi giữ nguyên ID → không bị động tới.
    Gom đủ danh sách ID trước rồi mới xóa (không xóa khi đang scroll). Trả về các ID đã xóa.
    """
    stale: list[str] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=_STALE_PAGE_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        stale.extend(str(r.id) for r in records if str(r.id) not in keep_ids)
        if offset is None:
            break
    for start in range(0, len(stale), _STALE_PAGE_SIZE):
        page: list[ExtendedPointId] = [*stale[start: start + _STALE_PAGE_SIZE]]
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=page),
            wait=True,
        )
    return stale


def _existing_point_ids(client: QdrantClient, collection_name: str, ids: list[str]) -> set[str]:
    """Ingest tăng dần — trả về các ID trong *ids* ĐÃ có trong collection.

    retrieve theo trang _STALE_PAGE_SIZE ID, không lấy payload/vector (chỉ cần biết tồn
    tại). Point ID tất định theo nội dung + embedding model nên ID đã có ⇒ vector đã có
    đúng cho chunk đó → không cần embed lại.
    """
    found: set[str] = set()
    for start in range(0, len(ids), _STALE_PAGE_SIZE):
        page: list[ExtendedPointId] = [*ids[start: start + _STALE_PAGE_SIZE]]
        records = client.retrieve(
            collection_name=collection_name,
            ids=page,
            with_payload=False,
            with_vectors=False,
        )
        found.update(str(r.id) for r in records)
    return found


@dataclass
class _PipelineResult:
    """Kết quả _run_async_pipeline cho một project."""

    # ID point của MỌI chunk hiện tại (mới + không đổi) — tập giữ lại khi dọn point cũ.
    all_ids: list[str]
    # (chunk, point_id) vừa embed + upsert ở lần ingest này (chunk mới/đã sửa).
    upserted: list[tuple]
    # (chunk, point_id) đã có sẵn trong Qdrant — không embed lại.
    unchanged: list[tuple]
    points_count: int
    # ID point cũ đã xóa — None nếu KHÔNG chạy bước dọn (reset=True hoặc có batch lỗi).
    stale_ids: list[str] | None
    failed_batches: int = 0
    batch_error: str | None = None


async def _run_async_pipeline(
    chunks: list[Document],
    collection_name: str,
    project: str,
    reset: bool,
) -> _PipelineResult:
    """
    Async embed + upsert pipeline: INGEST_EMBED_WORKERS batch chạy đồng thời.

    Cách hoạt động:
      • Ingest không-reset: tra các point_id hiện tại đã có trong collection
        (_existing_point_ids) → chỉ embed + upsert chunk có ID mới; chunk không đổi
        được bỏ qua. Không có chunk mới → không tạo HTTP client, không gọi Ollama.
      • reset=True: embed + upsert toàn bộ như trước; collection cũ bị xóa ở batch embed
        thành công đầu tiên (_CollectionGate) → Ollama down/model chưa pull thì mọi batch
        lỗi và index cũ được GIỮ NGUYÊN (failed_batches/batch_error trong response).
      • Không còn embed "health check": vector size đọc từ config collection có sẵn,
        hoặc lấy từ batch embed thật đầu tiên khi phải tạo collection (_CollectionGate).
      • asyncio.gather gửi tất cả batch cùng lúc.
      • Semaphore giới hạn Ollama request đồng thời = INGEST_EMBED_WORKERS.
      • Embed và upsert của mỗi batch chạy nối tiếp trong cùng coroutine,
        nhưng nhiều batch chạy song song → GPU và Qdrant luôn bận.
      • Tăng INGEST_EMBED_WORKERS = OLLAMA_NUM_PARALLEL để khai thác tối đa.
    """
    pairs = [(chunk, make_point_id(chunk)) for chunk in chunks]
    all_ids = [pid for _, pid in pairs]

    sync_client = QdrantClient(url=QDRANT_URL)

    # reset=True: KHÔNG xóa collection ở đây (gate xóa khi có batch embed thành công) và
    # KHÔNG đọc vector size / ID cũ — collection sẽ dựng lại, thường chính vì
    # EMBEDDING_MODEL đổi số chiều (đọc size cũ sẽ làm gate báo lệch dimension oan).
    vector_size: int | None = None
    existing_ids: set[str] = set()
    if not reset and sync_client.collection_exists(collection_name):
        vector_size = collection_vector_size(sync_client, collection_name)
        existing_ids = _existing_point_ids(sync_client, collection_name, all_ids)

    new_chunks = [chunk for chunk, pid in pairs if pid not in existing_ids]
    unchanged = [(chunk, pid) for chunk, pid in pairs if pid in existing_ids]
    logger.info(
        "Ingest project=%s: %d chunk mới/đã sửa cần embed, %d chunk không đổi (bỏ qua embed).",
        project, len(new_chunks), len(unchanged),
    )

    results: list[list[tuple] | BaseException] = []
    total_batches = 0
    gate = _CollectionGate(sync_client, collection_name, vector_size, drop_existing=reset)
    if not new_chunks:
        # reset mà project không có chunk nào → không có gì để embed: xóa ngay như trước.
        gate.drop_if_pending()
    else:
        # read = RAG_TIMEOUT (trước đây hardcode 600 nên RAG_TIMEOUT không có tác dụng lúc ingest).
        timeout = httpx.Timeout(connect=10.0, read=_EMBED_HTTP_TIMEOUT, write=60.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            qdrant = AsyncQdrantClient(url=QDRANT_URL)
            sem = asyncio.Semaphore(INGEST_EMBED_WORKERS)
            chunk_batches = list(batched(new_chunks, UPSERT_BATCH_SIZE))
            total_batches = len(chunk_batches)

            logger.info(
                "Async pipeline: %s batches, batch_size=%s, concurrency=%s",
                total_batches, UPSERT_BATCH_SIZE, INGEST_EMBED_WORKERS,
            )

            tasks = [
                _embed_and_upsert_batch_async(
                    http_client, qdrant, sem,
                    i + 1, total_batches, batch,
                    collection_name, project, gate,
                )
                for i, batch in enumerate(chunk_batches)
            ]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await qdrant.close()

    # R1: một batch lỗi KHÔNG làm hỏng cả project ingest. Bỏ qua batch lỗi và ghi
    # log số batch thất bại; các batch thành công vẫn được upsert. count phản ánh
    # đúng số point thực tế đã vào Qdrant (partial, thay vì mất sạch như asyncio.gather mặc định).
    # BaseException (không chỉ Exception): gather(return_exceptions=True) trả cả
    # CancelledError — trước đây nó lọt vào nhánh "thành công" và bị iterate.
    failed = [r for r in results if isinstance(r, BaseException)]
    if failed:
        logger.error(
            "Ingest project=%s: %d/%d batch thất bại (đã bỏ qua, phần còn lại vẫn upsert). Lỗi đầu tiên: %s",
            project, len(failed), total_batches, failed[0],
        )
    upserted: list[tuple] = [
        pair for r in results if not isinstance(r, BaseException) for pair in r
    ]
    if gate.drop_pending:
        logger.warning(
            "Reset project=%s: không batch embed nào thành công → GIỮ NGUYÊN collection cũ '%s'.",
            project, collection_name,
        )

    # Không còn embed "health check" tạo sẵn collection: nếu mọi batch lỗi (hoặc project
    # không có chunk nào) thì collection có thể chưa tồn tại → không scroll/count trên nó.
    collection_exists = sync_client.collection_exists(collection_name)

    # F11: upsert không bao giờ xóa point của chunk đã sửa/file đã xóa → dọn sau khi
    # upsert xong. Chỉ dọn khi MỌI batch thành công: nếu có batch lỗi, phiên bản mới
    # của chunk đó chưa vào Qdrant — giữ bản cũ còn hơn làm thủng index.
    # Tập giữ lại = MỌI chunk hiện tại (mới + không đổi): chunk không đổi không được
    # upsert lại ở lần này nên KHÔNG được dựng keep-set từ danh sách vừa upsert.
    stale_ids: list[str] | None = None
    if not reset:
        if failed:
            logger.warning(
                "Ingest project=%s: bỏ qua dọn point cũ vì có batch lỗi — chạy lại /ingest sau.",
                project,
            )
        elif not collection_exists:
            stale_ids = []  # không có collection → không có point cũ nào để dọn
        else:
            stale_ids = _delete_stale_points(sync_client, collection_name, set(all_ids))
            if stale_ids:
                logger.info("Ingest project=%s: đã xóa %d point cũ (chunk đã sửa/xóa).",
                            project, len(stale_ids))

    points_count = (
        sync_client.count(collection_name=collection_name, exact=True).count
        if collection_exists else 0
    )
    return _PipelineResult(
        all_ids=all_ids,
        upserted=upserted,
        unchanged=unchanged,
        points_count=points_count,
        stale_ids=stale_ids,
        failed_batches=len(failed),
        batch_error=f"{type(failed[0]).__name__}: {failed[0]}" if failed else None,
    )


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


def _plan_graph_writes(
    graph_module: ModuleType,
    project: str,
    pairs: list[tuple],
    reset: bool,
) -> tuple[list[tuple], list[tuple]]:
    """Ingest tăng dần cho Neo4j — trả về (pairs cần ghi, pairs mới cần trích xuất thực thể).

    - reset=True: graph của project vừa bị xóa → ghi + trích xuất toàn bộ như trước.
    - Không-reset: hỏi Neo4j chunk hiện tại nào ĐÃ có node (existing_chunk_ids). Chunk
      thiếu là chunk mới/đã sửa — hoặc mọi chunk khi project chưa có graph (vd. vừa bật
      GRAPH_ENABLED) hay graph dở dang do lần ingest trước lỗi → dựng bù đủ. Chỉ chunk
      thiếu được trích xuất thực thể; chunk đã có giữ nguyên node + MENTIONS.
    - save_project_graph chỉ nối NEXT_CHUNK giữa các pair được truyền vào (nhóm theo
      source_file) → với file có chunk thiếu, ghi lại CẢ chunk không đổi của file đó
      (MERGE idempotent) để chuỗi NEXT_CHUNK không đứt quanh chunk mới. File không có
      chunk thiếu không được ghi lại.
    """
    if reset:
        return pairs, pairs
    existing = graph_module.existing_chunk_ids(project, [pid for _, pid in pairs])
    missing = [(chunk, pid) for chunk, pid in pairs if pid not in existing]
    touched = {chunk.metadata.get("source_file", "unknown") for chunk, _ in missing}
    to_write = [
        (chunk, pid) for chunk, pid in pairs
        if chunk.metadata.get("source_file", "unknown") in touched
    ]
    return to_write, missing


def _ingest_project(project: str, reset: bool = False) -> dict:
    # F6: kiểm tra tên project TRƯỚC mọi thao tác filesystem/Qdrant/Neo4j.
    project_dir = str(project_dir_for(project))
    collection_name = get_collection_name(project)

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

    # Async pipeline: embed + upsert INGEST_EMBED_WORKERS batch đồng thời (chỉ chunk mới
    # khi không-reset).
    result = asyncio.run(
        _run_async_pipeline(chunks, collection_name, project, reset)
    )
    logger.info(
        "Project '%s' done. Points: %s | upserted: %s | không đổi: %s",
        project, result.points_count, len(result.upserted), len(result.unchanged),
    )

    # ── Tích hợp Neo4j graph ──────────────────────────────────────────────────
    graph_chunks_saved = 0
    if GRAPH_ENABLED and _graph_module is not None:
        try:
            _graph_module.setup_schema()
            # Chỉ chunk ĐANG có trong Qdrant (không đổi + vừa upsert thành công) mới được
            # phản chiếu sang Neo4j — chunk của batch lỗi chưa có vector thì chưa ghi.
            graph_pairs, new_graph_pairs = _plan_graph_writes(
                _graph_module, project, result.unchanged + result.upserted, reset)
            graph_chunks_saved = _graph_module.save_project_graph(
                project, graph_pairs)
            # F11: Qdrant vừa dọn point cũ → xóa Chunk/Document/Entity tương ứng
            # trong Neo4j để graph enrichment không trả lại nội dung đã bị xóa.
            # Tập giữ lại = MỌI chunk hiện tại (mới + không đổi) — nếu chỉ giữ chunk vừa
            # ghi thì Chunk node của chunk không đổi sẽ bị xóa oan.
            if result.stale_ids is not None:
                _graph_module.delete_stale_chunks(project, result.all_ids)

            if GRAPH_ENTITY_EXTRACTION and CHAT_MODEL and new_graph_pairs:
                logger.info(
                    "Entity extraction enabled for project '%s' (%d chunk mới)",
                    project, len(new_graph_pairs))
                # Timeout thật qua client_kwargs (xem make_chat_model) — trước đây không
                # có timeout nên một lời gọi treo làm /ingest treo vô hạn.
                llm = make_chat_model(CHAT_MODEL, temperature=0)
                # Mỗi Document gọi LLM một lần (không phải mỗi Chunk) — nhanh hơn nhiều.
                # Chỉ chunk mới: chunk không đổi giữ nguyên quan hệ MENTIONS đã có.
                doc_pairs: dict[str, list[tuple]] = {}
                for chunk, pid in new_graph_pairs:
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
        # Số chunk vừa embed + upsert (mới/đã sửa). Ingest lại khi không có gì đổi → 0.
        "upserted": len(result.upserted),
        # Số chunk đã có sẵn trong Qdrant (cùng point_id) → bỏ qua, không embed lại.
        "skipped_unchanged": len(result.unchanged),
        # R1: số batch embed/upsert lỗi (đã bỏ qua) + lỗi đầu tiên để thấy ngay nguyên
        # nhân (Ollama down, lệch dimension...) mà không cần đọc log.
        "failed_batches": result.failed_batches,
        "batch_error": result.batch_error,
        "points_count": result.points_count,
        "graph_chunks_saved": graph_chunks_saved,
        # F11: số point cũ (chunk đã sửa/xóa) đã bị xóa khỏi Qdrant ở lần ingest này.
        "stale_removed": len(result.stale_ids or []),
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
        "total_upserted": sum(r.get("upserted", 0) for r in results.values()),
        "total_skipped_unchanged": sum(r.get("skipped_unchanged", 0) for r in results.values()),
        "reset": reset,
        "details": results,
    }
