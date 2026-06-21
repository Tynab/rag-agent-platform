"""Cấu hình chung cho test suite (T1).

- Thêm thư mục của hai service vào sys.path để import module trực tiếp mà không
  cần đóng gói package (agent-api / rag-api / open-webui-tools).
- Đặt sẵn các biến môi trường tối thiểu mà một số module đọc lúc import, để test
  import được kể cả khi không có .env thật.

Lưu ý: cả agent-api lẫn rag-api đều có file tên `app.py` → ĐỪNG import `app` trong
unit test (xung đột tên). Chỉ test các module có tên duy nhất: artifacts, workflow,
agents (agent-api) và ingest (rag-api).
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for sub in ("agent-api", "rag-api", "open-webui-tools"):
    p = ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Env tối thiểu cho các module đọc cấu hình lúc import.
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("COLLECTION_NAME", "yan_raw_docs")
os.environ.setdefault("EMBEDDING_MODEL", "qwen3-embedding:8b")
os.environ.setdefault("CHAT_MODEL", "qwen3.6:35b")
os.environ.setdefault("RAG_TOP_K", "5")
os.environ.setdefault("RAG_API_URL", "http://localhost:8090")
os.environ.setdefault("RAW_DATA_DIR", str(ROOT / "data" / "raw"))
# rag-api/ingest.py đọc các biến này bằng _require_env (raise nếu thiếu) lúc import —
# phải set đủ để `import ingest` thành công khi CI đã cài service deps.
os.environ.setdefault("CHUNK_SIZE", "1000")
os.environ.setdefault("CHUNK_OVERLAP", "150")
os.environ.setdefault("UPSERT_BATCH_SIZE", "32")
