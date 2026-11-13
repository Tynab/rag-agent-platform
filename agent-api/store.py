"""
store.py — Kho lưu trạng thái workflow bền vững (SQLite, stdlib sqlite3)
=======================================================================

Thay cho dict in-memory cũ (mất sạch khi container restart). Mỗi WorkflowRecord được
lưu thành một dòng: JSON đầy đủ (cột ``data``) + các cột tóm tắt có index để lọc/sắp
xếp/liệt kê mà không phải parse JSON lớn (step_outputs của 15 agent có thể vài trăm KB).

Schema bảng ``workflows``
-------------------------
    workflow_id      TEXT PRIMARY KEY  — id workflow (uuid4)
    status           TEXT  (index)     — pending | running | completed | failed
    created_at       TEXT  (index)     — ISO-8601 UTC; sắp xếp mới nhất trước
    completed_at     TEXT              — ISO-8601 UTC hoặc NULL
    project          TEXT              — RAG project filter hoặc NULL
    completed_count  INTEGER           — số bước đã xong (cho GET /workflows)
    data             TEXT              — JSON đầy đủ của WorkflowRecord

Cột tóm tắt và ``data`` luôn được ghi trong CÙNG một câu lệnh → không thể lệch nhau.

Thread-safety
-------------
- Connection-per-call: mỗi thao tác mở một connection riêng rồi đóng ngay — dùng được
  từ event-loop thread (lifespan), threadpool của endpoint sync và BackgroundTasks.
- WAL mode: reader không chặn writer và ngược lại (UI poll liên tục trong lúc workflow ghi).
- ``_write_lock`` tuần tự hóa mọi thao tác ghi trong process → không có SQLITE_BUSY giữa
  các thread; ``busy_timeout`` phòng trường hợp process khác (sqlite3 CLI) đang giữ lock.
- Chỉ hỗ trợ MỘT process agent-api ghi vào file DB (uvicorn 1 worker như Dockerfile):
  recovery lúc khởi động coi mọi run pending/running là của process đã chết.

Retention
---------
Giữ tối đa ``MAX_STORED_WORKFLOWS`` (500) bản ghi mới nhất. Chỉ xóa bản ghi đã KẾT THÚC
(completed/failed) — không bao giờ xóa run đang pending/running (giữ quy tắc C4 cũ).

Module không làm I/O lúc import: đường dẫn DB đọc từ env WORKFLOW_DB_PATH khi tạo
``WorkflowStore`` và file/thư mục chỉ được tạo khi gọi ``init()``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent-api.store")

DEFAULT_DB_PATH = "/data/memory/workflows.sqlite3"
# Số bản ghi workflow tối đa giữ trong DB (bản ghi kết thúc cũ nhất bị xóa trước).
MAX_STORED_WORKFLOWS = 500
# Lỗi gán cho run bị bỏ dở (pending/running) khi agent-api khởi động lại.
INTERRUPTED_ERROR = "Bị gián đoạn do agent-api khởi động lại"

ACTIVE_STATUSES: tuple[str, ...] = ("pending", "running")
FINISHED_STATUSES: tuple[str, ...] = ("completed", "failed")

_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 30_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id     TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    project         TEXT,
    completed_count INTEGER NOT NULL DEFAULT 0,
    data            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);
"""

_UPSERT_SQL = """
INSERT INTO workflows (workflow_id, status, created_at, completed_at, project, completed_count, data)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(workflow_id) DO UPDATE SET
    status = excluded.status,
    created_at = excluded.created_at,
    completed_at = excluded.completed_at,
    project = excluded.project,
    completed_count = excluded.completed_count,
    data = excluded.data
"""

# Xóa bản ghi đã kết thúc nằm ngoài top-N mới nhất (tính trên TOÀN BỘ bảng, kể cả run
# đang chạy) — run pending/running không bao giờ bị xóa.
_PRUNE_SQL = """
DELETE FROM workflows
WHERE status IN ('completed', 'failed')
  AND workflow_id NOT IN (
      SELECT workflow_id FROM workflows ORDER BY created_at DESC, rowid DESC LIMIT ?
  )
"""


def db_path_from_env() -> str:
    """Đường dẫn file SQLite từ env WORKFLOW_DB_PATH (rỗng/không đặt → mặc định)."""
    return os.environ.get("WORKFLOW_DB_PATH", "").strip() or DEFAULT_DB_PATH


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summary_row(record: dict[str, Any]) -> tuple[Any, ...]:
    """Tách các cột tóm tắt + JSON đầy đủ từ dict WorkflowRecord (đã ở dạng JSON-able)."""
    workflow_id = record.get("workflow_id")
    status = record.get("status")
    created_at = record.get("created_at")
    if not workflow_id or not status or not created_at:
        raise ValueError("record thiếu workflow_id/status/created_at")
    return (
        str(workflow_id),
        str(status),
        str(created_at),
        record.get("completed_at"),
        record.get("project"),
        len(record.get("completed_steps") or []),
        json.dumps(record, ensure_ascii=False),
    )


class WorkflowStore:
    """Repository WorkflowRecord trên SQLite. Nhận/trả dict JSON-able (không phụ thuộc pydantic)."""

    def __init__(self, db_path: str | Path | None = None, max_records: int = MAX_STORED_WORKFLOWS) -> None:
        self.db_path = Path(db_path) if db_path is not None else Path(db_path_from_env())
        self.max_records = max_records
        self._write_lock = threading.Lock()

    # ── Kết nối ────────────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Mở connection riêng cho một thao tác; luôn đóng khi xong (``with conn`` chỉ commit)."""
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            yield conn
        finally:
            conn.close()

    def init(self) -> None:
        """Tạo thư mục, file DB, bật WAL và tạo schema (idempotent). Raise nếu không mở được DB."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as conn:
            # journal_mode=WAL lưu bền trong file DB — chỉ cần đặt một lần.
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if not mode or str(mode[0]).lower() != "wal":
                logger.warning("SQLite không bật được WAL (journal_mode=%s) — vẫn tiếp tục.", mode)
            with conn:
                conn.executescript(_SCHEMA)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        logger.info("Workflow store sẵn sàng: %s", self.db_path)

    # ── Ghi ────────────────────────────────────────────────────────────────────

    def save(self, record: dict[str, Any]) -> None:
        """Upsert *record* (dict JSON-able của WorkflowRecord) rồi áp dụng retention."""
        row = _summary_row(record)
        with self._write_lock, self._connect() as conn, conn:
            conn.execute(_UPSERT_SQL, row)
            conn.execute(_PRUNE_SQL, (self.max_records,))

    def recover_interrupted(self, error: str = INTERRUPTED_ERROR) -> list[str]:
        """Chuyển mọi run pending/running sang failed (gọi MỘT lần lúc khởi động).

        Run còn pending/running trong DB lúc process mới khởi động chắc chắn thuộc process
        đã chết (BackgroundTasks không sống sót qua restart) → không bao giờ hoàn tất.
        Bản ghi completed/failed không bị đụng tới. Lỗi cũ (nếu có) được giữ sau *error*.
        Trả về danh sách workflow_id đã chuyển trạng thái.
        """
        now = _utc_now_iso()
        recovered: list[str] = []
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._write_lock, self._connect() as conn, conn:
            rows = conn.execute(
                f"SELECT workflow_id, data FROM workflows WHERE status IN ({placeholders})",
                ACTIVE_STATUSES,
            ).fetchall()
            for workflow_id, data in rows:
                try:
                    record = json.loads(data)
                    if not isinstance(record, dict):
                        raise ValueError("data không phải object JSON")
                except Exception as exc:
                    logger.warning("Recovery: bản ghi %s hỏng (%s) — dựng lại tối thiểu.", workflow_id, exc)
                    record = {"workflow_id": workflow_id, "user_input": "", "project": None}
                    created = conn.execute(
                        "SELECT created_at FROM workflows WHERE workflow_id = ?", (workflow_id,)
                    ).fetchone()
                    record["created_at"] = created[0] if created else now
                prev_error = record.get("error")
                record["status"] = "failed"
                record["error"] = f"{error} | {prev_error}" if prev_error else error
                record["completed_at"] = record.get("completed_at") or now
                conn.execute(_UPSERT_SQL, _summary_row(record))
                recovered.append(workflow_id)
        if recovered:
            logger.warning(
                "Recovery: %d workflow bị gián đoạn do restart → failed: %s",
                len(recovered), ", ".join(recovered),
            )
        return recovered

    # ── Đọc ────────────────────────────────────────────────────────────────────

    def get(self, workflow_id: str) -> dict[str, Any] | None:
        """Trả về dict WorkflowRecord đầy đủ hoặc None nếu không có / JSON hỏng."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except Exception as exc:
            logger.warning("Bản ghi workflow %s hỏng JSON (bỏ qua): %s", workflow_id, exc)
            return None
        return data if isinstance(data, dict) else None

    def list_recent(self, limit: int) -> list[dict[str, Any]]:
        """Tóm tắt *limit* workflow mới nhất (mới nhất trước) — chỉ đọc cột tóm tắt, không parse JSON."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT workflow_id, status, project, completed_count, created_at, completed_at "
                "FROM workflows ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
        return [
            {
                "workflow_id": r[0],
                "status": r[1],
                "project": r[2],
                "completed_steps": r[3],
                "created_at": r[4],
                "completed_at": r[5],
            }
            for r in rows
        ]

    def count(self) -> int:
        """Tổng số bản ghi đang lưu."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()
        return int(row[0]) if row else 0
