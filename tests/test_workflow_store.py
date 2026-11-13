"""Unit test cho agent-api/store.py — kho WorkflowRecord trên SQLite (stdlib).

Không cần Docker/Ollama: mỗi test dùng file DB riêng trong tmp_path (không bao giờ /data).
"""

import sqlite3
import threading

import pytest
import store as store_mod
from store import INTERRUPTED_ERROR, MAX_STORED_WORKFLOWS, WorkflowStore


def _rec(wid: str, status: str = "completed", created_at: str = "2026-01-01T00:00:00+00:00", **extra) -> dict:
    rec = {
        "workflow_id": wid,
        "status": status,
        "project": "crm",
        "user_input": f"input {wid}",
        "tech_stack": None,
        "step_outputs": {"ba": f"BA {wid}"},
        "completed_steps": ["ba"],
        "failed_steps": [],
        "error": None,
        "artifacts": {},
        "created_at": created_at,
        "completed_at": None if status in ("pending", "running") else created_at,
    }
    rec.update(extra)
    return rec


def _ts(i: int) -> str:
    return f"2026-01-01T00:00:{i:02d}+00:00"


@pytest.fixture()
def db(tmp_path):
    s = WorkflowStore(tmp_path / "sub" / "wf.sqlite3")
    s.init()
    return s


def _raw_rows(path) -> dict[str, tuple]:
    conn = sqlite3.connect(path)
    try:
        return {r[0]: r for r in conn.execute("SELECT * FROM workflows")}
    finally:
        conn.close()


def test_constructor_does_no_io_and_init_creates_db_in_wal_mode(tmp_path):
    path = tmp_path / "nested" / "dir" / "wf.sqlite3"
    s = WorkflowStore(path)
    assert not path.parent.exists(), "tạo WorkflowStore không được đụng tới disk"
    s.init()
    s.init()  # idempotent
    assert path.exists()
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        cols = {r[1] for r in conn.execute("PRAGMA table_info(workflows)")}
        idx = {r[1] for r in conn.execute("PRAGMA index_list(workflows)")}
    finally:
        conn.close()
    assert {"workflow_id", "status", "created_at", "data"} <= cols
    assert {"idx_workflows_status", "idx_workflows_created_at"} <= idx


def test_db_path_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("WORKFLOW_DB_PATH", raising=False)
    assert store_mod.db_path_from_env() == "/data/memory/workflows.sqlite3"
    monkeypatch.setenv("WORKFLOW_DB_PATH", "")
    assert store_mod.db_path_from_env() == "/data/memory/workflows.sqlite3"
    monkeypatch.setenv("WORKFLOW_DB_PATH", str(tmp_path / "x.sqlite3"))
    assert WorkflowStore().db_path == tmp_path / "x.sqlite3"


def test_save_get_roundtrip_and_upsert(db):
    rec = _rec("a", status="running", step_outputs={"ba": "Tiếng Việt ✓"})
    db.save(rec)
    assert db.get("a") == rec
    assert db.get("missing") is None

    rec["status"] = "completed"
    rec["completed_steps"] = ["ba", "pm"]
    db.save(rec)
    assert db.get("a")["status"] == "completed"
    assert db.count() == 1
    [summary] = db.list_recent(10)
    assert summary == {
        "workflow_id": "a", "status": "completed", "project": "crm",
        "completed_steps": 2, "created_at": rec["created_at"], "completed_at": rec["completed_at"],
    }


def test_save_rejects_record_without_index_columns(db):
    with pytest.raises(ValueError):
        db.save({"workflow_id": "x"})


def test_list_recent_newest_first_with_insertion_tiebreak(db):
    db.save(_rec("old", created_at=_ts(1)))
    db.save(_rec("same-1", created_at=_ts(5)))
    db.save(_rec("same-2", created_at=_ts(5)))
    db.save(_rec("new", created_at=_ts(9)))
    assert [r["workflow_id"] for r in db.list_recent(10)] == ["new", "same-2", "same-1", "old"]
    assert [r["workflow_id"] for r in db.list_recent(2)] == ["new", "same-2"]


def test_retention_keeps_newest_and_never_drops_active_runs(tmp_path):
    assert MAX_STORED_WORKFLOWS == 500
    s = WorkflowStore(tmp_path / "wf.sqlite3", max_records=3)
    s.init()
    # Run đang chạy CŨ nhất — không bao giờ bị xóa dù nằm ngoài top-3.
    s.save(_rec("active-pending", status="pending", created_at=_ts(0)))
    s.save(_rec("active-running", status="running", created_at=_ts(1)))
    for i in range(2, 8):
        s.save(_rec(f"done-{i}", created_at=_ts(i)))
    ids = {r["workflow_id"] for r in s.list_recent(100)}
    assert ids == {"active-pending", "active-running", "done-5", "done-6", "done-7"}

    # Khi run đang chạy kết thúc, nó trở thành ứng viên retention như mọi bản ghi khác.
    s.save(_rec("active-running", status="completed", created_at=_ts(1)))
    ids = {r["workflow_id"] for r in s.list_recent(100)}
    assert ids == {"active-pending", "done-5", "done-6", "done-7"}


def test_recover_interrupted_only_touches_active_runs(db):
    db.save(_rec("p", status="pending", created_at=_ts(1)))
    db.save(_rec("r", status="running", created_at=_ts(2), error="ba: timeout"))
    db.save(_rec("c", status="completed", created_at=_ts(3)))
    db.save(_rec("f", status="failed", created_at=_ts(4), error="boom"))
    before = _raw_rows(db.db_path)

    recovered = db.recover_interrupted()
    assert sorted(recovered) == ["p", "r"]

    after = _raw_rows(db.db_path)
    assert after["c"] == before["c"] and after["f"] == before["f"], "bản ghi đã kết thúc phải giữ nguyên byte"

    p, r = db.get("p"), db.get("r")
    assert p["status"] == r["status"] == "failed"
    assert p["error"] == INTERRUPTED_ERROR == "Bị gián đoạn do agent-api khởi động lại"
    assert r["error"] == INTERRUPTED_ERROR  # đúng chuỗi, không ghép lỗi cũ
    assert p["completed_at"] and r["completed_at"]
    # Cột tóm tắt đồng bộ với JSON.
    statuses = {row["workflow_id"]: row["status"] for row in db.list_recent(10)}
    assert statuses == {"p": "failed", "r": "failed", "c": "completed", "f": "failed"}

    # Gọi lại: không còn gì để khôi phục.
    assert db.recover_interrupted() == []


def test_recover_interrupted_handles_corrupt_json(db):
    conn = sqlite3.connect(db.db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO workflows (workflow_id, status, created_at, data) VALUES (?, ?, ?, ?)",
                ("bad", "running", _ts(1), "{not json"),
            )
    finally:
        conn.close()
    assert db.recover_interrupted() == ["bad"]
    rec = db.get("bad")
    assert rec["status"] == "failed" and rec["error"] == INTERRUPTED_ERROR
    assert rec["created_at"] == _ts(1)


def test_get_ignores_corrupt_row(db):
    conn = sqlite3.connect(db.db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO workflows (workflow_id, status, created_at, data) VALUES (?, ?, ?, ?)",
                ("bad", "completed", _ts(1), "{not json"),
            )
    finally:
        conn.close()
    assert db.get("bad") is None


def test_concurrent_writers_and_readers(db):
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(20):
                rec = _rec(f"w{n}", status="running", created_at=_ts(n))
                rec["completed_steps"] = [f"s{j}" for j in range(i + 1)]
                db.save(rec)
        except BaseException as exc:  # pragma: no cover - chỉ để báo lỗi thread
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(40):
                db.list_recent(50)
                db.get("w0")
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert db.count() == 6
    assert all(r["completed_steps"] == 20 for r in db.list_recent(10))


def test_cancelled_is_finished_for_retention_and_recovery(tmp_path):
    assert "cancelled" in store_mod.FINISHED_STATUSES
    assert not set(store_mod.FINISHED_STATUSES) & set(store_mod.ACTIVE_STATUSES)
    s = WorkflowStore(tmp_path / "wf.sqlite3", max_records=2)
    s.init()
    s.save(_rec("old-cancelled", status="cancelled", created_at=_ts(0), error="Đã hủy"))
    s.save(_rec("old-pending", status="pending", created_at=_ts(1)))
    s.save(_rec("new-1", created_at=_ts(2)))
    s.save(_rec("new-2", status="cancelled", created_at=_ts(3)))
    # Bản ghi cancelled cũ bị retention xóa như completed/failed; run pending thì không.
    assert {r["workflow_id"] for r in s.list_recent(10)} == {"old-pending", "new-1", "new-2"}

    before = _raw_rows(s.db_path)["new-2"]
    assert s.recover_interrupted() == ["old-pending"]
    assert _raw_rows(s.db_path)["new-2"] == before, "recovery không được đụng tới run đã hủy"
