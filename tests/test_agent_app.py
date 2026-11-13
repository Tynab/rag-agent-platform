"""Test endpoint agent-api/app.py: SQLite store, crash recovery, lifespan, static mount.

agent-api/app.py trùng tên module `app` với rag-api → load bằng importlib dưới tên riêng
"agent_api_app" (không đụng sys.modules["app"]). LangGraph workflow được thay bằng graph
giả (không cần Ollama). Mỗi test trỏ WORKFLOW_DB_PATH / MEMORY_DIR / ARTIFACT_BASE vào
tmp_path — không bao giờ ghi vào /data.
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("workflow")
import artifacts  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from store import INTERRUPTED_ERROR, WorkflowStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "agent-api" / "static"


def _load_agent_app():
    name = "agent_api_app"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "agent-api" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


app_mod = _load_agent_app()
# Bản gốc (fixture `env` thay bằng no-op để test không gọi Ollama).
_REAL_VALIDATE_MODELS = app_mod._validate_models_on_startup

SUMMARY_KEYS = {"workflow_id", "status", "project", "completed_steps", "total_steps", "created_at", "completed_at"}


class _FakeGraph:
    """Thay LangGraph CompiledGraph: stream() phát lần lượt các chunk 'updates' cho trước."""

    def __init__(self, chunks, on_chunk=None, raise_after=None):
        self.chunks = chunks
        self.on_chunk = on_chunk
        self.raise_after = raise_after
        self.calls = []

    def stream(self, state, stream_mode="updates"):
        self.calls.append((state, stream_mode))
        for i, chunk in enumerate(self.chunks):
            yield chunk
            # Chạy sau khi app đã xử lý + persist chunk vừa yield.
            if self.on_chunk:
                self.on_chunk(i)
            if self.raise_after is not None and i == self.raise_after:
                raise RuntimeError("Ollama sập giữa chừng")


def _node(role, output, error=None):
    out = {"step_outputs": {role: output}, "completed_steps": [role]}
    if error:
        out["error"] = error
    return {role: out}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Cô lập app: DB/memory/artifact trong tmp_path, không gọi Ollama lúc startup."""
    db_path = tmp_path / "db" / "workflows.sqlite3"
    monkeypatch.setenv("WORKFLOW_DB_PATH", str(db_path))
    monkeypatch.setattr(app_mod, "_store", None)
    monkeypatch.setattr(app_mod, "MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path / "artifacts"))
    monkeypatch.setattr(app_mod, "_validate_models_on_startup", lambda: None)
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 0)
    monkeypatch.setattr(app_mod, "_WORKFLOW_MAX_SECONDS", 0)
    extracted = []
    monkeypatch.setattr(
        app_mod, "_extract_artifacts", lambda role, out, wid: extracted.append(role) or []
    )

    class Env:
        pass

    e = Env()
    e.tmp = tmp_path
    e.db_path = db_path
    e.extracted = extracted

    def use_graph(graph):
        monkeypatch.setattr(app_mod, "get_workflow", lambda: graph)
        return graph

    e.use_graph = use_graph
    yield e
    # Không test nào được chạm vào DB mặc định.
    assert app_mod._store is None or app_mod._store.db_path == db_path


def _run(client, **body):
    payload = {"user_input": "Xây app quản lý đơn hàng", "project": "crm", **body}
    r = client.post("/workflow/run", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "pending"
    return r.json()["workflow_id"]


# ── Chạy workflow ──────────────────────────────────────────────────────────────


def test_run_poll_completed_and_progress_is_persisted(env):
    snapshots = []

    def on_chunk(i):
        store = app_mod._get_store()
        [latest] = store.list_recent(1)
        snapshots.append(store.get(latest["workflow_id"]))

    graph = env.use_graph(_FakeGraph([_node("ba", "BA doc"), _node("pm", "PM doc")], on_chunk=on_chunk))
    with TestClient(app_mod.app) as client:
        wid = _run(client, tech_stack=["nestjs", "reactjs"])
        rec = client.get(f"/workflow/{wid}").json()

    assert rec["status"] == "completed"
    assert rec["step_outputs"] == {"ba": "BA doc", "pm": "PM doc"}
    assert rec["completed_steps"] == ["ba", "pm"]
    assert rec["failed_steps"] == [] and rec["error"] is None
    assert rec["completed_at"] and rec["tech_stack"] == ["nestjs", "reactjs"]
    assert set(rec) == set(app_mod.WorkflowRecord.model_fields)

    # Graph nhận state đầy đủ; tiến trình được ghi vào SQLite sau MỖI node.
    state, mode = graph.calls[0]
    assert mode == "updates" and state["user_input"] == "Xây app quản lý đơn hàng"
    assert [s["status"] for s in snapshots] == ["running", "running"]
    assert snapshots[0]["completed_steps"] == ["ba"] and snapshots[0]["completed_at"] is None
    assert snapshots[1]["step_outputs"] == {"ba": "BA doc", "pm": "PM doc"}

    # Episodic JSONL vẫn được ghi (vào MEMORY_DIR tạm).
    lines = (env.tmp / "memory" / "episodic" / "workflow_runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["completed"]


def test_error_marker_goes_to_failed_steps_not_artifacts(env):
    env.use_graph(_FakeGraph([
        _node("ba", "BA doc"),
        _node("fe", "[LỖI fe: Ollama timeout]", error="fe: Ollama timeout"),
        _node("be", "### FILE: src/main.ts\n```ts\nconsole.log(1)\n```"),
    ]))
    with TestClient(app_mod.app) as client:
        wid = _run(client)
        rec = client.get(f"/workflow/{wid}").json()

    assert rec["status"] == "completed"  # W6: lỗi từng node không flip cả workflow
    assert rec["failed_steps"] == ["fe"]
    assert "fe: Ollama timeout" in rec["error"]
    assert env.extracted == ["be"], "output [LỖI ...] không được trích artifact"


def test_stream_exception_marks_failed(env):
    env.use_graph(_FakeGraph([_node("ba", "BA doc"), _node("pm", "PM doc")], raise_after=0))
    with TestClient(app_mod.app) as client:
        wid = _run(client)
        rec = client.get(f"/workflow/{wid}").json()
    assert rec["status"] == "failed"
    assert rec["error"] == "Ollama sập giữa chừng"
    assert rec["completed_steps"] == ["ba"] and rec["completed_at"]


def test_regen_loop_result_is_persisted(env, monkeypatch):
    env.use_graph(_FakeGraph([_node("be", "BE v1"), _node("clarifier", "CL v1")]))
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"] if text == "CL v1" else [])
    monkeypatch.setattr(app_mod, "run_single_step", lambda role, **kw: f"{role.upper()} v2")
    with TestClient(app_mod.app) as client:
        wid = _run(client)
        rec = client.get(f"/workflow/{wid}").json()
    assert rec["status"] == "completed"
    assert rec["step_outputs"] == {"be": "BE v2", "clarifier": "CLARIFIER v2"}


def test_unknown_workflow_is_404(env):
    with TestClient(app_mod.app) as client:
        assert client.get("/workflow/does-not-exist").status_code == 404
        assert client.get("/workflow/does-not-exist/artifacts").status_code == 404


# ── GET /workflows ─────────────────────────────────────────────────────────────


def _seed(wid, status="completed", created_at="2026-01-01T00:00:00+00:00", **extra):
    rec = app_mod.WorkflowRecord(
        workflow_id=wid,
        status=status,
        project=extra.pop("project", "crm"),
        user_input=f"input {wid}",
        created_at=created_at,
        completed_at=None if status in ("pending", "running") else created_at,
        **extra,
    )
    app_mod._store_workflow(rec)
    return rec


def test_workflows_list_order_and_shape(env):
    env.use_graph(_FakeGraph([_node("ba", "BA doc")]))
    with TestClient(app_mod.app) as client:
        _seed("old", created_at="2025-01-01T00:00:00+00:00", completed_steps=["ba", "pm", "sa"])
        _seed("mid", status="failed", created_at="2025-06-01T00:00:00+00:00", project=None, error="x")
        new_id = _run(client)
        data = client.get("/workflows").json()
        limited = client.get("/workflows", params={"limit": 2}).json()
        assert client.get("/workflows", params={"limit": 0}).status_code == 422
        assert client.get("/workflows", params={"limit": 501}).status_code == 422

    assert set(data) == {"count", "workflows"}
    assert data["count"] == 3 == len(data["workflows"])
    assert [w["workflow_id"] for w in data["workflows"]] == [new_id, "mid", "old"]
    for w in data["workflows"]:
        assert set(w) == SUMMARY_KEYS
        assert w["total_steps"] == len(app_mod.WORKFLOW_STEPS)
    by_id = {w["workflow_id"]: w for w in data["workflows"]}
    assert by_id["old"]["completed_steps"] == 3 and by_id["old"]["status"] == "completed"
    assert by_id["mid"]["project"] is None and by_id["mid"]["status"] == "failed"
    assert by_id[new_id]["completed_steps"] == 1 and by_id[new_id]["completed_at"]
    assert [w["workflow_id"] for w in limited["workflows"]] == [new_id, "mid"] and limited["count"] == 2


def test_workflows_default_limit_is_50(env):
    with TestClient(app_mod.app) as client:
        for i in range(55):
            _seed(f"wf-{i:02d}", created_at=f"2026-01-01T00:{i:02d}:00+00:00")
        data = client.get("/workflows").json()
    assert data["count"] == 50
    assert data["workflows"][0]["workflow_id"] == "wf-54"


# ── Crash recovery (lifespan) ──────────────────────────────────────────────────


def test_restart_marks_unfinished_runs_failed_and_leaves_finished_untouched(env):
    # "Process cũ": ghi thẳng vào file DB như thể container chết giữa chừng.
    old = WorkflowStore(env.db_path)
    old.init()
    for wid, status, err in [("p", "pending", None), ("r", "running", "fe: timeout"),
                             ("c", "completed", None), ("f", "failed", "boom")]:
        rec = app_mod.WorkflowRecord(
            workflow_id=wid, status=status, project="crm", user_input="x", error=err,
            completed_steps=["ba"], created_at=f"2026-01-01T00:00:0{len(wid)}+00:00",
            completed_at="2026-01-01T01:00:00+00:00" if status in ("completed", "failed") else None,
        )
        old.save(rec.model_dump(mode="json"))

    def raw():
        conn = sqlite3.connect(env.db_path)
        try:
            return {r[0]: r for r in conn.execute("SELECT * FROM workflows")}
        finally:
            conn.close()

    before = raw()
    with TestClient(app_mod.app) as client:  # lifespan → recovery
        p = client.get("/workflow/p").json()
        r = client.get("/workflow/r").json()
        c = client.get("/workflow/c").json()
        f = client.get("/workflow/f").json()
        listed = {w["workflow_id"]: w["status"] for w in client.get("/workflows").json()["workflows"]}
    after = raw()

    assert p["status"] == "failed" and p["error"] == INTERRUPTED_ERROR
    assert r["status"] == "failed" and r["error"] == INTERRUPTED_ERROR
    assert p["completed_at"] and r["completed_at"]
    assert (c["status"], c["error"]) == ("completed", None)
    assert (f["status"], f["error"]) == ("failed", "boom")
    assert after["c"] == before["c"] and after["f"] == before["f"]
    assert listed == {"p": "failed", "r": "failed", "c": "completed", "f": "failed"}


def test_recovery_runs_only_at_startup(env):
    with TestClient(app_mod.app) as client:
        # Run đang chạy của CHÍNH process này: request thường không được "khôi phục" nó.
        _seed("live", status="running")
        assert client.get("/workflow/live").json()["status"] == "running"
        assert client.get("/workflows").json()["workflows"][0]["status"] == "running"
        assert client.get("/workflow/live/artifacts").json()["artifacts"] == {}
        assert client.get("/workflow/live").json()["status"] == "running"
    # Startup kế tiếp (process mới) mới chuyển nó sang failed.
    with TestClient(app_mod.app) as client:
        rec = client.get("/workflow/live").json()
    assert (rec["status"], rec["error"]) == ("failed", INTERRUPTED_ERROR)


# ── Fallback dựng lại từ disk ──────────────────────────────────────────────────


def _write_artifact(env, wid, role, rel, content):
    p = env.tmp / "artifacts" / wid / role / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_reconstruct_without_completion_line_reports_interrupted(env):
    wid = "old-run-1"
    _write_artifact(env, wid, "fe", "_output.md", "FE output")
    _write_artifact(env, wid, "fe", "src/App.tsx", "export {}")
    with TestClient(app_mod.app) as client:
        rec = client.get(f"/workflow/{wid}").json()
        arts = client.get(f"/workflow/{wid}/artifacts").json()
    assert rec["status"] == "failed"
    assert rec["error"] == INTERRUPTED_ERROR
    assert rec["completed_at"] is None
    assert rec["step_outputs"] == {"fe": "FE output"}
    assert set(arts["artifacts"]) == {"fe"}
    # Kết quả dựng lại chỉ để đọc — không ghi ngược vào store.
    assert app_mod._get_store().get(wid) is None


def test_reconstruct_with_completion_line_keeps_logged_status(env):
    wid = "old-run-2"
    _write_artifact(env, wid, "be", "_output.md", "BE output")
    log = env.tmp / "memory" / "episodic" / "workflow_runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "workflow_id": wid, "project": "crm", "user_input": "x", "completed_steps": ["ba", "be"],
        "status": "completed", "error": None, "timestamp": "2026-01-01T00:00:00+00:00",
    }) + "\n", encoding="utf-8")
    with TestClient(app_mod.app) as client:
        rec = client.get(f"/workflow/{wid}").json()
    assert rec["status"] == "completed" and rec["error"] is None
    assert rec["completed_steps"] == ["ba", "be"] and rec["project"] == "crm"


def test_store_record_wins_over_disk(env):
    _seed("both", status="failed", error="from-db")
    _write_artifact(env, "both", "fe", "_output.md", "disk")
    with TestClient(app_mod.app) as client:
        rec = client.get("/workflow/both").json()
    assert rec["error"] == "from-db" and rec["step_outputs"] == {}


# ── Artifact endpoint ──────────────────────────────────────────────────────────


def test_artifact_endpoint_reads_file_and_blocks_traversal(env):
    wid = "wf-art"
    _write_artifact(env, wid, "fe", "src/App.tsx", "export const ok = 1")
    # File bí mật đặt đúng nơi mỗi URL tấn công sẽ trỏ tới NẾU không bị chặn — 404 mới
    # thực sự chứng minh confinement (không phải chỉ vì file không tồn tại).
    secret = env.tmp / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    (env.tmp / "artifacts" / wid / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    (env.tmp / "fe").mkdir()
    (env.tmp / "fe" / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")

    with TestClient(app_mod.app) as client:
        ok = client.get(f"/workflow/{wid}/artifacts/fe/src/App.tsx")
        assert ok.status_code == 200
        body = ok.json()
        assert (body["path"], body["content"]) == ("src/App.tsx", "export const ok = 1") and body["language"]
        dl = client.get(f"/workflow/{wid}/artifacts/fe/src/App.tsx", params={"download": 1})
        assert dl.status_code == 200 and dl.content == b"export const ok = 1"
        assert dl.headers["content-disposition"] == 'attachment; filename="App.tsx"'

        attacks = [
            f"/workflow/{wid}/artifacts/fe/..%2F..%2F..%2Fsecret.txt",
            f"/workflow/{wid}/artifacts/fe/%2E%2E/%2E%2E/%2E%2E/secret.txt",
            f"/workflow/{wid}/artifacts/fe/..%2Fsecret.txt",
            f"/workflow/{wid}/artifacts/%2E%2E/secret.txt",
            f"/workflow/{wid}/artifacts/notarole/secret.txt",
            "/workflow/%2E%2E/artifacts/fe/secret.txt",
            f"/workflow/{wid}/artifacts/fe/{secret}",
        ]
        for url in attacks:
            resp = client.get(url)
            assert resp.status_code == 404, (url, resp.status_code, resp.text)
            assert "TOP-SECRET" not in resp.text


# ── Static + UI ────────────────────────────────────────────────────────────────


def test_ui_routes_serve_workflow_html(env):
    html = (STATIC_DIR / "workflow.html").read_bytes()
    with TestClient(app_mod.app) as client:
        for url in ("/ui", "/ui/"):
            r = client.get(url)
            assert r.status_code == 200 and r.content == html
            assert r.headers["content-type"].startswith("text/html")


def test_static_mount_serves_every_file_under_static(env):
    files = sorted(p for p in STATIC_DIR.rglob("*") if p.is_file())
    assert files, "agent-api/static trống?"
    with TestClient(app_mod.app) as client:
        for f in files:
            rel = f.relative_to(STATIC_DIR).as_posix()
            r = client.get(f"/ui/static/{quote(rel)}")
            assert r.status_code == 200, rel
            assert r.content == f.read_bytes(), rel
        assert client.get("/ui/static/vendor/khong-ton-tai.js").status_code == 404
        for url in ("/ui/static/..%2Fapp.py", "/ui/static/%2E%2E/app.py", "/ui/static/%2E%2E%2Fstore.py"):
            r = client.get(url)
            assert r.status_code == 404, url
            assert b"FastAPI" not in r.content


def test_static_mount_is_registered_under_expected_name():
    mounts = {getattr(r, "name", None): getattr(r, "path", None) for r in app_mod.app.routes}
    assert mounts.get("ui-static") == "/ui/static"
    assert Path(app_mod.app.url_path_for("ui-static", path="vendor/x.js")) == Path("/ui/static/vendor/x.js")


def test_dockerfile_copies_static_tree():
    dockerfile = (ROOT / "agent-api" / "Dockerfile").read_text(encoding="utf-8")
    # `COPY . .` (build context = agent-api/) mang theo static/ kèm mọi thư mục con (vendor/).
    assert any(line.split() in (["COPY", ".", "."], ["COPY", "static", "static"],
                                ["COPY", "static/", "static/"], ["COPY", "static/", "./static/"])
               for line in dockerfile.splitlines())
    assert not (ROOT / "agent-api" / ".dockerignore").exists() or "static" not in (
        ROOT / "agent-api" / ".dockerignore"
    ).read_text(encoding="utf-8")


def test_lifespan_runs_model_check_then_store_recovery(env, monkeypatch):
    src = (ROOT / "agent-api" / "app.py").read_text(encoding="utf-8")
    assert "on_event(" not in src, "dùng lifespan thay cho @app.on_event (deprecated)"
    calls = []
    monkeypatch.setattr(app_mod, "_validate_models_on_startup", lambda: calls.append("models"))
    real_init = app_mod._init_store_and_recover
    monkeypatch.setattr(app_mod, "_init_store_and_recover", lambda: (calls.append("store"), real_init()))
    with TestClient(app_mod.app) as client:
        assert calls == ["models", "store"]
        assert client.get("/health").json()["status"] == "ok"
    assert env.db_path.exists()


def test_model_check_failure_is_not_fatal(env, monkeypatch):
    import requests

    def boom(*a, **kw):
        raise requests.ConnectionError("ollama chưa lên")

    monkeypatch.setattr(app_mod, "_validate_models_on_startup", _REAL_VALIDATE_MODELS)
    monkeypatch.setattr(requests, "get", boom)
    with TestClient(app_mod.app) as client:
        assert client.get("/health").status_code == 200
