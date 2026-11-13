"""Test hàng đợi workflow (C2: WORKFLOW_CONCURRENCY, queue_position) và hủy (C1: cancel).

agent-api/app.py trùng tên module `app` với rag-api → load bằng importlib dưới tên riêng
"agent_api_app" (dùng chung với tests/test_agent_app.py). LangGraph được thay bằng graph giả
có "cổng" (threading.Event) cho từng node: node chỉ trả về khi test mở cổng — mô phỏng LLM
call đang chạy mà không cần Ollama. Mọi thao tác chờ đều có trần WAIT_TIMEOUT → test hỏng
sẽ fail thay vì treo cả suite. DB/memory/artifact nằm trong tmp_path.
"""

import importlib.util
import json
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("workflow")
import artifacts  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WAIT_TIMEOUT = 10.0


def _load_agent_app():
    name = "agent_api_app"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "agent-api" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


app_mod = _load_agent_app()


def _node(role, output):
    return {role: {"step_outputs": {role: output}, "completed_steps": [role]}}


class _GatedGraph:
    """Graph giả: node (user_input, role) báo 'đã vào' rồi chờ test mở cổng mới trả về.

    Graph dùng chung cho mọi run; mỗi run phân biệt bằng state["user_input"].
    """

    def __init__(self, roles=("ba", "pm")):
        self.roles = tuple(roles)
        self.calls = []  # user_input theo thứ tự stream() được gọi
        self._lock = threading.Lock()
        self._entered = {}
        self._gates = {}

    def _event(self, table, key):
        with self._lock:
            return table.setdefault(key, threading.Event())

    def stream(self, state, stream_mode="updates"):
        name = state["user_input"]
        with self._lock:
            self.calls.append(name)
        for role in self.roles:
            self._event(self._entered, (name, role)).set()
            if not self._event(self._gates, (name, role)).wait(WAIT_TIMEOUT):
                raise AssertionError(f"cổng {name}/{role} không được mở sau {WAIT_TIMEOUT}s")
            yield _node(role, f"{role} của {name}")

    def entered(self, name, role="ba"):
        return self._event(self._entered, (name, role)).is_set()

    def wait_entered(self, name, role="ba"):
        return self._event(self._entered, (name, role)).wait(WAIT_TIMEOUT)

    def release(self, name, role):
        self._event(self._gates, (name, role)).set()

    def release_all(self, name):
        for role in self.roles:
            self.release(name, role)

    def release_everything(self):
        with self._lock:
            gates = list(self._gates.values())
        for gate in gates:
            gate.set()


class _HookGraph:
    """Graph giả không chặn: gọi hooks[i](state) TRƯỚC khi yield node thứ i (node 'đang chạy')
    và after[i](state) khi app đã xử lý xong node i (sau kiểm tra hủy/W1 của node đó)."""

    def __init__(self, roles=("ba", "pm"), before=None, after=None):
        self.roles = tuple(roles)
        self.before = before or {}
        self.after = after or {}
        self.calls = []

    def stream(self, state, stream_mode="updates"):
        self.calls.append(state["user_input"])
        for i, role in enumerate(self.roles):
            if i in self.before:
                self.before[i](state)
            yield _node(role, f"{role} của {state['user_input']}")
            if i in self.after:
                self.after[i](state)


@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    """Cô lập app + hàng đợi mới cho mỗi test; teardown mở mọi cổng rồi chờ worker thoát."""
    monkeypatch.setenv("WORKFLOW_DB_PATH", str(tmp_path / "db" / "workflows.sqlite3"))
    monkeypatch.setattr(app_mod, "_store", None)
    monkeypatch.setattr(app_mod, "_run_queue", None)
    monkeypatch.setattr(app_mod, "_WORKFLOW_CONCURRENCY", 1)
    monkeypatch.setattr(app_mod, "MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path / "artifacts"))
    monkeypatch.setattr(app_mod, "_validate_models_on_startup", lambda: None)
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 0)
    monkeypatch.setattr(app_mod, "_WORKFLOW_MAX_SECONDS", 0)
    monkeypatch.setattr(app_mod, "_extract_artifacts", lambda role, out, wid: [])

    graphs = []

    def use_graph(graph):
        graphs.append(graph)
        monkeypatch.setattr(app_mod, "get_workflow", lambda: graph)
        return graph

    def set_concurrency(n):
        monkeypatch.setattr(app_mod, "_WORKFLOW_CONCURRENCY", n)
        monkeypatch.setattr(app_mod, "_run_queue", None)

    e = SimpleNamespace(tmp=tmp_path, use_graph=use_graph, set_concurrency=set_concurrency)
    yield e
    for graph in graphs:
        if isinstance(graph, _GatedGraph):
            graph.release_everything()
    queue = app_mod._run_queue
    if queue is not None:
        assert queue.close(timeout=WAIT_TIMEOUT), "worker hàng đợi vẫn còn chạy sau test"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _post(client, user_input):
    r = client.post("/workflow/run", json={"user_input": user_input, "project": "crm"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending" and "queue_position" in body
    return body


def _get(client, wid):
    r = client.get(f"/workflow/{wid}")
    assert r.status_code == 200, r.text
    return r.json()


def _cancel(client, wid):
    return client.post(f"/workflow/{wid}/cancel")


def _wait_finished(wid, timeout=WAIT_TIMEOUT):
    """Chờ run rời hàng đợi: đã chốt trạng thái, ghi DB, ghi JSONL và trả slot."""
    deadline = time.monotonic() + timeout
    queue = app_mod._get_run_queue()
    while queue.state(wid) is not None:
        assert time.monotonic() < deadline, f"workflow {wid} chưa kết thúc sau {timeout}s"
        time.sleep(0.005)


def _episodic(env):
    path = env.tmp / "memory" / "episodic" / "workflow_runs.jsonl"
    if not path.exists():
        return {}
    return {
        e["workflow_id"]: e
        for e in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _seed(wid, status, **extra):
    rec = app_mod.WorkflowRecord(
        workflow_id=wid,
        status=status,
        project="crm",
        user_input=f"input {wid}",
        created_at="2026-01-01T00:00:00+00:00",
        completed_at=None if status in ("pending", "running") else "2026-01-01T01:00:00+00:00",
        **extra,
    )
    app_mod._store_workflow(rec)
    return rec


# ── C2: hàng đợi ───────────────────────────────────────────────────────────────


def test_concurrency_1_queues_fifo_and_keeps_queue_position_accurate(qenv):
    graph = qenv.use_graph(_GatedGraph(("ba", "pm")))
    with TestClient(app_mod.app) as client:
        a = _post(client, "A")
        assert a["queue_position"] is None  # slot rảnh → chạy ngay, không xếp hàng
        assert graph.wait_entered("A", "ba")
        b = _post(client, "B")
        c = _post(client, "C")
        assert (b["queue_position"], c["queue_position"]) == (1, 2)
        A, B, C = a["workflow_id"], b["workflow_id"], c["workflow_id"]

        ra, rb, rc = _get(client, A), _get(client, B), _get(client, C)
        assert (ra["status"], ra["queue_position"]) == ("running", None)
        assert (rb["status"], rb["queue_position"]) == ("pending", 1)
        assert (rc["status"], rc["queue_position"]) == ("pending", 2)
        listed = {w["workflow_id"]: w["status"] for w in client.get("/workflows").json()["workflows"]}
        assert listed == {A: "running", B: "pending", C: "pending"}

        # A sang node thứ hai — B vẫn chờ, chưa ai khác được chạy.
        graph.release("A", "ba")
        assert graph.wait_entered("A", "pm")
        assert (_get(client, B)["status"], _get(client, B)["queue_position"]) == ("pending", 1)
        assert graph.calls == ["A"]

        # A xong → B nhận slot, C lên đầu hàng.
        graph.release_all("A")
        assert graph.wait_entered("B", "ba")
        _wait_finished(A)
        assert _get(client, A)["status"] == "completed"
        rb, rc = _get(client, B), _get(client, C)
        assert (rb["status"], rb["queue_position"]) == ("running", None)
        assert (rc["status"], rc["queue_position"]) == ("pending", 1)
        assert graph.calls == ["A", "B"]

        graph.release_all("B")
        graph.release_all("C")
        for wid in (B, C):
            _wait_finished(wid)
        final = [_get(client, wid) for wid in (A, B, C)]

    assert graph.calls == ["A", "B", "C"]
    for rec in final:
        assert rec["status"] == "completed" and rec["completed_steps"] == ["ba", "pm"]
        assert rec["queue_position"] is None and rec["cancel_requested"] is False and rec["error"] is None


def test_concurrency_2_runs_two_workflows_at_once(qenv):
    qenv.set_concurrency(2)
    graph = qenv.use_graph(_GatedGraph(("ba",)))
    with TestClient(app_mod.app) as client:
        a, b, c = _post(client, "A"), _post(client, "B"), _post(client, "C")
        # Cả A và B đang ở trong node (cổng còn đóng) cùng lúc; C xếp hàng vị trí 1.
        assert graph.wait_entered("A") and graph.wait_entered("B")
        assert (a["queue_position"], b["queue_position"], c["queue_position"]) == (None, None, 1)
        assert _get(client, a["workflow_id"])["status"] == "running"
        assert _get(client, b["workflow_id"])["status"] == "running"
        rc = _get(client, c["workflow_id"])
        assert (rc["status"], rc["queue_position"]) == ("pending", 1)
        assert not graph.entered("C")

        graph.release_all("B")  # B xong trước → C lấy slot của B dù A vẫn chạy
        assert graph.wait_entered("C")
        assert not graph.entered("A", "pm") and _get(client, a["workflow_id"])["status"] == "running"
        graph.release_all("A")
        graph.release_all("C")
        for body in (a, b, c):
            _wait_finished(body["workflow_id"])
            assert _get(client, body["workflow_id"])["status"] == "completed"
    assert sorted(graph.calls[:2]) == ["A", "B"] and graph.calls[2] == "C"


def test_workflow_concurrency_env_parsing(monkeypatch):
    cases = {None: 1, "": 1, "  ": 1, "abc": 1, "1.5": 1, "0": 1, "-3": 1, "1": 1, "2": 2, " 3 ": 3}
    for raw, expected in cases.items():
        if raw is None:
            monkeypatch.delenv("WORKFLOW_CONCURRENCY", raising=False)
        else:
            monkeypatch.setenv("WORKFLOW_CONCURRENCY", raw)
        assert app_mod._concurrency_from_env() == expected, raw


def test_worker_survives_runner_crash_and_bad_record(qenv, monkeypatch):
    graph = qenv.use_graph(_HookGraph(("ba",)))
    real_runner = app_mod._run_workflow_task
    real_load = app_mod._load_workflow
    seen = []

    def flaky_runner(wid, req, queue):
        seen.append(req.user_input)
        if req.user_input == "crash":
            raise RuntimeError("lỗi bất ngờ ngoài try của runner")
        return real_runner(wid, req, queue)

    def load(wid):
        rec = real_load(wid)
        if rec is not None and rec.user_input == "mất record" and threading.current_thread().name.startswith(
            "workflow-worker"
        ):
            return None
        return rec

    monkeypatch.setattr(app_mod, "_run_workflow_task", flaky_runner)
    monkeypatch.setattr(app_mod, "_load_workflow", load)
    with TestClient(app_mod.app) as client:
        ids = [_post(client, name)["workflow_id"] for name in ("crash", "mất record", "ok")]
        for wid in ids:
            _wait_finished(wid)
        ok = _get(client, ids[2])
    # Worker duy nhất (concurrency 1) vẫn sống và slot được trả sau cả hai sự cố.
    assert seen == ["crash", "mất record", "ok"]
    assert graph.calls == ["ok"] and ok["status"] == "completed"


# ── C1: hủy ────────────────────────────────────────────────────────────────────


def test_cancel_pending_run_is_immediate_and_never_takes_a_slot(qenv):
    graph = qenv.use_graph(_GatedGraph(("ba",)))
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        assert graph.wait_entered("A")
        B = _post(client, "B")["workflow_id"]
        C = _post(client, "C")["workflow_id"]

        r = _cancel(client, B)
        assert r.status_code == 200 and r.json() == {"workflow_id": B, "status": "cancelled"}
        rb = _get(client, B)
        assert rb["status"] == "cancelled" and rb["error"] == app_mod.CANCELLED_ERROR
        assert rb["completed_at"] and rb["queue_position"] is None
        assert rb["completed_steps"] == [] and rb["step_outputs"] == {}
        assert app_mod._get_run_queue().state(B) is None
        # B không còn được tính trong hàng: C lên vị trí 1.
        assert _get(client, C)["queue_position"] == 1

        # A xong → C (không phải B) nhận slot ngay.
        graph.release_all("A")
        assert graph.wait_entered("C")
        graph.release_all("C")
        for wid in (A, C):
            _wait_finished(wid)
        assert _get(client, C)["status"] == "completed"
        assert _get(client, B)["status"] == "cancelled"
        again = _cancel(client, B)
        assert again.status_code == 409 and "cancelled" in again.json()["detail"]

    assert graph.calls == ["A", "C"], "run bị hủy khi đang chờ không bao giờ được chạy"
    log = _episodic(qenv)
    assert log[B]["status"] == "cancelled" and log[B]["completed_steps"] == []


def test_cancel_running_stops_after_current_node(qenv):
    graph = qenv.use_graph(_GatedGraph(("ba", "pm", "sa")))
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        assert graph.wait_entered("A", "ba")
        B = _post(client, "B")["workflow_id"]

        r = _cancel(client, A)
        assert r.status_code == 200 and r.json() == {"workflow_id": A, "status": "running"}
        ra = _get(client, A)
        assert ra["status"] == "running" and ra["cancel_requested"] is True
        # Hủy lặp lại khi chưa kết thúc vẫn được chấp nhận (không phải 409).
        assert _cancel(client, A).json() == {"workflow_id": A, "status": "running"}

        # Node hiện tại (LLM call) chạy nốt rồi workflow mới dừng.
        graph.release("A", "ba")
        _wait_finished(A)
        ra = _get(client, A)
        assert not graph.entered("A", "pm"), "node kế tiếp không được chạy sau khi hủy"
        assert ra["status"] == "cancelled" and ra["cancel_requested"] is True
        assert ra["completed_steps"] == ["ba"] and ra["step_outputs"] == {"ba": "ba của A"}
        assert ra["error"] == app_mod.CANCELLED_ERROR and ra["completed_at"]
        assert _cancel(client, A).status_code == 409

        # Slot được trả → B chạy.
        assert graph.wait_entered("B", "ba")
        graph.release_all("B")
        _wait_finished(B)
        assert _get(client, B)["status"] == "completed"
    assert _episodic(qenv)[A]["status"] == "cancelled"


def test_cancel_requested_right_after_slot_runs_no_node(qenv, monkeypatch):
    graph = qenv.use_graph(_HookGraph(("ba",)))
    real_load = app_mod._load_workflow
    responses = []

    def load(wid):
        rec = real_load(wid)
        if threading.current_thread().name.startswith("workflow-worker"):
            # Worker vừa nhận slot, chưa chạy node nào — yêu cầu hủy đến đúng lúc này.
            responses.append(app_mod.cancel_workflow(wid))
        return rec

    monkeypatch.setattr(app_mod, "_load_workflow", load)
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert responses == [{"workflow_id": A, "status": "running"}]
    assert graph.calls == []
    assert ra["status"] == "cancelled" and ra["completed_steps"] == [] and ra["error"] == app_mod.CANCELLED_ERROR


def test_cancel_unknown_is_404_and_finished_is_409(qenv):
    with TestClient(app_mod.app) as client:
        r = _cancel(client, "khong-ton-tai")
        assert r.status_code == 404
        for wid, status in [("c", "completed"), ("f", "failed"), ("x", "cancelled")]:
            _seed(wid, status, error=None if status == "completed" else "lỗi cũ")
            r = _cancel(client, wid)
            assert r.status_code == 409, (wid, r.text)
            assert status in r.json()["detail"]
            assert _get(client, wid)["status"] == status  # không bị đổi
        # Run cũ chỉ còn trên disk (W2) luôn là đã kết thúc → 409, không phải 404.
        p = qenv.tmp / "artifacts" / "disk-only" / "fe" / "_output.md"
        p.parent.mkdir(parents=True)
        p.write_text("FE", encoding="utf-8")
        assert _cancel(client, "disk-only").status_code == 409


def test_cancel_orphan_active_record_without_worker(qenv):
    # Record pending/running mà không worker nào giữ (vd. lần ghi trạng thái cuối thất bại):
    # không có gì đang chạy → hủy ngay.
    with TestClient(app_mod.app) as client:
        _seed("orphan", "running", completed_steps=["ba"])
        r = _cancel(client, "orphan")
        assert r.status_code == 200 and r.json() == {"workflow_id": "orphan", "status": "cancelled"}
        rec = _get(client, "orphan")
    assert rec["status"] == "cancelled" and rec["completed_steps"] == ["ba"] and rec["completed_at"]


# ── C1: race giữa hủy và kết thúc ─────────────────────────────────────────────


def _only_workflow_id():
    [latest] = app_mod._get_store().list_recent(1)
    return latest["workflow_id"]


def test_cancel_accepted_during_last_node_always_ends_cancelled(qenv):
    responses = []
    # after[last]: app đã xử lý node cuối VÀ đã qua kiểm tra hủy của node đó; stream sắp xong.
    graph = qenv.use_graph(_HookGraph(
        ("ba", "pm"), after={1: lambda state: responses.append(app_mod.cancel_workflow(_only_workflow_id()))},
    ))
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert responses == [{"workflow_id": A, "status": "running"}]
    assert graph.calls == ["A"]
    # Hủy được chấp nhận (200) trước khi trạng thái cuối được chốt ⇒ cancelled, dù mọi node đã xong.
    assert ra["status"] == "cancelled" and ra["completed_steps"] == ["ba", "pm"]
    assert ra["cancel_requested"] is True and ra["error"] == app_mod.CANCELLED_ERROR


def test_cancel_after_final_status_decided_is_409_and_stays_completed(qenv, monkeypatch):
    qenv.use_graph(_HookGraph(("ba",)))
    real_log = app_mod._log_workflow_run
    seen = []

    def log_then_cancel(**kw):
        # Worker đã chốt + ghi DB trạng thái cuối nhưng chưa trả slot ("finishing").
        if threading.current_thread().name.startswith("workflow-worker"):
            wid = kw["workflow_id"]
            seen.append(app_mod._get_run_queue().state(wid))
            with pytest.raises(HTTPException) as exc:
                app_mod.cancel_workflow(wid)
            seen.append(exc.value.status_code)
        return real_log(**kw)

    monkeypatch.setattr(app_mod, "_log_workflow_run", log_then_cancel)
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
        assert _cancel(client, A).status_code == 409  # sau khi trả slot: vẫn 409 (đọc từ DB)
    assert seen == ["finishing", 409]
    assert ra["status"] == "completed" and ra["cancel_requested"] is False and ra["error"] is None


def test_cancel_racing_completion_is_always_consistent(qenv):
    """Hủy ở thời điểm ngẫu nhiên: 200 'cancelled' ⇒ chưa từng chạy; 200 'running' ⇒ kết
    thúc cancelled; 409 ⇒ completed nguyên vẹn. Không run nào kẹt, không slot nào bị rò."""
    graph = qenv.use_graph(_HookGraph(("ba", "pm")))
    rng = random.Random(1234)
    n = 30
    with TestClient(app_mod.app):
        reqs = [app_mod.WorkflowRunRequest(user_input=f"run-{i}") for i in range(n)]
        ids = [app_mod.start_workflow(req)["workflow_id"] for req in reqs]
        outcomes = {}

        def cancel_later(wid, delay):
            time.sleep(delay)
            try:
                outcomes[wid] = app_mod.cancel_workflow(wid)["status"]
            except HTTPException as exc:
                outcomes[wid] = exc.status_code

        threads = [threading.Thread(target=cancel_later, args=(wid, rng.uniform(0, 0.02))) for wid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT_TIMEOUT)
            assert not t.is_alive()
        for wid in ids:
            _wait_finished(wid)
        records = {wid: app_mod._get_store().get(wid) for wid in ids}
        queue = app_mod._get_run_queue()

    ran = set(graph.calls)
    for i, wid in enumerate(ids):
        rec, outcome = records[wid], outcomes[wid]
        if outcome == "cancelled":
            assert rec["status"] == "cancelled" and f"run-{i}" not in ran
        elif outcome == "running":
            assert rec["status"] == "cancelled" and rec["cancel_requested"] is True
        else:
            assert outcome == 409
            assert rec["status"] == "completed" and rec["cancel_requested"] is False
            assert rec["completed_steps"] == ["ba", "pm"]
    assert not queue._waiting and not queue._active


# ── C1: hủy tôn trọng W1 và Clarifier regen loop ──────────────────────────────


def test_cancel_takes_precedence_over_w1_budget(qenv, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(app_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(app_mod, "_WORKFLOW_MAX_SECONDS", 1)

    def during_ba(state):
        # Trong lúc node ba chạy: vừa vượt ngân sách vừa có yêu cầu hủy.
        clock[0] = 100.0
        app_mod.cancel_workflow(_only_workflow_id())

    graph = qenv.use_graph(_HookGraph(("ba", "pm"), before={0: during_ba}))
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert graph.calls == ["A"]
    assert ra["status"] == "cancelled" and ra["completed_steps"] == ["ba"]
    assert ra["error"] == app_mod.CANCELLED_ERROR  # không lẫn "[Timeout ...]"


def test_w1_budget_still_stops_without_cancel(qenv, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(app_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(app_mod, "_WORKFLOW_MAX_SECONDS", 1)
    qenv.use_graph(_HookGraph(("ba", "pm"), before={0: lambda state: clock.__setitem__(0, 100.0)}))
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert ra["status"] == "completed" and ra["completed_steps"] == ["ba"]
    assert "[Timeout" in ra["error"] and ra["cancel_requested"] is False


def test_regen_loop_honours_cancel(qenv, monkeypatch):
    qenv.use_graph(_HookGraph(("be", "clarifier")))
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 2)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be", "fe"])
    regen_calls = []

    def regen(role, **kw):
        regen_calls.append(role)
        if role == "be":
            # Hủy đến trong lúc agent đầu tiên của vòng regen đang chạy.
            assert app_mod.cancel_workflow(_only_workflow_id())["status"] == "running"
        return f"{role.upper()} v2"

    monkeypatch.setattr(app_mod, "run_single_step", regen)
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    # Agent đang chạy hoàn tất (output được giữ), không agent/clarifier nào chạy tiếp.
    assert regen_calls == ["be"]
    assert ra["status"] == "cancelled" and ra["error"] == app_mod.CANCELLED_ERROR
    assert ra["step_outputs"]["be"] == "BE v2" and ra["step_outputs"]["clarifier"] == "clarifier của A"


# ── Regen: lỗi không ghi đè output cũ; thành công gỡ khỏi failed_steps ─────────


def test_regen_error_output_keeps_previous_output(qenv, monkeypatch):
    qenv.use_graph(_HookGraph(("be", "clarifier")))
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"])

    def regen(role, **kw):
        if role == "be":
            return "[LỖI trong be] Không sinh được file nào (1 file lên kế hoạch đều thất bại)."
        return "CLARIFIER v2"

    monkeypatch.setattr(app_mod, "run_single_step", regen)
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert ra["status"] == "completed"
    assert ra["step_outputs"]["be"] == "be của A"          # bản cũ còn nguyên
    assert ra["step_outputs"]["clarifier"] == "CLARIFIER v2"
    assert "be" not in ra.get("failed_steps", [])


def test_regen_success_clears_failed_step(qenv, monkeypatch):
    class _FailingBe(_HookGraph):
        def stream(self, state, stream_mode="updates"):
            self.calls.append(state["user_input"])
            yield {"be": {"step_outputs": {"be": "[LỖI trong be] boom"}, "completed_steps": ["be"],
                          "error": "Bước 'be' thất bại: boom"}}
            yield _node("clarifier", "clarifier của A")

    qenv.use_graph(_FailingBe(("be", "clarifier")))
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"])
    monkeypatch.setattr(app_mod, "run_single_step", lambda role, **kw: f"{role.upper()} v2")
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert ra["step_outputs"]["be"] == "BE v2"
    assert "be" not in ra["failed_steps"]
    assert ra["error"] is None                              # không còn tóm tắt lỗi cũ


def test_regen_success_keeps_other_step_errors(qenv, monkeypatch):
    class _TwoFailing(_HookGraph):
        def stream(self, state, stream_mode="updates"):
            self.calls.append(state["user_input"])
            for role in ("be", "fe"):
                yield {role: {"step_outputs": {role: f"[LỖI trong {role}] boom"},
                              "completed_steps": [role], "error": f"Bước '{role}' thất bại: boom"}}
            yield _node("clarifier", "clarifier của A")

    qenv.use_graph(_TwoFailing(("be", "fe", "clarifier")))
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"])
    monkeypatch.setattr(app_mod, "run_single_step", lambda role, **kw: f"{role.upper()} v2")
    with TestClient(app_mod.app) as client:
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
    assert ra["failed_steps"] == ["fe"]
    assert ra["error"] == "Bước 'fe' thất bại: boom"


# ── Ghi trạng thái cuối có thử lại ─────────────────────────────────────────────


def test_final_persist_retries_transient_store_error(qenv, monkeypatch):
    qenv.use_graph(_HookGraph(("ba",)))
    monkeypatch.setattr(app_mod, "_FINAL_PERSIST_DELAYS", (0.0, 0.0))
    real_store = app_mod._store_workflow
    failures = {"left": 0}

    def flaky_store(rec):
        if rec.status.value in ("completed", "failed", "cancelled") and failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("database is locked")
        return real_store(rec)

    monkeypatch.setattr(app_mod, "_store_workflow", flaky_store)
    with TestClient(app_mod.app) as client:
        failures["left"] = 2                               # 2 lần đầu lỗi, lần 3 ghi được
        A = _post(client, "A")["workflow_id"]
        _wait_finished(A)
        ra = _get(client, A)
        assert ra["status"] == "completed" and ra["completed_at"]
        # Đã ghi xong trạng thái cuối → cancel phải 409, không đi nhánh "record mồ côi".
        assert _cancel(client, A).status_code == 409
    assert failures["left"] == 0


def test_final_persist_gives_up_without_raising(qenv, monkeypatch):
    monkeypatch.setattr(app_mod, "_FINAL_PERSIST_DELAYS", (0.0,))
    calls = []

    def always_fail(rec):
        calls.append(rec.status.value)
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_mod, "_store_workflow", always_fail)
    rec = app_mod.WorkflowRecord(
        workflow_id="x", status="completed", project="crm", user_input="u",
        created_at="2026-01-01T00:00:00+00:00",
    )
    app_mod._persist_final(rec)                            # không raise
    assert calls == ["completed", "completed"]
