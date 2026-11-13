"""Test đối kháng cho hàng đợi workflow (C2) và hủy (C1) của agent-api/app.py.

Bổ sung cho tests/test_workflow_queue.py — tấn công trực tiếp các bất biến:
- run bị hủy khi đang chờ không bao giờ nhận slot và không chạy node nào;
- WORKFLOW_CONCURRENCY=N ⇒ tối đa N run chạy node cùng lúc, kể cả khi nhiều thread
  submit + hủy đồng thời (stress, timing ngẫu nhiên, nhiều vòng);
- queue_position chính xác: snapshot lấy DƯỚI lock hàng đợi (nhiều GET rời rạc không phải
  một snapshot nhất quán) luôn là 1..k liên tục theo thứ tự FIFO;
- mỗi run kết thúc đúng MỘT trạng thái cuối (DB + đúng một dòng JSONL), completed/failed
  không bao giờ bị lật thành cancelled;
- recovery lúc startup chỉ đụng pending/running (kể cả dòng cancelled hỏng JSON);
- slot luôn được trả (node raise, hủy giữa chừng, vượt ngân sách W1, regen loop) và run
  thoát bất thường (không đọc được record, BaseException) vẫn chốt trạng thái cuối thay vì
  kẹt pending/running mãi mãi.

Một test dùng LangGraph THẬT (StateGraph tuyến tính trên SDLCState) để chứng minh node chạy
inline trong thread worker và stream.close() sau khi hủy không để node kế tiếp chạy.

agent-api/app.py trùng tên module `app` với rag-api → load bằng importlib dưới tên
"agent_api_app" (dùng chung với các file test khác). Mọi thao tác chờ có trần WAIT_TIMEOUT;
DB/memory/artifact nằm trong tmp_path.
"""

import importlib.util
import itertools
import json
import random
import sqlite3
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
from store import INTERRUPTED_ERROR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WAIT_TIMEOUT = 10.0
ROLES = ("ba", "pm", "sa")


def _load_agent_app():
    name = "agent_api_app"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "agent-api" / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


app_mod = _load_agent_app()
_SEQ = itertools.count(1)  # thứ tự toàn cục của sự kiện (node bắt đầu / hủy trả về)


def _seq():
    return next(_SEQ)


def _node(role, output):
    return {role: {"step_outputs": {role: output}, "completed_steps": [role]}}


def _event(table, lock, key):
    with lock:
        return table.setdefault(key, threading.Event())


class _Abort(BaseException):
    """BaseException KHÔNG phải Exception — lọt qua `except Exception` của runner."""


class _TrackingGraph:
    """Graph giả đếm số node đang chạy đồng thời trên MỌI run (max_active).

    "Node đang chạy" = từ lúc node bắt đầu tới lúc chunk của nó được yield (giống LangGraph:
    node chạy xong mới phát chunk). *behaviour* (theo user_input) cho phép một node raise
    hoặc chặn ở cổng (gates) — mặc định node ngủ ngẫu nhiên 0..max_sleep giây.
    """

    def __init__(self, roles=ROLES, max_sleep=0.0005, seed=0, behaviour=None):
        self.roles = tuple(roles)
        self.max_sleep = max_sleep
        self.behaviour = behaviour or {}
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.entries = []  # (seq, user_input, role, thread_name)
        self._entered = {}
        self._gates = {}

    def entered(self, name, role):
        return _event(self._entered, self._lock, (name, role))

    def gate(self, name, role):
        return _event(self._gates, self._lock, (name, role))

    def release_everything(self):
        with self._lock:
            gates = list(self._gates.values())
        for gate in gates:
            gate.set()

    def ran(self, name):
        with self._lock:
            return [e for e in self.entries if e[1] == name]

    def stream(self, state, stream_mode="updates"):
        name = state["user_input"]
        for role in self.roles:
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.entries.append((_seq(), name, role, threading.current_thread().name))
                delay = self._rng.uniform(0, self.max_sleep)
            try:
                self.entered(name, role).set()
                action = self.behaviour.get((name, role))
                if action == "gate":
                    if not self.gate(name, role).wait(WAIT_TIMEOUT):
                        raise AssertionError(f"cổng {name}/{role} không được mở")
                elif action is not None:
                    raise action
                elif delay:
                    time.sleep(delay)
            finally:
                with self._lock:
                    self.active -= 1
            yield _node(role, f"{role} của {name}")


@pytest.fixture()
def aenv(tmp_path, monkeypatch):
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

    e = SimpleNamespace(tmp=tmp_path, use_graph=use_graph, set_concurrency=set_concurrency, monkeypatch=monkeypatch)
    yield e
    for graph in graphs:
        release = getattr(graph, "release_everything", None)
        if release:
            release()
    queue = app_mod._run_queue
    if queue is not None:
        assert queue.close(timeout=WAIT_TIMEOUT), "worker hàng đợi vẫn còn chạy sau test"


# ── Helpers ────────────────────────────────────────────────────────────────────


def _submit(name):
    return app_mod.start_workflow(app_mod.WorkflowRunRequest(user_input=name))


def _cancel(wid):
    """Gọi endpoint hủy → (kết quả, seq). Kết quả là status hoặc mã lỗi HTTP.

    Seq được lấy NGAY sau quyết định, vẫn dưới lock hàng đợi (endpoint quyết định dưới chính
    lock này — re-entrant) → thứ tự seq giữa các lần hủy và node bắt đầu khớp thứ tự thật.
    """
    with app_mod._get_run_queue().lock:
        try:
            result = app_mod.cancel_workflow(wid)["status"]
        except HTTPException as exc:
            result = exc.status_code
        return result, _seq()


def _wait_finished(wid, timeout=WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    queue = app_mod._get_run_queue()
    while queue.state(wid) is not None:
        assert time.monotonic() < deadline, f"workflow {wid} chưa kết thúc sau {timeout}s"
        time.sleep(0.002)


def _wait_until(pred, what, timeout=WAIT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, f"hết {timeout}s chờ: {what}"
        time.sleep(0.002)


def _episodic_lines(env):
    path = env.tmp / "memory" / "episodic" / "workflow_runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _db(wid):
    return app_mod._get_store().get(wid)


def _raw_rows(env):
    conn = sqlite3.connect(env.tmp / "db" / "workflows.sqlite3")
    try:
        return {r[0]: r for r in conn.execute("SELECT * FROM workflows")}
    finally:
        conn.close()


def _record_runner_calls(env):
    """Bọc runner: ghi lại workflow_id mỗi lần worker GIAO slot cho một run."""
    real = app_mod._run_workflow_task
    calls = []
    lock = threading.Lock()

    def runner(wid, req, queue):
        with lock:
            calls.append(wid)
        return real(wid, req, queue)

    env.monkeypatch.setattr(app_mod, "_run_workflow_task", runner)
    return calls


def _check_queue_snapshot(queue, concurrency, submit_order, violations, sample):
    """Một snapshot DƯỚI lock hàng đợi: slot không vượt N, queue_position = 1..k liên tục
    theo FIFO, và GET /workflow/{id} trả đúng vị trí đó cho run đang chờ."""
    with queue.lock:
        waiting = [wid for wid, _ in queue._waiting]
        active = dict(queue._active)
        if len(active) > concurrency:
            violations.append(f"{len(active)} run giữ slot > {concurrency}")
        # FIFO: thứ tự trong hàng = thứ tự submit (submit_order chỉ được ghi dưới lock này).
        order = {wid: i for i, wid in enumerate(list(submit_order))}
        if [order[w] for w in waiting] != sorted(order[w] for w in waiting):
            violations.append("hàng đợi không theo thứ tự submit")
        positions = [queue.live_state(w)[0] for w in waiting]
        free = max(0, concurrency - len(active))
        # free run đầu hàng sắp nhận slot (None); phần còn lại xếp hàng 1, 2, 3, ...
        expected = [None] * min(free, len(waiting)) + list(range(1, len(waiting) - free + 1))
        if positions != expected:
            violations.append(f"queue_position {positions} != {expected}")
        for wid in set(active) & set(waiting):
            violations.append(f"{wid} vừa chờ vừa giữ slot")
        for i in sample:
            if i < len(waiting):
                rec = app_mod.get_workflow_status(waiting[i])
                if rec.status != app_mod.WorkflowStatus.pending or rec.queue_position != positions[i]:
                    violations.append(
                        f"GET {waiting[i]}: {rec.status.value}/{rec.queue_position} != pending/{positions[i]}"
                    )
        for wid in active:
            if queue.live_state(wid)[0] is not None:
                violations.append(f"run đang giữ slot {wid} vẫn có queue_position")


# ── Stress: submit + hủy đồng thời ────────────────────────────────────────────


@pytest.mark.parametrize("concurrency,seed", [(1, 11), (1, 12), (2, 21), (3, 31)])
def test_stress_concurrent_submits_and_cancels_keep_every_invariant(aenv, concurrency, seed):
    aenv.set_concurrency(concurrency)
    graph = aenv.use_graph(_TrackingGraph(seed=seed, max_sleep=0.001))
    runner_calls = _record_runner_calls(aenv)
    queue = app_mod._get_run_queue()

    # Ghi thứ tự submit DƯỚI lock hàng đợi (start_workflow gọi submit trong `with queue.lock`).
    submit_order = []
    real_submit = queue.submit

    def recording_submit(wid, req):
        pos = real_submit(wid, req)
        submit_order.append(wid)
        return pos

    queue.submit = recording_submit

    n_submitters, per_submitter = 5, 8
    names = {}  # wid -> user_input
    outcomes = {}  # wid -> [(kết quả, seq)]
    plans = []  # (thời điểm hủy, wid)
    lock = threading.Lock()
    violations = []
    submitting_done = threading.Event()
    stop_checker = threading.Event()
    errors = []

    def submitter(k):
        local = random.Random(seed * 100 + k)
        try:
            for i in range(per_submitter):
                name = f"s{k}-{i}"
                body = _submit(name)
                with lock:
                    names[body["workflow_id"]] = name
                    if local.random() < 0.4:
                        plans.append((time.monotonic() + local.uniform(0, 0.03), body["workflow_id"]))
                    if local.random() < 0.15:  # hủy lần hai (sau đó) — phải nhất quán với lần đầu
                        plans.append((time.monotonic() + local.uniform(0.01, 0.06), body["workflow_id"]))
                time.sleep(local.uniform(0, 0.002))
        except BaseException as exc:  # pragma: no cover - báo lỗi thread
            errors.append(exc)

    def canceller():
        try:
            while True:
                with lock:
                    due = sorted(p for p in plans if p[0] <= time.monotonic())
                    if due:
                        plans.remove(due[0])
                    pending_plans = bool(plans)
                if due:
                    wid = due[0][1]
                    result = _cancel(wid)
                    with lock:
                        outcomes.setdefault(wid, []).append(result)
                elif submitting_done.is_set() and not pending_plans:
                    return
                else:
                    time.sleep(0.0005)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def sniper():
        # Nhắm vào run ĐANG giữ slot: hủy ở thời điểm ngẫu nhiên (giữa node, giữa hai node,
        # đúng lúc chốt trạng thái cuối, hoặc vừa xong → 409) — đường "running" và race chốt.
        local = random.Random(seed + 3)
        seen = set()
        try:
            while not stop_checker.is_set():
                with queue.lock:
                    fresh = [w for w in queue._active if w not in seen]
                seen.update(fresh)
                for wid in fresh:
                    if local.random() < 0.5:
                        time.sleep(local.uniform(0, 0.004))
                        result = _cancel(wid)
                        with lock:
                            outcomes.setdefault(wid, []).append(result)
                time.sleep(0.0002)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def checker():
        local = random.Random(seed + 7)
        try:
            while not stop_checker.is_set():
                _check_queue_snapshot(queue, concurrency, submit_order, violations,
                                      sample=local.sample(range(6), 2))
                time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    submitters = [threading.Thread(target=submitter, args=(k,)) for k in range(n_submitters)]
    cancellers = [threading.Thread(target=canceller) for _ in range(3)]
    check = threading.Thread(target=checker)
    snipe = threading.Thread(target=sniper)
    for t in [check, snipe, *submitters, *cancellers]:
        t.start()
    for t in submitters:
        t.join(WAIT_TIMEOUT)
    submitting_done.set()
    for t in cancellers:
        t.join(WAIT_TIMEOUT)
    ids = list(names)
    for wid in ids:
        _wait_finished(wid)
    stop_checker.set()
    check.join(WAIT_TIMEOUT)
    snipe.join(WAIT_TIMEOUT)
    assert not any(t.is_alive() for t in [check, snipe, *submitters, *cancellers])
    assert not errors, errors
    assert not violations, violations[:5]
    assert len(ids) == n_submitters * per_submitter == len(submit_order)

    # ≤ N run chạy node cùng lúc; mỗi run chạy trong thread worker của hàng đợi.
    assert graph.max_active <= concurrency
    assert all(e[3].startswith("workflow-worker-") for e in graph.entries)
    assert not queue._waiting and not queue._active

    log = _episodic_lines(aenv)
    log_by_id = {}
    for entry in log:
        log_by_id.setdefault(entry["workflow_id"], []).append(entry)

    for wid in ids:
        name, rec = names[wid], _db(wid)
        results = outcomes.get(wid, [])
        accepted = [r for r in results if r[0] in ("cancelled", "running")]
        # Đúng MỘT trạng thái cuối: DB kết thúc + đúng một dòng JSONL khớp DB.
        assert rec["status"] in ("completed", "cancelled"), (name, rec["status"])
        assert rec["completed_at"] and rec["queue_position"] is None
        assert [e["status"] for e in log_by_id.get(wid, [])] == [rec["status"]], name
        assert 404 not in [r[0] for r in results]
        # Sau 409 (đã chốt) không thể có lần hủy nào được chấp nhận nữa.
        first_409 = min((s for r, s in results if r == 409), default=None)
        if first_409 is not None:
            assert all(s < first_409 for r, s in accepted), (name, results)
        if any(r == "cancelled" for r, _ in results):
            # Hủy khi đang chờ: không bao giờ nhận slot, không chạy node nào.
            assert wid not in runner_calls and not graph.ran(name), name
            assert rec["status"] == "cancelled" and rec["completed_steps"] == []
        if accepted:
            assert rec["status"] == "cancelled" and rec["cancel_requested"] is True, (name, results)
            assert rec["error"] == app_mod.CANCELLED_ERROR
            # Sau khi hủy được chấp nhận: tối đa MỘT node bắt đầu (node vừa qua kiểm tra hủy).
            first_ok = min(s for _, s in accepted)
            assert len([e for e in graph.ran(name) if e[0] > first_ok]) <= 1, (name, graph.ran(name), results)
        else:
            # Không có hủy nào được chấp nhận ⇒ completed nguyên vẹn, không bao giờ cancelled.
            assert rec["status"] == "completed", (name, results)
            assert rec["completed_steps"] == list(ROLES) and rec["cancel_requested"] is False
            assert rec["error"] is None
        if rec["status"] == "cancelled" and wid in runner_calls:
            assert rec["completed_steps"] == [e[2] for e in graph.ran(name)]
    # FIFO: với 1 slot, run nhận slot đúng thứ tự submit.
    if concurrency == 1:
        assert runner_calls == [w for w in submit_order if w in set(runner_calls)]


# ── Slot luôn được trả, mọi kiểu kết thúc ──────────────────────────────────────


def test_slot_released_on_every_exit_path_in_fifo_order(aenv, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(app_mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(app_mod, "_WORKFLOW_MAX_SECONDS", 50)
    monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
    monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"])
    started = []  # thứ tự run nhận slot

    def before_node(name, role):
        if role == "ba":
            started.append(name)
        if name == "budget" and role == "ba":
            clock[0] += 100  # node ba "chạy" quá ngân sách W1
        if name == "cancel-mid" and role == "ba":
            assert app_mod.cancel_workflow(wids[name])["status"] == "running"

    class Graph:
        def stream(self, state, stream_mode="updates"):
            name = state["user_input"]
            for role in ("ba", "pm", "clarifier"):
                before_node(name, role)
                if name == "node-raises" and role == "pm":
                    raise RuntimeError("Ollama sập")
                yield _node(role, f"{role} của {name}")

    def regen(role, **kw):
        name = kw["user_input"]
        if name == "regen-cancel":
            assert app_mod.cancel_workflow(wids[name])["status"] == "running"
        if name == "regen-raises":
            raise RuntimeError("regen sập")
        return f"{role} v2"

    monkeypatch.setattr(app_mod, "run_single_step", regen)
    aenv.use_graph(Graph())
    order = ["node-raises", "cancel-mid", "budget", "regen-cancel", "regen-raises", "normal"]
    wids = {}
    with TestClient(app_mod.app):
        queue = app_mod._get_run_queue()
        with queue.lock:  # giữ lock: worker chưa lấy được run nào cho tới khi xếp đủ hàng
            for name in order:
                wids[name] = _submit(name)["workflow_id"]
            positions = [queue.live_state(wids[n])[0] for n in order]
        assert positions == [None, 1, 2, 3, 4, 5]
        for name in order:
            _wait_finished(wids[name])
        recs = {name: _db(wids[name]) for name in order}
        assert not queue._waiting and not queue._active

    assert started == order  # FIFO, mỗi run nhận slot sau khi run trước trả slot
    assert recs["node-raises"]["status"] == "failed" and "Ollama sập" in recs["node-raises"]["error"]
    assert recs["cancel-mid"]["status"] == "cancelled" and recs["cancel-mid"]["completed_steps"] == ["ba"]
    assert recs["budget"]["status"] == "completed" and "[Timeout" in recs["budget"]["error"]
    assert recs["regen-cancel"]["status"] == "cancelled"
    assert recs["regen-cancel"]["step_outputs"]["clarifier"] == "clarifier của regen-cancel"
    assert recs["regen-raises"]["status"] == "completed"  # lỗi regen non-fatal
    assert recs["normal"]["status"] == "completed" and recs["normal"]["step_outputs"]["be"] == "be v2"
    assert all(r["completed_at"] and r["queue_position"] is None for r in recs.values())


# ── Run thoát bất thường vẫn phải chốt trạng thái cuối ────────────────────────


@pytest.mark.parametrize("failure", ["raise", "none"])
def test_unloadable_record_still_ends_terminal_and_frees_slot(aenv, failure):
    """Worker không đọc được record (lỗi SQLite tạm thời / dữ liệu hỏng): run không được kẹt
    'pending' mãi mãi (UI/OWUI poll vô hạn) — phải failed, ghi JSONL, và slot được trả."""
    graph = aenv.use_graph(_TrackingGraph(behaviour={("A", "ba"): "gate"}))
    real_load = app_mod._load_workflow
    broken = set()

    def load(wid):
        if wid in broken and threading.current_thread().name.startswith("workflow-worker"):
            if failure == "raise":
                raise sqlite3.OperationalError("disk I/O error")
            return None
        return real_load(wid)

    aenv.monkeypatch.setattr(app_mod, "_load_workflow", load)
    with TestClient(app_mod.app) as client:
        A = _submit("A")["workflow_id"]
        assert graph.entered("A", "ba").wait(WAIT_TIMEOUT)
        B = _submit("B")["workflow_id"]
        broken.add(B)
        C = _submit("C")["workflow_id"]
        graph.gate("A", "ba").set()
        for wid in (A, B, C):
            _wait_finished(wid)
        rb = client.get(f"/workflow/{B}").json()
        rc = client.get(f"/workflow/{C}").json()
    assert not graph.ran("B")
    assert rb["status"] == "failed" and rb["error"] == app_mod.UNLOADABLE_ERROR and rb["completed_at"]
    assert rb["user_input"] == "B" and rb["queue_position"] is None
    assert rc["status"] == "completed"
    assert [e["status"] for e in _episodic_lines(aenv) if e["workflow_id"] == B] == ["failed"]


def test_cancel_accepted_while_record_unloadable_still_ends_cancelled(aenv):
    graph = aenv.use_graph(_TrackingGraph())
    real_load = app_mod._load_workflow
    responses = []

    def load(wid):
        if threading.current_thread().name.startswith("workflow-worker"):
            # Run đã nhận slot ("running") → hủy được chấp nhận (200), rồi đọc record thất bại.
            responses.append(app_mod.cancel_workflow(wid)["status"])
            raise sqlite3.OperationalError("database is locked")
        return real_load(wid)

    aenv.monkeypatch.setattr(app_mod, "_load_workflow", load)
    with TestClient(app_mod.app) as client:
        A = _submit("A")["workflow_id"]
        _wait_finished(A)
        ra = client.get(f"/workflow/{A}").json()
    assert responses == ["running"] and not graph.entries
    assert ra["status"] == "cancelled" and ra["error"] == app_mod.CANCELLED_ERROR
    assert ra["cancel_requested"] is True and ra["completed_at"]


@pytest.mark.parametrize("where", ["node", "regen"])
def test_base_exception_never_leaves_record_running(aenv, monkeypatch, where):
    """BaseException (không phải Exception) lọt qua `except Exception` của runner: record
    không được ghi 'running' kèm completed_at (kẹt running mãi, JSONL ghi status running)."""
    if where == "node":
        graph = aenv.use_graph(_TrackingGraph(behaviour={("A", "pm"): _Abort("dừng")}))
    else:
        graph = aenv.use_graph(_TrackingGraph())
        monkeypatch.setattr(app_mod, "_CLARIFIER_REGEN_LOOPS", 1)
        monkeypatch.setattr(app_mod, "_parse_clarifier_regen_list", lambda text: ["be"])

        def regen(role, **kw):
            raise _Abort("dừng")

        monkeypatch.setattr(app_mod, "run_single_step", regen)
    with TestClient(app_mod.app) as client:
        A = _submit("A")["workflow_id"]
        B = _submit("B")["workflow_id"]
        for wid in (A, B):
            _wait_finished(wid)
        ra = client.get(f"/workflow/{A}").json()
        rb = client.get(f"/workflow/{B}").json()
        assert client.post(f"/workflow/{A}/cancel").status_code == 409  # đã kết thúc thật
    assert ra["status"] == "failed" and ra["completed_at"]
    assert ra["error"] == f"{app_mod.ABORTED_ERROR} (_Abort)" and ra["cancel_requested"] is False
    # Worker sống sót và slot được trả: B (cùng graph) chạy tiếp.
    assert graph.ran("B")
    assert rb["status"] == ("completed" if where == "node" else "failed")
    assert [e["status"] for e in _episodic_lines(aenv) if e["workflow_id"] == A] == ["failed"]


def test_cancel_accepted_before_base_exception_still_ends_cancelled(aenv):
    graph = aenv.use_graph(_TrackingGraph(behaviour={("A", "ba"): "gate", ("A", "pm"): _Abort("dừng")}))
    with TestClient(app_mod.app) as client:
        A = _submit("A")["workflow_id"]
        assert graph.entered("A", "ba").wait(WAIT_TIMEOUT)
        assert _cancel(A)[0] == "running"
        graph.gate("A", "ba").set()
        _wait_finished(A)
        ra = client.get(f"/workflow/{A}").json()
    # Hủy dừng run sau node ba — node pm (sẽ raise) không bao giờ chạy.
    assert [e[2] for e in graph.ran("A")] == ["ba"]
    assert ra["status"] == "cancelled" and ra["error"] == app_mod.CANCELLED_ERROR


# ── LangGraph thật ─────────────────────────────────────────────────────────────


def test_real_langgraph_cancel_stops_before_next_node_and_frees_slot(aenv):
    from langgraph.graph import END, StateGraph
    from workflow import SDLCState

    lock = threading.Lock()
    entered, gates = {}, {}
    ran = []  # (user_input, role, thread_name)
    active = [0, 0]  # [đang chạy, max]

    def make_node(role):
        def node(state):
            name = state["user_input"]
            with lock:
                ran.append((name, role, threading.current_thread().name))
                active[0] += 1
                active[1] = max(active[1], active[0])
            try:
                _event(entered, lock, (name, role)).set()
                if not _event(gates, lock, (name, role)).wait(WAIT_TIMEOUT):
                    raise AssertionError(f"cổng {name}/{role} không được mở")
            finally:
                with lock:
                    active[0] -= 1
            return {"step_outputs": {role: f"{role} của {name}"}, "completed_steps": [role]}

        return node

    builder = StateGraph(SDLCState)
    for role in ROLES:
        builder.add_node(role, make_node(role))
    builder.set_entry_point(ROLES[0])
    for a, b in zip(ROLES, ROLES[1:]):
        builder.add_edge(a, b)
    builder.add_edge(ROLES[-1], END)
    compiled = builder.compile()

    class _Release:
        def stream(self, *a, **kw):
            return compiled.stream(*a, **kw)

        def release_everything(self):
            with lock:
                for gate in gates.values():
                    gate.set()

    aenv.use_graph(_Release())
    with TestClient(app_mod.app) as client:
        A = _submit("A")["workflow_id"]
        assert _event(entered, lock, ("A", "ba")).wait(WAIT_TIMEOUT)
        B = _submit("B")["workflow_id"]
        assert client.get(f"/workflow/{B}").json()["queue_position"] == 1
        assert _cancel(A)[0] == "running"
        _event(gates, lock, ("A", "ba")).set()
        assert _event(entered, lock, ("B", "ba")).wait(WAIT_TIMEOUT)  # slot của A đã được trả
        _wait_finished(A)
        for role in ROLES:
            _event(gates, lock, ("B", role)).set()
        _wait_finished(B)
        ra = client.get(f"/workflow/{A}").json()
        rb = client.get(f"/workflow/{B}").json()

    # Node chạy INLINE trong thread worker (không có executor nền nào chạy tiếp sau close()).
    assert {t for _, _, t in ran} == {"workflow-worker-1"}
    assert [(n, r) for n, r, _ in ran] == [("A", "ba"), ("B", "ba"), ("B", "pm"), ("B", "sa")]
    assert active[1] == 1
    assert ra["status"] == "cancelled" and ra["completed_steps"] == ["ba"]
    assert rb["status"] == "completed" and rb["completed_steps"] == list(ROLES)


# ── Recovery lúc startup ───────────────────────────────────────────────────────


def test_startup_recovery_only_touches_active_rows(aenv):
    graph = aenv.use_graph(_TrackingGraph(behaviour={
        ("running-cancel", "ba"): "gate",
        ("node-fails", "pm"): RuntimeError("Ollama sập"),
    }))
    with TestClient(app_mod.app):
        rc = _submit("running-cancel")["workflow_id"]
        assert graph.entered("running-cancel", "ba").wait(WAIT_TIMEOUT)
        pc = _submit("pending-cancel")["workflow_id"]
        ok = _submit("completed")["workflow_id"]
        nf = _submit("node-fails")["workflow_id"]
        assert _cancel(pc)[0] == "cancelled"
        assert _cancel(rc)[0] == "running"
        graph.gate("running-cancel", "ba").set()
        for wid in (rc, pc, ok, nf):
            _wait_finished(wid)
        assert app_mod._get_run_queue().close(timeout=WAIT_TIMEOUT)

    statuses = {wid: _db(wid)["status"] for wid in (rc, pc, ok, nf)}
    assert statuses == {rc: "cancelled", pc: "cancelled", ok: "completed", nf: "failed"}
    # "Process cũ chết": thêm dòng còn dang dở + dòng kết thúc có JSON hỏng.
    conn = sqlite3.connect(aenv.tmp / "db" / "workflows.sqlite3")
    try:
        with conn:
            for wid, status, data in [
                ("bad-cancelled", "cancelled", "{not json"),
                ("bad-completed", "completed", "{not json"),
                ("bad-running", "running", "{not json"),
            ]:
                conn.execute(
                    "INSERT INTO workflows (workflow_id, status, created_at, data) VALUES (?, ?, ?, ?)",
                    (wid, status, "2026-01-01T00:00:00+00:00", data),
                )
    finally:
        conn.close()
    for wid, status in [("left-pending", "pending"), ("left-running", "running")]:
        app_mod._store_workflow(app_mod.WorkflowRecord(
            workflow_id=wid, status=status, project=None, user_input=wid,
            created_at="2026-01-02T00:00:00+00:00", cancel_requested=True,
        ))
    before = _raw_rows(aenv)
    terminal = [rc, pc, ok, nf, "bad-cancelled", "bad-completed"]

    with TestClient(app_mod.app) as client:  # startup mới → recovery
        after = _raw_rows(aenv)
        recovered = {wid: client.get(f"/workflow/{wid}").json() for wid in ("left-pending", "left-running")}
        bad = client.get("/workflow/bad-running").json()
    for wid in terminal:
        assert after[wid] == before[wid], f"recovery không được đụng tới {wid}"
    for rec in [*recovered.values(), bad]:
        assert rec["status"] == "failed" and rec["error"] == INTERRUPTED_ERROR and rec["completed_at"]

    # Startup lần nữa: không còn gì để khôi phục — mọi dòng giữ nguyên byte.
    with TestClient(app_mod.app):
        again = _raw_rows(aenv)
    assert again == after
