"""Open WebUI tool yan_agent_workflow — trạng thái kết thúc, hàng đợi, bước lỗi, hủy.

HTTP tới agent-api được thay bằng hàm giả (monkeypatch requests.get/post của module
tool); time.sleep bị vô hiệu để vòng poll chạy tức thì.
"""

import pytest

requests = pytest.importorskip("requests")
tool_mod = pytest.importorskip("yan_agent_workflow")

BASE = "http://agent-api:8091"
WF_ID = "wf-123"
AGENTS = {
    "ba": {"step_id": 1, "name": "BA", "model": "m"},
    "pm": {"step_id": 2, "name": "PM", "model": "m"},
    "be": {"step_id": 3, "name": "BE", "model": "m"},
}


class _Resp:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else str(body)

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


def _wf(status, **extra):
    body = {
        "workflow_id": WF_ID,
        "status": status,
        "project": None,
        "user_input": "x",
        "step_outputs": {},
        "completed_steps": [],
        "failed_steps": [],
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "completed_at": None,
    }
    body.update(extra)
    return body


@pytest.fixture()
def http(monkeypatch):
    """Ghi lại request; `states` là danh sách body lần lượt trả cho GET /workflow/{id}."""
    calls = {"get": [], "post": [], "states": [], "post_resp": _Resp(200, {"workflow_id": WF_ID, "status": "pending"})}

    def fake_get(url, timeout=None, **kw):
        calls["get"].append(url)
        if url == f"{BASE}/agents":
            return _Resp(200, AGENTS)
        if url == f"{BASE}/workflow/{WF_ID}":
            states = calls["states"]
            body = states.pop(0) if len(states) > 1 else states[0]
            return _Resp(200, body)
        return _Resp(404, {"detail": "not found"})

    def fake_post(url, json=None, timeout=None, **kw):
        calls["post"].append((url, json))
        return calls["post_resp"]

    monkeypatch.setattr(tool_mod.requests, "get", fake_get)
    monkeypatch.setattr(tool_mod.requests, "post", fake_post)
    monkeypatch.setattr(tool_mod.time, "sleep", lambda s: None)
    return calls


def _workflow_polls(calls):
    return [u for u in calls["get"] if u == f"{BASE}/workflow/{WF_ID}"]


@pytest.fixture()
def blocking_tool():
    t = tool_mod.Tools()
    t.valves.block_until_complete = True
    t.valves.poll_interval = 0
    t.valves.poll_max_attempts = 5
    return t


# ── Vòng poll: cancelled là trạng thái kết thúc ─────────────────────────────────

def test_poll_stops_immediately_on_cancelled(http, blocking_tool):
    http["states"] = [_wf("cancelled", completed_steps=["ba", "pm"], failed_steps=["pm"])]
    out = blocking_tool.run_sdlc_workflow("build x")
    assert len(_workflow_polls(http)) == 1  # trước đây poll tới hết poll_max_attempts
    assert "Workflow đã bị hủy" in out
    assert "2/3" in out
    assert "Bước lỗi: pm" in out
    assert "Timeout" not in out


def test_poll_pending_queue_then_completed_with_failed_steps(http, blocking_tool):
    http["states"] = [
        _wf("pending", queue_position=0),
        _wf("running", completed_steps=["ba"]),
        _wf("completed", completed_steps=["ba", "pm", "be"], failed_steps=["be"],
            step_outputs={"ba": "BA out", "pm": "PM out", "be": "[LỖI] boom"}),
    ]
    out = blocking_tool.run_sdlc_workflow("build x")
    assert len(_workflow_polls(http)) == 3
    assert "queue_position=0" in out  # 0 là vị trí hợp lệ, vẫn phải hiển thị
    assert "Workflow Hoàn thành" in out
    assert "1 bước lỗi:** be" in out


def test_poll_failed_reports_error_and_failed_steps(http, blocking_tool):
    http["states"] = [_wf("failed", error="Ollama chết", failed_steps=["ba"])]
    out = blocking_tool.run_sdlc_workflow("build x")
    assert len(_workflow_polls(http)) == 1
    assert "Workflow thất bại:** Ollama chết" in out
    assert "Bước lỗi: ba" in out


def test_poll_times_out_when_never_terminal(http, blocking_tool):
    http["states"] = [_wf("running")]
    out = blocking_tool.run_sdlc_workflow("build x")
    assert len(_workflow_polls(http)) == 5
    assert "Timeout sau 5 lần poll" in out


def test_non_blocking_submit_shows_queue_position_and_cancel_hint(http):
    http["post_resp"] = _Resp(200, {"workflow_id": WF_ID, "status": "pending", "queue_position": 2})
    out = tool_mod.Tools().run_sdlc_workflow("build x")
    assert "Vị trí hàng đợi:** 2" in out
    assert f"cancel_workflow('{WF_ID}')" in out
    assert _workflow_polls(http) == []


# ── get_workflow_result: summary ──────────────────────────────────────────────

@pytest.mark.parametrize("qp", [0, 3])
def test_summary_pending_shows_queue_position(http, qp):
    http["states"] = [_wf("pending", queue_position=qp)]
    out = tool_mod.Tools().get_workflow_result(WF_ID)
    assert f"Vị trí hàng đợi:** {qp}" in out
    assert f"cancel_workflow('{WF_ID}')" in out


def test_summary_pending_without_queue_position_from_older_backend(http):
    body = _wf("pending")
    del body["failed_steps"]  # agent-api cũ: không có field mới → không được KeyError
    http["states"] = [body]
    out = tool_mod.Tools().get_workflow_result(WF_ID)
    assert "đang chờ tới lượt chạy" in out


def test_summary_completed_marks_failed_steps(http):
    http["states"] = [_wf("completed", completed_steps=["ba", "be"], failed_steps=["be"],
                          step_outputs={"ba": "BA out", "be": "[LỖI] boom"})]
    out = tool_mod.Tools().get_workflow_result(WF_ID)
    assert "Bước lỗi (1):** be" in out
    assert "BE Agent — Backend Implementation** ⚠️" in out
    assert "cancel_workflow" not in out  # đã kết thúc → không gợi ý hủy


def test_summary_cancelled(http):
    http["states"] = [_wf("cancelled", completed_steps=["ba"], step_outputs={"ba": "BA out"})]
    out = tool_mod.Tools().get_workflow_result(WF_ID)
    assert "**Status:** cancelled" in out
    assert "Đã bị hủy" in out


def test_role_output_flags_failed_step(http):
    http["states"] = [_wf("completed", failed_steps=["be"], step_outputs={"be": "[LỖI] boom"})]
    out = tool_mod.Tools().get_workflow_result(WF_ID, role="be")
    assert "(bước lỗi)" in out and "[LỖI] boom" in out


# ── cancel_workflow ─────────────────────────────────────────────────────────────

def test_cancel_workflow_success(http):
    http["post_resp"] = _Resp(200, {"workflow_id": WF_ID, "status": "cancelled"})
    out = tool_mod.Tools().cancel_workflow(f"  {WF_ID}  ")
    assert http["post"] == [(f"{BASE}/workflow/{WF_ID}/cancel", None)]
    assert "đã bị hủy" in out


def test_cancel_workflow_accepted_but_still_running(http):
    http["post_resp"] = _Resp(200, {"workflow_id": WF_ID, "status": "running"})
    out = tool_mod.Tools().cancel_workflow(WF_ID)
    assert "Đã gửi yêu cầu hủy" in out and "running" in out


def test_cancel_workflow_not_found(http):
    http["post_resp"] = _Resp(404, {"detail": "Workflow 'x' không tìm thấy"})
    out = tool_mod.Tools().cancel_workflow(WF_ID)
    assert out.startswith("❌") and "không tìm thấy" in out


def test_cancel_workflow_conflict_shows_backend_detail(http):
    http["post_resp"] = _Resp(409, {"detail": "Workflow đã kết thúc (completed)"})
    out = tool_mod.Tools().cancel_workflow(WF_ID)
    assert "Không hủy được" in out
    assert "Workflow đã kết thúc (completed)" in out


def test_cancel_workflow_other_http_error(http):
    http["post_resp"] = _Resp(500, None, text="Internal Server Error")
    out = tool_mod.Tools().cancel_workflow(WF_ID)
    assert "HTTP 500" in out and "Internal Server Error" in out


def test_cancel_workflow_empty_id_does_not_call_api(http):
    assert tool_mod.Tools().cancel_workflow("   ").startswith("❌")
    assert http["post"] == []


def test_cancel_workflow_connection_error_points_to_compose_network(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(tool_mod.requests, "post", boom)
    out = tool_mod.Tools().cancel_workflow(WF_ID)
    assert "http://agent-api:8091" in out
    assert "rag-agent-platform_yan" in out
    assert "http://localhost:8091" not in out
    assert "http://host.docker.internal:8091" not in out


def test_cancel_workflow_is_a_public_tool_method():
    # Open WebUI chỉ expose method public của class Tools, mô tả lấy từ docstring.
    method = tool_mod.Tools.cancel_workflow
    assert ":param workflow_id:" in (method.__doc__ or "")


# ── Valve / helper ──────────────────────────────────────────────────────────────

def test_valve_description_points_to_service_name_not_host_fallback():
    field = tool_mod.Tools.Valves.model_fields["agent_api_url"]
    assert field.default == "http://agent-api:8091"
    assert "rag-agent-platform_yan" in field.description
    assert "http://host.docker.internal:8091" not in field.description
    assert "http://localhost:8091" not in field.description


def test_http_error_parts_handles_missing_response():
    assert tool_mod._http_error_parts(requests.exceptions.HTTPError("x")) == (None, "x")
