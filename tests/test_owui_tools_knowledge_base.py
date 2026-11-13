"""Open WebUI tool yan_knowledge_base — timeout căn theo rag-api, gợi ý URL, nguồn.

HTTP tới rag-api được thay bằng hàm giả (monkeypatch requests.post của module tool).
"""

import re
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
tool_mod = pytest.importorskip("yan_knowledge_base")

ROOT = Path(__file__).resolve().parent.parent


class _Resp:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)


def _active_env_int(name: str) -> int | None:
    """Giá trị KHÔNG bị comment của *name* trong .env (None nếu không có .env / không đặt)."""
    env = ROOT / ".env"
    if not env.is_file():
        return None
    m = re.search(rf"^{name}=(\d+)\s*$", env.read_text(encoding="utf-8"), re.MULTILINE)
    return int(m.group(1)) if m else None


def test_default_timeout_covers_rag_api_llm_timeout():
    # /ask chỉ trả về khi CHAT_MODEL sinh xong; tool bỏ cuộc sớm thì Ollama vẫn sinh
    # tiếp và chiếm slot duy nhất → Valve phải >= OLLAMA_REQUEST_TIMEOUT của rag-api.
    default = tool_mod.Tools.Valves.model_fields["timeout"].default
    assert default >= 1200
    active = _active_env_int("OLLAMA_REQUEST_TIMEOUT")
    if active is not None:
        assert default >= active
    assert "OLLAMA_REQUEST_TIMEOUT" in tool_mod.Tools.Valves.model_fields["timeout"].description


def test_valve_description_points_to_service_name_not_host_fallback():
    field = tool_mod.Tools.Valves.model_fields["rag_api_url"]
    assert field.default == "http://rag-api:8090"
    assert "rag-agent-platform_yan" in field.description
    assert "http://host.docker.internal:8090" not in field.description
    assert "http://localhost:8090" not in field.description


@pytest.fixture()
def post(monkeypatch):
    calls = []
    state = {"resp": _Resp(200, {"answer": "OK", "sources": []}), "exc": None}

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if state["exc"] is not None:
            raise state["exc"]
        return state["resp"]

    monkeypatch.setattr(tool_mod.requests, "post", fake_post)
    return calls, state


def test_ask_uses_valve_timeout_and_dedupes_sources(post):
    calls, state = post
    state["resp"] = _Resp(200, {"answer": "Trả lời", "sources": [
        {"source_file": "a.md", "module": "auth"},
        {"source_file": "b.md", "module": "billing"},
        {"source_file": "a.md", "module": "auth"},
        {"source_file": None},
    ]})
    t = tool_mod.Tools()
    out = t.ask_internal_docs("q", project="crm")
    assert calls[0]["url"] == "http://rag-api:8090/ask"
    assert calls[0]["timeout"] == t.valves.timeout
    assert calls[0]["json"] == {"question": "q", "project": "crm"}
    assert "**Nguồn:** a.md · b.md" in out  # khử trùng lặp, giữ thứ tự
    assert "**Module:** auth, billing" in out


def test_timeout_message_warns_generation_still_running(post):
    _, state = post
    state["exc"] = requests.exceptions.Timeout("read timeout")
    out = tool_mod.Tools().ask_internal_docs("q")
    assert out.startswith("❌")
    assert "slot" in out and "OLLAMA_REQUEST_TIMEOUT" in out


def test_connection_error_points_to_compose_network(post):
    _, state = post
    state["exc"] = requests.exceptions.ConnectionError("refused")
    out = tool_mod.Tools().ask_internal_docs("q")
    assert "http://rag-api:8090" in out
    assert "rag-agent-platform_yan" in out
    assert "http://localhost:8090" not in out
    assert "http://host.docker.internal:8090" not in out


def test_http_error_reports_status_and_body(post):
    _, state = post
    state["resp"] = _Resp(404, {"detail": "x"}, text='{"detail":"chưa ingest"}')
    out = tool_mod.Tools().ask_internal_docs("q")
    assert "HTTP 404" in out and "chưa ingest" in out


def test_http_error_parts_handles_missing_response():
    assert tool_mod._http_error_parts(requests.exceptions.HTTPError("x")) == (None, "x")
