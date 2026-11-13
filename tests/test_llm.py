"""Test llm.py — factory ChatOllama: timeout thật, trần num_predict, FILEGEN_THINK.

Không gọi Ollama: chỉ dựng ChatOllama (không validate model lúc init) rồi soi client
ollama/httpx bên trong và payload `_chat_params` mà ChatOllama sẽ gửi đi.
"""

import runpy
from types import SimpleNamespace

import pytest

llm = pytest.importorskip("llm")
httpx = pytest.importorskip("httpx")
messages = pytest.importorskip("langchain_core.messages")


# ── Timeout: phải nằm trên httpx client thật, không phải kwarg bị bỏ qua ─────────


def test_make_chat_sets_real_httpx_timeout_on_sync_and_async_clients(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_REQUEST_TIMEOUT", 42)
    chat = llm.make_chat("granite4.1:8b", base_url="http://ollama.test:11434")
    # ollama.Client giữ httpx.Client ở thuộc tính _client.
    assert chat._client._client.timeout == httpx.Timeout(42.0)
    assert chat._async_client._client.timeout == httpx.Timeout(42.0)
    assert chat._client._client.timeout.read == 42.0


def test_make_chat_default_timeout_comes_from_env_constant():
    chat = llm.make_chat("granite4.1:8b", base_url="http://ollama.test:11434")
    assert chat._client._client.timeout.read == float(llm.OLLAMA_REQUEST_TIMEOUT)
    assert chat._client._client.timeout.read is not None


@pytest.mark.parametrize("value", [0, -5])
def test_make_chat_non_positive_timeout_means_unlimited(monkeypatch, value):
    monkeypatch.setattr(llm, "OLLAMA_REQUEST_TIMEOUT", value)
    chat = llm.make_chat("m", base_url="http://ollama.test:11434")
    assert chat._client._client.timeout == httpx.Timeout(None)


def test_make_chat_base_url_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_BASE_URL", "http://fallback.test:11434")
    chat = llm.make_chat("m")
    assert chat.base_url == "http://fallback.test:11434"
    assert str(chat._client._client.base_url).startswith("http://fallback.test:11434")


def test_make_chat_client_kwargs_not_shared_between_instances():
    # ChatOllama._set_clients ghi auth header (từ userinfo trong URL) vào client_kwargs:
    # dict dùng chung sẽ rò header sang instance khác.
    with_auth = llm.make_chat("m", base_url="http://user:secret@ollama.test:11434")
    without = llm.make_chat("m", base_url="http://ollama.test:11434")
    assert "authorization" in with_auth._client._client.headers
    assert "authorization" not in without._client._client.headers


# ── Tham số sinh: num_predict / num_ctx / reasoning tới tận payload ──────────────


def _payload(chat):
    return chat._chat_params([messages.HumanMessage(content="hi")])


def test_make_chat_passes_generation_options_into_request_payload():
    chat = llm.make_chat("qwen3:8b", base_url="http://x:1", num_ctx=4096, num_predict=777, reasoning=False)
    params = _payload(chat)
    assert params["options"]["num_predict"] == 777
    assert params["options"]["num_ctx"] == 4096
    assert params["options"]["temperature"] == 0.1
    assert params["think"] is False


def test_make_chat_omits_unset_options():
    params = _payload(llm.make_chat("qwen3:8b", base_url="http://x:1"))
    assert "num_predict" not in params["options"]
    assert "num_ctx" not in params["options"]
    assert params["think"] is None   # không truyền reasoning → model tự quyết


# ── num_predict_for: trần theo loại lời gọi + headroom cho reasoning ───────────


@pytest.fixture
def caps(monkeypatch):
    monkeypatch.setattr(llm, "PLANNER_NUM_PREDICT", 1000)
    monkeypatch.setattr(llm, "FILEGEN_NUM_PREDICT", 3000)
    monkeypatch.setattr(llm, "REPORT_NUM_PREDICT", 5000)


@pytest.mark.parametrize("kind,base", [("planner", 1000), ("filegen", 3000), ("report", 5000)])
def test_num_predict_for_plain_model_uses_base_cap(caps, kind, base):
    assert llm.num_predict_for(kind, "granite4.1:8b") == base


@pytest.mark.parametrize("kind,base", [("planner", 1000), ("filegen", 3000), ("report", 5000)])
def test_num_predict_for_reasoning_model_gets_headroom(caps, kind, base):
    # Token <think> tính chung ngân sách num_predict → reasoning model được nhân headroom.
    assert llm.num_predict_for(kind, "qwen3.6:35b") == base * llm._THINKING_HEADROOM


def test_num_predict_for_explicit_reasoning_flag_overrides_model_name(caps):
    # reasoning=False tắt thinking → không cần headroom dù model là reasoning model.
    assert llm.num_predict_for("filegen", "qwen3.6:35b", reasoning=False) == 3000
    # reasoning=True bật thinking → cần headroom dù tên model không có trong danh sách.
    assert llm.num_predict_for("filegen", "granite4.1:8b", reasoning=True) == 3000 * llm._THINKING_HEADROOM


def test_num_predict_for_clamps_to_num_ctx(caps):
    assert llm.num_predict_for("report", "qwen3.6:35b", num_ctx=4096) == 4096
    assert llm.num_predict_for("planner", "granite4.1:8b", num_ctx=8192) == 1000


@pytest.mark.parametrize("value", [0, -1])
def test_num_predict_for_non_positive_means_no_cap(monkeypatch, value):
    monkeypatch.setattr(llm, "REPORT_NUM_PREDICT", value)
    assert llm.num_predict_for("report", "qwen3.6:35b", num_ctx=32768) is None


_LLM_ENV = ("PLANNER_NUM_PREDICT", "FILEGEN_NUM_PREDICT", "REPORT_NUM_PREDICT", "FILEGEN_THINK", "OLLAMA_REQUEST_TIMEOUT")


def _fresh_module_globals():
    """Chạy lại llm.py trong namespace riêng (không đụng module đã import)."""
    return runpy.run_path(llm.__file__)


def test_defaults_when_env_unset(monkeypatch):
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    g = _fresh_module_globals()
    # Khớp default trong docker-compose.yml / .env / README.
    assert (g["PLANNER_NUM_PREDICT"], g["FILEGEN_NUM_PREDICT"], g["REPORT_NUM_PREDICT"]) == (2048, 8192, 16384)
    assert g["FILEGEN_THINK"] is None
    assert g["OLLAMA_REQUEST_TIMEOUT"] == 1200
    # Với default model + OLLAMA_CONTEXT_LENGTH=32768: reasoning model có gấp đôi ngân
    # sách cho phần suy luận, planner (granite, không reasoning) giữ 2048.
    npf = g["num_predict_for"]
    assert npf("report", "qwen3.6:35b", num_ctx=32768) == 32768
    assert npf("filegen", "north-mini-code-1.0:q4_K_M", num_ctx=32768) == 16384
    assert npf("planner", "granite4.1:8b", num_ctx=8192) == 2048


def test_env_values_parsed_at_import(monkeypatch):
    monkeypatch.setenv("FILEGEN_THINK", "")          # mặc định của compose
    monkeypatch.setenv("PLANNER_NUM_PREDICT", "0")
    monkeypatch.setenv("REPORT_NUM_PREDICT", "")     # rỗng = default
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "90")
    g = _fresh_module_globals()
    assert g["FILEGEN_THINK"] is None
    assert g["PLANNER_NUM_PREDICT"] == 0
    assert g["REPORT_NUM_PREDICT"] == 16384
    assert g["OLLAMA_REQUEST_TIMEOUT"] == 90
    monkeypatch.setenv("FILEGEN_THINK", "False")
    assert _fresh_module_globals()["FILEGEN_THINK"] is False


# ── Đọc env ─────────────────────────────────────────────────────────────────────


def test_env_int_empty_or_unset_uses_default(monkeypatch):
    monkeypatch.delenv("X_TEST_NUM", raising=False)
    assert llm._env_int("X_TEST_NUM", 7) == 7
    monkeypatch.setenv("X_TEST_NUM", "")
    assert llm._env_int("X_TEST_NUM", 7) == 7
    monkeypatch.setenv("X_TEST_NUM", " 0 ")
    assert llm._env_int("X_TEST_NUM", 7) == 0
    monkeypatch.setenv("X_TEST_NUM", "-1")
    assert llm._env_int("X_TEST_NUM", 7) == -1


@pytest.mark.parametrize("raw,expected", [
    (None, None),
    ("", None),          # compose truyền ${FILEGEN_THINK:-} → chuỗi rỗng = chưa đặt
    ("   ", None),
    ("true", True),
    ("TRUE", True),
    (" True ", True),
    ("false", False),
    ("False", False),
    ("yes", None),       # không hợp lệ → bỏ qua (cảnh báo), không đoán
    ("1", None),
])
def test_parse_think(raw, expected):
    assert llm._parse_think(raw) is expected


def test_filegen_reasoning_reads_module_value_at_call_time(monkeypatch):
    monkeypatch.setattr(llm, "FILEGEN_THINK", True)
    assert llm.filegen_reasoning() is True
    monkeypatch.setattr(llm, "FILEGEN_THINK", None)
    assert llm.filegen_reasoning() is None


# ── Nhận diện model / chạm trần ─────────────────────────────────────────────────


@pytest.mark.parametrize("model,expected", [
    ("qwen3.6:35b", True),
    ("north-mini-code-1.0:q4_K_M", True),
    ("DeepSeek-R1:14b", True),
    ("granite4.1:8b", False),
    ("codegemma:2b", False),
])
def test_is_reasoning_model(model, expected):
    assert llm.is_reasoning_model(model) is expected


def test_hit_token_cap():
    assert llm.hit_token_cap(SimpleNamespace(content="x")) is False
    assert llm.hit_token_cap(SimpleNamespace(content="x", response_metadata={"done_reason": "stop"})) is False
    assert llm.hit_token_cap(SimpleNamespace(content="x", response_metadata={"done_reason": "length"})) is True
    assert llm.hit_token_cap(SimpleNamespace(content="x", response_metadata=None)) is False
