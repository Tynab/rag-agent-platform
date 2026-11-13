"""Mỗi lời gọi LLM trong workflow.py đi qua llm.make_chat với đúng timeout / trần token.

llm.ChatOllama được thay bằng recorder: ghi lại kwargs khởi tạo của từng instance và
trả lời theo kịch bản — không gọi Ollama thật.
"""

import dataclasses
from types import SimpleNamespace

import pytest

workflow = pytest.importorskip("workflow")
llm = pytest.importorskip("llm")

PLAIN_MODEL = "llama3.1:8b"          # không thuộc REASONING_MODELS
REASONING_MODEL = "qwen3.6:35b"


class Recorder:
    """Thay cho llm.ChatOllama: ghi kwargs khởi tạo, trả lời lần lượt từ *replies*."""

    def __init__(self, replies):
        self.instances: list[dict] = []
        self.prompts: list[str] = []
        self._replies = list(replies)

    def __call__(self, **kwargs):
        self.instances.append(kwargs)
        rec = self

        class _Chat:
            def invoke(self, messages):
                rec.prompts.append(messages[-1].content)
                reply = rec._replies.pop(0) if len(rec._replies) > 1 else rec._replies[0]
                if isinstance(reply, Exception):
                    raise reply
                if isinstance(reply, SimpleNamespace):
                    return reply
                return SimpleNamespace(content=reply)

        return _Chat()


@pytest.fixture
def caps(monkeypatch):
    monkeypatch.setattr(llm, "PLANNER_NUM_PREDICT", 1000)
    monkeypatch.setattr(llm, "FILEGEN_NUM_PREDICT", 3000)
    monkeypatch.setattr(llm, "REPORT_NUM_PREDICT", 5000)
    monkeypatch.setattr(llm, "OLLAMA_REQUEST_TIMEOUT", 321)
    monkeypatch.setattr(llm, "FILEGEN_THINK", None)
    monkeypatch.setattr(workflow, "OLLAMA_NUM_CTX", 65536)


def _agent(role, model):
    return dataclasses.replace(workflow.AGENTS[role], model=model)


def _install(monkeypatch, *replies):
    rec = Recorder(replies)
    monkeypatch.setattr(llm, "ChatOllama", rec)
    return rec


# ── Planner ─────────────────────────────────────────────────────────────────────


def test_planner_call_has_timeout_and_planner_cap(monkeypatch, caps):
    monkeypatch.setattr(workflow, "CODING_PLANNER_MODEL", "granite4.1:8b")
    rec = _install(monkeypatch, '[{"filename": "src/a.ts"}]')
    files = workflow._plan_code_files("fe", workflow.AGENTS["fe"], "http://ollama.test:11434", "task", ["react"], "ctx")
    assert [f["filename"] for f in files] == ["src/a.ts"]
    (kw,) = rec.instances
    assert kw["model"] == "granite4.1:8b"
    assert kw["base_url"] == "http://ollama.test:11434"
    assert kw["client_kwargs"] == {"timeout": 321.0}
    assert kw["num_ctx"] == 8192
    assert kw["num_predict"] == 1000
    assert "reasoning" not in kw     # FILEGEN_THINK không áp cho planner


def test_planner_with_reasoning_model_gets_headroom(monkeypatch, caps):
    monkeypatch.setattr(workflow, "CODING_PLANNER_MODEL", REASONING_MODEL)
    rec = _install(monkeypatch, '[{"filename": "src/a.ts"}]')
    workflow._plan_code_files("fe", workflow.AGENTS["fe"], "http://x", "task", None, "ctx")
    assert rec.instances[0]["num_predict"] == 2000


def test_planner_truncated_thinking_does_not_leak_draft_json(monkeypatch, caps):
    # Chạm trần giữa lúc suy luận: <think> không đóng, chứa JSON nháp → không được dùng.
    monkeypatch.setattr(workflow, "CODING_PLANNER_MODEL", REASONING_MODEL)
    _install(monkeypatch, SimpleNamespace(
        content='<think>maybe [{"filename": "draft.ts"}] or',
        response_metadata={"done_reason": "length"},
    ))
    assert workflow._plan_code_files("fe", workflow.AGENTS["fe"], "http://x", "task", None, "ctx") == []


# ── Sinh từng file ──────────────────────────────────────────────────────────────


def test_file_generation_uses_filegen_cap_and_timeout(monkeypatch, caps):
    rec = _install(monkeypatch, "```ts\nexport const a = 1;\n```")
    out = workflow._generate_one_file(
        "src/a.ts", "a", "typescript", _agent("fe", PLAIN_MODEL), "http://x", ["react"],
    )
    assert out == "export const a = 1;"
    (kw,) = rec.instances
    assert kw["num_predict"] == 3000
    assert kw["client_kwargs"] == {"timeout": 321.0}
    assert kw["num_ctx"] == 65536
    assert "reasoning" not in kw     # FILEGEN_THINK chưa đặt → không truyền


def test_file_generation_reasoning_model_gets_headroom_clamped_to_ctx(monkeypatch, caps):
    monkeypatch.setattr(workflow, "OLLAMA_NUM_CTX", 4096)
    rec = _install(monkeypatch, "```ts\nx\n```")
    workflow._generate_one_file("a.ts", "a", "typescript", _agent("fe", REASONING_MODEL), "http://x", None)
    assert rec.instances[0]["num_predict"] == 4096   # 3000 × 2 kẹp ở num_ctx


@pytest.mark.parametrize("think,expected_cap", [(True, 6000), (False, 3000)])
def test_filegen_think_applies_to_file_generation(monkeypatch, caps, think, expected_cap):
    monkeypatch.setattr(llm, "FILEGEN_THINK", think)
    rec = _install(monkeypatch, "```ts\nx\n```")
    # Model reasoning: FILEGEN_THINK=false tắt thinking → bỏ headroom; =true giữ headroom.
    workflow._generate_one_file("a.ts", "a", "typescript", _agent("fe", REASONING_MODEL), "http://x", None)
    kw = rec.instances[0]
    assert kw["reasoning"] is think
    assert kw["num_predict"] == expected_cap


def test_filegen_think_not_applied_to_planner_or_report(monkeypatch, caps):
    monkeypatch.setattr(llm, "FILEGEN_THINK", False)
    rec = _install(monkeypatch, "a long enough answer for the report path, yes")
    workflow._call_agent(_agent("ba", PLAIN_MODEL), "http://x", "ctx")
    rec2 = _install(monkeypatch, '[{"filename": "a.ts"}]')
    workflow._plan_code_files("fe", workflow.AGENTS["fe"], "http://x", "t", None, "c")
    assert "reasoning" not in rec.instances[0]
    assert "reasoning" not in rec2.instances[0]


# ── Báo cáo một lần (_call_agent) ───────────────────────────────────────────────


def test_report_call_uses_report_cap_and_retry_reuses_it(monkeypatch, caps):
    rec = _install(monkeypatch, "", "a long enough answer for the retry path")
    out = workflow._call_agent(_agent("ba", PLAIN_MODEL), "http://x", "ctx " * 100)
    assert out == "a long enough answer for the retry path"
    assert len(rec.prompts) == 2                     # lần chính + retry output ngắn
    (kw,) = rec.instances                           # retry dùng lại cùng instance
    assert kw["num_predict"] == 5000
    assert kw["client_kwargs"] == {"timeout": 321.0}


def test_report_call_reasoning_model_headroom(monkeypatch, caps):
    rec = _install(monkeypatch, "<think>plan</think>" + "a long enough answer " * 3)
    out = workflow._call_agent(_agent("ba", REASONING_MODEL), "http://x", "ctx")
    assert out.startswith("a long enough answer")
    assert rec.instances[0]["num_predict"] == 10000


def test_report_truncated_thinking_triggers_short_output_retry(monkeypatch, caps):
    # Chạm trần giữa <think> → sau khi bóc chỉ còn chuỗi rỗng → retry, không trả nháp.
    rec = _install(
        monkeypatch,
        SimpleNamespace(content="<think>" + "reasoning " * 50, response_metadata={"done_reason": "length"}),
        "final answer that is definitely long enough",
    )
    out = workflow._call_agent(_agent("ba", REASONING_MODEL), "http://x", "ctx")
    assert out == "final answer that is definitely long enough"
    assert len(rec.prompts) == 2


def test_artifact_fallback_single_call_uses_report_cap(monkeypatch, caps):
    monkeypatch.setattr(workflow, "_plan_code_files", lambda *a, **k: [])
    rec = _install(monkeypatch, "fallback single-call output long enough to keep")
    out = workflow._generate_artifacts_multi_turn(
        "be", _agent("be", PLAIN_MODEL), "http://x", "ctx", "task", ["nestjs"],
    )
    assert out.startswith("fallback single-call output")
    assert rec.instances[0]["num_predict"] == 5000


def test_no_cap_when_env_is_zero(monkeypatch, caps):
    monkeypatch.setattr(llm, "REPORT_NUM_PREDICT", 0)
    rec = _install(monkeypatch, "a long enough answer for the report path, yes")
    workflow._call_agent(_agent("ba", REASONING_MODEL), "http://x", "ctx")
    assert "num_predict" not in rec.instances[0]


# ── _strip_thinking ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("<think>a</think>answer", "answer"),
    ("<THINK>a</THINK> one <think>b</think> two", "one  two"),
    ("<think>cut off mid reasoning", ""),
    ("answer <think>trailing cut", "answer"),
    ("template-opened reasoning</think>answer", "answer"),
    ("no tags at all", "no tags at all"),
])
def test_strip_thinking(raw, expected):
    assert workflow._strip_thinking(raw) == expected


# ── Output rỗng sau retry → lỗi, không phải "thành công rỗng" ───────────────────


def test_report_empty_after_retry_raises(monkeypatch, caps):
    rec = _install(monkeypatch, "   ", "")
    with pytest.raises(workflow.EmptyAgentOutputError, match="output rỗng"):
        workflow._call_agent(_agent("ba", PLAIN_MODEL), "http://x", "ctx")
    assert len(rec.prompts) == 2                     # vẫn retry một lần trước khi bỏ cuộc


def test_report_truncated_thinking_twice_raises(monkeypatch, caps):
    cut = SimpleNamespace(content="<think>" + "reasoning " * 50, response_metadata={"done_reason": "length"})
    _install(monkeypatch, cut, cut)
    with pytest.raises(workflow.EmptyAgentOutputError):
        workflow._call_agent(_agent("ba", REASONING_MODEL), "http://x", "ctx")


def test_report_short_but_non_empty_after_retry_is_kept(monkeypatch, caps):
    # < 30 ký tự nhưng có nội dung → giữ nguyên (không đánh đồng với lỗi).
    _install(monkeypatch, "", "OK")
    assert workflow._call_agent(_agent("ba", PLAIN_MODEL), "http://x", "ctx") == "OK"


def test_node_marks_empty_report_as_failed_step(monkeypatch, caps):
    _install(monkeypatch, "")
    monkeypatch.setitem(workflow.AGENTS, "ba", _agent("ba", PLAIN_MODEL))
    result = workflow._build_node("ba")({"user_input": "Build an app", "step_outputs": {}})
    out = result["step_outputs"]["ba"]
    assert out.startswith("[LỖI trong ba]") and "output rỗng" in out
    assert result["error"].startswith("Bước 'ba' thất bại:")
    assert result["completed_steps"] == ["ba"]


def test_artifact_fallback_empty_single_call_fails_the_step(monkeypatch, caps):
    monkeypatch.setattr(workflow, "_plan_code_files", lambda *a, **k: [])
    _install(monkeypatch, "")
    with pytest.raises(workflow.EmptyAgentOutputError):
        workflow._generate_artifacts_multi_turn(
            "be", _agent("be", PLAIN_MODEL), "http://x", "ctx", "task", ["nestjs"],
        )


def test_report_empty_retry_falls_back_to_short_first_reply(monkeypatch, caps):
    rec = _install(monkeypatch, "OK done", "   ")
    out = workflow._call_agent(_agent("ba", PLAIN_MODEL), "http://x", "ctx")
    assert out == "OK done"                          # không đánh rơi nội dung lần đầu
    assert len(rec.prompts) == 2
