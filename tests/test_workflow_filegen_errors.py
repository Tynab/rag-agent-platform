"""Lỗi sinh file có cấu trúc (FileGenerationError) — không lưu thông báo lỗi thành source.

Hợp đồng với UI / runtime:
- File lỗi KHÔNG được ghi thành section "### FILE:" (artifacts.py sẽ lưu nó thành file).
- Output role có section bắt đầu đúng bằng "\\n## ⚠️ File generation errors\\n" liệt kê
  từng file lỗi + lỗi ngắn.
- Không file nào sinh được → output bắt đầu "[LỖI trong {role}]" (app.py → failed_steps).
- File lỗi không tính là cover TL task (checklist / follow-up).
"""

import dataclasses
from types import SimpleNamespace

import pytest

workflow = pytest.importorskip("workflow")
artifacts = pytest.importorskip("artifacts")
llm = pytest.importorskip("llm")

HEADER = "\n## ⚠️ File generation errors\n"
TECH = ["nestjs", "mongodb"]

TL_OUTPUT = """## 6. BE Task Board
| Task ID | Task | Module | Type | Est. (days) | Priority | Sprint |
|---|---|---|---|---|---|---|
| BE-01 | Bootstrap NestJS service | core | Setup | 1 | P0 | S1 |
| BE-02 | Implement POST /api/auth/login | auth | API Endpoint | 2 | P1 | S1 |

## 7. DBA Task Board
| # | Task | Type | Priority |
|---|---|---|---|
| 1 | Create users collection schema | Schema | P0 |
"""

MAIN = {"filename": "src/main.ts", "description": "bootstrap nestjs service", "language": "typescript"}
LOGIN = {"filename": "src/auth/login.controller.ts", "description": "login endpoint", "language": "typescript"}
USERS = {"filename": "src/users/users.module.ts", "description": "users module", "language": "typescript"}


def test_header_constant_is_the_exact_contract_string():
    assert workflow.FILE_GEN_ERRORS_HEADER == HEADER
    assert workflow.FILE_GEN_ERRORS_HEADER == "\n## ⚠️ File generation errors\n"


# ── _generate_one_file: raise thay vì trả comment lỗi ───────────────────────────


class _FakeChat:
    def __init__(self, reply):
        self._reply = reply

    def __call__(self, **kwargs):
        reply = self._reply

        class _Chat:
            def invoke(self, messages):
                if isinstance(reply, Exception):
                    raise reply
                return reply if isinstance(reply, SimpleNamespace) else SimpleNamespace(content=reply)

        return _Chat()


def _gen_one(monkeypatch, reply, model="llama3.1:8b"):
    monkeypatch.setattr(llm, "ChatOllama", _FakeChat(reply))
    agent = dataclasses.replace(workflow.AGENTS["be"], model=model)
    return workflow._generate_one_file("src/a.ts", "a", "typescript", agent, "http://x", TECH)


def test_generate_one_file_raises_on_llm_error(monkeypatch):
    with pytest.raises(workflow.FileGenerationError) as ei:
        _gen_one(monkeypatch, TimeoutError("timed out\nwhile reading"))
    err = ei.value
    assert err.filename == "src/a.ts"
    assert err.reason == "TimeoutError: timed out while reading"   # một dòng
    assert isinstance(err.__cause__, TimeoutError)


@pytest.mark.parametrize("reply", ["", "   \n", "```ts\n```"])
def test_generate_one_file_raises_on_empty_content(monkeypatch, reply):
    with pytest.raises(workflow.FileGenerationError, match="no file content"):
        _gen_one(monkeypatch, reply)


def test_generate_one_file_truncated_thinking_is_a_failure_not_a_file(monkeypatch):
    reply = SimpleNamespace(content="<think>" + "let me think " * 100, response_metadata={"done_reason": "length"})
    with pytest.raises(workflow.FileGenerationError) as ei:
        _gen_one(monkeypatch, reply, model="qwen3.6:35b")
    assert "hit FILEGEN_NUM_PREDICT" in ei.value.reason


def test_generate_one_file_success_still_returns_content(monkeypatch):
    assert _gen_one(monkeypatch, "```typescript\nexport const a = 1;\n```") == "export const a = 1;"


def test_short_error_is_single_line_and_truncated():
    msg = workflow._short_error(RuntimeError("x" * 500 + "\n```\n### FILE: evil.ts"))
    assert "\n" not in msg and "`" not in msg
    assert len(msg) <= workflow._FILE_ERR_MAX_CHARS
    assert msg.startswith("RuntimeError: xxx")
    assert workflow._short_error(RuntimeError()) == "RuntimeError"


# ── _generate_artifacts_multi_turn ─────────────────────────────────────────────


def _setup(monkeypatch, plans, failing, retry_ok=()):
    """Planner trả lần lượt *plans*; file trong *failing* raise (trừ lần thử lại nằm trong *retry_ok*)."""
    plan_calls: list[list[str]] = []
    gen_calls: list[str] = []
    plans = list(plans)

    def fake_plan(role, *args, tl_tasks=None, **kwargs):
        plan_calls.append([t["task"] for t in (tl_tasks or [])])
        return plans.pop(0) if plans else []

    def fake_gen(filename, description, language, *args, **kwargs):
        attempt = gen_calls.count(filename)
        gen_calls.append(filename)
        if filename in failing and not (attempt >= 1 and filename in retry_ok):
            raise workflow.FileGenerationError(filename, "ReadTimeout: timed out")
        return f"// content of {filename}"

    monkeypatch.setattr(workflow, "_plan_code_files", fake_plan)
    monkeypatch.setattr(workflow, "_generate_one_file", fake_gen)
    return plan_calls, gen_calls


def _run(role="be", tl_output=TL_OUTPUT):
    return workflow._generate_artifacts_multi_turn(
        role, workflow.AGENTS[role], "http://x", "ctx", "Build an app", TECH, "", tl_output=tl_output,
    )


def _file_sections(output):
    return [ln.removeprefix("### FILE: ") for ln in output.splitlines() if ln.startswith("### FILE: ")]


def _error_lines(output):
    _, _, tail = output.partition(HEADER)
    return [ln for ln in tail.splitlines() if ln.startswith("- `")]


def test_partial_failure_skips_file_section_and_lists_error(monkeypatch):
    plan_calls, _ = _setup(monkeypatch, [[MAIN, LOGIN, USERS], []], failing={LOGIN["filename"]})
    out = _run()

    assert not out.startswith("[LỖI")
    assert _file_sections(out) == [MAIN["filename"], USERS["filename"]]
    assert "Error generating" not in out
    assert out.count(HEADER) == 1
    (line,) = _error_lines(out)
    assert line == "- `src/auth/login.controller.ts` — ReadTimeout: timed out"

    # File lỗi không cover task → follow-up được gọi với task login; checklist ⏳.
    assert plan_calls[1] == ["Implement POST /api/auth/login"]
    checklist = out[out.index("Task Completion Checklist"):]
    assert "| BE-01 | Bootstrap NestJS service | P0 | ✅ Done | `src/main.ts` |" in checklist
    assert "| BE-02 | Implement POST /api/auth/login | P1 | ⏳ Partial | — |" in checklist
    # Section lỗi nằm trước checklist.
    assert out.index(HEADER) < out.index("Task Completion Checklist")


def test_partial_failure_artifacts_do_not_contain_failed_file(monkeypatch, tmp_path):
    _setup(monkeypatch, [[MAIN, LOGIN, USERS], []], failing={LOGIN["filename"]})
    out = _run()
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    saved = {a["filename"] for a in artifacts.extract_and_save("be", out, "wf1")}
    assert saved == {"_output.md", MAIN["filename"], USERS["filename"]}


def test_followup_retry_success_removes_file_from_error_list(monkeypatch):
    _, gen_calls = _setup(
        monkeypatch, [[MAIN, LOGIN], [LOGIN, MAIN]],
        failing={LOGIN["filename"]}, retry_ok={LOGIN["filename"]},
    )
    out = _run()
    assert gen_calls == [MAIN["filename"], LOGIN["filename"], LOGIN["filename"]]   # MAIN không sinh lại
    assert _file_sections(out) == [MAIN["filename"], LOGIN["filename"]]
    assert HEADER not in out
    assert "| BE-02 | Implement POST /api/auth/login | P1 | ✅ Done |" in out


def test_followup_retry_failure_listed_once(monkeypatch):
    _, gen_calls = _setup(monkeypatch, [[MAIN, LOGIN], [LOGIN, LOGIN]], failing={LOGIN["filename"]})
    out = _run()
    assert gen_calls.count(LOGIN["filename"]) == 2       # pha 1 + đúng một lần thử lại
    assert _file_sections(out) == [MAIN["filename"]]
    assert len(_error_lines(out)) == 1


def test_filename_never_both_section_and_error(monkeypatch):
    # Planner liệt kê trùng tên: lần đầu thành công → lần sau bị bỏ qua, không thành lỗi.
    _setup(monkeypatch, [[MAIN, dict(MAIN), USERS], []], failing=set())
    out = _run(tl_output=None)
    assert _file_sections(out) == [MAIN["filename"], USERS["filename"]]
    assert HEADER not in out


def test_all_files_failed_marks_step_as_error(monkeypatch):
    _setup(monkeypatch, [[MAIN, LOGIN], []], failing={MAIN["filename"], LOGIN["filename"]})
    out = _run()
    assert out.startswith("[LỖI trong be]")
    assert "### FILE:" not in out
    assert out.count(HEADER) == 1
    assert len(_error_lines(out)) == 2


def test_all_failed_without_tl_tasks(monkeypatch):
    _setup(monkeypatch, [[MAIN]], failing={MAIN["filename"]})
    out = _run(role="devsecops", tl_output=None)
    assert out.startswith("[LỖI trong devsecops]")
    assert HEADER in out


def test_error_text_cannot_inject_file_sections_or_fences(monkeypatch, tmp_path):
    # Thông điệp lỗi và tên file (do LLM đặt) có thể chứa backtick, xuống dòng, directive.
    evil_msg = "boom\n```ts\n### FILE: evil.ts\n```"
    evil_name = "x`y\n### FILE: evil2.ts"

    def fake_gen(filename, *args, **kwargs):
        if filename != MAIN["filename"]:
            raise workflow.FileGenerationError(filename, evil_msg)
        return "// ok"

    monkeypatch.setattr(workflow, "_plan_code_files", lambda *a, **k: [MAIN, LOGIN, {**USERS, "filename": evil_name}])
    monkeypatch.setattr(workflow, "_generate_one_file", fake_gen)
    out = _run(tl_output=None)
    errors = out.partition(HEADER)[2]
    assert "```" not in errors and "### FILE:" not in errors
    assert len(_error_lines(out)) == 2 and len(errors.strip().splitlines()) == 4   # mô tả + dòng trống + 2 lỗi
    assert _file_sections(out) == [MAIN["filename"]]
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    saved = {a["filename"] for a in artifacts.extract_and_save("be", out, "wf1")}
    assert saved == {"_output.md", MAIN["filename"]}


# ── Node / run_single_step ──────────────────────────────────────────────────────


def _state():
    return {"user_input": "Build an app", "step_outputs": {"tl": TL_OUTPUT}, "tech_stack": TECH}


def test_node_all_failed_sets_error_and_prefix(monkeypatch):
    _setup(monkeypatch, [[MAIN, LOGIN], []], failing={MAIN["filename"], LOGIN["filename"]})
    result = workflow._build_node("be")(_state())
    output = result["step_outputs"]["be"]
    assert output.startswith("[LỖI trong be]")
    assert result["error"].startswith("Bước 'be' thất bại: Không sinh được file nào")
    assert result["completed_steps"] == ["be"]


def test_node_partial_failure_is_not_a_step_error(monkeypatch):
    _setup(monkeypatch, [[MAIN, LOGIN], []], failing={LOGIN["filename"]})
    result = workflow._build_node("be")(_state())
    assert not result["step_outputs"]["be"].startswith("[LỖI")
    assert HEADER in result["step_outputs"]["be"]
    assert "error" not in result


def test_node_exception_path_unchanged(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(workflow, "_generate_artifacts_multi_turn", boom)
    result = workflow._build_node("be")(_state())
    assert result["step_outputs"]["be"] == "[LỖI trong be] ollama unreachable"
    assert result["error"] == "Bước 'be' thất bại: ollama unreachable"


def test_run_single_step_all_failed_returns_error_prefix(monkeypatch):
    _setup(monkeypatch, [[MAIN], []], failing={MAIN["filename"]})
    out = workflow.run_single_step(
        "be", "Build an app", prev_outputs={"tl": TL_OUTPUT}, tech_stack=TECH, rag_enabled=False,
    )
    assert out.startswith("[LỖI trong be]")
