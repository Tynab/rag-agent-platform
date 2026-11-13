"""Regression test cho các lỗi nhỏ của workflow.py (lane engine — small-bugs).

- (a) _deloop chèn marker cắt lặp đúng cú pháp comment của ngôn ngữ (không còn "//"
  cố định làm hỏng YAML/Python/SQL/Markdown); JSON không có marker.
- (b) Planner / fallback nhận TOÀN BỘ TL task (cắt chữ task, không cắt danh sách).
- (c) Context gửi planner và review snippet giữ đề bài + tech stack (_compact_context).
- (d) DA có gợi ý DB cho stack 'both'.

Không gọi Ollama thật: llm.ChatOllama được thay bằng fake ghi lại prompt.
"""

import json
from types import SimpleNamespace

import pytest

workflow = pytest.importorskip("workflow")
llm = pytest.importorskip("llm")

TECH = ["nestjs", "reactjs", "mongodb"]
USER_INPUT = "Build an order management app"
_FILLER = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 12
MARK = "... (repeated pattern truncated)"


class FakeChat:
    """Thay cho llm.ChatOllama: trả JSON kế hoạch cho prompt planner, nội dung file cho
    prompt sinh file; ghi lại mọi prompt (HumanMessage cuối) theo thứ tự gọi."""

    def __init__(self, plan_reply: str, file_reply: str = "```ts\nexport const a = 1;\n```"):
        self.plan_reply = plan_reply
        self.file_reply = file_reply
        self.plan_prompts: list[str] = []
        self.file_prompts: list[str] = []

    def __call__(self, **kwargs):
        fake = self

        class _Chat:
            def invoke(self, messages):
                prompt = messages[-1].content
                if "Return ONLY a JSON array" in prompt:
                    fake.plan_prompts.append(prompt)
                    return SimpleNamespace(content=fake.plan_reply)
                fake.file_prompts.append(prompt)
                return SimpleNamespace(content=fake.file_reply)

        return _Chat()


def _install(monkeypatch, plan_reply, file_reply="```ts\nexport const a = 1;\n```"):
    fake = FakeChat(plan_reply, file_reply)
    monkeypatch.setattr(llm, "ChatOllama", fake)
    return fake


# ── (a) _deloop: marker theo ngôn ngữ ───────────────────────────────────────────


def _repeated(line: str, n: int = 20) -> str:
    return "\n".join([line] * n)


@pytest.mark.parametrize("language,filename,marker", [
    ("typescript", "src/a.ts", "// " + MARK),
    ("yaml", "k8s/deploy.yml", "# " + MARK),
    ("python", "app/main.py", "# " + MARK),
    ("dockerfile", "Dockerfile", "# " + MARK),
    ("sql", "schema.sql", "-- " + MARK),
    ("markdown", "README.md", "<!-- " + MARK + " -->"),
    ("html", "index.html", "<!-- " + MARK + " -->"),
    ("css", "styles.css", "/* " + MARK + " */"),
    ("TypeScript (React)", "", "// " + MARK),        # language tự do của planner
    # language không nhận ra ("text" mặc định) → tra theo tên / đuôi file.
    ("text", "deploy.yaml", "# " + MARK),
    ("text", "Dockerfile", "# " + MARK),
    ("text", "db/001_init.sql", "-- " + MARK),
    ("", ".env.example", "# " + MARK),
])
def test_deloop_marker_uses_language_comment_syntax(language, filename, marker):
    out = workflow._deloop(_repeated("x = 1"), language, filename=filename).splitlines()
    assert out[:8] == ["x = 1"] * 8
    assert out[8] == marker
    assert len(out) == 9


@pytest.mark.parametrize("language,filename", [
    ("json", "package.json"),
    ("typescript", "tsconfig.json"),     # file .json: không bao giờ chèn comment
    ("text", "notes.txt"),               # không rõ cú pháp comment
    ("vue", "src/App.vue"),              # trộn HTML + script: không có cú pháp an toàn
    ("", ""),
])
def test_deloop_without_safe_comment_syntax_only_truncates(language, filename):
    out = workflow._deloop(_repeated('  "a",'), language, filename=filename).splitlines()
    assert out == ['  "a",'] * 8
    assert MARK not in "\n".join(out)


def test_deloop_marker_keeps_indentation_and_files_stay_valid():
    yaml = pytest.importorskip("yaml")
    yml = "items:\n" + _repeated("  - same") + "\nother: 1\n"
    out = workflow._deloop(yml, "yaml")
    assert "  # " + MARK in out.splitlines()
    assert yaml.safe_load(out) == {"items": ["same"] * 8, "other": 1}

    py = "def f():\n" + _repeated("    x = 1") + "\n    return x\n"
    out = workflow._deloop(py, "python")
    assert "    # " + MARK in out.splitlines()
    compile(out, "f.py", "exec")         # marker "//" cũ → SyntaxError

    js = "[\n" + _repeated('  "a",') + '\n  "b"\n]\n'
    assert json.loads(workflow._deloop(js, "json")) == ["a"] * 8 + ["b"]


def test_generate_one_file_passes_language_to_deloop(monkeypatch):
    body = "\n".join(["services:"] + ["  - api"] * 20)
    _install(monkeypatch, "[]", file_reply=f"```yaml\n{body}\n```")
    out = workflow._generate_one_file(
        "docker-compose.yml", "compose", "yaml", workflow.AGENTS["devsecops"], "http://x", TECH,
    )
    assert "  # " + MARK in out.splitlines()
    assert "//" not in out


# ── (b) planner / fallback nhận toàn bộ TL task ────────────────────────────────


def _be_board(tasks: list[str]) -> str:
    rows = "\n".join(f"| BE-{i:02d} | {t} | core | Setup | 1 | P1 | S1 |" for i, t in enumerate(tasks, 1))
    return (
        "## 6. BE Task Board\n"
        "| Task ID | Task | Module | Type | Est. (days) | Priority | Sprint |\n"
        "|---|---|---|---|---|---|---|\n" + rows + "\n\n## 7. DBA Task Board\n"
    )


_MANY_TASKS = [f"Implement widget{i:02d} endpoint" for i in range(1, 41)]


def _context(role="be", tech_stack=TECH, extra_context=None):
    step_outputs = {r: f"{r.upper()} " + _FILLER for r in workflow.WORKFLOW_STEPS}
    parts = workflow._build_context_parts(
        role=role, agent=workflow.AGENTS[role], user_input=USER_INPUT,
        step_outputs=step_outputs, tech_stack=tech_stack, dep_max_chars=3_000,
        extra_context=extra_context,
    )
    return "\n".join(parts)


def test_format_tl_task_lines_keeps_every_task():
    tasks = [{"task": t} for t in _MANY_TASKS]
    lines = workflow._format_tl_task_lines(tasks).splitlines()
    assert lines == [f"  - {t}" for t in _MANY_TASKS]


def test_format_tl_task_lines_truncates_text_not_list():
    tasks = [{"task": f"T{i:02d} " + "long words " * 40} for i in range(40)]
    out = workflow._format_tl_task_lines(tasks)
    lines = out.splitlines()
    assert len(lines) == 40
    assert all(ln.endswith("…") for ln in lines)
    assert [ln[4:7] for ln in lines] == [f"T{i:02d}" for i in range(40)]
    assert len(out) <= workflow._TL_TASKS_PROMPT_CHARS


def test_format_tl_task_lines_does_not_cut_when_board_fits():
    # Board nhỏ có task rất dài: vừa budget → gửi nguyên văn (không có trần độ rộng).
    long_task = "Implement order export " + "with filters and pagination " * 10
    assert len(long_task) > 250
    tasks = [{"task": "Bootstrap service"}, {"task": long_task}, {"task": "Health check"}]
    assert workflow._format_tl_task_lines(tasks).splitlines()[1] == f"  - {long_task.strip()}"

    # 40 task lệch độ dài: task ngắn nhường ngân sách → task 150 ký tự vẫn nguyên văn.
    mixed = [{"task": f"Short task {i:02d}"} for i in range(36)]
    longs = [f"Long task {i} " + "covering several keywords " * 5 for i in range(4)]
    mixed += [{"task": t} for t in longs]
    out = workflow._format_tl_task_lines(mixed)
    assert "…" not in out
    for t in longs:
        assert f"  - {t.strip()}" in out.splitlines()
    assert len(out) <= workflow._TL_TASKS_PROMPT_CHARS


def test_format_tl_task_lines_cuts_only_the_longest_tasks():
    short = [{"task": f"Short task {i:02d}"} for i in range(30)]
    huge = [{"task": f"Huge task {i} " + "x" * 2_000} for i in range(3)]
    out = workflow._format_tl_task_lines(short + huge)
    lines = out.splitlines()
    assert lines[:30] == [f"  - Short task {i:02d}" for i in range(30)]
    assert all(ln.endswith("…") and len(ln) > 200 for ln in lines[30:])
    assert len(out) <= workflow._TL_TASKS_PROMPT_CHARS


def test_format_tl_task_lines_pathological_board_stays_bounded():
    tasks = [{"task": f"Task number {i} " + "x" * 80} for i in range(1_000)]
    out = workflow._format_tl_task_lines(tasks)
    assert len(out) <= 2 * workflow._TL_TASKS_PROMPT_CHARS + 100
    assert out.splitlines()[-1].endswith("more Team Lead tasks not listed)")
    assert workflow._format_tl_task_lines([]) == ""


def test_planner_prompt_lists_all_tl_tasks(monkeypatch):
    fake = _install(monkeypatch, '[{"filename": "src/app.module.ts", "language": "typescript"}]')
    workflow._generate_artifacts_multi_turn(
        "be", workflow.AGENTS["be"], "http://x", _context(), USER_INPUT, TECH, "",
        tl_output=_be_board(_MANY_TASKS),
    )
    first_plan = fake.plan_prompts[0]
    for task in _MANY_TASKS:                       # tl_tasks[:15] cũ dừng ở widget15
        assert f"  - {task}\n" in first_plan


def test_fallback_single_call_lists_all_tl_tasks(monkeypatch):
    _install(monkeypatch, "no plan today")
    captured = {}
    monkeypatch.setattr(workflow, "_call_agent", lambda agent, url, ctx: captured.setdefault("ctx", ctx))
    out = workflow._generate_artifacts_multi_turn(
        "be", workflow.AGENTS["be"], "http://x", _context(extra_context="rag chunk " * 400),
        USER_INPUT, TECH, "", tl_output=_be_board(_MANY_TASKS),
    )
    fb = captured["ctx"]
    for task in _MANY_TASKS:
        assert f"- {task}" in fb
    assert USER_INPUT in fb and "- nestjs" in fb
    header = "\n\n## TL Task Board — bắt buộc cover\n"
    assert len(fb) <= 1_000 + len(header) + workflow._TL_TASKS_PROMPT_CHARS
    # Checklist cũng có đủ 40 task — nhất quán với danh sách gửi model.
    assert sum(1 for ln in out.splitlines() if ln.startswith("| BE-")) == 40


# ── (c) context planner / review snippet giữ đề bài + tech stack ──────────────


def test_planner_context_keeps_head_and_tail(monkeypatch):
    fake = _install(monkeypatch, '[{"filename": "src/app.module.ts", "language": "typescript"}]')
    ctx = _context(extra_context="rag chunk " * 400)
    workflow._generate_artifacts_multi_turn(
        "be", workflow.AGENTS["be"], "http://x", ctx, USER_INPUT, TECH, "", tl_output="",
    )
    summary = fake.plan_prompts[0].split("Context summary:\n", 1)[1]
    # "## ..." chỉ có trong context đã rút gọn (Task:/Tech stack: của template không có).
    assert summary.startswith("## Mục tiêu kinh doanh")
    assert "## Công nghệ bắt buộc" in summary and "- mongodb" in summary
    assert summary.endswith("rag chunk \n")             # đuôi context vẫn giữ
    assert len(summary) - 1 <= workflow._PLAN_CTX_CHARS


def test_plan_code_files_does_not_cut_compacted_tail(monkeypatch):
    fake = _install(monkeypatch, "[]")
    summary = workflow._compact_context(_context(extra_context="rag chunk " * 400), workflow._PLAN_CTX_CHARS)
    workflow._plan_code_files("be", workflow.AGENTS["be"], "http://x", USER_INPUT, TECH, summary)
    assert fake.plan_prompts[0].endswith(summary + "\n")


def test_review_role_file_context_keeps_head_and_tail(monkeypatch):
    fake = _install(
        monkeypatch, '[{"filename": "ARCHITECTURE.md", "language": "markdown"}]',
        file_reply="# Architecture\n\nNotes.",
    )
    ctx = _context(role="tech_lead", extra_context="rag chunk " * 400)
    workflow._generate_artifacts_multi_turn(
        "tech_lead", workflow.AGENTS["tech_lead"], "http://x", ctx, USER_INPUT, TECH, "",
    )
    snippet = fake.file_prompts[0].split("Review context (previous agents):\n", 1)[1]
    assert snippet.startswith("## Mục tiêu kinh doanh")
    assert "## Công nghệ bắt buộc" in snippet and "- nestjs" in snippet
    assert snippet.endswith("rag chunk ")
    assert len(snippet) <= workflow._REVIEW_CTX_CHARS


# ── (d) gợi ý DB cho DA khi stack 'both' ───────────────────────────────────────

BOTH = ["fastapi", "postgresql", "mongodb"]


def test_da_gets_db_hint_for_both_stack():
    assert workflow._detect_db_type(BOTH) == "both"
    da = _context(role="da", tech_stack=BOTH)
    assert "## Data Analysis: Mixed (SQL + NoSQL)" in da
    assert "aggregation pipeline" in da and "GROUP BY" in da
    # DBA vẫn giữ gợi ý 'both' như trước.
    assert "## DB Schema: Mixed (SQL + NoSQL)" in _context(role="dba", tech_stack=BOTH)


@pytest.mark.parametrize("stack,header", [
    (["postgresql", "redis"], "## Data Analysis: SQL\n"),         # redis là store phụ trợ
    (["mongodb"], "## Data Analysis: NoSQL Only\n"),
])
def test_da_single_store_hints_unchanged(stack, header):
    da = _context(role="da", tech_stack=stack)
    assert header in da
    assert "Mixed (SQL + NoSQL)" not in da
