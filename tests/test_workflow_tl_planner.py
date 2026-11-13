"""Regression test cho TL task board parsing (F7) và planner / fallback context (F8).

Không gọi Ollama thật: llm.ChatOllama, _plan_code_files và _call_agent được monkeypatch.
"""

from types import SimpleNamespace

import pytest

workflow = pytest.importorskip("workflow")
llm = pytest.importorskip("llm")

_FILLER = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 12

# TL output theo đúng template §1–§10 của TL agent, heading Markdown "## N.".
TL_OUTPUT = f"""## 1. Engineering Summary
{_FILLER}
## 2. Technical Research Spikes Required
| Spike ID | Title | Assigned Team | Description | Blocking For | Est. (days) | Must Resolve Before Sprint |
|---|---|---|---|---|---|---|
| SP-01 | OAuth flow research | BE | Google OAuth | BE #2 | 2 | S1 |
{_FILLER}
## 3. Dependency Map
| Task/Feature | Depends On | Team Owner | Blocks | Notes |
|---|---|---|---|---|
| Login API | SP-01 | BE | → see §4 FE Task Board #2 | contract |
| Orders API contract | Login API | BE | FE | none |
{_FILLER}
## 4. FE Task Board
| # | Task | Type | Est. (days) | Priority | Sprint | Depends On | Acceptance Criteria |
|---|------|------|-------------|----------|--------|------------|---------------------|
| 1 | Setup Vite React project | Setup | 1 | P0 | S1 | | builds |
| 2 | Implement login page form | Component | 2 | P0 | S1 | 1 | Designer §5 S-01 |

## 5. Mobile Task Board
| # | Task | Type | Est. (days) | Priority | Sprint | Depends On | Acceptance Criteria |
|---|------|------|-------------|----------|--------|------------|---------------------|
| 1 | Setup Flutter project skeleton | Setup | 1 | P0 | S1 | — | runs |

## 6. BE Task Board
| Task ID | Task | Module | Type | Est. (days) | Priority: P0/P1/P2 | Sprint | Depends On | Acceptance Criteria |
|---|---|---|---|---|---|---|---|---|
| BE-01 | Bootstrap NestJS service | core | Setup | 1 | **P0** | S1 | — | boots |
| BE-02 | Implement POST /api/auth/login | auth | API Endpoint | 2 | P1 (High) | S1 | BE-01 | 200 + JWT |

## 7. DBA Task Board
| # | Task | Type | Est. (days) | Priority | Sprint | Depends On | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Create users collection schema | Schema | 1 | P0 | S1 | — | tenantId first |

## 8. Sprint Allocation Plan
| Sprint | FE Focus | Mobile Focus | BE Focus | DBA Focus | Cross-team Milestones |
|---|---|---|---|---|---|
| S1 | Setup and login | Setup | Auth | Schema | Contract freeze |

## 10. Engineering Risks & Mitigations
| Risk | Team | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| OAuth delays | BE | M | H | spike early | TL |
"""

TECH = ["nestjs", "reactjs", "mongodb"]
USER_INPUT = "Build an order management app"


def _tasks(role, text):
    return [(t["id"], t["task"], t["priority"]) for t in workflow._extract_tl_tasks(role, text)]


# ── F7: _extract_tl_tasks ──────────────────────────────────────────────────────


def test_tl_section_stops_at_next_markdown_heading():
    # "## 5. Mobile" phải kết thúc section FE — không kéo task Mobile/BE/DBA/Sprint/Risk vào.
    assert _tasks("fe", TL_OUTPUT) == [
        ("1", "Setup Vite React project", "P0"),
        ("2", "Implement login page form", "P0"),
    ]
    assert _tasks("mobile", TL_OUTPUT) == [("1", "Setup Flutter project skeleton", "P0")]
    assert _tasks("dba", TL_OUTPUT) == [("1", "Create users collection schema", "P0")]


def test_tl_header_row_maps_columns_instead_of_counting_as_task():
    # Header "| Task ID | Task | ..." không bị đếm là task; tên task lấy từ cột "Task"
    # (không phải ID "BE-01"); priority lấy từ cột "Priority..." kể cả khi in đậm / có chú thích.
    assert _tasks("be", TL_OUTPUT) == [
        ("BE-01", "Bootstrap NestJS service", "P0"),
        ("BE-02", "Implement POST /api/auth/login", "P1"),
    ]


def test_tl_section_ignores_cross_reference_in_other_table():
    # "→ see §4 FE Task Board" nằm trong bảng §3 Dependency Map, xuất hiện TRƯỚC heading §4.
    tasks = _tasks("fe", TL_OUTPUT)
    assert all("API" not in t[1] and "contract" not in t[1] for t in tasks)


def test_tl_subheadings_stay_inside_section():
    text = (
        "### 4. FE Task Board\n#### Sprint 1\n"
        "| # | Task | Priority |\n|---|---|---|\n| 1 | Setup React app shell | P0 |\n"
        "#### Sprint 2\n| 2 | Orders list page | P1 |\n"
        "### 5. Mobile Task Board\n| # | Task |\n|---|---|\n| 1 | Flutter bootstrap app |\n"
    )
    assert [t[1] for t in _tasks("fe", text)] == ["Setup React app shell", "Orders list page"]


def test_tl_bold_and_plain_numbered_headings():
    bold = (
        "**4. FE Task Board**\n**Sprint 1**\n"
        "| # | Task | Type | Est | Priority |\n|---|---|---|---|---|\n"
        "| 1 | Setup React app shell | Setup | 1 | P0 |\n"
        "1. Note: follow the order above\n"
        "| 2 | Wire router skeleton | Routing | 1 | P1 |\n"
        "**5. Mobile Task Board**\n"
        "| # | Task | Type | Est | Priority |\n|---|---|---|---|---|\n"
        "| 1 | Flutter bootstrap app | Setup | 1 | P0 |\n"
    )
    assert [t[1] for t in _tasks("fe", bold)] == ["Setup React app shell", "Wire router skeleton"]
    assert [t[1] for t in _tasks("mobile", bold)] == ["Flutter bootstrap app"]

    # Bảng không có header → heuristic cũ theo vị trí cột vẫn hoạt động.
    plain = (
        "4. FE Task Board\n| 1 | Setup React app shell | Setup | 1 | P0 |\n"
        "5. Mobile Task Board\n| 1 | Flutter bootstrap app | Setup | 1 | P0 |\n"
    )
    assert _tasks("fe", plain) == [("1", "Setup React app shell", "P0")]


def test_tl_pipeless_table_header_is_not_a_heading():
    # Bảng GFM không có '|' đầu dòng: "# | Task | Priority" KHÔNG phải heading cấp 1.
    text = (
        "## 4. FE Task Board\n# | Task | Priority\n--- | --- | ---\n"
        "1 | Setup React app shell | P0\n2 | Orders list page | P1\n"
        "## 5. Mobile Task Board\n"
    )
    assert _tasks("fe", text) == [("1", "Setup React app shell", "P0"), ("2", "Orders list page", "P1")]


def test_tl_unknown_role_or_missing_section():
    assert workflow._extract_tl_tasks("tech_lead", TL_OUTPUT) == []
    assert workflow._extract_tl_tasks("fe", "## 1. Summary\nno boards here\n") == []


def _run_with_spy(monkeypatch, runner):
    """Chạy *runner* với planner/agent giả; trả về {role: [task]} mà planner nhận được."""
    seen: dict[str, list[str]] = {}

    def fake_plan(role, *args, tl_tasks=None, **kwargs):
        seen[role] = [t["task"] for t in (tl_tasks or [])]
        return []

    monkeypatch.setattr(workflow, "_plan_code_files", fake_plan)
    monkeypatch.setattr(workflow, "_call_agent", lambda agent, url, ctx: "OUT")
    runner()
    return seen


def _step_outputs():
    return {
        "ba": "BA " + _FILLER, "sa": "SA " + _FILLER, "ta": "TA " + _FILLER,
        "designer": "Designer " + _FILLER, "tl": TL_OUTPUT,
        "fe": "## FE Agent\n", "mobile": "## Mobile Agent\n", "dba": "## DBA Agent\n",
    }


@pytest.mark.parametrize("role,expected", [
    ("mobile", ["Setup Flutter project skeleton"]),
    ("be", ["Bootstrap NestJS service", "Implement POST /api/auth/login"]),
    ("dba", ["Create users collection schema"]),
])
def test_node_parses_raw_tl_output_not_truncated_context(monkeypatch, role, expected):
    # TL output > MAX_PREV_OUTPUT_CHARS: context chỉ giữ 3.000 ký tự đầu (§1–§4) nên
    # board §5–§7 chỉ còn trong step_outputs["tl"] thô.
    assert len(TL_OUTPUT) > workflow.MAX_PREV_OUTPUT_CHARS
    state = {"user_input": USER_INPUT, "step_outputs": _step_outputs(), "tech_stack": TECH}
    seen = _run_with_spy(monkeypatch, lambda: workflow._build_node(role)(state))
    assert seen[role] == expected


def test_run_single_step_parses_raw_tl_output(monkeypatch):
    seen = _run_with_spy(monkeypatch, lambda: workflow.run_single_step(
        "be", USER_INPUT, prev_outputs=_step_outputs(), tech_stack=TECH, rag_enabled=False,
    ))
    assert seen["be"] == ["Bootstrap NestJS service", "Implement POST /api/auth/login"]


# ── F8: planner JSON + context rút gọn ──────────────────────────────────────────


@pytest.mark.parametrize("reply,expected", [
    ('[{"filename": "app/users/[id]/page.tsx", "description": "detail", "language": "typescript"}]',
     ["app/users/[id]/page.tsx"]),
    ('[Assumption] App router.\n```json\n[{"filename": "src/App.tsx", "language": "typescript"}]\n```',
     ["src/App.tsx"]),
    ('[{"filename": "src/a.ts", "deps": ["x", "y"]}, {"filename": "src/b.ts"}]', ["src/a.ts", "src/b.ts"]),
    ('<think>draft: [{"filename": "draft.ts"}]</think>[{"filename": "final.ts"}]', ["final.ts"]),
    ("Sorry, I cannot produce a list.", []),
])
def test_plan_code_files_extracts_json_array(monkeypatch, reply, expected):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return SimpleNamespace(content=reply)

    monkeypatch.setattr(llm, "ChatOllama", FakeLLM)
    files = workflow._plan_code_files("fe", workflow.AGENTS["fe"], "http://x", "task", TECH, "ctx")
    assert [f["filename"] for f in files] == expected


def test_extract_json_array_skips_non_object_arrays():
    assert workflow._extract_json_array('["a", "b"]') is None
    assert workflow._extract_json_array("no json") is None
    assert workflow._extract_json_array('[{"filename": "a",}] [{"filename": "b"}]') == [{"filename": "b"}]


def _context(role="fe", user_input=USER_INPUT, tech_stack=TECH, **kw):
    parts = workflow._build_context_parts(
        role=role, agent=workflow.AGENTS[role], user_input=user_input,
        step_outputs=_step_outputs(), tech_stack=tech_stack, dep_max_chars=3_000, **kw,
    )
    return "\n".join(parts)


def test_compact_context_keeps_head_and_tail():
    ctx = _context(extra_context="rag chunk " * 400)
    out = workflow._compact_context(ctx, 1_500)
    assert len(out) <= 1_500
    assert USER_INPUT in out and "- nestjs" in out and "- mongodb" in out
    assert out.endswith("rag chunk ")          # đuôi context (phần gần nhất) vẫn giữ
    assert workflow._compact_context("short", 100) == "short"


def test_compact_context_long_user_input_keeps_tech_stack():
    ctx = _context(role="ba", user_input="U" * 5_000)
    out = workflow._compact_context(ctx, 1_500)
    assert len(out) <= 1_500
    assert out.startswith("## Mục tiêu kinh doanh")
    assert "- nestjs" in out and "- mongodb" in out


def test_fallback_single_call_keeps_user_input_stack_and_tl_tasks(monkeypatch):
    captured = {}
    monkeypatch.setattr(workflow, "_plan_code_files", lambda *a, **k: [])
    monkeypatch.setattr(workflow, "_call_agent", lambda agent, url, ctx: captured.setdefault("ctx", ctx))
    ctx = _context(extra_context="rag chunk " * 400)
    workflow._generate_artifacts_multi_turn(
        "be", workflow.AGENTS["be"], "http://x", ctx, USER_INPUT, TECH, "", tl_output=TL_OUTPUT,
    )
    fb = captured["ctx"]
    assert len(fb) <= 2_000
    assert USER_INPUT in fb and "- nestjs" in fb
    assert "Bootstrap NestJS service" in fb and "Implement POST /api/auth/login" in fb


def test_call_agent_short_output_retry_keeps_head(monkeypatch):
    prompts: list[str] = []
    replies = ["", "a long enough answer for the retry path"]

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            prompts.append(messages[-1].content)
            return SimpleNamespace(content=replies.pop(0))

    monkeypatch.setattr(llm, "ChatOllama", FakeLLM)
    ctx = _context(role="ba", extra_context="rag chunk " * 400)
    out = workflow._call_agent(workflow.AGENTS["ba"], "http://x", ctx)
    assert out == "a long enough answer for the retry path"
    assert len(prompts[1]) <= 1_500
    assert USER_INPUT in prompts[1] and "- nestjs" in prompts[1]
