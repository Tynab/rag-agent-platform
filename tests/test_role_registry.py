"""Snapshot hành vi per-role (role registry).

Chốt giá trị HIỆN TẠI của mọi bảng hành vi theo role trong workflow.py/artifacts.py
— refactor gom các bảng này về AgentConfig (agents.py) phải giữ nguyên hành vi:

- Bảng literal: _PER_DEP_CHARS, _ARTIFACT_ROLES (codegen), _TL_ROLE_SECTIONS,
  _REGEN_ELIGIBLE_ROLES (workflow) và ARTIFACT_ROLES (artifacts) — cả giá trị lẫn kiểu.
- Gợi ý ngày (planning roles) và gợi ý kiểu DB (dba/da) trong _build_context_parts.
- Review roles: role nào nhận context rút gọn khi sinh từng file.
- Dispatch codegen (2 pha) / report (_call_agent) ở cả run_single_step và node LangGraph.
- _compute_extra_instruction: role nào nhận extra instruction với từng loại stack.

Test chỉ dùng tên đã tồn tại trước refactor (không phụ thuộc field mới của AgentConfig).
"""

import artifacts
import pytest
import workflow

ALL_ROLES: list[str] = list(workflow.WORKFLOW_STEPS)
CODEGEN = {"fe", "mobile", "be", "dba", "devsecops", "tech_lead"}

USER_INPUT = "Xây dựng hệ thống đặt lịch khám bệnh"

# Stack mẫu: không có, rỗng, không DB, SQL, NoSQL, cả hai.
STACKS: dict[str, list[str] | None] = {
    "none": None,
    "empty": [],
    "nodb": ["react", "nestjs"],
    "sql": ["nestjs", "postgresql"],
    "nosql": ["nestjs", "mongodb"],
    "both": ["fastapi", "postgresql", "mongodb"],
}


# ── Bảng literal ───────────────────────────────────────────────────────────────


def test_per_dep_chars_snapshot():
    assert type(workflow._PER_DEP_CHARS) is dict
    assert workflow._PER_DEP_CHARS == {
        "tech_lead": 800,
        "devsecops": 1_000,
        "tester": 3_000,
        "clarifier": 1_500,
    }


def test_codegen_roles_snapshot():
    assert type(workflow._ARTIFACT_ROLES) is frozenset
    assert workflow._ARTIFACT_ROLES == frozenset(CODEGEN)


def test_artifact_extract_roles_snapshot():
    # Khác workflow._ARTIFACT_ROLES: DA chạy _call_agent (report) nhưng vẫn được trích code block.
    assert type(artifacts.ARTIFACT_ROLES) is frozenset
    assert artifacts.ARTIFACT_ROLES == frozenset(
        {"fe", "mobile", "be", "dba", "da", "tech_lead", "devsecops"}
    )


def test_tl_role_sections_snapshot():
    assert type(workflow._TL_ROLE_SECTIONS) is dict
    assert all(type(v) is list for v in workflow._TL_ROLE_SECTIONS.values())
    assert workflow._TL_ROLE_SECTIONS == {
        "fe":     ["FE Task Board", "4. FE Task", "§4 FE", "4. FE"],
        "mobile": ["Mobile Task Board", "5. Mobile Task", "§5 Mobile", "5. Mobile"],
        "be":     ["BE Task Board", "6. BE Task", "§6 BE", "6. BE"],
        "dba":    ["DBA Task Board", "7. DBA Task", "§7 DBA", "7. DBA"],
    }


def test_regen_eligible_roles_snapshot():
    assert type(workflow._REGEN_ELIGIBLE_ROLES) is frozenset
    assert workflow._REGEN_ELIGIBLE_ROLES == frozenset(
        {"fe", "mobile", "dba", "be", "da", "tech_lead", "devsecops"}
    )


def test_regen_role_regex_covers_exactly_eligible_roles():
    # Mỗi named group của _REGEN_ROLE_RE là một role re-gen được — hai nơi phải khớp.
    assert set(workflow._REGEN_ROLE_RE.groupindex) == set(workflow._REGEN_ELIGIBLE_ROLES)


# ── Gợi ý ngày / kiểu DB trong context ─────────────────────────────────────────

DATE_HEADER = "## Ngày hiện tại"
DB_HEADERS: dict[tuple[str, str], str] = {
    ("dba", "nosql"): "## DB Schema: NoSQL Only",
    ("dba", "sql"):   "## DB Schema: SQL Relational",
    ("dba", "both"):  "## DB Schema: Mixed (SQL + NoSQL)",
    ("da", "nosql"):  "## Data Analysis: NoSQL Only",
    ("da", "sql"):    "## Data Analysis: SQL",
    ("da", "both"):   "## Data Analysis: Mixed (SQL + NoSQL)",
}
ALL_DB_HEADERS = set(DB_HEADERS.values())


def _headers(role: str, stack: list[str] | None) -> set[str]:
    parts = workflow._build_context_parts(
        role=role,
        agent=workflow.AGENTS[role],
        user_input=USER_INPUT,
        step_outputs={},
        tech_stack=stack,
        dep_max_chars=workflow.MAX_PREV_OUTPUT_CHARS,
        rag_enabled=False,
    )
    return {ln.strip() for p in parts for ln in p.splitlines() if ln.startswith("## ")}


@pytest.mark.parametrize("stack_key", list(STACKS))
@pytest.mark.parametrize("role", ALL_ROLES)
def test_context_hints_snapshot(role, stack_key):
    headers = _headers(role, STACKS[stack_key])
    assert (DATE_HEADER in headers) == (role in {"ba", "pm", "sa", "ta"})
    expected_db = DB_HEADERS.get((role, stack_key))
    assert headers & ALL_DB_HEADERS == ({expected_db} if expected_db else set())


# ── Review roles: context rút gọn kèm mỗi lần sinh file ────────────────────────


@pytest.mark.parametrize("role", sorted(CODEGEN))
def test_review_context_snippet_snapshot(monkeypatch, role):
    snippets: list[str] = []

    def fake_plan(*args, **kwargs):
        return [{"filename": "src/main.ts", "description": "entry", "language": "typescript"}]

    def fake_gen(filename, description, language, agent, url, tech_stack, extra, ctx_snippet="", *a, **k):
        snippets.append(ctx_snippet)
        return "// ok"

    monkeypatch.setattr(workflow, "_plan_code_files", fake_plan)
    monkeypatch.setattr(workflow, "_generate_one_file", fake_gen)
    ctx = "\n".join(workflow._build_context_parts(
        role=role, agent=workflow.AGENTS[role], user_input=USER_INPUT, step_outputs={},
        tech_stack=STACKS["sql"], dep_max_chars=workflow.MAX_PREV_OUTPUT_CHARS,
    ))
    workflow._generate_artifacts_multi_turn(
        role, workflow.AGENTS[role], "http://x", ctx, USER_INPUT, STACKS["sql"], "", tl_output="",
    )
    assert len(snippets) == 1
    assert bool(snippets[0]) == (role in {"tech_lead", "devsecops"})


# ── Dispatch codegen / report ──────────────────────────────────────────────────


def _install_dispatch_fakes(monkeypatch) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []

    def fake_multi(role, agent, url, context, user_input, tech_stack, extra_instruction="", tl_output=None):
        calls.append(("codegen", role, extra_instruction))
        return f"CODEGEN {role}"

    def fake_call(agent, url, context, *a, **k):
        calls.append(("report", agent.role, ""))
        return f"REPORT {agent.role}"

    monkeypatch.setattr(workflow, "_generate_artifacts_multi_turn", fake_multi)
    monkeypatch.setattr(workflow, "_call_agent", fake_call)
    return calls


@pytest.mark.parametrize("stack_key", ["none", "nosql"])
@pytest.mark.parametrize("role", ALL_ROLES)
def test_run_single_step_dispatch_snapshot(monkeypatch, role, stack_key):
    calls = _install_dispatch_fakes(monkeypatch)
    stack = STACKS[stack_key]
    workflow.run_single_step(role, USER_INPUT, tech_stack=stack, rag_enabled=False)
    expected_mode = "codegen" if role in CODEGEN else "report"
    expected_extra = workflow._compute_extra_instruction(role, stack) if role in CODEGEN else ""
    assert calls == [(expected_mode, role, expected_extra)]


@pytest.mark.parametrize("stack_key", ["none", "nosql"])
@pytest.mark.parametrize("role", ALL_ROLES)
def test_node_dispatch_snapshot(monkeypatch, role, stack_key):
    calls = _install_dispatch_fakes(monkeypatch)
    stack = STACKS[stack_key]
    state = {
        "user_input": USER_INPUT,
        "step_outputs": {},
        "tech_stack": stack,
        "rag_enabled": False,
        "ollama_base_url": "http://x",
    }
    result = workflow._build_node(role)(state)
    expected_mode = "codegen" if role in CODEGEN else "report"
    expected_extra = workflow._compute_extra_instruction(role, stack) if role in CODEGEN else ""
    assert calls == [(expected_mode, role, expected_extra)]
    assert result["completed_steps"] == [role]


def test_dep_max_chars_used_by_run_single_step(monkeypatch):
    # dep_max_chars của từng role: override trong _PER_DEP_CHARS, còn lại MAX_PREV_OUTPUT_CHARS.
    seen: dict[str, int] = {}
    real = workflow._build_context_parts

    def spy(**kwargs):
        seen[kwargs["role"]] = kwargs["dep_max_chars"]
        return real(**kwargs)

    _install_dispatch_fakes(monkeypatch)
    monkeypatch.setattr(workflow, "_build_context_parts", spy)
    for role in ALL_ROLES:
        workflow.run_single_step(role, USER_INPUT, rag_enabled=False)
    assert seen == {
        **{r: workflow.MAX_PREV_OUTPUT_CHARS for r in ALL_ROLES},
        "tech_lead": 800, "devsecops": 1_000, "tester": 3_000, "clarifier": 1_500,
    }


# ── _compute_extra_instruction ─────────────────────────────────────────────────

# Role → các stack_key cho ra extra instruction KHÁC rỗng.
_EXTRA_NONEMPTY: dict[str, set[str]] = {
    "fe":        {"nodb", "sql", "nosql", "both"},
    "mobile":    {"nodb", "sql", "nosql", "both"},
    "be":        {"nodb", "sql", "nosql", "both"},
    "dba":       {"sql", "nosql", "both"},
    "devsecops": {"nodb", "sql", "nosql", "both"},
    # tech_lead nhận hướng dẫn kể cả khi không có tech stack (devsecops thì không).
    "tech_lead": set(STACKS),
}


@pytest.mark.parametrize("stack_key", list(STACKS))
@pytest.mark.parametrize("role", ALL_ROLES)
def test_extra_instruction_snapshot(role, stack_key):
    out = workflow._compute_extra_instruction(role, STACKS[stack_key])
    assert bool(out) == (stack_key in _EXTRA_NONEMPTY.get(role, set()))
    if role == "tech_lead":
        assert out.startswith("Write ARCHITECTURE.md or INTEGRATION.md")
    if role == "devsecops" and out:
        assert out.startswith("Write real Kubernetes manifests")
    if role in {"fe", "mobile", "be"} and out:
        assert out.startswith("Scaffold a ")
