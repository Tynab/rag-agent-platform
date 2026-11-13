"""Role registry: AgentConfig (agents.py) là nguồn sự thật duy nhất cho hành vi per-role.

- Mỗi bảng module-level trong workflow.py/artifacts.py khớp đúng field tương ứng của AGENTS.
- Bất biến giữa các field (codegen ⇒ trích artifact, review/TL board chỉ cho codegen...)
  để thêm role mới sai cấu hình bị bắt ở test thay vì lúc chạy workflow.
Giá trị cụ thể của từng bảng được chốt riêng trong test_role_registry.py.
"""

import dataclasses
import typing

import agents
import artifacts
import pytest
import workflow

AGENTS = agents.AGENTS
PLANNING_ROLES = {"ba", "pm", "sa", "ta", "designer", "tl", "tester"}


def _roles(pred) -> frozenset[str]:
    return frozenset(r for r, cfg in AGENTS.items() if pred(cfg))


# ── Bảng dẫn xuất ↔ field ──────────────────────────────────────────────────────


def test_codegen_table_matches_mode():
    assert workflow._ARTIFACT_ROLES == _roles(lambda c: c.mode == "codegen")


def test_extract_table_matches_field():
    assert artifacts.ARTIFACT_ROLES == _roles(lambda c: c.extract_artifacts)


def test_regen_table_matches_field():
    assert workflow._REGEN_ELIGIBLE_ROLES == _roles(lambda c: c.regen_eligible)


def test_hint_and_review_tables_match_fields():
    assert workflow._DATE_HINT_ROLES == _roles(lambda c: c.wants_date_hint)
    assert workflow._DB_HINT_ROLES == _roles(lambda c: c.wants_db_hint)
    assert workflow._REVIEW_ROLES == _roles(lambda c: c.review_context)


def test_tl_sections_table_matches_field():
    assert workflow._TL_ROLE_SECTIONS == {
        r: list(cfg.tl_sections) for r, cfg in AGENTS.items() if cfg.tl_sections
    }


@pytest.mark.parametrize("role", list(AGENTS))
def test_dep_max_chars_matches_field(role):
    cfg = AGENTS[role]
    expected = workflow.MAX_PREV_OUTPUT_CHARS if cfg.per_dep_chars is None else cfg.per_dep_chars
    assert workflow._PER_DEP_CHARS.get(role, workflow.MAX_PREV_OUTPUT_CHARS) == expected


def test_unknown_role_falls_back_to_defaults():
    # Role lạ (đến từ URL /agent/{role}, /artifacts/...) không có trong bảng nào.
    assert "ghost" not in workflow._ARTIFACT_ROLES
    assert "ghost" not in artifacts.ARTIFACT_ROLES
    assert workflow._PER_DEP_CHARS.get("ghost", workflow.MAX_PREV_OUTPUT_CHARS) == workflow.MAX_PREV_OUTPUT_CHARS
    assert workflow._find_tl_section("ghost", "## FE Task Board\n| T-01 | x |") == ""
    assert artifacts.extract_and_save("ghost", "```python\nprint(1)\n```", "wf") == []


# ── Bất biến giữa các field ────────────────────────────────────────────────────


def test_agentconfig_defaults_are_plain_report_role():
    cfg = agents.AgentConfig(step_id=99, role="x", name="X", model="m", system_prompt="p")
    assert cfg.mode == "report"
    assert not (cfg.extract_artifacts or cfg.regen_eligible or cfg.wants_date_hint
                or cfg.wants_db_hint or cfg.review_context)
    assert cfg.tl_sections == ()
    assert cfg.per_dep_chars is None


@pytest.mark.parametrize("role", list(AGENTS))
def test_registry_invariants(role):
    cfg = AGENTS[role]
    assert cfg.role == role
    assert cfg.mode in typing.get_args(agents.AgentMode)
    if cfg.mode == "codegen":
        # Output codegen là các section ### FILE: — phải được trích ra disk.
        assert cfg.extract_artifacts
    if cfg.review_context or cfg.tl_sections:
        # Review context và TL task board chỉ dùng trong quy trình sinh code 2 pha.
        assert cfg.mode == "codegen"
    if cfg.per_dep_chars is not None:
        assert cfg.per_dep_chars > 0
    assert isinstance(cfg.tl_sections, tuple)
    assert all(isinstance(s, str) and s for s in cfg.tl_sections)


def test_planning_roles_and_clarifier_never_regen():
    assert not workflow._REGEN_ELIGIBLE_ROLES & (PLANNING_ROLES | {"clarifier"})


def test_db_hint_roles_have_hint_text():
    # Câu chữ gợi ý DB chỉ có cho dba/da trong _build_context_parts — role mới bật
    # wants_db_hint mà không thêm câu chữ sẽ lặng lẽ không nhận gợi ý nào.
    assert workflow._DB_HINT_ROLES <= {"dba", "da"}


def test_registry_fields_are_dataclass_fields():
    names = {f.name for f in dataclasses.fields(agents.AgentConfig)}
    assert {
        "mode", "extract_artifacts", "regen_eligible", "tl_sections",
        "per_dep_chars", "wants_date_hint", "wants_db_hint", "review_context",
    } <= names
