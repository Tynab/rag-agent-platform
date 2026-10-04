"""Regression test cho các heuristic so khớp từ khoá trong agent-api/workflow.py.

- F1: _compute_extra_instruction chọn BE framework theo chuỗi con ("go" ⊂ "mongodb").
- F2: _parse_clarifier_regen_list khớp động từ "be", neo §10 vào trích dẫn "BA §10",
      chỉ lấy 1 role/cell (phụ thuộc hash seed).
- F9: _detect_db_type coi Redis/Elasticsearch là NoSQL chính → 'both' sai.
"""

import pytest

workflow = pytest.importorskip("workflow")

_HDR = (
    "## 10. Recommended Re-generation List\n"
    "| Priority | Agent Role | Reason for Re-generation | Sections to Regenerate |\n"
    "|---|---|---|---|\n"
)


# ── F1: chọn framework theo token, không theo chuỗi con ─────────────────────────


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        (["nestjs", "reactjs", "mongodb"], "NestJS"),       # "go" ⊂ "mongodb"
        (["nestjs", "mongoose", "reactjs"], "NestJS"),      # "go" ⊂ "mongoose"
        (["nestjs", "postgresql", "argocd"], "NestJS"),     # "go" ⊂ "argocd"
        (["express", "javascript"], "Express"),              # "java" ⊂ "javascript"
        (["nestjs", "postgresql", "python"], "NestJS"),     # framework rõ ràng thắng ngôn ngữ trần
        (["Go (Gin)", "postgresql"], "Go HTTP"),
        (["golang"], "Go HTTP"),
        (["Spring Boot", "mysql"], "Spring Boot"),
        (["kotlin"], "Spring Boot"),
        (["Django", "postgresql"], "Django"),
        (["python3.12"], "FastAPI"),
        (["express", "nestjs"], "NestJS"),                  # Nest chạy trên Express
        (["reactjs", "mongodb"], "NestJS"),                 # mặc định
    ],
)
def test_be_framework_matches_tokens_not_substrings(stack, expected):
    assert expected in workflow._compute_extra_instruction("be", stack)


def test_mobile_expo_not_matched_inside_export():
    out = workflow._compute_extra_instruction("mobile", ["flutter", "export-service"])
    assert "Flutter" in out and "React Native" not in out


def test_stack_tokens_normalization():
    toks = workflow._stack_tokens(["NestJS", "Vue3", "React-Native", "go1.22", "MongoDB Atlas"])
    assert {"nest", "vue", "react native", "go", "mongodb"} <= toks
    assert "go" not in workflow._stack_tokens(["mongodb", "django", "argocd", "google cloud"])


# ── F9: Redis/Elasticsearch/Memcached là store phụ trợ ─────────────────────────


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        (["nestjs", "postgresql", "redis", "k8s"], "sql"),
        (["django", "mysql", "elasticsearch"], "sql"),
        (["nestjs", "postgresql", "memcached"], "sql"),
        (["nestjs", "mongodb", "redis"], "nosql"),
        (["nestjs", "postgresql", "mongodb", "redis"], "both"),
        (["redis"], "nosql"),                               # không có DB chính → vẫn là NoSQL
        (["nestjs", "reactjs"], "unknown"),
        (None, "unknown"),
    ],
)
def test_detect_db_type_treats_cache_search_as_auxiliary(stack, expected):
    assert workflow._detect_db_type(stack) == expected


def test_dba_instruction_sql_with_redis_has_no_mongoose():
    out = workflow._compute_extra_instruction("dba", ["nestjs", "postgresql", "redis", "k8s"])
    assert "CREATE TABLE" in out and "Mongoose" not in out


# ── F2: Clarifier §10 regen list ───────────────────────────────────────────────


def test_regen_empty_table_with_verb_be_returns_nothing():
    # Bảng rỗng = không cần re-gen (theo prompt); câu văn chứa động từ "be" không được tính.
    out = _HDR + "\nNo agent needs to be regenerated at this time.\n"
    assert workflow._parse_clarifier_regen_list(out) == []


def test_regen_ignores_section_citations_before_real_heading():
    out = (
        "## 2. Assumption Register\n"
        "| # | Source (Agent §Section) | Assumption | Impact | Status | Question |\n"
        "|---|---|---|---|---|---|\n"
        "| 1 | BA §10 RTM | Login must be via SSO | BE auth may be wrong | Open | IdP? |\n"
        "| 2 | FE §3 | Token in cookie | DBA session table | Open | ? |\n\n"
        + _HDR + "| P1 | DevSecOps | secrets in plain env | §3 |\n"
    )
    assert workflow._parse_clarifier_regen_list(out) == ["devsecops"]


def test_regen_collects_all_roles_in_cell_deterministically():
    out = _HDR + "| P1 | FE, Mobile, DBA | DTO mismatch | §2 |\n"
    assert workflow._parse_clarifier_regen_list(out) == ["fe", "mobile", "dba"]


def test_regen_reads_only_agent_role_column():
    out = _HDR + "| P1 | DBA | orders table must be added so BE and FE can use it | §2 |\n"
    assert workflow._parse_clarifier_regen_list(out) == ["dba"]


def test_regen_display_names():
    out = _HDR + "| P1 | Tech Lead | missing ADR | §1 |\n| P2 | Backend (BE) | auth | §3 |\n"
    assert workflow._parse_clarifier_regen_list(out) == ["be", "tech_lead"]


@pytest.mark.parametrize(
    "row",
    [
        "| — | None | Nothing needs to be regenerated | — |\n",
        "| - | N/A — nothing to be re-generated | - | - |\n",
    ],
)
def test_regen_placeholder_rows_return_nothing(row):
    assert workflow._parse_clarifier_regen_list(_HDR + row) == []


def test_regen_numbered_list_item_10_is_not_section_10():
    # §7 là danh sách đánh số > 10 mục, không có §10 → không được đọc nhầm.
    out = "## 7. Top Critical Clarifications\n" + "".join(
        f"{i}. Does BE handle case {i}? Blocks FE.\n" for i in range(1, 12)
    )
    assert workflow._parse_clarifier_regen_list(out) == []


def test_regen_prose_fallback_skips_verb_be_and_negation():
    base = "## 10. Recommended Re-generation List\n"
    assert workflow._parse_clarifier_regen_list(base + "All outputs can be used as-is.\n") == []
    assert workflow._parse_clarifier_regen_list(
        base + "No re-generation needed; FE, BE and DBA outputs are consistent.\n"
    ) == []
    assert workflow._parse_clarifier_regen_list(
        base + "The be agent must be regenerated because the API is not defined.\n"
    ) == ["be"]
