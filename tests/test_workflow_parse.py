"""Unit test cho agent-api/workflow.py — các hàm parse/cleanup thuần (T2).

Cần dependency của agent-api (langgraph, langchain-ollama...). Nếu chưa cài, test
được SKIP qua importorskip.
"""

import pytest

workflow = pytest.importorskip("workflow")


def test_strip_thinking_removes_blocks():
    assert workflow._strip_thinking("<think>secret</think>answer") == "answer"
    assert "think" not in workflow._strip_thinking("a<think>x</think>b")


def test_truncate_has_no_marker():
    assert workflow._truncate("abcdef", 3) == "abc"   # không thêm '...'
    assert workflow._truncate("ab", 5) == "ab"


def test_parse_clarifier_table():
    out = (
        "## 10. Recommended Re-generation List\n"
        "| Agent Role | Reason |\n|---|---|\n| be | gap |\n| dba | missing |\n"
    )
    roles = workflow._parse_clarifier_regen_list(out)
    assert set(roles) == {"be", "dba"}
    # phải theo thứ tự WORKFLOW_STEPS (dba trước be)
    assert roles == [r for r in workflow.WORKFLOW_STEPS if r in {"be", "dba"}]


def test_parse_clarifier_prose_fallback():
    # W4: model liệt kê dạng câu văn thay vì bảng → vẫn bắt được role.
    out = "## 10. Recommended Re-generation\nWe should re-generate the be and dba agents.\n"
    roles = workflow._parse_clarifier_regen_list(out)
    assert "be" in roles and "dba" in roles


def test_parse_clarifier_empty():
    assert workflow._parse_clarifier_regen_list("không có section 10 nào ở đây") == []


def test_deloop_collapses_repeats():
    txt = "\n".join(["same line"] * 50)
    out_lines = workflow._deloop(txt, max_repeats=8).splitlines()
    assert len(out_lines) < 50
