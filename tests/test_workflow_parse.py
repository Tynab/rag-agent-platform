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


# ── F3: _strip_code_fence chỉ bóc lớp fence BAO NGOÀI ─────────────────────────────

_ARCH_MD = "# Architecture\n\nSetup:\n\n```bash\nnpm install\n```\n\nThen deploy."


def test_strip_code_fence_keeps_markdown_inner_fences():
    # ARCHITECTURE.md không bọc: ```bash là NỘI DUNG file, không được rút còn "npm install".
    assert workflow._strip_code_fence(_ARCH_MD, "markdown") == _ARCH_MD
    two = "```bash\nnpm i\n```\n\nThen:\n\n```bash\nnpm start\n```"
    assert workflow._strip_code_fence(two, "markdown") == two
    # Planner ghi sai language nhưng filename là .md → vẫn xử lý như file markdown.
    assert workflow._strip_code_fence(two, "text", "docs/README.md") == two


def test_strip_code_fence_unwraps_markdown_wrapper():
    # Bọc ```markdown cùng độ dài với fence bên trong → không bị cắt ở ``` đóng đầu tiên.
    assert workflow._strip_code_fence(f"```markdown\n{_ARCH_MD}\n```\n", "markdown") == _ARCH_MD
    assert workflow._strip_code_fence(f"Here it is:\n````md\n{_ARCH_MD}\n````\nDone.", "markdown") == _ARCH_MD
    assert workflow._strip_code_fence(f"```\n{_ARCH_MD}\n```", "markdown") == _ARCH_MD


def test_strip_code_fence_code_targets():
    assert workflow._strip_code_fence("```typescript\nconst a = 1;\n```", "typescript") == "const a = 1;"
    # Lời dẫn / lời kết quanh một block duy nhất.
    assert workflow._strip_code_fence("Here:\n```ts\nconst a = 1;\n```\nDone.", "typescript") == "const a = 1;"
    # Output bị cắt (không có fence đóng) vẫn bóc được fence mở.
    assert workflow._strip_code_fence("```python\nx = 1\n", "python") == "x = 1"
    # Nhiều block: ưu tiên block đúng ngôn ngữ.
    multi = "```bash\nnpm i\n```\n\n```yaml\nkey: v\n```"
    assert workflow._strip_code_fence(multi, "yaml") == "key: v"
    # Không có fence → giữ nguyên (chỉ strip khoảng trắng hai đầu).
    assert workflow._strip_code_fence("x = 1\n", "python") == "x = 1"


def test_file_section_roundtrip_through_artifacts(tmp_path, monkeypatch):
    # F4 end-to-end: section do workflow ghép phải được artifacts.py trích xuất nguyên vẹn.
    import artifacts

    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    out = "## Tech Lead\n" + "\n".join([
        workflow._file_section("ARCHITECTURE.md", "markdown", _ARCH_MD),
        workflow._file_section("src/a.ts", "typescript", "const a = 1;"),
    ])
    arts = artifacts.extract_and_save("tech_lead", out, "wf-rt")
    assert {a["filename"] for a in arts} == {"_output.md", "ARCHITECTURE.md", "src/a.ts"}
    saved = (tmp_path / "wf-rt" / "tech_lead" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert saved == _ARCH_MD + "\n"
