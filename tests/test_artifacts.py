"""Unit test cho agent-api/artifacts.py (T2).

artifacts.py chỉ dùng stdlib nên test này chạy được không cần cài dependency nào.
Bao phủ: làm sạch đường dẫn (_sanitize_relpath), sanitize segment (_safe_segment — C5),
trích xuất file theo directive, lọc role, và path confinement khi đọc artifact.
"""

import artifacts


def test_sanitize_relpath_blocks_traversal():
    assert artifacts._sanitize_relpath("../../etc/passwd") == "etc/passwd"
    assert artifacts._sanitize_relpath("a\\b\\c.ts") == "a/b/c.ts"
    # input rỗng phải cho ra đường dẫn confined, không có thành phần traversal.
    assert ".." not in artifacts._sanitize_relpath("")


def test_safe_segment_strips_separators():
    # C5: một segment không được chứa separator HAY trở thành traversal sau khi làm sạch.
    assert "/" not in artifacts._safe_segment("a/b/c")
    assert "\\" not in artifacts._safe_segment("a\\b")
    assert artifacts._safe_segment("..") == "_"   # '..' phải bị trung hòa
    assert artifacts._safe_segment(".") == "_"
    assert artifacts._safe_segment("") == "_"


def test_extract_file_directive(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    md = (
        "## FE Agent\n"
        "### FILE: src/pages/Login.tsx\n"
        "```tsx\n"
        "export const Login = () => null;\n"
        "```\n"
    )
    arts = artifacts.extract_and_save("fe", md, "wf1")
    paths = {a["path"] for a in arts}
    assert "fe/_output.md" in paths
    assert "fe/src/pages/Login.tsx" in paths
    saved = tmp_path / "wf1" / "fe" / "src" / "pages" / "Login.tsx"
    assert saved.exists()
    assert "export const Login" in saved.read_text(encoding="utf-8")


def test_extract_skips_non_artifact_role(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    # ba không thuộc ARTIFACT_ROLES → không trích xuất gì.
    assert artifacts.extract_and_save("ba", "### FILE: x.ts\n```ts\n1\n```", "wf2") == []


def test_read_artifact_path_confinement(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    artifacts.extract_and_save("be", "### FILE: a.ts\n```ts\nconst a = 1;\n```", "wf3")

    # workflow_id chứa traversal bị _safe_segment trung hòa → không đọc ra ngoài base.
    assert artifacts.read_artifact("../../etc", "be", "a.ts") is None

    res = artifacts.read_artifact("wf3", "be", "a.ts")
    assert res is not None
    assert "const a" in res[0]


# ── F4: ranh giới code block phải fence-aware (fence lồng nhau, fence dài hơn) ──────

_ARCH_MD = "# Architecture\n\n```bash\nnpm install\n```\n\nDeploy bằng Helm.\n"


def _saved_files(tmp_path, wf: str, role: str) -> dict[str, str]:
    base = tmp_path / wf / role
    return {
        p.relative_to(base).as_posix(): p.read_text(encoding="utf-8")
        for p in base.rglob("*")
        if p.is_file() and p.name != "_output.md"
    }


def test_find_fenced_blocks_respects_fence_length():
    # Fence mở ```` chỉ đóng bằng fence trơn >= 4 backtick → ```bash bên trong là nội dung.
    md = "intro\n````markdown\n" + _ARCH_MD + "````\ntail\n"
    blocks = artifacts.find_fenced_blocks(md)
    assert len(blocks) == 1
    assert blocks[0].lang == "markdown"
    assert blocks[0].body == _ARCH_MD
    assert md[blocks[0].end:] == "tail\n"


def test_find_fenced_blocks_nested_same_length_and_unclosed():
    # LLM hay bọc ```markdown quanh tài liệu có ```bash cùng độ dài → coi là block lồng nhau.
    md = "```markdown\n" + _ARCH_MD + "```\n\n```ts\nconst a = 1;\n"
    blocks = artifacts.find_fenced_blocks(md)
    assert [b.lang for b in blocks] == ["markdown", "ts"]
    assert blocks[0].body == _ARCH_MD
    # Block chưa đóng (output bị cắt do num_predict) kéo dài tới hết văn bản.
    assert blocks[1].body == "const a = 1;\n" and blocks[1].end == len(md)
    # Fence khác ký tự không đóng block; ```js``` (info chứa backtick) không phải fence.
    md2 = "~~~\n```\nx\n~~~\n```js```\n"
    blocks2 = artifacts.find_fenced_blocks(md2)
    assert len(blocks2) == 1 and blocks2[0].body == "```\nx\n"


def test_fence_for_is_longer_than_inner_fences():
    assert artifacts.fence_for("const a = 1;") == "```"
    assert artifacts.fence_for(_ARCH_MD) == "````"
    assert artifacts.fence_for("x\n  `````\n") == "``````"


def test_extract_file_with_inner_fence(tmp_path, monkeypatch):
    # F4: file markdown chứa ```bash không được bị cắt, phần sau không thành text-01.txt.
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    md = (
        "## Tech Lead\n"
        "### FILE: ARCHITECTURE.md\n````markdown\n" + _ARCH_MD + "````\n"   # fence dài hơn (workflow.py)
        "### FILE: docs/RUNBOOK.md\n```markdown\n" + _ARCH_MD + "```\n"     # LLM tự bọc cùng độ dài
        "### FILE: src/a.ts\n```ts\nconst a = 1;\n```\n"
    )
    artifacts.extract_and_save("tech_lead", md, "wf4")
    assert _saved_files(tmp_path, "wf4", "tech_lead") == {
        "ARCHITECTURE.md": _ARCH_MD,
        "docs/RUNBOOK.md": _ARCH_MD,
        "src/a.ts": "const a = 1;\n",
    }


def test_extract_free_blocks_still_named(tmp_path, monkeypatch):
    # Pass 2 (không có ### FILE:) vẫn lấy tên từ comment đầu block / bold phía trước.
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(tmp_path))
    md = (
        "```ts\n// filename: src/b.ts\nconst b = 2;\n```\n\n"
        "**schema.sql**\n```sql\nCREATE TABLE t (id int);\n```\n\n"
        "```yaml\nkey: v\n```\n"
    )
    artifacts.extract_and_save("dba", md, "wf5")
    assert _saved_files(tmp_path, "wf5", "dba") == {
        "src/b.ts": "const b = 2;\n",
        "schema.sql": "CREATE TABLE t (id int);\n",
        "yaml-01.yml": "key: v\n",
    }
