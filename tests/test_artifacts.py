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
