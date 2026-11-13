"""Unit test cho agent-api/artifacts.py — tên file và dọn thư mục role khi re-gen.

Chỉ dùng stdlib (artifacts.py không cần dependency ngoài). Bao phủ:
- directive "### FILE:" có trang trí Markdown quanh đường dẫn (`...`, **...**);
- tên tự động không trùng (Dockerfile thứ hai không bị dedupe bỏ mất);
- mỗi lần trích xuất dọn sạch thư mục {workflow}/{role} — file cũ không còn sinh ra
  biến mất — nhưng KHÔNG đụng role/workflow khác hay đích của symlink.
"""

import os

import artifacts
import pytest


@pytest.fixture
def base(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(artifacts, "ARTIFACT_BASE", str(root))
    return root


def _files(role_dir) -> dict[str, str]:
    return {
        p.relative_to(role_dir).as_posix(): p.read_text(encoding="utf-8")
        for p in role_dir.rglob("*")
        if p.is_file()
    }


# ── (e) trang trí Markdown quanh đường dẫn của directive ### FILE: ─────────────────


@pytest.mark.parametrize("header", [
    "### FILE: `src/a.ts`",
    "### FILE: **src/a.ts**",
    "### FILE: **`src/a.ts`**",
    '### FILE: "src/a.ts"',
    "### FILE: src/a.ts",
])
def test_file_directive_strips_markdown_decoration(base, header):
    arts = artifacts.extract_and_save("fe", f"{header}\n```ts\nconst a = 1;\n```\n", "wf")
    assert "fe/src/a.ts" in {a["path"] for a in arts}
    assert _files(base / "wf" / "fe").keys() == {"_output.md", "src/a.ts"}


def test_file_directive_keeps_underscores_in_real_names(base):
    # '_' KHÔNG phải trang trí: "__init__.py" phải giữ nguyên tên.
    md = "### FILE: `app/__init__.py`\n```python\nx = 1\n```\n### FILE: _private.ts\n```ts\nlet p;\n```\n"
    artifacts.extract_and_save("be", md, "wf")
    assert _files(base / "wf" / "be").keys() == {"_output.md", "app/__init__.py", "_private.ts"}


# ── (f) tên tự động không trùng ───────────────────────────────────────────────────


def test_second_unnamed_dockerfile_is_not_dropped(base):
    md = (
        "API image:\n```dockerfile\nFROM node:20\n```\n\n"
        "Web image:\n```dockerfile\nFROM nginx:1.27\n```\n\n"
        "```yaml\nkey: v\n```\n"
    )
    arts = artifacts.extract_and_save("devsecops", md, "wf")
    files = _files(base / "wf" / "devsecops")
    assert files["Dockerfile"] == "FROM node:20\n"
    assert files["Dockerfile.02"] == "FROM nginx:1.27\n"
    assert files["yaml-03.yml"] == "key: v\n"          # đánh số các block khác giữ ổn định
    assert {a["filename"] for a in arts} == {"_output.md", "Dockerfile", "Dockerfile.02", "yaml-03.yml"}


def test_unnamed_dockerfile_does_not_collide_with_named_one(base):
    md = "### FILE: Dockerfile\n```dockerfile\nFROM a\n```\n\n```dockerfile\nFROM b\n```\n"
    artifacts.extract_and_save("devsecops", md, "wf")
    files = _files(base / "wf" / "devsecops")
    assert files["Dockerfile"] == "FROM a\n"
    assert files["Dockerfile.01"] == "FROM b\n"


def test_default_name_skips_taken_names():
    assert artifacts._default_name("yaml", 1) == "yaml-01.yml"
    assert artifacts._default_name("yaml", 1, {"yaml-01.yml", "yaml-02.yml"}) == "yaml-03.yml"
    assert artifacts._default_name("dockerfile", 4) == "Dockerfile"
    assert artifacts._default_name("dockerfile", 4, {"Dockerfile", "Dockerfile.04"}) == "Dockerfile.05"


def test_dockerfile_variants_listed_as_dockerfile(base):
    md = "```dockerfile\nFROM a\n```\n```dockerfile\nFROM b\n```\n"
    artifacts.extract_and_save("devsecops", md, "wf")
    langs = {f["filename"]: f["language"] for f in artifacts.list_artifacts("wf")["devsecops"]}
    assert langs["Dockerfile"] == "dockerfile" and langs["Dockerfile.02"] == "dockerfile"
    assert langs["_output.md"] == "md"
    res = artifacts.read_artifact("wf", "devsecops", "Dockerfile.02")
    assert res == ("FROM b\n", "dockerfile")


# ── (g) mỗi lần trích xuất dọn sạch thư mục role ──────────────────────────────────

_RUN1 = (
    "### FILE: src/a.ts\n```ts\nconst a = 1;\n```\n"
    "### FILE: src/old/b.ts\n```ts\nconst b = 2;\n```\n"
)
_RUN2 = "### FILE: src/a.ts\n```ts\nconst a = 42;\n```\n"


def test_regen_removes_files_no_longer_produced(base):
    artifacts.extract_and_save("fe", _RUN1, "wf")
    assert _files(base / "wf" / "fe").keys() == {"_output.md", "src/a.ts", "src/old/b.ts"}

    arts = artifacts.extract_and_save("fe", _RUN2, "wf")
    assert _files(base / "wf" / "fe") == {"_output.md": _RUN2, "src/a.ts": "const a = 42;\n"}
    assert not (base / "wf" / "fe" / "src" / "old").exists()
    assert [a["filename"] for a in arts] == ["_output.md", "src/a.ts"]
    assert [f["filename"] for f in artifacts.list_artifacts("wf")["fe"]] == ["_output.md", "src/a.ts"]


def test_regen_leaves_other_roles_and_workflows_alone(base):
    artifacts.extract_and_save("fe", _RUN1, "wf")
    artifacts.extract_and_save("be", _RUN1, "wf")
    artifacts.extract_and_save("fe", _RUN1, "wf-other")

    artifacts.extract_and_save("fe", _RUN2, "wf")
    assert "src/old/b.ts" in _files(base / "wf" / "be")
    assert "src/old/b.ts" in _files(base / "wf-other" / "fe")


def test_empty_regen_output_keeps_previous_set(base):
    # extract_and_save trả [] cho output rỗng TRƯỚC khi dọn → bộ file tốt không bị xóa.
    artifacts.extract_and_save("fe", _RUN1, "wf")
    assert artifacts.extract_and_save("fe", "   \n", "wf") == []
    assert "src/old/b.ts" in _files(base / "wf" / "fe")


def test_output_md_written_first_and_not_overwritten_by_model_file(base):
    md = "### FILE: _output.md\n```markdown\nfake\n```\n### FILE: a.ts\n```ts\nx\n```\n"
    arts = artifacts.extract_and_save("fe", md, "wf")
    assert (base / "wf" / "fe" / "_output.md").read_text(encoding="utf-8") == md
    assert [a["filename"] for a in arts] == ["_output.md", "a.ts"]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="cần symlink")
def test_regen_unlinks_symlinks_without_touching_targets(base, tmp_path):
    outside_file = tmp_path / "keep.txt"
    outside_file.write_text("keep", encoding="utf-8")
    outside_dir = tmp_path / "keepdir"
    outside_dir.mkdir()
    (outside_dir / "inner.txt").write_text("inner", encoding="utf-8")

    artifacts.extract_and_save("fe", _RUN1, "wf")
    role_dir = base / "wf" / "fe"
    os.symlink(outside_file, role_dir / "link.txt")
    os.symlink(outside_dir, role_dir / "linkdir", target_is_directory=True)

    artifacts.extract_and_save("fe", _RUN2, "wf")
    assert not (role_dir / "link.txt").is_symlink() and not (role_dir / "linkdir").is_symlink()
    assert outside_file.read_text(encoding="utf-8") == "keep"
    assert (outside_dir / "inner.txt").read_text(encoding="utf-8") == "inner"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="cần symlink")
def test_role_dir_symlink_is_refused(base, tmp_path):
    # Thư mục role là symlink ra ngoài ARTIFACT_BASE → từ chối, không xóa/ghi gì ở đích.
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious.txt").write_text("precious", encoding="utf-8")
    (base / "wf").mkdir()
    os.symlink(victim, base / "wf" / "fe", target_is_directory=True)

    assert artifacts.extract_and_save("fe", _RUN2, "wf") == []
    assert _files(victim) == {"precious.txt": "precious"}


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="cần symlink")
def test_workflow_dir_symlink_outside_base_is_refused(base, tmp_path):
    victim = tmp_path / "victim-wf"
    (victim / "fe").mkdir(parents=True)
    (victim / "fe" / "precious.txt").write_text("precious", encoding="utf-8")
    os.symlink(victim, base / "wf", target_is_directory=True)

    assert artifacts.extract_and_save("fe", _RUN2, "wf") == []
    assert _files(victim / "fe") == {"precious.txt": "precious"}


def test_workflow_id_is_sanitized_before_clearing(base):
    # workflow_id lạ không được thoát khỏi ARTIFACT_BASE (cả khi ghi lẫn khi dọn).
    sibling = base.parent / "fe"
    sibling.mkdir()
    (sibling / "precious.txt").write_text("precious", encoding="utf-8")
    artifacts.extract_and_save("fe", _RUN2, "..")
    assert _files(sibling) == {"precious.txt": "precious"}
    assert (base / "_" / "fe" / "src" / "a.ts").exists()
