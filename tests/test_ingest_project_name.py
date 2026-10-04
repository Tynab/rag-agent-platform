"""Unit test F6 — kiểm tra tên project trước khi ghép vào RAW_DATA_DIR / tên collection.

Cần dependency của rag-api; tự SKIP qua importorskip nếu chưa cài.
"""

import pytest

ingest = pytest.importorskip("ingest")


@pytest.mark.parametrize("name", ["crm", "my-project", "yan_lib", "v1.2", "Auth2", "tài-liệu", "a..b"])
def test_validate_project_name_accepts_simple_names(name):
    assert ingest.validate_project_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "/", "../..", "a/b", "a\\b", "/etc", "~", ".git", "-x", "_x", " crm", "crm ", "a" * 129],
)
def test_validate_project_name_rejects_paths_and_odd_names(name):
    with pytest.raises(ingest.InvalidProjectError):
        ingest.validate_project_name(name)


def test_project_dir_for_rejects_symlink_escaping_raw_dir(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "crm").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (raw / "evil").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(ingest, "RAW_DATA_DIR", str(raw))

    assert ingest.project_dir_for("crm") == raw / "crm"
    with pytest.raises(ingest.InvalidProjectError):
        ingest.project_dir_for("evil")


def test_get_projects_skips_hidden_invalid_and_escaping_dirs(tmp_path):
    raw = tmp_path / "raw"
    for d in ("crm", "billing", ".git", "has space"):
        (raw / d).mkdir(parents=True)
    (raw / "notes.md").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (raw / "evil").symlink_to(outside, target_is_directory=True)

    assert ingest.get_projects(str(raw)) == ["billing", "crm"]


@pytest.mark.parametrize("name", [".", "..", "/", "../..", "crm/../.."])
def test_ingest_rejects_before_touching_filesystem(name, monkeypatch):
    def _must_not_load(*_a, **_k):
        raise AssertionError("load_documents không được gọi với project không hợp lệ")

    monkeypatch.setattr(ingest, "load_documents", _must_not_load)
    with pytest.raises(ingest.InvalidProjectError):
        ingest.ingest(project=name)
    with pytest.raises(ingest.InvalidProjectError):
        ingest.ingest(project=name, reset=True)
