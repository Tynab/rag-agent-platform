"""Kiểm tra vệ sinh source tree (T5) — chạy được không cần dependency.

- Không file .py nào được bắt đầu bằng UTF-8 BOM (A5): BOM vô hại với Python khi
  import từ file, nhưng gây nhiễu linter/diff và lệch chuẩn encoding của repo.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_BOM = b"\xef\xbb\xbf"


def test_no_utf8_bom_in_python_sources():
    bad = []
    for f in ROOT.rglob("*.py"):
        if ".git" in f.parts:
            continue
        if f.read_bytes()[:3] == _BOM:
            bad.append(str(f.relative_to(ROOT)))
    assert not bad, f"Các file .py còn UTF-8 BOM: {bad}"
