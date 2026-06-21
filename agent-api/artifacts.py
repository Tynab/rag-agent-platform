"""
artifacts.py — Trích xuất và lưu trữ file code từ output markdown của SDLC agents
==================================================================================

Mô tả
-----
Sau khi mỗi coding agent hoàn thành (fe, mobile, be, dba, da, tech_lead,
devsecops), toàn bộ output markdown được lưu và quét để tách riêng từng
file code, sau đó ghi ra disk theo layout chuẩn.

Layout thư mục artifact
-----------------------
    /data/artifacts/{workflow_id}/{role}/
        _output.md          — Toàn bộ markdown output gốc của agent (luôn có)
        src/pages/Login.tsx — File được trích xuất từ directive ### FILE:
        src/hooks/useAuth.ts
        schema.sql
        Dockerfile
        ...

Thứ tự ưu tiên nhận diện tên file trong markdown
-------------------------------------------------
1. Directive có cấu trúc (ưu tiên cao nhất):
       ### FILE: src/components/Button.tsx
       ```typescript
       ...
       ```
   Agent được yêu cầu dùng định dạng này trong system prompt.

2. Dòng comment đầu tiên của code block:
       ```typescript
       // filename: src/components/Button.tsx
       ...
       ```
   Hỗ trợ các prefix: //, #, *, --, /* và các biến thể như "file:", "filename:".

3. Bold hoặc heading ngay trước code block:
       **src/components/Button.tsx**
       ```typescript
       ...
       ```

4. Fallback — đặt tên tự động:
   Khi không tìm được tên file từ bất kỳ pattern nào, file được đặt tên
   theo ngôn ngữ và index: typescript-01.ts, python-02.py, yaml-03.yml...
   Dockerfile được xử lý đặc biệt (không có extension).

Bảo mật đường dẫn
-----------------
_sanitize_relpath() chuẩn hóa và làm sạch mọi đường dẫn do LLM sinh ra:
- Chuyển backslash Windows thành forward slash
- Loại bỏ path traversal: /../, /./
- Chỉ cho phép ký tự an toàn: chữ cái, chữ số, dấu chấm, dấu gạch ngang,
  dấu gạch dưới và dấu slash
- Đường dẫn rỗng sau làm sạch được thay bằng "file.txt"

Hằng số xuất khẩu
-----------------
- ARTIFACT_ROLES:  frozenset[str] — các role có artifact được trích xuất
- extract_and_save(role, output, workflow_id) → list[dict] — hàm chính
- list_artifacts(workflow_id) → dict[str, list[dict]] — liệt kê artifacts
- read_artifact(workflow_id, role, path) → str | None — đọc nội dung file

Lưu ý
-----
Mọi exception trong quá trình trích xuất đều được bắt và ghi log warning
— không làm dừng workflow chính. _output.md luôn được ghi trước, đảm bảo
output thô không bao giờ bị mất dù trích xuất file code thất bại.
"""
from __future__ import annotations

import logging
import os
import posixpath
import re
from pathlib import Path

logger = logging.getLogger("artifacts")

ARTIFACT_BASE: str = os.environ.get("ARTIFACT_DIR", "/data/artifacts")

# Chỉ trích xuất và lưu artifact cho các role coding này.
# Planning roles (ba, pm, sa, ta...) và tester không sinh code file nên không cần trích xuất.
ARTIFACT_ROLES: frozenset[str] = frozenset(
    {"fe", "mobile", "be", "dba", "da", "tech_lead", "devsecops"}
)

# Bảng ánh xạ từ tên ngôn ngữ code fence sang phần mở rộng file (không có dấu chấm).
# __dockerfile__ là sentinel đặc biệt — được xử lý riêng để tạo file "Dockerfile" không extension.
_LANG_EXT: dict[str, str] = {
    "typescript": "ts",   "ts": "ts",
    "tsx": "tsx",
    "javascript": "js",   "js": "js",
    "jsx": "jsx",
    "python": "py",       "py": "py",
    "sql": "sql",
    "yaml": "yml",        "yml": "yml",
    "json": "json",
    "dockerfile": "__dockerfile__",
    "bash": "sh",         "shell": "sh",   "sh": "sh",   "zsh": "sh",
    "markdown": "md",     "md": "md",
    "text": "txt",        "plaintext": "txt",
    "html": "html",       "css": "css",    "scss": "scss",
    "dart": "dart",       "kotlin": "kt",  "swift": "swift",
    "java": "java",       "go": "go",      "rust": "rs",  "rs": "rs",
    "toml": "toml",       "ini": "ini",    "xml": "xml",
    "hcl": "tf",          "terraform": "tf", "tf": "tf",
    "groovy": "groovy",   "properties": "properties",
    "nginx": "conf",      "conf": "conf",
    "env": "env",
}

# Pattern ưu tiên 1: ### FILE: path/to/file.ext rồi đến code block ngay sau.
# Đây là định dạng có cấu trúc cao nhất — agent được yêu cầu dùng pattern này.
_FILE_HDR_RE = re.compile(
    r"###\s+FILE:\s*([^\n]+?)\s*\n```(\w*)\n(.*?)```",
    re.DOTALL,
)

# Pattern dự phòng: khớp bất kỳ code block nào trong markdown.
_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

# Pattern nhận diện tên file từ dòng comment đầu tiên bên trong code block.
# Hỗ trợ các prefix comment: //, #, *, -- và biến thể "file:", "filename:".
_CMT_FNAME_RE = re.compile(
    r"^[/#*\-]+\s*(?:file(?:name)?:\s*)?([a-zA-Z0-9_.][^\s]*\.[a-zA-Z0-9]{1,10})\s*$",
    re.IGNORECASE,
)

# Pattern nhận diện tên file từ bold/heading Markdown ngay trước code block.
# Ví dụ: **src/Login.tsx** hoặc ### src/Login.tsx ngay trước ```typescript.
_PRE_FNAME_RE = re.compile(
    r"(?:\*{1,2}|`|#{1,4}\s+)([a-zA-Z0-9_.][^\s*`\n]*\.[a-zA-Z0-9]{1,10})`?\*{0,2}\s*:?\s*\n?\s*$"
)


def _sanitize_relpath(raw: str) -> str:
    """Chuẩn hóa và làm sạch đường dẫn tương đối do LLM sinh ra.

    Bảo vệ chống path traversal và ký tự không hợp lệ:
    1. Chuyển backslash Windows thành forward slash.
    2. Loại bỏ /../ và /./ bằng posixpath.normpath.
    3. Xóa các thành phần .. còn lại và dấu / đầu tiên.
    4. Chỉ cho phép: chữ cái, chữ số, dấu chấm, gạch ngang, gạch dưới, slash.
    5. Trả về "file.txt" nếu kết quả rỗng sau làm sạch.
    """
    # Chuyển backslash Windows → forward slash, sau đó resolve path traversal.
    normalized = posixpath.normpath(raw.replace("\\", "/"))
    # Loại bỏ .. còn lại và dấu / đầu tiên sau normpath.
    parts = [p for p in normalized.split("/") if p and p != ".."]
    p = "/".join(parts)
    # Chỉ cho phép ký tự an toàn: word chars, dấu chấm, gạch ngang, gạch dưới, slash.
    p = re.sub(r"[^\w.\-/]", "_", p)
    return p or "file.txt"


def _safe_segment(seg: str) -> str:
    """Làm sạch một path segment đơn lẻ (workflow_id, role) trước khi ghép vào đường dẫn.

    Chỉ giữ ký tự an toàn (word chars, dấu chấm, gạch ngang) — mọi separator ('/', '\\')
    bị thay bằng '_'. Vì dấu chấm được phép, phải XỬ LÝ RIÊNG '.' và '..' (nếu không
    chúng sống sót và trở thành traversal). Đây là lớp phòng thủ chiều sâu (C5): workflow_id
    và role đến trực tiếp từ URL nên không được tin tưởng, dù check resolve().relative_to()
    phía sau vẫn là chốt chặn cuối cùng.
    """
    cleaned = re.sub(r"[^\w.\-]", "_", seg)
    if cleaned in ("", ".", ".."):
        return "_"
    return cleaned


def _ext_for_lang(lang: str) -> str:
    v = _LANG_EXT.get(lang.lower(), lang.lower() or "txt")
    return "Dockerfile" if v == "__dockerfile__" else (v or "txt")


def _default_name(lang: str, idx: int) -> str:
    ext = _ext_for_lang(lang)
    if ext == "Dockerfile":
        return "Dockerfile"
    prefix = lang[:6] if lang else "file"
    return f"{prefix}-{idx:02d}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save(role: str, output: str, workflow_id: str) -> list[dict]:
    """Trích xuất các code block có tên file từ *output*, ghi ra disk, trả về danh sách metadata.

    Bao bọc _do_extract() bằng try/except để không làm crash workflow chính.
    Nếu role không nằm trong ARTIFACT_ROLES hoặc output rỗng, trả về [] ngay lập tức.
    Non-fatal: lỗi trích xuất chỉ ghi warning log, không ảnh hưởng đến workflow.
    """
    if role not in ARTIFACT_ROLES:
        return []
    if not output or not output.strip():
        return []
    try:
        return _do_extract(role, output, workflow_id)
    except Exception as exc:
        logger.warning("Artifact extraction failed for %s: %s", role, exc)
        return []


def _do_extract(role: str, output: str, workflow_id: str) -> list[dict]:
    base = Path(ARTIFACT_BASE) / workflow_id / role
    base.mkdir(parents=True, exist_ok=True)
    base_resolved = base.resolve()  # compute once; passed to every _save call

    artifacts: list[dict] = []
    seen: set[str] = set()

    # ── Pass 1: tìm và lưu các file theo directive ### FILE: (ưu tiên cao nhất) ───────────────
    structured_spans: list[tuple[int, int]] = []
    for m in _FILE_HDR_RE.finditer(output):
        raw_path = m.group(1)
        lang     = m.group(2).lower() or "text"
        content  = m.group(3)
        if not content.strip():
            continue
        rel = _sanitize_relpath(raw_path)
        _save(base, base_resolved, role, rel, content, lang, artifacts, seen)
        structured_spans.append((m.start(), m.end()))

    def _in_structured(start: int) -> bool:
        return any(s <= start <= e for s, e in structured_spans)

    # ── Pass 2: code block tự do (dự phòng khi không có ### FILE: directive) ─────────────
    counter = [0]
    for m in _BLOCK_RE.finditer(output):
        if _in_structured(m.start()):
            continue
        lang         = m.group(1) or "text"
        content      = m.group(2)
        if not content.strip():
            continue
        lines        = content.split("\n")
        filename     = None
        save_content = content

        # Thử tên file từ dòng comment đầu tiên bên trong code block.
        if lines:
            m2 = _CMT_FNAME_RE.match(lines[0].strip())
            if m2:
                filename     = m2.group(1).strip()
                save_content = "\n".join(lines[1:])

        # Thử tên file từ bold/heading Markdown ngay trước code block.
        if not filename:
            pre = output[: m.start()].rstrip()
            m3  = _PRE_FNAME_RE.search(pre)
            if m3:
                filename = m3.group(1).strip()

        if not filename:
            counter[0] += 1
            filename = _default_name(lang, counter[0])

        rel = _sanitize_relpath(filename)
        _save(base, base_resolved, role, rel, save_content, lang, artifacts, seen)

    # ── Luôn lưu toàn bộ output thô dưới dạng _output.md ──────────────────────────────
    # Đây là bản gốc đầy đủ không qua xử lý — đảm bảo output không bao giờ bị mất.
    md_path = base / "_output.md"
    md_path.write_text(output, encoding="utf-8")
    artifacts.insert(0, {
        "path": f"{role}/_output.md",
        "filename": "_output.md",
        "language": "markdown",
        "size": len(output.encode("utf-8")),
    })

    logger.info(
        "Artifacts — workflow=%s role=%s: %d files", workflow_id, role, len(artifacts)
    )
    return artifacts


def _save(
    base: Path,
    base_resolved: Path,
    role: str,
    rel: str,
    content: str,
    lang: str,
    artifacts: list,
    seen: set,
) -> None:
    """Ghi nội dung file vào disk tại đường dẫn *base/rel*.

    Kiểm tra path confinement trước khi ghi để đảm bảo file luôn nằm trong
    thư mục *base* — ngăn chặn path traversal qua workflow_id hoặc role giả mạo.
    Bỏ qua file trùng tên (rel đã có trong seen) để tránh ghi đè artifact.
    Non-fatal: lỗi ghi file chỉ ghi warning log, không dừng extraction.
    """
    if rel in seen:
        return
    seen.add(rel)
    dest = base / rel
    # Defense-in-depth: xác nhận đường dẫn đã resolve vẫn nằm trong base.
    # Bảo vệ khỏi các edge case không bị _sanitize_relpath chặn được.
    try:
        dest.resolve().relative_to(base_resolved)
    except ValueError:
        logger.warning("Path confinement blocked write outside base: %s", dest)
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        artifacts.append({
            "path": f"{role}/{rel}",
            "filename": rel,
            "language": lang,
            "size": len(content.encode("utf-8")),
        })
    except Exception as exc:
        logger.warning("Failed to write artifact %s: %s", dest, exc)


# ─────────────────────────────────────────────────────────────────────────────

def list_artifacts(workflow_id: str) -> dict[str, list[dict]]:
    """Quét thư mục artifact của *workflow_id*, trả về dict {role: [file_metadata]}.

    Mỗi phần tử metadata bao gồm: path, filename, language, size (bytes).
    Trả về {} nếu workflow chưa có artifact hoặc thư mục không tồn tại.
    """
    wf_dir = Path(ARTIFACT_BASE) / _safe_segment(workflow_id)
    if not wf_dir.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for role_dir in sorted(wf_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        files: list[dict] = []
        for f in sorted(role_dir.rglob("*")):
            if not f.is_file():
                continue
            rel  = f.relative_to(role_dir).as_posix()
            ext  = f.suffix.lstrip(".")
            files.append({
                "path": f"{role_dir.name}/{rel}",
                "filename": rel,
                "language": _LANG_EXT.get(ext, ext or "text"),
                "size": f.stat().st_size,
            })
        if files:
            result[role_dir.name] = files
    return result


def read_artifact(workflow_id: str, role: str, rel_path: str) -> tuple[str, str] | None:
    """Đọc và trả về nội dung của một file artifact cụ thể.

    Tham số:
        workflow_id: ID của workflow đã chạy.
        role: Tên role agent (ví dụ: "fe", "be").
        rel_path: Đường dẫn tương đối từ thư mục role (ví dụ: "src/Login.tsx").

    Trả về:
        (content, language): nội dung file và tên ngôn ngữ.
        None: nếu file không tồn tại hoặc path bị chặn do bảo mật.
    """
    safe = _sanitize_relpath(rel_path)
    base = Path(ARTIFACT_BASE)
    # workflow_id và role cũng đến từ URL — sanitize từng segment (C5) thay vì ghép thô.
    path = base / _safe_segment(workflow_id) / _safe_segment(role) / safe
    # Đảm bảo đường dẫn đã resolve nằm trong ARTIFACT_BASE để chặn traversal
    # qua workflow_id hoặc role parameter giả mạo từ HTTP request.
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        logger.warning("Path confinement blocked read outside ARTIFACT_BASE: %s", path)
        return None
    if not path.exists() or not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    ext  = path.suffix.lstrip(".")
    lang = _LANG_EXT.get(ext, ext or "text")
    return content, lang
