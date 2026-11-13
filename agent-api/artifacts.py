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

Mỗi lần trích xuất (lần chạy đầu hoặc Clarifier re-gen một role) XÓA SẠCH thư mục
{workflow_id}/{role}/ trước khi ghi — file của lần sinh trước mà lần này không còn
sinh ra sẽ không "sống sót" trong danh sách artifact. Chỉ xóa bên trong đúng thư mục
role đã resolve (sâu đúng 2 cấp dưới ARTIFACT_BASE, không phải symlink) — xem
_reset_role_dir().

Thứ tự ưu tiên nhận diện tên file trong markdown
-------------------------------------------------
1. Directive có cấu trúc (ưu tiên cao nhất):
       ### FILE: src/components/Button.tsx
       ```typescript
       ...
       ```
   Agent được yêu cầu dùng định dạng này trong system prompt. Trang trí Markdown
   quanh đường dẫn (### FILE: `src/a.ts`, ### FILE: **src/a.ts**) được bóc bỏ.
   Ranh giới code block được xác định bằng find_fenced_blocks() (fence-aware
   theo CommonMark): file chứa fence nội bộ (```bash trong ARCHITECTURE.md)
   được bọc bằng fence dài hơn (````markdown) và được trích xuất nguyên vẹn.

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
   theo ngôn ngữ và index: typesc-01.ts, python-02.py, yaml-03.yml...
   Dockerfile được xử lý đặc biệt (không có extension): block đầu tiên là
   "Dockerfile", các block sau "Dockerfile.02", "Dockerfile.03"... Tên tự động
   không bao giờ trùng file đã lưu (block thứ hai không bị dedupe bỏ mất).

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
                   (dẫn xuất từ AgentConfig.extract_artifacts trong agents.py)
- extract_and_save(role, output, workflow_id) → list[dict] — hàm chính
- list_artifacts(workflow_id) → dict[str, list[dict]] — liệt kê artifacts
- read_artifact(workflow_id, role, path) → str | None — đọc nội dung file
- find_fenced_blocks(text) → list[FencedBlock] — quét code block cấp ngoài cùng
- fence_for(content) → str — fence đủ dài để bọc content (dùng khi ghép markdown)

Lưu ý
-----
Mọi exception trong quá trình trích xuất đều được bắt và ghi log warning
— không làm dừng workflow chính. _output.md được ghi NGAY sau khi dọn thư mục
role (trước mọi file code), đảm bảo output thô không bao giờ bị mất dù trích
xuất file code thất bại; file code nào tự đặt tên "_output.md" bị bỏ qua.
"""
from __future__ import annotations

import logging
import os
import posixpath
import re
import shutil
from collections.abc import Container
from pathlib import Path
from typing import NamedTuple

from agents import AGENTS

logger = logging.getLogger("artifacts")

ARTIFACT_BASE: str = os.environ.get("ARTIFACT_DIR", "/data/artifacts")

# Chỉ trích xuất và lưu artifact cho các role coding này (AgentConfig.extract_artifacts
# trong agents.py: fe, mobile, be, dba, da, tech_lead, devsecops).
# Planning roles (ba, pm, sa, ta...) và tester không sinh code file nên không cần trích xuất.
ARTIFACT_ROLES: frozenset[str] = frozenset(
    role for role, cfg in AGENTS.items() if cfg.extract_artifacts
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

# Pattern ưu tiên 1: dòng "### FILE: path/to/file.ext" đứng ngay trước một code block.
# Đây là định dạng có cấu trúc cao nhất — agent được yêu cầu dùng pattern này.
# Chỉ khớp trên DÒNG ngay trước fence mở; ranh giới block do find_fenced_blocks() quyết
# định (F4: regex lazy cũ `(.*?)```` cắt file ở ``` đầu tiên — kể cả ```bash lồng bên
# trong ARCHITECTURE.md — phần còn lại bị lưu nhầm thành text-01.txt).
_FILE_HDR_RE = re.compile(r"###\s+FILE:\s*(.+?)\s*$")

# Ký tự trang trí Markdown LLM hay bọc quanh đường dẫn trong directive ### FILE:
# (`src/a.ts`, **src/a.ts**, "src/a.ts"). KHÔNG gồm '_' — "__init__.py" là tên thật.
_HDR_DECORATION = "`*\"' \t"

# Dòng fence theo CommonMark: thụt lề tối đa 3 space, >= 3 backtick hoặc tilde, rồi info string.
_FENCE_LINE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")
_FENCE_RUN_RE  = re.compile(r"^ {0,3}(`{3,})", re.MULTILINE)

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


def _strip_hdr_decoration(raw: str) -> str:
    """Bóc trang trí Markdown bao quanh đường dẫn bắt được từ directive ### FILE:.

    "### FILE: `src/a.ts`" / "### FILE: **src/a.ts**" → "src/a.ts". Nếu không bóc,
    _sanitize_relpath() biến backtick/dấu sao thành '_' → file lưu thành "_src/a.ts_".
    Chỉ bóc ở hai đầu (backtick, dấu sao, dấu nháy, khoảng trắng) — không đụng '_'.
    """
    return raw.strip(_HDR_DECORATION)


def _ext_for_lang(lang: str) -> str:
    v = _LANG_EXT.get(lang.lower(), lang.lower() or "txt")
    return "Dockerfile" if v == "__dockerfile__" else (v or "txt")


def _default_name(lang: str, idx: int, taken: Container[str] = frozenset()) -> str:
    """Tên file tự động cho code block không tìm được tên, KHÔNG trùng tên nào trong *taken*.

    *taken* là tập đường dẫn đã lưu (seen của _do_extract) — _save() bỏ qua tên trùng,
    nên tên tự động trùng nghĩa là block bị mất lặng lẽ. Dockerfile: block đầu tiên giữ
    tên chuẩn "Dockerfile", các block sau thành "Dockerfile.{idx:02d}" (quy ước
    Dockerfile.* mà docker build -f / BuildKit nhận ra). Ngôn ngữ khác:
    "{lang[:6]}-{idx:02d}.{ext}"; nếu vẫn trùng thì tăng index tới khi không trùng.
    """
    ext = _ext_for_lang(lang)
    if ext == "Dockerfile":
        if "Dockerfile" not in taken:
            return "Dockerfile"
        pattern = "Dockerfile.{:02d}"
    else:
        prefix = lang[:6] if lang else "file"
        pattern = prefix + "-{:02d}." + ext
    n = idx
    while pattern.format(n) in taken:
        n += 1
    return pattern.format(n)


def _path_language(name: str) -> str:
    """Ngôn ngữ hiển thị của một file artifact trên disk, suy từ tên file.

    Dockerfile không có extension (hoặc Dockerfile.02 do _default_name đặt) →
    "dockerfile"; còn lại tra _LANG_EXT theo extension như trước.
    """
    lower = name.lower()
    if lower == "dockerfile" or lower.startswith("dockerfile.") or lower.endswith(".dockerfile"):
        return "dockerfile"
    ext = posixpath.splitext(name)[1].lstrip(".")
    return _LANG_EXT.get(ext, ext or "text")


# ── Quét fenced code block (fence-aware, dùng chung với workflow.py) ────────────

class FencedBlock(NamedTuple):
    """Một fenced code block CẤP NGOÀI CÙNG trong văn bản markdown."""
    start: int   # offset đầu dòng fence mở
    end: int     # offset ngay sau dòng fence đóng (= len(text) nếu block chưa đóng)
    info: str    # info string sau fence mở, ví dụ "typescript" hoặc "ts title=a.ts"
    body: str    # nội dung giữa fence mở và fence đóng (giữ newline cuối)

    @property
    def lang(self) -> str:
        """Từ đầu tiên của info string (chỉ ký tự word), chữ thường — ví dụ "ts"."""
        m = re.match(r"\w*", self.info)   # \w* luôn khớp (có thể rỗng)
        return m.group(0).lower() if m else ""


def find_fenced_blocks(text: str) -> list[FencedBlock]:
    """Quét *text* theo từng dòng, trả về các fenced code block cấp ngoài cùng.

    Theo CommonMark: fence đóng phải CÙNG ký tự (` hoặc ~), dài >= fence mở và KHÔNG
    có info string; block chưa đóng kéo dài tới hết văn bản. Nhờ vậy nội dung chứa
    fence ngắn hơn (```bash bên trong ````markdown) không làm block bị cắt giữa chừng.

    Nới lỏng cho output LLM (hay bọc tài liệu markdown bằng ``` cùng độ dài với fence
    bên trong): trong một block, dòng fence cùng ký tự CÓ info string (```bash) được coi
    là mở block lồng nhau → fence trơn kế tiếp đóng block lồng đó chứ không đóng block
    ngoài. Fence trơn DÀI HƠN level trong cùng mà khớp đúng độ dài một level bên ngoài
    thì đóng thẳng tới level đó (block lồng bị bỏ ngỏ).
    """
    blocks: list[FencedBlock] = []
    stack: list[tuple[str, int]] = []   # (ký tự fence, độ dài) của các level đang mở
    start = body_start = pos = 0
    info = ""
    for line in text.splitlines(keepends=True):
        line_start, pos = pos, pos + len(line)
        m = _FENCE_LINE_RE.match(line.rstrip("\r\n"))
        if not m:
            continue
        fence, line_info = m.group(1), m.group(2)
        ch, n = fence[0], len(fence)
        if ch == "`" and "`" in line_info:
            continue   # CommonMark: info string của fence backtick không chứa backtick
        if not stack:
            stack.append((ch, n))
            start, body_start, info = line_start, pos, line_info
            continue
        top_ch, top_n = stack[-1]
        if line_info:
            if ch == top_ch:
                stack.append((ch, n))   # block lồng nhau (nới lỏng cho LLM)
            continue
        if ch != top_ch or n < top_n:
            continue   # fence ngắn hơn / khác ký tự → chỉ là nội dung
        idx = len(stack) - 1
        if n > top_n:
            idx = next((i for i in range(idx, -1, -1) if stack[i] == (ch, n)), idx)
        del stack[idx:]
        if not stack:
            blocks.append(FencedBlock(start, pos, info, text[body_start:line_start]))
    if stack:
        blocks.append(FencedBlock(start, len(text), info, text[body_start:]))
    return blocks


def fence_for(content: str) -> str:
    """Trả về fence backtick đủ dài để bọc *content* an toàn khi ghép markdown.

    Dài hơn mọi dãy backtick đầu dòng bên trong content (tối thiểu 3) — ví dụ
    ARCHITECTURE.md chứa ```bash sẽ được bọc bằng ```` để fence nội bộ không thể
    đóng block ngoài (quy tắc CommonMark, marked.js ở UI cũng render đúng).
    """
    longest = max((len(m.group(1)) for m in _FENCE_RUN_RE.finditer(content)), default=0)
    return "`" * max(3, longest + 1)


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


def _reset_role_dir(base: Path) -> Path:
    """Tạo (nếu chưa có) rồi XÓA SẠCH nội dung thư mục artifact của một role.

    Gọi mỗi lần trích xuất: app.py gọi extract_and_save đúng một lần cho mỗi role mỗi
    lần (re-)generate (sau node trong stream chính, và sau mỗi lần Clarifier re-gen
    role đó) — output mới là bộ file ĐẦY ĐỦ của role, nên file cũ không còn sinh ra
    phải biến mất thay vì tiếp tục hiện trong danh sách artifact.

    An toàn khi xóa (raise ValueError → extract_and_save bắt và ghi warning, không xóa gì):
    - *base* không được là symlink (không theo symlink ra ngoài để xóa).
    - Đường dẫn đã resolve phải nằm sâu ĐÚNG 2 cấp dưới ARTIFACT_BASE đã resolve
      ({workflow_id}/{role}) — không bao giờ xóa ARTIFACT_BASE hay thư mục workflow.
    - Chỉ xóa phần tử con: file và symlink bị unlink (KHÔNG theo symlink — đích ở ngoài
      còn nguyên), thư mục thật bị rmtree. Lỗi từng phần tử chỉ ghi warning.

    Trả về đường dẫn đã resolve của thư mục role (dùng cho check confinement của _save).
    """
    root = Path(ARTIFACT_BASE).resolve()
    if base.is_symlink():
        raise ValueError(f"artifact role dir là symlink, từ chối ghi/xóa: {base}")
    base.mkdir(parents=True, exist_ok=True)
    resolved = base.resolve()
    try:
        depth = len(resolved.relative_to(root).parts)
    except ValueError:
        depth = -1
    if depth != 2:
        raise ValueError(f"artifact role dir nằm ngoài {root}/<workflow>/<role>: {resolved}")
    for child in resolved.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            logger.warning("Không xóa được artifact cũ %s: %s", child, exc)
    return resolved


def _do_extract(role: str, output: str, workflow_id: str) -> list[dict]:
    # Sanitize từng segment như list_artifacts/read_artifact — cùng thư mục, và an toàn
    # cho bước xóa của _reset_role_dir dù caller truyền workflow_id lạ.
    base = Path(ARTIFACT_BASE) / _safe_segment(workflow_id) / _safe_segment(role)
    base_resolved = _reset_role_dir(base)  # compute once; passed to every _save call

    # Ghi output thô dưới dạng _output.md TRƯỚC mọi file code: đây là bản gốc đầy đủ
    # không qua xử lý (app.py khôi phục step_outputs từ file này) — trích xuất file code
    # lỗi giữa chừng cũng không làm mất nó. "_output.md" vào seen để file code nào tự
    # đặt tên trùng không ghi đè lên.
    (base_resolved / "_output.md").write_text(output, encoding="utf-8")
    artifacts: list[dict] = [{
        "path": f"{role}/_output.md",
        "filename": "_output.md",
        "language": "markdown",
        "size": len(output.encode("utf-8")),
    }]
    seen: set[str] = {"_output.md"}

    # Chỉ xét block cấp ngoài cùng — fence lồng bên trong (```bash trong ARCHITECTURE.md)
    # là nội dung của file, không phải file riêng.
    blocks = find_fenced_blocks(output)

    # ── Pass 1: tìm và lưu các file theo directive ### FILE: (ưu tiên cao nhất) ───────────────
    structured: set[int] = set()
    for i, b in enumerate(blocks):
        prev_line = output[: b.start].rstrip().rpartition("\n")[2]
        m = _FILE_HDR_RE.search(prev_line)
        if not m:
            continue
        structured.add(i)
        if not b.body.strip():
            continue
        rel = _sanitize_relpath(_strip_hdr_decoration(m.group(1)))
        _save(base, base_resolved, role, rel, b.body, b.lang or "text", artifacts, seen)

    # ── Pass 2: code block tự do (dự phòng khi không có ### FILE: directive) ─────────────
    counter = 0
    for i, b in enumerate(blocks):
        if i in structured:
            continue
        lang         = b.lang or "text"
        content      = b.body
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
            pre = output[: b.start].rstrip()
            m3  = _PRE_FNAME_RE.search(pre)
            if m3:
                filename = m3.group(1).strip()

        if not filename:
            counter += 1
            # Tên tự động không được trùng file đã lưu (Dockerfile thứ hai, hay tên
            # trùng file có directive) — _save() bỏ qua tên trùng, block sẽ mất lặng lẽ.
            filename = _default_name(lang, counter, seen)

        rel = _sanitize_relpath(filename)
        _save(base, base_resolved, role, rel, save_content, lang, artifacts, seen)

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
            files.append({
                "path": f"{role_dir.name}/{rel}",
                "filename": rel,
                "language": _path_language(f.name),
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
    return content, _path_language(path.name)
