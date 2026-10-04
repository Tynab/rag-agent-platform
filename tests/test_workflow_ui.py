"""Regression test cho SPA agent-api/static/workflow.html (G5: F5, F13).

- F5: dữ liệu do server/người dùng kiểm soát (project, workflow_id, currentId, filename)
  không được nội suy thô vào template HTML — phải qua esc() hoặc encodeURIComponent().
  Kiểm tra tĩnh, không cần dependency.
- F13: heuristic nhận diện sơ đồ ASCII không bọc ``` (looksLikeAsciiDiagram) và
  walkTokens promoteAsciiDiagram. Chạy chính mã JS trong trang bằng `node` (vm +
  stub DOM tối thiểu); tự skip nếu máy không có node.

F12 (race khi đổi workflow / poll trùng) cần DOM + fetch thật → kiểm bằng Playwright
thủ công, không nằm trong pytest.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "agent-api" / "static" / "workflow.html"

# Tên biến/field mang dữ liệu không tin cậy (server trả về hoặc lấy từ URL ?id=).
_TAINTED = re.compile(r"\b(project|workflow_id|currentId|filename)\b")
_SAFE_WRAP = re.compile(r"^\s*(esc|encodeURIComponent)\(")


def _inline_script() -> str:
    html = HTML.read_text(encoding="utf-8")
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, flags=re.S))


def test_tainted_fields_never_interpolated_raw():
    """Mọi `${...}` (trong cùng, không lồng) nhắc tới field không tin cậy phải được bọc esc()/encodeURIComponent()."""
    bad = [
        expr.strip()
        for expr in re.findall(r"\$\{([^{}`]*)\}", _inline_script())
        if _TAINTED.search(expr) and not _SAFE_WRAP.match(expr)
    ]
    assert not bad, f"Nội suy thô dữ liệu không tin cậy vào template: {bad}"


_NODE_RUNNER = r"""
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const src = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const el = () => ({ addEventListener() {}, style: {}, value: '' });
const ctx = { window: { location: { search: '' } }, URLSearchParams,
  document: { getElementById: el }, marked: { setOptions() {}, use() {} } };
vm.createContext(ctx);
vm.runInContext(src, ctx);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify({
  esc: input.esc.map(s => ctx.esc(s)),
  diagram: input.diagram.map(s => ctx.looksLikeAsciiDiagram(s)),
  promote: input.promote.map(t => { const tok = { ...t }; ctx.promoteAsciiDiagram(tok); return tok; }),
}));
"""


def _run_js(payload: dict) -> dict:
    node = shutil.which("node")
    if not node:
        pytest.skip("không có node trên PATH")
    proc = subprocess.run(
        [node, "-e", _NODE_RUNNER, str(HTML)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# Văn xuôi — KHÔNG được biến thành <pre> (F13: trước đây "[Confirmed] … e-mail" bị bắt nhầm).
_PROSE = [
    "[Confirmed] Users log in via e-mail",
    "[ASSUMPTION] FE-BE contract | REST | JSON",
    "[Assumption: K8s target] deploy --> cluster\n[Proposed] cache --> redis",
    "**[GAP]** BE §3 --> FE §4 mismatch\n[VAGUE] retry ---> unclear",
    "Normal prose - with a hyphen and a [link-like] thing - fine.",
    "Run `cat a | grep b | wc -l` then\nuse `x | y | z` again",
    "Một luồng một dòng: Client --> API --> DB",
    "Mapping key -> value\nand another -> here",
    "│ single box line │",
]

# Sơ đồ — PHẢI được nhận diện.
_DIAGRAMS = [
    "[Client] <----> [API]\n[API] <----> [DB]",
    "User ---> API: POST /login\nAPI ---> DB: SELECT user",
    "┌────────┐\n│ Client │\n└────────┘",
    "Client\n   |\n   v\n  API",
    "User          API\n |  POST /login |\n |------------->|",
    "+------+     +-----+\n| Web  |---->| API |\n+------+     +-----+",
]


def test_ascii_diagram_heuristic():
    res = _run_js({"esc": [], "diagram": _PROSE + _DIAGRAMS, "promote": []})["diagram"]
    got_prose, got_diag = res[: len(_PROSE)], res[len(_PROSE):]
    assert [p for p, r in zip(_PROSE, got_prose) if r] == [], "văn xuôi bị nhận nhầm là sơ đồ"
    assert [d for d, r in zip(_DIAGRAMS, got_diag) if not r] == [], "sơ đồ không được nhận diện"


def test_promote_uses_markdown_source_not_rendered_html():
    """Sơ đồ giữ nguyên nguồn ("<---->", khoảng trắng) — trước đây bị strip thành "Client  API"."""
    diag = {"type": "paragraph", "raw": "Client <----> API\n  API <----> DB\n\n", "tokens": [{"type": "text"}]}
    prose = {"type": "paragraph", "raw": "[Confirmed] Users log in via e-mail\n", "tokens": []}
    fence = {"type": "code", "raw": "a --> b\nc --> d", "text": "a --> b\nc --> d"}
    # Trong list item marked dùng token 'text' (raw bị ghép thừa '\n', nguồn đúng ở .text).
    first = {"type": "text", "raw": "First", "text": "First", "tokens": [{"type": "text"}]}
    item = {"type": "list_item", "tokens": [
        first,
        {"type": "text", "raw": "Client --> API\n\nAPI --> DB\n", "text": "Client --> API\nAPI --> DB",
         "tokens": [{"type": "text"}]},
    ]}
    out = _run_js({"esc": [], "diagram": [], "promote": [diag, prose, fence, item]})["promote"]
    assert out[0]["type"] == "code"
    assert out[0]["lang"] == "plaintext"
    assert out[0]["text"] == "Client <----> API\n  API <----> DB"
    assert out[0]["escaped"] is False and "tokens" not in out[0]
    assert out[1] == prose
    assert out[2] == fence
    assert out[3]["tokens"][0] == first
    assert out[3]["tokens"][1]["type"] == "code"
    assert out[3]["tokens"][1]["text"] == "Client --> API\nAPI --> DB"


def test_esc_covers_html_and_attribute_contexts():
    s = """<img src=x onerror='a("b")'>&"""
    out = _run_js({"esc": [s, None, 42], "diagram": [], "promote": []})["esc"]
    assert out[0] == "&lt;img src=x onerror=&#39;a(&quot;b&quot;)&#39;&gt;&amp;"
    assert out[1:] == ["null", "42"]
