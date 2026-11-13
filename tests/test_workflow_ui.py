"""Regression test cho SPA agent-api/static/workflow.html (G5: F5, F13).

- F5: dữ liệu do server/người dùng kiểm soát (project, workflow_id, currentId, filename)
  không được nội suy thô vào template HTML — phải qua esc() hoặc encodeURIComponent().
  Kiểm tra tĩnh, không cần dependency.
- F13: heuristic nhận diện sơ đồ ASCII không bọc ``` (looksLikeAsciiDiagram) và
  walkTokens promoteAsciiDiagram. Chạy chính mã JS trong trang bằng `node` (vm +
  stub DOM tối thiểu); tự skip nếu máy không có node.

- Offline: marked/DOMPurify/highlight.js/JSZip đi kèm repo trong agent-api/static/vendor/
  (không CDN); mọi file vendor được tham chiếu phải tồn tại, có LICENSE, khớp phiên bản.
  Thiếu thư viện (file vendor hỏng/bị chặn) → init không ném, output hiện text thô đã esc().

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
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const el = () => ({ addEventListener() {}, style: {}, value: '' });
// Thư viện vendor giả: input.libs chọn cái nào "đã tải" ("ok"), "broken" = API không như
// mong đợi (vd. bản marked khác major), vắng mặt = file vendor không tải được.
const calls = [];
const LIBS = {
  marked: {
    ok: { setOptions(o) { calls.push(['setOptions', o]); },
          use(o) { calls.push(['use', typeof o.walkTokens]); },
          parse(s) { if (s.includes('THROW')) throw new Error('boom'); return '<p>' + s + '</p>'; } },
    broken: {},
  },
  DOMPurify: { ok: { sanitize(h, cfg) { calls.push(['sanitize', cfg]); return 'CLEAN:' + h; } } },
  hljs: { ok: { highlightElement(el) { calls.push(['hl', el.id]); if (el.id === 'bad') throw new Error('x'); } } },
};
const ctx = { window: { location: { search: '' } }, URLSearchParams, console: { warn() {} },
  document: { getElementById: el } };
for (const [name, kind] of Object.entries(input.libs || { marked: 'ok' })) ctx[name] = LIBS[name][kind];
vm.createContext(ctx);
vm.runInContext(src, ctx);
process.stdout.write(JSON.stringify({
  esc: (input.esc || []).map(s => ctx.esc(s)),
  diagram: (input.diagram || []).map(s => ctx.looksLikeAsciiDiagram(s)),
  promote: (input.promote || []).map(t => { const tok = { ...t }; ctx.promoteAsciiDiagram(tok); return tok; }),
  render: (input.render || []).map(s => ctx.renderMarkdown(s)),
  highlight: (input.highlight || []).map(ids => ctx.highlightCode({
    querySelectorAll(sel) { calls.push(['sel', sel]); return ids.map(id => ({ id })); } })),
  calls,
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


# ── Offline: thư viện vendor đi kèm repo ───────────────────────────────────────

VENDOR = ROOT / "agent-api" / "static" / "vendor"
# Tên file bắt đầu bằng chữ/số → bỏ qua câu chú thích kiểu "nạp từ /ui/static/vendor/."
_VENDOR_REF = re.compile(r"/ui/static/vendor/([A-Za-z0-9][^\"'`\s)]*)")
# <tên>-<x.y.z>…: phiên bản nằm ngay trong tên file (nâng cấp = đổi tên → cache trình duyệt tự mất).
_VENDOR_NAME = re.compile(r"^(?P<lib>[a-z]+)-(?P<ver>\d+\.\d+\.\d+)[.-]")


def _vendor_refs() -> set[str]:
    return set(_VENDOR_REF.findall(HTML.read_text(encoding="utf-8")))


def test_no_cdn_or_external_assets():
    """README hứa chạy offline: không CDN, không src/href/url() tuyệt đối ra ngoài."""
    html = HTML.read_text(encoding="utf-8")
    assert "cdn.jsdelivr" not in html
    external = re.findall(r"""(?:\b(?:src|href)\s*=\s*["']?|url\(\s*["']?)((?:https?:)?//[^"'\s)>]+)""", html, re.I)
    assert not external, f"tài nguyên ngoài mạng: {external}"


def test_every_referenced_vendor_file_exists():
    refs = _vendor_refs()
    # Không rỗng — nếu regex hỏng thì test này không được "pass" vô nghĩa.
    for lib in ("marked", "dompurify", "highlight", "jszip"):
        assert any(r.startswith(f"{lib}-") for r in refs), f"không thấy {lib} trong {sorted(refs)}"
    missing = sorted(r for r in refs if not (VENDOR / r).is_file() or (VENDOR / r).stat().st_size == 0)
    assert not missing, f"file vendor được tham chiếu nhưng không có trong agent-api/static/vendor/: {missing}"


def test_vendor_files_are_versioned_licensed_and_all_used():
    refs = _vendor_refs()
    for name in sorted(refs):
        m = _VENDOR_NAME.match(name)
        assert m, f"tên file vendor phải có dạng <lib>-<x.y.z>…: {name}"
        lib, ver = m["lib"], m["ver"]
        assert list(VENDOR.glob(f"{lib}-{ver}.LICENSE*")), f"thiếu LICENSE cho {name}"
        if name.endswith(".js"):
            banner = (VENDOR / name).read_text(encoding="utf-8", errors="replace")[:1000]
            assert ver in banner, f"{name}: banner không ghi phiên bản {ver} — tên file sai phiên bản?"
    # Mọi file .js/.css trong vendor/ đều đang được dùng (không sót bản cũ sau nâng cấp).
    unused = sorted(p.name for p in VENDOR.iterdir() if p.suffix in (".js", ".css") and p.name not in refs)
    assert not unused, f"file vendor không còn được tham chiếu: {unused}"


def test_jszip_is_loaded_lazily_not_in_head():
    """JSZip (~100 KB) chỉ nạp khi bấm Download All — không có thẻ script tĩnh."""
    html = HTML.read_text(encoding="utf-8")
    static_srcs = re.findall(r"<script\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)", html, re.I)
    assert static_srcs, "không thấy thẻ script vendor nào?"
    assert not [s for s in static_srcs if "jszip" in s.lower()]
    assert any(r.startswith("jszip-") for r in _vendor_refs())


_FORBID = {"FORBID_TAGS": ["style", "form"], "FORBID_ATTR": ["id", "name"]}
_XSS = "<img src=x onerror=alert(1)> **đậm** & <b>"


@pytest.mark.parametrize(
    "libs",
    [{}, {"marked": "ok"}, {"DOMPurify": "ok"}, {"marked": "broken", "DOMPurify": "ok"}],
    ids=["none", "no-dompurify", "no-marked", "marked-broken"],
)
def test_missing_vendor_libs_fall_back_to_escaped_pre(libs):
    """Thiếu marked/DOMPurify: init không ném (trang vẫn poll) và output hiện text thô đã esc().

    Không bao giờ chèn HTML của marked khi thiếu DOMPurify.
    """
    out = _run_js({"libs": libs, "render": [_XSS]})
    html = out["render"][0]
    assert '<pre class="md-fallback">' in html
    assert "&lt;img src=x onerror=alert(1)&gt; **đậm** &amp; &lt;b&gt;</pre>" in html
    assert "<img" not in html and "<b>" not in html
    assert "CLEAN:" not in html and "<p><img" not in html
    assert not [c for c in out["calls"] if c[0] == "sanitize"]


def test_vendor_libs_present_render_sanitized_markdown():
    out = _run_js({"libs": {"marked": "ok", "DOMPurify": "ok"}, "render": ["hi", "THROW <i>x</i>"]})
    ok, thrown = out["render"]
    assert ok == "CLEAN:<p>hi</p>"
    # marked.parse ném lỗi → vẫn hiện được, dạng text thô đã esc().
    assert '<pre class="md-fallback">THROW &lt;i&gt;x&lt;/i&gt;</pre>' in thrown
    calls = out["calls"]
    assert ["setOptions", {"breaks": True, "gfm": True}] in calls
    assert ["use", "function"] in calls
    assert ["sanitize", _FORBID] in calls


@pytest.mark.parametrize("with_hljs", [True, False], ids=["hljs", "no-hljs"])
def test_highlight_only_explicit_languages_and_never_throws(with_hljs):
    """Khối ``` không ghi ngôn ngữ không bị hljs tự đoán; lỗi ở một khối không làm hỏng lượt render."""
    libs = {"marked": "ok", "DOMPurify": "ok", **({"hljs": "ok"} if with_hljs else {})}
    calls = _run_js({"libs": libs, "highlight": [["a", "bad", "b"]]})["calls"]
    hl = [c for c in calls if c[0] in ("sel", "hl")]
    if with_hljs:
        assert hl == [["sel", 'pre code[class*="language-"]'], ["hl", "a"], ["hl", "bad"], ["hl", "b"]]
    else:
        assert hl == []
