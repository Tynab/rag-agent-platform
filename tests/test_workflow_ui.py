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

- Trạng thái: bước lỗi (output "[LỖI…" / failed_steps) hiện ✗ đỏ + thông báo đầu tab,
  bước có section "File generation errors" hiện ⚠ (không phải ✗); status "cancelled" là
  trạng thái kết thúc; nút hủy (POST /workflow/{id}/cancel, 404/409 không xóa view);
  queue_position khi pending. Poll bền bỉ: chỉ 404 là lỗi cuối cùng, lỗi tạm thời giữ
  view + banner + backoff 2.5s → 30s, tab ẩn thì tạm dừng. Chạy bằng node với DOM giả,
  fetch theo kịch bản và setTimeout giả (_SM_RUNNER).

- Đồng bộ backend: mảng STEPS của UI (role, thứ tự, label, tên rút gọn) và các số "15 bước"
  hiển thị cứng khớp WORKFLOW_STEPS / AGENTS của agent-api/agents.py (đọc bằng regex, không cần node).
- Bàn phím: chip Recent, node dải tiến trình, tên file artifact là <button type="button">
  (Tab tới được; Enter/Space = click là hành vi gốc của trình duyệt), aria-label / aria-current,
  viền :focus-visible. Download All thu hồi object URL sau REVOKE_DELAY_MS, không ngay sau click.

F12 (race khi đổi workflow / poll trùng), layout/animation thật và phím Enter/Space thật cần
trình duyệt → kiểm bằng Playwright thủ công, không nằm trong pytest.
"""

import ast
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
  // hljs giả: biết đúng các grammar có trong vendor (common + grammar lẻ đã nạp), không có HCL.
  hljs: { ok: {
    getLanguage(n) {
      return ['python', 'typescript', 'ts', 'nginx', 'dockerfile', 'bash', 'ini', 'groovy', 'shell', 'yml']
        .includes(String(n).toLowerCase()) ? {} : undefined;
    },
    highlight(text, o) {
      calls.push(['hl', o.language, text]);
      if (text === 'BAD') throw new Error('x');
      return { value: '<span class="hljs-x">' + text + '</span>' };
    },
  } },
};
// IntersectionObserver giả (input.lazy): chỉ ghi lại target; ["intersect", i] mới "cuộn tới" khối i.
const io = { targets: [], cb: null, disconnected: 0 };
class FakeIO {
  constructor(cb, opts) { io.cb = cb; io.opts = opts; }
  observe(t) { if (!io.targets.includes(t)) io.targets.push(t); }   // như IO thật: observe lại là no-op
  unobserve(t) { io.targets = io.targets.filter(x => x !== t); }
  disconnect() { io.targets = []; io.disconnected++; }
}
const ctx = { window: { location: { search: '' }, addEventListener() {}, history: { pushState() {} } },
  URLSearchParams, console: { warn() {} }, document: { getElementById: el, addEventListener() {} } };
for (const [name, kind] of Object.entries(input.libs || { marked: 'ok' })) ctx[name] = LIBS[name][kind];
if (input.lazy) ctx.IntersectionObserver = FakeIO;
vm.createContext(ctx);
vm.runInContext(src, ctx);
// <pre><code class=…>text</code></pre> giả
function codeBlock([cls, text]) {
  const code = { className: cls, textContent: text, innerHTML: '', dataset: {}, added: [],
                 classList: { add(c) { code.added.push(c); } } };
  code.parentElement = { querySelector: () => code };
  return code;
}
const view = c => ({ html: c.innerHTML, cls: [...c.added] });
process.stdout.write(JSON.stringify({
  esc: (input.esc || []).map(s => ctx.esc(s)),
  diagram: (input.diagram || []).map(s => ctx.looksLikeAsciiDiagram(s)),
  promote: (input.promote || []).map(t => { const tok = { ...t }; ctx.promoteAsciiDiagram(tok); return tok; }),
  render: (input.render || []).map(s => ctx.renderMarkdown(s)),
  alias: (input.alias || []).map(n => ctx.hlLanguageOf(n)),
  // Mỗi phần tử: danh sách [class, text]; highlightCode chạy HAI lần trên cùng các khối
  // (khối đã tô không được tô lại). input.lazy: sau đó lần lượt "cuộn tới" các khối input.intersect.
  highlight: (input.highlight || []).map(blocks => {
    const codes = blocks.map(codeBlock);
    const root = { querySelectorAll(sel) { calls.push(['sel', sel]); return codes; } };
    ctx.highlightCode(root);
    ctx.highlightCode(root);
    const before = codes.map(view), observed = io.targets.length;
    for (const i of input.intersect || []) {
      io.cb([{ isIntersecting: false, target: codes[i].parentElement }]);
      io.cb([{ isIntersecting: true, target: codes[i].parentElement }]);
    }
    return { before, observed, after: codes.map(view), left: io.targets.length, opts: io.opts };
  }),
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


_HL_LIBS = {"marked": "ok", "DOMPurify": "ok", "hljs": "ok"}
# [class của <code>, nội dung] — khối trơn / không có grammar / bảng alias / khối ném lỗi
_HL_BLOCKS = [
    ["", "User ---> API\nAPI ---> DB"],            # ``` không ghi ngôn ngữ (sơ đồ) — không tự đoán
    ["language-plaintext", "+--+\n|  |"],           # sơ đồ ASCII đã được promoteAsciiDiagram đổi thành code
    ["language-tf", 'resource "x" "y" {}'],         # Terraform: không có grammar → để trơn
    ["language-python", "BAD"],                     # hljs ném lỗi ở khối này…
    ["language-Dockerfile", "FROM python:3.11"],    # …các khối sau vẫn được tô
    ["language-conf", "server { listen 80; }"],
    ["language-toString", "x"],                     # tên trùng prototype của Object → không resolve
    ["language-ts", "const a = 1"],
]


@pytest.mark.parametrize("with_hljs", [True, False], ids=["hljs", "no-hljs"])
def test_highlight_only_known_languages_and_never_throws(with_hljs):
    """Chỉ tô khối có ngôn ngữ hljs biết (qua bảng alias); khối trơn không bị tự đoán; lỗi ở một khối không làm hỏng lượt render."""
    libs = _HL_LIBS if with_hljs else {"marked": "ok", "DOMPurify": "ok"}
    out = _run_js({"libs": libs, "highlight": [_HL_BLOCKS]})
    hl = [c for c in out["calls"] if c[0] in ("sel", "hl")]
    if not with_hljs:
        assert hl == [] and out["highlight"][0]["after"] == [{"html": "", "cls": []}] * len(_HL_BLOCKS)
        return
    # Gọi highlightCode hai lần: khối đã tô (hoặc đã xét) không bị tô lại.
    assert hl == [
        ["sel", 'pre code[class*="language-"]'], ["hl", "python", "BAD"], ["hl", "dockerfile", "FROM python:3.11"],
        ["hl", "nginx", "server { listen 80; }"], ["hl", "ts", "const a = 1"],
        ["sel", 'pre code[class*="language-"]'],
    ]
    after = out["highlight"][0]["after"]
    painted = [i for i, b in enumerate(after) if b["cls"] == ["hljs"]]
    assert painted == [4, 5, 7]
    assert after[4]["html"] == '<span class="hljs-x">FROM python:3.11</span>'
    assert after[0] == after[1] == after[2] == after[3] == {"html": "", "cls": []}


def test_highlight_alias_table():
    names = ["Dockerfile", "docker", "CONF", "nginxconf", "hcl", "terraform", "tf", "env", "shell", "zsh",
             "Jenkinsfile", "text", "mermaid", "python", "yml", "constructor", "__proto__", "", None]
    got = dict(zip(map(str, names), _run_js({"libs": _HL_LIBS, "alias": names})["alias"]))
    assert got == {
        "Dockerfile": "dockerfile", "docker": "dockerfile", "CONF": "nginx", "nginxconf": "nginx",
        "hcl": None, "terraform": None, "tf": None, "env": "ini", "shell": "bash", "zsh": "bash",
        "Jenkinsfile": "groovy", "text": None, "mermaid": None, "python": "python", "yml": "yml",
        "constructor": None, "__proto__": None, "": None, "None": None,
    }


def test_highlight_is_lazy_until_the_block_scrolls_into_view():
    blocks = [["language-python", "a = 1"], ["language-tf", "x"], ["language-python", "b = 2"]]
    out = _run_js({"libs": _HL_LIBS, "lazy": True, "highlight": [blocks], "intersect": [2, 1]})
    h = out["highlight"][0]
    assert h["before"] == [{"html": "", "cls": []}] * 3, "chưa vào khung nhìn → chưa tô"
    assert h["observed"] == 3 and h["opts"] == {"rootMargin": "300px 0px"}
    assert [b["cls"] for b in h["after"]] == [[], [], ["hljs"]]   # chỉ khối đã cuộn tới (tf: không grammar)
    assert h["left"] == 1, "khối đã xét được bỏ khỏi observer"


# ── Trạng thái bước (failed / warn), cancelled, hàng đợi, poll bền bỉ ─────────────
#
# Chạy chính mã JS của trang trong node với DOM giả tối thiểu (mỗi id một phần tử cố định,
# classList thật), fetch theo kịch bản và setTimeout giả (không tự chạy — "tick" mới bắn
# timer poll). Kịch bản là danh sách thao tác; "snap" chụp trạng thái giao diện.

_SM_RUNNER = r"""
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[1], 'utf8');
const src = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));

function makeEl(id, tag = 'DIV') {
  const cls = new Set();
  let first = null;
  const el = {
    id, tagName: tag, style: {}, textContent: '', innerHTML: '', disabled: false, title: '', value: '',
    dataset: {}, scrollLeft: 0, children: [], parent: null, attrs: {},
    classList: {
      toggle(c, on) { if (on === undefined) on = !cls.has(c); if (on) cls.add(c); else cls.delete(c); return on; },
      add(c) { cls.add(c); }, remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); },
    },
    get className() { return [...cls].join(' '); },
    set className(v) { cls.clear(); String(v).split(/\s+/).filter(Boolean).forEach(c => cls.add(c)); },
    get firstChild() { return first || (first = makeEl('')); },
    appendChild(c) { c.parent = el; el.children.push(c); return c; },
    remove() { if (el.parent) el.parent.children = el.parent.children.filter(x => x !== el); el.parent = null; },
    setAttribute(k, v) { el.attrs[k] = String(v); }, removeAttribute(k) { delete el.attrs[k]; },
    getAttribute(k) { return k in el.attrs ? el.attrs[k] : null; }, hasAttribute(k) { return k in el.attrs; },
    addEventListener() {}, focus() {}, select() {}, setSelectionRange() {},
    click() { logs.clicks.push({ attached: !!el.parent, href: el.href, download: el.download,
                                 revokedBefore: logs.revoked.length }); },
    querySelector() { return makeEl(''); }, querySelectorAll() { return []; },
    getBoundingClientRect() { return { left: 0, right: 0, width: 0 }; },
  };
  return el;
}

const els = {}, listeners = {}, winListeners = {}, pushes = [];
const logs = { clicks: [], revoked: [], clip: [], zip: [] };
const document = {
  hidden: false,
  getElementById(id) { return els[id] || (els[id] = makeEl(id)); },
  createElement(tag) { return makeEl('', String(tag).toUpperCase()); },
  addEventListener(type, fn) { listeners[type] = fn; },
  body: makeEl('body'),
  head: makeEl('head'),
  activeElement: null,
};
// input.exec: true/false = execCommand('copy') trả về giá trị đó; vắng mặt = không có execCommand
if (input.exec !== undefined && input.exec !== null) {
  document.execCommand = cmd => {
    const ta = document.body.children[document.body.children.length - 1];
    logs.clip.push(['exec', cmd, ta && ta.value]);
    return input.exec;
  };
}
class FakeURL extends URL {
  static createObjectURL(b) { logs.zip.push(['blob', b]); return 'blob:zip-1'; }
  static revokeObjectURL(u) { logs.revoked.push(u); }
}

const timers = [];
let tid = 0;
const queue = [...(input.responses || [])];
// GET /workflow/{id}/artifacts: lần lượt lấy từ input.artifacts, hết thì dùng lại cái cuối.
const artQueue = [...(input.artifacts || [{}])];
let artLast = artQueue[0];
const fetchLog = [];
// {"defer": tên, "then": response}: request treo tới thao tác ["resolve", tên]. Như fetch
// thật, abort() của AbortController làm nó reject AbortError ngay — trừ khi "ignoreAbort".
const deferred = {};
let confirmAnswer = true;
function resp(r) {
  return { status: r.status, ok: r.status >= 200 && r.status < 300, statusText: r.statusText || '',
    json: async () => { if (r.body === undefined) throw new SyntaxError('not json'); return JSON.parse(JSON.stringify(r.body)); } };
}
const ctx = {
  window: { location: { search: '', href: 'http://agent.test/ui', toString() { return this.href; } },
            history: { pushState(st, t, u) { pushes.push(String(u)); setLocation(String(u)); } },
            matchMedia: () => ({ matches: false }),
            addEventListener(type, fn) { winListeners[type] = fn; },
            isSecureContext: !!input.secure,
            JSZip: class { folder(name) { return { file(n, c) { logs.zip.push(['file', name + '/' + n, c]); } }; }
                           async generateAsync(o) { return 'BLOB:' + o.type; } } },
  document, URL: FakeURL, URLSearchParams, AbortController,
  console: { warn() {}, error() {}, log() {} },
  setTimeout(fn, ms) { const t = { id: ++tid, fn, ms, done: false }; timers.push(t); return t.id; },
  clearTimeout(id) { const t = timers.find(x => x.id === id); if (t) t.done = true; },
  setInterval() { return 0; }, clearInterval() {},
  confirm() { return confirmAnswer; },
  async fetch(url, opts = {}) {
    fetchLog.push((opts.method || 'GET') + ' ' + url);
    if (url.endsWith('/artifacts')) {
      if (artQueue.length) artLast = artQueue.shift();
      return resp({ status: 200, body: { artifacts: artLast } });
    }
    if (url === '/workflows') return resp({ status: 200, body: { workflows: input.workflows || [] } });
    const r = url.endsWith('/cancel') ? input.cancel : queue.shift();
    if (!r) throw new Error('kịch bản hết response cho ' + url);
    if (r.network) throw new TypeError('Failed to fetch');
    if (r.defer) {
      return new Promise((resolve, reject) => {
        deferred[r.defer] = () => resolve(resp(r.then));
        if (opts.signal && !r.ignoreAbort) {
          opts.signal.addEventListener('abort', () => {
            const e = new Error('aborted'); e.name = 'AbortError'; reject(e);
          });
        }
      });
    }
    return resp(r);
  },
};
function setLocation(href) {
  const u = new URL(href, 'http://agent.test/ui');
  ctx.window.location.href = u.href;
  ctx.window.location.search = u.search;
}
if (input.url) setLocation(input.url);
// input.clipboard: 'ok' | 'reject' → có navigator.clipboard; vắng mặt → không có Clipboard API
if (input.clipboard) {
  ctx.navigator = { clipboard: { async writeText(t) {
    logs.clip.push(['api', t]);
    if (input.clipboard === 'reject') throw new Error('NotAllowedError');
  } } };
}
vm.createContext(ctx);
vm.runInContext(src, ctx);
const run = code => vm.runInContext(code, ctx);
const flush = async () => { for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r)); };
const pollFn = run('fetchAndUpdate');
const pollTimers = () => timers.filter(t => !t.done && t.fn === pollFn);

function snap() {
  const g = id => document.getElementById(id);
  const stepEls = run('_stepEls'), tabEls = run('_tabEls');
  return {
    view: g('wf-view').style.display, empty: g('empty-state').style.display,
    loading: g('wf-view').classList.contains('wf-loading'),
    project: g('meta-project').textContent,
    errBox: g('err-box').style.display, errText: g('err-box').textContent,
    banner: g('conn-banner').style.display, bannerText: g('conn-banner-text').textContent,
    pill: g('status-pill').className, label: g('status-label').textContent,
    poll: g('poll-indicator').style.display, pollCls: g('poll-indicator').className,
    pollText: g('poll-text').textContent,
    cancel: { display: g('cancel-btn').style.display, disabled: g('cancel-btn').disabled,
              text: g('cancel-btn').textContent },
    note: { display: g('status-note').style.display, text: g('status-note').textContent },
    stepsFailed: g('meta-steps-failed').textContent,
    timers: pollTimers().map(t => t.ms),
    steps: stepEls && stepEls.map(e => ({ cls: e.node.className, state: e.state,
      circle: e.circle.innerHTML || e.circle.textContent })),
    tabs: tabEls && tabEls.map(t => ({ cls: t.btn.className, icon: t.icon.textContent })),
    body: g('tab-body').innerHTML,
    fetches: fetchLog.length,
    toasts: g('toast-stack').children.map(t => t.textContent),
    toastCls: g('toast-stack').children.map(t => t.className),
    url: ctx.window.location.search, pushes: pushes.length,
    input: g('wid-input').value, copyBtn: g('copy-btn').style.display,
    revoked: [...logs.revoked], otherTimers: timers.filter(t => !t.done && t.fn !== pollFn).map(t => t.ms),
  };
}

(async () => {
  const out = [];
  for (const [op, arg] of input.script) {
    if (op === 'track') { ctx.trackWorkflow(arg); await flush(); }
    else if (op === 'tick') {
      const t = pollTimers().pop();
      if (!t) throw new Error('không có timer poll để tick');
      t.done = true; t.fn(); await flush();
    }
    else if (op === 'hide' || op === 'show') {
      document.hidden = op === 'hide';
      listeners.visibilitychange();
      await flush();
    }
    else if (op === 'confirm') confirmAnswer = arg;
    else if (op === 'cancel') { ctx.requestCancel(); await flush(); }
    else if (op === 'retry') { ctx.pollNow(); await flush(); }   // nút "Thử lại ngay"
    else if (op === 'resolve') { deferred[arg](); await flush(); }
    else if (op === 'recent') { ctx.showRecent(); await flush(); }
    else if (op === 'pop') { setLocation(arg); winListeners.popstate({ state: null }); await flush(); }
    else if (op === 'await') { out.push(await run(arg)); await flush(); }
    else if (op === 'flush') await flush();
    else if (op === 'timers') {
      for (const t of timers.filter(x => !x.done && x.fn !== pollFn && x.ms === arg)) { t.done = true; t.fn(); }
      await flush();
    }
    else if (op === 'tab') ctx.selectTab(arg);
    else if (op === 'eval') out.push(run(arg));
    else if (op === 'snap') out.push(snap());
    else throw new Error('op lạ: ' + op);
  }
  logs.bodyChildren = document.body.children.length;
  logs.pendingTimers = timers.filter(t => !t.done && t.fn !== pollFn).map(t => t.ms);
  process.stdout.write(JSON.stringify({ out, fetchLog, logs }));
})().catch(e => { console.error((e && e.stack) || e); process.exit(1); });
"""


def _run_sm(script: list, responses: list | None = None, cancel: dict | None = None,
            artifacts: list | None = None, url: str | None = None, **env) -> dict:
    """env: secure / clipboard ('ok' | 'reject') / exec (bool) — môi trường clipboard giả."""
    node = shutil.which("node")
    if not node:
        pytest.skip("không có node trên PATH")
    payload = {"script": script, "responses": responses or [], "cancel": cancel,
               "artifacts": artifacts, "url": url, **env}
    proc = subprocess.run(
        [node, "-e", _SM_RUNNER, str(HTML)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _rec(status: str, **kw) -> dict:
    rec = {
        "workflow_id": "wf-1", "status": status, "project": "demo", "user_input": "x",
        "step_outputs": {}, "completed_steps": [], "failed_steps": [], "error": None,
        "created_at": "2026-11-13T10:00:00+00:00", "completed_at": None,
    }
    rec.update(kw)
    return rec


def _ok(rec: dict) -> dict:
    return {"status": 200, "body": rec}


_RUNNING = _rec("running", completed_steps=["ba"], step_outputs={"ba": "ok"})
ROLES = ["ba", "pm", "sa", "ta", "designer", "tl", "fe", "mobile", "dba", "be", "da",
         "tech_lead", "tester", "devsecops", "clarifier"]


def _file_gen_errors_header() -> str:
    """Đọc FILE_GEN_ERRORS_HEADER từ agent-api/workflow.py (không import — tránh dependency)."""
    src = (ROOT / "agent-api" / "workflow.py").read_text(encoding="utf-8")
    m = re.search(r"^FILE_GEN_ERRORS_HEADER\s*=\s*(.+)$", src, re.M)
    assert m, "không thấy FILE_GEN_ERRORS_HEADER trong workflow.py"
    return ast.literal_eval(m.group(1))


def test_cancelled_is_a_terminal_status_with_its_own_badge():
    html = HTML.read_text(encoding="utf-8")
    assert re.search(r"cancelled:\s*\{\s*cls:\s*'badge-cancelled'", html)
    assert ".badge-cancelled" in html
    assert "cancelled:'⊘'" in html  # icon trong danh sách Recent


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_status_stops_polling_and_hides_cancel(status):
    res = _run_sm([["track", "wf-1"], ["snap"]], [_ok(_rec(status, completed_steps=["ba"]))])
    s = res["out"][0]
    assert s["timers"] == [], f"{status}: vẫn còn timer poll"
    assert s["poll"] == "none" and s["cancel"]["display"] == "none"
    assert s["pill"] == f"badge badge-{status}"


def test_step_states_failed_warn_done_with_output_as_source_of_truth():
    header = _file_gen_errors_header()
    outs = {r: f"# {r} ok" for r in ROLES[:11]}
    del outs["sa"]   # failed_steps có 'sa' nhưng không có output → vẫn là failed
    outs["pm"] = "[LỖI trong pm] Regen làm hỏng — không có trong failed_steps"
    outs["dba"] = "# DBA đã được regen sửa"   # failed_steps cũ còn 'dba' nhưng output đã tốt
    outs["fe"] = "### FILE: a.ts\n```ts\nx\n```" + header + "- b.ts: timeout"
    # Lỗi toàn bộ file: tiền tố [LỖI + CŨNG có section lỗi → phải là failed, không phải warn.
    outs["be"] = "[LỖI trong be] Không sinh được file nào (2 file lên kế hoạch đều thất bại).\n" + header + "- x"
    rec = _rec("completed", step_outputs=outs, completed_steps=ROLES[:11],
               failed_steps=["dba", "be", "sa"])
    res = _run_sm([["track", "wf-1"], ["snap"],
                   ["tab", "be"], ["snap"], ["tab", "fe"], ["snap"], ["tab", "ba"], ["snap"],
                   ["tab", "sa"], ["snap"]],
                  [_ok(rec)])
    s = res["out"][0]
    state = dict(zip(ROLES, (st["state"] for st in s["steps"])))
    assert state["ba"] == "done" and state["dba"] == "done" and state["da"] == "done"
    assert state["pm"] == "failed" and state["be"] == "failed" and state["sa"] == "failed"
    assert state["fe"] == "warn"
    assert state["tech_lead"] == "todo"
    steps = dict(zip(ROLES, s["steps"]))
    assert "icon-x" in steps["be"]["circle"] and "failed" in steps["be"]["cls"].split()
    assert "done" not in steps["be"]["cls"].split()   # không còn ✓ xanh cho bước lỗi
    assert "icon-check" in steps["fe"]["circle"] and "warn" in steps["fe"]["cls"].split()
    tabs = dict(zip(ROLES, s["tabs"]))
    assert tabs["be"]["icon"] == "✗" and "t-fail" in tabs["be"]["cls"].split()
    assert tabs["fe"]["icon"] == "⚠" and "t-warn" in tabs["fe"]["cls"].split()
    assert tabs["ba"]["icon"] == "✓" and "t-done" in tabs["ba"]["cls"].split()
    assert s["stepsFailed"] == " · ✗ 3 lỗi"

    be, fe, ba, sa = (o["body"] for o in res["out"][1:])
    assert "step-notice-err" in be and "Bước này thất bại" in be
    assert be.index("step-notice-err") < be.index('id="sub-output"')   # notice ở đầu tab
    assert "step-notice-warn" in fe and "step-notice-err" not in fe
    assert "step-notice" not in ba
    assert "thất bại" in sa


def test_pending_shows_queue_position_and_no_active_step():
    res = _run_sm(
        [["track", "wf-1"], ["snap"], ["tick"], ["snap"], ["tick"], ["snap"]],
        [_ok(_rec("pending", queue_position=2)), _ok(_rec("pending", queue_position=0)),
         _ok(_rec("pending"))],   # backend cũ: không có queue_position
    )
    s2, s0, old = res["out"]
    assert s2["note"] == {"display": "", "text": "Đang chờ trong hàng đợi (vị trí 2)"}
    assert "vị trí 2" in s2["body"]
    assert s0["note"]["text"] == "Đang chờ trong hàng đợi (vị trí 0)"   # 0 không bị coi là "không có"
    assert "vị trí 0" in s0["body"]
    assert old["note"]["text"] == "Đang chờ bắt đầu…"
    for s in (s2, s0, old):
        assert all(st["state"] == "todo" for st in s["steps"]), "run còn xếp hàng không có bước 'active'"
        assert s["cancel"] == {"display": "", "disabled": False, "text": "✕ Hủy workflow"}
        assert s["timers"] == [2500]


def test_cancel_running_then_cancel_requested_then_cancelled():
    requested = {**_RUNNING, "cancel_requested": True}
    done = _rec("cancelled", completed_steps=["ba", "pm"], step_outputs={"ba": "ok", "pm": "ok"},
                cancel_requested=True, completed_at="2026-11-13T10:05:00+00:00")
    res = _run_sm(
        [["track", "wf-1"], ["confirm", False], ["cancel"], ["snap"],
         ["confirm", True], ["cancel"], ["tab", "pm"], ["snap"], ["tick"], ["snap"]],
        [_ok(_RUNNING), _ok(requested), _ok(done)],
        cancel={"status": 200, "body": {"workflow_id": "wf-1", "status": "running"}},
    )
    log = res["fetchLog"]
    declined, requested_s, done_s = res["out"]
    assert declined["fetches"] == 1, "confirm bị từ chối vẫn gửi POST?"
    assert log.count("POST /workflow/wf-1/cancel") == 1
    # Sau POST: poll lại NGAY (không chờ timer 2.5s) → thấy cancel_requested.
    assert log[log.index("POST /workflow/wf-1/cancel") + 1] == "GET /workflow/wf-1"
    assert requested_s["label"] == "Cancelling…"
    assert requested_s["note"] == {"display": "", "text": "Đang hủy — sẽ dừng sau bước hiện tại"}
    assert requested_s["cancel"] == {"display": "", "disabled": True, "text": "Đang hủy…"}
    assert "Đã yêu cầu hủy" in requested_s["body"]   # tab của bước đang chạy
    assert requested_s["timers"] == [2500]
    assert done_s["pill"] == "badge badge-cancelled" and done_s["label"] == "Cancelled"
    assert done_s["timers"] == [] and done_s["cancel"]["display"] == "none"
    assert done_s["note"]["display"] == "none"


def test_cancel_pending_run_is_cancelled_immediately():
    res = _run_sm(
        [["track", "wf-1"], ["cancel"], ["snap"]],
        [_ok(_rec("pending", queue_position=1)), _ok(_rec("cancelled", cancel_requested=True))],
        cancel={"status": 200, "body": {"workflow_id": "wf-1", "status": "cancelled"}},
    )
    s = res["out"][0]
    assert s["pill"] == "badge badge-cancelled" and s["timers"] == []
    assert "đã bị hủy" in s["body"]


@pytest.mark.parametrize(("code", "toast", "kind"), [
    (404, "không tìm thấy", "toast-err"), (409, "đã kết thúc", "toast-warn"), (500, "HTTP 500", "toast-err"),
])
def test_cancel_error_keeps_view_and_reenables_button(code, toast, kind):
    """404/409 của POST /cancel chỉ báo toast + poll lại — không xóa view, không dừng theo dõi."""
    res = _run_sm([["track", "wf-1"], ["cancel"], ["snap"]], [_ok(_RUNNING), _ok(_RUNNING)],
                  cancel={"status": code, "body": {"detail": "x"}})
    s = res["out"][0]
    assert len(s["toasts"]) == 1 and toast in s["toasts"][0]
    assert s["toastCls"] == [f"toast {kind}"]
    assert s["view"] == "block" and s["errBox"] == "none"
    assert s["cancel"] == {"display": "", "disabled": False, "text": "✕ Hủy workflow"}
    assert s["timers"] == [2500]
    assert res["fetchLog"][-1] == "GET /workflow/wf-1"   # đã poll lại để lấy trạng thái thật


def test_poll_backoff_sequence():
    res = _run_sm([["eval", "[1, 2, 3, 4, 5, 6, 9].map(pollBackoffMs)"]])
    assert res["out"][0] == [2500, 5000, 10000, 20000, 30000, 30000, 30000]


def test_transient_errors_keep_view_back_off_and_recover():
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["snap"], ["tick"], ["snap"], ["tick"], ["snap"],
         ["tick"], ["snap"]],
        [_ok(_RUNNING), {"status": 502, "statusText": "Bad Gateway"}, {"network": True},
         {"status": 503, "body": {"detail": "down"}}, _ok(_RUNNING)],
    )
    e502, enet, e503, ok = res["out"]
    for s, ms in ((e502, 2500), (enet, 5000), (e503, 10000)):
        assert s["view"] == "block" and s["errBox"] == "none", "lỗi tạm thời không được xóa view"
        assert s["banner"] == "" and s["timers"] == [ms]
        assert s["pill"] == "badge badge-running"   # dữ liệu lần cuối vẫn hiển thị
        assert "retry" in s["pollCls"].split() and s["pollText"] == "Mất kết nối…"
    assert "HTTP 502" in e502["bannerText"] and "không kết nối được" in enet["bannerText"]
    assert ok["banner"] == "none" and ok["timers"] == [2500]
    assert "retry" not in ok["pollCls"].split() and ok["pollText"] == "Polling…"


def test_retry_now_skips_backoff_timer():
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["tick"], ["snap"], ["retry"], ["snap"]],
        [_ok(_RUNNING), {"status": 500}, {"status": 502}, _ok(_RUNNING)],
    )
    waiting, recovered = res["out"]
    assert waiting["timers"] == [5000] and waiting["banner"] == ""
    assert recovered["fetches"] == waiting["fetches"] + 1   # không chờ hết 5s
    assert recovered["timers"] == [2500] and recovered["banner"] == "none"


def test_render_error_is_not_treated_as_connection_loss():
    """Lỗi render (dữ liệu lạ / bug UI) báo ở err-box, giữ view, poll nhịp thường — không banner/backoff."""
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["snap"], ["tick"], ["snap"]],
        [_ok(_RUNNING), _ok({**_RUNNING, "completed_steps": 5}), _ok(_RUNNING)],
    )
    bad, ok = res["out"]
    assert bad["view"] == "block" and bad["errBox"] == "block" and "Lỗi hiển thị" in bad["errText"]
    assert bad["banner"] == "none" and bad["timers"] == [2500]
    assert ok["errBox"] == "none"


def test_404_is_fatal_and_visibility_does_not_resurrect_polling():
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["snap"], ["hide"], ["show"], ["snap"]],
        [_ok(_RUNNING), {"status": 404, "body": {"detail": "Workflow 'wf-1' không tồn tại"}}],
    )
    s, after = res["out"]
    assert s["view"] == "none" and s["empty"] == "flex" and s["errBox"] == "block"
    assert "không tồn tại" in s["errText"]
    assert s["timers"] == [] and s["poll"] == "none" and s["banner"] == "none"
    assert after["fetches"] == s["fetches"]


def test_hidden_tab_pauses_polling_and_resumes_immediately():
    res = _run_sm(
        [["track", "wf-1"], ["hide"], ["tick"], ["snap"], ["show"], ["snap"]],
        [_ok(_RUNNING), _ok(_RUNNING)],
    )
    hidden, shown = res["out"]
    assert hidden["fetches"] == 1 and hidden["timers"] == []   # bỏ lượt, không đặt timer mới
    assert shown["fetches"] == 2 and shown["timers"] == [2500]  # hiện lại → poll ngay


def test_hidden_tab_first_load_failure_waits_for_visibility():
    """Tab nền chưa từng tải được: lượt đầu vẫn chạy, lỗi thì chờ tab hiện lại (không retry nền)."""
    res = _run_sm(
        [["hide"], ["track", "wf-1"], ["tick"], ["snap"], ["show"], ["snap"]],
        [{"status": 502}, _ok(_RUNNING)],
    )
    hidden, shown = res["out"]
    assert hidden["fetches"] == 1 and hidden["timers"] == [] and hidden["loading"]
    assert shown["fetches"] == 2 and not shown["loading"] and shown["banner"] == "none"


# ── Chốt chặn thế hệ (pollGen / AbortController / single-flight) với request đang bay ──

_A = _rec("running", project="A", completed_steps=["ba"], step_outputs={"ba": "A out"})
_B = _rec("running", project="B", workflow_id="wf-2", completed_steps=["ba", "pm"],
          step_outputs={"ba": "B out", "pm": "B pm"})


def test_poll_now_while_in_flight_polls_again_right_after():
    res = _run_sm(
        [["track", "wf-1"], ["retry"], ["snap"], ["resolve", "g1"], ["snap"]],
        [{"defer": "g1", "then": _ok(_A)}],
    )
    flying, done = res["out"]
    assert flying["fetches"] == 1, "pollNow lúc đang bay không được mở request thứ hai"
    assert done["timers"] == [0], "response về → poll lại ngay (pollAgain), không chờ 2.5s"


@pytest.mark.parametrize("ignore_abort", [False, True], ids=["aborted", "late-response"])
def test_switching_workflow_ignores_previous_in_flight_response(ignore_abort):
    res = _run_sm(
        [["track", "wf-1"], ["track", "wf-2"], ["resolve", "a"], ["snap"]],
        [{"defer": "a", "then": _ok(_A), "ignoreAbort": ignore_abort}, _ok(_B)],
    )
    s = res["out"][0]
    assert s["project"] == "B" and s["timers"] == [2500], "chỉ một vòng poll, dữ liệu của B"
    assert not s["loading"]


def test_new_workflow_failed_first_load_never_shows_previous_workflow():
    res = _run_sm(
        [["track", "wf-1"], ["track", "wf-2"], ["snap"], ["tick"], ["snap"]],
        [_ok(_A), {"status": 502}, _ok(_B)],
    )
    failed, ok = res["out"]
    # Khung build-once còn dữ liệu của A → phải bị ẩn (wf-loading) tới lần render đầu của B.
    assert failed["loading"] and failed["view"] == "block"
    assert "chưa tải được" in failed["bannerText"] and "giữ dữ liệu" not in failed["bannerText"]
    assert not ok["loading"] and ok["project"] == "B" and ok["banner"] == "none"


def test_recent_during_backoff_stops_polling_and_clears_banner():
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["snap"], ["recent"], ["snap"]],
        [_ok(_A), {"status": 502}],
    )
    backoff, recent = res["out"]
    assert backoff["banner"] == "" and backoff["timers"] == [2500]
    assert recent["timers"] == [] and recent["banner"] == "none" and recent["poll"] == "none"


def test_cancel_reply_after_switching_workflow_does_not_leak():
    res = _run_sm(
        [["track", "wf-1"], ["cancel"], ["track", "wf-2"], ["resolve", "c"], ["snap"]],
        [_ok(_A), _ok(_B)],
        cancel={"defer": "c", "then": {"status": 200, "body": {"workflow_id": "wf-1", "status": "running"}}},
    )
    s = res["out"][0]
    assert s["project"] == "B" and s["label"] == "Running"
    assert s["cancel"] == {"display": "", "disabled": False, "text": "✕ Hủy workflow"}
    assert s["note"]["display"] == "none" and s["toasts"] == []
    assert res["fetchLog"].count("GET /workflow/wf-2") == 1   # không poll thêm vì hủy của wf-1


# ── Điều hướng (Back/Forward), thời gian, toast + clipboard, chữ ký tab-body, tải artifact ──

_TODAY = "2026-11-13T10:00:00+00:00"


def test_back_forward_follow_the_url_and_never_push_duplicates():
    res = _run_sm(
        [["flush"], ["snap"],
         ["track", "wf-1"], ["snap"],                 # Track lại cùng id → không push mục trùng
         ["track", "wf-2"], ["snap"],
         ["pop", "/ui?id=wf-1"], ["snap"],            # Back → về wf-1, ngừng poll wf-2
         ["pop", "/ui?id=wf-1#yeu-cau"], ["snap"],    # Back qua mục #anchor → không tải lại
         ["pop", "/ui"], ["snap"],                    # Back về trang ban đầu
         ["pop", "/ui?id=wf-2"], ["snap"]],           # Forward
        [_ok(_A), _ok(_A), _ok(_B), _ok(_A), _ok(_B)],
        url="/ui?id=wf-1",
    )
    load, again, b, back_a, anchor, empty, fwd = res["out"]
    assert load["project"] == "A" and load["pushes"] == 0, "tải trang với ?id= không được push thêm mục"
    assert again["pushes"] == 0 and again["url"] == "?id=wf-1"
    assert b["project"] == "B" and b["pushes"] == 1 and b["url"] == "?id=wf-2"
    assert back_a["project"] == "A" and back_a["pushes"] == 1 and back_a["input"] == "wf-1"
    assert back_a["timers"] == [2500], "chỉ còn đúng một vòng poll (của wf-1)"
    assert res["fetchLog"].count("GET /workflow/wf-2") == 2   # B lần đầu + Forward, không poll nền
    assert anchor["fetches"] == back_a["fetches"] and anchor["timers"] == [2500]
    assert empty["view"] == "none" and empty["empty"] == "flex" and empty["timers"] == []
    assert empty["copyBtn"] == "none" and empty["input"] == "" and empty["poll"] == "none"
    assert fwd["project"] == "B" and fwd["view"] == "block" and fwd["pushes"] == 1


def test_fmt_time_adds_the_date_when_not_today():
    res = _run_sm([["eval", """(() => {
      const now = new Date(2026, 10, 13, 15, 0, 0);
      const today = new Date(2026, 10, 13, 9, 5, 7), other = new Date(2026, 10, 10, 9, 5, 7);
      return {
        today: fmtTime(today.toISOString(), now), todayExp: today.toLocaleTimeString(),
        other: fmtTime(other.toISOString(), now),
        otherDate: other.toLocaleDateString(undefined, { year: 'numeric', month: '2-digit', day: '2-digit' }),
        bad: fmtTime('không-phải-ngày', now), none: fmtTime(null, now), full: fmtFullTime(null),
      };
    })()"""]])
    r = res["out"][0]
    assert r["today"] == r["todayExp"]
    assert r["otherDate"] in r["other"] and r["other"] != r["today"]
    assert r["bad"] == "không-phải-ngày" and r["none"] == "—" and r["full"] == ""


def test_toast_stack_keeps_only_the_newest():
    res = _run_sm([["eval", "['a','b','c','d','e','f'].forEach((m, i) => toast(m, i % 2 ? 'err' : 'ok'))"],
                   ["snap"]])
    s = res["out"][1]
    assert s["toasts"] == ["c", "d", "e", "f"]
    assert s["toastCls"] == ["toast toast-ok", "toast toast-err", "toast toast-ok", "toast toast-err"]
    assert sorted(res["logs"]["pendingTimers"]) == [3000, 3000, 3000, 6000, 6000, 6000]


@pytest.mark.parametrize(
    ("env", "via", "ok"),
    [
        ({"exec": True}, ["exec"], True),                                         # http://<ip> — không có Clipboard API
        ({"secure": True, "clipboard": "ok", "exec": True}, ["api"], True),
        ({"secure": True, "clipboard": "reject", "exec": True}, ["api", "exec"], True),
        ({"clipboard": "ok", "exec": True}, ["exec"], True),                      # có API nhưng không secure context
        ({}, [], False),                                                          # không cách nào sao chép được
    ],
    ids=["insecure-fallback", "secure-api", "api-rejected-fallback", "api-not-secure", "nothing"],
)
def test_copy_id_falls_back_to_exec_command_and_always_gives_feedback(env, via, ok):
    res = _run_sm([["track", "wf-1"], ["await", "copyId()"], ["snap"]], [_ok(_A)], **env)
    s = res["out"][1]
    clip = res["logs"]["clip"]
    assert [c[0] for c in clip] == via
    for c in clip:
        assert (c[1] if c[0] == "api" else c[2]) == "wf-1"
    assert res["logs"]["bodyChildren"] == 0, "textarea tạm phải được gỡ khỏi DOM"
    if ok:
        assert s["toasts"] == ["✓ Đã sao chép Workflow ID"] and s["toastCls"] == ["toast toast-ok"]
    else:
        assert s["toastCls"] == ["toast toast-err"] and "wf-1" in s["toasts"][0]   # hiện ID để chép tay


def test_tab_body_rerenders_on_same_length_output_and_changed_artifact_files():
    done = ROLES[:7]   # … fe → tab tự chọn là fe
    r1 = _rec("running", completed_steps=done, step_outputs={r: f"{r} out-1" for r in done})
    r2 = _rec("running", completed_steps=done, step_outputs={r: f"{r} out-2" for r in done})
    res = _run_sm(
        [["track", "wf-1"], ["snap"], ["await", "loadArtifacts('wf-1')"], ["snap"], ["tick"], ["snap"]],
        [_ok(r1), _ok(r2)],
        artifacts=[{"fe": [{"filename": "src/a.ts", "size": 10}]},
                   {"fe": [{"filename": "src/a.ts", "size": 2048}]}],
    )
    first, resized, regen = (o for o in res["out"] if isinstance(o, dict))
    assert "out-1" in first["body"] and "10 B" in first["body"]
    assert "2.0 KB" in resized["body"], "cùng số file nhưng đổi kích thước → phải render lại"
    assert "out-2" in regen["body"] and "out-1" not in regen["body"], "output cùng độ dài → phải render lại"


def test_live_artifact_reload_only_when_completed_artifact_roles_change():
    """Role artifact đã xong mà /artifacts không có mục nào (vd. không sinh file) → không gọi lại mỗi poll."""
    fe = _rec("running", completed_steps=ROLES[:7], step_outputs={r: "x" for r in ROLES[:7]})
    mob = _rec("running", completed_steps=ROLES[:8], step_outputs={r: "x" for r in ROLES[:8]})
    end = _rec("completed", completed_steps=ROLES[:8], step_outputs={r: "x" for r in ROLES[:8]})
    res = _run_sm(
        [["track", "wf-1"], ["tick"], ["tick"], ["tick"], ["snap"],
         ["tick"], ["tick"], ["snap"],
         ["eval", "_artLoadAt -= ART_RETRY_MS"], ["tick"], ["tick"], ["snap"],
         ["tick"], ["snap"]],
        [_ok(fe)] * 4 + [_ok(mob)] * 2 + [_ok(mob)] * 2 + [_ok(end)],
    )
    log = res["fetchLog"]
    n_art = []
    for o in res["out"]:
        if isinstance(o, dict):
            n_art.append(sum(1 for f in log[: o["fetches"]] if f.endswith("/artifacts")))
    assert n_art == [1, 2, 3, 4], f"số lần GET /artifacts sau mỗi giai đoạn: {n_art}"


def test_download_all_attaches_anchor_revokes_later_and_reports_missing_files():
    res = _run_sm(
        [["track", "wf-1"],
         ["eval", "artifactsCache = { fe: [{ filename: 'a.ts', size: 1 }, { filename: 'b.ts', size: 2 }] }; 0"],
         ["eval", "REVOKE_DELAY_MS"],
         ["await", "downloadRoleZip('fe')"], ["snap"], ["timers", 10000], ["snap"]],
        [_ok(_A), _ok({"path": "a.ts", "language": "ts", "content": "A"}), {"status": 404, "body": {}}],
    )
    logs = res["logs"]
    delay, before, after = res["out"][1], res["out"][3], res["out"][4]
    # Revoke sớm (ngay sau click, hoặc chỉ sau ~1s với zip lớn / máy chậm) có thể hủy tải.
    assert delay == 10000, f"REVOKE_DELAY_MS={delay} — đổi thì sửa luôn thao tác timers ở trên"
    assert logs["clicks"] == [{"attached": True, "href": "blob:zip-1", "download": "fe-artifacts.zip",
                               "revokedBefore": 0}]
    assert ["file", "fe/a.ts", "A"] in logs["zip"]
    assert logs["bodyChildren"] == 0, "anchor phải được gỡ sau khi click"
    # Đã click + gỡ anchor nhưng CHƯA thu hồi URL: còn đúng một timer REVOKE_DELAY_MS đang chờ.
    assert before["revoked"] == [], "object URL bị thu hồi ngay sau click()"
    assert delay in before["otherTimers"]
    assert after["revoked"] == ["blob:zip-1"], "timer REVOKE_DELAY_MS phải thu hồi object URL"
    assert delay not in after["otherTimers"]
    assert before["toasts"][-1].startswith("⚠") and "1/2" in before["toasts"][-1]
    assert "alert(" not in _inline_script()
    # Không helper tải xuống nào khác gọi revokeObjectURL trực tiếp (ngoài setTimeout).
    revokes = re.findall(r"^.*revokeObjectURL.*$", _inline_script(), re.M)
    assert revokes and all("setTimeout(" in line and "REVOKE_DELAY_MS" in line for line in revokes), revokes


def test_running_step_change_is_revealed_in_the_progress_strip():
    """Bước đang chạy đổi → cuộn nó vào khung dải tiến trình (sau reveal của bước được chọn)."""
    spy = ("window.__rev = []; revealInStrip = (strip, el) => "
           "window.__rev.push(strip.id + ':' + (_stepEls ? _stepEls.findIndex(e => e.node === el) : -1)); 0")
    at_ta = _rec("running", completed_steps=ROLES[:3], step_outputs={r: "x" for r in ROLES[:3]})
    at_ops = _rec("running", completed_steps=ROLES[:13], step_outputs={r: "x" for r in ROLES[:13]})
    res = _run_sm(
        [["track", "wf-1"], ["eval", spy], ["tab", "ba"], ["eval", "window.__rev.splice(0)"],
         ["tick"], ["eval", "window.__rev.splice(0)"], ["tick"], ["eval", "window.__rev.splice(0)"]],
        [_ok(at_ta), _ok(at_ops), _ok(at_ops)],
    )
    picked, moved, same = res["out"][1:]
    assert picked == ["steps-track:0", "tabs-nav:-1"]      # chọn tab BA: node BA + nút tab BA
    assert moved == ["steps-track:13"], "bước đang chạy mới (DevSecOps) phải được cuộn vào khung"
    assert same == [], "không đổi gì → không cuộn"


def _css_rule(selector: str) -> str:
    css = re.search(r"<style>(.*?)</style>", HTML.read_text(encoding="utf-8"), re.S).group(1)
    m = re.search(r"(?:^|[}/])\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    assert m, f"không thấy rule {selector}"
    return m.group(1)


def test_tables_wrap_with_sticky_header_and_tabs_stick_below_header():
    """Hợp đồng CSS (hành vi thật kiểm bằng Playwright): ô bảng xuống dòng, tiêu đề bảng và thanh tab dính."""
    cells = _css_rule(".md th, .md td")
    assert "nowrap" not in cells and "min-width" in cells
    assert "max-content" not in _css_rule(".md table") and "separate" in _css_rule(".md table")
    assert "sticky" in _css_rule(".md th") and "nowrap" in _css_rule(".md table.first-col-nowrap td:first-child")
    assert "max-height" in _css_rule(".md .table-wrap")
    tabs = _css_rule(".tabs-nav")
    assert "position: sticky" in tabs and "top: var(--header-h)" in tabs
    assert "overflow: clip" in _css_rule(".content-card")   # overflow:hidden làm sticky không dính


@pytest.mark.parametrize("exec_ok", [True, False], ids=["copied", "failed"])
def test_code_copy_button_flashes_and_announces(exec_ok):
    res = _run_sm(
        [["eval", "window.__b = document.createElement('button'); window.__b.textContent = 'Copy'; 0"],
         ["await", "copyCode(window.__b, 'print(1)')"], ["eval", "window.__b.textContent"], ["snap"],
         ["timers", 1500], ["eval", "window.__b.textContent"]],
        exec=exec_ok,
    )
    flashed, snap, restored = res["out"][2], res["out"][3], res["out"][4]
    assert res["logs"]["clip"] == [["exec", "copy", "print(1)"]]
    assert flashed == ("✓ Đã chép" if exec_ok else "✗ Lỗi") and restored == "Copy"
    assert snap["toastCls"] == ["toast toast-ok" if exec_ok else "toast toast-err"]


def test_extra_highlight_grammars_register_after_the_core_bundle():
    """Grammar lẻ (Dockerfile, nginx…) tự đăng ký vào hljs: phải nạp SAU bản core, và thật sự đăng ký được."""
    html = HTML.read_text(encoding="utf-8")
    srcs = re.findall(r"<script\b[^>]*\bsrc\s*=\s*[\"']/ui/static/vendor/([^\"']+)", html)
    core = srcs.index("highlight-11.12.0.min.js")
    extras = [s for s in srcs if s.startswith("highlight-11.12.0-") and s.endswith(".min.js")]
    assert {"highlight-11.12.0-dockerfile.min.js", "highlight-11.12.0-nginx.min.js"} <= set(extras)
    assert all(srcs.index(s) > core for s in extras)
    node = shutil.which("node")
    if not node:
        pytest.skip("không có node trên PATH")
    js = (
        "const vm = require('vm'), fs = require('fs'); const ctx = {}; vm.createContext(ctx);"
        "for (const f of process.argv.slice(1)) vm.runInContext(fs.readFileSync(f, 'utf8'), ctx);"
        "process.stdout.write(JSON.stringify(['dockerfile', 'nginx', 'dart', 'groovy', 'properties']"
        ".map(l => !!vm.runInContext('hljs', ctx).getLanguage(l))));"
    )
    files = [str(VENDOR / s) for s in srcs[core:core + 1 + len(extras)]]
    proc = subprocess.run([node, "-e", js, *files], capture_output=True, text=True, timeout=30, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [True] * 5


def test_table_first_column_stays_on_one_line_only_when_short():
    """Cột đầu nowrap chỉ khi mọi ô ngắn (ID / tên ngắn) — cột đầu là câu dài phải được xuống dòng."""
    res = _run_sm([["eval", """[
      shortFirstColumn(['FR-001', '  NFR-001  ', 'Password reset']),
      shortFirstColumn(['FR-001', 'The system will be deployed on a single VM with 16 GB RAM']),
      shortFirstColumn(['x'.repeat(32)]), shortFirstColumn(['x'.repeat(33)]), shortFirstColumn([]),
    ]"""]])
    assert res["out"][0] == [True, False, True, False, True]


def test_ui_artifact_roles_match_backend():
    """ARTIFACT_ROLES của UI (quyết định khi nào tải lại /artifacts) phải khớp agent-api/artifacts.py."""
    import artifacts

    m = re.search(r"const ARTIFACT_ROLES\s*=\s*new Set\(\[([^\]]*)\]\)", _inline_script())
    assert m, "không thấy ARTIFACT_ROLES trong workflow.html"
    assert set(re.findall(r"'([a-z_]+)'", m.group(1))) == set(artifacts.ARTIFACT_ROLES)


# ── Danh sách bước của UI khớp registry agent-api (R1a) ───────────────────────

def _ui_steps() -> list[dict]:
    """Đọc mảng STEPS trong workflow.html bằng regex (không cần node → test không bao giờ skip)."""
    m = re.search(r"const STEPS\s*=\s*\[(.*?)\];", _inline_script(), re.S)
    assert m, "không thấy const STEPS = [...] trong workflow.html"
    body = m.group(1)
    entries = re.findall(r"\{([^{}]*)\}", body)
    # Mỗi mục có đúng một `role:` — regex hỏng / mục viết kiểu khác thì fail chứ không bỏ sót.
    assert entries and len(entries) == len(re.findall(r"\brole\s*:", body)), "không parse được hết STEPS"
    steps = []
    for e in entries:
        fields = {k: v for k, _q, v in re.findall(r"(\w+)\s*:\s*(['\"])(.*?)\2", e)}
        assert {"role", "label", "name"} <= fields.keys(), f"mục STEPS thiếu role/label/name: {e.strip()}"
        steps.append(fields)
    return steps


def test_ui_steps_match_backend_workflow_steps_and_agents():
    """Thêm / bớt / đổi thứ tự role ở agents.py mà quên sửa STEPS của UI → fail (UI lệch backend)."""
    import agents

    ui = _ui_steps()
    roles = [s["role"] for s in ui]
    assert roles == list(agents.WORKFLOW_STEPS), "thứ tự role của STEPS (UI) khác WORKFLOW_STEPS"
    assert roles == sorted(agents.AGENTS, key=lambda r: agents.AGENTS[r].step_id), "khác thứ tự step_id của AGENTS"
    for i, s in enumerate(ui, 1):
        cfg = agents.AGENTS[s["role"]]
        assert cfg.role == s["role"] and cfg.step_id == i
        # AGENTS name: "<Label> Agent — <mô tả>". Label của UI phải đúng phần trước " Agent".
        m = re.match(r"(.+?) Agent — (.+)$", cfg.name)
        assert m, f"tên agent không đúng dạng '<Label> Agent — <mô tả>': {cfg.name}"
        assert s["label"] == m.group(1), f"{s['role']}: label UI {s['label']!r} ≠ {m.group(1)!r}"
        # name của UI là bản rút gọn của mô tả: mỗi từ là tiền tố của một từ trong mô tả
        # ("CI/CD & Infra" ⊂ "Infrastructure, CI/CD & Deployment") — đủ bắt role bị lệch vị trí.
        desc = [w.strip(",").lower() for w in m.group(2).split()]
        short = [w.lower() for w in s["name"].split() if w != "&"]
        assert short and all(any(d.startswith(w) for d in desc) for w in short), (
            f"{s['role']}: name UI {s['name']!r} không phải bản rút gọn của {m.group(2)!r}"
        )


def test_hardcoded_step_counts_match_workflow_steps():
    """Các con số "15" hiển thị cứng trong trang phải bằng số bước thật."""
    import agents

    html = HTML.read_text(encoding="utf-8")
    n = len(agents.WORKFLOW_STEPS)
    for pat in (r"(\d+)-Step SDLC Pipeline", r"/\s*(\d+)\s*hoàn thành", r"tiến trình (\d+) bước"):
        found = re.findall(pat, html)
        assert found, f"không thấy {pat!r} trong workflow.html"
        assert all(int(x) == n for x in found), f"{pat!r}: {found} ≠ {n} bước"


# ── Bàn phím: chip Recent, node dải tiến trình, tên file artifact (R2a) ─────────

_NODES = ("_stepEls.map(e => ({ tag: e.node.tagName, type: e.node.type, disabled: e.node.disabled,"
          " tabindex: e.node.getAttribute('tabindex'), label: e.node.getAttribute('aria-label'),"
          " current: e.node.getAttribute('aria-current'), click: typeof e.node.onclick }))")


def test_step_nodes_are_keyboard_buttons_with_state_label_and_aria_current():
    """Node là <button type=button> (Tab tới được; Enter/Space = click — hành vi gốc của trình duyệt,
    kiểm thật bằng Playwright), aria-label mang trạng thái, aria-current="step" theo bước đang chạy."""
    at_ta = _rec("running", completed_steps=ROLES[:3], step_outputs={r: "x" for r in ROLES[:3]})
    fin = _rec("completed", completed_steps=ROLES, step_outputs={r: "x" for r in ROLES})
    other = _rec("running", workflow_id="wf-2", completed_steps=["ba"],
                 step_outputs={"ba": "[LỖI trong ba] x"}, failed_steps=["ba"])
    res = _run_sm(
        [["track", "wf-1"], ["eval", _NODES],
         ["eval", "_stepEls[2].node.onclick(); [activeTab, userPickedTab]"],
         ["tick"], ["eval", _NODES],
         ["track", "wf-2"], ["eval", _NODES]],
        [_ok(at_ta), _ok(fin), _ok(other)],
    )
    running, picked, finished, switched = res["out"]
    assert len(running) == len(ROLES)
    for n in running:
        assert n["tag"] == "BUTTON" and n["type"] == "button", "node phải là <button type=button>"
        assert not n["disabled"] and n["tabindex"] in (None, "0") and n["click"] == "function"
    assert running[0]["label"] == "Bước 1/15: BA — Business Analysis (hoàn thành)"
    assert running[3]["label"] == "Bước 4/15: TA — Technical Architecture (đang chạy)"
    assert running[4]["label"].endswith("(chưa chạy)")
    assert [i for i, n in enumerate(running) if n["current"]] == [3] and running[3]["current"] == "step"
    assert picked == ["sa", True], "click (= Enter/Space trên <button>) phải chọn tab của bước"
    assert all(n["current"] is None for n in finished), "workflow xong → không còn bước hiện tại"
    assert all(n["label"].endswith("(hoàn thành)") for n in finished)
    # Node dựng một lần, dùng lại khi đổi workflow → aria-current / label phải theo workflow mới.
    assert [i for i, n in enumerate(switched) if n["current"]] == [1]
    assert switched[0]["label"].endswith("(thất bại)") and switched[1]["label"].endswith("(đang chạy)")


_WFS = [
    {"workflow_id": "wf-1", "status": "completed", "project": "<img src=x onerror=alert(1)>",
     "completed_steps": 15, "total_steps": 15, "created_at": _TODAY},
    {"workflow_id": "wf-2", "status": "running", "project": "B", "completed_steps": 2, "total_steps": 15,
     "created_at": _TODAY},
]
_CHIPS = ("document.getElementById('recent-grid').children.map(c => ({ tag: c.tagName, type: c.type,"
          " disabled: c.disabled, tabindex: c.getAttribute('tabindex'), current: c.getAttribute('aria-current'),"
          " html: c.innerHTML, click: typeof c.onclick }))")


def test_recent_chips_are_keyboard_buttons_and_mark_the_open_workflow():
    res = _run_sm(
        [["track", "wf-2"], ["recent"], ["eval", _CHIPS],
         ["eval", "document.getElementById('recent-grid').children[0].onclick(); 0"], ["flush"], ["snap"]],
        [_ok(_B), _ok(_A)],
        workflows=_WFS,
    )
    chips, snap = res["out"][0], res["out"][2]
    assert len(chips) == 2
    for c in chips:
        assert c["tag"] == "BUTTON" and c["type"] == "button", "chip phải là <button type=button>"
        assert not c["disabled"] and c["tabindex"] in (None, "0") and c["click"] == "function"
        assert "<div" not in c["html"], "<button> chỉ chứa phrasing content (span), không div"
    # Workflow đang mở (?id=wf-2) được đánh dấu; chip khác thì không.
    assert [c["current"] for c in chips] == [None, "true"]
    # Icon trạng thái ẩn với trình đọc màn hình, có tên trạng thái thay thế; project vẫn được esc().
    assert '<span aria-hidden="true">✓</span><span class="sr-only">Completed:</span>' in chips[0]["html"]
    assert "&lt;img src=x onerror=alert(1)&gt;" in chips[0]["html"] and "<img" not in chips[0]["html"]
    # Click (= Enter/Space trên <button>) mở workflow của chip.
    assert snap["view"] == "block" and snap["input"] == "wf-1" and snap["url"] == "?id=wf-1"
    assert res["fetchLog"][-1] == "GET /workflow/wf-1"


def test_file_tree_name_is_a_button_and_download_icon_has_a_label():
    res = _run_sm([["eval", "renderFileTree('fe', [{ filename: 'src/a\"<b>.ts', size: 3 }])"]])
    html = res["out"][0]
    item = re.search(r'<div class="ft-item" data-role="fe" data-path="src/a&quot;&lt;b&gt;.ts"\s+'
                     r'onclick="viewArtifact\(this\.dataset\.role, this\.dataset\.path\)">(.*?)</div>', html, re.S)
    assert item, "hàng file phải giữ data-role/data-path + onclick cũ"
    row = item.group(1)
    # Tên file là <button> bên trong hàng: Enter/Space → click nổi lên onclick của hàng.
    assert '<button type="button" class="ft-name ft-open">a&quot;&lt;b&gt;.ts</button>' in row
    assert 'role="button"' not in html, "hàng chứa link ⬇ — không được tự làm role=button"
    assert re.search(r'<a class="[^"]*ft-dl"[^>]*aria-label="Tải xuống a&quot;&lt;b&gt;.ts"', row, re.S)


def test_keyboard_focus_styles_and_button_resets():
    """Hợp đồng CSS (nhìn thật kiểm bằng Playwright): viền :focus-visible theo theme, button giữ dáng cũ."""
    for sel in (".recent-chip:focus-visible", ".step-node:focus-visible .step-circle", ".ft-open:focus-visible"):
        assert "outline: 2px solid var(--accent-l)" in _css_rule(sel), sel
    for sel in (".recent-chip", ".step-node"):
        rule = _css_rule(sel)
        assert "font: inherit" in rule and "appearance: none" in rule, sel
    assert "text-align: left" in _css_rule(".recent-chip")
    html = HTML.read_text(encoding="utf-8")
    assert 'tabindex="-1"' not in html
    # Giảm chuyển động: rule toàn cục vẫn còn (bao cả transition của viền focus / chip).
    assert re.search(r"@media \(prefers-reduced-motion: reduce\)\s*\{\s*\*, \*::before, \*::after", html)
