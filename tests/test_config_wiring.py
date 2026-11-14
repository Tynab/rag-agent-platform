"""Contract test cấu hình (C1) — code ↔ docker-compose.yml ↔ .env ↔ .env.secrets. Không cần Docker.

`env_file:` CHỈ được dùng cho file secret `.env.secrets` (D1b — gitignored, tạo bằng
scripts/init-secrets.sh, mẫu `.env.secrets.example`); mọi cấu hình còn lại đi qua khối
`environment:` của từng service. Một biến code có đọc nhưng compose quên truyền sẽ âm
thầm rơi về default trong code — dù `.env` có đặt giá trị (đã xảy ra với
WORKFLOW_MAX_SECONDS, INGEST_EMBED_RETRIES, RAG_TIMEOUT của rag-api). Test này chặn lỗi đó:

1. Mọi biến môi trường mà code của service đọc (`os.environ.get`, `os.environ[...]`,
   `os.getenv`, `_require_env`, helper kiểu `_env_int`, `"X" in os.environ`, kể cả khi
   tên biến đi qua hằng module-level) phải có trong `environment:` của service đó, hoặc
   là key của `.env.secrets.example` mà service nạp qua `env_file: .env.secrets` (và không
   bị ghi đè rỗng trong `environment:`).
2. Mọi `${VAR}` interpolate trong compose mà KHÔNG có default (`:-`/`-`) phải được
   định nghĩa trong `.env` (nếu không compose chỉ cảnh báo và thay bằng chuỗi rỗng).
3. Một số bất biến vận hành: cổng host không publish ra mọi interface, service
   stateful không bị Watchtower auto-update, `service_healthy` chỉ trỏ tới service
   có healthcheck.
4. Secrets (D1b): `.env` (được commit) không chứa giá trị secret, `.env.secrets` bị
   gitignore, mỗi secret có đúng service dùng nó, service khác ghi đè rỗng; script
   init-secrets.sh / rotate-neo4j-password.sh chạy đúng hợp đồng (không ghi đè, quyền
   600, không lộ secret ra argv/stdout).

PyYAML có sẵn trong CI qua dependency bắc cầu (langchain-core, uvicorn[standard]).
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.yml"
COMPOSE_CPU_FILE = ROOT / "docker-compose.cpu.yml"
ENV_FILE = ROOT / ".env"
SECRETS_FILE_NAME = ".env.secrets"
SECRETS_EXAMPLE_FILE = ROOT / ".env.secrets.example"
INIT_SECRETS_SCRIPT = ROOT / "scripts" / "init-secrets.sh"
ROTATE_NEO4J_SCRIPT = ROOT / "scripts" / "rotate-neo4j-password.sh"

# Secret trong .env.secrets → service DUY NHẤT được thấy giá trị. Service đọc bằng code
# (rag-api) được đối chiếu thêm với bộ quét env; neo4j/watchtower là image bên thứ ba.
SECRET_CONSUMERS: dict[str, str] = {
    "NEO4J_PASSWORD": "rag-api",
    "NEO4J_AUTH": "neo4j",
    "WATCHTOWER_HTTP_API_TOKEN": "watchtower",
}
# Tên key trông như secret → không được có giá trị trong .env (kể cả dòng comment).
_SECRET_KEY_RE = re.compile(r"TOKEN|PASSWORD|PASSWD|SECRET|API_KEY|PRIVATE_KEY", re.IGNORECASE)

# Thư mục source → tên service trong docker-compose.yml.
SERVICE_DIRS: dict[str, str] = {
    "agent-api": "agent-api",
    "rag-api": "rag-api",
}

# Biến code có đọc nhưng CỐ Ý không truyền qua compose (luôn dùng default trong code).
# Mỗi entry phải kèm lý do; entry thừa (đã wire hoặc code không còn đọc) làm test fail
# để allowlist không mục dần. Hiện chưa có ngoại lệ nào.
NOT_WIRED_ALLOWLIST: dict[str, dict[str, str]] = {
    "agent-api": {},
    "rag-api": {},
}

# Service giao diện người dùng được phép bind theo ${BIND_ADDR}; còn lại chỉ 127.0.0.1.
USER_FACING_SERVICES = {"agent-api", "open-webui"}
# Service giữ dữ liệu/trạng thái — không được để Watchtower recreate giữa chừng.
STATEFUL_SERVICES = ("ollama", "qdrant", "neo4j")

_VAR_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Helper đọc env tự viết: _require_env, _get_env, _env, _env_int, env_bool, ...
_ENV_HELPER_RE = re.compile(r"^_*(?:[a-z0-9]+_)*env(?:_[a-z0-9]+)*$")
_ENV_MAPPING_METHODS = {"get", "setdefault", "pop"}
# ${VAR}, ${VAR:-def}, ${VAR-def}, ${VAR:?err}, ${VAR?err}, ${VAR:+alt}, ${VAR+alt}, $VAR
_INTERP_RE = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)(?P<op>:?[-?+])?[^}]*\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


# ── Quét code: biến môi trường được đọc ────────────────────────────────────────


def _str_items(node: ast.AST) -> list[str]:
    """Chuỗi trong literal tuple/list/set (phần tử hoặc phần tử đầu của tuple con) hoặc key của dict."""
    if isinstance(node, ast.Dict):
        elts = [k for k in node.keys if k is not None]
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        elts = [e.elts[0] if isinstance(e, ast.Tuple) and e.elts else e for e in node.elts]
    else:
        return []
    return [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _module_constants(tree: ast.Module) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Hằng module-level: NAME = "X" và NAME = ("X", ...) / {"X": ...}."""
    strings: dict[str, str] = {}
    collections: dict[str, list[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings[target.id] = value.value
        elif items := _str_items(value):
            collections[target.id] = items
    return strings, collections


def _loop_bindings(tree: ast.Module, collections: dict[str, list[str]]) -> dict[str, list[str]]:
    """Biến vòng lặp/comprehension duyệt qua một hằng tập hợp (for k in _VARS / _MAP.items())."""
    bindings: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            continue
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr in {"items", "keys"}:
            it = it.func.value
        values = collections.get(it.id, []) if isinstance(it, ast.Name) else _str_items(it)
        if not values:
            continue
        target = node.target
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[0]
        if isinstance(target, ast.Name):
            bindings.setdefault(target.id, []).extend(values)
    return bindings


def _is_environ(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
        isinstance(node, ast.Name) and node.id == "environ"
    )


def _env_name_arg(call: ast.Call) -> ast.AST | None:
    """Trả về node chứa TÊN biến nếu *call* là một lần đọc env, ngược lại None."""
    func = call.func
    if not call.args:
        return None
    if isinstance(func, ast.Attribute):
        if func.attr in _ENV_MAPPING_METHODS and _is_environ(func.value):
            return call.args[0]
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    if name == "getenv" or _ENV_HELPER_RE.match(name):
        return call.args[0]
    return None


def env_reads_in_source(src: str) -> set[str]:
    """Tập tên biến môi trường (UPPER_CASE) mà đoạn source Python đọc."""
    tree = ast.parse(src)
    strings, collections = _module_constants(tree)
    loop_vars = _loop_bindings(tree, collections)

    def resolve(node: ast.AST | None) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name):
            if node.id in strings:
                return [strings[node.id]]
            return loop_vars.get(node.id, [])
        return []

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            found.update(resolve(_env_name_arg(node)))
        elif isinstance(node, ast.Subscript) and _is_environ(node.value):
            found.update(resolve(node.slice))
        elif isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            if any(_is_environ(c) for c in node.comparators):
                found.update(resolve(node.left))
    return {name for name in found if _VAR_NAME_RE.match(name)}


def env_reads_in_dir(directory: Path) -> dict[str, list[str]]:
    """{TÊN_BIẾN: [file đọc nó]} cho mọi file .py trong *directory*."""
    reads: dict[str, list[str]] = {}
    for f in sorted(directory.rglob("*.py")):
        for name in env_reads_in_source(f.read_text(encoding="utf-8")):
            reads.setdefault(name, []).append(str(f.relative_to(ROOT)))
    return reads


# ── Đọc compose / .env ─────────────────────────────────────────────────────────


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader + chấp nhận tag riêng của Compose (`!reset`, `!override`) trong file override."""


def _construct_compose_tag(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _construct_compose_tag)


def _load_compose(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load_compose(COMPOSE_FILE)


def _env_map(service: dict) -> dict[str, str | None]:
    """Khối `environment:` dạng {KEY: value}; `KEY` không có `=` (list) / `KEY:` rỗng (map) → None."""
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k): (None if v is None else str(v)) for k, v in env.items()}
    out: dict[str, str | None] = {}
    for item in env:
        key, sep, value = str(item).partition("=")
        out[key] = value if sep else None
    return out


def _env_keys(service: dict) -> set[str]:
    return set(_env_map(service))


def _env_files(service: dict) -> list[str]:
    """Đường dẫn trong `env_file:` (dạng chuỗi, list chuỗi hoặc list {path, required}), bỏ './'."""
    raw = service.get("env_file") or []
    items = [raw] if isinstance(raw, (str, dict)) else list(raw)
    paths = []
    for item in items:
        path = item.get("path", "") if isinstance(item, dict) else item
        paths.append(str(path).removeprefix("./"))
    return paths


def _secret_keys() -> set[str]:
    return set(_dotenv_values(SECRETS_EXAMPLE_FILE))


def _blanked_secrets(service: dict) -> set[str]:
    """Secret bị ghi đè rỗng trong `environment:` (environment thắng env_file) → service không thấy."""
    return {k for k, v in _env_map(service).items() if k in _secret_keys() and v == ""}


def _wired_keys(service: dict) -> set[str]:
    """Key mà container của service thực sự nhận được giá trị (environment + .env.secrets)."""
    blanked = _blanked_secrets(service)
    wired = _env_keys(service) - blanked
    if SECRETS_FILE_NAME in _env_files(service):
        wired |= _secret_keys() - blanked
    return wired


def _labels(service: dict) -> dict[str, str]:
    labels = service.get("labels") or {}
    if isinstance(labels, dict):
        return {k: str(v) for k, v in labels.items()}
    return {k: v for k, _, v in (str(item).partition("=") for item in labels)}


def _dotenv_keys(path: Path) -> set[str]:
    """Key được ĐỊNH NGHĨA (không tính dòng comment) trong file .env."""
    return set(_dotenv_values(path))


_DOTENV_LINE_RE = re.compile(r"^\s*(?P<comment>#\s*)?(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=(?P<value>.*)$")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _dotenv_values(path: Path, *, include_comments: bool = False) -> dict[str, str]:
    """{KEY: value} của file env (dòng cuối thắng). include_comments=True → tính cả `# KEY=value`."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _DOTENV_LINE_RE.match(raw)
        if not m or (m.group("comment") is not None and not include_comments):
            continue
        values[m.group("key")] = _unquote(m.group("value"))
    return values


def _strings(node) -> list[str]:
    """Mọi chuỗi (key lẫn value) trong cấu trúc YAML đã parse — comment đã bị loại."""
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return [node] if isinstance(node, str) else []


def _required_interpolations(doc: dict) -> set[str]:
    """Biến ${VAR} không có default (:-/-) → bắt buộc phải có trong .env."""
    required: set[str] = set()
    for s in _strings(doc):
        for m in _INTERP_RE.finditer(s.replace("$$", "")):
            name = m.group("braced") or m.group("bare")
            op = m.group("op") or ""
            if op.lstrip(":") in ("", "?"):
                required.add(name)
    return required


# ── Test ───────────────────────────────────────────────────────────────────────


def test_env_scanner_handles_indirection():
    """Bộ quét phải bắt được mọi dạng đọc env, kể cả qua hằng module-level/vòng lặp."""
    src = '''
import os
from os import environ, getenv

_DB_ENV = "WORKFLOW_DB_PATH"
_NUM_PREDICT_DEFAULTS = {"PLANNER_NUM_PREDICT": 2048, "FILEGEN_NUM_PREDICT": 8192}
_FLAGS = (("FILEGEN_THINK", ""), ("REPORT_NUM_PREDICT", "16384"))

A = os.environ.get("A_VAR", "x")
B = os.environ["B_VAR"]
C = os.getenv("C_VAR")
D = _require_env("D_VAR")
E = os.environ.get(_DB_ENV)
F = {k: int(os.environ.get(k, d)) for k, d in _NUM_PREDICT_DEFAULTS.items()}
G = environ.get(
    "G_VAR",
    "y",
)
H = getenv("H_VAR")
J = _env_int("J_VAR", 1)
K = "K_VAR" in os.environ
for name, default in _FLAGS:
    os.environ.setdefault(name, default)


def _require_env(param):
    return os.environ.get(param)
'''
    assert env_reads_in_source(src) == {
        "A_VAR", "B_VAR", "C_VAR", "D_VAR", "WORKFLOW_DB_PATH", "PLANNER_NUM_PREDICT",
        "FILEGEN_NUM_PREDICT", "G_VAR", "H_VAR", "J_VAR", "K_VAR", "FILEGEN_THINK",
        "REPORT_NUM_PREDICT",
    }


@pytest.mark.parametrize(
    ("service_dir", "expected"),
    [
        ("agent-api", {"LOG_LEVEL", "OLLAMA_BASE_URL", "RAG_API_URL"}),
        ("rag-api", {"LOG_LEVEL", "OLLAMA_BASE_URL", "QDRANT_URL", "EMBEDDING_MODEL"}),
    ],
)
def test_env_scanner_finds_known_reads(service_dir, expected):
    """Chốt chặn: nếu bộ quét hỏng (trả rỗng) thì test wiring bên dưới sẽ pass oan."""
    found = set(env_reads_in_dir(ROOT / service_dir))
    assert expected <= found, f"Bộ quét không thấy {sorted(expected - found)} trong {service_dir}/"


@pytest.mark.parametrize(("service_dir", "service"), sorted(SERVICE_DIRS.items()))
def test_every_env_var_read_by_code_is_wired_in_compose(compose, service_dir, service):
    reads = env_reads_in_dir(ROOT / service_dir)
    # environment: + secret nạp qua `env_file: .env.secrets` (trừ secret bị ghi đè rỗng).
    wired = _wired_keys(compose["services"][service])
    allow = NOT_WIRED_ALLOWLIST.get(service, {})

    missing = {name: files for name, files in reads.items() if name not in wired and name not in allow}
    assert not missing, (
        f"{service}: code đọc biến môi trường nhưng docker-compose.yml không truyền vào "
        f"`environment:` (thêm `NAME: ${{NAME:-<default>}}` và ghi chú trong .env/README; "
        f"secret thì thêm vào .env.secrets.example + SECRET_CONSUMERS + `env_file: .env.secrets`): "
        + ", ".join(f"{n} ({', '.join(f)})" for n, f in sorted(missing.items()))
    )

    stale = sorted(n for n in allow if n not in reads or n in wired)
    assert not stale, f"{service}: NOT_WIRED_ALLOWLIST có entry thừa (đã wire hoặc code không còn đọc): {stale}"


@pytest.mark.parametrize("compose_file", [COMPOSE_FILE, COMPOSE_CPU_FILE], ids=lambda p: p.name)
def test_required_compose_interpolations_are_defined_in_dotenv(compose_file):
    if not ENV_FILE.exists():
        pytest.skip(".env không có trong checkout")
    required = _required_interpolations(_load_compose(compose_file))
    missing = sorted(required - _dotenv_keys(ENV_FILE))
    assert not missing, (
        f"{compose_file.name} dùng ${{VAR}} không có default nhưng .env không định nghĩa: {missing} "
        "— thêm vào .env hoặc dùng ${VAR:-default}"
    )


def test_cpu_override_only_touches_existing_services(compose):
    override = _load_compose(COMPOSE_CPU_FILE)
    unknown = sorted(set(override.get("services") or {}) - set(compose["services"]))
    assert not unknown, f"docker-compose.cpu.yml override service không tồn tại: {unknown}"


def test_published_ports_are_not_bound_to_all_interfaces(compose):
    bad = []
    for name, svc in compose["services"].items():
        allowed = ("127.0.0.1",) + (("${BIND_ADDR:-127.0.0.1}",) if name in USER_FACING_SERVICES else ())
        for port in svc.get("ports") or []:
            if isinstance(port, dict):
                host_ip = str(port.get("host_ip", ""))
            else:
                host_ip = next((a for a in allowed if str(port).startswith(a + ":")), "")
            if host_ip not in allowed:
                bad.append(f"{name}: {port!r}")
    assert not bad, (
        "Cổng phải bind 127.0.0.1 (hạ tầng) hoặc ${BIND_ADDR:-127.0.0.1} (agent-api, open-webui): "
        + "; ".join(bad)
    )


def test_stateful_services_are_excluded_from_watchtower(compose):
    services = compose["services"]
    for name in STATEFUL_SERVICES:
        assert _labels(services[name]).get("com.centurylinklabs.watchtower.enable") == "false", (
            f"{name} là service stateful — phải đặt label com.centurylinklabs.watchtower.enable=false"
        )
    # Cron WATCHTOWER_SCHEDULE tính theo TZ của container; thiếu TZ = UTC.
    assert "TZ" in _env_keys(services["watchtower"])


def test_service_healthy_dependencies_have_healthcheck(compose):
    services = compose["services"]
    bad = []
    for name, svc in services.items():
        deps = svc.get("depends_on") or {}
        if not isinstance(deps, dict):
            continue
        for dep, cfg in deps.items():
            if (cfg or {}).get("condition") != "service_healthy":
                continue
            hc = services.get(dep, {}).get("healthcheck") or {}
            if not hc.get("test") or hc.get("disable"):
                bad.append(f"{name} → {dep}")
    assert not bad, f"depends_on service_healthy nhưng service đích không có healthcheck: {bad}"


def _built_services(compose: dict) -> dict[str, dict]:
    return {name: svc for name, svc in compose["services"].items() if svc.get("build")}


def _dockerfile(svc: dict) -> Path:
    build = svc["build"]
    context = build if isinstance(build, str) else build.get("context", ".")
    dockerfile = "Dockerfile" if isinstance(build, str) else build.get("dockerfile", "Dockerfile")
    return ROOT / context / dockerfile


def _healthcheck_port(test: list[str]) -> int:
    m = re.search(r"http://127\.0\.0\.1:(\d+)/health\b", " ".join(test))
    assert m, f"healthcheck phải gọi http://127.0.0.1:<port>/health: {test}"
    return int(m.group(1))


def test_built_service_healthchecks_do_not_depend_on_curl(compose):
    """R5a: image build local không cài curl/wget (không apt-get) → healthcheck dùng python của image."""
    built = _built_services(compose)
    assert set(built) == {"rag-api", "agent-api"}
    for name, svc in built.items():
        test = (svc.get("healthcheck") or {}).get("test")
        assert isinstance(test, list) and test[:3] == ["CMD", "python", "-c"], (
            f"{name}: healthcheck phải là [\"CMD\", \"python\", \"-c\", ...] (image không có curl): {test}"
        )
        assert not re.search(r"\b(curl|wget)\b", " ".join(test)), f"{name}: healthcheck không được gọi curl/wget"

        lines = [ln for ln in _dockerfile(svc).read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
        text = "\n".join(lines)
        assert re.search(r"^FROM\s+python:", text, re.M), f"{name}: base image phải có python (healthcheck dùng python)"
        assert not re.search(r"\b(curl|wget)\b", text), f"{name}: Dockerfile không cần cài curl/wget nữa"
        # Cổng healthcheck = cổng uvicorn trong Dockerfile = cổng container được publish.
        port = _healthcheck_port(test)
        assert f'"--port", "{port}"' in text, f"{name}: healthcheck gọi cổng {port} nhưng uvicorn không nghe cổng đó"
        assert any(str(p).endswith(f":{port}") for p in svc.get("ports") or []), f"{name}: cổng {port} không khớp ports:"


@pytest.mark.parametrize("service", ["rag-api", "agent-api"])
def test_built_service_healthcheck_command_detects_healthy_and_unhealthy(compose, service):
    """Chạy đúng lệnh `python -c` của healthcheck với server HTTP tạm: 200 → exit 0; 500 / không ai nghe → exit ≠ 0."""
    import http.server
    import socket
    import threading

    test = compose["services"][service]["healthcheck"]["test"]
    code, port = test[3], _healthcheck_port(test)

    class Handler(http.server.BaseHTTPRequestHandler):
        status = 200

        def do_GET(self):  # tên method do http.server quy định
            self.send_response(self.status if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        def log_message(self, *args):
            pass

    def run_check(target_port: int) -> int:
        patched = code.replace(f"127.0.0.1:{port}/", f"127.0.0.1:{target_port}/")
        assert patched != code
        return subprocess.run([sys.executable, "-c", patched], capture_output=True, timeout=30).returncode

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        live_port = server.server_address[1]
        assert run_check(live_port) == 0, "server trả 200 → healthcheck phải exit 0"
        Handler.status = 500
        assert run_check(live_port) != 0, "server trả 500 → healthcheck phải exit khác 0"
    finally:
        server.shutdown()
        server.server_close()

    with socket.socket() as s:  # lấy một cổng trống rồi đóng → không ai nghe
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]
    assert run_check(closed_port) != 0, "không kết nối được → healthcheck phải exit khác 0"


def test_neo4j_runs_without_plugins_and_code_does_not_need_them(compose):
    """D5a: rag-api chỉ dùng Cypher thuần → neo4j không cài plugin (APOC tải jar từ internet mỗi lần tạo container)."""
    assert "NEO4J_PLUGINS" not in _env_keys(compose["services"]["neo4j"]), (
        "neo4j không cần plugin — rag-api không gọi procedure apoc.*; bỏ NEO4J_PLUGINS"
    )
    uses_apoc = [
        str(f.relative_to(ROOT)) for f in sorted((ROOT / "rag-api").rglob("*.py"))
        if "apoc." in f.read_text(encoding="utf-8").lower()
    ]
    assert not uses_apoc, f"code gọi apoc.* nhưng neo4j không cài APOC (NEO4J_PLUGINS): {uses_apoc}"


# image: <repo>:${<X>_IMAGE_TAG:-<tag mặc định>} — repo không chứa tag/biến, default không rỗng.
_PINNABLE_IMAGE_RE = re.compile(r"^(?P<repo>[a-z0-9][a-z0-9._/-]*):\$\{(?P<var>[A-Z][A-Z0-9_]*_IMAGE_TAG):-(?P<default>[^}]+)\}$")


def test_third_party_images_use_a_configurable_tag(compose):
    """D4a: mọi image không build local phải pin được qua biến *_IMAGE_TAG (có default) trong .env."""
    bad, variables = [], {}
    for name, svc in compose["services"].items():
        if svc.get("build"):
            continue
        m = _PINNABLE_IMAGE_RE.match(str(svc.get("image", "")))
        if not m:
            bad.append(f"{name}: {svc.get('image')!r}")
            continue
        variables.setdefault(m.group("var"), []).append(name)
    assert not bad, f"image bên thứ ba phải có dạng <repo>:${{<X>_IMAGE_TAG:-<tag>}}: {bad}"
    shared = {v: s for v, s in variables.items() if len(s) > 1}
    assert not shared, f"mỗi service một biến *_IMAGE_TAG riêng: {shared}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    undocumented = sorted(v for v in variables if f"`{v}`" not in readme)
    assert not undocumented, f"thiếu mô tả trong bảng biến môi trường của README: {undocumented}"
    if ENV_FILE.exists():
        dotenv = _dotenv_values(ENV_FILE, include_comments=True)
        missing = sorted(v for v in variables if v not in dotenv)
        assert not missing, f"thiếu dòng (có thể comment) `# {missing[0]}=...` trong khối Phiên bản image của .env: {missing}"


# ── Secrets (D1b): .env.secrets ↔ compose ↔ scripts ─────────────────────────────


def test_dotenv_has_no_secret_values():
    """.env được commit vào git → không dòng nào (kể cả `# KEY=value`) được mang giá trị secret."""
    if not ENV_FILE.exists():
        pytest.skip(".env không có trong checkout")
    offending = []
    for lineno, raw in enumerate(ENV_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        m = _DOTENV_LINE_RE.match(raw)
        if not m:
            continue
        key = m.group("key")
        if (_SECRET_KEY_RE.search(key) or key in SECRET_CONSUMERS) and _unquote(m.group("value")):
            offending.append(f"{key} (dòng {lineno})")
    # Chỉ nêu tên key — không bao giờ in giá trị.
    assert not offending, f".env (được commit) chứa giá trị secret — chuyển sang .env.secrets: {offending}"


def test_secrets_example_lists_exactly_the_consumed_keys_without_values():
    values = _dotenv_values(SECRETS_EXAMPLE_FILE)
    assert set(values) == set(SECRET_CONSUMERS), (
        f".env.secrets.example phải liệt kê đúng các secret trong SECRET_CONSUMERS: "
        f"thừa {sorted(set(values) - set(SECRET_CONSUMERS))}, thiếu {sorted(set(SECRET_CONSUMERS) - set(values))}"
    )
    filled = sorted(k for k, v in values.items() if v)
    assert not filled, f".env.secrets.example là file mẫu được commit — giá trị phải để rỗng: {filled}"


def test_gitignore_ignores_secrets_file_but_not_example():
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("cần git checkout")

    def ignored(name: str) -> bool:
        # --no-index: xét pattern .gitignore kể cả với file đã track (không thì luôn "không ignore").
        res = subprocess.run([git, "check-ignore", "-q", "--no-index", name], cwd=ROOT, capture_output=True, text=True)
        assert res.returncode in (0, 1), res.stderr
        return res.returncode == 0

    assert ignored(SECRETS_FILE_NAME), ".gitignore phải chặn .env.secrets"
    assert ignored(".env.secrets.tmp.Ab12Cd"), ".gitignore phải chặn file tạm của init-secrets.sh"
    assert not ignored(".env.secrets.example"), ".env.secrets.example phải commit được (thêm !.env.secrets.example)"


def test_each_secret_reaches_only_its_consumer(compose):
    """Mỗi secret tới đúng service dùng nó; service khác nạp .env.secrets phải ghi đè rỗng.

    env_file nạp TOÀN BỘ file vào container. Với neo4j điều này là bắt buộc chứ không chỉ
    là least-privilege: entrypoint của image ghi mọi biến NEO4J_* có giá trị thành setting
    trong neo4j.conf, nên NEO4J_PASSWORD lọt vào sẽ thành dòng "PASSWORD=<mật khẩu>".
    """
    services = compose["services"]
    users = {name for name, svc in services.items() if SECRETS_FILE_NAME in _env_files(svc)}
    assert users == set(SECRET_CONSUMERS.values()), (
        f"Service nạp env_file .env.secrets phải đúng là {sorted(set(SECRET_CONSUMERS.values()))}, "
        f"hiện là {sorted(users)}"
    )
    for key, consumer in SECRET_CONSUMERS.items():
        assert key in _wired_keys(services[consumer]), (
            f"{consumer} phải nhận {key} từ .env.secrets (khai báo env_file và không ghi đè trong environment:)"
        )
    for name in sorted(users):
        own = {k for k, c in SECRET_CONSUMERS.items() if c == name}
        leaked = sorted(_secret_keys() - own - _blanked_secrets(services[name]))
        assert not leaked, f'{name} nạp .env.secrets nhưng không ghi đè rỗng secret của service khác: {leaked} (thêm `KEY: ""`)'
    assert "NEO4J_PASSWORD" in _blanked_secrets(services["neo4j"]), 'neo4j phải đặt NEO4J_PASSWORD: "" (neo4j.conf)'


def test_secrets_read_by_code_belong_to_that_service():
    """Secret của service có code trong repo phải thật sự được code đó đọc (danh sách không mục)."""
    dirs = {service: directory for directory, service in SERVICE_DIRS.items()}
    for key, consumer in SECRET_CONSUMERS.items():
        if consumer in dirs:
            assert key in env_reads_in_dir(ROOT / dirs[consumer]), f"{consumer} không còn đọc {key} — bỏ khỏi .env.secrets"


@pytest.mark.parametrize("compose_file", [COMPOSE_FILE, COMPOSE_CPU_FILE], ids=lambda p: p.name)
def test_compose_never_interpolates_secrets(compose_file):
    """${SECRET} chỉ lấy được từ .env (được commit) → secret phải đi qua env_file .env.secrets."""
    found = set()
    for s in _strings(_load_compose(compose_file)):
        for m in _INTERP_RE.finditer(s.replace("$$", "")):
            name = m.group("braced") or m.group("bare")
            if name in SECRET_CONSUMERS or _SECRET_KEY_RE.search(name):
                found.add(name)
    assert not found, f"{compose_file.name} interpolate secret {sorted(found)} — dùng env_file .env.secrets"


# ── scripts/init-secrets.sh và scripts/rotate-neo4j-password.sh ────────────────

_BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(_BASH is None or sys.platform == "win32", reason="cần bash (Linux/macOS)")


def _script_env(**extra: str) -> dict[str, str]:
    """Env của tiến trình test, bỏ mọi secret có sẵn (tránh script vô tình 'giữ' giá trị ngoài)."""
    skip = set(SECRET_CONSUMERS) | {"NEO4J_OLD_PASSWORD", "KEEP_NEO4J_PASSWORD"}
    env = {k: v for k, v in os.environ.items() if k not in skip}
    env.update(extra)
    return env


def _make_repo(tmp_path: Path, *scripts: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for script in scripts:
        shutil.copy2(script, repo / "scripts" / script.name)
    return repo


def _run_script(repo: Path, script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # Gọi qua `bash <file>` — không phụ thuộc bit thực thi của file trong checkout.
    assert _BASH is not None
    return subprocess.run(
        [_BASH, str(repo / "scripts" / script.name)],
        env=env, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
    )


def _run_init(repo: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return _run_script(repo, INIT_SECRETS_SCRIPT, _script_env(**env))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@needs_bash
def test_init_secrets_creates_every_key_with_strong_values_and_mode_600(tmp_path):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    res = _run_init(repo)
    assert res.returncode == 0, res.stderr

    secrets = repo / SECRETS_FILE_NAME
    assert _mode(secrets) == 0o600
    values = _dotenv_values(secrets)
    assert set(values) == _secret_keys(), "init-secrets.sh phải sinh đúng các key của .env.secrets.example"
    assert not sorted(k for k, v in values.items() if not v), "mọi key phải có giá trị"
    assert re.fullmatch(r"[0-9a-f]{64}", values["NEO4J_PASSWORD"])
    assert re.fullmatch(r"[0-9a-f]{64}", values["WATCHTOWER_HTTP_API_TOKEN"])
    assert values["NEO4J_PASSWORD"] != values["WATCHTOWER_HTTP_API_TOKEN"]
    assert values["NEO4J_AUTH"] == "neo4j/" + values["NEO4J_PASSWORD"]

    output = res.stdout + res.stderr
    assert not sorted(k for k, v in values.items() if v in output), "script không được in giá trị secret"
    assert sorted(p.name for p in repo.iterdir()) == [SECRETS_FILE_NAME, "scripts"], "không để lại file tạm"


@needs_bash
def test_init_secrets_never_overwrites_an_existing_file(tmp_path):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    assert _run_init(repo).returncode == 0
    secrets = repo / SECRETS_FILE_NAME
    first = secrets.read_bytes()

    again = _run_init(repo)
    assert again.returncode == 0, again.stderr
    assert secrets.read_bytes() == first, "chạy lại không được sinh giá trị mới"

    # File người dùng tự sửa (đủ key, hợp lệ): giữ nguyên nội dung + quyền, kể cả khi truyền
    # KEEP_NEO4J_PASSWORD hay có NEO4J_PASSWORD trong môi trường.
    edited = (
        "NEO4J_PASSWORD=user-edited-value\n"
        "NEO4J_AUTH=neo4j/user-edited-value\n"
        "WATCHTOWER_HTTP_API_TOKEN=user-edited-token\n"
    )
    secrets.write_text(edited, encoding="utf-8")
    secrets.chmod(0o640)
    res = _run_init(repo, KEEP_NEO4J_PASSWORD="Another-Pass-123", NEO4J_PASSWORD="Ambient-Pass-456")
    assert res.returncode == 0, res.stderr
    assert secrets.read_text(encoding="utf-8") == edited
    assert _mode(secrets) == 0o640
    assert "chmod 600" in res.stderr, "phải cảnh báo khi file secret mở quyền cho group/other"
    output = res.stdout + res.stderr
    for value in ("user-edited-value", "user-edited-token", "Another-Pass-123", "Ambient-Pass-456"):
        assert value not in output


_COMPLETE_SECRETS = {
    "NEO4J_PASSWORD": "Valid-Pass_123",
    "NEO4J_AUTH": "neo4j/Valid-Pass_123",
    "WATCHTOWER_HTTP_API_TOKEN": "Valid-Token_456",
}


@needs_bash
@pytest.mark.parametrize(
    ("case", "content", "expected_keys"),
    [
        # Người dùng `cp .env.secrets.example .env.secrets` rồi chạy script: mọi key rỗng.
        ("example-copy", None, set(SECRET_CONSUMERS)),
        ("missing-token", {k: v for k, v in _COMPLETE_SECRETS.items() if k != "WATCHTOWER_HTTP_API_TOKEN"},
         {"WATCHTOWER_HTTP_API_TOKEN"}),
        ("auth-mismatch", {**_COMPLETE_SECRETS, "NEO4J_AUTH": "neo4j/Other-Pass_999"}, {"NEO4J_AUTH"}),
        ("unsafe-password", {**_COMPLETE_SECRETS, "NEO4J_PASSWORD": "Valid$Pass_1",
                             "NEO4J_AUTH": "neo4j/Valid$Pass_1"}, {"NEO4J_PASSWORD"}),
        ("short-password", {**_COMPLETE_SECRETS, "NEO4J_PASSWORD": "Sh0rt_x", "NEO4J_AUTH": "neo4j/Sh0rt_x"},
         {"NEO4J_PASSWORD"}),
    ],
)
def test_init_secrets_rejects_an_existing_but_unusable_file_without_touching_it(tmp_path, case, content, expected_keys):
    """File đã có nhưng thiếu/rỗng key → thoát ≠ 0, nêu TÊN key, không sửa file, không in giá trị."""
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    secrets = repo / SECRETS_FILE_NAME
    if content is None:
        shutil.copy2(SECRETS_EXAMPLE_FILE, secrets)
    else:
        secrets.write_text("".join(f"{k}={v}\n" for k, v in content.items()), encoding="utf-8")
    secrets.chmod(0o600)
    before = secrets.read_bytes()

    res = _run_init(repo)
    assert res.returncode != 0, f"{case}: file không dùng được thì phải thoát ≠ 0"
    assert secrets.read_bytes() == before, "không được sửa file đã có"
    assert sorted(p.name for p in repo.iterdir()) == [SECRETS_FILE_NAME, "scripts"], "không để lại file tạm"
    for key in expected_keys:
        assert key in res.stderr, f"{case}: phải nêu tên key lỗi {key}"
    for key in set(SECRET_CONSUMERS) - expected_keys:
        assert f"- {key}" not in res.stderr, f"{case}: {key} hợp lệ, không được báo lỗi"
    output = res.stdout + res.stderr
    leaked = [v for v in (content or {}).values() if v and v.split("/")[-1] in output]
    assert not leaked, f"{case}: không được in giá trị secret"


@needs_bash
def test_init_secrets_ignores_an_ambient_neo4j_password(tmp_path):
    """NEO4J_PASSWORD còn export trong shell (vd. `set -a; . ./.env.secrets`) KHÔNG được giữ lại.

    Nếu không, luồng đổi mật khẩu của README (mv .env.secrets .env.secrets.old && init-secrets.sh
    trong cùng shell) sinh lại đúng mật khẩu cũ và rotate-neo4j-password.sh thành no-op.
    """
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    res = _run_init(repo, NEO4J_PASSWORD="Ambient-Old-Pass1")
    assert res.returncode == 0, res.stderr
    values = _dotenv_values(repo / SECRETS_FILE_NAME)
    assert re.fullmatch(r"[0-9a-f]{64}", values["NEO4J_PASSWORD"]), "phải sinh mật khẩu mới"
    assert values["NEO4J_AUTH"] == "neo4j/" + values["NEO4J_PASSWORD"]
    output = res.stdout + res.stderr
    assert "Ambient-Old-Pass1" not in output
    assert "KEEP_NEO4J_PASSWORD" in res.stderr, "phải nói rõ biến môi trường bị bỏ qua và cách giữ mật khẩu"


@needs_bash
def test_init_secrets_keeps_an_existing_neo4j_password(tmp_path):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    # KEEP_ thắng NEO4J_PASSWORD trong môi trường (bị bỏ qua).
    res = _run_init(repo, KEEP_NEO4J_PASSWORD="Old-Pass_1234", NEO4J_PASSWORD="Ambient-Pass-456")
    assert res.returncode == 0, res.stderr
    values = _dotenv_values(repo / SECRETS_FILE_NAME)
    assert values["NEO4J_PASSWORD"] == "Old-Pass_1234"
    assert values["NEO4J_AUTH"] == "neo4j/Old-Pass_1234"
    assert re.fullmatch(r"[0-9a-f]{64}", values["WATCHTOWER_HTTP_API_TOKEN"]), "token watchtower luôn sinh mới"
    assert "Old-Pass_1234" not in res.stdout + res.stderr
    assert "Ambient-Pass-456" not in res.stdout + res.stderr


@needs_bash
@pytest.mark.parametrize(
    "bad",
    ["short77", "neo4j", "has/slash1", "has space1", "dollar$sign1", "quote'1234", 'dq"12345', "hash#tag1234",
     "tilde~home1", "semi;colon1", "back`tick1"],
)
def test_init_secrets_rejects_neo4j_passwords_unsafe_for_env_files(tmp_path, bad):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    res = _run_init(repo, KEEP_NEO4J_PASSWORD=bad)
    assert res.returncode != 0
    assert sorted(p.name for p in repo.iterdir()) == ["scripts"], "lỗi thì không được tạo file nào"
    if bad != "neo4j":  # "neo4j" xuất hiện hợp lệ trong thông báo lỗi
        assert bad not in res.stdout + res.stderr


@needs_bash
def test_init_secrets_falls_back_to_urandom_without_openssl(tmp_path):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_openssl = fake_bin / "openssl"
    fake_openssl.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_openssl.chmod(0o755)
    res = _run_init(repo, PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    assert res.returncode == 0, res.stderr
    values = _dotenv_values(repo / SECRETS_FILE_NAME)
    assert re.fullmatch(r"[0-9a-f]{64}", values["NEO4J_PASSWORD"])
    assert re.fullmatch(r"[0-9a-f]{64}", values["WATCHTOWER_HTTP_API_TOKEN"])
    assert values["NEO4J_PASSWORD"] != values["WATCHTOWER_HTTP_API_TOKEN"]


@needs_bash
@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="chỉ kiểm tra được khi chạy bằng root")
def test_init_secrets_refuses_sudo_into_another_users_checkout(tmp_path):
    repo = _make_repo(tmp_path, INIT_SECRETS_SCRIPT)
    os.chown(repo, 65534, 65534)  # repo thuộc user thường (nobody), script chạy bằng root
    res = _run_init(repo)
    assert res.returncode != 0
    assert "sudo" in res.stderr
    assert not (repo / SECRETS_FILE_NAME).exists()


# docker giả: ghi argv / env / stdin của từng lần gọi và mô phỏng Neo4j: xác thực (mật khẩu
# hiện tại của "DB" nằm trong $FAKE_DB_PASSWORD_FILE; ALTER đổi nó sang mật khẩu mới), và quy
# tắc database `system` — chỉ nhận lệnh quản trị (ALTER/SHOW...), còn lệnh quản trị phải chạy
# trên `system` (Cypher thường như `RETURN 1` trên system bị Neo4j từ chối).
# FAKE_DOCKER_DAEMON=down: mọi lệnh lỗi như khi user không có quyền với docker.sock.
# FAKE_NEO4J_RUNNING=missing: daemon ổn nhưng không có container. FAKE_ALTER_NOOP=1: ALTER
# "thành công" mà mật khẩu không đổi (để kiểm tra bước xác minh sau ALTER).
_FAKE_DOCKER = r"""#!/usr/bin/env bash
set -u
log="$FAKE_DOCKER_LOG"
d="$log/$(printf '%03d' "$(find "$log" -mindepth 1 -maxdepth 1 | wc -l)")"
mkdir -p "$d"
printf '%s\n' "$@" > "$d/argv"
if [[ "${FAKE_DOCKER_DAEMON:-up}" == "down" ]]; then
  echo "permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock" >&2
  exit 1
fi
case "${1-}" in
  info)
    exit 0 ;;
  inspect)
    if [[ "${FAKE_NEO4J_RUNNING:-true}" == "missing" ]]; then
      echo "Error: No such object: ${!#}" >&2
      exit 1
    fi
    echo "${FAKE_NEO4J_RUNNING:-true}"
    exit 0 ;;
  exec)
    cat > "$d/stdin"
    printf '%s\n' "${NEO4J_USERNAME-}" "${NEO4J_PASSWORD-}" > "$d/env"
    db="${NEO4J_DATABASE:-neo4j}"
    args=("$@")
    for ((i = 0; i < ${#args[@]} - 1; i++)); do
      if [[ "${args[i]}" == "--database" || "${args[i]}" == "-d" ]]; then db="${args[i+1]}"; fi
    done
    if [[ "${NEO4J_PASSWORD-}" != "$(cat "$FAKE_DB_PASSWORD_FILE")" ]]; then
      echo "The client is unauthorized due to authentication failure." >&2
      exit 1
    fi
    admin=0
    if grep -qE '^[[:space:]]*(ALTER|SHOW|CREATE|DROP|GRANT|DENY|REVOKE)[[:space:]]' "$d/stdin"; then admin=1; fi
    if [[ "$db" == "system" && "$admin" == 0 ]]; then
      echo "Not a recognised system command or procedure. This Cypher command can only be executed in a user database." >&2
      exit 1
    fi
    if [[ "$db" != "system" && "$admin" == 1 ]]; then
      echo "This is an administration command and it should be executed against the system database." >&2
      exit 1
    fi
    if grep -q "ALTER CURRENT USER" "$d/stdin" && [[ "${FAKE_ALTER_NOOP:-0}" != 1 ]]; then
      printf '%s' "$FAKE_NEW_PASSWORD" > "$FAKE_DB_PASSWORD_FILE"
    fi
    exit 0 ;;
esac
exit 2
"""

_NEW_NEO4J_PASSWORD = "0123456789abcdef" * 4


def _rotate_setup(tmp_path: Path, *, db_password: str, auth: str | None = None) -> tuple[Path, dict[str, str], Path, Path]:
    repo = _make_repo(tmp_path, ROTATE_NEO4J_SCRIPT)
    (repo / ".env").write_text("NEO4J_USERNAME=neo4j\n", encoding="utf-8")
    (repo / SECRETS_FILE_NAME).write_text(
        f"NEO4J_PASSWORD={_NEW_NEO4J_PASSWORD}\n"
        f"NEO4J_AUTH={auth if auth is not None else 'neo4j/' + _NEW_NEO4J_PASSWORD}\n"
        f"WATCHTOWER_HTTP_API_TOKEN={'f' * 64}\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text(_FAKE_DOCKER, encoding="utf-8")
    (fake_bin / "docker").chmod(0o755)
    log = tmp_path / "docker-calls"
    log.mkdir()
    db = tmp_path / "db-password"
    db.write_text(db_password, encoding="utf-8")
    env = _script_env(
        PATH=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        FAKE_DOCKER_LOG=str(log), FAKE_DB_PASSWORD_FILE=str(db), FAKE_NEW_PASSWORD=_NEW_NEO4J_PASSWORD,
    )
    return repo, env, log, db


def _docker_calls(log: Path) -> list[dict]:
    calls = []
    for d in sorted(log.iterdir()):
        calls.append({
            "argv": (d / "argv").read_text(encoding="utf-8").splitlines(),
            "stdin": (d / "stdin").read_text(encoding="utf-8") if (d / "stdin").exists() else None,
            "env": (d / "env").read_text(encoding="utf-8").splitlines() if (d / "env").exists() else None,
        })
    return calls


@needs_bash
def test_rotate_neo4j_password_passes_secrets_via_env_and_stdin_only(tmp_path):
    old = "o'ld\\pass-1"  # nháy đơn + backslash: kiểm tra escape string literal Cypher
    repo, env, log, db = _rotate_setup(tmp_path, db_password=old)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": old})
    assert res.returncode == 0, res.stderr
    assert db.read_text(encoding="utf-8") == _NEW_NEO4J_PASSWORD

    calls = _docker_calls(log)
    for call in calls:
        argv = "\n".join(call["argv"])
        assert old not in argv and _NEW_NEO4J_PASSWORD not in argv, f"secret lọt vào argv của docker: {call['argv'][:2]}"
    alters = [c for c in calls if c["stdin"] and "ALTER" in c["stdin"]]
    assert len(alters) == 1
    alter = alters[0]
    assert alter["stdin"] == f"ALTER CURRENT USER SET PASSWORD FROM 'o\\'ld\\\\pass-1' TO '{_NEW_NEO4J_PASSWORD}';\n"
    assert alter["env"] == ["neo4j", old], "đăng nhập bằng mật khẩu CŨ qua biến môi trường"
    argv = alter["argv"]
    assert argv[:2] == ["exec", "-i"] and "cypher-shell" in argv
    assert argv[argv.index("--database") + 1] == "system"
    assert {"NEO4J_USERNAME", "NEO4J_PASSWORD"} <= set(argv), "docker exec -e KEY (không kèm giá trị)"
    probes = [c for c in calls if c["stdin"] == "RETURN 1;\n"]
    assert probes and all("system" not in c["argv"] for c in probes), "RETURN 1 phải chạy trên database mặc định, không phải system"
    output = res.stdout + res.stderr
    assert old not in output and _NEW_NEO4J_PASSWORD not in output

    # Chạy lại: DB đã dùng mật khẩu mới → không ALTER lần nữa.
    again = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": old})
    assert again.returncode == 0, again.stderr
    assert sum(1 for c in _docker_calls(log) if c["stdin"] and "ALTER" in c["stdin"]) == 1


@needs_bash
def test_rotate_neo4j_password_refuses_bad_input_without_touching_the_db(tmp_path):
    old = "Old-Pass_1234"
    # NEO4J_AUTH không khớp NEO4J_PASSWORD → dừng trước khi gọi docker.
    repo, env, log, db = _rotate_setup(tmp_path / "a", db_password=old, auth="neo4j/something-else")
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": old})
    assert res.returncode != 0 and "NEO4J_AUTH" in res.stderr
    assert _docker_calls(log) == []

    # Không có mật khẩu cũ và stdin không phải terminal → dừng, không ALTER.
    repo, env, log, db = _rotate_setup(tmp_path / "b", db_password=old)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, env)
    assert res.returncode != 0 and "NEO4J_OLD_PASSWORD" in res.stderr
    assert db.read_text(encoding="utf-8") == old

    # Sai mật khẩu cũ → cypher-shell lỗi → exit khác 0, DB giữ nguyên, không lộ secret.
    repo, env, log, db = _rotate_setup(tmp_path / "c", db_password=old)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": "Wrong-Pass-999"})
    assert res.returncode != 0
    assert db.read_text(encoding="utf-8") == old
    output = res.stdout + res.stderr
    assert "Wrong-Pass-999" not in output and _NEW_NEO4J_PASSWORD not in output


@needs_bash
def test_rotate_neo4j_password_fails_when_the_new_password_does_not_work_after_alter(tmp_path):
    """ALTER báo thành công nhưng mật khẩu mới vẫn không đăng nhập được → không được báo ✓."""
    old = "Old-Pass_1234"
    repo, env, log, db = _rotate_setup(tmp_path, db_password=old)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": old, "FAKE_ALTER_NOOP": "1"})
    assert res.returncode != 0
    assert "✓" not in res.stdout
    assert "mật khẩu mới" in res.stderr
    assert sum(1 for c in _docker_calls(log) if c["stdin"] and "ALTER" in c["stdin"]) == 1


@needs_bash
@pytest.mark.parametrize(
    ("fake_env", "expected", "not_expected"),
    [
        # Không có quyền với docker.sock: KHÔNG được bảo "container không chạy" (người dùng sẽ
        # `docker compose up -d neo4j` rồi gặp lại đúng lỗi đó).
        ({"FAKE_DOCKER_DAEMON": "down"}, "group docker", "docker compose up -d neo4j"),
        ({"FAKE_NEO4J_RUNNING": "missing"}, "docker compose up -d neo4j", "group docker"),
        ({"FAKE_NEO4J_RUNNING": "false"}, "docker compose up -d neo4j", "group docker"),
    ],
    ids=["daemon-unreachable", "no-container", "stopped"],
)
def test_rotate_neo4j_password_explains_why_docker_is_unusable(tmp_path, fake_env, expected, not_expected):
    old = "Old-Pass_1234"
    repo, env, log, db = _rotate_setup(tmp_path, db_password=old)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": old, **fake_env})
    assert res.returncode != 0
    assert expected in res.stderr and not_expected not in res.stderr, res.stderr
    assert not [c for c in _docker_calls(log) if c["argv"][:1] == ["exec"]], "không được chạy cypher-shell"
    assert db.read_text(encoding="utf-8") == old


@needs_bash
def test_rotate_neo4j_password_warns_when_old_and_new_password_are_equal(tmp_path):
    """Mật khẩu cũ = mật khẩu trong .env.secrets → không ALTER, nhưng phải nói rõ là CHƯA đổi."""
    repo, env, log, db = _rotate_setup(tmp_path, db_password=_NEW_NEO4J_PASSWORD)
    res = _run_script(repo, ROTATE_NEO4J_SCRIPT, {**env, "NEO4J_OLD_PASSWORD": _NEW_NEO4J_PASSWORD})
    assert res.returncode == 0, res.stderr
    assert "KHÔNG được đổi" in res.stderr
    assert not [c for c in _docker_calls(log) if c["argv"][:1] == ["exec"]]
    assert _NEW_NEO4J_PASSWORD not in res.stdout + res.stderr
