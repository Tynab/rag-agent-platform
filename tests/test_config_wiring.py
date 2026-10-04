"""Contract test cấu hình (C1) — code ↔ docker-compose.yml ↔ .env. Không cần Docker.

Compose KHÔNG dùng `env_file:`; mỗi service chỉ nhận đúng các key khai báo trong
khối `environment:`. Một biến code có đọc nhưng compose quên truyền sẽ âm thầm rơi
về default trong code — dù `.env` có đặt giá trị (đã xảy ra với WORKFLOW_MAX_SECONDS,
INGEST_EMBED_RETRIES, RAG_TIMEOUT của rag-api). Test này chặn lỗi đó:

1. Mọi biến môi trường mà code của service đọc (`os.environ.get`, `os.environ[...]`,
   `os.getenv`, `_require_env`, helper kiểu `_env_int`, `"X" in os.environ`, kể cả khi
   tên biến đi qua hằng module-level) phải có trong `environment:` của service đó.
2. Mọi `${VAR}` interpolate trong compose mà KHÔNG có default (`:-`/`-`) phải được
   định nghĩa trong `.env` (nếu không compose chỉ cảnh báo và thay bằng chuỗi rỗng).
3. Một số bất biến vận hành: cổng host không publish ra mọi interface, service
   stateful không bị Watchtower auto-update, `service_healthy` chỉ trỏ tới service
   có healthcheck.

PyYAML có sẵn trong CI qua dependency bắc cầu (langchain-core, uvicorn[standard]).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = ROOT / "docker-compose.yml"
COMPOSE_CPU_FILE = ROOT / "docker-compose.cpu.yml"
ENV_FILE = ROOT / ".env"

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


def _env_keys(service: dict) -> set[str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {str(item).split("=", 1)[0] for item in env}


def _labels(service: dict) -> dict[str, str]:
    labels = service.get("labels") or {}
    if isinstance(labels, dict):
        return {k: str(v) for k, v in labels.items()}
    return {k: v for k, _, v in (str(item).partition("=") for item in labels)}


def _dotenv_keys(path: Path) -> set[str]:
    """Key được ĐỊNH NGHĨA (không tính dòng comment) trong file .env."""
    keys: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        key = key.removeprefix("export ").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            keys.add(key)
    return keys


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
    wired = _env_keys(compose["services"][service])
    allow = NOT_WIRED_ALLOWLIST.get(service, {})

    missing = {name: files for name, files in reads.items() if name not in wired and name not in allow}
    assert not missing, (
        f"{service}: code đọc biến môi trường nhưng docker-compose.yml không truyền vào "
        f"`environment:` (thêm `NAME: ${{NAME:-<default>}}` và ghi chú trong .env/README): "
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
