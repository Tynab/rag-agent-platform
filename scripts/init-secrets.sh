#!/usr/bin/env bash
# scripts/init-secrets.sh — tạo file secret .env.secrets (D1b) với giá trị ngẫu nhiên mạnh.
#
# Chạy bằng user THƯỜNG (user chạy `docker compose`), KHÔNG dùng sudo:
#
#   scripts/init-secrets.sh
#
# Giữ mật khẩu Neo4j của volume neo4j_data đã có (NEO4J_AUTH chỉ áp dụng ở lần start đầu
# của volume rỗng, nên mật khẩu mới sinh sẽ KHÔNG khớp DB cũ):
#
#   NEO4J_PASSWORD='<mật khẩu Neo4j hiện tại>' scripts/init-secrets.sh
#
# Hoặc để script sinh mật khẩu mới rồi đổi mật khẩu trong DB bằng
# scripts/rotate-neo4j-password.sh (khuyến nghị khi mật khẩu cũ từng nằm trong git).
#
# Đảm bảo:
#   • Idempotent: .env.secrets đã tồn tại thì KHÔNG ghi đè (thoát 0, không đổi gì).
#   • File tạo ra có quyền 600 (umask 077) và thuộc user đang chạy script.
#   • Không bao giờ in giá trị secret ra màn hình hay đưa vào tham số dòng lệnh.
#   • Ghi qua file tạm rồi hard-link sang tên thật → không bao giờ để lại file dở dang,
#     không ghi đè file do tiến trình khác vừa tạo.
#
# Danh sách key phải khớp .env.secrets.example (tests/test_config_wiring.py kiểm tra).

set -euo pipefail
umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_FILE="$ROOT_DIR/.env.secrets"

die() {
  printf 'LỖI: %s\n' "$*" >&2
  exit 1
}

# Chạy bằng sudo trong repo của user thường → file thuộc root, `docker compose` (chạy
# bằng user thường) không đọc được. Root thật (repo thuộc root, vd. VM chỉ có root) thì cho phép.
if [[ "$(id -u)" -eq 0 && ! -O "$ROOT_DIR" ]]; then
  die "đang chạy bằng root/sudo nhưng thư mục $ROOT_DIR thuộc user khác — .env.secrets sẽ thuộc root.
     Chạy lại bằng user thường, KHÔNG có sudo: scripts/init-secrets.sh"
fi

if [[ -e "$SECRETS_FILE" || -L "$SECRETS_FILE" ]]; then
  echo "• $SECRETS_FILE đã tồn tại — giữ nguyên, KHÔNG ghi đè."
  if [[ -n "${NEO4J_PASSWORD:-}" ]]; then
    echo "  (bỏ qua NEO4J_PASSWORD truyền vào vì file đã có; muốn tạo lại thì đổi tên file cũ trước)"
  fi
  # ls thay vì stat/find: cú pháp stat khác nhau giữa GNU (Linux) và BSD (macOS).
  # shellcheck disable=SC2012
  mode="$(ls -ld -- "$SECRETS_FILE" | cut -c5-10)"
  if [[ "$mode" != "------" ]]; then
    echo "  ⚠ Quyền file đang mở cho group/other — nên chạy: chmod 600 .env.secrets" >&2
  fi
  exit 0
fi

# 32 byte ngẫu nhiên → 64 ký tự hex: an toàn cho env_file của Compose (không có $ # ' " /
# khoảng trắng), cho `. ./.env.secrets` của bash và cho NEO4J_AUTH (tách bằng `/`).
gen_secret() {
  local value
  value="$(openssl rand -hex 32 2>/dev/null)" || value=""
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    value="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  fi
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "không sinh được giá trị ngẫu nhiên (cần openssl hoặc /dev/urandom)"
  printf '%s' "$value"
}

# Mật khẩu do người dùng truyền vào phải dùng được ở MỌI nơi đọc file: env_file của Compose
# (interpolate `$`, cắt comment ` #`), `. ./.env.secrets` của bash (metachar shell, `~`),
# và regex NEO4J_AUTH của image neo4j (cấm `/`, tối thiểu 8 ký tự, khác "neo4j").
validate_neo4j_password() {
  local pw="$1"
  if [[ "${#pw}" -lt 8 || "$pw" == "neo4j" || ! "$pw" =~ ^[A-Za-z0-9._@%+=:,^-]+$ ]]; then
    die "NEO4J_PASSWORD truyền vào không dùng được trong .env.secrets (cần ≥ 8 ký tự, khác \"neo4j\",
     chỉ gồm chữ, số và . _ @ % + = : , ^ -). Hãy chạy KHÔNG truyền NEO4J_PASSWORD để sinh
     mật khẩu mới, rồi đổi mật khẩu trong DB bằng scripts/rotate-neo4j-password.sh."
  fi
}

if [[ -n "${NEO4J_PASSWORD:-}" ]]; then
  validate_neo4j_password "$NEO4J_PASSWORD"
  neo4j_password="$NEO4J_PASSWORD"
  neo4j_source="giữ mật khẩu truyền qua biến NEO4J_PASSWORD"
else
  neo4j_password="$(gen_secret)"
  neo4j_source="SINH MỚI"
fi
watchtower_token="$(gen_secret)"

tmp_file="$(mktemp "$SECRETS_FILE.tmp.XXXXXX")"
trap 'rm -f -- "$tmp_file"' EXIT

# printf là builtin của bash → giá trị không xuất hiện trong danh sách tiến trình.
{
  printf '%s\n' \
    "# .env.secrets — SECRET của rag-agent-platform. KHÔNG commit (đã có trong .gitignore)." \
    "# Tạo bởi scripts/init-secrets.sh lúc $(date '+%Y-%m-%d %H:%M:%S %z'). Mẫu: .env.secrets.example." \
    "# docker-compose.yml nạp file này qua env_file cho neo4j, rag-api và watchtower." \
    "" \
    "# Mật khẩu Neo4j mà rag-api dùng để đăng nhập (kèm NEO4J_USERNAME trong .env)."
  printf 'NEO4J_PASSWORD=%s\n' "$neo4j_password"
  printf '%s\n' \
    "# Mật khẩu khởi tạo của container neo4j — phải khớp NEO4J_PASSWORD. Chỉ áp dụng ở lần" \
    "# start ĐẦU của volume neo4j_data rỗng; volume đã có: scripts/rotate-neo4j-password.sh."
  printf 'NEO4J_AUTH=neo4j/%s\n' "$neo4j_password"
  printf '%s\n' "# Token Bearer cho HTTP API của watchtower (127.0.0.1:8080/v1/*)."
  printf 'WATCHTOWER_HTTP_API_TOKEN=%s\n' "$watchtower_token"
} > "$tmp_file"
chmod 600 "$tmp_file"

# `ln` thất bại nếu đích đã tồn tại → không bao giờ ghi đè. Filesystem không hỗ trợ hard
# link thì rơi về redirect với noclobber (O_EXCL), cũng không ghi đè.
if ! ln -- "$tmp_file" "$SECRETS_FILE" 2>/dev/null; then
  if [[ -e "$SECRETS_FILE" || -L "$SECRETS_FILE" ]]; then
    die "$SECRETS_FILE vừa được tạo bởi tiến trình khác — giữ nguyên, không ghi đè."
  fi
  (set -o noclobber; cat -- "$tmp_file" > "$SECRETS_FILE") || die "không tạo được $SECRETS_FILE"
  chmod 600 "$SECRETS_FILE"
fi

cat <<EOF
✓ Đã tạo $SECRETS_FILE (quyền 600) với NEO4J_PASSWORD, NEO4J_AUTH, WATCHTOWER_HTTP_API_TOKEN.
  Mật khẩu Neo4j: $neo4j_source. Giá trị secret không được in ra màn hình.

Bước tiếp theo (README mục 6):
  • Cài mới (volume neo4j_data chưa có dữ liệu):  docker compose up -d
  • Volume neo4j_data ĐÃ CÓ dữ liệu và mật khẩu Neo4j vừa SINH MỚI: DB vẫn dùng mật khẩu cũ
    (NEO4J_AUTH chỉ áp dụng cho volume rỗng). Đổi mật khẩu trong DB TRƯỚC khi up:
        scripts/rotate-neo4j-password.sh     # hỏi mật khẩu Neo4j CŨ (hoặc NEO4J_OLD_PASSWORD=...)
        docker compose up -d
  • Gọi HTTP API watchtower: nạp token bằng  set -a; . ./.env.secrets; set +a
EOF
