#!/usr/bin/env bash
# scripts/rotate-neo4j-password.sh — đổi mật khẩu Neo4j trên volume neo4j_data ĐÃ CÓ dữ liệu.
#
# Image neo4j chỉ áp dụng NEO4J_AUTH ở lần start ĐẦU của volume rỗng; với volume đã có dữ
# liệu, sửa .env.secrets KHÔNG đổi mật khẩu trong DB. Script này chạy
#
#   ALTER CURRENT USER SET PASSWORD FROM <cũ> TO <mới>
#
# bằng cypher-shell bên trong container neo4j ĐANG CHẠY (lệnh quản trị → database `system`;
# kiểm tra đăng nhập bằng `RETURN 1` → database mặc định).
#
#   Mật khẩu MỚI: NEO4J_PASSWORD trong .env.secrets (tạo bằng scripts/init-secrets.sh).
#   Mật khẩu CŨ : biến môi trường NEO4J_OLD_PASSWORD, hoặc nhập (ẩn) khi script hỏi.
#   User        : NEO4J_USERNAME trong .env (mặc định neo4j).
#   Container   : NEO4J_CONTAINER (mặc định neo4j — container_name trong docker-compose.yml).
#
# Dùng (user thường, thuộc group docker):
#
#   scripts/rotate-neo4j-password.sh
#   docker compose up -d          # rag-api (và neo4j) nhận mật khẩu mới từ .env.secrets
#
# Secret không nằm trên dòng lệnh của tiến trình nào: mật khẩu đăng nhập đi qua biến môi
# trường của `docker exec` (cypher-shell đọc NEO4J_USERNAME / NEO4J_PASSWORD), câu lệnh
# ALTER đi qua stdin. Script không in giá trị secret.

set -euo pipefail
umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_FILE="$ROOT_DIR/.env.secrets"
ENV_FILE="$ROOT_DIR/.env"
CONTAINER="${NEO4J_CONTAINER:-neo4j}"

die() {
  printf 'LỖI: %s\n' "$*" >&2
  exit 1
}

# Giá trị của dòng KEY=... CUỐI CÙNG trong file env (bỏ `export `, khoảng trắng hai đầu và
# một cặp nháy bao ngoài). Chỉ đọc, không `source` file → không thực thi gì trong file.
env_value() {
  local file="$1" key="$2" line value=""
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=(.*)$ ]]; then
      value="${BASH_REMATCH[2]}"
    fi
  done < "$file"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ "${#value}" -ge 2 && ( ("${value:0:1}" == "'" && "${value: -1}" == "'") || ("${value:0:1}" == '"' && "${value: -1}" == '"') ) ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

# Escape cho string literal Cypher trong nháy đơn.
cypher_quote() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\'/\\\'}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  printf "'%s'" "$s"
}

# stdin = Cypher; $1 = mật khẩu đăng nhập (truyền qua env, không qua argv); $2 = database
# (tùy chọn; bỏ trống = database mặc định của server). Lệnh quản trị (ALTER CURRENT USER...)
# phải chạy trên `system`, còn Cypher thường (RETURN 1) thì `system` từ chối.
run_cypher() {
  local args=(--format plain)
  if [[ -n "${2:-}" ]]; then
    args+=(--database "$2")
  fi
  NEO4J_USERNAME="$neo4j_user" NEO4J_PASSWORD="$1" \
    docker exec -i -e NEO4J_USERNAME -e NEO4J_PASSWORD "$CONTAINER" \
    cypher-shell "${args[@]}"
}

[[ -f "$SECRETS_FILE" ]] || die "chưa có $SECRETS_FILE — chạy scripts/init-secrets.sh trước."

new_password="$(env_value "$SECRETS_FILE" NEO4J_PASSWORD)"
neo4j_auth="$(env_value "$SECRETS_FILE" NEO4J_AUTH)"
[[ -n "$new_password" ]] || die "NEO4J_PASSWORD trong .env.secrets đang rỗng."
if [[ "${#new_password}" -lt 8 || "$new_password" == "neo4j" || ! "$new_password" =~ ^[A-Za-z0-9._@%+=:,^-]+$ ]]; then
  die "NEO4J_PASSWORD trong .env.secrets không hợp lệ (cần ≥ 8 ký tự, khác \"neo4j\", chỉ gồm chữ, số và . _ @ % + = : , ^ -)."
fi
if [[ "$neo4j_auth" != "neo4j/$new_password" ]]; then
  die "NEO4J_AUTH trong .env.secrets phải là neo4j/<NEO4J_PASSWORD> (hiện không khớp). Sửa lại cho khớp rồi chạy lại."
fi

neo4j_user="$(env_value "$ENV_FILE" NEO4J_USERNAME)"
neo4j_user="${neo4j_user:-neo4j}"

old_password="${NEO4J_OLD_PASSWORD:-}"
if [[ -z "$old_password" ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p "Mật khẩu Neo4j HIỆN TẠI (cũ) của user $neo4j_user: " old_password
    echo >&2
  fi
  [[ -n "$old_password" ]] || die "thiếu mật khẩu cũ — đặt NEO4J_OLD_PASSWORD hoặc chạy trong terminal để nhập."
fi

command -v docker >/dev/null 2>&1 || die "không tìm thấy lệnh docker."
if ! running="$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2>/dev/null)"; then
  # inspect lỗi vì không gọi được daemon (thiếu quyền / daemon tắt) khác hẳn với "chưa có container".
  if ! docker info >/dev/null 2>&1; then
    die "không gọi được Docker daemon (daemon chưa chạy, hoặc user $(id -un) chưa thuộc group docker).
     Thêm user vào group docker rồi đăng nhập lại, hoặc chạy: sudo scripts/rotate-neo4j-password.sh
     và NHẬP mật khẩu cũ khi được hỏi (sudo bỏ biến môi trường; đừng đưa mật khẩu vào tham số lệnh)."
  fi
  die "không có container $CONTAINER — khởi động trước: docker compose up -d neo4j"
fi
if [[ "$running" != "true" ]]; then
  die "container $CONTAINER không chạy — khởi động trước: docker compose up -d neo4j"
fi

if [[ "$old_password" == "$new_password" ]]; then
  {
    echo "⚠ Mật khẩu cũ trùng NEO4J_PASSWORD trong .env.secrets — mật khẩu Neo4j KHÔNG được đổi."
    echo "  Muốn đổi sang mật khẩu mới: mv .env.secrets .env.secrets.old && scripts/init-secrets.sh"
    echo "  rồi chạy lại script này với mật khẩu cũ (NEO4J_OLD_PASSWORD hoặc nhập khi được hỏi)."
  } >&2
  exit 0
fi

# Đã đổi từ lần chạy trước? (Đăng nhập được bằng mật khẩu mới thì dừng, không ALTER lại.)
if printf 'RETURN 1;\n' | run_cypher "$new_password" >/dev/null 2>&1; then
  echo "• DB đã dùng NEO4J_PASSWORD của .env.secrets — không cần đổi. Tiếp theo: docker compose up -d"
  exit 0
fi

echo "==> Đổi mật khẩu Neo4j của user $neo4j_user trong container $CONTAINER..."
if ! printf 'ALTER CURRENT USER SET PASSWORD FROM %s TO %s;\n' \
    "$(cypher_quote "$old_password")" "$(cypher_quote "$new_password")" \
    | run_cypher "$old_password" system >/dev/null; then
  die "cypher-shell báo lỗi (xem thông báo ở trên): sai mật khẩu cũ, user đang bị khóa tạm do đăng nhập
     sai nhiều lần (chờ vài giây rồi chạy lại), hoặc Neo4j chưa sẵn sàng (docker compose ps neo4j)."
fi

if ! printf 'RETURN 1;\n' | run_cypher "$new_password" >/dev/null; then
  die "đã chạy ALTER nhưng không đăng nhập được bằng mật khẩu mới — kiểm tra docker compose logs neo4j."
fi

cat <<'EOF'
✓ Đã đổi mật khẩu Neo4j sang NEO4J_PASSWORD trong .env.secrets (giá trị không được in ra).

Bước tiếp theo:
    docker compose up -d            # recreate rag-api/neo4j với .env.secrets mới
    docker compose restart rag-api  # chỉ cần nếu rag-api đã được recreate với mật khẩu mới từ trước
    curl -s http://localhost:8090/graph/status | jq .connected   # → true
EOF
