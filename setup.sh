#!/usr/bin/env bash
# setup.sh — Cài NVIDIA driver + nvidia-container-toolkit để Docker nhận GPU
# Chạy một lần trước khi `docker compose up` lần đầu
# Usage: sudo bash setup.sh

set -euo pipefail

# ── Kiểm tra quyền root ──────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "Vui lòng chạy với quyền root: sudo bash setup.sh"
  exit 1
fi

# ── Bước 0: Cài NVIDIA driver (nếu chưa có) ─────────────────────────────────
echo "==> [0/5] Kiểm tra NVIDIA driver..."
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "    Driver đã cài sẵn: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  echo "    Chưa có driver — tiến hành cài..."
  apt-get update
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers autoinstall

  echo ""
  echo "⚠ Driver vừa được cài. Cần REBOOT máy trước khi tiếp tục."
  echo "  Sau khi reboot, chạy lại: sudo bash setup.sh"
  echo "  Script sẽ bỏ qua bước cài driver và tiếp tục các bước còn lại."
  exit 0
fi

echo "==> [1/5] Thêm GPG key của nvidia-container-toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "==> [2/5] Thêm apt repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> [3/5] Cài nvidia-container-toolkit..."
apt-get update
apt-get install -y nvidia-container-toolkit

echo "==> [4/5] Cấu hình Docker runtime..."
nvidia-ctk runtime configure --runtime=docker

echo "==> [5/5] Restart Docker daemon..."
systemctl restart docker

echo ""
echo "✓ Setup GPU hoàn tất."
echo ""
# D1b: secret (mật khẩu Neo4j, token watchtower) nằm trong .env.secrets — KHÔNG tạo ở đây:
# script này chạy bằng root nên file sẽ thuộc root, docker compose của user thường không đọc được.
echo "Bước tiếp theo — chạy bằng user THƯỜNG${SUDO_USER:+ (${SUDO_USER})}, KHÔNG dùng sudo, trong thư mục repo:"
echo "  1. scripts/init-secrets.sh        # tạo .env.secrets (quyền 600) nếu chưa có"
echo "  2. docker compose up -d --build"
