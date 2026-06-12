#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[1/5] Installing Docker prerequisites..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl

echo "[2/5] Adding Docker's official Ubuntu repository..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# shellcheck disable=SC1091
. /etc/os-release
ARCH="$(dpkg --print-architecture)"
CODENAME="${UBUNTU_CODENAME:-$VERSION_CODENAME}"

echo \
  "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "[3/5] Installing Docker Engine and Compose..."
sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

echo "[4/5] Starting Docker..."
sudo service docker start

echo "[5/5] Starting Qdrant..."
cd "$REPO_ROOT"
sudo docker compose -f deploy/qdrant.compose.yml up -d
sudo docker compose -f deploy/qdrant.compose.yml ps

echo
echo "Qdrant installation complete."
echo "From Windows PowerShell, verify with:"
echo "  Invoke-RestMethod http://127.0.0.1:6333/healthz"
