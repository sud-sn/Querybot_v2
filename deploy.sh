#!/usr/bin/env bash
# deploy.sh — QueryBot production deployment script
# Usage:
#   First deploy : bash deploy.sh
#   Update only  : bash deploy.sh --update
set -euo pipefail

APP_DIR="/home/azureuser/querybot"
SERVICE="querybot"
VENV="$APP_DIR/venv"
PY="python3.11"

echo "============================================"
echo "  QueryBot Deploy"
echo "============================================"

# ── System packages ───────────────────────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    git curl unzip gnupg2

# ── Microsoft ODBC Driver 18 (required for Azure SQL) ─────────────────────────
echo "[2/7] Installing Microsoft ODBC Driver 18 for SQL Server..."
if ! odbcinst -q -d -n "ODBC Driver 18 for SQL Server" &>/dev/null; then
    # Remove any broken Microsoft repo entries
    sudo rm -f /etc/apt/sources.list.d/mssql-release.list

    # Add Microsoft signing key (Ubuntu 24.04 / Noble method)
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
        sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

    # Try Ubuntu 24.04 repo first, fall back to 22.04 if package not found
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
https://packages.microsoft.com/ubuntu/24.04/prod noble main" | \
        sudo tee /etc/apt/sources.list.d/mssql-release.list

    sudo apt-get update -qq 2>/dev/null || true

    if ! sudo ACCEPT_EULA=Y apt-get install -y -qq msodbcsql18 unixodbc-dev 2>/dev/null; then
        echo "  24.04 repo not available — trying 22.04 repo..."
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
https://packages.microsoft.com/ubuntu/22.04/prod jammy main" | \
            sudo tee /etc/apt/sources.list.d/mssql-release.list
        sudo apt-get update -qq
        sudo ACCEPT_EULA=Y apt-get install -y -qq msodbcsql18 unixodbc-dev
    fi

    echo "  ✓ ODBC Driver 18 installed"
else
    echo "  ✓ ODBC Driver 18 already installed"
fi

# ── Python virtual environment ────────────────────────────────────────────────
echo "[3/7] Setting up Python virtual environment..."
if [ ! -d "$VENV" ]; then
    $PY -m venv "$VENV"
    echo "  Created new venv at $VENV"
fi
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
echo "  Dependencies installed"

# ── Create data directory ─────────────────────────────────────────────────────
echo "[4/7] Creating data directory..."
mkdir -p "$APP_DIR/data"
mkdir -p "$APP_DIR/clients"
chmod 700 "$APP_DIR/data"

# ── Encryption key ────────────────────────────────────────────────────────────
echo "[5/7] Checking encryption key..."
KEY_FILE="$HOME/.querybot_key"
if [ ! -f "$KEY_FILE" ]; then
    echo "  Encryption key will be generated on first startup at $KEY_FILE"
    echo "  ⚠️  IMPORTANT: Back up $KEY_FILE after first startup."
    echo "     Losing this key means losing access to all stored credentials."
else
    echo "  Encryption key found at $KEY_FILE"
fi

# ── Systemd service ───────────────────────────────────────────────────────────
echo "[6/7] Installing systemd service..."
sudo cp "$APP_DIR/querybot.service" /etc/systemd/system/$SERVICE.service
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE
echo "  Service enabled: $SERVICE"

# ── Start service ─────────────────────────────────────────────────────────────
echo "[7/7] Starting service..."
sudo systemctl restart $SERVICE
sleep 3

STATUS=$(sudo systemctl is-active $SERVICE 2>/dev/null || echo "unknown")
if [ "$STATUS" = "active" ]; then
    echo "  ✓ Service running"
else
    echo "  ✗ Service failed to start. Check logs:"
    echo "    sudo journalctl -u $SERVICE -n 50"
    exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
VM_IP=$(curl -s --max-time 3 ifconfig.me 2>/dev/null || echo "YOUR-VM-IP")

echo ""
echo "============================================"
echo "  Deploy complete!"
echo "============================================"
echo ""
echo "  Admin UI   : http://$VM_IP:8000/admin"
echo "  Health     : http://$VM_IP:8000/health"
echo ""
echo "  Webhooks:"
echo "    Zoom     : http://$VM_IP:8000/webhook/zoom"
echo "    Teams    : http://$VM_IP:8000/webhook/teams"
echo "    Slack    : http://$VM_IP:8000/webhook/slack"
echo ""
echo "  View logs  : sudo journalctl -u $SERVICE -f"
echo "  Stop       : sudo systemctl stop $SERVICE"
echo "  Restart    : sudo systemctl restart $SERVICE"
echo ""
echo "  Next steps:"
echo "  1. Open http://$VM_IP:8000/admin in your browser"
echo "  2. Complete first-time setup (set password + API keys)"
echo "  3. Add a platform (Zoom/Teams/Slack) under Platforms"
echo "  4. Add a database connection under Databases"
echo "  5. Update your bot's webhook URL in the platform portal"
echo ""
