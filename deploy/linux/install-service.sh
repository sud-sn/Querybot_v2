#!/usr/bin/env bash
# Runs QueryBot as a systemd service: it starts at boot, restarts itself after a crash, and logs to the journal.
# Run it once on the server, from the repository, with sudo:
#
#   sudo bash deploy/linux/install-service.sh                       # service "querybot", port 8000, the sudo user
#   sudo bash deploy/linux/install-service.sh --port 8001 --user qbot --name querybot
#
# Safe to run again. Its settings live outside the repository, in /etc/querybot/<name>.env (root only), and are
# only ever added to, never changed:
#   - the four session and pseudonym secrets, once, random: sign-ins then survive a restart and pseudonyms stay
#     the same (left unset, each start made new ones). Never change PII_PSEUDONYM_SECRET once it is set;
#   - QUERYBOT_KEY_FILE, the key that encrypts stored passwords (the run user's ~/.querybot_key unless set);
#   - any Environment= line of the unit it replaces (PORTAL_BASE_URL, for instance).
# The unit it replaces is kept beside it as <name>.service.bak-<time>. Back up the env file and the key file
# together: without the key, stored database passwords cannot be read.
#
# Afterwards:  sudo systemctl status querybot   |   sudo journalctl -u querybot -f   |   sudo systemctl restart querybot
set -euo pipefail

NAME=querybot
PORT=""
RUN_USER="${SUDO_USER:-$(id -un)}"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --user) RUN_USER="$2"; shift 2 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "Run it with sudo: sudo bash $0 $*" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$APP_DIR/venv/bin/python"
[ -x "$PY" ] || { echo "No virtual environment at $APP_DIR/venv. Create it first:" >&2
                  echo "  python3 -m venv venv && venv/bin/pip install -r requirements.lock" >&2; exit 1; }
id "$RUN_USER" >/dev/null 2>&1 || { echo "No user $RUN_USER on this server (--user)." >&2; exit 1; }
RUN_GROUP="$(id -gn "$RUN_USER")"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
UNIT="/etc/systemd/system/$NAME.service"
ENV_FILE="/etc/querybot/$NAME.env"

# The port asked for; else the one the current unit runs on; else 8000.
if [ -z "$PORT" ] && [ -f "$UNIT" ]; then
  PORT="$(grep -oE -- '--port[ =]+[0-9]+' "$UNIT" | grep -oE '[0-9]+$' | head -1 || true)"
fi
PORT="${PORT:-8000}"

echo "== QueryBot service \"$NAME\": $APP_DIR, user $RUN_USER, port $PORT"
install -d -m 700 -o root -g root /etc/querybot

# Settings an earlier unit set inline move into the env file (kept as they were).
carried=()
if [ -f "$UNIT" ]; then
  while IFS= read -r line; do
    line="${line#Environment=}"; line="${line#\"}"; line="${line%\"}"
    case "$line" in PYTHONUNBUFFERED=*|"") ;; *=*) carried+=(--default "$line") ;; esac
  done < <(grep -E '^Environment=' "$UNIT" || true)
fi
"$PY" "$APP_DIR/deploy/envfile.py" "$ENV_FILE" \
  --secret SESSION_SECRET --secret PORTAL_SESSION_SECRET --secret ADMIN_SESSION_SECRET --secret PII_PSEUDONYM_SECRET \
  --default "QUERYBOT_KEY_FILE=$RUN_HOME/.querybot_key" ${carried[@]+"${carried[@]}"}
chown root:root "$ENV_FILE"
chmod 600 "$ENV_FILE"

if [ -f "$UNIT" ]; then
  cp -p "$UNIT" "$UNIT.bak-$(date +%Y%m%d-%H%M%S)"
fi
cat > "$UNIT" <<EOF
[Unit]
Description=QueryBot ($NAME)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
# One worker: the new core keeps each conversation's recent turns in this process.
ExecStart=$PY -m uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info
Restart=on-failure
RestartSec=5
# A Learn in progress is given a minute to stop; a later Learn reuses the tables it had read.
TimeoutStopSec=60
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$NAME
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$NAME" >/dev/null
systemctl restart "$NAME"

echo "== Waiting for http://localhost:$PORT/health"
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "== Running: $(curl -fsS "http://localhost:$PORT/health")"
    echo "   Logs: sudo journalctl -u $NAME -f    Restart: sudo systemctl restart $NAME"
    echo "   Back up together: $ENV_FILE and $(sed -nE 's/^QUERYBOT_KEY_FILE=//p' "$ENV_FILE" | tail -1)"
    exit 0
  fi
  sleep 2
done
echo "!! It did not answer within two minutes. The last log lines:" >&2
journalctl -u "$NAME" -n 40 --no-pager >&2 || true
exit 1
