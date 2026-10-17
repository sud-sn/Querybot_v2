#!/usr/bin/env bash
# Installs Solviz on a Linux server in one go: everything it needs, the service, and a health check.
# Ubuntu 22.04 or 24.04, or Debian 12. Run it from the repository, with sudo:
#
#   sudo bash deploy/linux/deploy.sh
#   sudo bash deploy/linux/deploy.sh --port 8001 --admin-email you@company.com --admin-name "Your Name"
#   sudo bash deploy/linux/deploy.sh --domain solviz.company.com --tls-email you@company.com
#
# Every step is safe to run again, so it also repairs a server that is missing a piece:
#   1. System packages: Python 3.11 or 3.12 with venv, git, curl, unixODBC, and Microsoft's ODBC Driver 18 for
#      SQL Server, which Azure SQL and SQL Server connections need (--no-odbc to leave it out).
#   2. The virtual environment ./venv with the exact versions in requirements.lock. PyTorch comes as its CPU build:
#      the default build carries about 3 GB of GPU libraries Solviz never uses.
#   3. The name recogniser's English model (en_core_web_md), and Chromium with the system libraries it needs, which
#      draws scheduled dashboards as pictures and PDF.
#   4. The service (deploy/linux/install-service.sh): settings in /etc/querybot/<name>.env, its secrets made once
#      and never changed; a systemd unit that starts at boot and restarts after a crash; a health check.
#   5. Optional: the first Super admin account (--admin-email; asks for the password), and HTTPS through nginx
#      with a Let's Encrypt certificate (--domain). The domain's DNS must already point at this server.
#
# Options:
#   --port N            the port Solviz listens on (default 8000; with --domain, only on this server)
#   --user NAME         the Linux user the service runs as (default: the user who ran sudo)
#   --name NAME         the service's name (default querybot)
#   --public-url URL    the address people open, used in emailed links (default: from --domain)
#   --domain NAME       serve https://NAME through nginx, with a certificate from Let's Encrypt
#   --tls-email EMAIL   the email Let's Encrypt writes to about the certificate (needed with --domain)
#   --admin-email EMAIL make (or reset) a Super admin account with this email
#   --admin-name NAME   that admin's name
#   --no-odbc           leave out the ODBC driver (no Azure SQL or SQL Server data sources)
#   --skip-system       install no system packages (a distribution this script does not know; install them first)
#
# Updates afterwards are one command, as the service's user:  bash deploy/linux/update.sh
set -euo pipefail

PORT=""
NAME=querybot
RUN_USER="${SUDO_USER:-$(id -un)}"
PUBLIC_URL=""
DOMAIN=""
TLS_EMAIL=""
ADMIN_EMAIL=""
ADMIN_NAME=""
WITH_ODBC=1
WITH_SYSTEM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --user) RUN_USER="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --public-url) PUBLIC_URL="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --tls-email) TLS_EMAIL="$2"; shift 2 ;;
    --admin-email) ADMIN_EMAIL="$2"; shift 2 ;;
    --admin-name) ADMIN_NAME="$2"; shift 2 ;;
    --no-odbc) WITH_ODBC=0; shift ;;
    --skip-system) WITH_SYSTEM=0; shift ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "Run it with sudo: sudo bash $0 $*" >&2; exit 1; }
if [ -n "$DOMAIN" ] && [ -z "$TLS_EMAIL" ]; then
  echo "--domain needs --tls-email: Let's Encrypt writes there before a certificate runs out." >&2; exit 2
fi
id "$RUN_USER" >/dev/null 2>&1 || { echo "No user $RUN_USER on this server (--user)." >&2; exit 1; }
if [ "$RUN_USER" = "root" ]; then
  echo "!! The service would run as root. Better: make a user for it, then --user <name>." >&2
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_GROUP="$(id -gn "$RUN_USER")"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
VENV="$APP_DIR/venv"
PY="$VENV/bin/python"
ENV_FILE="/etc/querybot/$NAME.env"
step() { echo; echo "== $*"; }

# Runs a command as the service's user, with that user's home (pip, browser and model caches live there).
as_user() {
  if [ "$RUN_USER" = "root" ]; then env HOME="$RUN_HOME" "$@"; else runuser -u "$RUN_USER" -- env HOME="$RUN_HOME" "$@"; fi
}

echo "Solviz: $APP_DIR, service \"$NAME\", user $RUN_USER"

# ── 1. System packages ─────────────────────────────────────────────────────────────────────────────────────────
PYBIN=""
pick_python() {   # Python 3.11, the version the test suite runs on, else 3.12; it must be able to make a venv
  PYBIN=""
  for candidate in python3.11 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "import venv, ensurepip" >/dev/null 2>&1; then
      PYBIN="$(command -v "$candidate")"
      return
    fi
  done
}
apt_has() { [ -n "$(apt-cache policy "$1" 2>/dev/null | sed -n 's/^ *Candidate: //p' | grep -v '(none)')" ]; }

if [ "$WITH_SYSTEM" -eq 1 ]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "!! This server has no apt-get. Install Python 3.11 or 3.12 (with venv), git, curl and unixODBC" >&2
    echo "   (and Microsoft's msodbcsql18 for SQL Server sources), then run again with --skip-system." >&2
    exit 1
  fi
  step "1/5 System packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates gnupg unixodbc >/dev/null
  pick_python
  if [ -z "$PYBIN" ]; then
    for version in 3.11 3.12; do
      if apt_has "python$version-venv"; then
        apt-get install -y -qq "python$version" "python$version-venv" >/dev/null
        break
      fi
    done
    pick_python
  fi
  if [ "$WITH_ODBC" -eq 1 ]; then
    bash "$APP_DIR/install_linux_odbc18.sh"
  fi
else
  step "1/5 System packages: skipped (--skip-system)"
  pick_python
fi
[ -n "$PYBIN" ] || { echo "!! No Python 3.11 or 3.12 with venv on this server. Install one, then run this again." >&2; exit 1; }
echo "Python: $PYBIN ($("$PYBIN" --version 2>&1))"

# ── 2. Virtual environment and libraries ───────────────────────────────────────────────────────────────────────
step "2/5 Virtual environment and libraries (the first time takes several minutes)"
# The service writes its database, uploads and logs under the application folder, so its user must own it.
if ! as_user test -w "$APP_DIR"; then
  echo "Giving $RUN_USER the application folder (the service writes its data there)"
  chown -R "$RUN_USER:$RUN_GROUP" "$APP_DIR"
fi
if [ -x "$PY" ]; then
  have="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo broken)"
  case "$have" in
    3.11|3.12) echo "Keeping the virtual environment (Python $have)" ;;
    *) moved="$VENV.old-$(date +%Y%m%d-%H%M%S)"
       echo "The virtual environment is Python $have; the libraries need 3.11 or 3.12. Moving it to $moved"
       mv "$VENV" "$moved" ;;
  esac
fi
if [ ! -x "$PY" ]; then
  as_user "$PYBIN" -m venv "$VENV"
fi
as_user "$PY" -m pip install --quiet --upgrade pip
TORCH="$(sed -nE 's/^torch==([^ ;]+).*/\1/p' "$APP_DIR/requirements.lock" | head -1)"
if [ -n "$TORCH" ] && ! as_user "$PY" -m pip install --quiet --retries 2 "torch==$TORCH" \
     --index-url https://download.pytorch.org/whl/cpu; then
  echo "!! PyTorch's own download site could not be reached: its default build comes from PyPI instead (larger)." >&2
fi
as_user "$PY" -m pip install --quiet -r "$APP_DIR/requirements.lock"
echo "Libraries installed from requirements.lock"

# ── 3. Name recogniser model and PDF browser ───────────────────────────────────────────────────────────────────
step "3/5 Name recogniser model and the PDF browser"
if as_user "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('en_core_web_md') else 1)"; then
  echo "en_core_web_md: already installed"
elif as_user "$PY" -m spacy download en_core_web_md >/dev/null 2>&1; then
  echo "en_core_web_md: installed"
else
  echo "!! The name recogniser's model could not be downloaded: names typed in questions are withheld by rules" >&2
  echo "   only until it is. To retry: sudo -u $RUN_USER $PY -m spacy download en_core_web_md" >&2
fi
if [ "$WITH_SYSTEM" -eq 1 ]; then
  "$PY" -m playwright install-deps chromium >/dev/null 2>&1 || \
    echo "!! Chromium's system libraries could not be installed: $PY -m playwright install-deps chromium" >&2
fi
if as_user "$PY" -m playwright install chromium >/dev/null 2>&1 && as_user "$PY" - <<'PYEOF' >/dev/null 2>&1
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    p.chromium.launch().close()
PYEOF
then
  echo "Chromium: installed and starts"
else
  echo "!! Chromium is not working: scheduled emails will go without pictures and PDF, with every number." >&2
fi

# ── 4. The service ─────────────────────────────────────────────────────────────────────────────────────────────
step "4/5 The service"
SERVICE_ARGS=(--name "$NAME" --user "$RUN_USER")
if [ -n "$PORT" ]; then SERVICE_ARGS+=(--port "$PORT"); fi
if [ -n "$DOMAIN" ]; then
  SERVICE_ARGS+=(--host 127.0.0.1)      # nginx on this server is the only way in
  PUBLIC_URL="${PUBLIC_URL:-https://$DOMAIN}"
fi
if [ -n "$PUBLIC_URL" ]; then SERVICE_ARGS+=(--set "PORTAL_BASE_URL=${PUBLIC_URL%/}"); fi
bash "$APP_DIR/deploy/linux/install-service.sh" "${SERVICE_ARGS[@]}"
UNIT="/etc/systemd/system/$NAME.service"
PORT="$(grep -oE -- '--port[ =]+[0-9]+' "$UNIT" | grep -oE '[0-9]+$' | head -1)"

# ── 5. First admin and HTTPS ───────────────────────────────────────────────────────────────────────────────────
step "5/5 First admin and HTTPS"
if [ -n "$ADMIN_EMAIL" ]; then
  if [ -t 0 ]; then
    # The admin command reads the same database and key as the service: run it as the service's user, from the
    # application folder, with the service's settings in its environment (never on its command line).
    (
      # Each line is NAME=value, exported as it stands.
      # shellcheck disable=SC2163
      while IFS= read -r -d '' setting; do export "$setting"; done < <("$PY" - "$ENV_FILE" "$APP_DIR/deploy/envfile.py" <<'PYEOF'
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("envfile", sys.argv[2])
envfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(envfile)
for name, value in envfile.read(pathlib.Path(sys.argv[1])).items():
    sys.stdout.write(f"{name}={value}\0")
PYEOF
      )
      cd "$APP_DIR"
      echo "Super admin $ADMIN_EMAIL: choose a password (at least 12 characters)"
      as_user "$PY" -m admin.create_admin "$ADMIN_EMAIL" "$ADMIN_NAME"
    ) || echo "!! The admin account was not made. Run this script again with --admin-email to retry." >&2
  else
    echo "!! --admin-email needs a terminal to ask for the password. Run this script again from one." >&2
  fi
else
  echo "No --admin-email: the first person to open /admin sets the admin password. Do that now, or run this"
  echo "script again with --admin-email you@company.com --admin-name \"Your Name\"."
fi

if [ -n "$DOMAIN" ]; then
  apt-get install -y -qq nginx certbot python3-certbot-nginx >/dev/null
  # Chat answers arrive over a WebSocket, Excel workbooks are uploaded, and Learn's pages can wait on a slow
  # database: the proxy passes upgrades, takes 100 MB bodies and waits ten minutes.
  cat > /etc/nginx/conf.d/solviz-websocket.conf <<'EOF'
map $http_upgrade $solviz_connection_upgrade { default upgrade; '' close; }
EOF
  cat > "/etc/nginx/sites-available/$NAME.conf" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    client_max_body_size 100m;
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \$solviz_connection_upgrade;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }
}
EOF
  ln -sf "/etc/nginx/sites-available/$NAME.conf" "/etc/nginx/sites-enabled/$NAME.conf"
  nginx -t >/dev/null 2>&1 || { nginx -t; echo "!! nginx refused the configuration above." >&2; exit 1; }
  systemctl enable --now nginx >/dev/null 2>&1 || true
  systemctl reload nginx
  if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$TLS_EMAIL" --redirect >/dev/null 2>&1; then
    echo "HTTPS: https://$DOMAIN (the certificate renews itself)"
  else
    echo "!! No certificate yet: check that $DOMAIN points at this server and ports 80 and 443 are open, then:" >&2
    echo "     sudo certbot --nginx -d $DOMAIN -m $TLS_EMAIL --agree-tos --redirect" >&2
    echo "   Until then Solviz answers on http://$DOMAIN" >&2
  fi
fi

if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q "Status: active"; then
  if [ -n "$DOMAIN" ]; then ufw allow "Nginx Full" >/dev/null; else ufw allow "$PORT/tcp" >/dev/null; fi
  echo "Firewall (ufw): opened $([ -n "$DOMAIN" ] && echo "80 and 443" || echo "port $PORT")"
fi

ADDRESS="${PUBLIC_URL:-http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT}"
ADDRESS="${ADDRESS%/}"
KEY_FILE="$(sed -nE 's/^QUERYBOT_KEY_FILE=//p' "$ENV_FILE" | tail -1)"
cat <<EOF

Solviz is installed and running.

  Admin console   $ADDRESS/admin
  Chat portal     $ADDRESS/portal/login?workspace=<workspace id>
  Health          curl -s http://localhost:$PORT/health

Next, in the admin console:
  1. Sign in (or set the admin password if this is the first visit).
  2. System > AI provider: choose the provider and paste its key (Azure OpenAI, Anthropic or OpenAI).
  3. Workspaces > New workspace: connect the data source, run discovery, then Learn.
  4. The workspace's People page: add users, their groups, their tables and any row rules.

Keep safe, together:  $ENV_FILE  and  $KEY_FILE
  (without the key file, saved data-source passwords cannot be read)
Service:  sudo systemctl status $NAME   |   sudo journalctl -u $NAME -f
Update:   bash deploy/linux/update.sh   (as $RUN_USER)
EOF
if [ -z "$DOMAIN" ]; then
  echo
  echo "This address is plain HTTP. Before real data goes in, put HTTPS in front: run this script again with"
  echo "--domain and --tls-email, or use your cloud's gateway. Open port $PORT in the cloud firewall if it is closed."
fi
