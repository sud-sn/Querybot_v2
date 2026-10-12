#!/usr/bin/env bash
# Brings a QueryBot server to the latest commit of a branch, installs what that commit needs and restarts the
# service. Run it as the user the service runs as (it asks for sudo only to restart):
#
#   bash deploy/linux/update.sh                     # branch core-v2, service "querybot"
#   bash deploy/linux/update.sh --branch main --name querybot
#
# The local branch is set to the branch on GitHub every time, so a server never stays on an older copy of it
# by mistake. Files changed on the server are kept unless the update changes them too, in which case git stops
# and nothing changes. The SQLite database is copied first (data/querybot.db.bak-<time>). The line to go back
# to where the server was is printed at the end.
set -euo pipefail

BRANCH=core-v2
NAME=querybot
while [ $# -gt 0 ]; do
  case "$1" in
    --branch) BRANCH="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$APP_DIR"
PY="$APP_DIR/venv/bin/python"
UNIT="/etc/systemd/system/$NAME.service"
PORT="$(grep -oE -- '--port[ =]+[0-9]+' "$UNIT" 2>/dev/null | grep -oE '[0-9]+$' | head -1 || true)"
PORT="${PORT:-8000}"
PREVIOUS_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
PREVIOUS="$(git rev-parse --short HEAD)"
echo "== Now at: $PREVIOUS_BRANCH $(git log --oneline -1)"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "== Files changed on this server (kept unless the update changes them too):"
  git status --short --untracked-files=no
fi
BACKUP="$("$PY" deploy/backup_db.py)"
if [ -n "$BACKUP" ]; then echo "== Database copied to $BACKUP"; fi

echo "== Fetching $BRANCH"
git fetch origin "$BRANCH"
git checkout -B "$BRANCH" "origin/$BRANCH"
echo "== Moved to: $(git log --oneline -1)"

echo "== Installing what this commit needs"
if [ -f requirements.lock ]; then "$PY" -m pip install --quiet -r requirements.lock; else "$PY" -m pip install --quiet -r requirements.txt; fi
# The browser that draws scheduled dashboards as pictures and PDF. Without it emails still go, with every number.
if "$PY" -m playwright install chromium >/dev/null 2>&1; then
  if ! "$PY" - <<'PYEOF' >/dev/null 2>&1
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    p.chromium.launch().close()
PYEOF
  then
    echo "!! Chromium is installed but will not start: it needs system libraries, once:" >&2
    echo "     sudo $PY -m playwright install-deps chromium" >&2
  fi
else
  echo "!! Chromium could not be downloaded; scheduled emails will go without pictures and PDF." >&2
fi

echo "== Restarting $NAME"
sudo systemctl restart "$NAME"
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
if curl -fsS "http://localhost:$PORT/health"; then echo; else echo "!! The health check did not answer." >&2; fi
echo "== Errors since the restart (none is good)"
sudo journalctl -u "$NAME" --since "3 min ago" --no-pager | grep -iE "error|traceback" | tail -20 || true

echo
echo "Done. If the release notes say so, run Learn again on each workspace (your decisions are kept)."
echo "To go back to where this server was:"
echo "  cd $APP_DIR && git checkout -B $PREVIOUS_BRANCH $PREVIOUS && sudo systemctl restart $NAME"
