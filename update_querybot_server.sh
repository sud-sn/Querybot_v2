#!/usr/bin/env bash
# Update the QueryBot test server to the latest core-v2 (5764b0a):
#   - dashboards look like the approved design: Edit, Share (keep it to yourself or share it with your team)
#     and a "..." menu with Follow (daily / weekly / monthly), Chat with dashboard, Tidy layout and About.
#   - a filter bar: Period (this/last month, quarter, year, year to date, last 12 months) and one filter per
#     field a chart is grouped by (Region, Category, Store...). The filters stay in the link you share.
#   - one number per KPI card ($543.1K, the full value on hover, red/green change "vs" the period before);
#     a tile of several numbers is split into a card each the first time its dashboard is opened.
#   - finer rows: the first time a dashboard is opened it is converted once by itself. Nothing to do.
#   - no tile is named by its question any more. New columns are added on start.
# Earlier (762e76d):
#   - dashboards: a tile shows its name and period only (no question); Add to dashboard has a "Name on the
#     dashboard" box (the chart's own name until you change it); Edit renames a tile (the pencil by its name).
# Earlier (818b612):
#   - maps: a measure by state, province or country is a shaded map; by ZIP code, a dot per ZIP on the states.
#     Nothing to set up: the outlines ship with the app (static/geo) and need no Learn.
#   - Settings (next to Log out, or /portal/settings): each reader picks a colour palette, straight or
#     smooth lines, values on line charts, and animation on or off. The new column is added on start.
#   - two groupings are stacked bars or a grid; two measures over many members are a scatter.
# Earlier (29ecd7a):
#   - charts and tiles: a tile is titled by its answer ("Net amount by store"; old tiles named only by the
#     measure take it when they refresh), number groupings in number order (or in ranges), the 10 largest and
#     the rest, a KPI with its change against the period before and a trend line, several numbers as one tile.
# Earlier (bcdff56):
#   - dashboards: tiles can be moved again (the drag handle works), a new dashboard lines up by itself
#     (headline numbers across the top, charts two to a row), a narrow window no longer breaks the saved
#     layout, the KPI number fits its tile, and Edit has a "Tidy layout" button to line up a messy dashboard.
# Earlier (75137c6): every field a question can name reaches the AI (numbers each member has, codes and
#   identifiers, free text); transit time from the ship and delivery dates. Needs a Learn run on each workspace.
# Earlier (514ea45): pointing phrases ("the number one customer", "the biggest mover"), quoted names found
#   whole, plain-word answers.
# Earlier (49285ad): a name narrows an answer only when it is written in quotes; chips quote their names.
# Earlier in core-v2 (9bcda51): links, query history, people's data per reader, attestations, People page.
#
# Run it on the server as chatbotadmin:   bash update_querybot_server.sh
# Nothing to install. If this server is older than 75137c6, RUN LEARN AGAIN on each workspace afterwards:
# the new fields come from Learn (your decisions and approvals are kept). Dashboards need nothing.
#
# From 9bcda51 on, query history needs a read grant, once, for the sign-in QueryBot uses (without it Learn skips this step
# and says so in its log; nothing else changes):
#   Snowflake:  GRANT IMPORTED PRIVILEGES ON DATABASE SNOWFLAKE TO ROLE <querybot role>;
#               (without it, Learn reads the database's own INFORMATION_SCHEMA history, the last 7 days)
#   Azure SQL:  GRANT VIEW DATABASE STATE TO [<querybot user>];   (Query Store must be on: it is by default)
#   Oracle:     GRANT SELECT ON V_$SQL TO <querybot user>;

set -euo pipefail

APP_DIR="$HOME/Querybot_v2"
TARGET="5764b0a"
SERVICE="querybot"
HEALTH_URL="http://localhost:8000/health"

cd "$APP_DIR"
PREVIOUS="$(git rev-parse --short HEAD)"
echo "== Now at:   $(git log --oneline -1)"

# Edits made on the server itself are kept by the checkout below unless they clash with the update,
# in which case git stops and changes nothing. List them so nothing is a surprise.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "== Files changed on this server (kept unless they clash with the update):"
  git status --short --untracked-files=no
fi

# A copy of the app database before updating (SQLite installs only; a Postgres one is untouched).
if [ -f data/querybot.db ]; then
  BACKUP="data/querybot.db.bak-$(date +%Y%m%d-%H%M%S)"
  cp -p data/querybot.db "$BACKUP"
  echo "== Database copied to $BACKUP"
fi

echo "== Fetching core-v2"
git fetch origin core-v2
if ! git cat-file -e "${TARGET}^{commit}" 2>/dev/null; then
  echo "!! $TARGET is not on core-v2 yet. Nothing was changed." >&2
  exit 1
fi
git checkout -B core2-test "$TARGET"
echo "== Moved to: $(git log --oneline -1)"

echo "== Restarting $SERVICE"
sudo systemctl restart "$SERVICE"
sleep 5
sudo systemctl status "$SERVICE" --no-pager | head -5

echo "== Health"
if curl -fsS "$HEALTH_URL"; then echo; else echo "!! The health check did not answer: see the log lines below." >&2; fi

echo "== Errors since the restart (none is good)"
sudo journalctl -u "$SERVICE" --since "2 min ago" --no-pager | grep -iE "error|traceback" | tail -20 || true

echo
echo "Done. If this server was older than 75137c6, run Learn again on each workspace (the new fields come from Learn)."
echo "A dashboard that already looks out of line: open it, press Edit, then Tidy layout."
echo "Then, in a workspace under compliance, review the people's data Learn proposed on its Compliance page."
echo "To go back to where this server was:"
echo "  cd $APP_DIR && git checkout -B core2-test $PREVIOUS && sudo systemctl restart $SERVICE"
