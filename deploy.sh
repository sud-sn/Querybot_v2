#!/usr/bin/env bash
# Solviz on Linux: the full install, or an update. Kept at the top of the repository for servers and notes that
# still call it; the scripts themselves are in deploy/linux/.
#
#   sudo bash deploy.sh [options]   # install, or repair, everything: sudo bash deploy/linux/deploy.sh --help
#   bash deploy.sh --update         # update to the latest release: deploy/linux/update.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--update" ]; then
  shift
  exec bash "$HERE/deploy/linux/update.sh" "$@"
fi
exec bash "$HERE/deploy/linux/deploy.sh" "$@"
