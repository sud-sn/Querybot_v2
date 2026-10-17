#!/usr/bin/env bash
# Install Microsoft ODBC Driver 18 for SQL Server on Ubuntu/Debian Linux.
# Run this once before installing Python requirements if you use Azure SQL:
#   bash install_linux_odbc18.sh
# deploy/linux/deploy.sh runs it for you.

set -euo pipefail

# sudo only when not root already: a minimal server run as root may have no sudo at all.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

DRIVER_NAME="ODBC Driver 18 for SQL Server"
KEYRING="/usr/share/keyrings/microsoft-prod.gpg"
LIST_FILE="/etc/apt/sources.list.d/mssql-release.list"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This installer is for apt-based Linux systems such as Ubuntu/Debian."
    exit 1
fi

if command -v odbcinst >/dev/null 2>&1 && odbcinst -q -d -n "$DRIVER_NAME" >/dev/null 2>&1; then
    echo "$DRIVER_NAME is already installed."
    exit 0
fi

echo "Installing prerequisites..."
$SUDO apt-get update -qq
$SUDO apt-get install -y curl gnupg2 ca-certificates unixodbc-dev

echo "Adding Microsoft package repository..."
curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | \
    $SUDO gpg --dearmor --yes -o "$KEYRING"

if [ -r /etc/os-release ]; then
    . /etc/os-release
else
    echo "Cannot detect Linux distribution from /etc/os-release."
    exit 1
fi

repo_written=0
if [ "${ID:-}" = "ubuntu" ]; then
    codename="${VERSION_CODENAME:-}"
    version="${VERSION_ID:-}"
    if [ "$version" = "24.04" ] || [ "$codename" = "noble" ]; then
        echo "deb [arch=amd64 signed-by=$KEYRING] https://packages.microsoft.com/ubuntu/24.04/prod noble main" | \
            $SUDO tee "$LIST_FILE" >/dev/null
        repo_written=1
    elif [ "$version" = "22.04" ] || [ "$codename" = "jammy" ]; then
        echo "deb [arch=amd64 signed-by=$KEYRING] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" | \
            $SUDO tee "$LIST_FILE" >/dev/null
        repo_written=1
    elif [ "$version" = "20.04" ] || [ "$codename" = "focal" ]; then
        echo "deb [arch=amd64 signed-by=$KEYRING] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | \
            $SUDO tee "$LIST_FILE" >/dev/null
        repo_written=1
    fi
elif [ "${ID:-}" = "debian" ]; then
    codename="${VERSION_CODENAME:-bookworm}"
    echo "deb [arch=amd64 signed-by=$KEYRING] https://packages.microsoft.com/debian/${VERSION_ID:-12}/prod $codename main" | \
        $SUDO tee "$LIST_FILE" >/dev/null
    repo_written=1
fi

if [ "$repo_written" -ne 1 ]; then
    echo "Unsupported Linux release: ${PRETTY_NAME:-unknown}."
    echo "Install msodbcsql18 manually from Microsoft docs, then rerun:"
    echo "  odbcinst -q -d -n \"$DRIVER_NAME\""
    exit 1
fi

echo "Installing $DRIVER_NAME..."
$SUDO apt-get update -qq
$SUDO env ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

echo "Verifying driver registration..."
odbcinst -q -d -n "$DRIVER_NAME"
echo "$DRIVER_NAME installed successfully."
