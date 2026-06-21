"""
core/qdrant_manager.py
──────────────────────
Detect how Qdrant is managed on the current host and expose
status-check + restart without any hard-coded assumptions.

Detection order (first match wins):
  1. Docker   — `docker inspect querybot-qdrant`
  2. systemd  — `systemctl list-units … qdrant*`
  3. Windows  — `sc query qdrant`
  4. unknown  — status-only, no restart

Entry points
────────────
  get_status()     → {"running": bool, "version": str}
  detect_manager() → "docker" | "systemd" | "windows-service" | "unknown"
  manager_label(m) → human-readable string
  restart(m)       → (success: bool, message: str)
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
import urllib.request

log = logging.getLogger("querybot.qdrant")

QDRANT_URL      = "http://localhost:6333"
CONTAINER_NAME  = "querybot-qdrant"


# ── Status ────────────────────────────────────────────────────────────────────

def get_status() -> dict:
    """Ping Qdrant HTTP API. Never raises — returns running=False on any error."""
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/", timeout=2) as r:
            data = json.loads(r.read())
            return {"running": True, "version": data.get("version", "")}
    except Exception:
        return {"running": False, "version": ""}


# ── Manager detection ─────────────────────────────────────────────────────────

def detect_manager() -> str:
    """
    Probe the host to find out how Qdrant is managed.
    Returns one of: 'docker', 'systemd', 'windows-service', 'unknown'.
    Safe to call repeatedly — each probe has a short timeout.
    """

    # 1. Docker — named container may be running or stopped
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            log.debug("Qdrant manager: docker (container=%s)", CONTAINER_NAME)
            return "docker"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. systemd (Linux / WSL)
    try:
        r = subprocess.run(
            ["systemctl", "list-units", "--type=service", "--all",
             "--no-pager", "--no-legend"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "qdrant" in r.stdout.lower():
            log.debug("Qdrant manager: systemd")
            return "systemd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Windows Service
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["sc", "query", "qdrant"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                log.debug("Qdrant manager: windows-service")
                return "windows-service"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    log.debug("Qdrant manager: unknown")
    return "unknown"


_MANAGER_LABELS = {
    "docker":          "Docker container",
    "systemd":         "systemd service",
    "windows-service": "Windows service",
    "unknown":         "Not detected",
}


def manager_label(manager: str) -> str:
    return _MANAGER_LABELS.get(manager, "Not detected")


# ── Restart ───────────────────────────────────────────────────────────────────

def restart(manager: str) -> tuple[bool, str]:
    """
    Restart Qdrant using the appropriate manager command.
    Returns (success, human-readable message).
    """
    try:
        if manager == "docker":
            r = subprocess.run(
                ["docker", "restart", CONTAINER_NAME],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                return True, f"Docker container '{CONTAINER_NAME}' restarted."
            return False, (r.stderr.strip() or r.stdout.strip() or "docker restart failed")

        elif manager == "systemd":
            r = subprocess.run(
                ["systemctl", "restart", "qdrant"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                return True, "systemd service 'qdrant' restarted."
            return False, (r.stderr.strip() or "systemctl restart failed")

        elif manager == "windows-service":
            subprocess.run(["sc", "stop", "qdrant"],
                           capture_output=True, timeout=10)
            time.sleep(2)
            r = subprocess.run(
                ["sc", "start", "qdrant"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                return True, "Windows service 'qdrant' restarted."
            return False, (r.stderr.strip() or "sc start failed")

        else:
            return False, "Cannot restart: Qdrant manager not detected on this host."

    except subprocess.TimeoutExpired:
        return False, "Restart command timed out (30 s)."
    except FileNotFoundError as exc:
        return False, f"Command not found: {exc}"
    except Exception as exc:
        return False, str(exc)
