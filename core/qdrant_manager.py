"""
core/qdrant_manager.py
──────────────────────
Detect how Qdrant is managed on the current host and expose
status-check + restart without any hard-coded assumptions.

Detection order (first match wins):
  1. Docker   — `docker inspect querybot-qdrant`
  2. systemd  — `systemctl list-units … qdrant*`
  3. Windows  — `sc query qdrant`
  4. process  — `pgrep -x qdrant`  (bare binary, Linux/macOS)
  5. unknown  — status-only, no restart

Entry points
────────────
  get_status()     → {"running": bool, "version": str}
  detect_manager() → "docker" | "systemd" | "windows-service" | "process" | "unknown"
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

    # 4. Standalone process — bare binary launched directly (./qdrant or /path/to/qdrant)
    try:
        r = subprocess.run(
            ["pgrep", "-x", "qdrant"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            log.debug("Qdrant manager: standalone process (pid=%s)", r.stdout.strip().split()[0])
            return "process"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    log.debug("Qdrant manager: unknown")
    return "unknown"


_MANAGER_LABELS = {
    "docker":          "Docker container",
    "systemd":         "systemd service",
    "windows-service": "Windows service",
    "process":         "Standalone process",
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

        elif manager == "process":
            # Standalone binary: find the PID, resolve its working directory,
            # kill it, then re-launch from the same directory.
            pid_r = subprocess.run(["pgrep", "-x", "qdrant"],
                                   capture_output=True, text=True, timeout=5)
            if pid_r.returncode != 0 or not pid_r.stdout.strip():
                return False, "Qdrant process not found — may have already stopped."
            pid = pid_r.stdout.strip().split()[0]
            cwd_r = subprocess.run(["readlink", "-f", f"/proc/{pid}/cwd"],
                                   capture_output=True, text=True, timeout=5)
            work_dir = cwd_r.stdout.strip() if cwd_r.returncode == 0 else ""
            if not work_dir:
                return False, f"Could not resolve working directory for PID {pid}."
            subprocess.run(["kill", pid], timeout=10)
            time.sleep(2)
            import os as _os
            subprocess.Popen(
                ["./qdrant"],
                cwd=work_dir,
                stdout=_os.open(_os.devnull, _os.O_WRONLY),
                stderr=_os.open(_os.devnull, _os.O_WRONLY),
                start_new_session=True,
            )
            return True, f"Qdrant restarted (PID {pid} killed, relaunched from {work_dir})."

        else:
            return False, "Cannot restart: Qdrant manager not detected on this host."

    except subprocess.TimeoutExpired:
        return False, "Restart command timed out (30 s)."
    except FileNotFoundError as exc:
        return False, f"Command not found: {exc}"
    except Exception as exc:
        return False, str(exc)
