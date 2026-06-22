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

# Populated by detect_manager() on Windows — holds the real service name
# (may differ from "qdrant", e.g. "Qdrant", "qdrant-service", etc.)
_windows_service_name: str = "qdrant"
# Populated by detect_manager() on Windows — holds the Scheduled Task name
_windows_task_name: str = "Qdrant"


def _find_windows_qdrant_service() -> "str | None":
    """PowerShell wildcard search for any service with 'qdrant' in name or display name."""
    try:
        r = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-Service | Where-Object { $_.Name -like '*qdrant*' -or $_.DisplayName -like '*qdrant*' } "
                "| Select-Object -First 1 -ExpandProperty Name",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            name = r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
            if name:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _find_windows_qdrant_task() -> "str | None":
    """PowerShell search for a Scheduled Task with 'qdrant' in its name."""
    try:
        r = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Get-ScheduledTask | Where-Object { $_.TaskName -like '*qdrant*' } "
                "| Select-Object -First 1 -ExpandProperty TaskName",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            name = r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
            if name:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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

    # 3. Windows Service — PowerShell wildcard so any service name containing
    #    "qdrant" is found (handles "Qdrant", "qdrant-service", etc.)
    if platform.system() == "Windows":
        svc = _find_windows_qdrant_service()
        if svc:
            global _windows_service_name
            _windows_service_name = svc
            log.debug("Qdrant manager: windows-service (name=%s)", svc)
            return "windows-service"

    # 3b. Windows Scheduled Task — common on Windows when not installed as a service
    if platform.system() == "Windows":
        task = _find_windows_qdrant_task()
        if task:
            global _windows_task_name
            _windows_task_name = task
            log.debug("Qdrant manager: windows-task (name=%s)", task)
            return "windows-task"

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
    "windows-task":    "Scheduled Task",
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

        elif manager == "windows-task":
            task = _windows_task_name
            # Stop the task (ignore failure — it may already be stopped)
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Stop-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue"],
                capture_output=True, timeout=15,
            )
            time.sleep(2)
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 f"Start-ScheduledTask -TaskName '{task}'"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                return True, f"Scheduled Task '{task}' restarted."
            return False, (r.stderr.strip() or f"Start-ScheduledTask '{task}' failed")

        elif manager == "windows-service":
            svc = _windows_service_name
            subprocess.run(["sc", "stop", svc],
                           capture_output=True, timeout=10)
            time.sleep(2)
            r = subprocess.run(
                ["sc", "start", svc],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                return True, f"Windows service '{svc}' restarted."
            return False, (r.stderr.strip() or f"sc start {svc} failed")

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
