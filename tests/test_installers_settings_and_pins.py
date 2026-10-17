"""What the one-command installers (deploy/linux/deploy.sh, deploy/windows/deploy.ps1) rely on in the repository's own
files: the Windows example settings come out of deploy/envfile.py with no placeholder left and no database that does
not exist; the Windows lock pins every package the Linux lock shares at the same version; and the torch version the
installers read from requirements.lock with sed is the one the lock pins."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
import envfile  # noqa: E402

SECRETS = ["SESSION_SECRET", "PORTAL_SESSION_SECRET", "ADMIN_SESSION_SECRET", "PII_PSEUDONYM_SECRET"]
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")


def _pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _PIN.match(line)
        if m:
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def test_the_windows_example_settings_leave_no_placeholder_once_the_installer_fills_them(tmp_path):
    # deploy.ps1 and install-querybot.ps1 ask envfile.py for exactly these; a copy of the example made by hand
    # must come out of it just as runnable as a file the installer started.
    env = tmp_path / ".env"
    shutil.copy(ROOT / ".env.windows.example", env)
    envfile.ensure(env, secrets_wanted=SECRETS, defaults={"QUERYBOT_KEY_FILE": "C:/Users/qb/.querybot_key"})
    values = dotenv_values(env)
    assert not [name for name, value in values.items() if value and "CHANGE_ME" in value]
    assert all(len(values[name]) == 64 for name in SECRETS)
    # No PostgreSQL that is not there: left unset, the store is SQLite in data\querybot.db.
    assert "DATABASE_URL" not in values


def test_the_windows_lock_pins_what_the_linux_lock_pins():
    linux, windows = _pins(ROOT / "requirements.lock"), _pins(ROOT / "requirements-windows.lock")
    shared = set(linux) & set(windows)
    assert {name for name in shared if linux[name] != windows[name]} == set()
    # Only the platforms' own packages differ.
    assert set(linux) - set(windows) <= {"uvloop", "cuda-bindings", "cuda-pathfinder"}
    assert set(windows) - set(linux) <= {"colorama", "pywin32", "tzdata"}
    # Everything the application asks for is pinned on Windows too.
    wanted = {re.sub(r"[-_.]+", "-", re.split(r"[\[<>=!~;\s]", line.strip(), maxsplit=1)[0]).lower()
              for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith(("#", "-"))}
    assert wanted - set(windows) == set()


@pytest.mark.skipif(shutil.which("sed") is None, reason="sed is what the Linux installers run")
def test_the_installers_read_the_locked_torch_version():
    out = subprocess.run(["sed", "-nE", r"s/^torch==([^ ;]+).*/\1/p", str(ROOT / "requirements.lock")],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [_pins(ROOT / "requirements.lock")["torch"]]
