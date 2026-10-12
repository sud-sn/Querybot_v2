"""The settings file a server runs QueryBot with (.env on Windows, /etc/querybot/<name>.env on Linux): what the
installers add to it, never what they change in it.

    python deploy/envfile.py FILE --secret SESSION_SECRET --secret ADMIN_SESSION_SECRET \
        --default QUERYBOT_KEY_FILE=/home/qb/.querybot_key

A --secret is set to 64 random hex characters, and a --default to its value, only when the file has no value for
it yet (missing, empty, or the CHANGE_ME of the example files). A value already there is kept as it is, whatever it
is: a secret that changes signs everyone out, and the pseudonym secret that changes turns every alias shown before
into a different one. Lines, comments and their order are kept; the file is written as UTF-8 with no byte-order
mark (python-dotenv reads a BOM as part of the first name) and, where the system has them, readable by its owner
only. It prints the names it added, never a value.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

PLACEHOLDERS = {"", "CHANGE_ME"}
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value.split(" #", 1)[0].strip()


def read(path: Path) -> dict[str, str]:
    """The file's names and values, the last one winning, as python-dotenv reads them."""
    found: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            m = _LINE.match(line)
            if m:
                found[m.group(1)] = _value(m.group(2))
    return found


def _quoted(value: str) -> str:
    """A value as a line can hold it: bare when it can be, else in double quotes."""
    if value == "" or re.fullmatch(r"[^\s'\"#\\]+", value):
        return value
    if "\\" in value and '"' not in value and "'" not in value:
        return f"'{value}'"        # a Windows path: single quotes keep its backslashes as they are
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ensure(path: Path, *, secrets_wanted: list[str] = (), defaults: dict[str, str] | None = None) -> list[str]:
    """Add the secrets and defaults the file has no value for; the names added, in order."""
    defaults = dict(defaults or {})
    have = read(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    wanted = [(name, None) for name in secrets_wanted] + list(defaults.items())
    added: list[str] = []
    for name, value in wanted:
        if have.get(name, "") not in PLACEHOLDERS or name in added:
            continue
        new = secrets.token_hex(32) if value is None else value
        line = f"{name}={_quoted(new)}"
        placeholder = [i for i, text in enumerate(lines)
                       if (m := _LINE.match(text)) and m.group(1) == name]
        if placeholder:
            lines[placeholder[-1]] = line            # CHANGE_ME or an empty value: filled where it stands
        else:
            lines.append(line)
        added.append(name)
    if added or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(lines) + ("\n" if lines else "")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(body.encode("utf-8"))
        if os.name == "posix":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("--secret", action="append", default=[], metavar="NAME",
                        help="set to 64 random hex characters when it has no value")
    parser.add_argument("--default", action="append", default=[], metavar="NAME=VALUE",
                        help="set to VALUE when it has no value")
    args = parser.parse_args(argv)
    defaults: dict[str, str] = {}
    for item in args.default:
        name, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            parser.error(f"--default needs NAME=VALUE, got {item!r}")
        defaults[name] = value
    added = ensure(args.file, secrets_wanted=args.secret, defaults=defaults)
    print(f"{args.file}: " + (f"added {', '.join(added)}" if added else "nothing to add, every setting has a value"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
