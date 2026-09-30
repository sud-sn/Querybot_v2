"""
core/release.py

The release this build is: the version an admin reads, the knowledge-base
format it builds, and the digest an answer's plan is kept under.

The version is VERSION at the repository's root, and CHANGELOG.md says what
each one changed and what an upgrade to it asks of an admin. /health reports
it, so the first thing a deploy checks is that the version it meant to deploy
is the one running.

A knowledge base is built once and read by every answer after, so a release
that changes what a build writes -- the table documents, the semantic model
beside them, the business terms taken from them -- leaves every workspace built
before it answering from the old kind until someone rebuilds. Nothing said so.
A finished build now records KB_FORMAT and the version that built it, and a
workspace a different format built is told to rebuild: in the startup log, on
the dashboard's inbox and on its setup page.

A question's validated SQL is kept, and the next time the question is asked
word for word it may be run again without being planned
(store.find_reusable_validated_sql_plan). The SQL is the work of the planners
and compilers, so it is only as right as the release that wrote it: an upgrade
that stops a member filter being left out -- "online sales in Canada" answered
with every country's sales -- reached no question already asked, whose old SQL
went on being reused for as long as the semantic contract stood still.

So a plan is reused only under the release that made it. A release is the code
that plans an answer, read byte for byte: every Python module of core/ and
store/, and the vocabulary packs. The digest is read once per process.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PLANNING_CODE = (("core", "*.py"), ("store", "*.py"), ("packs", "*.json"))


def release_of(root: Path) -> str:
    """The digest of the code under ``root`` that plans an answer."""
    digest = hashlib.sha256()
    for folder, pattern in _PLANNING_CODE:
        for path in sorted((root / folder).rglob(pattern)):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def code_release() -> str:
    """This process's release."""
    return release_of(_ROOT)


# The format of what a knowledge-base build writes. A release that changes it
# raises this and says so in its CHANGELOG.md entry; a workspace built under
# any other format is then told to rebuild. Builds from before the format was
# recorded read as 0.
KB_FORMAT = 1


@lru_cache(maxsize=1)
def product_version() -> str:
    """This build's version, as VERSION says; "unknown" without the file."""
    try:
        return (_ROOT / "VERSION").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def kb_build_record() -> dict:
    """What a finished knowledge-base build keeps in the workspace's state
    about the release that built it."""
    return {"kb_format": KB_FORMAT, "kb_built_version": product_version()}


def kb_rebuild_needed(client: dict) -> dict | None:
    """Why ``client``'s knowledge base must be rebuilt under this release, or
    None. ``client`` is a client row: its ``state``, and its ``state_data`` as
    stored. Only a built knowledge base needs rebuilding -- a workspace READY,
    with a kb_dir -- and one built by a newer release (a rollback) is told so
    as well: this release does not read what a later one wrote either."""
    if str(client.get("state") or "") != "READY":
        return None
    raw = client.get("state_data")
    try:
        state = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
    except ValueError:
        state = {}
    if not state.get("kb_dir"):
        return None
    try:
        built = int(state.get("kb_format") or 0)
    except (TypeError, ValueError):
        built = 0
    if built == KB_FORMAT:
        return None
    return {
        "built_format": built,
        "format": KB_FORMAT,
        "built_version": str(state.get("kb_built_version") or ""),
        "version": product_version(),
        "newer": built > KB_FORMAT,
    }
