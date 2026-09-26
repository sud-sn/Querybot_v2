"""
core/static_assets.py
─────────────────────
The URL of a file under static/, versioned by what the file holds.

Templates link every stylesheet, script and image as ``{{ asset('css/base.css') }}``,
which renders ``/static/css/base.css?v=<first 12 hex of its SHA-256>``. The
version used to be written by hand -- ``?v=20260926-ui-8`` in six places per
shell -- and bumped by whoever remembered: base.css once shipped with no version
at all, so returning browsers kept a stale copy through every deploy, and a
fixed rule looked unfixed. A version taken from the bytes changes exactly when
the file does, and never when it does not.

Fonts are the exception, and are linked bare: fonts.css names them in url(),
which cannot call this, and a preload is only used when its URL matches the one
the stylesheet fetches. A changed font ships under a new file name.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("querybot.static_assets")

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@lru_cache(maxsize=512)
def _digest(file: str, mtime_ns: int, size: int) -> str:
    # mtime and size are in the key so an edited file is hashed again; the
    # hash itself, not the timestamp, is what reaches the URL.
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()[:12]


def asset_url(path: str) -> str:
    """``/static/<path>?v=<content hash>`` for a file under static/."""
    rel = path.lstrip("/")
    if rel.startswith("static/"):
        rel = rel[len("static/"):]
    file = STATIC_DIR / rel
    try:
        stat = file.stat()
    except OSError:
        # A page must still render; the missing file is the defect, and it is
        # said here rather than as a 404 nobody reads.
        log.warning("static asset %s does not exist under %s; linked unversioned", rel, STATIC_DIR)
        return f"/static/{rel}"
    return f"/static/{rel}?v={_digest(str(file), stat.st_mtime_ns, stat.st_size)}"
