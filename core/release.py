"""
core/release.py

The release an answer's plan was made by.

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
