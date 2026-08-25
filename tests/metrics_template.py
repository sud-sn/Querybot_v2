"""One way to read the Metric Registry page.

The page was a 4,155-line monolith until 7a split it into
admin/templates/metrics/*.html. Three separate test modules string-scanned the
single file, each with its own inline read, which is exactly what made the
split feel risky enough to postpone for months.

This is that read, once, and IN RENDER ORDER. The order matters: several tests
assert that one marker appears before another -- window._qbDbType has to be set
before the main IIFE runs, for instance -- and those assertions are only
meaningful if the concatenation matches what the browser actually receives.

So the order is taken from the parent's own {% include %} statements rather
than from the filesystem. Sorting the directory alphabetically put _styles
after _scripts and inverted exactly that assertion, which is how this was
found. Following the parent means reordering the page reorders the
concatenation automatically, and a partial nobody includes cannot quietly keep
a deleted marker alive.
"""

from __future__ import annotations

import re
from pathlib import Path

_ADMIN_TEMPLATES = Path(__file__).resolve().parents[1] / "admin" / "templates"
_PARENT = _ADMIN_TEMPLATES / "client_metrics.html"
_INCLUDE = re.compile(r'{%-?\s*include\s+"(metrics/[^"]+)"\s*-?%}')


def metrics_template() -> str:
    """The whole Metric Registry page: the parent with its partials inlined."""
    parent = _PARENT.read_text(encoding="utf-8")

    def _inline(match: re.Match[str]) -> str:
        partial = _ADMIN_TEMPLATES / match.group(1)
        # An include naming a file that does not exist is a broken page, and a
        # silent passthrough here would let the tests keep passing over it.
        assert partial.is_file(), f"{_PARENT.name} includes missing {match.group(1)}"
        return partial.read_text(encoding="utf-8")

    return _INCLUDE.sub(_inline, parent)


def metrics_partials() -> list[Path]:
    """Every partial on disk, for tests that check nothing was orphaned."""
    directory = _ADMIN_TEMPLATES / "metrics"
    return sorted(directory.glob("*.html")) if directory.is_dir() else []
