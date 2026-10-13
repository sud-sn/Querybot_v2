"""The portal's Vue pieces (frontend/src) are built into static/dist, kept in the repository so the server never
needs Node -- and kept with their sources: a source changed without the build fails here.

The build writes the hash of the sources it was built from (frontend/scripts/stamp.mjs) to static/dist/SOURCES;
this computes the same hash, the same way (each file's path, then its text with \\n line ends, sorted by path).

Where Node is on the machine, the bar's own rules run too: the address a view is seen at (read back by
portal/dashboard_view.py), a filter bar that keeps its fields while other filters narrow the tiles, and the
calendar's days and spans, written as the server writes them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
DIST = ROOT / "static" / "dist"


def _sources_hash() -> str:
    files = [p for p in (FRONTEND / "src").rglob("*") if p.is_file()]
    files += [FRONTEND / "package.json", FRONTEND / "package-lock.json", FRONTEND / "vite.config.js"]
    digest = hashlib.sha256()
    for path in sorted(p.relative_to(FRONTEND).as_posix() for p in files):
        digest.update(path.encode() + b"\0")
        digest.update((FRONTEND / path).read_text(encoding="utf-8").replace("\r\n", "\n").encode() + b"\0")
    return digest.hexdigest()


def test_the_built_files_are_those_of_the_sources_kept_beside_them():
    assert (DIST / "SOURCES").read_text().strip() == _sources_hash(), (
        "frontend/src changed without a build: run `npm run build` in frontend/ and commit static/dist")


def test_the_dashboard_loads_its_built_bar_and_nothing_points_at_a_dev_server():
    built = (DIST / "dashboard.js").read_text(encoding="utf-8")
    assert len(built) > 10_000 and "localhost" not in built and "sourceMappingURL" not in built
    template = (ROOT / "portal" / "templates" / "portal_dashboard.html").read_text(encoding="utf-8")
    assert "asset('dist/dashboard.js')" in template and 'type="module"' in template
    assert "frontend/node_modules/" in (ROOT / ".gitignore").read_text()


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node runs the bar's own modules")


def _node(script: str):
    run = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60,
                         cwd=ROOT)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _module(name: str) -> str:
    return (FRONTEND / "src" / "dashboard" / name).as_uri()


@needs_node
def test_the_address_of_a_view_is_read_back_as_it_was_chosen():
    from starlette.datastructures import QueryParams

    from portal.dashboard_view import read_view

    state = {"dashboard_id": 7, "tab": "Sales & margin",
             "picker": {"period": "custom", "start": "2026-02-02", "end": "2026-03-15", "default": "this_month"},
             "filters": [{"field": "store.name", "selected": ["Downtown Store", "Eastview & Co"]},
                         {"field": "region.name", "selected": []}]}
    plain = {**state, "picker": {"period": "this_month", "start": "", "end": "", "default": "this_month"}}
    urls = _node(f"import {{ address }} from '{_module('view.js')}';"
                 f"console.log(JSON.stringify([address({json.dumps(state)}), address({json.dumps(plain)})]))")
    view = read_view(QueryParams(urlsplit(urls[0]).query))
    assert view == {"period": "custom", "start": "2026-02-02", "end": "2026-03-15",
                    "members": {"store.name": ["Downtown Store", "Eastview & Co"]}}
    assert dict(parse_qsl(urlsplit(urls[0]).query))["tab"] == "Sales & margin"
    assert "period" not in urls[1], "this month is what an address without one opens on"


@needs_node
def test_the_bar_keeps_its_fields_while_other_filters_narrow_the_tiles():
    before = {"dashboard_id": 7, "tab": "Overview", "filters": [
        {"field": "store.name", "source": "tile", "values": ["A", "B"], "selected": []},
        {"field": "segment", "source": "tile", "values": ["Retail", "Online"], "selected": []},
        {"field": "region.name", "source": "owner", "values": None, "selected": []}]}
    after = {"dashboard_id": 7, "tab": "Overview", "filters": [
        {"field": "store.name", "source": "tile", "values": ["A"], "selected": ["A"]},
        {"field": "product.name", "source": "tile", "values": ["P"], "selected": []}]}
    kept = _node(f"import {{ steady }} from '{_module('view.js')}';"
                 f"console.log(JSON.stringify(steady({json.dumps(before)}, {json.dumps(after)})))")
    assert [f["field"] for f in kept["filters"]] == ["store.name", "segment", "product.name"], \
        "in their places; the one its owner took off goes"
    assert kept["filters"][0]["selected"] == ["A"]
    other_tab = {**after, "tab": "Margin"}
    assert _node(f"import {{ steady }} from '{_module('view.js')}';"
                 f"console.log(JSON.stringify(steady({json.dumps(before)}, {json.dumps(other_tab)})))") == other_tab


@needs_node
def test_the_calendar_has_its_days_and_writes_spans_as_the_server_does():
    from portal.dashboard_view import span_label

    spans = [("2026-10-01", "2026-10-13"), ("2026-01-01", "2026-03-31"), ("2025-12-01", "2026-01-31"),
             ("2026-05-04", "2026-05-04"), ("2026-02-02", "2026-03-15"), ("2026-06-01", "2026-06-14")]
    script = (f"import * as D from '{_module('dates.js')}';"
              f"const spans = {json.dumps(spans)};"
              "console.log(JSON.stringify({en: spans.map(([a, b]) => D.spanLabel(a, b, 'en')),"
              " fr: spans.map(([a, b]) => D.spanLabel(a, b, 'fr')),"
              " feb: D.weeks('2026-02-01'), oct: D.weeks('2026-10-13').length,"
              " months: [D.addMonths('2026-01-31', 1), D.addMonths('2024-03-31', -1), D.addMonths('2026-12-15', 1)],"
              " days: [D.addDays('2026-02-28', 1), D.addDays('2024-02-28', 1), D.addDays('2026-01-01', -1)]}))")
    out = _node(script)
    for lang in ("en", "fr"):
        assert out[lang] == [span_label(dt.date.fromisoformat(a), dt.date.fromisoformat(b), lang) for a, b in spans]
    assert out["feb"][0] == [None] * 6 + ["2026-02-01"], "Monday first: 1 Feb 2026 was a Sunday"
    assert out["feb"][-1] == [f"2026-02-{d}" for d in range(23, 29)] + [None]
    assert sum(1 for week in out["feb"] for day in week if day) == 28 and out["oct"] == 5
    assert out["months"] == ["2026-02-28", "2024-02-29", "2027-01-15"]
    assert out["days"] == ["2026-03-01", "2024-02-29", "2025-12-31"]
