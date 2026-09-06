# -*- coding: utf-8 -*-
"""The two modelling surfaces that had endpoints and no way to reach them.

core/model_readiness.py and core/metric_coverage.py both shipped built and
tested, and both had only a JSON endpoint — so each answered a question no
admin could ask. That is the same failure the domains work had: code that is
correct, covered, and unreachable.

The readiness page is the one that says what to DO. Four quality scores already
existed and none of them answered that; this one orders remedies by how many
failing question shapes each would resolve, measured from the coverage reports
rather than weighted by guess.

The coverage panel is per metric, because a metric that compiles is not a
metric that works: the same definition is asked about as "top 10 by it", as
"how has it changed this quarter", and by a name nobody wrote down.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import admin.routes as routes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    return request


class _RealStore(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-surfaces-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-srf-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.client = {"account_id": self.account_id, "client_name": "Test Corp",
                       "state": "READY", "state_data": "{}"}

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _context(self):
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: (t, c)):
            return asyncio.run(routes.readiness_page(_request(), self.account_id))


class TestTheReadinessPageIsReachable(_RealStore):

    def test_the_route_exists_and_renders_its_template(self):
        template, context = self._context()
        self.assertEqual(template, "client_readiness.html")
        self.assertIn("report", context)

    def test_it_reports_the_same_thing_the_endpoint_does(self):
        """Page and API must not drift: both read build_report."""
        from core.model_readiness import build_report

        _template, context = self._context()
        direct = build_report(self.account_id)
        self.assertEqual(context["report"].metadata_version, direct.metadata_version)
        self.assertEqual(len(context["report"].items), len(direct.items))

    def test_every_backlog_kind_has_somewhere_to_go(self):
        """A row that says what to fix and not where is a dead end."""
        from core.model_readiness import _KIND_WEIGHT

        _template, context = self._context()
        for kind in _KIND_WEIGHT:
            self.assertIn(kind, context["destinations"], kind)
        for href in context["destinations"].values():
            self.assertTrue(href.startswith(f"/admin/clients/{self.account_id}/"))

    def test_a_failing_report_is_said_rather_than_shown_as_empty(self):
        # An empty backlog reads as "nothing to do", which is the opposite of
        # what a crashed report means.
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes.store, "get_client", return_value=self.client), \
                patch("core.model_readiness.build_report",
                      side_effect=RuntimeError("store down")), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: (t, c)):
            _template, context = asyncio.run(
                routes.readiness_page(_request(), self.account_id))
        self.assertIsNone(context["report"])
        self.assertIn("store down", context["error"])

    def test_an_unauthenticated_visitor_is_sent_to_login(self):
        with patch.object(routes, "_is_auth", return_value=False):
            response = asyncio.run(routes.readiness_page(_request(), self.account_id))
        self.assertIn("/admin/login", response.headers["location"])

    def test_the_nav_links_to_it_and_lights_up_on_it(self):
        nav = (ROOT / "admin" / "templates"
               / "_client_workspace_nav.html").read_text(encoding="utf-8")
        self.assertIn("{{ _b }}/readiness", nav)
        self.assertIn("'readiness'", nav)

    def test_the_route_the_nav_points_at_exists(self):
        paths = {getattr(r, "path", "") for r in routes.router.routes}
        self.assertIn("/admin/clients/{account_id}/readiness", paths)


class TestTheCoveragePanelIsReachable(unittest.TestCase):
    """Per metric, so it lives on the metrics page rather than its own."""

    REGISTRY = (ROOT / "admin" / "templates" / "metrics" / "_registry.html")
    SCRIPTS = (ROOT / "admin" / "templates" / "metrics" / "_scripts.html")
    STYLES = (ROOT / "admin" / "templates" / "metrics" / "_styles.html")

    def test_every_metric_row_offers_it(self):
        markup = self.REGISTRY.read_text(encoding="utf-8")
        self.assertIn("showMetricCoverage({{ m.id }}, this)", markup)
        self.assertIn('id="coverage-{{ m.id }}"', markup)

    def test_the_handler_calls_the_endpoint_that_exists(self):
        script = self.SCRIPTS.read_text(encoding="utf-8")
        self.assertIn('"/metrics/" + metricId + "/coverage"', script)
        paths = {getattr(r, "path", "") for r in routes.router.routes}
        self.assertIn("/admin/api/clients/{account_id}/metrics/{metric_id}/coverage",
                      paths)

    def test_it_takes_the_account_id_from_the_path_like_its_neighbours(self):
        # An undefined global here would throw on click and show nothing.
        script = self.SCRIPTS.read_text(encoding="utf-8")
        block = script[script.index("window.showMetricCoverage"):]
        block = block[:block.index("function renderCoverage")]
        self.assertIn("location.pathname.match", block)
        self.assertNotIn("ACCOUNT_ID", block)

    def test_a_failed_lookup_says_so_rather_than_showing_an_empty_panel(self):
        # An empty coverage panel reads as "no gaps", which is the opposite of
        # what a failed request means.
        script = self.SCRIPTS.read_text(encoding="utf-8")
        block = script[script.index("window.showMetricCoverage"):]
        self.assertIn(".catch(", block)
        self.assertIn("could not be checked", block)

    def test_the_styles_live_in_the_styles_partial(self):
        self.assertIn("mr-cov-", self.STYLES.read_text(encoding="utf-8"))
        self.assertNotIn("<style>", self.REGISTRY.read_text(encoding="utf-8"))


class TestTheCoverageRendererRunsInABrowser(unittest.TestCase):
    """The panel's own JavaScript, executed rather than read."""

    def setUp(self):
        import pytest

        self.dukpy = pytest.importorskip("dukpy")

    def _render(self, payload):
        import json

        from tests.js_lift import function as lift

        script = (ROOT / "admin" / "templates" / "metrics"
                  / "_scripts.html").read_text(encoding="utf-8")
        harness = f"""
function escHtml2(s){{ return (s||"").replace(/&/g,"&amp;")
  .replace(/</g,"&lt;").replace(/>/g,"&gt;"); }}
{lift(script, "function renderCoverage(data)")}
renderCoverage({json.dumps(payload)});
"""
        return self.dukpy.evaljs(harness)

    def test_a_complete_metric_says_so(self):
        html = self._render({"total": 8, "resolvable": 8, "complete": True,
                             "summary": "8 of 8", "gaps": []})
        self.assertIn("8 / 8", html)
        self.assertIn("Every generated question shape", html)

    def test_gaps_are_grouped_by_the_asset_that_would_close_them(self):
        # The fix is per asset, not per question: one synonym can close six
        # rows at once, and a flat list hides that.
        html = self._render({
            "total": 6, "resolvable": 2, "complete": False, "summary": "2 of 6",
            "gaps": [
                {"question": "top 10 by revenue", "missing": "a dimension"},
                {"question": "revenue by region", "missing": "a dimension"},
                {"question": "turnover this quarter", "missing": "a synonym"},
            ],
        })
        self.assertIn("Add: a dimension", html)
        self.assertIn("closes 2 questions", html)
        self.assertIn("Add: a synonym", html)
        self.assertIn("closes 1 question", html)
        self.assertNotIn("closes 1 questions", html)

    def test_a_long_gap_list_says_how_many_it_did_not_show(self):
        gaps = [{"question": f"q{i}", "missing": "a dimension"} for i in range(9)]
        html = self._render({"total": 9, "resolvable": 0, "complete": False,
                             "summary": "0 of 9", "gaps": gaps})
        self.assertIn("and 3 more", html)

    def test_a_question_cannot_inject_markup(self):
        html = self._render({
            "total": 1, "resolvable": 0, "complete": False, "summary": "x",
            "gaps": [{"question": "<img src=x onerror=alert(1)>",
                      "missing": "<b>bold</b>"}],
        })
        self.assertNotIn("<img", html)
        self.assertNotIn("<b>bold</b>", html)
        self.assertIn("&lt;img", html)


if __name__ == "__main__":
    unittest.main()
