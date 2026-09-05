# -*- coding: utf-8 -*-
"""Metric variation coverage — core/metric_coverage.py.

A metric that compiles is not a metric that works. The one someone defines in
chat on Tuesday is asked about on Wednesday as "top 10 customers by it" and on
Thursday by a name nobody wrote down, and nothing checked either.

The properties asserted hardest are the two that decide whether the report is
worth reading:

  * the score must not improve as the metric gets worse — a metric declaring
    no dimensions is still measured against the shapes that need one; and
  * a gap must name the asset to add, because a coverage report is a to-do
    list or it is nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.i18n import MESSAGES  # noqa: E402
from core.metric_coverage import (  # noqa: E402
    GRAINS,
    MAX_DIMENSIONS,
    CoverageReport,
    answerable_grains,
    check_variation,
    coverage_report,
    declared_dimensions,
    declared_examples,
    question_finds_metric,
    searchable_names,
    variations_for,
)

COMPLETE = {
    "name": "Net Revenue",
    "synonyms": "revenue, net sales",
    "allowed_dimensions": "Region, Customer",
    "grain": "day",
}
BARE = {"name": "Net Revenue", "allowed_dimensions": "", "grain": ""}
DATE_ROLE = [{"id": 1, "context_name": "invoice"}]


def _kinds(report):
    return {v.kind for v in report.variations}


# ══════════════════════════════════════════════════════════════════════════════
# Reading what the metric declares
# ══════════════════════════════════════════════════════════════════════════════

class TestDeclarations(unittest.TestCase):

    def test_dimensions_are_split_the_way_an_admin_types_them(self):
        # allowed_dimensions is unstructured free text, not JSON.
        for raw in ("Region, Customer", "Region;Customer", "Region|Customer",
                    "Region\nCustomer", "Region ,  Customer "):
            self.assertEqual(
                declared_dimensions({"allowed_dimensions": raw}),
                ["Region", "Customer"], raw)

    def test_no_dimensions_declared_reads_as_none(self):
        self.assertEqual(declared_dimensions({}), [])
        self.assertEqual(declared_dimensions({"allowed_dimensions": "  "}), [])

    def test_example_questions_split_on_lines_not_commas(self):
        # "revenue, net of returns, by region" is one question, not three.
        examples = declared_examples({
            "example_questions": "revenue, net of returns, by region\nwhat did we bill",
        })
        self.assertEqual(examples,
                         ["revenue, net of returns, by region", "what did we bill"])

    def test_a_daily_metric_rolls_up_but_a_monthly_one_does_not_split_down(self):
        self.assertEqual(answerable_grains({"grain": "day"}), GRAINS[::-1])
        monthly = answerable_grains({"grain": "month"})
        self.assertIn("year", monthly)
        self.assertNotIn("day", monthly)
        self.assertEqual(answerable_grains({"grain": "year"}), ("year",))

    def test_a_metric_with_no_grain_declares_none(self):
        self.assertEqual(answerable_grains({}), ())

    def test_every_synonym_can_find_the_metric(self):
        names = searchable_names(COMPLETE)
        self.assertIn("Net Revenue", names)
        self.assertIn("revenue", names)
        self.assertIn("net sales", names)


# ══════════════════════════════════════════════════════════════════════════════
# The question shapes
# ══════════════════════════════════════════════════════════════════════════════

class TestVariations(unittest.TestCase):

    def test_the_shapes_a_metric_has_to_survive_are_generated(self):
        report = coverage_report(COMPLETE, date_roles=DATE_ROLE)
        self.assertGreaterEqual(report.total, 20)
        self.assertEqual(
            _kinds(report),
            {"base", "grain", "dimension", "ranking", "share", "comparison", "trend"})

    def test_the_questions_read_like_something_a_user_would_type(self):
        questions = [v.question for v in variations_for(COMPLETE)]
        self.assertIn("top 10 Region by Net Revenue", questions)
        self.assertIn("Net Revenue by Customer", questions)
        self.assertIn("what is Net Revenue", questions)

    def test_a_metric_with_more_dimensions_is_checked_on_more_shapes(self):
        one = coverage_report({**COMPLETE, "allowed_dimensions": "Region"},
                              date_roles=DATE_ROLE)
        two = coverage_report(COMPLETE, date_roles=DATE_ROLE)
        self.assertGreater(two.total, one.total)

    def test_the_dimension_count_is_bounded(self):
        many = {**COMPLETE,
                "allowed_dimensions": ", ".join(f"D{i}" for i in range(20))}
        dimensions = {v.dimension for v in variations_for(many) if v.dimension}
        self.assertLessEqual(len(dimensions), MAX_DIMENSIONS)

    def test_a_metric_with_no_name_has_nothing_to_check(self):
        self.assertEqual(variations_for({"name": ""}), [])

    def test_a_metrics_own_examples_are_checked_too(self):
        report = coverage_report(
            {**COMPLETE, "example_questions": "how much did we bill last month"},
            date_roles=DATE_ROLE)
        self.assertIn("example", _kinds(report))
        self.assertIn("how much did we bill last month",
                      [v.question for v in report.variations])


# ══════════════════════════════════════════════════════════════════════════════
# The score must not reward doing nothing
# ══════════════════════════════════════════════════════════════════════════════

class TestTheScoreIsHonest(unittest.TestCase):

    def test_a_metric_with_no_dimensions_is_still_measured_on_them(self):
        # The regression this guards: skipping the shapes a bare metric
        # cannot answer shrinks its denominator, so the least-curated metric
        # scores best. "2 of 7" for a bare metric beside "25 of 25" for a
        # complete one is a report that rewards doing nothing.
        report = coverage_report({**BARE, "grain": "month"}, date_roles=DATE_ROLE)
        self.assertIn("dimension", _kinds(report))
        self.assertIn("ranking", _kinds(report))
        self.assertGreater(len(report.gaps), 0)

    def test_a_more_complete_metric_scores_a_higher_proportion(self):
        def ratio(metric, date_roles):
            report = coverage_report(metric, date_roles=date_roles)
            return report.resolvable / max(report.total, 1)

        bare = ratio(BARE, [])
        partial = ratio({**BARE, "grain": "month"}, DATE_ROLE)
        complete = ratio(COMPLETE, DATE_ROLE)
        self.assertLess(bare, partial)
        self.assertLess(partial, complete)
        self.assertEqual(complete, 1.0)

    def test_a_complete_metric_reports_no_gaps(self):
        report = coverage_report(COMPLETE, date_roles=DATE_ROLE)
        self.assertTrue(report.complete)
        self.assertEqual(report.gaps, ())
        self.assertEqual(report.resolvable, report.total)

    def test_resolvable_and_gaps_always_add_up(self):
        for metric, roles in ((COMPLETE, DATE_ROLE), (BARE, []),
                              ({**BARE, "grain": "month"}, DATE_ROLE)):
            report = coverage_report(metric, date_roles=roles)
            self.assertEqual(report.resolvable + len(report.gaps), report.total)


# ══════════════════════════════════════════════════════════════════════════════
# Gaps name what to add
# ══════════════════════════════════════════════════════════════════════════════

class TestGaps(unittest.TestCase):

    def test_a_metric_with_no_date_role_is_told_to_add_one(self):
        report = coverage_report({**COMPLETE, "grain": "month"}, date_roles=[])
        reasons = report.gaps_by_reason()
        self.assertIn("coverage.gap.no_date_role", reasons)
        gap = next(g for g in report.gaps
                   if g.reason_id == "coverage.gap.no_date_role")
        self.assertEqual(gap.missing, "date_role")
        self.assertIn("date role", gap.reason.lower())

    def test_a_metric_with_no_grain_is_told_to_set_one(self):
        report = coverage_report({**COMPLETE, "grain": ""}, date_roles=DATE_ROLE)
        self.assertIn("coverage.gap.no_grain", report.gaps_by_reason())

    def test_the_most_fundamental_missing_thing_is_reported_first(self):
        # A metric missing both a date role and a grain should be told about
        # the date role: fixing one gap must not immediately reveal a second
        # on the same question.
        report = coverage_report({**COMPLETE, "grain": ""}, date_roles=[])
        reasons = report.gaps_by_reason()
        self.assertIn("coverage.gap.no_date_role", reasons)
        self.assertNotIn("coverage.gap.no_grain", reasons)

    def test_an_unapproved_dimension_names_itself(self):
        from core.metric_coverage import Variation
        gap = check_variation(
            Variation(kind="dimension", message_id="", question="x",
                      dimension="Salesperson"),
            COMPLETE, date_roles=DATE_ROLE)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.missing, "Salesperson")
        self.assertIn("Salesperson", gap.reason)

    def test_an_example_question_that_cannot_find_the_metric_is_a_gap(self):
        # The commonest real failure: a human wrote the example from how the
        # business talks, and the metric is called something else.
        report = coverage_report(
            {**COMPLETE, "example_questions": "what did we bill in Q2"},
            date_roles=DATE_ROLE)
        self.assertIn("coverage.gap.name_not_found", report.gaps_by_reason())
        gap = next(g for g in report.gaps
                   if g.reason_id == "coverage.gap.name_not_found")
        self.assertEqual(gap.missing, "synonym")

    def test_an_example_that_uses_a_synonym_is_not_a_gap(self):
        report = coverage_report(
            {**COMPLETE, "example_questions": "how much net sales last month"},
            date_roles=DATE_ROLE)
        self.assertNotIn("coverage.gap.name_not_found", report.gaps_by_reason())

    def test_the_name_check_is_not_applied_to_generated_questions(self):
        # Every generated shape interpolates the metric's name today, so
        # checking them would verify that a template contains what the
        # template put there. The guard matters for the shape that does not:
        # a future "how has it changed" phrasing carries a pronoun, and
        # flagging it as a missing synonym would be wrong.
        from core.metric_coverage import Variation

        pronoun = Variation(kind="trend", message_id="coverage.q.trend",
                            question="how has it changed over the last year",
                            grain="month")
        self.assertIsNone(
            check_variation(pronoun, COMPLETE, date_roles=DATE_ROLE))

        # The same phrasing as an example -- a human's words -- IS flagged,
        # because there the metric genuinely cannot be found by name.
        as_example = Variation(kind="example", message_id="",
                               question="how has it changed over the last year")
        gap = check_variation(as_example, COMPLETE, date_roles=DATE_ROLE)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.missing, "synonym")

        report = coverage_report(COMPLETE, date_roles=DATE_ROLE)
        self.assertTrue(report.complete)


class TestNameMatching(unittest.TestCase):

    def test_a_whole_word_match_finds_the_metric(self):
        self.assertTrue(question_finds_metric("what is net revenue", COMPLETE))
        self.assertTrue(question_finds_metric("show NET REVENUE by region", COMPLETE))

    def test_a_substring_of_a_longer_word_does_not(self):
        self.assertFalse(
            question_finds_metric("revenuestream by region",
                                  {"name": "revenue", "synonyms": ""}))

    def test_a_synonym_finds_it_too(self):
        self.assertTrue(question_finds_metric("net sales last month", COMPLETE))

    def test_an_unrelated_question_does_not(self):
        self.assertFalse(question_finds_metric("how many orders shipped", COMPLETE))


# ══════════════════════════════════════════════════════════════════════════════
# Both languages
# ══════════════════════════════════════════════════════════════════════════════

class TestBothLanguages(unittest.TestCase):

    def test_every_question_shape_differs_between_the_languages(self):
        english = [v.question for v in variations_for(COMPLETE, lang="en")]
        french = [v.question for v in variations_for(COMPLETE, lang="fr")]
        self.assertEqual(len(english), len(french))
        # Compared line by line: a whole-list comparison passes when only one
        # shape in twenty was translated.
        untranslated = set(english) & set(french)
        self.assertFalse(untranslated, untranslated)

    def test_every_gap_reason_reads_in_both_languages(self):
        en = coverage_report(BARE, date_roles=[], lang="en")
        fr = coverage_report(BARE, date_roles=[], lang="fr")
        self.assertEqual(len(en.gaps), len(fr.gaps))
        self.assertFalse({g.reason for g in en.gaps} & {g.reason for g in fr.gaps})

    def test_the_summary_reads_in_both_languages(self):
        en = coverage_report(BARE, date_roles=[], lang="en")
        fr = coverage_report(BARE, date_roles=[], lang="fr")
        self.assertTrue(en.summary and fr.summary)
        self.assertNotEqual(en.summary, fr.summary)

    def test_french_takes_the_singular_at_one(self):
        # French puts "1 formulation" in the singular; English says "shapes"
        # either way. The distinction lives in the catalogue, not inline.
        self.assertIn("coverage.summary.one", MESSAGES)
        self.assertIn("coverage.summary.other", MESSAGES)
        self.assertNotEqual(MESSAGES["coverage.summary.one"]["fr"],
                            MESSAGES["coverage.summary.other"]["fr"])

    def test_every_coverage_id_exists_in_both_languages(self):
        ids = [k for k in MESSAGES if k.startswith("coverage.")]
        self.assertGreater(len(ids), 15)
        for msg_id in ids:
            self.assertTrue(MESSAGES[msg_id].get("en"), msg_id)
            self.assertTrue(MESSAGES[msg_id].get("fr"), msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], msg_id)


class TestFailingOpen(unittest.TestCase):

    def test_a_broken_metric_row_yields_an_empty_report_not_an_exception(self):
        # Coverage is advice. Advice that can break a metric save is worse
        # than no advice.
        self.assertIsInstance(coverage_report(None), CoverageReport)
        self.assertIsInstance(coverage_report({}), CoverageReport)
        self.assertEqual(coverage_report({}).total, 0)

    def test_a_report_for_a_metric_that_does_not_exist_is_empty(self):
        from unittest.mock import patch

        import core.metric_coverage as module

        with patch("store.list_metrics", return_value=[]):
            report = module.report_for_metric_id("acct", 999)
        self.assertEqual(report.total, 0)


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# Wiring
# ══════════════════════════════════════════════════════════════════════════════

class TestTheAdminApi(unittest.TestCase):
    """Write API to read API: a metric saved through the store must be the
    metric the coverage endpoint reports on, with nothing handed between."""

    def setUp(self):
        import os
        import tempfile
        import uuid

        import store
        self._dir = tempfile.mkdtemp(prefix="qb-cov-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "cov.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-cov-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        import os
        import shutil

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _save_metric(self, **overrides):
        import store
        fields = {
            "name": "Net Revenue", "synonyms": "revenue",
            "sql_template": "SUM(NET_AMT)", "description": "",
            "allowed_dimensions": "Region", "grain": "month",
        }
        fields.update(overrides)
        store.save_metric(self.account_id, fields)
        return next(m for m in store.list_metrics(self.account_id)
                    if m["name"] == fields["name"])

    def _get(self, metric_id):
        import asyncio
        from unittest.mock import MagicMock, patch

        import admin.routes as routes

        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            response = asyncio.run(
                routes.metric_coverage_api(req, self.account_id, metric_id))
        import json
        return json.loads(response.body)

    def test_a_saved_metric_is_reported_on(self):
        metric = self._save_metric()
        body = self._get(int(metric["id"]))
        self.assertGreater(body["total"], 0)
        self.assertTrue(body["summary"])
        self.assertEqual(body["resolvable"] + len(body["gaps"]), body["total"])

    def test_the_dimensions_saved_are_the_dimensions_checked(self):
        metric = self._save_metric(allowed_dimensions="Region, Customer")
        body = self._get(int(metric["id"]))
        questions = " ".join(a["question"] for a in body["answers"])
        # Nothing handed in — the endpoint had to find these on the row.
        self.assertIn("Customer", questions)
        self.assertIn("Region", questions)

    def test_a_metric_missing_a_date_role_is_told_which_asset_to_add(self):
        metric = self._save_metric()
        body = self._get(int(metric["id"]))
        self.assertIn("coverage.gap.no_date_role", body["by_reason"])
        missing = {g["missing"] for g in body["gaps"]}
        self.assertIn("date_role", missing)

    def test_an_unknown_metric_reports_nothing_rather_than_erroring(self):
        body = self._get(999999)
        self.assertEqual(body["total"], 0)
        self.assertEqual(body["gaps"], [])

    def test_an_unauthenticated_request_gets_nothing(self):
        import asyncio
        from unittest.mock import MagicMock, patch

        import admin.routes as routes
        from fastapi import HTTPException

        req = MagicMock()
        req.query_params = {}
        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(HTTPException):
                asyncio.run(routes.metric_coverage_api(req, self.account_id, 1))


class TestTheChatDraftCarriesIt(unittest.TestCase):
    """The chat path is where the user's ask lands: they define a metric in
    conversation and should be told there and then which questions it will
    answer, not discover it later from a failed query."""

    def test_the_draft_payload_includes_the_coverage(self):
        import inspect

        import gateway.webhooks as wh

        source = inspect.getsource(wh.ws_chat)
        self.assertIn("from core.metric_coverage import coverage_report", source)
        self.assertIn('"coverage": _coverage,', source)
        # Computed before the payload is sent, or it reaches nothing.
        computed_at = source.index("_report = coverage_report(draft.as_metric()")
        sent_at = source.index('"type": "assistant_metric_draft"')
        self.assertLess(computed_at, sent_at)

    def test_the_coverage_shape_the_payload_promises_is_what_the_engine_returns(self):
        # The payload claims total/resolvable/summary/gaps with question,
        # reason and missing. If the report's shape drifts, the browser gets
        # undefined and shows nothing, silently.
        report = coverage_report(
            {"name": "Net Revenue", "allowed_dimensions": "Region", "grain": "month"},
            date_roles=[])
        self.assertIsInstance(report.total, int)
        self.assertIsInstance(report.resolvable, int)
        self.assertIsInstance(report.summary, str)
        self.assertTrue(report.gaps)
        gap = report.gaps[0]
        self.assertTrue(gap.variation.question)
        self.assertTrue(gap.reason)
        self.assertTrue(gap.missing)
