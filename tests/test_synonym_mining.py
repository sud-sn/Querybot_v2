# -*- coding: utf-8 -*-
"""Learning the words readers use — core/synonym_mining.py.

A metric carries the vocabulary of whoever wrote it down. Readers use a
different one, and the gap only ever shows up as a failure: a question that
should have bound to a metric did not, the reader rephrased, and the second
attempt worked. Nobody recorded the first attempt, so the same word failed
again next week for the next person.

The three properties asserted hardest are the ones that keep this from being
noise dressed as insight:

  * only the DIFFERENCE counts — a word that survives into the successful
    question is one the metric already understands, and proposing it would add
    a synonym that changes nothing;
  * one occurrence is a typo, so nothing is proposed below MIN_OCCURRENCES; and
  * a proposal is never applied. Writing a synonym straight onto a governed
    metric from a mistyped question is the shape of a supply-chain attack.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.synonym_mining import (  # noqa: E402
    MAX_PHRASE_WORDS,
    MIN_OCCURRENCES,
    REPHRASE_WINDOW_MINUTES,
    Rephrase,
    candidates_from,
    known_vocabulary,
    propose,
    rephrase_pairs,
)

REVENUE = {
    "id": 7,
    "name": "Net Revenue",
    "synonyms": "revenue, sales",
    "sql_template": "SELECT SUM(NET_REVENUE_AMOUNT) FROM SALES.F_INVOICE",
}


def _row(question, success, at, user="u1"):
    return {"question": question, "success": 1 if success else 0,
            "created_at": at, "user_key": user}


class TestFindingTheRephrase(unittest.TestCase):

    def test_a_failure_followed_by_a_success_is_a_pair(self):
        pairs = rephrase_pairs([
            _row("what was turnover last month", False, "2026-01-01 09:00:00"),
            _row("what was revenue last month", True, "2026-01-01 09:01:00"),
        ])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].failed, "what was turnover last month")
        self.assertEqual(pairs[0].succeeded, "what was revenue last month")

    def test_a_success_with_no_failure_before_it_is_not_a_pair(self):
        self.assertEqual(rephrase_pairs([
            _row("what was revenue", True, "2026-01-01 09:00:00")]), [])

    def test_a_failure_nobody_retried_is_not_a_pair(self):
        self.assertEqual(rephrase_pairs([
            _row("what was turnover", False, "2026-01-01 09:00:00")]), [])

    def test_a_retry_the_next_morning_is_a_new_question(self):
        pairs = rephrase_pairs([
            _row("what was turnover", False, "2026-01-01 09:00:00"),
            _row("what was revenue", True, "2026-01-01 11:30:00"),
        ])
        self.assertEqual(pairs, [])

    def test_the_window_is_the_thing_that_excluded_it(self):
        # Guards the test above: the same pair inside the window IS one.
        pairs = rephrase_pairs([
            _row("what was turnover", False, "2026-01-01 09:00:00"),
            _row("what was revenue", True, "2026-01-01 11:30:00"),
        ], window_minutes=24 * 60)
        self.assertEqual(len(pairs), 1)

    def test_another_readers_success_is_not_this_readers_rephrase(self):
        pairs = rephrase_pairs([
            _row("what was turnover", False, "2026-01-01 09:00:00", user="u1"),
            _row("what was revenue", True, "2026-01-01 09:01:00", user="u2"),
        ])
        self.assertEqual(pairs, [])

    def test_a_failure_two_questions_back_was_abandoned_not_rephrased(self):
        pairs = rephrase_pairs([
            _row("what was turnover", False, "2026-01-01 09:00:00"),
            _row("how many customers", True, "2026-01-01 09:01:00"),
            _row("what was revenue", True, "2026-01-01 09:02:00"),
        ])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].succeeded, "how many customers")

    def test_re_asking_the_same_words_is_not_a_rephrase(self):
        pairs = rephrase_pairs([
            _row("what was revenue", False, "2026-01-01 09:00:00"),
            _row("What Was Revenue", True, "2026-01-01 09:00:30"),
        ])
        self.assertEqual(pairs, [])

    def test_a_missing_timestamp_does_not_lose_the_pair(self):
        # An unparseable date is not evidence the retry was late.
        pairs = rephrase_pairs([
            _row("what was turnover", False, ""),
            _row("what was revenue", True, ""),
        ])
        self.assertEqual(len(pairs), 1)

    def test_the_default_window_is_a_retry_not_a_session(self):
        self.assertLessEqual(REPHRASE_WINDOW_MINUTES, 30)


class TestOnlyTheDifferenceIsProposed(unittest.TestCase):

    def _candidates(self, failed, succeeded, metric=None):
        return candidates_from(Rephrase(failed=failed, succeeded=succeeded),
                               metric or REVENUE)

    def test_the_word_the_reader_dropped_is_a_candidate(self):
        self.assertIn("turnover",
                      self._candidates("what was turnover last month",
                                       "what was revenue last month"))

    def test_a_word_in_both_questions_is_not(self):
        # It survived the rephrase, so the metric already understands it.
        self.assertNotIn("month",
                         self._candidates("turnover last month", "revenue last month"))

    def test_a_word_the_metric_already_knows_is_not(self):
        for known in ("revenue", "sales"):
            self.assertNotIn(known, self._candidates(f"{known} figures", "net revenue"))

    def test_a_column_name_is_naming_the_thing_not_renaming_it(self):
        self.assertNotIn("net revenue amount",
                         self._candidates("net revenue amount please", "net revenue"))

    def test_a_stopword_is_never_a_synonym(self):
        for noise in ("what", "the", "combien", "quel"):
            self.assertNotIn(noise, self._candidates(f"{noise} zzz", "revenue"))

    def test_multi_word_phrases_are_offered_up_to_the_cap(self):
        found = self._candidates("what was gross turnover figure", "what was revenue")
        self.assertIn("gross turnover", found)
        self.assertTrue(all(len(p.split()) <= MAX_PHRASE_WORDS for p in found))

    def test_a_filter_value_is_not_proposed_as_a_name_for_the_measure(self):
        """A reader naming a slice is not renaming the metric.

        "bretagne numbers" for a metric already scoped to that region is the
        reader saying which rows they want. Accepting it as a synonym would
        bind a governed measure to one region's name for everybody.
        """
        metric = dict(REVENUE,
                      sql_template="SELECT SUM(AMT) FROM F WHERE REGION = 'Bretagne'")
        self.assertIn("bretagne", known_vocabulary(metric))
        self.assertNotIn("bretagne", self._candidates("bretagne numbers", "revenue", metric))

    def test_a_word_that_is_nowhere_in_the_metric_still_is(self):
        # Guards the test above against excluding everything.
        metric = dict(REVENUE,
                      sql_template="SELECT SUM(AMT) FROM F WHERE REGION = 'Bretagne'")
        self.assertIn("takings", self._candidates("takings numbers", "revenue", metric))


class TestRankingAndTheEvidenceBar(unittest.TestCase):

    def _propose(self, pairs, metric=REVENUE):
        return propose([Rephrase(failed=f, succeeded=s) for f, s in pairs],
                       lambda q: metric)

    def test_one_reader_saying_it_once_is_a_coincidence(self):
        self.assertEqual(self._propose([("turnover last month", "revenue last month")]), [])

    def test_saying_it_twice_is_a_vocabulary_gap(self):
        proposals = self._propose([
            ("turnover last month", "revenue last month"),
            ("turnover this year", "revenue this year"),
        ])
        self.assertEqual([p.phrase for p in proposals if p.phrase == "turnover"],
                         ["turnover"])

    def test_the_bar_is_what_excluded_the_single_case(self):
        self.assertGreaterEqual(MIN_OCCURRENCES, 2)

    def test_the_proposal_carries_its_evidence(self):
        proposals = self._propose([
            ("turnover last month", "revenue last month"),
            ("turnover this year", "revenue this year"),
        ])
        top = next(p for p in proposals if p.phrase == "turnover")
        self.assertEqual(top.occurrences, 2)
        self.assertEqual(top.metric_id, 7)
        self.assertEqual(top.metric_name, "Net Revenue")
        self.assertIn("turnover last month", top.examples)
        self.assertIn("Net Revenue", top.detail)

    def test_more_evidence_ranks_higher(self):
        proposals = self._propose([
            ("turnover q1", "revenue q1"), ("turnover q2", "revenue q2"),
            ("turnover q3", "revenue q3"),
            ("takings q1", "revenue q1"), ("takings q2", "revenue q2"),
        ])
        self.assertEqual(proposals[0].phrase, "turnover")
        self.assertGreater(proposals[0].occurrences,
                           next(p for p in proposals if p.phrase == "takings").occurrences)

    def test_a_question_that_bound_to_no_metric_proposes_nothing(self):
        self.assertEqual(
            propose([Rephrase(failed="turnover", succeeded="how many rows")] * 3,
                    lambda q: None),
            [])

    def test_a_matcher_that_raises_costs_that_pair_and_nothing_else(self):
        calls = {"n": 0}

        def _flaky(question):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("store down")
            return REVENUE

        pairs = [Rephrase(failed="turnover a", succeeded="revenue a"),
                 Rephrase(failed="turnover b", succeeded="revenue b"),
                 Rephrase(failed="turnover c", succeeded="revenue c")]
        proposals = propose(pairs, _flaky)
        self.assertEqual(next(p for p in proposals if p.phrase == "turnover").occurrences, 2)

    def test_nothing_here_writes_to_a_metric(self):
        """A proposal is a queue entry, not an edit.

        Writing a synonym straight onto a governed metric from a mistyped
        question would let anyone who can ask a question rename a measure.
        """
        import inspect

        import core.synonym_mining as mining

        source = inspect.getsource(mining)
        for writer in ("save_metric", "update_metric", "conn.execute(\"UPDATE",
                       "INSERT INTO metric"):
            self.assertNotIn(writer, source, writer)


class TestTheWholePassOverARealLog(unittest.TestCase):
    """mine() against a real query_log, with nothing handed in by the test."""

    def setUp(self):
        import os
        import tempfile
        import uuid

        import store

        self._dir = tempfile.mkdtemp(prefix="qb-synmine-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-syn-{uuid.uuid4().hex[:8]}"
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

    def _log(self, question, success, at, user="7"):
        import store

        with store.get_db() as conn:
            conn.execute(
                "INSERT INTO query_log (account_id, question, success, created_at, "
                "zoom_user_id) VALUES (?, ?, ?, ?, ?)",
                (self.account_id, question, 1 if success else 0, at, user))

    def test_an_empty_log_proposes_nothing_and_does_not_raise(self):
        from core.synonym_mining import mine

        self.assertEqual(mine(self.account_id), [])

    def test_the_pass_reads_pairs_out_of_the_real_table(self):
        from unittest.mock import patch

        from core.synonym_mining import mine

        self._log("what was turnover in q1", False, "2026-01-01 09:00:00")
        self._log("what was revenue in q1", True, "2026-01-01 09:01:00")
        self._log("turnover by region", False, "2026-01-01 10:00:00")
        self._log("revenue by region", True, "2026-01-01 10:00:30")

        with patch("store.match_metric", return_value=REVENUE):
            proposals = mine(self.account_id)
        self.assertTrue(proposals)
        self.assertEqual(proposals[0].phrase, "turnover")
        self.assertEqual(proposals[0].occurrences, 2)

    def test_a_broken_log_read_is_a_report_that_says_nothing_not_a_crash(self):
        from unittest.mock import patch

        from core.synonym_mining import mine

        with patch("store.get_db", side_effect=RuntimeError("db gone")):
            with self.assertLogs("querybot.synonym_mining", level="WARNING"):
                self.assertEqual(mine(self.account_id), [])


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# The admin surface
# ══════════════════════════════════════════════════════════════════════════════

class TestAcceptingAProposal(unittest.TestCase):
    """The human step. Mining never writes; this is what does.

    Write API to read API: the route appends the phrase, and then
    ``store.match_metric`` — the matcher the PIPELINE uses — is asked whether a
    question using that word now finds the metric. A test that asserted only
    "the synonyms column changed" would pass against a synonym the matcher can
    never see.
    """

    def setUp(self):
        import asyncio
        import os
        import tempfile
        import uuid

        import store

        self._dir = tempfile.mkdtemp(prefix="qb-synaccept-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-acc-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.metric_id = store.save_metric(self.account_id, {
            "name": "Net Revenue",
            "synonyms": "revenue",
            "sql_template": "SELECT SUM(NET_REVENUE_AMOUNT) AS VALUE FROM SALES.F_INVOICE",
            "formula_type": "query",
        })
        self._asyncio = asyncio

    def tearDown(self):
        import os
        import shutil

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _accept(self, phrase, authed=True):
        from unittest.mock import MagicMock, patch

        import admin.routes as routes

        request = MagicMock()
        request.query_params = {}
        request.session = {"admin_id": "a1"}
        with patch.object(routes, "_is_auth", return_value=authed):
            return self._asyncio.run(routes.accept_synonym(
                request, self.account_id, self.metric_id, phrase=phrase))

    def _synonyms(self):
        import store

        row = next(m for m in store.list_metrics(self.account_id, active_only=False)
                   if int(m["id"]) == int(self.metric_id))
        return str(row.get("synonyms") or "")

    def test_the_phrase_is_added_to_the_metric(self):
        self._accept("turnover")
        self.assertIn("turnover", self._synonyms())

    def test_the_matcher_the_pipeline_uses_now_finds_it(self):
        import store

        self.assertIsNone(store.match_metric(self.account_id, "what was turnover"))
        self._accept("turnover")
        matched = store.match_metric(self.account_id, "what was turnover")
        self.assertIsNotNone(matched)
        self.assertEqual(int(matched["id"]), int(self.metric_id))

    def test_the_existing_synonyms_are_kept(self):
        self._accept("turnover")
        self.assertIn("revenue", self._synonyms())

    def test_accepting_twice_does_not_duplicate_it(self):
        self._accept("turnover")
        self._accept("Turnover")
        self.assertEqual(self._synonyms().lower().count("turnover"), 1)

    def test_a_blank_phrase_is_refused(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            self._accept("   ")
        self.assertNotIn(",", self._synonyms())

    def test_an_unknown_metric_is_refused(self):
        from fastapi import HTTPException

        self.metric_id = 999999
        with self.assertRaises(HTTPException):
            self._accept("turnover")

    def test_an_unauthenticated_accept_writes_nothing(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            self._accept("turnover", authed=False)
        self.assertNotIn("turnover", self._synonyms())


class TestTheReviewPanel(unittest.TestCase):
    PANEL = (Path(__file__).resolve().parents[1] / "admin" / "templates"
             / "metrics" / "_synonym_proposals.html")

    def test_the_metrics_page_includes_it(self):
        page = (Path(__file__).resolve().parents[1] / "admin" / "templates"
                / "client_metrics.html").read_text(encoding="utf-8")
        self.assertIn("metrics/_synonym_proposals.html", page)

    def test_it_calls_the_endpoints_that_exist(self):
        import admin.routes as routes

        markup = self.PANEL.read_text(encoding="utf-8")
        self.assertIn("/synonym-proposals", markup)
        self.assertIn("/synonyms/accept", markup)
        paths = {getattr(r, "path", "") for r in routes.router.routes}
        self.assertIn("/admin/api/clients/{account_id}/synonym-proposals", paths)
        self.assertIn("/admin/clients/{account_id}/metrics/{metric_id}/synonyms/accept",
                      paths)

    def test_an_empty_queue_shows_nothing_rather_than_an_empty_card(self):
        markup = self.PANEL.read_text(encoding="utf-8")
        self.assertIn("if (!proposals.length) return;", markup)
        self.assertIn('style="display:none"', markup)

    def test_a_failed_accept_says_so_rather_than_going_quiet(self):
        # A button that silently stops reads as success.
        markup = self.PANEL.read_text(encoding="utf-8")
        self.assertIn("Could not add", markup)

    def test_a_mined_phrase_cannot_inject_markup(self):
        # The phrase comes from a reader's question.
        markup = self.PANEL.read_text(encoding="utf-8")
        self.assertIn("esc(p.phrase)", markup)
        self.assertIn("esc(p.detail)", markup)
        self.assertIn("esc(q)", markup)
