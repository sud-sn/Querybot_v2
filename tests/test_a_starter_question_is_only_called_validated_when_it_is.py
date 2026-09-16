"""
tests/test_a_starter_question_is_only_called_validated_when_it_is.py

The heading over the starter questions is a claim about where they came from,
and it was making a claim two of the three sources cannot support.

core.suggestions fills the panel from three tiers:

  1. Validated examples. Two different guarantees under one name, as the
     module's own comment says: a query_log example is a question a reader
     actually asked that actually came back with rows; a kb_stage2 example is
     SQL the model wrote during the knowledge-base build which core.examples
     COMPILE-checked, proving its tables and columns resolve and nothing at
     all about whether it returns anything.
  2. The metric registry, whose question text is synthesised as "What is our
     total {name}?" from a metric name. Never executed.
  3. Stage-2 query patterns -- text the model wrote, never executed.

The workspace guide printed "Validated questions for your current access:"
over all of it. So a tenant with no query history at all -- a new workspace,
which is every workspace on day one -- was shown six questions under a heading
promising they had been checked, and the ones that failed when clicked were the
ones the heading was vouching for.

Provenance now travels with the suggestion, and the strong heading is used only
when every question shown is one a reader asked and got rows from. Otherwise
the heading says what IS true of all three tiers: they are scoped to this
reader's access and the entity graph can reach what they name.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "QUERYBOT_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="qb-suggestion-provenance-"), "querybot.db"),
)

from core import i18n  # noqa: E402
from core.suggestions import get_suggestions  # noqa: E402
from core.workspace_guide import _safe_examples, render_workspace_guide  # noqa: E402

ACCOUNT = "provenance-acct"
ASKED_AND_ANSWERED = "Which customers placed the most orders last quarter?"
COMPILED_ONLY = "Show invoice totals by warehouse"


def _examples(*sources: str) -> list[dict]:
    """validated_examples rows, as store.get_validated_examples returns them."""
    rows = []
    if "query_log" in sources:
        rows.append({"question": ASKED_AND_ANSWERED, "sql_query": "SELECT 1",
                     "table_name": "DB.SALES.F_ORDERS", "source": "query_log"})
    if "kb_stage2" in sources:
        rows.append({"question": COMPILED_ONLY, "sql_query": "SELECT 1",
                     "table_name": "DB.SALES.F_ORDERS", "source": "kb_stage2"})
    return rows


class _Tiers:
    """Everything core.suggestions reads, patched at its boundary so the real
    tiering, ACL filter and graph gate all run."""

    def __init__(self, examples, metrics=()):
        self._patchers = [
            patch("store.get_validated_examples", return_value=list(examples)),
            patch("store.list_metrics", return_value=[dict(m) for m in metrics]),
            # The graph and date-role gates every tier passes through. Open,
            # so a suggestion that is dropped here is dropped for provenance
            # reasons and nothing else.
            patch("core.suggestions._graph_reachability_check",
                  return_value=lambda _q: True),
            patch("core.suggestions._date_scope_check",
                  return_value=lambda _q: True),
        ]

    def __enter__(self):
        for patcher in self._patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False


def _suggest(examples, metrics=(), n=6):
    with _Tiers(examples, metrics):
        return get_suggestions(ACCOUNT, "", None, n=n, schema_dir="")


class ProvenanceTravelsWithTheSuggestion(unittest.TestCase):

    def test_a_question_a_reader_asked_and_got_rows_from_is_proven(self):
        found = _suggest(_examples("query_log"))
        self.assertEqual([s["question"] for s in found], [ASKED_AND_ANSWERED])
        self.assertTrue(found[0]["proven"])

    def test_a_compile_checked_example_is_not_proven(self):
        """It proves the tables and columns resolve. It says nothing about
        whether the question returns a row, which is what the reader is being
        promised."""
        found = _suggest(_examples("kb_stage2"))
        self.assertEqual([s["question"] for s in found], [COMPILED_ONLY])
        self.assertFalse(found[0]["proven"])

    def test_a_metric_name_turned_into_a_sentence_is_not_proven(self):
        """Tier 2 synthesises the question from the metric's name. Nothing has
        ever run it, and the pipeline has to re-plan the sentence from its
        text like any other."""
        found = _suggest([], metrics=[
            {"name": "Net Revenue", "sql_template": "SELECT 1"},
        ])
        self.assertTrue(found)
        self.assertNotIn(ASKED_AND_ANSWERED, [s["question"] for s in found])
        for suggestion in found:
            self.assertFalse(suggestion["proven"], suggestion["question"])

    def test_every_suggestion_carries_the_key_at_all(self):
        """A missing key reads as false to the caller, which is right, but a
        tier that forgets it is a tier whose provenance nobody can inspect."""
        found = _suggest(_examples("query_log", "kb_stage2"), metrics=[
            {"name": "Net Revenue", "sql_template": "SELECT 1"},
        ])
        self.assertTrue(found)
        for suggestion in found:
            self.assertIn("proven", suggestion)
            self.assertIsInstance(suggestion["proven"], bool)


class TheGuideOnlyClaimsWhatItCanProve(unittest.TestCase):

    def _examples_and_claim(self, rows, metrics=()):
        with _Tiers(rows, metrics):
            return _safe_examples(ACCOUNT, "", "", None)

    def test_an_all_proven_list_may_be_called_validated(self):
        questions, proven = self._examples_and_claim(_examples("query_log"))
        self.assertEqual(questions, [ASKED_AND_ANSWERED])
        self.assertTrue(proven)

    def test_one_unproven_question_withdraws_the_claim_for_the_block(self):
        """The heading sits over the whole list, so it can only be as strong
        as its weakest member."""
        questions, proven = self._examples_and_claim(
            _examples("query_log", "kb_stage2"))
        self.assertEqual(len(questions), 2)
        self.assertFalse(proven)

    def test_an_empty_list_claims_nothing(self):
        questions, proven = self._examples_and_claim([])
        self.assertEqual(questions, [])
        self.assertFalse(proven)

    def test_a_suggestion_engine_that_fails_claims_nothing(self):
        """_safe_examples is fail-open by design. Failing open must not mean
        failing open on the promise as well."""
        with patch("core.suggestions.get_suggestions",
                   side_effect=RuntimeError("no knowledge base")):
            self.assertEqual(_safe_examples(ACCOUNT, "", "", None), ([], False))


class TheHeadingSaysWhichOneItIs(unittest.TestCase):
    """The rendered guide, in both languages."""

    USER = {"id": 1, "account_id": ACCOUNT, "role": "user"}

    def _rendered(self, kind: str, proven: bool, lang: str,
                  examples=(ASKED_AND_ANSWERED,)) -> str:
        patchers = [
            patch("core.workspace_guide.store.get_client", return_value={}),
            patch("core.workspace_guide.get_state", return_value={}),
            patch("core.workspace_guide.load_semantic_model", return_value={}),
            patch("core.workspace_guide.store.get_allowed_tables",
                  return_value={"DB.SALES.F_ORDERS"}),
            patch("core.workspace_guide.store.list_metrics", return_value=[]),
            patch("core.workspace_guide.store.list_terms", return_value=[]),
            patch("core.workspace_guide.store.list_dashboards", return_value=[]),
            patch("core.workspace_guide._safe_examples",
                  return_value=(list(examples), proven)),
        ]
        for patcher in patchers:
            patcher.start()
        token = i18n.activate_language(lang)
        try:
            return render_workspace_guide(kind, ACCOUNT, self.USER)[0]
        finally:
            i18n.deactivate_language(token)
            for patcher in reversed(patchers):
                patcher.stop()

    def test_the_capability_overview_withdraws_the_claim_too(self):
        """It prints the SAME list under its own heading, and that heading said
        "Try one of these validated questions" — the identical claim, missed
        when the other one was split."""
        for lang, claimed in (("en", "validated"), ("fr", "validée")):
            with self.subTest(lang=lang):
                unproven = self._rendered("capability_overview", False, lang)
                self.assertIn(ASKED_AND_ANSWERED, unproven)
                self.assertNotIn(claimed, unproven.lower())
                # ...and the strong claim is still available when earned.
                self.assertIn(
                    claimed, self._rendered(
                        "capability_overview", True, lang).lower())

    def test_an_empty_list_does_not_claim_validation_either(self):
        """"No VALIDATED starter questions are available" invites the reader to
        infer that unvalidated ones exist and are being withheld."""
        for lang, claimed in (("en", "validated"), ("fr", "validée")):
            with self.subTest(lang=lang):
                for kind in ("question_examples", "capability_overview"):
                    text = self._rendered(kind, False, lang, examples=())
                    self.assertNotIn(claimed, text.lower(), kind)
                    self.assertNotIn("guide.examples.none", text)

    def _heading(self, proven: bool, lang: str) -> str:
        patchers = [
            patch("core.workspace_guide.store.get_client", return_value={}),
            patch("core.workspace_guide.get_state", return_value={}),
            patch("core.workspace_guide.load_semantic_model", return_value={}),
            patch("core.workspace_guide.store.get_allowed_tables",
                  return_value={"DB.SALES.F_ORDERS"}),
            patch("core.workspace_guide.store.list_metrics", return_value=[]),
            patch("core.workspace_guide.store.list_terms", return_value=[]),
            patch("core.workspace_guide.store.list_dashboards", return_value=[]),
            patch("core.workspace_guide._safe_examples",
                  return_value=([ASKED_AND_ANSWERED], proven)),
        ]
        for patcher in patchers:
            patcher.start()
        token = i18n.activate_language(lang)
        try:
            return render_workspace_guide("question_examples", ACCOUNT, self.USER)[0]
        finally:
            i18n.deactivate_language(token)
            for patcher in reversed(patchers):
                patcher.stop()

    def test_proven_questions_get_the_strong_heading(self):
        self.assertIn("Validated questions", self._heading(True, "en"))
        self.assertIn("Questions validées", self._heading(True, "fr"))

    def test_unproven_questions_do_not(self):
        for lang, claimed in (("en", "Validated questions"),
                              ("fr", "Questions validées")):
            with self.subTest(lang=lang):
                text = self._heading(False, lang)
                self.assertNotIn(claimed, text)
                # ...but the questions are still offered, still bulleted, and
                # still introduced by a heading rather than appearing bare.
                self.assertIn(ASKED_AND_ANSWERED, text)
                self.assertIn(f"  • _{ASKED_AND_ANSWERED}_", text)
                heading = text.rsplit("\n  • ", 1)[0].rsplit("\n\n", 1)[-1]
                self.assertTrue(heading.startswith("*"), heading)
                self.assertNotIn("guide.", heading)

    def test_the_weaker_heading_is_translated_and_says_something(self):
        english = self._heading(False, "en")
        french = self._heading(False, "fr")
        self.assertIn("Questions you can ask", english)
        self.assertIn("Questions que vous pouvez poser", french)
        # A catalogue id rendering as itself means the key is missing.
        self.assertNotIn("guide.questions.available", french)


if __name__ == "__main__":
    unittest.main()
