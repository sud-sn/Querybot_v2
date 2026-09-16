"""
tests/test_the_menu_only_offers_what_the_question_touched.py

The last of the three ways "which measure do you want?" reached a reader whose
question named no measure at all.

_scored_terms_for_ambiguity_menu scores every term in this reader's scope by
word overlap with the question, and its result is the menu a constrained model
must pick the reader's options from — there is no other use for it; both
callers turn it straight into options. It ended with:

    chosen = positives if len(positives) >= 2 else scored

so when fewer than two terms scored, the ENTIRE scoped vocabulary was handed
over, capped at twenty. The model, asked to disambiguate and given twenty
choices, disambiguated. A reader asking how many pallets shipped from a
warehouse was invited to choose between net revenue and gross margin — terms
that had scored exactly zero against their sentence.

Only scoring terms are returned now. Under two of them, both callers fall
through to the plain CLEAR/AMBIGUOUS classifier, which asks in free text with
no options — the honest shape for "we do not know which of your terms you
meant, because none of them is yours". Terms an admin marked ambiguous are
untouched: Step 1 of check_ambiguity_glossary_first resolves those
deterministically, before this menu is built at all.

Every test executes the real scorer.
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
    os.path.join(tempfile.mkdtemp(prefix="qb-menu-scope-"), "querybot.db"),
)

from core.clarification import _scored_terms_for_ambiguity_menu  # noqa: E402

VOCABULARY = [
    {"id": 1, "term": "net revenue", "kind": "metric", "aliases": "sales value",
     "definition": "Invoiced amount after discounts", "canonical_expression": "SUM(A)",
     "tables_involved": ""},
    {"id": 2, "term": "gross margin", "kind": "metric", "aliases": "",
     "definition": "Revenue less cost of goods", "canonical_expression": "SUM(B)",
     "tables_involved": ""},
    {"id": 3, "term": "scrap rate", "kind": "metric", "aliases": "",
     "definition": "Rejected quantity over produced quantity",
     "canonical_expression": "SUM(C)", "tables_involved": ""},
]

UNRELATED = "how many pallets did we ship from the north warehouse yesterday"


def _menu(question: str) -> list[str]:
    with patch("store.list_terms", return_value=[dict(t) for t in VOCABULARY]):
        return [term["term"] for term in
                _scored_terms_for_ambiguity_menu("acct", question)]


class TheMenuIsBuiltFromTheQuestion(unittest.TestCase):

    def test_a_question_that_touches_nothing_produces_no_menu(self):
        """The reported defect. Every term here scores zero against these
        words, and a menu built from them is a menu of measures the reader
        never mentioned."""
        self.assertEqual(_menu(UNRELATED), [])

    def test_one_term_touched_is_not_a_choice_between_readings(self):
        """Under two, the callers fall through to the plain classifier, which
        asks with no options — so this list stopping at one is what stops a
        second, invented option being offered beside it.

        "net revenue" would not do as the example: "revenue" is a word in
        gross margin's own definition, so that question genuinely reaches two
        terms. Scoring over the definition is the feature, not a leak."""
        self.assertEqual(_menu("show me scrap rate for last month"), ["scrap rate"])

    def test_two_terms_touched_still_produce_a_menu(self):
        """The feature has to keep working: a question that genuinely reaches
        two measures is the case this menu exists for."""
        menu = _menu("compare net revenue and gross margin")
        self.assertEqual(sorted(menu), ["gross margin", "net revenue"])

    def test_a_term_is_reached_through_its_alias_and_its_definition(self):
        """Scoring is over the term, its aliases and its definition together,
        and narrowing the menu must not quietly narrow that."""
        self.assertIn("net revenue", _menu("what is our sales value this year"))
        self.assertIn("scrap rate", _menu("rejected quantity by work centre"))

    def test_the_better_match_is_first(self):
        """The cap is twenty, so order decides what a long vocabulary shows.

        The question is built to make the scores genuinely differ — it reaches
        "net revenue" through four words and "gross margin" through two. A
        question where the two tie cannot tell a correct ordering from a
        reversed one."""
        menu = _menu("revenue and margin and discounts and sales value")
        self.assertEqual(menu, ["net revenue", "gross margin"])

    def test_an_empty_vocabulary_is_an_empty_menu_not_a_crash(self):
        with patch("store.list_terms", return_value=[]):
            self.assertEqual(
                _scored_terms_for_ambiguity_menu("acct", UNRELATED), [])


class TheCallersSeeTheNarrowerMenu(unittest.TestCase):
    """The menu is only worth narrowing if the narrowing reaches the reader."""

    @staticmethod
    def _prompt_for(question: str, reply: str) -> str:
        """The prompt actually sent to the model for this question.

        Observing the PROMPT is the point. A verdict of CLEAR comes back empty
        whichever path was taken, so asserting on the verdict would pass with
        the narrowing reverted — the model was still handed the whole
        vocabulary, it just happened to be told to say CLEAR.
        """
        import asyncio

        from core import clarification as clar

        seen: list[str] = []

        async def _fake_complete(*args, **kwargs):
            seen.append("\n".join(str(a) for a in args) + str(kwargs))
            return reply, 10, 5

        async def _run():
            with patch("store.list_terms",
                       return_value=[dict(t) for t in VOCABULARY]), \
                 patch("core.llm.llm_complete", new=_fake_complete):
                return await clar._llm_ambiguity_check_constrained(
                    account_id="acct", question=question, context="",
                    provider="test", model="test", api_key="", extra_kwargs={},
                )

        result = asyncio.run(_run())
        return result, "\n".join(seen)

    def test_an_unrelated_question_never_puts_a_menu_in_front_of_the_model(self):
        """The reply is stubbed AMBIGUOUS on purpose: if the constrained menu
        were still being built, the model would be free to pick from it."""
        (is_ambiguous, _q, options), prompt = self._prompt_for(
            UNRELATED,
            '{"status":"AMBIGUOUS","question":"Which one?","option_ids":["t1","t2"]}',
        )
        self.assertEqual(options, [])
        for term in ("net revenue", "gross margin", "scrap rate"):
            self.assertNotIn(term, prompt.lower(), f"{term!r} was offered anyway")
        # And the ids the constrained prompt would have carried are absent, so
        # the model could not have been asked to choose between readings.
        self.assertNotIn("t1", prompt)
        self.assertNotIn("t2", prompt)

    def test_a_question_that_earns_a_menu_gets_one(self):
        """The mirror, so the test above cannot pass by the menu never being
        built at all."""
        (is_ambiguous, _q, options), prompt = self._prompt_for(
            "net revenue and gross margin by region",
            '{"status":"AMBIGUOUS","question":"Which one?","option_ids":["t1","t2"]}',
        )
        self.assertTrue(is_ambiguous)
        self.assertEqual([option["_term_id"] for option in options], [1, 2])
        self.assertIn("net revenue", prompt.lower())
        self.assertIn("gross margin", prompt.lower())



if __name__ == "__main__":
    unittest.main()
