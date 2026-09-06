# -*- coding: utf-8 -*-
"""The confidence panel, in the reader's language — core/answer_confidence.py.

Every string this module produces used to be an English literal, and the
module imported no i18n at all. It is not a corner of the product: the verdict,
its reasons and its watch-outs are on every answer card, and the panel a reader
opens to decide whether to trust a number was the one panel that never spoke
their language. A French user got "Confiance" as a heading over "SQL passed
schema validation."

The tests execute the scorer under an activated language rather than passing
`lang=` in, because both production callers pass nothing — they run inside the
activation `_handle_query_impl` sets up for the turn — and a test that hands
the language in cannot see a scorer that ignores it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.answer_confidence import build_answer_confidence  # noqa: E402
from core.i18n import MESSAGES, activate_language, deactivate_language  # noqa: E402


def _under(lang, **kwargs):
    token = activate_language(lang)
    try:
        return build_answer_confidence(**kwargs)
    finally:
        deactivate_language(token)


# Every branch that can add a line, and the arguments that reach it.
_BRANCHES = {
    "clean": dict(validation_code="ok", row_count=5, tables_used=["S.F"],
                  has_semantic_plan=True, has_graph_context=True),
    "no_rows": dict(validation_code="ok", row_count=0, tables_used=["S.F"]),
    "empty_table": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        empty_tables=["S.F"]),
    "retry": dict(validation_code="ok", row_count=3, retry_count=1, tables_used=["S.F"]),
    "bad_validation": dict(validation_code="unknown_column", row_count=3,
                           tables_used=["S.F"]),
    "null_metric": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        null_metric_issue=True),
    "derived_gap": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        derived_metric_gap="open order quantity"),
    "weak_retrieval": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                           weak_retrieval=True),
    "unscored": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                     retrieval_unscored=True),
    "planning_failed": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                            semantic_planning_failed=True),
    "suggested_graph": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                            has_graph_context=True, graph_scope="suggested_fallback"),
    "graph_failed": dict(validation_code="ok", row_count=3,
                         tables_used=["S.F", "S.D"], graph_resolution_failed=True),
    "fanout": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                   fanout_risk=True),
    "candidates_disagree": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                                candidate_selection={"reason": "verified_candidates_disagree"}),
    "no_candidate": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                         candidate_selection={"reason": "no_candidate_verified"}),
    "candidates_agree": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                             candidate_selection={"reason": "agreement_of_2"}),
    "areas_agree": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        corroboration={"checked": True, "agrees": True}),
    "areas_disagree": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                           corroboration={"checked": True, "agrees": False}),
    "verify_unavailable": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                               result_verification={"status": "unavailable"}),
    "verify_pass": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        result_verification={"status": "pass"}),
    "verify_fail": dict(validation_code="ok", row_count=3, tables_used=["S.F"],
                        result_verification={"status": "fail"}),
}


class TestEveryLineIsTranslated(unittest.TestCase):

    def test_no_branch_gives_a_french_reader_an_english_line(self):
        """The whole surface, branch by branch.

        Asserting only that French differs from English on ONE branch would
        pass against a module that translated the label and nothing else --
        which is roughly the state this replaced.
        """
        untranslated = []
        for name, kwargs in _BRANCHES.items():
            french = _under("fr", **kwargs)
            english = _under("en", **kwargs)
            for key in ("reasons", "warnings"):
                for fr_line, en_line in zip(french[key], english[key]):
                    if fr_line == en_line:
                        untranslated.append(f"{name}.{key}: {en_line[:70]}")
        self.assertEqual(untranslated, [], "\n".join(untranslated))

    def test_every_branch_actually_produced_a_line(self):
        # Guards the test above: a branch that emits nothing has no line to
        # compare, and would sit in the table looking covered.
        silent = [
            name for name, kwargs in _BRANCHES.items()
            if not (_under("fr", **kwargs)["reasons"] + _under("fr", **kwargs)["warnings"])
        ]
        self.assertEqual(silent, [])

    def test_the_verdict_label_is_translated(self):
        for kwargs in (_BRANCHES["clean"], _BRANCHES["fanout"], _BRANCHES["no_rows"]):
            self.assertNotEqual(_under("fr", **kwargs)["label"],
                                _under("en", **kwargs)["label"])

    def test_the_score_and_level_do_not_move_with_the_language(self):
        # Translation must not change the verdict, only its wording. Both
        # ways of choosing the language, because both are supported: the
        # ambient activation (what production uses) and an explicit lang=
        # (what a caller outside the request would pass).
        for name, kwargs in _BRANCHES.items():
            french, english = _under("fr", **kwargs), _under("en", **kwargs)
            self.assertEqual(french["score"], english["score"], name)
            self.assertEqual(french["level"], english["level"], name)

            passed_fr = build_answer_confidence(lang="fr", **kwargs)
            passed_en = build_answer_confidence(lang="en", **kwargs)
            self.assertEqual(passed_fr["score"], passed_en["score"], name)
            self.assertEqual(passed_fr["level"], passed_en["level"], name)

    def test_an_explicit_language_translates_without_an_activation(self):
        # A caller with a language and no activation -- the docstring promises
        # this and nothing else exercised it.
        kwargs = _BRANCHES["fanout"]
        self.assertNotEqual(
            build_answer_confidence(lang="fr", **kwargs)["warnings"],
            build_answer_confidence(lang="en", **kwargs)["warnings"],
        )
        self.assertNotEqual(
            build_answer_confidence(lang="fr", **kwargs)["label"],
            build_answer_confidence(lang="en", **kwargs)["label"],
        )

    def test_english_still_reads_the_way_it_did(self):
        clean = _under("en", **_BRANCHES["clean"])
        self.assertEqual(clean["label"], "High confidence")
        self.assertIn("SQL passed schema validation.", clean["reasons"])


class TestTheCountsAreWrittenForTheReader(unittest.TestCase):
    """French pluralises at zero and one, and groups thousands differently."""

    def _rows_line(self, lang, rows):
        result = _under(lang, validation_code="ok", row_count=rows, tables_used=["S.F"])
        return next((r for r in result["reasons"] if str(rows) in r or "ligne" in r or "row" in r), "")

    def test_french_takes_the_singular_at_one(self):
        line = self._rows_line("fr", 1)
        self.assertIn("ligne", line)
        self.assertNotIn("lignes", line)

    def test_french_takes_the_plural_above_one(self):
        self.assertIn("lignes", self._rows_line("fr", 2))

    def test_english_takes_the_plural_at_two(self):
        self.assertIn("rows", self._rows_line("en", 2))
        self.assertIn("row.", self._rows_line("en", 1))

    def test_the_id_pair_carries_the_shared_plural_rule(self):
        """Tested through plural() directly, because zero cannot reach it here.

        French takes the singular at zero as well as one, and that is the rule
        this call site uses -- but a row count of zero routes to the "no rows"
        branch instead, so the difference between the shared rule and
        English's own is not observable from build_answer_confidence. What is
        testable, and what this module owns, is that the id pair is well
        formed for both forms under the shared rule.
        """
        from core.i18n import plural

        self.assertIn("ligne", plural("confidence.reason.rows_returned", 0,
                                      lang="fr", rows="0"))
        self.assertNotIn("lignes", plural("confidence.reason.rows_returned", 0,
                                          lang="fr", rows="0"))
        self.assertIn("rows", plural("confidence.reason.rows_returned", 0,
                                     lang="en", rows="0"))

    def test_thousands_are_grouped_the_way_each_language_groups_them(self):
        # A narrow no-break space in French, a comma in English. "1,234" reads
        # as one-point-two-three-four to a French reader.
        self.assertIn("1 234", self._rows_line("fr", 1234))
        self.assertIn("1,234", self._rows_line("en", 1234))


class TestTheCatalogueIsComplete(unittest.TestCase):

    def test_every_confidence_id_exists_in_both_languages_and_differs(self):
        ids = [k for k in MESSAGES if k.startswith("confidence.")]
        self.assertGreaterEqual(len(ids), 25)
        for msg_id in ids:
            self.assertTrue(MESSAGES[msg_id].get("en"), msg_id)
            self.assertTrue(MESSAGES[msg_id].get("fr"), msg_id)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], msg_id)

    def test_no_line_leaks_a_raw_message_id(self):
        # t() returns the id itself when it is missing, so a typo shows up on
        # the card as "confidence.warn.something" rather than failing.
        for lang in ("en", "fr"):
            for name, kwargs in _BRANCHES.items():
                result = _under(lang, **kwargs)
                for line in result["reasons"] + result["warnings"] + [result["label"]]:
                    self.assertNotIn("confidence.", line, f"{lang} {name}: {line}")

    def test_placeholders_are_filled(self):
        for lang in ("en", "fr"):
            for name in ("derived_gap", "empty_table"):
                for line in _under(lang, **_BRANCHES[name])["warnings"]:
                    self.assertNotIn("{", line, f"{lang} {name}")


if __name__ == "__main__":
    unittest.main()
