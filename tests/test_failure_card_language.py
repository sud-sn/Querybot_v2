# -*- coding: utf-8 -*-
"""The failure and zero-row cards, in the reader's language.

Every one of these cards reached a French reader as English prose inside French
headings. The headings were already translated — the portal extracts each
section by an English wire label and renders `t('ui.chat.diag.reason')` above
the value — so what a reader saw was a French frame around an English answer to
"why did this fail".

Two things had to change together. The ~90 sentences moved into the catalogue,
and the card's kicker stopped being chosen by regex-matching the English
headline: that made the headline wire format, so translating it silently
downgraded every French failure to the "no rows" kicker. The server states the
`Kind:` now.

What must NOT be translated is asserted here too: the section labels are wire
format, and `cleaned` is the database's own error text, kept verbatim because
it is what support searches on.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.answer_formatter import (  # noqa: E402
    format_failure_business_response,
    format_zero_row_business_response,
)
from core.failure_messages import (  # noqa: E402
    build_query_timeout_guidance,
    sanitize_db_error,
    translate_failure,
)
from core.i18n import MESSAGES, activate_language, deactivate_language  # noqa: E402

# Every wire label the portal parses back out. Translating any of these breaks
# the card into an unparsed blob.
WIRE_LABELS = ("Most likely reason:", "Suggested next step:", "Why:",
               "Technical details:", "SQL tried:", "Confidence:", "Kind:")


def _under(lang, fn, *args, **kwargs):
    token = activate_language(lang)
    try:
        return fn(*args, **kwargs)
    finally:
        deactivate_language(token)


_FAILURES = {
    "missing table": dict(kind="execution",
                          exception_text="Invalid object name 'dbo.SALES'."),
    "login failed": dict(kind="execution", exception_text="Login failed for user 'qb'."),
    "timeout": dict(kind="execution", exception_text="Timeout expired."),
    "service limit": dict(kind="execution",
                          exception_text="Monthly free amount limit reached."),
    "unknown column": dict(kind="validation", code="unknown_column"),
    "fan-out join": dict(kind="validation", code="raw_fact_to_fact_join"),
    "access denied": dict(kind="validation", code="access_denied"),
    "cannot generate": dict(kind="validation", code="cannot_generate"),
    "unmapped code": dict(kind="validation", code="something_new"),
    "unknown kind": dict(kind="mystery", reason="no idea"),
}


class TestEveryFailureIsExplainedInTheReadersLanguage(unittest.TestCase):

    def test_no_failure_shape_gives_a_french_reader_an_english_explanation(self):
        untranslated = []
        for name, kwargs in _FAILURES.items():
            french = _under("fr", translate_failure, **kwargs)
            english = _under("en", translate_failure, **kwargs)
            for key in ("headline", "most_likely_reason", "suggested_next_step"):
                if french[key] == english[key]:
                    untranslated.append(f"{name}.{key}: {english[key][:70]}")
        self.assertEqual(untranslated, [], "\n".join(untranslated))

    def test_every_shape_actually_produced_prose(self):
        # Guards the test above: a shape that returns an empty string has
        # nothing to compare and would sit in the table looking covered.
        for name, kwargs in _FAILURES.items():
            rca = _under("fr", translate_failure, **kwargs)
            for key in ("headline", "most_likely_reason", "suggested_next_step"):
                self.assertTrue(str(rca[key]).strip(), f"{name}.{key}")

    def test_every_validator_code_has_a_reason_in_both_languages(self):
        from core.failure_messages import _VALIDATION_REASONS

        for code, msg_id in _VALIDATION_REASONS.items():
            self.assertIn(msg_id, MESSAGES, code)
            self.assertTrue(MESSAGES[msg_id].get("fr"), code)
            self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], code)

    def test_every_database_error_has_advice_in_both_languages(self):
        from core.failure_messages import _DB_ERROR_MAP

        for _pattern, stem in _DB_ERROR_MAP:
            for suffix in ("reason", "next_step"):
                msg_id = f"fail.db.{stem}.{suffix}"
                self.assertIn(msg_id, MESSAGES, msg_id)
                self.assertNotEqual(MESSAGES[msg_id]["en"], MESSAGES[msg_id]["fr"], msg_id)

    def test_the_suggested_terms_suffix_is_translated_and_keeps_the_terms(self):
        french = _under("fr", translate_failure, kind="validation",
                        code="unknown_column", suggestions=["revenue", "region"])
        english = _under("en", translate_failure, kind="validation",
                         code="unknown_column", suggestions=["revenue", "region"])
        self.assertIn("revenue, region", french["suggested_next_step"])
        self.assertNotEqual(french["suggested_next_step"],
                            english["suggested_next_step"])

    def test_the_timeout_guidance_is_translated(self):
        plan = {"temporal_policies": [
            {"fact_table": "S.F_SALES", "fact_column": "INVOICE_DATE_SK"}]}
        french = _under("fr", build_query_timeout_guidance, plan, timeout_seconds=30)
        english = _under("en", build_query_timeout_guidance, plan, timeout_seconds=30)
        self.assertNotEqual(french["reason"], english["reason"])
        self.assertNotEqual(french["next_step"], english["next_step"])
        # The table and column are data and must survive the translation.
        for text in (french["reason"], french["next_step"]):
            self.assertIn("S.F_SALES", text)
            self.assertIn("INVOICE_DATE_SK", text)


class TestWhatMustNotBeTranslated(unittest.TestCase):

    def test_the_database_error_text_is_kept_verbatim(self):
        """`cleaned` is the database's own words, and support searches on it."""
        raw = "[Microsoft][ODBC Driver 17] Invalid object name 'dbo.SALES'."
        french = _under("fr", sanitize_db_error, raw)
        english = _under("en", sanitize_db_error, raw)
        self.assertEqual(french["cleaned"], english["cleaned"])
        self.assertIn("Invalid object name", french["cleaned"])
        # ... while the advice around it is not.
        self.assertNotEqual(french["plain_reason"], english["plain_reason"])
        self.assertNotEqual(french["next_step"], english["next_step"])

    def test_the_error_text_survives_into_the_french_technical_note(self):
        rca = _under("fr", translate_failure, kind="execution",
                     exception_text="Invalid object name 'dbo.SALES'.")
        self.assertTrue(any("Invalid object name" in note
                            for note in rca["technical_notes"]))

    def test_the_section_labels_stay_english_in_both_languages(self):
        """They are wire format: the portal extracts by them and renders its own."""
        for lang in ("en", "fr"):
            card = _under(lang, format_failure_business_response,
                          rca=_under(lang, translate_failure, kind="validation",
                                     code="unknown_column"),
                          sql="SELECT 1")
            for label in ("Most likely reason:", "Suggested next step:", "Kind:"):
                self.assertIn(label, card, f"{lang} {label}")

    def test_the_zero_row_card_keeps_its_labels_too(self):
        for lang in ("en", "fr"):
            card = _under(lang, format_zero_row_business_response,
                          confidence={"label": "X", "score": 50,
                                      "warnings": ["w"], "reasons": []},
                          rca=_under(lang, translate_failure, kind="validation",
                                     code="unknown_column"),
                          sql="SELECT 1", sql_preview_fn=lambda s: s)
            for label in ("Confidence:", "Most likely reason:", "Why:", "SQL tried:"):
                self.assertIn(label, card, f"{lang} {label}")


class TestTheCardStatesItsKind(unittest.TestCase):
    """The kicker used to be inferred from the English headline."""

    def _kind(self, card):
        import re

        match = re.search(r"^Kind:\s*([a-z_]+)\s*$", card, re.M | re.I)
        return match.group(1) if match else ""

    def test_an_execution_failure_says_so(self):
        rca = translate_failure(kind="execution", exception_text="Timeout expired.")
        card = format_failure_business_response(rca=rca, sql="SELECT 1")
        self.assertEqual(self._kind(card), "execution")

    def test_a_validation_failure_says_so(self):
        rca = translate_failure(kind="validation", code="unknown_column")
        card = format_failure_business_response(rca=rca, sql="SELECT 1")
        self.assertEqual(self._kind(card), "validation")

    def test_a_zero_row_card_says_so(self):
        card = format_zero_row_business_response(
            confidence={"label": "X", "score": 50, "warnings": [], "reasons": []},
            rca={}, sql="SELECT 1", sql_preview_fn=lambda s: s)
        self.assertEqual(self._kind(card), "empty")

    def test_the_kind_is_stated_in_french_too(self):
        # It is a machine token: it must NOT be translated, or the portal's
        # match fails and the kicker falls back to "no rows".
        rca = _under("fr", translate_failure, kind="execution",
                     exception_text="Timeout expired.")
        card = _under("fr", format_failure_business_response, rca=rca, sql="SELECT 1")
        self.assertEqual(self._kind(card), "execution")

    def test_the_portal_reads_the_kind_rather_than_the_headline(self):
        page = (Path(__file__).resolve().parents[1]
                / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
        self.assertIn("/^Kind:", page)
        block = page[page.index("const kindMatch"):]
        block = block[:block.index("const kicker =")]
        self.assertIn("'execution'", block)
        self.assertIn("'validation'", block)
        self.assertIn("'empty'", block)

    def test_the_marker_does_not_leak_into_a_rendered_section(self):
        # It sits between the headline and Confidence:, and Confidence's
        # extraction stops at it.
        card = format_zero_row_business_response(
            confidence={"label": "X", "score": 50, "warnings": [], "reasons": []},
            rca={}, sql="SELECT 1", sql_preview_fn=lambda s: s)
        confidence_section = card.split("Confidence:")[1].split("Most likely reason:")[0]
        self.assertNotIn("Kind:", confidence_section)
        reason_section = card.split("Most likely reason:")[1].split("Suggested next step:")[0]
        self.assertNotIn("Kind:", reason_section)


if __name__ == "__main__":
    unittest.main()
