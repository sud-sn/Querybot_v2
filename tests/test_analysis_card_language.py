"""
tests/test_analysis_card_language.py

The analysis card, in both languages, executed rather than read.

build_analysis_response is the zero-latency fallback behind every action chip
under an answer -- Explain, Detail, Compare, Why, Predict, Decide. It runs
whenever the LLM path is unavailable and unconditionally for a regulated
tenant, so for some clients it is the only analysis they ever see. Every
sentence in it was concatenated in English.

One of those concatenations was already wrong in English. The card wrote

    f"{pct:.1f}% {direction} than the starting period"

with `direction` taken from a three-way "higher"/"lower"/"flat" -- so a series
that did not move produced "0.0% flat than the starting period". It went
unnoticed for as long as nobody had to translate it.

The scope phrase is the other one. The card wrote `scope["badge"].lower()` into
the middle of a sentence: lowercasing an English badge happens to read, and in
French an inline noun phrase needs its article, which carries the gender. That
is why infer_result_scope publishes a separate `inline` form.

Every test here calls the real function and asserts on what it returns.
"""

import re
import unittest

from core import i18n
from core.response_builder import (
    build_analysis_response,
    infer_result_scope,
    summarize_result_context,
    _regulated_analysis_fallback,
    _why_it_matters,
)

ACTIONS = ("explain", "analyze", "compare", "why", "predict", "decide")

# A catalogue id that was never translated comes back from lookup() as the id
# itself -- "analysis.explain.series" -- which reads as a broken template on
# the card rather than raising anywhere.
LOOKS_LIKE_AN_ID = re.compile(r"\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,}\b")


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


def _context(kind):
    """A real analysis contract, built by the real summariser."""
    if kind == "ranking":
        rows = [{"nationality": "UAE", "employee_count": 305},
                {"nationality": "Indian", "employee_count": 187},
                {"nationality": "Egyptian", "employee_count": 91},
                {"nationality": "Jordanian", "employee_count": 45}]
        question = "which nationality employees are higher in count"
        sql = ("SELECT nationality, COUNT(*) AS employee_count FROM employees "
               "GROUP BY nationality ORDER BY employee_count DESC")
    elif kind == "time_series":
        rows = [{"month": f"2025-{m:02d}", "revenue": 1000 + m * 90}
                for m in range(1, 9)]
        question = "show revenue by month"
        sql = ("SELECT month, SUM(amount) AS revenue FROM sales "
               "GROUP BY month ORDER BY month")
    elif kind == "flat_series":
        rows = [{"month": f"2025-{m:02d}", "revenue": 1000} for m in range(1, 9)]
        question = "show revenue by month"
        sql = ("SELECT month, SUM(amount) AS revenue FROM sales "
               "GROUP BY month ORDER BY month")
    elif kind == "numeric_table":
        rows = [{"invoice_id": i, "amount": i * 13.5, "tax": i * 1.1}
                for i in range(1, 7)]
        question = "list invoice amounts"
        sql = "SELECT invoice_id, amount, tax FROM invoices"
    elif kind == "top_one":
        rows = [{"nationality": "UAE", "employee_count": 305}]
        question = "which nationality employees are higher in count"
        sql = ("SELECT TOP 1 nationality, COUNT(*) AS employee_count FROM "
               "employees GROUP BY nationality ORDER BY employee_count DESC")
    elif kind == "empty":
        rows, question, sql = [], "list invoices", "SELECT * FROM invoices WHERE 1=0"
    else:
        raise AssertionError(kind)
    return summarize_result_context(rows, question, sql)


KINDS = ("ranking", "time_series", "flat_series", "numeric_table", "top_one", "empty")


def _text(card):
    """Everything on the card a reader actually sees."""
    return " ".join([card.get("title") or "", card.get("body") or "",
                     card.get("secondary") or "", card.get("next_step") or "",
                     *(card.get("bullets") or [])])


class TheWholeCardIsTranslated(unittest.TestCase):
    """Six actions across six result shapes: 36 cards, each rendered twice."""

    def _assert_every_field_translated(self, action, make_contract, label):
        # The contract is rebuilt inside each language, because
        # summarize_result_context translates the scope note as it builds it --
        # reusing one contract would test the card against a scope already
        # frozen in English, which is the same mistake as answering a French
        # reader out of a cached display_context.
        with _InLanguage("en"):
            english = build_analysis_response(action, make_contract())
        with _InLanguage("fr"):
            french = build_analysis_response(action, make_contract())
        self.assertTrue(english["title"] and english["body"], label)
        self.assertTrue(french["title"] and french["body"], label)
        # Field by field, not the concatenation: one hardcoded title among a
        # dozen translated sentences still leaves the two cards different, and
        # a whole-card comparison would call that translated.
        for field in ("title", "body", "secondary", "next_step"):
            if english[field]:
                self.assertNotEqual(english[field], french[field],
                                    f"{label}: {field} is still English")
        self.assertEqual(len(english["bullets"]), len(french["bullets"]), label)
        for i, (en, fr) in enumerate(zip(english["bullets"], french["bullets"])):
            self.assertNotEqual(en, fr, f"{label}: bullet {i} is still English")

    def test_every_action_and_shape_renders_in_both_languages(self):
        for kind in KINDS:
            for action in ACTIONS:
                with self.subTest(kind=kind, action=action):
                    self._assert_every_field_translated(
                        action, lambda k=kind: _context(k), f"{action}/{kind}")

    # The branches the six real result shapes above do not reach. A contract is
    # a plain dict, so these drive the real function down the remaining paths
    # rather than leaving them as the place an English sentence can hide.
    REMAINING_BRANCHES = [
        ("explain", {"mode": "table"}),
        ("explain", {"mode": "numeric_table", "row_count": 1,
                     "min_value": 1.0, "max_value": 1.0}),
        ("analyze", {"mode": "table"}),
        ("analyze", {"mode": "ranking",
                     "distribution_stats": {"category_count": 4, "spread": 10.0}}),
        ("analyze", {"mode": "ranking", "distribution_stats": {
            "top_3_share_pct": 91.2, "category_count": 4, "spread": 10.0,
            "std_dev": 3.4}}),
        ("compare", {"mode": "table"}),
        ("compare", {"mode": "ranking", "comparison_stats": {}}),
        ("compare", {"mode": "ranking", "comparison_stats": {"leader": "UAE"}}),
        ("compare", {"mode": "ranking", "comparison_stats": {
            "leader": "UAE", "runner_up": "Indian", "gap": 118.0,
            "leader_share_pct": 48.5}}),
        ("compare", {"mode": "time_series", "comparison_stats": {
            "last_period": "2025-08", "last_value": 1720.0,
            "first_value": 1090.0, "first_period": "2025-01",
            "pct_change": 57.8}}),
        ("why", {"mode": "time_series"}),
        ("why", {"mode": "time_series", "pct_change": -12.5}),
        ("why", {"mode": "ranking", "top_items": [{"label": "UAE", "value": 305}]}),
        ("why", {"mode": "empty"}),
        ("why", {"mode": "table"}),
        ("predict", {"mode": "table"}),
        ("predict", {"mode": "time_series", "row_count": 2}),
        ("decide", {"mode": "table"}),
    ]

    def test_every_remaining_branch_renders_in_both_languages(self):
        for action, contract in self.REMAINING_BRANCHES:
            label = f"{action}/{contract.get('mode')}/{sorted(contract)}"
            with self.subTest(branch=label):
                self._assert_every_field_translated(
                    action, lambda c=contract: dict(c), label)

    def test_no_card_leaks_an_untranslated_catalogue_id(self):
        """lookup() returns the id when an entry is missing, so a typo shows up
        on the card as "analysis.explain.series" rather than raising."""
        for lang in ("en", "fr"):
            for kind in KINDS:
                for action in ACTIONS:
                    with self.subTest(lang=lang, kind=kind, action=action):
                        with _InLanguage(lang):
                            card = build_analysis_response(action, _context(kind))
                        leaked = LOOKS_LIKE_AN_ID.findall(_text(card))
                        self.assertEqual(leaked, [], f"unresolved ids: {leaked}")

    def test_an_unsupported_action_still_says_something_readable(self):
        with _InLanguage("en"):
            english = build_analysis_response("teleport", _context("ranking"))
        with _InLanguage("fr"):
            french = build_analysis_response("teleport", _context("ranking"))
        self.assertTrue(french["title"] and french["body"])
        self.assertNotIn("analysis.", french["body"])
        self.assertNotEqual(english["body"], french["body"])

    def test_the_action_key_and_type_are_wire_tokens(self):
        """The browser routes on these. Translating them would break the card
        before anyone got to read it."""
        with _InLanguage("fr"):
            card = build_analysis_response("compare", _context("ranking"))
        self.assertEqual(card["action"], "compare")
        self.assertEqual(card["type"], "assistant_analysis")
        self.assertEqual(card["mode"], "ranking")


class TheScopePhraseReadsMidSentence(unittest.TestCase):
    """`scope["badge"].lower()` is an English trick. "Top result only" ->
    "top result only" happens to read; "Ligne la mieux classée uniquement" ->
    "ligne la mieux classée uniquement" does not, because French needs the
    article and the article carries the gender."""

    # Every badge a real result can produce, and rows/sql that produce it.
    SCOPES = {
        "returned": ([{"a": 1}], "SELECT a FROM t", "table"),
        "top_one": ([{"a": 1}], "SELECT TOP 1 a FROM t ORDER BY a DESC", "ranking"),
        "top_n": ([{"a": i} for i in range(3)],
                  "SELECT TOP 3 a FROM t ORDER BY a DESC", "ranking"),
        "full_distribution": ([{"a": i} for i in range(3)],
                              "SELECT a FROM t GROUP BY a", "ranking"),
        "full_series": ([{"a": i} for i in range(3)],
                        "SELECT a FROM t GROUP BY a", "time_series"),
        "preview": ([{"a": i} for i in range(500)], "SELECT a FROM t", "table"),
        "filtered_subset": ([{"a": 1}], "SELECT a FROM t WHERE a > 0", "table"),
    }

    def _scope(self, key, lang):
        rows, sql, mode = self.SCOPES[key]
        with _InLanguage(lang):
            return infer_result_scope(rows, "a question", sql, mode=mode)

    def test_every_badge_has_an_inline_form_in_both_languages(self):
        for key in self.SCOPES:
            for lang in ("en", "fr"):
                with self.subTest(badge=key, lang=lang):
                    scope = self._scope(key, lang)
                    self.assertEqual(scope["badge_key"], key)
                    inline = scope["inline"]
                    self.assertTrue(inline)
                    self.assertNotIn("answer.scope", inline)
                    # A noun phrase, not a badge: badges are title-case labels.
                    self.assertEqual(inline, inline.lstrip())

    def test_the_french_inline_form_is_not_the_english_one(self):
        for key in self.SCOPES:
            with self.subTest(badge=key):
                self.assertNotEqual(self._scope(key, "en")["inline"],
                                    self._scope(key, "fr")["inline"])

    def test_the_badge_key_is_a_wire_token_and_the_badge_is_not(self):
        """core/insight.py's narration prompt reads badge_key. If the French
        label reached it, a French portal would quietly change what the model
        is asked."""
        english = self._scope("top_one", "en")
        french = self._scope("top_one", "fr")
        self.assertEqual(english["badge_key"], french["badge_key"], "top_one")
        for field in ("badge", "note", "inline"):
            self.assertNotEqual(english[field], french[field], field)

    def test_the_card_puts_the_inline_form_in_its_sentence(self):
        """The end-to-end version: the phrase the scope publishes is the phrase
        the reader finds in the body."""
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                with _InLanguage(lang):
                    ctx = _context("top_one")
                    card = build_analysis_response("explain", ctx)
                self.assertIn(ctx["result_scope"]["inline"], card["body"])

    def test_the_english_says_the_result_is_one_row(self):
        with _InLanguage("en"):
            card = build_analysis_response("explain", _context("top_one"))
        self.assertIn("the top-ranked row only", card["body"])

    def test_the_french_says_the_result_is_one_row(self):
        with _InLanguage("fr"):
            card = build_analysis_response("explain", _context("top_one"))
        self.assertIn("uniquement la ligne la mieux classée", card["body"])


class TheFlatSeriesIsNotAComparative(unittest.TestCase):
    """The English defect this rewrite removes.

    _why_it_matters built "{pct}% {direction} than the starting period" from a
    three-way direction. "flat" is not a comparative, so a series that did not
    move read "0.0% flat than the starting period"."""

    def test_a_flat_series_does_not_read_as_a_comparison(self):
        with _InLanguage("en"):
            line = _why_it_matters(_context("flat_series"))
        self.assertNotIn("flat than", line)
        self.assertNotIn("0.0% flat", line)
        self.assertTrue(line.endswith("."), line)

    def test_a_moving_series_still_states_the_direction(self):
        with _InLanguage("en"):
            line = _why_it_matters(_context("time_series"))
        self.assertIn("57.8%", line)
        self.assertIn("higher", line)

    def test_all_three_directions_are_distinct_sentences_in_french(self):
        lines = set()
        with _InLanguage("fr"):
            for pct in (12.5, -12.5, 0.0):
                lines.add(_why_it_matters({"mode": "time_series", "pct_change": pct}))
        self.assertEqual(len(lines), 3, lines)
        for line in lines:
            self.assertNotIn("than", line)
            self.assertNotIn("analysis.why", line)

    def test_the_why_card_carries_that_sentence_in_both_languages(self):
        with _InLanguage("en"):
            english = build_analysis_response("why", _context("flat_series"))
        with _InLanguage("fr"):
            french = build_analysis_response("why", _context("flat_series"))
        self.assertNotIn("flat than", english["body"])
        self.assertTrue(french["body"])
        self.assertNotEqual(english["body"], french["body"])


class TheRegulatedFallbackIsTranslated(unittest.TestCase):
    """A regulated tenant never reaches the LLM path, so this static card is
    the whole of their analysis."""

    def test_it_speaks_the_readers_language(self):
        with _InLanguage("en"):
            english = _regulated_analysis_fallback("explain")
        with _InLanguage("fr"):
            french = _regulated_analysis_fallback("explain")
        self.assertTrue(french["title"] and french["body"])
        self.assertNotEqual(english["title"], french["title"])
        self.assertNotEqual(english["body"], french["body"])
        self.assertEqual(french["action"], "explain")
        self.assertEqual(french["type"], "assistant_analysis")
        self.assertNotIn("analysis.", french["body"])


class TheLanguageIsScopedToTheRequest(unittest.TestCase):
    def test_a_french_card_does_not_change_the_next_english_one(self):
        with _InLanguage("fr"):
            build_analysis_response("explain", _context("ranking"))
        card = build_analysis_response("explain", _context("ranking"))
        self.assertIn("ranks first", card["body"].lower())


if __name__ == "__main__":
    unittest.main()
