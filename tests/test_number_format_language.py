"""
tests/test_number_format_language.py

How a number is written, on both sides of the wire.

"1,234.56" is English. A French reader reads that comma as the decimal point,
so the same digits are off by a factor of a thousand with nothing on screen to
say so — the quietest possible wrong answer in a data product.

The rule lives in core/i18n.py because it is needed twice: the server writes
numbers into prose (the answer headline, the analysis card, the insight
summary) and the browser writes the same numbers into the table cells under
them. Two implementations of one rule is the shape that drifts, so the
JavaScript is EXECUTED here and compared to Python's answer for every case.

Three things this file pins that are not about French:

  * The page had THREE number formats live at once — toLocaleString('en-US'),
    toLocaleString(undefined) (the browser's own locale, not the reader's),
    and hand-built magnitude suffixes. On a French browser the chart tooltip
    already disagreed with the table beside it.
  * The table's sort comparator and its column aggregate both read back the
    text the cell formatter wrote. A formatter that groups with a space needs
    a parser that knows it, or "1 234,56" coerces to 123456 and the column
    sorts and totals wrong.
  * The CSV export is deliberately NOT localised. A decimal comma inside a
    comma-delimited file is ambiguous to every downstream parser.
"""

from __future__ import annotations

import json

import pytest

from core import i18n
from core.export import _format_cell
from core.response_builder import _format_number

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the browser's copy of "
           "the number rule; comparing the two by reading is the drift this "
           "file exists to catch",
)

from tests.browser_num import preamble  # noqa: E402

VALUES = [0, 1, 12, 999, 1000, 1234, 1234.5, 1234567.891, -1234.5, -0.5,
          0.5, 1000000, 999.999, 2.25, 1e9]


def _js(lang, expression):
    return dukpy.evaljs(preamble(lang) + "\n" + expression)


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. The contract itself
# ══════════════════════════════════════════════════════════════════════════════

class TestTheSeparators:

    def test_english_groups_with_a_comma_and_points_with_a_dot(self):
        spec = i18n.number_format("en")
        assert spec["group"] == ","
        assert spec["decimal"] == "."

    def test_french_groups_with_a_narrow_no_break_space(self):
        """U+202F, not a plain space and not U+00A0: a plain space wraps at
        the end of a line and splits the number across two."""
        spec = i18n.number_format("fr")
        assert spec["group"] == " "
        assert spec["decimal"] == ","

    def test_the_gap_before_a_sign_is_a_different_character(self):
        """French puts a NO-BREAK space before % and before a currency symbol,
        and the NARROW one between groups. Using one for both makes
        "12 345 %" look like a grouped number."""
        spec = i18n.number_format("fr")
        assert spec["percent_gap"] == " "
        assert spec["currency_gap"] == " "
        assert spec["group"] != spec["percent_gap"]

    def test_an_unknown_language_falls_back_to_english(self):
        assert i18n.number_format("klingon") == i18n.number_format("en")


class TestFormatDecimal:

    def test_english_is_unchanged(self):
        with _InLanguage("en"):
            assert i18n.format_decimal(1234567.891) == "1,234,567.89"
            assert i18n.format_decimal(1000) == "1,000"
            assert i18n.format_decimal(0.5) == "0.50"
            assert i18n.format_decimal(-1234.5) == "-1,234.50"

    def test_french_groups_and_points_its_own_way(self):
        with _InLanguage("fr"):
            assert i18n.format_decimal(1234567.891) == "1 234 567,89"
            assert i18n.format_decimal(1000) == "1 000"
            assert i18n.format_decimal(0.5) == "0,50"
            assert i18n.format_decimal(-1234.5) == "-1 234,50"

    def test_the_two_swaps_run_in_the_right_order(self):
        """Group first, decimal second. The other order swaps "." to a comma
        and then cannot tell the new commas from the group ones, so
        "1,234.56" comes out as "1 234 56"."""
        with _InLanguage("fr"):
            assert i18n.format_decimal(1234.56, 2) == "1\u202f234,56"
            assert i18n.format_decimal(1234567.89, 2) == "1\u202f234\u202f567,89"

    def test_grouping_can_be_turned_off_in_both(self):
        with _InLanguage("en"):
            assert i18n.format_decimal(1234.5, 2, grouping=False) == "1234.50"
        with _InLanguage("fr"):
            assert i18n.format_decimal(1234.5, 2, grouping=False) == "1234,50"

    def test_a_non_number_comes_back_as_it_arrived(self):
        with _InLanguage("fr"):
            assert i18n.format_decimal("abc") == "abc"
            assert i18n.format_decimal(None) == "None"
            assert i18n.format_decimal(float("inf")) == "inf"

    def test_format_count_is_format_decimal_with_no_places(self):
        with _InLanguage("fr"):
            assert i18n.format_count(1234567) == "1 234 567"
            assert i18n.format_count("nope") == "nope"


# ══════════════════════════════════════════════════════════════════════════════
# 2. The server's prose
# ══════════════════════════════════════════════════════════════════════════════

class TestTheServerSide:

    def test_plain_numbers(self):
        with _InLanguage("fr"):
            assert _format_number(1234567.891) == "1 234 567,89"
            assert _format_number(12) == "12"

    def test_currency_moves_the_symbol_in_french(self):
        """"$1 234,50" is not French. The symbol goes after the amount with a
        no-break space before it."""
        assert _format_number(1234.5, "currency") == "$1,234.50"
        with _InLanguage("fr"):
            assert _format_number(1234.5, "currency") == "1 234,50 $"
            assert _format_number(-1234.5, "currency") == "-1 234,50 $"
            assert _format_number(
                1234.5, "currency",
                {"type": "currency", "currency_code": "EUR"}) == "1 234,50 €"

    def test_accounting_negatives_keep_their_parentheses(self):
        with _InLanguage("fr"):
            assert _format_number(-1234.5, "currency",
                                  {"type": "currency", "accounting": True}) \
                == "(1 234,50 $)"

    def test_percentages_take_the_french_gap(self):
        assert _format_number(12.3456, "percentage") == "12.35%"
        with _InLanguage("fr"):
            assert _format_number(12.3456, "percentage") == "12,35 %"

    def test_the_compact_style_trims_the_language_own_decimal(self):
        """`.rstrip(".")` would leave the zero on "1,0M" in French."""
        spec = {"type": "number", "style": "compact"}
        assert _format_number(1234567, None, spec) == "1.23M"
        assert _format_number(1000000, None, spec) == "1M"
        with _InLanguage("fr"):
            assert _format_number(1234567, None, spec) == "1,23M"
            assert _format_number(1000000, None, spec) == "1M"

    def test_the_pre_rounding_threshold_is_preserved(self):
        """999.999 takes the ungrouped branch because the >= 1000 test runs on
        the value BEFORE rounding. A translation must not quietly change
        which branch a number lands in."""
        assert _format_number(999.999) == "1000.00"
        with _InLanguage("fr"):
            assert _format_number(999.999) == "1000,00"


# ══════════════════════════════════════════════════════════════════════════════
# 3. The browser, executed and compared to Python
# ══════════════════════════════════════════════════════════════════════════════

class TestTheBrowserAgreesWithTheServer:

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_qb_num_matches_format_decimal_for_every_value(self, lang):
        got = json.loads(_js(
            lang, f"JSON.stringify({json.dumps(VALUES)}.map(v => window.qbNum(v)));"))
        with _InLanguage(lang):
            expected = [i18n.format_decimal(v) for v in VALUES]
        assert got == expected

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_qb_num_matches_at_a_fixed_precision(self, lang):
        got = json.loads(_js(
            lang,
            f"JSON.stringify({json.dumps(VALUES)}"
            ".map(v => window.qbNum(v, {min: 2, max: 2})));"))
        with _InLanguage(lang):
            expected = [i18n.format_decimal(v, 2) for v in VALUES]
        assert got == expected

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_qb_num_matches_with_grouping_off(self, lang):
        got = json.loads(_js(
            lang,
            f"JSON.stringify({json.dumps(VALUES)}"
            ".map(v => window.qbNum(v, {min: 2, max: 2, grouping: false})));"))
        with _InLanguage(lang):
            expected = [i18n.format_decimal(v, 2, grouping=False) for v in VALUES]
        assert got == expected

    def test_a_non_number_comes_back_as_it_arrived(self):
        assert _js("fr", "window.qbNum('abc');") == "abc"
        assert _js("fr", "window.qbNum(1/0);") == "Infinity"


class TestTheBrowserCanReadBackWhatItWrote:
    """The table's sort comparator and its column aggregate both parse the
    RENDERED cell text. Format then parse has to be the identity."""

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_format_then_parse_round_trips(self, lang):
        got = json.loads(_js(
            lang,
            f"JSON.stringify({json.dumps(VALUES)}"
            ".map(v => window.qbParseNum(window.qbNum(v, {min: 2, max: 2}))));"))
        assert got == pytest.approx([round(float(v), 2) for v in VALUES])

    def test_a_french_grouped_number_is_not_read_as_a_bigger_one(self):
        """The defect: stripping every comma turned "1 234,56" into 123456."""
        assert _js("fr", "window.qbParseNum('1\\u202f234,56');") == pytest.approx(1234.56)
        assert _js("fr", "window.qbParseNum('1 234,56');") == pytest.approx(1234.56)

    def test_a_raw_server_value_still_parses_in_french(self):
        """Cells arrive from data.rows unformatted, so the parser sees both."""
        assert _js("fr", "window.qbParseNum('1234.56');") == pytest.approx(1234.56)
        assert _js("fr", "window.qbParseNum(1234.56);") == pytest.approx(1234.56)

    def test_currency_and_percent_decorations_are_stripped(self):
        for lang, text in [("en", "'$1,234.56'"), ("fr", "'1\\u202f234,56\\u00a0$'"),
                           ("en", "'12.5%'"), ("fr", "'12,5\\u00a0%'")]:
            assert not isinstance(_js(lang, f"window.qbParseNum({text});"), str)

    def test_text_is_not_a_number(self):
        for lang in ("en", "fr"):
            for text in ("'Paris, France'", "''", "'—'", "'abc'"):
                got = _js(lang, f"String(window.qbParseNum({text}));")
                assert got == "NaN", (lang, text, got)


# ══════════════════════════════════════════════════════════════════════════════
# 4. What stays English
# ══════════════════════════════════════════════════════════════════════════════

class TestTheCsvExportIsNotLocalised:
    """A CSV is an interchange format. A French decimal comma inside a
    comma-delimited file is ambiguous to every downstream parser, and the
    reader who exports it may not be the one who opens it."""

    def test_currency_cells_stay_english_in_french(self):
        with _InLanguage("fr"):
            assert _format_cell(1234.5, "currency") == "$1,234.50"

    def test_percentage_cells_stay_english_in_french(self):
        with _InLanguage("fr"):
            assert _format_cell(12.5, "percentage") == "12.50%"

    def test_plain_cells_are_untouched_in_both(self):
        for lang in ("en", "fr"):
            with _InLanguage(lang):
                assert _format_cell("North", "text") == "North"
                assert _format_cell(None, "number") == ""
