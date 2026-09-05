"""
tests/test_date_format_language.py

How a date is written, on both sides of the wire.

``strftime("%B")`` reads the process C locale — English on every server this
runs on, and not a per-request thing in any case. ``toLocaleDateString('en-US')``
is English by name. ``toLocaleTimeString([])`` follows the BROWSER's locale,
which is a third answer again: a French reader on an English-locale machine
got "2:30 PM" beside French prose.

The rule is needed twice for the same reason numbers are: the server writes a
date into the answer's prose and the browser writes the same date into the
table cell under it. So core/i18n.py owns the month names and the styles, and
the JavaScript is EXECUTED here and compared to Python's answer.

Two things this file pins that are not about French:

  * The display STYLE is never translated. dd-mm-yyyy versus mm-dd-yyyy is a
    per-column choice an administrator made about their own data; flipping it
    by language would silently change which number is the day — the one date
    bug nobody spots from the screen.
  * ISO dates stay ISO. The coverage caveats quote the date a source's data
    runs through, and "2026-07-20" is unambiguous in every language, where
    "20/07/2026" and "07/20/2026" are the same characters meaning different
    days.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from core import i18n
from core.response_builder import _format_display_value

dukpy = pytest.importorskip(
    "dukpy",
    reason="a JavaScript engine is required to EXECUTE the browser's copy of "
           "the date rule; comparing the two by reading is the drift this "
           "file exists to catch",
)

from tests.browser_num import preamble  # noqa: E402

STYLES = ("iso", "month_year_short", "month_year_long", "day_month_year",
          "month_day_year", "day_month_name_year", "day_month_short_year",
          "year", "month_name")
DATES = [date(2025, 1, 5), date(2025, 9, 3), date(2025, 12, 31),
         date(2026, 6, 1), date(1999, 10, 12)]


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


def _js(lang, expression):
    return dukpy.evaljs(preamble(lang) + "\n" + expression)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Month names
# ══════════════════════════════════════════════════════════════════════════════

class TestMonthNames:

    def test_english_is_what_strftime_produced(self):
        with _InLanguage("en"):
            assert i18n.month_name(1) == "January"
            assert i18n.month_name(9, short=True) == "Sep"
            assert i18n.month_name(12) == "December"

    def test_french_months_are_lower_case(self):
        """A proper-noun rule, not a typo: French does not capitalise month
        names."""
        with _InLanguage("fr"):
            assert i18n.month_name(1) == "janvier"
            assert i18n.month_name(9) == "septembre"

    def test_french_abbreviations_keep_their_full_stop(self):
        with _InLanguage("fr"):
            assert i18n.month_name(1, short=True) == "janv."
            assert i18n.month_name(9, short=True) == "sept."
            # Already short enough to need none.
            assert i18n.month_name(3, short=True) == "mars"
            assert i18n.month_name(5, short=True) == "mai"

    def test_an_impossible_month_comes_back_as_it_arrived(self):
        with _InLanguage("fr"):
            assert i18n.month_name(0) == "0"
            assert i18n.month_name(13) == "13"
            assert i18n.month_name("x") == "x"

    def test_every_month_exists_in_every_language(self):
        for lang in i18n.SUPPORTED_LANGUAGES:
            for index in range(1, 13):
                for kind in ("long", "short"):
                    msg_id = f"date.month.{kind}.{index}"
                    assert i18n.lookup(msg_id, lang=lang) != msg_id, msg_id


# ══════════════════════════════════════════════════════════════════════════════
# 2. The styles
# ══════════════════════════════════════════════════════════════════════════════

class TestTheStyles:

    def test_the_english_output_is_what_strftime_produced(self):
        d = date(2025, 9, 3)
        with _InLanguage("en"):
            assert i18n.format_date(d, "iso") == "2025-09"
            assert i18n.format_date(d, "month_year_short") == "Sep-25"
            assert i18n.format_date(d, "month_year_long") == "September 2025"
            assert i18n.format_date(d, "day_month_year") == "03-09-2025"
            assert i18n.format_date(d, "month_day_year") == "09-03-2025"
            assert i18n.format_date(d, "day_month_name_year") == "03-Sep-2025"
            assert i18n.format_date(d, "day_month_short_year") == "03 Sep 2025"
            assert i18n.format_date(d, "year") == "2025"
            assert i18n.format_date(d, "month_name") == "September"

    def test_french_changes_the_name_and_nothing_else(self):
        d = date(2025, 9, 3)
        with _InLanguage("fr"):
            assert i18n.format_date(d, "month_year_long") == "septembre 2025"
            assert i18n.format_date(d, "day_month_name_year") == "03-sept.-2025"
            assert i18n.format_date(d, "month_name") == "septembre"

    def test_the_numeric_styles_are_identical_in_both(self):
        """dd-mm-yyyy is a per-column decision, not a language one. Swapping
        it by language would change which number is the day."""
        for d in DATES:
            for style in ("iso", "day_month_year", "month_day_year", "year"):
                with _InLanguage("en"):
                    english = i18n.format_date(d, style)
                with _InLanguage("fr"):
                    french = i18n.format_date(d, style)
                assert english == french, (d, style)

    def test_an_unknown_style_falls_back_to_iso(self):
        with _InLanguage("fr"):
            assert i18n.format_date(date(2025, 9, 3), "teleport") == "2025-09"

    def test_a_non_date_comes_back_as_it_arrived(self):
        with _InLanguage("fr"):
            assert i18n.format_date("2025-09-03") == "2025-09-03"
            assert i18n.format_date(None) == "None"


class TestTheServersDisplayFormatter:
    """_format_display_value is what writes a date into a table cell and into
    the answer's own sentences."""

    def test_english_is_unchanged(self):
        assert _format_display_value(
            "2025-09-03", "date", {"type": "date", "style": "month_year_long"}
        ) == "September 2025"

    def test_french_names_the_month_in_french(self):
        with _InLanguage("fr"):
            assert _format_display_value(
                "2025-09-03", "date", {"type": "date", "style": "month_year_long"}
            ) == "septembre 2025"

    def test_a_value_that_is_not_a_date_is_left_to_the_number_formatter(self):
        with _InLanguage("fr"):
            assert _format_display_value(1234.5, "number") == "1\u202f234,50"


# ══════════════════════════════════════════════════════════════════════════════
# 3. The browser, executed and compared to Python
# ══════════════════════════════════════════════════════════════════════════════

class TestTheBrowserAgreesWithTheServer:

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_every_style_and_date_matches(self, lang):
        cases = [[d.year, d.month, d.day, style] for d in DATES for style in STYLES]
        got = json.loads(_js(
            lang,
            f"JSON.stringify({json.dumps(cases)}"
            ".map(c => window.qbDate(c[0], c[1], c[2], c[3])));"))
        with _InLanguage(lang):
            expected = [i18n.format_date(d, style) for d in DATES for style in STYLES]
        assert got == expected

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_month_names_match(self, lang):
        got = json.loads(_js(
            lang,
            "JSON.stringify([].concat("
            "[1,2,3,4,5,6,7,8,9,10,11,12].map(m => window.qbMonth(m, false)),"
            "[1,2,3,4,5,6,7,8,9,10,11,12].map(m => window.qbMonth(m, true))));"))
        with _InLanguage(lang):
            expected = ([i18n.month_name(m) for m in range(1, 13)]
                        + [i18n.month_name(m, short=True) for m in range(1, 13)])
        assert got == expected

    def test_an_unknown_style_falls_back_to_iso_in_the_browser_too(self):
        assert _js("fr", "window.qbDate(2025, 9, 3, 'teleport');") == "2025-09"


class TestThePagesCellFormatterAgreesWithTheServers:
    """One step further out than window.qbDate: the actual function that fills
    a table cell, lifted from portal_chat.html and executed, compared to the
    server function that fills the same column in the answer's prose."""

    CASES = [("2025-09-03", "month_year_long"), ("2025-09-03", "month_year_short"),
             ("2025-01-05", "day_month_name_year"), ("2025-01-05", "day_month_year"),
             ("2025-12-31", "month_day_year"), ("2026-06-01", "month_name"),
             ("2026-06-01", "year"), ("2025-09-03", "iso")]

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_every_style_renders_the_same_on_both_sides(self, lang):
        from tests.chat_js import run

        cases = [[raw, style] for raw, style in self.CASES]
        got = run(
            "JSON.stringify(" + json.dumps(cases) + ".map(c => "
            "_formatDisplayValue(c[0], 'date', {type:'date', style:c[1]}, 'when')));",
            lang=lang,
            functions=[
                "function _formatDisplayValue(value, fmt, spec = {}, column = '')",
                "function _parseDisplayDate(value)",
                "function _parseDisplayNumber(v)",
                "function _isPeriodLabel(raw, column)",
            ],
        )
        with _InLanguage(lang):
            expected = [
                _format_display_value(raw, "date", {"type": "date", "style": style})
                for raw, style in self.CASES
            ]
        assert got == expected


class TestTheClock:
    """The page read the clock convention from the BROWSER's locale, which is
    neither the reader's choice nor the server's."""

    def _time(self, lang, hour, minute):
        return _js(lang, f"window.qbTime(new Date(2025, 8, 3, {hour}, {minute}));")

    def test_english_is_twelve_hour(self):
        assert self._time("en", 14, 30) == "02:30 PM"
        assert self._time("en", 9, 5) == "09:05 AM"
        assert self._time("en", 0, 0) == "12:00 AM"
        assert self._time("en", 12, 0) == "12:00 PM"

    def test_french_is_twenty_four_hour(self):
        assert self._time("fr", 14, 30) == "14:30"
        assert self._time("fr", 9, 5) == "09:05"
        assert self._time("fr", 0, 0) == "00:00"

    def test_an_unusable_date_is_empty_not_a_crash(self):
        assert _js("fr", "window.qbTime(new Date('nope'));") == ""
        assert _js("fr", "String(window.qbTime(null));") == ""


class TestTheFreshnessBanner:
    """The one place a date is written into a sentence rather than a cell. It
    carried an English month, an English inline plural and an English word for
    the window all at once."""

    def _banner(self, lang, drift_days, **kw):
        """The real function, extracted from _handle_query_impl so it can be
        run on its own: it was six levels deep inside a 6,800-line coroutine,
        which is where a plural rule goes unnoticed."""
        from core.query_pipeline import business_date_banner

        anchor = {"value": date(2025, 9, 3).isoformat(), **kw}
        with _InLanguage(lang):
            return business_date_banner(
                "this_month", anchor,
                today=date(2025, 9, 3 + drift_days),
            )[1]

    def test_the_english_is_unchanged(self):
        assert self._banner("en", 5) == (
            "ℹ️ The most recent business data is **03 Sep 2025** (5 days ago), "
            'so "this month" is answered as of that date rather than the '
            "calendar date. _(read just now; if your data has just been "
            "reloaded, ask an administrator to refresh the business date.)_"
        )

    def test_the_french_names_the_month_and_counts_in_french(self):
        french = self._banner("fr", 5)
        assert "**03 sept. 2025**" in french
        assert "il y a 5 jours" in french
        assert "« ce mois-ci »" in french
        assert "Sep 2025" not in french

    def test_two_days_is_plural_in_both(self):
        assert "(2 days ago)" in self._banner("en", 2)
        assert "il y a 2 jours" in self._banner("fr", 2)

    def test_the_singular_form_exists_even_though_the_threshold_hides_it(self):
        """The banner only fires above one day of drift, so the singular is
        not reachable through it today. It is a counted sentence and keeps its
        pair: a threshold moved to >= 1 must not produce "1 days ago"."""
        for lang, expected in [("en", "(1 day ago)"), ("fr", "il y a 1 jour")]:
            with _InLanguage(lang):
                said = i18n.plural("date.anchor.drift", 1, date="x", window="y",
                                   source="z")
            assert expected in said
            assert "1 days" not in said
            assert "1 jours" not in said

    def test_a_cached_anchor_says_so(self):
        """A cached value stated as bare fact is how a stale answer passes for
        a current one."""
        assert "read from cache" in self._banner("en", 5, cached=True)
        assert "lu depuis le cache" in self._banner("fr", 5, cached=True)

    def test_a_fresh_window_says_nothing(self):
        """One day of drift is not news; the banner is for staleness."""
        assert self._banner("en", 1) == ""
        assert self._banner("fr", 1) == ""

    def test_no_anchor_at_all_still_discloses_the_relativity(self):
        from core.query_pipeline import business_date_banner

        for lang, marker in [("en", "most recent business date"),
                             ("fr", "date métier la plus récente")]:
            with _InLanguage(lang):
                drift, message = business_date_banner("today", {})
            assert drift is None
            assert marker in message

    def test_an_unparseable_anchor_degrades_instead_of_raising(self):
        """It used to raise ValueError into the surrounding handler, which
        skipped the whole banner -- on exactly the reader who most needed to
        be told the answer was data-relative."""
        from core.query_pipeline import business_date_banner

        with _InLanguage("fr"):
            drift, message = business_date_banner("today", {"value": "not-a-date"})
        assert drift is None
        assert "date métier la plus récente" in message

    def test_every_window_the_pipeline_gates_on_has_a_name(self):
        """The pipeline decides which kinds get a banner. A seventh added
        later without a catalogue entry would render as
        "date.window.last_month" in the message."""
        from core.query_pipeline import BUSINESS_DATE_WINDOW_KINDS

        kinds = sorted(BUSINESS_DATE_WINDOW_KINDS)
        assert len(kinds) >= 6, kinds
        for kind in kinds:
            for lang in i18n.SUPPORTED_LANGUAGES:
                msg_id = f"date.window.{kind}"
                assert i18n.lookup(msg_id, lang=lang) != msg_id, f"{msg_id} [{lang}]"


# ══════════════════════════════════════════════════════════════════════════════
# 4. What stays ISO
# ══════════════════════════════════════════════════════════════════════════════

class TestIsoDatesStayIso:
    """"2026-07-20" is unambiguous in every language. "20/07/2026" and
    "07/20/2026" are the same characters meaning different days, and a reader
    who copies one out of a caveat into a filter has no way to tell."""

    def test_the_coverage_caveat_quotes_the_date_verbatim(self):
        from unittest.mock import patch

        from core.date_coverage import check_date_coverage

        policy = {"amount": 7, "unit": "day", "fact_table": "DBO.F_ORDERS",
                  "fact_column": "ORDER_DATE", "date_key_type": "native"}
        with patch("core.contextual_dates.format_required_anchor",
                   return_value="(SELECT MAX(ORDER_DATE) FROM DBO.F_ORDERS)"), \
             patch("core.date_coverage.run_query", side_effect=[
                 [{"AnchorDate": "2026-07-20"}], [{"DaysWithData": 3}]]), \
             _InLanguage("fr"):
            gap = check_date_coverage(
                {"db_type": "azure_sql", "credentials": {}}, policy, "azure_sql")
        assert "2026-07-20" in gap.message
        assert "20/07/2026" not in gap.message

    def test_a_period_label_is_a_name_not_a_date(self):
        """core/result_renderer.py keeps 2025-03 as 2025-03. A bucket label is
        an identity, and the table beside it is keyed on the same string."""
        with _InLanguage("fr"):
            assert i18n.format_date(date(2025, 3, 1), "iso") == "2025-03"
