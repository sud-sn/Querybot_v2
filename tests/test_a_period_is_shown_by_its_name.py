"""
A period is shown by its name, the same everywhere.

A month reached the reader as a code. The portal's table printed "2025-03",
the headline and the sentences "2025-03", the chart's axis "Mar 2025": three
spellings of one month, none of them the one a reader says. A quarter fared
worse: 2025-04-01, the first day of the second quarter, was "2025-04" in the
table and "Apr 2025" on the chart -- April. A month series with a month
missing (no sale in May) failed the cadence check that marks a column as
periods, and so did a single month asked for, so their cells read as days:
"2025-01-01". The warehouse's own keys fared worst: the browser read the
month key 202503 as the year 202503, and could not read the day key 20250305
at all, while the chat's table printed that day as its month.

Now a period is named by the grain it was asked at -- the plan's, or the
question's own words in either language -- or else by its values: a month is
"March 2025" ("mars 2025"), a quarter "Q2 2025" ("T2 2025"), a year "2025",
and a date key the day it is. The same in the portal's table, a chat's table,
the headline, the sentences and the chart; and a French series named in
French is still read as a series.

The answer card, the chat reply and the chart are driven through the real
builders; the portal's cell formatter and chart are executed as JavaScript.
"""

from __future__ import annotations

import asyncio
import json
import re
import signal
from datetime import date, datetime
from decimal import Decimal

import pytest

from core import i18n
from core.response_builder import build_assistant_response, build_column_formats, summarize_result_context


@pytest.fixture(scope="module", autouse=True)
def _a_store():
    """The card reads the workspace's compliance profile from the store."""
    import store

    store.init_db()


class _InLanguage:
    def __init__(self, lang):
        self.lang = lang

    def __enter__(self):
        self._token = i18n.activate_language(self.lang)
        return self

    def __exit__(self, *exc):
        i18n.deactivate_language(self._token)
        return False


# Revenue by month, with no sale in May.
GAPPED = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (1, 2, 3, 4, 6)]
QUARTERS = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (1, 4)]


def _card(rows, question, lang="en", chart=None):
    with _InLanguage(lang):
        return build_assistant_response(
            question=question, rows=rows, sql="SELECT 1", duration_ms=1, chart=chart)


class TestTheCard:

    @pytest.mark.parametrize("lang,question,close,first", [
        ("en", "revenue by month in 2025", "June 2025 closed at 600.", "January 2025"),
        # A sentence starts with a capital; a French month is written without
        # one.
        ("fr", "revenus par mois en 2025", "Juin 2025 a terminé à 600.", "janvier 2025"),
    ])
    def test_a_month_series_with_a_month_missing(self, lang, question, close, first):
        card = _card(GAPPED, question, lang)
        assert card["data"]["column_formats"] == {"PERIOD": "date"}
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "month_year_long"}}
        assert card["answer"]["headline"] == close
        assert first in card["insight_summary"]
        assert "2025-0" not in card["insight_summary"]

    @pytest.mark.parametrize("lang,question,close", [
        ("en", "revenue by quarter", "Q2 2025 closed at 400."),
        ("fr", "revenus par trimestre", "T2 2025 a terminé à 400."),
    ])
    def test_a_quarter_is_named_as_one(self, lang, question, close):
        chart = {"x_key": "PERIOD", "y_keys": ["REVENUE"], "chart_type": "line"}
        card = _card(QUARTERS, question, lang, chart=chart)
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "quarter"}}
        assert card["answer"]["headline"] == close
        # The chart's axis is told too, and names the quarter, not its first month.
        assert card["chart"]["x_style"] == "quarter"

    @pytest.mark.parametrize("lang,question,close", [
        ("en", "revenue by month", "April 2025 closed at 2,000."),
        ("fr", "revenus par mois", "Avril 2025 a terminé à 2\u202f000."),
    ])
    def test_a_series_of_month_keys(self, lang, question, close):
        """202501..202504, in whatever order the warehouse returned them: the
        card was headed "202504 leads at 2,000", a ranking of codes, above
        sentences that read the same rows as a trend."""
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": 500.0 * month} for month in (3, 1, 4, 2)]
        assert _card(rows, question, lang)["answer"]["headline"] == close

    def test_a_month_key_reaches_the_browser_as_the_day_its_month_starts(self):
        """Every other number is sent grouped, "1,250"; 202501 was sent
        "202,501", which the browser cannot read as a month, and the cell
        showed that number (seen in the browser, TestThePortalTable). A period
        reaches the browser as the ISO day it starts on, which the page reads
        in its own time zone."""
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": 1500.0 * month} for month in (1, 2)]
        cells = _card(rows, "revenue by month")["data"]["rows"]
        assert [cell["PRD_KEY"] for cell in cells] == ["2025-01-01", "2025-02-01"]
        assert [cell["REVENUE"] for cell in cells] == ["1,500", "3,000"]

    def test_months_ranked_by_their_revenue_are_a_ranking(self):
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": 500.0 * month} for month in (4, 3, 2, 1)]
        with _InLanguage("en"):
            card = build_assistant_response(
                question="which month had the highest revenue", rows=rows, sql="SELECT 1", duration_ms=1,
                display_context={"result_operation": "sort"})
        assert card["answer"]["headline"] == "April 2025 leads at 2,000."

    def test_a_single_month_asked_for(self):
        rows = [{"PERIOD": date(2026, 3, 1), "STOCK_ON_HAND": 1250.0}]
        card = _card(rows, "stock on hand by month")
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "month_year_long"}}

    def test_a_single_date_asked_for_by_no_grain_is_left_as_it_is(self):
        rows = [{"PERIOD": date(2026, 3, 1), "STOCK_ON_HAND": 1250.0}]
        assert _card(rows, "stock on hand")["data"]["display_formats"] == {}

    def test_what_the_reader_asked_for_wins(self):
        with _InLanguage("en"):
            card = build_assistant_response(
                question="revenue by month", rows=GAPPED, sql="SELECT 1", duration_ms=1,
                display_formats={"PERIOD": {"type": "date", "style": "month_year_short"}})
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "month_year_short"}}


class _Text:
    """A chat with no card: Teams, Slack and Zoom are sent the table as text."""

    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, event, text, **kwargs):
        self.messages.append(str(text))


def _chat_reply(rows, question, lang="en"):
    from core.result_renderer import _send_results

    chat = _Text()
    with _InLanguage(lang):
        asyncio.run(_send_results({}, chat, question, rows, "SELECT 1", 900, None, "acct-periods",
                                  {"id": 1, "db_type": "azure_sql"}, question_id=None, cache_result=False))
    return "\n".join(chat.messages)


class TestTheChatTable:

    @pytest.mark.parametrize("lang,named", [("en", "January 2025"), ("fr", "janvier 2025")])
    def test_a_month_by_its_name(self, lang, named):
        reply = _chat_reply(GAPPED, "revenue by month", lang)
        assert named in reply
        assert "2025-01" not in reply

    def test_a_month_key(self):
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": 10.0 * month} for month in (1, 2, 3)]
        reply = _chat_reply(rows, "revenue by month")
        assert "March 2025" in reply and "202503" not in reply

    def test_a_day_key_is_the_day_not_its_month(self):
        rows = [{"INVOICE_DATE_KEY": 20260105 + day, "REVENUE": 10.0 * day} for day in range(3)]
        reply = _chat_reply(rows, "revenue by invoice date")
        assert all(f"2026-01-0{5 + day}" in reply for day in range(3))


class TestThePortalTable:
    """The browser formats the same cells from the same styles
    (portal_chat.html's _formatDisplayValue, executed)."""

    CASES = [("202503", "month_year_long", "PRD_KEY"), ("20250305", "iso_date", "INVOICE_DATE_KEY"),
             ("2025-04-01", "quarter", "PERIOD"), ("2025-01-01", "year", "YEAR_START")]

    @pytest.mark.parametrize("lang,expected", [
        ("en", ["March 2025", "2025-03-05", "Q2 2025", "2025"]),
        ("fr", ["mars 2025", "2025-03-05", "T2 2025", "2025"]),
    ])
    def test_every_period_shape(self, lang, expected):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify(" + json.dumps(self.CASES) + ".map(c => "
            "_formatDisplayValue(c[0], 'date', {type:'date', style:c[1]}, c[2])));",
            lang=lang,
            functions=[
                "function _formatDisplayValue(value, fmt, spec = {}, column = '')",
                "function _parseDisplayDate(value)",
                "function _parseDisplayNumber(v)",
                "function _isPeriodLabel(raw, column)",
            ],
        )
        assert got == expected


class TestTheChart:

    def _axis(self, labels, lang="en", style="quarter"):
        pytest.importorskip("dukpy")
        from tests.test_chart_annotation_language import _build

        payload = {"rows": [{"PERIOD": label, "REVENUE": 1.0} for label in labels], "x_key": "PERIOD",
                   "y_keys": ["REVENUE"], "chart_type": "line", "x_style": style,
                   "chart_spec": {"x": {"column": "PERIOD", "role": "temporal"}}}
        return lambda expression: _build("portal_chat.html", lang, payload, expression)

    @pytest.mark.parametrize("lang,first,second,tip", [
        ("en", "Q1\n2025", "Q2", "Q2 2025"),
        ("fr", "T1\n2025", "T2", "T2 2025"),
    ])
    def test_a_quarter_axis(self, lang, first, second, tip):
        draw = self._axis(["2025-01-01", "2025-04-01", "2025-07-01"], lang)
        assert draw("opt.xAxis.axisLabel.formatter('2025-01-01', 0)") == first
        assert draw("opt.xAxis.axisLabel.formatter('2025-04-01', 1)") == second
        tooltip = draw("opt.tooltip.formatter([{axisValue:'2025-04-01', value:5, seriesName:'REVENUE', "
                       "color:'#000'}])")
        assert tip in tooltip

    def test_a_year_axis(self):
        draw = self._axis(["2024-01-01", "2025-01-01"], style="year")
        assert draw("opt.xAxis.axisLabel.formatter('2025-01-01', 1)") == "2025"

    def test_a_month_bucket_is_its_month_not_its_first_day(self):
        draw = self._axis(["2025-01-01", "2025-02-01"], style="month_year_long")
        assert draw("opt.xAxis.axisLabel.formatter('2025-01-01', 0)") == "Jan\n2025"
        assert "January 2025" in draw(
            "opt.tooltip.formatter([{axisValue:'2025-01-01', value:5, seriesName:'REVENUE', color:'#000'}])")

    def test_a_chart_with_no_style_still_names_its_month_keys(self):
        draw = self._axis(["202501", "202502"], style="")
        assert draw("opt.xAxis.axisLabel.formatter('202501', 0)") == "Jan\n2025"


class TestAFrenchSeriesIsStillASeries:
    """Its labels are French names now, and every reader of a series reads
    them: what a period is, and its order."""

    def test_it_is_narrated_as_a_trend(self):
        rows = [{"PERIOD": date(2026, month, 1), "REVENUE": 100.0 * month} for month in range(1, 5)]
        with _InLanguage("fr"):
            context = summarize_result_context(rows, "revenus par mois")
        assert context["mode"] == "time_series"
        assert context["labels"] == ["janvier 2026", "février 2026", "mars 2026", "avril 2026"]

    def test_a_french_quarter_is_read_and_ordered(self):
        from core.analysis_evidence import period_order_key
        from core.insight import _looks_temporal
        from core.temporal_columns import parse_period_label

        assert parse_period_label("T2 2025") == date(2025, 4, 1)
        assert parse_period_label("février 2025") == date(2025, 2, 1)
        assert period_order_key("T2 2025") == (2025, 4, 0)
        assert _looks_temporal(["T1 2025", "T2 2025"]) is True
        # Terminals are not quarters, nor Marseille a month.
        assert _looks_temporal(["T1", "T2", "T3"]) is False
        assert _looks_temporal(["Marseille", "Lyon"]) is False


class TestTheGrain:

    @pytest.mark.parametrize("values,requested,column,grain", [
        ([date(2025, m, 1) for m in (1, 2, 3, 4, 6)], "", "PERIOD", "month"),
        ([date(2026, 3, 1)], "month", "PERIOD", "month"),
        ([date(2026, 3, 1)], "", "PERIOD", ""),
        ([date(2025, m, 1) for m in (1, 4, 7)], "", "PERIOD", "quarter"),
        # Asked by month, a month with no sale between is still a month.
        ([date(2025, m, 1) for m in (1, 4, 7)], "month", "PERIOD", "month"),
        ([date(2025, m, 1) for m in (1, 4)], "", "QUARTER_START", "quarter"),
        # Two Januaries are two years or two Januaries: neither is said.
        ([date(y, 1, 1) for y in (2024, 2025)], "", "PERIOD", ""),
        ([date(y, 1, 1) for y in (2023, 2024, 2025)], "", "PERIOD", "year"),
        ([date(2026, 1, d) for d in (3, 4, 5)], "", "ORDER_DATE", ""),
        ([20260105, 20260106], "", "INVOICE_DATE_KEY", "day"),
        ([202501, 202502], "", "PRD_KEY", "month"),
        (["2023", "2024"], "", "YEAR", "year"),
        (["Q1 2025", "Q2 2025"], "", "QTR", "quarter"),
        (["2025-01-01", "Halifax"], "", "PERIOD", ""),
    ])
    def test_of_a_column(self, values, requested, column, grain):
        from core.response_builder import period_grain

        assert period_grain(values, requested, column) == grain

    def test_the_sentences_and_the_table_agree(self):
        from core.response_builder import build_display_formats, narrative_period_labels

        formats = build_column_formats(GAPPED, {"period_grain": "month"})
        styles = build_display_formats(GAPPED, formats, requested_grain="month")
        with _InLanguage("en"):
            labels = narrative_period_labels([row["PERIOD"] for row in GAPPED], "month")
            cells = [i18n.format_date(row["PERIOD"], styles["PERIOD"]["style"]) for row in GAPPED]
        assert labels == cells == ["January 2025", "February 2025", "March 2025", "April 2025", "June 2025"]


def _sql_card(rows, question, sql, lang="en", **kwargs):
    with _InLanguage(lang):
        return build_assistant_response(question=question, rows=rows, sql=sql, duration_ms=1, **kwargs)


class TestWhichPeriodsTheyAre:
    """The grain asked for, then the column's own name, say which bucket a
    column of firsts is; with neither, only a run of whole quarters or years
    is one. A week's or a fiscal period's key is no calendar month, and a
    column named as a date keeps its days."""

    @pytest.mark.parametrize("rows,lang,headline", [
        # A MONTH column of January and April was "Q2 2025 closed at 400".
        ([{"ORDER_MONTH": date(2025, 1, 1), "SALES": 100.0}, {"ORDER_MONTH": date(2025, 4, 1), "SALES": 400.0}],
         "en", "April 2025 closed at 400."),
        # Its Januaries were "2025 closed at 300".
        ([{"ORDER_MONTH": date(year, 1, 1), "SALES": 100.0 * n} for n, year in enumerate((2023, 2024, 2025), 1)],
         "en", "January 2025 closed at 300."),
        ([{"MOIS": date(2025, 1, 1), "VENTES": 100.0}, {"MOIS": date(2025, 4, 1), "VENTES": 400.0}],
         "fr", "Avril 2025 a terminé à 400."),
    ])
    def test_a_month_column_is_months(self, rows, lang, headline):
        question = "historique des ventes" if lang == "fr" else "widget sales history"
        card = _card(rows, question, lang)
        assert card["answer"]["headline"] == headline
        assert list(card["data"]["display_formats"].values()) == [{"type": "date", "style": "month_year_long"}]

    @pytest.mark.parametrize("months", [(1, 4), (1, 7, 10)])
    def test_firsts_that_may_be_quarters_or_months_are_left_as_dates(self, months):
        rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 1.0 * month} for month in months]
        assert _card(rows, "revenue history")["data"]["display_formats"] == {}

    def test_a_windows_unit_is_no_grain(self):
        from core.response_builder import requested_period_grain

        # "the last 3 years" is a window, how far back the answer reaches.
        assert requested_period_grain("revenue for the last 3 years") == ""
        assert requested_period_grain("sales for the last 3 days") == "day"
        rows = [{"ORDER_MONTH": date(year, 1, 1), "REVENUE": 100.0 * n}
                for n, year in enumerate((2023, 2024, 2025), 1)]
        assert _card(rows, "January revenue for the last 3 years")["answer"]["headline"] == \
            "January 2025 closed at 300."

    def test_the_plan_names_the_buckets_its_sql_made(self):
        rows = [{"PERIOD": date(2025, 1, 1), "REVENUE": 1.0}, {"PERIOD": date(2025, 4, 1), "REVENUE": 2.0}]
        chart = {"x_key": "PERIOD", "y_keys": ["REVENUE"], "chart_type": "line"}
        with _InLanguage("en"):
            card = build_assistant_response(
                question="revenue", rows=rows, sql="SELECT 1", duration_ms=1, chart=chart,
                semantic_plan={"date_disclosures": [{"requested_grain": "quarter"}]})
        assert card["answer"]["headline"] == "Q2 2025 closed at 2."
        assert "Q1 2025" in card["insight_summary"]
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "quarter"}}
        assert card["chart"]["x_style"] == "quarter"

    @pytest.mark.parametrize("column,question", [("WEEK_KEY", "revenue by week"),
                                                 ("FISCAL_PERIOD", "revenue by fiscal period")])
    def test_a_week_or_fiscal_key_is_its_key(self, column, question):
        """WEEK_KEY 202501..202504 was headed "April 2025 closed at 400" and
        grouped "202,501" in its cells: a series of weeks, by their keys."""
        rows = [{column: 202500 + week, "REVENUE": 100.0 * week} for week in (4, 1, 3, 2)]
        card = _sql_card(rows, question, f"SELECT {column}, SUM(X) AS REVENUE FROM T GROUP BY {column}")
        assert card["answer"]["headline"] == "202504 closed at 400."
        assert "from 202501 to 202504" in card["insight_summary"]
        assert card["data"]["column_formats"] == {column: "text"}
        assert [row[column] for row in card["data"]["rows"]] == ["202504", "202501", "202503", "202502"]

    @pytest.mark.parametrize("column,lang,question,headline", [
        # A French warehouse's fiscal period, and the periods an accounting
        # system posts to: their keys are no calendar months either.
        ("PERIODE_FISCALE", "fr", "revenus par période fiscale", "202504 a terminé à 400."),
        ("Période fiscale", "fr", "revenus par période fiscale", "202504 a terminé à 400."),
        ("PERIODE_EXERCICE", "fr", "revenus par période de l'exercice", "202504 a terminé à 400."),
        ("MOIS_EXERCICE", "fr", "revenus par mois de l'exercice", "202504 a terminé à 400."),
        ("ACCOUNTING_PERIOD", "en", "revenue by accounting period", "202504 closed at 400."),
        ("GL_PERIOD", "en", "revenue by period", "202504 closed at 400."),
        ("POSTING_PERIOD", "en", "revenue by posting period", "202504 closed at 400."),
    ])
    def test_a_french_or_accounting_period_key_is_its_key(self, column, lang, question, headline):
        rows = [{column: 202500 + period, "REVENUE": 100.0 * period} for period in (4, 1, 3, 2)]
        card = _sql_card(rows, question, f"SELECT [{column}], SUM(X) AS REVENUE FROM T GROUP BY [{column}]", lang)
        assert card["answer"]["headline"] == headline
        assert card["data"]["column_formats"] == {column: "text"}
        assert [row[column] for row in card["data"]["rows"]] == ["202504", "202501", "202503", "202502"]

    @pytest.mark.parametrize("column,year_kind", [("EXERCICE", "an exercice"), ("ACCOUNTING_YEAR", "an accounting year")])
    def test_a_fiscal_year_is_still_a_year(self, column, year_kind):
        rows = [{column: year, "REVENUE": 100.0 * n} for n, year in enumerate((2023, 2024, 2025), 1)]
        card = _card(rows, "revenue by year")
        assert card["answer"]["headline"] == "2025 closed at 300."
        assert card["data"]["display_formats"] == {column: {"type": "date", "style": "year"}}

    @pytest.mark.parametrize("rows,question", [
        ([{"ORDER_DATE": date(2025, month, 1), "REVENUE": 1.0 * month} for month in (3, 6, 9)],
         "revenue by order date"),
        ([{"EFFECTIVE_DATE": date(year, 1, 1), "RATE": 1.0 * n} for n, year in enumerate((2022, 2023, 2024), 1)],
         "rate by effective date"),
        ([{"WEEK_START": date(2025, 9, 1), "REVENUE": 1.0}, {"WEEK_START": date(2025, 9, 8), "REVENUE": 2.0}],
         "revenue by week"),
    ])
    def test_a_date_keeps_its_days(self, rows, question):
        # "March 2025", "2024" and "September 2025" named days as months,
        # years and a month.
        assert _card(rows, question)["data"]["display_formats"] == {}

    @pytest.mark.parametrize("column", ["OrderMonth", "CAL_YM"])
    def test_a_key_named_for_its_month_in_any_case(self, column):
        rows = [{column: 202500 + month, "REVENUE": 100.0 * month} for month in (1, 2, 3, 4)]
        card = _card(rows, "revenue trend")
        assert card["answer"]["headline"] == "April 2025 closed at 400."
        assert card["data"]["display_formats"] == {column: {"type": "date", "style": "month_year_long"}}

    def test_a_number_in_a_column_named_for_no_period_is_no_month(self):
        rows = [{"BUCKET": 202500 + month, "REVENUE": 100.0 * month} for month in (1, 2, 3, 4)]
        card = _card(rows, "revenue trend")
        assert "April 2025" not in card["answer"]["headline"] + card["insight_summary"]
        assert card["data"]["column_formats"] == {}

    def test_a_french_period_key(self):
        rows = [{"PERIODE": 202500 + month, "REVENU": 100.0 * month} for month in (1, 2, 3)]
        assert _card(rows, "revenus", "fr")["answer"]["headline"] == "Mars 2025 a terminé à 300."

    @pytest.mark.parametrize("period", [lambda m: datetime(2025, m, 1), lambda m: f"2025-0{m}-01 00:00:00"])
    def test_a_bucket_at_midnight_is_its_month(self, period):
        rows = [{"PERIOD": period(month), "REVENUE": 100.0 * month} for month in (1, 2, 3, 4)]
        assert _card(rows, "revenue trend")["answer"]["headline"] == "April 2025 closed at 400."

    def test_a_time_of_day_is_no_period(self):
        from core.response_builder import period_grain

        assert period_grain([datetime(2025, 4, 1, 14, 30), datetime(2025, 5, 1, 9, 0)]) == ""

    def test_a_word_that_begins_with_a_month_is_no_month(self):
        from core.temporal_columns import parse_period_label

        rows = [{"GRADE": grade, "SALES": sales} for grade, sales in (("Octane 87", 900.0), ("Octane 89", 50.0),
                                                                       ("Octane 91", 20.0))]
        assert _card(rows, "sales by grade")["answer"]["headline"] == "Octane 87 leads at 900."
        assert parse_period_label("Junior 12") is None and parse_period_label("Marseille 13") is None
        assert parse_period_label("Sept 2025") == date(2025, 9, 1)

    def test_a_month_written_as_its_word_reaches_the_browser_as_its_day(self):
        rows = [{"MONTH": month, "REVENUE": value} for month, value in (("Jan-25", 1.0), ("Mar-25", 2.0),
                                                                          ("Jun-25", 3.0))]
        card = _card(rows, "revenue by month")
        # "Jan-25" was read by the browser as January 2001.
        assert [row["MONTH"] for row in card["data"]["rows"]] == ["2025-01-01", "2025-03-01", "2025-06-01"]


class TestPeriodsRankedByAFigure:
    """Periods the SQL lists by a figure are a ranking of them: put back in
    time order and read as a series, "top 3 months by revenue" said "June
    2025 closed at 700" and "Revenue trended down 12.5%"."""

    ROWS = [{"PRD_KEY": 202503, "REVENUE": 900.0}, {"PRD_KEY": 202501, "REVENUE": 800.0},
            {"PRD_KEY": 202506, "REVENUE": 700.0}]

    @pytest.mark.parametrize("question,sql", [
        ("top 3 months by revenue",
         "SELECT TOP 3 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY SUM(REV) DESC"),
        ("which months had the highest revenue",
         "SELECT PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY REVENUE DESC"),
        ("best months", "SELECT PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY 2 DESC"),
        # The months a subquery cut to the top of a figure, or a window ranked
        # by one, are a ranking whatever the statement outside them says.
        ("top 3 months by revenue",
         "SELECT * FROM (SELECT TOP 3 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY "
         "ORDER BY SUM(REV) DESC) t"),
        ("top 3 months by revenue",
         "SELECT PRD_KEY, REVENUE FROM (SELECT PRD_KEY, SUM(REV) AS REVENUE, "
         "ROW_NUMBER() OVER (ORDER BY SUM(REV) DESC) AS RN FROM F GROUP BY PRD_KEY) t WHERE RN <= 3 ORDER BY RN"),
        ("which months had the highest revenue",
         "SELECT PRD_KEY, REVENUE FROM (SELECT PRD_KEY, SUM(REV) AS REVENUE, "
         "RANK() OVER (ORDER BY SUM(REV) DESC) AS RK FROM F GROUP BY PRD_KEY) t WHERE RK <= 3 ORDER BY PRD_KEY"),
        ("top 3 months by revenue",
         "SELECT PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY SUM(REV) DESC "
         "OFFSET 0 ROWS FETCH FIRST 3 ROWS ONLY"),
    ])
    def test_a_ranking(self, question, sql):
        card = _sql_card(self.ROWS, question, sql)
        assert card["answer"]["headline"] == "March 2025 leads at 900."
        assert "trended" not in card["insight_summary"]

    @pytest.mark.parametrize("sql", [
        "SELECT PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY PRD_KEY",
        # A window's order, or a subquery's, lists nothing the reader sees.
        "SELECT PRD_KEY, SUM(REV) OVER (ORDER BY SUM(REV) DESC) AS REVENUE FROM F ORDER BY PRD_KEY",
        "SELECT * FROM (SELECT TOP 10 PRD_KEY, REV AS REVENUE FROM F ORDER BY REV DESC) s",
        # Last months, cut by their order and listed by it.
        "SELECT * FROM (SELECT TOP 12 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY "
        "ORDER BY PRD_KEY DESC) t ORDER BY PRD_KEY",
    ])
    def test_a_series(self, sql):
        card = _sql_card(self.ROWS, "revenue by month", sql)
        assert card["answer"]["headline"] == "June 2025 closed at 700."

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM (SELECT TOP 12 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY "
        "ORDER BY PRD_KEY DESC) t ORDER BY PRD_KEY",
        "SELECT PRD_KEY, REVENUE FROM (SELECT PRD_KEY, SUM(REV) AS REVENUE, "
        "ROW_NUMBER() OVER (ORDER BY PRD_KEY DESC) AS RN FROM F GROUP BY PRD_KEY) t WHERE RN <= 12 ORDER BY PRD_KEY",
    ])
    def test_a_cut_by_the_periods_own_order_is_a_series_whatever_is_asked(self, sql):
        """"Top 3 months by revenue" is a ranking; the last 12 months cut by
        their own order are the months in time."""
        card = _sql_card(self.ROWS, "top 3 months by revenue", sql)
        assert card["answer"]["headline"] == "June 2025 closed at 700."

    @pytest.mark.parametrize("sql", [
        "SELECT DATENAME(month, ORDER_DATE) + ' ' + DATENAME(year, ORDER_DATE) AS MONTH_NAME, SUM(X) AS NET_SALES "
        "FROM T GROUP BY DATENAME(month, ORDER_DATE), DATENAME(year, ORDER_DATE) ORDER BY MIN(ORDER_DATE)",
        "SELECT FORMAT(ORDER_DATE, 'MMMM yyyy') AS MONTH_NAME, SUM(X) AS NET_SALES FROM T "
        "GROUP BY FORMAT(ORDER_DATE, 'MMMM yyyy') ORDER BY MAX(ORDER_DATE) ASC",
    ])
    def test_the_first_or_last_date_of_a_period_lists_it_in_time(self, sql):
        """MIN(ORDER_DATE) is a date, not a figure: months listed by it are a
        series, and were a ranking that "leads at 900"."""
        rows = [{"MONTH_NAME": f"{name} 2025", "NET_SALES": sales}
                for name, sales in (("January", 800.0), ("February", 900.0), ("March", 700.0), ("April", 750.0))]
        card = _sql_card(rows, "net sales by month", sql)
        assert "leads" not in card["answer"]["headline"]
        assert card["answer"]["headline"] == "April 2025 closed at 750."

    def test_a_maximum_that_is_a_figure_still_ranks(self):
        # Listed by the price, the months come back in the price's order.
        rows = [{"PRD_KEY": 202500 + month, "TOP_PRICE": price} for month, price in ((2, 90.0), (3, 55.0), (1, 40.0))]
        sql = "SELECT PRD_KEY, MAX(PRICE) AS TOP_PRICE FROM F GROUP BY PRD_KEY ORDER BY MAX(PRICE) DESC"
        assert _sql_card(rows, "which months had the highest price", sql)["answer"]["headline"] == \
            "February 2025 leads at 90."

    def test_which_months_is_a_ranked_period(self):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain("which months had the highest revenue") == "month"


class TestAYear:
    """A year column is a date whose style is its year: the portal read the
    bare "2021" as midnight UTC -- 2020 in every time zone west of Greenwich
    -- the chat's table grouped it "2,021", and the table's footer totalled
    the years."""

    ROWS = [{"YEAR": year, "REVENUE": 1000.0 * n} for n, year in enumerate((2021, 2022, 2023, 2024), 1)]

    def test_the_chat_table(self):
        reply = _chat_reply(self.ROWS, "net sales by year")
        assert "2021" in reply and "2,021" not in reply

    def test_the_server_names_it_as_its_year(self):
        from core.response_builder import _format_display_value

        for lang in ("en", "fr"):
            with _InLanguage(lang):
                assert _format_display_value(2021, "date", {"type": "date", "style": "year"}) == "2021"
                assert _format_display_value("Q2 2025", "date", {"type": "date", "style": "year"}) in (
                    "Q2 2025", "T2 2025")

    @pytest.mark.parametrize("cell", ["2021-01-01", "2021"])
    def test_the_portal_west_of_greenwich(self, cell):
        pytest.importorskip("dukpy")
        import os
        import time

        from tests.chat_js import run

        before = os.environ.get("TZ")
        os.environ["TZ"] = "America/Toronto"
        time.tzset()
        try:
            got = run(
                f"JSON.stringify([_formatDisplayValue({json.dumps(cell)}, 'date', {{type:'date', style:'year'}},"
                " 'YEAR')]);",
                functions=["function _formatDisplayValue(value, fmt, spec = {}, column = '')",
                           "function _parseDisplayDate(value)", "function _parseDisplayNumber(v)",
                           "function _isPeriodLabel(raw, column)"])
        finally:
            if before is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = before
            time.tzset()
        assert got == ["2021"]

    def test_no_day_a_calendar_lacks(self):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run("JSON.stringify(['20250230', '2025-02-30', '20250228'].map(v => _parseDisplayDate(v) === null));",
                  functions=["function _parseDisplayDate(value)"])
        assert got == [True, True, False]

    def test_the_footer_totals_no_period(self):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "var years = [{dataset:{sort:'2021-01-01'}, textContent:'2021'}, {dataset:{sort:'2022-01-01'},"
            " textContent:'2022'}];"
            "var amounts = [{dataset:{}, textContent:'1,500'}, {dataset:{}, textContent:'2,000'}];"
            "JSON.stringify([_totalledColumn(years), _totalledColumn(amounts)]);",
            functions=["function _totalledColumn(cells, header)", "function _parseDisplayNumber(v)"])
        assert got == [False, True]

    def test_a_named_period_sorts_by_its_date(self):
        """"April 2025" before "January 2025" is the alphabet."""
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "var cells = [['April 2025', '2025-04-01'], ['January 2025', '2025-01-01'],"
            " ['February 2025', '2025-02-01']].map(c => ({textContent: c[0], dataset: {sort: c[1]}}));"
            "cells.sort((a, b) => _compareCellValues(_cellSortValue(a), _cellSortValue(b)));"
            "JSON.stringify(cells.map(c => c.textContent));",
            functions=["function _cellSortValue(cell)", "function _compareCellValues(av, bv)",
                       "function _parseDisplayNumber(v)"])
        assert got == ["January 2025", "February 2025", "April 2025"]


class TestWhatElseReadsThePeriods:

    def test_the_brief_names_them_at_the_grain_asked(self):
        from core.insight import compute_data_brief

        rows = [{"PERIOD": date(2025, 1, 1), "REVENUE": 1.0}, {"PERIOD": date(2025, 4, 1), "REVENUE": 2.0}]
        context = summarize_result_context(rows, "revenue", asked_grain="quarter")
        brief = compute_data_brief(rows, "revenue", context=context, asked_grain="quarter")
        assert (brief["time_series"]["first_period"], brief["time_series"]["last_period"]) == ("Q1 2025", "Q2 2025")

    def test_a_cell_that_is_no_period(self):
        from core.response_builder import _period_cell

        assert _period_cell(float("nan")) == ""
        assert _period_cell({"a": 1}) == '{"a":1}'
        assert _period_cell("Jan-25") == "2025-01-01"

    def test_a_french_quarter_is_no_code(self):
        from core.stat_signals import _TEMPORAL_VALUE_RE

        assert [bool(_TEMPORAL_VALUE_RE.match(v)) for v in ("T2 2025", "2025-T2", "T100", "T1-10", "t250")] == [
            True, True, False, False, False]

    def test_the_single_value_reply_in_a_chat(self):
        reply = _chat_reply([{"LAST_MONTH": 202503}], "latest month")
        assert "March 2025" in reply and "202503" not in reply


@pytest.fixture
def tile(monkeypatch):
    """A dashboard whose source returns the second and third quarters, and
    each of its tiles as the dashboard page draws them."""
    import uuid
    from types import SimpleNamespace

    import store

    account_id = f"periods-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account_id, "portal")
    owner_id, _ = store.create_user(account_id, "Owner", f"{account_id}@example.com", password="a-long-password-1")
    board = store.create_dashboard(account_id, owner_id, "t", "Revenue by quarter")
    db_id = store.save_db_config("azure_sql", f"Warehouse {account_id}", {
        "server": "dw.example.net", "database": "DW", "user": "reader", "password": "a-password"})
    source = store.create_data_source(
        board["id"], owner_id, account_id, name="Revenue", question="revenue by quarter",
        sql_query="SELECT PERIOD, SUM(REVENUE) AS REVENUE FROM F GROUP BY PERIOD", db_config_id=db_id)
    rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (4, 7)]

    def governed(*args, **kwargs):
        return SimpleNamespace(rows=list(rows), sql=args[2], decision=SimpleNamespace(cache_ttl_seconds=86400))

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)

    def draw(chart_type):
        from portal.routes import _refresh_chart
        from tests.dashboard_render import CHART

        return _refresh_chart({**CHART, "question": "revenue by quarter", "data_source_id": source["id"],
                               "user_id": owner_id, "chart_type": chart_type, "sql_query": source["sql_query"]},
                              store.get_db_config(db_id), store.get_user(owner_id))

    return draw


class TestTheDashboard:

    def test_a_table_tile(self, tile):
        drawn = tile("table")
        assert [row["PERIOD"]["d"] for row in drawn["table_rows"]] == ["Q2 2025", "Q3 2025"]

    def test_a_chart_tile(self, tile):
        drawn = tile("line")
        assert json.loads(drawn["chart_json"])["x_style"] == "quarter"


class TestTheForecastChart:
    """A forecast's axis and its tooltip name the quarters the table does."""

    def _draw(self, expression, lang="en"):
        pytest.importorskip("dukpy")
        from tests.test_chart_annotation_language import _build

        rows = [{"PERIOD": period, "REVENUE": value, "is_forecast": forecast}
                for period, value, forecast in (("2025-01-01", 1.0, False), ("2025-04-01", 2.0, False),
                                                ("2025-07-01", 3.0, True))]
        payload = {"rows": rows, "x_key": "PERIOD", "y_keys": ["REVENUE"], "chart_type": "forecast",
                   "x_style": "quarter"}
        return _build("portal_chat.html", lang, payload, expression)

    def test_its_axis(self):
        assert self._draw("opt.xAxis.axisLabel.formatter('2025-04-01', 1)") == "Q2"

    def test_its_tooltip(self):
        tip = self._draw("opt.tooltip.formatter([{dataIndex: 1, axisValue:'2025-04-01', value:2, "
                         "seriesName:'REVENUE', color:'#000'}])")
        assert "Q2 2025" in tip


class TestTheReplyCommandsName:

    @pytest.mark.parametrize("style,named", [("quarter", "quarter and year"), ("iso_date", "YYYY-MM-DD")])
    def test_a_period_style(self, style, named):
        from core.result_commands import _format_label

        with _InLanguage("en"):
            assert _format_label({"type": "date", "style": style}) == named


def test_a_period_key_kept_as_a_float():
    from core.response_builder import _period_cell

    assert _period_cell(202503.0) == "2025-03-01"


class TestAYearListedNewestFirst:
    """A bare-year series is read in time whatever order the SQL listed it in:
    "revenue by year" ORDER BY YEAR DESC said "trended down 12.5% from 2025 to
    2023. Peak: 800 in 2025" of revenue that rose."""

    ROWS = [{"YEAR": year, "REVENUE": revenue} for year, revenue in ((2025, 800.0), (2024, 750.0), (2023, 700.0))]

    @pytest.mark.parametrize("lang,question,phrase,headline", [
        ("en", "revenue by year", "trended up 14.3% from 2023 to 2025", "2025 closed at 800."),
        ("fr", "revenus par année", "a progressé de 14,3\u00a0% entre 2023 et 2025", "2025 a terminé à 800."),
    ])
    def test_the_trend_runs_forward(self, lang, question, phrase, headline):
        card = _sql_card(self.ROWS, question,
                         "SELECT YEAR, SUM(X) AS REVENUE FROM T GROUP BY YEAR ORDER BY YEAR DESC", lang)
        assert phrase in card["insight_summary"]
        assert card["answer"]["headline"] == headline


class TestATierIsNoQuarter:
    """T1 to T4 are the terminals of a French quarter only beside a year: a
    tier column T1..T4 is a list of categories, and was narrated "Revenue
    trended down 94.4% from T1 to T4" with a "3 consecutive periods of
    decline" callout."""

    ROWS = [{"TIER": tier, "REVENUE": revenue} for tier, revenue in (("T1", 900.0), ("T2", 500.0), ("T3", 100.0),
                                                                     ("T4", 50.0))]

    @pytest.mark.parametrize("lang,question", [("en", "revenue by tier"), ("fr", "revenus par niveau")])
    def test_a_tier_is_ranked_not_trended(self, lang, question):
        card = _sql_card(self.ROWS, question, "SELECT TIER, SUM(X) AS REVENUE FROM T GROUP BY TIER ORDER BY TIER", lang)
        assert "trended" not in card["insight_summary"] and "progress" not in card["insight_summary"]
        assert not [c for c in card["anomaly_callouts"] if "consecutive" in c["message"]]
        assert card["data_brief"]["mode"] != "time_series"

    def test_a_quarter_written_with_its_year_is_still_one(self):
        rows = [{"QTR": label, "REVENUE": revenue} for label, revenue in
                (("T1 2025", 100.0), ("T2 2025", 200.0), ("T3 2025", 300.0))]
        card = _card(rows, "revenus par trimestre", "fr")
        assert card["data_brief"]["mode"] == "time_series"


class TestWeekAndFiscalKeys:
    """A week's key (202530, the 30th week of 2025) and a fiscal period's
    (202513, the 13th of a year kept in thirteen periods) are a year and the
    period's number, which runs past 12. Read as a calendar month they were no
    period at all and the column a measure -- "Week Key ranges 202,530 to
    202,533, avg 202,531.50", "Σ Week Key: 810,126" under the table -- and the
    chart's axis named the keys below 13 as months while the table named
    them by their keys."""

    @pytest.mark.parametrize("column,keys", [
        ("WEEK_KEY", (202530, 202531, 202532, 202533)), ("WEEK_KEY", (202601, 202652, 202653)),
        ("FISCAL_PERIOD", (202513, 202514, 202515)), ("PERIODE_FISCALE", (202513, 202514)),
        ("ACCOUNTING_PERIOD", (202513, 202514)), ("SEMAINE", (202530, 202531)),
    ])
    def test_a_column_of_them_is_a_column_of_periods(self, column, keys):
        from core.temporal_columns import is_calendar_period_column

        assert is_calendar_period_column(column, list(keys))

    @pytest.mark.parametrize("column,keys", [
        ("WEEK_KEY", (202554, 202555)),          # no year has a 54th week
        ("WEEK_KEY", (202500, 202501)),          # nor a week 0
        ("FISCAL_PERIOD", (202518, 202519)),     # nor a year eighteen periods
        ("CUSTOMER_KEY", (202530, 202531)),      # a key no period names
        ("ORDER_QTY", (202530, 202531)),
        ("WEEK_KEY", (2025301, 2025302)),        # not a year and a number
        ("WEEK_QTY", (202530, 202531)),          # a measure's suffix vetoes
    ])
    def test_and_no_other_column_is(self, column, keys):
        from core.temporal_columns import is_calendar_period_column

        assert not is_calendar_period_column(column, list(keys))

    @pytest.mark.parametrize("lang,question,close,summary", [
        ("en", "revenue by week", "202533 closed at 100.", "from 202530 to 202533"),
        ("fr", "revenus par semaine", "202533 a terminé à 100.", "entre 202530 et 202533"),
    ])
    def test_a_series_of_weeks_is_read_in_time_by_its_keys(self, lang, question, close, summary):
        rows = [{"WEEK_KEY": 202500 + week, "REVENUE": revenue}
                for week, revenue in ((33, 100.0), (30, 200.0), (32, 300.0), (31, 400.0))]
        card = _sql_card(rows, question, "SELECT WEEK_KEY, SUM(X) AS REVENUE FROM T GROUP BY WEEK_KEY "
                         "ORDER BY WEEK_KEY DESC", lang)
        assert card["answer"]["headline"] == close
        assert summary in card["insight_summary"]
        assert card["data_brief"]["mode"] == "time_series"
        assert card["data"]["column_formats"] == {"WEEK_KEY": "text"}
        assert [row["WEEK_KEY"] for row in card["data"]["rows"]] == ["202533", "202530", "202532", "202531"]

    @pytest.mark.parametrize("keys", [(202501, 202502, 202503), (202530, 202531, 202532)])
    def test_the_chart_names_the_key_the_table_does(self, keys):
        rows = [{"WEEK_KEY": key, "REVENUE": 100.0 * n} for n, key in enumerate(keys, 1)]
        chart = {"x_key": "WEEK_KEY", "y_keys": ["REVENUE"], "chart_type": "line"}
        card = _sql_card(rows, "revenue by week", "SELECT WEEK_KEY, SUM(X) AS REVENUE FROM T GROUP BY WEEK_KEY",
                         chart=chart)
        assert card["chart"]["x_style"] == "key"

    def test_the_axis_reads_no_month_in_a_key(self):
        pytest.importorskip("dukpy")
        from tests.test_chart_annotation_language import _build

        payload = {"rows": [{"WEEK_KEY": key, "REVENUE": 1.0} for key in ("202501", "202502")], "x_key": "WEEK_KEY",
                   "y_keys": ["REVENUE"], "chart_type": "line", "x_style": "key",
                   "chart_spec": {"x": {"column": "WEEK_KEY", "role": "temporal"}}}
        draw = lambda expression: _build("portal_chat.html", "en", payload, expression)  # noqa: E731
        assert draw("opt.xAxis.axisLabel.formatter('202501', 0)") == "202501"
        assert "202502" in draw("opt.tooltip.formatter([{axisValue:'202502', value:5, seriesName:'REVENUE', "
                                "color:'#000'}])")

    def test_the_footer_totals_no_key(self):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "var keys = [{dataset:{}, textContent:'202530'}, {dataset:{}, textContent:'202531'}];"
            "var amounts = [{dataset:{}, textContent:'1,500'}, {dataset:{}, textContent:'2,000'}];"
            "JSON.stringify([_totalledColumn(keys, {dataset:{fmt:'text'}}), _totalledColumn(keys, {dataset:{}}),"
            " _totalledColumn(amounts, {dataset:{fmt:'number'}})]);",
            functions=["function _totalledColumn(cells, header)", "function _parseDisplayNumber(v)"])
        assert got == [False, True, True]


class TestThePortalTableInFrench:
    """The server writes every number the way English does ("12,000",
    "3,000.5"), whatever the page's language. The French page read those with
    its own separators -- a comma is its decimal point -- so "12,000" was 12,
    "3,000.5" no number at all: the cells stayed as sent, and the footer read
    "Σ Revenue: 13" under thirteen thousand."""

    FUNCTIONS = ["function _formatDisplayValue(value, fmt, spec = {}, column = '')", "function _parseDisplayDate(value)",
                 "function _parseDisplayNumber(v)", "function _parseServerNumber(v)",
                 "function _isPeriodLabel(raw, column)", "function _isANumericColumn(rows, header)"]

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_a_number_the_server_sent_is_formatted_in_the_pages_language(self, lang):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify([_formatDisplayValue('12,000', undefined, {}, 'REVENUE'),"
            " _formatDisplayValue('3,000.5', undefined, {}, 'REVENUE'),"
            " _formatDisplayValue('1,000', 'number', {fraction_digits: 0}, 'REVENUE'),"
            " _formatDisplayValue('-1,234,567.89', 'number', {}, 'REVENUE'),"
            " _formatDisplayValue('2,500', 'currency', {fraction_digits: 0}, 'REVENUE'),"
            " window.qbNum(12000, {max: 2}), window.qbNum(3000.5, {max: 2}), window.qbNum(1000, {max: 0}),"
            " window.qbNum(-1234567.89, {min: 2, max: 2}), window.qbNum(2500, {min: 0, max: 0})]);",
            lang=lang, functions=self.FUNCTIONS + ["function _qbMoney(body, symbol)"],
            consts=["_QB_CURRENCY_SYMBOL"])
        assert got[0] == got[5] and got[1] == got[6] and got[2] == got[7] and got[3] == got[8]
        assert got[9] in got[4]

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_what_the_page_wrote_reads_back_as_the_number_the_server_sent(self, lang):
        """The sort and the footer read the cell's text back: they must get the
        number the server sent, in either language."""
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify(['12,000', '3,000.5', '1,000', '-1,234,567.89', '0.25', '999']"
            ".map(v => _parseDisplayNumber(_formatDisplayValue(v, undefined, {}, 'REVENUE'))));",
            lang=lang, functions=self.FUNCTIONS)
        assert got == [12000, 3000.5, 1000, -1234567.89, 0.25, 999]

    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_a_column_of_them_is_a_column_of_numbers(self, lang):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify([_isANumericColumn([{R: '12,000'}, {R: '3,000.5'}], 'R'),"
            " _isANumericColumn([{R: 'North'}, {R: 'South'}], 'R'), _isANumericColumn([{R: ''}, {R: null}], 'R')]);",
            lang=lang, functions=self.FUNCTIONS)
        assert got == [True, False, False]


# ── Round 4: what the periods are is read from evidence, never from a hunch ─────

_SIX_MONTHS = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value}
               for month, value in zip(range(1, 7), (800.0, 850.0, 900.0, 700.0, 950.0, 1000.0))]
_TOP_CUSTOMERS = ("WITH top5 AS (SELECT TOP 5 CUSTOMER_ID FROM F GROUP BY CUSTOMER_ID ORDER BY SUM(NET_AMT) DESC) "
                  "SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F WHERE CUSTOMER_ID IN "
                  "(SELECT CUSTOMER_ID FROM top5) GROUP BY ORDER_MONTH ORDER BY ORDER_MONTH")
_TOP_PRODUCTS_LIMIT = ("SELECT ORDER_MONTH, SUM(amount) AS REVENUE FROM orders WHERE product_id IN "
                       "(SELECT product_id FROM orders GROUP BY product_id ORDER BY SUM(amount) DESC LIMIT 3) "
                       "GROUP BY ORDER_MONTH ORDER BY ORDER_MONTH")
_TOP_CUSTOMERS_NUMBERED = ("WITH r AS (SELECT CUSTOMER_ID, ROW_NUMBER() OVER (ORDER BY SUM(NET_AMT) DESC) AS RN "
                           "FROM F GROUP BY CUSTOMER_ID) SELECT ORDER_MONTH, SUM(F.NET_AMT) AS REVENUE FROM F "
                           "JOIN r ON r.CUSTOMER_ID = F.CUSTOMER_ID WHERE r.RN <= 10 GROUP BY ORDER_MONTH "
                           "ORDER BY ORDER_MONTH")
_BEST_SELLER = ("SELECT ORDER_MONTH, SUM(QTY) AS REVENUE FROM F WHERE ITEM_ID = (SELECT TOP 1 ITEM_ID FROM F "
                "GROUP BY ITEM_ID ORDER BY SUM(QTY) DESC) GROUP BY ORDER_MONTH ORDER BY ORDER_MONTH")


class TestTheMonthsOfTheTopMembersAreASeries:
    """A cut by a figure ranks what it picks. "The top 5 customers" are five
    customers, whose monthly totals are listed in time: read as a ranking of
    the months, June "leads at 1,000" and the trend, the drop and the gain
    were lost."""

    @pytest.mark.parametrize("question,sql", [
        ("monthly revenue for the top 5 customers", _TOP_CUSTOMERS),
        ("sales by month for the top 3 products", _TOP_PRODUCTS_LIMIT),
        ("revenue by month for the 10 largest customers", _TOP_CUSTOMERS_NUMBERED),
        ("monthly revenue of the best-selling item", _BEST_SELLER),
        ("revenue trend of our top 5 customers", _TOP_CUSTOMERS),
    ])
    def test_the_months_of_the_top_members(self, question, sql):
        card = _sql_card(_SIX_MONTHS, question, sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "June 2025 closed at 1,000."
        assert card["anomaly_callouts"]

    def test_in_french_too(self):
        card = _sql_card(_SIX_MONTHS, "chiffre d'affaires mensuel des 5 meilleurs clients", _TOP_CUSTOMERS, "fr")
        assert card["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH "
        "ORDER BY SUM(NET_AMT) DESC) t ORDER BY ORDER_MONTH",
        "SELECT ORDER_MONTH, REVENUE FROM (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE, "
        "ROW_NUMBER() OVER (ORDER BY SUM(NET_AMT) DESC) AS RN FROM F GROUP BY ORDER_MONTH) t WHERE RN <= 3 "
        "ORDER BY ORDER_MONTH",
        "SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH "
        "QUALIFY ROW_NUMBER() OVER (ORDER BY SUM(NET_AMT) DESC) <= 3",
        "SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH ORDER BY SUM(NET_AMT) DESC "
        "FETCH FIRST 3 ROWS ONLY",
        # A star hands on the columns of what it reads.
        "WITH m AS (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH), "
        "r AS (SELECT *, RANK() OVER (ORDER BY REVENUE DESC) AS RK FROM m) "
        "SELECT ORDER_MONTH, REVENUE FROM r WHERE RK <= 3 ORDER BY ORDER_MONTH",
        "SELECT ORDER_MONTH, REVENUE FROM (SELECT t.*, ROW_NUMBER() OVER (ORDER BY REVENUE DESC) AS RN FROM "
        "(SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH) t) u WHERE RN <= 3 "
        "ORDER BY ORDER_MONTH",
    ])
    def test_a_cut_of_the_months_themselves_still_ranks_them(self, sql):
        rows = [_SIX_MONTHS[5], _SIX_MONTHS[4], _SIX_MONTHS[2]]
        card = _sql_card(sorted(rows, key=lambda row: row["ORDER_MONTH"]), "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert "leads" in card["answer"]["headline"]


class TestTheFirstOrLastDateOfAPeriod:
    """ORDER BY MIN(x) of a date lists the periods in time, whatever the column
    is called: only a date word in its name was read as one, and MIN(ORDERDATE),
    MAX(created_at) and MIN(TXN_TS) ranked the months."""

    ROWS = [{"MONTH_NAME": f"{name} 2025", "NET_SALES": sales}
            for name, sales in (("January", 800.0), ("February", 850.0), ("March", 900.0), ("April", 700.0))]

    @pytest.mark.parametrize("column", ["ORDERDATE", "created_at", "TXN_TS", "POSTED_ON", "INVOICE_DATETIME",
                                        "ORDER_DATE"])
    @pytest.mark.parametrize("fn", ["MIN", "MAX"])
    def test_months_listed_by_it_are_in_time(self, column, fn):
        sql = ("SELECT FORMAT(d, 'MMMM yyyy') AS MONTH_NAME, SUM(x) AS NET_SALES FROM T "
               f"GROUP BY FORMAT(d, 'MMMM yyyy') ORDER BY {fn}({column})")
        card = _sql_card(self.ROWS, "net sales by month", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "April 2025 closed at 700."

    def test_listed_newest_first_it_is_still_a_series(self):
        sql = "SELECT FORMAT(d, 'MMMM yyyy') AS MONTH_NAME, SUM(x) AS NET_SALES FROM T GROUP BY 1 ORDER BY MAX(d) DESC"
        card = _sql_card(list(reversed(self.ROWS)), "net sales by month", sql)
        assert card["answer"]["headline"] == "April 2025 closed at 700."

    def test_in_french(self):
        rows = [{"MOIS": f"{name} 2025", "VENTES": sales}
                for name, sales in (("janvier", 800.0), ("février", 850.0), ("mars", 900.0), ("avril", 700.0))]
        sql = "SELECT FORMAT(d, 'MMMM yyyy') AS MOIS, SUM(x) AS VENTES FROM T GROUP BY 1 ORDER BY MAX(created_at)"
        card = _sql_card(rows, "ventes par mois", sql, "fr")
        assert card["answer"]["headline"] == "Avril 2025 a terminé à 700."

    def test_the_largest_amount_of_a_period_is_a_figure(self):
        rows = [{"MONTH_NAME": "March 2025", "TOP": 900.0}, {"MONTH_NAME": "February 2025", "TOP": 850.0},
                {"MONTH_NAME": "January 2025", "TOP": 800.0}, {"MONTH_NAME": "April 2025", "TOP": 700.0}]
        sql = ("SELECT FORMAT(d, 'MMMM yyyy') AS MONTH_NAME, MAX(amount) AS TOP FROM T GROUP BY 1 "
               "ORDER BY MAX(amount) DESC")
        assert _sql_card(rows, "largest sale by month", sql)["analysis_contract"]["mode"] == "ranking"


class TestEveryReaderOfAResultAgrees:
    """The analyst's prompt, a chat about a result, the chart's markers and the
    card read one result, and read it alike: periods listed by a figure are a
    ranking, and a series runs in time -- whatever order the SQL listed it in."""

    KEYS = [{"PRD_KEY": 202500 + month, "REVENUE": value} for month, value in ((3, 900.0), (1, 800.0), (2, 700.0))]

    def test_the_prompt_reads_a_ranking_of_month_keys(self):
        from core.insight import describe_result_for_prompt

        sql = "SELECT PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY ORDER BY SUM(REV) DESC"
        text = describe_result_for_prompt(self.KEYS, "revenue by month", sql)
        assert "Mode: ranking" in text and "Mode: time_series" not in text

    def test_a_brief_with_no_sql_reads_the_rows_order(self):
        from core.insight import compute_data_brief

        assert compute_data_brief(self.KEYS, "revenue by month")["mode"] == "ranking"
        in_time = [{"PRD_KEY": 202500 + month, "REVENUE": value} for month, value in ((1, 700.0), (2, 900.0), (3, 800.0))]
        assert compute_data_brief(in_time, "revenue by month")["mode"] == "time_series"

    def test_a_brief_with_no_sql_places_no_period_it_cannot_read(self):
        from core.insight import compute_data_brief

        weeks = [{"WEEK": f"Week {number}", "REVENUE": 100.0 * number} for number in range(1, 7)]
        assert compute_data_brief(weeks, "revenue by week")["mode"] == "time_series"

    def test_the_chart_marks_no_drop_or_gain_of_a_ranking(self):
        from core.chart import build_chart_annotations

        assert build_chart_annotations(self.KEYS, "revenue by month") is None

    @pytest.mark.parametrize("months,question", [
        (("avril", "mars", "février", "janvier"), "ventes par mois"),
        (("janvier", "février", "mars", "avril"), "ventes par mois"),
    ])
    def test_a_french_series_runs_forward_whatever_order_it_came_in(self, months, question):
        from core.insight import compute_data_brief

        value = {"janvier": 800.0, "février": 850.0, "mars": 900.0, "avril": 950.0}
        rows = [{"MOIS": f"{name} 2025", "VENTES": value[name]} for name in months]
        with _InLanguage("fr"):
            brief = compute_data_brief(rows, question)
        assert brief["time_series"]["direction"] == "increasing"
        assert (brief["time_series"]["first_period"], brief["time_series"]["last_period"]) == (
            "janvier 2025", "avril 2025")

    def test_a_french_quarter_series_listed_newest_first(self):
        rows = [{"TRIMESTRE": f"T{quarter} 2025", "VENTES": 100.0 * quarter} for quarter in (4, 3, 2, 1)]
        card = _card(rows, "ventes par trimestre", "fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert "a progressé" in card["insight_summary"]
        assert card["data_brief"]["time_series"]["direction"] == "increasing"


class TestAMonthsWordInALabelIsNoMonth:
    """A label that only holds a month's word -- a brand called "Mars Bar", a
    street "Rue de Juin" -- is a member, not a period: the units of four
    chocolate bars "trended down 88.9% from Mars Bar to Bounty"."""

    @pytest.mark.parametrize("labels,lang", [
        (("Mars Bar", "Snickers", "Twix", "Bounty"), "en"),
        (("Mars", "Nestle", "Lindt", "Hershey"), "en"),
        (("Mai Tai", "Mojito", "Daiquiri", "Negroni"), "en"),
        (("Rue de Juin", "Rue de Paris", "Rue Haute", "Rue Basse"), "fr"),
    ])
    def test_it_is_ranked(self, labels, lang):
        rows = [{"PRODUCT": label, "UNITS": units} for label, units in zip(labels, (900.0, 500.0, 300.0, 100.0))]
        card = _card(rows, "units by product" if lang == "en" else "unités par produit", lang)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["data_brief"]["mode"] == "ranking"
        assert "trended" not in card["insight_summary"] and "reculé" not in card["insight_summary"]

    @pytest.mark.parametrize("label,expected", [("mars", True), ("mars 2025", True), ("févr. 25", True),
                                                ("Mars Bar", False), ("Rue de Juin", False), ("Mai Tai", False)])
    def test_a_french_month_is_a_label_that_is_one_and_nothing_else(self, label, expected):
        from core.temporal_columns import FRENCH_MONTH_LABEL_RE

        assert bool(FRENCH_MONTH_LABEL_RE.fullmatch(label)) is expected


class TestWhatASinglePeriodValueReachesTheBrowserAs:
    """The KPI tile of a one-value answer is sent as the table's cell is: a
    period as the day it starts, a key as its digits. "2,021" is no date the
    browser can name, and it read it as the 21st of February 2001."""

    @pytest.mark.parametrize("rows,question,value", [
        ([{"FIRST_YEAR": 2021}], "first year with sales", "2021-01-01"),
        ([{"YEAR": 2021}], "year of the first sale", "2021-01-01"),
        ([{"FISCAL_YEAR": 2025}], "current fiscal year", "2025-01-01"),
        ([{"PRD_KEY": 202503}], "latest period with sales", "2025-03-01"),
        ([{"WEEK_KEY": 202530}], "latest week with sales", "202530"),
    ])
    def test_the_kpi_value(self, rows, question, value):
        assert _card(rows, question)["kpi"]["value"] == value

    def test_a_number_is_sent_as_before(self):
        assert _card([{"TOTAL_REVENUE": 1250.5}], "total revenue")["kpi"]["value"] == "1,250.5"

    def test_the_browser_reads_no_string_of_its_own(self):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run("JSON.stringify(['2,021', '21 Feb', 'Feb 21, 2001', '2021-03-05T10:00:00Z', '20210305', '2021']"
                  ".map(v => { const d = _parseDisplayDate(v); return d === null ? null : d.getFullYear(); }));",
                  functions=["function _parseDisplayDate(value)"])
        assert got == [None, None, None, 2021, 2021, 2021]


class TestTheGrainTheAnswerIsAskedAt:
    """The grain the question asks its periods at: its own grouping words, else
    the month or quarter it names, else the plan's; never a window's unit."""

    @pytest.mark.parametrize("question,plan,grain", [
        ("January revenue for the last 3 years", "year", "year|month:1"),
        ("March sales for each of the last 4 years", "year", "year|month:3"),
        ("Q1 revenue for each of the last 3 years", "year", "year|quarter:1"),
        ("revenue in the first quarter of the last 3 years", "year", "year|quarter:1"),
        ("revenue by month for the last 3 years", "year", "month"),
        ("revenue by year", "year", "year"),
        ("revenue for the last 3 years", "year", "year"),
        ("revenue for the last 3 years", "", ""),
        ("sales for the last 3 days", "", "day"),
        ("sales for the last 8 weeks", "", "week"),
        ("average daily sales by month", "", "day"),
    ])
    def test_the_asked_grain(self, question, plan, grain):
        from core.response_builder import requested_period_grain

        semantic_plan = {"date_disclosures": [{"requested_grain": plan}]} if plan else None
        assert requested_period_grain(question, semantic_plan) == grain

    def test_the_januaries_of_three_years_are_januaries(self):
        rows = [{"ORDER_MONTH": date(year, 1, 1), "REVENUE": 100.0 * n} for n, year in enumerate((2023, 2024, 2025), 1)]
        plan = {"date_disclosures": [{"requested_grain": "year"}]}
        card = _card(rows, "January revenue for the last 3 years")
        with _InLanguage("en"):
            card = build_assistant_response(question="January revenue for the last 3 years", rows=rows,
                                            sql="SELECT 1", duration_ms=1, semantic_plan=plan)
        assert card["answer"]["headline"] == "January 2025 closed at 300."
        assert "from January 2023 to January 2025" in card["insight_summary"]

    def test_the_first_quarters_of_three_years_are_first_quarters(self):
        rows = [{"PERIOD": date(year, 1, 1), "REVENUE": 100.0 * n} for n, year in enumerate((2023, 2024, 2025), 1)]
        plan = {"date_disclosures": [{"requested_grain": "year"}]}
        with _InLanguage("en"):
            card = build_assistant_response(question="Q1 revenue for each of the last 3 years", rows=rows,
                                            sql="SELECT 1", duration_ms=1, semantic_plan=plan)
        assert card["answer"]["headline"] == "Q1 2025 closed at 300."


class TestADateIsADateWhateverItsColumnIsCalled:
    """The books keep their own periods -- POSTING_PERIOD 202504, the fourth of
    the fiscal year -- and their keys are no calendar month. A date is one: a
    POSTING_MONTH of the first of each month is those months."""

    MONTHS = [{"REVENUE": 100.0 * month} for month in range(1, 5)]

    @pytest.mark.parametrize("column", ["POSTING_MONTH", "GL_MONTH", "ACCOUNTING_MONTH", "ACCT_MONTH",
                                        "POSTING_PERIOD", "FISCAL_MONTH_START", "POSTING_DATE", "GL_DATE",
                                        "WK_START"])
    def test_firsts_of_months_are_months(self, column):
        rows = [{column: date(2025, month, 1), **row} for month, row in enumerate(self.MONTHS, 1)]
        card = _card(rows, "net amount by month")
        assert card["answer"]["headline"] == "April 2025 closed at 400."
        assert card["data"]["display_formats"] == {column: {"type": "date", "style": "month_year_long"}}

    def test_in_french(self):
        rows = [{"MOIS_COMPTABLE": date(2025, month, 1), **row} for month, row in enumerate(self.MONTHS, 1)]
        assert _card(rows, "montant net par mois", "fr")["answer"]["headline"] == "Avril 2025 a terminé à 400."

    @pytest.mark.parametrize("column", ["POSTING_PERIOD", "GL_PERIOD", "ACCOUNTING_PERIOD", "PERIODE_FISCALE"])
    def test_their_keys_stay_their_digits(self, column):
        rows = [{column: 202500 + month, **row} for month, row in enumerate(self.MONTHS, 1)]
        card = _card(rows, "net amount by period")
        assert card["data"]["display_formats"] == {}
        assert card["answer"]["headline"] == "202504 closed at 400."


class TestADailyWordIsNoGrainOfMonthBuckets:
    """"Average daily sales by month" lists months, and so does "sales for the
    last 90 days" of a warehouse that keeps months: a month bucket read as a
    day was shown "2025-04-01"."""

    @pytest.mark.parametrize("question", [
        "average daily sales by month", "daily average revenue for each month", "weekly average sales by month",
        "sales for the last 90 days", "sales over the past 8 weeks", "average daily sales per month",
    ])
    def test_months_are_months(self, question):
        rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in range(1, 5)]
        card = _card(rows, question)
        assert card["answer"]["headline"] == "April 2025 closed at 400."

    def test_in_french(self):
        rows = [{"MOIS": date(2025, month, 1), "VENTES": 100.0 * month} for month in range(1, 5)]
        assert _card(rows, "ventes quotidiennes moyennes par mois", "fr")["answer"]["headline"] == \
            "Avril 2025 a terminé à 400."

    def test_dates_asked_by_the_day_that_are_not_a_months_stay_dates(self):
        rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (3, 6, 9)]
        assert _card(rows, "sales on the first of March, June and September by day")["data"]["display_formats"] == {}

    def test_days_asked_by_the_day_are_days(self):
        rows = [{"SALE_DATE": date(2025, 9, day), "REVENUE": 100.0 * day} for day in (1, 2, 3)]
        assert _card(rows, "sales for the last 3 days")["data"]["display_formats"] == {}


class TestADaysLabelIsNoMonthOfAnotherYear:
    """"Sep-30" is September 2030 or the 30th of September, and "Oct-01" the
    first of October or October 2001: only the grain asked for, or the
    column's own name, says which. Given neither, the label stays as the
    warehouse wrote it, and is never a month of a year a day is not."""

    @pytest.mark.parametrize("labels", [
        ("Sep-28", "Sep-29", "Sep-30", "Oct-01"),
        ("Sep-01", "Sep-02", "Sep-03"),
        ("Sep/28", "Sep/29", "Sep/30"),
        ("Jan-05", "Jan-12", "Jan-19", "Jan-26"),
    ])
    @pytest.mark.parametrize("column,question", [("PERIOD", "sales trend"), ("LABEL", "sales this month"),
                                                  ("SALE_DATE", "daily sales"), ("PERIOD", "sales for the last 3 days")])
    def test_the_label_is_left_as_written(self, labels, column, question):
        rows = [{column: label, "NET_SALES": 100.0 * n} for n, label in enumerate(labels, 1)]
        card = _card(rows, question)
        assert card["data"]["display_formats"] == {}
        said = f'{card["answer"]["headline"]} {card["insight_summary"]}'
        assert any(label in said for label in labels)
        assert not re.search(r"\b(?:19|20)\d\d\b", said), said

    @pytest.mark.parametrize("column,question", [("MONTH", "sales trend"), ("PERIOD", "sales by month"),
                                                  ("PERIOD", "monthly sales"), ("MONTH_NAME", "sales history")])
    def test_a_month_is_asked_for_or_named(self, column, question):
        rows = [{column: label, "NET_SALES": 100.0 * n} for n, label in enumerate(("Mar-25", "Apr-25", "May-25"), 1)]
        card = _card(rows, question)
        assert card["data"]["display_formats"] == {column: {"type": "date", "style": "month_year_long"}}
        assert card["answer"]["headline"] == "May 2025 closed at 300."

    def test_a_space_is_no_hyphen(self):
        from core.temporal_columns import parse_period_label

        assert parse_period_label("Sep 24") is None
        assert parse_period_label("Sep-24").year == 2024
        assert parse_period_label("Sep/24").year == 2024
        assert parse_period_label("September 2024").year == 2024

    def test_a_month_word_is_looked_up_whole(self):
        from core.temporal_columns import parse_period_label

        assert parse_period_label("Octane 87") is None and parse_period_label("Marseille 13") is None
        assert parse_period_label("Oct 2024").month == 10


class TestWhichUnitsAreNoRankedPeriod:
    """"Which months" asks what "which month" does, and plans its SQL by month.
    A measure counted in those units is no ranking of them: "what weeks of
    supply do we have by warehouse" asked for the weeks' worth of stock, and
    was planned by week."""

    @pytest.mark.parametrize("question,grain", [
        ("which months had the highest revenue", "month"),
        ("what weeks were the busiest", "week"),
        ("which quarters grew", "quarter"),
        ("which month of 2025 had the highest sales", "month"),
        ("what weeks of supply do we have by warehouse", ""),
        ("what days of supply do we have", ""),
        ("what years of service does each employee have", ""),
        ("what days sales outstanding do we have", ""),
        ("which days of the week are busiest", ""),
        ("which months of the year have the highest sales", ""),
        # A count of days, weeks or years that is a measure, or a cycle's unit.
        ("which days of week are busiest", ""),
        ("which days past due bucket has the most receivables", ""),
        ("what days late are our shipments", ""),
        ("which days overdue are most common", ""),
        ("what days to pay by customer", ""),
        ("which days to ship are longest", ""),
        ("what weeks on hand do we carry by item", ""),
        ("what days on hand by warehouse", ""),
        ("what months on hand of stock", ""),
        ("what months of age are our receivables", ""),
        ("what years of experience do our reps have", ""),
        ("what years in business do our customers have", ""),
        # What the verb after it says it is a listing of.
        ("which months had a loss", "month"),
        ("which months were below target", "month"),
        ("which quarters grew", "quarter"),
        ("which weeks had negative margin", "week"),
        ("which months in 2025 beat the target", "month"),
        ("which quarters of 2024 missed plan", "quarter"),
        ("which months during 2025 lost money", "month"),
    ])
    def test_the_period_a_question_ranks(self, question, grain):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(question) == grain

    def test_weeks_of_supply_is_planned_by_no_grain(self):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain("what weeks of supply do we have by warehouse") == ""


class TestAKeyInTheChatTable:
    """A key no calendar names is its digits in a chat's table too."""

    @pytest.mark.parametrize("rows,lang,digits", [
        ([{"WEEK_KEY": key, "REVENUE": 3000.0} for key in (202530, 202531)], "en", "202530"),
        ([{"FISCAL_PERIOD": key, "REVENUE": 1000.0} for key in (202510, 202513)], "en", "202513"),
        ([{"SEMAINE": key, "REVENUS": 3000.0} for key in (202530, 202531)], "fr", "202530"),
        ([{"PERIODE_FISCALE": key, "REVENUS": 100.0} for key in (202501, 202504)], "fr", "202501"),
    ])
    def test_it_is_never_grouped(self, rows, lang, digits):
        reply = _chat_reply(rows, "revenue by week" if lang == "en" else "revenus par semaine", lang)
        assert digits in reply
        assert f"{digits[:3]},{digits[3:]}" not in reply


class TestTheMonthsOfACutAreRankedInFrenchToo:

    def test_les_trois_meilleurs_mois_par_revenu(self):
        rows = [_SIX_MONTHS[2], _SIX_MONTHS[4], _SIX_MONTHS[5]]
        sql = ("SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(NET_AMT) AS REVENU FROM F GROUP BY ORDER_MONTH "
               "ORDER BY SUM(NET_AMT) DESC) t ORDER BY ORDER_MONTH")
        rows = [{"ORDER_MONTH": row["ORDER_MONTH"], "REVENU": row["REVENUE"]} for row in rows]
        card = _sql_card(rows, "les 3 meilleurs mois par revenu", sql, "fr")
        assert card["analysis_contract"]["mode"] == "ranking"
        assert "arrive en tête" in card["answer"]["headline"]


class TestTheChartMarksTheBiggestSwingWhereItHoldsThePeriod:
    """The brief names a period as the answer does, "February 2025"; a chart
    holds it as the warehouse sent it, "2025-02": the marker of the biggest drop
    and gain was dropped for every month the SQL wrote as yyyy-MM."""

    def test_a_month_written_yyyy_mm(self):
        from core.chart import build_chart_payload

        rows = [{"MONTH": f"2025-0{month}", "REVENUE": value}
                for month, value in ((1, 800.0), (2, 850.0), (3, 900.0), (4, 700.0), (5, 950.0))]
        annotations = {"biggest_period_drop": {"from_period": "March 2025", "to_period": "April 2025",
                                               "absolute_change": -200.0, "pct_change": -22.2},
                       "biggest_period_gain": {"from_period": "April 2025", "to_period": "May 2025",
                                               "absolute_change": 250.0, "pct_change": 35.7}}
        payload = build_chart_payload(rows, "line", title="revenue by month", annotations=annotations)
        assert payload["annotations"]["biggest_period_drop"]["period"] == "2025-04"
        assert payload["annotations"]["biggest_period_gain"]["period"] == "2025-05"


class TestARankingIsOfThePeriodsOnlyWhereTheQuestionAsksForOne:
    """A cut by a figure over the months ranks them only where the question
    asks for a ranking of them; and only the statement's outermost ORDER BY
    lists what the reader sees."""

    CUT = ("SELECT * FROM (SELECT TOP 6 ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH "
           "ORDER BY SUM(NET_AMT) DESC) t ORDER BY ORDER_MONTH")

    def test_a_cut_no_question_asks_a_ranking_of_is_a_series(self):
        card = _sql_card(_SIX_MONTHS, "revenue by month", self.CUT)
        assert card["analysis_contract"]["mode"] == "time_series"

    def test_the_same_cut_where_a_ranking_is_asked(self):
        card = _sql_card(_SIX_MONTHS, "the 6 best months by revenue", self.CUT)
        assert card["analysis_contract"]["mode"] == "ranking"

    def test_an_inner_order_by_a_figure_lists_nothing_the_reader_sees(self):
        sql = ("SELECT ORDER_MONTH, REVENUE FROM (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE, "
               "ROW_NUMBER() OVER (ORDER BY SUM(NET_AMT) DESC) AS RN FROM F GROUP BY ORDER_MONTH) t "
               "ORDER BY ORDER_MONTH")
        assert _sql_card(_SIX_MONTHS, "revenue by month", sql)["analysis_contract"]["mode"] == "time_series"

    def test_the_outermost_order_by_a_figure_ranks(self):
        sql = "SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH ORDER BY SUM(NET_AMT) DESC"
        rows = sorted(_SIX_MONTHS, key=lambda row: -row["REVENUE"])
        assert _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"] == "ranking"

    def test_a_position_or_an_alias_of_a_figure_ranks_too(self):
        rows = sorted(_SIX_MONTHS, key=lambda row: -row["REVENUE"])
        for order in ("ORDER BY 2 DESC", "ORDER BY REVENUE DESC"):
            sql = f"SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH {order}"
            assert _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"] == "ranking", order


class TestThePortalReadsWhatTheServerSentAsEnglish:
    """A percentage, a currency amount and a plain number are all sent the
    English way whatever the page's language: the French page reads them so and
    writes them its own way."""

    FUNCTIONS = TestThePortalTableInFrench.FUNCTIONS + ["function _qbMoney(body, symbol)"]

    @pytest.mark.parametrize("lang,expected", [("en", ["12.5%", "1,250.5%"]), ("fr", ["12,5 %", "1 250,5 %"])])
    def test_a_percentage(self, lang, expected):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify([_formatDisplayValue('12.5', 'percentage', {fraction_digits: 1}, 'MARGIN_PCT'),"
            " _formatDisplayValue('1,250.5', 'percentage', {fraction_digits: 1}, 'MARGIN_PCT')]);",
            lang=lang, functions=self.FUNCTIONS, consts=["_QB_CURRENCY_SYMBOL"])
        assert [value.replace(" ", " ").replace(" ", " ") for value in got] == expected

    def test_a_column_holds_numbers_where_any_cell_does(self):
        pytest.importorskip("dukpy")
        from tests.chat_js import run

        got = run(
            "JSON.stringify([_isANumericColumn([{R: '12,000'}, {R: null}], 'R'),"
            " _isANumericColumn([{R: '—'}, {R: '12'}], 'R'), _isANumericColumn([{R: 'abc'}, {R: 'def'}], 'R'),"
            " _isANumericColumn([{R: '1 234,5'}], 'R')]);",
            lang="fr", functions=TestThePortalTableInFrench.FUNCTIONS)
        assert got[:3] == [True, True, False]


class TestASeriesIsReadInTimeWhereverItIsRead:

    def test_the_context_reads_the_first_and_last_period_in_time(self):
        rows = [{"MOIS": f"{name} 2025", "VENTES": value}
                for name, value in (("avril", 950.0), ("mars", 900.0), ("février", 850.0), ("janvier", 800.0))]
        with _InLanguage("fr"):
            context = summarize_result_context(rows, "ventes par mois")
        assert context["mode"] == "time_series"
        assert (context["first_label"], context["last_label"]) == ("janvier 2025", "avril 2025")
        assert context["pct_change"] == pytest.approx(18.75, abs=0.01)
        # What the forecast and the analysis read is in the same order.
        assert context["labels"] == ["janvier 2025", "février 2025", "mars 2025", "avril 2025"]
        assert context["values"] == [800.0, 850.0, 900.0, 950.0]

    def test_a_label_a_day_reads_as_a_month_is_left_in_the_order_it_came(self):
        from core.response_builder import in_time_order

        labels = ["Sep-28", "Sep-29", "Sep-30", "Oct-01"]
        assert in_time_order(labels, [1, 2, 3, 4]) == (labels, [1, 2, 3, 4])
        assert in_time_order(["April 2025", "March 2025"], [2, 1]) == (["March 2025", "April 2025"], [1, 2])
        assert in_time_order(["Alpha", "Beta"], [2, 1]) == (["Alpha", "Beta"], [2, 1])

    def test_a_days_labels_keep_the_order_the_rows_came_in_the_brief(self):
        from core.insight import compute_data_brief

        rows = [{"PERIOD": label, "NET_SALES": 100.0 * n}
                for n, label in enumerate(("Sep-28", "Sep-29", "Sep-30", "Oct-01"), 1)]
        series = compute_data_brief(rows, "sales trend").get("time_series") or {}
        assert series.get("first_period") in (None, "Sep-28")

    def test_a_days_column_keeps_its_labels_even_where_months_are_asked(self):
        rows = [{"SALE_DATE": label, "NET_SALES": 100.0 * n} for n, label in enumerate(("Mar-25", "Apr-25", "May-25"), 1)]
        assert _card(rows, "sales by month")["data"]["display_formats"] == {}


# ── Round 5: a list by a date, by a window or by a sort key is no ranking ────────

_CLIMB = (80.0, 85.0, 90.0, 70.0, 95.0, 100.0)
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June"]


def _plain(text):
    """A French figure's and percentage's spaces, as ordinary ones."""
    return str(text).replace("\xa0", " ").replace("\u202f", " ")


def _a_climb_read_as_a_series(card, labels):
    """Six months that rose 25% with a drop in March and a gain in April."""
    first, last = labels[0], labels[-1]
    assert card["analysis_contract"]["mode"] == "time_series"
    assert card["answer"]["headline"] == f"{last} closed at 100."
    assert f"Revenue trended up 25.0% from {first} to {last}." in card["insight_summary"]
    assert [callout["message"] for callout in card["anomaly_callouts"]] == [
        f"Biggest drop: {labels[2]} → {labels[3]} (-22.2%)", f"Biggest gain: {labels[3]} → {labels[4]} (+35.7%)"]


class TestMonthsListedByTheirFirstDateWhereNoLabelSaysTheYear:
    """ORDER BY MIN(order_date) lists the months in time, and a label that
    carries no year -- "January", "Week 1", "FY25 Q1" -- cannot be placed in
    it: the rows say nothing of a listing by an amount. Read as a figure it
    made six months that rose "June leads at 100", with no trend, no callouts
    and "Mode: ranking" in the analyst's prompt -- while the chart, built with
    no SQL, drew the markers the card had dropped."""

    MONTH_SQL = ("SELECT DATENAME(month, order_date) AS MONTH_NAME, SUM(amount) AS REVENUE FROM orders "
                 "WHERE YEAR(order_date) = 2025 GROUP BY DATENAME(month, order_date) ORDER BY {order}")

    @staticmethod
    def _rows(labels, column="MONTH_NAME", measure="REVENUE", figures=_CLIMB):
        return [{column: label, measure: value} for label, value in zip(labels, figures)]

    @pytest.mark.parametrize("order", ["MIN(order_date)", "MAX(order_date) ASC", "MONTH(MIN(order_date))",
                                       "MIN(MONTH(order_date))", "MIN(ORDERDATE)"])
    def test_month_names(self, order):
        card = _sql_card(self._rows(_MONTH_NAMES), "monthly revenue this year", self.MONTH_SQL.format(order=order))
        _a_climb_read_as_a_series(card, _MONTH_NAMES)

    def test_month_abbreviations(self):
        labels = [month[:3] for month in _MONTH_NAMES]
        sql = ("SELECT LEFT(DATENAME(month, order_date), 3) AS MONTH_NAME, SUM(amount) AS REVENUE FROM orders "
               "GROUP BY LEFT(DATENAME(month, order_date), 3) ORDER BY MIN(MONTH(order_date))")
        _a_climb_read_as_a_series(_sql_card(self._rows(labels), "revenue by month", sql), labels)

    def test_week_numbers(self):
        labels = [f"Week {number}" for number in range(1, 7)]
        sql = ("SELECT CONCAT('Week ', DATEPART(week, order_date)) AS WEEK, SUM(amount) AS REVENUE FROM orders "
               "GROUP BY CONCAT('Week ', DATEPART(week, order_date)) ORDER BY MIN(order_date)")
        _a_climb_read_as_a_series(_sql_card(self._rows(labels, "WEEK"), "weekly revenue", sql), labels)

    @pytest.mark.parametrize("order", ["MIN(order_date)", "MAX(OrderDate)", "MIN(ORDERDATE)", "MIN(DOCDATE)", "MIN(ship_dt)",
                                       "MAX(created_at)", "MIN(TXN_TS)", "MIN(INVOICE_DATETIME)", "MIN(DATEKEY)",
                                       "MIN(t.order_date)", "MAX(CreatedAt)", "MIN(ShipDt)"])
    def test_days_written_mon_dd_across_two_months(self, order):
        """"Sep-27" reads as a month of 2027 and "Oct-01" of 2001: the labels
        are placed out of time, and the SQL that lists them by a date says they
        are listed in it."""
        labels = ["Sep-27", "Sep-28", "Sep-29", "Sep-30", "Oct-01", "Oct-02"]
        sql = ("SELECT FORMAT(order_date, 'MMM-dd') AS SALE_DAY, SUM(amount) AS REVENUE FROM orders t "
               f"GROUP BY FORMAT(order_date, 'MMM-dd') ORDER BY {order}")
        card = _sql_card(self._rows(labels, "SALE_DAY"), "daily revenue", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert "leads" not in card["answer"]["headline"]

    @pytest.mark.parametrize("measure", ["lifetime_value", "candidate_score", "runtime_secs", "update_count", "mandate_amt"])
    def test_a_maximum_of_a_measure_that_only_looks_like_a_date_still_ranks(self, measure):
        rows = [{"PRD_KEY": 202500 + month, "TOP_VALUE": value} for month, value in ((2, 90.0), (3, 55.0), (1, 40.0))]
        sql = f"SELECT PRD_KEY, MAX({measure}) AS TOP_VALUE FROM F GROUP BY PRD_KEY ORDER BY MAX({measure}) DESC"
        card = _sql_card(rows, "which months had the highest value", sql)
        assert card["answer"]["headline"] == "February 2025 leads at 90."

    def test_two_months_are_a_series_too(self):
        sql = self.MONTH_SQL.format(order="MIN(order_date)")
        card = _sql_card(self._rows(_MONTH_NAMES[:2]), "monthly revenue this year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "February closed at 85."

    def test_in_french(self):
        labels = ["janvier", "février", "mars", "avril", "mai", "juin"]
        sql = ("SELECT FORMAT(date_vente, 'MMMM', 'fr-FR') AS MOIS, SUM(montant) AS VENTES FROM ventes "
               "WHERE YEAR(date_vente) = 2025 GROUP BY FORMAT(date_vente, 'MMMM', 'fr-FR') ORDER BY MIN(date_vente)")
        card = _sql_card(self._rows(labels, "MOIS", "VENTES"), "ventes par mois cette année", sql, "fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "Juin a terminé à 100."
        assert "a progressé de 25,0 % entre janvier et juin" in _plain(card["insight_summary"])
        assert [_plain(callout["message"]) for callout in card["anomaly_callouts"]] == [
            "Plus forte baisse : mars → avril (-22,2 %)", "Plus forte hausse : avril → mai (+35,7 %)"]

    @pytest.mark.parametrize("labels,column,question,sql,headline,summary", [
        (("FY25 Q1", "FY25 Q2", "FY25 Q3", "FY25 Q4"), "FISCAL_QUARTER", "revenue by fiscal quarter",
         "SELECT FISCAL_QUARTER, SUM(x) AS REVENUE FROM t GROUP BY FISCAL_QUARTER ORDER BY MIN(posting_date)",
         "FY25 Q4 closed at 70.", "Revenue trended down 12.5% from FY25 Q1 to FY25 Q4."),
        (("Jan '25", "Feb '25", "Mar '25", "Apr '25"), "MONTH", "monthly revenue",
         "SELECT FORMAT(d, 'MMM ''yy') AS MONTH, SUM(x) AS REVENUE FROM t GROUP BY FORMAT(d, 'MMM ''yy') ORDER BY MIN(d)",
         "Apr '25 closed at 70.", "Revenue trended down 12.5% from Jan '25 to Apr '25."),
        (("March 3, 2025", "March 4, 2025", "March 5, 2025", "March 6, 2025"), "SALE_DAY", "daily revenue",
         "SELECT FORMAT(d, 'MMMM d, yyyy') AS SALE_DAY, SUM(x) AS REVENUE FROM t GROUP BY FORMAT(d, 'MMMM d, yyyy') "
         "ORDER BY MIN(d)",
         "March 6, 2025 closed at 70.", "Revenue trended down 12.5% from March 3, 2025 to March 6, 2025."),
    ])
    def test_labels_no_calendar_places(self, labels, column, question, sql, headline, summary):
        rows = self._rows(labels, column, figures=(80.0, 85.0, 90.0, 70.0))
        card = _sql_card(rows, question, sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == headline
        assert summary in card["insight_summary"]

    def test_the_prompt_and_the_chart_read_it_the_same_way(self):
        from core.chart import build_chart_annotations
        from core.insight import describe_result_for_prompt

        rows = self._rows(_MONTH_NAMES)
        sql = self.MONTH_SQL.format(order="MIN(order_date)")
        assert "Mode: time_series" in describe_result_for_prompt(rows, "monthly revenue this year", sql)
        assert sorted(build_chart_annotations(rows, "monthly revenue this year")) == [
            "biggest_period_drop", "biggest_period_gain"]

    def test_month_names_listed_by_their_total_are_a_ranking(self):
        rows = sorted(self._rows(_MONTH_NAMES), key=lambda row: -row["REVENUE"])
        card = _sql_card(rows, "which months had the highest revenue", self.MONTH_SQL.format(order="SUM(amount) DESC"))
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "June leads at 100."


class TestTheTopOfEachPeriodIsASeriesOfThePeriods:
    """A window PARTITIONed by the period ranks the products inside each month
    and cuts none of the months: "top product by revenue each month" lists six
    months in time, and was read as a ranking of them -- "June 2025 leads at
    1,000", no trend, no callouts."""

    PRODUCTS = ("Pumps", "Valves", "Pumps", "Hoses", "Pumps", "Valves")
    ROWS = [{"ORDER_MONTH": row["ORDER_MONTH"], "TOP_PRODUCT": product, "REVENUE": row["REVENUE"]}
            for row, product in zip(_SIX_MONTHS, PRODUCTS)]

    @pytest.mark.parametrize("sql", [
        # A window ranking the products of each month, the one kept.
        "SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE FROM (SELECT DATEFROMPARTS(YEAR(order_date), MONTH(order_date), 1) "
        "AS ORDER_MONTH, product AS TOP_PRODUCT, SUM(amount) AS REVENUE, ROW_NUMBER() OVER "
        "(PARTITION BY DATEFROMPARTS(YEAR(order_date), MONTH(order_date), 1) ORDER BY SUM(amount) DESC) AS RN "
        "FROM orders GROUP BY DATEFROMPARTS(YEAR(order_date), MONTH(order_date), 1), product) t "
        "WHERE RN = 1 ORDER BY ORDER_MONTH",
        "WITH ranked AS (SELECT DATE_TRUNC('month', order_date) AS ORDER_MONTH, product AS TOP_PRODUCT, "
        "SUM(amount) AS REVENUE, RANK() OVER (PARTITION BY DATE_TRUNC('month', order_date) ORDER BY SUM(amount) DESC) "
        "AS RK FROM orders GROUP BY 1, 2) SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE FROM ranked WHERE RK = 1 "
        "ORDER BY ORDER_MONTH",
        "SELECT DATE_TRUNC('month', order_date) AS ORDER_MONTH, product AS TOP_PRODUCT, SUM(amount) AS REVENUE "
        "FROM orders GROUP BY 1, 2 QUALIFY ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('month', order_date) "
        "ORDER BY SUM(amount) DESC) = 1 ORDER BY ORDER_MONTH",
        "SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE FROM (SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE, DENSE_RANK() OVER "
        "(PARTITION BY ORDER_MONTH ORDER BY REVENUE DESC) AS DR FROM monthly) t WHERE DR = 1 ORDER BY ORDER_MONTH",
    ])
    def test_the_best_of_each_month(self, sql):
        card = _sql_card(self.ROWS, "top product by revenue each month", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "June 2025 closed at 1,000."
        assert "Revenue trended up 25.0% from January 2025 to June 2025" in card["insight_summary"]
        assert card["anomaly_callouts"]

    def test_the_months_totalled_from_each_months_top_customers(self):
        sql = ("WITH monthly AS (SELECT DATE_TRUNC('month', order_date) AS ORDER_MONTH, customer_id, "
               "SUM(amount) AS REVENUE, DENSE_RANK() OVER (PARTITION BY DATE_TRUNC('month', order_date) "
               "ORDER BY SUM(amount) DESC) AS DR FROM orders GROUP BY 1, 2) "
               "SELECT ORDER_MONTH, SUM(REVENUE) AS REVENUE FROM monthly WHERE DR <= 10 GROUP BY ORDER_MONTH "
               "ORDER BY ORDER_MONTH")
        card = _sql_card(_SIX_MONTHS, "monthly revenue from each month's top 10 customers", sql)
        assert card["analysis_contract"]["mode"] == "time_series"

    def test_in_french(self):
        sql = ("SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE FROM (SELECT ORDER_MONTH, TOP_PRODUCT, REVENUE, "
               "ROW_NUMBER() OVER (PARTITION BY ORDER_MONTH ORDER BY REVENUE DESC) AS RN FROM monthly) t "
               "WHERE RN = 1 ORDER BY ORDER_MONTH")
        card = _sql_card(self.ROWS, "meilleur produit par mois", sql, "fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert "a progressé de 25,0 %" in _plain(card["insight_summary"])

    @pytest.mark.parametrize("sql", [
        "SELECT ORDER_MONTH, REVENUE FROM (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE, DENSE_RANK() OVER "
        "(ORDER BY SUM(NET_AMT) DESC) AS DR FROM F GROUP BY ORDER_MONTH) t WHERE DR <= 3 ORDER BY ORDER_MONTH",
        # The second key of an ORDER BY breaks ties; the first ranks.
        "SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH "
        "ORDER BY SUM(NET_AMT) DESC, ORDER_MONTH) t ORDER BY ORDER_MONTH",
        "SELECT ORDER_MONTH, REVENUE FROM (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE, ROW_NUMBER() OVER "
        "(ORDER BY SUM(NET_AMT) DESC, ORDER_MONTH) AS RN FROM F GROUP BY ORDER_MONTH) t WHERE RN <= 3 "
        "ORDER BY ORDER_MONTH",
    ])
    def test_a_window_with_no_partition_ranks_the_months_themselves(self, sql):
        rows = sorted([_SIX_MONTHS[2], _SIX_MONTHS[4], _SIX_MONTHS[5]], key=lambda row: row["ORDER_MONTH"])
        card = _sql_card(rows, "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "June 2025 leads at 1,000."


_QUARTER_FIRSTS = [date(2024, month, 1) for month in (1, 4, 7, 10)]
_JANUARIES = [date(year, 1, 1) for year in (2022, 2023, 2024, 2025)]


def _with_plan(grain):
    return {"date_disclosures": [{"label": "Order date", "requested_grain": grain}]} if grain else None


class TestAMonthOrQuarterNamedAsTheEdgeOfAWindowNamesNoPeriod:
    """"Revenue since January 2024" starts a window in January; it does not say
    that its periods are Januaries. A column of quarters or years keeps what it
    is: "October 2024" for the last of four quarters, and "January 2025" for a
    year's total, were the round-4 names."""

    @pytest.mark.parametrize("column,keys,question,plan,lang,headline", [
        ("QUARTER_START", _QUARTER_FIRSTS, "revenue since January 2024", "", "en", "Q4 2024 closed at 400."),
        ("QUARTER", _QUARTER_FIRSTS, "revenue trend since January 2024", "", "en", "Q4 2024 closed at 400."),
        ("PERIOD", _QUARTER_FIRSTS, "revenue since January 2024", "quarter", "en", "Q4 2024 closed at 400."),
        ("QUARTER_START", _QUARTER_FIRSTS, "revenue since Sept 2023", "", "en", "Q4 2024 closed at 400."),
        ("TRIMESTRE", _QUARTER_FIRSTS, "chiffre d'affaires depuis janvier 2024", "", "fr", "T4 2024 a terminé à 400."),
        ("ORDER_YEAR", _JANUARIES, "revenue since January 2022", "", "en", "2025 closed at 400."),
        ("PERIOD", _JANUARIES, "revenue since Jan 2022", "year", "en", "2025 closed at 400."),
        ("ORDER_YEAR", _JANUARIES, "revenue since Q3 2021", "", "en", "2025 closed at 400."),
        ("SALES_YEAR", _JANUARIES[1:], "compare Jan-Jun revenue for 2023, 2024 and 2025", "", "en",
         "2025 closed at 400."),
        ("ORDER_YEAR", _JANUARIES, "revenue through December 2025", "", "en", "2025 closed at 400."),
        # Words that are only a month's look-alike.
        ("QUARTER_START", _QUARTER_FIRSTS, "what may explain the revenue trend", "", "en", "Q4 2024 closed at 400."),
        ("QUARTER_START", _QUARTER_FIRSTS, "how does revenue march upward over time", "", "en",
         "Q4 2024 closed at 400."),
        ("QUARTER_START", _QUARTER_FIRSTS, "what did Jan sell over time", "", "en", "Q4 2024 closed at 400."),
        ("QUARTER_START", _QUARTER_FIRSTS, "revenue with 2 dec places over time", "", "en", "Q4 2024 closed at 400."),
    ])
    def test_the_periods_are_those_the_column_holds(self, column, keys, question, plan, lang, headline):
        rows = [{column: key, "REVENUE" if lang == "en" else "VENTES": 100.0 * n} for n, key in enumerate(keys, 1)]
        card = _sql_card(rows, question, "SELECT 1", lang, semantic_plan=_with_plan(plan))
        assert card["answer"]["headline"] == headline.replace("closed at 400", f"closed at {100 * len(keys)}")

    @pytest.mark.parametrize("column,question", [
        ("QUARTER_START", "revenue for the last 3 years"), ("PERIOD", "revenue for the last 3 years"),
        ("QUARTER_START", "revenue by year"), ("ORDER_QUARTER", "yearly revenue trend"),
    ])
    def test_a_year_asked_of_quarters_leaves_them_quarters(self, column, question):
        firsts = [date(year, month, 1) for year in (2023, 2024) for month in (1, 4, 7, 10)]
        rows = [{column: key, "REVENUE": 100.0 * n} for n, key in enumerate(firsts, 1)]
        card = _sql_card(rows, question, "SELECT 1", semantic_plan=_with_plan("year"))
        assert card["answer"]["headline"] == "Q4 2024 closed at 800."

    @pytest.mark.parametrize("question,plan,named", [
        ("January revenue for the last 4 years", "year", "January 2025"),
        ("Jan revenue for each of the last 4 years", "year", "January 2025"),
        ("revenue in January for each of the last 4 years", "year", "January 2025"),
        ("Q1 revenue for each of the last 4 years", "year", "Q1 2025"),
    ])
    def test_a_month_the_question_names_as_what_its_periods_are_still_names_them(self, question, plan, named):
        rows = [{"PERIOD": key, "REVENUE": 100.0 * n} for n, key in enumerate(_JANUARIES, 1)]
        card = _sql_card(rows, question, "SELECT 1", semantic_plan=_with_plan(plan))
        assert card["answer"]["headline"] == f"{named} closed at 400."

    @pytest.mark.parametrize("question,named", [
        ("sales in May for each of the last 3 years", "May 2025"),
        ("sales during May for the last 3 years", "May 2025"),
    ])
    def test_may_is_a_month_where_a_preposition_says_so(self, question, named):
        rows = [{"PERIOD": date(year, 5, 1), "SALES": 100.0 * n} for n, year in enumerate((2023, 2024, 2025), 1)]
        card = _sql_card(rows, question, "SELECT 1", semantic_plan=_with_plan("year"))
        assert card["answer"]["headline"] == f"{named} closed at 300."

    @pytest.mark.parametrize("question,grain", [
        ("sales in May for each of the last 3 years", "year|month:5"),
        ("Sept sales for each of the last 3 years", "year|month:9"),
        ("Aug revenue for the last 3 years", "year|month:8"),
        ("Feb revenue for each of the last 3 years", "year|month:2"),
        ("revenue in the third quarter of the last 3 years", "year|quarter:3"),
        # A month that is only the edge of a window, or one of several, names no period.
        ("what may explain the revenue trend", "year"),
        ("revenue since January 2024", "year"),
        ("January 2024 revenue by year", "year"),
        ("revenue from March for the last 3 years", "year"),
        ("Jan-Jun revenue for the last 3 years", "year"),
        ("Mar-Jun revenue for the last 3 years", "year"),
        ("revenue for January and March of the last 3 years", "year"),
        ("January revenue versus July revenue for the last 3 years", "year"),
        ("Q1 revenue versus Q3 revenue for each of the last 3 years", "year"),
        ("revenue since Q3 2021", "year"),
    ])
    def test_which_month_a_question_names(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan("year")) == grain

    @pytest.mark.parametrize("firsts,named,grain", [
        ([date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)], "year|month:1", "month"),
        ([date(2023, 1, 1), date(2025, 1, 1)], "year|month:1", "month"),
        ([date(2023, 4, 1), date(2024, 4, 1), date(2025, 4, 1)], "year|quarter:2", "quarter"),
        # Firsts that are not all the month named are not "that month each year".
        ([date(2023, 1, 1), date(2024, 4, 1), date(2025, 7, 1)], "year|month:1", ""),
        ([date(2023, 1, 1), date(2024, 1, 1), date(2025, 4, 1)], "year|month:1", ""),
    ])
    def test_the_firsts_a_named_month_names(self, firsts, named, grain):
        from core.response_builder import period_grain

        assert period_grain(firsts, named, "PERIOD") == grain

    def test_a_quarter_asked_of_months_leaves_them_months(self):
        rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (1, 2, 3)]
        card = _card(rows, "revenue by quarter")
        assert card["answer"]["headline"] == "March 2025 closed at 300."
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": "month_year_long"}}


class TestADaysLabelIsNoMonthWhereTheQuestionOnlyNamesOne:
    """"Sep-24" of a question that names September is the 24th of it, and
    "Oct-01" of "revenue for October" the first: a month named is no grain of
    the label, and it was read as "September 2027"."""

    @pytest.mark.parametrize("column,labels,question,last", [
        ("PERIOD", ("Sep-24", "Sep-25", "Sep-26", "Sep-27"), "sales in September", "Sep-27"),
        ("SALE_DAY", ("Sep-24", "Sep-25", "Sep-26", "Sep-27"), "sales in September", "Sep-27"),
        ("LABEL", ("Oct-01", "Oct-02", "Oct-03", "Oct-04"), "revenue for October", "Oct-04"),
        ("PERIOD", ("Jul-01", "Jul-08", "Jul-15", "Jul-22"), "Q3 sales", "Jul-22"),
    ])
    def test_the_label_is_left_as_written(self, column, labels, question, last):
        rows = [{column: label, "SALES": 100.0 * n} for n, label in enumerate(labels, 1)]
        card = _card(rows, question)
        assert card["data"]["display_formats"] == {}
        assert card["answer"]["headline"] == f"{last} closed at 400."
        assert not re.search(r"\b(?:19|20)\d\d\b", card["insight_summary"])

    @pytest.mark.parametrize("labels,question,style,headline", [
        (("Jan-25", "Apr-25", "Jul-25", "Oct-25"), "sales by quarter", "quarter", "Q4 2025 closed at 400."),
        (("Jan-22", "Jan-23", "Jan-24", "Jan-25"), "sales by year", "year", "2025 closed at 400."),
        (("Jan-25", "Feb-25", "Mar-25", "Apr-25"), "sales by month", "month_year_long", "April 2025 closed at 400."),
    ])
    def test_a_grouping_asked_for_says_what_the_label_is(self, labels, question, style, headline):
        rows = [{"PERIOD": label, "SALES": 100.0 * n} for n, label in enumerate(labels, 1)]
        card = _card(rows, question)
        assert card["data"]["display_formats"] == {"PERIOD": {"type": "date", "style": style}}
        assert card["answer"]["headline"] == headline


class TestAFrenchDayIsAPeriodOfADailySeries:
    """"1 mars 2025" is a day: a daily series of them was a ranking -- "4 mars
    2025 arrive en tête avec 400, sur 4 jours" -- once a French month counted
    only as a whole label."""

    DAYS = (("1 mars 2025", "2 mars 2025", "3 mars 2025", "4 mars 2025"),
            ("1er mars 2025", "2 mars 2025", "3 mars 2025", "4 mars 2025"),
            ("jeudi 27 février 2025", "vendredi 28 février 2025", "samedi 1 mars 2025", "dimanche 2 mars 2025"))

    @pytest.mark.parametrize("labels", DAYS)
    @pytest.mark.parametrize("column,question", [("JOUR", "ventes par jour"), ("DATE_VENTE", "ventes quotidiennes")])
    def test_a_daily_series(self, labels, column, question):
        rows = [{column: label, "VENTES": 100.0 * n} for n, label in enumerate(labels, 1)]
        card = _card(rows, question, "fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == f"{labels[-1][:1].upper()}{labels[-1][1:]} a terminé à 400."
        assert "a progressé de 300,0 % entre" in _plain(card["insight_summary"])

    @pytest.mark.parametrize("labels", DAYS)
    def test_the_analysts_brief_reads_it_so_too(self, labels):
        from core.insight import compute_data_brief

        rows = [{"JOUR": label, "VENTES": 100.0 * n} for n, label in enumerate(labels, 1)]
        with _InLanguage("fr"):
            assert compute_data_brief(rows, "ventes par jour")["mode"] == "time_series"

    @pytest.mark.parametrize("labels,expected", [
        (["janvier", "février", "mars", "avril"], True),
        (["1 mars 2025", "2 mars 2025"], True),
        (["jeudi 4 mars 2025", "vendredi 5 mars 2025"], True),
        (["Mars Bar", "Twix"], False),
        (["Mars", "Nestle", "Lindt"], False),
        (["12 Rue de Juin 2025", "Rue Haute"], False),
    ])
    def test_the_brief_places_a_whole_french_label_and_no_brand(self, labels, expected):
        from core.insight import _looks_temporal

        assert _looks_temporal(labels) is expected

    @pytest.mark.parametrize("label,expected", [
        ("1 mars 2025", True), ("1er mars 2025", True), ("jeudi 4 mars 2025", True), ("31 déc. 2025", True),
        ("mars 2025", False), ("Mars Bar", False), ("5 Mars Bars 2025", False), ("3 mars", False), ("12 Mars", False),
    ])
    def test_a_french_day_is_a_label_that_is_one_and_nothing_else(self, label, expected):
        from core.temporal_columns import FRENCH_DAY_LABEL_RE

        assert bool(FRENCH_DAY_LABEL_RE.fullmatch(label)) is expected


class TestTheChartKeepsItsMarkersWhereTheBriefNamesThePeriodsByTheQuestionsGrain:
    """The brief names the Januaries of "January revenue for the last 4 years" as
    the question does, "January 2024"; the chart holds them as "2024-01-01" and
    looks for them by the same grain. Looked up with none, it dropped the
    markers of the biggest drop and gain the card called out."""

    ROWS = [{"PERIOD": date(year, 1, 1), "REVENUE": value}
            for year, value in zip((2022, 2023, 2024, 2025), (100.0, 300.0, 200.0, 400.0))]

    @pytest.mark.parametrize("question", [
        "January revenue for the last 4 years", "Q1 revenue for each of the last 4 years", "revenue by month",
        "revenue by quarter", "revenue since January 2022",
    ])
    def test_the_markers_are_where_the_periods_are(self, question):
        from core.chart import build_chart_annotations, build_chart_payload

        brief = build_chart_annotations(self.ROWS, question)
        assert sorted(brief) == ["biggest_period_drop", "biggest_period_gain"]
        payload = build_chart_payload(self.ROWS, "line", title=question, question=question, annotations=brief)
        placed = {name: (marker or {}).get("period") for name, marker in (payload.get("annotations") or {}).items()}
        assert placed == {"biggest_period_drop": "2024-01-01", "biggest_period_gain": "2023-01-01"}


class TestAUnitCountIsNoGrainInFrenchToo:
    """"Quels jours de la semaine" is canonicalised to "which days of week",
    with no "the" for the exclusion of a cycle to see."""

    @pytest.mark.parametrize("question", [
        "quels jours de la semaine sont les plus chargés", "quels jours de la semaine livrons-nous",
    ])
    def test_a_cycle_asked_in_french_plans_no_grain(self, question):
        from core.contextual_dates import ranked_period_grain, requested_temporal_grain
        from core.question_normalizer import canonical_question

        canonical = canonical_question(question, "fr")
        assert "days of week" in canonical
        assert ranked_period_grain(canonical) == "" and requested_temporal_grain(canonical) == ""

    @pytest.mark.parametrize("question", [
        "which days of week are busiest", "which days past due bucket has the most receivables",
        "what years of experience do our reps have", "what weeks on hand do we carry by item",
        "what days on hand by warehouse", "what months of age are our receivables",
        "quels jours de la semaine sont les plus chargés",
    ])
    def test_their_date_plan_carries_no_grain(self, question):
        from core.contextual_dates import build_contextual_date_plan
        from core.question_normalizer import canonical_question

        binding = {"fact_table": "DW.FACT_SALES", "fact_column": "ORDER_DATE_KEY",
                   "date_key_type": "yyyymmdd_integer", "context_name": "Order date", "governance_status": "approved"}
        lang = "fr" if question.startswith("quels") else "en"
        plan = build_contextual_date_plan(binding, canonical_question(question, lang))
        assert not [policy for policy in plan.get("temporal_policies") or [] if policy.get("requested_grain")]


class TestAWeekOrFiscalPeriodKeyWrittenYyyyNnIsNoCalendarMonth:
    """"2025-01" of a week column is the first week of 2025, and of a fiscal
    period the first period of the year: sent as "2025-01-01" and named
    "January 2025", eight weeks became eight months."""

    KEYS = [f"2025-0{number}" for number in range(1, 9)]

    @pytest.mark.parametrize("column,question,lang,headline", [
        ("YEAR_WEEK", "weekly sales for the last 8 weeks", "en", "2025-08 closed at 800."),
        ("WEEK", "sales by week", "en", "2025-08 closed at 800."),
        ("FISCAL_WEEK", "revenue by fiscal week", "en", "2025-08 closed at 800."),
        ("FISCAL_PERIOD", "revenue by fiscal period", "en", "2025-08 closed at 800."),
        ("POSTING_PERIOD", "net amount by posting period", "en", "2025-08 closed at 800."),
        ("SEMAINE", "ventes par semaine", "fr", "2025-08 a terminé à 800."),
        ("PERIODE_FISCALE", "revenus par période fiscale", "fr", "2025-08 a terminé à 800."),
    ])
    def test_the_key_is_shown_as_written(self, column, question, lang, headline):
        rows = [{column: key, "REVENUE": 100.0 * number} for number, key in enumerate(self.KEYS, 1)]
        card = _card(rows, question, lang)
        assert card["data"]["display_formats"] == {}
        assert [row[column] for row in card["data"]["rows"]] == self.KEYS
        assert card["answer"]["headline"] == headline

    @pytest.mark.parametrize("column,key,lang,question,headline", [
        ("WEEK_KEY", 202530, "en", "which week had the peak", "Week Key: 202530"),
        ("FISCAL_PERIOD", 202513, "en", "which fiscal period was the last", "Fiscal Period: 202513"),
        ("SEMAINE", 202530, "fr", "quelle semaine était la meilleure", "Semaine : 202530"),
        ("YEAR_WEEK", "2025-08", "en", "which week was busiest", "Year Week: 2025-08"),
    ])
    def test_an_answer_that_is_one_key(self, column, key, lang, question, headline):
        card = _card([{column: key}], question, lang)
        assert card["answer"]["headline"] == headline
        assert card["insight_summary"] == f"{headline}."
        assert card["kpi"]["value"] == str(key)

    def test_in_the_chat_table_too(self):
        rows = [{"YEAR_WEEK": key, "REVENUE": 100.0 * number} for number, key in enumerate(self.KEYS, 1)]
        reply = _chat_reply(rows, "weekly sales for the last 8 weeks")
        assert "2025-08" in reply and "August 2025" not in reply

    @pytest.mark.parametrize("column", ["FISCAL_WEEK", "WEEK_START", "POSTING_PERIOD", "FISCAL_PERIOD"])
    def test_a_real_date_is_a_date_whatever_its_column_is_called(self, column):
        rows = [{column: date(2025, month, 1), "REVENUE": 100.0 * month} for month in range(1, 5)]
        card = _card(rows, "revenue by month")
        assert card["answer"]["headline"] == "April 2025 closed at 400."
        assert card["data"]["display_formats"] == {column: {"type": "date", "style": "month_year_long"}}


class TestASortKeyOfTheRowsIsNoFigureToRankThemBy:
    """ORDER BY SORT_ORDER lists the months in the order a calendar table keeps
    them: a column that counts the rows 1 to 6 is a sort key, and "June leads
    Sort Order at 6" was a ranking of months by the number of the month."""

    SQL = ("SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, MONTH(d) AS {column} FROM t "
           "GROUP BY DATENAME(month, d), MONTH(d) ORDER BY {column}")

    @pytest.mark.parametrize("column", ["SORT_ORDER", "MNTH", "RN", "MONTH_SEQ", "MONTH_NUM", "MONTH_NO"])
    def test_months_listed_by_their_number(self, column):
        rows = [{"MONTH_NAME": month, "REVENUE": value, column: number}
                for number, (month, value) in enumerate(zip(_MONTH_NAMES, _CLIMB), 1)]
        card = _sql_card(rows, "monthly revenue this year", self.SQL.format(column=column))
        _a_climb_read_as_a_series(card, _MONTH_NAMES)

    def test_a_number_that_counts_from_zero_is_a_sort_key_too(self):
        rows = [{"MONTH_NAME": month, "REVENUE": value, "SORT_ORDER": number}
                for number, (month, value) in enumerate(zip(_MONTH_NAMES, _CLIMB))]
        card = _sql_card(rows, "monthly revenue this year", self.SQL.format(column="SORT_ORDER"))
        assert card["analysis_contract"]["mode"] == "time_series"

    def test_a_position_of_a_sort_key_is_no_figure_either(self):
        rows = [{"MONTH_NAME": month, "REVENUE": value, "SORT_ORDER": number}
                for number, (month, value) in enumerate(zip(_MONTH_NAMES, _CLIMB), 1)]
        sql = ("SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, MONTH(d) AS SORT_ORDER FROM t "
               "GROUP BY DATENAME(month, d), MONTH(d) ORDER BY 3")
        assert _sql_card(rows, "monthly revenue this year", sql)["analysis_contract"]["mode"] == "time_series"

    def test_two_rows_counted_one_and_two_are_figures_not_a_sort_key(self):
        rows = [{"MONTH_NAME": "March", "REVENUE": 90.0, "ORDERS": 2}, {"MONTH_NAME": "January", "REVENUE": 80.0, "ORDERS": 1}]
        sql = ("SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t "
               "GROUP BY DATENAME(month, d) ORDER BY ORDERS DESC")
        card = _sql_card(rows, "which 2 months had the most orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("order", ["ORDERS DESC", "2 DESC"])
    def test_a_count_that_is_all_the_result_measures_is_what_it_ranks_by(self, order):
        """Three months counted 3, 2 and 1 orders, listed by the count, are
        no sort key: it is the only figure they have."""
        rows = [{"MONTH_NAME": month, "ORDERS": count} for month, count in (("March", 3), ("January", 2), ("June", 1))]
        sql = ("SELECT DATENAME(month, d) AS MONTH_NAME, COUNT(*) AS ORDERS FROM t "
               f"GROUP BY DATENAME(month, d) ORDER BY {order}")
        card = _sql_card(rows, "which 3 months had the most orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "March leads at 3."

    def test_a_count_of_the_months_orders_is_a_figure(self):
        orders = (14, 12, 11, 9, 6, 3)
        rows = [{"MONTH_NAME": month, "REVENUE": value, "ORDERS": count}
                for month, value, count in zip(_MONTH_NAMES, _CLIMB, orders)]
        sql = ("SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t "
               "GROUP BY DATENAME(month, d) ORDER BY ORDERS DESC")
        card = _sql_card(rows, "which months had the most orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"


class TestAFiscalYearStoredAsADateIsNoOneMonth:
    """A fiscal year that starts in July is stored as 2024-07-01: a year, whose
    date is the date, and no "July 2024" -- the month it is not."""

    @pytest.mark.parametrize("column,month,lang,question,headline", [
        ("FISCAL_YEAR", 7, "en", "revenue by fiscal year", "2024-07-01 closed at 400."),
        ("FY_START", 7, "en", "revenue by fiscal year", "2024-07-01 closed at 400."),
        ("FISCAL_YEAR_START", 7, "en", "revenue by fiscal year", "2024-07-01 closed at 400."),
        ("ACCOUNTING_YEAR", 4, "en", "revenue by accounting year", "2024-04-01 closed at 400."),
        ("EXERCICE", 7, "fr", "ventes par exercice", "2024-07-01 a terminé à 400."),
    ])
    def test_the_date_stays_the_date(self, column, month, lang, question, headline):
        rows = [{column: date(year, month, 1), "REVENUE" if lang == "en" else "VENTES": 100.0 * number}
                for number, year in enumerate((2021, 2022, 2023, 2024), 1)]
        card = _card(rows, question, lang)
        assert card["answer"]["headline"] == headline
        assert card["data"]["display_formats"] == {}

    @pytest.mark.parametrize("column", ["FISCAL_YEAR", "EXERCICE", "ACCOUNTING_YEAR"])
    def test_a_fiscal_year_of_januaries_or_whole_years_is_a_year(self, column):
        for keys in ([date(year, 1, 1) for year in (2022, 2023, 2024)], [2022, 2023, 2024]):
            rows = [{column: key, "REVENUE": 100.0 * number} for number, key in enumerate(keys, 1)]
            assert _card(rows, "revenue by fiscal year")["answer"]["headline"] == "2024 closed at 300."


class TestHowAStatementIsReadForTheCutOfItsPeriods:
    """Where the periods are cut by a figure inside the statement, in the
    dialects a warehouse writes, through a star over a join, and only where the
    question asks for a ranking of them."""

    TOP_THREE = sorted([_SIX_MONTHS[2], _SIX_MONTHS[4], _SIX_MONTHS[5]], key=lambda row: row["ORDER_MONTH"])

    def test_a_star_over_a_joined_source_hands_on_its_columns(self):
        sql = ("WITH m AS (SELECT ORDER_MONTH, SUM(NET_AMT) AS REVENUE FROM F GROUP BY ORDER_MONTH) "
               "SELECT * FROM (SELECT TOP 3 * FROM (SELECT 1 AS one) o JOIN m ON 1 = 1 ORDER BY m.REVENUE DESC) t "
               "ORDER BY ORDER_MONTH")
        assert _sql_card(self.TOP_THREE, "top 3 months by revenue", sql)["analysis_contract"]["mode"] == "ranking"

    def test_a_statement_only_snowflake_reads_is_read_for_an_azure_sql_workspace(self):
        sql = ("SELECT * FROM (SELECT d.value:month::string AS ORDER_MONTH, SUM(x) AS REVENUE FROM F, "
               "LATERAL FLATTEN(input => F.j) d GROUP BY 1 ORDER BY SUM(x) DESC LIMIT 3) t ORDER BY ORDER_MONTH")
        card = _sql_card(self.TOP_THREE, "top 3 months by revenue", sql, data_source="azure_sql")
        assert card["analysis_contract"]["mode"] == "ranking"

    def test_a_statement_only_t_sql_reads_is_read_for_a_snowflake_workspace(self):
        sql = ("SELECT * FROM (SELECT TOP 3 [ORDER_MONTH], SUM([NET_AMT]) AS [REVENUE] FROM F GROUP BY [ORDER_MONTH] "
               "ORDER BY [REVENUE] DESC) t ORDER BY [ORDER_MONTH]")
        card = _sql_card(self.TOP_THREE, "top 3 months by revenue", sql, data_source="snowflake")
        assert card["analysis_contract"]["mode"] == "ranking"

    def test_a_cut_in_a_statement_no_dialect_reads_is_not_found(self):
        sql = ("SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(x) AS REVENUE FROM F GROUP BY ORDER_MONTH "
               "ORDER BY SUM(x) DESC) t ORDER BY ORDER_MONTH ) ) (((")
        card = _sql_card(self.TOP_THREE, "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "time_series"

    def test_a_cut_that_only_the_verb_of_the_question_calls_a_ranking(self):
        """"Which months had a loss" names no ranking word: the months it lists
        are the ones a cut by the loss kept."""
        sql = ("SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(MARGIN) AS REVENUE FROM F GROUP BY ORDER_MONTH "
               "ORDER BY SUM(MARGIN) ASC) t ORDER BY ORDER_MONTH")
        assert _sql_card(self.TOP_THREE, "which months had a loss", sql)["analysis_contract"]["mode"] == "ranking"
        assert _sql_card(self.TOP_THREE, "revenue by month", sql)["analysis_contract"]["mode"] == "time_series"

    def test_only_the_outermost_order_by_lists_what_the_reader_sees(self):
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": value} for month, value in ((3, 900.0), (1, 800.0), (6, 700.0))]
        sql = ("SELECT * FROM (SELECT TOP 10 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY "
               "ORDER BY SUM(REV) DESC) s ORDER BY PRD_KEY")
        assert _sql_card(rows, "revenue by month", sql)["answer"]["headline"] == "June 2025 closed at 700."

    def test_an_order_by_inside_a_subquery_lists_nothing_where_the_statement_has_none(self):
        rows = [{"PRD_KEY": 202500 + month, "REVENUE": value} for month, value in ((3, 900.0), (1, 800.0), (6, 700.0))]
        sql = ("SELECT * FROM (SELECT TOP 10 PRD_KEY, SUM(REV) AS REVENUE FROM F GROUP BY PRD_KEY "
               "ORDER BY SUM(REV) DESC) s")
        assert _sql_card(rows, "revenue by month", sql)["answer"]["headline"] == "June 2025 closed at 700."


class TestWhereThereIsSqlItSaysWhatTheRowsDoNot:
    """The rows alone cannot tell a ranking from a series when they run
    newest first and biggest first at once: the SQL that listed them can."""

    ROWS = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((6, 1000.0), (5, 950.0), (3, 900.0))]
    SQL = "SELECT ORDER_MONTH, SUM(x) AS REVENUE FROM F GROUP BY ORDER_MONTH ORDER BY SUM(x) DESC"

    def test_the_context_is_decided_from_the_sql(self):
        assert summarize_result_context(self.ROWS, "top 3 months by revenue", self.SQL)["mode"] == "ranking"
        assert summarize_result_context(self.ROWS, "revenue by month")["mode"] == "time_series"

    def test_with_no_sql_rows_listed_by_their_value_low_to_high_are_a_ranking(self):
        from core.insight import compute_data_brief

        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((3, 700.0), (1, 800.0), (6, 900.0))]
        assert compute_data_brief(rows, "top months by revenue")["mode"] == "ranking"

    def test_with_no_sql_a_missing_figure_says_nothing(self):
        from core.response_builder import _listed_by_value

        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((3, 900.0), (1, None), (2, 800.0))]
        assert _listed_by_value(rows) is False


class TestAKeyIsNeverAFloatInTheChatTable:

    @pytest.mark.parametrize("key,digits", [(202530.0, "202530"), (202513.0, "202513")])
    def test_a_key_the_warehouse_sent_as_a_float(self, key, digits):
        rows = [{"WEEK_KEY": key, "REVENUE": 3000.0}, {"WEEK_KEY": key + 1, "REVENUE": 3100.0}]
        reply = _chat_reply(rows, "revenue by week")
        assert digits in reply and f"{digits}.0" not in reply


class TestTheBriefReadsAFrenchMonthWithNoYear:

    @pytest.mark.parametrize("labels", [("janvier", "février", "mars", "avril"), ("Mai", "Juin", "Juillet", "Août")])
    def test_a_series_of_french_months(self, labels):
        from core.insight import compute_data_brief

        rows = [{"MOIS": label, "VENTES": 100.0 * number} for number, label in enumerate(labels, 1)]
        with _InLanguage("fr"):
            assert compute_data_brief(rows, "ventes par mois")["mode"] == "time_series"


# ── Round 6 ──────────────────────────────────────────────────────────────────────────────────────────────────────

_EQUAL_STEPS = [
    ("MONTHNUM", range(4, 10)), ("MNTH", range(7, 13)), ("SORT_ORDER", range(10, 70, 10)),
    ("MONTH_ID", range(61, 67)), ("PERIOD_KEY", range(301, 307)), ("FISCAL_PERIOD", range(4, 10)),
    ("WEEK_NUM", range(27, 33)), ("DAY_NUM", range(10, 16)), ("YYYY", [2025] * 6)]
_BY_KEY = "SELECT {label}, SUM(x) AS REVENUE, {column} FROM t GROUP BY {label}, {column} ORDER BY {order}"


class TestASortKeyThatRunsInEqualSteps:
    """ORDER BY MONTHNUM lists months in the order a calendar keeps them, and so do a week's number, a
    surrogate key (MONTH_ID 61..66) and a sort order in tens. A column of whole numbers that runs in equal steps
    -- or is one number throughout, the YYYY of a year's months -- counts the rows, whatever it starts at: the
    months are a series, and no ranking by the number of the month ("June leads Month Number at 9")."""

    @pytest.mark.parametrize("column,numbers", _EQUAL_STEPS)
    def test_months_listed_by_it_are_a_series(self, column, numbers):
        rows = [{"MONTH_NAME": month, "REVENUE": value, column: number}
                for month, value, number in zip(_MONTH_NAMES, _CLIMB, numbers)]
        sql = _BY_KEY.format(label="MONTH_NAME", column=column, order=column)
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql), _MONTH_NAMES)

    @pytest.mark.parametrize("column,numbers", _EQUAL_STEPS)
    def test_and_so_are_months_that_are_dates(self, column, numbers):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value, column: number}
                for month, (value, number) in enumerate(zip(_CLIMB, numbers), 1)]
        sql = _BY_KEY.format(label="ORDER_MONTH", column=column, order=column)
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql),
                                  [f"{month} 2025" for month in _MONTH_NAMES])

    def test_listed_by_its_position(self):
        rows = [{"MONTH_NAME": month, "REVENUE": value, "SORT_ORDER": number}
                for month, value, number in zip(_MONTH_NAMES, _CLIMB, range(10, 70, 10))]
        sql = _BY_KEY.format(label="MONTH_NAME", column="SORT_ORDER", order="3")
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql), _MONTH_NAMES)

    def test_a_number_that_counts_down_is_a_sort_key_too(self):
        """The months newest first, by MONTHNUM DESC: the numbers run in equal steps in whichever order they
        stand, and the series is read in time."""
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value, "MONTHNUM": month}
                for month, value in reversed(list(enumerate(_CLIMB, 1)))]
        sql = _BY_KEY.format(label="ORDER_MONTH", column="MONTHNUM", order="MONTHNUM DESC")
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql),
                                  [f"{month} 2025" for month in _MONTH_NAMES])

    def test_three_rows_are_enough(self):
        rows = [{"MONTH_NAME": month, "REVENUE": value, "MONTHNUM": number}
                for month, value, number in (("April", 90.0, 4), ("May", 80.0, 5), ("June", 100.0, 6))]
        sql = _BY_KEY.format(label="MONTH_NAME", column="MONTHNUM", order="MONTHNUM")
        assert _sql_card(rows, "monthly revenue this year", sql)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("numbers", [(1, 2), (0, 1), (2, 1), (1, 0)])
    def test_two_rows_are_a_run_as_one_and_two_or_nought_and_one(self, numbers):
        """Two numbers that are not those could be anything the rows are counted by -- as they stand, or
        newest first."""
        rows = [{"PERIOD": label, "REVENUE": value, "SORT_ORDER": number}
                for label, value, number in zip(("FY25 Q1", "FY25 Q2"), (80.0, 90.0), numbers)]
        sql = _BY_KEY.format(label="PERIOD", column="SORT_ORDER", order="SORT_ORDER")
        assert _sql_card(rows, "revenue by quarter", sql)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("labelled", [True, False])
    def test_a_month_with_no_sale_leaves_a_gap_in_the_numbers(self, labelled):
        """No March: the number runs 1, 2, 4, 5, 6 -- still the months in the order of the calendar."""
        months = (1, 2, 4, 5, 6)
        values = (80.0, 85.0, 70.0, 95.0, 100.0)
        label = (lambda month: _MONTH_NAMES[month - 1]) if labelled else (lambda month: date(2025, month, 1))
        rows = [{"MONTH": label(month), "REVENUE": value, "MONTH_ID": number}
                for month, value, number in zip(months, values, (61, 62, 64, 65, 66))]
        sql = _BY_KEY.format(label="MONTH", column="MONTH_ID", order="MONTH_ID")
        card = _sql_card(rows, "monthly revenue this year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == ("June closed at 100." if labelled else "June 2025 closed at 100.")

    def test_a_count_the_question_ranks_the_months_by_is_a_figure(self):
        """"Top 5 months by number of orders" counts the months 5, 4, 3, 2, 1: the sort key of a calendar
        table, and the very figure the question ranks by."""
        rows = [{"ORDER_MONTH": date(2025, month, 1), "ORDERS": count, "REVENUE": value}
                for month, count, value in ((6, 5, 1000.0), (3, 4, 900.0), (5, 3, 700.0), (1, 2, 800.0), (2, 1, 950.0))]
        sql = ("SELECT TOP 5 ORDER_MONTH, COUNT(*) AS ORDERS, SUM(amount) AS REVENUE FROM orders "
               "GROUP BY ORDER_MONTH ORDER BY ORDERS DESC")
        card = _sql_card(rows, "top 5 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "June 2025 leads at 5."

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity"),
                                       10 ** 400, "NaN", "Infinity", 1e300, 0.5])
    def test_a_number_that_is_no_number_raises_nothing(self, value):
        """NaN, an infinity or a number too big for a float in a column the ORDER BY names: the context reader
        raised where the answer had always come back as a series."""
        rows = [{"PERIOD": date(2023 + n, 1, 1), "REVENUE": float(n + 1), "SORT_ORDER": (value if n == 1 else n + 1)}
                for n in range(3)]
        sql = "SELECT PERIOD, REVENUE, SORT_ORDER FROM t ORDER BY SORT_ORDER"
        context = summarize_result_context(rows, "January revenue for the last 3 years", sql=sql)
        assert context["mode"] == "time_series"


class TestPeriodsInTimeOrderAreASeriesUnlessTheQuestionRanksThem:
    """Rows that can be placed in time and come in its order, forwards or backwards, are listed by their
    period whatever the SQL sorted them by -- a key that does not run in equal steps, a figure that happens to
    rise with the months. Only a question that asks for them ranked makes them a ranking."""

    RISING = [(4, 700.0), (5, 950.0), (6, 1000.0)]
    SQL = "SELECT ORDER_MONTH, SUM(amount) AS REVENUE FROM orders GROUP BY ORDER_MONTH ORDER BY SUM(amount) {}"

    @pytest.mark.parametrize("question,mode", [
        ("revenue by month", "time_series"), ("monthly revenue for the last 3 months", "time_series"),
        ("top 3 months by revenue", "ranking"), ("which months had the highest revenue", "ranking"),
        ("which 3 months had the highest revenue", "ranking")])
    def test_in_time_order_forwards(self, question, mode):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in self.RISING]
        assert _sql_card(rows, question, self.SQL.format("ASC"))["analysis_contract"]["mode"] == mode

    @pytest.mark.parametrize("question,mode", [("revenue by month", "time_series"), ("best months by revenue", "ranking")])
    def test_in_time_order_backwards(self, question, mode):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in reversed(self.RISING)]
        assert _sql_card(rows, question, self.SQL.format("DESC"))["analysis_contract"]["mode"] == mode

    @pytest.mark.parametrize("question", ["revenue by month", "top 3 months by revenue"])
    def test_out_of_time_order_is_listed_by_the_figure(self, question):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((6, 1000.0), (4, 950.0), (5, 700.0))]
        assert _sql_card(rows, question, self.SQL.format("DESC"))["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("labels,lang,question", [
        (("April", "May", "June"), "en", "revenue by month"), (("avril", "mai", "juin"), "fr", "ventes par mois")])
    def test_month_names_in_the_order_of_the_calendar_are_in_time_too(self, labels, lang, question):
        rows = [{"MONTH_NAME": label, "REVENUE": value} for label, (_, value) in zip(labels, self.RISING)]
        sql = "SELECT MONTH_NAME, SUM(amount) AS REVENUE FROM orders GROUP BY MONTH_NAME ORDER BY SUM(amount)"
        assert _sql_card(rows, question, sql, lang)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("labels,lang,question", [
        (("June", "May", "April"), "en", "revenue by month"), (("juin", "mai", "avril"), "fr", "ventes par mois")])
    def test_month_names_newest_first_are_in_time_too(self, labels, lang, question):
        rows = [{"MONTH_NAME": label, "REVENUE": value} for label, (_, value) in zip(labels, reversed(self.RISING))]
        sql = "SELECT MONTH_NAME, SUM(amount) AS REVENUE FROM orders GROUP BY MONTH_NAME ORDER BY SUM(amount) DESC"
        assert _sql_card(rows, question, sql, lang)["analysis_contract"]["mode"] == "time_series"

    def test_month_names_in_any_other_order_are_a_fiscal_year_s_or_a_ranking(self):
        rows = [{"MONTH_NAME": label, "REVENUE": value}
                for label, value in (("June", 1000.0), ("April", 950.0), ("May", 700.0))]
        sql = "SELECT MONTH_NAME, SUM(amount) AS REVENUE FROM orders GROUP BY MONTH_NAME ORDER BY SUM(amount) DESC"
        assert _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("question,mode", [("revenue by month", "time_series"),
                                               ("which month had the highest revenue", "ranking")])
    def test_one_row_is_a_series_unless_it_is_asked_for_as_a_ranking(self, question, mode):
        rows = [{"ORDER_MONTH": date(2025, 6, 1), "REVENUE": 1000.0}]
        assert _sql_card(rows, question, self.SQL.format("DESC"))["analysis_contract"]["mode"] == mode


class TestAnAggregateIsReadToItsClosingParenthesis:
    """ORDER BY MIN(DATEFROMPARTS(YEAR(d), MONTH(d), 1)) is the first day of each month, however deep the
    functions inside go: the regex read one level of parentheses, counted nothing it could not match as a
    MIN or MAX, and took the key for a sum."""

    @pytest.mark.parametrize("key", [
        "MIN(DATEFROMPARTS(YEAR(d), MONTH(d), 1))",
        "MIN(TRY_CAST(CONCAT(LEFT(ym, 7), '-01') AS date))",
        "MAX(CAST(FORMAT(d, 'yyyyMM') AS INT))",
        "MIN(TO_DATE(TO_CHAR(d, 'YYYY-MM') || '-01'))",
        "MAX(COALESCE(NULLIF(DATEADD(day, 1, DATEFROMPARTS(YEAR(d), MONTH(d), 1)), ''), d))",
        "min ( CAST ( DATEFROMPARTS ( YEAR ( d ) , MONTH ( d ) , 1 ) AS date ) )",
    ])
    def test_months_listed_by_it_are_in_time(self, key):
        sql = f"SELECT MONTH_NAME, SUM(x) AS REVENUE FROM t GROUP BY MONTH_NAME ORDER BY {key}"
        rows = [{"MONTH_NAME": month, "REVENUE": value} for month, value in zip(_MONTH_NAMES, _CLIMB)]
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql), _MONTH_NAMES)
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in enumerate(_CLIMB, 1)]
        sql = f"SELECT ORDER_MONTH, SUM(x) AS REVENUE FROM t GROUP BY ORDER_MONTH ORDER BY {key}"
        _a_climb_read_as_a_series(_sql_card(rows, "monthly revenue this year", sql),
                                  [f"{month} 2025" for month in _MONTH_NAMES])

    @pytest.mark.parametrize("key", [
        "SUM(ROUND(CAST(x AS DECIMAL(10, 2)), 2))",
        "COUNT(DISTINCT COALESCE(NULLIF(customer_id, 0), -1))",
        "AVG(ABS(COALESCE(x, 0)))",
        "MAX(ABS(x)) + SUM(COALESCE(y, 0))",
    ])
    def test_a_figure_inside_functions_is_still_a_figure(self, key):
        sql = f"SELECT ORDER_MONTH, SUM(x) AS REVENUE FROM t GROUP BY ORDER_MONTH ORDER BY {key} DESC"
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value}
                for month, value in ((6, 1000.0), (4, 950.0), (5, 700.0))]
        assert _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("key,ranked", [
        ("MAX(COALESCE(NULLIF(amount, 0), 0))", True), ("MAX(order_date)", False),
        ("MIN(COALESCE(NULLIF(amount, NULL), order_date))", False),
        # A date word after the call is not its argument; one inside a string is no word at all.
        ("MAX(lead_time) OVER (PARTITION BY order_date)", True),
        ("MAX(CASE WHEN status = ')' THEN order_date END)", False)])
    def test_the_largest_amount_of_a_month_ranks_months_out_of_time_order_only(self, key, ranked):
        sql = f"SELECT ORDER_MONTH, SUM(x) AS REVENUE FROM t GROUP BY ORDER_MONTH ORDER BY {key} DESC"
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value}
                for month, value in ((6, 1000.0), (4, 950.0), (5, 700.0))]
        mode = _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"]
        assert mode == ("ranking" if ranked else "time_series")


class TestALengthOfTimeIsNoDate:
    """MAX(lead_time) and MAX(days_to_ship) are the longest time an order took, which ranks the months it
    took longest in: a time or a day in a name is as often a length as a date."""

    @pytest.mark.parametrize("key", ["MAX(lead_time)", "MAX(days_to_ship)", "MAX(response_time)", "MAX(transit_days)",
                                     "MAX(processing_time_hrs)", "MAX(ship_days)"])
    def test_the_months_with_the_longest_one_are_ranked(self, key):
        sql = f"SELECT ORDER_MONTH, {key} AS LONGEST FROM t GROUP BY ORDER_MONTH ORDER BY {key} DESC"
        rows = [{"ORDER_MONTH": date(2025, month, 1), "LONGEST": value}
                for month, value in ((6, 10.0), (4, 9.0), (5, 7.0))]
        card = _sql_card(rows, "which months had the longest delays", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "June 2025 leads at 10."

    @pytest.mark.parametrize("name,is_a_date", [
        ("order_date", True), ("OrderDate", True), ("ORDERDATE", True), ("DATEKEY", True), ("TXN_TS", True),
        ("created_at", True), ("INVOICE_DATETIME", True), ("posted_on", True), ("jour_commande", True),
        ("ship_dt", True), ("updated_at", True), ("modified_on", True), ("booked_on", True), ("event_timestamp", True),
        ("lead_time", False), ("days_to_ship", False), ("response_time", False), ("lifetime_value", False),
        ("candidate_score", False), ("runtime_secs", False), ("update_count", False), ("DAYS", False),
        ("DAY", False), ("TIME", False)])
    def test_which_names_say_a_date(self, name, is_a_date):
        from core.response_builder import _names_a_date

        assert _names_a_date(name) is is_a_date


_ORDER_YEARS = [{"ORDER_YEAR": date(year, 1, 1), "REVENUE": 100.0 * number}
                for number, year in enumerate((2022, 2023, 2024, 2025), 1)]


class TestAMonthBesideAYearAskedIsNoPeriodOfIt:
    """"Revenue by year for the January cohort" groups by year: January is what is filtered. So is the month of
    "yearly revenue excluding January" and of "annual revenue of products launched in Q1" -- and a list of
    months ("January, February and March") names none of them. A column named for years is years whatever
    month a question names."""

    @pytest.mark.parametrize("question", [
        "revenue by year for the January cohort", "yearly revenue excluding January",
        "annual revenue of products launched in Q1", "what did Jan sell each year",
        "revenue by year for customers who joined in March", "sales in March for each year",
        "revenue for January, February and March of each year", "Jan, Feb and Mar revenue by year",
        "January revenue by year", "Q1 revenue by year"])
    def test_a_year_column_stays_its_years(self, question):
        card = _sql_card(_ORDER_YEARS, question, "SELECT 1")
        assert card["answer"]["headline"] == "2025 closed at 400."
        assert card["data"]["display_formats"] == {"ORDER_YEAR": {"type": "date", "style": "year"}}

    @pytest.mark.parametrize("column", ["ORDER_YEAR", "FISCAL_YEAR", "YEAR", "SALES_YR"])
    def test_a_column_named_for_years_is_never_the_month_a_question_names(self, column):
        """Not even where the question says nothing of a year: the column does."""
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        assert _sql_card(rows, "January revenue for the last 4 years", "SELECT 1")["answer"]["headline"] == "2025 closed at 400."

    @pytest.mark.parametrize("question", [
        "revenue by fiscal year, our year starts in July", "revenue by fiscal year, our year begins in Q3",
        "fiscal year revenue, which starts in July"])
    def test_a_fiscal_year_that_starts_in_the_month_named_is_the_date_it_is(self, question):
        rows = [{"FISCAL_YEAR": date(year, 7, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2021, 2022, 2023, 2024), 1)]
        card = _sql_card(rows, question, "SELECT 1")
        assert card["answer"]["headline"] == "2024-07-01 closed at 400."

    @pytest.mark.parametrize("question,grain", [
        ("revenue by year for the January cohort", "year"), ("yearly revenue excluding January", "year"),
        ("what did Jan sell each year", "year|month:1"), ("annual revenue of products launched in Q1", "year"),
        ("sales in March for each year", "year|month:3"), ("revenue by quarter for the January cohort", "quarter"),
        # A list of months or quarters names none of them, with a comma or without.
        ("revenue for January, February and March over the last 4 years", "year"),
        ("Jan, Feb, Mar revenue for the last 3 years", "year"),
        ("Q1, Q2 revenue for the last 3 years", "year"),
        ("revenue for March - May over the last 3 years", "year"),
        # One month, then a comma and something else, is still one month.
        ("sales in January, by region, for the last 3 years", "year|month:1"),
        ("January, for the last 3 years", "year|month:1"),
        ("January revenue for the last 3 years", "year|month:1")])
    def test_which_grain_and_month_a_question_names(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan("year")) == grain

    @pytest.mark.parametrize("question,plan_grain", [
        ("January revenue for the last 3 years", "year"), ("Q1 revenue for each of the last 3 years", "year")])
    def test_a_plans_year_is_how_far_back_it_reaches_and_no_year_asked(self, question, plan_grain):
        """The plan's grain is the window's unit: the Januaries it names are still the periods."""
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan(plan_grain)).partition("|")[2] != ""


class TestAFiscalCalendarThatStartsInAnyMonth:
    """A fiscal year that starts in February, March, May, June, August, September, November or December is
    stored as the date it starts on, and so is its quarter: no month name says either. Only July, April and
    October -- firsts of calendar quarters -- were left as dates; the rest were named "February 2024", which
    a year of revenue is not."""

    @pytest.mark.parametrize("month", [2, 3, 5, 6, 8, 9, 11, 12, 7, 4, 10])
    def test_a_fiscal_year_is_the_date_it_starts_on(self, month):
        rows = [{"FISCAL_YEAR": date(year, month, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2021, 2022, 2023, 2024), 1)]
        card = _card(rows, "revenue by fiscal year")
        assert card["answer"]["headline"] == f"2024-{month:02d}-01 closed at 400."
        assert card["data"]["display_formats"] == {}

    @pytest.mark.parametrize("column", ["FISCAL_YEAR", "FY_START", "FISCAL_YEAR_START", "EXERCICE", "PERIOD"])
    def test_whatever_the_column_is_called(self, column):
        rows = [{column: date(year, 2, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2021, 2022, 2023, 2024), 1)]
        question = "revenue by year" if column == "PERIOD" else "revenue by fiscal year"
        assert _card(rows, question)["answer"]["headline"] == "2024-02-01 closed at 400."

    @pytest.mark.parametrize("months", [(2, 5, 8, 11), (3, 6, 9, 12)])
    @pytest.mark.parametrize("column", ["FISCAL_QUARTER", "FISCAL_QTR_START", "PERIOD"])
    def test_a_fiscal_quarter_is_the_date_it_starts_on(self, column, months):
        rows = [{column: date(2024, month, 1), "REVENUE": 100.0 * number} for number, month in enumerate(months, 1)]
        card = _card(rows, "revenue by quarter")
        assert card["answer"]["headline"] == f"2024-{months[-1]:02d}-01 closed at 400."
        assert card["data"]["display_formats"] == {}

    def test_firsts_three_months_apart_off_the_calendar_quarters_stay_dates_unasked(self):
        rows = [{"PERIOD": date(2024, month, 1), "REVENUE": 100.0 * number} for number, month in enumerate((2, 5, 8, 11), 1)]
        assert _card(rows, "revenue trend")["answer"]["headline"] == "2024-11-01 closed at 400."

    def test_a_single_first_beside_no_other_is_the_month_it_is(self):
        rows = [{"PERIOD": date(2024, 2, 1), "REVENUE": 100.0}]
        assert _card(rows, "revenue by quarter")["answer"]["headline"] == "February 2024 closed at 100."

    def test_months_asked_by_quarter_are_still_months(self):
        rows = [{"PERIOD": date(2025, month, 1), "REVENUE": 100.0 * month} for month in (1, 2, 4, 5)]
        assert _card(rows, "revenue by quarter")["answer"]["headline"] == "May 2025 closed at 500."

    def test_quarters_that_start_with_the_calendar_are_quarters(self):
        rows = [{"PERIOD": date(2024, month, 1), "REVENUE": 100.0 * number} for number, month in enumerate((1, 4, 7, 10), 1)]
        assert _card(rows, "revenue by quarter")["answer"]["headline"] == "Q4 2024 closed at 400."


class TestADayWithAMonthsNameIsADate:
    """"What were sales on January 1st" names January, and the first of it is a date, not the month: the
    whole month's sales are not the day's."""

    @pytest.mark.parametrize("question,column,day", [
        ("what were sales on January 1st", "ORDER_DATE", date(2025, 1, 1)),
        ("sales on 1 January", "ORDER_DATE", date(2025, 1, 1)),
        ("sales on the 1st of January", "ORDER_DATE", date(2025, 1, 1)),
        ("sales on the first of January", "ORDER_DATE", date(2025, 1, 1)),
        ("sales on the first day of November", "ORDER_DATE", date(2025, 11, 1)),
        ("sales on the last day of November", "ORDER_DATE", date(2025, 11, 1)),
        ("sales on Jan 1", "ORDER_DATE", date(2025, 1, 1)),
        ("sales on March 1st", "PERIOD", date(2025, 3, 1)),
        ("sales on 1st March", "PERIOD", date(2025, 3, 1)),
    ])
    def test_the_day_stays_the_day(self, question, column, day):
        card = _card([{column: day, "SALES": 900.0}], question)
        assert card["answer"]["headline"] == f"{day.isoformat()} closed at 900."
        assert card["data"]["display_formats"] == {}

    def test_in_french_too(self):
        card = _card([{"ORDER_DATE": date(2025, 3, 1), "VENTES": 900.0}], "ventes du 1er mars", "fr")
        assert card["answer"]["headline"] == "2025-03-01 a terminé à 900."

    def test_each_year_s_first_of_january(self):
        rows = [{"ORDER_DATE": date(year, 1, 1), "SALES": 100.0 * number} for number, year in enumerate((2023, 2024, 2025), 1)]
        assert _card(rows, "January 1st sales for each of the last 3 years")["answer"]["headline"] == "2025-01-01 closed at 300."

    @pytest.mark.parametrize("question,grain", [
        ("what were sales on January 1st", ""), ("sales on 1 January", ""), ("ventes du 1er mars", ""),
        ("sales on the first of January", ""), ("sales on the 15th of March", ""), ("sales on the first day of May", ""),
        ("sales on the last day of May", ""),
        # A number beside no month, or a month a number is not beside, is no day of it.
        ("January revenue for the last 3 years", "|month:1"),
        ("revenue in the last 3 Januaries of the top 10 regions", ""),
        ("top 10 regions for January revenue", "|month:1")])
    def test_which_month_is_named(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == grain

    def test_a_column_named_as_a_date_holds_dates_the_question_names_a_month_of(self):
        rows = [{"ORDER_DATE": date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        card = _sql_card(rows, "January revenue for the last 4 years", "SELECT 1")
        assert card["answer"]["headline"] == "2025-01-01 closed at 400."

    def test_a_column_that_is_not_a_date_holds_the_months_it_names(self):
        rows = [{"PERIOD": date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        card = _sql_card(rows, "January revenue for the last 4 years", "SELECT 1", semantic_plan=_with_plan("year"))
        assert card["answer"]["headline"] == "January 2025 closed at 400."

    def test_a_single_first_the_question_names_a_month_of_is_that_month(self):
        """One January, for "January revenue this year": the month whose firsts these are."""
        from core.response_builder import period_grain

        assert period_grain([date(2025, 1, 1)], "year|month:1", "PERIOD") == "month"
        card = _sql_card([{"PERIOD": date(2025, 1, 1), "REVENUE": 900.0}], "January revenue this year", "SELECT 1")
        assert card["answer"]["headline"] == "January 2025 closed at 900."


class TestAWindowsBoundIsNoMonthItsPeriodsAre:
    """A word that bounds a window -- "since", "by", "up to", "depuis", "avant", "jusqu'en" -- puts no month
    where the periods are."""

    @pytest.mark.parametrize("question", [
        "revenue by March for the last 3 years", "revenue up to March for the last 3 years",
        "revenue through March for the last 3 years", "revenue until March for the last 3 years",
        "revenue before March for the last 3 years", "revenue after March for the last 3 years",
        "revenue from March for the last 3 years", "revenue since March for the last 3 years",
        "revenue between March and June for the last 3 years", "revenue thru March for the last 3 years"])
    def test_in_english(self, question):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan("year")) == "year"

    @pytest.mark.parametrize("question", [
        "ventes depuis mars pour les 3 dernières années", "ventes avant mars pour les 3 dernières années",
        "ventes après mars pour les 3 dernières années", "ventes jusqu'en mars pour les 3 dernières années",
        "ventes entre mars et juin pour les 3 dernières années"])
    def test_in_french(self, question):
        from core.response_builder import requested_period_grain

        with _InLanguage("fr"):
            assert requested_period_grain(question, _with_plan("year")) == "year"

    def test_a_month_a_french_question_names_is_still_one(self):
        from core.response_builder import requested_period_grain

        with _InLanguage("fr"):
            assert requested_period_grain("ventes de mars pour les 3 dernières années", _with_plan("year")) == "year|month:3"


class TestADayWrittenOutHasItsPlaceInTime:
    """A French day listed newest first -- "4 mars 2025" down to "1 mars 2025" -- was read in the order the
    SQL listed it, which made sales that fell 75% "progress 300%" between two days; an English day, "4 March
    2025", did the same."""

    @pytest.mark.parametrize("label,expected", [
        ("4 March 2025", date(2025, 3, 4)), ("4 Mar 2025", date(2025, 3, 4)), ("March 4, 2025", date(2025, 3, 4)),
        ("Mar 4, 2025", date(2025, 3, 4)), ("March 4 2025", date(2025, 3, 4)), ("Mar 4 2025", date(2025, 3, 4)),
        ("04-Mar-2025", date(2025, 3, 4)),
        ("4/Mar/2025", date(2025, 3, 4)), ("1 mars 2025", date(2025, 3, 1)), ("1er mars 2025", date(2025, 3, 1)),
        ("jeudi 4 mars 2025", date(2025, 3, 4)), ("4 févr. 2025", date(2025, 2, 4)), ("4 fevrier 2025", date(2025, 2, 4)),
        ("12 décembre 2025", date(2025, 12, 12)), ("4 août 2025", date(2025, 8, 4)), ("4 JUIN 2025", date(2025, 6, 4)),
        # What is no day, or is one no calendar has, or has no year to place it by.
        ("31 févr. 2025", None), ("0 mars 2025", None), ("99 mars 2025", None), ("1 mars 25", None),
        ("Sep 24", None), ("Sep 24 25", None), ("March 2025", None), ("mars 2025", None), ("", None), (None, None),
        ("1 mars 1899", None), ("4 March 1899", None), ("4 March 2200", None), ("4 Marchh 2025", None)])
    def test_which_labels_are_a_day(self, label, expected):
        from core.temporal_columns import parse_day_label

        assert parse_day_label(label) == expected

    def test_a_french_day_needs_its_year_in_full(self):
        """"1 mars 25" is a label with a day in it; it is no day of a daily series."""
        from core.response_builder import _looks_temporal
        from core.temporal_columns import FRENCH_DAY_LABEL_RE

        assert FRENCH_DAY_LABEL_RE.fullmatch("1 mars 2025")
        assert not FRENCH_DAY_LABEL_RE.fullmatch("1 mars 25")
        assert not _looks_temporal(["1 mars 25", "2 mars 25"])

    @pytest.mark.parametrize("lang,label,question,trend,peak", [
        ("fr", "{day} mars 2025", "ventes par jour", "Ventes a reculé de 75,0 % entre 1 mars 2025 et 4 mars 2025.",
         "Pic : 400 en 1 mars 2025."),
        ("en", "{day} March 2025", "sales by day", "Sales trended down 75.0% from 1 March 2025 to 4 March 2025.",
         "Peak: 400 in 1 March 2025.")])
    def test_days_listed_newest_first_are_read_in_time(self, lang, label, question, trend, peak):
        measure = "VENTES" if lang == "fr" else "SALES"
        rows = [{"JOUR": label.format(day=day), measure: value} for day, value in zip((4, 3, 2, 1), (100.0, 200.0, 300.0, 400.0))]
        sql = f"SELECT JOUR, SUM(v) AS {measure} FROM t GROUP BY JOUR ORDER BY d DESC"
        card = _sql_card(rows, question, sql, lang)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert _plain(trend) in _plain(card["insight_summary"]) and _plain(peak) in _plain(card["insight_summary"])

    def test_days_listed_out_of_time_by_a_figure_are_a_ranking(self):
        rows = [{"JOUR": f"{day} mars 2025", "VENTES": value} for day, value in ((3, 500.0), (1, 400.0), (4, 100.0), (2, 300.0))]
        sql = "SELECT JOUR, SUM(v) AS VENTES FROM t GROUP BY JOUR ORDER BY SUM(v) DESC"
        assert _sql_card(rows, "ventes par jour", sql, "fr")["analysis_contract"]["mode"] == "ranking"


class TestAnExplicitFormatIsHonouredOfAKey:
    """A week's key is shown as its digits -- "202530", never "202,530" -- unless a display format asks
    for something else of the column."""

    @pytest.mark.parametrize("value,expected", [
        (202530, "202530"), (202530.0, "202530"), ("202530", "202530"), (Decimal("202530.00"), "202530"),
        (Decimal("202530"), "202530")])
    def test_a_key_is_its_digits(self, value, expected):
        from core.response_builder import _format_display_value

        assert _format_display_value(value, "text") == expected
        assert _format_display_value(value, "text", {"type": "text"}) == expected

    @pytest.mark.parametrize("spec,expected", [
        ({"type": "number", "fraction_digits": 2}, "202,530.00"), ({"type": "number", "fraction_digits": 0}, "202,530"),
        ({"type": "number", "fraction_digits": 0, "grouping": False}, "202530")])
    def test_a_format_the_column_was_given_is_kept(self, spec, expected):
        from core.response_builder import _format_display_value

        assert _format_display_value(202530, "text", spec) == expected

    @pytest.mark.parametrize("value,expected", [(Decimal("202530.50"), "202530.50"), (Decimal("Infinity"), "Infinity"),
                                                (Decimal("NaN"), "NaN")])
    def test_a_decimal_that_is_no_whole_number_is_left_as_it_is(self, value, expected):
        from core.response_builder import _format_display_value

        assert _format_display_value(value, "text") == expected


class TestWhichUnitsAreRankedWithAVerbOrAnOrdinalAfterThem:
    """"Which months had the most orders", "which 3 months are the busiest": every verb and every ranking
    word that turns "which months" into a ranking of the months."""

    @pytest.mark.parametrize("verb", [
        "had", "have", "has", "was", "were", "did", "do", "does", "saw", "see", "generated", "recorded", "brought",
        "made", "produced", "got", "came", "grew", "fell", "dropped", "declined", "improved", "ranked", "top", "best",
        "worst", "highest", "lowest", "most", "least", "busiest", "strongest", "weakest", "slowest", "fastest",
        "biggest", "largest", "smallest"])
    def test_a_verb_or_ranking_word(self, verb):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(f"which months {verb} sales") == "month"
        assert ranked_period_grain(f"what weeks {verb} the orders") == "week"

    @pytest.mark.parametrize("adjective", ["best", "worst", "highest", "lowest", "top", "busiest", "strongest",
                                           "weakest", "slowest", "fastest", "biggest", "largest", "smallest"])
    @pytest.mark.parametrize("verb", ["are", "is"])
    def test_is_or_are_the_best(self, verb, adjective):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(f"which months {verb} the {adjective}") == "month"
        assert ranked_period_grain(f"which months {verb} {adjective}") == "month"

    @pytest.mark.parametrize("tail", ["in 2025", "of 2024", "during 2023"])
    def test_a_year(self, tail):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(f"which months {tail} had sales") == "month"
        assert ranked_period_grain(f"which months {tail}") == "month"

    @pytest.mark.parametrize("window", ["last year", "this year", "past year", "last quarter", "this quarter",
                                        "past quarter", "last month", "this month", "past month"])
    def test_a_window(self, window):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(f"which months {window}") == "month"

    @pytest.mark.parametrize("count", ["2", "3", "12", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                       "ten", "twelve"])
    def test_a_count_before_them(self, count):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(f"which {count} months had the most orders") == "month"
        assert ranked_period_grain(f"what {count} days were the busiest") == "day"

    @pytest.mark.parametrize("question", [
        "what weeks of supply do we have by warehouse", "what days sales outstanding", "which days are past due",
        "what weeks on hand", "which days of the week are busiest", "which months of the year"])
    def test_a_measure_counted_in_them_is_no_ranking_of_them(self, question):
        from core.contextual_dates import ranked_period_grain

        assert ranked_period_grain(question) == ""


class TestAMonthNamedAlone:

    @pytest.mark.parametrize("label,month", [
        ("March", 3), ("MAR", 3), ("mar", 3), ("mars", 3), ("févr.", 2), ("Février", 2), ("août", 8), ("décembre", 12),
        ("Dec.", 12), ("janv.", 1), (" June ", 6), ("Sept", 9), ("sep", 9), ("juil.", 7)])
    def test_which_labels_are_a_month(self, label, month):
        from core.temporal_columns import parse_month_name

        assert parse_month_name(label) == month

    @pytest.mark.parametrize("label", ["", None, "Mars Bar", "Q1", "March 2025", "Marchh", "2025", "Week 1", "mars 2025"])
    def test_which_are_none(self, label):
        from core.temporal_columns import parse_month_name

        assert parse_month_name(label) is None


class TestADateObjectIsARealDateWhateverItsColumnIsCalled:
    """A fiscal period's key, written 2025-01, is no calendar month; the first of a month, as a date object
    or as its ISO day, is one in a column called POSTING_MONTH, GL_MONTH or FISCAL_PERIOD."""

    @pytest.mark.parametrize("column", ["POSTING_MONTH", "GL_MONTH", "FISCAL_MONTH", "FISCAL_PERIOD", "ACCOUNTING_PERIOD",
                                        "WEEK_START"])
    @pytest.mark.parametrize("written", [lambda d: d, lambda d: d.isoformat(), lambda d: datetime(d.year, d.month, d.day)])
    def test_firsts_of_months(self, column, written):
        rows = [{column: written(date(2025, month, 1)), "REVENUE": 100.0 * month} for month in (1, 2, 3)]
        card = _card(rows, "revenue by month")
        assert card["answer"]["headline"] == "March 2025 closed at 300."

    @pytest.mark.parametrize("column", ["POSTING_MONTH", "GL_MONTH", "FISCAL_PERIOD"])
    def test_a_key_written_with_a_hyphen_is_none(self, column):
        rows = [{column: f"2025-0{month}", "REVENUE": 100.0 * month} for month in (1, 2, 3)]
        card = _card(rows, "revenue by period")
        assert card["answer"]["headline"] == "2025-03 closed at 300."
        assert card["data"]["display_formats"] == {}


class TestTheTopOfACutByACountThatRunsInEqualSteps:
    def test_a_subquery_s_top_3_by_a_count_is_a_ranking_of_the_months(self):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "ORDERS": count, "REVENUE": value}
                for month, count, value in ((1, 3, 800.0), (2, 2, 950.0), (4, 1, 700.0))]
        sql = ("WITH m AS (SELECT TOP 3 ORDER_MONTH, COUNT(*) AS ORDERS, SUM(x) AS REVENUE FROM t GROUP BY ORDER_MONTH "
               "ORDER BY ORDERS DESC) SELECT ORDER_MONTH, ORDERS, REVENUE FROM m ORDER BY ORDER_MONTH")
        card = _sql_card(rows, "top 3 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"


class TestAMonthWithItsYearIsAWindowsEdgeWhateverGrainIsAsked:

    @pytest.mark.parametrize("question", [
        "January 2024 revenue for the last 3 years", "Q1 2024 revenue for the last 3 years",
        "revenue in January, 2024 for the last 3 years", "January 2024 revenue over the last 3 years",
        "revenue for Jan 2024 over the last 3 years", "revenue for the third quarter of 2023 over the last 3 years",
        "revenue for January of 2024 over the last 3 years"])
    def test_a_month_with_its_year_names_no_period(self, question):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan("year")) == "year"


class TestFirstsAskedByTheDayStayDays:
    """"Daily sales" of a column that is not called a date, on the first of January, March and July, are
    those days -- firsts that step in no whole quarter or year."""

    @pytest.mark.parametrize("question", ["daily sales", "sales by day", "sales by week", "weekly sales"])
    def test_firsts_that_skip_months_are_days(self, question):
        rows = [{"PERIOD": date(2025, month, 1), "SALES": 100.0 * number} for number, month in enumerate((1, 3, 7, 8), 1)]
        card = _card(rows, question)
        assert card["answer"]["headline"] == "2025-08-01 closed at 400."
        assert card["data"]["display_formats"] == {}

    def test_firsts_that_step_month_by_month_are_months(self):
        rows = [{"PERIOD": date(2025, month, 1), "SALES": 100.0 * month} for month in (1, 2, 3, 4)]
        assert _card(rows, "average daily sales by month")["answer"]["headline"] == "April 2025 closed at 400."


_WEEKS = [f"Week {number}" for number in range(27, 33)]


class TestASortKeyOfPeriodsThatCannotBePlacedInTime:
    """"Week 27" names no year, so only the sort key can say the weeks are listed in their order: where the
    months or the dates can be placed, the order of time says it."""

    @pytest.mark.parametrize("column,numbers", [
        ("WEEK_NUM", range(27, 33)), ("SORT_ORDER", range(10, 70, 10)), ("PERIOD_KEY", range(301, 307)),
        ("WEEK_ID", range(61, 67)), ("YYYY", [2025] * 6), ("SEQ", range(6)), ("RN", range(1, 7)),
        ("NUM", range(6, 0, -1))])
    def test_weeks_listed_by_it_are_a_series(self, column, numbers):
        rows = [{"WEEK_LABEL": week, "REVENUE": value, column: number} for week, value, number in zip(_WEEKS, _CLIMB, numbers)]
        sql = _BY_KEY.format(label="WEEK_LABEL", column=column, order=column)
        _a_climb_read_as_a_series(_sql_card(rows, "weekly revenue for the last 6 weeks", sql), _WEEKS)

    @pytest.mark.parametrize("order", ["3", "3 DESC", "WEEK_NUM DESC"])
    def test_listed_by_its_position_or_its_name(self, order):
        rows = [{"WEEK_LABEL": week, "REVENUE": value, "WEEK_NUM": number}
                for week, value, number in zip(_WEEKS, _CLIMB, range(27, 33))]
        sql = _BY_KEY.format(label="WEEK_LABEL", column="WEEK_NUM", order=order)
        _a_climb_read_as_a_series(_sql_card(rows, "weekly revenue for the last 6 weeks", sql), _WEEKS)

    @pytest.mark.parametrize("order", ["1", "1 DESC", "9", "WEEK_LABEL", "WEEK_LABEL DESC"])
    def test_a_position_of_the_weeks_themselves_or_of_nothing_is_no_figure(self, order):
        rows = [{"WEEK_LABEL": week, "REVENUE": value} for week, value in zip(_WEEKS, _CLIMB)]
        sql = f"SELECT WEEK_LABEL, SUM(x) AS REVENUE FROM t GROUP BY WEEK_LABEL ORDER BY {order}"
        _a_climb_read_as_a_series(_sql_card(rows, "weekly revenue for the last 6 weeks", sql), _WEEKS)

    def test_a_position_of_the_figure_ranks(self):
        rows = [{"WEEK_LABEL": week, "REVENUE": value} for week, value in zip(_WEEKS, (950.0, 900.0, 850.0, 700.0, 500.0, 300.0))]
        sql = "SELECT WEEK_LABEL, SUM(x) AS REVENUE FROM t GROUP BY WEEK_LABEL ORDER BY 2 DESC"
        assert _sql_card(rows, "weekly revenue for the last 6 weeks", sql)["analysis_contract"]["mode"] == "ranking"

    def test_a_count_that_is_all_the_result_measures_is_what_it_ranks_by_unasked(self):
        """Orders counted 3, 2, 1, listed by the count: a run of equal steps, and the only figure they have."""
        rows = [{"MONTH_NAME": month, "ORDERS": count} for month, count in (("March", 3), ("January", 2), ("June", 1))]
        sql = ("SELECT DATENAME(month, d) AS MONTH_NAME, COUNT(*) AS ORDERS FROM t "
               "GROUP BY DATENAME(month, d) ORDER BY ORDERS DESC")
        card = _sql_card(rows, "orders by month", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "March leads at 3."

    def test_an_outer_order_by_a_figure_ranks_whatever_an_inner_one_lists(self):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((6, 1000.0), (4, 950.0), (5, 700.0))]
        sql = ("SELECT ORDER_MONTH, REVENUE FROM (SELECT TOP 100 ORDER_MONTH, SUM(x) AS REVENUE FROM F "
               "GROUP BY ORDER_MONTH ORDER BY ORDER_MONTH) t ORDER BY REVENUE DESC")
        assert _sql_card(rows, "revenue by month", sql)["analysis_contract"]["mode"] == "ranking"


class TestARankingWordAboutSomethingElseRanksNoMonth:
    """"Monthly revenue for the top 5 customers" and "our best customer by month" list months in time: the
    ranking is of the customers."""

    @pytest.mark.parametrize("question", [
        "monthly revenue for the top 5 customers", "monthly revenue trend of our best customer",
        "revenue by month for the worst performing region", "monthly sales of the biggest customer",
        "monthly revenue for the top customer", "sales by month of our highest spending account"])
    def test_months_listed_by_a_sort_key_are_a_series(self, question):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value, "MONTHNUM": month}
                for month, value in enumerate(_CLIMB, 1)]
        sql = _BY_KEY.format(label="ORDER_MONTH", column="MONTHNUM", order="MONTHNUM")
        _a_climb_read_as_a_series(_sql_card(rows, question, sql), [f"{name} 2025" for name in _MONTH_NAMES])

    @pytest.mark.parametrize("question", [
        "monthly revenue for the top 5 customers", "monthly revenue trend of our best customer"])
    def test_a_whole_year_of_months_listed_by_its_number_is_a_series(self, question):
        rows = [{"MONTH_NAME": name, "REVENUE": 100.0 + 10 * number, "MONTHNUM": number}
                for number, name in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                               "September", "October", "November", "December"], 1)]
        sql = _BY_KEY.format(label="MONTH_NAME", column="MONTHNUM", order="MONTHNUM")
        assert _sql_card(rows, question, sql)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("question", [
        "monthly revenue for the top 5 customers", "monthly revenue trend of our best customer"])
    def test_months_in_time_order_listed_by_the_revenue_are_a_series(self, question):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value} for month, value in ((4, 700.0), (5, 950.0), (6, 1000.0))]
        sql = "SELECT ORDER_MONTH, SUM(amount) AS REVENUE FROM orders GROUP BY ORDER_MONTH ORDER BY SUM(amount)"
        assert _sql_card(rows, question, sql)["analysis_contract"]["mode"] == "time_series"


class TestWhichQuestionsRankThePeriodsThemselves:

    @pytest.mark.parametrize("question", [
        "top 5 months by number of orders", "the 6 best months by revenue", "best months by revenue",
        "which months had the most orders", "which 3 months had the most orders", "highest revenue month",
        "what was the best month", "bottom 3 weeks by sales", "rank months by revenue", "ranking of months by orders",
        "worst quarter", "days with the most sales", "weeks with the lowest revenue", "best-performing month",
        "what were our 3 best quarters", "top month", "busiest weeks this year", "show the top 10 days by orders",
        "the five best years", "weeks that had the highest orders", "quarters having the lowest sales",
        "ranked months by sales", "our ranks of the months", "rank the months", "rank all weeks by sales",
        "rank our months by sales"])
    def test_periods_ranked(self, question):
        from core.response_builder import _asks_to_rank_periods

        assert _asks_to_rank_periods(question)

    @pytest.mark.parametrize("word", ["top", "bottom", "best", "worst", "highest", "lowest", "biggest", "largest", "smallest",
                                      "busiest", "slowest", "fastest", "strongest", "weakest"])
    @pytest.mark.parametrize("noun", ["day", "week", "month", "quarter", "year", "days", "weeks", "months", "quarters", "years"])
    def test_each_word_before_each_period(self, word, noun):
        from core.response_builder import _asks_to_rank_periods

        assert _asks_to_rank_periods(f"the {word} {noun}")
        assert _asks_to_rank_periods(f"{word} 3 {noun} by sales")
        assert _asks_to_rank_periods(f"{word} sales {noun}")

    @pytest.mark.parametrize("count", ["2", "10", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                       "twelve"])
    def test_a_count_between(self, count):
        from core.response_builder import _asks_to_rank_periods

        assert _asks_to_rank_periods(f"top {count} months")
        assert _asks_to_rank_periods(f"{count} best months")

    @pytest.mark.parametrize("preposition", ["by", "per", "each", "in", "for", "of", "and", "or", "with", "on", "at", "to",
                                             "from", "during", "over"])
    def test_a_ranking_word_and_a_preposition_before_a_period_rank_no_period(self, preposition):
        """"Best per year" is the best of something in each year."""
        from core.response_builder import _asks_to_rank_periods

        assert not _asks_to_rank_periods(f"best {preposition} month")
        assert not _asks_to_rank_periods(f"top customers, highest {preposition} quarter")

    def test_a_question_that_cannot_be_canonicalised_is_still_read_as_written(self, monkeypatch):
        from core import question_normalizer
        from core.response_builder import _asks_to_rank_periods

        def boom(*args, **kwargs):
            raise RuntimeError("no canonical form")

        monkeypatch.setattr(question_normalizer, "canonical_question", boom)
        assert _asks_to_rank_periods("the 6 best months by revenue")
        assert not _asks_to_rank_periods("monthly revenue for the top 5 customers")

    @pytest.mark.parametrize("tail", ["most", "highest", "lowest", "least", "fewest", "best", "worst", "biggest", "largest",
                                      "smallest"])
    @pytest.mark.parametrize("lead", ["with", "that had", "having"])
    def test_each_period_with_the_most(self, lead, tail):
        from core.response_builder import _asks_to_rank_periods

        assert _asks_to_rank_periods(f"months {lead} the {tail} orders")
        assert _asks_to_rank_periods(f"weeks {lead} the {tail} sales")

    @pytest.mark.parametrize("question", [
        "monthly revenue for the top 5 customers", "monthly revenue trend of our best customer", "revenue by month",
        "top 3 products by revenue each month", "monthly revenue of the worst performing region",
        "top 10 customers by month", "most recent month", "highest revenue by region each quarter",
        "best selling product by week", "what is the trend for the top product per year", "show revenue for the last 6 months",
        "revenue for the best region in the last 3 months", "top customers for the month of June",
        "quarterly sales of the biggest customer", "top 10 customers per quarter", "top regions in the year",
        "best customers for each month", "worst products over the week", "lowest prices and months", "rank customers by month",
        "ranking of regions by year", "months"])
    def test_something_else_ranked(self, question):
        from core.response_builder import _asks_to_rank_periods

        assert not _asks_to_rank_periods(question)

    @pytest.mark.parametrize("lang,question", [
        ("fr", "les 3 meilleurs mois par revenu"), ("fr", "quel mois a eu le plus de commandes"),
        ("fr", "les 5 pires semaines")])
    def test_in_french(self, lang, question):
        from core.response_builder import _asks_to_rank_periods

        with _InLanguage(lang):
            assert _asks_to_rank_periods(question)

    def test_not_the_customers_in_french(self):
        from core.response_builder import _asks_to_rank_periods

        with _InLanguage("fr"):
            assert not _asks_to_rank_periods("chiffre d'affaires mensuel des 5 meilleurs clients")


# ── A series that crosses a new year, a month named beside a grain, a day beside a year ─────────────────────────────

_FROM_OCTOBER = ["October", "November", "December", "January", "February", "March", "April", "May", "June", "July", "August",
                 "September"]
_FROM_OCTOBER_REVENUE = (700.0, 720.0, 760.0, 650.0, 680.0, 800.0, 820.0, 850.0, 900.0, 880.0, 950.0, 1000.0)
_FROM_OCTOBER_YEARS = [2024] * 3 + [2025] * 9
_FROM_OCTOBER_NUMBERS = [10, 11, 12] + list(range(1, 10))
_WEEKS_ACROSS_A_YEAR = ["Week 50", "Week 51", "Week 52", "Week 1", "Week 2", "Week 3"]
_WEEK_KEYS = [202450, 202451, 202452, 202501, 202502, 202503]


def _twelve_months_across_a_new_year_are_a_series(card, first="October", last="September"):
    assert card["analysis_contract"]["mode"] == "time_series"
    assert card["answer"]["headline"] == f"{last} closed at 1,000."
    assert f"Revenue trended up 42.9% from {first} to {last}." in card["insight_summary"]
    assert [callout["message"] for callout in card["anomaly_callouts"]] == [
        "Biggest drop: December → January (-14.5%)", "Biggest gain: February → March (+17.6%)"]


def _weeks_across_a_new_year_are_a_series(card):
    assert card["analysis_contract"]["mode"] == "time_series"
    assert card["answer"]["headline"] == "Week 3 closed at 800."
    assert "Revenue trended up 14.3% from Week 50 to Week 3." in card["insight_summary"]


class TestAMonthSeriesAcrossANewYearIsASeries:
    """October to September, listed by year and month or by a key that counts months on, has no month name
    that stands after its successor: read as listed by the figure, it was ranked, and "September leads at 1,000"
    replaced "September closed at 1,000. Revenue trended up 42.9%"."""

    @pytest.mark.parametrize("year", ["Y", "YYYY", "AN", "CAL_Y", "YR"])
    def test_months_listed_by_a_year_and_a_month_the_statement_takes_from_the_date(self, year):
        rows = [{"MONTH_NAME": name, "REVENUE": value, year: number, "M": month}
                for name, value, number, month in zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                      _FROM_OCTOBER_NUMBERS)]
        sql = (f"SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, YEAR(d) AS {year}, MONTH(d) AS M FROM t "
               f"GROUP BY DATENAME(month, d), YEAR(d), MONTH(d) ORDER BY {year}, M")
        _twelve_months_across_a_new_year_are_a_series(_sql_card(rows, "monthly revenue for the last 12 months", sql))

    def test_the_same_columns_read_from_a_table_are_in_time_by_the_names_of_the_months(self):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "Y": number, "M": month}
                for name, value, number, month in zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                      _FROM_OCTOBER_NUMBERS)]
        sql = ("SELECT c.MONTH_NAME, SUM(x) AS REVENUE, c.Y AS Y, c.M AS M FROM t JOIN c ON 1 = 1 "
               "GROUP BY c.MONTH_NAME, c.Y, c.M ORDER BY Y, M")
        _twelve_months_across_a_new_year_are_a_series(_sql_card(rows, "monthly revenue for the last 12 months", sql))

    def test_in_french_too(self):
        names = ["octobre", "novembre", "décembre", "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
                 "septembre"]
        rows = [{"NOM_MOIS": name, "CA": value, "AN": number, "NUM_MOIS": month}
                for name, value, number, month in zip(names, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS, _FROM_OCTOBER_NUMBERS)]
        sql = ("SELECT NOM_MOIS, SUM(x) AS CA, AN, NUM_MOIS FROM t GROUP BY NOM_MOIS, AN, NUM_MOIS "
               "ORDER BY AN, NUM_MOIS")
        card = _sql_card(rows, "chiffre d'affaires mensuel des 12 derniers mois", sql, "fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert _plain(card["answer"]["headline"]) == "Septembre a terminé à 1 000."

    def test_a_window_of_six_months_across_the_new_year(self):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "Y": number, "M": month}
                for name, value, number, month in zip(_FROM_OCTOBER[:6], _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                      _FROM_OCTOBER_NUMBERS)]
        sql = ("SELECT c.MONTH_NAME, SUM(x) AS REVENUE, c.Y AS Y, c.M AS M FROM t JOIN c ON 1 = 1 "
               "GROUP BY c.MONTH_NAME, c.Y, c.M ORDER BY Y, M")
        card = _sql_card(rows, "monthly revenue for the last 6 months", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "March closed at 800."

    @pytest.mark.parametrize("key", ["SortOrder", "SORT_KEY", "KEY", "ORD", "SEQ", "RN"])
    def test_months_listed_by_a_key_the_statement_takes_from_the_date(self, key):
        rows = [{"MONTH_NAME": name, "REVENUE": value, key: number * 100 + month}
                for name, value, number, month in zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                      _FROM_OCTOBER_NUMBERS)]
        sql = (f"SELECT DATENAME(month, d) AS MONTH_NAME, SUM(x) AS REVENUE, YEAR(d) * 100 + MONTH(d) AS {key} FROM t "
               f"GROUP BY DATENAME(month, d), YEAR(d), MONTH(d) ORDER BY {key}")
        _twelve_months_across_a_new_year_are_a_series(_sql_card(rows, "monthly revenue for the last 12 months", sql))

    @pytest.mark.parametrize("key", ["SortOrder", "SORT_KEY", "KEY", "ORD", "SEQ", "RN"])
    def test_months_listed_by_a_stored_key_of_year_and_month(self, key):
        rows = [{"MONTH_NAME": name, "REVENUE": value, key: number * 100 + month}
                for name, value, number, month in zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                      _FROM_OCTOBER_NUMBERS)]
        sql = f"SELECT MONTH_NAME, SUM(x) AS REVENUE, {key} FROM t GROUP BY MONTH_NAME, {key} ORDER BY {key}"
        _twelve_months_across_a_new_year_are_a_series(_sql_card(rows, "monthly revenue for the last 12 months", sql))

    @pytest.mark.parametrize("year,week", [("Y", "W"), ("AN", "SEMAINE"), ("YYYY", "WK")])
    def test_weeks_listed_by_a_year_and_a_week_the_statement_takes_from_the_date(self, year, week):
        rows = [{"WEEK_LABEL": label, "REVENUE": value, year: key // 100, week: key % 100}
                for label, value, key in zip(_WEEKS_ACROSS_A_YEAR, _FROM_OCTOBER_REVENUE, _WEEK_KEYS)]
        sql = (f"SELECT WEEK_LABEL, SUM(x) AS REVENUE, YEAR(d) AS {year}, DATEPART(week, d) AS {week} FROM t "
               f"GROUP BY WEEK_LABEL, YEAR(d), DATEPART(week, d) ORDER BY {year}, {week}")
        _weeks_across_a_new_year_are_a_series(_sql_card(rows, "weekly revenue for the last 6 weeks", sql))

    @pytest.mark.parametrize("key", ["SORT_KEY", "YW", "SEQ_KEY"])
    def test_weeks_listed_by_a_stored_key_of_year_and_week(self, key):
        rows = [{"WEEK_LABEL": label, "REVENUE": value, key: number}
                for label, value, number in zip(_WEEKS_ACROSS_A_YEAR, _FROM_OCTOBER_REVENUE, _WEEK_KEYS)]
        sql = f"SELECT WEEK_LABEL, SUM(x) AS REVENUE, {key} FROM t GROUP BY WEEK_LABEL, {key} ORDER BY {key}"
        _weeks_across_a_new_year_are_a_series(_sql_card(rows, "weekly revenue for the last 6 weeks", sql))

    def test_a_week_the_statement_takes_from_the_date_with_one_missing_is_a_series(self):
        """No sale in week 51: the keys stand two apart, and only what the statement says they are says so."""
        labels = ["Week 50", "Week 52", "Week 1", "Week 2"]
        keys = [202450, 202452, 202501, 202502]
        rows = [{"WEEK_LABEL": label, "REVENUE": value, "YW": key} for label, value, key in zip(labels, (700.0, 760.0, 650.0, 680.0), keys)]
        sql = ("SELECT WEEK_LABEL, SUM(x) AS REVENUE, YEAR(d) * 100 + DATEPART(week, d) AS YW FROM t "
               "GROUP BY WEEK_LABEL, YEAR(d), DATEPART(week, d) ORDER BY YW")
        card = _sql_card(rows, "weekly revenue for the last 6 weeks", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "Week 2 closed at 680."

    def test_two_months_across_the_new_year_are_a_series(self):
        rows = [{"MONTH_NAME": "December", "REVENUE": 900.0, "SORT_KEY": 202412}, {"MONTH_NAME": "January", "REVENUE": 800.0, "SORT_KEY": 202501}]
        sql = "SELECT MONTH_NAME, SUM(x) AS REVENUE, SORT_KEY FROM t GROUP BY MONTH_NAME, SORT_KEY ORDER BY SORT_KEY"
        assert _sql_card(rows, "monthly revenue for the last 2 months", sql)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("months,order", [
        (["October", "November", "December", "January"], "in"), (["January", "December", "November", "October"], "in"),
        (["December", "January"], "in"), (["March"], "in"), (["January", "February", "March"], "in"),
        # In the calendar's order or the reverse of it, with a month skipped.
        (["March", "April", "June"], "in"), (["June", "April", "March"], "in"),
        (_FROM_OCTOBER, "in"), (_FROM_OCTOBER[::-1], "in"),
        # Two years of names, one after another.
        (["November", "December", "January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December", "January"], "in"),
        # A month that stands out of its place, or one that is skipped, says nothing.
        (["October", "December", "January"], ""), (["October", "November", "January"], ""),
        (["December", "October", "November"], ""), (["March", "January", "February"], ""),
        (["October", "November", "October"], ""), (["November", "November", "December", "January"], "")])
    def test_month_names_are_in_time_one_month_after_another_from_any_month(self, months, order):
        from core.response_builder import _period_order

        rows = [{"MONTH_NAME": month, "REVENUE": 100.0 + number} for number, month in enumerate(months)]
        assert _period_order(rows) == order

    @pytest.mark.parametrize("keys,run", [
        ([202411, 202412, 202501], True), ([202412, 202501, 202502, 202503], True), ([202501, 202412, 202411], True),
        ([202410, 202501, 202504], True), ([202401, 202501, 202601], True), ([202412, 202501], False),
        ([202410, 202411, 202501], False), ([202411, 202412, 202412], False), ([202411, 202413, 202415], True),
        ([202401, 202402, 202403], True), ([190001, 190002, 190003], True),
        # A figure that holds a year and a month is as it was; so is one with a month that is none, or a year out of range.
        ([100, 200, 400], False), ([219912, 220001, 220002], False), ([189911, 189912, 190001], False),
        ([202313, 202407, 202413], False), ([202400, 202406, 202500], False)])
    def test_a_key_of_year_and_month_that_runs_on_across_a_new_year_is_a_run(self, keys, run):
        from core.response_builder import _is_a_sequence

        assert _is_a_sequence([{"KEY": key} for key in keys], "KEY") is run

    @pytest.mark.parametrize("keys,run", [
        ([202451, 202452, 202501], True), ([202452, 202453, 202501, 202502], True), ([202450, 202451, 202452, 202501], True),
        ([202453, 202501, 202502], True), ([202502, 202501, 202452, 202451], True),
        ([202450, 202452, 202501], False), ([202451, 202453, 202501], False), ([202451, 202452, 202502], False),
        ([202451, 202501, 202502], False), ([202454, 202501, 202502], False), ([202451, 202452, 202552], False),
        # Numbers that only look like a year and a week: below 1900 or above 2199.
        ([451, 452, 501], False), ([219951, 219952, 220001], False), ([189951, 189952, 190001], False),
        # A new year follows in the next year only.
        ([202452, 202601, 202602], False)])
    def test_a_key_of_year_and_week_that_runs_on_across_a_new_year_is_a_run(self, keys, run):
        from core.response_builder import _is_a_sequence

        assert _is_a_sequence([{"KEY": key} for key in keys], "KEY") is run

    def test_a_ranking_by_the_figure_is_still_a_ranking(self):
        """Months out of every order of time, listed by what they sold, are ranked."""
        rows = [{"MONTH_NAME": name, "REVENUE": value} for name, value in (("October", 900.0), ("December", 800.0), ("January", 700.0))]
        sql = "SELECT MONTH_NAME, SUM(x) AS REVENUE FROM t GROUP BY MONTH_NAME ORDER BY SUM(x) DESC"
        card = _sql_card(rows, "revenue by month", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "October leads at 900."

    @pytest.mark.parametrize("expression,figure", [
        ("YEAR(d)", False), ("MONTH(d)", False), ("YEAR(d) * 100 + MONTH(d)", False), ("DATEPART(week, d)", False),
        ("DATEPART(year, d) * 100 + DATEPART(month, d)", False), ("EXTRACT(MONTH FROM d)", False),
        ("DAY(d)", False), ("QUARTER(d)", False), ("DAYOFWEEK(d)", False), ("DATE_PART('month', d)", False),
        ("DAYOFYEAR(d)", False), ("DAYOFMONTH(d)", False), ("WEEKOFYEAR(d)", False), ("ISOWEEK(d)", False),
        ("YEARWEEK(d)", False), ("ISO_WEEK(d)", False), ("WEEKISO(d)", False), ("WEEK(d)", False),
        ("YEAROFWEEK(d)", False), ("YEAROFWEEKISO(d)", False), ("CAST(FORMAT(d, 'yyyyMM') AS INT)", False),
        ("CAST(TO_CHAR(d, 'YYYYMM') AS INT)", False), ("CAST(DATE_FORMAT(d, '%Y%m') AS INT)", False),
        ("CAST(STRFTIME('%Y%m', d) AS INT)", False), ("DATE_PART('month', d)", False),
        ("DATEFROMPARTS(YEAR(d), MONTH(d), 1)", False), ("ROW_NUMBER() OVER (ORDER BY YEAR(d), MONTH(d))", False),
        ("SUM(x)", True), ("COUNT(*)", True), ("AVG(x)", True), ("YEAR(d) + SUM(x)", True), ("x", True),
        ("ROUND(SUM(x), 2)", True), ("DENSE_RANK() OVER (ORDER BY SUM(x) DESC)", True), ("x * 100", True),
        ("DATEDIFF(day, d, e)", True), ("DATEDIFF(day, MIN(d), MAX(d))", True)])
    def test_a_column_the_statement_defines_as_a_part_of_a_date_is_no_figure(self, expression, figure):
        from core.response_builder import _ordered_by_a_figure

        rows = [{"LABEL": name, "REVENUE": value, "KEY": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4.5, 9.5, 5.5, 7.5))]
        sql = f"SELECT LABEL, SUM(x) AS REVENUE, {expression} AS KEY FROM t GROUP BY LABEL ORDER BY KEY"
        assert _ordered_by_a_figure(sql, rows) is figure

    @pytest.mark.parametrize("expression,db_type", [
        ("DAYOFWEEKISO(d)", "snowflake"), ("DATEPART(week, d)", "snowflake"), ("DATEPART(week, d)", "oracle"),
        ("CAST(DATENAME(year, d) AS INT)", "snowflake"), ("CAST(TO_CHAR(d, 'YYYYMM') AS INT)", "oracle"),
        ("YEAROFWEEK(d)", "snowflake"), ("YEAR(d) * 100 + MONTH(d)", "snowflake"), ("EXTRACT(MONTH FROM d)", "oracle")])
    def test_a_part_of_a_date_in_the_dialect_the_statement_is_written_in(self, expression, db_type):
        from core.response_builder import _ordered_by_a_figure

        rows = [{"LABEL": name, "REVENUE": value, "KEY": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4.5, 9.5, 5.5, 7.5))]
        sql = f"SELECT LABEL, SUM(x) AS REVENUE, {expression} AS KEY FROM t GROUP BY LABEL ORDER BY KEY"
        assert _ordered_by_a_figure(sql, rows, False, db_type) is False

    def test_the_statement_is_read_once_however_many_keys_ask(self, monkeypatch):
        from core import response_builder
        from core.response_builder import _ordered_by_a_figure

        reads = []
        real = response_builder._parsed_sql
        monkeypatch.setattr(response_builder, "_parsed_sql", lambda sql, db_type="azure_sql": reads.append(sql) or real(sql, db_type))
        response_builder._select_definitions.cache_clear()
        rows = [{"LABEL": name, "REVENUE": value, "KEY": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4.5, 9.5, 5.5, 7.5))]
        sql = "SELECT LABEL, SUM(x) AS REVENUE, YEAR(d) AS KEY FROM t GROUP BY LABEL ORDER BY KEY"
        for _ in range(3):
            assert _ordered_by_a_figure(sql, rows) is False
        assert reads == [sql]

    def test_a_column_defined_in_a_subquery_or_a_cte_is_read_through_it(self):
        from core.response_builder import _ordered_by_a_figure

        rows = [{"LABEL": name, "REVENUE": value, "Y": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4, 9, 5, 7))]
        subquery = ("SELECT LABEL, REVENUE, Y FROM (SELECT LABEL, SUM(x) AS REVENUE, YEAR(d) AS Y FROM t GROUP BY LABEL, YEAR(d)) s "
                    "ORDER BY Y")
        cte = ("WITH s AS (SELECT LABEL, SUM(x) AS REVENUE, YEAR(d) AS Y FROM t GROUP BY LABEL, YEAR(d)) "
               "SELECT LABEL, REVENUE, Y FROM s ORDER BY Y")
        for sql in (subquery, cte, subquery.replace("ORDER BY Y", "ORDER BY s.Y"), subquery.replace("ORDER BY Y", 'ORDER BY "Y"')):
            assert _ordered_by_a_figure(sql, rows) is False, sql

    def test_a_name_that_two_columns_share_is_a_figure_if_either_is(self):
        from core.response_builder import _ordered_by_a_figure

        rows = [{"LABEL": name, "REVENUE": value, "Y": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4, 9, 5, 7))]
        sql = ("SELECT LABEL, REVENUE, Y FROM (SELECT LABEL, SUM(x) AS REVENUE, SUM(z) AS Y FROM t GROUP BY LABEL) s "
               "JOIN (SELECT YEAR(d) AS Y FROM u) v ON 1 = 1 ORDER BY Y")
        assert _ordered_by_a_figure(sql, rows) is True

    def test_a_statement_that_cannot_be_read_defines_nothing(self):
        from core.response_builder import _select_definitions

        assert _select_definitions("SELECT FROM WHERE (", "azure_sql") == {}

    @pytest.mark.parametrize("values,std_dev", [
        ([1.0, 2.0, 3.0], 1.0), ([1.0, 2.0, 4.0], 1.53), ([1.0, 2.0], None), ([], None), ([1.0, float("nan"), 3.0], None),
        ([1.0, float("inf"), 3.0], None), ([1.0, float("-inf"), 3.0], None), ([2.0, 2.0, 2.0], 0.0)])
    def test_the_spread_of_values_is_none_where_one_is_no_number(self, values, std_dev):
        from core.response_builder import _std_dev

        assert _std_dev(values) == std_dev

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_measure_that_is_no_number_is_read_without_a_crash(self, bad):
        from core.response_builder import summarize_result_context

        rows = [{"MONTH_NAME": "January", "REVENUE": bad, "SORT_ORDER": 1}, {"MONTH_NAME": "February", "REVENUE": 1.0, "SORT_ORDER": 2},
                {"MONTH_NAME": "March", "REVENUE": 2.0, "SORT_ORDER": 3}]
        ctx = summarize_result_context(rows, "top 3 months by revenue",
                                       "SELECT MONTH_NAME, REVENUE, SORT_ORDER FROM t ORDER BY REVENUE DESC")
        assert ctx["mode"] == "ranking"
        assert ctx["distribution_stats"]["std_dev"] is None
        plain = summarize_result_context([{"REVENUE": bad}, {"REVENUE": 1.0}, {"REVENUE": 2.0}], "revenue", "SELECT REVENUE FROM t")
        assert plain["mode"] == "numeric_table"
        assert plain["distribution_stats"]["std_dev"] is None


class TestARankingWordAboutAnotherPeriodRanksNoMonth:
    """"Monthly revenue in our best year" and "daily sales during our busiest week" rank a year and a week; their
    months stand in time, counted 1, 2, 3 by a key. Only periods that are out of time order, or are two, are
    ranked by it."""

    _BY_NAME = "SELECT MONTH_NAME, SUM(x) AS REVENUE, MONTHNUM FROM t GROUP BY MONTH_NAME, MONTHNUM ORDER BY MONTHNUM"

    @pytest.mark.parametrize("question", [
        "monthly revenue in our best year", "monthly revenue for the year with the most orders",
        "daily sales during our busiest week", "monthly sales for customers in the top quarter by spend",
        "monthly revenue with the highest year-over-year growth", "monthly revenue with the highest days sales outstanding",
        "best day of the week to post monthly revenue", "monthly revenue with the lowest months of inventory"])
    def test_months_in_time_counted_by_a_key_are_a_series(self, question):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "MONTHNUM": number}
                for number, (name, value) in enumerate(zip(_MONTH_NAMES, _CLIMB), 1)]
        _a_climb_read_as_a_series(_sql_card(rows, question, self._BY_NAME), _MONTH_NAMES)

    @pytest.mark.parametrize("column,numbers", [
        ("MONTHNUM", range(4, 10)), ("MNTH", range(4, 10)), ("DAY_NUM", range(10, 16)), ("SORT_ORDER", range(10, 70, 10))])
    def test_dated_months_counted_by_a_key_are_a_series(self, column, numbers):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value, column: number}
                for month, value, number in zip(range(4, 10), _CLIMB, numbers)]
        sql = _BY_KEY.format(label="ORDER_MONTH", column=column, order=column)
        card = _sql_card(rows, "monthly revenue in our best year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "September 2025 closed at 100."

    def test_three_months_in_time_counted_by_a_key_are_a_series(self):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "MONTHNUM": number}
                for number, (name, value) in enumerate(zip(_MONTH_NAMES[3:], (90.0, 70.0, 100.0)), 4)]
        card = _sql_card(rows, "monthly revenue in our best year", self._BY_NAME)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "June closed at 100."

    def test_weeks_that_cannot_be_placed_are_ranked_by_a_count_the_question_ranks_by(self):
        """Orders counted 5, 4, 3, 2, 1 beside a revenue that is no run: the count is what the weeks are listed by."""
        rows = [{"WEEK_LABEL": week, "REVENUE": revenue, "ORDERS": orders}
                for week, revenue, orders in (("Week 7", 900.0, 5), ("Week 3", 750.0, 4), ("Week 9", 720.0, 3),
                                              ("Week 1", 400.0, 2), ("Week 5", 310.0, 1))]
        sql = ("SELECT WEEK_LABEL, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY WEEK_LABEL "
               "ORDER BY ORDERS DESC")
        card = _sql_card(rows, "top 5 weeks by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "Week 7 leads at 900."

    @pytest.mark.parametrize("year,month", [("Y", "M"), ("AN", "NUM_MOIS"), ("YYYY", "MM")])
    @pytest.mark.parametrize("labels", ["names", "dates"])
    def test_months_in_time_listed_by_a_year_and_a_month_column_are_a_series_whatever_year_is_asked(self, year, month, labels):
        """The year column holds a number, and the months across a new year stand in time: "our best year" is no
        ranking of them."""
        label = {"names": "MONTH_NAME", "dates": "ORDER_MONTH"}[labels]
        rows = []
        for index, (name, value, number, numeral) in enumerate(zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                                    _FROM_OCTOBER_NUMBERS)):
            shown = name if labels == "names" else date(number, numeral, 1)
            rows.append({label: shown, "REVENUE": value, year: number, month: numeral})
        sql = (f"SELECT c.{label}, SUM(x) AS REVENUE, c.{year} AS {year}, c.{month} AS {month} FROM t JOIN c ON 1 = 1 "
               f"GROUP BY c.{label}, c.{year}, c.{month} ORDER BY {year}, {month}")
        card = _sql_card(rows, "monthly revenue in our best year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"].startswith("September")

    @pytest.mark.parametrize("order", ["SUM(x) DESC", "REVENUE DESC", "2 DESC", "ROUND(SUM(x), 0) DESC"])
    def test_months_in_time_listed_by_what_they_add_up_are_ranked_where_the_question_ranks_them(self, order):
        rows = [{"MONTH_NAME": name, "REVENUE": value} for name, value in (("January", 900.0), ("February", 800.0), ("March", 700.0))]
        sql = f"SELECT MONTH_NAME, SUM(x) AS REVENUE FROM t GROUP BY MONTH_NAME ORDER BY {order}"
        card = _sql_card(rows, "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "January leads at 900."

    def test_months_in_time_listed_by_the_measure_of_a_table_are_ranked_where_the_question_ranks_them(self):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "SORT_ORDER": number}
                for number, (name, value) in enumerate((("January", 900.0), ("February", 800.0), ("March", 700.0)), 1)]
        sql = "SELECT MONTH_NAME, REVENUE, SORT_ORDER FROM monthly ORDER BY REVENUE DESC"
        card = _sql_card(rows, "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "January leads at 900."

    @pytest.mark.parametrize("keys", ["Y, M", "YYYY, M", "M"])
    def test_months_in_time_listed_by_the_year_and_month_columns_of_a_table_are_a_series_whatever_year_is_asked(self, keys):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "Y": number, "M": numeral, "YYYY": number}
                for name, value, number, numeral in zip(_FROM_OCTOBER, _FROM_OCTOBER_REVENUE, _FROM_OCTOBER_YEARS,
                                                        _FROM_OCTOBER_NUMBERS)]
        sql = f"SELECT MONTH_NAME, REVENUE, Y, M, YYYY FROM monthly ORDER BY {keys}"
        card = _sql_card(rows, "monthly revenue in our best year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("sql", [
        # A measure the statement names twice, once as what it adds up and once as the column it passes on.
        "SELECT MONTH_NAME, REVENUE AS REVENUE FROM (SELECT MONTH_NAME, SUM(x) AS REVENUE FROM t GROUP BY MONTH_NAME) s "
        "ORDER BY REVENUE DESC",
        # A table's own columns: the result's first measure, whatever periods stand before it, qualified or not.
        "SELECT ORDER_YEAR, MONTH_NAME, REVENUE FROM monthly ORDER BY REVENUE DESC",
        "SELECT m.ORDER_YEAR, m.MONTH_NAME, m.REVENUE FROM monthly m ORDER BY m.REVENUE DESC",
        'SELECT [ORDER_YEAR], [MONTH_NAME], [REVENUE] FROM [monthly] ORDER BY [REVENUE] DESC'])
    def test_months_in_time_listed_by_the_measure_are_ranked_however_the_statement_names_it(self, sql):
        rows = [{"ORDER_YEAR": 2025, "MONTH_NAME": name, "REVENUE": value}
                for name, value in (("January", 900.0), ("February", 800.0), ("March", 700.0))]
        if not sql.startswith("SELECT ORDER_YEAR") and "m.ORDER_YEAR" not in sql and "[ORDER_YEAR]" not in sql:
            rows = [{"MONTH_NAME": row["MONTH_NAME"], "REVENUE": row["REVENUE"]} for row in rows]
        card = _sql_card(rows, "top 3 months by revenue", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "January leads at 900."

    def test_the_cut_reads_a_column_the_statement_takes_from_a_date_as_no_figure(self):
        rows = [{"ORDER_MONTH": date(2025 if month > 0 else 2024, abs(month) or 12, 1), "REVENUE": value, "Y": 2025}
                for month, value in ((4, 80.0), (5, 85.0), (6, 90.0), (7, 70.0), (8, 95.0), (9, 100.0))]
        rows[0]["Y"] = 2024
        sql = ("SELECT * FROM (SELECT TOP 6 ORDER_MONTH, SUM(x) AS REVENUE, YEAR(d) AS Y FROM t GROUP BY ORDER_MONTH, YEAR(d) "
               "ORDER BY Y DESC) s ORDER BY ORDER_MONTH")
        card = _sql_card(rows, "the last 6 months of revenue for our top customer", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "September 2025 closed at 100."

    def test_the_statements_own_dialect_reads_what_a_column_is_in_the_ranking_too(self):
        from core.response_builder import _ranked_by_a_figure

        rows = [{"LABEL": name, "REVENUE": value, "KEY": key}
                for name, value, key in zip(("Week 4", "Week 9", "Week 5", "Week 7"), (10.0, 20.0, 30.0, 25.0), (4.5, 9.5, 5.5, 7.5))]
        sql = "SELECT LABEL, SUM(x) AS REVENUE, DAYOFWEEKISO(d) AS KEY FROM t GROUP BY LABEL ORDER BY KEY"
        assert _ranked_by_a_figure("revenue by week", sql, rows, "snowflake") is False
        assert _ranked_by_a_figure("revenue by week", sql, rows, "azure_sql") is True

    def test_months_in_time_listed_by_a_count_that_is_a_name_for_one_are_ranked_where_asked(self):
        rows = [{"MONTH_NAME": name, "REVENUE": value, "ORDERS": count}
                for name, value, count in (("January", 900.0, 50), ("February", 700.0, 30), ("March", 650.0, 20))]
        sql = ("SELECT MONTH_NAME, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY MONTH_NAME "
               "ORDER BY ORDERS DESC")
        assert _sql_card(rows, "top 3 months by number of orders", sql)["analysis_contract"]["mode"] == "ranking"

    def test_two_months_still_rank_by_the_key_the_question_ranks_by(self):
        """Two rows are a run only as 1, 2 or 0, 1: of two months the ask is read as it always was."""
        rows = [{"MONTH_NAME": "March", "ORDERS": 5, "MONTHNUM": 5}, {"MONTH_NAME": "June", "ORDERS": 4, "MONTHNUM": 4}]
        sql = "SELECT MONTH_NAME, COUNT(*) AS ORDERS, MONTHNUM FROM t GROUP BY MONTH_NAME, MONTHNUM ORDER BY MONTHNUM DESC"
        assert _sql_card(rows, "the best 2 months by orders", sql)["analysis_contract"]["mode"] == "ranking"

    def test_months_out_of_time_order_are_still_ranked_by_a_count_the_question_ranks_by(self):
        rows = [{"MONTH_NAME": name, "ORDERS": count} for name, count in (("March", 5), ("June", 4), ("January", 3), ("May", 2), ("April", 1))]
        sql = "SELECT MONTH_NAME, COUNT(*) AS ORDERS FROM t GROUP BY MONTH_NAME ORDER BY ORDERS DESC"
        card = _sql_card(rows, "top 5 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "March leads at 5."

    _CUT = ("SELECT * FROM (SELECT TOP 6 ORDER_MONTH, SUM(x) AS REVENUE, {key} FROM t GROUP BY ORDER_MONTH, {group} "
            "ORDER BY {order} DESC) s ORDER BY {order}")

    @pytest.mark.parametrize("column,order,group,numbers", [
        ("DENSE_RANK() OVER (ORDER BY YEAR(d), MONTH(d)) AS SEQ", "SEQ", "d", range(1, 7)),
        ("MONTHNUM", "MONTHNUM", "MONTHNUM", range(4, 10)),
        ("MONTH_SEQ", "MONTH_SEQ", "MONTH_SEQ", range(64, 70))])
    def test_the_last_months_of_a_top_customer_are_a_series(self, column, order, group, numbers):
        name = column.rpartition(" AS ")[2]
        rows = [{"ORDER_MONTH": date(2025, month, 1), "REVENUE": value, name: number}
                for month, value, number in zip(range(4, 10), _CLIMB, numbers)]
        sql = self._CUT.format(key=column, group=group, order=order)
        card = _sql_card(rows, "the last 6 months of revenue for our top customer", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "September 2025 closed at 100."

    def test_a_cut_by_a_count_the_question_ranks_by_is_a_ranking(self):
        rows = [{"ORDER_MONTH": date(2025, month, 1), "ORDERS": count, "RN": number}
                for number, (month, count) in enumerate(((3, 50), (6, 40), (9, 30)), 1)]
        sql = ("SELECT * FROM (SELECT ORDER_MONTH, COUNT(*) AS ORDERS, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS RN "
               "FROM t GROUP BY ORDER_MONTH) s WHERE RN <= 3 ORDER BY RN")
        assert _sql_card(rows, "top 3 months by number of orders", sql)["analysis_contract"]["mode"] == "ranking"


class TestAMonthNamedAsWhatThePeriodsAreIsTheirNameWhateverGrainIsAsked:
    """"January revenue by year" lists Januaries: "January 2025", not "2025" -- unless the column says its rows are
    years or dates, or the month qualifies something other than the periods."""

    @pytest.mark.parametrize("question,grain", [
        ("January revenue by year", "year|month:1"), ("January sales each year", "year|month:1"),
        ("revenue for January of each year", "year|month:1"), ("compare January revenue by year", "year|month:1"),
        ("what did Jan sell each year", "year|month:1"), ("sales in March for each year", "year|month:3"),
        ("Q1 revenue by year", "year|quarter:1"), ("first quarter sales for each year", "year|quarter:1"),
        ("revenue by year for January", "year|month:1"), ("January revenue by quarter", "quarter|month:1"),
        ("January revenue for the last 3 years", "year|month:1")])
    def test_which_grain_and_month_the_question_names(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question, _with_plan("year")) == grain

    @pytest.mark.parametrize("column", ["PERIOD", "ORDER_MONTH", "MONTH_START", "BUCKET"])
    @pytest.mark.parametrize("question", [
        "January revenue by year", "January sales each year", "revenue for January of each year",
        "compare January revenue by year"])
    def test_januaries_are_named_januaries(self, column, question):
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        card = _sql_card(rows, question, "SELECT 1")
        assert card["answer"]["headline"] == "January 2025 closed at 400."

    @pytest.mark.parametrize("column", ["QUARTER_START", "ORDER_QUARTER", "QTR", "PERIOD"])
    @pytest.mark.parametrize("question", ["Q1 revenue by year", "first quarter sales for each year"])
    def test_first_quarters_are_named_first_quarters(self, column, question):
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        assert _sql_card(rows, question, "SELECT 1")["answer"]["headline"] == "Q1 2025 closed at 400."

    @pytest.mark.parametrize("column", ["ORDER_DATE", "SALE_DATE", "DATE_VENTE"])
    def test_a_column_named_as_a_date_keeps_its_dates_whatever_it_is_asked_by(self, column):
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        for question in ("January revenue by year", "Q1 revenue by year"):
            assert _sql_card(rows, question, "SELECT 1")["answer"]["headline"] == "2025-01-01 closed at 400."

    @pytest.mark.parametrize("column", ["ORDER_DATE", "SALE_DATE", "PERIOD"])
    def test_a_date_column_asked_by_year_with_no_month_named_is_years(self, column):
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        assert _sql_card(rows, "revenue by year", "SELECT 1")["answer"]["headline"] == "2025 closed at 400."

    def test_a_date_column_of_months_stays_months_where_one_of_them_is_named(self):
        """Only the firsts of one month a year are the dates a question names a month of; here the question names
        January and asks for the months."""
        rows = [{"ORDER_DATE": date(2025, month, 1), "REVENUE": 100.0 * month} for month in range(1, 7)]
        card = _sql_card(rows, "monthly revenue including January", "SELECT 1")
        assert card["answer"]["headline"] == "June 2025 closed at 600."

    @pytest.mark.parametrize("lang,question,grain", [
        ("fr", "chiffre d'affaires de janvier par année", "year|month:1"), ("fr", "ventes du T1 par année", "year|quarter:1")])
    def test_in_french(self, lang, question, grain):
        from core.response_builder import requested_period_grain

        with _InLanguage(lang):
            assert requested_period_grain(question) == grain


class TestAMonthThatQualifiesSomethingElseNamesNoPeriod:
    """"Revenue from the January cohort over the last 4 years" is the cohort's: the months are not Januaries."""

    @pytest.mark.parametrize("noun", ["cohort", "cohorts", "promotion", "promotions", "promo", "promos", "campaign",
                                      "campaigns", "launch", "launches", "intake", "intakes", "class", "classes"])
    def test_a_month_before_a_noun_that_is_a_group_of_customers_or_a_plan(self, noun):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"revenue from the January {noun} over the last 4 years") == ""
        assert requested_period_grain(f"Q1 {noun} revenue by year") == "year"

    @pytest.mark.parametrize("words", [
        "excluding", "excluding the", "except", "except the", "without", "without the", "who joined in", "who joined in the",
        "who acquired in", "launched in", "who registered in", "who enrolled in", "who onboarded in", "hired in",
        "who arrived in", "who signed up in", "whose year starts in", "where the year starts in", "starting in",
        "that begins in", "beginning in", "which ends in", "ending in", "who start in the", "who begin in", "who end in"])
    def test_a_month_after_what_leaves_it_out_or_dates_a_start(self, words):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"revenue for customers {words} January over the last 4 years") == ""

    @pytest.mark.parametrize("question", [
        "revenue from the January cohort over the last 4 years", "revenue of customers acquired in January, last 4 years",
        "revenue excluding January for the last 4 years", "how much did the January promotion bring in over the last 4 years",
        "revenue for the January campaign over the last 4 years"])
    def test_the_periods_of_such_a_question_are_not_januaries(self, question):
        rows = [{"PERIOD": date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        card = _sql_card(rows, question, "SELECT 1")
        assert card["answer"]["headline"] == "2025 closed at 400."

    def test_one_row_with_a_month_left_out(self):
        card = _sql_card([{"PERIOD": date(2025, 1, 1), "REVENUE": 900.0}], "revenue for 2025 excluding January", "SELECT 1")
        assert card["answer"]["headline"] == "2025-01-01 closed at 900."

    @pytest.mark.parametrize("question", [
        "January revenue for the last 4 years", "revenue for January over the last 4 years",
        "revenue in January over the last 4 years", "orders shipped in January over the last 4 years"])
    def test_a_month_that_qualifies_nothing_else_still_names_its_periods(self, question):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == "|month:1"

    def test_what_is_left_out_must_stand_right_before_the_month(self):
        from core.response_builder import requested_period_grain

        assert requested_period_grain("revenue excluding refunds for January over the last 4 years") == "|month:1"
        assert requested_period_grain("revenue except in March over the last 4 years") == "|month:3"

    @pytest.mark.parametrize("verb", ["joined", "acquired", "launched", "registered", "enrolled", "hired", "arrived", "opened",
                                      "released", "introduced", "signed", "signed up", "created", "inscrits"])
    def test_a_verb_that_dates_a_group_needs_the_word_that_dates_it(self, verb):
        """"Customers who joined in January" are January's group; "who joined January" says the month and no more -- the
        question is read as before."""
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"revenue by year for customers who {verb} January") == "year|month:1"
        assert requested_period_grain(f"revenue by year for customers who {verb} in January") == "year"

    @pytest.mark.parametrize("word", ["subclass", "classic", "promotional", "launchpad", "intaken"])
    def test_a_word_that_only_begins_as_one_of_them_is_none(self, word):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"January {word} revenue over the last 4 years") == "|month:1"


    @pytest.mark.parametrize("words", [
        # What is left out, in English and in the French a question is canonicalised from ("hors", "sauf", "sans" and
        # "en excluant" stay as they are).
        "excluding", "excl.", "excl", "except", "without", "minus", "less", "apart from", "aside from", "other than",
        "not counting", "not including", "leaving out", "omitting", "skipping", "ignoring", "barring", "besides",
        "but not", "save", "ex", "hors", "sauf", "sans", "mais pas", "excepte", "a part", "a share", "en excluant",
        "a exclusion of", "exclusion of"])
    def test_a_month_or_quarter_after_more_words_that_leave_it_out_names_no_period(self, words):
        from core.response_builder import requested_period_grain

        for month in ("January", "Q1"):
            assert requested_period_grain(f"revenue by year {words} {month}") == "year"
            assert requested_period_grain(f"revenue by year {words} the {month}") == "year"

    @pytest.mark.parametrize("words", [
        # What dates a group of customers, accounts or products, said with a verb or by the group's name.
        "started in", "began in", "opened in", "released in", "introduced in", "signed in", "signed up in", "created in",
        "acquis en", "arrives en", "inscrit en", "inscrits en", "inscrits of", "cohort of", "cohorts of", "cohorte of",
        "promotion of", "promotions of", "campaign of", "campagne of", "intake of"])
    def test_a_month_or_quarter_after_what_dates_a_group_names_no_period(self, words):
        from core.response_builder import requested_period_grain

        for month in ("January", "Q1"):
            assert requested_period_grain(f"revenue by year for customers {words} {month}") == "year"

    @pytest.mark.parametrize("words", [
        "starts", "starting", "begins", "beginning", "ends", "ending", "commencant", "commence",
        "starts in", "starting in", "ending in", "commencant en", "commence en"])
    def test_a_month_a_year_starts_or_ends_in_names_no_period_with_or_without_in(self, words):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"revenue by fiscal year {words} April") == "year"

    @pytest.mark.parametrize("noun", ["forecast", "forecasts", "target", "targets", "budget", "budgets", "plan", "plans",
                                      "goal", "goals", "quota", "quotas", "benchmark", "benchmarks"])
    def test_a_month_or_quarter_before_what_the_periods_are_compared_with_names_no_period(self, noun):
        from core.response_builder import requested_period_grain

        for month in ("January", "Q1"):
            assert requested_period_grain(f"revenue by year against the {month} {noun}") == "year"

    @pytest.mark.parametrize("question", [
        "chiffre d'affaires par ann\u00e9e hors janvier", "chiffre d'affaires par ann\u00e9e sauf janvier",
        "chiffre d'affaires par ann\u00e9e sans janvier", "chiffre d'affaires par ann\u00e9e \u00e0 l'exclusion de janvier",
        "chiffre d'affaires par ann\u00e9e en excluant janvier", "chiffre d'affaires par ann\u00e9e de la cohorte de janvier",
        "chiffre d'affaires par ann\u00e9e except\u00e9 janvier", "chiffre d'affaires par ann\u00e9e \u00e0 part janvier",
        "chiffre d'affaires par ann\u00e9e, mais pas janvier",
        "chiffre d'affaires par ann\u00e9e pour la promotion de janvier",
        "chiffre d'affaires par ann\u00e9e des clients acquis en janvier",
        "chiffre d'affaires par ann\u00e9e des clients arriv\u00e9s en janvier",
        "chiffre d'affaires par ann\u00e9e des inscrits de janvier", "chiffre d'affaires par ann\u00e9e hors T1",
        "chiffre d'affaires par exercice commen\u00e7ant en avril"])
    def test_in_french_a_month_that_qualifies_something_else_names_no_period(self, question):
        from core.response_builder import requested_period_grain

        with _InLanguage("fr"):
            assert requested_period_grain(question) == "year"

    @pytest.mark.parametrize("lang,question,headline", [
        ("en", "revenue by year, not counting January", "2025 closed at 400."),
        ("en", "revenue by year, other than January", "2025 closed at 400."),
        ("en", "revenue by year for customers who started in January", "2025 closed at 400."),
        ("en", "revenue by year against the Q1 target", "2025 closed at 400."),
        ("en", "revenue by fiscal year starting April", "2025 closed at 400."),
        ("fr", "chiffre d'affaires par ann\u00e9e hors janvier", "2025 a termin\u00e9 \u00e0 400.")])
    def test_the_years_of_such_a_question_are_years_not_januaries(self, lang, question, headline):
        rows = [{"PERIOD": date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        assert _sql_card(rows, question, "SELECT 1", lang=lang)["answer"]["headline"] == headline

    @pytest.mark.parametrize("question,grain", [
        # A word that only ends as a cue does is none: weekends, trends, spending, unless.
        ("revenue by year on weekends in January", "year|month:1"), ("revenue by year, sales trends in March", "year|month:3"),
        ("revenue by year, average spending in January", "year|month:1"), ("revenue by year, unless January", "year|month:1"),
        ("revenue by year for the cohorts in January", "year|month:1"), ("revenue by year for hairless January", "year|month:1"),
        # The cues are read in any case.
        ("Revenue by year Excluding January", "year"), ("REVENUE BY YEAR EXCLUDING JANUARY", "year"),
        ("revenue by year Not Counting Q1", "year"), ("Revenue by year, OTHER THAN January", "year"),
        ("revenue by year for customers Who Joined In March", "year"), ("revenue by fiscal year STARTING April", "year")])
    def test_a_cue_is_a_whole_word_in_any_case(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == grain

    def test_a_noun_that_is_a_qualifier_must_follow_the_month_directly(self):
        from core.response_builder import requested_period_grain

        assert requested_period_grain("January revenue for the cohort over the last 4 years") == "|month:1"
        assert requested_period_grain("Q1 revenue against the target over the last 4 years") == "|quarter:1"
        assert requested_period_grain("January revenue over the last 4 years") == "|month:1"

    @pytest.mark.parametrize("word", ["planet", "planning", "targeted", "budgetary", "goalie", "quotation", "forecasting"])
    def test_a_word_that_only_begins_as_a_compared_with_noun_is_none(self, word):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(f"January {word} revenue over the last 4 years") == "|month:1"


class TestADayTheQuestionNamesIsTheGrainAsked:
    """"Sales on January 1 each year" are the first of January's: a year is how often they are asked for."""

    @pytest.mark.parametrize("question", [
        "sales on January 1 each year", "sales on January 1st each year", "sales on 1 January each year",
        "sales on the first of January each year", "sales on the 1st of January for each year",
        "sales on the first day of November each year", "revenue by year on March 15"])
    def test_the_grain_is_the_day(self, question):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == "day"

    def test_in_french(self):
        from core.response_builder import requested_period_grain

        with _InLanguage("fr"):
            assert requested_period_grain("ventes le 1er janvier de chaque année") == "day"
            assert requested_period_grain("chiffre d'affaires par année du 1er mars au 31 mars") == "year"
            assert requested_period_grain("chiffre d'affaires par année jusqu'au 30 juin") == "year"
            assert requested_period_grain("chiffre d'affaires par année entre le 1er mars et le 31 mars") == "year"
            assert requested_period_grain("chiffre d'affaires par année du 15 mars au 1er avril") == "year"
            assert requested_period_grain("chiffre d'affaires par année pour le 15 mars et le 1er avril") == "year"

    @pytest.mark.parametrize("question", [
        "revenue by year from January 1, 2022", "revenue by year since March 1", "revenue by year until June 30",
        "revenue by year between January 1 and March 31", "revenue by year from 15 January to 15 March",
        "January 5 - March 10 revenue by year", "revenue by year before March 1st", "revenue by year after June 15",
        "revenue by year from 1st March to 31 March", "revenue by year from March 1st to March 31st",
        "revenue by year since the 15th of June", "revenue by year until 30 June",
        # No bound word before the range: the days beside its months say it is one.
        "revenue by year for 1st March and the 31st March", "revenue by year for 1st March to 31st March",
        "revenue by year for 15 March to 1st April",
        # An article stands between the bound word and the month.
        "revenue by year until the March report", "revenue by year since the January launch"])
    def test_a_day_that_is_the_edge_of_a_window_is_no_day_asked_for(self, question):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == "year"

    @pytest.mark.parametrize("question,grain", [("monthly revenue on March 15", "month"), ("sales on March 15 by quarter", "quarter")])
    def test_only_a_year_asked_gives_way_to_a_day(self, question, grain):
        from core.response_builder import requested_period_grain

        assert requested_period_grain(question) == grain

    @pytest.mark.parametrize("column", ["ORDER_DATE", "SALE_DATE", "DATE_VENTE", "PERIOD", "BUCKET"])
    @pytest.mark.parametrize("question", ["sales on January 1 each year", "sales on the first of January each year"])
    def test_the_firsts_of_january_stay_the_days_they_are(self, column, question):
        rows = [{column: date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        assert _sql_card(rows, question, "SELECT 1")["answer"]["headline"] == "2025-01-01 closed at 400."

    def test_a_year_asked_by_the_window_it_names_is_still_the_year(self):
        rows = [{"PERIOD": date(year, 1, 1), "REVENUE": 100.0 * number} for number, year in enumerate((2022, 2023, 2024, 2025), 1)]
        card = _sql_card(rows, "revenue by year from January 1, 2022", "SELECT 1")
        assert card["answer"]["headline"] == "2025 closed at 400."


def _weeks_of_two_years() -> list[dict]:
    return [{"WEEK_LABEL": label, "REVENUE": revenue, "Y": year, "W": week}
            for label, year, week, revenue in (("Week 50", 2024, 50, 700.0), ("Week 51", 2024, 51, 720.0),
                                               ("Week 52", 2024, 52, 760.0), ("Week 1", 2025, 1, 650.0),
                                               ("Week 2", 2025, 2, 680.0), ("Week 3", 2025, 3, 800.0))]


class TestRowsListedByAPartOfTheirDatesAreASeriesHoweverTheKeyIsWritten:
    """A statement that lists its rows by what it defines as a part of a date -- YEAR(d) AS Y, DATEPART(week, d) AS W,
    MIN(YEAR(d) * 100 + DATEPART(week, d)) AS SORT_KEY -- lists them by their period, whether it writes the key by its
    name, by its position (ORDER BY 3) or as the first or last of a group."""

    @pytest.mark.parametrize("order", ["3, 4", "3", "4", "Y, W", "Y", "c.Y, c.W", "3 DESC, 4 DESC"])
    def test_weeks_listed_by_the_position_of_their_year_and_week_are_a_series(self, order):
        sql = ("SELECT WEEK_LABEL, SUM(x) AS REVENUE, YEAR(d) AS Y, DATEPART(week, d) AS W FROM t c "
               f"GROUP BY WEEK_LABEL, YEAR(d), DATEPART(week, d) ORDER BY {order}")
        card = _sql_card(_weeks_of_two_years(), "weekly revenue for the last 6 weeks", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "Week 3 closed at 800."

    def test_a_position_that_names_a_figure_is_still_one(self):
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "ORDERS": orders}
                for name, revenue, orders in (("February", 800.0, 70), ("March", 700.0, 60), ("January", 900.0, 50))]
        sql = "SELECT MONTH_NAME, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY MONTH_NAME ORDER BY 3 DESC"
        card = _sql_card(rows, "top 3 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("position", ["9", "3", "0"])
    def test_a_position_outside_the_columns_names_nothing(self, position):
        from core.response_builder import _key_is_a_figure

        rows = [{"MONTH_NAME": "March", "REVENUE": 900.0}, {"MONTH_NAME": "January", "REVENUE": 800.0}]
        assert _key_is_a_figure(position, rows, False, {"revenue": ("aggregate",)}) is False

    @pytest.mark.parametrize("key", ["MIN(YEAR(d) * 100 + DATEPART(week, d))", "MAX(YEAR(d) * 100 + DATEPART(week, d))",
                                     "MIN(YEAR(d)) * 100 + MIN(DATEPART(week, d))"])
    def test_weeks_listed_by_the_first_or_last_of_a_date_part_are_a_series(self, key):
        """Week 51 has no row: the keys 202450, 202452, 202501 are no run, and are no figure either."""
        rows = [{"WEEK_LABEL": label, "REVENUE": revenue, "SORT_KEY": number}
                for label, number, revenue in (("Week 50", 202450, 700.0), ("Week 52", 202452, 760.0), ("Week 1", 202501, 650.0),
                                               ("Week 2", 202502, 680.0), ("Week 3", 202503, 800.0))]
        sql = f"SELECT WEEK_LABEL, SUM(x) AS REVENUE, {key} AS SORT_KEY FROM t GROUP BY WEEK_LABEL ORDER BY SORT_KEY"
        card = _sql_card(rows, "weekly sales of product X for the last 6 weeks", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "Week 3 closed at 800."

    def test_a_difference_of_date_parts_is_a_length_of_time_and_a_figure(self):
        rows = [{"MONTH_NAME": name, "AGE": age} for name, age in (("March", 61.0), ("January", 58.0), ("February", 52.0))]
        sql = "SELECT MONTH_NAME, MAX(YEAR(GETDATE()) - YEAR(birth)) AS AGE FROM t GROUP BY MONTH_NAME ORDER BY AGE DESC"
        card = _sql_card(rows, "top 3 months by the age of the oldest customer", sql)
        assert card["analysis_contract"]["mode"] == "ranking"
        assert card["answer"]["headline"] == "March leads at 61."

    def test_what_a_statement_defines_as_the_first_or_last_of_a_date_part_or_a_length_of_time(self):
        from core.response_builder import _select_definitions

        defined = _select_definitions(
            "SELECT a, MIN(YEAR(d) * 100 + MONTH(d)) AS FIRST_KEY, MAX(MONTH(d)) AS LAST_MONTH, "
            "MAX(YEAR(GETDATE()) - YEAR(birth)) AS AGE, MAX(YEAR(d)) - MIN(YEAR(d)) AS SPAN, MAX(amount) AS BIGGEST, "
            "MIN(order_date) AS FIRST_DAY, MIN(YEAR(d)) * 100 + SUM(x) AS MIXED, MIN(YEAR(d)) * 100 + MIN(MONTH(d)) AS BOTH "
            "FROM t GROUP BY a")
        assert defined == {"first_key": ("date",), "last_month": ("date",), "age": ("aggregate",), "span": ("aggregate",),
                           "biggest": ("aggregate",), "first_day": ("aggregate",), "mixed": ("aggregate",), "both": ("date",)}

    @pytest.mark.parametrize("window,kind", [
        ("DENSE_RANK() OVER (ORDER BY YEAR(d), MONTH(d))", "date"), ("RANK() OVER (ORDER BY YEAR(d), MONTH(d))", "date"),
        ("ROW_NUMBER() OVER (ORDER BY YEAR(d), MONTH(d))", "date"), ("RANK() OVER (ORDER BY SUM(x) DESC)", "aggregate"),
        ("DENSE_RANK() OVER (ORDER BY SUM(x) DESC, YEAR(d))", "aggregate"), ("ROW_NUMBER() OVER (ORDER BY name)", "other"),
        ("SUM(x) OVER (ORDER BY YEAR(d))", "aggregate"), ("RANK() OVER (PARTITION BY k)", "aggregate"),
        ("ROW_NUMBER() OVER ()", "other"), ("DENSE_RANK() OVER (PARTITION BY k ORDER BY YEAR(d))", "date"),
        # Only the order says what a rank counts, not what it is partitioned by.
        ("ROW_NUMBER() OVER (PARTITION BY COUNT(*) ORDER BY YEAR(d))", "date"),
        ("RANK() OVER (PARTITION BY COUNT(*) ORDER BY YEAR(d))", "date"),
        ("ROW_NUMBER() OVER (PARTITION BY k ORDER BY SUM(x) DESC)", "aggregate")])
    def test_what_a_statement_defines_as_a_rank(self, window, kind):
        from core.response_builder import _select_definitions

        assert _select_definitions(f"SELECT {window} AS RK FROM t") == {"rk": (kind,)}


class TestQuarterKeysThatRunOnAcrossANewYear:
    """A quarter's key, YYYYQ -- 20243, 20244, 20251 -- is a sort key like a month's or a week's, whatever it is called."""

    @staticmethod
    def _is_a_sequence(keys) -> bool:
        from core.response_builder import _is_a_sequence

        return _is_a_sequence([{"K": key} for key in keys], "K")

    @pytest.mark.parametrize("keys", [
        [20243, 20244, 20251], [20244, 20251, 20252, 20253], [20251, 20252, 20253, 20254], [20254, 20261, 20262],
        [19004, 19011, 19012], [21992, 21993, 21994], [21504, 21511, 21512],
        # The first and the last quarter a key can name, among others that run on in unequal steps of the numbers.
        [19001, 19002, 19003, 19004, 19011], [21984, 21991, 21992, 21993, 21994]])
    def test_quarters_in_equal_steps_are_a_run(self, keys):
        assert self._is_a_sequence(keys)

    @pytest.mark.parametrize("keys", [
        [20243, 20244, 20252], [20243, 20251, 20252], [20243, 20245, 20251], [20240, 20241, 20243], [20241, 20243, 20244],
        [18993, 18994, 19001], [21993, 21994, 22001], [20243, 20244, 20251, 20253],
        # A key with a fifth quarter or a quarter 0 is none, whatever steps it would make as quarters.
        [20243, 20244, 20245, 20252], [20243, 20250, 20251], [20244, 20250, 20251, 20252]])
    def test_quarters_with_a_gap_or_out_of_range_are_not(self, keys):
        assert not self._is_a_sequence(keys)

    @pytest.mark.parametrize("column", ["QUARTER_KEY", "YEAR_QTR", "QTR_ID", "YQ", "SORT_KEY"])
    def test_quarters_listed_by_a_year_and_quarter_key_are_a_series(self, column):
        labels = ("FY24 Q3", "FY24 Q4", "FY25 Q1", "FY25 Q2", "FY25 Q3", "FY25 Q4")
        keys = (20243, 20244, 20251, 20252, 20253, 20254)
        rows = [{"QUARTER": label, "REVENUE": revenue, column: key}
                for label, key, revenue in zip(labels, keys, (100.0, 200.0, 300.0, 400.0, 700.0, 800.0))]
        sql = f"SELECT QUARTER, SUM(x) AS REVENUE, {column} FROM t GROUP BY QUARTER, {column} ORDER BY {column}"
        card = _sql_card(rows, "revenue by fiscal quarter", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "FY25 Q4 closed at 800."

    def test_a_month_key_and_a_week_key_are_read_to_the_end_of_the_range(self):
        # 2150 is far from now, and inside the range a key may take: only a key beyond 2199 is none.
        assert self._is_a_sequence([215011, 215012, 215101])
        assert self._is_a_sequence([219911, 219912, 220001]) is False
        from core.response_builder import _weeks_in_a_row

        assert _weeks_in_a_row([215051, 215052, 215101])
        assert not _weeks_in_a_row([219952, 220001])

    def test_a_week_zero_is_no_week_that_a_new_year_follows(self):
        assert not self._is_a_sequence([202500, 202601, 202602])


class TestTheCutOfPeriodsIsGivenTheAskOfTheirOwnRanking:
    """"The last 6 months of revenue in our best year" cuts its months by their key, MONTHNUM DESC, inside the statement:
    the question's "best year" is no ask to rank those months, whatever the key is written as."""

    _MONTHS = [(date(2025, month, 1), revenue) for month, revenue in zip(range(4, 10), (80.0, 85.0, 90.0, 70.0, 95.0, 100.0))]

    @pytest.mark.parametrize("question", ["the last 6 months of revenue in our best year", "monthly revenue in our best year"])
    @pytest.mark.parametrize("inner,column,numbers", [
        ("ORDER_MONTH, SUM(x) AS REVENUE, MONTHNUM FROM t GROUP BY ORDER_MONTH, MONTHNUM ORDER BY MONTHNUM DESC",
         "MONTHNUM", range(4, 10)),
        ("ORDER_MONTH, SUM(x) AS REVENUE, MONTH_SEQ FROM t GROUP BY ORDER_MONTH, MONTH_SEQ ORDER BY MONTH_SEQ DESC",
         "MONTH_SEQ", range(64, 70)),
        ("ORDER_MONTH, SUM(x) AS REVENUE, DENSE_RANK() OVER (ORDER BY YEAR(d), MONTH(d)) AS SEQ FROM t "
         "GROUP BY ORDER_MONTH, YEAR(d), MONTH(d) ORDER BY SEQ DESC", "SEQ", range(64, 70)),
        ("ORDER_MONTH, SUM(x) AS REVENUE, MONTHNUM FROM t GROUP BY ORDER_MONTH, MONTHNUM ORDER BY 3 DESC",
         "MONTHNUM", range(4, 10))])
    def test_months_cut_by_a_key_are_a_series(self, question, inner, column, numbers):
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue, column: number}
                for (month, revenue), number in zip(self._MONTHS, numbers)]
        card = _sql_card(rows, question, f"SELECT * FROM (SELECT TOP 6 {inner}) s ORDER BY {column}")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "September 2025 closed at 100."

    def test_months_cut_by_what_they_add_up_to_are_still_ranked_where_the_question_ranks_them(self):
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue, "ORDERS": orders}
                for (month, revenue), orders in zip(self._MONTHS[:3], (30, 20, 10))]
        sql = ("SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY ORDER_MONTH "
               "ORDER BY ORDERS DESC) s ORDER BY ORDERS DESC")
        card = _sql_card(rows, "top 3 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"


class TestAColumnNamedANIsAYear:
    """A French calendar table calls its year AN or AN_CIVIL, as ANNEE: where it holds years it is one, and the weeks and
    months a statement lists by it and by their number run in time across a new year."""

    @pytest.mark.parametrize("name,values,expected", [
        ("AN", [2022, 2023, 2024, 2025], True), ("AN_CIVIL", [2022, 2023, 2024, 2025], True), ("an", [2024, 2025], True),
        ("AN_FISCAL", [2022, 2023, 2024, 2025], True), ("AN", [202401, 202402, 202403], True),
        # An amount for the year, a count, a number that is no year, a word that only ends as it does.
        ("CA_AN", [1000.0, 1200.0, 900.0, 1500.0], False), ("AN", [3, 5, 8, 13], False), ("MAN", [2022, 2023, 2024, 2025], False),
        ("BAN_ID", [2022, 2023, 2024, 2025], False), ("ANNEE", [2022, 2023, 2024, 2025], True)])
    def test_which_columns_named_an_are_periods(self, name, values, expected):
        from core.temporal_columns import is_calendar_period_column

        assert is_calendar_period_column(name, values) is expected

    _WEEKS = [(2024, 50), (2024, 51), (2024, 52), (2025, 1), (2025, 2), (2025, 3)]
    _VALUES = (700.0, 720.0, 760.0, 650.0, 680.0, 800.0)

    @pytest.mark.parametrize("year", ["AN", "AN_CIVIL", "ANNEE", "EXERCICE"])
    def test_weeks_listed_by_a_year_and_a_week_column_are_a_series(self, year):
        label = "Semaine {w}"
        rows = [{"LIBELLE_SEMAINE": label.format(w=week), "VENTES": value, year: number, "SEMAINE": week}
                for (number, week), value in zip(self._WEEKS, self._VALUES)]
        sql = (f"SELECT c.LIBELLE_SEMAINE, SUM(f.montant) AS VENTES, c.{year}, c.SEMAINE FROM ventes f JOIN calendrier c "
               f"ON c.jour = f.jour GROUP BY c.LIBELLE_SEMAINE, c.{year}, c.SEMAINE ORDER BY c.{year}, c.SEMAINE")
        card = _sql_card(rows, "ventes hebdomadaires des 6 derni\u00e8res semaines", sql, lang="fr")
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"].startswith(label.format(w=3))

    @pytest.mark.parametrize("year", ["AN", "AN_CIVIL", "ANNEE"])
    def test_months_listed_by_a_year_and_a_month_column_are_a_series(self, year):
        months = ["oct.", "nov.", "d\u00e9c.", "janv.", "f\u00e9vr.", "mars"]
        numbers = [(2024, 10), (2024, 11), (2024, 12), (2025, 1), (2025, 2), (2025, 3)]
        rows = [{"LIBELLE_MOIS": f"{name} {str(number)[2:]}", "VENTES": value, year: number, "MOIS": month}
                for name, (number, month), value in zip(months, numbers, self._VALUES)]
        sql = (f"SELECT c.LIBELLE_MOIS, SUM(f.montant) AS VENTES, c.{year}, c.MOIS FROM ventes f JOIN calendrier c "
               f"ON c.jour = f.jour GROUP BY c.LIBELLE_MOIS, c.{year}, c.MOIS ORDER BY c.{year}, c.MOIS")
        card = _sql_card(rows, "ventes mensuelles des 6 derniers mois", sql, lang="fr")
        assert card["analysis_contract"]["mode"] == "time_series"


class TestTheFirstOrLastOfAFigureIsAFigure:
    """MIN or MAX of a part of a date is a key; MIN or MAX of a figure that only reads a date part to pick it -- the amount of
    December, the price of 2025 -- is the figure itself, and months listed by it are ranked by it."""

    @pytest.mark.parametrize("expression", [
        "MAX(CASE WHEN MONTH(d) = 12 THEN amount END)", "MIN(CASE WHEN YEAR(d) = 2025 THEN price END)",
        "MAX(IIF(DATEPART(quarter, d) = 4, amount, 0))", "MAX(amount * MONTH(d))",
        "ROUND(MAX(CASE WHEN MONTH(d) = 12 THEN amount END), 0)", "COALESCE(MAX(CASE WHEN MONTH(d) = 12 THEN amount END), 0)",
        "MAX(YEAR(GETDATE()) - YEAR(birth))", "MAX(YEAR(d)) - MIN(YEAR(d))", "MAX(YEAR(d) * 12 + MONTH(d) + amount)",
        "RANK() OVER (ORDER BY MAX(CASE WHEN MONTH(d) = 12 THEN amount END))", "SUM(YEAR(d))", "AVG(MONTH(d))", "COUNT(YEAR(d))",
        # The first or last of what is no part of a date, a constant among them, is a figure too.
        "MAX(1)", "MIN(0)", "MAX('x')", "MIN(GETDATE())"])
    def test_what_reads_a_date_part_beside_a_figure_is_a_figure(self, expression):
        from core.response_builder import _select_definitions

        assert _select_definitions(f"SELECT a, {expression} AS K FROM t GROUP BY a") == {"k": ("aggregate",)}

    @pytest.mark.parametrize("expression", [
        "MIN(YEAR(d) * 100 + MONTH(d))", "MAX(YEAR(d) * 100 + MONTH(d))", "MIN(YEAR(d) * 12 + MONTH(d) - 1)",
        "MIN(YEAR(d) * 12 + MONTH(d)) - 1", "MAX(DATEPART(week, d))", "MIN(YEAR(d)) * 100 + MIN(MONTH(d))",
        "MIN(YEAR(CAST(d AS DATE)) * 100 + MONTH(CAST(d AS DATE)))", "MIN(YEARWEEK(d))"])
    def test_what_takes_the_first_or_last_of_a_date_part_is_a_key(self, expression):
        from core.response_builder import _select_definitions

        assert _select_definitions(f"SELECT a, {expression} AS K FROM t GROUP BY a") == {"k": ("date",)}

    @pytest.mark.parametrize("figure", [
        "MAX(CASE WHEN MONTH(d) = 12 THEN amount END)", "MAX(amount * MONTH(d))", "MIN(CASE WHEN YEAR(d) = 2025 THEN price END)"])
    def test_months_out_of_time_order_listed_by_it_are_ranked(self, figure):
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "PEAK": peak}
                for name, revenue, peak in (("March", 700.0, 900.0), ("January", 650.0, 800.0), ("February", 720.0, 610.0))]
        sql = f"SELECT MONTH_NAME, SUM(x) AS REVENUE, {figure} AS PEAK FROM t GROUP BY MONTH_NAME ORDER BY PEAK DESC"
        card = _sql_card(rows, "top 3 months by peak amount", sql)
        assert card["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("key", ["²", "³", "\u0663", "3\u00b2"])
    def test_a_position_written_with_other_digits_is_no_position(self, key):
        from core.response_builder import _key_is_a_figure, _named_by

        rows = [{"MONTH_NAME": "March", "REVENUE": 900.0, "ORDERS": 30}, {"MONTH_NAME": "January", "REVENUE": 800.0, "ORDERS": 20}]
        assert _named_by(key, rows) == key.lower()
        assert _key_is_a_figure(key, rows, False, {}) is False

    def test_a_position_is_counted_from_1_and_no_further_than_the_columns(self):
        from core.response_builder import _named_by

        rows = [{"MONTH_NAME": "March", "REVENUE": 900.0}]
        assert [_named_by(key, rows) for key in ("0", "1", "2", "3")] == ["0", "month_name", "revenue", "3"]

    def test_an_order_by_a_superscript_is_read_without_a_crash(self):
        rows = [{"MONTH_NAME": "March", "REVENUE": 900.0}, {"MONTH_NAME": "January", "REVENUE": 800.0}]
        card = _sql_card(rows, "revenue by month", "SELECT MONTH_NAME, SUM(x) AS REVENUE FROM t GROUP BY MONTH_NAME ORDER BY \u00b2")
        assert card["answer"]["headline"]


class TestACutsOwnPositionIsItsOwnColumn:
    """A subquery that keeps its top periods orders by its own columns: ORDER BY 2 is the second column it selects, not the
    second of the result outside it."""

    _MONTHS = [(date(2025, month, 1), revenue) for month, revenue in zip(range(4, 10), (80.0, 85.0, 90.0, 70.0, 95.0, 100.0))]

    def test_a_position_names_the_cuts_own_sort_key(self):
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue, "MONTHNUM": number}
                for (month, revenue), number in zip(self._MONTHS, range(4, 10))]
        sql = ("SELECT s.ORDER_MONTH, s.REVENUE, s.MONTHNUM FROM (SELECT TOP 6 ORDER_MONTH, MONTHNUM, SUM(x) AS REVENUE FROM t "
               "GROUP BY ORDER_MONTH, MONTHNUM ORDER BY 2 DESC) s ORDER BY s.MONTHNUM")
        card = _sql_card(rows, "monthly revenue in our best year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"
        assert card["answer"]["headline"] == "September 2025 closed at 100."

    def test_a_position_names_the_cuts_own_count(self):
        rows = [{"ORDER_MONTH": month, "ORDERS": orders, "REVENUE": revenue}
                for (month, revenue), orders in zip((self._MONTHS[2], self._MONTHS[0], self._MONTHS[1]), (30, 20, 10))]
        sql = ("SELECT s.ORDER_MONTH, s.ORDERS, s.REVENUE FROM (SELECT TOP 3 ORDER_MONTH, SUM(x) AS REVENUE, COUNT(*) AS ORDERS "
               "FROM t GROUP BY ORDER_MONTH ORDER BY 3 DESC) s ORDER BY s.ORDERS DESC")
        card = _sql_card(rows, "top 3 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"

    def test_an_unaliased_aggregate_at_a_position_is_a_figure(self):
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue, "ORDERS": orders}
                for (month, revenue), orders in zip((self._MONTHS[2], self._MONTHS[0], self._MONTHS[1]), (30, 20, 10))]
        sql = ("SELECT * FROM (SELECT TOP 3 ORDER_MONTH, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY ORDER_MONTH "
               "ORDER BY COUNT(*) DESC) s ORDER BY ORDERS DESC")
        assert _sql_card(rows, "top 3 months by number of orders", sql)["analysis_contract"]["mode"] == "ranking"

    @pytest.mark.parametrize("key", ["\u00b2", "\u00b3"])
    def test_a_cut_ordered_by_a_superscript_is_read_without_a_crash(self, key):
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue} for month, revenue in self._MONTHS]
        sql = f"SELECT * FROM (SELECT TOP 6 ORDER_MONTH, SUM(x) AS REVENUE FROM t GROUP BY ORDER_MONTH ORDER BY {key} DESC) s"
        assert _sql_card(rows, "monthly revenue in our best year", sql)["answer"]["headline"]

    def test_a_star_leaves_a_position_to_be_read_as_before(self):
        from core.response_builder import _a_selects_own_key
        import sqlglot

        select = sqlglot.parse_one("SELECT TOP 3 * FROM t ORDER BY 3 DESC", read="tsql")
        assert _a_selects_own_key(select, select.args["order"].expressions[0].this) == "3"

    @pytest.mark.parametrize("sql,key,expected", [
        ("SELECT a, SUM(x) AS REVENUE, MONTHNUM FROM t ORDER BY 3", "3", "MONTHNUM"),
        ("SELECT a, SUM(x) AS REVENUE, MONTHNUM AS MN FROM t ORDER BY 3", "3", "MN"),
        ("SELECT a, SUM(x) AS REVENUE, COUNT(*) FROM t ORDER BY 3", "3", "COUNT(*)"),
        ("SELECT t.a, t.b FROM t ORDER BY 2", "2", "b"), ("SELECT a, b FROM t ORDER BY 3", "3", "3"),
        ("SELECT a, b FROM t ORDER BY 0", "0", "0"), ("SELECT a, b FROM t ORDER BY b", "b", "b"),
        ("SELECT a, * FROM t ORDER BY 2", "2", "2")])
    def test_what_a_position_names(self, sql, key, expected):
        from core.response_builder import _a_selects_own_key
        import sqlglot

        select = sqlglot.parse_one(sql, read="tsql")
        assert _a_selects_own_key(select, select.args["order"].expressions[0].this) == expected

    @pytest.mark.parametrize("text", ["\u00b2", "\u0663", "3\u00b2"])
    def test_a_key_written_with_other_digits_is_no_position_of_a_select(self, text):
        from core.response_builder import _a_selects_own_key
        import sqlglot

        class _Key:
            def sql(self):
                return text

        select = sqlglot.parse_one("SELECT a, b FROM t ORDER BY 1", read="tsql")
        assert _a_selects_own_key(select, _Key()) == text

    def test_a_key_the_statement_defines_as_a_plain_column_is_no_figure_of_a_cut(self):
        """MONTHNUM AS MN is a sort key whatever it is called: only a count or a sum the statement defines is a figure."""
        rows = [{"ORDER_MONTH": month, "REVENUE": revenue, "MN": number} for (month, revenue), number in zip(self._MONTHS, range(4, 10))]
        sql = ("SELECT * FROM (SELECT TOP 6 ORDER_MONTH, SUM(x) AS REVENUE, MONTHNUM AS MN FROM t GROUP BY ORDER_MONTH, MONTHNUM "
               "ORDER BY MN DESC) s ORDER BY MN")
        card = _sql_card(rows, "monthly revenue in our best year", sql)
        assert card["analysis_contract"]["mode"] == "time_series"

    _OUT_OF_ORDER = [("March", 700.0, 30), ("January", 650.0, 20), ("February", 720.0, 10)]
    _IN_ORDER = [("January", 700.0, 30), ("February", 650.0, 20), ("March", 720.0, 10)]
    _CUT = "SELECT * FROM (SELECT TOP 3 MONTH_NAME, REVENUE, ORDERS FROM monthly ORDER BY ORDERS DESC) s"

    def test_the_ask_of_the_question_makes_the_orders_of_a_table_a_figure_beside_another_figure(self):
        """3, 2, 1 orders are a count to rank by where the question ranks the months and they are out of time order -- and
        a sort key where another figure is what the result measures and the months are not out of order."""
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "ORDERS": orders} for name, revenue, orders in self._OUT_OF_ORDER]
        assert _sql_card(rows, "top 3 months by number of orders", self._CUT)["analysis_contract"]["mode"] == "ranking"
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "ORDERS": orders} for name, revenue, orders in self._IN_ORDER]
        assert _sql_card(rows, "top 3 months by number of orders", self._CUT)["analysis_contract"]["mode"] == "time_series"

    @pytest.mark.parametrize("key", ["m.ORDERS", "[ORDERS]", '"ORDERS"', "M.orders"])
    def test_a_count_the_statement_defines_is_a_figure_under_any_spelling_of_its_name(self, key):
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "ORDERS": orders} for name, revenue, orders in self._IN_ORDER]
        sql = ("SELECT TOP 3 * FROM (SELECT MONTH_NAME, SUM(x) AS REVENUE, COUNT(*) AS ORDERS FROM t GROUP BY MONTH_NAME) m "
               f"ORDER BY {key} DESC")
        assert _sql_card(rows, "top 3 months by number of orders", sql)["analysis_contract"]["mode"] == "ranking"

    def test_a_cut_by_a_sort_key_of_a_table_ranks_the_months_out_of_time_order_that_the_question_ranks(self):
        """The ask of the question counts on a cut as on an order: months out of time order, cut by the orders a table counts
        3, 2, 1, are ranked where the question asks for them ranked."""
        rows = [{"MONTH_NAME": name, "REVENUE": revenue, "ORDERS": orders}
                for name, revenue, orders in (("March", 900.0, 30), ("January", 800.0, 20), ("February", 700.0, 10))]
        sql = ("SELECT * FROM (SELECT TOP 3 MONTH_NAME, REVENUE, ORDERS FROM monthly ORDER BY ORDERS DESC) s")
        card = _sql_card(rows, "top 3 months by number of orders", sql)
        assert card["analysis_contract"]["mode"] == "ranking"



class TestALongExpressionIsReadInOnePass:
    """A select list of many figures taken one from another -- a P&L's SUM(revenue) - SUM(cogs) - SUM(opex) ... -- is read
    once. Reading each part of it again for every way to reach it took twice as long for every term added."""

    @staticmethod
    def _read_within(seconds, sql):
        from core.response_builder import _select_definitions

        def give_up(*_):
            raise TimeoutError(f"not read in {seconds}s")

        _select_definitions.cache_clear()
        previous = signal.signal(signal.SIGALRM, give_up)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return _select_definitions(sql)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="the guard is a SIGALRM")
    @pytest.mark.parametrize("expression,kind", [
        (" - ".join(f"SUM(a{i})" for i in range(40)), "aggregate"),
        (" - ".join(f"MIN(YEAR(d{i}))" for i in range(40)), "aggregate"),
        (" + ".join(f"MIN(YEAR(d{i}) * 100 + MONTH(d{i}))" for i in range(40)), "date"),
        ("MAX(" + " - ".join(f"YEAR(d{i})" for i in range(40)) + ")", "aggregate"),
        ("ROUND(" + " - ".join(f"SUM(a{i})" for i in range(40)) + ", 2)", "aggregate")])
    def test_a_long_chain_of_terms_is_read_in_a_blink(self, expression, kind):
        assert self._read_within(5, f"SELECT g, {expression} AS K FROM t GROUP BY g") == {"k": (kind,)}
