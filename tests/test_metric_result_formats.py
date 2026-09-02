import asyncio
import unittest
from datetime import date as _date, timedelta as _timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from core.response_builder import build_assistant_response, build_column_formats
from core.result_cache import ResultCache
from core.result_renderer import (
    _detect_column_format,
    _format_value,
    _rows_to_table,
    _send_results,
)


class _PlainAdapter:
    def __init__(self):
        self.messages = []

    async def send_message(self, event, message):
        self.messages.append(message)


class _RichAdapter:
    def __init__(self):
        self.payloads = []
        self.cached_formats = {}

    def cache_result(self, rows, question, sql, db_cfg, rag_context, **kwargs):
        self.cached_formats = kwargs.get("column_formats") or {}

    async def send_assistant_response(self, event, payload):
        self.payloads.append(payload)


class MetricResultFormatTests(unittest.TestCase):
    def test_column_name_detection_handles_camel_case(self):
        self.assertEqual(_detect_column_format("TotalRevenue"), "currency")
        self.assertEqual(_detect_column_format("GrossProfit"), "currency")
        self.assertEqual(_detect_column_format("RetentionRate"), "percent")
        self.assertEqual(_detect_column_format("ProfitMargin"), "percent")
        self.assertEqual(_detect_column_format("TotalEmployees"), "number")

    def test_plural_duration_words_do_not_fall_through_to_currency(self):
        # Regression: "AVG_DAYS_TO_PAY" tokenizes to {avg, days, to, pay}.
        # _DIMENSION_KEYWORDS only had the singular "day", not "days", so
        # this fell through to the currency check where "pay" matched,
        # rendering a plain-number row-calculated metric as "$1.00" instead
        # of "1". Confirmed against a real "Avg Days To Pay" metric whose
        # result_format is explicitly "number".
        self.assertEqual(_detect_column_format("AVG_DAYS_TO_PAY"), "number")
        self.assertEqual(_detect_column_format("DAYS_SALES_OUTSTANDING"), "number")
        self.assertEqual(_detect_column_format("WEEKS_ON_HAND"), "number")
        # Currency/percent detection for real money/rate columns must be unaffected
        self.assertEqual(_detect_column_format("TOTAL_SALES"), "currency")
        self.assertEqual(_detect_column_format("TOTAL_REVENUE"), "currency")

    def test_decimal_values_honor_inferred_and_explicit_formats(self):
        value = Decimal("52677.25")
        self.assertEqual(_format_value(value, "TotalRevenue"), "$52,677.25")
        self.assertEqual(
            _format_value(value, "TotalRevenue", "number"),
            "52,677.25",
        )
        self.assertEqual(
            _format_value(Decimal("12.5"), "MetricValue", "percentage"),
            "12.50%",
        )

    def test_plain_table_uses_explicit_metric_formats(self):
        table = _rows_to_table(
            [{"MetricValue": Decimal("52677.25"), "OrderCount": 12}],
            {"MetricValue": "currency", "OrderCount": "number"},
        )
        self.assertIn("$52,677.25", table)
        self.assertIn("12", table)

    def test_currency_metric_formats_measure_not_numeric_dimension(self):
        rows = [
            {"Warehouse": 1000450, "TotalRevenue": 52677.25},
            {"Warehouse": 1000547, "TotalRevenue": 40650.14},
        ]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(formats, {"TotalRevenue": "currency"})

    def test_percentage_metric_format_maps_percentage_alias(self):
        rows = [{"Division": "A", "ProfitPct": 12.5}]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Profit percentage", "result_format": "percentage"}],
            },
        )
        self.assertEqual(formats, {"ProfitPct": "percentage"})

    def test_llm_metric_context_does_not_format_unrelated_count(self):
        rows = [{"Warehouse": 1000450, "OrderCount": 12}]
        formats = build_column_formats(
            rows,
            display_context={
                "format_scope": "metric_context",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(formats, {})

    def test_response_payload_includes_generic_column_formats(self):
        rows = [{"Customer": "A", "MarginRate": 18.25}]
        payload = build_assistant_response(
            question="show margin rate by customer",
            rows=rows,
            sql="SELECT Customer, MarginRate FROM result",
            duration_ms=25,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Margin Rate", "result_format": "percentage"}],
            },
        )
        self.assertEqual(payload["data"]["column_formats"], {"MarginRate": "percentage"})
        self.assertEqual(payload["data"]["currency_columns"], [])
        self.assertIn("18.25%", payload["answer"]["headline"])

    def test_single_value_currency_metric_formats_headline(self):
        payload = build_assistant_response(
            question="what is total revenue",
            rows=[{"TotalRevenue": 52677.25}],
            sql="SELECT SUM(x) AS TotalRevenue FROM t",
            duration_ms=25,
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "currency"}],
            },
        )
        self.assertEqual(payload["answer"]["short_value"], "$52,677.25")
        self.assertIn("$52,677.25", payload["answer"]["headline"])

    def test_single_value_decimal_result_is_json_serializable(self):
        # Regression: pyodbc/Azure SQL returns decimal.Decimal for SUM() on a
        # numeric/decimal column. summarize_result_context's single-value
        # branch (one row, one column — exactly "what is the total ordered
        # quantity") stored rows[0][col] raw, unlike every other branch of
        # that function which normalizes through _to_float/_to_float_z. The
        # Decimal rode straight into analysis_contract and crashed
        # ws.send_json's JSON encoder downstream in gateway/web_adapter.py —
        # the query succeeded but the user got total silence with no error
        # ever reaching them (only a server-log line).
        import json
        payload = build_assistant_response(
            question="what is the total ordered quantity",
            rows=[{"TOTAL_ORDERED_QUANTITY": Decimal("48213.500")}],
            sql="SELECT SUM(qty) AS TOTAL_ORDERED_QUANTITY FROM orders",
            duration_ms=1200,
        )
        json.dumps(payload)  # must not raise TypeError
        self.assertEqual(payload["analysis_contract"]["value"], 48213.5)
        self.assertIsInstance(payload["analysis_contract"]["value"], float)

    def test_single_value_text_result_stays_string_not_lost(self):
        import json
        payload = build_assistant_response(
            question="what is the top customer name",
            rows=[{"TOP_CUSTOMER": "Acme Industries"}],
            sql="SELECT TOP 1 name AS TOP_CUSTOMER FROM customers",
            duration_ms=10,
        )
        json.dumps(payload)
        self.assertEqual(payload["analysis_contract"]["value"], "Acme Industries")

    def test_single_value_zero_decimal_not_dropped(self):
        # _to_float_z's docstring warns the `_to_float(v) or str(v)` idiom
        # silently zeroes-out legitimate falsy numeric values; confirm the
        # fix uses an explicit None-check instead.
        import json
        payload = build_assistant_response(
            question="what is the total returns",
            rows=[{"TOTAL_RETURNS": Decimal("0")}],
            sql="SELECT SUM(qty) AS TOTAL_RETURNS FROM returns",
            duration_ms=10,
        )
        json.dumps(payload)
        self.assertEqual(payload["analysis_contract"]["value"], 0.0)

    def test_result_cache_preserves_explicit_formats(self):
        cache = ResultCache()
        cache.store(
            "s1",
            [{"Warehouse": 1000450, "TotalRevenue": 52677.25, "MarginPct": 10.5}],
            column_formats={"TotalRevenue": "currency", "MarginPct": "percentage"},
        )
        self.assertEqual(
            cache.get_column_formats("s1"),
            {"TotalRevenue": "currency", "MarginPct": "percentage"},
        )
        self.assertEqual(cache.get_currency_columns("s1"), ["TotalRevenue"])

    def test_registry_number_overrides_currency_sounding_alias(self):
        formats = build_column_formats(
            [{"TotalRevenue": Decimal("52677.25")}],
            display_context={
                "format_scope": "metric_registry",
                "metrics": [{"name": "Total Revenue", "result_format": "number"}],
            },
        )
        self.assertEqual(formats, {"TotalRevenue": "number"})

    def test_plain_metric_response_uses_registry_format_with_decimal(self):
        adapter = _PlainAdapter()
        asyncio.run(
            _send_results(
                SimpleNamespace(schema_hint=""),
                adapter,
                "what is total revenue",
                [{"MetricValue": Decimal("52677.25")}],
                "SELECT 52677.25 AS MetricValue",
                25,
                None,
                "acct",
                {"db_type": "azure_sql"},
                display_context={
                    "format_scope": "metric_registry",
                    "metrics": [
                        {"name": "Total Revenue", "result_format": "currency"}
                    ],
                },
            )
        )
        self.assertIn("$52,677.25", adapter.messages[0])

    def test_rich_response_and_cache_receive_inferred_format(self):
        adapter = _RichAdapter()
        asyncio.run(
            _send_results(
                SimpleNamespace(schema_hint=""),
                adapter,
                "show total revenue",
                [{"TotalRevenue": Decimal("52677.25")}],
                "SELECT 52677.25 AS TotalRevenue",
                25,
                None,
                "acct",
                {"db_type": "azure_sql"},
            )
        )
        self.assertEqual(adapter.cached_formats, {"TotalRevenue": "currency"})
        self.assertEqual(
            adapter.payloads[0]["data"]["column_formats"],
            {"TotalRevenue": "currency"},
        )

    def test_rich_kpi_response_receives_dashboard_token(self):
        adapter = _RichAdapter()
        with patch("core.result_renderer._create_pin_token", return_value="kpi-token") as create:
            asyncio.run(
                _send_results(
                    SimpleNamespace(schema_hint="", platform="portal"),
                    adapter,
                    "what is total revenue",
                    [{"TotalRevenue": Decimal("52677.25")}],
                    "SELECT 52677.25 AS TotalRevenue",
                    25,
                    {"id": 7},
                    "acct",
                    {"id": 11, "db_type": "azure_sql"},
                )
            )
        payload = adapter.payloads[0]
        self.assertIsNotNone(payload["kpi"])
        self.assertEqual(payload["pin_token"], "kpi-token")
        self.assertEqual(payload["dashboard_item_type"], "kpi")
        self.assertEqual(create.call_args.kwargs["chart_type"], "kpi")

    def test_rich_table_response_receives_dashboard_token(self):
        adapter = _RichAdapter()
        rows = [{"Customer": "A"}, {"Customer": "B"}]
        with patch("core.result_renderer._create_pin_token", return_value="table-token") as create:
            asyncio.run(
                _send_results(
                    SimpleNamespace(schema_hint="", platform="portal"),
                    adapter,
                    "list customers",
                    rows,
                    "SELECT Customer FROM customers",
                    25,
                    {"id": 7},
                    "acct",
                    {"id": 11, "db_type": "azure_sql"},
                    display_context={"chart_type_override": "table"},
                )
            )
        payload = adapter.payloads[0]
        self.assertIsNone(payload["kpi"])
        self.assertEqual(payload["pin_token"], "table-token")
        self.assertEqual(payload["dashboard_item_type"], "table")
        self.assertEqual(create.call_args.kwargs["chart_type"], "table")


if __name__ == "__main__":
    unittest.main()


class MonthBucketsAreDeclaredDatesTests(unittest.TestCase):
    """A month bucket must not be displayed as the first of the month.

    Live defect, tenant Emco_test, 2026-08-25: "what is my revenue by month
    this year" returned six correct monthly rows whose period cells rendered
    as "2026-01-01", "2026-02-01" ... A month displayed as a specific day
    reads as one day's figure.

    The portal draws its own tables, and it ALREADY formats a declared date
    column as YYYY-MM. Nothing declared this one: `build_column_formats` only
    recognised a compact ERP integer (202601 / 20260131 -- that regex admits
    no separators), so a genuine DATE column, which reaches the browser as the
    ISO string "2026-01-01", was never marked and was printed verbatim.

    These call `build_column_formats` -- the real function whose output is
    sent to the browser as `column_formats` -- rather than asserting on
    template text.
    """

    def _monthly(self, column="PERIOD"):
        return [
            {column: _date(2026, month, 1), "TOTAL_REVENUE": Decimal(1000 * month)}
            for month in range(1, 7)
        ]

    def test_a_monthly_date_column_is_declared_a_date(self):
        self.assertEqual(build_column_formats(self._monthly()), {"PERIOD": "date"})

    def test_the_iso_string_form_is_declared_too(self):
        """By the time rows reach the browser the date has been isoformat()ed,
        so the string form is the one that actually matters."""
        rows = [
            {"PERIOD": f"2026-{month:02d}-01", "TOTAL_REVENUE": Decimal(month)}
            for month in range(1, 7)
        ]
        self.assertEqual(build_column_formats(rows), {"PERIOD": "date"})

    def test_a_period_repeated_per_category_still_counts(self):
        """"revenue by month by region" repeats each month once per region.
        Reading the cadence rather than the row order is what makes this work."""
        rows = [
            {"PERIOD": _date(2026, month, 1), "REGION": region, "REV": Decimal(1)}
            for month in range(1, 7)
            for region in ("N", "S", "E", "W")
        ]
        self.assertEqual(build_column_formats(rows).get("PERIOD"), "date")

    def test_a_true_daily_column_is_left_alone(self):
        """Load-bearing. The portal falls through to YYYY-MM for any date style
        it does not recognise, so declaring a daily column a date would collapse
        every day of a month onto one label."""
        rows = [
            {"DMS_DT": _date(2026, 1, day), "REV": Decimal(day)}
            for day in range(1, 20)
        ]
        self.assertEqual(build_column_formats(rows), {})

    def test_irregular_real_dates_are_left_alone(self):
        rows = [
            {"ORDER_DT": value, "REV": Decimal(1)}
            for value in (_date(2026, 1, 5), _date(2026, 1, 17),
                          _date(2026, 3, 2), _date(2026, 7, 29))
        ]
        self.assertEqual(build_column_formats(rows), {})

    def test_a_column_not_named_like_a_date_is_left_alone(self):
        rows = [
            {"AMOUNT": _date(2026, month, 1), "REV": Decimal(month)}
            for month in range(1, 7)
        ]
        self.assertEqual(build_column_formats(rows), {})

    def test_plain_numbers_in_a_day_named_column_are_not_dates(self):
        """DAYS_LATE = 5 is a count, not a calendar value."""
        rows = [{"DAYS_LATE": 5, "REV": Decimal(1)}, {"DAYS_LATE": 12, "REV": Decimal(2)}]
        self.assertEqual(build_column_formats(rows), {})

    def test_the_compact_erp_integer_path_still_works(self):
        """The branch that already existed must be untouched."""
        rows = [{"PRD_DMS_KEY": 202601, "REV": Decimal(1)},
                {"PRD_DMS_KEY": 202602, "REV": Decimal(2)}]
        self.assertEqual(build_column_formats(rows), {"PRD_DMS_KEY": "date"})

    def test_an_explicit_caller_format_still_wins(self):
        """Inference never overrides a declared format."""
        self.assertEqual(
            build_column_formats(self._monthly(), explicit_formats={"PERIOD": "text"}),
            {"PERIOD": "text"},
        )

    # ── The distinction that matters: BUCKET SHAPE, not cadence ─────────────
    #
    # The first version of this fix asked whether the values stepped about a
    # month apart. That is true of plenty of genuine day-precision data, and
    # relabelling it "2026-01" erases the exact thing the reader needs. A
    # governed bucket is always the FIRST day of its period, which is a shape
    # the server itself emits (DATEFROMPARTS(YEAR(x), MONTH(x), 1)), so that is
    # what these test.

    def test_invoices_due_on_the_fifteenth_keep_their_day(self):
        rows = [{"DUE_DATE": _date(2026, month, 15), "AMT": Decimal(1)}
                for month in range(1, 7)]
        self.assertEqual(build_column_formats(rows), {})

    def test_month_end_balance_dates_keep_their_day(self):
        """A snapshot's as-of date is the whole point of the snapshot, and
        these step ~30 days exactly like a bucket does."""
        rows = [{"BAL_DT": _date(2026, month, 1) - _timedelta(days=1), "AMT": Decimal(1)}
                for month in range(2, 8)]
        self.assertEqual(build_column_formats(rows), {})

    def test_a_daily_tail_below_the_sample_window_is_not_collapsed(self):
        """The format is applied to EVERY row, so it cannot be decided from a
        sample. Twenty monthly rows followed by a daily tail would otherwise
        merge 51 distinct dates onto 21 labels."""
        rows = (
            [{"ACTIVITY_DT": _date(2024, m, 1), "AMT": Decimal(1)} for m in range(1, 13)]
            + [{"ACTIVITY_DT": _date(2025, m, 1), "AMT": Decimal(1)} for m in range(1, 9)]
            + [{"ACTIVITY_DT": _date(2026, 1, d), "AMT": Decimal(1)} for d in range(1, 32)]
        )
        self.assertEqual(build_column_formats(rows), {})

    def test_a_descending_result_is_still_recognised(self):
        rows = [{"PERIOD": _date(2026, month, 1), "AMT": Decimal(1)}
                for month in range(6, 0, -1)]
        self.assertEqual(build_column_formats(rows), {"PERIOD": "date"})

    def test_quarter_buckets_are_recognised(self):
        rows = [{"PERIOD": _date(2026, month, 1), "AMT": Decimal(1)}
                for month in (1, 4, 7, 10)]
        self.assertEqual(build_column_formats(rows), {"PERIOD": "date"})

    def test_weekly_and_year_buckets_are_left_alone(self):
        """Weekly buckets are not month-firsts; year buckets are, but the
        shared date renderer would print 2026-01-01 as "2026-01", which is no
        better than the day stamp it replaced."""
        weekly = [{"WK_DT": _date(2026, 1, d), "AMT": Decimal(1)} for d in (5, 12, 19, 26)]
        yearly = [{"YR_DT": _date(y, 1, 1), "AMT": Decimal(1)} for y in (2024, 2025, 2026)]
        self.assertEqual(build_column_formats(weekly), {})
        self.assertEqual(build_column_formats(yearly), {})

    def test_a_null_in_any_rendered_row_stands_the_column_down(self):
        rows = [{"PERIOD": _date(2026, 1, 1), "AMT": Decimal(1)},
                {"PERIOD": None, "AMT": Decimal(2)},
                {"PERIOD": _date(2026, 2, 1), "AMT": Decimal(3)}]
        self.assertEqual(build_column_formats(rows), {})

    def test_the_text_channel_prints_the_same_month_the_browser_does(self):
        """One classification, two separate renderers.

        `_FORMAT_ALIASES` has no "date" key, so this channel silently dropped
        the hint and fell through to str(val): the portal showed "2026-01" and
        Teams / /api/ask showed "2026-01-01" for the same cell. That is commit
        b57e03b's split in mirror image, and only a test that runs BOTH sides
        catches it.
        """
        rows = self._monthly()
        table = _rows_to_table(rows, build_column_formats(rows))
        self.assertIn("2026-01", table)
        self.assertNotIn("2026-01-01", table)

    def test_an_undeclared_day_column_is_untouched_by_the_text_channel(self):
        rows = [{"DMS_DT": _date(2026, 1, 17), "AMT": Decimal(1)}]
        self.assertIn("2026-01-17", _rows_to_table(rows, build_column_formats(rows)))


class NarrationShowsTheSamePeriodAsTheTableTests(unittest.TestCase):
    """Periods reach the user through THREE paths, not two.

    Live on EMCO, 2026-09-02: one answer's KPI headline read "2026-06 closed at
    $7,439,558.42" while its Key insights, three lines below, read "trended flat
    0.7% from 2026-01-01 to 2026-06-01". The table and KPI go through the
    display formatter via column_formats; the sentences written ABOUT the
    series never did.

    These execute `narrative_period_labels`, the function both narrators now
    build their label list from.
    """

    def _labels(self, values):
        from core.response_builder import narrative_period_labels
        return narrative_period_labels(values)

    def test_month_buckets_are_narrated_as_months(self):
        self.assertEqual(
            self._labels([_date(2026, m, 1) for m in range(1, 7)]),
            ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"],
        )

    def test_the_iso_string_form_is_covered(self):
        """By narration time the value has usually been stringified."""
        self.assertEqual(
            self._labels([f"2026-{m:02d}-01" for m in range(1, 4)]),
            ["2026-01", "2026-02", "2026-03"],
        )

    def test_a_true_daily_series_is_left_alone(self):
        """Collapsing real days onto month labels would make consecutive rows
        read as duplicates of each other."""
        self.assertEqual(
            self._labels([_date(2026, 1, d) for d in range(1, 5)]),
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        )

    def test_dates_that_merely_step_monthly_are_left_alone(self):
        """Invoices due on the 15th, and month-END balance dates, both step
        ~30 days. Cadence cannot tell those from a bucket; day == 1 can."""
        for label, values in (
            ("due on the 15th", [_date(2026, m, 15) for m in range(1, 5)]),
            ("month end", [_date(2026, m, 1) - _timedelta(days=1) for m in range(2, 6)]),
        ):
            with self.subTest(shape=label):
                self.assertEqual(self._labels(values), [str(v) for v in values])

    def test_non_temporal_labels_pass_through(self):
        self.assertEqual(self._labels(["Halifax", "Toronto"]), ["Halifax", "Toronto"])

    def test_a_single_period_is_not_reformatted(self):
        """One label establishes no cadence, so there is nothing to infer."""
        self.assertEqual(self._labels([_date(2026, 1, 1)]), ["2026-01-01"])

    def test_the_trend_sentence_itself_says_the_month(self):
        """Executes the real narrator rather than scanning it for a call.

        The whole defect was a formatted headline sitting above an unformatted
        sentence, so the assertion has to be on the sentence a user reads.
        """
        from core.insight import compute_data_brief

        rows = [{"PERIOD": _date(2026, m, 1), "REVENUE": Decimal(1000 * m)}
                for m in range(1, 7)]
        brief = str(compute_data_brief(rows, "revenue by month") or "")
        self.assertIn("2026-01", brief)
        self.assertNotIn("2026-01-01", brief)

    def test_a_daily_brief_still_shows_the_day(self):
        from core.insight import compute_data_brief

        rows = [{"ORDER_DT": _date(2026, 1, d), "REVENUE": Decimal(10 * d)}
                for d in range(1, 8)]
        brief = str(compute_data_brief(rows, "revenue by day") or "")
        self.assertIn("2026-01-0", brief)
