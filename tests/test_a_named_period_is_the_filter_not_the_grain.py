"""
tests/test_a_named_period_is_the_filter_not_the_grain.py

"revenue in March 2025" was answered with $253,684.68 — the total for
31 March alone, headlined "2025-03-31 closed at $253,684.68". The WHERE clause
was right (`inv.CAL_YR = 2025 AND inv.MTH_NO = 3`, the whole month); the SQL
also carried `inv.DMS_DT AS InvoiceDate ... GROUP BY inv.DMS_DT`, so the month
came back as 31 daily rows and the narrative reported the last one.

In the same session, "revenue during 2024" generated

    SELECT SUM(SOP_CUS_IVC_LIN_AMT) FROM ...CUS_ORD_IVC_FCT AS cus
    JOIN ...DT_DMS AS inv ON cus.CUS_IVC_DT_DMS_KEY = inv.DT_DMS_KEY
    WHERE inv.CAL_YR = 2024

— correct, governed, and REFUSED as field_plan_mismatch.

Those are the two halves of one cause. The validator's acceptance set for a
planned date field is {the planned column} plus that date dimension's
*calendar attributes*, and `infer_calendar_attributes` matched only fully
spelled names — CALENDAR_YEAR, MONTH_NUMBER, QUARTER_NO. EMCO's calendar
dimension ships CAL_YR, MTH_NO, QTR_NO, WK_OF_YR, DAY_OF_MTH: five of seven
attributes unresolved. With CAL_YR unacceptable, filtering on it failed
validation, and the only way to pass was to project the raw date column —
which forces a GROUP BY on it and turns a period TOTAL into a daily series.

A named period is a filter. It becomes a grain only when the reader asks for
one ("by day", "daily", "trend").
"""

from __future__ import annotations

import unittest

from core.contextual_dates import infer_calendar_attributes
from core.validator import validate_sql_detailed

SCHEMA = "EMDW_DMART"
FACT = f"{SCHEMA}.CUS_ORD_IVC_FCT"
DIM = f"{SCHEMA}.DT_DMS"

# The live EMCO calendar dimension, column-for-column.
TABLE_COLUMNS = {
    FACT: {c: "" for c in [
        "CUS_IVC_DT_DMS_KEY", "CUS_DMS_KEY", "SOP_CUS_IVC_LIN_AMT",
    ]},
    DIM: {c: "" for c in [
        "DT_DMS_KEY", "DMS_DT", "CAL_YR", "MTH_NO", "MTH_NM", "MTH_START_DT",
        "QTR_NO", "QTR_NM", "WK_OF_YR", "DAY_OF_MTH", "DAY_NM", "DAY_OF_WK_NO",
    ]},
}
KNOWN_TABLES = set(TABLE_COLUMNS)


def _plan():
    """Built the way production builds it: the calendar attributes are
    INFERRED from the live schema, never handed in by the test. If the
    inference regresses, this plan loses them and the assertions below fail —
    which is the point."""
    attributes = infer_calendar_attributes(
        DIM, TABLE_COLUMNS, date_value_column="DMS_DT",
    )
    return {"semantic_plan": {
        "enabled": True,
        "fields": [
            {"term": "revenue", "table": FACT, "column": "SOP_CUS_IVC_LIN_AMT",
             "role": "measure", "enforcement": "required"},
            {"term": "invoice date", "table": DIM, "column": "DMS_DT",
             "role": "date", "enforcement": "required",
             "calendar_attributes": attributes},
        ],
    }}


def _validate(sql: str):
    return validate_sql_detailed(
        sql, KNOWN_TABLES, "mssql", None, TABLE_COLUMNS, _plan(),
    )


class TheCalendarAttributesOfARealDateDimension(unittest.TestCase):

    def test_every_attribute_of_the_live_dimension_resolves(self):
        attributes = infer_calendar_attributes(
            DIM, TABLE_COLUMNS, date_value_column="DMS_DT")
        self.assertEqual(attributes.get("year"), "CAL_YR")
        self.assertEqual(attributes.get("month_number"), "MTH_NO")
        self.assertEqual(attributes.get("quarter"), "QTR_NO")
        self.assertEqual(attributes.get("week"), "WK_OF_YR")
        self.assertEqual(attributes.get("day"), "DAY_OF_MTH")
        self.assertEqual(attributes.get("date"), "DMS_DT")

    def test_a_fully_spelled_dimension_still_resolves(self):
        """The abbreviations were added, not substituted."""
        spelled = {DIM: {c: "" for c in
                         ["DATE_SK", "FULL_DATE", "CALENDAR_YEAR",
                          "MONTH_NUMBER", "QUARTER_NO"]}}
        attributes = infer_calendar_attributes(
            DIM, spelled, date_value_column="FULL_DATE")
        self.assertEqual(attributes.get("year"), "CALENDAR_YEAR")
        self.assertEqual(attributes.get("month_number"), "MONTH_NUMBER")


class APeriodFilterOnACalendarAttributeIsAccepted(unittest.TestCase):

    # Production emits null-aware aggregates (a separate validator rule),
    # so the fixtures do too — otherwise the assertion fails for a reason
    # that has nothing to do with calendar attributes.
    _AGG = ("SELECT COUNT_BIG(*) AS MatchedRows, "
            "COUNT(cus.SOP_CUS_IVC_LIN_AMT) AS NonNullMetricRows, "
            "COALESCE(SUM(cus.SOP_CUS_IVC_LIN_AMT), 0) AS Revenue ")
    _JOIN = (f"FROM {FACT} AS cus "
             f"JOIN {DIM} AS inv ON cus.CUS_IVC_DT_DMS_KEY = inv.DT_DMS_KEY ")

    def test_a_whole_year_filtered_on_the_year_attribute(self):
        """The exact SQL the product generated for 'revenue during 2024' and
        then refused."""
        sql = self._AGG + self._JOIN + "WHERE inv.CAL_YR = 2024"
        result = _validate(sql)
        self.assertTrue(result.ok, f"{result.code}: {result.reason}")

    def test_a_whole_month_filtered_on_year_and_month(self):
        """'revenue in March 2025' — and note there is no GROUP BY here: the
        month is the filter, and the answer is one number."""
        sql = self._AGG + self._JOIN + "WHERE inv.CAL_YR = 2025 AND inv.MTH_NO = 3"
        result = _validate(sql)
        self.assertTrue(result.ok, f"{result.code}: {result.reason}")

    def test_the_raw_date_column_is_still_accepted(self):
        """A genuine daily question is unaffected."""
        sql = ("SELECT inv.DMS_DT, SUM(cus.SOP_CUS_IVC_LIN_AMT) AS Revenue "
               + self._JOIN + "WHERE inv.CAL_YR = 2025 AND inv.MTH_NO = 3 "
               "GROUP BY inv.DMS_DT")
        result = _validate(sql)
        self.assertTrue(result.ok, f"{result.code}: {result.reason}")

    def test_sql_that_touches_no_governed_date_is_still_refused(self):
        """The widened acceptance must not become no acceptance at all."""
        sql = f"SELECT SUM(cus.SOP_CUS_IVC_LIN_AMT) AS Revenue FROM {FACT} AS cus"
        result = _validate(sql)
        self.assertFalse(
            result.ok,
            "a query ignoring the governed date dimension must still fail")


if __name__ == "__main__":
    unittest.main()
