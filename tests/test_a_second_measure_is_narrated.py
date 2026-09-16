# -*- coding: utf-8 -*-
""""Net sales and returns by warehouse" described the sales and never said the word returns.

A ranking with two measures narrated the first: leader, share, count. The
second column -- the one that made the question a comparison -- was in the
table and nowhere in the prose. The insight sentence now carries one more
clause for it: which entry leads the second measure, at what value, and with
what share when the measure adds up.

Measured on the real composer before the fix, three warehouses with sales and
returns:

    Montreal leads at 412,000 (55.5% of total) across 3 warehouse names.

Every test executes build_assistant_response.
"""

from __future__ import annotations

import unittest

from core.i18n import activate_language, deactivate_language
from core.response_builder import build_assistant_response

ROWS = [
    {"WHS_NM": "Montreal", "NET_SLS_AMT": 412_000.0, "RTN_AMT": 12_000.0},
    {"WHS_NM": "Quebec", "NET_SLS_AMT": 233_000.0, "RTN_AMT": 19_500.0},
    {"WHS_NM": "Laval", "NET_SLS_AMT": 98_000.0, "RTN_AMT": 7_900.0},
]
SQL = (
    "WITH s AS (SELECT WHS_DMS_KEY, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM EMDW_DMART.CUS_ORD_IVC_FCT GROUP BY WHS_DMS_KEY), "
    "r AS (SELECT WHS_DMS_KEY, SUM(RTN_AMT) AS RTN_AMT FROM EMDW_DMART.CUS_RTN_FCT GROUP BY WHS_DMS_KEY) "
    "SELECT w.WHS_NM, s.NET_SLS_AMT, r.RTN_AMT FROM EMDW_DMART.WHS_DMS w "
    "LEFT JOIN s ON s.WHS_DMS_KEY = w.WHS_DMS_KEY LEFT JOIN r ON r.WHS_DMS_KEY = w.WHS_DMS_KEY"
)


def compose(rows, question, lang="en"):
    token = activate_language(lang)
    try:
        return build_assistant_response(question=question, rows=rows, sql=SQL,
                                        duration_ms=10, data_source="azure_sql")
    finally:
        deactivate_language(token)


class TestTheSecondMeasureIsNarrated(unittest.TestCase):

    def test_the_insight_names_the_leader_of_the_second_measure(self):
        insight = compose(ROWS, "net sales and returns by warehouse")["insight_summary"]
        self.assertIn("Montreal leads at 412,000", insight)
        self.assertIn("Quebec leads Rtn Amount at 19,500", insight)
        self.assertIn("49.5% of total", insight)

    def test_in_french(self):
        insight = compose(ROWS, "ventes nettes et retours par entrepôt", lang="fr")["insight_summary"]
        self.assertIn("Quebec arrive en tête pour Rtn Amount avec 19 500", insight)

    def test_a_single_measure_reads_exactly_as_before(self):
        rows = [{k: v for k, v in r.items() if k != "RTN_AMT"} for r in ROWS]
        insight = compose(rows, "net sales by warehouse")["insight_summary"]
        self.assertEqual(insight, "Montreal leads at 412,000 (55.5% of total) across 3 warehouse names.")

    def test_a_second_measure_that_does_not_add_up_gets_no_share(self):
        rows = [dict(r, GRS_MRG_PCT=p) for r, p in zip(ROWS, (31.2, 28.9, 35.1))]
        rows = [{k: v for k, v in r.items() if k != "RTN_AMT"} for r in rows]
        insight = compose(rows, "net sales and margin by warehouse")["insight_summary"]
        self.assertIn("Laval leads Gross Margin Percent at 35.10", insight)
        self.assertNotIn("of total", insight.split("Laval leads")[1])

    def test_a_period_column_is_not_a_second_measure(self):
        rows = [dict(r, IVC_YR=2025) for r in ROWS]
        rows = [{k: v for k, v in r.items() if k != "RTN_AMT"} for r in rows]
        insight = compose(rows, "net sales by warehouse")["insight_summary"]
        self.assertNotIn("Ivc Yr", insight)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
