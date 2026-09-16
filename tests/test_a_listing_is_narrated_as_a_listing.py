# -*- coding: utf-8 -*-
""""Show invoices for customer C001" was narrated as a race between invoices.

Twelve invoice lines -- number, date, customer, item, quantity, unit price,
amount, warehouse -- came back from a SELECT with no GROUP BY, and the card
read the first text column against the first numeric one and wrote a ranking:

    INV11 leads at 21 (11.3% of total) across 12 invoice nos.
    Volume is spread across the field — no single entry exceeds 12%; broadly diversified.

Nothing in that is false and none of it is what an analyst would say about
a listing, which is how many records there are and what they add up to.

A result is a listing when its SQL aggregated nothing, the question did not
ask for a top N, and the rows are not sorted by a measure. That is decided
where the narrative mode is decided, so the headline, the insight sentence
and the decision signal agree. Measured on the real composer.
"""

from __future__ import annotations

import unittest

from core.i18n import activate_language, deactivate_language
from core.response_builder import build_assistant_response, summarize_result_context

INVOICE_LINES = [
    {"IVC_NO": f"INV{i:02d}", "IVC_DT": f"2025-03-{i + 1:02d}", "CUS_NM": "Client 1",
     "ITM_NM": f"Item {i % 3}", "QTY": 10 + i, "UNIT_PRC": 12.5,
     "NET_SLS_AMT": (10 + i) * 12.5, "WHS_NM": "Montreal"}
    for i in range(12)
]
LISTING_SQL = "SELECT TOP 20 IVC_NO, IVC_DT, CUS_NM, ITM_NM, QTY, UNIT_PRC, NET_SLS_AMT, WHS_NM FROM EMDW_DMART.CUS_ORD_IVC_FCT f WHERE CUS_NO = 'C001' ORDER BY IVC_DT"
RANKING_SQL = "SELECT TOP 20 WHS_NM, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM f GROUP BY WHS_NM ORDER BY 2 DESC"
WAREHOUSES = [{"WHS_NM": w, "NET_SLS_AMT": v} for w, v in [("Montreal", 412_000.0), ("Quebec", 233_000.0), ("Laval", 98_000.0)]]


def compose(rows, question, sql, lang="en"):
    token = activate_language(lang)
    try:
        return build_assistant_response(question=question, rows=rows, sql=sql,
                                        duration_ms=10, data_source="azure_sql")
    finally:
        deactivate_language(token)


class TestAListingIsNotARace(unittest.TestCase):

    def test_the_mode_is_a_table_of_records(self):
        ctx = summarize_result_context(INVOICE_LINES, "show invoices for customer C001", LISTING_SQL)
        self.assertEqual(ctx["mode"], "text_table")
        self.assertTrue(ctx.get("listing"))

    def test_the_headline_counts_and_does_not_rank(self):
        payload = compose(INVOICE_LINES, "show invoices for customer C001", LISTING_SQL)
        self.assertEqual(payload["answer"]["headline"], "Found 12 results for: show invoices for customer C001")
        self.assertNotIn("leads", payload["answer"]["headline"])

    def test_the_insight_says_what_the_records_add_up_to(self):
        payload = compose(INVOICE_LINES, "show invoices for customer C001", LISTING_SQL)
        self.assertEqual(payload["insight_summary"],
                         "12 records — Net Sls Amount totals 2,325 (125 to 262.50 per record).")

    def test_no_diversification_verdict_over_a_customers_invoices(self):
        payload = compose(INVOICE_LINES, "show invoices for customer C001", LISTING_SQL)
        self.assertEqual(payload["decision_signal"], {})
        self.assertEqual(payload["anomaly_callouts"], [])

    def test_the_amount_is_narrated_ahead_of_the_quantity(self):
        """Invoice lines carry a quantity before the amount. The quantity adds
        up too, and it is not what the reader came for."""
        payload = compose(INVOICE_LINES, "show invoices for customer C001", LISTING_SQL)
        self.assertNotIn("Qty", payload["insight_summary"])

    def test_in_french(self):
        payload = compose(INVOICE_LINES, "afficher les factures du client C001", LISTING_SQL, lang="fr")
        self.assertEqual(payload["insight_summary"],
                         "12 enregistrements — Net Sls Amount totalise 2 325 (125 à 262,50 par enregistrement).")

    def test_a_listing_with_no_additive_measure_gives_the_range(self):
        rows = [{"CUS_NO": f"C{i:03d}", "CUS_NM": f"Client {i}", "CRD_LMT_PCT": 10.0 + i} for i in range(4)]
        payload = compose(rows, "list customers with their credit limit percent",
                          "SELECT CUS_NO, CUS_NM, CRD_LMT_PCT FROM c")
        self.assertEqual(payload["insight_summary"],
                         "4 records — Crd Lmt Percent ranges 10 to 13, avg 11.50.")

    def test_a_single_record_reads_in_the_singular(self):
        payload = compose(INVOICE_LINES[:1], "show invoice INV00", LISTING_SQL)
        self.assertEqual(payload["insight_summary"], "1 record — Net Sls Amount totals 125.")


class TestWhatIsStillARanking(unittest.TestCase):

    def test_an_aggregated_result_is_a_ranking(self):
        payload = compose(WAREHOUSES, "net sales by warehouse", RANKING_SQL)
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")
        self.assertIn("leads at 412,000", payload["answer"]["headline"])

    def test_a_top_n_over_records_is_a_ranking(self):
        """"Top 5 invoices by amount": no GROUP BY, but the reader asked for
        the biggest, and the biggest leads."""
        rows = sorted(INVOICE_LINES, key=lambda r: -r["NET_SLS_AMT"])[:5]
        sql = "SELECT TOP 5 IVC_NO, NET_SLS_AMT FROM f ORDER BY NET_SLS_AMT DESC"
        rows = [{"IVC_NO": r["IVC_NO"], "NET_SLS_AMT": r["NET_SLS_AMT"]} for r in rows]
        payload = compose(rows, "top 5 invoices by amount", sql)
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")

    def test_an_aggregate_grouped_by_an_identifier_is_a_ranking(self):
        """"Net sales by customer number": the key column is there, and so
        is the GROUP BY. One row per customer, and the biggest leads."""
        rows = [{"CUS_NO": f"C{i:03d}", "NET_SLS_AMT": 90_000.0 - i * 10_000} for i in range(5)]
        payload = compose(rows, "net sales by customer number",
                          "SELECT CUS_NO, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM f GROUP BY CUS_NO")
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")
        self.assertIn("C000 leads at 90,000", payload["answer"]["headline"])

    def test_a_top_n_ordered_by_ordinal_is_a_ranking(self):
        """The reader asked for the top five and the model ordered by
        position, not by name. The question decides."""
        rows = [{"IVC_NO": f"INV{i:02d}", "NET_SLS_AMT": 900.0 - i * 100} for i in range(5)]
        payload = compose(rows, "top 5 invoices by amount",
                          "SELECT TOP 5 IVC_NO, NET_SLS_AMT FROM f ORDER BY 2 DESC")
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")

    def test_records_sorted_by_a_measure_are_a_ranking(self):
        rows = [{"IVC_NO": r["IVC_NO"], "NET_SLS_AMT": r["NET_SLS_AMT"]}
                for r in sorted(INVOICE_LINES, key=lambda r: -r["NET_SLS_AMT"])]
        payload = compose(rows, "invoices by amount", "SELECT IVC_NO, NET_SLS_AMT FROM f ORDER BY NET_SLS_AMT DESC")
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")

    def test_without_the_sql_nothing_changes(self):
        """The fallback scopes that are built without SQL keep their old
        reading: the rule needs the SQL to know nothing was aggregated."""
        ctx = summarize_result_context(INVOICE_LINES, "show invoices for customer C001", "")
        self.assertEqual(ctx["mode"], "ranking")

    def test_a_label_and_value_read_off_a_view_is_a_ranking(self):
        """No GROUP BY because the view already aggregated. No identifier
        column either: one row per region with its revenue is the ranking it
        looks like, and the reader is owed the leader."""
        rows = [{"REGION": r, "REVENUE": v} for r, v in [("North", 900.0), ("South", 50.0), ("East", 20.0)]]
        payload = compose(rows, "revenue by region", "SELECT region, revenue FROM region_sales_vw")
        self.assertEqual(payload["analysis_contract"]["mode"], "ranking")
        self.assertIn("North leads at 900", payload["answer"]["headline"])

    def test_records_are_known_by_their_key(self):
        """The same eight columns without the invoice number: nothing says
        these are records, and the old reading stands."""
        rows = [{k: v for k, v in r.items() if k != "IVC_NO"} for r in INVOICE_LINES]
        ctx = summarize_result_context(rows, "show invoices for customer C001", LISTING_SQL)
        self.assertNotEqual(ctx["mode"], "text_table")

    def test_a_series_is_still_a_series(self):
        months = [{"IVC_MTH": f"2025-{m:02d}", "NET_SLS_AMT": 100_000.0 + m * 3_000} for m in range(1, 13)]
        payload = compose(months, "net sales by month", "SELECT IVC_MTH, SUM(x) AS NET_SLS_AMT FROM f GROUP BY IVC_MTH ORDER BY 1")
        self.assertEqual(payload["analysis_contract"]["mode"], "time_series")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
