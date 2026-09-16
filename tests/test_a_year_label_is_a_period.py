# -*- coding: utf-8 -*-
""""2,025 closed at 1,900,000."

A series over an integer year column narrated its last period through the
number formatter, so the year picked up a thousands separator: "2,025" in
English, "2 025" in French. The insight sentence two lines below said "from
2021 to 2025", because it reads the labels the context formatted for prose.
The headline now reads the same labels.
"""

from __future__ import annotations

import unittest

from core.i18n import activate_language, deactivate_language
from core.response_builder import build_assistant_response

YEARS = [{"IVC_YR": y, "NET_SLS_AMT": v}
         for y, v in [(2021, 1.1e6), (2022, 1.3e6), (2023, 1.25e6), (2024, 1.6e6), (2025, 1.9e6)]]
MONTHS = [{"IVC_MTH": f"2025-{m:02d}", "NET_SLS_AMT": 100_000.0 + m * 3_000} for m in range(1, 13)]
MONTH_DATES = [{"IVC_MTH": f"2025-{m:02d}-01", "NET_SLS_AMT": 100_000.0 + m * 3_000} for m in range(1, 13)]


def compose(rows, question, lang="en"):
    token = activate_language(lang)
    try:
        return build_assistant_response(question=question, rows=rows,
                                        sql="SELECT ... GROUP BY 1 ORDER BY 1",
                                        duration_ms=10, data_source="azure_sql")
    finally:
        deactivate_language(token)


class TestAYearIsAPeriod(unittest.TestCase):

    def test_the_english_headline_names_the_year(self):
        answer = compose(YEARS, "net sales by year")["answer"]
        self.assertEqual(answer["headline"], "2025 closed at 1,900,000.")

    def test_the_french_headline_names_the_year(self):
        answer = compose(YEARS, "ventes nettes par année", lang="fr")["answer"]
        self.assertEqual(answer["headline"], "2025 a terminé à 1 900 000.")

    def test_the_value_still_takes_the_number_format(self):
        answer = compose(YEARS, "net sales by year")["answer"]
        self.assertEqual(answer["short_value"], "1,900,000")

    def test_a_month_label_is_untouched(self):
        answer = compose(MONTHS, "net sales by month")["answer"]
        self.assertEqual(answer["headline"], "2025-12 closed at 136,000.")

    def test_a_month_bucket_stored_as_a_date_reads_as_the_month(self):
        """The same rule the table and the insight sentence already apply:
        a first-of-month date series is a month series."""
        answer = compose(MONTH_DATES, "net sales by month")["answer"]
        self.assertEqual(answer["headline"], "2025-12 closed at 136,000.")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
