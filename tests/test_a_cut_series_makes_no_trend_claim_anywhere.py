# -*- coding: utf-8 -*-
"""The trend sentence was withheld on a cut series. The decision signal was not.

03bb5b6 taught the insight sentence that a claim about the WHOLE series needs
the whole series: on a daily result cut at the row cap it says nothing about
direction, because the window ends where the cap fell and not where the data
does. The decision signal below it -- built from the same brief, two functions
later -- still read

    Sustained downward trend (-12% overall) — worth investigating before it compounds.

on exactly that result. The ranking branch of the same function already
checks `was_limited`; the time-series branch never did. And the badge on a
series cut by a binding TOP said "Returned result", which names nothing: a
ranking cut the same way says "Top 20 only".

Measured on the real composer before the fix, 200 daily rows under TOP 200:

    insight_summary = ""                                   (correct)
    decision_signal = "Sustained downward trend (-12% ...)" (a claim about rows it never saw)
    badge           = "Returned result"

Every test executes build_assistant_response, the composer the gateway calls.
"""

from __future__ import annotations

import unittest

from core.i18n import activate_language, deactivate_language
from core.response_builder import build_assistant_response

DAYS = [
    {"IVC_DT": f"2025-{1 + d // 28:02d}-{d % 28 + 1:02d}", "NET_SLS_AMT": 5_000.0 - d * 3}
    for d in range(200)
]
MONTHS = [{"IVC_MTH": f"2025-{m:02d}", "NET_SLS_AMT": 100_000.0 - m * 3_000} for m in range(1, 13)]
TOP200 = "SELECT TOP 200 IVC_DT, SUM(x) AS NET_SLS_AMT FROM f GROUP BY IVC_DT ORDER BY 1"
PLAIN = "SELECT IVC_DT, SUM(x) AS NET_SLS_AMT FROM f GROUP BY IVC_DT ORDER BY 1"


def compose(rows, question, sql, lang="en"):
    token = activate_language(lang)
    try:
        return build_assistant_response(question=question, rows=rows, sql=sql,
                                        duration_ms=10, data_source="azure_sql")
    finally:
        deactivate_language(token)


class TestACutSeriesMakesNoTrendClaimAnywhere(unittest.TestCase):

    def test_the_decision_signal_is_withheld_under_a_binding_top(self):
        payload = compose(DAYS, "net sales by day", TOP200)
        self.assertEqual(payload["insight_summary"], "")
        self.assertEqual(payload["decision_signal"], {})

    def test_and_under_the_preview_cap_with_no_top_at_all(self):
        payload = compose(DAYS, "net sales by day", PLAIN)
        self.assertEqual(payload["decision_signal"], {})

    def test_in_french_too(self):
        payload = compose(DAYS, "ventes nettes par jour", TOP200, lang="fr")
        self.assertEqual(payload["decision_signal"], {})

    def test_a_complete_series_keeps_its_signal(self):
        """Twelve months under a TOP 20 that never bit: the whole series,
        and a real sustained decline the reader is owed."""
        payload = compose(MONTHS, "net sales by month",
                          "SELECT TOP 20 IVC_MTH, SUM(x) AS NET_SLS_AMT FROM f GROUP BY IVC_MTH ORDER BY 1")
        self.assertTrue(payload["decision_signal"].get("line"))
        self.assertEqual(payload["decision_signal"]["basis"], "decline")


class TestTheBadgeNamesTheCut(unittest.TestCase):

    def test_a_series_cut_by_a_binding_top_says_so(self):
        scope = compose(DAYS, "net sales by day", TOP200)["result_scope"]
        self.assertEqual(scope["badge_key"], "partial_series")
        self.assertEqual(scope["badge"], "First 200 periods only")
        self.assertIn("200", scope["note"])

    def test_the_same_badge_in_french(self):
        scope = compose(DAYS, "ventes nettes par jour", TOP200, lang="fr")["result_scope"]
        self.assertEqual(scope["badge"], "200 premières périodes uniquement")

    def test_a_series_cut_by_the_cap_alone_is_still_a_preview(self):
        """No TOP in the SQL; the display cap did the cutting. That is the
        preview badge's own case and it keeps it."""
        scope = compose(DAYS, "net sales by day", PLAIN)["result_scope"]
        self.assertEqual(scope["badge_key"], "preview")

    def test_a_complete_series_is_still_a_full_series(self):
        scope = compose(MONTHS, "net sales by month",
                        "SELECT TOP 20 IVC_MTH, SUM(x) AS NET_SLS_AMT FROM f GROUP BY IVC_MTH ORDER BY 1")["result_scope"]
        self.assertEqual(scope["badge_key"], "full_series")

    def test_the_inline_form_reads_as_a_noun_phrase(self):
        scope = compose(DAYS, "net sales by day", TOP200)["result_scope"]
        self.assertEqual(scope["inline"], "the first 200 periods only")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
