# -*- coding: utf-8 -*-
"""A TOP 20 over three warehouses silenced the summary.

``infer_result_scope`` decides whether a result was truncated, and everything
that makes a claim about the WHOLE reads that decision: the trend sentence,
the decision signal, the ranking's runner-up clause, median and quartiles. The
rule that withholds them on a truncated result is right, and it landed in
03bb5b6 after a live TOP 5 called itself "broadly diversified".

The decision underneath it was wrong. It read the mere presence of a row
limit in the SQL as the fact of truncation:

    was_limited = explicit_limit is not None or preview_cap_hit

The Azure prompt tells the model "Default to SELECT TOP 20", so every
generated T-SQL query carries a limit whether or not the question asked for
one. A three-warehouse breakdown under that default TOP 20 was therefore
"limited", its distribution "incomplete", and -- because the same branch set
``is_top_n`` -- a top-twenty request the reader never made. Measured before
the fix, on the real function:

    TOP 20,  3 rows -> was_limited=True  is_top_n=True  n=20
    TOP 20,  8 rows -> was_limited=True  is_top_n=True  n=20
    TOP 20, 19 rows -> was_limited=True  is_top_n=True  n=20

So the day the truncation rule shipped, every complete Azure ranking lost its
decision signal and every complete Azure series lost its trend sentence. The
rule was correct; the flag it trusted was not.

Fewer rows back than the limit is proof the limit never bit. Exactly the limit
is treated as bound -- it may have been, and the cost of assuming so is one
withheld sentence rather than a claim about rows that were cut. And a top-N
framing now needs a limit that bound OR a question that asked for one; a
defensive TOP that did nothing, on a question that never asked, is neither.

The preview cap is decoupled at the same time: TOP 500 returning 200 rows was
cut by the cap, not by the TOP, and used to read as neither.

Every test below executes the real scope detector or the real narrator with
the SQL the pipeline passes. The two live cases the truncation rule was written
for -- five rows under TOP 5, two hundred days under TOP 200 -- are kept here
as the boundary the fix must not cross.
"""

from __future__ import annotations

import unittest

import pytest

from core.i18n import activate_language, deactivate_language
from core.insight import compute_data_brief
from core.response_builder import (
    _build_anomaly_callouts,
    _build_decision_signal,
    _build_insight_summary,
    build_answer,
    infer_result_scope,
    summarize_result_context,
)


def _ranking(n: int, col: str = "WHS_DSC") -> list[dict]:
    return [{col: f"Site {i:02d}", "NET_SLS_AMT": 100_000.0 - i * 1_000} for i in range(n)]


def _scope(rows, question, sql, mode="ranking"):
    return infer_result_scope(rows, question, sql=sql, mode=mode)


TOP20 = "SELECT TOP 20 WHS_DSC, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM f GROUP BY WHS_DSC ORDER BY 2 DESC"
TOP5 = "SELECT TOP 5 CUS_NM, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM f GROUP BY CUS_NM ORDER BY 2 DESC"
PLAIN = "SELECT WHS_DSC, SUM(NET_SLS_AMT) AS NET_SLS_AMT FROM f GROUP BY WHS_DSC ORDER BY 2 DESC"


class TestALimitCountsOnlyWhenTheResultReachedIt:

    @pytest.mark.parametrize("returned", [1, 3, 8, 19])
    def test_fewer_rows_than_the_top_is_a_complete_result(self, returned):
        scope = _scope(_ranking(returned), "net sales by warehouse", TOP20)
        assert scope["was_limited"] is False
        assert scope["is_complete_distribution"] is True

    @pytest.mark.parametrize("returned", [20, 21])
    def test_the_limit_or_more_is_treated_as_bound(self, returned):
        """Exactly n may or may not have been cut; the safe reading is that it
        was. Beyond n is impossible for a real TOP and reads the same way."""
        scope = _scope(_ranking(returned), "net sales by warehouse", TOP20)
        assert scope["was_limited"] is True
        assert scope["is_complete_distribution"] is False

    def test_the_live_top_five_is_still_a_truncation(self):
        """The case 03bb5b6 was written for. Five rows under TOP 5 bound."""
        scope = _scope(_ranking(5, "CUS_NM"), "top 5 customers by net sales", TOP5)
        assert scope["was_limited"] is True
        assert scope["is_top_n"] is True
        assert scope["n"] == 5

    def test_a_limit_over_an_empty_result_did_not_bind(self):
        scope = _scope([], "net sales by warehouse", TOP20)
        assert scope["was_limited"] is False

    @pytest.mark.parametrize("sql", [
        "SELECT TOP (20) a FROM f",
        "SELECT a FROM f ORDER BY a OFFSET 0 ROWS FETCH NEXT 20 ROWS ONLY",
        "SELECT a FROM f LIMIT 20",
    ])
    def test_every_limit_spelling_uses_the_same_rule(self, sql):
        assert _scope(_ranking(3), "net sales by warehouse", sql)["was_limited"] is False
        assert _scope(_ranking(20), "net sales by warehouse", sql)["was_limited"] is True


class TestATopNFramingNeedsAReasonToExist:

    def test_a_defensive_top_on_a_plain_question_is_not_a_top_n(self):
        """The reader asked "by warehouse". Nothing about that is a top twenty,
        and the ranking explanation must not drop its runner-up clause or the
        badge call it one."""
        scope = _scope(_ranking(3), "net sales by warehouse", TOP20)
        assert scope["is_top_n"] is False
        assert scope["n"] is None

    def test_a_question_that_asked_keeps_its_framing_even_when_complete(self):
        """"Top 5 customers" returning all three that exist: complete, and
        still the reader's top-5 framing."""
        scope = _scope(_ranking(3, "CUS_NM"), "top 5 customers by net sales", TOP5)
        assert scope["is_top_n"] is True
        assert scope["n"] == 5
        assert scope["was_limited"] is False

    def test_a_limit_that_bound_is_a_top_n_whatever_the_wording(self):
        """Five rows under TOP 5: the SQL decided the shape, so the framing
        holds even for wording the intent detector does not read."""
        scope = _scope(_ranking(5, "CUS_NM"), "les 5 meilleurs clients", TOP5)
        assert scope["is_top_n"] is True


class TestThePreviewCapIsItsOwnTruncation:

    def test_a_larger_top_does_not_hide_the_cap(self):
        """TOP 500 returning 200 rows was cut by the preview cap. Before, the
        presence of the TOP disabled the cap check and the limit "bound" by
        virtue of existing -- right answer, wrong reason, and the badge said
        top-500 instead of preview."""
        scope = _scope(_ranking(200), "net sales by customer", "SELECT TOP 500 a FROM f")
        assert scope["was_limited"] is True
        assert scope["is_preview"] is True
        assert scope["is_top_n"] is False

    def test_no_limit_at_the_cap_is_a_preview(self):
        scope = _scope(_ranking(200), "net sales by customer", PLAIN)
        assert scope["was_limited"] is True
        assert scope["is_preview"] is True

    def test_below_the_cap_with_no_limit_is_complete(self):
        scope = _scope(_ranking(50), "net sales by customer", PLAIN)
        assert scope["was_limited"] is False
        assert scope["is_preview"] is False


class TestWhatTheReaderGetsBack(unittest.TestCase):
    """The narrators, executed the way the card executes them."""

    WAREHOUSES = [
        {"WHS_DSC": "Montreal", "NET_SLS_AMT": 412_000.0},
        {"WHS_DSC": "Quebec", "NET_SLS_AMT": 233_000.0},
        {"WHS_DSC": "Laval", "NET_SLS_AMT": 98_000.0},
    ]
    MONTHS = [{"IVC_MTH": f"2025-{m:02d}", "REVENUE": 100.0 + m * 3} for m in range(1, 13)]
    DAYS = [
        {"IVC_DT": f"2025-{1 + d // 28:02d}-{d % 28 + 1:02d}", "REVENUE": 100.0 - d * 0.3}
        for d in range(200)
    ]

    def setUp(self):
        self._token = activate_language("en")

    def tearDown(self):
        deactivate_language(self._token)

    def _signal(self, rows, question, sql):
        ctx = summarize_result_context(rows, question, sql=sql)
        scope = _scope(rows, question, sql, mode=ctx.get("mode", "ranking"))
        brief = compute_data_brief(rows, question, result_scope=scope, context=ctx)
        return (_build_decision_signal(ctx, brief, _build_anomaly_callouts(brief)) or {}).get("line", "")

    def _trend(self, rows, question, sql):
        scope = _scope(rows, question, sql, mode="time_series")
        brief = compute_data_brief(rows, question, result_scope=scope)
        ctx = summarize_result_context(rows, question, sql)
        ctx["mode"] = "time_series"
        return _build_insight_summary(rows, ctx, brief).strip()

    def test_a_complete_ranking_under_a_default_top_keeps_its_decision_signal(self):
        """Three warehouses, the whole population, Montreal at 56%. That is a
        real single point of dependency and the reader is owed the sentence."""
        line = self._signal(self.WAREHOUSES, "net sales by warehouse", TOP20)
        self.assertTrue(line, "the decision signal was withheld on a complete result")
        self.assertIn("Montreal", line)

    def test_and_the_same_rows_under_a_binding_top_do_not(self):
        top3 = TOP20.replace("TOP 20", "TOP 3")
        self.assertEqual(self._signal(self.WAREHOUSES, "top 3 warehouses", top3), "")

    def test_a_complete_series_under_a_default_top_keeps_its_trend(self):
        self.assertTrue(self._trend(self.MONTHS, "revenue by month",
                                    "SELECT TOP 20 IVC_MTH, SUM(x) AS REVENUE FROM f GROUP BY IVC_MTH"))

    def test_the_live_two_hundred_day_cap_still_makes_no_trend_claim(self):
        self.assertEqual(self._trend(self.DAYS, "revenue by day",
                                     "SELECT TOP 200 IVC_DT, SUM(x) AS REVENUE FROM f GROUP BY IVC_DT"), "")

    def test_the_runner_up_clause_returns_on_a_plain_question(self):
        """build_answer drops "N above the next result" on a top-N, because on
        a slice the runner-up is an artefact of the cut. On a complete
        breakdown it is a fact about the data."""
        scope = _scope(self.WAREHOUSES, "net sales by warehouse", TOP20)
        answer = build_answer(self.WAREHOUSES, "net sales by warehouse", result_scope=scope)
        self.assertIn("above the next result", answer.get("comparison", ""))

    def test_and_stays_dropped_on_a_top_n_that_bound(self):
        rows = _ranking(5, "CUS_NM")
        scope = _scope(rows, "top 5 customers by net sales", TOP5)
        answer = build_answer(rows, "top 5 customers by net sales", result_scope=scope)
        self.assertNotIn("above the next result", answer.get("comparison", ""))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
