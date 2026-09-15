"""
tests/test_claims_respect_a_truncated_result.py

Every share the answer card quotes is computed by dividing by the sum of the
rows that CAME BACK — core/insight.py builds ``category_breakdown["total"]``
from the collapsed result, not from the population. On a ``TOP 5`` that
denominator is the top-5 subtotal, so the card said:

    "redacted segment leads at $3,959,025.04 (20.2% of total) across 5
     customer names."
    "Volume is spread across the field — no single entry exceeds 20%;
     broadly diversified."

Both are about the user's own filter. Near-equal shares among the five largest
customers is what a top-5 always looks like, so "broadly diversified" is a
tautology; and the dominance branch one line above can warn about a single
point of dependence over a population it never saw.

The flag needed to suppress all three was already there. ``infer_result_scope``
computes ``was_limited`` from an explicit LIMIT/TOP or a preview cap, stores it
in the brief, and the ranking EXPLANATION at response_builder.py:2349 already
reads ``is_top_n`` to drop its runner-up clause for exactly this reason. The
decision-signal chain never asked, and neither did the trend sentence — which
on a 200-row preview read "trended down 36.6% from 2025-01-02 to 2025-07-20",
a window ending where the row cap fell rather than where the data does, one
line under a banner withholding median and quartiles because "computing them
over a partial result would give a misleading answer".

Two rules, one flag:
  * a claim about the WHOLE (share, concentration, dominance, trend, peak) is
    withheld when the result was truncated;
  * a claim about the ROWS RETURNED (one observation, two endpoints, the gap
    between two of the result's own values) is not — it was always honest.
"""

from __future__ import annotations

from core.insight import compute_data_brief
from core.response_builder import (
    _build_anomaly_callouts,
    _build_decision_signal,
    _build_insight_summary,
    infer_result_scope,
    summarize_result_context,
)

FULL_SQL = "SELECT CUS_NM, SUM(NET_SLS_AMT) AS REVENUE FROM F GROUP BY CUS_NM"
TOP5_SQL = "SELECT TOP 5 CUS_NM, SUM(NET_SLS_AMT) AS REVENUE FROM F GROUP BY CUS_NM"

# Five near-equal customers: the leader holds 20.2%, which is what a top-5
# slice of a long tail looks like from the inside.
RANKING_ROWS = [
    {"CUS_NM": "Rideau Valley Trading", "REVENUE": 3_959_025.04},
    {"CUS_NM": "Nova Scotia Service", "REVENUE": 3_930_000.00},
    {"CUS_NM": "Prairie Equipment", "REVENUE": 3_900_000.00},
    {"CUS_NM": "Coastal Supply", "REVENUE": 3_870_000.00},
    {"CUS_NM": "Northern Yard", "REVENUE": 3_940_000.00},
]

QUESTION = "top 5 customers by revenue"


def signal_for(rows, sql, question=QUESTION):
    brief = compute_data_brief(
        rows, question,
        result_scope=infer_result_scope(rows, question, sql, mode="ranking"),
    )
    ctx = summarize_result_context(rows, question, sql)
    return _build_decision_signal(ctx, brief, _build_anomaly_callouts(brief))


class TestAShareClaimNeedsTheWholePopulation:

    def test_the_scope_flag_is_actually_derived_from_the_sql(self):
        """Ground truth first: if this is False the rest of the file proves
        nothing, because the guard would never engage."""
        assert infer_result_scope(RANKING_ROWS, QUESTION, TOP5_SQL,
                                  mode="ranking")["was_limited"] is True
        assert infer_result_scope(RANKING_ROWS, QUESTION, FULL_SQL,
                                  mode="ranking")["was_limited"] is False

    def test_no_diversification_verdict_on_a_top_n(self):
        assert signal_for(RANKING_ROWS, TOP5_SQL) == {}

    def test_the_same_rows_still_get_a_verdict_when_complete(self):
        """The guard must key on truncation, not on the shape of the data —
        otherwise it silently removes the feature for everyone."""
        signal = signal_for(RANKING_ROWS, FULL_SQL)
        assert signal, "a complete distribution must still get its signal"
        assert signal.get("basis") in {"spread", "dominance", "concentration"}

    def test_the_spread_sentence_states_a_bound_it_can_support(self):
        """'no single entry exceeds N%' is an upper bound, so it rounds UP.
        With :.0f a 20.2% leader printed 'exceeds 20%' directly above its own
        '20.2% of total' bullet — the card contradicting itself by a rounding
        mode."""
        signal = signal_for(RANKING_ROWS, FULL_SQL)
        if signal.get("basis") != "spread":
            return  # fixture drifted into another branch; nothing to assert
        breakdown = compute_data_brief(RANKING_ROWS, QUESTION)["category_breakdown"]
        leader_share = breakdown["leader_share_pct"]
        quoted = int("".join(c for c in signal["line"] if c.isdigit()) or 0)
        assert quoted >= leader_share, (
            f"claimed no entry exceeds {quoted}% while the leader holds "
            f"{leader_share}%")


class TestATrendClaimNeedsTheWholeSeries:

    SERIES = [{"IVC_MTH": f"2025-{m:02d}", "REVENUE": 100.0 - m} for m in range(1, 13)]
    SERIES_SQL = "SELECT IVC_MTH, SUM(NET_SLS_AMT) AS REVENUE FROM F GROUP BY IVC_MTH"
    CAPPED_SQL = "SELECT TOP 200 IVC_MTH, SUM(NET_SLS_AMT) AS REVENUE FROM F GROUP BY IVC_MTH"

    def summary_for(self, sql):
        question = "revenue by month"
        scope = infer_result_scope(self.SERIES, question, sql, mode="time_series")
        brief = compute_data_brief(self.SERIES, question, result_scope=scope)
        ctx = summarize_result_context(self.SERIES, question, sql)
        ctx["mode"] = "time_series"
        return _build_insight_summary(self.SERIES, ctx, brief)

    def test_a_complete_series_still_gets_its_trend_sentence(self):
        assert self.summary_for(self.SERIES_SQL).strip() != ""

    def test_a_truncated_series_makes_no_trend_claim(self):
        assert self.summary_for(self.CAPPED_SQL) == ""
