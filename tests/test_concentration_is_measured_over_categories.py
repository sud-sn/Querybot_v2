"""
tests/test_concentration_is_measured_over_categories.py

The answer card called a 91.9% concentration "broadly diversified", and called
three categories a 100% concentration risk.

core/insight.py computes the same statistic twice. ``category_breakdown
["top_3_share_pct"]`` is the share held by the three leading CATEGORIES, over the
collapsed rows; ``numeric_summaries[col]["top_3_concentration_pct"]`` was the
three leading ROWS of the raw result. On anything grouped by two things those are
three periods of one category, and insight.py's own comment says so -- it records
five warehouses over three months reporting a leader share of 25.0 and a top-3
share of 25.0, "the top three accounting for exactly as much as the leader alone,
which cannot happen".

That was fixed for the breakdown. The three consumers that make the CLAIM were
not. Measured, five warehouses over three months where Montreal, Toronto and
Calgary hold 91.9% of the total:

    collapsed top-3 share      91.9    <- the fixed field, unread
    raw-row top_3_concentration 45.0   <- what the callout read
    callouts                   []
    decision signal            "Volume is spread across the field — no single
                                entry exceeds 45%; broadly diversified."

The advisory line told EMCO their business was diversified while three
warehouses carried 92% of it. And on three categories, where the raw and the
collapsed numbers happen to agree, the claim itself is vacuous: the top three of
three is 100% of the total by construction, and the card announced "100.0% of
total — highly concentrated" as a finding.

So two things are fixed together, and both are needed: the claim reads the
COLLAPSED share, and it is withheld below the category floor the evidence engine
already uses for the same judgement. The raw field is REMOVED rather than left
unread -- it outlived the fix that replaced it once already, and a field nobody
should read is one a future reader picks up anyway.
"""

from __future__ import annotations

import pytest

from core.analysis_evidence import MIN_CATEGORIES_FOR_CONCENTRATION
from core.insight import _format_brief_for_prompt, compute_data_brief
from core.response_builder import (
    _build_anomaly_callouts,
    _build_decision_signal,
    summarize_result_context,
)

# Montreal + Toronto + Calgary = 91.9 of 100.
SHARES = {"Montreal": 45.0, "Toronto": 28.0, "Calgary": 18.9,
          "Halifax": 5.0, "Regina": 3.1}
MONTHS = ("2025-04", "2025-05", "2025-06")


def two_dimensional() -> list[dict]:
    """Five warehouses over three months: fifteen rows, five categories."""
    return [
        {"WHS_DSC": warehouse, "IVC_MTH": month,
         "NET_SLS_AMT": share * 1_000_000 / len(MONTHS)}
        for warehouse, share in SHARES.items()
        for month in MONTHS
    ]


def one_dimensional(count: int) -> list[dict]:
    """`count` categories, the leader holding 45% and the rest even."""
    rest = (100.0 - 45.0) / max(1, count - 1)
    values = [45.0] + [rest] * (count - 1)
    return [{"WHS_DSC": f"W{index}", "NET_SLS_AMT": value * 1_000_000}
            for index, value in enumerate(values)]


def card(rows, question="net sales by warehouse"):
    brief = compute_data_brief(rows, question)
    ctx = summarize_result_context(rows, question)
    callouts = _build_anomaly_callouts(brief)
    return {
        "brief": brief,
        "categories": (brief.get("category_breakdown") or {}).get("category_count"),
        "share": (brief.get("category_breakdown") or {}).get("top_3_share_pct"),
        "callouts": [c["type"] for c in callouts],
        "messages": [c["message"] for c in callouts],
        "signal": _build_decision_signal(ctx, brief, callouts),
        "prompt": _format_brief_for_prompt(brief),
    }


class TestTheConcentrationIsTheCategoriesConcentration:

    def test_the_collapsed_share_is_the_true_one(self):
        """Ground truth, computed from the fixture rather than read from the
        code: three of five warehouses hold 91.9%."""
        assert round(sum(sorted(SHARES.values(), reverse=True)[:3]), 1) == 91.9
        assert card(two_dimensional())["share"] == 91.9

    def test_the_callout_states_it(self):
        result = card(two_dimensional())
        assert "concentration" in result["callouts"], result["messages"]
        assert "91.9" in " ".join(result["messages"])

    def test_and_never_states_the_raw_row_share(self):
        """45.0 is the leading warehouse's own three months. Seeing it in a
        concentration claim is the defect."""
        result = card(two_dimensional())
        assert "45.0% of total" not in " ".join(result["messages"])

    def test_the_advisory_line_says_concentrated_not_diversified(self):
        signal = card(two_dimensional())["signal"]
        assert signal.get("basis") == "concentration", signal
        assert "diversified" not in (signal.get("line") or "")

    def test_the_narration_prompt_is_shown_the_same_number(self):
        """The model wrote prose beside a card that disagreed with it."""
        prompt = card(two_dimensional())["prompt"]
        assert "Top 3 categories hold 91.9% of total" in prompt
        assert "top 3 items account for 45.0%" not in prompt


class TestTheClaimNeedsEnoughCategoriesToMeanAnything:

    def test_the_floor_is_the_evidence_engines_own(self):
        assert MIN_CATEGORIES_FOR_CONCENTRATION == 5

    @pytest.mark.parametrize("count", [2, 3, 4])
    def test_below_the_floor_no_concentration_is_claimed(self, count):
        result = card(one_dimensional(count))
        assert "concentration" not in result["callouts"], result["messages"]
        assert result["signal"].get("basis") != "concentration"

    def test_the_top_three_of_three_is_arithmetic_not_a_finding(self):
        result = card(one_dimensional(3))
        assert result["share"] == 100.0, "the number itself is still computed"
        assert "100.0" not in " ".join(result["messages"])
        assert "100%" not in (result["signal"].get("line") or "")

    def test_and_the_model_is_not_shown_it_either(self):
        """A model shown "top 3 = 100%" writes a concentration warning about
        it, and the warning reads as a finding."""
        assert "Top 3 categories hold 100.0%" not in card(one_dimensional(3))["prompt"]

    def test_at_the_floor_a_real_concentration_is_claimed(self):
        result = card(two_dimensional())
        assert result["categories"] == MIN_CATEGORIES_FOR_CONCENTRATION
        assert "concentration" in result["callouts"]

    def test_a_genuinely_spread_result_claims_nothing(self):
        result = card(one_dimensional(8))
        assert result["share"] is not None and result["share"] < 80
        assert "concentration" not in result["callouts"]
        assert result["signal"].get("basis") != "concentration"


class TestTheMisreadFieldIsGone:
    """Removed, not merely unread. It was replaced once for the breakdown and
    survived in numeric_summaries, where three more consumers found it."""

    @pytest.mark.parametrize("rows", [
        two_dimensional(), one_dimensional(3), one_dimensional(8),
    ])
    def test_no_numeric_summary_carries_it(self, rows):
        summaries = compute_data_brief(rows, "net sales by warehouse").get(
            "numeric_summaries") or {}
        assert summaries, "ground truth: summaries are still computed"
        for column, stats in summaries.items():
            assert "top_3_concentration_pct" not in stats, column

    def test_the_summaries_still_carry_what_they_are_for(self):
        summaries = compute_data_brief(
            two_dimensional(), "net sales by warehouse")["numeric_summaries"]
        stats = summaries["NET_SLS_AMT"]
        for key in ("total", "min", "max", "mean", "median", "std_dev"):
            assert key in stats, key


class TestTheCardAndTheEngineAgree:
    """Three classifiers make a concentration judgement about one result: this
    card, the advisory line, and core/analysis_evidence.py. They must not
    disagree in front of the reader."""

    def test_the_engine_also_withholds_pareto_below_the_floor(self):
        from core.analysis_evidence import concentration_findings

        rows = one_dimensional(4)
        kinds = {finding.kind for finding in
                 concentration_findings(rows, "NET_SLS_AMT", "WHS_DSC")}
        assert "concentration_pareto" not in kinds

    def test_and_reports_it_at_the_floor(self):
        from core.analysis_evidence import concentration_findings

        rows = [{"WHS_DSC": f"W{i}", "NET_SLS_AMT": v * 1_000_000}
                for i, v in enumerate([45.0, 28.0, 18.9, 5.0, 3.1])]
        kinds = {finding.kind for finding in
                 concentration_findings(rows, "NET_SLS_AMT", "WHS_DSC")}
        assert kinds, "the engine should report something about this shape"
