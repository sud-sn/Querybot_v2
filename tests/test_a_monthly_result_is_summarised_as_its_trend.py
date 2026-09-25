# -*- coding: utf-8 -*-
"""A period written as a number is the time axis of a result, not a measure.

A mart keyed by period stores the month as an integer -- 202401 -- and a year
as 2025. Both parse as numbers, so the two layers that explain a result
counted them as measures, while the answer card above them already knew
better. On "stock on hand by month" the summary under the answer read

    Prd Key and On Hnd Qty rise and fall together (r = 0.95 over 6 rows).
    Prd Key is unusually uniform across rows, close to 202,403.50 throughout.

and the follow-up questions beside it offered "Why are Period Key values so
uniform across all rows?" -- while the one thing the result says, that stock
rose 43% over the six months, was never computed, because the axis it needs
was not a label.

The summary and the questions now take the rule the answer card uses, from
the same place (core.response_builder._measure_and_label_cols), and name the
measure the way the rest of the product does. Synthetic rows; no customer data.
"""

from __future__ import annotations

from core.analysis_evidence import build_evidence
from core.analysis_narrative import build_narrative, humanise_column
from core.schema_enrichment import display_label
from core.stat_signals import _suggestion_pairs, compute_signals

MONTHLY = [
    {"PRD_KEY": 202401 + i, "ON_HND_QTY": value}
    for i, value in enumerate([1200.0, 1150.0, 1400.0, 1380.0, 1600.0, 1720.0])
]
YEARLY = [
    {"SLS_YR": year, "NET_AMT": value}
    for year, value in ((2021, 100.0), (2022, 140.0), (2023, 150.0), (2024, 210.0))
]
# Two sites a month: grouped by two things, so not a series.
BY_SITE_AND_MONTH = [
    {"PRD_KEY": month, "SITE": site, "QTY": qty}
    for month, site, qty in ((202401, "A", 10.0), (202401, "B", 30.0),
                             (202402, "A", 12.0), (202402, "B", 33.0))
]


def _kinds_on(evidence, column):
    return {finding.kind for finding in evidence.findings if finding.columns[:1] == (column,)}


class TestTheSummary:

    def test_a_monthly_result_is_summarised_as_its_trend(self):
        evidence = build_evidence(MONTHLY)
        assert [(f.kind, f.columns) for f in evidence.findings] == [
            ("trend_up", ("ON_HND_QTY", "PRD_KEY"))]
        text = " ".join(build_narrative(evidence).sentences)
        assert "43.3%" in text and "1,200" in text and "1,720" in text

    def test_the_period_is_never_described_as_a_quantity(self):
        for rows, period in ((MONTHLY, "PRD_KEY"), (YEARLY, "SLS_YR"), (BY_SITE_AND_MONTH, "PRD_KEY")):
            evidence = build_evidence(rows)
            assert _kinds_on(evidence, period) == set(), (period, evidence.findings)
            text = " ".join(build_narrative(evidence).sentences)
            assert "202,4" not in text and "2,02" not in text, text

    def test_a_yearly_result_is_summarised_as_its_trend(self):
        evidence = build_evidence(YEARLY)
        assert ("trend_up", ("NET_AMT", "SLS_YR")) in [(f.kind, f.columns) for f in evidence.findings]

    def test_a_result_by_site_and_month_is_about_the_sites(self):
        evidence = build_evidence(BY_SITE_AND_MONTH)
        kinds = {finding.kind for finding in evidence.findings}
        assert "concentration_leader" in kinds
        assert not {kind for kind in kinds if kind.startswith("trend")}

    def test_the_summary_a_workspace_gets_without_the_model_reads_the_same(self):
        from core.response_builder import _regulated_analysis_fallback

        summary = _regulated_analysis_fallback("analyze", MONTHLY)
        assert summary["finding_kinds"] == ["trend_up"]
        assert "43.3%" in summary["body"] and "202,4" not in summary["body"]

    def test_the_measure_is_named_as_everywhere_else(self):
        """The summary said "On Hnd Qty rose" beside follow-up questions that
        said "On Hnd Quantity"."""
        assert humanise_column("ON_HND_QTY") == display_label("ON_HND_QTY") != "On Hnd Qty"
        text = " ".join(build_narrative(build_evidence(MONTHLY)).sentences)
        assert display_label("ON_HND_QTY") in text


class TestTheFollowUpQuestions:

    # What compute_data_brief says about these columns: numeric, by value.
    TYPES = {"PRD_KEY": "numeric", "ON_HND_QTY": "numeric"}

    def test_a_monthly_result_offers_its_trend(self):
        signals = compute_signals(MONTHLY)
        assert [(s["type"], s["col"]) for s in signals] == [("temporal", "PRD_KEY")]
        questions = [pair["question"] for pair in _suggestion_pairs(signals, list(self.TYPES), self.TYPES)]
        assert questions and all("upward" in question for question in questions), questions

    def test_no_question_is_about_the_period_as_a_number(self):
        for rows in (MONTHLY, YEARLY):
            signals = compute_signals(rows)
            period = next(iter(rows[0]))
            assert all(s["type"] == "temporal" for s in signals if s.get("col") and period in s["col"]), signals
            columns = list(rows[0])
            questions = " ".join(p["question"] for p in _suggestion_pairs(signals, columns, None))
            assert "uniform" not in questions and "move together" not in questions, questions
