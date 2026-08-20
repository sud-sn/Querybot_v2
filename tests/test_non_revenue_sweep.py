"""
tests/test_non_revenue_sweep.py

Found by asking the live warehouse about things other than revenue — inventory,
suppliers, profit centres, customer types — and about the product itself.

  B9   A supplier or profit-centre breakdown read as a time series, because
       "Nova" contains "nov" and "Maritime" contains "mar".
  B10  "which items have the highest on hand quantity" was answered
       conversationally, about a table sitting in the workspace.
  B12  "what tables and data can you answer questions about" was answered with
       a governed SQL refusal about a missing semantic mapping.

B10 and B12 are the same defect from opposite ends: a hand-written list of
business nouns decides whether a turn is a data request, and it neither knows
what this workspace holds nor which words belong to questions about the product.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import core.dispatcher as dispatcher
from core.stat_signals import _is_temporal_col, compute_signals


# ── B9: a business name is not a month ───────────────────────────────────────


class TestOnlyRealPeriodsCountAsTemporal:
    @pytest.mark.parametrize("label,values", [
        ("profit centres", ["Nova Scotia Service", "Maritime Equipment",
                            "Mayfield Depot", "Septic Systems Div",
                            "Ontario West", "Calgary Yard"]),
        ("suppliers", ["Pacific Import Partners 8", "Marine Supply Co",
                       "Novatech Holdings", "Decatur Industrial"]),
        ("warehouses", ["Toronto Distribution Centre", "Winnipeg Branch Store",
                        "Calgary Distribution Centre"]),
    ])
    def test_business_names_are_not_periods(self, label, values):
        """Month abbreviations used to match as bare substrings anywhere in the
        text, so a result grouped by profit centre was read as a time series and
        the answer asked "what period drove the biggest change?" about a result
        with no period in it."""
        rows = [{"LABEL": v} for v in values]
        assert _is_temporal_col("LABEL", rows) is False

    @pytest.mark.parametrize("col,values", [
        ("MONTH", ["2026-01", "2026-02", "2026-03"]),
        ("MONTH_NAME", ["January", "February", "March"]),
        ("PERIOD", ["Q1 2026", "Q2 2026"]),
        ("YR", ["2024", "2025", "2026"]),
    ])
    def test_real_periods_still_count(self, col, values):
        assert _is_temporal_col(col, [{col: v} for v in values]) is True

    def test_a_period_named_column_counts_whatever_its_values(self):
        """The column NAME is the strong signal and is checked first."""
        assert _is_temporal_col("INVOICE_DATE", [{"INVOICE_DATE": "x"}]) is True

    def test_one_stray_month_word_cannot_carry_a_column(self):
        """A majority is required, so a single "May Brothers Ltd" among five
        customer names does not turn the result into a trend."""
        rows = [{"CUS": n} for n in [
            "May Brothers Ltd", "Northern Supply", "Rideau Valley Trading",
            "Pacific Import", "Atlantic Hardware",
        ]]
        assert _is_temporal_col("CUS", rows) is False

    def test_a_categorical_breakdown_gets_no_trend_signal(self):
        rows = [
            {"PROFIT_CENTRE": "Nova Scotia Service", "AMOUNT": 57526105.35},
            {"PROFIT_CENTRE": "Maritime Equipment", "AMOUNT": 55931664.40},
            {"PROFIT_CENTRE": "Ontario West", "AMOUNT": 51002311.10},
            {"PROFIT_CENTRE": "Calgary Yard", "AMOUNT": 48771002.00},
        ]
        kinds = {s["type"] for s in compute_signals(rows)}
        assert "temporal" not in kinds
        assert "flat_trend" not in kinds


# ── B10 / B12: what counts as a data request ─────────────────────────────────


WORKSPACE = {
    "terms": [{"term": "on hand quantity", "aliases": "stock on hand, inventory"},
              {"term": "supplier", "aliases": "vendor"}],
    "metrics": [{"name": "Revenue", "synonyms": "sales"}],
    "graph": {"entities": [{"entity_name": "Item"}, {"entity_name": "Warehouse"},
                           {"entity_name": "Profit Centre"}]},
}


def _with_workspace():
    return (
        patch("store.list_terms", return_value=WORKSPACE["terms"]),
        patch("store.list_metrics", return_value=WORKSPACE["metrics"]),
        patch("store.get_full_graph", return_value=WORKSPACE["graph"]),
    )


class TestTheWorkspaceDecidesWhatItHolds:
    """The shape half of the test is a hand-written noun list that grew by
    incident — customers, orders, invoices, claims, prescriptions — with no
    entry for items, stock, suppliers or warehouses. Growing it per client is
    the wrong shape of fix; the workspace already states what it holds."""

    @pytest.mark.parametrize("question", [
        "which items have the highest on hand quantity",
        "show me stock by warehouse",
        "who are my top suppliers",
        "which profit centre spent the most",
    ])
    def test_a_question_naming_something_the_workspace_defines_is_a_data_request(
        self, question,
    ):
        terms, metrics, graph = _with_workspace()
        with terms, metrics, graph:
            assert dispatcher._looks_like_data_request(question, "acct") is True

    def test_the_hardcoded_nouns_still_work_without_an_account(self):
        assert dispatcher._looks_like_data_request("what is my revenue last month") is True

    def test_a_greeting_is_not_a_data_request_either_way(self):
        terms, metrics, graph = _with_workspace()
        with terms, metrics, graph:
            assert dispatcher._looks_like_data_request("hello there", "acct") is False

    def test_workspace_nouns_exclude_system_words(self):
        """"data", "name", "value" and friends appear in every schema and would
        match almost any sentence."""
        terms, metrics, graph = _with_workspace()
        with terms, metrics, graph:
            nouns = dispatcher._workspace_nouns("acct")
        assert "item" in nouns and "supplier" in nouns and "warehouse" in nouns
        assert not (nouns & {"name", "data", "value", "code", "date", "total"})

    def test_an_unreadable_workspace_falls_back_rather_than_failing(self):
        with patch("store.list_terms", side_effect=RuntimeError("no db")):
            assert dispatcher._workspace_nouns("acct") == set()


class TestAQuestionAboutTheProductIsNotAQuery:
    """The noun list contains "data", "info", "records", "fields", "columns" —
    exactly the words a capability question uses. So "what tables and data can
    you answer questions about" matched the fast path, skipped the analyst,
    entered SQL generation and came back "I cannot compile a trusted query
    until the semantic layer resolves the business event dataset to analyse"."""

    @pytest.mark.parametrize("question", [
        "what tables and data can you answer questions about",
        "what can you do",
        "what data do you have",
        "which metrics are available",
        "how does this work",
        "what are your capabilities",
        "who are you",
        "are you able to forecast",
    ])
    def test_meta_questions_go_to_the_analyst(self, question):
        assert dispatcher._looks_like_data_request(question, "acct") is False

    @pytest.mark.parametrize("question", [
        "show me all customer records",
        "what is my revenue last month",
        "list the top 10 orders by amount",
        "how many invoices this month",
        "which rows have missing data",
        "show me the data for Toronto",
    ])
    def test_real_data_questions_are_unaffected(self, question):
        """The guard must not swallow a question that merely uses the word
        "data" or "records" about the business."""
        assert dispatcher._looks_like_data_request(question) is True
