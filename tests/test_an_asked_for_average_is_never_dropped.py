"""
An average, a minimum or a maximum the question asks for is never dropped.

The governed compilers answer a metric question from the metric registry
without a model: the metric's own formula, a window and a date. They never
read which aggregate the question asked for. "Total and average revenue for
the last 6 months", over a Revenue metric of SUM(amount), compiled to
SELECT SUM(amount) -- the total alone, labelled as the answer -- and so did
"average revenue" and "min and max order value", grouped by region or not,
and "compare average revenue this month vs last month" through the
comparison repair.

Now a question that asks for an aggregate its metric does not compute is
declined by both compilers and goes to the governed planner, which writes what
was asked. A metric that does compute it (AVG for an average) still compiles,
and so does one whose own name holds the word: "average order value" over an
Average Order Value ratio asks for that metric, in English or French.

The real compilers, on a synthetic sales fact and calendar.
"""

from __future__ import annotations

import pytest

from core.analytical_request_plan import compile_analytical_request_plan
from core.contextual_dates import build_contextual_date_plan
from core.pipeline_helpers import (
    _outer_aggregate,
    attempt_governed_temporal_metric_repair,
    compile_governed_temporal_metric_sql,
)
from core.question_normalizer import canonical_question

FACT, DIM = "dbo.SALES_FACT", "dbo.DT_DMS"
KNOWN = {FACT, DIM, "SALES_FACT", "DT_DMS"}
COLUMNS = {
    FACT: {"NET_SLS_AMT": "decimal", "ORD_DT_KEY": "int", "ORD_VAL_AMT": "decimal", "ORD_ID": "int",
           "REGION_NM": "varchar"},
    DIM: {"DT_KEY": "int", "CAL_DT": "date"},
}
BINDING = {
    "fact_table": FACT, "fact_column": "ORD_DT_KEY",
    "dimension_table": DIM, "dimension_key": "DT_KEY", "date_value_column": "CAL_DT",
    "date_key_type": "surrogate_fk", "date_role": "Order Date", "context_name": "Order Date",
    "governance_status": "approved", "resolution_source": "metric_default",
}


def _metric(name, formula, synonyms=""):
    return {"name": name, "formula_type": "expression", "sql_template": formula, "synonyms": synonyms,
            "base_table": FACT, "_resolved_source_tables": [FACT]}


def _context(question, metrics, lang="en"):
    canonical = canonical_question(question, lang)
    plan = build_contextual_date_plan(BINDING, canonical)
    plan.setdefault("source_scope", {"selected_fact": FACT})
    return {"question": question, "canonical_question": canonical, "semantic_plan": plan,
            "metric_formulas": metrics,
            "analytical_request_plan": compile_analytical_request_plan(canonical, plan, matched_metrics=metrics),
            "top_n": None}


def _compiled(question, metrics, lang="en"):
    return compile_governed_temporal_metric_sql("azure_sql", KNOWN, None, COLUMNS, _context(question, metrics, lang))


REVENUE = _metric("Revenue", "SUM(NET_SLS_AMT)")
ORDER_VALUE = _metric("Order Value", "SUM(ORD_VAL_AMT)")


class TestTheCompiler:

    def test_a_total_still_compiles(self):
        sql = _compiled("total revenue for the last 6 months", [REVENUE])
        assert "SUM(NET_SLS_AMT)" in sql

    @pytest.mark.parametrize("question, metrics, lang", [
        ("total and average revenue for the last 6 months", [REVENUE], "en"),
        ("average revenue for the last 6 months", [REVENUE], "en"),
        ("min and max order value for the last 6 months", [ORDER_VALUE], "en"),
        ("minimum order value for the last 6 months", [ORDER_VALUE], "en"),
        ("maximum order value for the last 6 months", [ORDER_VALUE], "en"),
        ("median order value for the last 6 months", [ORDER_VALUE], "en"),
        ("average revenue by region for the last 6 months", [REVENUE], "en"),
        ("chiffre d'affaires moyen des 6 derniers mois", [REVENUE], "fr"),
        ("valeur de commande maximale des 6 derniers mois", [ORDER_VALUE], "fr"),
        ("average revenue for the last 6 months", [_metric("Revenue", "SUM(NET_SLS_AMT)", "sales, turnover")], "en"),
    ])
    def test_an_aggregate_the_metric_does_not_compute_is_not_answered_with_its_formula(
            self, question, metrics, lang):
        assert _compiled(question, metrics, lang) == ""

    def test_a_metric_that_computes_the_average_answers_it(self):
        sql = _compiled("average order value for the last 6 months", [_metric("Order Value", "AVG(ORD_VAL_AMT)")])
        assert "AVG(ORD_VAL_AMT)" in sql

    @pytest.mark.parametrize("question, name, synonyms, lang", [
        ("average order value for the last 6 months", "Average Order Value", "", "en"),
        ("aov for the last 6 months", "Order Value Ratio", "aov, average basket", "en"),
        ("average basket for the last 6 months", "Order Value Ratio", "aov, average basket", "en"),
        ("panier moyen des 6 derniers mois", "Panier moyen", "", "fr"),
    ])
    def test_a_metric_whose_own_name_holds_the_word_answers_it(self, question, name, synonyms, lang):
        ratio = "SUM(ORD_VAL_AMT) / NULLIF(COUNT(DISTINCT ORD_ID), 0)"
        sql = _compiled(question, [_metric(name, ratio, synonyms)], lang)
        assert "COUNT(DISTINCT ORD_ID)" in sql


class TestTheComparisonRepair:

    def _repaired(self, question):
        return attempt_governed_temporal_metric_repair("SELECT 1", "azure_sql", KNOWN, None, COLUMNS,
                                                       _context(question, [REVENUE]))

    def test_a_comparison_of_totals_is_still_repaired(self):
        assert "SUM(NET_SLS_AMT)" in self._repaired("compare revenue this month vs last month")

    def test_a_comparison_of_averages_is_not_answered_with_totals(self):
        assert self._repaired("compare average revenue this month vs last month") == ""


@pytest.mark.parametrize("formula, outer", [
    ("SUM(NET_SLS_AMT)", "SUM"),
    ("avg(ORD_VAL_AMT - DSC_AMT)", "AVG"),
    ("COUNT(DISTINCT ORD_ID)", "COUNT"),
    ("SUM(A) / NULLIF(SUM(B), 0)", ""),
    ("SUM(A) + 1", ""),
    ("NET_SLS_AMT", ""),
])
def test_what_a_formula_computes_last(formula, outer):
    assert _outer_aggregate(formula) == outer
