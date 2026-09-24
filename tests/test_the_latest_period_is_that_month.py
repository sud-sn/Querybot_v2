"""
The latest period of a yyyymm key is that month, not the second of January.

"What is the stock on hand" on a monthly period fact resolves its window from
the newest period present, probed once and cached. The probe read MAX of the
period key -- 202212 -- and the coercion that turns a probed value into a date
tried each format in turn, and "%Y%m%d" parses six digits by backtracking:
year 2022, month 1, day 2. The anchor became 2022-01-02. Every relative window
then measured from January -- "this month" was January, "last 3 months" ended
in January -- and the "data as of" banner named the wrong month. When the
newest year had only its whole-year row (202300), MAX picked that too.

Separately, the governed compiler used the raw period key as though it were a
date: `fact_rows.PRD_DMS_KEY > DATEADD(month, -3, anchor)`, an integer against
a date, which no warehouse accepts. The compiler now decodes it, the way the
SQL prompt always told the model to.

Synthetic tables in a mart's naming convention; no customer data.
"""

from __future__ import annotations

import pytest

from core.date_anchor import (
    _coerce_anchor,
    build_anchor_probe_sql,
    clear_cache,
    resolve_business_anchor,
)
from core.pipeline_helpers import (
    attempt_governed_temporal_metric_repair,
    compile_governed_temporal_metric_sql,
)
from core.validator import validate_sql_detailed

FACT = "MART.ITM_BAL_PRD_FCT"

PERIOD_POLICY = {
    "anchor_policy": "latest_available",
    "kind": "last_n",
    "amount": 3,
    "unit": "month",
    "fact_table": FACT,
    "fact_column": "PRD_DMS_KEY",
    "date_table": FACT,
    "date_column": "PRD_DMS_KEY",
    "date_key_type": "yyyymm_integer",
    "temporal_grain": "month",
    "business_role": "Period",
}


@pytest.fixture(autouse=True)
def _store():
    """The anchor is persisted; give it the tables it writes to."""
    import store.db as _db

    _db.init_db()
    clear_cache("acct-anchor", persistent=True)
    yield
    clear_cache("acct-anchor", persistent=True)


class TestAProbedPeriodIsThatMonth:

    def test_a_yyyymm_value_is_the_first_of_its_month(self):
        assert _coerce_anchor(202212, "yyyymm_integer") == "2022-12-01"
        assert _coerce_anchor("202203", "yyyymm_integer") == "2022-03-01"

    def test_a_whole_year_row_is_not_an_anchor(self):
        assert _coerce_anchor(202200, "yyyymm_integer") == ""

    def test_six_digits_are_never_read_as_a_day(self):
        """The defect itself: 202212 was 2022-01-02."""
        assert _coerce_anchor(202212) != "2022-01-02"

    def test_a_yyyymmdd_value_still_reads_as_a_day(self):
        assert _coerce_anchor(20221231) == "2022-12-31"
        assert _coerce_anchor(20221231, "yyyymmdd_integer") == "2022-12-31"

    def test_the_probe_never_picks_a_whole_year_row(self):
        sql = build_anchor_probe_sql(PERIOD_POLICY, "azure_sql")
        assert "% 100 BETWEEN 1 AND 12" in sql

    def test_the_resolved_anchor_is_the_latest_month(self):
        seen: list[str] = []

        def run_probe(sql):
            seen.append(sql)
            return [{"max_business_date": 202212}]

        anchor = resolve_business_anchor(
            "acct-anchor", PERIOD_POLICY, "azure_sql", run_probe,
        )
        assert anchor["value"] == "2022-12-01"
        assert seen and "% 100 BETWEEN 1 AND 12" in seen[-1]


class TestTheCompilerDecodesThePeriodKey:

    def _compile(self):
        context = {
            "semantic_plan": {"temporal_policies": [PERIOD_POLICY]},
            "metric_formulas": [{
                "name": "Purchased quantity", "formula_type": "expression",
                "sql_template": "SUM(fact_rows.PCH_QTY)",
            }],
            "question": "purchased quantity for the last 3 months",
            "canonical_question": "purchased quantity for the last 3 months",
        }
        return compile_governed_temporal_metric_sql(
            "azure_sql", {FACT}, {FACT},
            {FACT: {"PRD_DMS_KEY": "int", "PCH_QTY": "decimal"}}, context,
        )

    def test_the_window_compares_dates_not_integers(self):
        sql = self._compile()
        assert sql, "the compiler declined"
        assert "TRY_CONVERT(date, CONVERT(varchar(6), fact_rows.[PRD_DMS_KEY]) + '01', 112)" in sql
        assert "fact_rows.[PRD_DMS_KEY] >" not in sql

    def test_the_compiled_query_keeps_month_rows_only(self):
        sql = self._compile()
        result = validate_sql_detailed(
            sql, {FACT}, "azure_sql", None,
            {FACT: {"PRD_DMS_KEY": "int", "PCH_QTY": "decimal"}},
            {"semantic_plan": {
                "temporal_policies": [PERIOD_POLICY],
                "period_row_policies": [{"fact_table": FACT, "fact_column": "PRD_DMS_KEY"}],
            }},
        )
        assert result.ok, (result.code, result.reason)


WAREHOUSE = "MART.WHS_DMS"
COLUMNS = {
    FACT: {"PRD_DMS_KEY": "int", "PCH_QTY": "decimal", "WHS_DMS_KEY": "int"},
    WAREHOUSE: {"WHS_DMS_KEY": "int", "WHS_DSC": "varchar"},
}
PURCHASES = {
    "name": "Purchased quantity", "formula_type": "expression",
    "sql_template": "SUM(PCH_QTY)", "base_table": FACT,
}
DECODED = "TRY_CONVERT(date, CONVERT(varchar(6), fact_rows.[PRD_DMS_KEY]) + '01', 112)"


def _governed_verdict(sql: str, plan: dict):
    return validate_sql_detailed(
        sql, set(COLUMNS), "azure_sql", set(COLUMNS), COLUMNS,
        {"semantic_plan": {
            **plan,
            "period_row_policies": [{"fact_table": FACT, "fact_column": "PRD_DMS_KEY"}],
        }},
    )


class TestEveryGovernedCompilerDecodesIt:
    """The ranking compiler and the period-comparison compiler, too."""

    def test_a_ranking_over_the_last_three_months(self):
        plan = {
            "fields": [
                {"term": "Purchased quantity", "table": FACT, "column": "PCH_QTY",
                 "role": "measure"},
                {"term": "Warehouse", "table": WAREHOUSE, "column": "WHS_DSC",
                 "role": "display_dimension", "display_required": True},
            ],
            "joins": [{"from": FACT, "to": WAREHOUSE,
                       "conditions": [["WHS_DMS_KEY", "WHS_DMS_KEY"]],
                       "enforcement": "required"}],
            "temporal_policies": [PERIOD_POLICY],
        }
        sql = compile_governed_temporal_metric_sql(
            "azure_sql", set(COLUMNS), set(COLUMNS), COLUMNS, {
                "question": "top 5 warehouses by purchased quantity in the last 3 months",
                "top_n": {"limit": 5, "direction": "descending", "tie_policy": "exactly_n"},
                "metric_formulas": [PURCHASES],
                "semantic_plan": plan,
                "analytical_request_plan": {
                    "status": "compiled", "intent": "ranking", "source_fact": FACT,
                    "source_facts": [FACT], "top_n": 5, "output_shape": "table",
                },
            },
        )
        assert "TOP (5)" in sql, "the ranking compiler declined"
        assert f"WHERE {DECODED} >" in sql
        result = _governed_verdict(sql, plan)
        assert result.ok, (result.code, result.reason)

    def test_this_month_against_last_month(self):
        plan = {"temporal_policies": [PERIOD_POLICY]}
        sql = attempt_governed_temporal_metric_repair(
            "SELECT 1", "azure_sql", set(COLUMNS), set(COLUMNS), COLUMNS, {
                "question": "compare purchased quantity this month versus last month",
                "metric_formulas": [PURCHASES],
                "semantic_plan": plan,
            },
        )
        assert "period_comparison" in sql, "the comparison compiler declined"
        assert f"WHERE {DECODED} >=" in sql
        result = _governed_verdict(sql, plan)
        assert result.ok, (result.code, result.reason)
