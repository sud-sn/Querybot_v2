"""
Stock and balances are read at the last snapshot of each period, never summed.

A daily stock table holds the whole stock again every day. The governed
compilers answered a stock measure the way they answer revenue -- SUM over
every row in the window -- so "stock value for the last 6 months" added 182
daily snapshots together, about 180 times the stock, and "by month" gave each
month the sum of its thirty-odd snapshots. By warehouse, the same.

Now a level (the measure classes core/analysis_contract.py calls
semi-additive) is restricted to the last snapshot of the window, or of each
period when the answer is a series. A flow over the same window is still
summed. A stock measure asked beside a flow in one query, or with no date to
take a snapshot on, is left to the governed planner.

The real compilers, and their SQL executed: transpiled from T-SQL by sqlglot
and run by DuckDB on synthetic daily balances for two warehouses.
"""

from __future__ import annotations

import datetime

import duckdb
import pytest
import sqlglot

from core.analytical_request_plan import compile_analytical_request_plan
from core.contextual_dates import build_contextual_date_plan
from core.pipeline_context import _merge_semantic_plans
from core.pipeline_helpers import compile_governed_temporal_metric_sql
from core.question_normalizer import canonical_question

BAL, SNAP, DT, WHS = "MART.ITM_BAL_DLY_FCT", "MART.ITM_BAL_SNP_FCT", "MART.DT_DMS", "MART.WHS_DMS"
KNOWN = {BAL, SNAP, DT, WHS, "ITM_BAL_DLY_FCT", "ITM_BAL_SNP_FCT", "DT_DMS", "WHS_DMS"}
COLUMNS = {
    BAL: {"BAL_DT_DMS_KEY": "int", "WHS_DMS_KEY": "int", "BAL_VAL_AMT": "decimal", "RCV_QTY": "decimal"},
    SNAP: {"SNAP_DT": "date", "WHS_DMS_KEY": "int", "BAL_VAL_AMT": "decimal"},
    DT: {"DT_DMS_KEY": "int", "CAL_DT": "date"},
    WHS: {"WHS_DMS_KEY": "int", "WHS_NM": "varchar"},
}
FIRST_DAY, DAYS = datetime.date(2025, 7, 1), 274          # through 2026-03-31
LAST_NORTH = 1000 + DAYS - 1                              # North's stock on the last day
KEYED_DATE = {"fact_table": BAL, "fact_column": "BAL_DT_DMS_KEY", "dimension_table": DT,
              "dimension_key": "DT_DMS_KEY", "date_value_column": "CAL_DT", "date_key_type": "surrogate_fk",
              "date_role": "Balance Date", "context_name": "Balance Date",
              "governance_status": "approved", "resolution_source": "metric_default"}
NATIVE_DATE = {"fact_table": SNAP, "fact_column": "SNAP_DT", "dimension_table": "", "dimension_key": "",
               "date_value_column": "SNAP_DT", "date_key_type": "native_date", "date_role": "Snapshot Date",
               "context_name": "Snapshot Date", "governance_status": "approved",
               "resolution_source": "metric_default"}
BY_WAREHOUSE = {"enabled": True, "fields": [{
    "term": "warehouse", "table": WHS, "column": "WHS_NM", "role": "display_dimension",
    "display_required": True, "enforcement": "required", "source_table": BAL, "source_key_column": "WHS_DMS_KEY"}],
    "joins": [{"from": BAL, "to": WHS, "conditions": [("WHS_DMS_KEY", "WHS_DMS_KEY")], "enforcement": "required"}]}


def _metric(name, formula, table=BAL):
    return {"name": name, "formula_type": "expression", "sql_template": formula,
            "base_table": table, "_resolved_source_tables": [table]}


STOCK = _metric("Stock Value", "SUM(BAL_VAL_AMT)")
RECEIVED = _metric("Units Received", "SUM(RCV_QTY)")


@pytest.fixture(scope="module")
def warehouse():
    """North holds 1000 on the first day and one more each day after; South
    holds 500 throughout. Ten units are received at North each day."""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA MART")
    con.execute("CREATE TABLE MART.DT_DMS (DT_DMS_KEY INTEGER, CAL_DT DATE)")
    con.execute("CREATE TABLE MART.WHS_DMS (WHS_DMS_KEY INTEGER, WHS_NM VARCHAR)")
    con.execute("CREATE TABLE MART.ITM_BAL_DLY_FCT (BAL_DT_DMS_KEY INTEGER, WHS_DMS_KEY INTEGER, "
                "BAL_VAL_AMT DECIMAL(18,2), RCV_QTY DECIMAL(18,2))")
    con.execute("CREATE TABLE MART.ITM_BAL_SNP_FCT (SNAP_DT DATE, WHS_DMS_KEY INTEGER, BAL_VAL_AMT DECIMAL(18,2))")
    con.execute("INSERT INTO MART.WHS_DMS VALUES (1, 'North'), (2, 'South')")
    for day in range(DAYS):
        date = FIRST_DAY + datetime.timedelta(days=day)
        key = int(date.strftime("%Y%m%d"))
        con.execute("INSERT INTO MART.DT_DMS VALUES (?, ?)", [key, date])
        con.execute("INSERT INTO MART.ITM_BAL_DLY_FCT VALUES (?, 1, ?, 10), (?, 2, 500, 0)", [key, 1000 + day, key])
        con.execute("INSERT INTO MART.ITM_BAL_SNP_FCT VALUES (?, 1, ?), (?, 2, 500)", [date, 1000 + day, date])
    return con


def _compiled(question, metrics, binding=KEYED_DATE, extra_plan=None, anchor=None):
    canonical = canonical_question(question, "en")
    plan = build_contextual_date_plan(binding, canonical) if binding else {}
    plan = _merge_semantic_plans(extra_plan, plan) if extra_plan else plan
    plan["source_scope"] = {"selected_fact": binding["fact_table"] if binding else BAL}
    context = {"question": question, "canonical_question": canonical, "semantic_plan": plan,
               "metric_formulas": metrics, "top_n": None,
               "analytical_request_plan": compile_analytical_request_plan(canonical, plan, matched_metrics=metrics)}
    if anchor:
        context["resolved_date_anchor"] = {"value": anchor, "fact_table": binding["fact_table"],
                                           "fact_column": binding["fact_column"]}
    return compile_governed_temporal_metric_sql("azure_sql", KNOWN, None, COLUMNS, context)


def _run(con, sql):
    assert sql, "the governed compiler declined"
    return [tuple(float(v) if isinstance(v, (int, float)) or hasattr(v, "as_tuple") else v for v in row)
            for row in con.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0]).fetchall()]


class TestAStockMeasure:

    def test_over_a_window_is_the_last_snapshot(self, warehouse):
        assert _run(warehouse, _compiled("stock value for the last 6 months", [STOCK])) == [
            (LAST_NORTH + 500,)]

    def test_by_month_is_each_months_last_snapshot(self, warehouse):
        rows = _run(warehouse, _compiled("stock value by month for the last 3 months", [STOCK]))
        assert [value for _period, value in rows] == [1000 + 214 + 500, 1000 + 242 + 500, LAST_NORTH + 500]

    def test_by_warehouse_is_each_warehouses_stock_on_the_last_snapshot(self, warehouse):
        rows = _run(warehouse, _compiled("stock value by warehouse for the last 6 months", [STOCK],
                                         extra_plan=BY_WAREHOUSE))
        assert sorted(rows) == [("North", LAST_NORTH), ("South", 500)]

    def test_on_a_dated_snapshot_table_with_a_known_anchor(self, warehouse):
        sql = _compiled("stock value for the last 6 months", [_metric("Stock Value", "SUM(BAL_VAL_AMT)", SNAP)],
                        binding=NATIVE_DATE, anchor="2026-03-31")
        assert _run(warehouse, sql) == [(LAST_NORTH + 500,)]


class TestWhatStillAdds:

    def test_a_flow_over_the_same_window_is_summed(self, warehouse):
        rows = _run(warehouse, _compiled("units received for the last 6 months", [RECEIVED]))
        days_in_window = (datetime.date(2026, 3, 31) - datetime.date(2025, 9, 30)).days
        assert rows == [(10.0 * days_in_window,)]


class TestWhatIsLeftToThePlanner:

    def test_stock_beside_a_flow_in_one_query(self):
        assert _compiled("stock value and units received by warehouse for the last 6 months",
                         [STOCK, RECEIVED], extra_plan=BY_WAREHOUSE) == ""

    def test_stock_with_no_date_to_take_a_snapshot_on(self):
        assert _compiled("stock value by warehouse", [STOCK], binding=None, extra_plan=BY_WAREHOUSE) == ""
