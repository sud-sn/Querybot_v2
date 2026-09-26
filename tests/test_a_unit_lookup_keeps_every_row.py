"""
Looking up a row's unit of measure never drops the row.

A total of a quantity is kept per unit, and the unit is the item's, read
through the join from the fact to the item. Where the entity graph resolved
that edge for the question, the compiler writes the join it requires. Where it
did not -- "available quantity by warehouse" resolves the warehouse and the
date, not the item -- the compiler wrote an inner join, so a fact row whose
item was missing from the item table left the total without a word. The lookup
is now an outer join there: such a row is kept, under no unit.

The compiled SQL runs here against a small DuckDB warehouse; no customer data.
"""

from __future__ import annotations

import duckdb
import sqlglot
from sqlglot import exp

FACT = "MART.ITM_BAL_DLY_FCT"
_COLUMNS = {
    FACT: {"ITM_BAL_DLY_FCT_KEY": "bigint", "ITM_DMS_KEY": "int", "ITM_BAL_EFC_DT_DMS_KEY": "int",
           "ON_HND_QTY": "decimal"},
    "MART.ITM_DMS": {"ITM_DMS_KEY": "int", "UNT_OF_MSR": "nvarchar"},
    "MART.DT_DMS": {"DT_DMS_KEY": "int", "DMS_DT": "date"},
}
_UNITS = [{"kind": "units_of_measure", "fact_table": FACT, "quantities": ["ON_HND_QTY"],
           "unit_columns": [{"table": "MART.ITM_DMS", "column": "UNT_OF_MSR"}],
           "item_columns": [{"table": FACT, "column": "ITM_DMS_KEY"}],
           "unit_joins": [{"table": "MART.ITM_DMS", "fact_column": "ITM_DMS_KEY", "key": "ITM_DMS_KEY"}]}]
_DATE_EDGE = {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": "DT_DMS",
              "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]], "join_type": "LEFT"}
_ITEM_EDGE = {"from_schema": "MART", "from_table": "ITM_BAL_DLY_FCT", "to_schema": "MART", "to_table": "ITM_DMS",
              "conditions": [["ITM_DMS_KEY", "ITM_DMS_KEY"]], "join_type": "INNER"}


def _stock_on_hand(*edges: dict) -> str:
    from core.pipeline_helpers import compile_governed_temporal_metric_sql

    context = {
        "question": "what is our stock on hand", "canonical_question": "what is our stock on hand",
        "metric_formulas": [{"name": "Stock on hand", "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY)",
                             "base_table": FACT}],
        "analytical_request_plan": {"status": "compiled", "intent": "metric_query", "source_fact": FACT,
                                    "source_facts": [FACT]},
        "graph_context": {"resolved_edges": list(edges)},
        "semantic_plan": {
            "fields": [], "unit_policies": _UNITS,
            "joins": [{"from": FACT, "to": "MART.DT_DMS", "conditions": [["ITM_BAL_EFC_DT_DMS_KEY", "DT_DMS_KEY"]],
                       "enforcement": "required"}],
            "temporal_policies": [{"kind": "latest_snapshot", "amount": 1, "unit": "day", "fact_table": FACT,
                                   "fact_column": "ITM_BAL_EFC_DT_DMS_KEY", "dimension_table": "MART.DT_DMS",
                                   "dimension_key": "DT_DMS_KEY", "date_column": "DMS_DT",
                                   "date_key_type": "surrogate_fk", "role_alias": "balance_date"}],
        },
    }
    return compile_governed_temporal_metric_sql("azure_sql", set(_COLUMNS), set(_COLUMNS), _COLUMNS, context)


def _run(sql: str) -> dict:
    """{unit: total} from the compiled SQL over three rows, one of them for an
    item the item table does not hold."""
    con = duckdb.connect()
    con.execute('CREATE TABLE "ITM_BAL_DLY_FCT" ("ITM_BAL_DLY_FCT_KEY" BIGINT, "ITM_DMS_KEY" INTEGER, '
                '"ITM_BAL_EFC_DT_DMS_KEY" INTEGER, "ON_HND_QTY" DOUBLE)')
    con.execute('CREATE TABLE "ITM_DMS" ("ITM_DMS_KEY" INTEGER, "UNT_OF_MSR" VARCHAR)')
    con.execute('CREATE TABLE "DT_DMS" ("DT_DMS_KEY" INTEGER, "DMS_DT" DATE)')
    con.executemany('INSERT INTO "ITM_BAL_DLY_FCT" VALUES (?, ?, ?, ?)',
                    [(1, 101, 20260331, 10), (2, 102, 20260331, 4), (3, 999, 20260331, 5)])
    con.executemany('INSERT INTO "ITM_DMS" VALUES (?, ?)', [(101, "EA"), (102, "EA")])
    con.execute('INSERT INTO "DT_DMS" VALUES (20260331, DATE \'2026-03-31\')')
    tree = sqlglot.parse_one(sql, read="tsql")
    for table in tree.find_all(exp.Table):
        table.set("db", None)
        table.set("catalog", None)
    return {unit: total for unit, total in con.execute(tree.sql(dialect="duckdb")).fetchall()}


class TestTheUnitLookup:

    def test_a_row_whose_item_is_missing_is_kept(self):
        sql = _stock_on_hand(_DATE_EDGE)
        assert "LEFT JOIN [MART].[ITM_DMS] AS unit_item" in sql
        assert _run(sql) == {"EA": 14, None: 5}

    def test_the_join_the_graph_resolved_is_the_join(self):
        # The validator holds the SQL to the resolved edge; the graph's
        # decision stands where it made one.
        assert "\n    JOIN [MART].[ITM_DMS] AS unit_item" in _stock_on_hand(_DATE_EDGE, _ITEM_EDGE)


class TestTheGraphsJoinType:

    def test_an_unresolved_edge_takes_what_the_caller_names(self):
        from core.pipeline_helpers import _graph_join_type

        context = {"graph_context": {"resolved_edges": [_DATE_EDGE]}}
        assert _graph_join_type(context, FACT, "MART.ITM_DMS", unresolved="LEFT JOIN") == "LEFT JOIN"
        assert _graph_join_type(context, FACT, "MART.ITM_DMS") == "JOIN"
