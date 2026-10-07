"""The same question gives the same numbers on Azure SQL, Snowflake and Oracle.

Every golden question is compiled for each warehouse, translated back to DuckDB
by sqlglot, run on the test warehouse and compared with the reference rows; so
are the shapes met on real warehouses (a formula metric, a whole-year row).
Some idioms the translator cannot carry back (T-SQL's @@DATEFIRST weekday,
string + concatenation, Oracle's TO_CHAR formats): such a query is not run here,
it is read back in its own dialect by the golden evaluation. Every query that
runs must give the reference rows, and most must run.
"""

from __future__ import annotations

import datetime as dt

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import Untranslatable, _Words, golden, learn, same_rows
from evals.core2.framework import materialize

# The share of golden questions each warehouse's SQL must carry back and run.
MUST_RUN = {"snowflake": 1.0, "tsql": 0.8, "oracle": 0.55}


def _back(sql: str, dialect: str) -> str | None:
    try:
        return sqlglot.transpile(sql, read=dialect, write="duckdb")[0]
    except sqlglot.errors.SqlglotError:
        return None


@pytest.mark.parametrize("dialect", sorted(MUST_RUN))
@pytest.mark.parametrize("name", [n for n in domains.available() if golden(n)])
def test_every_golden_question_gives_the_reference_rows_in_each_dialect(name, dialect):
    domain = domains.build(name)
    built, model = learn(domain, "descriptive")
    reference = DuckDBWarehouse(materialize(domain, "descriptive").con)
    cases = golden(name)
    today = dt.date.fromisoformat(str(cases["today"]))
    words = _Words(model, built)
    ran, total, wrong = 0, 0, []
    for case in cases["questions"]:
        if not case.get("plan") or not case.get("reference_sql"):
            continue
        try:
            logical = resolve(words.plan(case["plan"]), model, Context(today=today))
        except Untranslatable:
            continue
        total += 1
        back = _back(compile_query(logical, model, dialect).sql, dialect)
        if back is None:
            continue
        try:
            got = DuckDBWarehouse(built.con).query(back)
        except Exception:  # noqa: BLE001 - an idiom DuckDB has no word for: read back by the golden evaluation
            continue
        ran += 1
        expected = reference.query(case["reference_sql"])
        diff = same_rows(expected.columns, expected.rows, got.columns, got.rows,
                         order_matters=bool((case.get("expect") or {}).get("order_matters")))
        if diff:
            wrong.append(f"{case['id']}: {diff}\n{back}")
    assert not wrong, "\n\n".join(wrong)
    assert total and ran >= MUST_RUN[dialect] * total, f"only {ran} of {total} ran"


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle"])
def test_a_formula_and_a_whole_year_row_mean_the_same_on_each_warehouse(dialect):
    from core2.model.imports import Legacy, decisions
    from core2.model.overrides import apply_overrides
    from tests.test_core2_reads_real_warehouse_shapes import TODAY, _warehouse

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb

    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, schema="main"), options=BuildOptions(workers=1))
    report = decisions(model, Legacy(metrics=[
        {"id": 1, "name": "Inventory value", "is_active": 1, "metric_status": "validated",
         "base_table": "STOCK_DAY_FACT", "result_format": "currency", "sql_template": "SUM(ON_HND_QTY * ITM_CST)"}]))
    apply_overrides(model, [{"object_key": d.object_key, "field": d.field, "value": d.value} for d in report.decisions])
    value = next(m.slug for m in model.measures.values() if m.business_name == "Inventory value")
    sold = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(".sld_qty"))
    checks = [
        ({"intent": "value", "measures": [value], "time": {
            "window": {"kind": "between", "start": "2026-09-03", "end": "2026-09-03"},
            "compare": {"kind": "window", "window": {"kind": "between", "start": "2026-08-17", "end": "2026-08-17"}}}},
         "SELECT SUM(CASE WHEN snapshot_date = DATE '2026-09-03' THEN on_hnd_qty * itm_cst END), "
         "SUM(CASE WHEN snapshot_date = DATE '2026-08-17' THEN on_hnd_qty * itm_cst END) FROM stock_day_fact"),
        ({"intent": "value", "measures": [sold]},
         "SELECT SUM(sld_qty) FROM stock_month_fact WHERE period_key % 100 BETWEEN 1 AND 12"),
    ]
    for plan, reference_sql in checks:
        logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
        back = _back(compile_query(logical, model, dialect).sql, dialect)
        assert back is not None, dialect          # plain SQL in every dialect: it must carry back and run
        got = warehouse.query(back)
        want = warehouse.query(reference_sql)
        assert [float(v) for v in want.rows[0]] == pytest.approx([float(got.rows[0][i]) for i in range(len(want.rows[0]))])
