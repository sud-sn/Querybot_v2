"""Targets and budgets kept by a year and a quarter number are dated by that quarter.

A pharmacy chain's store targets were kept per store per quarter in two numbers (target year, target quarter), a
bank's budgets per branch per fiscal quarter (FiscalYear, FiscalQuarter): no date column at all. Learn read both
numbers as keys, the tables had no date, and "scripts against target in Q3 2026" or "originations against budget
for fiscal 2025" could not pick the quarter: the plan dropped the period and set every quarter's target against
it, or left the target out.

Now a year number and a quarter number on the same table are its quarter, the first day of which is the date its
rows are counted on: calendar quarters, or the fiscal year's when the names say fiscal (a fiscal year named by
the year it ends, as everywhere else). A period takes the quarters it touches, each in full, and says so when it
cuts one; a grouping by month is refused in words, a quarter having no months. "The stores that missed target"
(the actual below its own target) passes the planner's check as the target's own filter, as the resolver reads it,
where it went back to the AI as a figure compared with a name. Every number is compared with SQL written by hand
against invented rows.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import CompileError, compile_query
from core2.model.overrides import apply_overrides
from core2.plan.ir import Plan
from core2.plan.planner import check
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 10, 21)


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE stores AS SELECT range + 1 AS store_id, 'Store ' || chr(65 + range::INTEGER) || repeat('x', range % 3)
               AS store_name FROM range(6);
        CREATE TABLE sales AS SELECT range + 1 AS sale_id, range % 6 + 1 AS store_id,
               DATE '2024-01-01' + (range % 1000)::INTEGER AS sale_date,
               round(20 + (range * 37 % 400) + (range % 7) * 3.5, 2) AS sale_amount FROM range(9000);
        CREATE TABLE store_targets AS SELECT s.range + 1 AS store_id, y.range + 2024 AS tgt_yr, q.range + 1 AS tgt_qtr,
               (150000 + (s.range % 3) * 9000 + q.range * 4000 + y.range * 6000 + 0.25)::DECIMAL(12, 2) AS sales_tgt
        FROM range(6) s, range(3) y, range(4) q;
        CREATE TABLE store_budgets AS SELECT s.range + 1 AS store_id, y.range + 2024 AS fiscal_year,
               q.range + 1 AS fiscal_quarter,
               (140000 + (s.range % 2) * 7000 + q.range * 5000 + y.range * 8000 + 0.5)::DECIMAL(12, 2) AS sales_budget
        FROM range(6) s, range(4) y, range(4) q;
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-yq", options=BuildOptions(workers=1))
    return model, warehouse, con


def _table(model, name):
    return next(t for k, t in model.tables.items() if k.endswith(f".{name}"))


def _measure(model, table, column):
    return next(m for m in model.measures.values() if m.table.endswith(f".{table}")
                and getattr(m.expr, "column", None) and m.expr.column.endswith(f".{column}"))


def _rows(model, warehouse, plan: dict, today=TODAY) -> tuple[list[dict], object]:
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=today))
    for dialect in ("snowflake", "tsql", "oracle", "postgres"):
        compile_query(logical, model, dialect)          # every warehouse can write it
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    return [dict(zip(result.columns, row)) for row in result.rows], logical


def test_a_year_and_a_quarter_are_the_tables_date(shop):
    model, _, _ = shop
    for table, fiscal in (("store_targets", False), ("store_budgets", True)):
        t = _table(model, table)
        assert t.default_date, f"{table} has no date"
        role = model.date_roles[t.default_date]
        assert role.granularity == "quarter" and role.quarter_column and role.fiscal is fiscal, role
        assert model.columns[role.column].name in ("tgt_yr", "fiscal_year")
        assert model.columns[role.quarter_column].name in ("tgt_qtr", "fiscal_quarter")
    calendar = model.date_roles[_table(model, "store_targets").default_date]
    assert (calendar.first, calendar.last) == (dt.date(2024, 1, 1), dt.date(2026, 10, 1))


def test_the_targets_of_one_quarter(shop):
    model, warehouse, con = shop
    target = _measure(model, "store_targets", "sales_tgt")
    rows, _ = _rows(model, warehouse, {"intent": "breakdown", "measures": [target.slug], "group_by": ["store"],
                                       "time": {"window": {"kind": "between", "start": "2026-07-01",
                                                           "end": "2026-09-30"}}})
    truth = dict(con.execute("""select store_name, sum(sales_tgt) from store_targets join stores using (store_id)
        where tgt_yr = 2026 and tgt_qtr = 3 group by 1""").fetchall())
    got = {r["store_name"]: float(r[target.slug]) for r in rows}
    assert got == pytest.approx({k: float(v) for k, v in truth.items()})


def test_sales_against_target_for_a_quarter(shop):
    model, warehouse, con = shop
    actual, target = _measure(model, "sales", "sale_amount"), _measure(model, "store_targets", "sales_tgt")
    rows, _ = _rows(model, warehouse, {"intent": "breakdown", "group_by": ["store"],
                                       "target": {"actual": actual.slug, "target": target.slug},
                                       "time": {"window": {"kind": "between", "start": "2025-04-01",
                                                           "end": "2025-06-30"}}})
    truth = {name: (float(a), float(t)) for name, a, t in con.execute("""
        with a as (select store_id, sum(sale_amount) v from sales
                   where sale_date between date '2025-04-01' and date '2025-06-30' group by 1),
             t as (select store_id, sum(sales_tgt) v from store_targets where tgt_yr = 2025 and tgt_qtr = 2 group by 1)
        select store_name, a.v, t.v from stores join a using (store_id) join t using (store_id)""").fetchall()}
    got = {r["store_name"]: (float(r[actual.slug]), float(r[target.slug])) for r in rows}
    assert got == pytest.approx(truth)


def test_fiscal_budgets_are_read_by_the_workspaces_fiscal_year(shop):
    # Fiscal years from November, named by the year they end: FY2026 Q3 is May to July 2026.
    model, warehouse, con = shop
    model = model.model_copy(deep=True)
    model.settings.fiscal_year_start_month = 11
    budget = _measure(model, "store_budgets", "sales_budget")
    rows, _ = _rows(model, warehouse, {"intent": "breakdown", "measures": [budget.slug], "group_by": ["store"],
                                       "time": {"window": {"kind": "between", "start": "2026-05-01",
                                                           "end": "2026-07-31"}}})
    truth = dict(con.execute("""select store_name, sum(sales_budget) from store_budgets join stores using (store_id)
        where fiscal_year = 2026 and fiscal_quarter = 3 group by 1""").fetchall())
    assert {r["store_name"]: float(r[budget.slug]) for r in rows} == pytest.approx(
        {k: float(v) for k, v in truth.items()})
    # The whole fiscal year: its four quarters.
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                       "time": {"window": {"kind": "between", "start": "2025-11-01",
                                                           "end": "2026-10-31"}}})
    (whole,) = con.execute("select sum(sales_budget) from store_budgets where fiscal_year = 2026").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(whole))


def test_targets_by_quarter_over_a_year(shop):
    model, warehouse, con = shop
    target = _measure(model, "store_targets", "sales_tgt")
    rows, _ = _rows(model, warehouse, {"intent": "trend", "measures": [target.slug],
                                       "time": {"grain": "quarter", "window": {"kind": "between",
                                                                              "start": "2025-01-01",
                                                                              "end": "2025-12-31"}}})
    truth = {q: float(v) for q, v in con.execute(
        "select tgt_qtr, sum(sales_tgt) from store_targets where tgt_yr = 2025 group by 1").fetchall()}
    got = {(r["period"].month - 1) // 3 + 1: float(r[target.slug]) for r in rows}
    assert got == pytest.approx(truth)


def test_a_period_that_cuts_a_quarter_counts_it_whole_and_says_so(shop):
    model, warehouse, con = shop
    target = _measure(model, "store_targets", "sales_tgt")
    rows, logical = _rows(model, warehouse, {"intent": "value", "measures": [target.slug],
                                             "time": {"window": {"kind": "between", "start": "2026-07-01",
                                                                 "end": "2026-07-31"}}})
    (q3,) = con.execute("select sum(sales_tgt) from store_targets where tgt_yr = 2026 and tgt_qtr = 3").fetchone()
    assert float(rows[0][target.slug]) == pytest.approx(float(q3))
    said = " ".join(logical.notes + logical.cautions)
    assert "kept by quarter" in said and "Q3 2026" in said, said


def test_a_quarter_has_no_months(shop):
    model, _, _ = shop
    target = _measure(model, "store_targets", "sales_tgt")
    plan = Plan.model_validate({"kind": "query", "intent": "trend", "measures": [target.slug],
                                "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01",
                                                                     "end": "2025-12-31"}}})
    with pytest.raises((ResolveError, CompileError), match="kept by quarter"):
        compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")


def test_a_table_with_a_real_date_keeps_it(shop):
    # The sales table has a year-like number nowhere and its own date: nothing changes for it.
    model, _, _ = shop
    role = model.date_roles[_table(model, "sales").default_date]
    assert model.columns[role.column].name == "sale_date" and not role.quarter_column


def test_an_admin_dates_a_table_by_its_year_and_quarter(shop):
    # Learn missed it: the admin adds the date on the Dates page, and the targets of one quarter are read by it.
    model, warehouse, con = shop
    model = model.model_copy(deep=True)
    table = _table(model, "store_targets")
    found = model.date_roles.pop(table.default_date)
    table.default_date = None
    added = found.model_dump(mode="json") | {"key": "admin:target_quarter", "slug": "admin_target_quarter",
                                             "granularity": "year", "is_default": False}
    stray = added | {"key": "admin:stray", "slug": "admin_stray",
                     "quarter_column": next(c for c in model.columns if c.endswith(".store_budgets.fiscal_quarter"))}
    notes = apply_overrides(model, [
        {"object_key": "date_role:admin:target_quarter", "field": "define", "value": added},
        {"object_key": "date_role:admin:target_quarter", "field": "is_default", "value": True},
        {"object_key": "date_role:admin:stray", "field": "define", "value": stray}])
    assert "admin:stray" not in model.date_roles and any("not in the data" in n for n in notes), notes
    role = model.date_roles["admin:target_quarter"]
    assert role.granularity == "quarter" and table.default_date == "admin:target_quarter"
    target = _measure(model, "store_targets", "sales_tgt")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [target.slug],
                                       "time": {"window": {"kind": "between", "start": "2025-10-01",
                                                           "end": "2025-12-31"}}})
    (q4,) = con.execute("select sum(sales_tgt) from store_targets where tgt_yr = 2025 and tgt_qtr = 4").fetchone()
    assert float(rows[0][target.slug]) == pytest.approx(float(q4))


def test_the_stores_that_missed_their_quarters_target(shop):
    # The plan the AI writes for "which stores missed target in Q2 2025": passed by the planner's check, then read.
    model, warehouse, con = shop
    actual, target = _measure(model, "sales", "sale_amount"), _measure(model, "store_targets", "sales_tgt")
    plan = {"intent": "breakdown", "group_by": ["store"], "target": {"actual": actual.slug, "target": target.slug},
            "filters": [{"field": actual.slug, "op": "lt", "values": [target.slug]}],
            "time": {"window": {"kind": "between", "start": "2025-04-01", "end": "2025-06-30"}}}
    problems, _ = check(Plan.model_validate({"kind": "query", **plan}), model, TODAY, "which stores missed target")
    assert problems == []
    rows, _ = _rows(model, warehouse, plan)
    truth = {name for (name,) in con.execute("""
        with a as (select store_id, sum(sale_amount) v from sales
                   where sale_date between date '2025-04-01' and date '2025-06-30' group by 1),
             t as (select store_id, sum(sales_tgt) v from store_targets where tgt_yr = 2025 and tgt_qtr = 2 group by 1)
        select store_name from stores join a using (store_id) join t using (store_id) where a.v < t.v""").fetchall()}
    assert truth and {r["store_name"] for r in rows} == truth
    # Any other figure compared with a name still goes back to be written as a difference.
    other = {**plan, "filters": [{"field": actual.slug, "op": "lt", "values": [target.slug]}], "target": None,
             "measures": [actual.slug, target.slug]}
    problems, _ = check(Plan.model_validate({"kind": "query", **other}), model, TODAY,
                        "stores whose sales were below target")
    assert problems and "another figure" in problems[0]


def test_a_table_with_its_own_date_keeps_it_and_its_fiscal_numbers_stay_groupings():
    # Revenue dated by its invoice, with the fiscal year and quarter worked out from it beside: counted by the
    # invoice, "revenue in March 2025" is March's, and the fiscal year is still a grouping. Two numbers of
    # different things (an order's year, a shipment's quarter) are no quarter.
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE fact_revenue AS SELECT range + 1 AS revenue_id, DATE '2024-11-01' + (range % 700)::INTEGER
               AS invoice_date, round(10 + range % 90 + 0.25, 2) AS revenue FROM range(4000);
        ALTER TABLE fact_revenue ADD COLUMN fiscal_year INTEGER;
        ALTER TABLE fact_revenue ADD COLUMN fiscal_quarter INTEGER;
        UPDATE fact_revenue SET fiscal_year = year(invoice_date) + (CASE WHEN month(invoice_date) >= 11 THEN 1 ELSE 0 END),
               fiscal_quarter = ((month(invoice_date) - 11 + 12) % 12) // 3 + 1;
        CREATE TABLE shipments AS SELECT range + 1 AS shipment_id, 2024 + range % 2 AS order_year,
               range % 4 + 1 AS ship_quarter, (range % 50 + 0.5)::DECIMAL(10, 2) AS shipped_weight FROM range(400);
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-yq-own", options=BuildOptions(workers=1))
    revenue = _table(model, "fact_revenue")
    assert model.columns[model.date_roles[revenue.default_date].column].name == "invoice_date"
    assert not any(r.quarter_column for r in model.date_roles.values()), [r.name for r in model.date_roles.values()]
    assert any(model.columns[a.column].name == "fiscal_year" for a in model.attributes.values())
    measure = _measure(model, "fact_revenue", "revenue")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [measure.slug],
                                       "time": {"window": {"kind": "between", "start": "2025-03-01",
                                                           "end": "2025-03-31"}}})
    (march,) = con.execute("select sum(revenue) from fact_revenue where invoice_date between date '2025-03-01' "
                           "and date '2025-03-31'").fetchone()
    assert float(rows[0][measure.slug]) == pytest.approx(float(march))
    assert _table(model, "shipments").default_date is None


def test_a_fiscal_quarter_with_no_fiscal_year_set_is_asked_and_said(shop):
    # Learn found no fiscal year: the budgets' quarters are read as the calendar's, and both the admin (a review
    # item) and the reader (a caution) are told, never a silent calendar reading of fiscal numbers.
    model, warehouse, _ = shop
    assert model.settings.fiscal_year_start_month is None
    role = model.date_roles[_table(model, "store_budgets").default_date]
    assert role.status == "proposed" and any(r.key == f"fiscal_year:{role.key}" for r in model.review)
    budget = _measure(model, "store_budgets", "sales_budget")
    _, logical = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                          "time": {"window": {"kind": "between", "start": "2026-01-01",
                                                              "end": "2026-03-31"}}})
    assert any("fiscal year is not set" in c for c in logical.cautions), logical.cautions


def test_the_quarters_dates_follow_the_fiscal_year_set_after_learn(shop):
    # Set on the Dates page after Learn (November): FY2024 Q1 starts on 1 Nov 2023, FY2027 Q4 on 1 Aug 2027.
    model, warehouse, con = shop
    model = model.model_copy(deep=True)
    apply_overrides(model, [{"object_key": "settings:workspace", "field": "fiscal_year_start_month", "value": 11}])
    role = model.date_roles[_table(model, "store_budgets").default_date]
    assert (role.first, role.last) == (dt.date(2023, 11, 1), dt.date(2027, 8, 1))
    calendar = model.date_roles[_table(model, "store_targets").default_date]
    assert (calendar.first, calendar.last) == (dt.date(2024, 1, 1), dt.date(2026, 10, 1))
    # A question worked out without the fiscal year (as a list of members is) still reads the workspace's.
    budget = _measure(model, "store_budgets", "sales_budget")
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": [budget.slug],
                                           "time": {"window": {"kind": "between", "start": "2026-05-01",
                                                               "end": "2026-07-31"}}}), model,
                      Context(today=TODAY))
    logical.fiscal_start = None
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    (q3,) = con.execute("select sum(sales_budget) from store_budgets where fiscal_year = 2026 and fiscal_quarter = 3"
                        ).fetchone()
    assert float(result.rows[0][0]) == pytest.approx(float(q3))
