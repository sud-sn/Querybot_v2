"""Budgets kept by a fiscal year and a period number (1 to 12) are dated by that month.

An invented wholesaler's ERP kept its sales budget per branch per fiscal period: FiscalYear (named by the year it
ends) and FiscalPeriod (1 is the fiscal year's first month), with no date column. Learn read a year and a quarter
number as a table's date, never a year and a month number: the budget had no date, and "sales vs budget fytd",
"fy27 budget total" and "budget vs actual by branch fy26" were told "Sales budget has no date".

A year number and a period or month number on the same table are now its month, the first day of which is the date
its rows count on: the calendar's months, or the fiscal year's when the names say fiscal. A period takes the months
it touches, each in full, and says so when it cuts one; a grouping by week is refused in words. Every number is
compared with SQL written by hand against invented rows.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import CompileError, compile_query
from core2.plan.catalog import catalog_text
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 10, 24)


@pytest.fixture(scope="module")
def erp():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE branches AS SELECT 'B' || (range + 1) AS branch_cd, ['Dallas', 'Atlanta', 'Denver', 'Toronto'][range + 1]
               || ' Branch' AS branch_name FROM range(4);
        CREATE TABLE sales AS SELECT range + 1 AS sale_id, 'B' || (range % 4 + 1) AS branch_cd,
               DATE '2024-07-01' + (range % 845)::INTEGER AS sale_date,
               round(40 + (range * 37 % 900) + (range % 7) * 3.5, 2) AS sale_amount FROM range(12000);
        CREATE TABLE sales_budget AS SELECT 'B' || (b.range + 1) AS branch_cd, y.range + 2025 AS fiscal_year,
               p.range + 1 AS fiscal_period,
               (90000 + b.range * 7000 + p.range * 1500 + y.range * 11000 + 0.25)::DECIMAL(12, 2) AS budget_amt
        FROM range(4) b, range(3) y, range(12) p;
        CREATE TABLE rep_quota AS SELECT 'B' || (b.range + 1) AS branch_cd, y.range + 2025 AS quota_yr,
               m.range + 1 AS quota_mth, (20000 + b.range * 900 + m.range * 300 + 0.5)::DECIMAL(12, 2) AS quota_amt
        FROM range(4) b, range(2) y, range(12) m;
    """)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-ym", options=BuildOptions(workers=1))
    model.settings.fiscal_year_start_month = 7
    from core2.model.overrides import apply_overrides

    apply_overrides(model, [])          # the fiscal year as the workspace sets it dates each period again
    return model, warehouse, con


def _table(model, name):
    return next(t for k, t in model.tables.items() if k.endswith(f".{name}"))


def _measure(model, table, column):
    return next(m for m in model.measures.values() if m.table.endswith(f".{table}")
                and getattr(m.expr, "column", None) and m.expr.column.endswith(f".{column}"))


def _between(start: str, end: str) -> dict:
    return {"window": {"kind": "between", "start": start, "end": end}}


def _rows(model, warehouse, plan: dict) -> tuple[list[dict], object]:
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    for dialect in ("snowflake", "tsql", "oracle", "postgres"):
        compile_query(logical, model, dialect)          # every warehouse can write it
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    return [dict(zip(result.columns, row)) for row in result.rows], logical


def test_a_fiscal_year_and_a_period_are_the_budgets_date(erp):
    model, _, _ = erp
    role = model.date_roles[_table(model, "sales_budget").default_date]
    assert role.granularity == "month" and role.period_months == 1 and role.fiscal, role
    assert model.columns[role.column].name == "fiscal_year"
    assert model.columns[role.quarter_column].name == "fiscal_period"
    # FY2025 starts in July 2024; FY2027's twelfth period is June 2027.
    assert (role.first, role.last) == (dt.date(2024, 7, 1), dt.date(2027, 6, 1))
    # Named by the period, with or without the AI's names.
    from core2.bootstrap import labels

    assert role.name == "Fiscal period"
    named = model.model_copy(deep=True)
    table = _table(named, "sales_budget").key
    labels._apply(named, table, {"name": "Sales budget", "columns": {
        "fiscal_year": {"name": "Fiscal year"}, "fiscal_period": {"name": "Fiscal period (month)"}}})
    assert named.date_roles[role.key].name == "Fiscal period (month)"
    quota = model.date_roles[_table(model, "rep_quota").default_date]
    assert quota.granularity == "month" and quota.period_months == 1 and not quota.fiscal
    assert (quota.first, quota.last) == (dt.date(2025, 1, 1), dt.date(2026, 12, 1))


def test_the_budget_of_one_month(erp):
    model, warehouse, con = erp
    budget = _measure(model, "sales_budget", "budget_amt")
    rows, _ = _rows(model, warehouse, {"intent": "breakdown", "measures": [budget.slug], "group_by": ["branch"],
                                       "time": _between("2026-09-01", "2026-09-30")})
    truth = dict(con.execute("""select branch_name, sum(budget_amt) from sales_budget join branches using (branch_cd)
        where fiscal_year = 2027 and fiscal_period = 3 group by 1""").fetchall())
    assert {r["branch_name"]: float(r[budget.slug]) for r in rows} == pytest.approx(
        {k: float(v) for k, v in truth.items()})


def test_a_fiscal_years_budget_and_its_first_quarter(erp):
    model, warehouse, con = erp
    budget = _measure(model, "sales_budget", "budget_amt")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                       "time": _between("2026-07-01", "2027-06-30")})
    (whole,) = con.execute("select sum(budget_amt) from sales_budget where fiscal_year = 2027").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(whole))
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                       "time": _between("2026-07-01", "2026-09-30")})
    (q1,) = con.execute("select sum(budget_amt) from sales_budget where fiscal_year = 2027 "
                        "and fiscal_period between 1 and 3").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(q1))


def test_the_budget_by_month_over_a_fiscal_year(erp):
    model, warehouse, con = erp
    budget = _measure(model, "sales_budget", "budget_amt")
    rows, _ = _rows(model, warehouse, {"intent": "trend", "measures": [budget.slug],
                                       "time": {"grain": "month", **_between("2025-07-01", "2026-06-30")}})
    truth = {p: float(v) for p, v in con.execute(
        "select fiscal_period, sum(budget_amt) from sales_budget where fiscal_year = 2026 group by 1").fetchall()}
    got = {(r["period"].month - 7) % 12 + 1: float(r[budget.slug]) for r in rows}
    assert got == pytest.approx(truth)
    assert min(r["period"] for r in rows) == dt.date(2025, 7, 1)


def test_sales_against_budget_by_branch_for_a_month(erp):
    model, warehouse, con = erp
    actual, budget = _measure(model, "sales", "sale_amount"), _measure(model, "sales_budget", "budget_amt")
    rows, _ = _rows(model, warehouse, {"intent": "breakdown", "group_by": ["branch"],
                                       "target": {"actual": actual.slug, "target": budget.slug},
                                       "time": _between("2026-08-01", "2026-08-31")})
    truth = {name: (float(a), float(t)) for name, a, t in con.execute("""
        with a as (select branch_cd, sum(sale_amount) v from sales
                   where sale_date between date '2026-08-01' and date '2026-08-31' group by 1),
             t as (select branch_cd, sum(budget_amt) v from sales_budget where fiscal_year = 2027 and fiscal_period = 2
                   group by 1)
        select branch_name, a.v, t.v from branches join a using (branch_cd) join t using (branch_cd)""").fetchall()}
    got = {r["branch_name"]: (float(r[actual.slug]), float(r[budget.slug])) for r in rows}
    assert got == pytest.approx(truth)


def test_a_period_that_cuts_a_month_counts_it_whole_and_says_so(erp):
    model, warehouse, con = erp
    budget = _measure(model, "sales_budget", "budget_amt")
    rows, logical = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                             "time": _between("2026-09-15", "2026-10-10")})
    (both,) = con.execute("select sum(budget_amt) from sales_budget where fiscal_year = 2027 "
                          "and fiscal_period in (3, 4)").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(both))
    said = " ".join(logical.notes + logical.cautions)
    assert "kept by month" in said and "Sep 2026" in said and "Oct 2026" in said, said


def test_a_month_has_no_weeks(erp):
    model, _, _ = erp
    budget = _measure(model, "sales_budget", "budget_amt")
    plan = Plan.model_validate({"kind": "query", "intent": "trend", "measures": [budget.slug],
                                "time": {"grain": "week", **_between("2026-07-01", "2026-09-30")}})
    with pytest.raises((ResolveError, CompileError), match="kept by month"):
        compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")


def test_a_calendar_year_and_month_number(erp):
    model, warehouse, con = erp
    quota = _measure(model, "rep_quota", "quota_amt")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [quota.slug],
                                       "time": _between("2026-03-01", "2026-03-31")})
    (march,) = con.execute("select sum(quota_amt) from rep_quota where quota_yr = 2026 and quota_mth = 3").fetchone()
    assert float(rows[0][quota.slug]) == pytest.approx(float(march))


def test_the_planner_is_told_the_budget_is_kept_by_fiscal_period(erp):
    model, _, _ = erp
    text = catalog_text(model)
    line = next(l for l in text.splitlines() if "| Fiscal period" in l)
    assert "by fiscal period" in line and "2024-07-01 to 2027-06-01" in line, line


def test_a_quarterly_table_with_a_number_of_other_things_stays_quarterly():
    """Found in review: a store's quarterly targets with visits_per_week (1 to 7) beside them were dated by "visits
    per week" read as the month. A year and quarter named alike are the table's quarter, whatever else it holds; a
    period column carrying other words never dates a year it does not share them with."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE store_targets AS SELECT s.range + 1 AS store_id, y.range + 2024 AS year,
                   q.range + 1 AS quarter, 1 + (s.range + q.range) % 7 AS visits_per_week,
                   1 + (s.range * 3 + q.range) % 12 AS review_month,
                   (100000 + s.range * 500 + q.range * 900)::DECIMAL(12, 2) AS target_amt
                   FROM range(6) s, range(2) y, range(4) q""")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-q", options=BuildOptions(workers=1))
    role = model.date_roles[_table(model, "store_targets").default_date]
    assert role.period_months == 3 and role.granularity == "quarter", role
    assert model.columns[role.quarter_column].name == "quarter"


def _learn(sql: str, client: str):
    con = duckdb.connect()
    con.execute(sql)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id=client, options=BuildOptions(workers=1)), \
        warehouse, con


def _year_roles(model, name):
    return [r for r in model.date_roles.values() if r.table == _table(model, name).key and r.quarter_column]


def test_a_model_year_or_a_vintage_beside_a_month_is_no_date():
    """Found in review: a car's model year beside the month it was registered, and a loan's vintage year beside its
    period on the books, were read as the year and month the rows count on."""
    model, _, _ = _learn("""CREATE TABLE vehicle_registrations AS SELECT 2015 + (r.range % 9) AS model_year,
                            1 + r.range % 12 AS month, 10 + (r.range * 7) % 90 AS registrations FROM range(400) r;
                            CREATE TABLE loan_vintage AS SELECT 2016 + (r.range % 8) AS vintage_year,
                            1 + r.range % 12 AS period, (1000 + r.range * 13 % 900)::DECIMAL(12, 2) AS balance
                            FROM range(400) r""", "t-vintage")
    assert not _year_roles(model, "vehicle_registrations") and not _year_roles(model, "loan_vintage")


def test_a_calendar_of_thirteen_periods_is_no_month():
    """Found in review: a retailer's 13 four-week periods were read as months, period 13 left out of every total."""
    model, _, _ = _learn("""CREATE TABLE period_sales AS SELECT 'S' || (s.range + 1) AS store_cd,
                            y.range + 2025 AS fiscal_year, p.range + 1 AS fiscal_period,
                            (9000 + s.range * 300 + p.range * 50 + 0.5)::DECIMAL(12, 2) AS sales_amt
                            FROM range(4) s, range(2) y, range(13) p""", "t-13")
    assert not [r for r in _year_roles(model, "period_sales") if r.period_months == 1]


def test_a_period_numbered_one_to_four_year_after_year_is_a_quarter():
    model, warehouse, con = _learn("""CREATE TABLE quarterly_budget AS SELECT 'B' || (b.range + 1) AS branch_cd,
                                      y.range + 2025 AS fiscal_year, p.range + 1 AS fiscal_period,
                                      (50000 + b.range * 900 + p.range * 700)::DECIMAL(12, 2) AS budget_amt
                                      FROM range(3) b, range(3) y, range(4) p""", "t-qp")
    (role,) = _year_roles(model, "quarterly_budget")
    assert role.period_months == 3 and role.granularity == "quarter", role


def test_a_ledger_budget_with_opening_and_closing_periods_is_dated_by_its_months():
    """Found in review: a budget kept by periods 0 (opening) to 13 (closing adjustments) had no date at all, and no
    figures. Periods 1 to 12 are its months; 0 and 13 count in a total over all time, in no month."""
    model, warehouse, con = _learn("""CREATE TABLE gl_budget AS SELECT 'A' || (a.range + 1) AS account_cd,
                                      y.range + 2025 AS fiscal_year, p.range AS fiscal_period,
                                      (1000 + a.range * 90 + p.range * 30 + 0.5)::DECIMAL(12, 2) AS budget_amt
                                      FROM range(5) a, range(2) y, range(14) p""", "t-gl")
    model.settings.fiscal_year_start_month = 7
    from core2.model.overrides import apply_overrides

    apply_overrides(model, [])
    (role,) = _year_roles(model, "gl_budget")
    assert role.period_months == 1
    budget = _measure(model, "gl_budget", "budget_amt")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                       "time": _between("2025-07-01", "2026-06-30")})
    (fy26,) = con.execute("select sum(budget_amt) from gl_budget where fiscal_year = 2026 "
                          "and fiscal_period between 1 and 12").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(fy26))
    rows, _ = _rows(model, warehouse, {"intent": "trend", "measures": [budget.slug],
                                       "time": {"grain": "month", **_between("2025-07-01", "2026-06-30")}})
    assert len(rows) == 12 and all(r["period"] is not None for r in rows), rows
    # Over all time, by month: the opening and closing periods are in no month (never a month 0 or a 13th month).
    rows, _ = _rows(model, warehouse, {"intent": "trend", "measures": [budget.slug], "time": {"grain": "month"}})
    months = {r["period"]: float(r[budget.slug]) for r in rows if r["period"] is not None}
    assert len(months) == 24 and min(months) == dt.date(2024, 7, 1) and max(months) == dt.date(2026, 6, 1), months


def test_a_fiscal_year_named_by_the_year_it_starts():
    """Found in review: FY2026 period 1 was dated April 2025 where the calendar names a fiscal year by the year it
    starts (FY2026 runs April 2026 to March 2027), so "FY2026's budget" gave FY2027's rows."""
    from core2.model.schema import Calendar

    model, warehouse, con = _learn("""CREATE TABLE plan_budget AS SELECT 'D' || (d.range + 1) AS dept_cd,
                                      y.range + 2025 AS fiscal_year, p.range + 1 AS fiscal_period,
                                      (100 + (p.range % 4) * 10 + y.range * 1000 + 0.5)::DECIMAL(12, 2) AS plan_amt
                                      FROM range(3) d, range(3) y, range(12) p""", "t-start")
    model.settings.fiscal_year_start_month = 4
    model.calendars["cal"] = Calendar(table="cal", key_column=None, date_column=None, fiscal_year_start_month=4,
                                      fiscal_year_named_by="start")
    from core2.model.overrides import apply_overrides

    apply_overrides(model, [])
    (role,) = _year_roles(model, "plan_budget")
    assert role.first == dt.date(2025, 4, 1), role
    budget = _measure(model, "plan_budget", "plan_amt")
    rows, _ = _rows(model, warehouse, {"intent": "value", "measures": [budget.slug],
                                       "time": _between("2026-04-01", "2027-03-31")})
    (fy26,) = con.execute("select sum(plan_amt) from plan_budget where fiscal_year = 2026").fetchone()
    assert float(rows[0][budget.slug]) == pytest.approx(float(fy26))
