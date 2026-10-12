"""Rows in force on each date from their start and end dates: headcount at each month end, hires and leavers, and
the end rate, equal to the warehouse's own monthly headcount and to SQL written by hand.

Asked of the invented HR warehouse through the whole answer path (a scripted plan, as the planner writes it). An
employee is in force at a month's end when hired on or before its last day and not terminated by then; one still
working has a placeholder termination date far ahead, which is no end. The warehouse keeps its own headcount at
each month end: the two must agree.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1_2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-06-30"}
ACTIVE = {"start": "hire_date", "end": "termination_date"}


@pytest.fixture(scope="module")
def hr():
    return learn(domains.build("hr"), "descriptive")


def _ask(learned, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _rows(payload: dict) -> dict:
    return {str(r["period"])[:10]: r for r in payload["export_rows"]}


def test_headcount_at_each_month_end_matches_the_warehouses_own(hr):
    built, _ = hr
    own = dict(built.con.execute("""SELECT date_trunc('month', month_end_date)::DATE, COUNT(*) FROM headcount_monthly
        WHERE month_end_date >= DATE '2025-01-01' AND month_end_date < DATE '2025-07-01' GROUP BY 1""").fetchall())
    got = _ask(hr, {"active": ACTIVE, "time": {"window": H1_2025, "grain": "month"}})
    assert {k: r["active"] for k, r in _rows(got).items()} == {str(k): v for k, v in own.items()}
    assert "in force at the end of each month" in got["definition"]
    assert "an empty or far-off termination date is none" in got["definition"]


def test_hires_leavers_and_the_end_rate_by_quarter(hr):
    built, _ = hr
    rows = built.con.execute("""
        WITH q AS (SELECT * FROM (VALUES (DATE '2025-01-01', DATE '2025-04-01'), (DATE '2025-04-01', DATE '2025-07-01'))
                   AS t(lo, hi))
        SELECT lo,
          (SELECT COUNT(*) FROM employees WHERE hire_date >= lo AND hire_date < hi),
          (SELECT COUNT(*) FROM employees WHERE termination_date >= lo AND termination_date < hi),
          (SELECT COUNT(*) FROM employees WHERE hire_date < lo AND NOT (termination_date >= DATE '1901-01-01'
                                                                         AND termination_date < lo))
        FROM q""").fetchall()
    expected = {str(lo): (s, e, b) for lo, s, e, b in rows}
    every = _rows(_ask(hr, {"active": {**ACTIVE, "status": "all"}, "time": {"window": H1_2025, "grain": "quarter"}}))
    rate = _rows(_ask(hr, {"active": {**ACTIVE, "status": "end_rate"},
                           "time": {"window": H1_2025, "grain": "quarter"}}))
    for quarter, (starts, ends, before) in expected.items():
        assert (every[quarter]["starts"], every[quarter]["ends"]) == (starts, ends)
        assert float(rate[quarter]["end_rate"]) == pytest.approx(100.0 * ends / before)


def test_headcount_by_each_employees_own_department(hr):
    # By the department the employee list holds for each (today's), not the one a monthly snapshot held then.
    built, _ = hr
    own = built.con.execute("""SELECT d.department_name, COUNT(*) FROM employees e
        JOIN departments d ON e.department_id = d.department_id
        WHERE e.hire_date < DATE '2025-04-01' AND NOT (e.termination_date >= DATE '1901-01-01'
                                                     AND e.termination_date < DATE '2025-04-01')
        GROUP BY 1""").fetchall()
    got = _ask(hr, {"active": ACTIVE, "group_by": ["department.name"],
                    "time": {"window": {"kind": "between", "start": "2025-03-01", "end": "2025-03-31"},
                             "grain": "month"}})
    assert {r["department_name"]: r["active"] for r in got["export_rows"]} == dict(own)


def test_two_dates_of_different_rows_are_refused(hr):
    built, model = hr
    plan = Plan.model_validate({"kind": "query", "active": {"start": "hire_date", "end": "pay_date"},
                                "time": {"window": H1_2025, "grain": "month"}})
    with pytest.raises(ResolveError, match="two dates of the same rows"):
        resolve(plan, model, Context(today=TODAY))


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
@pytest.mark.parametrize("status", ["active", "all", "end_rate"])
def test_the_sql_reads_on_every_warehouse(hr, dialect, status):
    built, model = hr
    plan = Plan.model_validate({"kind": "query", "active": {**ACTIVE, "status": status},
                                "group_by": ["department.name"], "time": {"window": H1_2025, "grain": "month"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


def test_active_subscriptions_with_an_empty_end_match_the_monthly_snapshot():
    learned = learn(domains.build("subscriptions"), "descriptive")
    built, _ = learned
    own = dict(built.con.execute("""SELECT date_trunc('month', month_end_date)::DATE, COUNT(*) FROM mrr_monthly
        WHERE month_end_date >= DATE '2025-01-01' AND month_end_date < DATE '2025-07-01' GROUP BY 1""").fetchall())
    got = _ask(learned, {"active": {"start": "start_date", "end": "end_date"},
                         "time": {"window": H1_2025, "grain": "month"}})
    shown = {k: r["active"] for k, r in _rows(got).items()}
    # Every month the snapshot was loaded agrees; the one whose load never ran is counted from the rows alone.
    assert {k: shown[k] for k in map(str, own)} == {str(k): v for k, v in own.items()}
    (february,), = built.con.execute("SELECT COUNT(*) FROM subscriptions WHERE start_date < DATE '2025-03-01' AND "
                                     "(end_date IS NULL OR end_date >= DATE '2025-03-01')").fetchone(),
    assert "2025-02-01" not in map(str, own) and shown["2025-02-01"] == february
