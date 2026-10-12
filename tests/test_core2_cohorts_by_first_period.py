"""Members grouped by the period of their first activity, and how many are still active each period after it,
equal to SQL written by hand.

"Retention by month of first order" is asked of the invented retail warehouse through the whole answer path (a
scripted plan, as the planner writes it). A customer's cohort is the month of their first order ever; each cell
is the share of that cohort with orders n months later (0: the first month itself, always 100%). A month that has
happened with no one active shows 0, not nothing; no month after the last with any orders is shown.

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
Q1_2023 = {"kind": "between", "start": "2023-01-01", "end": "2023-03-31"}
MP = """SELECT DISTINCT o.customer_id AS k, date_trunc('month', d.full_date)::DATE AS p FROM order_lines o
        JOIN calendar d ON o.order_date_key = d.date_key
        WHERE o.customer_id IN (SELECT customer_id FROM customers)
          AND d.full_date >= DATE '1901-01-01' AND d.full_date < DATE '9000-01-01'"""


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _cells(learned) -> dict:
    rows = learned[0].con.execute(f"""
        WITH mp AS ({MP}), f AS (SELECT k, MIN(p) AS first FROM mp GROUP BY k),
        cells AS (SELECT f.first AS cohort, date_diff('month', f.first, mp.p) AS age, COUNT(*) AS n FROM mp
                  JOIN f USING (k) GROUP BY 1, 2)
        SELECT cohort, age, n FROM cells WHERE cohort >= DATE '2023-01-01' AND cohort < DATE '2023-04-01'
          AND age <= 6""").fetchall()
    return {(str(c), a): n for c, a, n in rows}


def test_retention_by_month_of_first_order(retail):
    cells = _cells(retail)
    got = _ask(retail, {"cohort": {"of": "customer", "measure": "number_of_orders", "periods": 6},
                        "time": {"window": Q1_2023, "grain": "month"}})
    shown = {(str(r["period"])[:10], int(r["periods_after"])): float(r["retained"]) for r in got["export_rows"]}
    cohorts = {c for c, _ in cells}
    assert {c for c, _ in shown} == cohorts and len(cohorts) == 3
    for (cohort, age), share in shown.items():
        size = cells[(cohort, 0)]
        assert share == pytest.approx(100.0 * cells.get((cohort, age), 0) / size), (cohort, age)
    assert all(shown[(c, 0)] == pytest.approx(100.0) for c in cohorts)
    assert set(a for _, a in shown) == set(range(7))
    assert "by the month of their first order ever" in got["definition"]


def test_counts_instead_of_shares(retail):
    cells = _cells(retail)
    got = _ask(retail, {"cohort": {"of": "customer", "measure": "number_of_orders", "periods": 6, "value": "count"},
                        "time": {"window": Q1_2023, "grain": "month"}})
    shown = {(str(r["period"])[:10], int(r["periods_after"])): float(r["members"]) for r in got["export_rows"]}
    assert shown == {k: float(cells.get(k, 0)) for k in shown}


def test_a_grouping_is_asked_as_a_filter(retail):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "cohort": {"of": "customer", "measure": "number_of_orders"},
                                "group_by": ["customer.segment"], "time": {"window": Q1_2023, "grain": "month"}})
    with pytest.raises(ResolveError, match="one group of members at a time"):
        resolve(plan, model, Context(today=TODAY))


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle"])
def test_the_sql_reads_on_every_warehouse(retail, dialect):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "cohort": {"of": "customer", "measure": "number_of_orders"},
                                "time": {"window": Q1_2023, "grain": "quarter"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "duckdb"])
def test_the_sql_passes_the_governed_executors_shape_rules(retail, dialect):
    """The profitability stress test: on a governed workspace the grid's CROSS JOIN was refused as a cartesian
    product the question did not ask for. The cohorts and the periods after their first are joined on the one
    condition that keeps them: periods that have happened."""
    from core.validator import _production_shape_errors

    built, model = retail
    plan = Plan.model_validate({"kind": "query", "cohort": {"of": "customer", "measure": "number_of_orders"},
                                "time": {"window": Q1_2023, "grain": "quarter"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    tree = sqlglot.parse_one(compiled.sql, read=dialect)
    assert _production_shape_errors(tree, {"production_sql": True}) == []
    assert " CROSS JOIN " not in compiled.sql.upper()
