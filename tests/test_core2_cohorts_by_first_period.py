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


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
def test_the_sql_reads_on_every_warehouse(retail, dialect):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "cohort": {"of": "customer", "measure": "number_of_orders"},
                                "time": {"window": Q1_2023, "grain": "quarter"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres", "duckdb"])
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


def _pct(value: float) -> str:
    """A share as the answer writes it: whole above 10%, never rounded up to a full 100%."""
    if round(value) == 100 and value != 100:
        return f"{value:.1f}%"
    return f"{value:.0f}%" if value >= 10 else f"{value:.1f}%"


def test_the_headline_says_what_the_average_cohort_kept(retail):
    # Not "0 led in every month ...": the months after the first are no members to lead. What the grid is read for,
    # worked out by hand: the share of each cohort with orders a month later and at the last month shown, averaged.
    cells = _cells(retail)
    got = _ask(retail, {"cohort": {"of": "customer", "measure": "number_of_orders", "periods": 6},
                        "time": {"window": Q1_2023, "grain": "month"}})
    cohorts = sorted({c for c, _ in cells})

    def average(age: int) -> float:
        return sum(100.0 * cells.get((c, age), 0) / cells[(c, 0)] for c in cohorts) / len(cohorts)
    assert got["answer"]["headline"] == (
        f"On average {_pct(average(1))} of customers had orders again 1 month after their first order, and "
        f"{_pct(average(6))} 6 months after, across the 3 monthly cohorts from Jan 2023 to Mar 2023.")
    assert "led in" not in json.dumps(got)          # nor in the findings under it


def test_counted_cohorts_say_how_many_came_and_how_many_came_back(retail):
    cells = _cells(retail)
    got = _ask(retail, {"cohort": {"of": "customer", "measure": "number_of_orders", "periods": 6, "value": "count"},
                        "time": {"window": Q1_2023, "grain": "month"}})
    cohorts = sorted({c for c, _ in cells})
    came = sum(cells[(c, 0)] for c in cohorts)

    def back(age: int) -> float:
        return 100.0 * sum(cells.get((c, age), 0) for c in cohorts) / came
    assert got["answer"]["headline"] == (
        f"{came:,} customers had their first order from Jan 2023 to Mar 2023; {_pct(back(1))} of them had orders "
        f"again 1 month after it, and {_pct(back(6))} 6 months after.")


def test_the_grid_is_drawn_as_a_grid(retail):
    # A cohort by the months after it is a grid, a cell for each, darker for more: never a line per month after
    # the first drawn over the cohorts, all of them at the top of the chart.
    cells = _cells(retail)
    got = _ask(retail, {"cohort": {"of": "customer", "measure": "number_of_orders", "periods": 6},
                        "time": {"window": Q1_2023, "grain": "month"}})
    chart = got["chart"]
    assert chart["chart_type"] == "heatmap" and chart["allowed_types"] == ["heatmap"]
    assert chart["y_keys"] == [f"Month {k}" for k in range(7)]
    assert [r[chart["x_key"]] for r in chart["rows"]] == ["2023-01", "2023-02", "2023-03"]
    first = chart["rows"][0]
    size = cells[("2023-01-01", 0)]
    assert first["Month 2"] == pytest.approx(100.0 * cells.get(("2023-01-01", 2), 0) / size)
    assert chart["grouped_measure"] == "retained" and chart["column_formats"]["Month 2"] == chart["column_formats"]["retained"]
    assert not chart["chart_warnings"]


def _grid(cohorts: int, ages: int, value: str = "share") -> dict:
    """The grid drawn for invented cohorts, a month apart from Jan 2024, each with the months after it it has had."""
    from types import SimpleNamespace as Ns

    from core2.answer.builder import _cohort_grid

    firsts = [dt.date(2024 + i // 12, i % 12 + 1, 1) for i in range(cohorts)]
    records = [{"period": d.isoformat(), "periods_after": str(a), "retained": 100.0 - 3 * a}
               for i, d in enumerate(firsts) for a in range(min(ages, cohorts - i))]
    first = Ns(name="period", role="period", grain="month", ordinal=False)
    age = Ns(name="periods_after", role="attribute", grain=None, ordinal=True)
    logical = Ns(shape=Ns(value=value), fiscal_start=1, conditions=[], intent="breakdown")
    return _cohort_grid(logical, first, age, Ns(name="retained"), records, {"retained": "percentage"},
                        {"period": "First month", "periods_after": "Months after the first",
                         "retained": "Share of customers still with orders"})


def test_a_long_run_of_cohorts_shows_the_latest_and_says_so():
    chart = _grid(24, 4)
    assert [r["period"] for r in chart["rows"]] == [f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(9, 24)]
    assert chart["y_keys"] == [f"Month {k}" for k in range(4)]
    assert chart["chart_warnings"] == ["Showing the latest 15 of 24 first months: every one is in the table."]
    # A month the cohort has not had yet is an empty cell, not a nought.
    assert chart["rows"][-1] == {"period": "2025-12", "Month 0": 100.0, "Month 1": None, "Month 2": None,
                                 "Month 3": None}


def test_a_shares_month_0_gives_way_when_the_months_do_not_fit():
    # Month 0 is all of each cohort by definition: past twelve columns it is the one left out, unsaid.
    assert _grid(14, 13)["y_keys"] == [f"Month {k}" for k in range(1, 13)]
    assert not _grid(14, 13)["chart_warnings"]
    # Counted cohorts keep it (it is each cohort's size), and say which months are drawn.
    counted = _grid(14, 13, value="count")
    assert counted["y_keys"] == [f"Month {k}" for k in range(12)]
    assert counted["chart_warnings"] == ["Showing months 0 to 11 after the first: every one is in the table."]
