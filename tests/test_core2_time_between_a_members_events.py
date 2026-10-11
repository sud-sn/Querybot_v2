"""Time between one member's events, each equal to a query written by hand.

"Average days between orders per customer", "the median gap by month", "days from a customer's first order to
their second", "the share of orders followed by another within 30 days": asked of the invented retail warehouse
through the whole answer path (a scripted plan, as the planner writes it), compared with SQL written here against
the raw tables. Each customer's days with an order count once each (an order of several lines is one event), a
gap is from each order back to the one before (which may be before the window), and each answer says so.

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
Q1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}
DAYS = """SELECT DISTINCT o.customer_id AS k, d.full_date AS day FROM order_lines o
          JOIN calendar d ON o.order_date_key = d.date_key
          WHERE o.customer_id IN (SELECT customer_id FROM customers)
            AND d.full_date >= DATE '1901-01-01' AND d.full_date < DATE '9000-01-01'"""
SEQ = f"""SELECT k, day, LAG(day) OVER (PARTITION BY k ORDER BY day) AS prev,
                 LEAD(day) OVER (PARTITION BY k ORDER BY day) AS nxt,
                 ROW_NUMBER() OVER (PARTITION BY k ORDER BY day) AS n FROM ({DAYS})"""
IN_Q1 = "day >= DATE '2026-01-01' AND day < DATE '2026-04-01'"


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


def _sql(learned, sql: str) -> list[tuple]:
    return learned[0].con.execute(sql).fetchall()


def _only(payload: dict) -> float:
    rows = payload["export_rows"]
    assert len(rows) == 1
    return float(next(iter(rows[0].values())))


def test_average_and_median_days_between_a_customers_orders(retail):
    (average, gaps), = _sql(retail, f"SELECT AVG(date_diff('day', prev, day)), COUNT(*) FROM ({SEQ}) "
                                    f"WHERE prev IS NOT NULL AND {IN_Q1}")
    (median,), = _sql(retail, f"SELECT quantile_cont(date_diff('day', prev, day), 0.5) FROM ({SEQ}) "
                              f"WHERE prev IS NOT NULL AND {IN_Q1}")
    got = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders"}, "time": Q1})
    assert _only(got) == pytest.approx(float(average))
    assert got["answer"]["headline"].startswith("Average days between orders in Q1 2026")
    assert f"(the {gaps:,} gaps)" in got["definition"] and "a first order has no gap" in got["definition"]
    mid = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders",
                                                "summary": "median"}, "time": Q1})
    assert _only(mid) == pytest.approx(float(median))


def test_the_gap_by_month_and_by_the_customers_own_segment(retail):
    by_month = dict(_sql(retail, f"SELECT date_trunc('month', day)::DATE, AVG(date_diff('day', prev, day)) "
                                 f"FROM ({SEQ}) WHERE prev IS NOT NULL AND {IN_Q1} GROUP BY 1"))
    got = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders"},
                        "time": {**Q1, "grain": "month"}})
    shown = {str(r["period"])[:10]: float(r["days"]) for r in got["export_rows"]}
    assert shown == pytest.approx({str(k): float(v) for k, v in by_month.items()})
    by_segment = dict(_sql(retail, f"SELECT c.segment, AVG(date_diff('day', prev, day)) FROM ({SEQ}) s "
                                   f"JOIN customers c ON s.k = c.customer_id WHERE prev IS NOT NULL AND {IN_Q1} "
                                   "GROUP BY 1"))
    seg = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders"},
                        "group_by": ["customer.segment"], "time": Q1})
    assert {r["customer_segment"]: float(r["days"]) for r in seg["export_rows"]} == pytest.approx(
        {k: float(v) for k, v in by_segment.items()})


def test_days_from_the_first_order_to_the_second(retail):
    firsts = f"""SELECT k, MIN(CASE WHEN n = 1 THEN day END) AS f, MIN(CASE WHEN n = 2 THEN day END) AS s
                 FROM ({SEQ}) GROUP BY k"""
    (average, counted, reached), = _sql(retail, f"SELECT AVG(date_diff('day', f, s)), COUNT(*), COUNT(s) "
                                                f"FROM ({firsts}) WHERE f >= DATE '2023-01-01' AND "
                                                "f < DATE '2024-01-01'")
    assert counted > 100
    got = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders", "nth": 2},
                        "time": {"window": {"kind": "between", "start": "2023-01-01", "end": "2023-12-31"}}})
    assert _only(got) == pytest.approx(float(average))
    assert f"{reached:,} of the {counted:,} customers have one" in got["definition"]


def test_the_share_of_orders_followed_by_another_within_30_days(retail):
    (share,), = _sql(retail, f"SELECT AVG(CASE WHEN nxt IS NOT NULL AND date_diff('day', day, nxt) <= 30 "
                             f"THEN 100.0 ELSE 0 END) FROM ({SEQ}) WHERE {IN_Q1}")
    got = _ask(retail, {"measures": [], "gap": {"of": "customer", "measure": "number_of_orders",
                                                "within_days": 30}, "time": Q1})
    assert _only(got) == pytest.approx(float(share))
    assert got["answer"]["headline"].startswith("Orders followed by another within 30 days")


def test_a_grouping_of_the_rows_would_cut_a_sequence_and_is_refused(retail):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "gap": {"of": "customer", "measure": "number_of_orders"},
                                "group_by": ["product.name"], "time": Q1})
    with pytest.raises(ResolveError, match="one sequence"):
        resolve(plan, model, Context(today=TODAY))


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle"])
@pytest.mark.parametrize("gap", [{}, {"summary": "median"}, {"nth": 2}, {"within_days": 30}])
def test_the_sql_reads_on_every_warehouse(retail, dialect, gap):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "gap": {"of": "customer", "measure": "number_of_orders", **gap},
                                "time": {**Q1, "grain": "month"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)
    assert "LAG(" in compiled.sql.upper() and "ROW_NUMBER()" in compiled.sql.upper()
