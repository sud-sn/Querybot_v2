"""New, returning, won back and lapsed customers in each period, and the churn rate, equal to SQL written by hand.

Asked of the invented retail warehouse through the whole answer path (a scripted plan, as the planner writes it).
A customer counts once a period: new when their first order ever was in it, returning when they ordered the period
before too, won back when they ordered before but not the period before, lapsed when they ordered the period
before and not in this one; the churn rate is the lapsed over those who ordered the period before. No period after
the last with any orders is shown (everyone would look lapsed), and each answer says what each word means.

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
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
DAYS = """SELECT DISTINCT o.customer_id AS k, d.full_date AS day FROM order_lines o
          JOIN calendar d ON o.order_date_key = d.date_key
          WHERE o.customer_id IN (SELECT customer_id FROM customers)
            AND d.full_date >= DATE '1901-01-01' AND d.full_date < DATE '2026-01-01'"""


def _periods(unit: str) -> str:
    return f"""SELECT k, p, LAG(p) OVER (PARTITION BY k ORDER BY p) AS prev, LEAD(p) OVER (PARTITION BY k ORDER BY p)
               AS nxt, MIN(p) OVER (PARTITION BY k) AS first FROM (SELECT DISTINCT k, date_trunc('{unit}', day)::DATE
               AS p FROM ({DAYS}))"""


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


def _expected(learned, unit: str, step: str) -> dict:
    seq = _periods(unit)
    rows = learned[0].con.execute(f"""
        WITH s AS ({seq}),
        months AS (SELECT DISTINCT p AS m FROM s WHERE p >= DATE '2025-01-01' AND p < DATE '2026-01-01')
        SELECT m,
          (SELECT COUNT(*) FROM s WHERE p = m),
          (SELECT COUNT(*) FROM s WHERE p = m AND p = first),
          (SELECT COUNT(*) FROM s WHERE p = m AND prev = CAST(m - INTERVAL {step} AS DATE)),
          (SELECT COUNT(*) FROM s WHERE CAST(p + INTERVAL {step} AS DATE) = m AND (nxt IS NULL OR nxt <> m)),
          (SELECT COUNT(*) FROM s WHERE CAST(p + INTERVAL {step} AS DATE) = m)
        FROM months ORDER BY m""").fetchall()
    return {str(m): {"active": a, "new": n, "returning": r, "won_back": a - n - r, "lapsed": lp, "before": b}
            for m, a, n, r, lp, b in rows}


def test_customers_new_returning_won_back_and_lapsed_by_month(retail):
    expected = _expected(retail, "month", "1 MONTH")
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders"},
                        "time": {"window": Y2025, "grain": "month"}})
    shown = {str(r["period"])[:10]: r for r in got["export_rows"]}
    assert set(shown) == set(expected) and len(expected) == 12
    for month, want in expected.items():
        row = shown[month]
        assert (row["active"], row["new"], row["returning"], row["won_back"], row["lapsed"]) == (
            want["active"], want["new"], want["returning"], want["won_back"], want["lapsed"]), month
    assert "new: their first order ever was in the month" in got["definition"]
    assert "lapsed: orders the month before, none in this one" in got["definition"]


def test_members_counted_by_their_own_list_are_followed_by_what_they_did(retail):
    """"How many new customers did we gain each month?" was planned with the number of customers: dated by a date
    on the customer list, it found none (the profitability stress test). A member's activity is the rows of what
    it did: the same answer as by its orders."""
    _, model = retail
    own = next(m.slug for m in model.measures.values()
               if m.table == model.entities["customer"].table and getattr(m.expr, "agg", "") in ("count", "count_distinct"))
    by_orders = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "new"},
                              "time": {"window": Y2025, "grain": "month"}})
    by_list = _ask(retail, {"lifecycle": {"of": "customer", "measure": own, "status": "new"},
                            "time": {"window": Y2025, "grain": "month"}})
    assert by_list["export_rows"] == by_orders["export_rows"] and by_orders["export_rows"]


def test_the_churn_rate_by_quarter(retail):
    expected = _expected(retail, "quarter", "3 MONTH")
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "churn_rate"},
                        "time": {"window": Y2025, "grain": "quarter"}})
    shown = {str(r["period"])[:10]: float(r["churn_rate"]) for r in got["export_rows"]}
    assert shown == pytest.approx({q: 100.0 * w["lapsed"] / w["before"] for q, w in expected.items()})
    assert "churn rate: the lapsed as a share of the customers with orders the quarter before" in got["definition"]


def test_no_period_after_the_last_with_activity(retail):
    built, model = retail
    (last,), = built.con.execute("SELECT MAX(date_trunc('month', d.full_date))::DATE FROM order_lines o JOIN "
                                 "calendar d ON o.order_date_key = d.date_key WHERE d.full_date < DATE '9000-01-01'"
                                 ).fetchall()
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "lapsed"},
                        "time": {"window": {"kind": "between", "start": str(last.replace(day=1)),
                                            "end": "2027-12-31"}, "grain": "month"}})
    assert [str(r["period"])[:10] for r in got["export_rows"]] == [str(last)]


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
@pytest.mark.parametrize("status", ["all", "churn_rate"])
def test_the_sql_reads_on_every_warehouse(retail, dialect, status):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "lifecycle": {"of": "customer", "measure": "number_of_orders",
                                                               "status": status},
                                "time": {"window": Y2025, "grain": "quarter"}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)
    assert "UNION ALL" in compiled.sql.upper()
