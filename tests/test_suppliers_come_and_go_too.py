"""Suppliers come and go too: "new suppliers by month", "spend with returning suppliers" are asked of a purchasing
warehouse as "new customers" are of a shop's, equal to SQL written by hand.

Found writing the Phase 5 condition books: a purchasing warehouse had no members whose activity could be followed,
so every question about new, returning or repeat suppliers was refused. Suppliers and vendors are parties for the
analyses of who comes back; they are never a dashboard's "top customers". Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2 import recipe_book
from core2.plan.analyses import activities
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
# Suppliers on the supplier list (an order line naming an unknown supplier is no supplier of ours).
FIRST = ("SELECT supplier_id AS k, MIN(order_date) AS first FROM purchase_order_lines "
         "WHERE supplier_id IN (SELECT supplier_id FROM suppliers) GROUP BY supplier_id")


@pytest.fixture(scope="module")
def purchasing():
    return learn(domains.build("purchasing"), "descriptive")


def _ask(learned, question: str, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def test_suppliers_purchase_orders_are_an_activity(purchasing):
    found = {(a.entity.slug, a.measure.slug) for a in activities(purchasing[1])}
    assert ("supplier", "number_of_po_numbers") in found


def test_new_suppliers_month_by_month(purchasing):
    want = dict(purchasing[0].con.execute(f"""
        WITH f AS ({FIRST}) SELECT date_trunc('month', first)::DATE, COUNT(*) FROM f
        WHERE first < DATE '2024-04-01' GROUP BY 1""").fetchall())
    got = _ask(purchasing, "How many new suppliers did we order from each month in Q1 2024?",
               {"lifecycle": {"of": "supplier", "measure": "number_of_po_numbers", "status": "new"},
                "time": {"grain": "month", "window": {"kind": "between", "start": "2024-01-01", "end": "2024-03-31"}}})
    value = next(k for k in got["export_rows"][0] if k != "period")
    shown = {str(r["period"])[:10]: float(r[value]) for r in got["export_rows"] if float(r[value] or 0)}
    assert shown == pytest.approx({str(k)[:10]: float(v) for k, v in want.items()})


def test_spend_with_returning_suppliers(purchasing):
    built, model = purchasing
    amount = next(m.slug for m in model.measures.values() if m.business_name == "Line amount")
    want = built.con.execute(f"""
        WITH f AS ({FIRST}) SELECT SUM(l.line_amount) FROM purchase_order_lines l JOIN f ON l.supplier_id = f.k
        WHERE l.order_date >= DATE '2024-03-01' AND l.order_date < DATE '2024-04-01'
          AND f.first < DATE '2024-03-01'""").fetchone()[0]
    got = _ask(purchasing, "Line amount with returning suppliers in March 2024",
               {"measures": [amount], "segment": {"of": "supplier", "measure": "number_of_po_numbers",
                                                  "status": "returning"},
                "time": {"window": {"kind": "between", "start": "2024-03-01", "end": "2024-03-31"}}})
    assert float(got["kpi"]["value"]) == pytest.approx(float(want))


def test_suppliers_are_never_a_dashboards_top_customers():
    assert not {"supplier", "vendor"} & set(recipe_book.CUSTOMER)


def test_offered_unasked_only_for_customers_and_the_like(purchasing):
    # A stock movement's supplier is asked of ("new suppliers"), never offered on Learn's page or a dashboard.
    from core2.plan.analyses import activity

    assert activity(purchasing[1]) is None or activity(purchasing[1]).entity.slug != "supplier"
