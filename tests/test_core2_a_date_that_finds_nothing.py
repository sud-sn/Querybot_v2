"""A date that finds nothing, and a snapshot counted by its own date.

Found asking real conversations of a real model (gpt-4o):

* 'Shipments with delivery status "Delivery Exception" in 2025' kept the delivered date of the question before it
  ("How many shipments were delivered in 2026?") and said "No rows match": a shipment never delivered has no
  delivered date. Nothing by a date the question did not name, the rows having none, is answered by their
  table's main date, and says so; a question that names that date ("delivered in 2026") keeps its answer;
* "How many warehouses and items are in each snapshot?" was counted by the item list's own date and refused
  ("Warehouse has no date"): a question about snapshots takes the snapshot table's date, and the warehouses of a
  snapshot are those its rows name, counted once each.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.planner import snapshot_date
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2026 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}}


def _ask(learned, replies: list, *questions: str) -> list[dict]:
    built, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    session = Session()
    return [answer_question(q, services, session) for q in questions]


@pytest.fixture(scope="module")
def retail():
    """The retail warehouse with its open orders never delivered: no delivered date."""
    built, model = learn(domains.build("retail"), "descriptive")
    built.con.execute("UPDATE order_lines SET delivered_date_key = NULL WHERE status_code = 'O'")
    return built, model


def _open_lines(when: str) -> dict:
    return {"kind": "query", "intent": "value", "measures": ["number_of_order_lines"], "follow_up": "refine",
            "filters": [{"field": "order_line.status_code", "op": "eq", "values": ["O"]}],
            "time": {**Y2026, "date": when}}


def test_nothing_by_a_date_the_rows_lack_is_answered_by_their_main_date(retail):
    built, model = retail
    delivered = {"kind": "query", "intent": "value", "measures": ["number_of_order_lines"],
                 "time": {**Y2026, "date": "delivered_date"}}
    first, payload = _ask(retail, [delivered, _open_lines("delivered_date")],
                          "How many order lines were delivered in 2026?", 'Order lines with status "O" in 2026')
    expected = built.con.execute(
        "SELECT COUNT(*) FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key WHERE o.status_code = 'O' "
        "AND d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-12-31'").fetchone()[0]
    assert expected and payload["export_rows"][0]["number_of_order_lines"] == expected
    assert "No rows match" not in payload["answer"]["headline"]
    said = " ".join(payload["trust"]["date_context"])
    assert "None of these rows has a delivered date: shown by order date" in said
    assert payload["plan"]["time"]["date"] == "order_date"


def test_a_question_that_names_the_date_keeps_its_answer(retail):
    payload = _ask(retail, [_open_lines("delivered_date")], 'Order lines with status "O" delivered in 2026')[0]
    assert "No rows match" in payload["answer"]["headline"] or not payload["export_rows"][0]["number_of_order_lines"]


def test_rows_that_have_the_date_only_not_then_keep_their_answer(retail):
    """Shipped order lines by ship date in 2025: they have ship dates, all in 2026. None then is the answer."""
    plan = {**_open_lines("ship_date"), "filters": [{"field": "order_line.status_code", "op": "eq", "values": ["S"]}],
            "time": {"date": "ship_date", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    payload = _ask(retail, [plan], 'Order lines with status "S" in 2025')[0]
    assert not payload["export_rows"] or not payload["export_rows"][0].get("number_of_order_lines")
    assert payload["plan"]["time"]["date"] == "ship_date"


@pytest.fixture(scope="module")
def inventory():
    return learn(domains.build("inventory"), "descriptive")


def test_each_snapshot_takes_the_snapshot_tables_date(inventory):
    model = inventory[1]
    ai = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["number_of_warehouses"],
                              "time": {"grain": "day", "date": "posted_date"}})
    assert snapshot_date(ai, model, "How many warehouses are in each snapshot?").time.date == "balance_date"
    monthly = ai.model_copy(update={"time": ai.time.model_copy(update={"grain": "month"})})
    assert snapshot_date(monthly, model, "How many warehouses are in each monthly snapshot?").time.date == "period"
    assert snapshot_date(ai, model, "How many warehouses posted movements each day?") is ai


def test_the_warehouses_of_each_snapshot_are_those_its_rows_name(inventory):
    built, model = inventory
    plan = {"kind": "query", "intent": "trend", "measures": ["number_of_warehouses"],
            "time": {"grain": "day", "date": "posted_date"}}
    payload = _ask(inventory, [plan], "How many warehouses are in each snapshot?")[0]
    rows = payload["export_rows"]
    expected = dict(built.con.execute("SELECT CAST(balance_date AS DATE), COUNT(DISTINCT warehouse_id) "
                                      "FROM daily_balances GROUP BY 1").fetchall())
    got = {dt.date.fromisoformat(str(next(v for k, v in r.items() if k.startswith("period")))[:10]):
           r["number_of_warehouses"] for r in rows}
    assert got and all(expected[day] == n for day, n in got.items())
    assert payload["answer"]["headline"].count("with a daily balance") <= 1
