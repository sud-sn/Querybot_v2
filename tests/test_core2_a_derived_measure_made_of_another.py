"""A derived measure made of an earlier one: a gross margin % of a gross profit worked out first.

Asked of the AdventureWorks sample with a real model (gpt-4o), "Which product category has the highest gross
margin percentage?" was planned as sales over cost, x 100, and answered 267%: a derived measure could only name
two of the model's measures, and a margin % needs the sales less the cost first. A part of a derived measure may
now be an earlier one's name; that one is worked out inside it and not shown on its own. Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(retail, plan: dict, question: str) -> dict:
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


def test_a_margin_percent_is_the_profit_over_the_sales(retail):
    plan = {"kind": "query", "intent": "rank", "group_by": ["store.name"],
            "derived": [{"name": "Gross profit", "op": "difference", "measures": ["net_amount", "cost_amount"]},
                        {"name": "Gross margin %", "op": "ratio", "measures": ["Gross profit", "net_amount"],
                         "scale": 100}],
            "sort": [{"by": "Gross margin %", "desc": True}], "limit": 3}
    payload = _ask(retail, plan, "Which stores have the highest gross margin percentage?")
    rows = payload["export_rows"]
    expected = retail[0].con.execute(
        "SELECT s.store_name, 100.0 * SUM(l.net_amount - l.cost_amount) / SUM(l.net_amount) AS m FROM order_lines l "
        "JOIN stores s ON l.store_id = s.store_id GROUP BY 1 ORDER BY m DESC LIMIT 3").fetchall()
    figures = [next(v for k, v in r.items() if isinstance(v, (int, float)) and not k.endswith("code")) for r in rows]
    assert [r["store_name"] for r in rows] == [e[0] for e in expected], rows
    assert figures == pytest.approx([float(e[1]) for e in expected], rel=1e-6)
    assert all(not any("profit" in k for k in r) for r in rows), rows      # worked out inside, not shown
    assert 0 < figures[0] < 100


def test_an_earlier_derived_on_another_table_is_combined_too(retail):
    plan = {"kind": "query", "intent": "value",
            "derived": [{"name": "Net after refunds", "op": "difference", "measures": ["net_amount", "refund_amount"]},
                        {"name": "Kept share", "op": "ratio", "measures": ["Net after refunds", "net_amount"],
                         "scale": 100}]}
    payload = _ask(retail, plan, "What share of net sales did we keep after refunds?")
    con = retail[0].con
    net = float(con.execute("SELECT SUM(net_amount) FROM order_lines").fetchone()[0])
    refunds = float(con.execute("SELECT SUM(refund_amount) FROM returns").fetchone()[0])
    value = next(v for v in payload["export_rows"][0].values() if isinstance(v, (int, float)))
    assert value == pytest.approx(100.0 * (net - refunds) / net, rel=1e-6)


def test_a_derived_figure_named_among_the_measures_is_that_figure(retail):
    """ "What was our gross profit in 2025?" was written with gross_profit among the measures as well as derived,
    and refused twice: "no measure called gross_profit". It is the figure worked out."""
    plan = {"kind": "query", "intent": "value", "measures": ["gross_profit"],
            "derived": [{"name": "gross_profit", "op": "difference", "measures": ["net_amount", "cost_amount"]}]}
    payload = _ask(retail, plan, "What was our gross profit?")
    expected = retail[0].con.execute("SELECT SUM(net_amount) - SUM(cost_amount) FROM order_lines").fetchone()[0]
    assert payload["export_rows"] and next(
        v for v in payload["export_rows"][0].values() if isinstance(v, (int, float))) == pytest.approx(float(expected))


def test_an_earlier_derived_the_reader_also_asked_for_is_shown(retail):
    plan = {"kind": "query", "intent": "value", "measures": ["Gross profit"],
            "derived": [{"name": "Gross profit", "op": "difference", "measures": ["net_amount", "cost_amount"]},
                        {"name": "Gross margin %", "op": "ratio", "measures": ["Gross profit", "net_amount"],
                         "scale": 100}]}
    payload = _ask(retail, plan, "Gross profit and gross margin %?")
    row = payload["export_rows"][0]
    assert any("profit" in k for k in row) and any("margin" in k for k in row), row
