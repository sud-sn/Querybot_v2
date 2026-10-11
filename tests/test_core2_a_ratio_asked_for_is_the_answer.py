"""A ratio the question asks for is the answer, under a name in words, and money per unit reads as money.

Found asking real conversations of a real model (gpt-4o):

* "What is our average selling price per unit by item group?" was planned as the amount, the quantity and their
  ratio: the answer led with the amount, showed the ratio last, named "average_selling_price_per_unit", as a plain
  number. The ratio is now the answer ("Average selling price per unit: ... leads with $27.85"); its parts are
  shown only when the question asks for them too;
* "What is the average days supply per fill by pharmacy?" was planned as each fill's days supply averaged, and
  also split by fill: one row per fill. A figure per row of a count is never also split by those rows;
* "... for each?" after a ranking asks the figure for each member on screen;
* "window": null and "per": null were written for "not set", and the plan was refused.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.followup import read_turn
from core2.plan.ir import Plan
from core2.plan.planner import parse, ratio_alone, settle
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
PRICE = {"kind": "query", "intent": "rank", "measures": ["net_amount", "quantity"], "group_by": ["store.name"],
         "derived": [{"name": "average_selling_price_per_unit", "op": "ratio", "measures": ["net_amount", "quantity"]}],
         "sort": [{"by": "net_amount", "desc": True}]}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def test_the_ratio_asked_for_is_the_answer_sorted_by_and_named_in_words(retail):
    plan = ratio_alone(Plan.model_validate(PRICE), retail[1], "What is our average selling price per unit by store?")
    assert plan.measures == []
    assert [d.name for d in plan.derived] == ["Average selling price per unit"]
    assert plan.sort[0].by == "Average selling price per unit"


@pytest.mark.parametrize("question", [
    "Net amount, quantity and price per unit by store",
    "Show net amount and quantity by store",                  # no ratio asked for: as planned
])
def test_parts_asked_for_too_are_kept(retail, question):
    assert ratio_alone(Plan.model_validate(PRICE), retail[1], question).measures == ["net_amount", "quantity"]


def test_money_per_unit_reads_as_money(retail):
    built, model = retail
    replies = [PRICE]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps(replies.pop(0)), index=MemberIndex(), today=TODAY)
    payload = answer_question("What is our average selling price per unit by store?", services, Session())
    rows = payload["export_rows"]
    assert list(rows[0]) == ["store_name", "average_selling_price_per_unit"]
    prices = [r["average_selling_price_per_unit"] for r in rows]
    assert prices == sorted(prices, reverse=True)
    expected = built.con.execute(
        "SELECT SUM(net_amount) / SUM(quantity) FROM order_lines o JOIN stores s ON o.store_id = s.store_id "
        "GROUP BY s.store_id ORDER BY 1 DESC LIMIT 1").fetchone()[0]
    assert prices[0] == pytest.approx(float(expected))
    assert payload["answer"]["headline"].startswith("Average selling price per unit:")
    assert f"${float(expected):,.2f}" in payload["answer"]["headline"]


def test_a_figure_per_row_of_a_count_is_not_also_split_by_those_rows(retail):
    plan = Plan.model_validate({"kind": "query", "measures": ["quantity"], "group_by": ["store.name", "order_line"],
                                "per": {"of": "number_of_order_lines", "summary": "avg"}})
    assert ratio_alone(plan, retail[1], "Average quantity per order line by store").group_by == ["store.name"]
    # Split by a member of another table: kept.
    other = plan.model_copy(update={"group_by": ["store.name", "product.name"]})
    assert ratio_alone(other, retail[1], "Average quantity per order line by store and product").group_by == [
        "store.name", "product.name"]


def test_for_each_asks_the_figure_for_each_member_on_screen(retail):
    model = retail[1]
    before = Plan.model_validate({"kind": "query", "intent": "rank", "measures": ["net_amount"],
                                  "group_by": ["store.name"], "sort": [{"by": "net_amount", "desc": True}],
                                  "limit": 5})
    assert read_turn("What is the average quantity per order line for each?", previous=before,
                     model=model).kind == "refine"
    ai = Plan.model_validate({"kind": "query", "measures": ["quantity"], "follow_up": "refine",
                              "per": {"of": "number_of_order_lines", "summary": "avg"}})
    assert settle(ai, model, "What is the average quantity per order line for each?", before).group_by == [
        "store.name"]


def test_a_part_written_as_null_is_not_set():
    plan = parse(json.dumps({"kind": "query", "measures": ["net_amount"], "time": {"window": None, "grain": None},
                             "per": None, "rolling": None, "chart": None}))
    assert plan.measures == ["net_amount"] and plan.time.window.kind == "all" and plan.per is None
