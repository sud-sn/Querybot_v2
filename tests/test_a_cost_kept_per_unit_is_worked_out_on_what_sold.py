"""A cost or a price kept once per product is worth something only times the quantity on each line that sold it.

Found replaying a real model's (gpt-4o) plans for an analyst's report: "For 2024, give me revenue, cost (at each
product's standard cost), margin in $ and margin % for every product category" was planned with a standard cost
added up (or averaged) on its own: $813 of cost for every bike sold, and a margin of 99.99%. The cost at each
product's standard cost is each line's quantity times its product's standard cost, added up; the margin is made
of that. The follow-up "drop the cost column and add the units sold" kept the margin by name but was planned
with the margin made of the cost the AI first named: a figure worked out above and named again keeps its parts.

Asked of the invented retail warehouse (a product's list price) through the whole answer path, the AI stood in
by the plan it wrote; every figure is compared with SQL written by hand.

Synthetic data only.
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
IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
GAP = {"name": "Discount from list", "op": "difference", "measures": ["product.list_price", "net_amount"]}
GAP_PCT = {"name": "Discount %", "op": "ratio", "measures": ["Discount from list", "product.list_price"],
           "scale": 100.0}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _services(retail, plans: list[dict]) -> Services:
    built, model = retail
    answers = [json.dumps({"kind": "query", **p}) for p in plans]
    # Each turn's plan, as the AI wrote it; asked again on the same turn, the AI writes the same.
    return Services(model=model, warehouse=DuckDBWarehouse(built.con),
                    complete=lambda s, t: answers.pop(0) if len(answers) > 1 else answers[0],
                    index=MemberIndex(), today=TODAY)


def _hand(retail) -> dict[str, tuple[float, float]]:
    rows = retail[0].con.execute("""
        SELECT c.category_name, SUM(o.net_amount), SUM(o.quantity * p.list_price)
        FROM order_lines o JOIN products p ON p.product_id = o.product_id
             JOIN categories c ON c.category_id = p.category_id JOIN calendar d ON d.date_key = o.order_date_key
        WHERE d.full_date >= DATE '2025-01-01' AND d.full_date < DATE '2026-01-01'
        GROUP BY 1""").fetchall()
    return {name: (float(net), float(listed)) for name, net, listed in rows}


def _by_category(payload: dict) -> dict[str, dict]:
    return {next(v for k, v in r.items() if "categor" in k): r for r in payload["export_rows"]}


def _first(retail, session: Session) -> dict:
    plan = {"intent": "breakdown", "measures": ["net_amount", "product.list_price", "Discount from list"],
            "derived": [GAP, GAP_PCT], "group_by": ["category.name"], "time": IN_2025}
    return answer_question("For 2025, give me net sales, the value at each product's list price, the discount from "
                           "list in $ and as a % for every category", _services(retail, [plan]), session)


def test_the_value_at_each_products_list_price_is_its_quantity_times_that_price(retail):
    payload = _first(retail, Session())
    expected = _hand(retail)
    rows = _by_category(payload)
    assert set(expected) <= set(rows) <= {*expected, "Unknown"}
    for name, (net, listed) in expected.items():
        row = rows[name]
        value = next(v for k, v in row.items() if "list_price" in k and "discount" not in k)
        assert float(value) == pytest.approx(listed, rel=1e-6), name
        gap = next(v for k, v in row.items() if k.startswith("discount_from"))
        assert float(gap) == pytest.approx(listed - net, rel=1e-6), name
        pct = row["discount"]                     # "Discount %"
        assert float(pct) == pytest.approx(100 * (listed - net) / listed, rel=1e-6), name
    # Some sales lines have no product on file: they have no list price, and the answer says so.
    assert any("with no product on file" in n for n in payload["trust"]["date_context"]), payload["trust"]


def test_a_follow_up_that_drops_the_column_keeps_the_figures_made_of_it(retail):
    session = Session()
    services_plans = [
        {"intent": "breakdown", "measures": ["net_amount", "product.list_price", "Discount from list"],
         "derived": [GAP, GAP_PCT], "group_by": ["category.name"], "time": IN_2025},
        # As the AI wrote the follow-up: the gap kept by name, made of the list price it first named.
        {"intent": "breakdown", "measures": ["net_amount", "quantity", "Discount from list"],
         "derived": [GAP, GAP_PCT], "group_by": ["category.name"], "time": IN_2025, "follow_up": "refine"}]
    services = _services(retail, services_plans)
    answer_question("For 2025, give me net sales, the value at each product's list price, the discount from list in "
                    "$ and as a % for every category", services, session)
    payload = answer_question("drop the list price column and add the units sold", services, session)
    assert all("list_price" not in k for k in payload["export_rows"][0]), "the column taken off"
    # Units sold are kept apart by their unit of measure: each category's rows, added up.
    gaps: dict[str, float] = {}
    for row in payload["export_rows"]:
        name = row["category_name"]
        gaps[name] = gaps.get(name, 0.0) + float(row["discount_from_list"])
    for name, (net, listed) in _hand(retail).items():
        assert gaps[name] == pytest.approx(listed - net, rel=1e-6), name




def test_a_figure_the_reader_makes_of_something_else_is_theirs(retail):
    # "Work it out on the gross amount instead": the discount is made of the gross amount now, not the list price.
    session = Session()
    gap = {"name": "Discount from list", "op": "difference", "measures": ["gross_amount", "net_amount"]}
    gap_pct = {"name": "Discount %", "op": "ratio", "measures": ["Discount from list", "gross_amount"], "scale": 100.0}
    services = _services(retail, [
        {"intent": "breakdown", "measures": ["net_amount", "product.list_price", "Discount from list"],
         "derived": [GAP, GAP_PCT], "group_by": ["category.name"], "time": IN_2025},
        {"intent": "breakdown", "measures": ["net_amount", "gross_amount", "Discount from list"],
         "derived": [gap, gap_pct], "group_by": ["category.name"], "time": IN_2025, "follow_up": "refine"}])
    answer_question("For 2025, give me net sales, the value at each product's list price, the discount from list in "
                    "$ and as a % for every category", services, session)
    payload = answer_question("work the discount from list out on the gross amount instead, not the list price",
                              services, session)
    hand = {name: (float(gross), float(net)) for name, gross, net in retail[0].con.execute("""
        SELECT c.category_name, SUM(o.gross_amount), SUM(o.net_amount)
        FROM order_lines o JOIN products p ON p.product_id = o.product_id
             JOIN categories c ON c.category_id = p.category_id JOIN calendar d ON d.date_key = o.order_date_key
        WHERE d.full_date >= DATE '2025-01-01' AND d.full_date < DATE '2026-01-01'
        GROUP BY 1""").fetchall()}
    rows = _by_category(payload)
    for name, (gross, net) in hand.items():
        assert float(rows[name]["discount_from_list"]) == pytest.approx(gross - net, rel=1e-6), name

def test_a_cost_each_line_keeps_as_booked_is_left_as_the_ai_planned_it():
    # Here each sale also keeps its own cost at standard, at the standard cost of the day it was booked: that is the
    # figure asked for, never one worked out again from the product's standard cost today.
    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.plan import analyst_math
    from core2.plan.ir import Plan

    built, _ = learn(domains.build("retail"), "descriptive")
    con = built.con
    con.execute("ALTER TABLE products ADD COLUMN standard_cost DECIMAL(12,2)")
    con.execute("UPDATE products SET standard_cost = ROUND(list_price * 0.45, 2)")
    con.execute("ALTER TABLE order_lines ADD COLUMN standard_cost_amount DECIMAL(14,2)")
    con.execute("UPDATE order_lines o SET standard_cost_amount = ROUND(o.quantity * p.standard_cost * 0.9, 2) "
                "FROM products p WHERE p.product_id = o.product_id")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                        client_id="eval-retail-std", options=BuildOptions(workers=1))
    question = "For 2025, give me net sales and cost (at each product's standard cost) for every category"

    def plan(cost: str) -> Plan:
        return Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount", cost],
                                    "group_by": ["category.name"], "time": IN_2025})

    booked = plan("standard_cost_amount")
    assert analyst_math.valued_at_unit_cost(booked, model, question) is booked
    # The product's own standard cost, added up as it is, is still each line's quantity times it.
    kept = analyst_math.valued_at_unit_cost(plan("product.standard_cost"), model, question)
    assert [f.formula for f in kept.figures] == ["SUM([Order line · Quantity] * [Product · Standard cost])"], kept


def test_a_stock_valued_at_standard_cost_is_money_not_the_quantitys_unit():
    # "Closing stock value at standard cost by warehouse": the quantity is kept apart by its unit, the value is money
    # ("leads with 685,165 EA" it read).
    built, model = learn(domains.build("inventory"), "descriptive")
    plan = {"intent": "breakdown", "measures": ["closing_quantity", "item.standard_cost"],
            "group_by": ["warehouse.name"], "time": IN_2025}
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0) if len(answers) > 1 else answers[0],
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("For 2025, closing quantity and closing stock value at standard cost by warehouse",
                              services, Session())
    assert [f["format"] for f in payload["figures"]] == ["currency"], payload["figures"]
    headline = payload["answer"]["headline"]
    assert "leads with $" in headline, headline



def test_a_follow_up_naming_the_cost_by_its_own_words_keeps_the_figure_above():
    # "Drop the cost column and add the units sold" after a margin made of the cost at standard cost: the AI wrote
    # the margin of the cost measure it first named ("standard cost"), which the figure stands for.
    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.plan.ir import Plan
    from core2.plan.thread import derived_as_above

    built, _ = learn(domains.build("retail"), "descriptive")
    con = built.con
    con.execute("ALTER TABLE products ADD COLUMN standard_cost DECIMAL(12,2)")
    con.execute("UPDATE products SET standard_cost = ROUND(list_price * 0.45, 2)")
    con.execute("ALTER TABLE order_lines ADD COLUMN standard_cost_amount DECIMAL(14,2)")
    con.execute("UPDATE order_lines SET standard_cost_amount = cost_amount")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                        client_id="eval-retail-std2", options=BuildOptions(workers=1))
    figure = {"name": "Cost at standard cost", "formula": "SUM([Order line · Quantity] * [Product · Standard cost])",
              "format": "currency"}
    margin = {"name": "Margin $", "op": "difference", "measures": ["net_amount", "Cost at standard cost"]}
    above = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                                 "figures": [figure], "derived": [margin], "group_by": ["category.name"]})

    def again(cost: str) -> Plan:
        return Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount", "quantity"],
                                    "group_by": ["category.name"], "follow_up": "refine",
                                    "derived": [{**margin, "measures": ["net_amount", cost]}]})

    assert derived_as_above(again("standard_cost_amount"), above, model).derived[0].measures == \
        ["net_amount", "Cost at standard cost"]
    assert derived_as_above(again("cost_amount"), above, model).derived[0].measures == ["net_amount", "cost_amount"], \
        "a cost that is not what the figure stands for stays the AI's"
    # However the AI spells it, a part by the figure's own words is the figure.
    for spelled in ("standard cost", "Standard_cost", "product.standard cost"):
        assert derived_as_above(again(spelled), above, model).derived[0].measures[1] == "Cost at standard cost", spelled
    # The reader names the part: theirs ("use the standard cost amount booked on each line instead").
    asked = "use the standard cost amount booked on each line instead"
    assert derived_as_above(again("standard_cost_amount"), above, model, asked).derived[0].measures[1] == \
        "standard_cost_amount"
    for chosen in ("switch the margin to the standard cost amount", "replace the cost with the standard cost amount",
                   "margin should be net sales minus the standard cost amount"):
        assert derived_as_above(again("standard_cost_amount"), above, model, chosen).derived[0].measures[1] == \
            "standard_cost_amount", chosen
    # Named only to be shown beside it: still the figure ("same for 2024 with the standard cost amount shown too").
    for shown in ("same for 2024 with the standard cost amount shown too",
                  "same for orders from 2024 with the standard cost amount shown too", "add the standard cost amount"):
        assert derived_as_above(again("standard_cost_amount"), above, model, shown).derived[0].measures[1] == \
            "Cost at standard cost", shown
    dropped = "drop the standard cost amount column and add the units sold"
    assert derived_as_above(again("standard_cost_amount"), above, model, dropped).derived[0].measures[1] == \
        "Cost at standard cost", "named only to be taken off"


def test_a_cost_column_with_no_profile_is_worked_out_all_the_same():
    built, model = learn(domains.build("inventory"), "descriptive")
    key = next(k for k, c in model.columns.items() if c.name == "standard_cost" and "item" in c.table)
    model.columns[key] = model.columns[key].model_copy(update={"profile": None})
    plan = {"intent": "breakdown", "measures": ["closing_quantity", "item.standard_cost"],
            "group_by": ["warehouse.name"], "time": IN_2025}
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: answers.pop(0) if len(answers) > 1 else answers[0],
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("For 2025, closing quantity and closing stock value at standard cost by warehouse",
                              services, Session())
    assert payload.get("export_rows"), payload.get("answer")
