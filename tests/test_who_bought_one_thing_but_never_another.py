"""Who bought one thing in a period but never another: the members, and their figures for the thing they bought.

Found replaying a real model's (gpt-4o) plans for an analyst's request: "which customers bought Dish Soap in 2025 but
have never bought Laundry Pods? give their segment, city and what they spent on dish soap in 2025". The AI wrote
Dish Soap and 2025 as the question's own filter and period, and Laundry Pods as the rows they had none of. So:

* the absence was looked for among Dish Soap rows that were also Laundry Pods (no row is both): every Dish Soap
  buyer was listed;
* the period went to the absence (no Laundry Pods in 2025 only), and what they spent was counted over all time;
* the sales with no customer on file were listed as a customer ("Unknown");
* the sentence said "had none ever" without saying of what.

Now the rows they had (Dish Soap, in 2025) count the figures, the rows they had none of are looked for at any time
when the question says "never" (in the same period when it does not), rows with no member are no member, and the
sentence names both. Each answer is compared with SQL written by hand on the invented retail warehouse.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan import analyst_math
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
BOUGHT, NEVER = "Dish Soap Large", "Olive Oil Small"
IN_2025 = {"date": "order_date", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    con = built.con
    # A sale with no customer on file: no customer who had none.
    (key,), = con.execute("SELECT MIN(date_key) FROM calendar WHERE full_date >= DATE '2025-06-01'").fetchall()
    (product,), = con.execute(f"SELECT product_id FROM products WHERE product_name = '{BOUGHT}'").fetchall()
    con.execute("INSERT INTO order_lines (order_line_id, order_number, line_number, customer_id, product_id, store_id, "
                "order_date_key, quantity, unit_price, gross_amount, discount_amount, net_amount, cost_amount, "
                f"status_code) SELECT MAX(order_line_id) + 1, 'WALKIN-1', 1, NULL, {product}, 1, {key}, 2, 50, 100, 0, "
                "100, 60, 'D' FROM order_lines")
    return built, model


def _ask(retail, question: str, plan: dict) -> dict:
    built, model = retail
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert "export_rows" in payload, payload.get("answer")
    return payload


def _plan() -> dict:
    # As the AI wrote it: the thing bought and its period as the question's own, the other as the absence.
    return {"intent": "list", "measures": ["net_amount"], "group_by": ["customer.segment", "customer.city"],
            "filters": [{"field": "product.name", "op": "eq", "values": [BOUGHT]}], "time": IN_2025,
            "absent": {"of": "customer", "measure": "number_of_order_lines",
                       "without": [{"field": "product.name", "op": "eq", "values": [NEVER]}]}}


def _hand(retail, *, never_in_2025_only: bool) -> list[tuple]:
    in_2025 = "d.full_date >= DATE '2025-01-01' AND d.full_date < DATE '2026-01-01'"
    other = f"AND {in_2025.replace('d.', 'd2.')}" if never_in_2025_only else ""
    return sorted((name, round(float(spent), 2)) for name, spent in retail[0].con.execute(f"""
        SELECT c.customer_name, ROUND(SUM(o.net_amount), 2)
        FROM order_lines o JOIN products p ON p.product_id = o.product_id
             JOIN calendar d ON d.date_key = o.order_date_key JOIN customers c ON c.customer_id = o.customer_id
        WHERE p.product_name = '{BOUGHT}' AND {in_2025}
          AND NOT EXISTS (SELECT 1 FROM order_lines o2 JOIN products p2 ON p2.product_id = o2.product_id
                          JOIN calendar d2 ON d2.date_key = o2.order_date_key
                          WHERE o2.customer_id = o.customer_id AND p2.product_name = '{NEVER}' {other})
        GROUP BY c.customer_id, c.customer_name""").fetchall())


def _rows(payload: dict) -> list[tuple]:
    return sorted((r["customer_name"], round(float(r["net_amount"]), 2)) for r in payload["export_rows"])


def test_bought_in_a_period_but_never_the_other_lists_those_with_what_they_spent_then(retail):
    payload = _ask(retail, f'which customers bought "{BOUGHT}" in 2025 but have never bought "{NEVER}"? give their '
                           f"segment, city and what they spent on it in 2025", _plan())
    expected = _hand(retail, never_in_2025_only=False)
    assert expected and _rows(payload) == expected
    assert all(r["customer_name"] not in (None, "", "Unknown") for r in payload["export_rows"])
    headline = payload["answer"]["headline"]
    assert BOUGHT in headline and NEVER in headline and "at any time" in headline, headline


def test_bought_but_not_the_other_in_the_same_period(retail):
    payload = _ask(retail, f'which customers bought "{BOUGHT}" in 2025 but not "{NEVER}"? with what they spent on it',
                   _plan())
    expected = _hand(retail, never_in_2025_only=True)
    assert expected != _hand(retail, never_in_2025_only=False), "the two questions have different answers here"
    assert _rows(payload) == expected



@pytest.mark.parametrize("asked", [
    f'which customers bought "{BOUGHT}" but never bought "{NEVER}" in 2025? with what they spent',
    # "While" and "though" go on with the never ("never bought it while it was on sale in 2025").
    f'which customers bought "{BOUGHT}" but never bought "{NEVER}" while it was on sale in 2025? with what they spent',
    f'clients qui ont acheté "{BOUGHT}" mais n\'ont jamais acheté de "{NEVER}" qui sont sortis en 2025, avec ce qu\'ils ont '
    "dépensé"])
def test_a_period_written_on_the_never_is_the_nevers_own(retail, asked):
    # "... but never bought Olive Oil Small in 2025": the AI read 2025 for the whole question; "never" leaves it be.
    assert _rows(_ask(retail, asked, _plan())) == _hand(retail, never_in_2025_only=True)


@pytest.mark.parametrize("asked", [
    f'which customers have never bought "{NEVER}" but bought "{BOUGHT}" in 2025? with what they spent on it',
    f'customers who never bought "{NEVER}" and who bought "{BOUGHT}" in 2025, with what they spent on it',
    # The request goes on after the never: its period is the bought one's.
    f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" plus what they spent on it in 2025',
    f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" and the amount they spent on it in 2025',
    f'clients qui n\'ont jamais acheté "{NEVER}" mais ont acheté "{BOUGHT}" en 2025, avec ce qu\'ils ont dépensé',
    f'customers who never bought "{NEVER}" even though they bought "{BOUGHT}" in 2025, with what they spent on it',
    f'customers who never bought "{NEVER}" despite buying "{BOUGHT}" in 2025, with what they spent on it'])
def test_a_period_after_the_never_clause_is_the_bought_ones(retail, asked):
    assert _rows(_ask(retail, asked, _plan())) == _hand(retail, never_in_2025_only=False)

@pytest.mark.parametrize("asked,own", [
    (f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" since Monday', True),
    (f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" since Christmas', True),
    (f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" since the summer', True),
    (f'clients qui ont acheté "{BOUGHT}" en 2025 mais n\'ont jamais acheté "{NEVER}" ce mois-ci', True),
    (f'clients qui ont acheté "{BOUGHT}" en 2025 mais n\'ont jamais acheté "{NEVER}" le mois dernier', True),
    (f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}" since joining', False),
    (f'customers who bought "{BOUGHT}" in 2025 but never bought "{NEVER}"', False)])
def test_the_never_keeps_a_period_it_says_and_looks_at_all_time_otherwise(retail, asked, own):
    # "Never ... since Monday" has a period of its own; "never ... since joining" is any time they were a customer.
    out = analyst_math.bought_but_never(Plan.model_validate({"kind": "query", **_plan()}), retail[1], asked)
    assert (out.time.window.kind != "all") is own, asked


def test_an_absence_with_no_filter_of_its_own_field_is_left_as_written():
    plan = Plan.model_validate({"kind": "query", "intent": "list", "measures": [],
                                "time": {"window": {"kind": "last", "unit": "day", "n": 90}},
                                "absent": {"of": "customer", "measure": "number_of_orders"}})
    assert analyst_math.bought_but_never(plan, None, "customers with no order in the last 90 days") is plan
