"""What business users asked in the second round of the stress test, answered.

* "Net sales of Lakeshore": the store is "Lakeshore Store". A name made of some of a member's own words names it when
  no other member has them ("Apples" names neither "Apples Large" nor "Apples Small"), and a name the planner put
  on the wrong grouping (the region) is read on the one grouping that has it (the store).
* "How many new customers did we add in 2023?": the customers created in 2023, by their own date -- not those whose
  first order fell then (every customer is "new" in the data's first month).
* "Did we hit our target for the quarter?" after a Q1 answer: Q1, not the quarter we are in.
* "Which stores missed target?": each store whose figure is below its target, said as the difference below 0;
  the headline says none reached it and which came closest, never "furthest above it".
* "Gross margin %" worked out from "Gross margin" is shown: the sign keeps the two names apart.
* "The return rate for Frozen, as refunds over net sales": a rate whose condition only says the filter again is the
  ratio of the two figures named.

The invented retail shop of the stress test.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan import followup
from core2.plan.ir import Plan, Window
from core2.plan.planner import _one_period, settle
from core2.plan.values import MemberIndex
from core2.service import Services, Session, _answer_question, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1_26 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}


@pytest.fixture(scope="module")
def shop():
    built, model = learn(domains.build("retail"), "descriptive")
    index = MemberIndex()
    for slug, sql in (("store.name", "SELECT store_name FROM stores"), ("region.name", "SELECT region_name FROM regions"),
                      ("category.name", "SELECT category_name FROM categories"),
                      ("product.name", "SELECT product_name FROM products")):
        index.add(slug, [r[0] for r in built.con.execute(sql).fetchall()])
    return built, model, index


def _ask(shop, question: str, plan: dict, session: Session | None = None) -> dict:
    built, model, index = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=index, today=TODAY)
    return answer_question(question, services, session or Session())


def _run(shop, plan: dict) -> dict:
    built, model, index = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=None, index=index, today=TODAY)
    return _answer_question("q", services, Session(), planned=Plan.model_validate({"kind": "query", **plan}))


def test_a_member_is_named_by_its_own_words_when_no_other_has_them(shop):
    index = shop[2]
    assert index.stored("store.name", "Lakeshore") == "Lakeshore Store"
    assert index.stored("store.name", "old town") == "Old Town Store"
    assert index.stored("store.name", "Store") is None              # a word of every store names none
    assert index.stored("product.name", "Apples") is None           # Apples Large or Apples Small?
    assert ("store.name", "Lakeshore Store") in index.homes("Lakeshore")


def test_a_member_put_on_the_wrong_grouping_is_read_on_the_one_that_has_it(shop):
    built = shop[0]
    payload = _ask(shop, 'Net sales of "Lakeshore" in Q1 2026', {
        "intent": "value", "measures": ["net_amount"], "time": {"window": Q1_26},
        "filters": [{"field": "region.name", "op": "eq", "values": ["Lakeshore"]}]})
    want = built.con.execute(
        "SELECT SUM(l.net_amount) FROM order_lines l JOIN stores s ON l.store_id = s.store_id "
        "JOIN calendar c ON l.order_date_key = c.date_key WHERE s.store_name = 'Lakeshore Store' "
        "AND c.full_date BETWEEN '2026-01-01' AND '2026-03-31'").fetchone()[0]
    assert payload["data"]["rows"][0]["net_amount"] == pytest.approx(float(want))


def test_customers_added_in_a_year_are_counted_by_their_own_date(shop):
    built = shop[0]
    payload = _ask(shop, "How many new customers did we add in 2023?", {
        "intent": "value", "measures": ["number_of_customers"],
        "lifecycle": {"of": "customer", "measure": "number_of_customers", "status": "new"},
        "time": {"window": {"kind": "between", "start": "2023-01-01", "end": "2023-12-31"}}})
    want = built.con.execute("SELECT COUNT(*) FROM customers WHERE created_at >= '2023-01-01' "
                             "AND created_at < '2024-01-01'").fetchone()[0]
    assert payload["data"]["rows"] == [{"number_of_customers": want}]


def test_the_quarter_of_a_follow_up_is_the_one_on_screen(shop):
    model = shop[1]
    previous = Plan.model_validate({"kind": "query", "intent": "value", "measures": ["net_amount"],
                                    "time": {"window": Q1_26}})
    plan = Plan.model_validate({"kind": "query", "intent": "value", "target": {"actual": "net_amount",
                                                                               "target": "target_amount"},
                                "follow_up": "refine", "time": {"window": {"kind": "this", "unit": "quarter"}}})
    settled = settle(plan, model, "Did we hit our sales target for the quarter?", previous)
    assert settled.time.window == previous.time.window
    year = settle(plan.model_copy(update={"time": plan.time.model_copy(update={"window": Window(
        kind="this", unit="year")})}), model, "and for the year?", previous)
    assert year.time.window.kind == "this"                              # a year is no quarter
    assert followup.the_period("sales for the quarter before") is None
    assert _one_period(Window.model_validate(Q1_26), "quarter") and not _one_period(Window.model_validate(Q1_26), "month")


def test_the_stores_that_missed_target_are_those_below_it(shop):
    built = shop[0]
    for missed in ({"field": "net_amount", "op": "lt", "values": ["target_amount"], "total": True},
                   {"field": "of_target", "op": "lt", "values": [100], "total": True},
                   {"field": "Difference", "op": "lt", "values": [0], "total": True}):
        payload = _run(shop, {"intent": "breakdown", "target": {"actual": "net_amount", "target": "target_amount"},
                              "group_by": ["store"], "filters": [missed], "time": {"window": Q1_26}})
        rows = payload["data"]["rows"]
        assert rows and all(r["of_target"] < 100 for r in rows), missed
        assert payload["answer"]["headline"].startswith(f"None of the {len(rows)} stores reached")
        assert "came closest" in payload["answer"]["headline"] and "furthest above" not in payload["answer"]["headline"]
    everyone = _run(shop, {"intent": "breakdown", "target": {"actual": "net_amount", "target": "target_amount"},
                           "group_by": ["store"], "time": {"window": Q1_26}})
    missed_all = built.con.execute("SELECT COUNT(*) FROM stores").fetchone()[0]
    assert len(everyone["data"]["rows"]) == missed_all > len(rows)


def test_a_figure_compared_with_another_by_name_is_never_sent_as_text(shop):
    payload = _run(shop, {"intent": "breakdown", "measures": ["net_amount", "cost_amount"], "group_by": ["store"],
                          "filters": [{"field": "net_amount", "op": "lt", "values": ["cost_amount"], "total": True}],
                          "time": {"window": Q1_26}})
    assert payload.get("data") is None or not payload["data"].get("rows")
    assert "could not run" not in json.dumps(payload)


def test_a_percentage_worked_out_from_a_figure_of_the_same_name_is_shown(shop):
    margin = [{"name": "Gross margin", "op": "difference", "measures": ["net_amount", "cost_amount"]},
              {"name": "Gross margin %", "op": "ratio", "measures": ["Gross margin", "net_amount"], "scale": 100}]
    payload = _run(shop, {"intent": "value", "derived": margin, "time": {"window": Q1_26}})
    (row,) = payload["data"]["rows"]
    assert list(row) == ["gross_margin"] and 30 < row["gross_margin"] < 50          # a percentage, not dollars
    both = _run(shop, {"intent": "breakdown", "derived": margin, "group_by": ["category.name"],
                       "sort": [{"by": "Gross margin", "desc": True}], "time": {"window": Q1_26}})
    assert {"gross_margin", "gross_margin_pct"} <= set(both["data"]["rows"][0])


def test_a_rate_that_restates_the_filter_is_the_ratio_of_the_two_figures(shop):
    built = shop[0]
    frozen = [{"field": "category.name", "op": "eq", "values": ["Frozen"]}]
    payload = _ask(shop, 'What was the return rate for "Frozen" in 2024, as refunds over net sales?', {
        "intent": "value", "measures": ["refund_amount", "net_amount"],
        "rates": [{"name": "Return rate for Frozen", "measure": "refund_amount", "where": frozen}],
        "filters": frozen, "time": {"window": {"kind": "between", "start": "2024-01-01", "end": "2024-12-31"}}})
    (row,) = payload["data"]["rows"]
    refunds = built.con.execute(
        "SELECT SUM(r.refund_amount) FROM returns r JOIN products p ON p.product_id = r.product_id "
        "JOIN categories k ON k.category_id = p.category_id JOIN calendar c ON r.return_date_key = c.date_key "
        "WHERE k.category_name = 'Frozen' AND c.full_date BETWEEN '2024-01-01' AND '2024-12-31'").fetchone()[0]
    sales = built.con.execute(
        "SELECT SUM(o.net_amount) FROM order_lines o JOIN products p ON p.product_id = o.product_id "
        "JOIN categories k ON k.category_id = p.category_id JOIN calendar c ON o.order_date_key = c.date_key "
        "WHERE k.category_name = 'Frozen' AND c.full_date BETWEEN '2024-01-01' AND '2024-12-31'").fetchone()[0]
    assert list(row.values())[0] == pytest.approx(float(refunds) / float(sales) * 100, rel=1e-4)


def test_the_rows_left_out_by_default_are_one_click_away():
    """ "How many bookings did we take in 2025?" leaves out the cancelled, duplicate and voided bookings, as the
    bookings table does by default (and the answer says so); a chip gives the count with them, with no AI."""
    built, model = learn(domains.build("home_services"), "descriptive")
    plan = {"kind": "query", "intent": "value", "measures": ["number_of_bookings"],
            "time": {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        return json.dumps(plan)

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(),
                        today=TODAY)
    session = Session()
    first = answer_question("How many bookings did we take in 2025?", services, session)
    chip = next(c for c in first["follow_up_suggestions"] if c["label"].startswith("Including"))
    assert "Cancelled" in chip["label"]
    every = built.con.execute("SELECT COUNT(*) FROM bookings b JOIN calendar c ON b.booked_date_key = c.date_key "
                              "WHERE c.full_date BETWEEN '2025-01-01' AND '2025-12-31'").fetchone()[0]
    calls = len(asked)
    second = answer_question(chip["question"], services, session)
    assert second["data"]["rows"] == [{"number_of_bookings": every}] and len(asked) == calls
    assert first["data"]["rows"][0]["number_of_bookings"] < every
