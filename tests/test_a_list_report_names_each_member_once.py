"""A list lists its members once each, and a report built on it stays that list.

Found running an analyst's threads on a sample workbook with a real model (gpt-4o): "Give me all the customers who
bought Dish Soap in the last 90 days", then "Add their segment, their city and the store they bought from", "Also
add the order date and the quantity" and "Show that as a chart"; and "List every order over $200 in September 2026
with the customer, store, products and order amount":

* the Dish Soap sold to walk-ins, with no customer on file, was listed as one more customer ("Unknown");
* a customer on several rows (two stores, two days) was counted once per row ("42 customers" for 30), and two
  customers who share a name were counted as one;
* the order date added to the list came back as the quantity day by day, the list gone;
* "show that as a chart" was planned anew, as a line of quantity by day; drawn from the list, two customers of one
  name were one bar;
* every order over $200 was a row per product, each under its own amount: the lines are keyed by order and line
  number and no table lists the orders, so nothing named an order;
* "the customers who bought in the last 20 days" listed every customer when the customers' table has no date: the
  period was dropped, and the answer did not say so.

Every answer goes through the whole answer path, the AI stood in by the plan it wrote, and is compared with SQL
written by hand on an invented shop's warehouse (tests/shop_data.py).

Synthetic data only.
"""

from __future__ import annotations

import json

import pytest
import sqlglot

from core2.answer.builder import build_answer
from core2.compile.compiler import compile_query
from core2.plan.planner import plan_question
from core2.resolve.resolver import Context, resolve
from core2.service import Session
from tests import shop_data
from tests.shop_data import TODAY, ask, headline, rows

DISH = [{"field": "product.name", "op": "eq", "values": ["Dish Soap"]}]
LAST_90 = {"window": {"kind": "last", "unit": "day", "n": 90}}
IN_90 = "o.order_date >= DATE '2026-05-27' AND o.order_date <= DATE '2026-08-24'"
LIVE = "o.status <> 'Cancelled'"
DISH_SOLD = (f"FROM order_lines o JOIN products p USING (product_id) WHERE p.product_name = 'Dish Soap' AND {LIVE} "
             f"AND {IN_90}")
REPORT = ["customer.name", "customer.segment", "customer.city", "store.name"]
IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def shop():
    return shop_data.build()


def _sql(shop, query: str) -> list[tuple]:
    return shop[0].execute(query).fetchall()


def _thread(shop) -> tuple[Session, list[dict]]:
    """The Dish Soap thread, each turn's plan as the AI wrote it."""
    session, answers = Session(), []
    answers.append(ask(shop, "Give me all the customers who bought Dish Soap in the last 90 days",
                       {"intent": "list", "group_by": ["customer.name"], "filters": DISH, "time": LAST_90}, session))
    answers.append(ask(shop, "Add their segment, their city and the store they bought from",
                       {"intent": "list", "group_by": REPORT, "filters": DISH, "time": LAST_90, "follow_up": "refine"},
                       session))
    # As the AI wrote it: the quantity added, the date it was asked for left out.
    answers.append(ask(shop, "Also add the order date and the quantity",
                       {"intent": "list", "measures": ["quantity"], "group_by": REPORT, "filters": DISH,
                        "time": LAST_90, "follow_up": "refine"}, session))
    # As the AI wrote it: a new plan, a line of quantity by day.
    answers.append(ask(shop, "Show that as a chart",
                       {"intent": "trend", "measures": ["quantity"], "filters": DISH,
                        "time": {**LAST_90, "grain": "day"}, "follow_up": "refine"}, session))
    return session, answers


@pytest.fixture(scope="module")
def thread(shop):
    return _thread(shop)[1]


def test_a_list_of_customers_lists_the_customers_never_the_walk_ins(shop, thread):
    first = thread[0]
    bought = _sql(shop, f"SELECT DISTINCT c.customer_name, c.customer_id {DISH_SOLD.replace('USING (product_id)', 'USING (product_id) JOIN customers c USING (customer_id)')}")
    walk_ins = _sql(shop, f"SELECT COUNT(*) {DISH_SOLD} AND o.customer_id IS NULL")[0][0]
    assert walk_ins > 0, "the shop sold Dish Soap to walk-ins in the period"
    listed = {(r["customer_name"], int(r["customer_name_code"])) for r in rows(first)}
    assert listed == {(name, cid) for name, cid in bought}
    assert len(rows(first)) == len(bought), "a row each, and none for the sales with no customer"
    assert headline(first).startswith(f"{len(bought)} customers "), headline(first)
    assert any("no customer recorded are left out" in n for n in first["trust"]["date_context"]), \
        first["trust"]["date_context"]


def test_a_member_on_several_rows_is_counted_once_and_two_of_one_name_are_two(shop, thread):
    report = thread[1]
    pairs = _sql(shop, "SELECT DISTINCT c.customer_id, c.segment, c.city, s.store_name "
                       + DISH_SOLD.replace("USING (product_id)", "USING (product_id) JOIN customers c USING "
                                           "(customer_id) JOIN stores s USING (store_id)"))
    customers = {cid for cid, *_ in pairs}
    assert len(rows(report)) == len(pairs) > len(customers)
    assert headline(report).startswith(f"{len(pairs)} rows for {len(customers)} customers "), headline(report)
    # The two customers named Ari Lane are told apart by their number in the sentence.
    assert "Ari Lane (customer 1" in headline(report) and "Ari Lane (customer 2" in headline(report)


def test_a_date_added_to_a_list_is_a_column_of_each_row(shop, thread):
    dated = thread[2]
    labels = dated["data"]["header_labels"]
    assert "Order date" in labels.values(), labels
    period = next(k for k, v in labels.items() if v == "Order date")
    hand = _sql(shop, "SELECT o.order_date, c.customer_id, s.store_name, SUM(o.quantity) "
                + DISH_SOLD.replace("USING (product_id)", "USING (product_id) JOIN customers c USING (customer_id) "
                                    "JOIN stores s USING (store_id)") + " GROUP BY 1, 2, 3")
    got = {(str(r[period])[:10], int(r["customer_name_code"]), r["store_name"]): int(r["quantity"])
           for r in rows(dated)}
    assert got == {(d.isoformat(), cid, store): int(q) for d, cid, store, q in hand}
    assert headline(dated).startswith(f"{len(hand)} rows for "), headline(dated)


def test_a_date_added_to_a_list_planned_as_a_series_is_still_the_list(shop):
    session = Session()
    ask(shop, "Give me all the customers who bought Dish Soap in the last 90 days",
        {"intent": "list", "group_by": REPORT, "filters": DISH, "time": LAST_90}, session)
    series = ask(shop, "Also add the order date and the quantity",
                 {"intent": "trend", "measures": ["quantity"], "group_by": REPORT, "filters": DISH,
                  "time": {**LAST_90, "grain": "month"}, "follow_up": "refine"}, session)
    assert "Order date" in series["data"]["header_labels"].values()
    assert not series.get("chart") or series["chart"].get("chart_type") != "line"


def test_show_that_as_a_chart_draws_the_list_above_a_bar_per_customer(shop, thread):
    dated, drawn = thread[2], thread[3]
    assert rows(drawn) == rows(dated), "the same answer, drawn"
    chart = drawn["chart"]
    assert chart["chart_type"] == "bar", chart.get("chart_type")
    bars = {r["customer_name"]: r["quantity"] for r in chart["rows"]}
    hand = _sql(shop, "SELECT c.customer_id, c.customer_name, SUM(o.quantity) "
                + DISH_SOLD.replace("USING (product_id)", "USING (product_id) JOIN customers c USING (customer_id)")
                + " GROUP BY 1, 2")
    named = {name for _, name, _ in hand}
    for cid, name, quantity in hand:
        shared = sum(1 for _, n, _ in hand if n == name) > 1
        assert bars[f"{name} ({cid})" if shared else name] == pytest.approx(float(quantity)), name
    assert len(bars) == len(hand) > len(named), "the two customers named Ari Lane are two bars"


def _orders_over(shop, amount: float) -> dict[int, tuple[float, str]]:
    found = _sql(shop, f"""SELECT o.order_id, SUM(o.net_amount), STRING_AGG(DISTINCT p.product_name, ', '
                               ORDER BY p.product_name)
                           FROM order_lines o JOIN products p USING (product_id)
                           WHERE {LIVE} AND o.order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'
                           GROUP BY 1 HAVING SUM(o.net_amount) > {amount}""")
    return {int(order): (float(net), products) for order, net, products in found}


EVERY_ORDER = {"asked": [{"words": "every order over $300", "kind": "condition", "plan": "filters"},
                         {"words": "customer", "kind": "field", "plan": "customer"},
                         {"words": "store", "kind": "field", "plan": "store"},
                         {"words": "products", "kind": "field", "plan": "product"},
                         {"words": "order amount", "kind": "figure", "plan": "net_amount"}],
               "intent": "list", "measures": ["net_amount"], "group_by": ["customer", "store", "product"],
               "filters": [{"field": "net_amount", "op": "gt", "values": [300]}], "time": IN_2025}
EVERY_ORDER_Q = "List every order over $300 in 2025 with the customer, store, products and order amount"


def test_every_order_over_an_amount_is_a_row_per_order_its_products_listed(shop):
    payload = ask(shop, EVERY_ORDER_Q, EVERY_ORDER)
    expected = _orders_over(shop, 300)
    got = rows(payload)
    order = next(k for k in got[0] if k.endswith("order_id"))
    products = next(k for k in got[0] if k.startswith("product"))
    assert len(got) == len(expected), "a row for each order, never one for each of its products"
    for r in got:
        net, listed = expected[int(r[order])]
        assert float(r["net_amount"]) == pytest.approx(net), r[order]
        assert r[products] == listed, r[order]
    assert headline(payload).startswith(f"{len(expected)} orders in 2025"), headline(payload)
    assert "order 1003" in headline(payload) or f"order {sorted(expected)[0]}" in headline(payload)


def test_every_order_grouped_by_its_number_and_its_products_is_still_a_row_per_order(shop):
    # As gpt-4o wrote it once "listed" was kept from it: the order's number beside the products, all grouped by.
    payload = ask(shop, EVERY_ORDER_Q, {**EVERY_ORDER, "group_by": ["customer", "store", "product",
                                                                    "order_line.order_id"]})
    expected = _orders_over(shop, 300)
    got = rows(payload)
    assert len(got) == len(expected), headline(payload)
    order = next(k for k in got[0] if k.endswith("order_id"))
    products = next(k for k in got[0] if k.startswith("product"))
    for r in got:
        net, listed = expected[int(r[order])]
        assert float(r["net_amount"]) == pytest.approx(net) and r[products] == listed, r[order]


@pytest.mark.parametrize("dialect,words", [("snowflake", "LISTAGG"), ("tsql", "STRING_AGG"),
                                           ("postgres", "STRING_AGG"), ("duckdb", "LISTAGG"),
                                           # Oracle's list stops at its 4,000 bytes rather than fail the question.
                                           ("oracle", "ON OVERFLOW TRUNCATE")])
def test_an_orders_products_are_listed_in_every_dialect(shop, dialect, words):
    _con, model, _index = shop
    planned = plan_question(model, EVERY_ORDER_Q, lambda s, t: json.dumps({"kind": "query", **EVERY_ORDER}),
                            today=TODAY).plan
    assert planned.listed, planned
    compiled = compile_query(resolve(planned, model, Context(today=TODAY)), model, dialect)
    assert words in compiled.sql.upper(), compiled.sql
    sqlglot.parse_one(compiled.sql, read=dialect)          # SQL the warehouse reads


def test_the_lines_table_numbers_its_orders_by_the_number_in_its_key(shop):
    from core2.bootstrap.build import numbered_documents

    _con, model, _index = shop
    numbered = [a for a in model.attributes.values()
                if a.kind == "identifier" and model.columns[a.column].name == "order_id"]
    # The order lines' own order numbers, never the returns' (they are lines of the order lines).
    assert [model.tables[model.columns[a.column].table].name for a in numbered] == ["order_lines"]
    # Lines that point at a table of their orders are numbered by that table: no number of their own.
    bare = model.model_copy(deep=True)
    for a in numbered:
        del bare.attributes[a.slug]
    join = next(j for j in bare.joins.values() if j.from_table == numbered[0].column.rsplit(".", 1)[0])
    stores = next(k for k, t in bare.tables.items() if t.name == "stores")
    bare.joins["orders"] = join.model_copy(update={"key": "orders", "from_columns": [numbered[0].column],
                                                   "to_table": stores})
    numbered_documents(bare)
    assert not any(model.columns[a.column].name == "order_id" for a in bare.attributes.values() if a.kind == "identifier")


def test_a_dated_list_of_members_reads_the_rows_that_happened_in_the_period(shop):
    # 3 and 4 Jan 2025: a walk-in and two customers bought; the customers' own table has no date.
    payload = ask(shop, "Which customers bought between 3 and 4 January 2025?",
                  {"intent": "list", "group_by": ["customer.name"],
                   "time": {"window": {"kind": "between", "start": "2025-01-03", "end": "2025-01-04"}}})
    hand = _sql(shop, f"SELECT DISTINCT customer_id FROM order_lines o WHERE {LIVE} AND o.customer_id IS NOT NULL "
                      "AND o.order_date BETWEEN DATE '2025-01-03' AND DATE '2025-01-04'")
    assert 0 < len(hand) < len(shop_data.CUSTOMERS)
    assert {int(r["customer_name_code"]) for r in rows(payload)} == {cid for (cid,) in hand}


def test_a_list_cut_at_the_limit_says_it_shows_the_first_rows(shop):
    con, model, _index = shop
    planned = plan_question(model, EVERY_ORDER_Q, lambda s, t: json.dumps({"kind": "query", **EVERY_ORDER}),
                            today=TODAY).plan
    logical = resolve(planned, model, Context(today=TODAY))
    compiled = compile_query(logical, model, "duckdb")
    result = con.execute(compiled.sql)
    columns = [d[0] for d in result.description]
    payload = build_answer(EVERY_ORDER_Q, logical, compiled, columns, result.fetchall()[:40], truncated=True)
    assert payload["answer"]["headline"].startswith("The first 40 rows in 2025"), payload["answer"]["headline"]
    assert payload["data"]["totals_complete"] is False


def test_the_ai_is_never_offered_the_field_code_lists_by():
    # Offered it, the AI put the order date and the quantity in it ("also add the order date and the quantity"),
    # and the plan was refused twice.
    from core2.plan.ir import schema_text

    assert '"listed"' not in schema_text()


def test_a_date_and_a_figure_the_ai_lists_are_the_lists_day_and_its_figure(shop):
    session = Session()
    ask(shop, "Give me all the customers who bought Dish Soap in the last 90 days",
        {"intent": "list", "group_by": REPORT, "filters": DISH, "time": LAST_90}, session)
    # As the AI wrote it, copying "listed" from a plan it had seen.
    dated = ask(shop, "Also add the order date and the quantity",
                {"intent": "list", "group_by": REPORT, "filters": DISH, "time": LAST_90, "follow_up": "refine",
                 "listed": ["order_date", "quantity", "no_such_field"]}, session)
    assert "Order date" in dated["data"]["header_labels"].values(), dated["answer"]["headline"]
    hand = _sql(shop, "SELECT COUNT(*) FROM (SELECT DISTINCT o.order_date, c.customer_id, s.store_name "
                + DISH_SOLD.replace("USING (product_id)", "USING (product_id) JOIN customers c USING (customer_id) "
                                    "JOIN stores s USING (store_id)") + ")")[0][0]
    assert len(rows(dated)) == hand and all("quantity" in r for r in rows(dated))


# ── found in review ───────────────────────────────────────────────────────


def test_a_dated_list_reads_the_date_the_ai_named(shop):
    # The natural plan names the date the question means: the rows that happened between the two days.
    payload = ask(shop, "Which customers bought between 3 and 4 January 2025?",
                  {"intent": "list", "group_by": ["customer.name"],
                   "time": {"window": {"kind": "between", "start": "2025-01-03", "end": "2025-01-04"},
                            "date": "order_date"}})
    hand = _sql(shop, f"SELECT DISTINCT customer_id FROM order_lines o WHERE {LIVE} AND o.customer_id IS NOT NULL "
                      "AND o.order_date BETWEEN DATE '2025-01-03' AND DATE '2025-01-04'")
    assert {int(r["customer_name_code"]) for r in rows(payload)} == {cid for (cid,) in hand}


def test_a_listed_field_beside_a_comparison_is_no_figure_to_compare(shop):
    # Copied from the plan above onto a comparison, the products listed were compared as an amount: SQL that
    # fails on every database ("the database refused it").
    payload = ask(shop, "and vs 2024", {**EVERY_ORDER, "listed": ["product"],
                                         "group_by": ["order_line.order_id", "customer", "store"],
                                         "time": {**IN_2025, "compare": {"kind": "same_period_last_year"}}})
    got = rows(payload)
    assert got, payload["answer"]["headline"]
    assert "products" in got[0] and not any(k.startswith("products_") for k in got[0]), sorted(got[0])
    assert "net_amount_prior" in got[0]


def test_a_list_by_store_then_customer_drawn_is_a_bar_per_store(shop):
    session = Session()
    ask(shop, "Dish Soap sold in the last 90 days by store and customer",
        {"intent": "list", "measures": ["quantity"], "group_by": ["store.name", "customer.name"], "filters": DISH,
         "time": LAST_90}, session)
    drawn = ask(shop, "Show that as a chart", {"intent": "trend", "measures": ["quantity"], "filters": DISH,
                                                "time": {**LAST_90, "grain": "day"}, "follow_up": "refine"}, session)
    hand = dict(_sql(shop, "SELECT s.store_name, SUM(o.quantity) " + DISH_SOLD.replace(
        "USING (product_id)", "USING (product_id) JOIN stores s USING (store_id)") + " GROUP BY 1"))
    bars = {r["store_name"]: r["quantity"] for r in drawn["chart"]["rows"]}
    assert bars == pytest.approx({k: float(v) for k, v in hand.items()}), "one bar per store, never per customer"


@pytest.mark.parametrize("question,plan", [
    ("add a line of quantity by date", {"intent": "trend", "chart": "line"}),
    ("show me the day-by-day quantity", {"intent": "trend"})])
def test_a_series_asked_for_after_a_list_stays_a_series(shop, question, plan):
    session = Session()
    ask(shop, "Give me all the customers who bought Dish Soap in the last 90 days",
        {"intent": "list", "group_by": ["customer.name"], "filters": DISH, "time": LAST_90}, session)
    series = ask(shop, question, {**plan, "measures": ["quantity"], "filters": DISH,
                                  "time": {**LAST_90, "grain": "day"}, "follow_up": "refine"}, session)
    assert series["plan"]["intent"] == "trend", series["answer"]["headline"]


def test_a_date_that_narrows_a_list_is_no_column_of_it(shop):
    session = Session()
    ask(shop, "List the customers who bought Dish Soap in the last 90 days and how many they bought",
        {"intent": "list", "measures": ["quantity"], "group_by": ["customer.name"], "filters": DISH,
         "time": LAST_90}, session)
    july = ask(shop, "and only those with an order date in July",
               {"intent": "list", "measures": ["quantity"], "group_by": ["customer.name"], "filters": DISH,
                "time": {"window": {"kind": "between", "start": "2026-07-01", "end": "2026-07-31"},
                         "date": "order_date"}, "follow_up": "refine"}, session)
    hand = _sql(shop, "SELECT DISTINCT o.customer_id FROM order_lines o JOIN products p USING (product_id) WHERE "
                      f"p.product_name = 'Dish Soap' AND {LIVE} AND o.customer_id IS NOT NULL AND o.order_date "
                      "BETWEEN DATE '2026-07-01' AND DATE '2026-07-31'")
    assert len(rows(july)) == len(hand), "a row for each customer, never one for each day"


def test_only_a_lists_own_date_is_headed_by_its_name():
    from core2.answer.builder import _header
    from core2.compile.compiler import OutColumn

    # A weekly shape's periods are labelled by their grain: still "Week of", never "Week".
    assert _header(OutColumn("period", "Week", "period", grain="week")) == "Week of"
    assert _header(OutColumn("period", "Month", "period", grain="month")) == "Period"
    assert _header(OutColumn("period", "Order date", "period", grain="day", dated=True)) == "Order date"


def test_a_balance_listed_by_day_is_never_drawn_added_up(shop):
    con, model, index = shop
    held = model.model_copy(deep=True)
    quantity = next(m for m in held.measures.values() if m.slug == "quantity")
    quantity.additivity, quantity.time_aggregation = "semi_additive", "last"    # as stock on hand is
    shop, session = (con, held, index), Session()
    ask(shop, "Give me all the customers who bought Dish Soap in the last 90 days",
        {"intent": "list", "group_by": ["customer.name"], "filters": DISH, "time": LAST_90}, session)
    dated = ask(shop, "Also add the order date and the quantity",
                {"intent": "list", "measures": ["quantity"], "group_by": ["customer.name"], "filters": DISH,
                 "time": LAST_90, "follow_up": "refine"}, session)
    assert "period" in rows(dated)[0], "the list by day"
    drawn = ask(shop, "Show that as a chart", {"intent": "trend", "measures": ["quantity"], "filters": DISH,
                                                "time": {**LAST_90, "grain": "day"}, "follow_up": "refine"}, session)
    assert not drawn.get("chart"), "a balance's days never add up to a bar"


def test_the_rows_with_no_customer_have_no_code_and_no_place(shop):
    from core2.answer.geo import off_map_zips, place_kind

    # Customers who share a name are told apart by their number; the walk-ins have none.
    by_customer = ask(shop, "net sales by customer in 2025", {"intent": "breakdown", "measures": ["net_amount"],
                                                               "group_by": ["customer.name"], "time": IN_2025})
    walk_ins = [r for r in rows(by_customer) if r["customer_name"] == "No customer recorded"]
    assert len(walk_ins) == 1 and walk_ins[0]["customer_name_code"] in ("", None), walk_ins
    assert off_map_zips(["No customer recorded", "Unknown"]) == []
    states = ["California", "Texas", "Nevada", "Oregon", "No customer recorded"]
    assert place_kind("State", "state", states) == place_kind("State", "state", states[:-1])


def test_a_customer_on_several_days_is_one_customer_and_namesakes_beside_are_told_apart(shop, thread):
    by_day = thread[2]
    assert "period" in rows(by_day)[0], "the list by day"
    on_days = {r["customer_name"] for r in rows(by_day)} - {"Ari Lane"}
    assert any(sum(1 for r in rows(by_day) if r["customer_name"] == n) > 1 for n in on_days), "one on several days"
    # A customer who bought on several days is one customer: no number beside the name.
    assert not any(f"{n} (customer " in headline(by_day) for n in on_days), headline(by_day)
    beside = ask(shop, "Dish Soap sold in the last 90 days by store and customer",
                 {"intent": "list", "measures": ["quantity"], "group_by": ["store.name", "customer.name"],
                  "filters": DISH, "time": LAST_90})
    said = headline(beside)
    if "customer Ari Lane" in said:
        assert "customer Ari Lane, customer 1" in said or "customer Ari Lane, customer 2" in said, said


@pytest.mark.parametrize("question,form", [
    ("show that as a chart", "bar"), ("as a pie", "pie"), ("show that as a line", "line"), ("line chart please", "line"),
    ("affiche-le en graphique", "bar"), ("en camembert", "pie"), ("montre ça sous forme de courbe", "line"),
    ("affichez-les en barres s'il vous plaît", "bar"), ("en tableau", "table"),
    # A row of the answer, never a form: the turn is read as it was written.
    ("show that line", None), ("show that area", None), ("show that map", None), ("show the table", None),
    # Found in review: "et en ligne ?" (and online?) and "et en carte ?" (by card?) are questions, never a line or a
    # map; asked for as a form, they still are one.
    ("et en ligne ?", None), ("en ligne ?", None), ("et en carte ?", None), ("et par carte ?", None),
    ("sous forme de carte", "map"), ("affiche-le en carte", "map"), ("montre-le en ligne", "line"),
    ("en graphique en lignes", "line")])
def test_how_to_show_the_list_is_read_in_english_and_french(question, form):
    from core2.plan.ir import Plan
    from core2.plan.thread import shown_as

    above = Plan.model_validate({"kind": "query", "intent": "list", "measures": ["quantity"],
                                 "group_by": ["customer.name"]})
    written = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["quantity"],
                                   "time": {"grain": "day"}})
    shown = shown_as(written, above, question)
    if form is None:
        assert shown is written, question
    else:
        assert shown.intent == "list" and shown.chart == form, (question, shown.chart)
