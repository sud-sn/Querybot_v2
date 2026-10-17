"""The arithmetic an analyst reaches for, each asked of the invented retail warehouse through the whole answer path
(a scripted plan, as the planner writes it) and compared with SQL written here against the raw tables.

* "Bought in one period but not the next", with the figures asked for: the customers who ordered in Q1 2026 and
  not in Q2 2026, with their segment and what they spent in Q1. The planner writes the first period as the
  question's own and the second on the rows they had none of; the second was dropped, and the answer said
  "every customer with orders in Q1 2026 had orders in Q1 2026 too".
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
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
Q2 = {"kind": "between", "start": "2026-04-01", "end": "2026-06-30"}
Q4_25 = {"kind": "between", "start": "2025-10-01", "end": "2025-12-31"}
ORDERED = ("SELECT 1 FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key WHERE o.customer_id = "
           "c.customer_id AND d.full_date BETWEEN DATE '{0}' AND DATE '{1}'")


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


def _order_date(learned) -> str:
    return next(r.slug for r in learned[1].date_roles.values() if r.name.lower() == "order date")


def _between(field: str, window: dict) -> dict:
    return {"field": field, "op": "between", "values": [window["start"], window["end"]]}


def _lapsed(learned) -> list[tuple]:
    return learned[0].con.execute(f"""
        SELECT c.customer_name, c.segment, CAST(ROUND(SUM(o.net_amount), 2) AS DOUBLE)
        FROM customers c JOIN order_lines o ON o.customer_id = c.customer_id
        JOIN calendar d ON o.order_date_key = d.date_key
        WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
          AND NOT EXISTS ({ORDERED.format('2026-04-01', '2026-06-30')})
        GROUP BY c.customer_id, 1, 2""").fetchall()


@pytest.mark.parametrize("written", ["window is the first period", "both periods on the rows", "window second"])
def test_bought_in_one_quarter_not_the_next_with_their_segment_and_spend(retail, written):
    date = _order_date(retail)
    absent = {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)]}
    plans = {
        "window is the first period": {"time": {"window": Q1},
                                       "absent": {**absent, "had": [_between(date, Q1)]}},
        "both periods on the rows": {"absent": {**absent, "had": [_between(date, Q1)]}},
        "window second": {"time": {"window": Q2},
                          "absent": {"of": "customer", "measure": "number_of_orders", "had": [_between(date, Q1)]}},
    }
    plan = {"intent": "list", "measures": ["net_amount"], "group_by": ["customer", "customer.segment"],
            **plans[written]}
    want = _lapsed(retail)
    got = _ask(retail, plan)
    rows = got["export_rows"]
    seen = sorted((r["customer_name"], r["customer_segment"], round(float(r["net_amount"]), 2)) for r in rows)
    assert seen == sorted(want) and want
    assert got["answer"]["headline"].startswith(f"{len(want)} customers with orders in Q1 2026 had none in Q2 2026."), \
        got["answer"]["headline"]
    assert got["definition"].startswith("Listed: the customers with orders in Q1 2026, minus those with orders in "
                                        "Q2 2026."), got["definition"]


def test_a_list_of_who_had_none_shows_their_own_fields(retail):
    date = _order_date(retail)
    plan = {"intent": "list", "group_by": ["customer", "customer.segment", "customer.city"], "time": {"window": Q1},
            "absent": {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)],
                       "had": [_between(date, Q1)]}}
    want = retail[0].con.execute(f"""
        SELECT c.customer_name, c.segment, c.city FROM customers c
        WHERE EXISTS ({ORDERED.format('2026-01-01', '2026-03-31')})
          AND NOT EXISTS ({ORDERED.format('2026-04-01', '2026-06-30')})""").fetchall()
    rows = _ask(retail, plan)["export_rows"]
    assert sorted((r["customer_name"], r["customer_segment"], r["customer_city"]) for r in rows) == sorted(want)
    assert want


def test_how_many_lapsed_in_each_segment(retail):
    """Counted, not listed: the number of customers in each segment who ordered in Q1 2026 and not in Q2 2026."""
    date = _order_date(retail)
    plan = {"intent": "count", "measures": ["number_of_customers"], "group_by": ["customer.segment"],
            "time": {"date": date, "window": Q1},
            "absent": {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)]}}
    want = {}
    for _, segment, _ in _lapsed(retail):
        want[segment] = want.get(segment, 0) + 1
    rows = _ask(retail, plan)["export_rows"]
    got = {r["customer_segment"]: int(next(v for k, v in r.items() if k != "customer_segment")) for r in rows}
    assert got == want and want


def test_how_many_lapsed_in_each_region_counts_each_one_under_their_own_region(retail):
    """ "How many customers made a card purchase in Q1 2026 but none in Q2 2026? Break it down by region" listed
    the customers by name and dropped the regions. Each is counted once, under the region of their own home
    store -- not under each region they bought in."""
    date = _order_date(retail)
    plan = {"intent": "count", "group_by": ["region"],
            "absent": {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)],
                       "had": [_between(date, Q1)]}}
    want = dict(retail[0].con.execute(f"""
        SELECT r.region_name, COUNT(*) FROM customers c JOIN stores s ON c.home_store_id = s.store_id
        JOIN regions r ON s.region_id = r.region_id
        WHERE EXISTS ({ORDERED.format('2026-01-01', '2026-03-31')})
          AND NOT EXISTS ({ORDERED.format('2026-04-01', '2026-06-30')}) GROUP BY 1""").fetchall())
    got = _asked(retail, "How many customers ordered in Q1 2026 but none in Q2 2026? Break it down by region", plan)
    counts = {next(v for v in r.values() if isinstance(v, str)):
              int(float(next(v for v in r.values() if not isinstance(v, str)))) for r in got["export_rows"]}
    assert counts == want and want
    assert sum(want.values()) == len(_lapsed(retail))


@pytest.mark.parametrize("words,counted", [("how many orders they placed in Q1", True),
                                           ("how much they spent in Q1", False)])
def test_how_many_they_did_is_a_count_never_their_money(retail, words, counted):
    """ "Branches that originated a 7/1 ARM in 2024 but none in 2025 - and how many they did in 2024" was answered
    with the dollars lent: "how many" is the count of the rows the words name, each order once (not each line)."""
    date = _order_date(retail)
    plan = {"asked": [{"words": "customers", "kind": "field", "plan": "customer"},
                      {"words": words, "kind": "figure", "plan": "net_amount"}],
            "intent": "list", "measures": ["net_amount"], "group_by": ["customer"], "time": {"window": Q1},
            "absent": {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)],
                       "had": [_between(date, Q1)]}}
    figure = "COUNT(DISTINCT o.order_number)" if counted else "CAST(ROUND(SUM(o.net_amount), 2) AS DOUBLE)"
    want = dict(retail[0].con.execute(f"""
        SELECT c.customer_name, {figure} FROM customers c JOIN order_lines o ON o.customer_id = c.customer_id
        JOIN calendar d ON o.order_date_key = d.date_key
        WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
          AND NOT EXISTS ({ORDERED.format('2026-04-01', '2026-06-30')}) GROUP BY c.customer_id, 1""").fetchall())
    got = _asked(retail, f"Customers who ordered in Q1 2026 but none in Q2 2026 - and {words}", plan)
    column = "number_of_orders" if counted else "net_amount"
    seen = {r["customer_name"]: round(float(r[column]), 2) for r in got["export_rows"]}
    assert seen == {k: round(float(v), 2) for k, v in want.items()} and want


def test_the_same_period_twice_is_said_not_answered(retail):
    """"Bought in Q1 but not in Q1": nothing can be listed, and the question is asked back, not answered empty."""
    date = _order_date(retail)
    built, model = retail
    answers = [json.dumps({"kind": "query", "intent": "list", "time": {"window": Q1},
                           "absent": {"of": "customer", "measure": "number_of_orders",
                                      "had": [_between(date, Q1)]}})] * 3
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert "export_rows" not in payload or not payload["export_rows"], payload
    assert "name the period they had none in" in json.dumps(payload), payload.get("answer")


# ── a share of each group's own total ──────────────────────────────────────

SALES_Q1 = ("FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key "
            "LEFT JOIN stores s ON o.store_id = s.store_id LEFT JOIN regions r ON s.region_id = r.region_id "
            "LEFT JOIN products p ON o.product_id = p.product_id "
            "LEFT JOIN categories c ON p.category_id = c.category_id "
            "WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'")


def _share_column(row: dict) -> float:
    return float(next(v for k, v in row.items() if k.endswith("_share")))


def test_each_categorys_share_of_its_own_regions_sales(retail):
    """"What % of each region's sales came from each category": each share is of the region's total, so every
    region's shares add up to 100 -- never of the grand total."""
    want = {(region, category): share for region, category, share in retail[0].con.execute(f"""
        SELECT region, category, 100.0 * amount / SUM(amount) OVER (PARTITION BY region) FROM (
            SELECT COALESCE(r.region_name, 'Unknown') AS region, COALESCE(c.category_name, 'Unknown') AS category,
                   SUM(o.net_amount) AS amount {SALES_Q1} GROUP BY 1, 2)""").fetchall()}
    got = _ask(retail, {"intent": "share", "measures": ["net_amount"], "group_by": ["region", "category"],
                        "share_of": "region", "time": {"window": Q1}})
    shares = {(r["region_name"], r["category_name"]): _share_column(r) for r in got["export_rows"]}
    assert set(shares) == set(want) and want
    scale = 100.0 if max(shares.values()) <= 1.0 else 1.0
    for key, value in want.items():
        assert shares[key] * scale == pytest.approx(value, abs=0.01), key
    assert any("own total" in n for n in got.get("notes", []) + [got.get("definition", "")]) or \
        "own total" in json.dumps(got), "the answer says each share is of its region's own total"


def test_one_grouping_share_is_of_everything(retail):
    got = _ask(retail, {"intent": "share", "measures": ["net_amount"], "group_by": ["region"], "share_of": "region",
                        "time": {"window": Q1}})
    total = sum(_share_column(r) for r in got["export_rows"])
    assert total == pytest.approx(1.0, abs=1e-3) or total == pytest.approx(100.0, abs=0.01)


# ── groups kept by their change ────────────────────────────────────────────

def _by_product(learned, window: dict) -> dict:
    return dict(learned[0].con.execute(f"""
        SELECT COALESCE(p.product_name, 'Unknown'), SUM(o.net_amount) FROM order_lines o
        JOIN calendar d ON o.order_date_key = d.date_key LEFT JOIN products p ON o.product_id = p.product_id
        WHERE d.full_date BETWEEN DATE '{window["start"]}' AND DATE '{window["end"]}' GROUP BY 1""").fetchall())


@pytest.mark.parametrize("kept", ["fell", "rose", "down more than 10%"])
def test_only_the_products_that_fell_or_rose(retail, kept):
    """"Which products sold less in Q2 than in Q1, and by how much": only those that fell, each with both quarters
    and the change -- the answer listed every product, the ones that rose with them."""
    # Two whole quarters: the quarter still under way (today is in Q2) is compared like for like.
    now, before = _by_product(retail, Q1), _by_product(retail, Q4_25)
    change = {p: float(now.get(p) or 0) - float(before.get(p) or 0) for p in {*now, *before}}
    pct = {p: change[p] / float(before[p]) for p in change if before.get(p)}
    if kept == "fell":
        flt, want = {"field": "change", "op": "lt", "values": [0]}, {p for p, c in change.items() if c < 0}
    elif kept == "rose":
        flt, want = {"field": "change", "op": "gt", "values": [0]}, {p for p, c in change.items() if c > 0}
    else:
        flt, want = {"field": "pct_change", "op": "lt", "values": [-10]}, {p for p, c in pct.items() if c < -0.10}
    got = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["product"], "filters": [flt],
                        "time": {"window": Q1, "compare": {"kind": "window", "window": Q4_25}},
                        "sort": [{"by": "change", "desc": kept == "rose"}]})
    assert {r["product_name"] for r in got["export_rows"]} == want and want
    words = {"fell": "that fell", "rose": "that rose", "down more than 10%": "down more than 10%"}[kept]
    assert words in json.dumps(got), got["answer"]["headline"]


# ── read from the question's words where the AI's plan missed it ───────────

def _asked(learned, question: str, plan: dict) -> dict:
    """The question put through the whole answer path, the AI writing ``plan`` for it."""
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})] * 3
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def test_a_rate_on_every_category_is_each_categorys_share_of_its_region(retail):
    """The plan gpt-4o wrote for "What percentage of each region's revenue came from each product category?": a
    "rate" whose condition is every category, 100% for each. It is each category's share of its region."""
    categories = [c for c, in retail[0].con.execute("SELECT category_name FROM categories").fetchall()]
    plan = {"intent": "share", "measures": ["net_amount"], "group_by": ["region", "category"],
            "rates": [{"name": "Revenue share by category", "measure": "net_amount",
                       "where": [{"field": "category", "op": "in", "values": categories}]}],
            "time": {"window": Q1}}
    got = _asked(retail, "What percentage of each region's revenue came from each product category in Q1 2026?",
                 plan)
    by_region: dict[str, float] = {}
    for r in got["export_rows"]:
        by_region[r["region_name"]] = by_region.get(r["region_name"], 0.0) + _share_column(r)
    assert by_region and all(v == pytest.approx(100.0, abs=0.05) or v == pytest.approx(1.0, abs=5e-4)
                             for v in by_region.values()), by_region


@pytest.mark.parametrize("question", [
    "What % of each region's sales came from each category in Q1 2026?",
    "Show each category's share of its region's net sales, Q1 2026, by region and category",
])
def test_the_words_name_the_group_each_share_is_of(retail, question):
    plan = {"intent": "share", "measures": ["net_amount"], "group_by": ["region", "category"],
            "time": {"window": Q1}}
    got = _asked(retail, question, plan)
    by_region: dict[str, float] = {}
    for r in got["export_rows"]:
        by_region[r["region_name"]] = by_region.get(r["region_name"], 0.0) + _share_column(r)
    assert len(by_region) > 1 and all(v == pytest.approx(100.0, abs=0.05) for v in by_region.values()), by_region


@pytest.mark.parametrize("question, kept", [
    ("Which products sold less in Q1 2026 than in Q4 2025, and by how much?", "fell"),
    ("List the products whose net sales fell from Q4 2025 to Q1 2026", "fell"),
    ("which products sold more in Q1 2026 than in Q4 2025?", "rose"),
    ("Which products had more than 10% lower sales in Q1 2026 than in Q4 2025?", "down 10"),
    ("Which product grew the most from Q4 2025 to Q1 2026?", "all"),
])
def test_the_question_keeps_only_those_that_fell_or_rose(retail, question, kept):
    plan = {"intent": "compare", "measures": ["net_amount"], "group_by": ["product"],
            "time": {"window": Q1, "compare": {"kind": "window", "window": Q4_25}}}
    now, before = _by_product(retail, Q1), _by_product(retail, Q4_25)
    change = {p: float(now.get(p) or 0) - float(before.get(p) or 0) for p in {*now, *before}}
    want = {"fell": {p for p, c in change.items() if c < 0}, "rose": {p for p, c in change.items() if c > 0},
            "down 10": {p for p, c in change.items() if before.get(p) and c / float(before[p]) < -0.10},
            "all": set(change)}[kept]
    got = _asked(retail, question, plan)
    assert {r["product_name"] for r in got["export_rows"]} == want and want


# ── a ratio of two figures from two tables ─────────────────────────────────

def test_refunds_as_a_percentage_of_sales_by_category_each_by_its_own_date(retail):
    """gpt-4o planned "refunds as a percentage of net sales" as a rate of net sales on every category: 100% for
    each. It is the refunds (by their return date) over the sales (by their order date), category by category."""
    categories = [c for c, in retail[0].con.execute("SELECT category_name FROM categories").fetchall()]
    plan = {"intent": "breakdown", "group_by": ["category"], "time": {"window": Q1},
            "rates": [{"name": "Return rate", "measure": "net_amount",
                       "where": [{"field": "category", "op": "in", "values": categories}]}]}
    want = dict(retail[0].con.execute("""
        WITH s AS (SELECT COALESCE(c.category_name, 'Unknown') AS cat, SUM(o.net_amount) AS sales FROM order_lines o
                   JOIN calendar d ON o.order_date_key = d.date_key LEFT JOIN products p ON o.product_id = p.product_id
                   LEFT JOIN categories c ON p.category_id = c.category_id
                   WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' GROUP BY 1),
             r AS (SELECT COALESCE(c.category_name, 'Unknown') AS cat, SUM(x.refund_amount) AS refunds FROM returns x
                   JOIN calendar d ON x.return_date_key = d.date_key
                   LEFT JOIN order_lines o ON x.order_line_id = o.order_line_id
                   LEFT JOIN products p ON o.product_id = p.product_id
                   LEFT JOIN categories c ON p.category_id = c.category_id
                   WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' GROUP BY 1)
        SELECT s.cat, 100.0 * r.refunds / s.sales FROM s JOIN r USING (cat)""").fetchall())
    got = _asked(retail, "Return rate by product category in Q1 2026: refunds as a percentage of net sales", plan)
    ratio = {r["category_name"]: float(r["return_rate"]) for r in got["export_rows"]
             if r.get("return_rate") is not None}
    assert set(want) <= set(ratio) <= {*want, "Unknown"} and want    # rows of no known category said apart
    for cat, value in want.items():
        assert ratio[cat] == pytest.approx(float(value), abs=0.001), cat
    assert not any(v == pytest.approx(100.0) for v in ratio.values())
    # Beside each ratio, the two figures it divides; the answer still led by the ratio.
    row = next(r for r in got["export_rows"] if r.get("category_name") in want)
    assert {"net_amount"} <= set(row) and any("refund" in k for k in row), row
    assert got["answer"]["headline"].lower().startswith("return rate"), got["answer"]["headline"]


# ── a condition on the total of each document ──────────────────────────────

def test_orders_over_an_amount_are_each_orders_own_total(retail):
    """"List every order over $500 in Q1 2026 with the customer and the order amount": gpt-4o grouped by customer
    with the condition on their total; each order's own total passes it or not."""
    plan = {"intent": "list", "measures": ["net_amount"], "group_by": ["customer"],
            "filters": [{"field": "net_amount", "op": "gt", "values": [500]}], "time": {"window": Q1}}
    want = {(o, c) for o, c in retail[0].con.execute("""
        SELECT o.order_number, cu.customer_name FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
        JOIN customers cu ON o.customer_id = cu.customer_id
        WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
        GROUP BY 1, 2 HAVING SUM(o.net_amount) > 500""").fetchall()}
    got = _asked(retail, "List every order over $500 in Q1 2026 with the customer and the order amount", plan)
    rows = got["export_rows"]
    number = next(k for k in rows[0] if "order_number" in k)
    assert {(r[number], r["customer_name"]) for r in rows} == want and want


# ── two periods named: the later against the earlier, the calendar's ──────

@pytest.mark.parametrize("question, written", [
    # gpt-4o's plan for the bank's "Total orig $ fell from 2024 to 2025": both years against the two before them.
    ("Total net sales from Q4 2025 to Q1 2026 - how much of the change came from each region?",
     {"window": {"kind": "between", "start": "2025-10-01", "end": "2026-03-31"}, "compare": {"kind": "previous_period"}}),
    # And "H1 2026 against H1 2025" read as a fiscal year's halves; here the quarters, read the other way round.
    ("Which regions sold less in Q1 2026 than in Q4 2025?",
     {"window": {"kind": "between", "start": "2025-10-01", "end": "2025-12-31"},
      "compare": {"kind": "window", "window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}}),
])
def test_two_named_periods_set_the_later_against_the_earlier(retail, question, written):
    plan = {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"], "time": written}
    got = _asked(retail, question, plan)
    want = dict(retail[0].con.execute("""
        SELECT COALESCE(r.region_name, 'Unknown'),
               SUM(CASE WHEN d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31' THEN o.net_amount END)
             - SUM(CASE WHEN d.full_date BETWEEN DATE '2025-10-01' AND DATE '2025-12-31' THEN o.net_amount END)
        FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
        LEFT JOIN stores s ON o.store_id = s.store_id LEFT JOIN regions r ON s.region_id = r.region_id
        GROUP BY 1""").fetchall())
    change = {r["region_name"]: float(r["net_amount_change"]) for r in got["export_rows"]}
    if "sold less" in question:
        want = {k: v for k, v in want.items() if v < 0}
    assert set(change) == set(want) and want
    for region, value in want.items():
        assert change[region] == pytest.approx(float(value), abs=0.01), region


def test_how_many_customers_spent_over_an_amount_in_each_segment(retail):
    """gpt-4o's plan for "How many customers deposited more than $50,000 in Q2 2026? Split it by segment" kept the
    deposits above $50,000 by segment (refused). It counts each segment's customers whose own total passes it."""
    plan = {"intent": "count", "group_by": ["customer.segment"], "time": {"window": Q1},
            "filters": [{"field": "net_amount", "op": "gt", "values": [2000]}]}
    want = dict(retail[0].con.execute("""
        SELECT segment, COUNT(*) FROM (
            SELECT c.segment, o.customer_id FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE d.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
            GROUP BY 1, 2 HAVING SUM(o.net_amount) > 2000) GROUP BY 1""").fetchall())
    got = _asked(retail, "How many customers spent more than $2,000 with us in Q1 2026? Split it by segment", plan)
    counts = {r["customer_segment"]: int(float(next(v for k, v in r.items() if k != "customer_segment")))
              for r in got["export_rows"]}
    assert counts == want and want


# ── members the data keeps only a number for ───────────────────────────────


@pytest.fixture(scope="module")
def no_patient_list():
    """The compounding pharmacy without its table of patients: each prescription holds the patient's number, and
    the data lists nobody (as a pharmacy chain keeps it)."""
    import dataclasses

    domain = domains.build("compounding_pharmacy")
    domain = dataclasses.replace(domain, tables=[t for t in domain.tables if t.name != "patients"], truth=dataclasses.replace(
        domain.truth, joins=[j for j in domain.truth.joins if "patients" not in (j.from_table, j.to_table)]))
    return learn(domain, "descriptive")


def _fill_date(learned) -> str:
    return next(r.slug for r in learned[1].date_roles.values() if r.name.lower() == "fill date")


FILLED = ("SELECT 1 FROM fills f2 JOIN prescriptions p2 ON f2.rx_number = p2.rx_number WHERE p2.patient_id = "
          "p.patient_id AND f2.fill_date BETWEEN DATE '{0}' AND DATE '{1}'")


@pytest.mark.parametrize("of", ["number_of_patients", "prescriber"])
def test_patients_with_a_fill_in_q1_and_none_in_q2_counted_by_prescriber(no_patient_list, of):
    """ "How many patients filled a statin in Q1 2026 but didn't fill one in Q2 2026? by store" was planned as the
    drugs with no fills (the data has no list of patients, and the AI named the closest list it had), and "every
    region with patients in Q4 2024 had patients in Q4 2025 too". The patients are counted by the number their
    prescriptions hold, on the fills of each period by the fill's own date -- and never listed."""
    date = _fill_date(no_patient_list)
    plan = {"intent": "count", "group_by": ["prescriber"],
            "absent": {"of": of, "measure": "number_of_fills", "without": [_between(date, Q2)],
                       "had": [_between(date, Q1)]}}
    want = dict(no_patient_list[0].con.execute(f"""
        SELECT pr.prescriber_name, COUNT(DISTINCT p.patient_id) FROM fills f
        JOIN prescriptions p ON f.rx_number = p.rx_number JOIN prescribers pr ON p.prescriber_npi = pr.prescriber_npi
        WHERE f.fill_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
          AND NOT EXISTS ({FILLED.format('2026-04-01', '2026-06-30')}) GROUP BY 1""").fetchall())
    got = _asked(no_patient_list, "How many patients had a fill in Q1 2026 but none in Q2 2026? By prescriber", plan)
    counts = {next(v for v in r.values() if isinstance(v, str)):
              int(float(next(v for v in r.values() if not isinstance(v, str)))) for r in got["export_rows"]}
    assert counts == want and want
    assert not any("patient_id" in k or "member_key" in k for r in got["export_rows"] for k in r)
    assert "patients" in got["answer"]["headline"].lower()


@pytest.mark.parametrize("question,kept", [
    ("drop the count and the average, and show each balance as % of the region's total instead", True),
    ("and how did that change against last year?", False),
    # Asking for a change in other words keeps the comparison the AI set beside it.
    ("which ones went up?", False), ("which stores are dropping?", False), ("rank them by the drop", False),
    ("who were the biggest movers?", False), ("which gained the most?", False), ("and QoQ?", False),
    ("show the delta", False), ("Lesquels ont baissé ?", False), ("Et en glissement annuel ?", False),
    ("Et la variation ?", False), ("Comparé à avant ?", False)])
def test_a_follow_up_naming_no_comparison_keeps_the_one_period_on_screen(retail, question, kept):
    """ "Drop the count and the average, and show each balance as % of the region's total instead", after balances at
    the end of June 2026, was answered as the year to date against the year before, which no one asked for. A turn
    naming no period and no comparison keeps the period on screen; one asking how it changed has its comparison."""
    from core2.plan.ir import Plan
    from core2.plan.planner import settle

    previous = Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": ["net_amount"],
                                    "group_by": ["region", "category"], "time": {"window": Q1}})
    written = Plan.model_validate({
        "kind": "query", "intent": "compare", "measures": ["net_amount"], "group_by": ["region", "category"],
        "share_of": "region", "follow_up": "refine", "sort": [{"by": "change", "desc": False}],
        "time": {"window": {"kind": "to_date", "unit": "year"}, "compare": {"kind": "same_period_last_year"}}})
    settled = settle(written, retail[1], question, previous)
    if kept:
        assert settled.time.compare is None and settled.time.window == previous.time.window
        assert settled.intent == "share" and not any(s.by == "change" for s in settled.sort)
    else:
        assert settled.time.compare is not None


def test_a_follow_up_keeps_the_period_its_comparison_was_against(retail):
    """ "Only the ones that fell, and by how much", after H1 2026 against H1 2025 by branch, was answered for January to
    September 2026 against H1 2025: the comparison kept, its period moved. Neither moves when the turn names none."""
    from core2.plan.ir import Plan
    from core2.plan.planner import settle

    h1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
    h1_25 = {"kind": "window", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-06-30"}}
    previous = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                    "group_by": ["store"], "time": {"window": h1, "compare": h1_25}})
    written = Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                   "group_by": ["store"], "follow_up": "refine",
                                   "time": {"window": {"kind": "to_date", "unit": "year"}, "compare": h1_25}})
    settled = settle(written, retail[1], "only the ones that fell, and by how much", previous)
    assert settled.time.window == previous.time.window and settled.time.compare == previous.time.compare


# ── what part of each group's own total one member is ──────────────────────

@pytest.mark.parametrize("question,within", [
    ('What share of each region\'s net sales came from "Bakery" in Q1 2026?', True),
    ('What share of "Bakery" net sales did each region have in Q1 2026?', False)])
def test_what_share_of_each_regions_sales_came_from_one_category(retail, question, within):
    """ "For the Nebraska stores, what share of each store's front store sales came from Beauty in Q2 2026?" kept the
    sales to Beauty and gave each store's part of all of Beauty. "Came from Beauty" is Beauty's part of each store's
    own sales; "each region's share of Bakery sales" is still each region's part of all of Bakery."""
    plan = {"intent": "share", "measures": ["net_amount"], "group_by": ["region"], "share_of": "region",
            "filters": [{"field": "category", "op": "eq", "values": ["Bakery"]}], "time": {"window": Q1}}
    if not within:
        plan.pop("share_of")
    rows = retail[0].con.execute(f"""
        SELECT r.region_name, SUM(CASE WHEN c.category_name = 'Bakery' THEN o.net_amount END), SUM(o.net_amount)
        {SALES_Q1} GROUP BY 1""").fetchall()
    bakery = sum(float(b or 0) for _, b, _ in rows)
    want = {r: 100.0 * float(b or 0) / float(t) if within else 100.0 * float(b or 0) / bakery for r, b, t in rows}
    got = _asked(retail, question, plan)
    seen = {r["region_name"]: float(next(v for k, v in r.items() if k != "region_name" and isinstance(v, float)
                                         and ("share" in k or "%" in k or "bakery" in k)))
            for r in got["export_rows"]}
    assert set(seen) == set(want) and want
    for region, value in want.items():
        assert seen[region] == pytest.approx(value, abs=0.05), (region, got["export_rows"])


def test_so_many_per_hundred_of_another_is_their_ratio(retail):
    """ "Flu shots per 100 scripts by region" was planned as the flu shots times the scripts."""
    returns = next(m.slug for m in retail[1].measures.values() if m.table.endswith(".returns")
                   and getattr(m.expr, "agg", "") == "count" and not m.hidden)
    plan = {"intent": "breakdown", "group_by": ["region"], "time": {"window": Q1},
            "derived": [{"name": "Returns per 100 orders", "op": "product",
                         "measures": [returns, "number_of_orders"], "scale": 100.0}]}
    got = _asked(retail, "Returns per 100 orders by region in Q1 2026", plan)
    plan_out = got["plan"]
    assert plan_out["derived"][0]["op"] == "ratio" and plan_out["derived"][0]["scale"] == 100.0, plan_out


# ── what the rules leave alone (an adversarial review's cases) ──────────────

JAN_MAY = {"kind": "between", "start": "2026-01-01", "end": "2026-05-31"}
JAN_MAY_25 = {"kind": "window", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-05-31"}}


@pytest.mark.parametrize("rule,question,plan", [
    ("two_periods_named", "Sales by region, Jan-May 2026 vs Jan-May 2025",
     {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"],
      "time": {"window": JAN_MAY, "compare": JAN_MAY_25}}),
    ("two_periods_named", "Sales by region year to date 2026 vs 2025",
     {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"],
      "time": {"window": {"kind": "to_date", "unit": "year"}, "compare": {"kind": "same_period_last_year"}}}),
    ("two_periods_named", "How did Q1 2026 sales compare with the previous quarter, for products launched in Q3 2025?",
     {"intent": "compare", "measures": ["net_amount"], "group_by": ["product"],
      "time": {"window": Q1, "compare": {"kind": "previous_period"}}}),
    ("kept_by_change", "Compare revenue and returns by store, Q1 2026 vs Q4 2025: which stores had more returns?",
     {"intent": "compare", "measures": ["net_amount", "refund_amount"], "group_by": ["store"],
      "time": {"window": Q1, "compare": {"kind": "window", "window": Q4_25}}}),
    ("kept_by_change", "Sales by store in Q1 2026 vs Q4 2025: which stores have lower margins than the average",
     {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
      "time": {"window": Q1, "compare": {"kind": "window", "window": Q4_25}}}),
    ("total_at_its_level", "Which stores had orders above 100 in Q1 2026?",
     {"intent": "list", "group_by": ["store"], "time": {"window": Q1},
      "filters": [{"field": "number_of_orders", "op": "gt", "values": [100]}]}),
    ("total_at_its_level", "Which customers had orders totaling more than $2,000 in Q1 2026?",
     {"intent": "list", "group_by": ["customer"], "time": {"window": Q1},
      "filters": [{"field": "net_amount", "op": "gt", "values": [2000]}]}),
    ("members_over_a_total", "How many customers spent more than $2,000 in Q1 2026, and what did they spend in total?",
     {"intent": "count", "measures": ["net_amount"], "group_by": [], "time": {"window": Q1},
      "filters": [{"field": "net_amount", "op": "gt", "values": [2000]}]}),
    ("share_in_group", "Net sales split by store and category in each region for Q1 2026",
     {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store", "category", "region"],
      "time": {"window": Q1}}),
])
def test_a_rule_leaves_a_plan_it_does_not_fit_alone(retail, rule, question, plan):
    """Each rule rewrites only the plans its words fit: a like-for-like part of two years, a period a condition
    holds, a change kept with two figures shown, a count of orders, the members a question asks about, the figures
    asked beside a count, and "split" with no share asked."""
    from core2.plan import analyst_math
    from core2.plan.ir import Plan

    written = Plan.model_validate({"kind": "query", **plan})
    assert getattr(analyst_math, rule)(written, retail[1], question) == written


def test_a_share_of_a_rate_on_its_grouping_is_of_that_rates_figure(retail):
    """A "rate" whose only condition is the grouping, beside another figure: the share is of the rate's own figure
    (a share is worked out on the first measure shown), never of the order count beside it."""
    from core2.plan import analyst_math
    from core2.plan.ir import Plan

    cats = [c for c, in retail[0].con.execute("SELECT category_name FROM categories").fetchall()]
    written = Plan.model_validate({
        "kind": "query", "intent": "breakdown", "measures": ["number_of_orders"], "group_by": ["region", "category"],
        "time": {"window": Q1}, "rates": [{"name": "Category share", "measure": "net_amount",
                                           "where": [{"field": "category", "op": "in", "values": cats}]}]})
    out = analyst_math.share_in_group(written, retail[1], "What % of each region's sales came from each category?")
    assert out.measures[0] == "net_amount" and out.share_of == "region", out


def test_an_ai_figure_over_two_fact_tables_counts_each_on_its_own_rows(retail):
    """ "Returns per 100 orders by category" written as one figure, the returns over the distinct orders: the orders
    are counted on the order lines, by their own date -- counted on the returns, every category read 100."""
    plan = {"intent": "breakdown", "group_by": ["category"], "time": {"window": Q1},
            "figures": [{"name": "Returns per 100 orders", "format": "number",
                         "formula": "(COUNT([Return]) / COUNT(DISTINCT [Order line · Order number])) * 100"}]}
    got = _asked(retail, "Returns per 100 orders by category in Q1 2026", plan)
    values = [float(v) for r in got["export_rows"] if r.get("category_name") != "Unknown"
              for k, v in r.items() if k.startswith("returns_per")]
    assert len(values) >= 5 and all(0 < v < 50 for v in values), got["export_rows"]


def test_the_lapsed_listed_with_a_field_of_another_table_keep_it(retail):
    """ "The customers who ordered in Q1 2026 but not in Q2, with their region": a region is their home store's, on
    another table; it was dropped from the list without a word. Each one is listed with it (and their orders)."""
    date = _order_date(retail)
    plan = {"intent": "list", "group_by": ["customer", "region"],
            "absent": {"of": "customer", "measure": "number_of_orders", "without": [_between(date, Q2)],
                       "had": [_between(date, Q1)]}}
    want = dict(retail[0].con.execute(f"""
        SELECT c.customer_name, r.region_name FROM customers c JOIN stores s ON c.home_store_id = s.store_id
        JOIN regions r ON s.region_id = r.region_id
        WHERE EXISTS ({ORDERED.format('2026-01-01', '2026-03-31')})
          AND NOT EXISTS ({ORDERED.format('2026-04-01', '2026-06-30')})""").fetchall())
    got = _asked(retail, "Customers who ordered in Q1 2026 but not in Q2 2026, with their region", plan)
    seen = {r["customer_name"]: r["region_name"] for r in got["export_rows"]}
    assert seen == want and want


@pytest.fixture(scope="module")
def pharmacy():
    return learn(domains.build("compounding_pharmacy"), "descriptive")


WROTE = ("SELECT 1 FROM prescriptions p2 JOIN formula_ingredients fi2 ON p2.formula_id = fi2.formula_id "
         "JOIN ingredients i2 ON fi2.ingredient_id = i2.ingredient_id WHERE p2.prescriber_npi = pr.prescriber_npi "
         "AND i2.ingredient_name = 'Gabapentin powder' AND p2.written_date BETWEEN DATE '{0}' AND DATE '{1}'")


def test_members_kept_by_number_are_counted_by_the_date_they_had_some_on(no_patient_list):
    """ "Patients with a prescription written in Q1 2026 but no fill in Q2 2026, by prescriber": those who had some
    are counted by the written date the question named, never by the fill date of the rows they are counted on."""
    con = no_patient_list[0].con
    want = dict(con.execute(f"""SELECT pr.prescriber_name, COUNT(DISTINCT p.patient_id) FROM fills f
        JOIN prescriptions p ON f.rx_number = p.rx_number JOIN prescribers pr ON p.prescriber_npi = pr.prescriber_npi
        WHERE p.written_date BETWEEN DATE '2026-01-01' AND DATE '2026-03-31'
          AND NOT EXISTS ({FILLED.format('2026-04-01', '2026-06-30')}) GROUP BY 1""").fetchall())
    plan = {"intent": "count", "group_by": ["prescriber"],
            "absent": {"of": "number_of_patients", "measure": "number_of_fills",
                       "without": [_between("fill_date", Q2)], "had": [_between("written_date", Q1)]}}
    got = _asked(no_patient_list, "How many patients had a prescription written in Q1 2026 but no fill in Q2 2026? "
                 "By prescriber", plan)
    counts = {next(v for v in r.values() if isinstance(v, str)):
              int(float(next(v for v in r.values() if not isinstance(v, str)))) for r in got["export_rows"]}
    assert counts == want and want


def test_rows_without_a_member_across_a_bridge_have_none_of_it(pharmacy):
    """ "Prescriptions without gabapentin by specialty": those whose formula holds no gabapentin at all (NOT EXISTS),
    never those holding some other ingredient besides it -- every prescription."""
    con = pharmacy[0].con
    want = dict(con.execute("""SELECT pr.specialty, COUNT(*) FROM prescriptions p
        JOIN prescribers pr ON p.prescriber_npi = pr.prescriber_npi WHERE NOT EXISTS (SELECT 1 FROM formula_ingredients fi
        JOIN ingredients i ON fi.ingredient_id = i.ingredient_id WHERE fi.formula_id = p.formula_id
        AND i.ingredient_name = 'Gabapentin powder') GROUP BY 1""").fetchall())
    got = _asked(pharmacy, 'Number of prescriptions without "Gabapentin powder" by specialty', {
        "intent": "breakdown", "measures": ["number_of_prescriptions"], "group_by": ["prescriber.specialty"],
        "filters": [{"field": "ingredient.name", "op": "ne", "values": ["Gabapentin powder"]}]})
    assert {r["prescriber_specialty"]: int(float(r["number_of_prescriptions"])) for r in got["export_rows"]} == want


@pytest.mark.parametrize("figures", [False, True])
def test_who_had_none_by_a_condition_across_a_bridge(pharmacy, figures):
    """ "Which prescribers wrote gabapentin scripts in Q2 2026 but none in Q3 2026?" (gabapentin an ingredient, held
    by each formula through its list of ingredients) could not be worked out: the condition on the ingredient was
    put on the rows themselves, outside the EXISTS its table sits in, and named a table the query did not have."""
    cond = [{"field": "ingredient.name", "op": "eq", "values": ["Gabapentin powder"]}]
    plan = {"intent": "list", "group_by": ["prescriber"],
            "absent": {"of": "prescriber", "measure": "number_of_prescriptions",
                       "without": [*cond, _between("written_date", Q2)], "had": [*cond, _between("written_date", Q1)]}}
    if figures:
        plan["measures"] = ["number_of_prescriptions"]
    want = {n for n, in pharmacy[0].con.execute(f"""
        SELECT pr.prescriber_name FROM prescribers pr
        WHERE EXISTS ({WROTE.format('2026-01-01', '2026-03-31')})
          AND NOT EXISTS ({WROTE.format('2026-04-01', '2026-06-30')})""").fetchall()}
    got = _asked(pharmacy, 'Which prescribers wrote "Gabapentin powder" scripts in Q1 2026 but none in Q2 2026?', plan)
    assert {r["prescriber_name"] for r in got["export_rows"]} == want and want
