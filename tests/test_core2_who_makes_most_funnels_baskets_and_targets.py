"""Analytics wave 3: who makes most of a figure, funnels, what is bought together, and a figure against its target,
each equal to SQL written by hand.

Asked of the invented retail warehouse through the whole answer path (a scripted plan, as the planner writes it):

* ABC classes of products by net amount (A the products making the first 80%, B the next 15%, C the rest: a
  class holds the product that takes the total past its edge), the customers making the first 80%, deciles;
* order lines through order date, ship date and delivered date: how many reached each, the share of the first and
  of the stage before, and the average days from the stage before;
* products on the same order, in pairs: the orders with both, their share, how often the first's orders hold the
  second, and the lift; or the products bought with one quoted product;
* net amount against target amount by store and by month: the difference, the % of target, how many reached it.

Each compiles on every warehouse; each says how it was counted; one asked of what it cannot be is told why.
Learn offers these questions where the data gives them, and the planner is shown how to ask for them.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.analyses import suggestions
from core2.plan.ir import Plan
from core2.plan.planner import stable_prompt
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
LINES = """SELECT ol.*, c.full_date AS ordered FROM order_lines ol JOIN calendar c ON ol.order_date_key = c.date_key
           WHERE c.full_date >= DATE '2025-01-01' AND c.full_date < DATE '2026-01-01'"""


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, plan: dict, question: str = "q") -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _sql(learned, sql: str) -> list[tuple]:
    return learned[0].con.execute(sql).fetchall()


def _per_member(learned, key: str, table: str, label: str) -> list[tuple[str, float]]:
    """Each member's net amount in 2025, the largest first (ties by name), as the answer ranks them: a member the
    list lacks (a planted product number no product has) is one of them, under its number."""
    return [(name, float(v)) for name, v in _sql(learned, f"""
        SELECT COALESCE(x.{label}, CAST(l.{key} AS VARCHAR) || ' (no name)') AS member, SUM(l.net_amount)
        FROM ({LINES}) l LEFT JOIN {table} x ON l.{key} = x.{key}
        GROUP BY 1 HAVING SUM(l.net_amount) IS NOT NULL ORDER BY 2 DESC, 1""")]


def _classes(values: list[float], cuts: list[float]) -> list[str]:
    total, running, out = sum(values), 0.0, []
    for v in values:
        before = running * 100 / total
        out.append(next((c for c, cut in zip("ABC", cuts) if before < cut), "ABC"[len(cuts)]))
        running += v
    return out


# ── who makes most ─────────────────────────────────────────────────────────


def test_abc_classes_of_products_by_net_amount(retail):
    members = _per_member(retail, "product_id", "products", "product_name")
    classes = _classes([v for _, v in members], [80, 95])
    total = sum(v for _, v in members)
    got = _ask(retail, {"intent": "breakdown", "pareto": {"of": "product", "measure": "net_amount"},
                        "time": {"window": Y2025}})
    rows = {r["class"]: r for r in got["export_rows"]}
    assert list(rows) == ["A", "B", "C"]
    for c in "ABC":
        mine = [v for (_, v), k in zip(members, classes) if k == c]
        assert rows[c]["products"] == len(mine)
        assert float(rows[c]["net_amount"]) == pytest.approx(sum(mine))
        assert float(rows[c]["share"]) == pytest.approx(sum(mine) * 100 / total)
    a = classes.count("A")
    assert got["answer"]["headline"].startswith(f"A: {a} of {len(members)} products ({a / len(members):.0%}) made "
                                                f"{float(rows['A']['share']):.0f}% of net amount in 2025; B: ")
    assert "A the products making the first 80%; B the products making the next 15%; C the rest" in got["definition"]


def test_each_product_with_its_class_and_running_share(retail):
    members = _per_member(retail, "product_id", "products", "product_name")
    classes = _classes([v for _, v in members], [80, 95])
    got = _ask(retail, {"intent": "rank", "pareto": {"of": "product", "measure": "net_amount", "members": True},
                        "time": {"window": Y2025}})
    rows = got["export_rows"]
    assert [r["product_name"] for r in rows] == [n for n, _ in members]
    assert [r["class"] for r in rows] == classes
    assert float(rows[-1]["running_share"]) == pytest.approx(100.0)
    assert got["answer"]["headline"].startswith(f"{classes.count('A')} of {len(members)} products are class A")


def test_the_customers_making_the_first_80_percent(retail):
    members = _per_member(retail, "customer_id", "customers", "customer_name")
    classes = _classes([v for _, v in members], [80])
    first = [n for (n, _), c in zip(members, classes) if c == "A"]
    got = _ask(retail, {"intent": "rank", "pareto": {"of": "customer", "measure": "net_amount", "mode": "pareto",
                                                     "cuts": [80]}, "time": {"window": Y2025}})
    assert [r["customer_name"] for r in got["export_rows"]] == first
    assert got["answer"]["headline"].startswith(f"{len(first)} of {len(members)} customers made the first 80% of "
                                                f"net amount in 2025; the largest, {members[0][0]}")
    assert "the one that takes it past 80% included" in got["definition"]



def test_the_first_80_percent_is_counted_in_full_when_the_list_is_cut(retail, monkeypatch):
    """Only the first rows of a long list are read back: the count of members making the first 80% is the
    warehouse's, never the number of rows that came back."""
    import core2.service as service

    members = _per_member(retail, "customer_id", "customers", "customer_name")
    first = [n for (n, _), c in zip(members, _classes([v for _, v in members], [80])) if c == "A"]
    assert len(first) > 3
    monkeypatch.setattr(service, "MAX_ROWS", 3)
    got = _ask(retail, {"intent": "rank", "pareto": {"of": "customer", "measure": "net_amount", "mode": "pareto",
                                                     "cuts": [80]}, "time": {"window": Y2025}})
    assert len(got["export_rows"]) == 3
    assert got["answer"]["headline"].startswith(f"{len(first)} of {len(members)} customers made the first 80% of ")

def test_customers_by_decile(retail):
    members = _per_member(retail, "customer_id", "customers", "customer_name")
    n = len(members)
    # NTILE: the first n % 10 groups hold one more.
    sizes = [n // 10 + (1 if i < n % 10 else 0) for i in range(10)]
    total = sum(v for _, v in members)
    got = _ask(retail, {"intent": "breakdown", "pareto": {"of": "customer", "measure": "net_amount", "mode": "ntile"},
                        "time": {"window": Y2025}})
    rows = got["export_rows"]
    assert [int(r["decile"]) for r in rows] == list(range(1, 11))
    assert [r["customers"] for r in rows] == sizes
    top = sum(v for _, v in members[:sizes[0]])
    assert float(rows[0]["share"]) == pytest.approx(top * 100 / total)
    assert got["answer"]["headline"].startswith(f"The top decile of customers ({sizes[0]} of {n}) made")


# ── funnels ────────────────────────────────────────────────────────────────


def test_order_lines_through_ship_and_delivery(retail):
    # A date far back or far ahead (1900, 9999) stands for none yet: not a stage reached.
    (first, shipped, delivered, to_ship, to_deliver), = _sql(retail, f"""
        WITH r AS (SELECT l.ordered, CASE WHEN s.full_date > DATE '1901-01-01' AND s.full_date < DATE '9000-01-01'
                                          THEN s.full_date END AS shipped,
                          CASE WHEN d.full_date > DATE '1901-01-01' AND d.full_date < DATE '9000-01-01'
                               THEN d.full_date END AS delivered
                   FROM ({LINES}) l LEFT JOIN calendar s ON l.ship_date_key = s.date_key
                                    LEFT JOIN calendar d ON l.delivered_date_key = d.date_key)
        SELECT COUNT(*), COUNT(shipped), COUNT(CASE WHEN shipped IS NOT NULL THEN delivered END),
               AVG(date_diff('day', ordered, shipped)),
               AVG(CASE WHEN shipped IS NOT NULL THEN date_diff('day', shipped, delivered) END) FROM r""")
    got = _ask(retail, {"intent": "breakdown", "funnel": {"stages": ["order_date", "ship_date", "delivered_date"]},
                        "time": {"window": Y2025}})
    rows = got["export_rows"]
    assert [r["stage"] for r in rows] == ["Order date", "Ship date", "Delivered date"]
    assert [r["reached"] for r in rows] == [first, shipped, delivered]
    assert float(rows[1]["of_first"]) == pytest.approx(shipped * 100 / first)
    assert float(rows[2]["of_previous"]) == pytest.approx(delivered * 100 / shipped)
    assert rows[0]["of_previous"] is None and rows[0]["days"] is None
    assert float(rows[1]["days"]) == pytest.approx(float(to_ship))
    assert float(rows[2]["days"]) == pytest.approx(float(to_deliver))
    assert got["answer"]["headline"].startswith(f"Of {first:,} order lines with an order date in 2025, "
                                                f"{delivered:,} ({delivered * 100 / first:.0f}%) reached delivered date")
    assert "a stage is reached when its date and every one before it are filled" in got["definition"]


# ── bought together ────────────────────────────────────────────────────────

PAIRS = f"""WITH b AS (SELECT DISTINCT p.product_name AS item, l.order_number AS basket
                     FROM ({LINES}) l JOIN products p ON l.product_id = p.product_id),
              n AS (SELECT COUNT(DISTINCT basket) AS n FROM b),
              i AS (SELECT item, COUNT(*) AS n FROM b GROUP BY item)
         SELECT a.item, o.item, COUNT(*) AS together, COUNT(*) * 100.0 / MAX(n.n),
                COUNT(*) * 100.0 / MAX(ia.n), COUNT(*) * MAX(n.n) * 1.0 / (MAX(ia.n) * MAX(io.n))
         FROM b a JOIN b o ON a.basket = o.basket AND {{pair}}
         JOIN i ia ON ia.item = a.item JOIN i io ON io.item = o.item CROSS JOIN n
         GROUP BY a.item, o.item ORDER BY together DESC, a.item, o.item"""


def test_products_bought_together(retail):
    pairs = _sql(retail, PAIRS.format(pair="a.item < o.item"))
    got = _ask(retail, {"intent": "rank", "together": {"of": "product", "basket": "number_of_orders"},
                        "time": {"window": Y2025}, "limit": 10})
    rows = got["export_rows"]
    assert len(rows) == 10
    want = {(a, b): (n, s, c, lift) for a, b, n, s, c, lift in pairs}
    top = pairs[0][2]
    assert rows[0]["together"] == top
    for r in rows:
        n, support, confidence, lift = want[(r["item"], r["with_item"])]
        assert r["together"] == n
        assert float(r["support"]) == pytest.approx(float(support), rel=1e-4)       # shown to 5 figures
        assert float(r["confidence"]) == pytest.approx(float(confidence), rel=1e-4)
        assert float(r["lift"]) == pytest.approx(float(lift), rel=1e-4)
    assert got["answer"]["headline"].startswith(f"Bought together most often in 2025: {rows[0]['item']} and "
                                                f"{rows[0]['with_item']}, on {top} orders")
    assert "the lift (above 1" in got["definition"]


def test_what_is_bought_with_one_product(retail):
    built, model = retail
    (anchor,), = _sql(retail, f"SELECT p.product_name FROM ({LINES}) l JOIN products p USING (product_id) "
                              "GROUP BY 1 ORDER BY COUNT(DISTINCT l.order_number) DESC, 1 LIMIT 1")
    pairs = _sql(retail, PAIRS.format(pair=f"a.item = '{anchor}' AND o.item <> a.item"))
    got = _ask(retail, {"intent": "rank", "together": {"of": "product", "basket": "number_of_orders"},
                        "filters": [{"field": "product", "op": "eq", "values": [anchor]}],
                        "time": {"window": Y2025}, "limit": 5}, question=f'what is bought with "{anchor}" in 2025')
    rows = got["export_rows"]
    assert {r["item"] for r in rows} == {anchor}
    assert [r["together"] for r in rows] == [n for _, _, n, *_ in pairs[:5]]
    assert got["answer"]["headline"].startswith(f"Bought most often with {anchor} in 2025:")


# ── against target ─────────────────────────────────────────────────────────


def test_net_amount_against_target_by_store(retail):
    want = _sql(retail, f"""
        SELECT s.store_name, a.net, t.tgt FROM stores s
        JOIN (SELECT store_id, SUM(net_amount) AS net FROM ({LINES}) GROUP BY 1) a USING (store_id)
        JOIN (SELECT store_id, SUM(target_amount) AS tgt FROM sales_targets
              WHERE period_key BETWEEN 202501 AND 202512 GROUP BY 1) t USING (store_id)""")
    got = _ask(retail, {"intent": "breakdown", "target": {"actual": "net_amount", "target": "target_amount"},
                        "group_by": ["store.name"], "time": {"window": Y2025}})
    rows = {r["store_name"]: r for r in got["export_rows"]}
    for store, net, tgt in want:
        r = rows[store]
        assert float(r["net_amount"]) == pytest.approx(float(net))
        assert float(r["target_amount"]) == pytest.approx(float(tgt))
        assert float(r["difference"]) == pytest.approx(float(net) - float(tgt))
        assert float(r["of_target"]) == pytest.approx(float(net) * 100 / float(tgt))
    hits = sum(1 for _, net, tgt in want if net >= tgt)
    best = max(want, key=lambda w: w[1] / w[2])
    assert got["answer"]["headline"].startswith(f"{hits} of {len(want)} stores reached their target amount in 2025; "
                                                f"{best[0]} is furthest above it")
    assert got["definition"] == ("% of target: net amount as a share of target amount (100% or more is reached); "
                                 "difference: net amount minus target amount.")


def test_by_month_a_month_with_no_figure_yet_is_not_one_that_missed(retail):
    got = _ask(retail, {"intent": "trend", "target": {"actual": "net_amount", "target": "target_amount"},
                        "time": {"grain": "month", "window": {"kind": "this", "unit": "year"}}})
    months = [r for r in got["export_rows"] if r["net_amount"] is not None]
    assert len(months) == 6                     # the year's targets run to December; the figures to June
    # June is still under way: said to be partly covered, and not compared as a whole month.
    whole = [r for r in months if not str(r["period"]).startswith("2026-06")]
    assert f"in {sum(1 for r in whole if float(r['of_target']) >= 100)} of 5 months" in got["answer"]["headline"]
    assert any("Jun 2026 is only partly covered" in c for c in got["coverage_caveats"])


def test_a_quarter_covered_in_part_is_not_compared(retail):
    """A quarter the window (or the data) covers only in part is not compared as a whole one: from mid-February,
    the first quarter is neither the best nor the lowest, nor counted."""
    got = _ask(retail, {"intent": "trend", "target": {"actual": "net_amount", "target": "target_amount"},
                        "time": {"grain": "quarter", "window": {"kind": "between", "start": "2025-02-15",
                                                                "end": "2025-12-31"}}})
    rows = got["export_rows"]
    assert str(rows[0]["period"]).startswith("2025-01-01") and rows[0]["of_target"] is not None
    whole = rows[1:]
    hits = sum(1 for r in whole if float(r["of_target"]) >= 100)
    assert f"in {hits} of 3 quarters" in got["answer"]["headline"]
    assert "Q1 2025" not in got["answer"]["headline"]


def test_classes_stages_and_pairs_offer_no_member_to_open(retail):
    """A decile, a class, a stage or a pair is no member a question can name: no "Monthly ... for "1"" chip,
    no findings counted over its rows, no click on its bars; a bar per pair of products, named by both."""
    deciles = _ask(retail, {"intent": "breakdown", "pareto": {"of": "customer", "measure": "net_amount",
                                                              "mode": "ntile"}, "time": {"window": Y2025}})
    assert deciles["key_insights"] == []
    assert not [s for s in deciles["follow_up_suggestions"] if "for \"" in s["question"]]
    assert not (deciles.get("chart") or {}).get("drill")
    pairs = _ask(retail, {"intent": "rank", "together": {"of": "product", "basket": "number_of_orders"},
                          "time": {"window": Y2025}, "limit": 5})
    rows, chart = pairs["export_rows"], pairs["chart"]
    assert chart["y_keys"] == ["together"] and chart["x_pair"]
    assert [r[chart["x_key"]] for r in chart["rows"]] == [f"{r['item']} + {r['with_item']}" for r in rows]
    assert pairs["key_insights"] == [] and not chart.get("drill")


# ── every warehouse, and what cannot be asked ──────────────────────────────

PLANS = {
    "abc": {"pareto": {"of": "product", "measure": "net_amount"}},
    "deciles": {"pareto": {"of": "customer", "measure": "net_amount", "mode": "ntile", "members": True}},
    "funnel": {"funnel": {"stages": ["order_date", "ship_date", "delivered_date"]}},
    "together": {"together": {"of": "product", "basket": "number_of_orders"}},
    "target": {"target": {"actual": "net_amount", "target": "target_amount"}, "time": {"grain": "month"}},
}


@pytest.mark.parametrize("name", sorted(PLANS))
@pytest.mark.parametrize("dialect", ["tsql", "oracle", "snowflake", "duckdb"])
def test_each_compiles_on_every_warehouse(retail, name, dialect):
    _, model = retail
    logical = resolve(Plan.model_validate({"kind": "query", **PLANS[name]}), model, Context(today=TODAY))
    sql = compile_query(logical, model, dialect).sql
    sqlglot.parse_one(sql, read=dialect)


@pytest.mark.parametrize("plan, words", [
    ({"pareto": {"of": "product", "measure": "net_amount"}, "group_by": ["store.name"]}, "every member at once"),
    ({"pareto": {"of": "product", "measure": "margin_percent"}}, "is not a total"),
    ({"funnel": {"stages": ["order_date", "ship_date"]}, "group_by": ["store.name"]}, "all together"),
    ({"funnel": {"stages": ["order_date", "order_date"]}}, "a date of its own"),
    ({"together": {"of": "product", "basket": "net_amount"}}, "does not count baskets"),
    ({"target": {"actual": "net_amount", "target": "net_amount"}}, "needs two figures"),
])
def test_what_cannot_be_asked_is_told_why(retail, plan, words):
    _, model = retail
    with pytest.raises(ResolveError) as told:
        resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    assert words in told.value.message


def test_learn_offers_them_and_the_planner_is_shown_how(retail):
    _, model = retail
    offered = dict(suggestions(model, this_year=2026))
    assert "Customers making 80% of net amount in 2026" in offered
    assert "Order lines through order date, ship date and delivered date in 2026" in offered
    assert "Products bought together in 2026" in offered
    assert "Net amount against target amount by month in 2025" in offered      # the last whole year of targets
    prompt = stable_prompt(model)
    for words in ("- pareto:", "- funnel:", "- together:", "- target:", "Q: ABC analysis of customers by net amount",
                  "Q: which products are bought together"):
        assert words in prompt, words
    hr = learn(domains.build("hr"), "descriptive")[1]
    assert not any(k in plan for _, plan in suggestions(hr, this_year=2026)
                   for k in ("pareto", "funnel", "together", "target"))     # no employees by their pay


def test_the_sales_dashboard_shows_net_amount_against_its_target(retail):
    from core2 import recipes

    name, tiles = recipes.executive(retail[1], today=TODAY)
    tile = next(t for t in tiles if t.title == "Net amount against target amount by month")
    assert tile.tab == "Sales target" and tile.plan["target"] == {"actual": "net_amount", "target": "target_amount"}


def test_a_designed_tile_keeps_the_analysis_its_title_names():
    """A dashboard designed from a list: a tile whose question lost the analysis its title asks for ("ABC classes
    of products" asked as "net amount by product class", a funnel asked as a breakdown by two dates) is asked in
    its title's words; a plain tile keeps its question (with the period the designer gave it)."""
    from core2.dashboards import _tiles

    tiles = _tiles([
        {"title": "ABC classes of products by net amount", "kind": "breakdown",
         "question": "Net amount by product class"},
        {"title": "Order funnel from order date to delivered date", "kind": "breakdown",
         "question": "Number of orders by order date and delivered date"},
        {"title": "Net amount against target by month", "kind": "trend",
         "question": "Net amount and target amount by month"},
        {"title": "Products bought together", "kind": "table", "question": "Products bought together on orders"},
        {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month for the last 12 months"},
    ], 16)
    assert [t.question for t in tiles] == [
        "ABC classes of products by net amount", "Order funnel from order date to delivered date",
        "Net amount against target by month", "Products bought together on orders",
        "Net amount by month for the last 12 months"]


def test_a_month_with_no_figure_yet_has_no_difference(retail):
    """Targets set ahead: a month with a target and no figure yet has no difference (never minus its whole target),
    and no % of target."""
    got = _ask(retail, {"intent": "trend", "target": {"actual": "net_amount", "target": "target_amount"},
                        "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01",
                                                                     "end": "2026-12-31"}}})
    ahead = [r for r in got["export_rows"] if r["net_amount"] is None]
    assert ahead and all(r["target_amount"] is not None for r in ahead)
    assert all(r["difference"] is None and r["of_target"] is None for r in ahead)
    known = [r for r in got["export_rows"] if r["net_amount"] is not None]
    assert all(float(r["difference"]) == pytest.approx(float(r["net_amount"]) - float(r["target_amount"]))
               for r in known)


def test_a_funnel_is_drawn_as_how_many_reached_each_stage(retail):
    got = _ask(retail, {"intent": "breakdown", "funnel": {"stages": ["order_date", "ship_date", "delivered_date"]},
                        "time": {"window": Y2025}})
    chart = got["chart"]
    assert chart["y_keys"] == ["reached"] and chart["share_key"] == "of_first"
    assert "pie" not in chart["allowed_types"] and "donut" not in chart["renderable_types"]
    assert [r[chart["x_key"]] for r in chart["rows"]] == ["Order date", "Ship date", "Delivered date"]


def test_twenty_pairs_are_twenty_bars_never_the_other_products(retail):
    got = _ask(retail, {"intent": "rank", "together": {"of": "product", "basket": "number_of_orders"},
                        "time": {"window": Y2025}})
    chart = got["chart"]
    assert len(got["export_rows"]) == len(chart["rows"]) == 20 and chart["other"] is None
    assert chart["title"].startswith("Orders with both by pair of products")


def test_a_designed_question_names_fields_and_dates_in_words(retail):
    """The design wrote catalog names into a tile's question ("by customer.segment", "from order_date to
    ship_date"): asked by those names, a period or a field the design meant for the whole tile is read as one
    table's column. They are asked in words; a measure keeps its name (it says whose figure it is)."""
    from core2.dashboards import design

    built, model = retail
    reply = json.dumps({"name": "Orders", "tiles": [
        {"title": "Net amount by segment", "kind": "breakdown", "question": "net_amount by customer.segment"},
        {"title": "Order funnel", "kind": "breakdown", "question": "Funnel of orders from order_date to ship_date"},
        {"title": "Net amount by month", "kind": "trend", "question": "Net amount by month"}]})
    name, tiles = design(model, "net amount by segment, the order funnel and net amount by month",
                         lambda stable, tail: reply)
    assert [t.question for t in tiles] == [
        f"net_amount by {model.attributes['customer.segment'].business_name.lower()}",
        "Funnel of orders from order date to ship date", "Net amount by month"]


def test_against_target_by_the_target_tables_own_quarter_is_by_quarter(retail):
    """A target kept a row a period, with its quarter as a column of its own ("by quarter" read as that column): the
    figure has no such column, so both are counted by quarter, each by its own date. Any other grouping stays."""
    from core2.model.schema import Attribute
    from core2.resolve.shapes import with_target

    built, model = retail
    quarter = Attribute(slug="sales_target.quarter", column="memory.main.sales_targets.period_key",
                        business_name="Quarter")
    year = Attribute(slug="sales_target.year", column="memory.main.sales_targets.period_key", business_name="Year")
    model = model.model_copy(update={"attributes": {**model.attributes, quarter.slug: quarter, year.slug: year}})
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                "target": {"actual": "net_amount", "target": "target_amount"},
                                "group_by": ["sales_target.year", "sales_target.quarter", "store.name"]})
    made, _ = with_target(plan, model)
    assert made.time.grain == "quarter" and made.group_by == ["store.name"]
    plain = Plan.model_validate({"kind": "query", "intent": "breakdown", "group_by": ["store.name"],
                                 "target": {"actual": "net_amount", "target": "target_amount"}})
    assert with_target(plain, model)[0].time.grain is None
