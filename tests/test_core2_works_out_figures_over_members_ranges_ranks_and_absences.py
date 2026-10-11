"""Figures worked out over the rows of a plain answer, each equal to a query written by hand.

"Average orders per customer", "median spend per customer", "how many customers had 3 or more orders", "90th
percentile days to deliver", "customers by spend band", "orders by delivery time", "top 3 products in each store",
"running total this year", "3-month moving average", "customers with no order in the last 90 days": each is asked
of the invented retail and pharmacy warehouses through the whole answer path (a scripted plan, as the planner
writes it), and its figures are compared with SQL written here against the raw tables. Each says what it worked
out, and its SQL means the same on Snowflake, SQL Server and Oracle, without SQL Server's pitfalls.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal

import pytest
import sqlglot
from sqlglot import exp

from core2.compile.compiler import compile_query
from core2.compile.shapes import fill_edges
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}
ORDERS_Q1 = ("FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key "
             "WHERE d.full_date >= DATE '2026-01-01' AND d.full_date < DATE '2026-04-01'")
DELIVERED = ("CASE WHEN dd.full_date >= DATE '1901-01-01' AND dd.full_date < DATE '9000-01-01' "
             "THEN date_diff('day', d.full_date, dd.full_date) END")


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.fixture(scope="module")
def pharmacy():
    return learn(domains.build("compounding_pharmacy"), "descriptive")


def _ask(learned, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _sql(learned, sql: str) -> list[tuple]:
    return learned[0].con.execute(sql).fetchall()


def _only(payload: dict) -> float:
    rows = payload["export_rows"]
    assert len(rows) == 1
    return float(next(iter(rows[0].values())))


# ── per member, then over the members ──────────────────────────────────────


def test_average_orders_per_customer_over_those_with_orders_and_over_everyone(retail):
    per = f"SELECT o.customer_id, COUNT(DISTINCT o.order_number) AS n {ORDERS_Q1} " \
          "AND o.customer_id IN (SELECT customer_id FROM customers) GROUP BY 1"
    (average, counted), = _sql(retail, f"SELECT AVG(n), COUNT(*) FROM ({per})")
    (listed,), = _sql(retail, "SELECT COUNT(*) FROM customers")
    (total,), = _sql(retail, f"SELECT SUM(n) FROM ({per})")

    with_orders = _ask(retail, {"measures": ["number_of_orders"], "per": {"of": "customer"}, "time": Q1})
    assert _only(with_orders) == pytest.approx(average)
    assert with_orders["answer"]["headline"].startswith("Average number of orders per customer in Q1 2026")
    assert f"the {counted:,} customers with orders" in with_orders["definition"]

    everyone = _ask(retail, {"measures": ["number_of_orders"], "per": {"of": "customer", "everyone": True},
                             "time": Q1})
    assert listed > counted and _only(everyone) == pytest.approx(total / listed)
    assert f"all {listed:,} customers listed (one with none counted as 0)" in everyone["definition"]
    assert "counting every customer listed" in everyone["answer"]["headline"]


@pytest.mark.parametrize("summary, p, quantile", [("median", None, 0.5), ("percentile", 75, 0.75),
                                                   ("min", None, None), ("max", None, None)])
def test_each_customers_spend_summarised_in_each_segment(retail, summary, p, quantile):
    per = f"SELECT c.segment, o.customer_id, SUM(o.net_amount) AS s {ORDERS_Q1.replace('WHERE', 'JOIN customers c ON o.customer_id = c.customer_id WHERE')} GROUP BY 1, 2"
    how = f"quantile_cont(s, {quantile})" if quantile else f"{summary.upper()}(s)"
    expected = dict(_sql(retail, f"SELECT segment, {how} FROM ({per}) GROUP BY 1"))
    got = _ask(retail, {"measures": ["net_amount"], "group_by": ["customer.segment"],
                        "per": {"of": "customer", "summary": summary, **({"p": p} if p else {})}, "time": Q1})
    shown = {r["customer_segment"]: r[next(k for k in r if k != "customer_segment")] for r in got["export_rows"]}
    assert set(shown) == set(expected)
    for segment, value in expected.items():
        assert float(shown[segment]) == pytest.approx(float(value), abs=0.01)


def test_customers_with_three_or_more_orders_counted_and_as_a_share_of_everyone(retail):
    per = f"SELECT o.customer_id, COUNT(DISTINCT o.order_number) AS n {ORDERS_Q1} " \
          "AND o.customer_id IN (SELECT customer_id FROM customers) GROUP BY 1"
    (three, counted), = _sql(retail, f"SELECT COUNT(*) FILTER (WHERE n >= 3), COUNT(*) FROM ({per})")
    (none_or_few,), = _sql(retail, f"SELECT COUNT(*) FILTER (WHERE n < 2) FROM ({per})")
    (listed,), = _sql(retail, "SELECT COUNT(*) FROM customers")

    count = _ask(retail, {"measures": ["number_of_orders"],
                          "per": {"of": "customer", "summary": "count", "op": "gte", "values": [3]}, "time": Q1})
    assert _only(count) == three
    assert count["answer"]["headline"] == f"Customers with 3 or more orders in Q1 2026: {three}."
    share = _ask(retail, {"measures": ["number_of_orders"], "time": Q1, "per": {
        "of": "customer", "summary": "share", "op": "gte", "values": [3], "everyone": True}})
    assert _only(share) == pytest.approx(100 * three / listed)
    # Fewer than 2 orders, over everyone: a customer with no orders at all has fewer than 2 too.
    few = _ask(retail, {"measures": ["number_of_orders"], "time": Q1, "per": {
        "of": "customer", "summary": "count", "op": "lt", "values": [2], "everyone": True}})
    assert _only(few) == none_or_few + (listed - counted)


def test_ninetieth_percentile_days_to_deliver_each_line_by_store(retail):
    expected = dict(_sql(retail, f"""
        SELECT s.store_name, quantile_cont({DELIVERED}, 0.9)
        FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
        LEFT JOIN calendar dd ON o.delivered_date_key = dd.date_key LEFT JOIN stores s ON o.store_id = s.store_id
        WHERE d.full_date >= DATE '2026-01-01' AND d.full_date < DATE '2026-04-01' GROUP BY 1"""))
    got = _ask(retail, {"durations": [{"name": "Days from order to delivery", "start": "order_date",
                                       "end": "delivered_date"}], "group_by": ["store.name"],
                        "per": {"summary": "percentile", "p": 90}, "time": Q1})
    shown = {r["store_name"]: r["percentile_days_from_order_to_delivery"] for r in got["export_rows"]}
    assert {k for k in expected if k} == {k for k in shown if k != "Unknown"}
    for store, value in expected.items():
        assert float(shown[store or "Unknown"]) == pytest.approx(float(value))
    assert "90th percentile of days from order to delivery" in got["answer"]["headline"]
    assert got["definition"].startswith("The days from order to delivery of each order line in Q1 2026, then the "
                                        "value 90% of them are at or below")


def test_average_order_value_and_fills_per_prescription(retail, pharmacy):
    (order_value,), = _sql(retail, f"SELECT AVG(s) FROM (SELECT o.order_number, SUM(o.net_amount) AS s {ORDERS_Q1} "
                                   "GROUP BY 1)")
    got = _ask(retail, {"measures": ["net_amount"], "per": {"of": "number_of_orders"}, "time": Q1})
    assert _only(got) == pytest.approx(float(order_value))
    assert got["answer"]["headline"].startswith("Average net amount per order in Q1 2026")

    (fills,), = _sql(pharmacy, "SELECT AVG(n) FROM (SELECT rx_number, COUNT(*) AS n FROM fills GROUP BY 1)")
    got = _ask(pharmacy, {"measures": ["number_of_fills"], "per": {"of": "number_of_prescriptions"}})
    assert _only(got) == pytest.approx(float(fills))
    (patients,), = _sql(pharmacy, "SELECT AVG(n) FROM (SELECT r.patient_id, COUNT(*) AS n FROM fills f JOIN "
                                  "prescriptions r ON f.rx_number = r.rx_number JOIN patients p ON "
                                  "r.patient_id = p.patient_id GROUP BY 1)")
    got = _ask(pharmacy, {"measures": ["number_of_fills"], "per": {"of": "patient"}})
    assert _only(got) == pytest.approx(float(patients))


# ── ranges ─────────────────────────────────────────────────────────────────


def test_customers_by_spend_range_and_orders_by_delivery_time(retail):
    per = f"SELECT o.customer_id, SUM(o.net_amount) AS s {ORDERS_Q1} " \
          "AND o.customer_id IN (SELECT customer_id FROM customers) GROUP BY 1"
    expected = dict(_sql(retail, f"""SELECT CASE WHEN s < 500 THEN 'Under $500' WHEN s < 1000 THEN
        '$500 to under $1,000' WHEN s < 2500 THEN '$1,000 to under $2,500' ELSE '$2,500 and over' END, COUNT(*)
        FROM ({per}) GROUP BY 1"""))
    got = _ask(retail, {"measures": ["net_amount"], "bands": {"of": "net_amount", "per": "customer",
                                                              "edges": [2500, 500, 1000]}, "time": Q1})
    assert [r["band"] for r in got["export_rows"]] == ["Under $500", "$500 to under $1,000",
                                                        "$1,000 to under $2,500", "$2,500 and over"]
    assert {r["band"]: r["customers"] for r in got["export_rows"]} == expected
    most = max(expected, key=expected.get)
    assert got["answer"]["headline"].startswith("Customers in Q1 2026 by net amount per customer: the most, "
                                                f"{expected[most]} (") and got["answer"]["headline"].endswith(
        f"in {most}.")

    days = dict(_sql(retail, f"""SELECT CASE WHEN {DELIVERED} IS NULL THEN 'Unknown' WHEN {DELIVERED} < 3
        THEN 'Under 3 days' WHEN {DELIVERED} < 8 THEN '3–7 days' ELSE '8 days and over' END,
        COUNT(DISTINCT o.order_number) FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key
        LEFT JOIN calendar dd ON o.delivered_date_key = dd.date_key
        WHERE d.full_date >= DATE '2026-01-01' AND d.full_date < DATE '2026-04-01' GROUP BY 1"""))
    got = _ask(retail, {"measures": ["number_of_orders"], "durations": [
        {"name": "Days from order to delivery", "start": "order_date", "end": "delivered_date", "measure": False}],
        "bands": {"of": "Days from order to delivery", "edges": [3, 8]}, "time": Q1})
    assert {r["band"]: r["number_of_orders"] for r in got["export_rows"]} == days
    assert [r["band"] for r in got["export_rows"]][:3] == ["Under 3 days", "3–7 days", "8 days and over"]
    assert f"{days['Unknown']} orders have no value for it" in got["answer"]["headline"]


def test_ranges_named_by_no_one_are_round_and_split_the_data(retail):
    got = _ask(retail, {"measures": ["net_amount"], "bands": {"of": "net_amount", "per": "customer"}, "time": Q1})
    names = [r["band"] for r in got["export_rows"]]
    assert 3 <= len(names) <= 6 and names[0].startswith("Under $") and names[-1].endswith(" and over")
    edges = [float(n.split("Under $")[1].replace(",", "")) for n in names[:1]] + \
        [float(n.split(" to under $")[0].lstrip("$").replace(",", "")) for n in names[1:-1]]
    for edge in edges:          # 1, 2, 2.5 or 5 times a power of ten
        lead = edge / 10 ** (len(str(int(edge))) - 1)
        assert lead in (1, 2, 2.5, 5), edge
    assert max(r["customers"] for r in got["export_rows"]) < 0.6 * sum(r["customers"] for r in got["export_rows"])


# ── a rank inside each member of a grouping ────────────────────────────────


def test_the_first_three_products_in_each_store(retail):
    expected = _sql(retail, f"""SELECT store_name, product_name, s, rn FROM (
        SELECT s.store_name, p.product_name, SUM(o.net_amount) AS s, ROW_NUMBER() OVER (PARTITION BY s.store_name
        ORDER BY SUM(o.net_amount) DESC, p.product_name) AS rn
        {ORDERS_Q1.replace('WHERE', 'JOIN stores s ON o.store_id = s.store_id JOIN products p ON o.product_id = p.product_id WHERE')}
        GROUP BY 1, 2) WHERE rn <= 3 ORDER BY 1, 4""")
    got = _ask(retail, {"intent": "rank", "measures": ["net_amount"], "group_by": ["store.name", "product.name"],
                        "within": {"group": "store.name", "top": 3}, "time": Q1})
    rows = [(r["store_name"], r["product_name"], r["net_amount"], r["rank"]) for r in got["export_rows"]]
    assert [(a, b, d) for a, b, _, d in rows] == [(a, b, d) for a, b, _, d in expected]
    assert [c for *_, c, _ in rows] == pytest.approx([float(c) for *_, c, _ in expected])
    first = expected[0]
    assert f"{first[0]}: {first[1]} (" in got["answer"]["headline"]
    assert "the first 3 products in each store" in got["answer"]["headline"]


# ── a series accumulated or smoothed ───────────────────────────────────────


MONTHLY = ("SELECT date_trunc('month', d.full_date) AS m, SUM(o.net_amount) AS s FROM order_lines o JOIN calendar d "
           "ON o.order_date_key = d.date_key WHERE d.full_date >= DATE '{start}' AND d.full_date < DATE '2026-06-01' "
           "GROUP BY 1 ORDER BY 1")


def test_running_total_this_year_and_a_three_month_moving_average(retail):
    months = [float(s) for _, s in _sql(retail, MONTHLY.format(start="2025-11-01"))]
    assert len(months) == 7
    running = _ask(retail, {"intent": "trend", "measures": ["net_amount"], "rolling": {"kind": "running", "reset": "year"},
                            "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01",
                                                                   "end": "2026-05-31"}}})
    rows = running["export_rows"]
    assert [r["period"] for r in rows] == ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"]
    assert [r["net_amount_running"] for r in rows] == pytest.approx(
        [sum(months[2:3 + i]) for i in range(5)])
    assert [r["net_amount"] for r in rows] == pytest.approx(months[2:])
    assert running["data"]["header_labels"]["net_amount_running"] == "Net amount, year to date"
    assert running["definition"] == "Each month adds to the total since the start of each year."
    assert running["key_insights"] == []                     # a running total only grows: nothing to find in that

    moving = _ask(retail, {"intent": "trend", "measures": ["net_amount"],
                           "rolling": {"kind": "moving_average", "periods": 3},
                           "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01",
                                                                  "end": "2026-05-31"}}})
    assert [r["net_amount_moving_average"] for r in moving["export_rows"]] == pytest.approx(
        [sum(months[i:i + 3]) / 3 for i in range(5)])      # January's average reads November and December
    assert "2025-11-01" in moving["trust"]["sql"]


def test_a_month_with_no_rows_adds_nothing_and_a_short_start_has_no_average_yet():
    from core2.answer.shaped import _work
    from core2.resolve.shapes import RollingShape
    from core2.resolve.time import Range

    days = [dt.date(2026, m, 1) for m in range(1, 6)]
    shape = RollingShape(kind="moving_sum", periods=2, reset=None, grain="month", shown=Range(None, None),
                         read=Range(None, None), columns={}, labels={}, adds={})
    assert _work(shape, [5.0, 0.0, 3.0, 4.0, 0.0], days, 1) == [None, 5.0, 3.0, 7.0, 4.0]
    shape.kind = "moving_average"           # a level with a missing month: averaged over the months it has
    assert _work(shape, [5.0, None, 3.0, None, None], days, 1) == [None, 5.0, 3.0, 3.0, None]
    shape.kind, shape.reset = "running", "fiscal_year"
    assert _work(shape, [1.0, 2.0, 3.0, 4.0, 5.0], days, 4) == [1.0, 3.0, 6.0, 4.0, 9.0]   # starts again in April


# ── who had none ───────────────────────────────────────────────────────────


def test_customers_with_no_order_in_the_last_ninety_days(retail):
    expected = sorted(n for n, in _sql(retail, """SELECT c.customer_name FROM customers c WHERE NOT EXISTS (
        SELECT 1 FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key WHERE o.customer_id =
        c.customer_id AND d.full_date >= DATE '2026-03-17' AND d.full_date < DATE '2026-06-15')"""))
    got = _ask(retail, {"intent": "list", "absent": {"of": "customer", "measure": "number_of_orders"},
                        "time": {"window": {"kind": "last", "n": 90, "unit": "day"}}})
    assert sorted(r["customer_name"] for r in got["export_rows"]) == expected and expected
    assert got["answer"]["headline"] == f"{len(expected)} customers had no orders from 17 Mar 2026 to 14 Jun 2026."
    assert got["definition"].startswith("Listed: every customer listed, minus those with orders from 17 Mar 2026")

    # Who bought at one store before, and had nothing there in the last 90 days.
    store = "Airport Store"
    expected = sorted(n for n, in _sql(retail, f"""SELECT c.customer_name FROM customers c WHERE NOT EXISTS (
        SELECT 1 FROM order_lines o JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON
        o.store_id = s.store_id WHERE o.customer_id = c.customer_id AND s.store_name = '{store}' AND
        d.full_date >= DATE '2026-03-17' AND d.full_date < DATE '2026-06-15') AND EXISTS (SELECT 1 FROM
        order_lines o JOIN stores s ON o.store_id = s.store_id WHERE o.customer_id = c.customer_id AND
        s.store_name = '{store}')"""))
    got = _ask(retail, {"intent": "list", "time": {"window": {"kind": "last", "n": 90, "unit": "day"}}, "absent": {
        "of": "customer", "measure": "number_of_orders",
        "without": [{"field": "store.name", "op": "eq", "values": [store]}],
        "had": [{"field": "store.name", "op": "eq", "values": [store]}]}})
    assert sorted(r["customer_name"] for r in got["export_rows"]) == expected and expected


def test_a_list_with_no_one_missing_says_everyone_had_some(pharmacy):
    (missing,), = _sql(pharmacy, """SELECT COUNT(*) FROM prescribers p WHERE NOT EXISTS (SELECT 1 FROM fills f
        JOIN prescriptions r ON f.rx_number = r.rx_number WHERE r.prescriber_npi = p.prescriber_npi AND
        f.fill_date >= DATE '2026-03-17' AND f.fill_date < DATE '2026-06-15')""")
    got = _ask(pharmacy, {"intent": "list", "absent": {"of": "prescriber", "measure": "number_of_fills"},
                          "time": {"window": {"kind": "last", "n": 90, "unit": "day"}}})
    assert missing == 0 and got["export_rows"] == []
    assert got["answer"]["headline"] == "Every prescriber listed had fills from 17 Mar 2026 to 14 Jun 2026."


# ── what is not worked out yet is said, never guessed ──────────────────────


@pytest.mark.parametrize("plan, words", [
    ({"measures": ["number_of_orders"], "per": {"of": "customer"}, "rolling": {"kind": "running"}},
     "one analytical shape at a time"),
    ({"measures": ["number_of_orders"], "per": {"of": "customer"},
      "time": {"window": {"kind": "this", "unit": "month"}, "compare": {"kind": "previous_period"}}},
     "ask for each period on its own"),
    ({"measures": ["margin_percent"], "rolling": {"kind": "running"}, "time": {"grain": "month"}},
     "moving average instead"),
    ({"measures": ["net_amount"], "per": {"of": "customer", "summary": "percentile"}}, "which one"),
    ({"measures": ["net_amount"], "per": {"of": "customer", "summary": "count"}}, "the condition"),
    ({"measures": ["net_amount"], "per": {"of": "customer", "summary": "median", "everyone": True}},
     "members with rows"),
])
def test_what_is_not_worked_out_is_refused_with_why(retail, plan, words):
    with pytest.raises(ResolveError) as refused:
        resolve(Plan.model_validate({"kind": "query", **plan}), retail[1], Context(today=TODAY))
    assert words in refused.value.message


def test_a_plan_without_a_shape_reads_as_it_always_did():
    plan = Plan.model_validate({"kind": "query", "measures": ["net_amount"], "group_by": ["store.name"]})
    assert not plan.shaped()
    assert set(plan.model_dump(mode="json", exclude_defaults=True)) == {"measures", "group_by"}


# ── the same on every warehouse ────────────────────────────────────────────


SHAPES = {
    "median per customer by segment": {"measures": ["net_amount"], "group_by": ["customer.segment"],
                                       "per": {"of": "customer", "summary": "median"}, "time": Q1},
    "share of everyone with 3+": {"measures": ["number_of_orders"], "time": Q1, "per": {
        "of": "customer", "summary": "share", "op": "gte", "values": [3], "everyone": True}},
    "p90 days by store": {"durations": [{"name": "Days from order to delivery", "start": "order_date",
                                         "end": "delivered_date"}], "group_by": ["store.name"],
                          "per": {"summary": "percentile", "p": 90}, "time": Q1},
    "spend ranges": {"measures": ["net_amount"], "bands": {"of": "net_amount", "per": "customer",
                                                           "edges": [500, 1000, 2500]}, "time": Q1},
    "delivery ranges": {"measures": ["number_of_orders"], "durations": [
        {"name": "Days from order to delivery", "start": "order_date", "end": "delivered_date", "measure": False}],
        "bands": {"of": "Days from order to delivery", "edges": [3, 8]}, "time": Q1},
    "top 3 per store": {"measures": ["net_amount"], "group_by": ["store.name", "product.name"],
                        "within": {"group": "store.name", "top": 3}, "time": Q1},
    "moving average": {"measures": ["net_amount"], "rolling": {"kind": "moving_average", "periods": 3},
                       "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01",
                                                              "end": "2026-05-31"}}},
    "no order in 90 days": {"intent": "list", "absent": {"of": "customer", "measure": "number_of_orders",
                                                         "active_before": True},
                            "time": {"window": {"kind": "last", "n": 90, "unit": "day"}}},
}


def _compiled(learned, plan: dict, dialect: str):
    built, model = learned
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    logical = fill_edges(logical, model, "duckdb", lambda sql: built.con.execute(sql).fetchall())
    return compile_query(logical, model, dialect)


def _cell(v):
    """A value as both warehouses' rows can be compared: numbers rounded, a day as its date (a timestamp of
    midnight after translation), nothing as empty."""
    if v is None:
        return ""
    if isinstance(v, (dt.date, dt.datetime)):
        return str(v)[:10]
    if isinstance(v, (int, float, Decimal)) and not isinstance(v, bool):
        return round(float(v), 4)
    return str(v)


def _rounded(rows):
    return sorted((tuple(_cell(v) for v in row) for row in rows), key=repr)


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_each_shape_means_the_same_on_every_warehouse(retail, name, dialect):
    con = retail[0].con
    plan = SHAPES[name]
    sql = _compiled(retail, plan, dialect).sql
    tree = sqlglot.parse_one(sql, read=dialect)        # the warehouse's own SQL reads back
    if dialect == "tsql":
        for select in tree.find_all(exp.Select):
            # SQL Server refuses an ORDER BY inside a subquery (without TOP), a GROUP BY on a column alias, and
            # averages whole numbers into whole numbers.
            if select is not tree:
                assert not select.args.get("order") or select.args.get("limit"), select.sql(dialect="tsql")
            aliases = {e.alias for e in select.expressions if isinstance(e, exp.Alias)}
            for g in (select.args.get("group").expressions if select.args.get("group") else []):
                assert not (isinstance(g, exp.Column) and not g.table and g.name in aliases), g.sql("tsql")
        for avg in tree.find_all(exp.Avg):
            assert isinstance(avg.this, exp.Cast), avg.sql(dialect="tsql")
    try:
        back = sqlglot.transpile(sql, read=dialect, write="duckdb")[0]
        got = con.execute(back).fetchall()
    except Exception:  # noqa: BLE001 - an idiom DuckDB has no word for; it parsed in its own dialect above
        pytest.skip(f"{dialect} SQL does not carry back to DuckDB")
    want = con.execute(_compiled(retail, plan, "duckdb").sql).fetchall()
    assert _rounded(got) == _rounded(want)
