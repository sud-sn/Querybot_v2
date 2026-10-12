"""Learn works out a value no column holds: a quantity times its price or cost per unit, added up.

A stock table often keeps how much of each product is on hand and what one unit costs, and no value: the
stock value is the two multiplied on each row. Learn found the quantity and summed the unit cost, so "what
is our stock worth?" was answered with the units on hand, or with every unit cost added together -- a
figure that means nothing. Order lines kept as a quantity and a unit price, with no line amount, had no
sales figure at all.

Now Learn reads a cost or a price as one per unit by its name (a unit price, a cost beside its own
average), by its rows (it is there where nothing is on hand, and does not rise with the quantity), or by
staying the same for one product while quantities vary. A per-unit figure is averaged, never summed, and
the value -- the quantity times it on each row, added up -- is proposed as a measure with the words people
use for it: stock value for a balance at cost, the value at the unit price for order lines. A table that
already holds an amount (quantity x price on its own row) gets no second one, and a price empty where the
quantity is not makes none.

Learn also read a quantity that is nought on most rows as "another column divided by a third", because
nought divided by anything is nought: a ratio is now read only where the column holds something. And an
order's lines whose quantities never repeat within the order made (order, quantity) look like a second key,
which hid the quantity: a column named as a quantity or an amount stays a measure on a table with its own key.

And a short description that repeats now and then (13 for 15 item types) names a table's members, before
their bare codes.

A small warehouse in DuckDB, invented data; figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import SqlExpr
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)
MONTH_ENDS = [dt.date(2025, m, 28) for m in range(7, 13)] + [dt.date(2026, m, 28) for m in range(1, 7)]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE products (product_id INTEGER PRIMARY KEY, product_name VARCHAR, type_code VARCHAR)")
    con.execute("CREATE TABLE product_types (type_code VARCHAR PRIMARY KEY, type_description VARCHAR)")
    con.execute("CREATE TABLE depots (depot_id INTEGER PRIMARY KEY, depot_name VARCHAR)")
    con.execute("CREATE TABLE stock_balances (balance_date DATE, product_id INTEGER, depot_id INTEGER, "
                "on_hand_qty INTEGER, reserved_qty INTEGER, product_cost DECIMAL(12, 2), "
                "avg_product_cost DECIMAL(12, 2))")
    con.execute("CREATE TABLE order_lines (order_id INTEGER, line_no INTEGER, order_date DATE, product_id INTEGER, "
                "quantity INTEGER, unit_price DECIMAL(12, 2))")
    con.execute("CREATE TABLE invoice_lines (invoice_id INTEGER, line_no INTEGER, invoice_date DATE, "
                "product_id INTEGER, qty INTEGER, price DECIMAL(12, 2), line_amount DECIMAL(14, 2))")
    # Fifteen product types, two pairs sharing a description: still the names readers use.
    described = ["Raw material", "Packaging", "Finished good", "Spare part", "Consumable", "Tooling", "Chemical",
                 "Label", "Pallet", "Fastener", "Electrical", "Filter", "Lubricant", "Packaging", "Spare part"]
    types = [(f"T{i:02d}", d) for i, d in enumerate(described)]
    con.executemany("INSERT INTO product_types VALUES (?, ?)", types)
    con.executemany("INSERT INTO products VALUES (?, ?, ?)",
                    [(p, f"Product {p:03d}", types[p % 15][0]) for p in range(1, 61)])
    con.executemany("INSERT INTO depots VALUES (?, ?)", [(1, "Northgate"), (2, "Riverside"), (3, "Hillcrest")])
    cost = {p: round(2.5 + (p * 37 % 90) + p / 100, 2) for p in range(1, 61)}
    balances = []
    for i, day in enumerate(MONTH_ENDS):
        for p in range(1, 61):
            for d in (1, 2, 3):
                seed = (p * 31 + d * 17 + i * 7) % 100
                on_hand = 0 if seed < 50 else seed * 3 + d          # half the rows hold nothing
                reserved = (seed % 9) + 1 if seed > 94 else 0      # nought on nearly every row
                balances.append((day, p, d, on_hand, reserved, cost[p], round(cost[p] * 1.01, 2)))
    con.executemany("INSERT INTO stock_balances VALUES (?, ?, ?, ?, ?, ?, ?)", balances)
    lines, amounts = [], []
    for o in range(1, 401):
        day = dt.date(2025, 7, 1) + dt.timedelta(days=o % 360)
        for n in range(1, 3):
            p = (o * 7 + n * 13) % 60 + 1
            q = (o + n) % 12 + 1
            lines.append((o, n, day, p, q, round(cost[p] * 1.6, 2)))
            amounts.append((o, n, day, p, q, round(cost[p] * 1.4, 2), round(q * cost[p] * 1.4, 2)))
    con.executemany("INSERT INTO order_lines VALUES (?, ?, ?, ?, ?, ?)", lines)
    con.executemany("INSERT INTO invoice_lines VALUES (?, ?, ?, ?, ?, ?, ?)", amounts)
    return con


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-value", options=BuildOptions(workers=1))
    return con, model


def _table(model, name):
    return next(k for k, t in model.tables.items() if t.name == name)


def _column_measure(model, table, column):
    key = _table(model, table)
    return next(m for m in model.measures.values()
                if m.table == key and getattr(m.expr, "column", "") and m.expr.column.endswith("." + column))


def _values(model, table):
    key = _table(model, table)
    return [m for m in model.measures.values() if m.table == key and isinstance(m.expr, SqlExpr)]


def _ask(con, model, plan):
    answer = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answer,
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _figures(payload) -> list[float]:
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    assert headers, payload["answer"]["headline"] + " / " + str(payload.get("trust", {}).get("stopped"))
    return [float(r[headers[-1]]) for r in data.get("rows") or []]


# ── a cost per unit on a stock table ──────────────────────────────────────────


def test_a_unit_cost_is_averaged_never_summed(learned):
    _, model = learned
    for column in ("product_cost", "avg_product_cost"):
        m = _column_measure(model, "stock_balances", column)
        assert (m.expr.agg, m.additivity) == ("avg", "non_additive"), (column, m.evidence)


def test_the_stock_value_is_the_quantity_times_the_unit_cost_taken_at_each_snapshot(learned):
    _, model = learned
    [value] = _values(model, "stock_balances")
    assert "on_hand_qty" in value.expr.sql and "product_cost" in value.expr.sql and "avg_" not in value.expr.sql
    assert (value.additivity, value.time_aggregation, value.format) == ("semi_additive", "last", "currency")
    assert "stock value" in value.synonyms.get("en", [])
    assert any(e.kind == "value_of_units" for e in value.evidence)


def test_the_stock_value_answers_with_the_last_snapshots_figure(learned):
    con, model = learned
    [value] = _values(model, "stock_balances")
    payload = _ask(con, model, {"intent": "value", "measures": [value.slug]})
    want = con.execute("SELECT SUM(on_hand_qty * product_cost) FROM stock_balances "
                       "WHERE balance_date = (SELECT MAX(balance_date) FROM stock_balances)").fetchone()[0]
    assert _figures(payload) == [pytest.approx(float(want), rel=1e-6)]


def test_a_quantity_nought_on_most_rows_is_a_quantity_not_a_ratio(learned):
    _, model = learned
    m = _column_measure(model, "stock_balances", "reserved_qty")
    assert m.format != "percent" and m.expr.agg == "sum", m.evidence
    assert not any(e.kind == "ratio" for e in m.evidence)


# ── order lines ───────────────────────────────────────────────────────────────


def test_order_lines_with_no_amount_get_their_value_at_the_unit_price(learned):
    con, model = learned
    [value] = _values(model, "order_lines")
    assert "quantity" in value.expr.sql and "unit_price" in value.expr.sql
    assert (value.additivity, value.format) == ("additive", "currency")
    payload = _ask(con, model, {"intent": "value", "measures": [value.slug]})
    want = con.execute("SELECT SUM(quantity * unit_price) FROM order_lines").fetchone()[0]
    assert _figures(payload) == [pytest.approx(float(want), rel=1e-6)]


def test_lines_that_hold_their_amount_get_no_second_one(learned):
    _, model = learned
    assert _values(model, "invoice_lines") == []
    amount = _column_measure(model, "invoice_lines", "line_amount")
    assert (amount.expr.agg, amount.additivity) == ("sum", "additive")


def test_a_price_empty_where_the_quantity_is_not_makes_no_value():
    con = duckdb.connect()
    con.execute("CREATE TABLE month_balances (period_date DATE, item_id INTEGER, closing_qty INTEGER, "
                "item_cost DECIMAL(12, 2))")
    rows = [(day, i, (i * 13 + k) % 40 + 1, (5.0 + i) if k == 11 and i < 10 else None)
            for k, day in enumerate(MONTH_ENDS) for i in range(1, 81)]
    con.executemany("INSERT INTO month_balances VALUES (?, ?, ?, ?)", rows)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-empty", options=BuildOptions(workers=1))
    assert _values(model, "month_balances") == []


# ── names ─────────────────────────────────────────────────────────────────────


def test_a_description_that_repeats_now_and_then_names_the_members(learned):
    _, model = learned
    key = _table(model, "product_types")
    entity = next(e for e in model.entities.values() if e.table == key)
    assert model.columns[entity.label_column].name == "type_description"


def test_lines_that_each_sell_one_unit_still_have_a_sales_amount():
    """Every line of a music store sells one track (quantity 1): the sales are the unit prices added up, named as
    the sales they are, not left as a count of lines."""
    import duckdb

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.warehouse.runner import DuckDBWarehouse

    import datetime as dt

    con = duckdb.connect()
    con.execute("CREATE TABLE tracks (track_id INTEGER PRIMARY KEY, track_name VARCHAR, genre VARCHAR)")
    con.executemany("INSERT INTO tracks VALUES (?, ?, ?)",
                    [(i, f"Track {i}", ["Rock", "Jazz", "Blues", "Pop"][i % 4]) for i in range(400)])
    con.execute("CREATE TABLE invoices (invoice_id INTEGER PRIMARY KEY, invoice_date DATE, customer_id INTEGER)")
    con.executemany("INSERT INTO invoices VALUES (?, ?, ?)",
                    [(i, dt.date(2024, 1, 1) + dt.timedelta(days=i % 700), i % 60) for i in range(800)])
    con.execute("CREATE TABLE invoice_lines (invoice_line_id INTEGER PRIMARY KEY, invoice_id INTEGER, "
                "track_id INTEGER, unit_price DECIMAL(10,2), quantity INTEGER)")
    con.executemany("INSERT INTO invoice_lines VALUES (?, ?, ?, ?, ?)",
                    [(i, i // 3, i % 400, 0.99 if i % 5 else 1.99, 1) for i in range(1, 2201)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-one-each", options=BuildOptions(workers=1))
    sales = [m for m in model.measures.values() if m.business_name == "Sales amount"]
    assert len(sales) == 1 and "revenue" in sales[0].synonyms.get("en", [])
    unit = next(m for m in model.measures.values() if m.business_name.lower() == "unit price")
    assert unit.expr.agg == "avg"
