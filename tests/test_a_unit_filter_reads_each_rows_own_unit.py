"""A filter on the unit of measure reads the unit each row is counted in, not its item's usual unit.

A movements table counts each row's quantity in its own unit (EA, KG, BOX), and the item list keeps each
item's usual unit under the same name. Asked for "units sold, only the ones counted in EA", the AI may name
the item list's unit: the answer then keeps rows by their item's unit, which differs from the row's own where
an item is sold in a second unit, and keeps nothing at all where the item list does not hold the row's item
(an extract of active items). The filter now goes on the unit column the measure itself is counted in, when
the one named has the same name on a table the rows reach. Other columns sharing a name (an order's status,
a customer's status) are left as the plan names them.

A small warehouse in DuckDB, invented data; figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE items (item_id INTEGER PRIMARY KEY, item_name VARCHAR, uom VARCHAR)")
    con.execute("CREATE TABLE sales_lines (line_id INTEGER PRIMARY KEY, sale_date DATE, item_id INTEGER, "
                "sold_qty DECIMAL(12, 2), uom VARCHAR)")
    # Items 1-30 are on the list (usually counted in EA or KG); items 31-40 are not on it at all.
    con.executemany("INSERT INTO items VALUES (?, ?, ?)",
                    [(i, f"Item {i:02d}", "EA" if i % 2 else "KG") for i in range(1, 31)])
    rows = []
    for n in range(1, 801):
        item = n % 40 + 1
        unit = ("EA", "KG", "BOX")[n % 3]          # each row's own unit, whatever the item's usual one
        rows.append((n, dt.date(2026, 1, 1) + dt.timedelta(days=n % 150), item, float(n % 17 + 1), unit))
    con.executemany("INSERT INTO sales_lines VALUES (?, ?, ?, ?, ?)", rows)
    return con


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-unit", options=BuildOptions(workers=1))
    return con, model


def _quantity(model):
    return next(m for m in model.measures.values() if getattr(m.expr, "column", "") and
                m.expr.column.endswith(".sold_qty"))


def _item_unit(model):
    items = next(k for k, t in model.tables.items() if t.name == "items")
    return next(a.slug for a in model.attributes.values()
                if model.columns[a.column].table == items and model.columns[a.column].name == "uom")


def test_the_quantity_is_counted_in_its_rows_own_unit(learned):
    _, model = learned
    assert model.columns[_quantity(model).unit_column].name == "uom"
    assert model.tables[model.columns[_quantity(model).unit_column].table].name == "sales_lines"


def test_a_filter_on_the_items_unit_keeps_the_rows_counted_in_it(learned):
    con, model = learned
    plan = {"kind": "query", "intent": "value", "measures": [_quantity(model).slug],
            "filters": [{"field": _item_unit(model), "op": "eq", "values": ["EA"]}]}
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("units sold, only the ones counted in EA", services, Session())
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    assert headers, payload["answer"]["headline"]
    got = [float(r[headers[-1]]) for r in data["rows"]]
    want = con.execute("SELECT SUM(sold_qty) FROM sales_lines WHERE uom = 'EA'").fetchone()[0]
    assert got == [pytest.approx(float(want))]
