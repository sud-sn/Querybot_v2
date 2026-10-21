"""The number of an order made from its lines' key on load gives way to a link the admin adds to a table of orders.

A model learned before Learn numbered the documents of a lines table (order lines keyed by order and line, no
table of orders found) is given the order number as what the orders are told apart by when it is loaded
(core2/bootstrap/service.load_model). It was made before the admin's decisions were applied: a link the admin
added from the lines to a table of orders was not seen, and the number was made on every load anyway, beside the
orders table that numbers them.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import Join
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture
def account(tmp_path, monkeypatch):
    import store

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    name = f"acct-docs-{uuid.uuid4().hex[:8]}"
    store.upsert_client(name, "web")
    return name


def _learned():
    con = duckdb.connect()
    con.execute("CREATE TABLE order_lines (order_id INTEGER, line INTEGER, order_date DATE NOT NULL, quantity INTEGER, "
                "amount DECIMAL(10,2), PRIMARY KEY (order_id, line))")
    # The orders by a number of their own the lines do not hold: no link found.
    con.execute("CREATE TABLE web_orders (ref INTEGER PRIMARY KEY, channel VARCHAR NOT NULL)")
    con.executemany("INSERT INTO order_lines VALUES (?, ?, ?, ?, ?)",
                    [(1000 + o, n, dt.date(2025, 1, 1) + dt.timedelta(days=o), 1 + (o + n) % 3, 10.0 * (o + n))
                     for o in range(400) for n in (1, 2)])
    con.executemany("INSERT INTO web_orders VALUES (?, ?)", [(r, "web" if r % 2 else "app") for r in range(1, 401)])
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-docs", options=BuildOptions(workers=1))


def _number(model) -> list[str]:
    lines = next(k for k, t in model.tables.items() if t.name == "order_lines")
    column = next(k for k, c in model.columns.items() if c.table == lines and c.name == "order_id")
    return [s for s, a in model.attributes.items() if a.column == column and a.kind == "identifier"]


def test_the_number_made_on_load_goes_when_the_admin_links_the_lines_to_their_orders(account):
    import store
    from core2.bootstrap.service import load_model

    model = _learned()
    made = _number(model)
    assert made, "Learn numbers the orders of a lines table with no table of orders linked"
    for slug in made:          # as learned before Learn numbered them
        del model.attributes[slug]
    store.save_core2_model(account, None, model.model_dump_json())
    assert _number(load_model(account, None)), "numbered on load"

    lines = next(k for k, t in model.tables.items() if t.name == "order_lines")
    orders = next(k for k, t in model.tables.items() if t.name == "web_orders")
    order_id = next(k for k, c in model.columns.items() if c.table == lines and c.name == "order_id")
    ref = next(k for k, c in model.columns.items() if c.table == orders and c.name == "ref")
    link = Join(key="lines_to_orders", from_table=lines, from_columns=[order_id], to_table=orders, to_columns=[ref],
                trust="admin", match_rate=1.0, to_unique=True, target_checked=True)
    store.set_core2_override(account, None, f"join:{link.key}", "define", link.model_dump(mode="json"))
    loaded = load_model(account, None)
    assert "lines_to_orders" in loaded.joins
    assert not _number(loaded), "the table of orders numbers them"


def test_the_number_learn_made_goes_too_when_the_admin_links_the_lines_to_their_orders(account):
    # Found in review: Learn now makes the number itself, before the admin's decisions exist, so on load it was
    # already there, never "made" on load, and stayed beside the admin's link to the orders.
    import store
    from core2.bootstrap.service import load_model

    model = _learned()
    assert _number(model), "Learn numbers the orders"
    store.save_core2_model(account, None, model.model_dump_json())
    lines = next(k for k, t in model.tables.items() if t.name == "order_lines")
    orders = next(k for k, t in model.tables.items() if t.name == "web_orders")
    order_id = next(k for k, c in model.columns.items() if c.table == lines and c.name == "order_id")
    ref = next(k for k, c in model.columns.items() if c.table == orders and c.name == "ref")
    link = Join(key="lines_to_orders", from_table=lines, from_columns=[order_id], to_table=orders, to_columns=[ref],
                trust="admin", match_rate=1.0, to_unique=True, target_checked=True)
    store.set_core2_override(account, None, f"join:{link.key}", "define", link.model_dump(mode="json"))
    loaded = load_model(account, None)
    assert "lines_to_orders" in loaded.joins
    assert not _number(loaded), "the table of orders numbers them"
