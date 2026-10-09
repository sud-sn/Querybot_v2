"""Every link the database declares is learned, and a same-named code finds its table.

The accuracy benchmark (evals/core2/benchmark.py) showed Learn dropping links the database itself
declares: a device id that is part of an interface's two-column key was taken for a line number,
a link to a prescriber's NPI (unique, but not the table's own key) had no target, a language id
holding one value in every row was skipped, and a key of two columns was read as two links to
half a key each. A circuit code beside the service id was never a target, though tickets name it.

Each of those is learned here, from a small warehouse built for it, and a question that needs the
two-column link is answered with the right numbers. Two things that only look like links stay
unlinked: a line number, and two tables' postal codes.

Invented data only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import InvForeignKey, from_duckdb
from core2.warehouse.runner import DuckDBWarehouse

DECLARED = [
    ("interfaces", "device_id", "devices", "device_id", "fk_if_device", 1),
    ("films", "language_id", "languages", "language_id", "fk_film_language", 1),
    ("prescriptions", "prescriber_npi", "prescribers", "prescriber_npi", "fk_rx_prescriber", 1),
    ("shipments", "order_number", "order_lines", "order_number", "fk_ship_line", 1),
    ("shipments", "line_number", "order_lines", "line_number", "fk_ship_line", 2),
]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE devices (device_id INTEGER PRIMARY KEY, hostname VARCHAR)")
    con.execute("INSERT INTO devices SELECT i, 'sw' || lpad(i::VARCHAR, 3, '0') FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE interfaces (interface_id INTEGER PRIMARY KEY, device_id INTEGER, interface_name VARCHAR)")
    con.execute("INSERT INTO interfaces SELECT i, (i - 1) // 3 + 1, 'Eth1/' || ((i - 1) % 3 + 1) FROM range(1, 121) t(i)")
    con.execute("CREATE TABLE languages (language_id INTEGER PRIMARY KEY, language_name VARCHAR)")
    con.execute("INSERT INTO languages VALUES (1, 'English'), (2, 'Italian'), (3, 'Japanese'), (4, 'Mandarin'), "
                "(5, 'French'), (6, 'German')")
    con.execute("CREATE TABLE films (film_id INTEGER PRIMARY KEY, title VARCHAR, language_id INTEGER)")
    con.execute("INSERT INTO films SELECT i, 'Film ' || i, 1 FROM range(1, 51) t(i)")
    con.execute("CREATE TABLE prescribers (prescriber_id INTEGER PRIMARY KEY, prescriber_npi VARCHAR, "
                "prescriber_name VARCHAR)")
    con.execute("INSERT INTO prescribers SELECT i, (1000000000 + i * 7919)::VARCHAR, 'Dr. Example ' || i "
                "FROM range(1, 31) t(i)")
    con.execute("CREATE TABLE prescriptions (rx_number VARCHAR PRIMARY KEY, prescriber_npi VARCHAR, "
                "written_date DATE, quantity INTEGER)")
    con.execute("INSERT INTO prescriptions SELECT 'RX' || (700000 + i), (1000000000 + (i % 30 + 1) * 7919)::VARCHAR, "
                "DATE '2026-01-01' + (i % 150)::INTEGER, i % 9 + 1 FROM range(1, 201) t(i)")
    con.execute("CREATE TABLE orders (order_number VARCHAR PRIMARY KEY, customer_name VARCHAR, order_date DATE)")
    con.execute("INSERT INTO orders SELECT 'SO-' || (200000 + i), 'Customer ' || (i % 12), "
                "DATE '2026-01-01' + (i % 120)::INTEGER FROM range(1, 81) t(i)")
    con.execute("CREATE TABLE order_lines (order_number VARCHAR, line_number INTEGER, product VARCHAR, "
                "quantity INTEGER, PRIMARY KEY (order_number, line_number))")
    con.execute("INSERT INTO order_lines SELECT 'SO-' || (200000 + o), l, "
                "CASE (o + l) % 3 WHEN 0 THEN 'Bolts' WHEN 1 THEN 'Nuts' ELSE 'Washers' END, (o * l) % 7 + 1 "
                "FROM range(1, 81) a(o), range(1, 4) b(l)")
    con.execute("CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, order_number VARCHAR, line_number INTEGER, "
                "shipped_date DATE, shipped_quantity INTEGER)")
    con.execute("INSERT INTO shipments SELECT i, 'SO-' || (200000 + (i - 1) % 80 + 1), (i - 1) // 80 + 1, "
                "DATE '2026-02-01' + (i % 90)::INTEGER, i % 5 + 1 FROM range(1, 201) t(i)")
    con.execute("CREATE TABLE services (service_id INTEGER PRIMARY KEY, circuit_code VARCHAR, service_type VARCHAR)")
    con.execute("INSERT INTO services SELECT i, 'CIR-' || lpad(i::VARCHAR, 5, '0'), "
                "CASE i % 3 WHEN 0 THEN 'Internet' WHEN 1 THEN 'MPLS' ELSE 'SD-WAN' END FROM range(1, 61) t(i)")
    con.execute("CREATE TABLE tickets (ticket_id INTEGER PRIMARY KEY, circuit_code VARCHAR, opened_date DATE)")
    con.execute("INSERT INTO tickets SELECT i, 'CIR-' || lpad((i % 60 + 1)::VARCHAR, 5, '0'), "
                "DATE '2026-03-01' + (i % 60)::INTEGER FROM range(1, 301) t(i)")
    con.execute("CREATE TABLE suppliers (supplier_id INTEGER PRIMARY KEY, supplier_name VARCHAR, postal_code VARCHAR)")
    con.execute("INSERT INTO suppliers SELECT i, 'Supplier ' || i, 'P' || (10000 + i * 37) FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name VARCHAR, postal_code VARCHAR)")
    # Four customers share a postal code with a supplier; the rest live elsewhere.
    con.execute("INSERT INTO customers SELECT i, 'Customer ' || i, "
                "CASE WHEN i <= 4 THEN 'P' || (10000 + i * 37) ELSE 'Q' || (20000 + i) END FROM range(1, 101) t(i)")
    return con


def _declared() -> list[dict]:
    return [{"constraint_name": name, "parent_schema": "main", "parent_table": parent, "parent_col": col,
             "ref_schema": "main", "ref_table": ref, "ref_col": ref_col, "ordinal": ordinal, "enforced": False}
            for parent, col, ref, ref_col, name, ordinal in DECLARED]


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse, declared_fks=_declared())
    model = build_model(warehouse, inventory, client_id="links", options=BuildOptions(workers=1))
    return con, warehouse, inventory, model


def _links(model) -> dict[tuple, object]:
    out = {}
    for j in model.joins.values():
        out[(model.tables[j.from_table].name, tuple(model.columns[c].name for c in j.from_columns),
             model.tables[j.to_table].name, tuple(model.columns[c].name for c in j.to_columns))] = j
    return out


def test_a_two_column_declared_key_is_read_as_one_key(learned):
    _, _, inventory, _ = learned
    by_name = {fk.name: fk for fk in inventory.foreign_keys}
    key = by_name["fk_ship_line"]
    assert isinstance(key, InvForeignKey) and key.columns == ["order_number", "line_number"]
    assert key.ref_columns == ["order_number", "line_number"]
    assert len([fk for fk in inventory.foreign_keys if fk.name == "fk_ship_line"]) == 1


@pytest.mark.parametrize("link", [
    ("interfaces", ("device_id",), "devices", ("device_id",)),               # part of a two-column unique key
    ("films", ("language_id",), "languages", ("language_id",)),               # one value in every row
    ("prescriptions", ("prescriber_npi",), "prescribers", ("prescriber_npi",)),   # a unique column, not the key
    ("shipments", ("order_number", "line_number"), "order_lines", ("order_number", "line_number")),
])
def test_every_declared_link_is_learned_and_usable(learned, link):
    _, _, _, model = learned
    found = _links(model).get(link)
    assert found is not None, sorted(_links(model))
    assert found.trust in ("verified", "declared") and any(e.kind == "declared_fk" for e in found.evidence)


def test_the_order_number_of_a_two_column_key_still_points_at_its_order(learned):
    _, _, _, model = learned
    assert ("shipments", ("order_number",), "orders", ("order_number",)) in _links(model)


def test_a_same_named_code_finds_the_table_it_identifies(learned):
    _, _, _, model = learned
    found = _links(model).get(("tickets", ("circuit_code",), "services", ("circuit_code",)))
    assert found is not None and found.trust == "verified"


@pytest.mark.parametrize("not_a_link", [
    ("customers", ("postal_code",), "suppliers", ("postal_code",)),     # a few shared values, a shared name
    ("order_lines", ("line_number",), None, None),                       # numbers lines, points nowhere
])
def test_what_only_looks_like_a_link_stays_unlinked(learned, not_a_link):
    _, _, _, model = learned
    table, columns, *_ = not_a_link
    assert not [k for k in _links(model) if k[0] == table and k[1] == columns
                and (not_a_link[2] is None or k[2] == not_a_link[2])], sorted(_links(model))


def test_a_question_over_the_two_column_link_has_the_right_numbers(learned):
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    con, _, _, model = learned
    shipped = next(m for m in model.measures.values()
                   if getattr(m.expr, "column", None) and model.columns[m.expr.column].name == "shipped_quantity")
    product = next(slug for slug, a in model.attributes.items()
                   if model.columns[a.column].name == "product" and model.tables[model.columns[a.column].table].name
                   == "order_lines")
    plan = Plan(intent="breakdown", measures=[shipped.slug], group_by=[product])
    logical = resolve(plan, model, Context(today=dt.date(2026, 6, 15)))
    got = sorted(con.execute(compile_query(logical, model, "duckdb").sql).fetchall())
    expected = sorted(con.execute(
        "SELECT l.product, SUM(s.shipped_quantity) FROM shipments s JOIN order_lines l "
        "ON l.order_number = s.order_number AND l.line_number = s.line_number GROUP BY l.product").fetchall())
    assert [(p, float(v)) for p, v in got] == [(p, float(v)) for p, v in expected]
