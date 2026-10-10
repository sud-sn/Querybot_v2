"""Links the database does not declare, to a document's own number, a code kept as text, and a line.

The accuracy benchmark (evals/core2/benchmark.py) showed Learn missing three kinds of link where no
foreign key is declared: a booking line's booking number points at the booking's own number (unique,
but not the table's key); a fill's pharmacist code '0007' is pharmacist 7, kept as text against a
whole-number key; and a claim names a prescription and its fill number, the fills' key of two columns.
Each is learned here, and a question that follows the code kept as text, or the line of two columns,
has the numbers the hand-written SQL gives. Two things stay unlinked: a code that is no number, and
a movement's day and item, which key a balance but are no document and its line.

Invented data only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest
import sqlglot
from sqlglot import exp

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse import dialect as D
from core2.warehouse.runner import DuckDBWarehouse


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE bookings (booking_id INTEGER PRIMARY KEY, booking_number VARCHAR, customer VARCHAR, "
                "booked_date DATE)")
    con.execute("INSERT INTO bookings SELECT i, 'BK-' || (500000 + i * 7), 'Customer ' || (i % 40), "
                "DATE '2026-01-01' + (i % 150)::INTEGER FROM range(1, 801) t(i)")
    con.execute("CREATE TABLE booking_lines (booking_number VARCHAR, line_number INTEGER, service_type VARCHAR, "
                "line_amount DECIMAL(10,2), PRIMARY KEY (booking_number, line_number))")
    con.execute("INSERT INTO booking_lines SELECT 'BK-' || (500000 + b * 7), l, "
                "CASE (b + l) % 3 WHEN 0 THEN 'Repair' WHEN 1 THEN 'Install' ELSE 'Inspect' END, (b * l) % 90 + 10 "
                "FROM range(1, 801) a(b), range(1, 5) c(l) WHERE l <= b % 4 + 1")
    # One visit for four bookings in five: the visit's booking number is unique, and still the booking's.
    con.execute("CREATE TABLE visits (visit_id INTEGER PRIMARY KEY, booking_number VARCHAR, minutes INTEGER)")
    con.execute("INSERT INTO visits SELECT row_number() OVER (), 'BK-' || (500000 + i * 7), 30 + i % 200 "
                "FROM range(1, 801) t(i) WHERE i % 5 <> 0")
    con.execute("CREATE TABLE pharmacists (pharmacist_id INTEGER PRIMARY KEY, pharmacist_name VARCHAR, "
                "license_state VARCHAR)")
    con.execute("INSERT INTO pharmacists SELECT i, 'Pharmacist ' || i, CASE i % 2 WHEN 0 THEN 'NY' ELSE 'NJ' END "
                "FROM range(1, 13) t(i)")
    con.execute("CREATE TABLE prescriptions (rx_number VARCHAR PRIMARY KEY, written_date DATE, drug VARCHAR)")
    con.execute("INSERT INTO prescriptions SELECT 'RX' || (700000 + r), DATE '2025-06-01' + (r % 200)::INTEGER, "
                "'Drug ' || (r % 9) FROM range(1, 1201) t(r)")
    con.execute("CREATE TABLE fills (rx_number VARCHAR, fill_number INTEGER, fill_date DATE, pharmacist_code VARCHAR, "
                "quantity INTEGER, PRIMARY KEY (rx_number, fill_number))")
    con.execute("INSERT INTO fills SELECT 'RX' || (700000 + r), f, DATE '2025-06-01' + (r % 200 + f * 30)::INTEGER, "
                "lpad(((r + f) % 12 + 1)::VARCHAR, 4, '0'), 30 * f FROM range(1, 1201) a(r), range(1, 5) b(f) "
                "WHERE f <= r % 4 + 1")
    con.execute("CREATE TABLE claims (claim_id INTEGER PRIMARY KEY, rx_number VARCHAR, fill_number INTEGER, "
                "claim_date DATE, paid_amount DECIMAL(10,2))")
    con.execute("INSERT INTO claims SELECT row_number() OVER (), rx_number, fill_number, fill_date + 2, quantity * 1.5 "
                "FROM fills WHERE hash(rx_number) % 5 <> 0")     # every fill of four prescriptions in five
    # A code that is no number beside pharmacists' whole-number keys.
    con.execute("CREATE TABLE audits (audit_id INTEGER PRIMARY KEY, pharmacist_code VARCHAR, audit_date DATE, "
                "findings INTEGER)")
    con.execute("INSERT INTO audits SELECT i, 'PH-' || (i % 12 + 1), DATE '2026-01-01' + i::INTEGER, i % 4 "
                "FROM range(1, 301) t(i)")
    # A day and an item key a balance; a movement of that item on that day is not one of its lines.
    con.execute("CREATE TABLE items (item_id INTEGER PRIMARY KEY, item_name VARCHAR)")
    con.execute("INSERT INTO items SELECT i, 'Item ' || i FROM range(1, 21) t(i)")
    con.execute("CREATE TABLE daily_balances (day_key INTEGER, item_id INTEGER, on_hand INTEGER, "
                "PRIMARY KEY (day_key, item_id))")
    con.execute("INSERT INTO daily_balances SELECT 20260101 + d, i, (d * i) % 300 FROM range(0, 28) a(d), range(1, 21) b(i)")
    con.execute("CREATE TABLE movements (movement_id INTEGER PRIMARY KEY, day_key INTEGER, item_id INTEGER, "
                "moved_quantity INTEGER)")
    con.execute("INSERT INTO movements SELECT m, 20260101 + m % 28, m % 20 + 1, m % 7 + 1 FROM range(1, 2001) t(m)")
    return con


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="codes", options=BuildOptions(workers=1))
    return con, model


def _links(model) -> dict[tuple, object]:
    return {(model.tables[j.from_table].name, tuple(model.columns[c].name for c in j.from_columns),
             model.tables[j.to_table].name, tuple(model.columns[c].name for c in j.to_columns)): j
            for j in model.joins.values() if j.trust != "rejected"}


@pytest.mark.parametrize("link", [
    ("booking_lines", ("booking_number",), "bookings", ("booking_number",)),      # the booking's own number
    ("visits", ("booking_number",), "bookings", ("booking_number",)),
    ("fills", ("pharmacist_code",), "pharmacists", ("pharmacist_id",)),           # '0007' is pharmacist 7
    ("claims", ("rx_number", "fill_number"), "fills", ("rx_number", "fill_number")),   # a fill of a prescription
    ("claims", ("rx_number",), "prescriptions", ("rx_number",)),
])
def test_links_no_one_declared_are_learned(learned, link):
    _, model = learned
    found = _links(model).get(link)
    assert found is not None, sorted(_links(model))
    assert found.trust == "verified"


@pytest.mark.parametrize("not_a_link", [
    ("audits", ("pharmacist_code",)),                 # PH-7 is no number
    ("movements", ("day_key", "item_id")),            # a balance's day and item, not a document's line
    ("fills", ("rx_number", "fill_number")),          # a fill with no claim: the claim points at the fill
    ("fills", ("fill_number", "rx_number")),
    ("bookings", ("booking_number",)),                # the visit's copy of it is the booking's, not the visit's
])
def test_what_only_looks_like_a_link_stays_unlinked(learned, not_a_link):
    _, model = learned
    table, columns = not_a_link
    assert not [k for k in _links(model) if k[0] == table and k[1] == columns], sorted(_links(model))


def _answer(con, model, measure_column: str, measure_table: str, group_column: str, group_table: str):
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    measure = next(m for m in model.measures.values()
                   if getattr(m.expr, "column", None) and model.columns[m.expr.column].name == measure_column
                   and model.tables[m.table].name == measure_table)
    group = next(slug for slug, a in model.attributes.items()
                 if model.columns[a.column].name == group_column
                 and model.tables[model.columns[a.column].table].name == group_table)
    logical = resolve(Plan(intent="breakdown", measures=[measure.slug], group_by=[group]), model,
                      Context(today=dt.date(2026, 6, 15)))
    sql = compile_query(logical, model, "duckdb").sql
    return sorted((k, float(v)) for k, v in con.execute(sql).fetchall()), sql


def test_a_question_over_a_code_kept_as_text_has_the_right_numbers(learned):
    con, model = learned
    got, sql = _answer(con, model, "quantity", "fills", "license_state", "pharmacists")
    expected = sorted((k, float(v)) for k, v in con.execute(
        "SELECT p.license_state, SUM(f.quantity) FROM fills f JOIN pharmacists p "
        "ON p.pharmacist_id = CAST(f.pharmacist_code AS INTEGER) GROUP BY 1").fetchall())
    assert got == expected
    # DuckDB would read '0007' as 7 by itself; Azure SQL stops the whole query at the first code that is no number.
    assert "LTRIM(" in sql.upper()


def test_a_question_over_the_fill_of_a_claim_has_the_right_numbers(learned):
    """A claim reaches its pharmacist through its fill: the key of two columns, then the code kept as text."""
    con, model = learned
    got, _ = _answer(con, model, "paid_amount", "claims", "license_state", "pharmacists")
    expected = sorted((k, float(v)) for k, v in con.execute(
        "SELECT p.license_state, SUM(c.paid_amount) FROM claims c "
        "JOIN fills f ON f.rx_number = c.rx_number AND f.fill_number = c.fill_number "
        "JOIN pharmacists p ON p.pharmacist_id = CAST(f.pharmacist_code AS INTEGER) GROUP BY 1").fetchall())
    assert got == expected


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "tsql", "oracle"])
def test_a_code_kept_as_text_is_compared_without_its_leading_zeros_on_every_warehouse(dialect):
    sql = D.same_number(exp.column("code", table="f"), exp.column("id", table="t"), dialect).sql(dialect=dialect)
    sqlglot.parse_one(sql, read=dialect)
    if dialect == "duckdb":
        con = duckdb.connect()
        for text, number, equal in [("0007", 7, True), (" 7", 7, True), ("000", 0, True), ("70", 7, False),
                                    ("PH-7", 7, False), ("0070", 70, True)]:
            got = con.execute("SELECT " + D.same_number(exp.Literal.string(text), exp.Literal.number(number),
                                                        dialect).sql(dialect=dialect)).fetchone()[0]
            assert bool(got) is equal, (text, number)
