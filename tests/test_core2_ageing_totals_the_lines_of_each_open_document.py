"""Ageing totals the amount owed on the lines of each open invoice.

"Receivables ageing with the amount owed in each bucket" was answered with each bucket's freight: the due and paid
dates are on the invoice, the amounts on its lines, and ageing only totalled a figure of the invoice's own rows, so
the planner took the one amount the invoice has. A figure of the rows that point at each open invoice (its lines) is
added up per invoice and then per range of days past due; the lines' own default filters still hold.

Invented data only; every figure checked against SQL written by hand.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import ColumnFilter
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)
BANDS = """CASE WHEN past <= 0 THEN 'Not yet due' WHEN past <= 30 THEN '1-30 days' WHEN past <= 60 THEN '31-60 days'
                WHEN past <= 90 THEN '61-90 days' ELSE 'Over 90 days' END"""


@pytest.fixture(scope="module")
def books():
    con = duckdb.connect()
    con.execute("CREATE TABLE clients (client_id INTEGER, client_name VARCHAR, segment VARCHAR)")
    con.execute("INSERT INTO clients SELECT i, 'Client ' || i, CASE WHEN i % 3 = 0 THEN 'Trade' ELSE 'Retail' END "
                "FROM range(1, 81) t(i)")
    # Invoices dated over two years, due 30 days later; most paid, the last few months' often not.
    con.execute("CREATE TABLE invoices (invoice_id INTEGER, client_id INTEGER, invoice_date DATE, due_date DATE, "
                "paid_date DATE, freight DECIMAL(10,2))")
    con.execute("""INSERT INTO invoices SELECT i, i % 80 + 1, DATE '2024-07-01' + CAST(i % 715 AS INTEGER),
                   DATE '2024-07-31' + CAST(i % 715 AS INTEGER),
                   CASE WHEN i % 715 < 560 OR i % 4 = 0 THEN DATE '2024-07-25' + CAST(i % 715 AS INTEGER) END,
                   5 + i % 40 FROM range(1, 2401) t(i)""")
    con.execute("CREATE TABLE invoice_lines (invoice_id INTEGER, line_no INTEGER, product VARCHAR, quantity INTEGER, "
                "line_amount DECIMAL(12,2), line_status VARCHAR)")
    con.execute("""INSERT INTO invoice_lines SELECT o, l, 'Item ' || ((o + l) % 25), (o + l) % 6 + 1,
                   40 + (o * 7 + l * 13) % 500, CASE WHEN (o + l) % 17 = 0 THEN 'Void' ELSE 'Billed' END
                   FROM range(1, 2401) a(o), range(1, 1 + 1 + CAST(hash(a.o) % 4 AS INTEGER)) b(l)""")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="books", options=BuildOptions(workers=1))
    lines = next(t for t in model.tables.values() if t.name == "invoice_lines")
    status = next(c.key for c in model.columns.values() if c.table == lines.key and c.name == "line_status")
    lines.default_filters = [ColumnFilter(column=status, op="ne", values=["Void"])]
    return con, model


def _slug(model, table: str, column: str) -> str:
    key = next(t.key for t in model.tables.values() if t.name == table)
    return next(m.slug for m in model.measures.values()
                if m.table == key and getattr(m.expr, "column", "") == f"{key}.{column}"
                and getattr(m.expr, "agg", "") == "sum")


def _date(model, table: str, column: str) -> str:
    key = next(t.key for t in model.tables.values() if t.name == table)
    return next(r.slug for r in model.date_roles.values() if r.table == key and r.column == f"{key}.{column}")


def _ask(books, plan: dict, question: str) -> dict:
    con, model = books
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def test_the_amount_owed_on_open_invoices_by_days_past_due(books):
    con, model = books
    amount = _slug(model, "invoice_lines", "line_amount")
    plan = {"kind": "query", "intent": "breakdown", "measures": [amount],
            "on_time": {"done": _date(model, "invoices", "paid_date"), "due": _date(model, "invoices", "due_date"),
                        "ageing": True}}
    payload = _ask(books, plan, "Receivables ageing with the amount owed in each bucket")
    want = {band: (n, float(owed)) for band, n, owed in con.execute(f"""
        SELECT {BANDS}, COUNT(*), SUM(COALESCE(owed, 0)) FROM (
            SELECT i.invoice_id, date_diff('day', i.due_date, DATE '{TODAY}') AS past,
                   (SELECT SUM(l.line_amount) FROM invoice_lines l
                    WHERE l.invoice_id = i.invoice_id AND l.line_status <> 'Void') AS owed
            FROM invoices i WHERE i.paid_date IS NULL) GROUP BY 1""").fetchall()}
    assert len(want) >= 3, want
    rows = payload["export_rows"]
    figure = next(k for k in rows[0] if k not in ("days_past_due", "open"))
    got = {r["days_past_due"]: (r["open"], float(r[figure])) for r in rows}
    assert got == pytest.approx(want)


def test_a_figure_of_the_invoices_own_rows_still_ages_as_before(books):
    con, model = books
    freight = _slug(model, "invoices", "freight")
    plan = {"kind": "query", "intent": "breakdown", "measures": [freight],
            "on_time": {"done": _date(model, "invoices", "paid_date"), "due": _date(model, "invoices", "due_date"),
                        "ageing": True}}
    payload = _ask(books, plan, "Open invoices and their freight by days past due")
    want = {band: (n, float(f)) for band, n, f in con.execute(f"""
        SELECT {BANDS}, COUNT(*), SUM(freight) FROM (
            SELECT freight, date_diff('day', due_date, DATE '{TODAY}') AS past FROM invoices WHERE paid_date IS NULL)
        GROUP BY 1""").fetchall()}
    got = {r["days_past_due"]: (r["open"], float(r[freight])) for r in payload["export_rows"]}
    assert got == pytest.approx(want)


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
def test_the_sql_reads_on_every_warehouse(books, dialect):
    import sqlglot

    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    _, model = books
    plan = Plan.model_validate({"kind": "query", "intent": "breakdown",
                                "measures": [_slug(model, "invoice_lines", "line_amount")],
                                "on_time": {"done": _date(model, "invoices", "paid_date"),
                                            "due": _date(model, "invoices", "due_date"), "ageing": True}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)
    assert "owed_rows" in compiled.sql.lower()
