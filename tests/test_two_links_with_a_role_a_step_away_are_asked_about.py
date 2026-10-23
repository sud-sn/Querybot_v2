"""Two links with a role a step away are asked about, as two direct ones are.

An invoice holds its bill-to and its payer customer and no plain one. A question on the invoice itself asked which; a
question on its lines, a step away, took whichever link matched better (the payer, its values all found, where a few
bill-to customers are not on file) and said nothing. Now:

* "line amount by customer" asks which, bill-to or payer, however well each matches;
* the reader's choice is followed, from the lines as from the invoice, and is offered by the role that sets it
  apart: Bill to customer or Payer customer, never the order every route passes ("Sales order, Bill to customer"),
  which no choice could name; a choice two roles set apart (the original invoice's payer customer) is followed by
  both;
* one link with a role and no other (an order's bill-to) is still taken unasked, and so is a plain link beside a
  link with a role (the customer beside the payer);
* two routes through different tables (the invoice's payer, the shipment's ship-to) are ranked as before, the better
  matched first: the question is which route, not which of two links between the same tables.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import random

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2025, 6, 30)


def _warehouse(*customers: str, missing: str | None = None) -> duckdb.DuckDBPyConnection:
    """120 customers; 2,000 invoices holding ``customers`` (a few of ``missing`` not on file); 8,000 lines."""
    rng = random.Random(len(customers))
    con = duckdb.connect()
    con.execute("CREATE TABLE DIM_CUST (CUST_KEY INTEGER, CUST_NM VARCHAR)")
    con.executemany("INSERT INTO DIM_CUST VALUES (?,?)", [(i, f"Customer {i:03d}") for i in range(1, 121)])
    con.execute(f"CREATE TABLE FCT_INVOICE (INVOICE_KEY INTEGER, {', '.join(f'{c} INTEGER' for c in customers)}, "
                "INVOICE_DT DATE)")
    rows = []
    for i in range(1, 2001):
        first = rng.randint(1, 120)
        keys = [first if rng.random() < 0.6 else rng.randint(1, 120) for _ in customers]
        if missing and i % 300 == 7:
            keys[customers.index(missing)] = 900 + i % 5
        rows.append((i, *keys, dt.date(2024, 1, 1) + dt.timedelta(days=i % 365)))
    con.executemany(f"INSERT INTO FCT_INVOICE VALUES ({', '.join('?' * len(rows[0]))})", rows)
    con.execute("CREATE TABLE FCT_INVOICE_LINE (INVOICE_LINE_KEY INTEGER, INVOICE_KEY INTEGER, LINE_NO INTEGER, "
                "LINE_AMT DECIMAL(12,2))")
    con.executemany("INSERT INTO FCT_INVOICE_LINE VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 2000), i % 4 + 1, rng.randint(100, 9999) / 100) for i in range(1, 8001)])
    return con


def _model(con: duckdb.DuckDBPyConnection, client: str):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id=client,
                       options=BuildOptions(workers=1, today=lambda: dt.datetime(2025, 6, 30, tzinfo=dt.timezone.utc)))


def _by_customer(con, model, measure: str, via: str | None = None) -> dict[str, float]:
    plan = Plan.model_validate({"intent": "breakdown", "measures": [measure], "group_by": ["customer"],
                                "via": {"customer": via} if via else {}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")
    return {k: float(v) for k, v in con.execute(compiled.sql).fetchall() if k is not None}


def _line_amount_by(con, column: str) -> dict[str, float]:
    return {k: float(v) for k, v in con.execute(
        "SELECT c.CUST_NM, SUM(l.LINE_AMT) FROM FCT_INVOICE_LINE l JOIN FCT_INVOICE f ON f.INVOICE_KEY = l.INVOICE_KEY "
        f"JOIN DIM_CUST c ON c.CUST_KEY = f.{column} GROUP BY c.CUST_NM").fetchall()}


@pytest.mark.parametrize("missing", [None, "BILL_TO_CUST_KEY", "PYR_CUST_KEY"])
def test_from_the_lines_the_reader_is_asked_which_whichever_matches_better(missing):
    con = _warehouse("BILL_TO_CUST_KEY", "PYR_CUST_KEY", missing=missing)
    model = _model(con, f"eval-two-roles-{(missing or 'none').lower()}")
    roles = {model.columns[j.from_columns[0]].name: j.role for j in model.joins.values()
             if model.tables[j.to_table].name == "DIM_CUST"}
    assert roles == {"BILL_TO_CUST_KEY": "Bill to customer", "PYR_CUST_KEY": "Payer customer"}
    for measure in ("line_amount", "number_of_invoices"):       # from the lines, and from the invoice itself
        with pytest.raises(ResolveError) as asked:
            _by_customer(con, model, measure)
        assert asked.value.kind == "ambiguous"
        assert sorted(asked.value.options) == ["Bill to customer", "Payer customer"]


def test_the_readers_choice_is_followed_from_the_lines():
    con = _warehouse("BILL_TO_CUST_KEY", "PYR_CUST_KEY", missing="BILL_TO_CUST_KEY")
    model = _model(con, "eval-two-roles-chosen")
    assert _by_customer(con, model, "line_amount", "Bill to customer") == pytest.approx(
        _line_amount_by(con, "BILL_TO_CUST_KEY"))
    assert _by_customer(con, model, "line_amount", "Payer customer") == pytest.approx(
        _line_amount_by(con, "PYR_CUST_KEY"))


@pytest.mark.parametrize("customers, taken", [(("BILL_TO_CUST_KEY",), "BILL_TO_CUST_KEY"),
                                              (("CUST_KEY", "PYR_CUST_KEY"), "CUST_KEY")])
def test_one_link_with_a_role_or_a_plain_one_beside_it_is_still_taken(customers, taken):
    con = _warehouse(*customers)
    model = _model(con, f"eval-two-roles-taken-{len(customers)}")
    assert _by_customer(con, model, "line_amount") == pytest.approx(_line_amount_by(con, taken))


def test_routes_through_different_tables_are_ranked_as_before():
    # The invoice's one customer is its payer, the shipment's its ship-to (a few not on file).
    con = _warehouse("PYR_CUST_KEY")
    rng = random.Random(4)
    con.execute("CREATE TABLE FCT_SHIPMENT (SHIPMENT_KEY INTEGER, SHIP_TO_CUST_KEY INTEGER, SHIP_DT DATE)")
    con.executemany("INSERT INTO FCT_SHIPMENT VALUES (?,?,?)", [
        (i, 900 + i % 5 if i % 300 == 7 else rng.randint(1, 120), dt.date(2024, 1, 1) + dt.timedelta(days=i % 365))
        for i in range(1, 1501)])
    con.execute("ALTER TABLE FCT_INVOICE_LINE ADD COLUMN SHIPMENT_KEY INTEGER")
    con.execute("UPDATE FCT_INVOICE_LINE SET SHIPMENT_KEY = 1 + (INVOICE_LINE_KEY * 7) % 1500")
    model = _model(con, "eval-two-roles-two-tables")
    assert _by_customer(con, model, "line_amount") == pytest.approx(_line_amount_by(con, "PYR_CUST_KEY"))


def test_a_step_with_a_role_every_route_shares_is_no_part_of_the_choice():
    # The lines point at their orders by SALES_ORDER_KEY, their one order link: it carries a role, on both routes.
    con = _warehouse("BILL_TO_CUST_KEY", "PYR_CUST_KEY", missing="BILL_TO_CUST_KEY")
    con.execute("ALTER TABLE FCT_INVOICE_LINE RENAME COLUMN INVOICE_KEY TO SALES_INVOICE_KEY")
    model = _model(con, "eval-two-roles-shared-step")
    with pytest.raises(ResolveError) as asked:
        _by_customer(con, model, "line_amount")
    assert sorted(asked.value.options) == ["Bill to customer", "Payer customer"]
    for option, column in (("Bill to customer", "BILL_TO_CUST_KEY"), ("Payer customer", "PYR_CUST_KEY")):
        assert _by_customer(con, model, "line_amount", option) == pytest.approx({k: float(v) for k, v in con.execute(
            "SELECT c.CUST_NM, SUM(l.LINE_AMT) FROM FCT_INVOICE_LINE l JOIN FCT_INVOICE f "
            f"ON f.INVOICE_KEY = l.SALES_INVOICE_KEY JOIN DIM_CUST c ON c.CUST_KEY = f.{column} "
            "GROUP BY c.CUST_NM").fetchall()})


def test_a_choice_two_roles_set_apart_is_followed_by_both():
    # The lines hold their invoice and their original invoice, neither a plain link: four routes, two roles each.
    con = _warehouse("BILL_TO_CUST_KEY", "PYR_CUST_KEY", missing="BILL_TO_CUST_KEY")
    con.execute("ALTER TABLE FCT_INVOICE_LINE RENAME COLUMN INVOICE_KEY TO SALES_INVOICE_KEY")
    con.execute("ALTER TABLE FCT_INVOICE_LINE ADD COLUMN ORIG_INVOICE_KEY INTEGER")
    con.execute("UPDATE FCT_INVOICE_LINE SET ORIG_INVOICE_KEY = 1 + (INVOICE_LINE_KEY * 13) % 2000")
    model = _model(con, "eval-two-roles-two-steps")
    with pytest.raises(ResolveError) as asked:
        _by_customer(con, model, "line_amount")
    options = asked.value.options
    assert len(options) == 4 and all(", " in o for o in options), options
    for option in options:
        invoice = "SALES_INVOICE_KEY" if option.startswith("Sales") else "ORIG_INVOICE_KEY"
        customer = "BILL_TO_CUST_KEY" if "Bill to" in option else "PYR_CUST_KEY"
        assert _by_customer(con, model, "line_amount", option) == pytest.approx({k: float(v) for k, v in con.execute(
            f"SELECT c.CUST_NM, SUM(l.LINE_AMT) FROM FCT_INVOICE_LINE l JOIN FCT_INVOICE f ON f.INVOICE_KEY = l.{invoice} "
            f"JOIN DIM_CUST c ON c.CUST_KEY = f.{customer} GROUP BY c.CUST_NM").fetchall()}), option
