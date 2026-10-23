"""The one link named for the table it points at, and nothing more, is that table's own link.

Found on a sales warehouse whose invoices hold the customer and the payer customer: "net sales by customer" was
answered with "Customer can be reached more than one way", and two buttons, Customer and Pyr customer. Where a
table joins another more than one way, Learn named every link for its column, the plain one too ("Customer"), and
a link carrying a role is never taken unasked. Now:

* the link whose column names only the table it points at (CUST_KEY, CUSTOMER_ID to the customers) keeps no role:
  a question about the customer takes it, and the payer, ship-to or parent customer is asked for by name;
* a model learned before is read the same way when it is loaded, before the admin's decisions, which still win;
* an answer saved, pinned or followed up with the reader's earlier choice still runs: "Customer" goes the way a
  question naming no role does, from the invoice or from its lines, and "Pyr customer" is the payer customer; a role
  only spelled like another ("Prior customer" for the "Priority customer") is asked about, never taken;
* where every link carries a role (a bill-to and a ship-to customer, no plain one), the reader is still asked, and
  so where another link to the customers has a name that reads as nothing (C07): the plain one keeps its role;
* the party just before the customer reads in whole words: Payer customer, Ultimate customer (a PRN customer is left
  as it is written: in a pharmacy's data PRN is a dose taken as needed), unless another link is named so already;
  before an order, a PYR is a prior year's.

Dates are untouched: every date of a row keeps its role, and the date roles decide which one a question uses.
Invented data only.
"""

from __future__ import annotations

import datetime as dt
import random

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.service import own_links
from core2 import ids
from core2.compile.compiler import compile_query
from core2.model.overrides import apply_overrides
from core2.plan.ir import Plan
from core2.resolve import paths as P
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2025, 6, 30)


def _warehouse(customer: str, *others: str) -> duckdb.DuckDBPyConnection:
    """120 customers; 3,000 invoices, each holding ``customer`` and ``others`` for the customers, the others the
    same customer six times in ten."""
    rng = random.Random(len(others) * 7 + len(customer))
    con = duckdb.connect()
    con.execute("CREATE TABLE DIM_CUST (CUST_KEY INTEGER, CUST_NM VARCHAR, CUST_TYP VARCHAR)")
    con.executemany("INSERT INTO DIM_CUST VALUES (?,?,?)",
                    [(i, f"Customer {i:03d}", rng.choice(["Retail", "Trade", "Online"])) for i in range(1, 121)])
    columns = ", ".join(f"{c} INTEGER" for c in (customer, *others))
    con.execute(f"CREATE TABLE FCT_INVOICE (INVOICE_KEY INTEGER, {columns}, NET_AMT DECIMAL(12,2), INVOICE_DT DATE)")
    rows = []
    for i in range(1, 3001):
        c = rng.randint(1, 120)
        rows.append((i, c, *[c if rng.random() < 0.6 else rng.randint(1, 120) for _ in others],
                     rng.randint(100, 99999) / 100, dt.date(2024, 1, 1) + dt.timedelta(days=i % 365)))
    con.executemany(f"INSERT INTO FCT_INVOICE VALUES ({', '.join('?' * len(rows[0]))})", rows)
    return con


def _model(con: duckdb.DuckDBPyConnection, client: str):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse, declared_fks=[]), client_id=client,
                       options=BuildOptions(workers=1, today=lambda: dt.datetime(2025, 6, 30, tzinfo=dt.timezone.utc)))


def _tables(model):
    by_name = {t.name: k for k, t in model.tables.items()}
    return by_name["FCT_INVOICE"], by_name["DIM_CUST"]


def _roles(model) -> dict[str, str | None]:
    invoices, customers = _tables(model)
    return {model.columns[j.from_columns[0]].name: j.role for j in model.joins.values()
            if j.from_table == invoices and j.to_table == customers}


def _by_customer(model, via: str | None = None):
    plan = Plan.model_validate({"intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer"],
                                "via": {"customer": via} if via else {}})
    return compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")


def _sales_by(con, column: str) -> dict[str, float]:
    return {k: float(v) for k, v in con.execute(
        f"SELECT c.CUST_NM, SUM(f.NET_AMT) FROM FCT_INVOICE f JOIN DIM_CUST c ON c.CUST_KEY = f.{column} "
        "GROUP BY c.CUST_NM").fetchall()}


def _answer(con, compiled) -> dict[str, float]:
    return {k: float(v) for k, v in con.execute(compiled.sql).fetchall()}


@pytest.mark.parametrize("other, role", [("PYR_CUST_KEY", "Payer customer"), ("SHIP_TO_CUST_KEY", "Ship to customer"),
                                         ("PRN_CUST_KEY", "Prn customer"), ("ULT_CUST_KEY", "Ultimate customer"),
                                         ("END_CUST_KEY", "End customer")])
def test_the_customer_beside_another_customer_is_the_plain_link(other, role):
    con = _warehouse("CUST_KEY", other)
    model = _model(con, f"eval-own-link-{other.lower()}")
    assert _roles(model) == {"CUST_KEY": None, other: role}
    invoices, customers = _tables(model)
    assert model.columns[P.best_path(model, invoices, customers).joins[0].from_columns[0]].name == "CUST_KEY"
    assert _answer(con, _by_customer(model)) == pytest.approx(_sales_by(con, "CUST_KEY"))


def test_a_customer_named_in_full_is_the_plain_link_too():
    con = _warehouse("CUSTOMER_ID", "PAYER_CUSTOMER_ID")
    con.execute("ALTER TABLE DIM_CUST RENAME COLUMN CUST_KEY TO CUSTOMER_ID")
    model = _model(con, "eval-own-link-in-full")
    assert _roles(model) == {"CUSTOMER_ID": None, "PAYER_CUSTOMER_ID": "Payer customer"}


def test_with_no_plain_link_the_reader_is_still_asked_which():
    con = _warehouse("BILL_TO_CUST_KEY", "SHIP_TO_CUST_KEY")
    model = _model(con, "eval-own-link-none-plain")
    assert _roles(model) == {"BILL_TO_CUST_KEY": "Bill to customer", "SHIP_TO_CUST_KEY": "Ship to customer"}
    with pytest.raises(ResolveError) as asked:
        _by_customer(model)
    assert asked.value.kind == "ambiguous"
    assert sorted(asked.value.options) == ["Bill to customer", "Ship to customer"]


def test_every_date_keeps_its_role():
    con = _warehouse("CUST_KEY", "PYR_CUST_KEY")
    con.execute("CREATE TABLE DIM_DT (DT_KEY INTEGER, CAL_DT DATE, CAL_YR INTEGER, CAL_MTH INTEGER)")
    con.execute("INSERT INTO DIM_DT SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER), d, year(d), month(d) "
                "FROM (SELECT DATE '2023-01-01' + INTERVAL (i) DAY AS d FROM range(0, 1100) t(i))")
    con.execute("ALTER TABLE FCT_INVOICE ADD COLUMN DT_KEY INTEGER")
    con.execute("ALTER TABLE FCT_INVOICE ADD COLUMN DUE_DT_KEY INTEGER")
    con.execute("UPDATE FCT_INVOICE SET DT_KEY = CAST(strftime(INVOICE_DT, '%Y%m%d') AS INTEGER), "
                "DUE_DT_KEY = CAST(strftime(INVOICE_DT + INTERVAL 30 DAY, '%Y%m%d') AS INTEGER)")
    model = _model(con, "eval-own-link-dates")
    invoices = _tables(model)[0]
    dated = [j for j in model.joins.values() if j.from_table == invoices and j.to_calendar]
    assert len(dated) == 2 and all(j.role for j in dated), [(j.key, j.role) for j in dated]


def _as_stored_before(model):
    """The model as Learn named it before: the plain link named for its column too, the payer's cut short."""
    old = model.model_copy(deep=True)
    for j in old.joins.values():
        name = old.columns[j.from_columns[0]].name
        if name == "CUST_KEY":
            j.role = "Customer"
        elif name == "PYR_CUST_KEY":
            j.role = "Pyr customer"
    return old


def test_a_model_learned_before_is_read_the_same_way_and_the_admin_still_decides():
    con = _warehouse("CUST_KEY", "PYR_CUST_KEY")
    model = _as_stored_before(_model(con, "eval-own-link-before"))
    invoices, customers = _tables(model)
    plain = next(j for j in model.joins.values() if model.columns[j.from_columns[0]].name == "CUST_KEY")
    with pytest.raises(ResolveError):
        _by_customer(model)
    own_links(model)
    assert model.joins[plain.key].role is None
    assert P.best_path(model, invoices, customers).joins[0].key == plain.key
    assert _answer(con, _by_customer(model)) == pytest.approx(_sales_by(con, "CUST_KEY"))
    # The admin names the customer link on the relationships page: their own link, the learned one turned off.
    mine = plain.model_copy(update={"key": ids.join_key(plain.from_table, plain.from_columns, plain.to_table,
                                                         plain.to_columns, "Sold to customer"),
                                    "role": "Sold to customer", "provenance": "admin", "trust": "admin"})
    assert apply_overrides(model, [{"object_key": f"join:{mine.key}", "field": "define",
                                    "value": mine.model_dump(mode="json")},
                                   {"object_key": f"join:{plain.key}", "field": "trust", "value": "rejected"}]) == []
    assert model.joins[mine.key].role == "Sold to customer" and model.joins[plain.key].trust == "rejected"


def test_an_answer_kept_with_the_readers_earlier_choice_still_runs():
    # Saved, pinned or followed up: the plan keeps the button the reader pressed ("Customer", "Pyr customer").
    con = _warehouse("CUST_KEY", "PYR_CUST_KEY")
    learned = _model(con, "eval-own-link-kept")
    before = _as_stored_before(learned)
    assert _answer(con, _by_customer(before, "Customer")) == pytest.approx(_sales_by(con, "CUST_KEY"))
    own_links(before)
    for model in (before, learned):
        assert _answer(con, _by_customer(model, "Customer")) == pytest.approx(_sales_by(con, "CUST_KEY"))
        assert _answer(con, _by_customer(model, "Pyr customer")) == pytest.approx(_sales_by(con, "PYR_CUST_KEY"))
        assert _answer(con, _by_customer(model, "Payer customer")) == pytest.approx(_sales_by(con, "PYR_CUST_KEY"))


def _with_lines(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """9,000 invoice lines, each on one of the 3,000 invoices."""
    rng = random.Random(2)
    con.execute("CREATE TABLE FCT_INVOICE_LINE (INVOICE_LINE_KEY INTEGER, INVOICE_KEY INTEGER, LINE_NO INTEGER, "
                "LINE_AMT DECIMAL(12,2))")
    con.executemany("INSERT INTO FCT_INVOICE_LINE VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 3000), i % 4 + 1, rng.randint(100, 9999) / 100) for i in range(1, 9001)])
    return con


def _line_amount_by(con, column: str) -> dict[str, float]:
    return {k: float(v) for k, v in con.execute(
        "SELECT c.CUST_NM, SUM(l.LINE_AMT) FROM FCT_INVOICE_LINE l JOIN FCT_INVOICE f ON f.INVOICE_KEY = l.INVOICE_KEY "
        f"JOIN DIM_CUST c ON c.CUST_KEY = f.{column} GROUP BY c.CUST_NM").fetchall()}


def test_an_answer_on_the_lines_kept_with_the_readers_earlier_choice_still_runs():
    # The figure's table is a step from the customers: its lines reach them through the invoice, either way.
    con = _with_lines(_warehouse("CUST_KEY", "PYR_CUST_KEY"))
    learned = _model(con, "eval-own-link-lines")
    before = _as_stored_before(learned)
    stored = before.model_copy(deep=True)
    own_links(stored)

    def by_customer(model, via):
        plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"], "group_by": ["customer"],
                                    "via": {"customer": via}})
        return _answer(con, compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb"))

    for model in (before, stored, learned):
        assert by_customer(model, "Customer") == pytest.approx(_line_amount_by(con, "CUST_KEY"))
        assert by_customer(model, "Pyr customer") == pytest.approx(_line_amount_by(con, "PYR_CUST_KEY"))
    plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"], "group_by": ["customer"]})
    for model in (stored, learned):
        assert _answer(con, compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb")) \
            == pytest.approx(_line_amount_by(con, "CUST_KEY"))


def test_from_the_lines_the_customer_is_the_plain_one_however_well_the_payer_matches():
    # A few invoices name a customer not on file: the payer customer matches every row, the customer nearly all.
    con = _with_lines(_warehouse("CUST_KEY", "PYR_CUST_KEY"))
    con.execute("UPDATE FCT_INVOICE SET CUST_KEY = 900 + INVOICE_KEY % 5 WHERE INVOICE_KEY % 400 = 7")
    learned = _model(con, "eval-own-link-lines-orphans")
    stored = _as_stored_before(learned)
    own_links(stored)

    def by_customer(model, via=None):
        plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"], "group_by": ["customer"],
                                    "via": {"customer": via} if via else {}})
        return {k: v for k, v in _answer(con, compile_query(resolve(plan, model, Context(today=TODAY)), model,
                                                            "duckdb")).items() if k is not None}

    for model in (learned, stored):
        assert by_customer(model) == pytest.approx(_line_amount_by(con, "CUST_KEY"))
        assert by_customer(model, "Customer") == pytest.approx(_line_amount_by(con, "CUST_KEY"))
        assert by_customer(model, "Pyr customer") == pytest.approx(_line_amount_by(con, "PYR_CUST_KEY"))


def test_a_route_through_the_only_link_with_a_role_is_ranked_like_any_other():
    # The orders' one customer is their bill-to customer; the contracts' customer is a link not every row finds.
    rng = random.Random(5)
    con = duckdb.connect()
    con.execute("CREATE TABLE DIM_CUST (CUST_KEY INTEGER, CUST_NM VARCHAR)")
    con.executemany("INSERT INTO DIM_CUST VALUES (?,?)", [(i, f"Customer {i:03d}") for i in range(1, 101)])
    con.execute("CREATE TABLE DIM_CONTRACT (CONTRACT_KEY INTEGER, CONTRACT_NM VARCHAR, CUST_KEY INTEGER)")
    con.executemany("INSERT INTO DIM_CONTRACT VALUES (?,?,?)", [
        (i, f"Contract {i:03d}", 900 + i if rng.random() < 0.06 else rng.randint(1, 100)) for i in range(1, 301)])
    con.execute("CREATE TABLE FCT_ORDER (ORDER_KEY INTEGER, BILL_TO_CUST_KEY INTEGER, ORDER_DT DATE)")
    con.executemany("INSERT INTO FCT_ORDER VALUES (?,?,?)", [
        (i, rng.randint(1, 100), dt.date(2024, 1, 1) + dt.timedelta(days=i % 365)) for i in range(1, 1501)])
    con.execute("CREATE TABLE FCT_ORDER_LINE (ORDER_LINE_KEY INTEGER, ORDER_KEY INTEGER, CONTRACT_KEY INTEGER, "
                "LINE_NO INTEGER, LINE_AMT DECIMAL(12,2))")
    con.executemany("INSERT INTO FCT_ORDER_LINE VALUES (?,?,?,?,?)", [
        (i, rng.randint(1, 1500), rng.randint(1, 300), i % 4 + 1, rng.randint(100, 99999) / 100) for i in range(1, 6001)])
    model = _model(con, "eval-own-link-bill-to")
    plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"], "group_by": ["customer"]})
    got = _answer(con, compile_query(resolve(plan, model, Context(today=TODAY)), model, "duckdb"))
    assert got == pytest.approx({k: float(v) for k, v in con.execute(
        "SELECT c.CUST_NM, SUM(l.LINE_AMT) FROM FCT_ORDER_LINE l JOIN FCT_ORDER o ON o.ORDER_KEY = l.ORDER_KEY "
        "JOIN DIM_CUST c ON c.CUST_KEY = o.BILL_TO_CUST_KEY GROUP BY c.CUST_NM").fetchall()})


def test_a_kept_customer_says_the_other_route_as_a_question_naming_none_does():
    # The lines reach the customers through their invoice or their shipment, as well one way as the other.
    con = _with_lines(_warehouse("CUST_KEY", "PYR_CUST_KEY"))
    rng = random.Random(9)
    con.execute("CREATE TABLE FCT_SHIPMENT (SHIPMENT_KEY INTEGER, CUST_KEY INTEGER, SHIP_DT DATE)")
    con.executemany("INSERT INTO FCT_SHIPMENT VALUES (?,?,?)",
                    [(i, rng.randint(1, 120), dt.date(2024, 1, 1) + dt.timedelta(days=i % 365)) for i in range(1, 1001)])
    con.execute("ALTER TABLE FCT_INVOICE_LINE ADD COLUMN SHIPMENT_KEY INTEGER")
    con.execute("UPDATE FCT_INVOICE_LINE SET SHIPMENT_KEY = 1 + (INVOICE_LINE_KEY * 7) % 1000")
    model = _model(con, "eval-own-link-two-routes")

    def cautions(via):
        plan = Plan.model_validate({"intent": "breakdown", "measures": ["line_amount"], "group_by": ["customer"],
                                    "via": {"customer": via} if via else {}})
        return resolve(plan, model, Context(today=TODAY)).cautions

    said = cautions(None)
    assert said and cautions("Customer") == said


def test_a_party_is_read_so_only_just_before_a_party():
    # PYR_BILL_CUST_KEY says more between the PYR and the customer; a PYR order is a prior year's, no payer's.
    con = _warehouse("CUST_KEY", "PYR_BILL_CUST_KEY")
    assert _roles(_model(con, "eval-own-link-pyr-bill")) == {"CUST_KEY": None, "PYR_BILL_CUST_KEY": "Pyr bill customer"}
    rng = random.Random(1)
    con = duckdb.connect()
    con.execute("CREATE TABLE DIM_ORDER (ORD_KEY INTEGER, ORD_NO VARCHAR)")
    con.executemany("INSERT INTO DIM_ORDER VALUES (?,?)", [(i, f"O{i:05d}") for i in range(1, 501)])
    con.execute("CREATE TABLE FCT_RETURN (RTN_KEY INTEGER, ORD_KEY INTEGER, PYR_ORD_KEY INTEGER, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO FCT_RETURN VALUES (?,?,?,?)", [
        (i, rng.randint(1, 500), rng.randint(1, 500), rng.randint(100, 9999) / 100) for i in range(1, 2001)])
    model = _model(con, "eval-own-link-pyr-order")
    assert {model.columns[j.from_columns[0]].name: j.role for j in model.joins.values() if not j.to_calendar} \
        == {"ORD_KEY": None, "PYR_ORD_KEY": "Pyr order"}
    # A budget's PYR account is last year's, an account of the ledger; an ULT payor is the ultimate one.
    for table, key, other, fact, role in [("DIM_ACCOUNT", "ACCT_KEY", "PYR_ACCT_KEY", "FCT_BUDGET", "Pyr account"),
                                          ("DIM_PAYOR", "PAYOR_KEY", "ULT_PAYOR_KEY", "FCT_CLAIM", "Ultimate payor")]:
        con = duckdb.connect()
        con.execute(f"CREATE TABLE {table} ({key} INTEGER, NM VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?,?)", [(i, f"N{i:04d}") for i in range(1, 301)])
        con.execute(f"CREATE TABLE {fact} (ROW_KEY INTEGER, {key} INTEGER, {other} INTEGER, AMT DECIMAL(10,2))")
        con.executemany(f"INSERT INTO {fact} VALUES (?,?,?,?)", [
            (i, rng.randint(1, 300), rng.randint(1, 300), rng.randint(100, 9999) / 100) for i in range(1, 2001)])
        model = _model(con, f"eval-own-link-{table.lower()}")
        assert {model.columns[j.from_columns[0]].name: j.role for j in model.joins.values() if not j.to_calendar} \
            == {key: None, other: role}


def test_a_party_cut_short_beside_the_same_party_in_full_keeps_its_own_name():
    con = _warehouse("CUST_KEY", "PYR_CUST_KEY", "PAYER_CUST_KEY")
    model = _model(con, "eval-own-link-two-payers")
    assert _roles(model) == {"CUST_KEY": None, "PYR_CUST_KEY": "Pyr customer", "PAYER_CUST_KEY": "Payer customer"}
    assert _answer(con, _by_customer(model, "Pyr customer")) == pytest.approx(_sales_by(con, "PYR_CUST_KEY"))
    assert _answer(con, _by_customer(model, "Payer customer")) == pytest.approx(_sales_by(con, "PAYER_CUST_KEY"))
    with pytest.raises(ResolveError) as asked:
        _by_customer(model, "Payr customer")      # either of them: asked, never guessed
    assert asked.value.kind == "unknown"


def test_a_role_only_spelled_like_the_one_asked_for_is_asked_about():
    con = _warehouse("CUST_KEY", "PRI_CUST_KEY")
    model = _model(con, "eval-own-link-priority")
    assert _roles(model) == {"CUST_KEY": None, "PRI_CUST_KEY": "Priority customer"}
    with pytest.raises(ResolveError) as asked:
        _by_customer(model, "Prior customer")
    assert asked.value.kind == "unknown" and asked.value.options == ["Priority customer"]
    with pytest.raises(ResolveError):
        _by_customer(model, "Custodian")         # only spelled like the customer: no customer link of its own
    assert _answer(con, _by_customer(model, "Priority customer")) == pytest.approx(_sales_by(con, "PRI_CUST_KEY"))


def test_beside_a_link_whose_name_reads_as_nothing_the_plain_link_keeps_its_role():
    con = _warehouse("CUST_KEY", "C07")
    warehouse = DuckDBWarehouse(con)
    declared = [{"parent_table": "FCT_INVOICE", "parent_col": "C07", "ref_table": "DIM_CUST", "ref_col": "CUST_KEY"}]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=declared), client_id="eval-own-link-c07",
                        options=BuildOptions(workers=1))
    roles = _roles(model)
    assert roles.get("C07") is None and roles.get("CUST_KEY") == "Customer", roles
    old = model.model_copy(deep=True)
    own_links(old)
    assert _roles(old) == roles, "a model learned before keeps it too"


def test_a_link_an_admin_made_keeps_its_role():
    # Learn found the payer customer alone; the admin's own link to the customer, under a name of their own.
    con = _warehouse("CUST_KEY", "PYR_CUST_KEY")
    model = _model(con, "eval-own-link-admin")
    plain = next(j for j in model.joins.values() if model.columns[j.from_columns[0]].name == "CUST_KEY")
    del model.joins[plain.key]
    mine = plain.model_copy(update={"key": plain.key + "#mine", "role": "Customer", "provenance": "admin",
                                    "trust": "admin"})
    model.joins[mine.key] = mine
    own_links(model)
    assert model.joins[mine.key].role == "Customer"
