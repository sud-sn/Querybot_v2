"""A table that keeps each member once per version is read as its members, not its rows.

A warehouse often keeps a customer once per version: each change of segment adds a row with the same customer
number, and one row of each is the current one -- marked by a flag (IS_CURRENT = 1), or by a validity period
still open (VALID_TO empty, or 9999-12-31). Read as an ordinary list, a customer with three versions was three
customers: counted three times, listed three times, and a link on the customer number found three rows for
each order -- which the join check (J5) now refuses rather than count twice.

Learn now recognises such a table, checking each claim against the data: the number repeats, and the rows the
marker keeps hold each number once, for nearly every number. A link on the member's number keeps its current
row, and the answer says so; a fact holding one version's own row id keeps that version (as it was then). The
members are counted, and told apart, by their number. A table that only looks the part -- a flag on a list
whose numbers never repeat, or two "current" rows for some member -- is left as it is.

A small warehouse in DuckDB, invented data; every total is checked against hand-written SQL.
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
OPEN = {"flag": None, "empty end": None, "far-off end": dt.date(9999, 12, 31)}
# How a flag is kept: its type, and its values for the current version and the others.
FLAGS = {"number": ("INTEGER", "1", "0"), "true or false": ("BOOLEAN", "TRUE", "FALSE"), "Y or N": ("VARCHAR", "'Y'", "'N'")}
# The current rows, as hand-written SQL, for each marker.
CURRENT = {("flag", "number"): "c.is_current = 1", ("flag", "true or false"): "c.is_current",
           ("flag", "Y or N"): "c.is_current = 'Y'", ("empty end", None): "c.valid_to IS NULL",
           ("far-off end", None): "c.valid_to >= DATE '2900-01-01'"}


def _warehouse(marker: str = "flag", *, two_current: bool = False, kept: str = "number") -> duckdb.DuckDBPyConnection:
    """Forty customers, one to three versions each (the segment changes); orders by customer number, payments by
    the version's own row id."""
    con = duckdb.connect()
    con.execute("CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name VARCHAR)")
    con.execute("INSERT INTO regions VALUES (1, 'North'), (2, 'South'), (3, 'West')")
    flag_type, yes, no = FLAGS[kept]
    flag = f", is_current {flag_type}" if marker == "flag" else ""
    con.execute("CREATE TABLE customer_versions (customer_sk INTEGER PRIMARY KEY, customer_id INTEGER, "
                f"customer_name VARCHAR, segment VARCHAR, region_id INTEGER, valid_from DATE, valid_to DATE{flag})")
    rows, sk = [], 0
    for c in range(1, 41):
        versions = 1 + c % 3
        for v in range(1, versions + 1):
            sk += 1
            start = dt.date(2024, 1, 1) + dt.timedelta(days=120 * (v - 1))
            last = v == versions
            doubled = two_current and c % 5 == 0 and v == versions - 1     # a second "current" row, open too
            end = (OPEN[marker] if last or doubled else start + dt.timedelta(days=119)) if marker != "flag" else (
                None if last or doubled else start + dt.timedelta(days=119))
            ends = "NULL" if end is None else f"DATE '{end.isoformat()}'"
            current = f", {yes if last or doubled else no}" if flag else ""
            segment = ("Retail", "Trade", "Online")[(c + v) % 3]
            rows.append(f"({sk}, {c}, 'Customer {c:02d}', '{segment}', {1 + c % 3}, DATE '{start.isoformat()}', "
                        f"{ends}{current})")
    con.execute("INSERT INTO customer_versions VALUES " + ", ".join(rows))
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE, customer_id INTEGER, "
                "order_total DECIMAL(12, 2))")
    con.execute("INSERT INTO orders SELECT i, DATE '2025-01-01' + CAST(i % 360 AS INTEGER), 1 + i % 40, "
                "round(20 + (i * 37) % 400, 2) FROM range(1, 801) t(i)")
    con.execute("CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, payment_date DATE, customer_sk INTEGER, "
                "payment_amount DECIMAL(12, 2))")
    con.execute(f"INSERT INTO payments SELECT i, DATE '2025-01-01' + CAST(i % 360 AS INTEGER), 1 + i % {sk}, "
                "round(10 + (i * 17) % 300, 2) FROM range(1, 601) t(i)")
    return con


def _learn(con):
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="versions", options=BuildOptions(workers=1))


@pytest.fixture(scope="module")
def flagged():
    con = _warehouse("flag")
    return con, _learn(con)


def _table(model, name):
    return next(t for t in model.tables.values() if t.name == name)


def _column(model, table, name):
    return next(c.key for c in model.columns.values() if model.tables[c.table].name == table and c.name == name)


def _attribute(model, table, name):
    return next(a.slug for a in model.attributes.values() if a.column == _column(model, table, name))


def _ask(con, model, plan):
    answer = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answer,
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _rows(payload) -> dict:
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    assert headers, payload["answer"]["headline"] + " / " + str(payload.get("trust", {}).get("stopped"))
    return {tuple(str(r[h]) for h in headers[:-1]): float(r[headers[-1]]) for r in data.get("rows") or []}


# ── what Learn recognises ─────────────────────────────────────────────────


@pytest.mark.parametrize("marker,current", [
    ("flag", [("is_current", "eq", [1])]),
    ("empty end", [("valid_to", "is_null", [])]),
    ("far-off end", [("valid_to", "gte", ["2900-01-01"])]),
])
def test_learn_recognises_a_table_of_versions_by_what_marks_the_current_one(marker, current):
    con = _warehouse(marker)
    model = _learn(con)
    held = _table(model, "customer_versions").versions
    assert held is not None, marker
    assert held.business_key == _column(model, "customer_versions", "customer_id")
    assert [(model.columns[c.column].name, c.op, c.values) for c in held.current] == current
    assert held.members == 40
    if marker != "flag":
        assert (model.columns[held.valid_from].name, model.columns[held.valid_to].name) == ("valid_from", "valid_to")


@pytest.mark.parametrize("kept,value", [("true or false", True), ("Y or N", "Y")])
def test_a_flag_kept_as_true_or_false_or_as_a_letter_is_read_too(kept, value):
    model = _learn(_warehouse("flag", kept=kept))
    held = _table(model, "customer_versions").versions
    assert held is not None and [(c.op, c.values) for c in held.current] == [("eq", [value])]


def test_a_list_whose_numbers_never_repeat_is_not_one(flagged):
    _, model = flagged
    assert all(t.versions is None for t in model.tables.values() if t.name != "customer_versions")


def test_two_current_rows_for_some_member_are_not_believed():
    model = _learn(_warehouse("flag", two_current=True))
    assert _table(model, "customer_versions").versions is None


# ── questions ─────────────────────────────────────────────────────────────


def test_a_link_on_the_members_number_keeps_its_current_version(flagged):
    con, model = flagged
    link = next(j for j in model.joins.values() if model.tables[j.from_table].name == "orders"
                and model.tables[j.to_table].name == "customer_versions")
    assert [model.columns[c.column].name for c in link.conditions] == ["is_current"]
    assert link.to_unique and link.cardinality == "many_to_one"
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["order_total"],
                                "group_by": [_attribute(model, "customer_versions", "segment")]})
    want = {(s,): float(v) for s, v in con.execute(
        "SELECT c.segment, SUM(o.order_total) FROM orders o JOIN customer_versions c "
        "ON c.customer_id = o.customer_id AND c.is_current = 1 GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)
    right = float(con.execute("SELECT SUM(order_total) FROM orders").fetchone()[0])
    assert sum(_rows(payload).values()) == pytest.approx(right), "an order was counted more than once"
    notes = " ".join(payload["trust"].get("notes") or []) + json.dumps(payload.get("trust"))
    assert "as each is now (its current version" in notes


@pytest.mark.parametrize("marker,kept", [("flag", "true or false"), ("flag", "Y or N"), ("empty end", None),
                                         ("far-off end", None)])
def test_every_kind_of_marker_gives_each_order_its_customers_current_segment(marker, kept):
    con = _warehouse(marker, kept=kept or "number")
    model = _learn(con)
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["order_total"],
                                "group_by": [_attribute(model, "customer_versions", "segment")]})
    want = {(s,): float(v) for s, v in con.execute(
        "SELECT c.segment, SUM(o.order_total) FROM orders o JOIN customer_versions c "
        f"ON c.customer_id = o.customer_id AND {CURRENT[(marker, kept)]} GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)
    right = float(con.execute("SELECT SUM(order_total) FROM orders").fetchone()[0])
    assert sum(_rows(payload).values()) == pytest.approx(right), "an order was counted more than once"


def test_a_fact_holding_a_versions_own_row_keeps_that_version(flagged):
    con, model = flagged
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["payment_amount"],
                                "group_by": [_attribute(model, "customer_versions", "segment")]})
    want = {(s,): float(v) for s, v in con.execute(
        "SELECT c.segment, SUM(p.payment_amount) FROM payments p JOIN customer_versions c USING (customer_sk) "
        "GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


def test_the_members_are_counted_once_whatever_their_versions(flagged):
    con, model = flagged
    counts = [m for m in model.measures.values() if m.table == _table(model, "customer_versions").key
              and m.expr.model_dump().get("agg") == "count_distinct"]
    assert len(counts) == 1 and counts[0].expr.column == _column(model, "customer_versions", "customer_id")
    payload = _ask(con, model, {"intent": "value", "measures": [counts[0].slug]})
    assert payload["kpi"]["value"] == 40


def test_orders_by_customer_name_list_each_customer_once(flagged):
    con, model = flagged
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["order_total"],
                                "group_by": [_attribute(model, "customer_versions", "customer_name")]})
    got = _rows(payload)
    names = [k[0] for k in got]
    assert len(names) == len(set(names)) == 40, "a customer was listed once per version"
    want = dict(con.execute("SELECT c.customer_name, SUM(o.order_total) FROM orders o JOIN customer_versions c "
                            "ON c.customer_id = o.customer_id AND c.is_current = 1 GROUP BY 1").fetchall())
    assert {k[0]: v for k, v in got.items()} == pytest.approx({k: float(v) for k, v in want.items()})


def test_payments_by_customer_name_add_up_every_version_under_its_customer(flagged):
    """Payments hold each version's own row: a customer's payments under every version are still one customer."""
    con, model = flagged
    payload = _ask(con, model, {"intent": "breakdown", "measures": ["payment_amount"],
                                "group_by": [_attribute(model, "customer_versions", "customer_name")]})
    got = _rows(payload)
    names = [k[0] for k in got]
    assert len(names) == len(set(names)), "a customer was listed once per version"
    want = dict(con.execute("SELECT c.customer_name, SUM(p.payment_amount) FROM payments p "
                            "JOIN customer_versions c USING (customer_sk) GROUP BY 1").fetchall())
    assert {k[0]: v for k, v in got.items()} == pytest.approx({k: float(v) for k, v in want.items()})
    # Shown beside each name: the customer's own number, never one version's row id.
    assert sorted(int(k[1]) for k in got) == list(range(1, 41))


def test_the_members_are_named_by_their_name_though_it_repeats_once_per_version(flagged):
    _, model = flagged
    entity = next(e for e in model.entities.values() if e.table == _table(model, "customer_versions").key)
    assert entity.label_column == _column(model, "customer_versions", "customer_name")
    assert entity.code_column == _column(model, "customer_versions", "customer_id")


def test_top_customers_are_ranked_as_customers_not_as_versions(flagged):
    """Asked about the customers themselves, every version's payments land on the one customer."""
    con, model = flagged
    entity = next(e for e in model.entities.values() if e.table == _table(model, "customer_versions").key)
    payload = _ask(con, model, {"intent": "rank", "measures": ["payment_amount"], "group_by": [entity.slug],
                                "sort": [{"by": "payment_amount", "desc": True}], "limit": 5})
    got = _rows(payload)
    want = con.execute("SELECT c.customer_name, SUM(p.payment_amount) s FROM payments p JOIN customer_versions c "
                       "USING (customer_sk) GROUP BY c.customer_id, c.customer_name ORDER BY s DESC LIMIT 6").fetchall()
    assert want[4][1] != want[5][1], "the fifth place is tied: the test data needs changing"
    assert {k[0]: v for k, v in got.items()} == pytest.approx({n: float(v) for n, v in want[:5]})


# ── every warehouse ───────────────────────────────────────────────────────


@pytest.mark.parametrize("dialect,far_off,flag", [
    ("snowflake", "VALID_TO >= CAST('2900-01-01' AS DATE)", "IS_CURRENT = TRUE"),
    ("oracle", "VALID_TO >= TO_DATE('2900-01-01', 'YYYY-MM-DD')", "IS_CURRENT = 1"),
    ("tsql", "VALID_TO >= CAST('2900-01-01' AS DATE)", "IS_CURRENT = 1"),
])
def test_learns_check_writes_dates_and_flags_as_each_warehouse_reads_them(dialect, far_off, flag):
    """Oracle reads a date in text by the session's own format, unless TO_DATE names it; SQL Server and Oracle
    keep true or false as 1 or 0."""
    from core2.bootstrap.versions import condition_sql
    assert condition_sql("VALID_TO", "gte", ["2900-01-01"], dialect).sql(dialect=dialect) == far_off
    assert condition_sql("IS_CURRENT", "eq", [True], dialect).sql(dialect=dialect) == flag


def test_a_true_or_false_flag_stays_one_when_the_model_is_saved_and_read_back():
    from core2.model.schema import SemanticModel
    model = _learn(_warehouse("flag", kept="true or false"))
    again = SemanticModel.model_validate_json(model.model_dump_json())
    held = _table(again, "customer_versions").versions
    assert held.current[0].values == [True] and isinstance(held.current[0].values[0], bool)
    link = next(j for j in again.joins.values() if again.tables[j.to_table].name == "customer_versions"
                and again.tables[j.from_table].name == "orders")
    assert link.conditions[0].values == [True] and isinstance(link.conditions[0].values[0], bool)
