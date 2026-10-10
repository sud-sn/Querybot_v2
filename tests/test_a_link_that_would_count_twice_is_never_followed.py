"""A link that would count a row more than once is never followed, and the answer says why.

Questions follow links from a table to one row of another (an order to its store). A link whose target holds
more than one row for some key -- a customer number kept once for each segment the customer belongs to, which
the database declares as a foreign key all the same -- counts each order once per segment, and its totals come
out too high with nothing said. Learn never measured it: every link it found was written as pointing at one row.

Now Learn measures each link's target (one row per key, or up to how many), and a link that repeats is never
followed: a question that needs it is refused, saying which rows would be counted more than once. So is a
figure of a header split by its lines (an order total by the products on its lines). A link no one measured --
declared, or brought over from today's setup, in a model learned before this -- is checked against the data
the first time a question follows it, once. An admin's condition that keeps one row per key (each customer's
primary segment) makes it safe again, and a link from today's setup written from the "one" side is turned the
way a question follows it. (A table keeping each customer once per version, with its current row marked, is
another matter: tests/test_a_history_table_is_read_as_its_members.py.)

A small warehouse in DuckDB, invented data; totals are checked against hand-written SQL.
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
from core2.service import Services, Session, _link_check, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 30)
YEAR = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}


def _fk(parent: str, column: str, ref: str, ref_column: str) -> dict:
    return {"constraint_name": f"fk_{parent}_{column}", "parent_schema": "main", "parent_table": parent,
            "parent_col": column, "ref_schema": "main", "ref_table": ref, "ref_col": ref_column, "ordinal": 1}


DECLARED = [_fk("orders", "customer_id", "customer_segments", "customer_id"),
            _fk("orders", "store_id", "stores", "store_id"),
            _fk("stores", "region_id", "regions", "region_id"),
            _fk("order_lines", "order_id", "orders", "order_id")]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE regions (region_id INTEGER PRIMARY KEY, region_name VARCHAR)")
    con.execute("INSERT INTO regions VALUES (1, 'North'), (2, 'South'), (3, 'West')")
    con.execute("CREATE TABLE stores (store_id INTEGER PRIMARY KEY, store_name VARCHAR, region_id INTEGER)")
    con.execute("INSERT INTO stores VALUES " + ", ".join(f"({i}, 'Store {i:02d}', {1 + i % 3})" for i in range(1, 13)))
    con.execute("CREATE TABLE customer_segments (customer_id INTEGER, segment_no INTEGER, segment VARCHAR, "
                "is_primary INTEGER, PRIMARY KEY (customer_id, segment_no))")
    rows = []
    for c in range(1, 41):
        belongs = 1 + c % 3          # each customer in one to three segments
        for v in range(1, belongs + 1):
            segment = ("Retail", "Trade", "Online")[(c + v) % 3]
            rows.append(f"({c}, {v}, '{segment}', {int(v == belongs)})")
    con.execute("INSERT INTO customer_segments VALUES " + ", ".join(rows))
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE, customer_id INTEGER, "
                "store_id INTEGER, order_total DECIMAL(12, 2))")
    con.execute("INSERT INTO orders SELECT i, DATE '2026-01-01' + CAST(i % 180 AS INTEGER), 1 + i % 40, 1 + i % 12, "
                "round(20 + (i * 37) % 400, 2) FROM range(1, 601) t(i)")
    con.execute("CREATE TABLE order_lines (order_id INTEGER, line_no INTEGER, product VARCHAR, quantity INTEGER, "
                "line_amount DECIMAL(12, 2), PRIMARY KEY (order_id, line_no))")
    con.execute("INSERT INTO order_lines SELECT o.order_id, l.n, 'Product ' || ((o.order_id + l.n) % 7), "
                "1 + (o.order_id + l.n) % 5, round(5 + ((o.order_id * l.n) * 13) % 90, 2) "
                "FROM orders o, range(1, 4) l(n) WHERE l.n <= 1 + o.order_id % 3")
    return con


@pytest.fixture(scope="module")
def learned():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=DECLARED), client_id="links",
                        options=BuildOptions(workers=1))
    return con, model


def _join(model, from_name: str, to_name: str):
    found = [j for j in model.joins.values()
             if model.tables[j.from_table].name == from_name and model.tables[j.to_table].name == to_name]
    assert found, (from_name, to_name, [(model.tables[j.from_table].name, model.tables[j.to_table].name)
                                        for j in model.joins.values()])
    return found[0]


def _slug(model, table: str, column: str) -> str:
    """The grouping a plan names for a column (its attribute's slug)."""
    return next(a.slug for a in model.attributes.values() if a.column in model.columns
                and model.tables[model.columns[a.column].table].name == table
                and model.columns[a.column].name == column)


_AMOUNT = {"orders": "order_total", "order_lines": "line_amount"}


def _measure(model, table: str) -> str:
    """The plan's name for the table's amount, added up."""
    return next(m.slug for m in model.measures.values() if model.tables[m.table].name == table and not m.hidden
                and m.expr.model_dump().get("agg") == "sum" and m.slug == _AMOUNT[table])


def _ask(con, model, plan: dict, *, check: bool = True):
    answer = json.dumps({"kind": "query", **plan})
    warehouse = DuckDBWarehouse(con)
    services = Services(model=model, warehouse=warehouse, complete=lambda s, t: answer, index=MemberIndex(),
                        today=TODAY)
    if not check:
        import core2.service as service

        real = service._link_check
        service._link_check = lambda services: None
        try:
            return answer_question("q", services, Session())
        finally:
            service._link_check = real
    return answer_question("q", services, Session())


def _rows(payload) -> dict:
    data = payload.get("data") or {}
    headers = data.get("headers") or []
    return {tuple(str(r[h]) for h in headers[:-1]): float(r[headers[-1]]) for r in data.get("rows") or []}


# ── what Learn measures ───────────────────────────────────────────────────


def test_learn_measures_a_declared_link_whose_target_repeats(learned):
    _, model = learned
    to_versions = _join(model, "orders", "customer_segments")
    assert (to_versions.to_unique, to_versions.max_fanout, to_versions.cardinality, to_versions.target_checked) == (
        False, 3.0, "many_to_many", True)
    assert any(e.kind == "fanout" for e in to_versions.evidence)


@pytest.mark.parametrize("link", [("orders", "stores"), ("stores", "regions"), ("order_lines", "orders")])
def test_learn_measures_the_links_that_find_one_row(learned, link):
    _, model = learned
    j = _join(model, *link)
    assert (j.to_unique, j.max_fanout, j.cardinality, j.target_checked) == (True, 1.0, "many_to_one", True)


# ── at question time ──────────────────────────────────────────────────────


def test_a_link_that_finds_one_row_is_followed_and_its_totals_are_right(learned):
    con, model = learned
    total = _measure(model, "orders")
    payload = _ask(con, model, {"intent": "breakdown", "measures": [total],
                                "group_by": [_slug(model, "regions", "region_name")], "time": {"window": YEAR}})
    want = {(name,): float(v) for name, v in con.execute(
        "SELECT r.region_name, SUM(o.order_total) FROM orders o JOIN stores s USING (store_id) "
        "JOIN regions r USING (region_id) GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


def test_a_link_whose_target_repeats_is_refused_and_says_why(learned):
    con, model = learned
    total = _measure(model, "orders")
    payload = _ask(con, model, {"intent": "breakdown", "measures": [total],
                                "group_by": [_slug(model, "customer_segments", "segment")], "time": {"window": YEAR}})
    stopped = payload["trust"]["stopped"]
    assert payload.get("data") is None, "a total through a repeating link was answered"
    assert stopped == ("Segment cannot split orders: some orders match more than one customer segment, so they "
                       "would be counted more than once."), stopped


def test_a_header_figure_is_not_split_by_its_lines(learned):
    con, model = learned
    total = _measure(model, "orders")
    payload = _ask(con, model, {"intent": "breakdown", "measures": [total],
                                "group_by": [_slug(model, "order_lines", "product")], "time": {"window": YEAR}})
    stopped = payload["trust"]["stopped"]
    assert payload.get("data") is None
    assert stopped == ("Product cannot split orders: each order has several order lines, so it would be counted "
                       "once for each."), stopped


def test_the_lines_own_figure_is_split_by_its_lines(learned):
    con, model = learned
    amount = _measure(model, "order_lines")
    payload = _ask(con, model, {"intent": "breakdown", "measures": [amount],
                                "group_by": [_slug(model, "order_lines", "product")]})
    want = {(p,): float(v) for p, v in con.execute(
        "SELECT product, SUM(line_amount) FROM order_lines GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


# ── a model learned before Learn measured ─────────────────────────────────


def _as_before(model):
    """The model as one learned before this release: no link measured, every learned one said to find one row."""
    old = model.model_copy(deep=True)
    for j in old.joins.values():
        j.cardinality = "one_to_one" if j.cardinality == "one_to_many" else (
            "many_to_one" if j.cardinality == "many_to_many" else j.cardinality)
        j.to_unique, j.max_fanout, j.target_checked = True, 1.0, False
    return old


def test_an_unmeasured_link_is_checked_against_the_data_and_refused_when_it_repeats(learned):
    con, model = learned
    old = _as_before(model)
    total = _measure(old, "orders")
    plan = {"intent": "breakdown", "measures": [total],
            "group_by": [_slug(old, "customer_segments", "segment")], "time": {"window": YEAR}}
    unchecked = _ask(con, old, plan, check=False)
    wrong = sum(_rows(unchecked).values())
    right = float(con.execute("SELECT SUM(order_total) FROM orders").fetchone()[0])
    assert wrong > right * 1.5, "the stand-in for a model from before does not show the double count"
    payload = _ask(con, old, plan)
    assert payload.get("data") is None and "would be counted more than once" in payload["trust"]["stopped"]


def test_the_check_is_asked_once_and_an_unmeasured_link_that_finds_one_row_is_followed(learned):
    import core2.service as service

    con, model = learned
    old = _as_before(model)
    old.version = 987654
    asked: list[str] = []

    class Counting(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if "HAVING" in sql.upper():
                asked.append(sql)
            return super().query(sql, max_rows=max_rows)

    total = _measure(old, "orders")
    plan = {"intent": "breakdown", "measures": [total],
            "group_by": [_slug(old, "regions", "region_name")], "time": {"window": YEAR}}
    answer = json.dumps({"kind": "query", **plan})
    for _ in range(3):
        services = Services(model=old, warehouse=Counting(con), complete=lambda s, t: answer, index=MemberIndex(),
                            today=TODAY)
        payload = answer_question("q", services, Session())
        assert payload.get("data") is not None
    assert len(asked) == 2, asked          # orders -> stores, stores -> regions: once each, then remembered
    assert len(service._LINK_REPEATS) >= 2


def test_a_check_the_warehouse_refuses_leaves_the_link_followed_and_is_logged(learned, caplog):
    con, model = learned
    old = _as_before(model)
    old.version = 55555

    class Refusing(DuckDBWarehouse):
        def query(self, sql, *, max_rows=None):
            if "HAVING" in sql.upper():
                raise RuntimeError("permission denied")
            return super().query(sql, max_rows=max_rows)

    check = _link_check(Services(model=old, warehouse=Refusing(con), complete=lambda s, t: "",
                                 index=MemberIndex(), today=TODAY))
    with caplog.at_level("WARNING", logger="querybot.core2"):
        assert check(_join(old, "orders", "stores")) is False
    assert "could not check" in caplog.text


# ── an admin's condition, and today's setup ───────────────────────────────


def test_a_condition_that_keeps_one_row_per_key_makes_the_link_safe(learned):
    con, model = learned
    fixed = model.model_copy(deep=True)
    j = _join(fixed, "orders", "customer_segments")
    primary = next(c.key for c in fixed.columns.values()
                   if fixed.tables[c.table].name == "customer_segments" and c.name == "is_primary")
    j.conditions = [ColumnFilter(column=primary, op="eq", values=[1])]
    j.cardinality, j.to_unique, j.max_fanout, j.target_checked, j.provenance = (
        "many_to_one", True, 1.0, False, "admin")
    total = _measure(fixed, "orders")
    payload = _ask(con, fixed, {"intent": "breakdown", "measures": [total],
                                "group_by": [_slug(fixed, "customer_segments", "segment")],
                                "time": {"window": YEAR}})
    want = {(s,): float(v) for s, v in con.execute(
        "SELECT c.segment, SUM(o.order_total) FROM orders o JOIN customer_segments c "
        "ON c.customer_id = o.customer_id AND c.is_primary = 1 GROUP BY 1").fetchall()}
    assert _rows(payload) == pytest.approx(want)


def test_an_admin_link_saved_over_the_warning_is_not_followed():
    from core2.model.links import LinkSpec, as_join
    from core2.resolve.paths import repeats

    spec = LinkSpec("t.orders", "t.customers", [("t.orders.customer_id", "t.customers.customer_id")], [], "",
                    "many_to_one", True)

    class Model:
        columns = {"t.orders.customer_id": type("C", (), {"name": "customer_id"})(),
                   "t.customers.customer_id": type("C", (), {"name": "customer_id"})()}
        tables = {"t.customers": type("T", (), {"kind": "dimension"})()}

    assert repeats(as_join(Model, spec, {"match": 1.0, "twice": 12, "most": 3}))
    assert not repeats(as_join(Model, spec, {"match": 1.0, "twice": 0, "most": 1}))


def test_a_question_never_follows_an_admin_link_saved_over_the_warning(learned):
    """An admin's link keeps "many to one" and records what its check found: up to three rows per key."""
    con, model = learned
    saved = model.model_copy(deep=True)
    j = _join(saved, "orders", "customer_segments")
    j.cardinality, j.to_unique, j.max_fanout, j.target_checked, j.provenance, j.trust = (
        "many_to_one", False, 3.0, True, "admin", "admin")
    payload = _ask(con, saved, {"intent": "breakdown", "measures": [_measure(saved, "orders")],
                                "group_by": [_slug(saved, "customer_segments", "segment")],
                                "time": {"window": YEAR}})
    assert payload.get("data") is None and "would be counted more than once" in payload["trust"]["stopped"]


@pytest.mark.parametrize("kind,walked,direction", [
    ("one_to_many", True, ("orders", "stores")),
    ("many_to_one", True, ("orders", "stores")),
    ("many_to_many", False, ("orders", "stores")),
])
def test_a_link_from_todays_setup_is_followed_only_the_way_that_counts_each_row_once(learned, kind, walked,
                                                                                      direction):
    from core2.model.imports import Legacy, _links, _Names, Report
    from core2.resolve.paths import repeats

    _, model = learned
    bare = model.model_copy(deep=True)
    bare.joins = {}
    if kind == "one_to_many":
        rel = {"from_entity": "Store", "to_entity": "Order", "from_column": "store_id", "to_column": "store_id"}
    else:
        rel = {"from_entity": "Order", "to_entity": "Store", "from_column": "store_id", "to_column": "store_id"}
    legacy = Legacy(entities=[{"entity_name": "Order", "table_name": "orders"},
                              {"entity_name": "Store", "table_name": "stores"}],
                    relationships=[{**rel, "relationship_type": kind, "status": "confirmed", "is_active": 1}])
    report = Report()
    _links(bare, legacy, _Names(bare), report)
    (defined,) = [d for d in report.decisions if d.field == "define"]
    j = defined.value
    assert (bare.tables[j["from_table"]].name, bare.tables[j["to_table"]].name) == direction
    assert repeats(type(next(iter(model.joins.values()))).model_validate(j)) is (not walked)
