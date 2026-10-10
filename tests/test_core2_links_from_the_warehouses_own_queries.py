"""The joins a warehouse's own people and tools write are links Learn may use.

A customer's support rep is an employee, and an employee reports to one; neither name says which
table it points at, and their few small numbers fit several tables' keys by chance, so names and
values alone leave them out (evals/core2/benchmark.py: chinook and northwind). The warehouse's query
history holds the analysts' joins: Learn reads the SELECT statements its sign-in may see, keeps only
which columns two tables are compared on, and tests each pair on the data like any other link.

What a query holds besides its joins (a patient's name in a filter) is never kept. A join written once,
a join to a subquery (Learn's own tests), a join of two lists on a shared value, and a join the data
does not bear out are not links. A warehouse that will not show its history leaves the step out.

Invented data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model, learn
from core2.bootstrap.history import _READS, read_query_log
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse, QueryResult, assert_read_only

HISTORY = [
    # An analyst's report, twice with other filters: the rep of each customer, and the manager of each rep.
    "SELECT c.company, e.last_name FROM customer AS c JOIN employee AS e ON c.support_rep_id = e.employee_id "
    "WHERE c.country = 'Patientia Secretname'",
    "select c.company, e.title from main.customer c inner join main.employee e on e.employee_id = c.support_rep_id",
    "SELECT e.last_name, m.last_name AS manager FROM employee e LEFT JOIN employee m ON e.reports_to = m.employee_id",
    "SELECT e.last_name FROM employee e, employee boss WHERE e.reports_to = boss.employee_id AND boss.title = 'GM'",
    # Learn's own test of a candidate: a table joined to the target's keys in a subquery. Never counted.
    "SELECT COUNT(1) FROM customer AS f LEFT JOIN (SELECT DISTINCT media_type_id FROM media_type) AS t "
    "ON f.support_rep_id = t.media_type_id",
    "SELECT COUNT(1) FROM customer AS f LEFT JOIN (SELECT DISTINCT media_type_id FROM media_type) AS t "
    "ON f.support_rep_id = t.media_type_id",
    # Written once: a mistake, or a one-off (the zones happen to number 1 to 4, as the territories do).
    "SELECT * FROM customer c JOIN sales_territory s ON c.zone = s.territory_id",
    # Two lists on a shared value: no key on either side.
    "SELECT * FROM customer c JOIN employee e ON c.country = e.country",
    "SELECT * FROM customer c JOIN employee e ON c.country = e.country",
    # A habit the data does not bear out: invoice totals are not media types.
    "SELECT * FROM invoice i JOIN media_type m ON i.total_units = m.media_type_id",
    "SELECT * FROM invoice i JOIN media_type m ON m.media_type_id = i.total_units",
    "this is not SQL at all (",
]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE employee (employee_id INTEGER PRIMARY KEY, last_name VARCHAR, title VARCHAR, "
                "reports_to INTEGER, country VARCHAR)")
    con.execute("INSERT INTO employee VALUES (1, 'Adams', 'GM', NULL, 'Canada'), (2, 'Edwards', 'Sales Manager', 1, "
                "'Canada'), (3, 'Peacock', 'Agent', 2, 'Canada'), (4, 'Park', 'Agent', 2, 'Canada'), "
                "(5, 'Johnson', 'Agent', 2, 'Canada'), (6, 'Mitchell', 'IT Manager', 1, 'Canada'), "
                "(7, 'King', 'IT Staff', 6, 'Canada'), (8, 'Callahan', 'IT Staff', 6, 'Canada')")
    con.execute("CREATE TABLE media_type (media_type_id INTEGER PRIMARY KEY, media_name VARCHAR)")
    con.execute("INSERT INTO media_type VALUES (1, 'MPEG'), (2, 'Protected AAC'), (3, 'Protected MPEG-4'), "
                "(4, 'Purchased AAC'), (5, 'AAC')")
    con.execute("CREATE TABLE sales_territory (territory_id INTEGER PRIMARY KEY, territory_name VARCHAR)")
    con.execute("INSERT INTO sales_territory VALUES (1, 'North'), (2, 'East'), (3, 'South'), (4, 'West')")
    con.execute("CREATE TABLE customer (customer_id INTEGER PRIMARY KEY, company VARCHAR, country VARCHAR, "
                "support_rep_id INTEGER, zone INTEGER)")
    con.execute("INSERT INTO customer SELECT i, 'Company ' || i, CASE i % 3 WHEN 0 THEN 'Canada' ELSE 'Brazil' END, "
                "3 + i % 3, i % 4 + 1 FROM range(1, 60) t(i)")
    con.execute("CREATE TABLE invoice (invoice_id INTEGER PRIMARY KEY, customer_id INTEGER, invoice_date DATE, "
                "total_units INTEGER, total DECIMAL(10,2))")
    con.execute("INSERT INTO invoice SELECT i, i % 59 + 1, DATE '2025-01-01' + i::INTEGER, i % 30 + 1, "
                "(i % 20) * 1.99 FROM range(1, 413) t(i)")
    return con


def _links(model) -> dict[tuple, object]:
    return {(model.tables[j.from_table].name, tuple(model.columns[c].name for c in j.from_columns),
             model.tables[j.to_table].name): j for j in model.joins.values() if j.trust != "rejected"}


@pytest.fixture(scope="module")
def models():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    without = build_model(warehouse, inventory, client_id="h0", options=BuildOptions(workers=1))
    with_history = build_model(warehouse, from_duckdb(warehouse), client_id="h1",
                               options=BuildOptions(workers=1, query_history=lambda _w: list(HISTORY)))
    return without, with_history


@pytest.mark.parametrize("link", [
    ("customer", ("support_rep_id",), "employee"),        # a support rep is an employee
    ("employee", ("reports_to",), "employee"),            # an employee reports to one
])
def test_the_joins_people_write_are_links(models, link):
    without, with_history = models
    assert link not in _links(without), "names and values alone cannot tell"
    found = _links(with_history).get(link)
    assert found is not None, sorted(_links(with_history))
    assert found.trust == "verified"
    assert any(e.kind == "query_history" and "2 of the warehouse's own queries" in e.detail for e in found.evidence)


@pytest.mark.parametrize("not_a_link", [
    ("customer", ("support_rep_id",), "media_type"),      # only Learn's own tests join it so
    ("customer", ("zone",), "sales_territory"),           # written once
    ("customer", ("country",), "employee"),               # two lists on a shared value
    ("invoice", ("total_units",), "media_type"),          # written twice; the data says no
])
def test_what_history_only_seems_to_say_is_not_a_link(models, not_a_link):
    _, with_history = models
    assert not_a_link not in _links(with_history)


def test_nothing_of_a_query_but_its_joins_is_kept(models):
    _, with_history = models
    assert "Patientia" not in with_history.model_dump_json()


def test_without_a_grant_to_read_the_history_learn_goes_on():
    class Refusing:
        dialect, db_type = "snowflake", "snowflake"

        def __init__(self):
            self.asked = []

        def query(self, sql, *, max_rows=None) -> QueryResult:
            self.asked.append(sql)
            raise RuntimeError("SQL access control error: Insufficient privileges")

    warehouse = Refusing()
    assert read_query_log(warehouse) == []
    assert len(warehouse.asked) == len(_READS["snowflake"])


def test_a_history_query_that_breaks_does_not_stop_learn():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)

    def broken(_w):
        raise RuntimeError("the history view went away")

    found = learn(warehouse, from_duckdb(warehouse), BuildOptions(workers=1, query_history=broken))
    assert found.joins


@pytest.mark.parametrize("dialect", sorted(_READS))
def test_reading_the_history_is_read_only_on_every_warehouse(dialect):
    for sql in _READS[dialect]:
        assert_read_only(sql, dialect)
