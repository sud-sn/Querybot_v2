"""The AI's second opinion on columns named like links that no name or value placed.

On public and synthetic samples, a customer's support rep (an employee's number) and a batch's technician (a
pharmacist's number) were never linked: nothing in the names says which table, and a few small numbers fit several
tables' keys. So Learn asks the AI which of the tables that could hold the values each such column refers to, by
names alone, and keeps a link only where every filled row finds a member. What leaves for the AI is names, never a
value; a reading the data contradicts is dropped; a failed call changes nothing.

Invented data only.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model, learn
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.second_opinion import SYSTEM, second_opinion, unplaced
from core2.resolve.paths import usable
from core2.warehouse.runner import DuckDBWarehouse


def _warehouse() -> DuckDBWarehouse:
    con = duckdb.connect()
    con.execute("CREATE TABLE employee (employee_id INTEGER, last_name VARCHAR, title VARCHAR)")
    con.execute("INSERT INTO employee SELECT i, 'Surname' || i, ['Agent', 'Manager'][i % 2 + 1] FROM range(1, 10) t(i) "
                "WHERE i <> 6")
    con.execute("CREATE TABLE customer (customer_id INTEGER, company VARCHAR, support_rep_id INTEGER, "
                "account_rep_id INTEGER)")
    # Reps 3, 4 and 5 look after every customer; the account reps include a 6, inside the employees' numbers
    # (1..9) but nobody's.
    con.execute("INSERT INTO customer SELECT i, 'Company ' || i, i % 3 + 3, CASE WHEN i % 7 = 0 THEN 6 ELSE i % 4 + 2 END "
                "FROM range(1, 61) t(i)")
    con.execute("CREATE TABLE pharmacist (pharmacist_id INTEGER, pharmacist_name VARCHAR)")
    con.execute("INSERT INTO pharmacist SELECT i, 'Pharmacist ' || i FROM range(1, 19) t(i)")
    con.execute("CREATE TABLE batch (batch_id INTEGER, technician_id INTEGER, status_id INTEGER, units INTEGER, "
                "home_base_cd VARCHAR)")
    con.execute("INSERT INTO batch SELECT i, i % 8 + 11, i % 4 + 1, 10 + i % 30, 'D-' || lpad(CAST(i % 12 + 1 AS VARCHAR), "
                "2, '0') FROM range(1, 301) t(i)")
    # Depots by their code: the only table whose keys could hold a batch's home base.
    con.execute("CREATE TABLE depot (depot_cd VARCHAR, depot_name VARCHAR)")
    con.execute("INSERT INTO depot SELECT 'D-' || lpad(CAST(i AS VARCHAR), 2, '0'), 'Depot ' || i FROM range(1, 21) t(i)")
    # A list's own alternate key, named for it: its own number, never a reading to ask for.
    con.execute("CREATE TABLE category (category_id INTEGER, category_alternate_key INTEGER, category_name VARCHAR)")
    con.execute("INSERT INTO category SELECT i, i, 'Category ' || i FROM range(1, 13) t(i)")
    return DuckDBWarehouse(con)


class _AI:
    """Reads names as a person would; remembers what it was sent."""

    def __init__(self, answer: dict[str, str | None]):
        self.answer, self.sent = answer, []

    def __call__(self, system: str, user: str) -> str:
        self.sent.append((system, user))
        if system != SYSTEM:
            return "{}"           # the table names asked for later are not this test's
        return json.dumps({"links": self.answer})


_READING = {"customer.support_rep_id": "employee", "customer.account_rep_id": "employee",
            "batch.technician_id": "pharmacist", "batch.status_id": None, "batch.home_base_cd": "depot"}


@pytest.fixture(scope="module")
def learned():
    warehouse = _warehouse()
    inventory = from_duckdb(warehouse)
    ai = _AI(_READING)
    findings = learn(warehouse, inventory, BuildOptions(workers=1, query_history=None, labeler=ai))
    links = {(j.from_table.rsplit(".", 1)[-1], j.from_column): j for j in findings.joins if j.trust != "rejected"}
    return links, ai


def test_without_the_ai_names_and_values_leave_them_unlinked():
    warehouse = _warehouse()
    findings = learn(warehouse, from_duckdb(warehouse), BuildOptions(workers=1, query_history=None))
    columns = {(j.from_table.rsplit(".", 1)[-1], j.from_column) for j in findings.joins if j.trust != "rejected"}
    assert ("customer", "support_rep_id") not in columns and ("batch", "technician_id") not in columns


def test_the_ai_reads_a_support_rep_as_an_employee_and_every_row_finds_one(learned):
    links, _ = learned
    rep = links[("customer", "support_rep_id")]
    assert rep.to_table.rsplit(".", 1)[-1] == "employee" and rep.to_column == "employee_id"
    assert rep.trust == "proposed" and rep.match_rate == 1.0
    assert any(e.kind == "ai_opinion" for e in rep.evidence)
    assert (rep.role or "").startswith("Support rep")
    tech = links[("batch", "technician_id")]
    assert tech.to_table.rsplit(".", 1)[-1] == "pharmacist"


def test_a_reading_only_values_that_fit_one_table_confirm_is_used_at_once(learned):
    links, _ = learned
    rep, base = links[("customer", "support_rep_id")], links[("batch", "home_base_cd")]
    # 3, 4 and 5 are numbers in every table here: the AI's reading alone chooses, so it waits for the admin.
    assert any(e.kind == "ambiguous" for e in rep.evidence)
    # Only the depots hold codes like D-07: used at once.
    assert base.to_table.rsplit(".", 1)[-1] == "depot" and not any(e.kind == "ambiguous" for e in base.evidence)


def test_a_reading_the_data_contradicts_is_dropped_and_a_null_is_no_link(learned):
    links, _ = learned
    assert ("customer", "account_rep_id") not in links     # its 6 is nobody
    assert ("batch", "status_id") not in links


def test_only_names_leave_never_a_value(learned):
    _, ai = learned
    asked = next(user for system, user in ai.sent if system == SYSTEM)
    assert "support_rep_id" in asked and "employee" in asked and "pharmacist" in asked
    for value in ("Surname", "Company", "Pharmacist 1", "Agent", "Manager"):
        assert value not in asked


def test_a_failed_call_changes_nothing():
    warehouse = _warehouse()
    inventory = from_duckdb(warehouse)
    findings = learn(warehouse, inventory, BuildOptions(workers=1, query_history=None))

    def broken(system: str, user: str) -> str:
        raise RuntimeError("the AI is down")

    assert second_opinion(warehouse, inventory, findings.profiles, findings.keys, findings.calendars, findings.joins,
                          broken) == []
    asked = unplaced(inventory, findings.profiles, findings.keys, findings.calendars, findings.joins)
    columns = {(k.rsplit(".", 1)[-1], c) for k, c, _ in asked}
    assert columns >= {("customer", "support_rep_id"), ("batch", "technician_id")}
    assert ("category", "category_alternate_key") not in columns


def test_the_admin_is_asked_to_confirm_what_the_ai_read():
    warehouse = _warehouse()
    model = build_model(warehouse, from_duckdb(warehouse),
                        options=BuildOptions(workers=1, query_history=None, labeler=_AI(_READING)))
    asked = [r for r in model.review if r.key.startswith("join:") and "support_rep_id" in r.key]
    assert len(asked) == 1 and "AI" in asked[0].question and "until confirmed" in asked[0].choice_made
    rep = next(j for j in model.joins.values() if [model.columns[c].name for c in j.from_columns] == ["support_rep_id"])
    assert not usable(rep)
    base = next(j for j in model.joins.values() if [model.columns[c].name for c in j.from_columns] == ["home_base_cd"])
    assert usable(base)
