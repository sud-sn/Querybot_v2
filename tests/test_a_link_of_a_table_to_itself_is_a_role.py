"""A table's link to itself -- an employee's manager, an account's parent -- is a role a question can name.

Learn found "Manager" (employees.manager_id -> employees) and "Parent account", but no question could follow
them: a path from a table to itself was always the table itself, so "how many people report to each manager",
"staff whose manager is at level L5" and "expenses by parent account" were refused, and the planner made up
filters instead (managers as job level L5). The catalog now lists the role, a grouping or a filter that names
it follows the link, and "by manager" is no longer read as each employee counting itself.

The synthetic HR and finance warehouses (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.catalog import catalog_text
from core2.plan.values import MemberIndex
from core2.resolve import paths as P
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn


@pytest.fixture(scope="module")
def hr():
    return learn(domains.build("hr"), "descriptive")[:2]


def _ask(learned, plan):
    built, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 15))
    return answer_question("q", services, Session())


def test_the_catalog_offers_the_role(hr):
    line = next(l for l in catalog_text(hr[1]).splitlines() if l.startswith("- employee |"))
    assert "roles (name one in via): Manager" in line


def test_staff_whose_manager_is_at_a_level(hr):
    built, _ = hr
    payload = _ask(hr, {"intent": "value", "measures": ["number_of_employees"],
                        "filters": [{"field": "employee.job_level", "op": "eq", "values": ["L5"]}],
                        "via": {"employee.job_level": "role:Manager"}})
    want = built.con.execute("SELECT COUNT(*) FROM employees e JOIN employees m ON e.manager_id = m.employee_id "
                             "WHERE m.job_level = 'L5'").fetchone()[0]
    assert payload["data"]["rows"][0]["number_of_employees"] == want
    assert "role:" not in json.dumps(payload.get("conditions") or payload["answer"])


def test_people_reporting_to_each_manager(hr):
    built, _ = hr
    payload = _ask(hr, {"intent": "rank", "measures": ["number_of_employees"], "group_by": ["employee.number"],
                        "via": {"employee.number": "role:Manager"}, "limit": 3,
                        "sort": [{"by": "number_of_employees", "desc": True}]})
    want = built.con.execute("SELECT m.employee_number, COUNT(*) n FROM employees e JOIN employees m "
                             "ON e.manager_id = m.employee_id GROUP BY 1 ORDER BY n DESC, 1 LIMIT 3").fetchall()
    assert [(r["employee_number"], r["number_of_employees"]) for r in payload["data"]["rows"]] == want


def test_the_rows_whose_role_names_no_one_are_named_by_the_role(hr):
    # Found on an invented network-equipment business with a real model: "bookings by each rep's manager" put the
    # sales of the reps who have no manager on a line "No sales rep recorded", though every sale has its rep: it is
    # the manager that is missing. The pay lines of the staff with no manager are "No manager recorded".
    built, _ = hr
    payload = _ask(hr, {"intent": "breakdown", "measures": ["overtime_hours"], "group_by": ["employee.number"],
                        "via": {"employee.number": "role:Manager"}})
    found = {r["employee_number"]: r["overtime_hours"] for r in payload["data"]["rows"]}
    (unmanaged,), = built.con.execute("SELECT SUM(p.overtime_hours) FROM payroll p JOIN employees e "
                                      "USING (employee_id) WHERE e.manager_id IS NULL").fetchall()
    assert unmanaged and "No employee recorded" not in found, sorted(found)[-3:]
    assert found.get("No manager recorded") == unmanaged, sorted(found)[-3:]

def test_a_parent_account_is_reached_from_the_ledger():
    model = learn(domains.build("finance"), "descriptive")[1]
    accounts = next(t.key for t in model.tables.values() if t.key.endswith(".accounts"))
    ledger = [t.key for t in model.tables.values() if t.kind == "fact"
              and P.all_paths(model, t.key, accounts) and t.key != accounts]
    assert ledger and "Parent account" in P.role_names(model, ledger[0], accounts)
    assert P.with_role(model, ledger[0], accounts, "parent account")
