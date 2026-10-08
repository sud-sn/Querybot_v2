"""Members of a period are those its events name, never those a date of their own falls in.

"How many customers bought in Q2?" counted the customer list by a date on the
customer itself (on a test server, when its credit limit last changed): the
customers whose credit changed in Q2, 0 where 60 had bought. Now a count of a
list's members over a period is counted on the events that name them:

* a date of an event the question names ("bought" is the order date): the
  distinct customers on the order lines dated in the period, with a note;
* a date of the member itself the question names ("customers created in
  2025"): the list itself, by that date, as before;
* no date named, one kind of event: that one, with a note; several: asked,
  offering the dates by name, and the reply is followed;
* no period: the whole list, as before.

A measure's own date is used only on its own table (a date lives on its table).
A trend with the same value in every period says so instead of naming a peak.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _count(con, model, plan: dict):
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "count", "measures": ["number_of_customers"],
                                           **plan}), model, Context(today=TODAY))
    rows = DuckDBWarehouse(con).query(compile_query(logical, model, "duckdb").sql).rows
    return logical, rows


def _bought(con, date_key: str, table: str = "order_lines") -> int:
    return con.execute(f"SELECT COUNT(DISTINCT o.customer_id) FROM {table} o JOIN calendar c ON c.date_key = "
                       f"o.{date_key} WHERE c.full_date BETWEEN DATE '2026-01-01' AND DATE '2026-06-30'").fetchone()[0]


def test_customers_who_bought_in_a_period_are_counted_on_the_order_lines(retail):
    con, model = retail
    logical, rows = _count(con, model, {"time": {"date": "order_date", "window": H1}})
    want = _bought(con, "order_date_key")
    created = con.execute("SELECT COUNT(*) FROM customers WHERE created_at BETWEEN DATE '2026-01-01' "
                          "AND DATE '2026-06-30'").fetchone()[0]
    assert rows[0][0] == want and want != created, (want, created)
    assert any("counts the customers on order line rows by order date" in n for n in logical.notes), logical.notes
    assert not any("counted by Created at" in n for n in logical.notes)


def test_a_breakdown_and_a_trend_of_buyers_are_counted_the_same_way(retail):
    con, model = retail
    _, rows = _count(con, model, {"intent": "breakdown", "group_by": ["customer.segment"],
                                  "time": {"date": "order_date", "window": H1}})
    want = dict(con.execute(
        "SELECT cu.segment, COUNT(DISTINCT o.customer_id) FROM order_lines o JOIN calendar c ON c.date_key = "
        "o.order_date_key JOIN customers cu ON cu.customer_id = o.customer_id WHERE c.full_date BETWEEN "
        "DATE '2026-01-01' AND DATE '2026-06-30' GROUP BY 1").fetchall())
    assert {r[0]: r[1] for r in rows} == want
    _, rows = _count(con, model, {"intent": "trend", "time": {"date": "order_date", "grain": "month", "window": H1}})
    assert len(rows) == 6 and sum(1 for r in rows if r[1]) == 6


def test_a_date_of_the_customer_itself_counts_the_list_by_it(retail):
    con, model = retail
    logical, rows = _count(con, model, {"time": {"date": "created_at", "window": H1}})
    created = con.execute("SELECT COUNT(*) FROM customers WHERE created_at >= DATE '2026-01-01' "
                          "AND created_at < DATE '2026-07-01'").fetchone()[0]
    assert rows[0][0] == created
    assert not any("counts the customers on" in n for n in logical.notes)


def test_without_a_period_the_whole_list_is_counted(retail):
    con, model = retail
    _, rows = _count(con, model, {})
    assert rows[0][0] == con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


def test_without_a_named_date_two_kinds_of_event_are_asked_about(retail):
    con, model = retail
    with pytest.raises(ResolveError) as raised:
        _count(con, model, {"time": {"window": H1}})
    assert raised.value.kind == "ambiguous" and raised.value.field == "time.date"
    assert sorted(raised.value.options) == ["Order date", "Return date"]
    assert "Customers in a period are counted by what they did in it" in raised.value.message


def test_without_a_named_date_one_kind_of_event_is_used_and_said(retail):
    con, model = retail
    one = model.model_copy(deep=True)
    for join in one.joins.values():
        if join.from_table.endswith("returns") and join.to_table.endswith("customers"):
            join.trust = "rejected"
    logical, rows = _count(con, one, {"time": {"window": H1}})
    assert rows[0][0] == _bought(con, "order_date_key")
    assert any("by order date" in n for n in logical.notes)


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def test_the_reply_names_the_date_and_is_followed(retail):
    con, model = retail
    plan = {"kind": "query", "intent": "count", "measures": ["number_of_customers"], "time": {"window": H1}}
    warehouse = DuckDBWarehouse(con)
    services = Services(model=model, warehouse=warehouse, complete=Recorded(json.dumps(plan)), index=MemberIndex(),
                        today=TODAY)
    session = Session()
    asked = answer_question("how many customers in the first half of 2026?", services, session)
    assert sorted(asked["clarify"]["options"]) == ["Order date", "Return date"]
    answered = answer_question("return date", services, session)
    assert answered["kpi"]["value"] == _bought(con, "return_date_key", "returns")
    assert "Return date" not in json.dumps(answered.get("clarify") or {})


def test_a_flat_trend_says_so_instead_of_naming_a_peak(retail):
    con, model = retail
    stores = next(m.slug for m in model.measures.values() if m.slug == "number_of_stores")
    plan = {"kind": "query", "intent": "trend", "measures": [stores],
            "time": {"date": "order_date", "grain": "month", "window": H1}}
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=Recorded(json.dumps(plan)),
                        index=MemberIndex(), today=TODAY)
    answer = answer_question("how many stores sold each month this year?", services, Session())
    selling = con.execute("SELECT COUNT(DISTINCT store_id) FROM order_lines").fetchone()[0]
    assert answer["answer"]["headline"].endswith(f"by month: {selling} in every month."), answer["answer"]
    assert "highest" not in answer["answer"]["headline"]
