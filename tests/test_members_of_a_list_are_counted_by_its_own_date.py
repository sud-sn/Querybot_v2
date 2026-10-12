"""Members of a list counted for a period are counted by the list's own date when the question names it.

"How many new customers signed up in 2025?" and "how many subscriptions were cancelled in 2025?" were asked
back "counted by what they did: Invoice date or Month end date?", and the list's own dates (the customer's
signup, the subscription's end) were never offered: the click counted the customers invoiced in 2025. The
question's words now settle it -- signed up, joined, started, created take the list's start date; cancelled,
churned, ended, left its end date -- and when it is still asked, the list's own dates are among the choices.
"How many customers bought in Q2?" is still counted by what they did.

The synthetic subscriptions warehouse (invented data); figures are checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.planner import settle
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

Y25 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}


@pytest.fixture(scope="module")
def subs():
    built = materialize(domains.build("subscriptions"), "descriptive")
    warehouse = DuckDBWarehouse(built.con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks), client_id="t-own-date",
                        options=BuildOptions(workers=1))
    return built.con, model


def _count(measure):
    return Plan.model_validate({"kind": "query", "intent": "value", "measures": [measure], "time": {"window": Y25}})


@pytest.mark.parametrize("question, measure, date", [
    ("How many new customers signed up in 2025?", "number_of_customers", "signup_timestamp"),
    ("How many customers joined in 2025?", "number_of_customers", "signup_timestamp"),
    ("How many subscriptions were cancelled in 2025?", "number_of_subscriptions", "end_date"),
    ("Combien d'abonnements résiliés en 2025 ?", "number_of_subscriptions", "end_date"),
    ("How many subscriptions started in 2025?", "number_of_subscriptions", "start_date"),
])
def test_the_questions_words_name_the_lists_own_date(subs, question, measure, date):
    assert settle(_count(measure), subs[1], question, None).time.date == date


def test_what_members_did_is_still_counted_by_their_activity(subs):
    assert settle(_count("number_of_customers"), subs[1], "How many customers were invoiced in 2025?",
                  None).time.date is None


def test_customers_who_signed_up_are_counted_by_their_signup(subs):
    con, model = subs
    plan = settle(_count("number_of_customers"), model, "How many new customers signed up in 2025?", None)
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan.model_dump(mode="json")}),
                        index=MemberIndex(), today=dt.date(2026, 6, 14))
    payload = answer_question("How many new customers signed up in 2025?", services, Session())
    want = con.execute("SELECT COUNT(*) FROM customers WHERE signup_ts >= TIMESTAMP '2025-01-01' "
                       "AND signup_ts < TIMESTAMP '2026-01-01'").fetchone()[0]
    assert payload["data"]["rows"][0]["number_of_customers"] == want


def test_when_asked_the_lists_own_dates_are_offered(subs):
    con, model = subs
    plan = _count("number_of_customers")
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan.model_dump(mode="json")}),
                        index=MemberIndex(), today=dt.date(2026, 6, 14))
    payload = answer_question("How many customers in 2025?", services, Session())
    options = (payload.get("clarify") or {}).get("options") or []
    assert "Invoice date" in options and "Signup timestamp" in options     # what they did, or their own signup
