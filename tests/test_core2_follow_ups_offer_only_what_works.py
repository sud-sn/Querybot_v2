"""Follow-ups: the chips under an answer, a reply to a question back, "show it as a pie".

* The chips are concrete next questions that the new core answers: a "why" only
  for a bounded period, a forecast only for a series at a grain that has one,
  never a comparison of a series (issue G3), and the member leading a ranking by
  name. Each is written so the planner reads it like a reader's own question.
* A reply to "which one?" picks an option by name, number or an unambiguous start
  (issue B2): ``choose`` decides, and a reply about something else picks nothing.
* "As a pie", "just the table" change how the same answer is shown; a pie of a
  time series is refused with a note, never drawn.
* A filter can name the link it means, as a grouping can ("orders shipped to
  Client 07"); without one, a filter over two links is asked about.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.suggestions import follow_ups
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question, choose
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from tests.test_core2_asks_which_role_and_follows_it import _orders

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


class _AI:
    def __init__(self, *plans: dict):
        self.plans = [json.dumps(p) for p in plans]

    def __call__(self, stable: str, tail: str) -> str:
        return self.plans.pop(0)


def _ask(con, model, plan: dict, question: str = "q") -> dict:
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=_AI(plan), index=MemberIndex(),
                        today=TODAY)
    return answer_question(question, services, Session())


def _chips(payload: dict) -> list[str]:
    return [c["question"] for c in payload["follow_up_suggestions"]]


def test_a_series_offers_why_for_its_last_complete_period_and_a_forecast_never_a_comparison(retail):
    con, model = retail
    payload = _ask(con, model, {"kind": "query", "intent": "trend", "measures": ["net_amount"],
                                "time": {"grain": "month", "window": {"kind": "this", "unit": "year"}}})
    chips = _chips(payload)
    assert payload["data"]["rows"][-1][payload["data"]["headers"][0]] == "2026-06-01"   # June is on screen...
    assert chips[0] == "Why did net amount change in May 2026?"      # ...but partial: not offered as a period
    assert "Forecast net amount for the next 3 months" in chips
    assert not any("compare" in c.lower() or "previous period" in c.lower() for c in chips)


def test_a_ranking_offers_the_leading_member_by_name(retail):
    con, model = retail
    payload = _ask(con, model, {"kind": "query", "intent": "rank", "measures": ["net_amount"],
                                "group_by": ["customer"], "sort": [{"by": "net_amount", "desc": True}], "limit": 5,
                                "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}})
    leader = payload["data"]["rows"][0][payload["data"]["headers"][0]]
    chips = _chips(payload)
    assert chips[0] == f"Monthly net amount for {leader}"
    assert "Why did net amount change in Q1 2026?" in chips


def test_a_total_over_all_time_offers_no_why(retail):
    con, model = retail
    payload = _ask(con, model, {"kind": "query", "intent": "value", "measures": ["net_amount"]})
    assert not any(c.startswith("Why") for c in _chips(payload))
    assert "Net amount by month for the last 12 months" in _chips(payload)


@pytest.mark.parametrize("reply, picked", [
    ("Ship to customer", "Ship to customer"), ("ship-to customer", "Ship to customer"),
    ("the second one", "Ship to customer"), ("2", "Ship to customer"), ("first", "Bill to customer"),
    ("the last", "Ship to customer"), ("ship", "Ship to customer"), ("I mean bill to customer", "Bill to customer"),
    ("what about returns", None), ("customer", None), ("3", None), ("", None),
])
def test_a_reply_picks_an_option_by_name_number_or_unambiguous_start(reply, picked):
    assert choose(reply, ["Bill to customer", "Ship to customer"]) == picked


def test_a_breakdown_can_be_shown_as_a_pie_or_a_table(retail):
    con, model = retail
    plan = {"kind": "query", "intent": "breakdown", "measures": ["net_amount"], "group_by": ["region"],
            "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}}
    assert _ask(con, model, {**plan, "chart": "pie"})["chart"]["chart_type"] == "pie"
    assert _ask(con, model, {**plan, "chart": "table"})["chart"] is None
    assert _ask(con, model, plan)["chart"]["chart_type"] == "bar"


def test_a_pie_of_a_time_series_is_refused_with_a_note(retail):
    con, model = retail
    payload = _ask(con, model, {"kind": "query", "intent": "trend", "measures": ["net_amount"], "chart": "pie",
                                "time": {"grain": "month", "window": {"kind": "last", "unit": "month", "n": 6}}})
    assert payload["chart"]["chart_type"] == "line"
    assert any("A pie chart does not fit" in n for n in payload["trust"]["date_context"])


def test_a_filter_names_the_link_it_means_or_is_asked_about():
    warehouse, model = _orders()
    measure = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(
        "order_amount"))
    customer = next(e.slug for e in model.entities.values() if e.table.endswith("customers"))
    roles = sorted(j.role for j in model.joins.values() if j.to_table.endswith("customers") and j.role)
    plan = {"kind": "query", "intent": "value", "measures": [measure],
            "filters": [{"field": customer, "op": "eq", "values": ["Client 07"]}]}
    with pytest.raises(ResolveError) as asked:
        resolve(Plan.model_validate(plan), model, Context(today=TODAY))
    assert asked.value.kind == "ambiguous" and asked.value.field == customer
    logical = resolve(Plan.model_validate({**plan, "via": {customer: roles[1]}}), model, Context(today=TODAY))
    got = warehouse.query(compile_query(logical, model, "duckdb").sql).rows[0][0]
    want = warehouse.query("SELECT SUM(o.order_amount) FROM orders o JOIN customers c "
                           "ON c.customer_id = o.ship_to_customer_id WHERE c.customer_name = 'Client 07'").rows[0][0]
    assert got == pytest.approx(want)


def test_the_chips_are_what_the_service_sends(retail):
    con, model = retail
    plan = Plan.model_validate({"kind": "query", "intent": "trend", "measures": ["net_amount"],
                                "time": {"grain": "quarter"}})
    logical = resolve(plan, model, Context(today=TODAY))
    payload = _ask(con, model, plan.model_dump(mode="json", exclude_defaults=True))
    assert payload["follow_up_suggestions"] == follow_ups(plan, logical, payload, model, None)
    assert all(set(c) == {"label", "question"} for c in payload["follow_up_suggestions"])
