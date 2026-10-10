"""An answer names every condition that narrowed what it counts.

On a test server, "average delay from requested to confirmed delivery in 2025"
was answered "1.0 days": right for the Wholesale customers the conversation had
been about, read as everyone's (whose average is 0.006 days). "5,356 invoice
lines in 2025" was right for those invoiced more than 10 days after ordering,
and read as all of them. The conditions were in the notes only.

Now the sentence, the value card, the workspace KPI and the next questions
offered all carry them, worded as a reader says them:

* a member: "for segment Retail", "excluding segment Online", "for customer X";
* days between two dates: "where days from order to ship is above 3 days";
* a condition on a total: "with net amount above $1,000";
* another date: "with ship date in Mar 2026"; the date counted by, when the
  question named one that is not the default: "by ship date";
* members counted by what they did: "Number of customers with an order line";
* a stock read at its last snapshot: "at the last snapshot";
* a grouping reached through a named link: "Customer name (Ship to customer)";
* "why" and forecast answers too.

Counts are shown whole (5,356, never 5,356.00) and days to a tenth.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, resolve
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


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def _ask(retail, plan: dict, question: str = "q") -> dict:
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=Recorded(json.dumps({"kind": "query", **plan})), index=MemberIndex(), today=TODAY)
    # The reader quotes the members the plan narrows to: a name narrows an answer only in quotes.
    named = " ".join(f'"{v}"' for f in plan.get("filters", []) for v in f["values"] if isinstance(v, str))
    return answer_question(f"{question} {named}".strip() if question == "q" else question, services, Session())


@pytest.mark.parametrize("plan, said", [
    ({"intent": "value", "measures": ["net_amount"], "time": {"window": H1},
      "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]},
     "Net amount from Jan 2026 to Jun 2026, for segment Retail: $"),
    ({"intent": "value", "measures": ["net_amount"], "time": {"window": H1},
      "filters": [{"field": "customer.segment", "op": "not_in", "values": ["Online", "Wholesale"]}]},
     "Net amount from Jan 2026 to Jun 2026, excluding segment Online and Wholesale: $"),
    ({"intent": "count", "measures": ["number_of_order_lines"], "time": {"window": H1},
      "durations": [{"name": "Days from order to ship", "start": "order_date", "end": "ship_date", "measure": False}],
      "filters": [{"field": "Days from order to ship", "op": "gt", "values": [3]}]},
     "Number of order lines from Jan 2026 to Jun 2026, where days from order to ship is above 3 days: "),
    ({"intent": "rank", "measures": ["net_amount"], "group_by": ["store"], "time": {"window": H1},
      "filters": [{"field": "net_amount", "op": "gt", "values": [1000]}]},
     "Net amount from Jan 2026 to Jun 2026, with net amount above $1,000: "),
    ({"intent": "value", "measures": ["net_amount"], "time": {"window": H1},
      "filters": [{"field": "ship_date", "op": "between", "values": ["2026-03-01", "2026-03-31"]}]},
     "Net amount from Jan 2026 to Jun 2026, with ship date in March 2026: $"),
    ({"intent": "value", "measures": ["net_amount"], "time": {"date": "ship_date", "window": H1}},
     "Net amount from Jan 2026 to Jun 2026 by ship date: $"),
    ({"intent": "count", "measures": ["number_of_customers"], "time": {"date": "order_date", "window": H1}},
     "Number of customers with an order line from Jan 2026 to Jun 2026: "),
])
def test_the_sentence_names_what_narrowed_the_answer(retail, plan, said):
    answer = _ask(retail, plan)
    assert answer["answer"]["headline"].startswith(said), answer["answer"]["headline"]


def test_the_value_card_the_kpi_and_the_next_questions_keep_the_condition(retail):
    answer = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": H1},
                           "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]})
    assert answer["answer"]["comparison"] == "from Jan 2026 to Jun 2026, for segment Retail"
    assert answer["kpi"]["note"] == "from Jan 2026 to Jun 2026, for segment Retail"
    chips = [c["question"] for c in answer["follow_up_suggestions"]]
    assert chips and all('for segment "Retail"' in c for c in chips), chips   # quoted: a click keeps it


def test_a_count_is_shown_whole_and_days_to_a_tenth(retail):
    answer = _ask(retail, {"intent": "value", "measures": ["number_of_order_lines"], "time": {"window": H1}})
    assert answer["kpi"]["display_format"] == {"fraction_digits": 0}
    days = _ask(retail, {"intent": "breakdown", "group_by": ["store"], "time": {"window": H1},
                         "durations": [{"name": "Days to ship", "start": "order_date", "end": "ship_date"}]})
    assert days["data"]["display_formats"]["days_to_ship"] == {"fraction_digits": 1}


def test_why_and_forecast_name_the_condition(retail):
    member = [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]
    why = _ask(retail, {"intent": "drivers", "measures": ["net_amount"], "filters": member,
                        "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}})
    assert why["answer"]["headline"].startswith("Net amount for segment Retail "), why["answer"]["headline"]
    forecast = _ask(retail, {"intent": "forecast", "measures": ["net_amount"], "filters": member,
                             "time": {"grain": "month"}, "forecast": {"periods": 3}})
    assert forecast["answer"]["headline"].startswith("Net amount for segment Retail was "), forecast["answer"]
    assert all("for segment Retail" in c["question"] for c in forecast["follow_up_suggestions"])


def test_a_stock_says_it_is_its_last_snapshot():
    built, model = learn(domains.build("inventory"), "descriptive")
    snapshot = next(m for m in model.measures.values() if m.additivity == "semi_additive")
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": [snapshot.slug]}),
                      model, Context(today=TODAY))
    assert [c.kind for c in logical.conditions] == ["snapshot"]
    series = resolve(Plan.model_validate({"kind": "query", "intent": "trend", "measures": [snapshot.slug],
                                          "time": {"grain": "month"}}), model, Context(today=TODAY))
    assert not series.conditions, "a series of snapshots is read period by period"


def test_a_grouping_through_a_named_link_says_which():
    from tests.test_core2_asks_which_role_and_follows_it import _orders

    warehouse, model = _orders()
    measure = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(
        "order_amount"))
    customer = next(e.slug for e in model.entities.values() if e.table.endswith("customers"))
    role = sorted(j.role for j in model.joins.values() if j.to_table.endswith("customers") and j.role)[1]
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [measure],
                                           "group_by": [customer], "via": {customer: role}}),
                      model, Context(today=TODAY))
    assert logical.groups[0].label.endswith(f"({role})"), logical.groups[0].label
