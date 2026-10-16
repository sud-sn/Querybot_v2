""""By month" is a series of months, not the months of the year added up across years.

"Net amount by month for the Ontario region" (a tile of a dashboard asked for in the chat) was planned by month of
the year: January of every year together, December of every year together -- twelve bars that looked like one year
and were four. A plain "by month" (monthly, per month, month by month) is a month after month series; the month of
the year is kept when the question says so ("by month of the year", "seasonality", "which month is busiest").

Invented retail data only.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
BY_NAME = {"intent": "trend", "measures": ["net_amount"], "group_by": ["time:month_of_year"]}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _rows(learned, question: str, plan: dict) -> list[dict]:
    built, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("export_rows"), payload.get("answer")
    return payload["export_rows"]


@pytest.mark.parametrize("question", ["Net sales by month", "monthly net sales", "Net sales month by month"])
def test_by_month_is_month_after_month(retail, question):
    rows = _rows(retail, question, BY_NAME)
    months = {str(r.get("period"))[:7] for r in rows}
    assert len(rows) > 12 and all(len(m) == 7 and m[4] == "-" for m in months), rows[:3]


@pytest.mark.parametrize("question", ["Net sales by month of the year", "Which month of the year sells the most?",
                                      "seasonality of net sales by month"])
def test_the_month_of_the_year_asked_for_stays(retail, question):
    rows = _rows(retail, question, BY_NAME)
    assert len(rows) <= 12, rows
