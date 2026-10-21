"""A rate worked out over things the data does not record is said to be missing, never made up.

"What is the conversion rate on our website?" was answered "Conversion rate: 40.8%": the AI named the share of
orders placed on the web store a conversion rate. A conversion rate is over visits, sessions or leads; a bounce rate
over sessions; a click-through rate over impressions. When no table, field or figure of the data holds any of them,
the answer says so, as it does for a Net Promoter Score.

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
WEB_SHARE = {"intent": "value", "rates": [{"name": "Conversion rate", "measure": "number_of_orders",
                                          "where": [{"field": "store.name", "op": "eq", "values": ["Online"]}]}]}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, question: str, plan: dict) -> dict:
    built, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    return answer_question(question, services, Session())


@pytest.mark.parametrize("question", ["What is the conversion rate on our website?", "website CTR last month",
                                      "Quel est notre taux de conversion ?"])
def test_a_rate_over_visits_the_data_has_none_of_is_said_missing(retail, question):
    got = _ask(retail, question, WEB_SHARE)
    text = str((got.get("answer") or {}).get("headline") or got.get("text") or got.get("message") or "")
    # Said in the reader's language (core2/answer/beyond.py), never with a rate made up.
    assert ("cannot answer" in text or "ne peux pas répondre" in text) and "%" not in text, text
    assert not (got.get("export_rows") or []), got.get("export_rows")


def test_a_share_of_orders_asked_as_a_share_is_answered(retail):
    plan = {**WEB_SHARE, "rates": [{**WEB_SHARE["rates"][0], "name": "Share of orders"}]}
    got = _ask(retail, "What share of our orders were placed online?", plan)
    assert got.get("export_rows"), got.get("answer")
