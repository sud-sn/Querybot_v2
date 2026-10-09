"""An average or a ratio has no share of a total: its values over the groups do not add up.

"Rank profit centres by gross profit per invoice in 2025" was answered "Manitoba Service leads with
2,870.72 (14% of the total)", and offered "Share of gross profit per invoice by profit centre": eight
per-invoice figures added together are no total, so neither said anything. A share is worked out,
said and offered only for a sum or a count; asked for an average or a ratio, the answer says why
there is none. Invented retail data.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question

YEAR = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}


@pytest.fixture(scope="module")
def retail():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    return learn(domains.build("retail"), "descriptive")


def _ask(retail, plan: dict) -> dict:
    from core2.warehouse.runner import DuckDBWarehouse

    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(),
                        today=dt.date(2026, 6, 15))
    return answer_question("q", services, Session())


def _chips(payload: dict) -> list[str]:
    return [s["question"] for s in payload.get("follow_up_suggestions") or []]


@pytest.mark.parametrize("measure", ["unit_price", "margin_percent"])
def test_a_ranking_of_an_average_says_no_share_and_offers_none(retail, measure):
    payload = _ask(retail, {"intent": "rank", "measures": [measure], "group_by": ["region"], "time": {"window": YEAR}})
    assert payload["data"]["rows"] and "of the total" not in payload["answer"]["headline"], payload["answer"]
    assert not any(c.startswith("Share of") for c in _chips(payload)), _chips(payload)
    assert not any("of the total" in k for k in payload.get("key_insights") or []), payload.get("key_insights")


def test_a_sum_still_says_its_share_and_offers_it(retail):
    payload = _ask(retail, {"intent": "rank", "measures": ["net_amount"], "group_by": ["region"],
                            "time": {"window": YEAR}})
    assert "of the total)" in payload["answer"]["headline"], payload["answer"]
    assert any(c.startswith("Share of") for c in _chips(payload)), _chips(payload)


def test_a_share_asked_of_an_average_is_not_worked_out_and_says_why(retail):
    payload = _ask(retail, {"intent": "share", "measures": ["unit_price"], "group_by": ["region"],
                            "time": {"window": YEAR}})
    chart = payload.get("chart") or {}
    assert chart.get("chart_type") != "pie" and not chart.get("share_key"), chart
    assert not any("share" in h for h in payload["data"]["headers"]), payload["data"]["headers"]
    said = " ".join(payload["trust"].get("date_context") or [])
    assert "does not add up" in said, said


def test_a_share_asked_of_a_sum_is_still_worked_out(retail):
    payload = _ask(retail, {"intent": "share", "measures": ["net_amount"], "group_by": ["region"],
                            "time": {"window": YEAR}})
    assert any("share" in h for h in payload["data"]["headers"]), payload["data"]["headers"]
