"""A question about a period after the data ends says the data ends before it, never "no value".

Found by the sector stress test on the TPC-DS retail benchmark (October 2026): "What will our total sales be in
2004?" was answered "Extended Sales Price in 2004: no value." -- as if the sales of 2004 had been counted and
found empty. The data ends in January 2003: the answer says so, and offers a period it covers or a forecast.

Invented rows only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from tests.test_a_new_core_answer_can_be_pinned import retail  # noqa: F401


def _ask(retail, plan: dict, today=dt.date(2026, 6, 15)):  # noqa: F811
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, u: json.dumps(plan),
                        index=MemberIndex(), today=today)
    return answer_question("What will our net amount be in 2030?", services, Session())


def _text(payload) -> str:
    return str((payload.get("answer") or {}).get("headline") or payload.get("text") or "")


def test_a_year_after_the_data_says_the_data_ends_before_it(retail):  # noqa: F811
    payload = _ask(retail, {"kind": "query", "measures": ["net_amount"],
                            "time": {"window": {"kind": "between", "start": "2030-01-01", "end": "2030-12-31"}}})
    assert "The data ends on" in _text(payload) and "no value" not in _text(payload), _text(payload)
    assert "Ask about a period it covers" in payload["text"]


def test_the_conditions_asked_under_are_still_said(retail):  # noqa: F811
    payload = _ask(retail, {"kind": "query", "measures": ["net_amount"],
                            "time": {"window": {"kind": "between", "start": "2030-01-01", "end": "2030-12-31"}},
                            "filters": [{"field": "customer.segment", "op": "ne", "values": ["Mail order"]}]})
    assert "excluding segment Mail order" in _text(payload) and "The data ends on" in _text(payload), _text(payload)


def test_a_year_the_data_covers_is_still_answered(retail):  # noqa: F811
    payload = _ask(retail, {"kind": "query", "measures": ["net_amount"],
                            "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}})
    assert (payload.get("data") or {}).get("rows") and "The data ends on" not in _text(payload)
