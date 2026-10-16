"""A comparison of two days the question never names is the year it names.

"Revenue on orders with a promotion vs without one in 2025" was planned as the promotion orders of 31 Dec 2025
against those of 1 Jan 2025: "$106.9K, up +1565.7%". The "vs" was between two kinds of order, not two days; the
question names one period, 2025, and no day. A comparison of two single days that the question names no day of
(no date, no month, no "today") is answered for the one year it names, with no comparison.

Invented retail data; every figure is checked against SQL written by hand.
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
DAYS = {"window": {"kind": "between", "start": "2025-12-31", "end": "2025-12-31"},
        "compare": {"kind": "window", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-01-01"}}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, question: str, plan: dict) -> dict:
    built, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _year_2025(learned, cancelled_out: bool) -> float:
    # The product leaves cancelled lines out where the model says so; the hand total follows its SQL.
    out = "AND o.status_code <> 'C'" if cancelled_out else ""
    return float(learned[0].con.execute(
        "SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar c ON o.order_date_key = c.date_key "
        f"WHERE c.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' {out}").fetchone()[0])


def test_two_days_no_one_named_are_the_year_it_names(retail):
    plan = {"intent": "compare", "measures": ["net_amount"], "time": DAYS}
    got = _ask(retail, "Net sales on web orders vs store orders in 2025", plan)
    rows = got["export_rows"]
    assert len(rows) == 1 and not any("pct_change" in k or "prior" in k for k in rows[0]), rows
    sql = (got.get("trust") or {}).get("sql") or ""
    assert float(rows[0]["net_amount"]) == pytest.approx(_year_2025(retail, "'C'" in sql)), sql


def test_two_days_the_question_names_are_compared(retail):
    plan = {"intent": "compare", "measures": ["net_amount"], "time": DAYS}
    got = _ask(retail, "Net sales on 31 December 2025 vs 1 January 2025", plan)
    assert any("prior" in k or "pct_change" in k for k in got["export_rows"][0]), got["export_rows"]
