"""Two quarters compared by name are those quarters.

"Why did Ontario revenue drop in Q3 2026 compared with Q3 2025?" was once planned as August to October of each year
(the same question, asked again, was planned as July to September). A quarter named with its year and no "fiscal"
is the calendar's; with two named, each side of the comparison is put right.

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
OFF = {"window": {"kind": "between", "start": "2025-08-01", "end": "2025-10-31"},
       "compare": {"kind": "window", "window": {"kind": "between", "start": "2024-08-01", "end": "2024-10-31"}}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _total(learned, start: str, end: str, cancelled_out: bool) -> float:
    out = "AND o.status_code <> 'C'" if cancelled_out else ""
    return float(learned[0].con.execute(
        "SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar c ON o.order_date_key = c.date_key "
        f"WHERE c.full_date BETWEEN DATE '{start}' AND DATE '{end}' {out}").fetchone()[0])


def test_two_named_quarters_are_compared_as_named(retail):
    built, model = retail
    plan = {"kind": "query", "intent": "compare", "measures": ["net_amount"], "time": OFF}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    got = answer_question("Net sales in Q3 2025 compared with Q3 2024", services, Session())
    row = got["export_rows"][0]
    sql = (got.get("trust") or {}).get("sql") or ""
    out = "'C'" in sql
    assert float(row["net_amount"]) == pytest.approx(_total(retail, "2025-07-01", "2025-09-30", out)), sql
    assert float(row["net_amount_prior"]) == pytest.approx(_total(retail, "2024-07-01", "2024-09-30", out)), sql


def test_fiscal_quarters_stay_the_planners(retail):
    built, model = retail
    plan = {"kind": "query", "intent": "compare", "measures": ["net_amount"], "time": OFF}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    got = answer_question("Net sales in fiscal Q3 2025 compared with fiscal Q3 2024", services, Session())
    sql = (got.get("trust") or {}).get("sql") or ""
    assert float(got["export_rows"][0]["net_amount"]) == pytest.approx(
        _total(retail, "2025-08-01", "2025-10-31", "'C'" in sql)), sql


@pytest.mark.parametrize("question", ["q3 25 vs q3 24 rev", "Q3'25 vs Q3'24 net sales"])
def test_quarters_with_two_digit_years_are_the_quarters_named(retail, question):
    built, model = retail
    plan = {"kind": "query", "intent": "compare", "measures": ["net_amount"], "time": OFF}
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: json.dumps(plan),
                        index=MemberIndex(), today=TODAY)
    got = answer_question(question, services, Session())
    row = got["export_rows"][0]
    out = "'C'" in ((got.get("trust") or {}).get("sql") or "")
    assert float(row["net_amount"]) == pytest.approx(_total(retail, "2025-07-01", "2025-09-30", out))
    assert float(row["net_amount_prior"]) == pytest.approx(_total(retail, "2024-07-01", "2024-09-30", out))
