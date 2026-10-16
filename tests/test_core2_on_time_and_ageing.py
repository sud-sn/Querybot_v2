"""Rows against the date they were due by: the share on time, and the rows not done yet by days past due, equal to
SQL written by hand.

Asked of the invented purchasing warehouse through the whole answer path (a scripted plan, as the planner writes
it). A receipt is on time when received on or before its order line's promised date (plus any grace days): the
promised date is on the line each receipt points at. An order line is open while no receipt points at it; open
lines are put in ranges of days past their promised date on the day they are aged on (today, or the end of the
window asked for). Each answer says how it was counted.

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
import sqlglot

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
# A line whose supplier the supplier list does not hold is counted under "Unknown", as every answer does.
LATE = """SELECT r.receipt_id, COALESCE(s.supplier_name, 'Unknown') AS supplier_name,
                 date_trunc('month', r.received_date)::DATE AS m, date_diff('day', l.promised_date, r.received_date) AS late
          FROM receipts r JOIN purchase_order_lines l ON r.po_line_id = l.po_line_id
          LEFT JOIN suppliers s ON l.supplier_id = s.supplier_id
          WHERE r.received_date >= DATE '2025-01-01' AND r.received_date < DATE '2026-01-01'"""
OPEN = """SELECT l.*, date_diff('day', l.promised_date, DATE '{day}') AS past FROM purchase_order_lines l
          WHERE NOT EXISTS (SELECT 1 FROM receipts r WHERE r.po_line_id = l.po_line_id)"""
BANDS = """CASE WHEN past <= 0 THEN 'Not yet due' WHEN past <= 30 THEN '1-30 days' WHEN past <= 60 THEN '31-60 days'
                WHEN past <= 90 THEN '61-90 days' ELSE 'Over 90 days' END"""


@pytest.fixture(scope="module")
def purchasing():
    return learn(domains.build("purchasing"), "descriptive")


def _ask(learned, plan: dict, question: str = "q") -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _sql(learned, sql: str) -> list[tuple]:
    return learned[0].con.execute(sql).fetchall()


def test_the_share_of_receipts_on_time_in_2025(purchasing):
    (share, counted, late), = _sql(purchasing, f"SELECT AVG(CASE WHEN late <= 0 THEN 100.0 ELSE 0 END), COUNT(*), "
                                                f"COUNT(*) FILTER (WHERE late > 0) FROM ({LATE})")
    got = _ask(purchasing, {"on_time": {"done": "received_date", "due": "promised_date"}, "time": {"window": Y2025}})
    assert float(got["export_rows"][0]["on_time"]) == pytest.approx(float(share))
    assert f"Of {counted:,}, {late:,} were late." in got["definition"]
    assert "on time when its received date is on or before its promised date" in got["definition"]


def test_the_share_late_when_the_question_asks_how_many_were_late(purchasing):
    """"What share of order lines had a confirmed delivery date later than the requested one?" was answered with the
    share on time (the profitability stress test): the figure asked for is the share late, whatever the AI wrote."""
    (share, counted, late), = _sql(purchasing, f"SELECT AVG(CASE WHEN late > 0 THEN 100.0 ELSE 0 END), COUNT(*), "
                                                f"COUNT(*) FILTER (WHERE late > 0) FROM ({LATE})")
    on_time = {"on_time": {"done": "received_date", "due": "promised_date"}, "time": {"window": Y2025}}
    got = _ask(purchasing, on_time, "What share of 2025 receipts came in later than the promised date?")
    assert float(got["export_rows"][0]["late"]) == pytest.approx(float(share))
    assert got["answer"]["headline"].startswith("Late in 2025:"), got["answer"]["headline"]
    assert f"Of {counted:,}, {late:,} were late." in got["definition"]
    assert "late when its received date is after its promised date" in got["definition"]
    # Said the other way round, the AI's late share is put back to the share on time.
    back = _ask(purchasing, {**on_time, "on_time": {**on_time["on_time"], "late": True}},
                "What share of 2025 receipts were on time?")
    assert float(back["export_rows"][0]["on_time"]) == pytest.approx(100.0 - float(share))


def test_the_most_late_supplier_comes_first(purchasing):
    expected = dict(_sql(purchasing, f"SELECT supplier_name, AVG(CASE WHEN late > 0 THEN 100.0 ELSE 0 END) "
                                     f"FROM ({LATE}) GROUP BY 1"))
    got = _ask(purchasing, {"on_time": {"done": "received_date", "due": "promised_date"},
                            "group_by": ["supplier.name"], "time": {"window": Y2025}},
               "Which suppliers delivered late most often in 2025?")
    shown = [(r["supplier_name"], float(r["late"])) for r in got["export_rows"]]
    assert dict(shown) == pytest.approx({k: float(v) for k, v in expected.items()})
    assert [v for _, v in shown] == sorted((v for _, v in shown), reverse=True)


def test_on_time_by_supplier_and_by_month_with_three_days_grace(purchasing):
    expected = dict(_sql(purchasing, f"SELECT supplier_name, AVG(CASE WHEN late <= 3 THEN 100.0 ELSE 0 END) "
                                     f"FROM ({LATE}) GROUP BY 1"))
    got = _ask(purchasing, {"on_time": {"done": "received_date", "due": "promised_date", "grace_days": 3},
                            "group_by": ["supplier.name"], "time": {"window": Y2025}})
    assert {r["supplier_name"]: float(r["on_time"]) for r in got["export_rows"]} == pytest.approx(
        {k: float(v) for k, v in expected.items()})
    by_month = dict(_sql(purchasing, f"SELECT m, AVG(CASE WHEN late <= 0 THEN 100.0 ELSE 0 END) FROM ({LATE}) "
                                     "GROUP BY 1"))
    series = _ask(purchasing, {"on_time": {"done": "received_date", "due": "promised_date"},
                               "time": {"window": Y2025, "grain": "month"}})
    assert {str(r["period"])[:10]: float(r["on_time"]) for r in series["export_rows"]} == pytest.approx(
        {str(k): float(v) for k, v in by_month.items()})


def test_open_lines_and_their_amount_by_days_past_due_today(purchasing):
    expected = {band: (n, float(amount)) for band, n, amount in _sql(
        purchasing, f"SELECT {BANDS}, COUNT(*), SUM(line_amount) FROM ({OPEN.format(day=TODAY)}) GROUP BY 1")}
    got = _ask(purchasing, {"measures": ["line_amount"], "on_time": {"done": "received_date", "due": "promised_date",
                                                                     "ageing": True}})
    shown = {r["days_past_due"]: (r["open"], float(r["line_amount"])) for r in got["export_rows"]}
    assert shown == pytest.approx(expected)
    assert [r["days_past_due"] for r in got["export_rows"]] == [b for b in ("Not yet due", "1-30 days", "31-60 days",
                                                                           "61-90 days", "Over 90 days")
                                                              if b in expected]
    assert "with no row with a received date pointing at it yet" in got["definition"]
    assert "15 June 2026" in got["definition"]


def test_ageing_on_the_last_day_of_the_window_asked_for(purchasing):
    day = dt.date(2025, 12, 31)
    expected = dict(_sql(purchasing, f"SELECT {BANDS}, COUNT(*) FROM ({OPEN.format(day=day)}) GROUP BY 1"))
    got = _ask(purchasing, {"on_time": {"done": "received_date", "due": "promised_date", "ageing": True},
                            "time": {"window": {"kind": "until", "end": "2025-12-31"}}})
    assert {r["days_past_due"]: r["open"] for r in got["export_rows"]} == expected


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
@pytest.mark.parametrize("ageing", [False, True])
def test_the_sql_reads_on_every_warehouse(purchasing, dialect, ageing):
    built, model = purchasing
    plan = Plan.model_validate({"kind": "query", "on_time": {"done": "received_date", "due": "promised_date",
                                                             "ageing": ageing},
                                "group_by": ["supplier.name"], "time": {"window": Y2025} if not ageing else {}})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


def test_how_many_were_late_is_the_number_late_with_the_share_beside_it(purchasing):
    """ "How many orders were delivered late in 2025?" was answered "Late in 2025: 7.7%": the share, where the reader
    asked for a number. Asked with a count (the planner puts the rows' count beside the analysis), the answer leads
    with how many were late, by supplier too, and keeps the share beside it."""
    (counted, late, share), = _sql(purchasing, f"SELECT COUNT(*), COUNT(*) FILTER (WHERE late > 0), "
                                                f"AVG(CASE WHEN late > 0 THEN 100.0 ELSE 0 END) FROM ({LATE})")
    plan = {"measures": ["number_of_receipts"], "on_time": {"done": "received_date", "due": "promised_date",
                                                            "late": True}, "time": {"window": Y2025}}
    got = _ask(purchasing, plan, "How many receipts came in late in 2025?")
    row = got["export_rows"][0]
    figures = [k for k in row if k not in ("supplier_name",)]
    assert row[figures[0]] == late, row
    assert any(float(row[k]) == pytest.approx(float(share)) for k in figures[1:]), row
    assert got["answer"]["headline"].replace(",", "").find(str(late)) >= 0, got["answer"]["headline"]
    by_supplier = dict(_sql(purchasing, f"SELECT supplier_name, COUNT(*) FILTER (WHERE late > 0) FROM ({LATE}) GROUP BY 1"))
    got = _ask(purchasing, {**plan, "group_by": ["supplier.name"]}, "How many receipts came in late by supplier in 2025?")
    first = next(k for k in got["export_rows"][0] if k != "supplier_name")
    assert {r["supplier_name"]: r[first] for r in got["export_rows"]} == by_supplier
    # The share alone, as before, when no count is asked for.
    share_only = _ask(purchasing, {"on_time": plan["on_time"], "time": {"window": Y2025}},
                      "What share of 2025 receipts came in late?")
    assert list(share_only["export_rows"][0]) == ["late"]
