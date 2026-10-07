"""An answer says what in the data is worth checking, for what it touched only.

* A month far from every other (a quantity typed a thousand times too large) is
  named when the answer covers it, and only then.
* An average of a measure that is the same on every row says so: it tells nothing
  apart. A sum of one still counts rows, and says nothing.
* A count of a list's members says how many are ever used.
* A status value on most rows is the normal state (C for closed), never offered
  as a cancellation to leave out; the minority value (X) is.
* A table joined only to keep units apart is not described as a path the reader
  chose; a grouping the reader asked for still is.
* A period of a series with no row at all (a month-end snapshot never loaded) is
  named as a gap, not left for the reader to take as a zero; a top N, which leaves
  periods out on purpose, says nothing.
* A count by a date that most rows lack (subscriptions still running have no end
  date) counts the rows that have it, and says how many do not.
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


@pytest.fixture(scope="module")
def purchasing():
    built, model = learn(domains.build("purchasing"), "descriptive")
    return built.con, model


@pytest.fixture(scope="module")
def subscriptions():
    built, model = learn(domains.build("subscriptions"), "descriptive")
    return built.con, model


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _payload(learned, plan: dict) -> dict:
    con, model = learned
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(
        {"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert payload["data"]["rows"], payload
    return payload


def _notes(learned, plan: dict) -> list[str]:
    return _payload(learned, plan)["trust"]["date_context"]


def _between(start: str, end: str, **time) -> dict:
    return {"window": {"kind": "between", "start": start, "end": end}, **time}


def test_a_month_far_from_every_other_is_named_when_the_answer_covers_it(purchasing):
    con, model = purchasing
    typed = con.execute("SELECT MAX(line_amount) FROM purchase_order_lines WHERE order_date >= DATE '2025-11-01' "
                        "AND order_date < DATE '2025-12-01'").fetchone()[0]
    assert typed > 100_000                                  # the line typed a thousand times too large
    for time in (_between("2025-01-01", "2025-12-31", grain="month"), _between("2025-11-03", "2025-11-07"), {}):
        notes = _notes(purchasing, {"intent": "value", "measures": ["line_amount"], "time": time})
        assert any(n.startswith("Nov 2025 stands out: line amount of") for n in notes), (time, notes)
    for time in (_between("2026-01-01", "2026-03-31"), _between("2025-12-01", "2025-12-31")):
        notes = _notes(purchasing, {"intent": "value", "measures": ["line_amount"], "time": time})
        assert not any("stands out" in n for n in notes), (time, notes)
    # Compared with it, it is covered too.
    notes = _notes(purchasing, {"intent": "compare", "measures": ["line_amount"], "time": {
        **_between("2025-12-01", "2025-12-31"), "compare": {"kind": "previous_period"}}})
    assert any(n.startswith("Nov 2025 stands out") for n in notes), notes


def test_a_measure_the_same_on_every_row_says_so(retail):
    notes = _notes(retail, {"intent": "value", "measures": ["margin_percent"], "time": _between("2025-01-01",
                                                                                                "2025-12-31")})
    assert "Margin percent is 38 on every row: it tells nothing apart." in notes
    notes = _notes(retail, {"intent": "value", "measures": ["net_amount"]})
    assert not any("on every row" in n for n in notes)
    con, model = retail
    margin = next(m for m in model.measures.values() if m.slug == "margin_percent")
    summed = model.model_copy(deep=True)
    summed.measures[margin.key].expr = margin.expr.model_copy(update={"agg": "sum"})
    notes = _notes((con, summed), {"intent": "value", "measures": ["margin_percent"]})
    assert not any("on every row" in n for n in notes), notes


def test_a_count_of_a_lists_members_says_how_many_are_used(purchasing):
    notes = _notes(purchasing, {"intent": "count", "measures": ["number_of_items"]})
    assert "80 items are listed; 55 appear in purchase order lines." in notes
    notes = _notes(purchasing, {"intent": "count", "measures": ["number_of_purchase_order_lines"]})
    assert not any("are listed" in n for n in notes)


def test_only_a_minority_status_is_offered_as_one_to_leave_out(purchasing):
    con, model = purchasing
    flag = next(q for q in model.quality if q.kind == "status_column")
    assert flag.data["cancel_like"] == ["X"]                # C is "closed": 90% of lines, the normal state
    notes = _notes(purchasing, {"intent": "value", "measures": ["line_amount"]})
    assert "Includes rows whose status is X; an admin can leave them out by default." in notes


def test_a_table_joined_only_to_keep_units_apart_is_not_called_a_path(purchasing):
    by_month = _notes(purchasing, {"intent": "trend", "measures": ["received_quantity"], "time": _between(
        "2026-01-01", "2026-05-31", grain="month")})
    assert any("shown per unit of measure" in n for n in by_month)
    assert not any(n.startswith("Item:") for n in by_month), by_month
    by_category = _notes(purchasing, {"intent": "breakdown", "measures": ["received_quantity"],
                                      "group_by": ["item.category"]})
    assert any(n.startswith("Item: each row's own") for n in by_category), by_category


def test_a_period_with_no_row_is_named_a_gap_not_left_to_read_as_zero(subscriptions):
    con, model = subscriptions
    assert not con.execute("SELECT COUNT(*) FROM mrr_monthly WHERE month_end_date = DATE '2025-02-28'").fetchone()[0]
    by_month = {"intent": "trend", "measures": ["mrr"], "time": _between("2025-01-01", "2025-12-31", grain="month")}
    payload = _payload(subscriptions, by_month)
    assert len(payload["data"]["rows"]) == 11
    assert "No data for Feb 2025: shown as a gap, not as zero." in payload["coverage_caveats"]
    assert payload["answer"]["headline"].startswith("MRR in 2025, by month:")        # an acronym, as written
    whole = _payload(subscriptions, {**by_month, "time": _between("2025-03-01", "2025-12-31", grain="month")})
    assert not any(c.startswith("No data") for c in whole["coverage_caveats"])
    # Before the data starts (January 2024) a month has no row because nothing was there yet.
    early = _payload(subscriptions, {**by_month, "time": _between("2023-07-01", "2024-06-30", grain="month")})
    assert len(early["data"]["rows"]) == 6
    assert not any(c.startswith("No data") for c in early["coverage_caveats"]), early["coverage_caveats"]
    top = _payload(subscriptions, {**by_month, "intent": "rank", "sort": [{"by": "mrr", "desc": True}], "limit": 3})
    assert not any(c.startswith("No data") for c in top["coverage_caveats"])


def test_a_count_by_a_date_most_rows_lack_says_how_many_lack_it(subscriptions):
    con, model = subscriptions
    ended = con.execute("SELECT COUNT(*) FROM subscriptions WHERE end_date IS NOT NULL").fetchone()[0]
    running = con.execute("SELECT AVG(CASE WHEN end_date IS NULL THEN 1.0 ELSE 0 END) FROM subscriptions").fetchone()[0]
    payload = _payload(subscriptions, {"intent": "count", "measures": ["number_of_subscriptions"],
                                       "time": {"date": "end_date"}})
    assert payload["kpi"]["value"] == ended
    assert f"Rows with no end date ({float(running):.0%}) are left out." in payload["trust"]["date_context"]
    payload = _payload(subscriptions, {"intent": "count", "measures": ["number_of_subscriptions"]})
    assert payload["kpi"]["value"] == con.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
