"""The sentence and the chart say what the rows show (issues A6, A7, E4, F1, F3, G4).

* A quantity held in several units is answered per unit (E4): "1,200 EA and 300 M",
  never 1,500. The table has every unit; the sentence and the chart use the unit
  most of the data is in, and say so. A question that already names the unit
  (grouped by it, or one unit) is answered as asked, and a reader who may not use
  the unit's table gets the total with a note that it adds different units.
* A ranking of periods ("which month had the most") names the period that leads
  and is drawn as ranked bars, never as a time line in ranked order (F1).
* A comparison by member says which way each moved: when every member fell, none
  "rose most" (G4).
* A series is described by its complete periods: a partly covered year or week is
  named, not compared (A7); weeks are Monday weeks, read "the week of 4 May 2026"
  (A6); the highest period is named, and nothing is called flat (F3).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.answer.builder import fmt
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn, same_rows
from tests.test_core2_answers_a_question_end_to_end import _domain

TODAY = dt.date(2026, 6, 15)
LATEST = "b.balance_date = (SELECT MAX(balance_date) FROM daily_balances)"


class _AI:
    def __init__(self, plan: dict):
        self.plan = json.dumps(plan)

    def __call__(self, stable: str, tail: str) -> str:
        return self.plan


def _ask(model, con, plan: dict, **kw) -> tuple[dict, DuckDBWarehouse]:
    warehouse = DuckDBWarehouse(con)
    services = Services(model=model, warehouse=warehouse, complete=_AI({"kind": "query", **plan}),
                        index=MemberIndex(), today=TODAY, **kw)
    return answer_question("q", services, Session()), warehouse


def _same(reference, sql: str, warehouse: DuckDBWarehouse) -> str:
    want, ran = reference.query(sql), warehouse.query(warehouse.log[-1])
    return same_rows(want.columns, want.rows, ran.columns, ran.rows, order_matters=False)


@pytest.fixture(scope="module")
def inventory():
    _, built, model, reference = _domain("inventory")
    return built.con, model, reference


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


ON_HAND = {"intent": "value", "measures": ["on_hand_quantity"], "time": {"date": "balance_date"}}


# ── units ──────────────────────────────────────────────────────────────────


def test_stock_on_hand_is_answered_per_unit_never_added_across_units(inventory):
    con, model, reference = inventory
    payload, warehouse = _ask(model, con, ON_HAND)
    per_unit = (f"SELECT i.unit_of_measure, SUM(b.on_hand_qty) FROM daily_balances b "
                f"LEFT JOIN items i ON i.item_id = b.item_id WHERE {LATEST} GROUP BY 1")
    assert not _same(reference, per_unit, warehouse)
    headline = payload["answer"]["headline"]
    for unit, total in reference.query(per_unit).rows:
        assert f"{fmt(total, 'number')} {unit}" in headline, headline
    notes = payload["trust"]["date_context"]
    assert any("shown per unit of measure" in n for n in notes)
    assert not any("adds up different units" in n for n in notes)
    chips = [c["question"] for c in payload["follow_up_suggestions"]]   # the unit is not a member to drill into
    assert not any("EA" in c or "BOX" in c or "unit" in c.lower() for c in chips), chips


def test_a_breakdown_keeps_units_apart_and_draws_the_unit_most_items_are_in(inventory):
    con, model, reference = inventory
    payload, warehouse = _ask(model, con, {**ON_HAND, "intent": "breakdown", "group_by": ["warehouse.name"]})
    assert not _same(reference, f"SELECT w.warehouse_name, i.unit_of_measure, SUM(b.on_hand_qty) "
                                f"FROM daily_balances b LEFT JOIN warehouses w ON w.warehouse_id = b.warehouse_id "
                                f"LEFT JOIN items i ON i.item_id = b.item_id WHERE {LATEST} GROUP BY 1, 2", warehouse)
    data, chart = payload["data"], payload["chart"]
    unit = next(h for h in data["headers"] if {r[h] for r in data["rows"]} <= {"EA", "BOX", "M", "Unknown"})
    name, value = chart["x_key"], chart["y_keys"][0]
    # EA: the unit of 42 of the 60 items, so of most stock rows.
    assert chart["title"].endswith("(EA)") and any("Only EA is drawn" in w for w in chart["chart_warnings"])
    assert {r[name]: r[value] for r in chart["rows"]} == {r[name]: r[value] for r in data["rows"] if r[unit] == "EA"}
    assert " EA" in payload["answer"]["headline"] and "of the total" not in payload["answer"]["headline"]
    assert any("counted in 3 units" in c for c in payload["coverage_caveats"])
    chips = [c["question"] for c in payload["follow_up_suggestions"]]
    assert chips and not any("EA" in c or "BOX" in c or "unit" in c.lower() for c in chips), chips


def test_a_question_that_names_the_unit_is_answered_as_asked(inventory):
    con, model, reference = inventory
    by_unit, warehouse = _ask(model, con, {**ON_HAND, "intent": "breakdown",
                                           "group_by": ["warehouse.name", "item.unit_of_measure"]})
    assert not _same(reference, f"SELECT w.warehouse_name, i.unit_of_measure, SUM(b.on_hand_qty) "
                                f"FROM daily_balances b LEFT JOIN warehouses w ON w.warehouse_id = b.warehouse_id "
                                f"LEFT JOIN items i ON i.item_id = b.item_id WHERE {LATEST} GROUP BY 1, 2", warehouse)
    assert len(by_unit["data"]["headers"]) == 3 and not by_unit["coverage_caveats"]
    one, warehouse = _ask(model, con, {**ON_HAND, "filters": [
        {"field": "item.unit_of_measure", "op": "eq", "values": ["M"]}]})
    assert not _same(reference, f"SELECT SUM(b.on_hand_qty) FROM daily_balances b JOIN items i "
                                f"ON i.item_id = b.item_id WHERE {LATEST} AND i.unit_of_measure = 'M'", warehouse)
    assert one["kpi"] is not None


def test_a_reader_who_may_not_use_the_unit_gets_the_total_with_a_note(inventory):
    con, model, reference = inventory
    allowed = {k for k, t in model.tables.items() if t.name != "items"}
    payload, warehouse = _ask(model, con, ON_HAND, allowed_tables=allowed)
    assert not _same(reference, f"SELECT SUM(b.on_hand_qty) FROM daily_balances b WHERE {LATEST}", warehouse)
    assert any("adds up different units" in n for n in payload["trust"]["date_context"])
    assert "items" not in warehouse.log[-1].lower()


# ── rankings of periods ────────────────────────────────────────────────────


def _months_2025(con, n: int) -> list[tuple]:
    return con.execute(
        "SELECT date_trunc('month', c.full_date) AS month, SUM(o.net_amount) AS v FROM order_lines o "
        "JOIN calendar c ON c.date_key = o.order_date_key WHERE c.full_date BETWEEN '2025-01-01' AND '2025-12-31' "
        f"GROUP BY 1 ORDER BY v DESC LIMIT {n}").fetchall()


RANKED = {"intent": "rank", "measures": ["net_amount"], "sort": [{"by": "net_amount", "desc": True}], "limit": 3,
          "time": {"grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}


def test_a_ranking_of_months_names_the_leading_month_and_is_drawn_as_ranked_bars(retail):
    con, model = retail
    payload, _ = _ask(model, con, RANKED)
    top = _months_2025(con, 3)
    (first, a), (second, b) = top[0], top[1]
    assert payload["answer"]["headline"] == (f"{first:%b %Y} had the highest net amount in 2025: "
                                             f"{fmt(a, 'currency')}, then {second:%b %Y} ({fmt(b, 'currency')}).")
    chart = payload["chart"]
    assert chart["chart_type"] == "bar" and chart["chart_spec"]["x"]["role"] == "dimension"
    assert [r[chart["x_key"]] for r in chart["rows"]] == [f"{month:%b %Y}" for month, _ in top]


def test_a_ranking_of_months_is_not_drawn_as_a_time_line(retail):
    con, model = retail
    payload, _ = _ask(model, con, {**RANKED, "chart": "line"})
    assert payload["chart"]["chart_type"] == "bar"
    assert any("A line chart does not fit" in n for n in payload["trust"]["date_context"])


# ── comparisons ────────────────────────────────────────────────────────────


def _regions(con, start: str, end: str) -> dict[str, float]:
    rows = con.execute(
        "SELECT r.region_name, SUM(o.net_amount) FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key "
        "JOIN stores s ON s.store_id = o.store_id JOIN regions r ON r.region_id = s.region_id "
        f"WHERE c.full_date BETWEEN '{start}' AND '{end}' GROUP BY 1").fetchall()
    return {name: float(v) for name, v in rows}


def _compare(month: str) -> dict:
    start = dt.date.fromisoformat(month)
    end = (start + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    return {"intent": "compare", "measures": ["net_amount"], "group_by": ["region"],
            "time": {"window": {"kind": "between", "start": str(start), "end": str(end)},
                     "compare": {"kind": "previous_period"}}}


def test_when_every_member_fell_none_is_said_to_have_risen(retail):
    con, model = retail
    now, before = _regions(con, "2025-01-01", "2025-01-31"), _regions(con, "2024-12-01", "2024-12-31")
    change = {r: now[r] - before[r] for r in now}
    assert max(change.values()) < 0                      # every region fell in January 2025
    worst = min(change, key=change.__getitem__)
    headline = _ask(model, con, _compare("2025-01-01"))[0]["answer"]["headline"]
    assert headline == (f"Net amount in January 2025 against December 2024: all {len(change)} regions fell; "
                        f"{worst} fell most ({fmt(change[worst], 'currency')}).")


def test_a_rise_is_signed_and_a_fall_is_signed(retail):
    con, model = retail
    now, before = _regions(con, "2026-05-01", "2026-05-31"), _regions(con, "2026-04-01", "2026-04-30")
    change = {r: now[r] - before[r] for r in now}
    best, worst = max(change, key=change.__getitem__), min(change, key=change.__getitem__)
    assert change[best] > 0 > change[worst]
    headline = _ask(model, con, _compare("2026-05-01"))[0]["answer"]["headline"]
    assert f"{best} rose most (+{fmt(change[best], 'currency')})" in headline
    assert f"{worst} fell most ({fmt(change[worst], 'currency')})" in headline


# ── series ─────────────────────────────────────────────────────────────────


def test_a_partly_covered_year_is_named_not_compared(retail):
    con, model = retail
    payload, _ = _ask(model, con, {"intent": "trend", "measures": ["net_amount"], "time": {"grain": "year"}})
    headline = payload["answer"]["headline"]
    assert "2026" not in headline and "in 2025; highest" in headline
    assert any(c.startswith("2026 is only partly covered") for c in payload["coverage_caveats"])


def test_weeks_are_monday_weeks_and_read_as_the_week_of(retail):
    con, model = retail
    payload, _ = _ask(model, con, {"intent": "trend", "measures": ["net_amount"], "time": {
        "grain": "week", "window": {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}}})
    period = payload["data"]["headers"][0]
    assert all(dt.date.fromisoformat(r[period]).weekday() == 0 for r in payload["data"]["rows"])
    headline = payload["answer"]["headline"]
    assert "in the week of 4 May 2026" in headline and "Week of 27 Apr" not in headline    # partly in May
    assert any(c.startswith("Week of 27 Apr 2026 is only partly covered") for c in payload["coverage_caveats"])


def test_a_series_with_a_spike_names_its_highest_period_and_is_never_called_flat(retail):
    con, model = retail
    payload, _ = _ask(model, con, {"intent": "trend", "measures": ["net_amount"], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}})
    (peak, value), = _months_2025(con, 1)
    headline = payload["answer"]["headline"]
    assert f"highest {fmt(value, 'currency')} in {peak:%b %Y}" in headline and "flat" not in headline


def test_a_series_listed_newest_first_still_reads_and_draws_forward(retail):
    con, model = retail
    plan = {"intent": "trend", "measures": ["net_amount"], "sort": [{"by": "period", "desc": True}], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}}
    payload, _ = _ask(model, con, plan)
    period = payload["data"]["headers"][0]
    assert payload["data"]["rows"][0][period] == "2025-12-01"                   # the table as asked
    assert " in Jan 2025, " in payload["answer"]["headline"]                  # the sentence from the start
    xs = [r[payload["chart"]["x_key"]] for r in payload["chart"]["rows"]]
    assert payload["chart"]["chart_type"] == "line" and xs == sorted(xs)       # the line runs forward
