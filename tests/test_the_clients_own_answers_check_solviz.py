"""The client's own questions, checked against the figures in their own reports.

The setup check keeps the client team's real questions with the answer their own report gives, and until now
nothing used them: a wrong number was found when a manager noticed it in a meeting. Now each question is asked as
a reader would ask it, every figure the report gives is looked for in the answer (read in code, rounded the way
the report rounds it), and the admin sees a scorecard with each question side by side. The check runs again by
itself: everything asked afresh after a Learn, the answers drawn again (no AI) once decisions stop changing, and
a new question as it arrives.

The real store, routes, service and governed SQL on the invented retail warehouse; only the AI is stood in, as
the plan it would write for each question. Every expected figure is worked out with SQL written by hand.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import threading
import time
from unittest.mock import patch

import pytest

from core2 import answers_check as ac
from tests.test_core2_answers_in_the_portal import learned  # noqa: F401 - the fixture
from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _page as _learned_page,
    _request,
    workspace,  # noqa: F401 - the fixture
)

IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
SALES = {"intent": "value", "measures": ["net_amount"], "time": IN_2025}
BY_REGION = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["region"], "time": IN_2025}
ORDERS = {"intent": "value", "measures": ["number_of_orders"], "time": IN_2025}


class _Planner:
    """The AI, stood in: the plan it would write for each question, by words in it (an unknown question cannot be
    planned). ``asked`` keeps every question it was given."""

    def __init__(self, plans: dict[str, dict]):
        self.plans, self.asked = plans, []

    def __call__(self, account_id, client, *, question, question_id=""):
        self.asked.append(question)
        plan = next((p for words, p in self.plans.items() if words in question.lower()), None)

        def complete(stable: str, tail: str) -> str:
            return json.dumps({"kind": "query", **plan} if plan else {"kind": "unsupported", "reason": "no such data"})
        return complete


def _sql(built, sql: str) -> float:
    (value,) = built.con.execute(sql).fetchone()
    return float(value)


SALES_2025 = """SELECT SUM(o.net_amount) FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key
                WHERE c.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' {extra}"""


def _db_id(store) -> int:
    return int(store.get_client(ACCOUNT)["db_config_id"])


def _add(store, question: str, expected: str, **extra) -> int:
    return store.add_core2_test_question(ACCOUNT, _db_id(store), question=question, expected=expected, **extra)


def _latest(store) -> dict[int, dict]:
    return {qid: rows[0] for qid, rows in store.core2_answer_results(ACCOUNT, _db_id(store)).items()}


@pytest.fixture(autouse=True)
def _fresh_debounce():
    ac._SEEN.clear()
    yield
    ac._SEEN.clear()


# ── the report's figures ───────────────────────────────────────────────────


def test_a_reports_figures_are_read_with_their_rounding_their_member_and_never_a_date():
    (m,) = ac.expected_figures("$3.09M")
    assert (m.value, m.step, m.percent) == (3_090_000, 10_000, False)
    north, south = ac.expected_figures("North: 41, South: 37")
    assert (north.label, north.value, north.step) == ("North", 41, 1)
    assert (south.label, south.value) == ("South", 37)
    (share,) = ac.expected_figures("12.5%")
    assert share.percent and share.value == 12.5 and share.step == pytest.approx(0.1)
    (orders,) = ac.expected_figures("41 orders in March 2025")
    assert orders.value == 41, "a month and its year say when, they are no figure"
    (q2,) = ac.expected_figures("Q2 2025: 1,250")
    assert (q2.label, q2.value) == ("Q2 2025", 1250)
    (loss,) = ac.expected_figures("(1,200)")
    assert loss.value == -1200
    (lower,) = ac.expected_figures("1.2m")
    assert lower.value == 1_200_000
    (count,) = ac.expected_figures("41 m")
    assert count.value == 41, "a lone lower-case m beside a whole number is not a million"
    assert ac.expected_figures("Mostly online") == []
    # A member's later figures are still that member's; a figure said again in brackets is no loss.
    loans, share = ac.expected_figures("Kenji Watanabe: 45 loans (4.0%)")
    assert (loans.label, loans.value) == ("Kenji Watanabe", 45) and (share.label, share.value) == ("Kenji Watanabe", 4)
    assert [(f.label, f.value) for f in ac.expected_figures("North: $1.2M, 12%")] == [("North", 1.2e6), ("North", 12)]
    assert [f.value for f in ac.expected_figures("$3.09M ($3,093,601)")] == [3_090_000, 3_093_601]
    # Digits inside a name are no figure.
    assert [(f.label, f.value) for f in ac.expected_figures("Lincoln 27th St: 496")] == [("Lincoln 27th St", 496)]


def test_a_figure_matches_as_the_report_rounds_it_and_close_is_within_one_percent():
    payload = {"answer": {"headline": "h"}, "data": {"rows": [{"v": 3_093_601.41}], "column_formats": {"v": "currency"},
                                                      "header_labels": {"v": "Net amount"}}}
    assert ac.compare("$3.09M", payload)["verdict"] == "matches"
    assert ac.compare("$3.1M", payload)["verdict"] == "matches"
    assert ac.compare("3,100,000", payload)["verdict"] == "close"
    assert ac.compare("$3.2M", payload)["verdict"] == "different"
    assert ac.compare("", payload)["verdict"] == "unchecked"
    # Solviz's figure is shown as the report writes it, and in full.
    (kept,) = ac.compare("$3.09M", payload)["figures"]
    assert ac.said(kept["got"], kept, kept["format"]) == "$3.09M ($3,093,601)"
    # A share is shown in points, as the answer's table shows it.
    shares = {"answer": {}, "data": {"rows": [{"s": 12.47}], "column_formats": {"s": "percentage"}}}
    assert ac.compare("12.5%", shares)["verdict"] == "matches"
    (kept,) = ac.compare("12.5%", shares)["figures"]
    assert ac.said(kept["got"], kept, kept["format"]) == "12.47%"
    regions = {"answer": {}, "data": {"rows": [{"r": "North", "v": 1_178_255.27, "s": 38.1},
                                               {"r": "West", "v": 600_861.10, "s": 19.4}],
                                      "column_formats": {"r": "text", "v": "currency", "s": "percentage"},
                                      "header_labels": {"v": "Net sales", "s": "Share"}}}
    assert ac.compare("North: $1.18M, 38.1%, West: $601K, 19.4%", regions)["verdict"] == "matches"
    for total in ("Total: $1.78M", "Total sales: $1.78M"):
        assert ac.compare(total, regions)["verdict"] == "matches", "a total is the rows added up"
    missing = ac.compare("North / Retail: $1.18M, 38.1%", regions)
    assert missing["verdict"] == "different" and all(f["got"] is None for f in missing["figures"]), \
        "the member's later figures go with it"
    # The report's period goes with a question that names none, never twice.
    assert ac.asked_text({"question": "What were net sales?", "period": "2025"}) == "What were net sales, for 2025?"
    for named in ("Net sales by store in the first half of 2026", "Net sales in Q2", "Net sales in March"):
        assert ac.asked_text({"question": named, "period": "H1 2026"}) == named
    asked_back = {"answer": {"headline": "Which store?"}, "clarify": {"question": "Which store?", "options": ["A"]},
                  "data": None}
    verdict = ac.compare("41", asked_back)
    assert verdict["verdict"] == "unanswered" and verdict["why"] == "Solviz asked back instead of answering: " \
        "Which store? (A)"


def _table(rows, formats, headers=None):
    return {"answer": {"headline": "h"}, "data": {"rows": rows, "column_formats": formats,
                                                  "header_labels": headers or {}}}


_REGIONS = {"r": "text", "v": "number"}


@pytest.mark.parametrize("expected, payload", [
    # A member is the row it names, never one holding its name ("North East" beside "North").
    ("North: 41", _table([{"r": "North", "v": 30}, {"r": "North East", "v": 41}], _REGIONS, {"v": "Orders"})),
    # A count is never read from a share column, nor a percentage from a count.
    ("North: 38, West: 19", _table([{"r": "North", "o": 41, "s": 38.09}, {"r": "West", "o": 27, "s": 19.42}],
                                   {"r": "text", "o": "number", "s": "percentage"},
                                   {"o": "Orders", "s": "Share of net amount"})),
    ("12%", _table([{"n": 12}], {"n": "number"}, {"n": "Cancelled orders"})),
    ("12%", _table([{"n": 1200}], {"n": "number"}, {"n": "Cancelled orders"})),
    # A share kept as 0.25 shows as 0.3%: no 25%.
    ("25%", _table([{"m": 0.25}], {"m": "percentage"}, {"m": "Margin"})),
    # Measure words and a member: the member's row.
    ("North sales: 41", _table([{"r": "North", "v": 30}, {"r": "South", "v": 41}], _REGIONS, {"v": "Sales"})),
    # A member written after its figure.
    ("41 in North, 37 in South", _table([{"r": "North", "v": 37}, {"r": "South", "v": 41}], _REGIONS)),
    # One row for another member.
    ("North: 41", _table([{"r": "South", "v": 41}], _REGIONS, {"v": "Orders"})),
    # A figure beside several rows, or a total, is the rows added up, never one of them.
    ("12", _table([{"m": "2025-01-01", "v": 5}, {"m": "2025-02-01", "v": 12}], {"m": "date", "v": "number"})),
    ("Total: $1.78M", _table([{"r": "North", "v": 1_780_000}, {"r": "South", "v": 1_200_000}],
                             {"r": "text", "v": "currency"}, {"v": "Net sales"})),
    ("100", _table([{"r": "North", "n": 40, "s": 46.0}, {"r": "South", "n": 47, "s": 54.0}],
                   {"r": "text", "n": "number", "s": "percentage"}, {"n": "New customers", "s": "Share"})),
    # "$2M" is no $1.55M.
    ("$2M", _table([{"v": 1_550_000}], {"v": "currency"})),
    # "Q2" in an answer holding two years' Q2.
    ("Q2: 41", _table([{"d": "2024-04-01", "v": 41}, {"d": "2025-04-01", "v": 30}], {"d": "date", "v": "number"})),
    # The figure the report's question names first, not another column.
    ("41", _table([{"o": 50, "i": 41}], {"o": "number", "i": "number"}, {"o": "Orders", "i": "Items"})),
    # A report's figures in another order are each read from a column of their own kind: never two members' figures
    # spread over one row's columns, money from a count, a figure off a column of zeros, or two years against what
    # the headers say.
    ("North: 41, South: 37", _table([{"o": 37, "i": 41}], {"o": "number", "i": "number"}, {"o": "Orders", "i": "Items"})),
    ("Jan 2025: 500, Feb 2025: 750", _table([{"m": "Jan", "o": 200, "i": 300}, {"m": "Feb", "o": 300, "i": 450}],
                                            {"m": "text", "o": "number", "i": "number"}, {"o": "Orders", "i": "Items"})),
    ("Live Oak: $7,077, 2,048", _table([{"b": "Live Oak", "f": 2047.87, "n": 7077}],
                                       {"b": "text", "f": "currency", "n": "number"}, {"f": "Fee amount", "n": "Loans"})),
    ("Live Oak: 7,077, 0", _table([{"b": "Live Oak", "r": 2047.87, "f": 7076.59, "w": 0}],
                                  {"b": "text", "r": "currency", "f": "currency", "w": "currency"},
                                  {"r": "Reversal amount", "f": "Fee amount", "w": "Write-off"})),
    ("2024: 25,480,000, 2025: 33,983,000", _table([{"r": "Lakeshore", "c": 25_480_000, "b": 33_983_000}],
                                                  {"r": "text", "c": "currency", "b": "currency"},
                                                  {"c": "Loan amount", "b": "Loan amount, a year before"})),
    # A member called Growth is no "change"; a label of header words names the column, never a row holding one.
    ("Growth change: 12", _table([{"f": "Growth", "a": 1000, "c": 5}, {"f": "Income", "a": 2000, "c": 7}],
                                 {"f": "text", "a": "number", "c": "number"}, {"a": "Assets", "c": "Assets change"})),
    ("Loans: 30,319,500", _table([{"p": "Consumer Loans", "c": 30_319_500}, {"p": "Mortgages", "c": 161_728_000}],
                                 {"p": "text", "c": "currency"}, {"c": "Loan amount"})),
])
def test_a_figure_the_answer_does_not_give_is_never_read_as_a_match(expected, payload):
    # Each of these read "matches" before: a wrong answer set beside a right report and passed.
    assert ac.compare(expected, payload)["verdict"] == "different"


def test_a_figure_is_read_as_reports_write_it():
    assert [(f.label, f.value) for f in ac.expected_figures("North - 41, South - 37")] == [("North", 41), ("South", 37)]
    assert [f.value for f in ac.expected_figures("$1.2M – $1.3M")] == [1_200_000, 1_300_000]
    assert [(f.label, f.value, f.percent) for f in ac.expected_figures("North (12%)")] == [("North", 12, True)]
    assert [(f.label, f.value) for f in ac.expected_figures("Total: 2048 (12.5%)")] == [("Total", 2048), ("Total", 12.5)]
    # A comparison beside the figure is no figure of its own.
    assert [f.value for f in ac.expected_figures("North: 41 (2024: 37)")] == [41]
    assert [f.value for f in ac.expected_figures("Net sales were $3.09M, up 4.1% on 2024")] == [3_090_000]
    assert [f.value for f in ac.expected_figures("$1.2M (+5% YoY)")] == [1_200_000]
    # A member's own number, and a number running into letters, are no figures.
    assert [(f.label, f.value) for f in ac.expected_figures("Store 12: 496")] == [("Store 12", 496)]
    assert ac.expected_figures("4.2x") == [] and ac.expected_figures("12.5pp") == []
    assert [(f.label, f.value) for f in ac.expected_figures("North was 41")] == [("North", 41)]
    assert [(f.label, f.value) for f in ac.expected_figures("North 41 / South 37")] == [("North", 41), ("South", 37)]
    # French figures.
    assert [(f.value, f.percent) for f in ac.expected_figures("Marge : 12,5 %")] == [(12.5, True)]
    assert [f.value for f in ac.expected_figures("1 200 €")] == [1200]
    # "by month" says how the answer is split, never which period it is for.
    assert ac.asked_text({"question": "What were sales by month?", "period": "Q2 2025"}) == \
        "What were sales by month, for Q2 2025?"


def test_figures_that_are_right_still_match():
    regions = _table([{"r": "North", "o": 41, "s": 38.09}, {"r": "North East", "o": 30, "s": 27.1},
                      {"r": "West", "o": 27, "s": 19.42}], {"r": "text", "o": "number", "s": "percentage"},
                     {"o": "Orders", "s": "Share of net amount"})
    assert ac.compare("North: 41 (38.1%), West: 27", regions)["verdict"] == "matches"
    assert ac.compare("North - 41, North East - 30", regions)["verdict"] == "matches"
    assert ac.compare("Total: 98", regions)["verdict"] == "matches"
    two = _table([{"v": 3_093_601, "c": 1_918_033}], {"v": "currency", "c": "currency"},
                 {"v": "Net amount", "c": "Cost amount"})
    assert ac.compare("$3.09M, $1.92M", two)["verdict"] == "matches", "figures in the order of the answer's columns"
    assert ac.compare("Cost: $1.92M, Net: $3.09M", two)["verdict"] == "matches", "or by the column they name"
    assert ac.compare("Net sales were $3.09M ($3,093,601), up 4.1% on 2024", two)["verdict"] == "matches"
    months = _table([{"m": "2025-01-01", "v": 500}, {"m": "2025-02-01", "v": 750}, {"m": "2026-01-01", "v": 9}],
                    {"m": "date", "v": "number"})
    assert ac.compare("2025: 1,250", months)["verdict"] == "matches", "a period over several rows is their total"
    assert ac.compare("Jan 2026: 9", months)["verdict"] == "matches"


def test_a_reports_figures_in_its_own_order_match_the_answers_columns():
    # Reports list a member's figures in their own order: last year before this year, fees before reversals.
    fees = _table([{"b": "Live Oak", "r": 2047.87, "f": 7076.59, "p": 28.94}],
                  {"b": "text", "r": "currency", "f": "currency", "p": "percentage"},
                  {"r": "Reversal amount", "f": "Fee amount", "p": "Reversals as % of fees"})
    assert ac.compare("Live Oak: 7,077, 2,048, 28.9%", fees)["verdict"] == "matches"
    year = _table([{"r": "Lakeshore", "p": "Mortgages", "c": 33_983_000, "b": 25_480_000, "d": 8_503_000}],
                  {"r": "text", "p": "text", "c": "currency", "b": "currency", "d": "currency"},
                  {"c": "Loan amount", "b": "Loan amount, a year before", "d": "Loan amount: change"})
    assert ac.compare("2024: 25,480,000, 2025: 33,983,000, Growth: 8,503,000", year)["verdict"] == "matches"
    # A member whose name shares a word with a header ("Consumer Loans", "Loan amount") is still that row.
    loans = _table([{"p": "Consumer Loans", "c": 30_319_500, "b": 28_067_400, "d": 2_252_100},
                    {"p": "Mortgages", "c": 161_728_000, "b": 154_125_000, "d": 7_603_000}],
                   {"p": "text", "c": "currency", "b": "currency", "d": "currency"},
                   {"c": "Loan amount", "b": "Loan amount, a year before", "d": "Loan amount: change"})
    assert ac.compare("Consumer Loans: 28,067,400, 30,319,500, 2,252,100", loans)["verdict"] == "matches"
    # Each figure from a column of its own: the same figure twice, or one the row does not hold, is no match.
    assert ac.compare("Live Oak: 7,077, 7,077", fees)["verdict"] == "different"
    assert ac.compare("Live Oak: 7,077, 2,100", fees)["verdict"] == "different"
    assert ac.compare("Mortgages: 28,067,400, 30,319,500", loans)["verdict"] == "different"
    assert ac.compare("Live Oak: 7,077, 28.94", fees)["verdict"] == "different", "a share is never a plain figure"
    # Header words that also name a row: the column; a member called Growth is that row.
    assert ac.compare("Loans: 192,047,500", loans)["verdict"] == "matches", "the loans added up"
    funds = _table([{"f": "Growth", "a": 1000, "c": 5}, {"f": "Income", "a": 2000, "c": 7}],
                   {"f": "text", "a": "number", "c": "number"}, {"a": "Assets", "c": "Assets change"})
    assert ac.compare("Growth change: 5", funds)["verdict"] == "matches"
    assert ac.compare("Growth assets: 1,000", funds)["verdict"] == "matches"


# ── a run, through the real portal path ────────────────────────────────────


def test_each_question_is_asked_as_a_reader_would_and_set_beside_the_reports_figures(learned):  # noqa: F811
    store, built, _ = learned
    total = _sql(built, SALES_2025.format(extra=""))
    by_region = dict(built.con.execute("""
        SELECT r.region_name, SUM(o.net_amount) FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key
        JOIN stores s ON s.store_id = o.store_id JOIN regions r ON r.region_id = s.region_id
        WHERE c.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' GROUP BY 1 ORDER BY 2 DESC""").fetchall())
    top, second = list(by_region)[:2]
    ids = {
        "matches": _add(store, "What were net sales?", f"${total / 1e6:.2f}M", period="2025"),
        "close": _add(store, "What were net sales in 2025 to the nearest thousand?", f"{total * 1.004:,.0f}"),
        "different": _add(store, "What were net sales in 2025 in the old pack?", f"${total * 1.3 / 1e6:.1f}M"),
        "regions": _add(store, "What were net sales by region in 2025?",
                        f"{top}: ${float(by_region[top]) / 1e3:,.0f}K, {second}: ${float(by_region[second]) / 1e3:,.0f}K"),
        "missing": _add(store, "What were net sales by region in 2025 with Atlantis?", "Atlantis: $1.0M"),
        "unanswered": _add(store, "How many unicorns were sold?", "12"),
        "unchecked": _add(store, "Which region leads net sales by region in 2025?", "Mostly the big stores"),
    }
    planner = _Planner({"by region": BY_REGION, "net sales": SALES})
    with patch("core2.bootstrap.ai.workspace_planner", planner):
        check_id = ac.run(ACCOUNT, mode="ask", reason="admin", started_by="Ana")
    check = store.latest_core2_answer_check(ACCOUNT, _db_id(store))
    assert check["id"] == check_id and check["status"] == "done" and check["done_count"] == len(ids)
    assert "What were net sales?, for 2025?" not in planner.asked and "What were net sales, for 2025?" in planner.asked, \
        "the report's period goes with a question that names none"
    got = _latest(store)
    verdicts = {name: got[qid]["verdict"] for name, qid in ids.items()}
    assert verdicts == {"matches": "matches", "close": "close", "different": "different", "regions": "matches",
                        "missing": "different", "unanswered": "unanswered", "unchecked": "unchecked"}
    (figure,) = got[ids["matches"]]["detail"]["figures"]
    assert figure["got"] == pytest.approx(total) and figure["expected"] == f"${total / 1e6:.2f}M"
    regions = got[ids["regions"]]["detail"]["figures"]
    assert [f["label"] for f in regions] == [top, second]
    assert all(f["where"].startswith(f["label"]) for f in regions), "each figure is read off its own member's row"
    assert "Atlantis is not in Solviz's answer" in got[ids["missing"]]["detail"]["figures"][0]["note"]
    assert got[ids["matches"]]["asked"] == 1 and got[ids["matches"]]["plan"]["measures"] == ["net_amount"]
    assert ac.summary(ACCOUNT, _db_id(store)) == {
        "total": 7, "counts": {"matches": 2, "close": 1, "different": 2, "unanswered": 1, "unchecked": 1},
        "waiting": 0, "checked": 6}


# ── when it runs by itself ─────────────────────────────────────────────────


def test_after_a_learn_every_question_is_asked_again_and_a_new_one_as_it_arrives(learned):  # noqa: F811
    store, built, _ = learned
    db = _db_id(store)
    first = _add(store, "What were net sales in 2025?", "$3.09M")
    assert ac.due(ACCOUNT, db) == ("ask", "first", None)
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        ac.run(ACCOUNT, mode="ask", reason="first")
        assert ac.due(ACCOUNT, db) is None, "nothing changed: nothing to run"
        orders = _add(store, "How many orders in 2025?", "1,000")
        assert ac.due(ACCOUNT, db) == ("ask", "new_questions", [orders])
        ac.run(ACCOUNT, mode="ask", reason="new_questions", only=[orders])
        assert ac.due(ACCOUNT, db) is None
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)                    # Learn again
    assert ac.due(ACCOUNT, db) == ("ask", "learn", None)
    # One question asked again alone, after the Learn, says nothing of the others: they are still asked again.
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        ac.run(ACCOUNT, mode="ask", reason="admin", only=[first])
    assert ac.due(ACCOUNT, db) == ("ask", "learn", None)


def test_once_decisions_stop_changing_the_answers_are_drawn_again_with_no_ai(learned):  # noqa: F811
    store, built, _ = learned
    db = _db_id(store)
    shipped = _sql(built, SALES_2025.format(extra="AND o.status_code <> 'C'"))
    asked = _add(store, "What were net sales in 2025, cancelled orders left out?", f"${shipped:,.0f}")
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        ac.run(ACCOUNT, mode="ask", reason="first")
    assert _latest(store)[asked]["verdict"] == "different", "Solviz counts the cancelled orders: the report does not"
    # The admin decides net amount leaves cancelled orders out.
    store.set_core2_override(ACCOUNT, db, "measure:m:memory.main.order_lines.net_amount", "filters",
                             [{"column": "memory.main.order_lines.status_code", "op": "not_in", "values": ["C"]}])
    t = 1000.0
    assert ac.due(ACCOUNT, db, now=t) is None, "decisions just changed: wait for the next"
    assert ac.due(ACCOUNT, db, now=t + ac.SETTLE_SECONDS / 2) is None
    store.set_core2_override(ACCOUNT, db, "measure:m:memory.main.order_lines.net_amount", "business_name", "Net sales")
    assert ac.due(ACCOUNT, db, now=t + ac.SETTLE_SECONDS + 1) is None, "another change starts the wait again"
    assert ac.due(ACCOUNT, db, now=t + 2 * ac.SETTLE_SECONDS + 2) == ("replay", "decisions", None)

    def no_ai(*args, **kwargs):
        def complete(stable: str, tail: str) -> str:
            raise AssertionError("a replay never asks the AI")
        return complete

    with patch("core2.bootstrap.ai.workspace_planner", no_ai):
        ac.run(ACCOUNT, mode="replay", reason="decisions")
    result = _latest(store)[asked]
    assert result["verdict"] == "matches" and result["asked"] == 0
    assert result["detail"]["figures"][0]["got"] == pytest.approx(shipped)
    assert ac.due(ACCOUNT, db, now=t + 3 * ac.SETTLE_SECONDS) is None


def test_one_run_at_a_time_and_a_run_a_restart_stopped_is_let_go(workspace, monkeypatch):  # noqa: F811
    store, _, _ = workspace
    db = _db_id(store)
    claim = lambda: store.claim_core2_answer_check(ACCOUNT, db, mode="ask", reason="admin", stamp=[1], total=1)  # noqa: E731
    first = claim()
    assert first and claim() is None, "one run at a time for a workspace"
    monkeypatch.setenv(store.core2_store._RUNNER_ENV, "after-a-restart")
    second = claim()
    assert second and second != first
    stopped = [c for c in [store.latest_core2_answer_check(ACCOUNT, db)] if c]
    assert stopped[0]["id"] == second
    with store.get_db() as conn:
        row = conn.execute("SELECT status, message FROM core2_answer_check WHERE id = ?", (first,)).fetchone()
    assert row["status"] == "failed" and "restarted" in row["message"]


def test_nothing_a_restart_stopped_holds_the_check_for_good(learned, monkeypatch):  # noqa: F811
    store, _, _ = learned
    db = _db_id(store)
    qid = _add(store, "What were net sales in 2025?", "$3.09M")
    # A run this process was minding, then a restart: the page lets it go as soon as it is read.
    held = store.claim_core2_answer_check(ACCOUNT, db, mode="ask", reason="admin", stamp=[1], total=1)
    monkeypatch.setenv(store.core2_store._RUNNER_ENV, "after-a-restart")
    assert "data-running=\"1\"" not in _shown(), "the page shows no run as running"
    check = store.latest_core2_answer_check(ACCOUNT, db)
    assert check["id"] == held and check["status"] == "failed"
    store.update_core2_answer_check(held, done_count=1, status="done")
    assert store.latest_core2_answer_check(ACCOUNT, db)["status"] == "failed", "a run let go stays let go"
    # A Learn a restart stopped stays 'running' in the store: the check goes on all the same.
    store.start_core2_build(ACCOUNT, db)
    monkeypatch.setenv(store.core2_store._RUNNER_ENV, "after-another-restart")
    assert ac.due(ACCOUNT, db) == ("ask", "first", None)
    # A run claimed for a question taken off before it started is closed, never left running.
    claimed = store.claim_core2_answer_check(ACCOUNT, db, mode="ask", reason="admin", stamp=[1], total=1,
                                             whole=False)
    store.delete_core2_test_question(ACCOUNT, db, qid)
    assert ac.run(ACCOUNT, mode="ask", reason="admin", only=[qid], check_id=claimed) is None
    assert store.latest_core2_answer_check(ACCOUNT, db)["status"] == "done"


def test_the_minute_scheduler_starts_the_runs_that_are_due(learned):  # noqa: F811
    store, _, _ = learned
    _add(store, "What were net sales in 2025?", "$3.09M")
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        from core.notification_scheduler import run_due_notifications_once

        run_due_notifications_once()
        _wait(store)
    check = store.latest_core2_answer_check(ACCOUNT, _db_id(store))
    assert check["status"] == "done" and check["reason"] == "first" and check["started_by"] == "Solviz"
    assert ac.run_due_answer_checks() == 0, "checked on the model as it is: nothing more is due"


def _wait(store, seconds: float = 60) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check = store.latest_core2_answer_check(ACCOUNT, _db_id(store)) or {}
        if check.get("status") != "running" and not any(t.name.startswith("answers-check-")
                                                         for t in threading.enumerate()):
            return
        time.sleep(0.2)
    raise AssertionError("the answers check did not finish")


# ── the admin page ─────────────────────────────────────────────────────────


def _admin(handler, path: str, fields: dict | None = None, method: str = "POST", query: str = "", **kw):
    from admin import core2_answers_check as page

    with patch.object(page, "_is_auth", return_value=True):
        return asyncio.run(handler(_request(path, fields, method=method, query=query), ACCOUNT, **kw))


def _text(page: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", page).split()).replace(" :", ":")


def _shown(query: str = "") -> str:
    from admin import core2_answers_check as page

    response = _admin(page.answers_check_page, f"/admin/clients/{ACCOUNT}/answers-check", method="GET", query=query)
    return html.unescape(response.body.decode())


def test_the_page_shows_the_scorecard_and_each_question_beside_the_report_and_takes_new_ones(learned):  # noqa: F811
    store, built, _ = learned
    from admin import core2_answers_check as page

    base = f"/admin/clients/{ACCOUNT}/answers-check"
    assert "No questions yet" in _shown()
    total = _sql(built, SALES_2025.format(extra=""))
    response = _admin(page.answers_check_add, f"{base}/questions",
                      {"question": "What were  net sales?", "expected": f"${total / 1e6:.2f}M", "period": "2025",
                       "report": "Monthly sales pack"})
    assert response.status_code == 303 and response.headers["location"].endswith("?saved=added")
    wrong = _add(store, "What were net sales in 2025 in the old pack?", "$9.9M")
    assert "Not checked yet" in _shown()
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        response = _admin(page.answers_check_run, f"{base}/run", {})
        assert response.headers["location"].endswith("?saved=running")
        _wait(store)
    shown = _shown()
    assert "1 of 2 questions with a figure match the client's reports." in shown
    assert "What were net sales?" in shown and "Monthly sales pack · 2025" in shown
    assert "asked as “What were net sales, for 2025?”" in shown
    assert f"${total / 1e6:.2f}M (${total:,.0f})" in shown, "Solviz's figure as the report writes it, and in full"
    # Worst first: what needs an admin's eye heads the list.
    assert shown.index("in the old pack") < shown.index("What were net sales?</h3>")
    assert "Last checked" in shown and "because an admin asked" in shown
    # The report fixed, the one question asked again alone: the page says what it was.
    with store.get_db() as conn:
        conn.execute("UPDATE core2_test_question SET expected = ? WHERE id = ?", (f"${total / 1e6:.1f}M", wrong))
    with patch("core2.bootstrap.ai.workspace_planner", _Planner({"net sales": SALES})):
        _admin(page.answers_check_run, f"{base}/run", {"question": str(wrong)})
        _wait(store)
    shown = _shown()
    assert "was different" in shown and "2 of 2 questions with a figure match" in shown
    # The Learn page's setup section points here with the scorecard.
    assert "Answers check: 2 of 2 test questions with a figure match the client's reports." in _text(_learned_page())
    response = _admin(page.answers_check_delete, f"{base}/questions/{wrong}/delete", {}, question_id=wrong)
    assert response.headers["location"].endswith("?saved=removed")
    assert wrong not in [q["id"] for q in store.list_core2_test_questions(ACCOUNT, _db_id(store))]
    assert wrong not in store.core2_answer_results(ACCOUNT, _db_id(store)), "its results go with it"
    status = json.loads(_admin(page.answers_check_status, f"{base}/status", method="GET").body)
    assert status["status"] == "done" and status["total"] == 1


def test_before_a_learn_the_page_says_so_and_runs_nothing(workspace):  # noqa: F811
    store, _, _ = workspace
    from admin import core2_answers_check as page

    _add(store, "What were net sales in 2025?", "$3.09M")
    assert "has not learned this workspace's database yet" in _shown()
    response = _admin(page.answers_check_run, f"/admin/clients/{ACCOUNT}/answers-check/run", {})
    assert "Learn%20the%20database%20first" in response.headers["location"]
    assert store.latest_core2_answer_check(ACCOUNT, _db_id(store)) is None
    assert ac.due(ACCOUNT, _db_id(store)) is None


def test_the_page_is_a_data_stewards():
    from admin.access import STEWARD, rule_for

    assert rule_for(f"/clients/{ACCOUNT}/answers-check")[0] == STEWARD
    assert rule_for(f"/clients/{ACCOUNT}/answers-check/questions/3/delete")[0] == STEWARD
