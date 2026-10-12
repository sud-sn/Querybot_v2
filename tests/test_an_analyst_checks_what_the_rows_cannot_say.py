"""The note says what an analyst checks next, from up to three small queries more.

"Net amount in Q1 2026: $543K" said nothing of whether that was good. The note now says how far it got against its
target, how it compares with the same quarter a year before, and which store moved it most since the quarter
before; a month that stands out as a one-off is explained by the store that made it. Each figure is checked here
against hand-written SQL. The checks are asked in the chat only (a dashboard tile keeps its own line), a check that
fails costs its fact and never the answer, and the chart can be seen without the one-off.

The invented retail shop of the test domains.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2 import service
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
Y2025 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
SALES = ("SELECT {cols} FROM order_lines l JOIN calendar c ON l.order_date_key = c.date_key "
         "JOIN stores s ON l.store_id = s.store_id WHERE c.full_date BETWEEN '{start}' AND '{end}' {group}")


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _ask(retail, plan: dict) -> dict:
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(
        {"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _facts(payload: dict) -> dict[str, dict]:
    return {f["kind"]: f for f in (payload.get("reading") or {}).get("facts") or []}


def _sum(con, start: str, end: str) -> float:
    return float(con.execute(SALES.format(cols="SUM(l.net_amount)", start=start, end=end, group="")).fetchone()[0])


def test_one_figure_is_read_against_its_target_last_year_and_what_moved_it(retail):
    con, _ = retail
    facts = _facts(_ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": Q1}}))
    assert {"target", "last_year", "moved_by"} <= set(facts), set(facts)
    now = _sum(con, "2026-01-01", "2026-03-31")
    target = float(con.execute("SELECT SUM(target_amount) FROM sales_targets WHERE period_key BETWEEN 202601 "
                               "AND 202603").fetchone()[0])
    got, aim, share = facts["target"]["figures"]
    assert got == pytest.approx(now, rel=1e-6) and aim == pytest.approx(target, rel=1e-6)
    assert share == pytest.approx(now / target, rel=1e-3)
    assert facts["target"]["section"] == ("watch" if now < target else "stands_out")
    before = _sum(con, "2025-01-01", "2025-03-31")
    assert facts["last_year"]["figures"][2] == pytest.approx(before, rel=1e-6)
    assert (" is up " if now > before else " is down ") in facts["last_year"]["en"]
    moves = {name: float(cur) - float(prev) for name, cur, prev in con.execute(
        "WITH a AS (" + SALES.format(cols="s.store_name AS n, SUM(l.net_amount) AS v", start="2026-01-01",
                                     end="2026-03-31", group="GROUP BY 1") + "), b AS ("
        + SALES.format(cols="s.store_name AS n, SUM(l.net_amount) AS v", start="2025-10-01", end="2025-12-31",
                       group="GROUP BY 1") + ") SELECT a.n, a.v, b.v FROM a JOIN b USING (n)").fetchall()}
    net = sum(moves.values())
    leader = max(moves, key=lambda n: moves[n] if net > 0 else -moves[n])
    assert facts["moved_by"]["members"] == [leader]
    assert facts["moved_by"]["figures"][0] == pytest.approx(moves[leader], rel=1e-6)
    for f in (facts["target"], facts["last_year"], facts["moved_by"]):
        assert f["fr"] and f["how"] and f["how_fr"] and f["query"].lstrip().upper().startswith(("SELECT", "WITH"))


def test_a_one_off_month_is_explained_by_the_member_that_made_it(retail):
    con, _ = retail
    payload = _ask(retail, {"intent": "trend", "measures": ["net_amount"], "time": {"window": Y2025, "grain": "month"}})
    one_off = payload["reading"]["one_offs"][0]
    start = one_off["period"][:10]
    end = (dt.date.fromisoformat(start).replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    stores = con.execute(SALES.format(cols="s.store_name, SUM(l.net_amount)", start=start, end=end.isoformat(),
                                      group="GROUP BY 1 ORDER BY 2 DESC")).fetchall()
    fact = _facts(payload)["one_off_why"]
    total = sum(float(v) for _, v in stores)
    assert fact["members"] == [stores[0][0]]
    assert fact["figures"] == pytest.approx([float(stores[0][1]), total, float(stores[0][1]) / total], rel=1e-3)
    assert stores[0][0] in fact["en"] and stores[0][0] in fact["fr"] and "Aug" not in fact["fr"]


def test_a_dashboard_tile_asks_nothing_more(retail):
    con, model = retail
    asked: list[str] = []

    class Counting(DuckDBWarehouse):
        def query(self, sql, max_rows=None):
            asked.append(sql)
            return super().query(sql, max_rows=max_rows)

    services = Services(model=model, warehouse=Counting(con), complete=None, index=MemberIndex(), today=TODAY)
    ctx = service.Context(today=TODAY, allowed_tables=None, max_rows=service.MAX_ROWS)
    plan = service.parse_plan({"kind": "query", "intent": "trend", "measures": ["net_amount"],
                               "time": {"window": Y2025, "grain": "month"}})
    payload = service._compute("q", plan, services, ctx, question_id="", started=0.0)
    assert not {"one_off_why", "last_year", "target", "moved_by"} & set(_facts(payload))
    assert len(asked) == 1


def test_a_check_that_fails_costs_its_fact_and_never_the_answer(retail, monkeypatch):
    from core2.answer import extra

    def broken(*args, **kwargs):
        raise RuntimeError("the warehouse went away")

    monkeypatch.setattr(extra, "_against_target", broken)
    monkeypatch.setattr(extra, "_last_year", broken)
    payload = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": Q1}})
    assert payload["data"]["rows"] and "moved_by" in _facts(payload)
    assert not {"target", "last_year"} & set(_facts(payload))


def test_the_chart_is_seen_without_the_one_off_and_says_what_the_rest_add_up_to():
    from tests.chat_js import run

    msg = {"reading": {"one_offs": [{"period": "2025-08-01", "label": "Aug 2025", "value": 60.0, "typical": 10.0}]},
           "chart": {"x_key": "period", "y_keys": ["net_amount"], "column_formats": {"net_amount": "number"},
                     "rows": [{"period": "2025-07-01", "net_amount": 10}, {"period": "2025-08-01", "net_amount": 60},
                              {"period": "2025-09-01", "net_amount": 14}]}}
    money = {"reading": {"one_offs": [{"period": "2025-08-01", "label": "Aug 2025", "label_fr": "août 2025"}]},
             "chart": {"x_key": "period", "y_keys": ["net_amount"], "column_formats": {"net_amount": "currency"},
                       "rows": [{"period": "2025-07-01", "net_amount": 1_000_000.25},
                                {"period": "2025-08-01", "net_amount": 9_000_000},
                                {"period": "2025-09-01", "net_amount": 985_915.72}]}}
    functions = ["function _oneOffs(msg)", "function _oneOffToggleHtml(msg)", "function _inLang(entry, key)",
                 "function _withoutOneOffs(msg, payload, hide)", "function _shortFigure(value, fmt)",
                 "function _shortAmount(value, fmt, spec = {})", "function escHtml(str)",
                 "function _formatDisplayValue(value, fmt, spec = {}, column = '')",
                 "function _parseDisplayDate(value)", "function _parseDisplayNumber(v)",
                 "function _parseServerNumber(v)", "function _qbMoney(body, symbol)"]
    out = run(f"const m = {json.dumps(msg)}, d = {json.dumps(money)}; JSON.stringify([_oneOffToggleHtml(m), "
              "_withoutOneOffs(m, m.chart, true), _withoutOneOffs(m, m.chart, false), "
              "_withoutOneOffs(d, d.chart, true)]);",
              functions=functions, consts=["_QB_CURRENCY_SYMBOL"])
    button, hidden, shown, dollars = out
    assert 'data-one-offs="hide"' in button and "Without the one-off" in button
    assert [r["net_amount"] for r in hidden["payload"]["rows"]] == [10, None, 14]
    assert hidden["said"].startswith("Without Aug 2025: 24") and " in all, 12" in hidden["said"]   # 60 is left out
    assert [r["net_amount"] for r in shown["payload"]["rows"]] == [10, 60, 14]
    # Said short, as the note says its figures: never "$1,986,915.97".
    assert dollars["said"] == "Without Aug 2025: $1.99M in all, $993K on average.", dollars["said"]
    french = run(f"const d = {json.dumps(money)}; window.QB_LANG = 'fr'; "
                 "JSON.stringify(_withoutOneOffs(d, d.chart, true));",
                 lang="fr", functions=functions, consts=["_QB_CURRENCY_SYMBOL"])
    assert french["said"].startswith("Sans août 2025 : 1,99"), french["said"]


def test_a_cause_stands_only_where_it_names_who_moved_the_figure(retail):
    from core2.answer import notes

    payload = _ask(retail, {"intent": "trend", "measures": ["net_amount"], "time": {"window": Y2025, "grain": "month"}})
    leader = _facts(payload)["one_off_why"]["members"][0]
    said = {"stands_out": [], "watch": [],
            "why": [f"The August spike was driven by {leader}.",
                    "Sales were steady because the stores kept their prices."]}     # no member a fact names
    kept = notes.checked(said, payload["reading"])
    assert kept["why"] == [f"The August spike was driven by {leader}."]


def test_a_sentence_says_one_fact_never_two_put_together():
    from core2.answer import notes

    reading = {"facts": [
        {"kind": "one_off", "section": "stands_out", "en": "Aug 2025 was a one-off at $1.42M.", "figures": [1_420_000]},
        {"kind": "target", "section": "stands_out", "en": "2025 reached $3.09M against a target of $2.35M.",
         "figures": [3_090_000, 2_350_000]}]}
    said = {"stands_out": ["August 2025 was a one-off at $1.42M, against a target of $2.35M.",   # two facts in one
                           "The year reached $3.09M against its target of $2.35M in 2025."], "why": [], "watch": []}
    assert notes.checked(said, reading)["stands_out"] == [said["stands_out"][1]]
