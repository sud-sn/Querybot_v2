""""Why did it change?" and "what comes next?", answered from the data.

A why question compares two periods and breaks the change down by the groupings
that carry it: the change is the warehouse's (checked against hand-written SQL),
a grouping's members' changes add up to the total, the grouping named first is
the one where the fewest members carry the most of the change, an even change is
said to be spread out, a ratio is never decomposed as if it added up, a "why"
about a daily series compares weeks rather than two single days (issue G2), and a
current period the data covers only in part is compared with the same part of
the one before (A7).

A forecast projects complete periods only, dates its periods like the past ones,
names a partial period and an outlier instead of fitting them (F2), recovers a
known trend and season, and says when there is too little history.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.answer import drivers as D
from core2.answer.forecast import answer_forecast, project, unusual
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse, Guarded
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _plan(**kw) -> Plan:
    return Plan.model_validate({"kind": "query", **kw})


def _why(con, model, plan: Plan, **ctx) -> dict:
    return D.answer_drivers("why?", plan, model=model, warehouse=Guarded(DuckDBWarehouse(con)),
                            ctx=Context(today=TODAY, **ctx))


def _net(con, start: str, end: str, by: str = "") -> dict:
    """Net amount by order date in [start, end), by a grouping expression (hand-written)."""
    rows = con.execute(f"""
        SELECT {by or "'all'"} AS k, SUM(o.net_amount)
        FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key
        LEFT JOIN stores s ON s.store_id = o.store_id LEFT JOIN regions r ON r.region_id = s.region_id
        WHERE c.full_date >= DATE '{start}' AND c.full_date < DATE '{end}' GROUP BY 1""").fetchall()
    return {k: float(v or 0) for k, v in rows}


def test_a_why_question_gives_the_change_and_the_groupings_that_carry_it(retail):
    con, model = retail
    plan = _plan(intent="drivers", measures=["net_amount"],
                 time={"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}})
    payload = _why(con, model, plan)
    now, before = _net(con, "2026-03-01", "2026-04-01")["all"], _net(con, "2026-02-01", "2026-03-01")["all"]
    change = payload["drivers"]["change"]
    assert change == pytest.approx(now - before)
    headline = payload["answer"]["headline"]
    assert headline.startswith("Net amount rose") and "February 2026" in headline and "March 2026" in headline

    # Region: the members' changes are the warehouse's and add up to the total (no row lost on the way).
    by_now = _net(con, "2026-03-01", "2026-04-01", "r.region_name")
    by_before = _net(con, "2026-02-01", "2026-03-01", "r.region_name")
    regions = [r for r in payload["data"]["rows"] if r["grouping"] == "Region"]
    assert {r["member"] for r in regions} == {k or "Unknown" for k in {*by_now, *by_before}}
    for r in regions:
        assert r["change"] == pytest.approx(by_now.get(r["member"], 0) - by_before.get(r["member"], 0))
    assert sum(r["change"] for r in regions) == pytest.approx(change)

    # The grouping named first needs the fewest members: one region carries more than the whole rise.
    first = payload["drivers"]["groupings"][0]
    leader = max(by_now, key=lambda k: by_now.get(k, 0) - by_before.get(k, 0))
    assert first["label"].startswith("Region") and first["leaders"][0]["member"] == leader
    assert f"By region: {leader}" in headline and "more than the whole rise" in headline
    json.dumps(payload)


def test_a_change_spread_evenly_is_not_pinned_on_a_member():
    con = duckdb.connect()
    stores = pd.DataFrame({"store_id": range(1, 7), "store_name": [f"Store {i}" for i in range(1, 7)]})
    rng = np.random.default_rng(1)
    january = [(store, int(day), round(float(rng.uniform(20, 80)) * store, 2))
               for store in range(1, 7) for day in rng.integers(1, 29, 40)]
    rows = [{"sale_id": i + 1, "store_id": store, "sale_date": dt.date(2026, month, day), "amount": amount * lift}
            for month, lift in ((1, 1.0), (2, 1.1))
            for i, (store, day, amount) in enumerate(january, start=len(january) * (month - 1))]
    for name, frame in (("stores", stores), ("sales", pd.DataFrame(rows))):
        con.register("_f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _f")
        con.unregister("_f")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    amount = next(m for m in model.measures.values() if (getattr(m.expr, "column", "") or "").endswith("amount"))
    assert amount.additivity == "additive"            # sales, not a balance: every store up 10%
    plan = _plan(intent="drivers", measures=[amount.slug],
                 time={"window": {"kind": "between", "start": "2026-02-01", "end": "2026-02-28"}})
    payload = D.answer_drivers("why?", plan, model=model, warehouse=Guarded(warehouse),
                               ctx=Context(today=dt.date(2026, 3, 5)))
    headline = payload["answer"]["headline"]
    assert "rose" in headline and "(+10.0%)" in headline
    assert "spread out" in headline and "By store" not in headline, headline
    # The largest stores still carry most of the rise: only the unchanged mix keeps them from being "the cause".
    assert payload["drivers"]["groupings"][0]["explained"] >= 0.5


def test_a_ratio_is_never_decomposed_as_if_it_added_up(retail):
    con, model = retail
    plan = _plan(intent="drivers", measures=["margin_percent"],
                 time={"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}})
    payload = _why(con, model, plan)
    assert "share_of_change" not in payload["data"]["headers"]
    if "By " in payload["answer"]["headline"]:
        assert "do not add up" in payload["answer"]["headline"]


def test_a_why_about_a_daily_series_compares_weeks_not_two_days(retail):
    con, model = retail
    plan = _plan(intent="drivers", measures=["net_amount"],
                 time={"grain": "day", "window": {"kind": "between", "start": "2026-05-25", "end": "2026-06-07"}})
    payload = _why(con, model, plan)
    notes = " ".join(payload["trust"]["date_context"])
    assert "the last 7 days of the series against the 7 before" in notes
    now, before = _net(con, "2026-06-01", "2026-06-08")["all"], _net(con, "2026-05-25", "2026-06-01")["all"]
    assert payload["drivers"]["change"] == pytest.approx(now - before)


def test_a_partly_covered_month_is_compared_with_the_same_days_of_the_one_before(retail):
    con, model = retail      # the data runs to 14 June 2026
    plan = _plan(intent="drivers", measures=["net_amount"], time={"window": {"kind": "this", "unit": "month"}})
    payload = _why(con, model, plan)
    now, before = _net(con, "2026-06-01", "2026-06-15")["all"], _net(con, "2026-05-01", "2026-05-15")["all"]
    assert payload["drivers"]["change"] == pytest.approx(now - before)
    assert any("compared over their first 14 days" in n for n in payload["trust"]["date_context"])


def test_a_grouping_the_reader_may_not_use_is_not_checked(retail):
    con, model = retail
    plan = _plan(intent="drivers", measures=["net_amount"],
                 time={"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}})
    allowed = {k for k in model.tables if not k.endswith("regions")}
    payload = _why(con, model, plan, allowed_tables=allowed)
    assert all(not g["label"].startswith("Region") for g in payload["drivers"]["groupings"])
    assert "By region" not in payload["answer"]["headline"] and payload["drivers"]["change"]


def test_a_forecast_projects_complete_months_and_names_what_it_left_out(retail):
    con, model = retail
    plan = _plan(intent="forecast", measures=["net_amount"], time={"grain": "month"}, forecast={"periods": 3})
    payload = answer_forecast("forecast?", plan, model=model, warehouse=Guarded(DuckDBWarehouse(con)),
                              ctx=Context(today=TODAY))
    rows = payload["data"]["rows"] if len(payload["data"]["rows"]) < 200 else []
    future = [r for r in payload["chart"]["rows"] if r["is_forecast"]]
    assert [r["period"] for r in future] == ["2026-06-01", "2026-07-01", "2026-08-01"]
    assert all(r["forecast_low"] <= r["value"] <= r["forecast_high"] for r in future)
    headline = payload["answer"]["headline"]
    assert "in May 2026, the last complete month" in headline and "Forecast for Jun 2026 to Aug 2026" in headline
    caveats = " ".join(payload["coverage_caveats"])
    assert "Jun 2026 is only partly covered" in caveats and "Aug 2025" in caveats   # the planted spike
    # The spike did not make every August a peak: next August is in line with the months around it.
    by_period = {r["period"]: r["value"] for r in future}
    assert by_period["2026-08-01"] < 1.3 * by_period["2026-07-01"]
    assert payload["chart"]["chart_type"] == "forecast" and payload["forecast_meta"]["interval"] == 0.95
    assert not rows or rows[-1]["actual"] is None
    json.dumps(payload)


def test_a_known_trend_and_season_are_recovered():
    season = [10, 8, 12, 15, 20, 25, 30, 28, 22, 16, 12, 30]
    series = [1000 + 5 * i + season[i % 12] for i in range(36)]
    fit = project(series, 12, 6)
    truth = [1000 + 5 * i + season[i % 12] for i in range(36, 42)]
    assert fit.method == "trend and seasonality"
    for got, want in zip(fit.values, truth):
        assert got == pytest.approx(want, rel=0.01)


def test_one_spike_is_named_and_does_not_shape_the_projection():
    series = [100.0 + 2 * i for i in range(18)]
    series[10] = 900.0
    assert unusual(series, None) == [10]
    fit = project(series, None, 2, skip=[10])
    assert fit.values[0] == pytest.approx(136, rel=0.02)


def test_too_little_history_is_said_not_forecast():
    con = duckdb.connect()
    frame = pd.DataFrame({"sale_id": range(1, 41), "amount": [10.0] * 40,
                          "sale_date": [dt.date(2026, 1 + i // 10, 1 + i % 10) for i in range(40)]})
    con.register("_f", frame)
    con.execute("CREATE TABLE sales AS SELECT * FROM _f")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    amount = next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", "") or "").endswith("amount"))
    payload = answer_forecast("forecast?", _plan(intent="forecast", measures=[amount], time={"grain": "month"}),
                              model=model, warehouse=Guarded(warehouse), ctx=Context(today=dt.date(2026, 6, 1)))
    assert payload["answer"]["headline"].startswith("Not enough history to forecast")
    assert payload["chart"] is None


def test_a_measure_never_below_zero_is_not_projected_below_it():
    fit = project([100.0 - 15 * i for i in range(7)], None, 6)
    assert min(fit.values + fit.low) >= 0


class _Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def test_the_service_answers_why_and_forecast_plans(retail):
    con, model = retail
    why = {"kind": "query", "intent": "drivers", "measures": ["net_amount"],
           "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}}
    ahead = {"kind": "query", "intent": "forecast", "measures": ["net_amount"], "time": {"grain": "month"},
             "forecast": {"periods": 2}}
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=_Recorded(json.dumps(why),
                        json.dumps(ahead)), index=MemberIndex(), today=TODAY)
    session = Session()
    explained = answer_question("why did net sales change in March?", services, session)
    assert explained["drivers"]["groupings"] and explained["plan"]["intent"] == "drivers"
    projected = answer_question("and the next two months?", services, session)
    assert projected["chart"]["chart_type"] == "forecast" and projected["plan"]["intent"] == "forecast"
    assert len(session.turns) == 2


def test_a_day_range_against_last_year_is_the_same_days_a_year_earlier():
    from core2.resolve.time import Range, shift

    r = shift(Range(dt.date(2026, 6, 3), dt.date(2026, 6, 11)), "same_period_last_year")
    assert (r.start, r.end) == (dt.date(2025, 6, 3), dt.date(2025, 6, 11))
    r = shift(Range(dt.date(2024, 2, 29), dt.date(2024, 3, 1)), "same_period_last_year")
    assert (r.start, r.end) == (dt.date(2023, 2, 28), dt.date(2023, 3, 1))
