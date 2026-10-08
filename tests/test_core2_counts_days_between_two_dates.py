"""Days between two dates of the same rows: "average days from order to invoice".

A plan names a duration: two date slugs of one table, the earlier event first,
and how to add the days up (average, longest, shortest, total). It is a measure
like any other (by a grouping, as a trend, ranked by its name), or it keeps
rows ("invoiced more than 10 days after ordering"). On every warehouse:

* days are whole calendar days, computed from the dates however they are
  stored (a calendar key, a DATE, a timestamp, a yyyymmdd number);
* a row missing either date, or holding a placeholder (1900-01-01), is left
  out of the days, and still counted by every other measure of the answer;
* two dates of different tables, the same date twice, or a date kept by month
  are refused with the reason.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import numpy as np
import pandas as pd
import pytest
import sqlglot

from core2.answer.builder import fmt
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex, mask
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y25 = {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}
TO_SHIP = {"name": "Days from order to ship", "start": "order_date", "end": "ship_date", "agg": "avg"}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _run(con, model, plan: dict, dialect: str = "duckdb"):
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    compiled = compile_query(logical, model, dialect)
    return logical, compiled, (DuckDBWarehouse(con).query(compiled.sql).rows if dialect == "duckdb" else None)


DAYS = ("FROM order_lines o JOIN calendar c ON c.date_key = o.order_date_key "
        "LEFT JOIN calendar s ON s.date_key = o.ship_date_key "
        "WHERE c.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'")
REAL = "s.full_date >= DATE '1901-01-01' AND s.full_date < DATE '9000-01-01'"


def test_the_average_days_between_two_calendar_dates(retail):
    con, model = retail
    logical, _, rows = _run(con, model, {"intent": "value", "durations": [TO_SHIP], "time": {"window": Y25}})
    want = con.execute(f"SELECT AVG(CASE WHEN {REAL} THEN DATE_DIFF('day', c.full_date, s.full_date) END) "
                       f"{DAYS}").fetchone()[0]
    assert rows[0][0] == pytest.approx(want)
    assert any(n.startswith("Days from order to ship: days from order date to ship date") for n in logical.notes)


@pytest.mark.parametrize("agg, sql", [("max", "MAX"), ("min", "MIN"), ("sum", "SUM")])
def test_the_longest_shortest_and_total(retail, agg, sql):
    con, model = retail
    _, _, rows = _run(con, model, {"intent": "value", "durations": [{**TO_SHIP, "agg": agg}], "time": {"window": Y25}})
    want = con.execute(f"SELECT {sql}(CASE WHEN {REAL} THEN DATE_DIFF('day', c.full_date, s.full_date) END) "
                       f"{DAYS}").fetchone()[0]
    assert rows[0][0] == pytest.approx(want)


def test_by_a_grouping_ranked_by_its_name(retail):
    con, model = retail
    _, _, rows = _run(con, model, {"intent": "rank", "durations": [TO_SHIP], "group_by": ["store"],
                                   "sort": [{"by": "Days from order to ship", "desc": True}], "limit": 3,
                                   "time": {"window": Y25}})
    want = con.execute(f"SELECT st.store_name, AVG(CASE WHEN {REAL} THEN DATE_DIFF('day', c.full_date, s.full_date) END) "
                       f"{DAYS.replace('WHERE', 'JOIN stores st ON st.store_id = o.store_id WHERE')} "
                       "GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 3").fetchall()
    assert [(r[0], pytest.approx(r[-1])) for r in rows] == [(w[0], pytest.approx(w[1])) for w in want]


def test_rows_kept_by_their_days(retail):
    con, model = retail
    lines = next(m.slug for m in model.measures.values() if m.table.endswith("order_lines") and m.kind == "count")
    logical, _, rows = _run(con, model, {"intent": "count", "measures": [lines],
                                         "durations": [{**TO_SHIP, "measure": False}],
                                         "filters": [{"field": "days from order to ship", "op": "gt", "values": [3]}],
                                         "time": {"window": Y25}})
    want = con.execute(f"SELECT COUNT(*) {DAYS} AND {REAL} AND DATE_DIFF('day', c.full_date, s.full_date) > 3"
                       ).fetchone()[0]
    assert rows[0][0] == want and len(rows[0]) == 1, "a duration that only keeps rows is not shown"
    assert any("whose days from order to ship is above 3 days" in n for n in logical.notes), logical.notes


def test_another_measure_beside_the_days_keeps_every_row():
    built, model = learn(domains.build("retail"), "descriptive")
    con = built.con
    con.execute("UPDATE order_lines SET ship_date_key = NULL WHERE order_line_id % 7 = 0")   # not shipped yet
    _, _, alone = _run(con, model, {"intent": "value", "measures": ["net_amount"], "time": {"window": Y25}})
    _, _, beside = _run(con, model, {"intent": "value", "measures": ["net_amount"], "durations": [TO_SHIP],
                                     "time": {"window": Y25}})
    want = con.execute(f"SELECT AVG(CASE WHEN {REAL} THEN DATE_DIFF('day', c.full_date, s.full_date) END) "
                       f"{DAYS}").fetchone()[0]
    assert beside[0][0] == pytest.approx(alone[0][0]), "lines not shipped yet still count in net sales"
    assert beside[0][1] == pytest.approx(want)


@pytest.mark.parametrize("dialect, spelled", [("tsql", "DATEDIFF(DAY"), ("snowflake", "DATEDIFF(DAY"),
                                             ("oracle", "TRUNC(")])
def test_every_warehouse_spells_it(retail, dialect, spelled):
    con, model = retail
    _, compiled, _ = _run(con, model, {"intent": "breakdown", "durations": [TO_SHIP], "group_by": ["store"],
                                       "filters": [{"field": "Days from order to ship", "op": "lte", "values": [10]}],
                                       "time": {"window": Y25}}, dialect)
    assert spelled in compiled.sql.upper()
    sqlglot.parse_one(compiled.sql, read=dialect)


# ── dates stored as DATEs, some missing, some placeholders ─────────────────


@pytest.fixture(scope="module")
def plain():
    rng = np.random.default_rng(5)
    n = 400
    order = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    invoice = pd.Series(order + pd.to_timedelta(rng.integers(1, 20, n), unit="D"))
    invoice[rng.random(n) < 0.1] = pd.NaT                          # not invoiced yet
    invoice[rng.random(n) < 0.03] = pd.Timestamp("1900-01-01")     # a placeholder
    frame = pd.DataFrame({"order_id": np.arange(1, n + 1), "order_date": order.date,
                          "invoice_date": pd.to_datetime(invoice).dt.date,
                          "amount": np.round(rng.uniform(10, 500, n), 2)})
    con = duckdb.connect()
    con.register("_f", frame)
    con.execute("CREATE TABLE orders AS SELECT order_id, CAST(order_date AS DATE) AS order_date, "
                "CAST(invoice_date AS DATE) AS invoice_date, amount FROM _f")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    return con, model


TO_INVOICE = {"name": "Days to invoice", "start": "order_date", "end": "invoice_date", "agg": "avg"}


def test_missing_and_placeholder_dates_are_left_out_of_the_days_only(plain):
    con, model = plain
    _, _, rows = _run(con, model, {"intent": "value", "measures": ["amount"], "durations": [TO_INVOICE]})
    want_days = con.execute("SELECT AVG(DATE_DIFF('day', order_date, invoice_date)) FROM orders "
                            "WHERE invoice_date >= DATE '1901-01-01'").fetchone()[0]
    want_amount = con.execute("SELECT SUM(amount) FROM orders").fetchone()[0]
    assert rows[0][0] == pytest.approx(float(want_amount)) and rows[0][1] == pytest.approx(want_days)
    assert 0 < rows[0][1] < 20, "a 1900 placeholder would make the average negative"


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        self.stable = stable
        return self.answers.pop(0)


def test_the_answer_says_days(plain):
    con, model = plain
    plan = {"kind": "query", "intent": "value", "measures": ["amount"], "durations": [TO_INVOICE]}
    ai = Recorded(json.dumps(plan))
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=ai, index=MemberIndex(), today=TODAY)
    answer = answer_question("total amount and average days to invoice", services, Session())
    assert " days to invoice" in answer["answer"]["headline"] and "days days" not in answer["answer"]["headline"]
    assert "durations" in ai.stable and "Days between two" in ai.stable           # the rule and the schema
    assert '"durations": [{"name": "Days from order date to invoice date"' in ai.stable   # the worked example


def test_days_read_in_a_sentence():
    assert fmt(9.5, "days") == "9.5 days" and fmt(14, "days") == "14 days" and fmt(1, "days") == "1 day"
    assert fmt(0.04, "days") == "0.04 days", "a change of hours is not 0.0 days"


# ── refused, with the reason ───────────────────────────────────────────────


@pytest.mark.parametrize("duration, says", [
    ({"start": "order_date", "end": "return_date"}, "needs two dates of the same rows"),
    ({"start": "order_date", "end": "order_date"}, "needs two different dates"),
    ({"start": "order_date", "end": "shipped_on"}, "no date called shipped_on"),
])
def test_dates_that_cannot_make_a_duration_are_refused(retail, duration, says):
    con, model = retail
    with pytest.raises(ResolveError) as raised:
        _run(con, model, {"intent": "value", "durations": [{"name": "Days", **duration}]})
    assert says in raised.value.message


def test_a_date_kept_by_month_has_no_days(retail):
    con, model = retail
    monthly = model.model_copy(deep=True)
    next(r for r in monthly.date_roles.values() if r.slug == "ship_date").granularity = "month"
    with pytest.raises(ResolveError) as raised:
        _run(con, monthly, {"intent": "value", "durations": [TO_SHIP]})
    assert "is kept by month, so days cannot be counted from it" in raised.value.message


def test_days_in_a_filter_are_never_masked_as_member_names():
    from core2.plan.planner import _plan_shown

    plan = Plan.model_validate({"kind": "query", "measures": ["amount"],
                                "durations": [{**TO_INVOICE, "measure": False}],
                                "filters": [{"field": "Days to invoice", "op": "gt", "values": [10]},
                                            {"field": "customer.name", "op": "eq", "values": ["Acme"]}]})
    shown = json.loads(_plan_shown(plan, None, mask("q", [])))
    assert shown["filters"][0]["values"] == [10] and shown["filters"][1]["values"] != ["Acme"]


@pytest.mark.parametrize("plan, question", [
    ({"intent": "drivers", "durations": [TO_SHIP],
      "time": {"window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}},
     "why did the days from order to ship change in March?"),
    ({"intent": "forecast", "durations": [TO_SHIP], "time": {"grain": "month"}, "forecast": {"periods": 3}},
     "forecast the days from order to ship for the next 3 months"),
])
def test_why_and_forecast_take_a_duration_as_their_measure(retail, plan, question):
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=Recorded(json.dumps({"kind": "query", **plan})), index=MemberIndex(), today=TODAY)
    answer = answer_question(question, services, Session())
    assert not answer.get("unsupported") and "needs the measure" not in answer["answer"]["headline"], answer["answer"]
    headline = answer["answer"]["headline"]
    assert headline.startswith("Days from order to ship") and " 0.0 days" not in headline, headline
    assert " in all" not in headline, "averages of days are never added up"
