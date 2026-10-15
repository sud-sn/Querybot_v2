"""Members by what they did over any period, as one figure or by week, month, quarter or year, equal to SQL written
by hand.

"How many returning customers in the last 2 months" was answered with one figure per month, and "returning
customers by week" was refused. Over a period P, with "the period before" the same length just before it:

* active: with activity in P;
* new: their first activity ever in P;
* returning: with activity in P and some before it (new and returning make up the active);
* won back: returning, with none in the period before (back after a break);
* lapsed: with activity in the period before, none in P;
* repeat: activity on 2 or more days in P; one-time: on one day only;
* loyal: activity in every month (or week, quarter) of P;
* churn rate: the lapsed over those active the period before.

Asked of the invented retail warehouse through the whole answer path (a scripted plan, as the planner writes it).
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
LAST_2_MONTHS = {"kind": "last", "unit": "month", "n": 2}           # April and May 2026
DAYS = """SELECT DISTINCT o.customer_id AS k, d.full_date AS day FROM order_lines o
          JOIN calendar d ON o.order_date_key = d.date_key
          WHERE o.customer_id IN (SELECT customer_id FROM customers)"""


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _ask(learned, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    grain = (plan.get("time") or {}).get("grain")
    # The reader's words: a series is asked for by its period ("by month"); with none, one figure for the period.
    payload = answer_question(f"customers by {grain}" if grain else "q", services, Session())
    assert payload.get("type") == "assistant_response" and "export_rows" in payload, payload.get("answer")
    return payload


def _members(learned, start: dt.date, end: dt.date, unit: str = "month") -> dict:
    """Each figure over [start, end), the period before being the same length just before it, by hand."""
    before = _before(start, end)
    whole = len(learned[0].con.execute(f"""SELECT DISTINCT date_trunc('{unit}', full_date) FROM calendar
                                            WHERE full_date >= DATE '{start}' AND full_date < DATE '{end}'""").fetchall())
    row = learned[0].con.execute(f"""
        WITH d AS ({DAYS} AND d.full_date < DATE '{end}'),
        m AS (SELECT k, MIN(day) AS first,
                     MAX(CASE WHEN day >= DATE '{start}' THEN 1 ELSE 0 END) AS now,
                     MAX(CASE WHEN day >= DATE '{before}' AND day < DATE '{start}' THEN 1 ELSE 0 END) AS was,
                     COUNT(DISTINCT CASE WHEN day >= DATE '{start}' THEN day END) AS days,
                     COUNT(DISTINCT CASE WHEN day >= DATE '{start}' THEN date_trunc('{unit}', day) END) AS periods
              FROM d GROUP BY k)
        SELECT SUM(now), SUM(CASE WHEN first >= DATE '{start}' THEN 1 ELSE 0 END),
               SUM(CASE WHEN now = 1 AND first < DATE '{start}' THEN 1 ELSE 0 END),
               SUM(CASE WHEN now = 1 AND first < DATE '{start}' AND was = 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN was = 1 AND now = 0 THEN 1 ELSE 0 END), SUM(was),
               SUM(CASE WHEN days >= 2 THEN 1 ELSE 0 END), SUM(CASE WHEN days = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN periods = {whole} THEN 1 ELSE 0 END)
        FROM m""").fetchone()
    active, new, returning, won_back, lapsed, was, repeat, one_time, loyal = (int(v or 0) for v in row)
    return {"active": active, "new": new, "returning": returning, "won_back": won_back, "lapsed": lapsed,
            "churn_rate": 100.0 * lapsed / was if was else None, "repeat": repeat, "one_time": one_time,
            "loyal": loyal}


def _before(start: dt.date, end: dt.date) -> dt.date:
    """Where the period before starts: whole months stay whole months, other periods go back as many days."""
    if start.day == 1 and end.day == 1:
        index = start.year * 12 + start.month - 1 - ((end.year - start.year) * 12 + end.month - start.month)
        return dt.date(index // 12, index % 12 + 1, 1)
    return start - (end - start)


@pytest.mark.parametrize("status", ["active", "new", "returning", "won_back", "lapsed", "churn_rate", "repeat",
                                    "one_time"])
def test_each_as_one_figure_for_a_period(retail, status):
    expected = _members(retail, dt.date(2026, 4, 1), dt.date(2026, 6, 1))
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": status},
                        "time": {"window": LAST_2_MONTHS}})
    rows = got["export_rows"]
    assert len(rows) == 1, rows
    value = next(v for k, v in rows[0].items() if k == status or k.startswith(status))
    assert float(value) == pytest.approx(expected[status]), status


def test_new_and_returning_make_up_the_active(retail):
    expected = _members(retail, dt.date(2026, 4, 1), dt.date(2026, 6, 1))
    assert expected["new"] + expected["returning"] == expected["active"]
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "all"},
                        "time": {"window": LAST_2_MONTHS}})
    row = got["export_rows"][0]
    assert {k: int(row[k]) for k in ("active", "new", "returning", "won_back", "lapsed")} == {
        k: expected[k] for k in ("active", "new", "returning", "won_back", "lapsed")}
    said = got["definition"]
    assert "returning: they had orders before it too" in said
    assert "won back: returning, with none in the 2 months before" in said


def test_returning_over_two_months_from_today(retail):
    # Rolling: 16 April to 15 June, the period before the 61 days just before it.
    expected = _members(retail, dt.date(2026, 4, 16), dt.date(2026, 6, 16))
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "returning"},
                        "time": {"window": {"kind": "rolling", "unit": "month", "n": 2}}})
    assert len(got["export_rows"]) == 1
    assert float(got["kpi"]["value"]) == pytest.approx(expected["returning"])


def test_loyal_members_were_active_every_month(retail):
    expected = _members(retail, dt.date(2025, 12, 1), dt.date(2026, 6, 1))
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "loyal"},
                        "time": {"window": {"kind": "last", "unit": "month", "n": 6}}})
    assert float(got["kpi"]["value"]) == pytest.approx(expected["loyal"])
    assert "in every month of it" in got["definition"]


def test_by_week(retail):
    weeks = retail[0].con.execute(f"""
        WITH s AS (SELECT k, p, LAG(p) OVER (PARTITION BY k ORDER BY p) AS prev, MIN(p) OVER (PARTITION BY k) AS first
                   FROM (SELECT DISTINCT k, date_trunc('week', day)::DATE AS p FROM ({DAYS}) WHERE day < DATE '2026-06-01'))
        SELECT p, COUNT(*), SUM(CASE WHEN p = first THEN 1 ELSE 0 END), SUM(CASE WHEN p > first THEN 1 ELSE 0 END),
               SUM(CASE WHEN p > first AND (prev IS NULL OR prev <> CAST(p - INTERVAL 7 DAY AS DATE)) THEN 1 ELSE 0 END)
        FROM s WHERE p >= DATE '2026-03-02' AND p < DATE '2026-06-01' GROUP BY p ORDER BY p""").fetchall()
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "all"},
                        "time": {"grain": "week", "window": {"kind": "between", "start": "2026-03-02",
                                                             "end": "2026-05-31"}}})
    shown = {str(r["period"])[:10]: r for r in got["export_rows"]}
    assert len(weeks) == 13 and len(shown) == 13
    for week, active, new, returning, won_back in weeks:
        row = shown[str(week)[:10]]
        assert (int(row["active"]), int(row["new"]), int(row["returning"]), int(row["won_back"])) == (
            active, new, returning, won_back), week


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle", "postgres"])
@pytest.mark.parametrize("time", [{"window": LAST_2_MONTHS}, {"grain": "week", "window": LAST_2_MONTHS}])
def test_the_sql_reads_on_every_warehouse(retail, dialect, time):
    built, model = retail
    plan = Plan.model_validate({"kind": "query", "lifecycle": {"of": "customer", "measure": "number_of_orders",
                                                               "status": "all"}, "time": time})
    compiled = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect)
    sqlglot.parse_one(compiled.sql, read=dialect)


def _asked(learned, question: str, plan: dict) -> dict:
    built, model = learned
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question(question, services, Session())
    assert payload.get("type") == "assistant_response", payload.get("answer")
    return payload


def test_a_period_with_no_series_asked_is_one_figure(retail):
    # The probe: one figure per month for "how many returning customers in the last 2 months".
    expected = _members(retail, dt.date(2026, 4, 1), dt.date(2026, 6, 1))
    got = _asked(retail, "How many returning customers did we have in the last 2 months?",
                 {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "returning"},
                  "time": {"grain": "month", "window": LAST_2_MONTHS}})
    assert len(got["export_rows"]) == 1
    assert float(got["kpi"]["value"]) == pytest.approx(expected["returning"])


def test_repeat_customers_by_month(retail):
    want = dict(retail[0].con.execute(f"""
        SELECT p, SUM(CASE WHEN n >= 2 THEN 1 ELSE 0 END) FROM (
            SELECT k, date_trunc('month', day)::DATE AS p, COUNT(DISTINCT day) AS n FROM ({DAYS}) GROUP BY k, p)
        WHERE p >= DATE '2026-01-01' AND p < DATE '2026-06-01' GROUP BY p""").fetchall())
    got = _asked(retail, "Repeat customers by month this year",
                 {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "repeat"},
                  "time": {"grain": "month", "window": {"kind": "between", "start": "2026-01-01",
                                                        "end": "2026-05-31"}}})
    shown = {str(r["period"])[:10]: int(r["repeat"]) for r in got["export_rows"]}
    assert shown == {str(k)[:10]: int(v) for k, v in want.items()}


def test_all_of_them_said_in_one_sentence(retail):
    e = _members(retail, dt.date(2026, 4, 1), dt.date(2026, 6, 1))
    got = _ask(retail, {"lifecycle": {"of": "customer", "measure": "number_of_orders", "status": "all"},
                        "time": {"window": LAST_2_MONTHS}})
    assert got["answer"]["headline"] == (
        f"{e['active']:,} customers had orders from Apr 2026 to May 2026: {e['new']:,} new and {e['returning']:,} "
        f"returning ({e['won_back']:,} of them won back); {e['lapsed']:,} lapsed.")


@pytest.mark.parametrize("question", ["Customers active in the last 2 months, by store",
                                      "How many customers bought from each store in the last 2 months?",
                                      "Clients actifs sur les 2 derniers mois par magasin"])
def test_active_members_are_counted_on_their_activity(retail, question):
    # The probe asked which of a customer's own dates to count by; "active" is what they did: orders in the period.
    want = dict(retail[0].con.execute("""
        SELECT s.store_name, COUNT(DISTINCT o.customer_id) FROM order_lines o
        JOIN calendar d ON o.order_date_key = d.date_key JOIN stores s ON o.store_id = s.store_id
        WHERE d.full_date >= DATE '2026-04-01' AND d.full_date < DATE '2026-06-01'
          AND o.customer_id IN (SELECT customer_id FROM customers)
        GROUP BY s.store_name""").fetchall())
    got = _asked(retail, question, {"measures": ["number_of_customers"], "group_by": ["store.name"],
                                    "time": {"window": LAST_2_MONTHS}})
    rows = got["export_rows"]
    name = next(k for k in rows[0] if "store" in k or "name" in k)
    value = next(k for k in rows[0] if k != name)
    assert {r[name]: int(r[value]) for r in rows} == {k: int(v) for k, v in want.items()}
