"""A calendar table's own facts are time fields: holidays, business days and fiscal weeks; and so is the hour.

Day of week, month and quarter are worked out from any date. A calendar table can also say what no date
arithmetic gives: which days are holidays (and their names), which are business days, which fiscal week a day is
in. Learn finds them from the data (a business day is "no" on every weekend day and "yes" on most weekdays; a
holiday and a fiscal week by their names), and a question can group by them or keep only some days ("on business
days", "excluding holidays"); a date with a time can be grouped by the hour. A date whose calendar has none of
them says so instead of guessing.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest
import sqlglot

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve, time_fields
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 12, 31)
HOLIDAYS = {dt.date(2025, 1, 1): "New Year's Day", dt.date(2025, 7, 4): "Independence Day",
            dt.date(2025, 12, 25): "Christmas Day", dt.date(2026, 1, 1): "New Year's Day",
            dt.date(2026, 5, 25): "Memorial Day", dt.date(2026, 7, 3): "Independence Day (observed)",
            dt.date(2026, 12, 25): "Christmas Day"}


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE dim_date (date_key INTEGER PRIMARY KEY, cal_date DATE, is_weekend INTEGER, "
                "holiday_flag VARCHAR, holiday_desc VARCHAR, workday_ind INTEGER, fiscal_week_num INTEGER)")
    rows, day = [], dt.date(2025, 1, 1)
    while day <= dt.date(2026, 12, 31):
        weekend = day.isoweekday() >= 6
        fiscal_start = dt.date(day.year if day.month >= 4 else day.year - 1, 4, 1)
        rows.append((int(day.strftime("%Y%m%d")), day, int(weekend), "Y" if day in HOLIDAYS else "N",
                     HOLIDAYS.get(day, ""), int(not weekend and day not in HOLIDAYS),
                     (day - fiscal_start).days // 7 + 1))
        day += dt.timedelta(days=1)
    con.executemany("INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute("CREATE TABLE stores (store_id INTEGER PRIMARY KEY, store_name VARCHAR)")
    con.execute("INSERT INTO stores VALUES (1, 'Harbour Store'), (2, 'Hill Store'), (3, 'Market Store')")
    con.execute("""CREATE TABLE sales AS
        SELECT row_number() OVER () AS sale_id, d.date_key, 1 + (i % 3) AS store_id,
               CAST(10 + (i * 7 + extract(day FROM d.cal_date)) % 90 AS DECIMAL(10, 2)) AS amount,
               d.cal_date + INTERVAL (8 + (i * 5) % 11) HOUR + INTERVAL (i * 13 % 60) MINUTE AS sold_at
        FROM dim_date d, range(0, 6) t(i)""")
    return con


@pytest.fixture(scope="module")
def shop():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, schema="main"), options=BuildOptions(workers=1))
    return con, model


def _ask(shop, plan: dict) -> dict:
    con, model = shop
    answers = [json.dumps({"kind": "query", **plan})]
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answers.pop(0),
                        index=MemberIndex(), today=TODAY)
    payload = answer_question("q", services, Session())
    assert "export_rows" in payload, payload.get("answer")
    return payload


def _amount(model) -> str:
    return next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", "") or "").endswith(".amount")
                and getattr(m.expr, "agg", "") == "sum")


def test_learn_finds_the_calendars_own_days_from_the_data(shop):
    _, model = shop
    calendar = next(iter(model.calendars.values()))
    held = {a: model.columns[c].name for a, c in calendar.attributes.items()}
    assert {k: held[k] for k in ("business_day", "holiday", "holiday_name", "fiscal_week")} == {
        "business_day": "workday_ind", "holiday": "holiday_flag", "holiday_name": "holiday_desc",
        "fiscal_week": "fiscal_week_num"}
    assert calendar.flags == {"business_day": 1, "holiday": "Y"}
    assert set(time_fields(model)) >= {"time:business_day", "time:holiday", "time:holiday_name",
                                       "time:fiscal_week", "time:hour_of_day"}


def test_only_business_days_and_without_holidays(shop):
    con, model = shop
    amount = _amount(model)
    year = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}}
    (business,), = con.execute("SELECT SUM(amount) FROM sales s JOIN dim_date d USING (date_key) WHERE "
                               "d.workday_ind = 1 AND d.cal_date >= DATE '2026-01-01'").fetchall()
    got = _ask(shop, {"measures": [amount], "time": year,
                      "filters": [{"field": "time:business_day", "op": "eq", "values": [True]}]})
    assert float(got["export_rows"][0][amount]) == pytest.approx(float(business))
    assert "for business days" in got["answer"]["headline"]
    (other,), = con.execute("SELECT SUM(amount) FROM sales s JOIN dim_date d USING (date_key) WHERE "
                            "d.holiday_flag = 'N' AND d.cal_date >= DATE '2026-01-01'").fetchall()
    got = _ask(shop, {"measures": [amount], "time": year,
                      "filters": [{"field": "time:holiday", "op": "ne", "values": [True]}]})
    assert float(got["export_rows"][0][amount]) == pytest.approx(float(other))
    assert "for days that are not holidays" in got["answer"]["headline"]


def test_grouped_by_holiday_name_fiscal_week_and_hour(shop):
    con, model = shop
    amount = _amount(model)
    expected = dict(con.execute("SELECT holiday_desc, SUM(amount) FROM sales s JOIN dim_date d USING (date_key) "
                                "WHERE holiday_desc <> '' AND d.cal_date >= DATE '2026-01-01' GROUP BY 1").fetchall())
    got = _ask(shop, {"measures": [amount], "group_by": ["time:holiday_name"],
                      "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}}})
    shown = {r["holiday_name"]: r[amount] for r in got["export_rows"]}
    assert "Not a holiday" in shown
    assert {k: float(v) for k, v in shown.items() if k != "Not a holiday"} == pytest.approx(
        {k: float(v) for k, v in expected.items()})

    weeks = dict(con.execute("SELECT fiscal_week_num, SUM(amount) FROM sales s JOIN dim_date d USING (date_key) "
                             "WHERE d.cal_date >= DATE '2026-04-01' AND d.cal_date < DATE '2026-07-01' "
                             "GROUP BY 1").fetchall())
    got = _ask(shop, {"measures": [amount], "group_by": ["time:fiscal_week"],
                      "time": {"window": {"kind": "between", "start": "2026-04-01", "end": "2026-06-30"}}})
    assert "Fiscal week 1" in [r["fiscal_week"] for r in got["export_rows"]]
    assert {int(r["fiscal_week"].split()[-1]): float(r[amount]) for r in got["export_rows"]} == pytest.approx(
        {int(k): float(v) for k, v in weeks.items()})

    timed = next(r.slug for r in model.date_roles.values() if model.columns[r.column].name == "sold_at")
    hours = dict(con.execute("SELECT hour(sold_at), SUM(amount) FROM sales WHERE sold_at >= TIMESTAMP '2026-06-01' "
                             "AND sold_at < TIMESTAMP '2026-07-01' GROUP BY 1").fetchall())
    got = _ask(shop, {"measures": [amount], "group_by": ["time:hour_of_day"], "time": {
        "date": timed, "window": {"kind": "between", "start": "2026-06-01", "end": "2026-06-30"}}})
    assert sorted(r["hour_of_day"] for r in got["export_rows"]) == ["08:00", "11:00", "12:00", "13:00", "17:00",
                                                                     "18:00"]
    assert {int(r["hour_of_day"][:2]): float(r[amount]) for r in got["export_rows"]} == pytest.approx(
        {int(k): float(v) for k, v in hours.items()})
    shown = _ask(shop, {"measures": [amount], "time": {"date": timed, "window": {
        "kind": "between", "start": "2026-06-01", "end": "2026-06-30"}},
        "filters": [{"field": "time:hour_of_day", "op": "between", "values": [9, 12]}]})
    assert float(shown["export_rows"][0][amount]) == pytest.approx(sum(float(v) for k, v in hours.items() if 9 <= k <= 12))


@pytest.mark.parametrize("dialect", ["snowflake", "tsql", "oracle"])
def test_every_warehouse_reads_the_calendars_own_days(shop, dialect):
    con, model = shop
    plan = Plan.model_validate({"kind": "query", "measures": [_amount(model)], "group_by": ["time:holiday_name"],
                                "filters": [{"field": "time:business_day", "op": "eq", "values": [False]}]})
    sql = compile_query(resolve(plan, model, Context(today=TODAY)), model, dialect).sql
    sqlglot.parse_one(sql, read=dialect)
    assert "holiday_desc" in sql.lower() and "workday_ind" in sql.lower()


def test_a_date_without_those_days_says_so():
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    _, model = learn(domains.build("retail"), "descriptive")
    assert "time:holiday" not in time_fields(model)
    for field in ("time:holiday", "time:business_day"):
        with pytest.raises(ResolveError) as refused:
            resolve(Plan.model_validate({"kind": "query", "measures": ["net_amount"], "group_by": [field]}), model,
                    Context(today=TODAY))
        assert "does not say which days are" in refused.value.message
    with pytest.raises(ResolveError) as refused:
        resolve(Plan.model_validate({"kind": "query", "measures": ["net_amount"], "group_by": ["time:hour_of_day"]}),
                model, Context(today=TODAY))
    assert "has no time of day" in refused.value.message
