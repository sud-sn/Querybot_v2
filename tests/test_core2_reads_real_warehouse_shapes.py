"""Shapes met on real warehouses, learned and answered right.

* A monthly table keyed YYYYMM may hold a row for a whole year (month 00, 202600).
  It is not a month: never added to its months, even in an all-time total.
* A series that stops for years and starts again is forecast from the stretch
  since it started again, and the answer says so; a long gap is not zeros, and a
  series over it names the stretch with no data.
* "Why did it change?" does not check a grouping most rows cannot reach (a link
  that matches few rows): its answer would be "Unknown".
* Today's metrics written as formulas (SUM(ON_HAND_QTY * ITEM_COST)) come over:
  checked, then compiled by the new core, in totals and in comparisons.
* A month whose data starts on the 2nd (a holiday on the 1st) is a whole month;
  a month still under way is not.
* NUM_OF_RCT is a number of receipts, added up (never a receipt number counted),
  and a forecast of whole numbers says whole numbers.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.imports import Legacy, decisions
from core2.model.overrides import apply_overrides
from core2.plan.values import MemberIndex
from core2.resolve.time import partial_periods
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 10, 7)
MONTHS = [(2022, m) for m in range(1, 8)] + [(2025, 10), (2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 9)]


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE warehouse_dim (warehouse_id INTEGER PRIMARY KEY, warehouse_name VARCHAR)")
    con.execute("INSERT INTO warehouse_dim VALUES (1, 'North Depot'), (2, 'South Depot'), (3, 'East Depot')")
    con.execute("CREATE TABLE item_dim (item_id INTEGER PRIMARY KEY, item_name VARCHAR, uom VARCHAR)")
    # Only items 1-8 of 20 are listed: the item link matches 40% of stock rows.
    con.execute("INSERT INTO item_dim SELECT i, 'Part ' || i, CASE WHEN i % 3 = 0 THEN 'FT' ELSE 'EA' END "
                "FROM range(1, 9) t(i)")
    con.execute("CREATE TABLE stock_month_fact (period_key INTEGER, item_id INTEGER, warehouse_id INTEGER, "
                "sld_qty DECIMAL(12,1), pch_qty DECIMAL(12,1), num_of_rct INTEGER, cur_on_hnd_qty DECIMAL(12,1))")
    rows = []
    for year, month in MONTHS:
        for item in range(1, 21):
            for wh in (1, 2, 3):
                sold = (item * 7 + wh * 3 + month * 5 + year) % 40 + (25 if (year, month) == (2026, 3) and wh == 2 else 0)
                rows.append((year * 100 + month, item, wh, sold, sold // 2 + 3, (item + month) % 3, 100 + item * wh))
    for year in (2022, 2026):     # a whole-year row per item and warehouse: month 00
        for item in range(1, 21):
            for wh in (1, 2, 3):
                total = sum(r[3] for r in rows if r[0] // 100 == year and r[1] == item and r[2] == wh)
                rows.append((year * 100, item, wh, total, 0, 0, None))
    con.executemany("INSERT INTO stock_month_fact VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute("CREATE TABLE stock_day_fact (snapshot_date DATE, item_id INTEGER, warehouse_id INTEGER, "
                "on_hnd_qty DECIMAL(12,1), alc_on_hnd_qty DECIMAL(12,1), itm_cst DECIMAL(10,2))")
    con.execute("""INSERT INTO stock_day_fact
                   SELECT d, i, w, 10 * i + w + day(d), i + w, 2.5 * i + w
                   FROM (VALUES (DATE '2026-06-26'), (DATE '2026-08-17'), (DATE '2026-09-03')) s(d),
                        range(1, 21) t(i), range(1, 4) u(w)""")
    return con


@pytest.fixture(scope="module")
def stock():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, schema="main"), options=BuildOptions(workers=1))
    return con, model


def _ask(con, model, plan: dict) -> dict:
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: json.dumps(
        {"kind": "query", **plan}), index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _measure(model, column: str) -> str:
    return next(m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(column))


def test_a_whole_year_row_is_never_added_to_its_months(stock):
    con, model = stock
    sold = _measure(model, ".sld_qty")
    every_month = con.execute("SELECT SUM(sld_qty) FROM stock_month_fact WHERE period_key % 100 <> 0").fetchone()[0]
    with_year_rows = con.execute("SELECT SUM(sld_qty) FROM stock_month_fact").fetchone()[0]
    assert with_year_rows > every_month
    payload = _ask(con, model, {"intent": "value", "measures": [sold]})
    assert payload["kpi"]["value"] == pytest.approx(float(every_month))
    assert any("rows for a whole year (month 00) are left out" in n for n in payload["trust"]["date_context"])
    in_2026 = con.execute("SELECT SUM(sld_qty) FROM stock_month_fact WHERE period_key BETWEEN 202601 AND 202612"
                          ).fetchone()[0]
    payload = _ask(con, model, {"intent": "value", "measures": [sold], "time": {
        "window": {"kind": "between", "start": "2026-01-01", "end": "2026-12-31"}}})
    assert payload["kpi"]["value"] == pytest.approx(float(in_2026))
    period = next(r for r in model.date_roles.values() if r.column.endswith(".period_key"))
    assert period.first == dt.date(2022, 1, 1) and period.whole_year_share > 0


def test_a_forecast_starts_after_a_long_gap(stock):
    con, model = stock
    payload = _ask(con, model, {"intent": "forecast", "measures": [_measure(model, ".sld_qty")],
                                "time": {"grain": "month"}, "forecast": {"periods": 3}})
    notes = payload["trust"]["date_context"]
    assert any(n.startswith("Only the data since Oct 2025 is used") for n in notes), notes
    assert not any("counted as zero" in n for n in notes)
    assert "of 11 months" in payload["answer"]["headline"]


def test_a_why_does_not_check_a_grouping_most_rows_cannot_reach(stock):
    con, model = stock
    payload = _ask(con, model, {"intent": "drivers", "measures": [_measure(model, ".sld_qty")], "time": {
        "window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}, "compare": {"kind": "previous_period"}}})
    checked = next(n for n in payload["trust"]["date_context"] if n.startswith("Groupings checked"))
    assert "warehouse" in checked.lower() and "item" not in checked.lower().replace("groupings checked", "")
    assert "South Depot" in payload["answer"]["headline"]          # the planted rise in March 2026


def _with_metrics(model, *metrics: dict):
    report = decisions(model, Legacy(metrics=[{"is_active": 1, "metric_status": "validated", "id": i,
                                               "formula_type": "expression", **m} for i, m in enumerate(metrics)]))
    copy = model.model_copy(deep=True)
    assert not apply_overrides(copy, [{"object_key": d.object_key, "field": d.field, "value": d.value}
                                      for d in report.decisions])
    return copy, report


def test_formula_metrics_come_over_and_answer_totals_and_comparisons(stock):
    con, model = stock
    applied, report = _with_metrics(
        model,
        {"name": "Inventory value", "base_table": "dbo.STOCK_DAY_FACT", "result_format": "currency",
         "sql_template": "SUM([ON_HND_QTY] * [ITM_CST])"},
        {"name": "Available quantity", "base_table": "dbo.STOCK_DAY_FACT", "result_format": "number",
         "sql_template": "SUM(ON_HND_QTY) - COALESCE(SUM(ALC_ON_HND_QTY), 0)"},
        {"name": "Units sold", "base_table": "dbo.STOCK_MONTH_FACT", "result_format": "number",
         "sql_template": "SUM(SLD_QTY)", "synonyms": "units sold, quantity sold"},
        {"name": "Number of receipts", "base_table": "dbo.STOCK_MONTH_FACT", "result_format": "number",
         "sql_template": "SUM(NUM_OF_RCT)"},
        {"name": "Odd one", "base_table": "dbo.STOCK_MONTH_FACT", "result_format": "number",
         "sql_template": "SUM(SLD_QTY) + (SELECT MAX(warehouse_id) FROM warehouse_dim)"},
    )
    by_name = {m.business_name: m for m in applied.measures.values()}
    value, available = by_name["Inventory value"], by_name["Available quantity"]
    assert value.additivity == "semi_additive" and available.additivity == "semi_additive"
    assert by_name["Number of receipts"].additivity == "additive"
    assert by_name["Units sold"].slug == _measure(model, ".sld_qty")             # matched, renamed: not a copy
    assert any("Odd one" in m and "SUBQUERY" in m for m in report.missed)

    latest = "snapshot_date = DATE '2026-09-03'"
    payload = _ask(con, applied, {"intent": "value", "measures": [value.slug]})
    want = con.execute(f"SELECT SUM(on_hnd_qty * itm_cst) FROM stock_day_fact WHERE {latest}").fetchone()[0]
    assert payload["kpi"]["value"] == pytest.approx(float(want))
    payload = _ask(con, applied, {"intent": "value", "measures": [available.slug]})
    want = con.execute(f"SELECT SUM(on_hnd_qty) - SUM(alc_on_hnd_qty) FROM stock_day_fact WHERE {latest}").fetchone()[0]
    assert payload["kpi"]["value"] == pytest.approx(float(want))

    payload = _ask(con, applied, {"intent": "value", "measures": [value.slug], "time": {
        "window": {"kind": "between", "start": "2026-09-03", "end": "2026-09-03"},
        "compare": {"kind": "window", "window": {"kind": "between", "start": "2026-08-17", "end": "2026-08-17"}}}})
    now, before = con.execute("SELECT SUM(CASE WHEN snapshot_date = DATE '2026-09-03' THEN on_hnd_qty * itm_cst END), "
                              "SUM(CASE WHEN snapshot_date = DATE '2026-08-17' THEN on_hnd_qty * itm_cst END) "
                              "FROM stock_day_fact").fetchone()
    row = payload["data"]["rows"][0]
    assert row[value.slug] == pytest.approx(float(now)) and row[f"{value.slug}_prior"] == pytest.approx(float(before))
    assert payload["answer"]["comparison"].endswith("against 17 Aug 2026")
    assert f"{'up' if now >= before else 'down'}" in payload["answer"]["comparison"]


@pytest.mark.parametrize("first, last, today, partial", [
    (dt.date(2025, 1, 2), dt.date(2026, 6, 30), dt.date(2026, 10, 7), []),              # a holiday on 1 January
    (dt.date(2025, 1, 10), dt.date(2026, 6, 30), dt.date(2026, 10, 7), [dt.date(2025, 1, 1)]),
    (dt.date(2025, 1, 1), dt.date(2026, 6, 29), dt.date(2026, 10, 7), []),              # June is over
    (dt.date(2025, 1, 1), dt.date(2026, 6, 29), dt.date(2026, 6, 30), [dt.date(2026, 6, 1)]),   # June under way
])
def test_a_month_is_whole_unless_its_data_really_stops_short(first, last, today, partial):
    starts = [dt.date(2025, m, 1) for m in range(1, 13)] + [dt.date(2026, m, 1) for m in range(1, 7)]
    assert partial_periods(starts, "month", first_data=first, last_data=last, today=today) == partial


def test_a_series_that_stops_for_years_names_the_stretch_it_has_no_data_for(stock):
    con, model = stock
    payload = _ask(con, model, {"intent": "trend", "measures": [_measure(model, ".sld_qty")], "time": {"grain": "month"}})
    assert "No data from Aug 2022 to Sep 2025: shown as a gap, not as zero." in payload["coverage_caveats"]
    months = {r["period"][:7] for r in payload["data"]["rows"]}
    assert "2022-07" in months and "2025-10" in months and not any("2023" in m for m in months)


def test_a_forecast_of_whole_numbers_kept_as_a_number_says_whole_numbers(stock):
    # Today's metric "Number of receipts" is formatted as a number: its history is whole
    # numbers, so the forecast is "about 47 a month", not 46.68.
    con, model = stock
    copy = model.model_copy(deep=True)
    receipts = next(m for m in copy.measures.values() if (getattr(m.expr, "column", None) or "").endswith(".num_of_rct"))
    receipts.format = "number"
    payload = _ask(con, copy, {"intent": "forecast", "measures": [receipts.slug], "time": {"grain": "month"},
                               "forecast": {"periods": 3}})
    headline = payload["answer"]["headline"]
    assert receipts.business_name == "Number of receipts"          # NUM_OF_RCT: a count to add up
    assert [m.expr.agg for m in copy.measures.values()                 # never also a receipt number counted
            if (getattr(m.expr, "column", None) or "").endswith(".num_of_rct")] == ["sum"]
    assert "Forecast for Sep 2026 to Nov 2026" in headline and not re.search(r"\d\.\d", headline), headline


def test_where_values_stay_out_of_prompts_a_formula_or_filter_value_does_too(stock):
    # A formula's text values and a measure filter's values are values like any other: a
    # workspace that keeps values out of the AI's prompt keeps these out too.
    from core2.model.schema import ColumnFilter
    from core2.plan.catalog import catalog_text

    con, model = stock
    applied, report = _with_metrics(model, {
        "name": "North stock", "base_table": "dbo.STOCK_DAY_FACT", "result_format": "number",
        "sql_template": "SUM(CASE WHEN CAST(WAREHOUSE_ID AS VARCHAR) = 'North Depot' THEN ON_HND_QTY END)"})
    assert not report.missed, report.missed
    sold = next(m for m in applied.measures.values() if (getattr(m.expr, "column", None) or "").endswith(".sld_qty"))
    period = next(c.key for c in applied.columns.values() if c.name == "warehouse_id" and c.table == sold.table)
    sold.expr = sold.expr.model_copy(update={"filters": [ColumnFilter(column=period, op="ne", values=["Depot 9"])]})
    kept = catalog_text(applied, values_allowed=True)
    assert "North Depot" in kept and "Depot 9" in kept
    out = catalog_text(applied, values_allowed=False)
    assert "North Depot" not in out and "Depot 9" not in out
    assert "the formula SUM(CASE WHEN CAST(" in out and "(a value)" in out
