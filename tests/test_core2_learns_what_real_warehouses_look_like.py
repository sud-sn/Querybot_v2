"""What real warehouses do that tidy synthetic ones do not, and what core2 must make of it.

Each case was first seen in a customer's warehouse and is rebuilt here from
invented rows: an invoice table whose requested-delivery date repeats once per
customer and day (it is still invoices, not balances), item names that repeat
(two items called the same must never be merged), columns that only describe the
load or are never filled (they are not groupings), abbreviations whose meaning
depends on the column's type, a description that matches its own table's codes,
a stamp named for when rows were entered (a load time, batched or not), and a list of
people with a birth date (never what the list is dated by) and 9999-12-31 for "not left",
a department number that is unique beside a name only by chance (a link, not a line number),
and numbers of events on a month-end table (NUM_OF_DLV: deliveries, added up over months).
"""

from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd
import pytest

from core2.bootstrap import names
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import compile_query
from core2.model.schema import SemanticModel
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2026, 6, 15)


def _load(con: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    con.register("_frame", frame)
    con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _frame')
    con.unregister("_frame")


def _warehouse() -> tuple[DuckDBWarehouse, SemanticModel]:
    rng = np.random.default_rng(11)
    con = duckdb.connect()
    days = pd.date_range("2025-01-01", "2026-06-12", freq="D")
    _load(con, "CAL_DMS", pd.DataFrame({
        "DT_DMS_KEY": (days.year * 10000 + days.month * 100 + days.day).astype("int64"),
        "DMS_DT": days.date, "YR": days.year, "MTH_NO": days.month, "DAY_OF_WK_NO": days.dayofweek + 1}))
    codes = [f"R{i:02d}" for i in range(38)]
    _load(con, "RGN_DMS", pd.DataFrame({
        # Descriptions mostly left equal to the codes: they match their own table's codes.
        "RGN_DMS_KEY": np.arange(1, 39), "RGN_CD": codes,
        "RGN_DSC": [c if i % 20 else f"Region {c}" for i, c in enumerate(codes)]}))
    _load(con, "CUS_DMS", pd.DataFrame({
        "CUS_DMS_KEY": np.arange(1, 61), "CUS_CD": [f"C{1000 + i}" for i in range(60)],
        "RGN_DMS_KEY": rng.integers(1, 39, 60),
        "CUS_NM": [f"Customer {chr(65 + i % 26)}{i}" for i in range(60)],
        "AZ_LST_UPD_USR": "loader", "AZ_EXT_ID": rng.choice(["A", "B"], 60), "CUS_FR_NM": None,
        "CUS_TYP_DSC": rng.choice(["Retail trade", "Wholesale trade", "Public sector"], 60)}))
    item_names = [f"Part {i:03d}" for i in range(40)]
    item_names[7] = item_names[3] = "Hose Clamp Set"          # two items, one name
    _load(con, "ITM_DMS", pd.DataFrame({
        "ITM_DMS_KEY": np.arange(1, 41), "ITM_CD": [f"IT-{5000 + i}" for i in range(40)], "ITM_NM": item_names,
        "ABC_CLS_CD": rng.choice(["A", "B", "C"], 40), "ITM_GRS_WT": np.round(rng.uniform(0.1, 90, 40), 3),
        "PRU_GRP_DMS_KEY": rng.integers(1, 48, 40)}))
    # At most one line per customer and requested day, on about two days in three:
    # the requested date recurs per customer the way a daily snapshot's date does.
    customer, day = np.nonzero(rng.random((60, len(days))) < 0.65)
    lines = pd.DataFrame({"CUS_DMS_KEY": customer + 1, "RQS": days[day]})
    n = len(lines)
    invoiced = lines["RQS"] - pd.to_timedelta(rng.integers(0, 10, n), unit="D")
    lines = lines.assign(
        CUS_ORD_IVC_FCT_KEY=np.arange(1, n + 1),
        IVC_NO=[f"INV-{700000 + i // 3}" for i in range(n)],
        IVC_LIN_NO=np.arange(n) % 3 + 1,
        ITM_DMS_KEY=rng.integers(1, 41, n),
        CUS_IVC_DT_DMS_KEY=(invoiced.dt.year * 10000 + invoiced.dt.month * 100 + invoiced.dt.day).astype("int64"),
        RQS_DLY_DT_DMS_KEY=(lines["RQS"].dt.year * 10000 + lines["RQS"].dt.month * 100
                            + lines["RQS"].dt.day).astype("int64"),
        IVC_QTY=rng.integers(1, 40, n).astype(float),
        IVC_LIN_AMT=np.round(rng.uniform(10, 900, n), 2),
    ).drop(columns=["RQS"])
    _load(con, "CUS_ORD_IVC_FCT", lines)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    return warehouse, model


_cache: dict[str, tuple[DuckDBWarehouse, SemanticModel]] = {}


def _built() -> tuple[DuckDBWarehouse, SemanticModel]:
    if "w" not in _cache:
        _cache["w"] = _warehouse()
    return _cache["w"]


def _table(model: SemanticModel, name: str):
    return next(t for t in model.tables.values() if t.name == name)


def test_invoices_with_a_repeating_secondary_date_are_transactions():
    _, model = _built()
    invoices = _table(model, "CUS_ORD_IVC_FCT")
    assert invoices.kind == "fact"
    amount = next(m for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(".ivc_lin_amt"))
    assert amount.additivity == "additive"
    default = model.date_roles[invoices.default_date or ""]
    assert model.columns[default.column].name == "CUS_IVC_DT_DMS_KEY"


def test_names_that_repeat_still_name_members_and_never_merge_two():
    warehouse, model = _built()
    item = next(e for e in model.entities.values() if model.tables[e.table].name == "ITM_DMS")
    assert model.columns[item.label_column or ""].name == "ITM_NM"
    plan = Plan.model_validate({"intent": "breakdown", "measures": [next(
        m.slug for m in model.measures.values() if (getattr(m.expr, "column", None) or "").endswith(".ivc_qty"))],
        "group_by": [item.slug]})
    logical = resolve(plan, model, Context(today=TODAY))
    result = warehouse.query(compile_query(logical, model, "duckdb").sql)
    label = next(i for i, c in enumerate(result.columns) if c == logical.groups[0].name)
    clamps = [r for r in result.rows if r[label] == "Hose Clamp Set"]
    assert len(clamps) == 2, "two items called the same were merged into one row"
    assert any(g.kind == "member_code" for g in logical.groups)


def test_columns_about_the_load_or_never_filled_are_not_groupings():
    _, model = _built()
    offered = {model.columns[a.column].name for a in model.attributes.values()}
    for column in ("AZ_LST_UPD_USR", "AZ_EXT_ID", "CUS_FR_NM", "PRU_GRP_DMS_KEY", "ITM_GRS_WT"):
        assert column not in offered, column
    assert {"CUS_NM", "ITM_NM", "CUS_TYP_DSC"} <= offered


def test_abbreviations_read_by_the_column_they_are_in():
    assert names.readable("WHS_DSC", "text") == "Warehouse description"
    assert names.readable("IVC_DSC_AMT", "decimal") == "Invoice discount amount"
    assert names.readable("ABC_CLS_CD", "text") == "ABC class code"
    assert names.readable("CLS_DT", "date") == "Closed date"
    _, model = _built()
    column = next(c for c in model.columns.values() if c.name == "CUS_TYP_DSC")
    assert column.business_name == "Customer type description"


def test_a_column_never_links_to_its_own_table_unless_it_names_a_hierarchy():
    _, model = _built()
    assert not [j for j in model.joins.values() if j.from_table == j.to_table]


@pytest.mark.parametrize("runs", [True, False])
def test_a_stamp_named_for_when_rows_were_entered_is_a_load_time(runs):
    # When the row was entered, not when the business happened: its name says so. Written by
    # posting runs at 06:00, 12:30 and 18:00, the times prove it too, and the evidence says why.
    rng = np.random.default_rng(5)
    con = duckdb.connect()
    days = pd.bdate_range("2025-01-01", "2026-06-12")
    posted = days.repeat(rng.poisson(4, len(days)))
    n = len(posted)
    minutes = rng.choice([6 * 60, 12 * 60 + 30, 18 * 60], n) if runs else rng.integers(7 * 60, 19 * 60, n)
    _load(con, "gl_lines", pd.DataFrame({
        "line_id": np.arange(1, n + 1), "posting_date": posted.date,
        "entered_at": posted + pd.to_timedelta(rng.integers(0, 3, n), unit="D") + pd.to_timedelta(minutes, unit="m"),
        "amount": np.round(rng.uniform(10, 900, n), 2)}))
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    entered = next(r for r in model.date_roles.values() if model.columns[r.column].name == "entered_at")
    assert entered.kind == "audit" and not entered.is_default
    clustered = any(e.kind == "load_clustering" and "3 times of day" in e.detail for e in entered.evidence)
    assert clustered == runs
    assert any(q.kind == "load_timestamp" and q.object == entered.column for q in model.quality) == runs


def test_a_person_is_listed_by_when_they_joined_never_by_their_birth_date():
    # Every date of the employee list is filled; the birth date comes first. It is a date in
    # a person's life, not something the business did: the hire date dates the list. An
    # employee still working has the termination date 9999-12-31, a placeholder, not a date.
    rng = np.random.default_rng(3)
    con = duckdb.connect()
    n = 400
    hired = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), unit="D")
    left = hired + pd.to_timedelta(rng.integers(30, 1000, n), unit="D")          # by 2026
    _load(con, "staff", pd.DataFrame({
        "staff_id": np.arange(1, n + 1), "staff_name": [f"Person {i}" for i in range(n)],
        "birth_date": (hired - pd.to_timedelta(rng.integers(8000, 20000, n), unit="D")).date,
        "hire_date": hired.date,
        "termination_date": np.where(rng.random(n) < 0.7, pd.Timestamp("9999-12-31").date(), left.date)}))
    _load(con, "timesheets", pd.DataFrame({
        "staff_id": rng.integers(1, n + 1, 3000), "work_date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 500, 3000), unit="D")).date, "hours": rng.integers(1, 10, 3000)}))
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    staff = _table(model, "staff")
    assert model.columns[model.date_roles[staff.default_date or ""].column].name == "hire_date"
    left_role = next(r for r in model.date_roles.values() if model.columns[r.column].name == "termination_date")
    assert left_role.last is not None and left_role.last.year < 2030 and 0.6 < left_role.placeholder_share < 0.8
    assert any(q.kind == "placeholder_dates" and q.object == left_role.column for q in model.quality)


def test_a_link_unique_beside_a_name_by_chance_is_a_link_not_a_line_number():
    # A few names repeat, each time in another department: name and department are unique
    # together, the way an order number and its line number are. A line number starts again
    # at 1 in each order; a department number does not, and stays a link to the departments.
    rng = np.random.default_rng(8)
    con = duckdb.connect()
    _load(con, "dept", pd.DataFrame({"dept_id": np.arange(1, 11), "dept_name": [f"Team {c}" for c in "ABCDEFGHIJ"]}))
    n = 500
    names_ = [f"Person {i}" for i in range(n - 20)] + [f"Person {i}" for i in range(20)]
    first = rng.integers(2, 11, n - 20)
    dept = np.concatenate([first, (first[:20] % 10) + 1])     # a repeated name is in another department
    _load(con, "people", pd.DataFrame({"person_id": np.arange(1, n + 1), "person_name": names_, "dept_id": dept,
                                       "hired_on": (pd.Timestamp("2020-01-01") + pd.to_timedelta(
                                           rng.integers(0, 2000, n), unit="D")).date}))
    lines = pd.DataFrame({"order_no": np.repeat([f"SO-{i}" for i in range(300)], 3), "line_no": np.tile([1, 2, 3], 300),
                          "person_id": rng.integers(1, n + 1, 900), "amount": np.round(rng.uniform(5, 500, 900), 2),
                          "order_date": np.repeat((pd.Timestamp("2025-01-01") + pd.to_timedelta(
                              rng.integers(0, 300, 300), unit="D")).date, 3)})
    lines.insert(0, "line_id", np.arange(1, 901))
    _load(con, "order_lines", lines)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    links = {(model.columns[j.from_columns[0]].name, model.tables[j.to_table].name) for j in model.joins.values()
             if j.trust != "rejected"}
    assert ("dept_id", "dept") in links
    people = _table(model, "people")
    assert people.kind == "dimension"
    line_no = next(c for c in model.columns.values() if c.name == "line_no")
    assert not any(m for m in model.measures.values() if getattr(m.expr, "column", None) == line_no.key)


def test_a_number_of_events_on_a_month_end_table_is_counted_and_added_up():
    # A month-end balance table also counts the month's deliveries and stock counts
    # (NUM_OF_DLV, NUM_OF_PHY_INV): numbers of events, added up over months, named as
    # what was counted. The stock on hand beside them is a level, taken at month end.
    rng = np.random.default_rng(4)
    con = duckdb.connect()
    _load(con, "ITM_DMS", pd.DataFrame({"ITM_DMS_KEY": np.arange(1, 41), "ITM_NM": [f"Part {i}" for i in range(40)]}))
    months = [y * 100 + m for y in (2025, 2026) for m in range(1, 13) if y * 100 + m <= 202608]
    rows = [(p, i, int(rng.integers(0, 500)), int(rng.integers(0, 4)), int(rng.integers(0, 2)))
            for p in months for i in range(1, 41)]
    _load(con, "ITM_BAL_PRD_FCT", pd.DataFrame(rows, columns=["PRD_KEY", "ITM_DMS_KEY", "ON_HND_QTY", "NUM_OF_DLV",
                                                              "NUM_OF_PHY_INV"]))
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t", options=BuildOptions(workers=1))
    by_column = {model.columns[m.expr.column].name: m for m in model.measures.values()
                 if getattr(m.expr, "column", None)}
    assert _table(model, "ITM_BAL_PRD_FCT").kind == "snapshot"
    assert by_column["ON_HND_QTY"].additivity == "semi_additive"
    assert (by_column["NUM_OF_DLV"].business_name, by_column["NUM_OF_DLV"].additivity) == (
        "Number of deliveries", "additive")
    assert (by_column["NUM_OF_PHY_INV"].business_name, by_column["NUM_OF_PHY_INV"].additivity) == (
        "Number of physical inventories", "additive")
