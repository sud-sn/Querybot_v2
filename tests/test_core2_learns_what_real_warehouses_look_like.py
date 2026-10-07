"""What real warehouses do that tidy synthetic ones do not, and what core2 must make of it.

Each case was first seen in a customer's warehouse and is rebuilt here from
invented rows: an invoice table whose requested-delivery date repeats once per
customer and day (it is still invoices, not balances), item names that repeat
(two items called the same must never be merged), columns that only describe the
load or are never filled (they are not groupings), abbreviations whose meaning
depends on the column's type, and a description that matches its own table's codes.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np
import pandas as pd

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
