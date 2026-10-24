"""An ERP's order header is the orders: counted, its amounts kept in the currency they are in, its copies set aside.

Found on an invented wholesaler's ERP reporting database in SQL Server 2022, learned through discovery and asked by a
reader with gpt-4o (the production shape of a header and its lines):

- The order header holds no figure of its own, only keys, codes, a document number and its dates; a summary of the
  open orders (a view) holds one row for each of its orders. Learn read the header as a bridge between tables: it
  had no "number of orders", and "# orders mtd" was answered with the number of open orders.
- Each line keeps its amount twice, in the order's currency (ExtAmtDoc) and in the company's (ExtAmtBase). The
  base amount was split by the order's currency code and shown as "USD 145.15M, CAD 12.39M", USD labelled CAD;
  the document amount, whose currency is on the header, was added up across currencies.
- A backup copy of the header (SalesOrderHeader_bak_20240115) and the load's staging copy of the lines
  (stg_SalesOrderLine_Load) were learned as tables of their own, under the same names as the real ones.
- Each line holds what it sold for and what it cost, and no gross profit was learned: "gp$ last qtr" was told
  "Your data does not hold gross profit data".
- Orders the application deletes stay in the table, marked IsDeleted = 1, and were counted in every total.

A small warehouse in DuckDB, invented data; figures checked against hand-written SQL.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

TODAY = dt.date(2025, 12, 31)
CAD = 0.73


def _warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("""CREATE TABLE DimDate AS SELECT CAST(strftime(d, '%Y%m%d') AS INTEGER) AS DateKey, CAST(d AS DATE) AS FullDate,
                   year(d) AS CalendarYear, month(d) AS CalendarMonth, strftime(d, '%Y-%m') AS YearMonth,
                   dayname(d) AS DayName FROM range(DATE '2025-01-01', DATE '2026-01-01', INTERVAL 1 DAY) t(d)""")
    con.execute("ALTER TABLE DimDate ADD PRIMARY KEY (DateKey)")
    con.execute("CREATE TABLE Branch (BranchCd VARCHAR PRIMARY KEY, BranchName VARCHAR, CurrencyCd VARCHAR)")
    con.execute("INSERT INTO Branch VALUES ('DAL01', 'Dallas Branch', 'USD'), ('ATL01', 'Atlanta Branch', 'USD'), "
                "('TOR01', 'Toronto Branch', 'CAD')")
    con.execute("CREATE TABLE Customer (CustAcctNo VARCHAR PRIMARY KEY, CustName VARCHAR)")
    con.execute("INSERT INTO Customer SELECT 'C' || (10000 + i), ['Adams', 'Baker', 'Carter', 'Dawson', 'Ellis'][1 + i % 5] "
                "|| ' Electric ' || i FROM range(1, 41) t(i)")
    con.execute("CREATE TABLE lkpOrderStatus (StatusCd VARCHAR PRIMARY KEY, StatusDesc VARCHAR)")
    con.execute("INSERT INTO lkpOrderStatus VALUES ('OP', 'Open'), ('CL', 'Closed'), ('CX', 'Cancelled')")
    con.execute("""CREATE TABLE SalesOrderHeader AS SELECT 4000000 + i AS SalesOrderID, 'SO-' || lpad(CAST(i AS VARCHAR), 7, '0')
                   AS SalesOrderNo, 'C' || (10001 + i % 40) AS CustAcctNo, ['DAL01', 'ATL01', 'TOR01'][1 + i % 3] AS BranchCd,
                   CAST(strftime(DATE '2025-01-01' + INTERVAL (i % 360) DAY, '%Y%m%d') AS INTEGER) AS OrderDateKey,
                   CAST(strftime(DATE '2025-01-01' + INTERVAL (i % 360 + 2) DAY, '%Y%m%d') AS INTEGER) AS ShipDateKey,
                   CASE WHEN i % 29 = 0 THEN 'CX' WHEN i % 360 > 350 THEN 'OP' ELSE 'CL' END AS OrderStatusCd,
                   CASE WHEN i % 3 = 2 THEN 'CAD' ELSE 'USD' END AS CurrencyCd, CASE WHEN i % 97 = 0 THEN 1 ELSE 0 END AS IsDeleted
                   FROM range(1, 721) t(i)""")
    con.execute("ALTER TABLE SalesOrderHeader ADD PRIMARY KEY (SalesOrderID)")
    con.execute("""CREATE TABLE SalesOrderLine AS SELECT 90000000 + 10 * h.SalesOrderID + n AS SalesOrderLineID, h.SalesOrderID,
                   n AS LineNo, 1 + (h.SalesOrderID * 7 + n) % 30 AS OrdQty,
                   ROUND((1 + (h.SalesOrderID * 7 + n) % 30) * (10 + n * 3.5) * CASE WHEN h.CurrencyCd = 'CAD' THEN 1.32 ELSE 1 END, 2) AS ExtAmtDoc,
                   ROUND((1 + (h.SalesOrderID * 7 + n) % 30) * (10 + n * 3.5) * CASE WHEN h.CurrencyCd = 'CAD' THEN 1.32 * {CAD} ELSE 1 END, 2)
                   AS ExtAmtBase, ROUND((1 + (h.SalesOrderID * 7 + n) % 30) * (10 + n * 3.5) * 0.74, 2) AS ExtCostBase,
                   CASE WHEN (h.SalesOrderID + n) % 41 = 0 THEN 'X' ELSE 'S' END AS LineStatus
                   FROM SalesOrderHeader h, range(1, 1 + 1 + h.SalesOrderID % 4) t(n)""".replace("{CAD}", str(CAD)))
    con.execute("ALTER TABLE SalesOrderLine ADD PRIMARY KEY (SalesOrderLineID)")
    # One invoice for each order, its amounts in the order's currency and in the company's, with the currency code.
    con.execute("""CREATE TABLE Invoice AS SELECT 7000000 + h.SalesOrderID - 4000000 AS InvoiceID, 'INV-' || h.SalesOrderID AS InvoiceNo,
                   h.SalesOrderID, h.ShipDateKey AS InvoiceDateKey, h.CurrencyCd, SUM(l.ExtAmtDoc) AS SubtotalAmtDoc,
                   SUM(l.ExtAmtBase) AS SubtotalAmtBase FROM SalesOrderHeader h JOIN SalesOrderLine l ON l.SalesOrderID = h.SalesOrderID
                   WHERE h.OrderStatusCd = 'CL' GROUP BY ALL""")
    con.execute("ALTER TABLE Invoice ADD PRIMARY KEY (InvoiceID)")
    # A summary of the open orders, one row for each of them (a view in SQL Server).
    con.execute("""CREATE TABLE vw_OpenOrders AS SELECT h.SalesOrderID, h.SalesOrderNo, h.OrderDateKey, SUM(l.ExtAmtBase) AS OpenAmtUSD
                   FROM SalesOrderHeader h JOIN SalesOrderLine l ON l.SalesOrderID = h.SalesOrderID WHERE h.OrderStatusCd = 'OP'
                   GROUP BY ALL""")
    # The copies production databases keep beside their tables.
    con.execute("CREATE TABLE SalesOrderHeader_bak_20240115 AS SELECT * FROM SalesOrderHeader WHERE SalesOrderID < 4000300")
    con.execute("CREATE TABLE stg_SalesOrderLine_Load AS SELECT *, 88123 AS LoadBatchID FROM SalesOrderLine WHERE SalesOrderID > 4000650")
    return con


@pytest.fixture(scope="module")
def erp():
    con = _warehouse()
    warehouse = DuckDBWarehouse(con)
    fks = [{"parent_table": "SalesOrderLine", "parent_col": "SalesOrderID", "ref_table": "SalesOrderHeader",
            "ref_col": "SalesOrderID", "constraint_name": "FK_line_order"},
           {"parent_table": "SalesOrderHeader", "parent_col": "OrderDateKey", "ref_table": "DimDate", "ref_col": "DateKey",
            "constraint_name": "FK_order_date"}]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=fks), client_id="eval-erp",
                        options=BuildOptions(workers=1))
    return con, model


def _table(model, name: str):
    return next(t for k, t in model.tables.items() if k.endswith("." + name.casefold()))


def _measure(model, table: str, column: str):
    return next(m for m in model.measures.values() if m.table.endswith("." + table.casefold())
                and str(getattr(m.expr, "column", "") or "").endswith("." + column.casefold()) and not m.hidden)


def test_the_order_header_is_the_orders_and_they_are_counted(erp):
    _, model = erp
    header = _table(model, "SalesOrderHeader")
    assert header.kind != "bridge", header.evidence
    counted = [m for m in model.measures.values() if m.table == header.key and getattr(m.expr, "agg", "") in ("count", "count_distinct")
               and not m.hidden]
    assert counted, "no number of orders"
    assert model.date_roles[counted[0].default_date].column.endswith(".orderdatekey")


def test_an_amount_in_the_companys_currency_is_one_currency(erp):
    _, model = erp
    for table, column in (("SalesOrderLine", "ExtAmtBase"), ("SalesOrderLine", "ExtCostBase"), ("Invoice", "SubtotalAmtBase")):
        base = _measure(model, table, column)
        assert base.unit_column is None, (column, base.unit_column)
    # The invoice's own amount in the order's currency is kept apart by its own currency code.
    assert _measure(model, "Invoice", "SubtotalAmtDoc").unit_column.endswith("invoice.currencycd")


def test_an_amount_in_the_orders_currency_is_kept_apart_by_it(erp):
    _, model = erp
    doc = _measure(model, "SalesOrderLine", "ExtAmtDoc")
    assert doc.unit_column and doc.unit_column.endswith("salesorderheader.currencycd"), doc.unit_column


def test_an_amount_is_kept_apart_by_its_own_currency_never_a_customers():
    """Found in review: a payment's amount took the currency code of the branch it was paid at (one step away, but
    not its document), a tax base kept in the order's currency (TaxBaseAmtDoc) was read as the company's because
    "base" was in its name, a home delivery fee as the company's because of "home", and a line's amounts were split
    by the header's first currency code, the company's own (BaseCurrencyCd, always USD), not the order's."""
    con = duckdb.connect()
    con.execute("CREATE TABLE Branch (BranchCd VARCHAR PRIMARY KEY, BranchName VARCHAR, CurrencyCd VARCHAR)")
    con.execute("INSERT INTO Branch VALUES ('DAL01', 'Dallas Branch', 'USD'), ('TOR01', 'Toronto Branch', 'CAD')")
    con.execute("""CREATE TABLE OrderHdr AS SELECT 5000 + i AS OrderID, 'SO-' || i AS OrderNo, 'USD' AS BaseCurrencyCd,
                   ['DAL01', 'TOR01'][1 + i % 2] AS BranchCd, DATE '2025-01-01' + INTERVAL (i % 90) DAY AS OrderDate,
                   CASE WHEN i % 3 = 0 THEN 'CAD' ELSE 'USD' END AS CurrencyCd FROM range(1, 301) t(i)""")
    con.execute("ALTER TABLE OrderHdr ADD PRIMARY KEY (OrderID)")
    con.execute("""CREATE TABLE OrderLn AS SELECT 10 * h.OrderID + n AS OrderLnID, h.OrderID, n AS LineNo,
                   ROUND(20 + (h.OrderID * 7 + n) % 90 * 1.5, 2) AS ExtAmtDoc, ROUND(15 + (h.OrderID * 7 + n) % 90, 2) AS ExtAmtBase,
                   ROUND(18 + (h.OrderID * 3 + n) % 70 * 1.25, 2) AS TaxBaseAmtDoc,
                   ROUND(4 + (h.OrderID + n) % 9 * 0.75, 2) AS HomeDeliveryFeeAmt
                   FROM OrderHdr h, range(1, 3) t(n)""")
    con.execute("ALTER TABLE OrderLn ADD PRIMARY KEY (OrderLnID)")
    con.execute("""CREATE TABLE Payment AS SELECT 800 + i AS PaymentID, ['DAL01', 'TOR01'][1 + i % 2] AS BranchCd,
                   DATE '2025-01-01' + INTERVAL (i % 90) DAY AS PaidDate, ROUND(100 + (i * 37 % 120) * 3.25, 2) AS PaidAmt
                   FROM range(1, 201) t(i)""")
    con.execute("ALTER TABLE Payment ADD PRIMARY KEY (PaymentID)")
    warehouse = DuckDBWarehouse(con)
    fks = [{"parent_table": "OrderLn", "parent_col": "OrderID", "ref_table": "OrderHdr", "ref_col": "OrderID",
            "constraint_name": "FK_ln_hdr"},
           {"parent_table": "OrderHdr", "parent_col": "BranchCd", "ref_table": "Branch", "ref_col": "BranchCd",
            "constraint_name": "FK_hdr_branch"},
           {"parent_table": "Payment", "parent_col": "BranchCd", "ref_table": "Branch", "ref_col": "BranchCd",
            "constraint_name": "FK_pmt_branch"}]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=fks), client_id="eval-cur",
                        options=BuildOptions(workers=1))
    assert _measure(model, "OrderLn", "ExtAmtBase").unit_column is None
    for column in ("ExtAmtDoc", "TaxBaseAmtDoc", "HomeDeliveryFeeAmt"):
        unit = _measure(model, "OrderLn", column).unit_column
        assert unit and unit.endswith("orderhdr.currencycd"), (column, unit)
    assert _measure(model, "Payment", "PaidAmt").unit_column is None


def test_a_backup_copy_and_a_staging_copy_are_set_aside(erp):
    _, model = erp
    for name in ("SalesOrderHeader_bak_20240115", "stg_SalesOrderLine_Load"):
        copy = _table(model, name)
        assert copy.hidden, name
        assert not [m for m in model.measures.values() if m.table == copy.key and not m.hidden], name
    assert not _table(model, "SalesOrderHeader").hidden and not _table(model, "SalesOrderLine").hidden



def _inventory(*tables: tuple):
    from core2.bootstrap.inventory import InvColumn, InvTable, Inventory

    return Inventory(tables={f"{schema}.{name}": InvTable("", schema, name, [InvColumn(c, "int", "integer") for c in cols],
                                                          row_count=(rows or [None])[0])
                             for schema, name, cols, *rows in tables})


def test_a_table_named_like_another_is_no_copy_of_it():
    """Found in review: LandingPages, LandingFee, TruckLoads_2024, SavedSearches, Address2 and Sales_2023 were set
    aside as copies of Pages, Fee, Trucks, Searches, Address and Sales. A copy says so at the end of its name (or a
    staging word first, or a staging schema), and keeps the other's columns."""
    from core2.bootstrap.joins import copies_of

    cols = ["id", "name", "amount", "created_at", "region"]
    real = _inventory(
        ("dbo", "Pages", cols), ("dbo", "LandingPages", cols), ("dbo", "Fee", cols), ("dbo", "LandingFee", cols),
        ("dbo", "Trucks", cols), ("dbo", "TruckLoads_2024", cols), ("dbo", "Searches", cols),
        ("dbo", "SavedSearches", cols), ("dbo", "Address", cols), ("dbo", "Address2", cols), ("dbo", "Sales", cols),
        ("dbo", "Sales_2023", cols), ("dbo", "Material", cols), ("dbo", "RawMaterial_2024", cols),
        ("dbo", "Customer", cols), ("dbo", "Customer_Extract_2024", cols),
        # And in a second review: an archive's, the previous ones' and the saved ones' rows are their own, a month's
        # stamp is a month's table, and in a load's schema only a load's last word says it is a copy, on no more rows.
        ("dbo", "Orders", cols, 5000), ("dbo", "Orders_Archive", cols), ("dbo", "Sales_202312", cols),
        ("dbo", "CustomerAddress", cols), ("dbo", "CustomerAddress_Prev", cols), ("dbo", "Carts", cols),
        ("dbo", "Carts_Saved", cols), ("dbo", "Materials", cols), ("etl", "RawMaterials", cols),
        ("dbo", "Products", cols), ("etl", "ImportedProducts", cols), ("dbo", "Trucks", cols, 40),
        ("etl", "TruckLoads", cols, 9000))
    assert copies_of(real) == {}
    copies = _inventory(
        ("dbo", "Orders", cols), ("dbo", "Orders_Old", cols), ("dbo", "Orders_20240101", cols),
        ("dbo", "STG_ORDERS", cols + ["load_batch"]), ("dbo", "ORDERS_BAK", cols), ("dbo", "Orders_Old_2023", cols),
        ("dbo", "Customers", cols), ("stg", "Customers", cols), ("dbo", "Lines", cols), ("stg", "Lines_Load", cols),
        # A load of a table named so itself is its copy, never a copy of the trucks.
        ("dbo", "Trucks", cols), ("dbo", "TruckLoads", cols), ("stg", "TruckLoads", cols),
        # Named as a copy, but another table's columns: not set aside.
        ("dbo", "Payments", cols), ("dbo", "Payments_Old", ["pmt_no", "paid_on", "paid_amt"]))
    assert copies_of(copies) == {"dbo.Orders_Old": "dbo.Orders", "dbo.Orders_20240101": "dbo.Orders",
                                 "dbo.STG_ORDERS": "dbo.Orders", "dbo.ORDERS_BAK": "dbo.Orders",
                                 "dbo.Orders_Old_2023": "dbo.Orders", "stg.Customers": "dbo.Customers",
                                 "stg.Lines_Load": "dbo.Lines", "stg.TruckLoads": "dbo.TruckLoads"}


def test_what_a_line_sold_for_less_what_it_cost_is_its_gross_profit(erp):
    con, model = erp
    gp = next((m for m in model.measures.values() if not m.hidden and (
        m.business_name.casefold() == "gross profit" or "gross profit" in [w.casefold() for ws in m.synonyms.values() for w in ws])),
        None)
    assert gp is not None, sorted(m.business_name for m in model.measures.values())
    assert gp.table.endswith(".salesorderline") and gp.format == "currency"
    services = Services(model=model, warehouse=DuckDBWarehouse(con), index=MemberIndex(), today=TODAY,
                        complete=lambda s, t: json.dumps({"kind": "query", "intent": "value", "measures": [gp.slug]}))
    payload = answer_question("gross profit", services, Session())
    # Read as the orders, the header's cancelled orders are left out by default (its status names them Cancelled),
    # and so are the orders it marks deleted.
    assert sorted((f.column.rsplit(".", 1)[-1], f.op, f.values) for f in _table(model, "SalesOrderHeader").default_filters) == \
        [("isdeleted", "not_in", [1]), ("orderstatuscd", "not_in", ["CX"])]
    (want,) = con.execute("SELECT SUM(l.ExtAmtBase - l.ExtCostBase) FROM SalesOrderLine l JOIN SalesOrderHeader h "
                          "ON h.SalesOrderID = l.SalesOrderID WHERE h.OrderStatusCd <> 'CX' AND h.IsDeleted = 0").fetchone()
    got = (payload.get("kpi") or {}).get("value")
    assert got is not None and float(got) == pytest.approx(float(want), rel=1e-6), payload.get("answer")


def test_a_table_set_aside_is_listed_and_brought_back_whole(erp, tmp_path, monkeypatch):
    """The Relationships page lists the tables set aside and why; brought back, a copy answers again with its own
    figures and fields, and set aside by an admin, a table takes its figures with it."""
    from unittest.mock import patch

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    _, model = erp
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from admin import core2_relationships, routes
    from core2.bootstrap.service import load_model
    from core2.model import links

    store.init_db()
    account = "acct-erp-aside"
    store.upsert_client(account, "Wholesale (invented)")
    store.save_core2_model(account, None, model.model_dump_json())
    backup, copy = _table(model, "SalesOrderHeader_bak_20240115"), _table(model, "stg_SalesOrderLine_Load")
    listed = {t["key"]: t for t in links.view(load_model(account, None))["set_aside"]}
    assert "copy of SalesOrderHeader " in listed[backup.key]["why"], listed
    assert "copy of SalesOrderLine " in listed[copy.key]["why"], listed
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), patch.object(core2_relationships, "_is_auth", return_value=True):
        web = TestClient(app)
        assert web.post(f"/admin/clients/{account}/relationships/save-table", json={"table": copy.key, "hidden": False}).json()["ok"]
        back = load_model(account, None)
        assert not back.tables[copy.key].hidden
        assert [m for m in back.measures.values() if m.table == copy.key and not m.hidden]
        assert not [c for c in back.columns.values() if c.table == copy.key and c.hidden]
        assert back.tables[backup.key].hidden          # the other copy stays set aside
        line = _table(model, "SalesOrderLine")
        assert web.post(f"/admin/clients/{account}/relationships/save-table", json={"table": line.key, "hidden": True}).json()["ok"]
        aside = load_model(account, None)
        assert aside.tables[line.key].hidden
        assert not [m for m in aside.measures.values() if m.table == line.key and not m.hidden]
        assert line.key in {t["key"] for t in links.view(aside)["set_aside"]}


def test_the_orders_are_counted_as_orders_not_headers(erp):
    """A reader asks for "# orders": the count of the order header is "Number of sales orders", never of "sale order
    headers" ("sales" in "sales order" is no plural to undo, and a header is the document it heads), with or without
    the AI's names."""
    from core2.bootstrap import labels

    _, model = erp
    header = _table(model, "SalesOrderHeader")
    rows = model.measures[f"m:{header.key}.rows"]
    assert rows.business_name == "Number of sales orders"
    named = model.model_copy(deep=True)
    labels._apply(named, header.key, {"name": "Sales order header", "columns": {}})
    assert named.measures[rows.key].business_name == "Number of sales orders"


def test_the_plural_kept_in_a_name_changes_no_word_learn_compares():
    """Found in review: keeping "sales" and "accounts" plural before the noun they describe, so the orders are
    counted as "sales orders", was done where every table's words are read, and the accounts receivable were no
    longer a table of accounts (a party). Only the name shown keeps the plural."""
    from core2.bootstrap import names
    from core2.bootstrap.inventory import InvTable
    from core2.bootstrap.joins import _of_parties

    assert names.core_table("SalesOrderHeader") == ["sale", "order", "header"]
    assert names.core_table("AccountsReceivable") == ["account", "receivable"]
    assert names.table_noun("SalesOrderHeader") == ["sales", "order", "header"]
    # A header of sales alone counts sales, never "saleses".
    assert names.counted_noun(names.readable(" ".join(names.table_noun("tbl_sales_hdr")))).lower() == "sale"
    assert _of_parties(InvTable("", "dbo", "AccountsReceivable", []), InvTable("", "dbo", "Invoice", []))


def test_rows_marked_deleted_or_void_are_left_out_by_default():
    """A row an application deletes or voids often stays, marked by a flag (IsDeleted, is_void). A flag set on few rows
    leaves them out of every total by default, said on the answer, an admin able to count them again; a flag set on
    most rows is no deletion (every row would go) and nothing is left out."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE Payments AS SELECT i AS PaymentID, 100 + (i * 37) % 500 + 0.25 AS Amount,
                   DATE '2025-01-01' + INTERVAL (i % 300) DAY AS PaidOn, i % 31 = 0 AS IsVoid,
                   CASE WHEN i % 5 = 0 THEN 'N' ELSE 'Y' END AS IsRemoved FROM range(1, 2001) t(i)""")
    con.execute("ALTER TABLE Payments ADD PRIMARY KEY (PaymentID)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="eval-void", options=BuildOptions(workers=1))
    filters = {(f.column.rsplit(".", 1)[-1], f.op, tuple(f.values)) for f in _table(model, "Payments").default_filters}
    assert filters == {("isvoid", "not_in", (True,))}, filters
    amount = _measure(model, "Payments", "Amount")
    services = Services(model=model, warehouse=warehouse, index=MemberIndex(), today=TODAY,
                        complete=lambda s, t: json.dumps({"kind": "query", "intent": "value", "measures": [amount.slug]}))
    payload = answer_question("total payments", services, Session())
    (want,) = con.execute("SELECT SUM(Amount) FROM Payments WHERE NOT IsVoid").fetchone()
    assert float(payload["kpi"]["value"]) == pytest.approx(float(want)), payload.get("answer")
    said = payload["trust"].get("date_context") or []
    assert "Payment: rows where Is void is True are left out (a default filter)." in said, said
    assert "Including the rows left out by default" in [c["label"] for c in payload["follow_up_suggestions"]]
    review = next(r for r in model.review if r.key.endswith(".isvoid"))
    assert review.choice_made == "left out: True" and not review.alternatives[1:], review


def test_a_deletion_flag_is_one_by_its_whole_name_and_its_values():
    """Found in review: any 2-valued column with a deletion word was taken from the status rule (a void status of
    ACTIVE and VOIDED lost its leave-out), counts and facts with the word were read as deletions (VoidCount,
    DeviceRemoved, RemovedFromCart), and real flags were missed (DEL_FLAG, DeletionInd 'X' beside nothing,
    IS_DELETED holding 1 or nothing, a 0/1 flag kept as a decimal)."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE Tickets AS SELECT i AS TicketID, DATE '2025-01-01' + INTERVAL (i % 300) DAY AS OpenedOn,
                   100 + (i * 37) % 500 + 0.25 AS Amount,
                   CASE WHEN i % 23 = 0 THEN 'Y' ELSE 'N' END AS DEL_FLAG,
                   CASE WHEN i % 29 = 0 THEN 'X' END AS DeletionInd,
                   CASE WHEN i % 31 = 0 THEN 1 END AS IS_DELETED,
                   CASE WHEN i % 7 = 0 THEN 1 ELSE 0 END AS VoidCount,
                   CASE WHEN i % 11 = 0 THEN 'Y' ELSE 'N' END AS DeviceRemoved,
                   i % 13 = 0 AS RemovedFromCart,
                   CASE WHEN i % 17 = 0 THEN 'VOIDED' ELSE 'ACTIVE' END AS VoidStatus,
                   CASE WHEN i % 37 = 0 THEN 1 WHEN i % 3 = 0 THEN NULL ELSE 0 END AS is_void,
                   CASE WHEN i % 41 = 0 THEN 'Y' WHEN i % 3 = 1 THEN NULL ELSE 'N' END AS purged_flag,
                   (i * 7) % 40 AS Removed FROM range(1, 2001) t(i)""")
    con.execute("ALTER TABLE Tickets ADD PRIMARY KEY (TicketID)")
    con.execute("""CREATE TABLE Refunds AS SELECT i AS RefundID, DATE '2025-01-01' + INTERVAL (i % 300) DAY AS RefundedOn,
                   20 + (i * 13) % 300 + 0.5 AS RefundAmt, CAST(CASE WHEN i % 19 = 0 THEN 1 ELSE 0 END AS DECIMAL(1, 0))
                   AS IsDeleted FROM range(1, 2001) t(i)""")
    con.execute("ALTER TABLE Refunds ADD PRIMARY KEY (RefundID)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="eval-del", options=BuildOptions(workers=1))
    left_out = {(f.column.rsplit(".", 1)[-1], f.op, tuple(f.values)) for f in _table(model, "Tickets").default_filters}
    assert {c for c, _, _ in left_out} == {"del_flag", "deletionind", "is_deleted", "voidstatus", "is_void",
                                           "purged_flag"}, left_out
    assert ("voidstatus", "not_in", ("VOIDED",)) in left_out
    assert {(c, v) for c, _, v in left_out if c != "voidstatus"} == {
        ("del_flag", ("Y",)), ("deletionind", ("X",)), ("is_deleted", (1,)), ("is_void", (1,)),
        ("purged_flag", ("Y",))}, left_out
    # A count of items removed is a figure, whatever its name says.
    assert [m for m in model.measures.values() if str(getattr(m.expr, "column", "")).endswith(".removed")]
    assert not [m for m in model.measures.values() if str(getattr(m.expr, "column", "")).endswith(".is_deleted")
                and not m.hidden]
    refunds = {(f.column.rsplit(".", 1)[-1], tuple(f.values)) for f in _table(model, "Refunds").default_filters}
    assert refunds == {("isdeleted", (1,))}, refunds


def test_a_percentage_kept_as_a_share_of_one_is_shown_in_points():
    """DiscPct holds 0.05 for 5%: "avg disc %" was answered "0.0%" (0.0144 read as 0.0144%). A percentage whose values
    are shares of one is averaged as points (1.44%); one kept in points already (TaxPct 8.25) stays as it is. Found in
    review: a margin below nothing (-0.2 to 0.6) and a rate (0.031) were left as shares of one, and an exchange
    rate (0.74) is no share of anything."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE OrderLines AS SELECT i AS LineID, 50 + (i * 13) % 400 + 0.5 AS LineAmount,
                   [0.0, 0.0, 0.0, 0.05, 0.10, 0.15][1 + i % 6] AS DiscPct, [5.0, 7.25, 8.25][1 + i % 3] AS TaxPct,
                   [-0.2, 0.15, 0.35, 0.6][1 + i % 4] AS MarginPct, [0.021, 0.031, 0.044][1 + i % 3] AS ConversionRate,
                   [0.731, 0.744, 0.752][1 + i % 3] AS ExchRate, [0.05, 0.0725, 0.0825][1 + i % 3] AS TaxRate,
                   [0.11, 0.21, 0.30][1 + i % 3] AS KwhRate,
                   DATE '2025-01-01' + INTERVAL (i % 300) DAY AS LineDate FROM range(1, 1201) t(i)""")
    con.execute("ALTER TABLE OrderLines ADD PRIMARY KEY (LineID)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="eval-pct", options=BuildOptions(workers=1))
    disc = next(m for m in model.measures.values() if m.table.endswith(".orderlines") and "disc" in m.key)
    tax = next(m for m in model.measures.values() if m.table.endswith(".orderlines") and "taxpct" in m.key)
    assert disc.format == tax.format == "percent"
    margin, conversion, exch, tax_rate, kwh = (
        next(m for m in model.measures.values() if m.table.endswith(".orderlines") and m.key.endswith(f".{c}"))
        for c in ("marginpct", "conversionrate", "exchrate", "taxrate", "kwhrate"))
    assert margin.format == conversion.format == tax_rate.format == "percent", (tax_rate.format,)
    assert exch.format != "percent" and kwh.format != "percent", (exch.format, kwh.format)
    for measure, sql in ((disc, "SELECT AVG(DiscPct) * 100 FROM OrderLines"), (tax, "SELECT AVG(TaxPct) FROM OrderLines"),
                         (margin, "SELECT AVG(MarginPct) * 100 FROM OrderLines"),
                         (conversion, "SELECT AVG(ConversionRate) * 100 FROM OrderLines"),
                         (tax_rate, "SELECT AVG(TaxRate) * 100 FROM OrderLines"),
                         (exch, "SELECT AVG(ExchRate) FROM OrderLines")):
        services = Services(model=model, warehouse=warehouse, index=MemberIndex(), today=TODAY,
                            complete=lambda s, t, slug=measure.slug: json.dumps({"kind": "query", "intent": "value",
                                                                                 "measures": [slug]}))
        payload = answer_question("average", services, Session())
        (want,) = con.execute(sql).fetchone()
        assert float(payload["kpi"]["value"]) == pytest.approx(float(want)), (measure.key, payload["answer"]["headline"])


def test_gross_profit_takes_the_net_amount_beside_a_gross_one():
    """Found in review: lines holding a gross amount, a discount, a net amount and a cost had their gross profit
    worked out from the gross amount (overstated by every discount). The net amount, the one called sales, is used."""
    from evals.core2 import domains
    from evals.core2.compile_eval import learn

    _, model = learn(domains.build("retail"), "descriptive")
    gp = next((m for m in model.measures.values() if m.slug == "gross_profit"), None)
    assert gp is not None
    sold = model.measures[gp.expr.args[0].measure]
    assert model.columns[sold.expr.column].name == "net_amount", sold.key


def _lines(columns: str):
    con = duckdb.connect()
    con.execute(f"""CREATE TABLE OrderLines AS SELECT i AS LineID, DATE '2025-01-01' + INTERVAL (i % 90) DAY AS LineDate,
                    {columns} FROM range(1, 1501) t(i)""")
    con.execute("ALTER TABLE OrderLines ADD PRIMARY KEY (LineID)")
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="eval-gp", options=BuildOptions(workers=1))


def _gross_profit(model):
    gp = next((m for m in model.measures.values() if m.slug == "gross_profit"), None)
    return gp and tuple(model.columns[model.measures[a.measure].expr.column].name for a in gp.expr.args)


def test_gross_profit_is_worked_out_only_where_the_table_does_not_keep_one():
    """Found in review: an amount after its discount was dropped as a discount (the profit taken from the amount
    before it), a table keeping its own profit and margin got a second "Gross profit" and "Gross margin %", and a cost
    kept as a negative amount was taken off, adding it to the profit."""
    after = _lines("ROUND(50 + (i * 37) % 400 + 0.25, 2) AS LineAmount, ROUND(45 + (i * 37) % 380 + 0.75, 2) AS "
                   "AmtAfterDiscount, ROUND(30 + (i * 37) % 250 + 0.5, 2) AS LineCost")
    assert _gross_profit(after) == ("AmtAfterDiscount", "LineCost")
    kept = _lines("ROUND(50 + (i * 37) % 400 + 0.25, 2) AS SalesAmt, ROUND(30 + (i * 37) % 250 + 0.5, 2) AS CostAmt, "
                  "ROUND(20 + (i * 37) % 150 + 0.25, 2) AS GP_Amt, ROUND(0.1 + (i % 30) / 100.0, 3) AS GP_Pct")
    assert _gross_profit(kept) is None
    assert not [m for m in kept.measures.values() if m.slug == "gross_margin_pct"]
    negative = _lines("ROUND(50 + (i * 37) % 400 + 0.25, 2) AS SalesAmount, -ROUND(30 + (i * 37) % 250 + 0.5, 2) "
                      "AS CostAmount")
    assert _gross_profit(negative) is None
    # A net discount is a discount, never what was sold.
    net_discount = _lines("ROUND(50 + (i * 37) % 400 + 0.25, 2) AS Amount, ROUND(2 + (i * 37) % 20 + 0.25, 2) AS "
                          "NetDiscAmt, ROUND(30 + (i * 37) % 250 + 0.5, 2) AS Cost")
    assert _gross_profit(net_discount) == ("Amount", "Cost")
