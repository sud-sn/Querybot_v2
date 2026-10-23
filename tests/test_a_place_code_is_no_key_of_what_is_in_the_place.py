"""A code of a place is no key of what is in the place: nothing points at a warehouse by its province.

Found on a sales warehouse with six warehouses, each in its own province: every column of the warehouses was unique,
the province code among them, and the customers' province code was linked to the warehouses by it (named the same,
every value found, used at once). A customer is in a province, not in a warehouse: "customers by warehouse", or a
figure whose table holds no warehouse of its own, would have gone through the province. A unique code named for a
place or a currency (province, state, country, region, city, postal code, currency, however it is cut short: PRV,
CNTRY, CCY; a PROV is as often a provider) is now no target on a table of other things, each with a code or a name of
its own (WHS_CD, WHS_NM, or a bare CODE and NAME), pointed at by a key of their own (the sales' WHS_KEY), unless the
table is named for a place: there it says where each row is, unique by accident or one row a place. Still linked:

* the warehouses' own code, by the same name (a shipment's WHS_CD), and their keys;
* a place's own code on its own table, whatever the table is called: the provinces, however few; a geography's
  state code, a DIM_CCY's currency, the ZIPCODES, a nation's country code; and on a table with nothing else to call
  its rows (one exchange rate for each currency); a code named for something else last (a language's
  COUNTRY_LANG_CD); a table nothing else points at (a tax rate for each province); a clinic's provider code;
* a lookup's own code, whatever its table is called: a category's CATEGORY_CD, an account type's TYP_CD, a
  status's STATUS_CD, a country's ISO_CD (an ISO code names no other place);
* a status table's bare CODE, by a name that says the table (an order's ORDER_STATUS_CODE);
* a code on a table of many rows (a product's SKU_CD among 500 products).

Invented data only.
"""

from __future__ import annotations

import datetime as dt
import random

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse

PROVINCES = ["ON", "QC", "BC", "AB", "MB", "NS", "NB", "NL", "PE", "SK", "YT", "NT", "NU"]


def _links(con: duckdb.DuckDBPyConnection, client: str, declared=()) -> dict[tuple[str, str], str]:
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=list(declared)), client_id=client,
                        options=BuildOptions(workers=1))
    name = lambda key: f"{model.tables[model.columns[key].table].name}.{model.columns[key].name}"  # noqa: E731
    return {(name(j.from_columns[0]), name(j.to_columns[0])): j.trust for j in model.joins.values()
            if not j.to_calendar}


def _day(i: int) -> dt.date:
    return dt.date(2024, 1, 1) + dt.timedelta(days=i % 365)


@pytest.fixture(scope="module")
def depots():
    rng = random.Random(3)
    con = duckdb.connect()
    con.execute("CREATE TABLE WAREHOUSES (WHS_KEY INTEGER, WHS_CD VARCHAR, WHS_NM VARCHAR, CTY_NM VARCHAR, "
                "PRV_CD VARCHAR)")
    con.executemany("INSERT INTO WAREHOUSES VALUES (?,?,?,?,?)",
                    [(i + 1, f"W{i + 1:02d}", f"Depot {i + 1}", f"City {i}", p) for i, p in enumerate(PROVINCES[:6])])
    con.execute("CREATE TABLE CUSTOMERS (CUS_KEY INTEGER, CUS_NM VARCHAR, PRV_CD VARCHAR)")
    con.executemany("INSERT INTO CUSTOMERS VALUES (?,?,?)",
                    [(i, f"Customer {i:03d}", rng.choice(PROVINCES[:6])) for i in range(1, 61)])
    con.execute("CREATE TABLE SALES (SALE_KEY INTEGER, CUS_KEY INTEGER, WHS_KEY INTEGER, AMT DECIMAL(10,2), "
                "SALE_DT DATE)")
    con.executemany("INSERT INTO SALES VALUES (?,?,?,?,?)",
                    [(i, rng.randint(1, 60), rng.randint(1, 6), rng.randint(100, 99999) / 100, _day(i))
                     for i in range(1, 3001)])
    con.execute("CREATE TABLE SHIPMENTS (SHP_KEY INTEGER, WHS_CD VARCHAR, QTY INTEGER, SHP_DT DATE)")
    con.executemany("INSERT INTO SHIPMENTS VALUES (?,?,?,?)",
                    [(i, f"W{rng.randint(1, 6):02d}", rng.randint(1, 50), _day(i)) for i in range(1, 501)])
    return _links(con, "eval-place-codes-depots")


def test_a_province_shared_with_a_few_warehouses_is_no_link(depots):
    assert ("CUSTOMERS.PRV_CD", "WAREHOUSES.PRV_CD") not in depots, depots


def test_nor_a_state_with_one_store_in_each():
    rng = random.Random(8)
    states = [f"S{i:02d}" for i in range(50)]
    con = duckdb.connect()
    con.execute("CREATE TABLE STORES (STORE_KEY INTEGER, STORE_CD VARCHAR, STORE_NM VARCHAR, STATE_CD VARCHAR)")
    con.executemany("INSERT INTO STORES VALUES (?,?,?,?)",
                    [(i + 1, f"ST{i + 1:03d}", f"Store {i + 1}", s) for i, s in enumerate(states)])
    con.execute("CREATE TABLE CUSTOMERS (CUS_KEY INTEGER, CUS_NM VARCHAR, STATE_CD VARCHAR)")
    con.executemany("INSERT INTO CUSTOMERS VALUES (?,?,?)",
                    [(i, f"Customer {i:04d}", rng.choice(states)) for i in range(1, 2001)])
    con.execute("CREATE TABLE SALES (SALE_KEY INTEGER, CUS_KEY INTEGER, STORE_KEY INTEGER, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO SALES VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 2000), rng.randint(1, 50), rng.randint(100, 99999) / 100)
                     for i in range(1, 5001)])
    links = _links(con, "eval-place-codes-stores")
    assert ("CUSTOMERS.STATE_CD", "STORES.STATE_CD") not in links, links
    assert ("SALES.STORE_KEY", "STORES.STORE_KEY") in links, links


@pytest.mark.parametrize("table, columns, column, codes, rows, pointer", [
    ("WAREHOUSES", "WHS_KEY INTEGER, WHS_CD VARCHAR, WHS_NM VARCHAR, PRV_CD VARCHAR", "PRV_CD", PROVINCES[:6], 60,
     "WHS_KEY"),
    ("WAREHOUSES", "ID INTEGER, CODE VARCHAR, NAME VARCHAR, PROVINCE_CODE VARCHAR", "PROVINCE_CODE", PROVINCES[:6], 60,
     "WAREHOUSE_ID"),
    ("BRANCHES", "BRANCH_KEY INTEGER, BRANCH_NM VARCHAR, CNTRY_CD VARCHAR", "CNTRY_CD",
     ["CA", "US", "MX", "GB", "FR", "DE", "IT", "ES"], 400, "BRANCH_KEY"),
    ("TRADING_DESKS", "DESK_KEY INTEGER, DESK_NM VARCHAR, CCY_CD VARCHAR", "CCY_CD", ["CAD", "USD", "EUR", "GBP", "JPY"],
     500, "DESK_KEY"),
    # An account in each currency: "current" is no currency.
    ("CURRENT_ACCOUNTS", "ACCT_KEY INTEGER, ACCT_NM VARCHAR, CCY_CD VARCHAR", "CCY_CD", ["CAD", "USD", "EUR", "GBP"],
     500, "ACCT_KEY")])
def test_nor_a_place_cut_short_on_a_table_of_other_things(table, columns, column, codes, rows, pointer):
    rng = random.Random(len(table))
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {table} ({columns})")
    width = len(columns.split(","))
    con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * width)})",
                    [(i + 1, *[f"{table[:3]} {i + 1}"] * (width - 2), code) for i, code in enumerate(codes)])
    con.execute(f"CREATE TABLE CLIENTS (CLIENT_KEY INTEGER, CLIENT_NM VARCHAR, {column} VARCHAR)")
    con.executemany("INSERT INTO CLIENTS VALUES (?,?,?)",
                    [(i, f"Client {i:04d}", rng.choice(codes)) for i in range(1, rows + 1)])
    # What the table's rows are pointed at by: a key of their own.
    con.execute(f"CREATE TABLE TRADES (TRADE_KEY INTEGER, {pointer} INTEGER, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO TRADES VALUES (?,?,?)",
                    [(i, rng.randint(1, len(codes)), rng.randint(100, 99999) / 100) for i in range(1, 1001)])
    links = _links(con, f"eval-place-codes-{table.lower()}-{column.lower()}")
    assert (f"CLIENTS.{column}", f"{table}.{column}") not in links, links
    assert any(a == f"TRADES.{pointer}" for a, _ in links), links


@pytest.mark.parametrize("table, columns, column, codes, rows, pointer", [
    ("PROVINCES", "PRV_KEY INTEGER, PRV_CD VARCHAR, PRV_NM VARCHAR", "PRV_CD", PROVINCES, 1500, "PRV_KEY"),
    ("DIM_GEOGRAPHY", "GEO_KEY INTEGER, STATE_CD VARCHAR, STATE_NM VARCHAR", "STATE_CD",
     [a + b for a in "ABCDEFGHIJ" for b in "KLMNO"], 2000, "GEO_KEY"),
    ("DIM_CCY", "CCY_KEY INTEGER, CURRENCY_CD VARCHAR, CCY_NM VARCHAR", "CURRENCY_CD",
     ["CAD", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "MXN", "CNY", "INR", "BRL", "SEK"], 1500, "CCY_KEY"),
    ("ZIPCODES", "ZIP_ID INTEGER, ZIP_CD VARCHAR, CITY_NM VARCHAR", "ZIP_CD", [f"{i:05d}" for i in range(10001, 10301)],
     3000, "ZIP_ID"),
    ("POSTALCODES", "PC_KEY INTEGER, POSTAL_CD VARCHAR, POSTALCODE_NM VARCHAR", "POSTAL_CD",
     [f"K{i}A {i}B{i}" for i in range(10, 90)], 3000, "PC_KEY"),
    ("DIM_NATION", "NATION_KEY INTEGER, COUNTRY_CD VARCHAR, NATION_NM VARCHAR", "COUNTRY_CD",
     ["CA", "US", "MX", "GB", "FR", "DE", "IT", "ES", "JP", "CN"], 1500, "NATION_KEY"),
    ("EXCHANGE_RATES", "RATE_ID INTEGER, CURRENCY_CD VARCHAR, RATE DECIMAL(10,4)", "CURRENCY_CD",
     ["CAD", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "MXN"], 1500, "RATE_ID"),
    # A language of a country is a language: the code is named for what it is last.
    ("LANGUAGES", "LANGUAGE_KEY INTEGER, COUNTRY_LANG_CD VARCHAR, LANGUAGE_NM VARCHAR", "COUNTRY_LANG_CD",
     ["en-CA", "fr-CA", "en-US", "es-MX", "fr-FR", "de-DE", "it-IT", "es-ES"], 1500, "LANGUAGE_KEY"),
    # A tax rate for each province, which nothing else points at: read by its province.
    ("TAX_RATES", "TAX_RATE_ID INTEGER, PRV_CD VARCHAR, TAX_RATE_NM VARCHAR", "PRV_CD", PROVINCES, 1500, None),
    # A clinic's PROV_CD is its provider's code, no province.
    ("CLINICS", "CLINIC_ID INTEGER, PROV_CD VARCHAR, CLINIC_NM VARCHAR", "PROV_CD", [f"P{i:03d}" for i in range(12)],
     1500, "CLINIC_ID")])
def test_a_places_own_table_still_links_whatever_it_is_called(table, columns, column, codes, rows, pointer):
    rng = random.Random(len(table))
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {table} ({columns})")
    con.executemany(f"INSERT INTO {table} VALUES (?,?,?)",
                    [(i + 1, code, 1 + i / 10 if "DECIMAL" in columns else f"Name {code}")
                     for i, code in enumerate(codes)])
    con.execute(f"CREATE TABLE SALES (SALE_KEY INTEGER, {column} VARCHAR, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO SALES VALUES (?,?,?)",
                    [(i, rng.choice(codes), rng.randint(100, 99999) / 100) for i in range(1, rows + 1)])
    if pointer:     # the table's rows pointed at by a key of their own as well
        con.execute(f"CREATE TABLE TRADES (TRADE_KEY INTEGER, {pointer} INTEGER, QTY INTEGER)")
        con.executemany("INSERT INTO TRADES VALUES (?,?,?)",
                        [(i, rng.randint(1, len(codes)), rng.randint(1, 9)) for i in range(1, 1001)])
    links = _links(con, f"eval-place-codes-{table.lower()}")
    assert (f"SALES.{column}", f"{table}.{column}") in links, links
    assert not pointer or any(a == f"TRADES.{pointer}" for a, _ in links), links


def test_a_place_code_the_database_declares_still_links():
    rng = random.Random(12)
    con = duckdb.connect()
    con.execute("CREATE TABLE WAREHOUSES (WHS_KEY INTEGER, WHS_CD VARCHAR, WHS_NM VARCHAR, PRV_CD VARCHAR)")
    con.executemany("INSERT INTO WAREHOUSES VALUES (?,?,?,?)",
                    [(i + 1, f"W{i + 1:02d}", f"Depot {i + 1}", p) for i, p in enumerate(PROVINCES[:6])])
    con.execute("CREATE TABLE CUSTOMERS (CUS_KEY INTEGER, CUS_NM VARCHAR, PRV_CD VARCHAR)")
    con.executemany("INSERT INTO CUSTOMERS VALUES (?,?,?)",
                    [(i, f"Customer {i:03d}", rng.choice(PROVINCES[:6])) for i in range(1, 61)])
    con.execute("CREATE TABLE SALES (SALE_KEY INTEGER, CUS_KEY INTEGER, WHS_KEY INTEGER, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO SALES VALUES (?,?,?,?)",
                    [(i, rng.randint(1, 60), rng.randint(1, 6), rng.randint(100, 99999) / 100) for i in range(1, 2001)])
    declared = [{"parent_table": "CUSTOMERS", "parent_col": "PRV_CD", "ref_table": "WAREHOUSES", "ref_col": "PRV_CD"}]
    assert ("CUSTOMERS.PRV_CD", "WAREHOUSES.PRV_CD") in _links(con, "eval-place-codes-declared", declared)


@pytest.mark.parametrize("lookup, key, column, codes", [
    ("PRODUCT_CATEGORY", "CATEGORY_ID", "CATEGORY_CD", ["BEV", "SNK", "DRY", "FRZ", "PRD", "DLY", "BKY", "MEA"]),
    ("ACCOUNT_TYPES", "ACCT_TYP_ID", "TYP_CD", ["CHQ", "SAV", "LOC", "MTG", "CRD"]),
    ("DIM_STATUS", "STATUS_KEY", "STATUS_CD", ["OPN", "SHP", "CLS", "CNL", "HLD"]),
    ("COUNTRIES", "COUNTRY_KEY", "ISO_CD", ["CA", "US", "MX", "GB", "FR", "DE", "IT", "ES", "JP", "CN", "BR", "AU"])])
def test_a_lookups_own_code_still_links(lookup, key, column, codes):
    rng = random.Random(len(lookup))
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {lookup} ({key} INTEGER, {column} VARCHAR, LABEL_NM VARCHAR)")
    con.executemany(f"INSERT INTO {lookup} VALUES (?,?,?)", [(i + 1, c, f"Label {c}") for i, c in enumerate(codes)])
    con.execute(f"CREATE TABLE ITEMS (ITEM_ID INTEGER, {column} VARCHAR, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO ITEMS VALUES (?,?,?)",
                    [(i, rng.choice(codes), rng.randint(100, 99999) / 100) for i in range(1, 1501)])
    assert (f"ITEMS.{column}", f"{lookup}.{column}") in _links(con, f"eval-place-codes-{lookup.lower()}")


def test_the_warehouses_own_code_and_keys_still_link(depots):
    assert {("SHIPMENTS.WHS_CD", "WAREHOUSES.WHS_CD"), ("SALES.WHS_KEY", "WAREHOUSES.WHS_KEY"),
            ("SALES.CUS_KEY", "CUSTOMERS.CUS_KEY")} <= set(depots), depots


def test_a_status_tables_bare_code_is_its_own():
    rng = random.Random(4)
    con = duckdb.connect()
    con.execute("CREATE TABLE ORDER_STATUS (STATUS_ID INTEGER, CODE VARCHAR, NAME VARCHAR)")
    con.executemany("INSERT INTO ORDER_STATUS VALUES (?,?,?)", [(1, "OPN", "Open"), (2, "SHP", "Shipped"),
                                                              (3, "CLS", "Closed"), (4, "CNL", "Cancelled"),
                                                              (5, "HLD", "On hold")])
    con.execute("CREATE TABLE ORDERS (ORDER_ID INTEGER, ORDER_STATUS_CODE VARCHAR, AMT DECIMAL(10,2), ORDER_DT DATE)")
    con.executemany("INSERT INTO ORDERS VALUES (?,?,?,?)",
                    [(i, rng.choice(["OPN", "SHP", "CLS", "CNL", "HLD"]), rng.randint(100, 9999) / 100, _day(i))
                     for i in range(1, 2001)])
    assert ("ORDERS.ORDER_STATUS_CODE", "ORDER_STATUS.CODE") in _links(con, "eval-place-codes-status")


def test_a_province_code_still_points_at_the_provinces():
    rng = random.Random(5)
    con = duckdb.connect()
    con.execute("CREATE TABLE PROVINCES (PRV_KEY INTEGER, PRV_CD VARCHAR, PRV_NM VARCHAR)")
    con.executemany("INSERT INTO PROVINCES VALUES (?,?,?)",
                    [(i + 1, p, f"Province {p}") for i, p in enumerate(PROVINCES)])
    con.execute("CREATE TABLE CUSTOMERS (CUS_KEY INTEGER, CUS_NM VARCHAR, PRV_CD VARCHAR)")
    con.executemany("INSERT INTO CUSTOMERS VALUES (?,?,?)",
                    [(i, f"Customer {i:03d}", rng.choice(PROVINCES)) for i in range(1, 201)])
    assert ("CUSTOMERS.PRV_CD", "PROVINCES.PRV_CD") in _links(con, "eval-place-codes-provinces")


def test_a_code_among_many_rows_still_links_by_its_name():
    rng = random.Random(6)
    con = duckdb.connect()
    con.execute("CREATE TABLE PRODUCTS (PRODUCT_KEY INTEGER, SKU_CD VARCHAR, PRODUCT_NM VARCHAR)")
    con.executemany("INSERT INTO PRODUCTS VALUES (?,?,?)", [(i, f"SKU{i:05d}", f"Product {i}") for i in range(1, 501)])
    con.execute("CREATE TABLE SALES (SALE_KEY INTEGER, SKU_CD VARCHAR, QTY INTEGER, AMT DECIMAL(10,2))")
    con.executemany("INSERT INTO SALES VALUES (?,?,?,?)",
                    [(i, f"SKU{rng.randint(1, 500):05d}", rng.randint(1, 9), rng.randint(100, 9999) / 100)
                     for i in range(1, 4001)])
    assert ("SALES.SKU_CD", "PRODUCTS.SKU_CD") in _links(con, "eval-place-codes-skus")
