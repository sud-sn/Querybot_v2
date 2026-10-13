"""A date kept as a key into a calendar table has the calendar's first and last day as its range.

Found by the sector stress test on the TPC-DS retail benchmark (October 2026): every sale is dated by a key into
the date table (ss_sold_date_sk, 2,450,816 and up), no date written out. Learn kept no first or last day for
these dates, so the data looked current: "this month" on the retail dashboard was October 2026, with nothing in
it, where the data ends in 2003, and "the latest full month" was September 2026. The range is the calendar's
dates at the key's lowest and highest values.

The weekly stock table of the same data writes its columns INV_DATE_SK, INV_QUANTITY_ON_HAND: "inv" on the
inventory table is the inventory, and its date was named "Invoice date" (INV read as an invoice anywhere else).

Invented rows only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse

JULIAN_2001 = 2451911       # the date table's key for 1 January 2001, as TPC-DS numbers its days


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE date_dim AS SELECT {JULIAN_2001} - 366 + range AS d_date_sk,
               DATE '2000-01-01' + range::INTEGER AS d_date, year(DATE '2000-01-01' + range::INTEGER) AS d_year,
               month(DATE '2000-01-01' + range::INTEGER) AS d_moy, dayname(DATE '2000-01-01' + range::INTEGER) AS d_day_name
        FROM range(1500);
        CREATE TABLE item AS SELECT range + 1 AS i_item_sk, 'Item ' || range || repeat('x', range % 4) AS i_item_desc
        FROM range(200);
        CREATE TABLE store_sales AS SELECT {JULIAN_2001} + (range % 700) AS ss_sold_date_sk,
               range % 200 + 1 AS ss_item_sk, range + 1 AS ss_ticket_number, range % 9 + 1 AS ss_quantity,
               round(3 + (range * 7) % 300 + 0.5, 2) AS ss_net_paid FROM range(20000);
        CREATE TABLE inventory AS SELECT {JULIAN_2001} + w.range * 7 AS inv_date_sk, i.range + 1 AS inv_item_sk,
               (i.range * 13 + w.range * 5) % 400 AS inv_quantity_on_hand FROM range(60) w, range(200) i;
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-cal", options=BuildOptions(workers=1))


def test_a_sale_dated_by_a_calendar_key_has_the_calendars_range(model):
    sales = next(t for k, t in model.tables.items() if k.endswith(".store_sales"))
    role = model.date_roles[sales.default_date]
    assert (role.first, role.last) == (dt.date(2001, 1, 1), dt.date(2001, 1, 1) + dt.timedelta(days=699))


def test_a_date_written_with_its_tables_initials_is_named_for_the_table(model):
    stock = next(t for k, t in model.tables.items() if k.endswith(".inventory"))
    assert model.date_roles[stock.default_date].name == "Inventory date"
    assert "invoice" not in " ".join(c.business_name.lower() for c in model.columns.values() if c.table == stock.key)


def test_only_a_shortening_read_as_another_word_is_the_tables():
    from core2.bootstrap.names import own_words

    assert own_words("inventory", "inv_date_sk") == "inventory_date_sk"
    assert own_words("payroll", "pay_date") == "pay_date", "pay is a word of its own on the payroll"
