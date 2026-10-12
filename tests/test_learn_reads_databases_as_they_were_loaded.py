"""Learn reads a database as it was loaded, and a period that has ended is compared whole.

Found on the public sample databases (as many real ones are loaded, from files or another database):

* a key kept as a floating-point number (payment.rental_id 76.0) was never a link, so payments never reached
  their rentals: a column named as a key holding only whole numbers is read as a whole number;
* a list known only by its number (a store with no name or code) was refused, "Store has no name to show": it is
  told apart by its number;
* "2025 against 2024", asked in 2026 of data whose last sale was on 22 December 2025, trimmed both years to 22
  December, and the change turned from a fall to a rise: an ended period whose data stops a few quiet days before
  its end is compared whole (a load that stopped months early is still compared like for like).

Invented data only.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.resolve.resolver import _like_for_like
from core2.resolve.time import Range
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def rentals():
    con = duckdb.connect()
    con.execute("CREATE TABLE store (store_id INTEGER PRIMARY KEY, manager_staff_id INTEGER, address_id INTEGER)")
    con.executemany("INSERT INTO store VALUES (?, ?, ?)", [(1, 1, 1), (2, 2, 2)])
    con.execute("CREATE TABLE rental (rental_id INTEGER PRIMARY KEY, rental_date DATE, store_id INTEGER)")
    con.executemany("INSERT INTO rental VALUES (?, ?, ?)",
                    [(i, dt.date(2005, 5, 1) + dt.timedelta(days=i % 90), 1 + i % 2) for i in range(1, 1601)])
    con.execute("CREATE TABLE payment (payment_id INTEGER PRIMARY KEY, rental_id DOUBLE, amount DECIMAL(6,2), "
                "payment_date DATE)")
    con.executemany("INSERT INTO payment VALUES (?, ?, ?, ?)",
                    [(i, float(1 + (i * 7) % 1600), 2.99, dt.date(2005, 5, 1) + dt.timedelta(days=i % 90))
                     for i in range(1, 1501)])
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-loaded", options=BuildOptions(workers=1))


def test_a_key_kept_as_a_floating_number_is_a_link(rentals):
    assert any(j.from_table.endswith(".payment") and j.to_table.endswith(".rental") for j in rentals.joins.values())


def test_a_list_known_only_by_its_number_can_be_grouped_by(rentals):
    store = next(e for e in rentals.entities.values() if e.table.endswith(".store"))
    assert any(a.entity == store.slug and a.column in store.key_columns for a in rentals.attributes.values())


Y25, Y24 = Range(dt.date(2025, 1, 1), dt.date(2026, 1, 1)), Range(dt.date(2024, 1, 1), dt.date(2025, 1, 1))


def test_an_ended_year_with_a_few_quiet_days_is_compared_whole():
    window, compare, note = _like_for_like(Y25, Y24, dt.date(2025, 12, 22), "same_period_last_year",
                                           today=dt.date(2026, 10, 12))
    assert (window, compare, note) == (Y25, Y24, "")


def test_a_load_that_stopped_months_early_is_still_like_for_like():
    window, compare, note = _like_for_like(Y25, Y24, dt.date(2025, 6, 30), "same_period_last_year",
                                           today=dt.date(2026, 10, 12))
    assert window.end == dt.date(2025, 7, 1) and compare.end == dt.date(2024, 7, 1) and note


def test_the_current_year_is_still_like_for_like():
    window, _, note = _like_for_like(Range(dt.date(2026, 1, 1), dt.date(2027, 1, 1)), Y25, dt.date(2026, 10, 10),
                                     "same_period_last_year", today=dt.date(2026, 10, 12))
    assert window.end == dt.date(2026, 10, 11) and note
