"""A designed dashboard tile's title says what its figure means only when the figure, its table or the reader did.

Found by the sector stress test on a retail bank's database (October 2026): asked for an executive dashboard, the
AI design titled the sum of every transaction's amount -- deposits, withdrawals, transfers and fees alike --
"Total Revenue", "Revenue Trend", "Revenue by Region" and "Top 10 Branches by Revenue", on a tab "Revenue". The
questions asked for the amount; only the titles called it revenue, and a bank's revenue is no such sum. The tiles
are named for their figure ("Total Amount"), on a tab named for its table, unless the figure, its table or the
reader's request says revenue.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.dashboards import Tile, meant
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE branches AS SELECT range + 1 AS branch_id, 'Branch ' || range || repeat('!', range % 3) AS branch_name,
               ['North', 'South', 'East'][range % 3 + 1] AS region FROM range(12);
        CREATE TABLE transactions AS SELECT range + 1 AS transaction_id, range % 12 + 1 AS branch_id,
               DATE '2025-01-01' + (range % 500)::INTEGER AS transaction_date,
               ['Deposit', 'Withdrawal', 'Transfer', 'Fee'][range % 4 + 1] AS transaction_type,
               round(5 + (range * 37) % 2000 + 0.25, 2) AS amount FROM range(6000);
        CREATE TABLE sales AS SELECT range + 1 AS sale_id, range % 12 + 1 AS branch_id,
               DATE '2025-01-01' + (range % 500)::INTEGER AS sale_date,
               round(20 + (range * 13) % 900 + 0.5, 2) AS amount FROM range(3000);
    """)
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="t-meant", options=BuildOptions(workers=1))


def _slug(model, table):
    return next(m.slug for m in model.measures.values() if m.table.endswith("." + table) and
                getattr(m.expr, "column", "").endswith(".amount"))


@pytest.mark.parametrize("title, want", [
    ("Total Revenue", "Total Amount"),
    ("Revenue Trend", "Amount Trend"),
    ("Top 10 Branches by Revenue", "Top 10 Branches by Amount"),
    ("Net revenue by region", "Amount by region"),
    ("Branches ranked by net revenue", "Branches ranked by amount"),
    ("Amount by month", "Amount by month"),
])
def test_a_sum_of_transactions_is_not_called_revenue(model, title, want):
    tile = Tile(title=title, kind="number", question="Amount for the last fiscal year", tab="Revenue",
                plan={"measures": [_slug(model, "transactions")]})
    meant(model, tile, "Build an executive dashboard for the bank")
    assert tile.title.casefold() == want.casefold() and tile.title[:1] == want[:1]
    assert tile.tab == "Transactions"


def test_the_amount_of_a_sales_table_is_sales(model):
    tile = Tile(title="Total Sales", kind="number", question="Amount this year", tab="Sales",
                plan={"measures": [_slug(model, "sales")]})
    meant(model, tile, "Build an executive dashboard")
    assert (tile.title, tile.tab) == ("Total Sales", "Sales")


def test_the_readers_own_word_is_kept(model):
    tile = Tile(title="Revenue by month", kind="trend", question="Amount by month", tab="Overview",
                plan={"measures": [_slug(model, "transactions")]})
    meant(model, tile, "I need revenue by month and transactions by branch")
    assert (tile.title, tile.tab) == ("Revenue by month", "Overview")
