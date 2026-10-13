"""Columns named the benchmark way -- each table's initials before every column, words glued in lower case -- are
read for what they say.

Found by the sector stress test (October 2026), on a public distribution warehouse whose every column carries its
table's initials (O_CUSTKEY on the orders, C_CUSTKEY on the customers, L_SUPPKEY on the order lines) and on a
public retail chain warehouse written the same way (SS_ on store sales, CR_ on catalog returns):

* "custkey" and "suppkey" were read as one word each, never as a customer key and a supplier key, so the orders
  were never linked to their customers nor the order lines to their suppliers, though the parts and orders, whose
  keys start with their table's whole name ("partkey"), were;
* a table's initials were read as a word of the name: CR_ (catalog returns) became "credit", CD_ (customer
  demographics) "code", and the links of the catalog returns were named "Credit call center", "Credit reason";
* a promotion's P_COST was linked to the customers, "cost" read as "customer" cut short to its consonants;
* an item's I_MANAGER_ID was linked to the items, as an item's parent item.

Invented rows only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap import names
from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse

REGIONS = ["NORTH", "SOUTH", "EAST", "WEST", "CENTRE"]


@pytest.fixture(scope="module")
def learned():
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE region AS SELECT range AS r_regionkey, ['{"','".join(REGIONS)}'][range + 1] AS r_name
        FROM range(5);
        CREATE TABLE nation AS SELECT range AS n_nationkey, 'Nation ' || range || repeat('x', range % 7) AS n_name,
               range % 5 AS n_regionkey FROM range(25);
        CREATE TABLE customer AS SELECT range + 1 AS c_custkey, 'Customer#' || lpad((range + 1)::VARCHAR, 6, '0') AS c_name,
               range % 25 AS c_nationkey, round((range * 37) % 9000 - 900 + 0.25, 2) AS c_acctbal,
               ['BUILDING', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE', 'AUTOMOBILE'][range % 5 + 1] AS c_mktsegment
        FROM range(400);
        CREATE TABLE supplier AS SELECT range + 1 AS s_suppkey, 'Supplier#' || lpad((range + 1)::VARCHAR, 6, '0') AS s_name,
               range % 25 AS s_nationkey FROM range(40);
        CREATE TABLE part AS SELECT range + 1 AS p_partkey, 'part ' || range || repeat(' steel', range % 4) AS p_name,
               round(900 + range % 300 + 0.99, 2) AS p_retailprice FROM range(300);
        CREATE TABLE orders AS SELECT range + 1 AS o_orderkey, (range * 7) % 400 + 1 AS o_custkey,
               DATE '2024-01-01' + (range % 700)::INTEGER AS o_orderdate, round(1000 + (range * 13) % 5000 + 0.5, 2) AS o_totalprice,
               ['O', 'F', 'P'][range % 3 + 1] AS o_orderstatus FROM range(2000);
        CREATE TABLE lineitem AS SELECT range // 4 + 1 AS l_orderkey, range % 4 + 1 AS l_linenumber,
               (range * 11) % 300 + 1 AS l_partkey, (range * 3) % 40 + 1 AS l_suppkey, range % 50 + 1 AS l_quantity,
               round(100 + (range * 17) % 9000 + 0.5, 2) AS l_extendedprice FROM range(8000);
        CREATE TABLE partsupp AS SELECT range % 300 + 1 AS ps_partkey, (range * 7) % 40 + 1 AS ps_suppkey,
               range % 900 + 10 AS ps_availqty FROM range(1200);
        CREATE TABLE promotion AS SELECT range + 1 AS pr_promokey, 'Promo ' || range || repeat('!', range % 5) AS pr_name,
               (100 + range * 3)::DECIMAL(15, 2) AS pr_cost FROM range(90);
        CREATE TABLE item AS SELECT range + 1 AS i_itemkey, 'Item ' || range || repeat(' blue', range % 3) AS i_product_name,
               range % 60 + 1 AS i_manager_id, round(5 + range % 90 + 0.99, 2) AS i_current_price FROM range(900);
        CREATE TABLE store_sales AS SELECT range + 1 AS ss_ticket_number, range % 900 + 1 AS ss_itemkey,
               range % 90 + 1 AS ss_promokey, range % 400 + 1 AS ss_custkey, round(1 + range % 200 + 0.5, 2) AS ss_net_paid
        FROM range(6000);
    """)
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    name_of = {key: t.name for key, t in inventory.tables.items()}
    return {(name_of[j.from_table], j.from_column): (name_of[j.to_table], j.to_column, j.trust, j.role)
            for j in found.joins if j.trust != "rejected"}


@pytest.mark.parametrize("table, column, target, key", [
    ("orders", "o_custkey", "customer", "c_custkey"),        # "custkey": the customer key, glued
    ("lineitem", "l_suppkey", "supplier", "s_suppkey"),
    ("lineitem", "l_partkey", "part", "p_partkey"),
    ("lineitem", "l_orderkey", "orders", "o_orderkey"),
    ("customer", "c_nationkey", "nation", "n_nationkey"),
    ("partsupp", "ps_partkey", "part", "p_partkey"),          # "part" names the parts, not PARTSUPP itself
    ("partsupp", "ps_suppkey", "supplier", "s_suppkey"),
    ("store_sales", "ss_custkey", "customer", "c_custkey"),
    ("store_sales", "ss_itemkey", "item", "i_itemkey"),
])
def test_a_key_named_with_its_tables_initials_links_to_its_table(learned, table, column, target, key):
    assert learned.get((table, column), (None,))[:3] == (target, key, "verified"), learned


def test_a_tables_initials_are_never_the_name_of_a_links_role(learned):
    roles = {k: v[3] for k, v in learned.items() if v[3]}
    assert not roles, roles


def test_a_cost_is_no_customer(learned):
    assert ("promotion", "pr_cost") not in learned, learned


def test_an_items_manager_is_no_item(learned):
    assert learned.get(("item", "i_manager_id"), ("",))[0] != "item", learned


@pytest.mark.parametrize("short, long_", [("cost", "customer"), ("count", "contract"), ("store", "storage"),
                                          ("part", "partsupp"), ("count", "country")])
def test_a_whole_word_is_no_abbreviation_of_another(short, long_):
    assert not names.same_word(short, long_)


@pytest.mark.parametrize("short, long_", [("cust", "customer"), ("supp", "supplier"), ("dept", "department"),
                                          ("prod", "product"), ("order", "orders"), ("ship", "shipment"),
                                          ("ship", "shipping"), ("return", "returned")])
def test_a_cut_short_word_still_reads_as_its_word(short, long_):
    assert names.same_word(short, long_)


@pytest.mark.parametrize("column, words", [("o_custkey", ["o", "cust"]), ("l_suppkey", ["l", "supp"]),
                                           ("CustomerID", ["customer"]), ("customerid", ["customer"]),
                                           ("monkey", ["monkey"]), ("paid", ["paid"])])
def test_a_key_word_glued_to_its_noun_is_read_apart(column, words):
    assert names.core_column(column) == words


# ── what the numbers are ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def measured():
    """A distribution warehouse named the benchmark way, and a pharmacy's batches and tests (invented rows)."""
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE orders AS SELECT range + 1 AS o_orderkey, DATE '2024-01-01' + (range % 700)::INTEGER AS o_orderdate,
               round(1000 + (range * 13) % 997 + 0.5, 2) AS o_totalprice, 0 AS o_shippriority FROM range(2000);
        CREATE TABLE lineitem AS SELECT range // 4 + 1 AS l_orderkey, range % 4 + 1 AS l_linenumber,
               range % 50 + 1 AS l_quantity, round(100 + (range * 17) % 1999 + 0.5, 2) AS l_extendedprice,
               (range % 7 + 1) % 5 AS l_shippriority, DATE '2024-01-03' + (range % 700)::INTEGER AS l_shipdate
        FROM range(8000);
        CREATE TABLE staff AS SELECT range + 1 AS staff_id, 'First' || range AS first_name, 'Last' || range AS last_name
        FROM range(30);
        CREATE TABLE batches AS SELECT range + 1 AS batch_id, DATE '2024-01-01' + (range % 500)::INTEGER AS compounded_date,
               range % 30 + 1 AS compounded_by, (range * 7) % 30 + 1 AS verified_by, 100 + range % 400 AS quantity_made,
               round(40 + range % 90 + 0.25, 2) AS price_charged FROM range(3000);
        CREATE TABLE quality_tests AS SELECT range + 1 AS test_id, range % 3000 + 1 AS batch_id,
               DATE '2024-02-01' + (range % 500)::INTEGER AS sent_date, round(0.5 + (range % 97) / 10, 2) AS result_value
        FROM range(1200);
    """)
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    name_of = {key: t.name for key, t in inventory.tables.items()}
    links = {(name_of[j.from_table], j.from_column): (name_of[j.to_table], j.to_column, j.trust)
             for j in found.joins if j.trust != "rejected"}
    return links, {(name_of[m.table], m.column): m for m in found.measures if m.column}


@pytest.mark.parametrize("table, column", [("lineitem", "l_linenumber"), ("lineitem", "l_shippriority"),
                                           ("orders", "o_shippriority"), ("batches", "compounded_by"),
                                           ("batches", "verified_by")])
def test_a_glued_line_number_a_priority_or_who_did_it_is_never_added_up(measured, table, column):
    assert (table, column) not in measured[1], measured[1][(table, column)]


@pytest.mark.parametrize("table, column", [("lineitem", "l_extendedprice"), ("orders", "o_totalprice"),
                                           ("batches", "price_charged")])
def test_a_lines_or_a_documents_price_is_an_amount(measured, table, column):
    m = measured[1][(table, column)]
    assert (m.agg, m.additivity, m.format) == ("sum", "additive", "currency"), m


def test_a_tests_result_is_averaged(measured):
    m = measured[1][("quality_tests", "result_value")]
    assert (m.agg, m.additivity) == ("avg", "non_additive") and m.format != "currency", m


@pytest.mark.parametrize("table, column, target, key", [("batches", "compounded_by", "staff", "staff_id"),
                                                        ("batches", "verified_by", "staff", "staff_id")])
def test_who_did_it_links_to_the_staff(measured, table, column, target, key):
    assert measured[0].get((table, column)) == (target, key, "verified"), measured[0]
