"""A grouping is counted in plain words, whatever its field is called.

On a public sample warehouse the AI named fields "Category name (english)" and "Promotion name (ES)", and answers
read "all 3 category name (english)s rose" and "across 4 promotion name (ES)s". The language, or anything else in
brackets after the name, is no part of what is counted: "all 3 categories", "across 4 promotions".

Synthetic data only.
"""

from __future__ import annotations

import json

import pytest

from core2.answer.builder import _noun, _thing


@pytest.mark.parametrize("label, one, many", [
    ("Category name (english)", "category", "categories"),
    ("Promotion name (ES)", "promotion", "promotions"),
    ("Country name (English)", "country", "countries"),
    ("Warehouse name", "warehouse", "warehouses"),
    ("Day of week", "day of week", "days of week"),
    ("GL account", "GL account", "GL accounts"),
])
def test_a_grouping_is_counted_in_plain_words(label, one, many):
    assert _thing(label) == one and _noun(label) == many


def test_a_durations_name_is_said_in_words():
    """ "days_between_order_and_ship: 7 days" was the headline of how long orders take to ship."""
    from core2.plan.ir import Plan
    from core2.plan.planner import ratio_alone

    plan = Plan.model_validate({"kind": "query", "intent": "value", "durations": [
        {"name": "days_between_order_and_ship", "start": "order_date", "end": "ship_date"}],
        "sort": [{"by": "days_between_order_and_ship", "desc": True}],
        "filters": [{"field": "days_between_order_and_ship", "op": "gt", "values": [3]}]})
    named = ratio_alone(plan, None, "How many days between order and ship?")
    assert named.durations[0].name == "Days between order and ship"
    assert named.sort[0].by == "Days between order and ship" and named.filters[0].field == "Days between order and ship"


def test_a_count_of_a_tables_members_is_named_after_the_table():
    """ "Number of entities: 701" counted the stores, whose key is named for business entities."""
    import duckdb

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.bootstrap.labels import label
    from core2.warehouse.runner import DuckDBWarehouse

    con = duckdb.connect()
    con.execute("CREATE TABLE Store (BusinessEntityID INTEGER, Name VARCHAR)")
    con.execute("INSERT INTO Store SELECT i, 'Store ' || i || repeat('x', i % 5) FROM range(1, 51) t(i)")
    con.execute("CREATE TABLE Visit (VisitID INTEGER, BusinessEntityID INTEGER, Minutes INTEGER, VisitDate DATE)")
    con.execute("INSERT INTO Visit SELECT i, i % 50 + 1, 10 + i % 90, DATE '2025-01-01' + CAST(i % 300 AS INTEGER) "
                "FROM range(1, 2001) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="stores", options=BuildOptions(workers=1))
    reply = {"tables": {"Store": {"name": "Store", "columns": {"BusinessEntityID": {"name": "Entity ID"},
                                                                "Name": {"name": "Store name"}}}}}
    label(model, lambda system, user: json.dumps(reply))
    store = next(t.key for t in model.tables.values() if t.name == "Store")
    counts = [m.business_name for m in model.measures.values() if m.table == store and m.kind == "count"]
    assert counts == ["Number of stores"], counts


def test_one_column_of_a_two_column_key_is_not_named_after_the_table():
    """ "Number of internet sales" counted the order numbers of a table keyed by order number and line number:
    a count of one column of a two-column key is of that column's things (the orders), not of the table's rows."""
    import duckdb

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.bootstrap.labels import label
    from core2.warehouse.runner import DuckDBWarehouse

    con = duckdb.connect()
    con.execute("CREATE TABLE WebSale (OrderNumber VARCHAR, LineNumber INTEGER, Amount DECIMAL(10,2), "
                "OrderDate DATE)")
    con.execute("INSERT INTO WebSale SELECT 'SO' || (40000 + i // 3), i % 3 + 1, 5 + i % 70, "
                "DATE '2025-01-01' + CAST(i // 3 % 300 AS INTEGER) FROM range(0, 3000) t(i)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="web", options=BuildOptions(workers=1))
    reply = {"tables": {"WebSale": {"name": "Internet sale", "columns": {"OrderNumber": {"name": "Order number"},
                                                                       "LineNumber": {"name": "Line number"}}}}}
    label(model, lambda system, user: json.dumps(reply))
    sale = next(t for t in model.tables.values() if t.name == "WebSale")
    assert len(sale.primary_key) == 2, sale.primary_key
    counts = {m.business_name for m in model.measures.values() if m.table == sale.key and m.kind == "count"}
    assert "Number of orders" in counts and "Number of internet sales" in counts, counts
    distinct = [m.business_name for m in model.measures.values()
                if m.table == sale.key and m.kind == "count" and m.expr.agg == "count_distinct"]
    assert "Number of internet sales" not in distinct, distinct
