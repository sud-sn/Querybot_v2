"""
A second small warehouse to ask questions of: a sales star in camel case.

tests/answer_harness.py keeps a distribution mart in the underscore, abbreviated
naming of one ERP family. This one is spelled the other common way -- a Kimball
star in PascalCase, in full words: FactInternetSales, DimProduct, OrderDateKey
-- with a snowflaked product (product, subcategory, category), a customer, a
sales territory, and a calendar keyed yyyymmdd that the fact reaches as its
order, due and ship dates. Rows are invented.

The tenant is set up as answer_harness sets up its own, by discovery and an
admin: the schema files written by the Azure discovery writer, the value index,
the join graph, every join checked against the rows, the admin choosing the
star-schema vocabulary, accepting the joins the data vouches for, approving the
date roles with the order date as the everyday one, and registering the
business's metrics. Questions go through the same boundaries: DuckDB for the
warehouse, a marker for the SQL writer.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from pathlib import Path

from tests import answer_harness as harness

ACCOUNT = f"acct-star-{os.urandom(3).hex()}"
CONNECTION = {"db_type": "azure_sql", "name": "star",
              "credentials": {"server": "star", "user": "u", "password": "p", "database": "SalesDW"}}
_saved: dict = {}


def _table(key: str, *columns: tuple[str, str]) -> dict:
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 0, "comment": "", "schema": "dbo", "database": "SalesDW"}


SCHEMA = {
    "SalesDW.dbo.DimDate": _table(
        "DateKey", ("DateKey", "int"), ("FullDateAlternateKey", "date"), ("DayNumberOfWeek", "tinyint"),
        ("EnglishDayNameOfWeek", "nvarchar"), ("FrenchDayNameOfWeek", "nvarchar"), ("MonthNumberOfYear", "tinyint"),
        ("EnglishMonthName", "nvarchar"), ("FrenchMonthName", "nvarchar"), ("CalendarQuarter", "tinyint"),
        ("CalendarYear", "smallint")),
    "SalesDW.dbo.DimProductCategory": _table(
        "ProductCategoryKey", ("ProductCategoryKey", "int"), ("EnglishProductCategoryName", "nvarchar"),
        ("FrenchProductCategoryName", "nvarchar")),
    "SalesDW.dbo.DimProductSubcategory": _table(
        "ProductSubcategoryKey", ("ProductSubcategoryKey", "int"), ("EnglishProductSubcategoryName", "nvarchar"),
        ("FrenchProductSubcategoryName", "nvarchar"), ("ProductCategoryKey", "int")),
    # A product's prices are money on the product's own row -- numbers, but not
    # what the product table holds facts about.
    "SalesDW.dbo.DimProduct": _table(
        "ProductKey", ("ProductKey", "int"), ("ProductAlternateKey", "nvarchar"), ("EnglishProductName", "nvarchar"),
        ("FrenchProductName", "nvarchar"), ("ProductSubcategoryKey", "int"), ("Color", "nvarchar"),
        ("StandardCost", "money"), ("ListPrice", "money"), ("DealerPrice", "money")),
    "SalesDW.dbo.DimCustomer": _table(
        "CustomerKey", ("CustomerKey", "int"), ("CustomerAlternateKey", "nvarchar"), ("FirstName", "nvarchar"),
        ("LastName", "nvarchar"), ("Gender", "nvarchar"), ("YearlyIncome", "money")),
    # A territory's names say "sales" and are text: no measure.
    "SalesDW.dbo.DimSalesTerritory": _table(
        "SalesTerritoryKey", ("SalesTerritoryKey", "int"), ("SalesTerritoryRegion", "nvarchar"),
        ("SalesTerritoryCountry", "nvarchar"), ("SalesTerritoryGroup", "nvarchar")),
    "SalesDW.dbo.FactInternetSales": _table(
        "SalesOrderNumber", ("ProductKey", "int"), ("OrderDateKey", "int"), ("DueDateKey", "int"),
        ("ShipDateKey", "int"), ("CustomerKey", "int"), ("SalesTerritoryKey", "int"),
        ("SalesOrderNumber", "nvarchar"), ("SalesOrderLineNumber", "tinyint"), ("OrderQuantity", "smallint"),
        ("UnitPrice", "money"), ("SalesAmount", "money"), ("TotalProductCost", "money")),
}

_MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
              "novembre", "décembre"]
_DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

CATEGORIES = {1: ("Camping", "Camping"), 2: ("Climbing", "Escalade"), 3: ("Clothing", "Vêtements")}
SUBCATEGORIES = {1: ("Tents", "Tentes", 1), 2: ("Stoves", "Réchauds", 1), 3: ("Ropes", "Cordes", 2),
                 4: ("Helmets", "Casques", 2), 5: ("Jackets", "Vestes", 3), 6: ("Gloves", "Gants", 3)}
# key: (code, English name, French name, subcategory, list price, standard cost)
PRODUCTS = {
    1: ("TE-1", "Ridge Tent", "Tente Ridge", 1, 420.0, 250.0), 2: ("TE-2", "Summit Tent", "Tente Summit", 1, 610.0, 380.0),
    3: ("ST-1", "Trail Stove", "Réchaud Trail", 2, 95.0, 50.0), 4: ("RO-1", "Alpine Rope", "Corde Alpine", 3, 210.0, 120.0),
    5: ("HE-1", "Canyon Helmet", "Casque Canyon", 4, 80.0, 44.0), 6: ("HE-2", "Peak Helmet", "Casque Peak", 4, 130.0, 70.0),
    7: ("JA-1", "Storm Jacket", "Veste Storm", 5, 300.0, 170.0), 8: ("GL-1", "Grip Gloves", "Gants Grip", 6, 35.0, 15.0),
}
# Two customers share a first name: a customer is never told apart by it.
CUSTOMERS = {1: ("C001", "Alex", "Martin", "F", 60000.0), 2: ("C002", "Sam", "Roy", "M", 85000.0),
             3: ("C003", "Robin", "Lee", "F", 120000.0), 4: ("C004", "Alex", "Weber", "M", 45000.0)}
TERRITORIES = {1: ("Northwest", "United States", "North America"), 2: ("Canada", "Canada", "North America"),
               3: ("France", "France", "Europe")}


def _key(day: dt.date) -> int:
    return day.year * 10000 + day.month * 100 + day.day


def orders() -> list[tuple]:
    """Order lines: two orders a month from January 2024 to December 2025,
    each shipped a few days after it was placed -- across a month's end for
    the second -- and due two weeks after. (order number, line, day, ship
    day, product, customer, territory, quantity)"""
    found = []
    number = 5000
    for year in (2024, 2025):
        for month in range(1, 13):
            for n, day_of_month in enumerate((5, 27)):
                number += 1
                placed = dt.date(year, month, day_of_month)
                shipped = placed + dt.timedelta(days=3 + 4 * n)
                for line in range(1, 2 + (month + n) % 2 + 1):
                    product = 1 + (month * 3 + line + n + year) % len(PRODUCTS)
                    customer = 1 + (month + line + n) % len(CUSTOMERS)
                    territory = 1 + (customer + n) % len(TERRITORIES)
                    quantity = 1 + (month + line) % 3
                    found.append((f"SO{number}", line, placed, shipped, product, customer, territory, quantity))
    return found


def rows() -> dict[str, list[tuple]]:
    days = []
    day = dt.date(2024, 1, 1)
    while day <= dt.date(2025, 12, 31):
        days.append((_key(day), day.isoformat(), day.isoweekday() % 7 + 1, _DAYS_EN[day.weekday()],
                     _DAYS_FR[day.weekday()], day.month, _MONTHS_EN[day.month - 1], _MONTHS_FR[day.month - 1],
                     (day.month - 1) // 3 + 1, day.year))
        day += dt.timedelta(days=1)
    facts = []
    for number, line, placed, shipped, product, customer, territory, quantity in orders():
        _code, _en, _fr, _sub, price, cost = PRODUCTS[product]
        facts.append((product, _key(placed), _key(placed + dt.timedelta(days=14)), _key(shipped), customer, territory,
                      number, line, quantity, price, quantity * price, quantity * cost))
    return {
        "DimDate": days,
        "DimProductCategory": [(k, en, fr) for k, (en, fr) in CATEGORIES.items()],
        "DimProductSubcategory": [(k, en, fr, c) for k, (en, fr, c) in SUBCATEGORIES.items()],
        "DimProduct": [(k, code, en, fr, sub, "Black", cost, price, round(price * 0.6, 2))
                       for k, (code, en, fr, sub, price, cost) in PRODUCTS.items()],
        "DimCustomer": [(k, code, first, last, gender, income)
                        for k, (code, first, last, gender, income) in CUSTOMERS.items()],
        "DimSalesTerritory": [(k, *names) for k, names in TERRITORIES.items()],
        "FactInternetSales": facts,
    }


# What the business reports, as its administrator registers it.
METRICS = [
    {"name": "Sales Amount", "sql_template": "SUM(SalesAmount)", "base_table": "dbo.FactInternetSales",
     "formula_type": "expression", "category": "sales",
     "synonyms": "sales amount, sales, revenue, online sales, montant des ventes, ventes, chiffre d'affaires",
     "description": "Online sales to customers."},
    {"name": "Order Quantity", "sql_template": "SUM(OrderQuantity)", "base_table": "dbo.FactInternetSales",
     "formula_type": "expression", "category": "sales",
     "synonyms": "order quantity, units ordered, units sold, quantité commandée, unités vendues",
     "description": "Units ordered online."},
    {"name": "Number of Orders", "sql_template": "COUNT(DISTINCT SalesOrderNumber)",
     "base_table": "dbo.FactInternetSales", "formula_type": "expression", "category": "sales",
     "synonyms": "number of orders, order count, orders, nombre de commandes, commandes",
     "description": "Online orders; an order has one or more lines."},
    # Measures named for a dimension's members: their names are not a breakdown.
    {"name": "Number of Buying Customers", "sql_template": "COUNT(DISTINCT CustomerKey)",
     "base_table": "dbo.FactInternetSales", "formula_type": "expression", "category": "sales",
     "synonyms": "buying customers, active customers, clients acheteurs",
     "description": "Distinct customers with at least one online order."},
    {"name": "Number of Products Sold", "sql_template": "COUNT(DISTINCT ProductKey)",
     "base_table": "dbo.FactInternetSales", "formula_type": "expression", "category": "sales",
     "synonyms": "products sold, distinct products sold, produits vendus",
     "description": "Distinct products on at least one online order line."},
]


def build_tenant(root: Path) -> harness.Warehouse:
    """The warehouse and a tenant over it, set up by discovery and an admin."""
    from unittest.mock import patch

    import store
    from core.graph_autopopulate import auto_populate_from_schema
    from core.relationship_validator import profile_suggested_relationships
    from core.semantic_contract import write_contract
    from core.semantic_model import load_semantic_model, patch_date_role, set_default_date_role, write_semantic_model
    from core.unknown_members import detect_unknown_members
    from core.value_index import build_value_index
    from core.vocab_packs import forget_account_vocab, vocab_for_account

    warehouse = harness.Warehouse(root / "warehouse.duckdb", SCHEMA, rows())
    store.init_db()
    store.upsert_client(ACCOUNT, "Star Harness Ltd")
    store.save_compliance_profile(ACCOUNT, mode="standard", industry="standard")
    db_id = store.save_db_config(CONNECTION["db_type"], CONNECTION["name"], CONNECTION["credentials"])
    _saved["id"] = db_id
    # The admin's choice for a warehouse in this style: the Kimball vocabulary.
    store.update_client_meta(ACCOUNT, db_config_id=db_id, chat_ui_enabled=1,
                             erp_packs=json.dumps(["generic_star_schema"]))
    schema_dir = root / "schema"
    harness._schema_files(schema_dir, warehouse, SCHEMA)
    store.update_client_state(ACCOUNT, "READY", {"schema_dir": str(schema_dir), "kb_dir": str(schema_dir)})
    forget_account_vocab(ACCOUNT)

    def probe(db_type, raw_cfg, sql, *, timeout_seconds=20, max_rows=200):
        return [tuple(row.values()) for row in warehouse.query(sql, max_rows)]

    with patch("core.relationship_validator.run_probe", probe):
        build_value_index(ACCOUNT, {}, "azure_sql", str(schema_dir), vocab=vocab_for_account(ACCOUNT),
                          run_query_fn=lambda creds, db_type, sql, max_rows=200: warehouse.query(sql, max_rows))
        auto_populate_from_schema(ACCOUNT, str(schema_dir))
        profile_suggested_relationships(ACCOUNT)
        detect_unknown_members(ACCOUNT)
    write_semantic_model(schema_dir=str(schema_dir), kb_dir=str(schema_dir), account_id=ACCOUNT)
    harness._accept_what_the_data_vouches_for(ACCOUNT)
    for role in load_semantic_model(str(schema_dir)).get("date_roles") or []:
        if not str(role.get("fact_table") or "").upper().endswith("FACTINTERNETSALES"):
            continue
        patch_date_role(kb_dir=str(schema_dir), fact_table=role["fact_table"], fact_column=role["fact_column"],
                        dimension_table=role.get("dimension_table") or "",
                        dimension_key=role.get("dimension_key") or "",
                        business_role=role.get("business_role") or "", name=role.get("name") or "",
                        date_value_column=role.get("date_value_column") or "",
                        date_key_type=role.get("date_key_type") or "", status="approved")
        if role["fact_column"] == "OrderDateKey":
            set_default_date_role(str(schema_dir), role["fact_table"], "OrderDateKey")
    for metric in METRICS:
        store.save_metric(ACCOUNT, dict(metric))
    write_contract(ACCOUNT, str(schema_dir))
    store.delete_db_config(db_id)
    return warehouse


def saved_connection() -> dict:
    return {"id": _saved.get("id"), **CONNECTION}


def ask(warehouse: harness.Warehouse, question: str, lang: str = "en") -> dict:
    return harness.ask(warehouse, question, lang, account=ACCOUNT, connection=saved_connection())


_built: dict = {}


@contextlib.contextmanager
def tenant_in(root: Path):
    """The tenant, built once per store as answer_harness builds its own."""
    import store

    previous = Path.cwd()
    if "warehouse" not in _built or store.get_client(ACCOUNT) is None:
        os.chdir(root)
        try:
            _built["warehouse"] = build_tenant(root)
        except BaseException:
            os.chdir(previous)
            raise
        _built["root"] = root
    os.chdir(_built["root"])
    try:
        yield _built["warehouse"]
    finally:
        os.chdir(previous)
