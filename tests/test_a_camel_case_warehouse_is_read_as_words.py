"""
A camel-case warehouse is read as its words.

A warehouse named the common Kimball way -- FactInternetSales, DimProduct,
ProductKey, OrderDateKey -- was discovered as fifteen dimensions and not one
join. Every rule discovery reads a name by was written for underscore names
(PRODUCT_KEY, DIM_PRODUCT, _FCT), so a camel-case key was no key, a Fact table
no fact and DimDate no calendar; its date keys reached no calendar and its
tables were shown as "Factinternetsales". Of forty questions asked in English
and French of a made-up retailer's warehouse in this style, seven were
answered.

Discovery now reads a name as its words, whatever its case -- ProductKey is
PRODUCT_KEY and DimDate is DIM_DATE -- in the tables' roles, the keys and the
tables they reach, the calendar and the date keys that reach it, the labels of
date roles and the names tables are shown by. With it: a measure holds a
number (a territory's SalesTerritoryRegion is text); a column typed as a date
is the calendar's day, whatever its name ends with; a table named a fact is
never the calendar, read in its own case; a calendar's own key is no key to one
of its roles; and a date key already joined as a role is not joined again. An
underscore warehouse reads exactly as before.

tests/star_harness.py is a small warehouse in this style, set up by discovery
and an admin as tests/answer_harness.py sets up its own.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("camel-case")) as built:
        yield built


def _lines(year: int | None = None):
    for number, _line, placed, _shipped, product, _customer, territory, quantity in star.orders():
        if year is None or placed.year == year:
            yield number, placed, product, territory, quantity * star.PRODUCTS[product][4]


def _answered(answer: dict) -> list[dict]:
    assert answer["model_wrote_sql"] is False
    assert [body for kind, body in answer["replies"] if kind == "clarify"] == []
    return answer["rows"]


def _by(rows: list[dict], label) -> dict:
    """The one number of each row, by the label ``label`` reads off it."""
    out = {}
    for row in rows:
        numbers = [value for key, value in row.items() if isinstance(value, (int, float)) and key != "UNT_OF_MSR"]
        out[label(row)] = numbers[-1]
    return out


def _month(row: dict) -> str:
    value = next(iter(row.values()))
    return str(value)[:7]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Sales amount by month in 2025", "en"),
        ("Montant des ventes par mois en 2025", "fr"),
    ])
    def test_by_month_on_the_order_date(self, warehouse, question, lang):
        expected: dict = {}
        for _number, placed, _product, _territory, amount in _lines(2025):
            expected[f"{placed:%Y-%m}"] = expected.get(f"{placed:%Y-%m}", 0) + amount
        assert _by(_answered(star.ask(warehouse, question, lang)), _month) == expected

    @pytest.mark.parametrize("question,lang", [
        ("How many orders did we receive in 2025?", "en"),
        ("Combien de commandes avons-nous reçues en 2025 ?", "fr"),
    ])
    def test_orders_in_a_year(self, warehouse, question, lang):
        rows = _answered(star.ask(warehouse, question, lang))
        assert [list(row.values()) for row in rows] == [[len({number for number, *_ in _lines(2025)})]]

    def test_by_territory(self, warehouse):
        expected: dict = {}
        for _number, _placed, _product, territory, amount in _lines():
            region = star.TERRITORIES[territory][0]
            expected[region] = expected.get(region, 0) + amount
        rows = _answered(star.ask(warehouse, "Sales amount by sales territory region"))
        assert _by(rows, lambda row: next(iter(row.values()))) == expected


def _normalized() -> dict:
    from core.schema import _normalize_schema

    return _normalize_schema(star.SCHEMA)


def _edges(graph: dict) -> set[tuple[str, str, str]]:
    return {(rel["from_entity"], rel["from_column"], rel["to_entity"]) for rel in graph["relationships"]}


class TestDiscovery:

    def test_the_tables_roles(self):
        from core.table_role_classifier import classify_schema_tables

        roles = {fqn.split(".")[-1]: role.role for fqn, role in classify_schema_tables(_normalized()).items()}
        assert roles == {"DimDate": "date_dimension", "DimProductCategory": "dimension",
                         "DimProductSubcategory": "dimension", "DimProduct": "dimension",
                         "DimCustomer": "dimension", "DimSalesTerritory": "dimension",
                         "FactInternetSales": "fact", "FactProductInventory": "fact",
                         "FactProductInventoryMonthly": "fact"}

    def test_the_joins(self):
        from core.schema import build_entity_graph

        assert _edges(build_entity_graph(_normalized())) == {
            ("FactInternetSales", "ProductKey", "DimProduct"),
            ("FactInternetSales", "CustomerKey", "DimCustomer"),
            ("FactInternetSales", "SalesTerritoryKey", "DimSalesTerritory"),
            ("DimProduct", "ProductSubcategoryKey", "DimProductSubcategory"),
            ("DimProductSubcategory", "ProductCategoryKey", "DimProductCategory"),
            ("FactInternetSales", "OrderDateKey", "Order Date"),
            ("FactInternetSales", "DueDateKey", "Due Date"),
            ("FactInternetSales", "ShipDateKey", "Shipping Date"),
            ("FactProductInventory", "ProductKey", "DimProduct"),
            ("FactProductInventory", "DateKey", "Date"),
            ("FactProductInventoryMonthly", "ProductKey", "DimProduct"),
            ("FactProductInventoryMonthly", "MonthEndDateKey", "Month End Date"),
        }

    def test_the_tables_names(self):
        from core.schema import build_entity_graph

        names = {entity["entity_name"]: entity["display_name"] for entity in build_entity_graph(_normalized())["entities"]}
        assert (names["FactInternetSales"], names["DimProductSubcategory"], names["DimDate"]) == (
            "Internet Sales", "Product Subcategory", "Date")

    def test_the_date_keys_reach_the_calendar(self):
        from core.semantic_model import _date_roles

        schema = _normalized()
        fact = "SalesDW.dbo.FactInternetSales"
        assert sorted((role["fact_column"], role["name"], role["dimension_table"], role["dimension_key"],
                       role["date_value_column"]) for role in _date_roles(schema, fact, schema[fact])) == [
            ("DueDateKey", "Due Date", "dbo.DimDate", "DateKey", "FullDateAlternateKey"),
            ("OrderDateKey", "Order Date", "dbo.DimDate", "DateKey", "FullDateAlternateKey"),
            ("ShipDateKey", "Shipping Date", "dbo.DimDate", "DateKey", "FullDateAlternateKey"),
        ]


class TestAPlainDateKey:
    """A snapshot dated by the calendar's own plain key: its role is the
    Date, and the calendar reaches nothing through its own key."""

    SCHEMA = {
        "SalesDW.dbo.DimDate": star.SCHEMA["SalesDW.dbo.DimDate"],
        "SalesDW.dbo.FactProductInventory": star._table(
            "ProductKey", ("ProductKey", "int"), ("DateKey", "int"), ("UnitsBalance", "int"), ("UnitCost", "money")),
    }

    def test_one_join_by_the_date(self):
        from core.schema import _normalize_schema, build_entity_graph

        assert _edges(build_entity_graph(_normalize_schema(self.SCHEMA))) == {
            ("FactProductInventory", "DateKey", "Date")}


class TestTheWords:

    @pytest.mark.parametrize("name,words", [
        ("ProductKey", "PRODUCT_KEY"), ("DimProduct", "DIM_PRODUCT"), ("SKUCount", "SKU_COUNT"),
        ("customerId", "CUSTOMER_ID"), ("ON_HND_QTY", "ON_HND_QTY"), ("ITM_BAL_DLY_FCT", "ITM_BAL_DLY_FCT"),
    ])
    def test_a_name_as_its_words(self, name, words):
        from core.identifier_intelligence import identifier_words

        assert identifier_words(name) == words

    @pytest.mark.parametrize("table,kind", [
        ("FactInternetSales", "fact_table"), ("DimProduct", "dimension_table"), ("ProductDim", "dimension_table"),
        ("FactoryOutput", None), ("ITM_BAL_DLY_FCT", "fact_table"), ("WHS_DMS", "dimension_table"),
    ])
    def test_a_tables_kind_by_its_name(self, table, kind):
        from core.naming_convention import match_table_suffix

        rule = match_table_suffix(table)
        assert (rule.table_type if rule else None) == kind

    @pytest.mark.parametrize("table,shown", [
        ("FactInternetSales", "Internet Sales"), ("DIM_CUSTOMER", "Customer"), ("ITM_BAL_DLY_FCT", "Itm Bal Dly Fct"),
    ])
    def test_a_tables_shown_name(self, table, shown):
        from core.schema import _display_name_from_table

        assert _display_name_from_table(table) == shown

    def test_a_plain_date_key_is_the_date(self):
        from core.date_roles import detect_date_role

        assert detect_date_role("DateKey").label == detect_date_role("DATE_KEY").label == "Date"

    def test_the_calendars_day_is_the_column_typed_as_a_date(self):
        from core.date_roles import find_date_value_column

        assert find_date_value_column([{"name": "DateKey", "type": "int"},
                                       {"name": "FullDateAlternateKey", "type": "date"}]) == "FullDateAlternateKey"
        # A key that is not typed as a date is still no date.
        assert find_date_value_column([{"name": "DateKey", "type": "int"}, {"name": "LoadKey", "type": "int"}]) == ""

    @pytest.mark.parametrize("spelling", ["SALES_{}", "Sales{}"])
    def test_a_text_column_is_no_measure(self, spelling):
        from core.table_role_classifier import classify_table

        texts = [{"name": spelling.format(word), "type": "nvarchar"} for word in ("REGION", "COUNTRY", "GROUP")]
        role = classify_table("TERRITORIES", {"columns": [{"name": "TERRITORY_KEY", "type": "int"}, *texts],
                                               "pk_columns": ["TERRITORY_KEY"]})
        assert role.role == "dimension"

    def test_the_calendars_key_wherever_it_is_listed(self):
        from core.date_roles import find_date_dimension_key

        assert find_date_dimension_key([{"name": "FullDateAlternateKey", "type": "date"},
                                        {"name": "DateKey", "type": "int"}]) == "DateKey"

    def test_a_calendar_by_its_name_or_by_its_columns(self):
        from core.date_roles import is_date_dimension_table

        # Named a calendar, whatever its key is called.
        assert is_date_dimension_table("dbo.DimDate", [{"name": "DateSK", "type": "int"},
                                                       {"name": "TheDate", "type": "date"}]) is True
        # Not named one, but keyed and dated as one.
        assert is_date_dimension_table("dbo.TimeTable", [{"name": "DateKey", "type": "int"},
                                                         {"name": "TheDate", "type": "date"}]) is True

    def test_a_calendar_entity_by_its_table(self):
        """A graph discovered before names were read as words keeps the name
        it was shown by; its calendar is still a calendar."""
        from core.graph_resolver import is_date_role_entity

        assert is_date_role_entity({"entity_name": "DimDate", "display_name": "Dimdate", "table_name": "DimDate"})
        assert not is_date_role_entity({"entity_name": "DimProduct", "display_name": "Dimproduct",
                                        "table_name": "DimProduct"})

    def test_a_fact_is_never_the_calendar(self):
        from core.date_roles import is_date_dimension_table

        columns = [{"name": "ProductKey", "type": "int"}, {"name": "DateKey", "type": "int"},
                   {"name": "MovementDate", "type": "date"}, {"name": "UnitsBalance", "type": "int"}]
        assert is_date_dimension_table("dbo.FactProductInventory", columns) is False
        assert is_date_dimension_table("dbo.DimDate", [{"name": "DateKey", "type": "int"},
                                                       {"name": "FullDateAlternateKey", "type": "date"},
                                                       {"name": "CalendarYear", "type": "smallint"},
                                                       {"name": "MonthNumberOfYear", "type": "tinyint"}]) is True
