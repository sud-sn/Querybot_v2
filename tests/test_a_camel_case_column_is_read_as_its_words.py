"""
A camel-case column is read as its words.

Once a camel-case warehouse's tables and joins were found, its columns still
read as nothing: the semantic model called SalesAmount, OrderQuantity and
UnitsBalance plain attributes, ProductKey no key, and EnglishProductName no
name to show a product by -- its rules are written for underscore short forms
(_AMT, _QTY, _KEY, _NM, _DMS_KEY). The model held no measure and no breakdown,
and a key's label was spelled from its physical name ("Product Key").

A column is now read as its words, and a last word spelled out as its short
form: SalesAmount is an amount as SALES_AMT is, UnitsBalance a balance,
EnglishProductName a name, ProductKey a key to the product, which is what it
is called. A key in any convention (CUSTOMER_KEY, customer_id) keys a
dimension; the source system's own key for a member (CustomerAlternateKey) is
its business code. And what a member is shown by must tell members apart: a
person's first or last name is never it -- grouped by first name, two
customers called Alex are one -- nor is a person's official number its code.
An underscore warehouse's columns read exactly as before.

tests/star_harness.py is a warehouse in this style; two of its customers share
a first name.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("camel-case-columns")) as built:
        yield built


def _totals(index: int) -> dict:
    """Sales amount by product (index 4) or customer (index 5) of an order line."""
    totals: dict = {}
    for line in star.orders():
        key = line[index]
        totals[key] = totals.get(key, 0) + line[7] * star.PRODUCTS[line[4]][4]
    return totals


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Top 3 products by sales amount", "en"),
        ("Les 3 meilleurs produits par montant des ventes", "fr"),
    ])
    def test_the_top_products(self, warehouse, question, lang):
        answer = star.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        best = sorted(_totals(4).items(), key=lambda item: -item[1])[:3]
        # Each product by its name in the reader's language.
        assert [(row["PRODUCT"], row["SALES_AMOUNT"]) for row in answer["rows"]] == [
            (star.PRODUCTS[key][2 if lang == "fr" else 1], total) for key, total in best]

    def test_customers_are_never_told_apart_by_a_first_name(self, warehouse):
        answer = star.ask(warehouse, "Top 3 customers by sales amount")
        # Grouped by each customer's own key, whatever the customer is called.
        grouped_by = answer["sql"].split("GROUP BY", 1)[-1].split("ORDER BY")[0]
        assert "CustomerAlternateKey" in grouped_by and "FirstName" not in grouped_by.split(",")[0]
        # Whatever is answered is a customer's own total.
        per_customer = set(_totals(5).values())
        assert all(set(row.values()) & per_customer for row in answer["rows"])


class TestTheColumns:

    @pytest.mark.parametrize("column,role", [
        ("SalesAmount", "measure"), ("OrderQuantity", "measure"), ("UnitsBalance", "semi_additive"),
        ("EnglishProductName", "display"), ("ProductKey", "surrogate_fk"), ("SalesOrderNumber", "identifier"),
        ("BusinessType", "type"), ("ON_HND_QTY", "measure"), ("WHS_DSC", "display"), ("NET_SLS_AMT", "measure"),
    ])
    def test_a_columns_rule(self, column, role):
        from core.naming_convention import match_column_suffix

        assert match_column_suffix(column).role == role

    @pytest.mark.parametrize("column,data_type,role", [
        ("ProductKey", "int", "dimension_key"), ("customer_id", "int", "dimension_key"),
        ("CustomerAlternateKey", "nvarchar", "identifier"), ("CustomerAlternateKey", "", "identifier"),
        ("RegionID", "varchar", "identifier"), ("PriceBand", "nvarchar", "attribute"),
        ("SalesAmount", "money", "measure"),
        ("UnitsBalance", "int", "measure"), ("SalesTerritoryRegion", "nvarchar", "attribute"),
        ("WHS_DMS_KEY", "int", "dimension_key"), ("ITM_BAL_DLY_FCT_KEY", "bigint", "surrogate_key"),
        ("ON_HND_QTY", "decimal", "measure"),
    ])
    def test_a_columns_role(self, column, data_type, role):
        from core.schema_enrichment import _role_for_column

        assert _role_for_column(column, data_type)[0] == role

    @pytest.mark.parametrize("key,stem", [("CustomerDimKey", "CUSTOMER"), ("ProductSK", "PRODUCT"),
                                          ("WHS_DMS_KEY", "WHS")])
    def test_a_keys_stem(self, key, stem):
        from core.vocab_packs import strip_dimension_key_suffix

        assert strip_dimension_key_suffix(key) == stem


class TestTheLabels:

    @pytest.mark.parametrize("table,name", [
        ("DimProductCategory", "Product Category"), ("FactInternetSales", "Internet Sales"), ("ITM_GRP_DMS", "Itm Grp"),
    ])
    def test_a_tables_entity(self, table, name):
        from core.semantic_model import _entity_name

        assert _entity_name(table) == name

    @pytest.mark.parametrize("key,role", [("ProductKey", "product"), ("SalesTerritoryKey", "sales_territory")])
    def test_a_keys_role(self, key, role):
        from core.semantic_model import _business_role_from_column

        assert _business_role_from_column(key) == role


class TestWhatAMemberIsShownBy:

    @pytest.mark.parametrize("columns,shown", [
        (["ProductKey", "EnglishProductName", "FrenchProductName"], "EnglishProductName"),
        (["CustomerKey", "FirstName", "LastName", "EmailAddress"], ""),
        (["EmployeeKey", "FirstName", "Surname", "Title"], ""),
        (["CUSTOMER_KEY", "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS"], ""),
        (["EMPLOYEE_KEY", "SURNAME", "TITLE"], ""),
        (["WHS_DMS_KEY", "WHS_CD", "WHS_DSC"], "WHS_DSC"),
    ])
    def test_its_name(self, columns, shown):
        from core.semantic_model import _display_field_for_columns

        assert _display_field_for_columns(columns) == shown

    def test_the_name_named_for_the_entity_first(self):
        from core.semantic_model import _display_field_for_columns

        assert _display_field_for_columns(["ProductKey", "ModelName", "ProductName"], "ProductKey") == "ProductName"

    @pytest.mark.parametrize("columns,code", [
        (["CustomerKey", "CustomerAlternateKey", "FirstName"], "CustomerAlternateKey"),
        (["EmployeeKey", "EmployeeNationalIDAlternateKey", "FirstName"], ""),
        (["WHS_DMS_KEY", "WHS_CD", "WHS_DSC"], "WHS_CD"),
    ])
    def test_its_code(self, columns, code):
        from core.semantic_model import _code_field_for_columns

        assert _code_field_for_columns(columns) == code
