"""
A member is shown by its own name, never by the name of something it belongs to.

What a dimension's members are shown by was the first column named for the
member itself, or else any column called a name or a description. On a made-up
retailer's warehouse the employee has a first and a last name -- neither of
which tells employees apart -- and a department name, so the employees were
shown by their department: "reseller sales by employee" was compiled grouped by
DepartmentName and answered with one row, "Sales", every employee's sales
together under it. A geography was shown by its state (every city of a state
one member), and the calendar by its month name (every January one date).

A label is the member's own only when its name names the member: its words,
less the language and the fullness of the copy (English, French, full, short,
legal...), are the member's -- its table's entity, or what its key is named
for. DepartmentName in the employee table names a department; EnglishProductName
in the product table, the product. A member with no name of its own has none to
show: its breakdown is declined rather than answered wrong, and the department
is still a breakdown of its own.
"""

from __future__ import annotations

import json

import pytest


def _table(key: str, *columns: tuple[str, str]) -> dict:
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 10, "comment": "", "schema": "dbo", "database": "DW"}


SCHEMA = {
    "DW.dbo.DimEmployee": _table("EmployeeKey", ("EmployeeKey", "int"), ("FirstName", "nvarchar"),
                                 ("LastName", "nvarchar"), ("DepartmentName", "nvarchar"), ("Title", "nvarchar")),
    "DW.dbo.DimReseller": _table("ResellerKey", ("ResellerKey", "int"), ("ResellerName", "nvarchar"),
                                 ("BusinessType", "nvarchar")),
    "DW.dbo.DimDate": _table("DateKey", ("DateKey", "int"), ("FullDateAlternateKey", "date"),
                             ("EnglishMonthName", "nvarchar"), ("CalendarYear", "smallint")),
    "DW.dbo.FactResellerSales": _table("SalesOrderNumber", ("ResellerKey", "int"), ("EmployeeKey", "int"),
                                       ("OrderDateKey", "int"), ("SalesOrderNumber", "nvarchar"),
                                       ("SalesAmount", "money")),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    from core.semantic_model import load_semantic_model, write_semantic_model

    folder = tmp_path_factory.mktemp("own-name")
    (folder / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    write_semantic_model(schema_dir=str(folder), kb_dir=str(folder / "kb"))
    return {table["table"]: table for table in load_semantic_model(str(folder / "kb"))["tables"]}


def _shown_by(model: dict, table: str, key: str) -> str:
    return next(dimension.get("display_column") for dimension in model[table]["dimensions"]
                if dimension.get("source_key") == key)


class TestTheModel:

    def test_an_employee_is_not_shown_by_the_department(self, model):
        assert _shown_by(model, "FactResellerSales", "EmployeeKey") == ""

    def test_a_reseller_is_shown_by_its_own_name(self, model):
        assert _shown_by(model, "FactResellerSales", "ResellerKey") == "ResellerName"

    def test_a_date_is_not_shown_by_its_month(self, model):
        assert _shown_by(model, "DimDate", "DateKey") == ""

    def test_the_employee_table_is_not_shown_by_its_department(self, model):
        assert "DepartmentName" not in {
            d.get("display_column") for d in model["DimEmployee"]["dimensions"] if d.get("name") == "Employee"}

    def test_the_department_is_a_breakdown_of_its_own(self, model):
        assert "DepartmentName" in {d.get("display_column") for d in model["DimEmployee"]["dimensions"]}


class TestTheLabel:

    @pytest.mark.parametrize("columns,key,entity,shown", [
        (["EmployeeKey", "FirstName", "LastName", "DepartmentName"], "EmployeeKey", "Employee", ""),
        (["GeographyKey", "City", "StateProvinceName", "EnglishCountryRegionName"], "GeographyKey", "Geography", ""),
        (["ResellerKey", "ResellerName", "BusinessType"], "ResellerKey", "Reseller", "ResellerName"),
        (["ProductKey", "EnglishProductName", "ModelName"], "ProductKey", "Product", "EnglishProductName"),
        (["StoreKey", "FullName", "RegionName"], "StoreKey", "Store", "FullName"),
        (["CUSTOMER_ID", "CUSTOMER_NAME", "SEGMENT_NAME"], "CUSTOMER_ID", "Customer", "CUSTOMER_NAME"),
        (["WHS_DMS_KEY", "WHS_CD", "WHS_DSC"], "WHS_DMS_KEY", "Whs", "WHS_DSC"),
        (["SupplierKey", "Name", "CityName"], "SupplierKey", "Supplier", "Name"),
        # A key's own words name the member where the table's do not.
        (["SalesRepKey", "SalesRepName"], "SalesRepKey", "Employee", "SalesRepName"),
    ])
    def test_its_own(self, columns, key, entity, shown):
        from core.semantic_model import _display_field_for_columns

        assert _display_field_for_columns(columns, key, entity) == shown

    def test_with_no_member_to_read_it_against_nothing_is_ruled_out(self):
        from core.semantic_model import _display_field_for_columns

        assert _display_field_for_columns(["EmployeeKey", "DepartmentName"]) == "DepartmentName"
