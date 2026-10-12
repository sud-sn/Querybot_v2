"""The names of things (categories, stores, months) reach the member list; people's names do not.

The new core keeps common values out of the AI's view and out of the list of names it matches a question
against when the old masking rule calls a column sensitive. That rule reads any column called "name" as a
person's name and any "city" as part of an address, so category, store, region, product and month names were
all held back: "snacks" never found the category Snacks, "Lakeshore" never found Lakeshore Store, and the
planner never saw a store's name. A column the rule reads as a name or a city is now held back only when it,
or the table it is on, is about people (customers, contacts, employees, patients); a column that names a
thing (a category name, a store's city, a company name) is the business's own vocabulary. Contact details,
ID numbers and free text stay held back as before, and Learn's own reading of the values still keeps people's
names from the AI.

Invented columns only.
"""

from __future__ import annotations

import pytest

from core2.bootstrap.inventory import InvColumn, Inventory, InvTable
from core2.bootstrap.service import _a_things_name, values_gate


def _table(name: str, *columns: str) -> InvTable:
    return InvTable(database="shop", schema="sales", name=name, source_key=f"SALES.{name}",
                    columns=[InvColumn(name=c, raw_type="nvarchar", data_type="text") for c in columns])


@pytest.mark.parametrize("table, column", [("categories", "category_name"), ("stores", "store_name"),
                                           ("regions", "region_name"), ("calendar", "month_name"),
                                           ("stores", "city"), ("Customers", "CompanyName"),
                                           ("suppliers", "supplier_name"), ("employees", "department_name")])
def test_a_things_name_is_not_a_persons(table, column):
    assert _a_things_name(table, column, "city" if column == "city" else "name")


@pytest.mark.parametrize("table, column", [("customers", "customer_name"), ("customers", "city"),
                                           ("Customers", "ContactName"), ("patients", "patient_name"),
                                           ("staff", "name"), ("technicians", "technician_name")])
def test_a_persons_name_or_city_stays_held(table, column):
    assert not _a_things_name(table, column, "city" if column == "city" else "name")


def test_contact_details_stay_held_whatever_the_table(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))
    inventory = Inventory(tables={
        "stores": _table("stores", "store_name", "city", "email", "phone"),
        "customers": _table("customers", "customer_name", "city", "segment"),
    })
    allowed = values_gate("acct-test", inventory, {})
    assert allowed("stores", "store_name") and allowed("stores", "city")
    assert not allowed("stores", "email") and not allowed("stores", "phone")
    assert not allowed("customers", "customer_name") and not allowed("customers", "city")
    assert allowed("customers", "segment")
