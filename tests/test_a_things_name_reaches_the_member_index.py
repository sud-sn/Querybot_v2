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


@pytest.mark.parametrize("table, column", [("ingredients", "ingredient_name"), ("formulas", "compound_name"),
                                           ("fills", "drug_name"), ("products", "generic_name")])
def test_a_drugs_name_in_a_catalogue_is_a_things(table, column):
    # "VersaBase cream" is an ingredient the pharmacy buys: the planner must see it to put "lots of 'VersaBase
    # cream'" on the ingredient, not on another field it then refuses as having no such name.
    assert _a_things_name(table, column, "drug_name")


@pytest.mark.parametrize("table, column", [("patients", "medication_name"), ("patient_meds", "drug_name"),
                                           ("members", "rx_name")])
def test_a_drugs_name_about_a_person_stays_held(table, column):
    assert not _a_things_name(table, column, "drug_name")


def test_a_drugs_name_reaches_the_index_beside_its_catalogue(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))
    inventory = Inventory(tables={
        "ingredients": _table("ingredients", "ingredient_name", "fda_listed_name"),
        "patients": _table("patients", "patient_name", "medication_name"),
    })
    allowed = values_gate("acct-test", inventory, {})
    assert allowed("ingredients", "ingredient_name")
    assert not allowed("patients", "patient_name") and not allowed("patients", "medication_name")


# Discovery masks too: its own patterns ("auto") wrote store_name and city into masked_fields of the sample
# workbook's stores sheet, and Learn held every masked column back, so "Lakeside" and "Dish Soap" were never found.

def _masked(name: str, mode: str, kinds: dict[str, str], *columns: str) -> InvTable:
    table = _table(name, *columns)
    table.masked, table.mask_mode, table.mask_kinds = set(kinds), mode, dict(kinds)
    return table


def test_a_things_name_discovery_masked_by_its_own_patterns_reaches_the_index(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))
    inventory = Inventory(tables={
        "stores": _masked("stores", "auto", {"store_name": "name", "city": "city"}, "store_name", "city", "region"),
        "products": _masked("products", "auto", {"product_name": "name"}, "product_name", "category"),
        "customers": _masked("customers", "auto", {"customer_name": "name", "city": "city"},
                             "customer_name", "city", "segment"),
    })
    allowed = values_gate("acct-test", inventory, {})
    assert allowed("stores", "store_name") and allowed("stores", "city") and allowed("products", "product_name")
    assert not allowed("customers", "customer_name") and not allowed("customers", "city")


# A distributor's customers are businesses: "Hearthstone Home Centres", "Stonebridge Mercantile", "Quincaillerie
# Saint-Roch". Discovery's own patterns mask any "customer name", and that mask was final: the values were never
# read, so no customer was ever found by name ("revenue for Hearthstone Home Centres" narrowed to nothing).
BUSINESSES = ["Hearthstone Home Centres", "Lakeshore Home & Hardware", "Mosaic Living", "Stonebridge Mercantile",
              "Bayside Housewares", "Quincaillerie Saint-Roch", "Comptoir Cuisine du Plateau", "Harbourfront Hotel",
              "Grand Pacific Lodge", "Apex Corporate Gifts", "Keystone Promotions Inc.", "Prairie Outfitters",
              "Aurora Online Mall", "NorthBay Deals", "Polar Value Stores", "Tamarack Kitchenware",
              "Boutique Culinaire Saint-Denis", "Auberge du Lac", "Maison Laval Plus", "Golden Horseshoe Home"]
PEOPLE = ["Olivia Smith", "Liam Tremblay", "Chloé Gagnon", "Raj Patel", "Mei Chen", "Kofi Okonkwo", "Leah Fraser",
          "Diego Santos", "Nora Lavoie", "Ethan Murphy"]


def test_a_customer_name_discovery_masked_reaches_the_index_when_its_values_are_businesses(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))
    kinds = {"customer_name": "name", "contact_name": "name", "city": "city"}
    inventory = Inventory(tables={
        "customers": _masked("customers", "auto", kinds, "customer_name", "contact_name", "city", "segment"),
        "buyers": _masked("buyers", "auto", {"customer_name": "name"}, "customer_name", "segment"),
    })
    values = {("customers", "customer_name"): BUSINESSES, ("customers", "contact_name"): PEOPLE,
              ("buyers", "customer_name"): PEOPLE}
    allowed = values_gate("acct-test", inventory, {}, sample=lambda table, column: values.get((table, column), []))
    assert allowed("customers", "customer_name")
    assert not allowed("customers", "contact_name") and not allowed("buyers", "customer_name")
    assert not allowed("customers", "city")
    # A mask the admin chose is final, whatever the values are.
    inventory.tables["customers"].mask_mode = "selective"
    allowed = values_gate("acct-test", inventory, {}, sample=lambda table, column: values.get((table, column), []))
    assert not allowed("customers", "customer_name")


def test_a_column_an_admin_masked_stays_held_whatever_it_names(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))
    inventory = Inventory(tables={
        "stores": _masked("stores", "selective", {"store_name": "name"}, "store_name", "region"),
        "products": _table("products", "product_name"),
    })
    inventory.tables["products"].mask_all, inventory.tables["products"].mask_mode = True, "all"
    allowed = values_gate("acct-test", inventory, {})
    assert not allowed("stores", "store_name") and allowed("stores", "region")
    assert not allowed("products", "product_name")


def test_a_model_read_again_keeps_the_same_rule():
    from core2.bootstrap.service import held_by_masking, mask_as_discovered
    from core2.model.schema import Column, SemanticModel, Table

    stores = _masked("stores", "auto", {"store_name": "name", "email": "email"}, "store_name", "email")
    assert not held_by_masking(stores, "store_name") and held_by_masking(stores, "email")
    model = SemanticModel(client_id="t", db_type="duckdb", tables={"stores": Table(key="stores", name="stores")},
                          columns={f"stores.{c}": Column(key=f"stores.{c}", table="stores", name=c, data_type="text",
                                                                   values_allowed=True)
                                   for c in ("store_name", "email")})
    mask_as_discovered(model, Inventory(tables={"stores": stores}))
    assert model.columns["stores.store_name"].values_allowed and not model.columns["stores.email"].values_allowed


def test_a_customer_name_learn_found_to_be_businesses_stays_named_when_the_model_is_read_again():
    # Learn read "Hearthstone Home Centres" and the rest and found businesses; the model each question is answered
    # with applies discovery's masks again, and held the same column back: "What did Hearthstone Home Centres buy?"
    # was answered for every customer.
    from core2.bootstrap.service import mask_as_discovered
    from core2.model.schema import Column, SemanticModel, Table

    names = ("customer_name", "contact_name", "email")
    customers = _masked("customers", "auto", {"customer_name": "name", "contact_name": "name", "email": "email"},
                        *names)
    learned = {"customer_name": True, "contact_name": False, "email": False}

    def model() -> SemanticModel:
        return SemanticModel(client_id="t", db_type="duckdb", tables={"customers": Table(key="customers",
                                                                                         name="customers")},
                             columns={f"customers.{c}": Column(key=f"customers.{c}", table="customers", name=c,
                                                               data_type="text", values_allowed=learned[c])
                                      for c in names})

    read_again = model()
    mask_as_discovered(read_again, Inventory(tables={"customers": customers}))
    assert read_again.columns["customers.customer_name"].values_allowed
    assert not read_again.columns["customers.contact_name"].values_allowed
    assert not read_again.columns["customers.email"].values_allowed
    # A mask the admin chose after Learn holds it back at once.
    customers.mask_mode = "selective"
    read_again = model()
    mask_as_discovered(read_again, Inventory(tables={"customers": customers}))
    assert not read_again.columns["customers.customer_name"].values_allowed
