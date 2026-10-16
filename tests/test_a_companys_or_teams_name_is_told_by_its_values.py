"""A customer or buyer that is a company or a team is named to the AI; a person never is.

The values gate holds back any "name" column on a table about customers, buyers or owners, as a person's name.
Where the customers are businesses ("Harbor Trading", "Quarry Supply Co") and the buyers are buying teams
("Facilities Desk", "Capital Projects"), that kept their names from the planner: "for 'Harbor Trading'" was
refused as a name the data does not have. Learn now reads a sample of such a column's own values, in the app,
and lets it through only when nearly all of them are the names of businesses or teams and none reads as a
person's ("Dr. …", "…, MD"). A column of people's names, a contact's name, or any column read without a sample
stays held back.

Invented tables and names only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap import personal
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.service import values_gate, value_sampler
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture()
def warehouse():
    con = duckdb.connect()
    con.execute("create table buyers (buyer_id integer, buyer_name varchar)")
    con.execute("insert into buyers values (1, 'Facilities Desk'), (2, 'Capital Projects'), "
                "(3, 'Fleet Buying Team'), (4, 'Indirect Spend')")
    con.execute("create table customers (customer_id integer, customer_name varchar, segment varchar)")
    con.execute("insert into customers values " + ", ".join(
        f"({i}, '{name}', 'Trade')" for i, name in enumerate(
            ["Harbor Trading", "Pine Ridge Foods", "Quarry Supply Co", "Bellweather Grocers", "Tamarack Wholesale",
             "Juno & Vale Market", "Saltmarsh Provisions", "Corvid Distribution Ltd", "Brindle Goods",
             "Ashgrove Retail"])))
    con.execute("create table clients (client_id integer, client_name varchar)")
    con.execute("insert into clients values (1, 'Lena Marsh'), (2, 'Omar Fitch'), (3, 'Ivy Collard'), "
                "(4, 'Theo Brandt'), (5, 'Ruth Okonkwo')")
    con.execute("create table owners (owner_id integer, owner_name varchar)")
    con.execute("insert into owners values (1, 'Dr. Ansel Moore'), (2, 'Harbor Trading'), (3, 'Pine Ridge Foods'), "
                "(4, 'Quarry Supply Co'), (5, 'Tamarack Wholesale'), (6, 'Brindle Goods')")
    con.execute("create table partners (partner_id integer, partner_name varchar, contact_name varchar)")
    con.execute("insert into partners values (1, 'Harbor Trading', 'Billing Desk'), "
                "(2, 'Brindle Goods', 'Accounts Team'), (3, 'Ashgrove Retail', 'Service Desk')")
    yield DuckDBWarehouse(con)
    con.close()


@pytest.fixture()
def no_clearance(monkeypatch):
    import core.value_index as value_index

    monkeypatch.setattr(value_index, "_clearance_gate", lambda account_id: (None, ""))


def test_businesses_and_teams_are_named_where_people_are_not(warehouse, no_clearance):
    inventory = from_duckdb(warehouse)
    allowed = values_gate("acct-test", inventory, {}, sample=value_sampler(warehouse, inventory))
    key = {t.name: k for k, t in inventory.tables.items()}
    assert allowed(key["buyers"], "buyer_name")
    assert allowed(key["customers"], "customer_name")
    assert not allowed(key["clients"], "client_name")          # people
    assert not allowed(key["owners"], "owner_name")            # one reads as a person's: held back
    assert not allowed(key["partners"], "contact_name")        # a contact is a person, whatever is written
    assert allowed(key["partners"], "partner_name")


def test_without_a_sample_a_customers_name_stays_held(warehouse, no_clearance):
    inventory = from_duckdb(warehouse)
    allowed = values_gate("acct-test", inventory, {})
    key = {t.name: k for k, t in inventory.tables.items()}
    assert not allowed(key["buyers"], "buyer_name") and not allowed(key["customers"], "customer_name")


def test_each_column_is_read_once(warehouse, no_clearance):
    inventory = from_duckdb(warehouse)
    read: list[tuple[str, str]] = []
    sample = value_sampler(warehouse, inventory)

    def counting(table_key: str, column: str) -> list[str]:
        read.append((table_key, column))
        return sample(table_key, column)

    allowed = values_gate("acct-test", inventory, {}, sample=counting)
    key = {t.name: k for k, t in inventory.tables.items()}
    for _ in range(3):
        allowed(key["customers"], "customer_name")
        allowed(key["customers"], "segment")
    assert read == [(key["customers"], "customer_name")]


@pytest.mark.parametrize("values", [
    ["Harbor Trading", "Pine Ridge Foods", "Quarry Supply Co", "Brindle Goods"],
    ["Facilities Desk", "Capital Projects", "Fleet Buying Team", "Indirect Spend", "MRO Desk"],
    ["Delta Legal 1", "Ember Design 2", "Iris Labs 1", "Kestrel Clinics 3", "Orchid Energy 1"],
])
def test_names_of_businesses_and_teams(values):
    assert personal.organisations_named(values)


@pytest.mark.parametrize("values", [
    ["Lena Marsh", "Omar Fitch", "Ivy Collard", "Theo Brandt"],
    ["Luis Robinson, PA", "Andrew Wong, DO", "Harbor Trading", "Pine Ridge Foods", "Brindle Goods"],
    ["Dr. Riley Oakley 2", "Dr. Emerson Ingram 8"],
    ["Harbor Trading", "Lena Marsh", "Omar Fitch", "Ivy Collard", "Theo Brandt"],
    ["Harbor Trading", "Brindle Goods"],                     # too few to tell
    [],
])
def test_not_told_apart_from_people(values):
    assert not personal.organisations_named(values)
