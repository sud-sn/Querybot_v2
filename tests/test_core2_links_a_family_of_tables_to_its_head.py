"""Tables that share one key point at the one that says most about it; counts and a table's own codes point nowhere.

On a public sample of a manufacturer's operational database, with its keys withheld, Learn linked a person's email
address to their password (both kept once per person, under the same number) and left the person table pointing
at the email table; a reseller's number of employees pointed at the employees; and the product categories' own
alternate key (1, 2, 3, 4) pointed at the products. Questions that follow such links answer with wrong numbers.

So: tables holding the same members under one key are a family whose head says most about them (the person, with
names and titles): the others point at it, and it points at the smallest table holding more members (every
business entity). A table holding fewer (the employees) points at the head of the smallest family holding them.
A name that starts by counting (NumberEmployees) is a figure, and a unique column named for its own table (the
categories' alternate key) is that table's code. What must link still links: a customer's type, by its name.

Invented data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def links():
    con = duckdb.connect()
    # Business entities 1..1200: people are 1..1000, vendors 1001..1200.
    con.execute("CREATE TABLE BusinessEntity (BusinessEntityID INTEGER, ModifiedDate DATE)")
    con.execute("INSERT INTO BusinessEntity SELECT i, DATE '2024-01-01' FROM range(1, 1201) t(i)")
    con.execute("CREATE TABLE Person (BusinessEntityID INTEGER, PersonType VARCHAR, Title VARCHAR, FirstName VARCHAR, "
                "LastName VARCHAR, Suffix VARCHAR)")
    con.execute("INSERT INTO Person SELECT i, ['EM', 'IN', 'SC'][i % 3 + 1], ['Mr.', 'Ms.'][i % 2 + 1], "
                "'First' || i, 'Last' || (i % 400), NULL FROM range(1, 1001) t(i)")
    con.execute("CREATE TABLE EmailAddress (BusinessEntityID INTEGER, EmailAddress VARCHAR)")
    con.execute("INSERT INTO EmailAddress SELECT i, 'person' || i || '@example.com' FROM range(1, 1001) t(i)")
    con.execute("CREATE TABLE Password (BusinessEntityID INTEGER, PasswordHash VARCHAR)")
    con.execute("INSERT INTO Password SELECT i, md5(i::VARCHAR) FROM range(1, 1001) t(i)")
    con.execute("CREATE TABLE Employee (BusinessEntityID INTEGER, JobTitle VARCHAR, VacationHours INTEGER)")
    con.execute("INSERT INTO Employee SELECT i * 4, ['Buyer', 'Engineer', 'Technician'][i % 3 + 1], i % 90 "
                "FROM range(1, 201) t(i)")
    # An employee's pay history: several rows per employee, pointing at the employees, not every entity.
    con.execute("CREATE TABLE EmployeePayHistory (BusinessEntityID INTEGER, RateChangeDate DATE, Rate DECIMAL(8,2))")
    con.execute("INSERT INTO EmployeePayHistory SELECT (i % 200 + 1) * 4, DATE '2020-01-01' + CAST(i AS INTEGER), "
                "10 + i % 30 FROM range(0, 500) t(i)")
    # Exchange rates all from one currency, to several: both columns point at the currencies, by name.
    con.execute("CREATE TABLE Currency (CurrencyCode VARCHAR, Name VARCHAR)")
    con.execute("INSERT INTO Currency VALUES ('USD', 'US Dollar'), ('EUR', 'Euro'), ('GBP', 'Pound'), ('CAD', 'Dollar')")
    con.execute("CREATE TABLE CurrencyRate (CurrencyRateID INTEGER, FromCurrencyCode VARCHAR, ToCurrencyCode VARCHAR, "
                "AverageRate DOUBLE)")
    con.execute("INSERT INTO CurrencyRate SELECT i, 'USD', ['EUR', 'GBP', 'CAD'][i % 3 + 1], 1 + i % 7 / 10 "
                "FROM range(1, 301) t(i)")
    con.execute("CREATE TABLE Vendor (BusinessEntityID INTEGER, Name VARCHAR, CreditRating INTEGER)")
    con.execute("INSERT INTO Vendor SELECT i, 'Vendor ' || i, i % 5 + 1 FROM range(1001, 1201) t(i)")
    # Resellers and how many people work at each (2..100, all inside the employees' numbers).
    con.execute("CREATE TABLE Reseller (ResellerKey INTEGER, ResellerName VARCHAR, NumberEmployees INTEGER)")
    con.execute("INSERT INTO Reseller SELECT i, 'Shop ' || i, (i * 37) % 99 + 2 FROM range(1, 401) t(i)")
    con.execute("CREATE TABLE EmployeeKeys (EmployeeKey INTEGER, Login VARCHAR)")
    con.execute("INSERT INTO EmployeeKeys SELECT i, 'login' || i FROM range(1, 297) t(i)")
    # Products 1..600 in categories 1..4, whose own alternate key is 1..4 too.
    con.execute("CREATE TABLE ProductCategory (ProductCategoryKey INTEGER, ProductCategoryAlternateKey INTEGER, "
                "CategoryName VARCHAR)")
    con.execute("INSERT INTO ProductCategory VALUES (1, 1, 'Bikes'), (2, 2, 'Components'), (3, 3, 'Clothing'), "
                "(4, 4, 'Accessories')")
    con.execute("CREATE TABLE Product (ProductKey INTEGER, ProductName VARCHAR, ProductCategoryKey INTEGER)")
    con.execute("INSERT INTO Product SELECT i, 'Product ' || i, i % 4 + 1 FROM range(1, 601) t(i)")
    # Reviews of products, numbered 1..4 within a table whose key the data does not show (rows of a comment left
    # empty): the review's own number still points at no product.
    con.execute("CREATE TABLE ProductReview (ProductReviewID INTEGER, ProductKey INTEGER, Rating INTEGER)")
    con.execute("INSERT INTO ProductReview VALUES (1, 709, 5), (NULL, NULL, NULL), (2, 37, 4), (NULL, NULL, NULL), "
                "(3, 37, 2), (4, 98, 5)")
    # Customers and their types, linked by name.
    con.execute("CREATE TABLE CustomerType (CustomerTypeID INTEGER, TypeName VARCHAR)")
    con.execute("INSERT INTO CustomerType VALUES (1, 'Retail'), (2, 'Wholesale'), (3, 'Online')")
    con.execute("CREATE TABLE Customer (CustomerID INTEGER, CustomerName VARCHAR, CustomerTypeID INTEGER)")
    con.execute("INSERT INTO Customer SELECT i, 'Customer ' || i, i % 3 + 1 FROM range(1, 801) t(i)")
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    found = learn(warehouse, inventory, BuildOptions(workers=1))
    name_of = {key: t.name for key, t in inventory.tables.items()}
    out: dict[tuple[str, str], set[str]] = {}
    for j in found.joins:
        if j.trust != "rejected":
            out.setdefault((name_of[j.from_table], j.from_column), set()).add(name_of[j.to_table])
    return out


@pytest.mark.parametrize("table, target", [
    ("EmailAddress", "Person"),
    ("Password", "Person"),
    ("Person", "BusinessEntity"),
    ("Employee", "Person"),
    ("Vendor", "BusinessEntity"),
])
def test_a_family_points_at_its_head(links, table, target):
    assert links.get((table, "BusinessEntityID")) == {target}, links


def test_a_members_history_points_at_the_smallest_table_holding_them(links):
    assert links.get(("EmployeePayHistory", "BusinessEntityID")) == {"Employee"}, links


def test_a_code_holding_one_value_links_by_its_name(links):
    assert links.get(("CurrencyRate", "FromCurrencyCode")) == {"Currency"}, links
    assert links.get(("CurrencyRate", "ToCurrencyCode")) == {"Currency"}, links


def test_a_count_of_employees_points_nowhere(links):
    assert ("Reseller", "NumberEmployees") not in links, links


def test_a_tables_own_alternate_key_points_nowhere(links):
    assert ("ProductCategory", "ProductCategoryAlternateKey") not in links, links
    assert links.get(("Product", "ProductCategoryKey")) == {"ProductCategory"}, links


def test_a_tables_own_number_points_nowhere_even_without_a_key(links):
    assert ("ProductReview", "ProductReviewID") not in links, links
    assert links.get(("ProductReview", "ProductKey")) == {"Product"}, links


def test_a_customers_type_still_links(links):
    assert links.get(("Customer", "CustomerTypeID")) == {"CustomerType"}, links
