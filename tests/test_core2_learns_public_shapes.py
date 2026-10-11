"""What Learn reads from shapes found on public sample warehouses, keys withheld.

A sales warehouse kept its products and employees once per version, so no name was unique: both were read as
facts, never named, and a ranking of products answered by category; its subcategory names came in Spanish beside
the English ones; a salesperson was named by a row's GUID; a stock count was split by the products' weight unit,
and the units moved in and out on the stock table were taken at the last day like a balance. A health-records
sample named its patients FIRST and LAST and kept a driver's licence and home coordinates; summed its diagnosis
codes and department numbers; counted claims by the day an illness began; and left its diagnoses unclassified.

So: a table others point at that names its own members is a dimension, whatever figures and dates it keeps; a
name kept in several languages is read in English; a GUID names nothing; a unit column for a weight is the unit
of a weight only; units in and out on a stock table add up; FIRST and LAST name a person, MAIDEN and MIDDLE too,
DRIVERS and LAT/LON are personal; a row's codes and numbers are never figures; a claim is dated by its service; a
dated event with nothing to add is counted.

Invented data only.
"""

from __future__ import annotations

import uuid

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def model():
    con = duckdb.connect()
    # Products kept once per version: a name repeats on each version; English and Spanish names; prices and dates.
    con.execute("CREATE TABLE Subcategory (SubcategoryKey INTEGER, EnglishSubcategoryName VARCHAR, "
                "SpanishSubcategoryName VARCHAR)")
    con.execute("INSERT INTO Subcategory VALUES (1, 'Mountain Bikes', 'Bicicleta de montaña'), "
                "(2, 'Road Bikes', 'Bicicleta de carretera'), (3, 'Helmets', 'Cascos'), (4, 'Gloves', 'Guantes')")
    con.execute("CREATE TABLE Product (ProductKey INTEGER, ProductAlternateKey VARCHAR, EnglishProductName VARCHAR, "
                "SubcategoryKey INTEGER, ListPrice DECIMAL(10,2), Weight DECIMAL(8,2), WeightUnitMeasureCode VARCHAR, "
                "StartDate DATE, EndDate DATE)")
    con.execute("INSERT INTO Product SELECT i + 1, 'P-' || (i % 150), 'Product ' || (i % 150), i % 4 + 1, "
                "10 + (i * 7) % 900, (i % 30) + 0.5, ['LB', 'G'][i % 2 + 1], DATE '2020-01-01' + CAST((i // 150) * 365 AS INTEGER), "
                "CASE WHEN i // 150 < 2 THEN DATE '2020-12-31' + CAST((i // 150) * 365 AS INTEGER) END "
                "FROM range(0, 450) t(i)")
    # Resellers kept once per version, whose versions' dates contradict each other (an old one ending before it
    # starts): no versions to read, the reseller's own name still names it.
    con.execute("CREATE TABLE Reseller (ResellerKey INTEGER, ResellerAlternateKey VARCHAR, ResellerName VARCHAR, "
                "AnnualSales DECIMAL(12,2), StartDate DATE, EndDate DATE)")
    con.execute("INSERT INTO Reseller SELECT i + 1, 'AW' || (i % 80), 'Shop ' || (i % 80), 1000 * (i % 80), "
                "DATE '2012-07-01' + CAST((i // 80) * 365 AS INTEGER), CASE WHEN i < 80 THEN DATE '2008-12-27' END "
                "FROM range(0, 100) t(i)")
    con.execute("CREATE TABLE Employee (EmployeeKey INTEGER, FirstName VARCHAR, LastName VARCHAR, Title VARCHAR, "
                "HireDate DATE, VacationHours INTEGER, BaseRate DECIMAL(8,2), EmergencyContactName VARCHAR)")
    con.execute("INSERT INTO Employee SELECT i, 'Given' || (i % 60), 'Family' || (i % 80), "
                "['Sales Representative', 'Sales Manager'][i % 2 + 1], DATE '2015-01-01' + CAST(i * 11 AS INTEGER), i % 99, 10 + i % 40, "
                "'Contact ' || i || repeat('y', i % 6) FROM range(1, 121) t(i)")
    con.execute("CREATE TABLE SalesPerson (SalesPersonKey INTEGER, Territory VARCHAR, Bonus DECIMAL(10,2), rowguid VARCHAR)")
    con.executemany("INSERT INTO SalesPerson VALUES (?, ?, ?, ?)",
                    [(i, ["North", "South", "East"][i % 3], 1000 + i * 10, str(uuid.UUID(int=i * 7919 + 13)).upper())
                     for i in range(1, 18)])
    con.execute("CREATE TABLE Sales (OrderNumber VARCHAR, SalesOrderLineNumber INTEGER, RevisionNumber INTEGER, "
                "ProductKey INTEGER, EmployeeKey INTEGER, SalesPersonKey INTEGER, ResellerKey INTEGER, OrderDate DATE, "
                "Quantity INTEGER, "
                "SalesAmount DECIMAL(12,2))")
    con.execute("INSERT INTO Sales SELECT 'SO' || (i // 3), i % 3 + 1, 1, i % 450 + 1, i % 120 + 1, i % 17 + 1, i % 100 + 1, "
                "DATE '2022-01-01' + CAST(i % 900 AS INTEGER), i % 9 + 1, (i % 97) * 3.5 + 20 FROM range(1, 9001) t(i)")
    con.execute("CREATE TABLE ProductInventory (ProductKey INTEGER, DateKey DATE, UnitsIn INTEGER, UnitsOut INTEGER, "
                "UnitsBalance INTEGER)")
    con.execute("INSERT INTO ProductInventory SELECT p, DATE '2023-01-01' + CAST(d AS INTEGER), CASE WHEN d % 7 = 0 THEN 20 ELSE 0 END, "
                "CASE WHEN d % 3 = 0 THEN 5 ELSE 0 END, 100 + p % 50 + d % 11 FROM range(1, 451) a(p), range(0, 60) b(d)")
    # Patients named FIRST and LAST, keyed by text; claims and diagnoses of them.
    pid = [str(uuid.UUID(int=(9 << 64) + i * 104729 + 7)) for i in range(150)]
    con.execute("CREATE TABLE patients (Id VARCHAR, BIRTHDATE DATE, FIRST VARCHAR, MIDDLE VARCHAR, LAST VARCHAR, "
                "MAIDEN VARCHAR, DRIVERS VARCHAR, GENDER VARCHAR, LAT DOUBLE, LON DOUBLE, INCOME INTEGER)")
    con.executemany("INSERT INTO patients VALUES (?, DATE '1960-01-01' + ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(p, i * 97, f"Given{i % 70}", f"M{i % 9}", f"Family{i % 90}", None if i % 2 else f"Maiden{i}",
                      f"S{99900000 + i * 13}", "FM"[i % 2], 42.1 + i / 1000, -71.0 - i / 1000, 20000 + i * 300)
                     for i, p in enumerate(pid)])
    con.execute("CREATE TABLE claims (Id VARCHAR, PATIENTID VARCHAR, DEPARTMENTID INTEGER, DIAGNOSIS1 BIGINT, "
                "CURRENTILLNESSDATE DATE, SERVICEDATE DATE, OUTSTANDING1 DOUBLE)")
    con.executemany("INSERT INTO claims VALUES (?, ?, ?, ?, DATE '2023-01-01' + ?, DATE '2023-01-01' + ?, ?)",
                    [(str(uuid.UUID(int=(11 << 64) + i * 31 + 1)), pid[i % 150], i % 12 + 1, 44054006 + (i % 40) * 1000,
                      i % 700, i % 700 + i % 30 + 1, (i % 50) * 12.5) for i in range(3000)])
    con.execute("CREATE TABLE conditions (START DATE, STOP DATE, PATIENT VARCHAR, CODE BIGINT, DESCRIPTION VARCHAR)")
    con.executemany("INSERT INTO conditions VALUES (DATE '2020-01-01' + ?, NULL, ?, ?, ?)",
                    [(i * 3 % 1500, pid[i % 150], 100000 + i % 25,
                      ["Hypertension", "Diabetes mellitus type 2 (disorder)", "Gingivitis", "Stress (finding)",
                       "Asthma"][i % 5]) for i in range(1200)])
    warehouse = DuckDBWarehouse(con)
    return build_model(warehouse, from_duckdb(warehouse), client_id="public-shapes", options=BuildOptions(workers=1))


def _table(model, name):
    return next(t for t in model.tables.values() if t.name == name)


def _column(model, table, name):
    return model.columns[f"{_table(model, table).key}.{name.casefold()}"]


def _label(model, table):
    entity = next(e for e in model.entities.values() if e.table == _table(model, table).key)
    return model.columns[entity.label_column].name if entity.label_column else None


@pytest.mark.parametrize("table, kind", [("Product", "dimension"), ("Employee", "dimension"), ("Sales", "fact"),
                                          ("patients", "dimension"), ("conditions", "fact")])
def test_tables_are_read_for_what_they_are(model, table, kind):
    assert _table(model, table).kind == kind


@pytest.mark.parametrize("table, label", [("Product", "EnglishProductName"), ("Subcategory", "EnglishSubcategoryName"),
                                           ("Reseller", "ResellerName"),
                                           ("Employee", "FirstName+LastName"), ("patients", "FIRST+LAST")])
def test_members_are_named_in_english_by_their_own_name(model, table, label):
    assert _label(model, table) == label


def test_members_kept_once_per_unmarked_version_are_told_apart_by_their_own_code(model):
    """A stock ranking of products split each product into its versions: their row numbers told the versions
    apart. The short code that repeats just as the names do is each member's own."""
    entity = next(e for e in model.entities.values() if e.table == _table(model, "Reseller").key)
    assert entity.code_column and model.columns[entity.code_column].name == "ResellerAlternateKey"


def test_a_guid_names_nothing(model):
    entity = next((e for e in model.entities.values() if e.table == _table(model, "SalesPerson").key), None)
    assert entity is None or entity.label_column is None or model.columns[entity.label_column].name != "rowguid"


@pytest.mark.parametrize("column, personal", [("FIRST", "name"), ("LAST", "name"), ("MAIDEN", "name"),
                                               ("DRIVERS", "detail"), ("LAT", "detail"), ("LON", "detail")])
def test_a_patients_names_and_details_are_personal(model, column, personal):
    assert _column(model, "patients", column).personal == personal


def _measures_on(model, table):
    key = _table(model, table).key
    return {model.columns[m.expr.column].name: m for m in model.measures.values()
            if m.table == key and getattr(m.expr, "column", None) and getattr(m.expr, "agg", "") in ("sum", "avg")}


def test_codes_and_row_numbers_are_never_figures(model):
    claims, sales = _measures_on(model, "claims"), _measures_on(model, "Sales")
    assert "DEPARTMENTID" not in claims and "DIAGNOSIS1" not in claims and "OUTSTANDING1" in claims
    assert "SalesOrderLineNumber" not in sales and "RevisionNumber" not in sales and "SalesAmount" in sales


def test_stock_counts_are_not_split_by_a_weight_unit_and_movements_add_up(model):
    stock = _measures_on(model, "ProductInventory")
    assert stock["UnitsBalance"].unit_column is None
    assert stock["UnitsBalance"].additivity == "semi_additive"
    assert stock["UnitsIn"].additivity == "additive" and stock["UnitsOut"].additivity == "additive"


def test_a_claim_is_dated_by_its_service(model):
    default = next(r for r in model.date_roles.values() if r.table == _table(model, "claims").key and r.is_default)
    assert model.columns[default.column].name == "SERVICEDATE"
