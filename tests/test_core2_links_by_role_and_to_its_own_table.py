"""Links named by a role: a second role of a table the row already points at, a step up the table's own members,
a code two tables share, and an offer number that is no line number.

On a public sample of a manufacturer's operational database, with its keys withheld, Learn missed a bill of
materials' component (a product, beside the assembly it already linked to the products by name), and took a
special offer's number for a line number of each product's offers, so the offers were never linked. On synthetic
warehouses named in codes, an uplink device (UPLNK_DVC_KEY on the devices), a manager kept as REPORTS_TO and a
prescriber's NPI kept on the prescriptions were not linked either.

So: a column named for a table's key with a role before it, on that table itself, points at another member of it;
so does one named by a step up alone (reports to, parent). A code of one length, unique in one table and named
alike in another, is the first table's code. A row already pointing at a table by name may point at it again
under another role, found by its values, when every value finds a member. A number named as a pointer (an offer's
ID) numbers lines only where it runs without a gap. What must not link still does not: a run of numbers that stops
short of the products, and a second role whose values miss some products.

Invented data only.
"""

from __future__ import annotations

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.warehouse.runner import DuckDBWarehouse


@pytest.fixture(scope="module")
def found():
    con = duckdb.connect()
    con.execute("CREATE TABLE Product (ProductID INTEGER, Name VARCHAR, ListPrice DECIMAL(10,2))")
    con.execute("INSERT INTO Product SELECT i, 'Part ' || i, 5 + i % 40 FROM range(1, 401) t(i)")
    # A bill of materials: each assembly (by name a product) made of components (products too, by values alone).
    con.execute("CREATE TABLE BillOfMaterials (BillOfMaterialsID INTEGER, ProductAssemblyID INTEGER, "
                "ComponentID INTEGER, PerAssemblyQty DECIMAL(8,2))")
    con.execute("INSERT INTO BillOfMaterials SELECT i, 300 + i % 60, (i * 7) % 400 + 1, 1 + i % 4 "
                "FROM range(1, 1201) t(i)")
    # Work orders: a product by name, and a scrap reason number 1..16 that is a run of numbers, not a product.
    con.execute("CREATE TABLE WorkOrder (WorkOrderID INTEGER, ProductID INTEGER, ScrapReasonID INTEGER, "
                "OrderQty INTEGER)")
    con.execute("INSERT INTO WorkOrder SELECT i, i % 400 + 1, i % 16 + 1, 10 + i % 50 FROM range(1, 3001) t(i)")
    # A transfer between products: from a product by name, to one by values; some go to products that are gone.
    con.execute("CREATE TABLE ProductSwap (SwapID INTEGER, ProductID INTEGER, ReplacementID INTEGER)")
    con.execute("INSERT INTO ProductSwap SELECT i, i % 400 + 1, CASE WHEN i % 5 = 0 THEN 900 + i ELSE (i * 3) % 400 + 1 "
                "END FROM range(1, 801) t(i)")
    # Offers, and the products each offer covers: offer 1 covers every product, the others a few, with gaps.
    con.execute("CREATE TABLE SpecialOffer (SpecialOfferID INTEGER, Description VARCHAR, DiscountPct DECIMAL(4,2))")
    con.execute("INSERT INTO SpecialOffer SELECT i, 'Offer ' || i, i / 100 FROM range(1, 17) t(i)")
    con.execute("CREATE TABLE SpecialOfferProduct (SpecialOfferID INTEGER, ProductID INTEGER, rowguid VARCHAR)")
    con.execute("INSERT INTO SpecialOfferProduct SELECT 1, i, md5('a' || i) FROM range(1, 401) t(i)")
    con.execute("INSERT INTO SpecialOfferProduct SELECT [2, 3, 5, 7, 9, 12, 13, 16][i % 8 + 1], (i * 11) % 400 + 1, "
                "md5('b' || i) FROM range(1, 300) t(i) WHERE i % 8 <> 0 OR i % 3 = 0")
    # Devices hanging off devices, in codes: the uplink is a device.
    con.execute("CREATE TABLE DVC_DIM (DVC_KEY INTEGER, DVC_NM VARCHAR, UPLNK_DVC_KEY INTEGER)")
    con.execute("INSERT INTO DVC_DIM SELECT i, 'dev-' || i, CASE WHEN i <= 4 THEN NULL ELSE (i % 4) + 1 END "
                "FROM range(1, 121) t(i)")
    # Employees and whom they report to, with no key word in the name.
    con.execute("CREATE TABLE employees (employee_id INTEGER, last_name VARCHAR, title VARCHAR, reports_to INTEGER)")
    con.execute("INSERT INTO employees SELECT i, 'Name' || i, ['Rep', 'Lead', 'Head'][i % 3 + 1], "
                "CASE WHEN i = 1 THEN NULL WHEN i <= 4 THEN 1 ELSE (i % 3) + 2 END FROM range(1, 41) t(i)")
    # Prescribers, keyed by their own number, and the prescriptions that name them by their NPI.
    con.execute("CREATE TABLE PRSC_DIM (PRSCR_KEY INTEGER, PRSCR_NPI VARCHAR, PRSCR_NM VARCHAR)")
    con.execute("INSERT INTO PRSC_DIM SELECT i, lpad(CAST(1000000000 + 7919 * i AS VARCHAR), 10, '0'), 'Dr ' || i "
                "FROM range(1, 121) t(i)")
    con.execute("CREATE TABLE RX_FCT (RX_NO VARCHAR, PRSCR_NPI VARCHAR, QTY INTEGER)")
    con.execute("INSERT INTO RX_FCT SELECT 'RX' || i, lpad(CAST(1000000000 + 7919 * ((i * 13) % 120 + 1) AS VARCHAR), "
                "10, '0'), 1 + i % 9 FROM range(1, 2001) t(i)")
    # Each product's photo, kept once per product under its number: the components are products, not photos.
    con.execute("CREATE TABLE ProductProductPhoto (ProductID INTEGER, ProductPhotoID INTEGER)")
    con.execute("INSERT INTO ProductProductPhoto SELECT i, 1 + i % 50 FROM range(1, 401) t(i)")
    # Three phone types whose names are all four letters, and shifts named alike: no code, no link.
    con.execute("CREATE TABLE PhoneNumberType (PhoneNumberTypeID INTEGER, Name VARCHAR)")
    con.execute("INSERT INTO PhoneNumberType VALUES (1, 'Cell'), (2, 'Home'), (3, 'Work')")
    con.execute("CREATE TABLE Shift (ShiftID INTEGER, Name VARCHAR, StartHour INTEGER)")
    con.execute("INSERT INTO Shift VALUES (1, 'Days', 7), (2, 'Eves', 15), (3, 'Late', 23)")
    # Orders, some placed again from an earlier one; the customer's own reference is no order of this table.
    con.execute("CREATE TABLE SalesOrder (SalesOrderID INTEGER, OriginalSalesOrderID INTEGER, "
                "CustomerSalesOrderID INTEGER, Amount DECIMAL(10,2))")
    con.execute("INSERT INTO SalesOrder SELECT i, CASE WHEN i % 9 = 0 THEN i - 5 END, "
                "CASE WHEN i % 3 = 0 THEN 5000 + i ELSE i % 700 + 1 END, 10 + i % 90 FROM range(1, 801) t(i)")
    # Order lines numbered 1..6000, and stock moves whose reference is an order, a work order or a purchase, by
    # type: every reference sits inside the lines' numbers, and none is a line. A move's vendor, by values alone,
    # is a vendor: the name says what each vendor row is.
    con.execute("CREATE TABLE SalesOrderDetail (SalesOrderDetailID INTEGER, SalesOrderID INTEGER, Qty INTEGER)")
    con.execute("INSERT INTO SalesOrderDetail SELECT i, 1 + i // 4, 1 + i % 5 FROM range(1, 6001) t(i)")
    con.execute("CREATE TABLE sales_vendor (BusinessEntityID INTEGER, VendorName VARCHAR)")
    con.execute("INSERT INTO sales_vendor SELECT i, 'Vendor ' || i FROM range(1, 81) t(i)")
    con.execute("CREATE TABLE StockMove (StockMoveID INTEGER, ReferenceOrderID INTEGER, VendorID INTEGER, Qty INTEGER)")
    con.execute("INSERT INTO StockMove SELECT i, (i * 37) % 5000 + 1, i % 80 + 1, 1 + i % 9 FROM range(1, 4001) t(i)")
    warehouse = DuckDBWarehouse(con)
    inventory = from_duckdb(warehouse)
    findings = learn(warehouse, inventory, BuildOptions(workers=1, query_history=None))
    links = {(_t(j.from_table), j.from_column): j for j in findings.joins if j.trust != "rejected"}
    keys = {_t(k): v for k, v in findings.keys.items()}
    return links, keys


def _t(table_key: str) -> str:
    return table_key.rsplit(".", 1)[-1].casefold()


def _to(links, table: str, column: str) -> tuple[str, str] | None:
    j = links.get((table.casefold(), column))
    return (_t(j.to_table), j.to_column) if j is not None else None


def test_a_component_beside_its_assembly_is_a_product_too(found):
    links, _ = found
    assert _to(links, "BillOfMaterials", "ProductAssemblyID") == ("product", "ProductID")
    component = links[("billofmaterials", "ComponentID")]
    assert (_t(component.to_table), component.to_column) == ("product", "ProductID")
    assert component.trust == "verified" and (component.role or "").startswith("Component")


def test_a_run_of_numbers_or_a_role_missing_members_is_no_product(found):
    links, _ = found
    assert _to(links, "WorkOrder", "ScrapReasonID") is None       # 1..16: a run that stops short of the products
    assert _to(links, "ProductSwap", "ReplacementID") is None     # a fifth of its values are no product


def test_an_offer_number_with_gaps_is_no_line_number_and_links_to_its_offer(found):
    links, keys = found
    assert "SpecialOfferID" not in keys["specialofferproduct"].line_numbers
    assert _to(links, "SpecialOfferProduct", "SpecialOfferID") == ("specialoffer", "SpecialOfferID")


def test_a_role_before_the_tables_own_key_points_at_its_own_members(found):
    links, _ = found
    uplink = links[("dvc_dim", "UPLNK_DVC_KEY")]
    assert (_t(uplink.to_table), uplink.to_column) == ("dvc_dim", "DVC_KEY")
    assert uplink.trust == "verified"


def test_reports_to_points_at_the_employees_themselves(found):
    links, _ = found
    assert _to(links, "employees", "reports_to") == ("employees", "employee_id")


def test_a_code_two_tables_share_is_the_one_where_it_is_unique(found):
    links, _ = found
    npi = links[("rx_fct", "PRSCR_NPI")]
    assert (_t(npi.to_table), npi.to_column) == ("prsc_dim", "PRSCR_NPI")
    assert npi.trust == "verified"


def test_a_second_role_goes_to_the_table_its_named_link_keeps(found):
    links, _ = found
    component = links[("billofmaterials", "ComponentID")]
    assert _t(component.to_table) == "product"
    assert not any(e.kind == "ambiguous" for e in component.evidence)


def test_names_of_one_length_in_a_tiny_list_are_no_code(found):
    links, _ = found
    assert _to(links, "Shift", "Name") is None


def test_a_role_named_self_link_needs_every_row_to_find_a_member(found):
    links, _ = found
    assert _to(links, "SalesOrder", "OriginalSalesOrderID") == ("salesorder", "SalesOrderID")
    assert _to(links, "SalesOrder", "CustomerSalesOrderID") is None     # a third are no order of this table


def test_a_word_shared_with_a_table_counts_only_when_it_names_its_rows(found):
    links, _ = found
    assert _to(links, "StockMove", "ReferenceOrderID") is None          # an order, not an order line
    assert _to(links, "StockMove", "VendorID") == ("sales_vendor", "BusinessEntityID")
