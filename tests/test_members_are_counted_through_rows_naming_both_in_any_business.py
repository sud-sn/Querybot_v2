"""Two lists that do not reach each other are counted through the rows that name both, in any business.

A patient has no doctor of their own; a prescription names both. The same shape is in every business: the customers
of a product category (each order line names its customer and its product, the product its category), the
suppliers a buyer orders from, the accounts a cost centre posts to (on journal lines, never on budgets), the
bookings and customers of a technician (each booking line names its technician), the patients a pharmacist
dispensed to (a fill names its prescription, the prescription its patient). Each count here is the distinct
members on those rows, checked against SQL written by hand; every way of asking (one member, a period, top N, by
month, more than N, on average, compared, as a share, listed) gives the same members; what the counted table
leaves out by default it leaves out here too; a reader who may not see the rows is never counted on them; and what
no rows tie together, or what sits under the counted list, is still refused.

Invented data only; the public DVD rental warehouse, where cached, for a bridge (a film's categories).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.compile.compiler import compile_query
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, ResolveError, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
_LEARNED: dict[str, tuple] = {}


def _domain(name: str):
    if name not in _LEARNED:
        built, model = learn(domains.build(name), "descriptive")
        _LEARNED[name] = (built.con, model)
    return _LEARNED[name]


def _run(name: str, plan: dict, allowed: set[str] | None = None):
    con, model = _domain(name)
    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model,
                      Context(today=TODAY, allowed_tables=allowed))
    compiled = compile_query(logical, model, "duckdb")
    return logical, compiled.sql, DuckDBWarehouse(con).query(compiled.sql).rows


def _by_member(rows) -> list[tuple]:
    """(member, count) of each row with a member, the count last (a person's code may sit between)."""
    return sorted((r[0], r[-1]) for r in rows if r[0] is not None)


# (business, counted members, grouped by, the rows counted on, the same by hand)
CASES = [
    ("retail", "number_of_customers", "category", "order_line", """
        SELECT c.category_name, COUNT(DISTINCT o.customer_id) FROM order_lines o
        JOIN products p ON o.product_id = p.product_id JOIN categories c ON p.category_id = c.category_id
        GROUP BY 1"""),
    ("retail", "number_of_products", "region", "order_line", """
        SELECT r.region_name, COUNT(DISTINCT o.product_id) FROM order_lines o
        JOIN stores s ON o.store_id = s.store_id JOIN regions r ON s.region_id = r.region_id GROUP BY 1"""),
    ("retail", "number_of_stores", "product", "order_line", """
        SELECT p.product_name, COUNT(DISTINCT o.store_id) FROM order_lines o
        JOIN products p ON o.product_id = p.product_id GROUP BY 1"""),
    ("finance", "number_of_accounts", "division", "journal_line", """
        SELECT d.division_name, COUNT(DISTINCT j.account_id) FROM journal_lines j
        JOIN cost_centres c ON j.cost_centre_id = c.cost_centre_id JOIN divisions d ON c.division_id = d.division_id
        GROUP BY 1"""),
    ("purchasing", "number_of_suppliers", "buyer", "purchase_order_line", """
        SELECT b.buyer_name, COUNT(DISTINCT l.supplier_id) FROM purchase_order_lines l
        JOIN buyers b ON l.buyer_id = b.buyer_id GROUP BY 1"""),
    ("inventory", "number_of_suppliers", "warehouse", "stock_movement", """
        SELECT w.warehouse_name, COUNT(DISTINCT m.supplier_id) FROM stock_movements m
        JOIN warehouses w ON m.warehouse_id = w.warehouse_id GROUP BY 1"""),
    # The lines of bookings left out by default (cancelled, duplicate, voided) are left out with them.
    ("home_services", "number_of_customers", "technician", "booking_line", """
        SELECT t.technician_name, COUNT(DISTINCT b.customer_id) FROM booking_lines l
        JOIN bookings b ON l.booking_number = b.booking_number
        JOIN technicians t ON l.technician_id = t.technician_id WHERE b.status_id NOT IN (4, 6, 7) GROUP BY 1"""),
    ("compounding_pharmacy", "number_of_patients", "pharmacist", "fill", """
        SELECT ph.pharmacist_name, COUNT(DISTINCT p.patient_id) FROM fills f
        JOIN prescriptions p ON f.rx_number = p.rx_number
        JOIN pharmacists ph ON CAST(f.pharmacist_code AS INTEGER) = ph.pharmacist_id GROUP BY 1"""),
    ("bank", "number_of_customers", "product", "loan_application", """
        SELECT p.product_name, COUNT(DISTINCT a.customer_id) FROM loan_applications a
        JOIN products p ON a.product_id = p.product_id GROUP BY 1"""),
]


@pytest.mark.parametrize("name,measure,by,rows,by_hand", CASES, ids=[f"{c[0]}:{c[1]}-by-{c[2]}" for c in CASES])
def test_each_business_counts_the_members_on_the_rows_that_name_both(name, measure, by, rows, by_hand):
    con, _ = _domain(name)
    logical, sql, got = _run(name, {"intent": "breakdown", "measures": [measure], "group_by": [by]})
    assert f"COUNT(DISTINCT" in sql and f"AS {rows} " in sql
    assert any("which reach both" in n for n in logical.notes)
    assert _by_member(got) == sorted(con.execute(by_hand).fetchall())


def test_by_a_field_of_the_rows_themselves():
    """A patient has no unit of measure; each fill has one and names its prescription, the prescription its
    patient: the patients of a unit are the distinct patients of its fills. A prescription has several fills: its
    count by their unit counts each prescription once per unit."""
    con, _ = _domain("compounding_pharmacy")
    _, sql, got = _run("compounding_pharmacy", {"intent": "breakdown", "measures": ["number_of_patients"],
                                                "group_by": ["fill.unit_of_measure"]})
    assert _by_member(got) == sorted(con.execute("""
        SELECT f.unit_of_measure, COUNT(DISTINCT p.patient_id) FROM fills f
        JOIN prescriptions p ON f.rx_number = p.rx_number GROUP BY 1""").fetchall())
    _, _, got = _run("compounding_pharmacy", {"intent": "breakdown", "measures": ["number_of_prescriptions"],
                                              "group_by": ["fill.unit_of_measure"]})
    assert _by_member(got) == sorted(con.execute("""
        SELECT unit_of_measure, COUNT(DISTINCT rx_number) FROM fills GROUP BY 1""").fetchall())


def test_across_a_recipe_on_its_own_lines():
    """A formula's ingredients are on its recipe lines: the ingredients of a dosage form are the distinct
    ingredients of its formulas' lines, never only those of formulas someone was prescribed or holds stock of."""
    con, _ = _domain("compounding_pharmacy")
    logical, sql, got = _run("compounding_pharmacy", {"intent": "breakdown", "measures": ["number_of_ingredients"],
                                                      "group_by": ["formula.dosage_form"]})
    assert "formula_ingredients" in sql and "prescriptions" not in sql
    assert _by_member(got) == sorted(con.execute("""
        SELECT f.dosage_form, COUNT(DISTINCT fi.ingredient_id) FROM formula_ingredients fi
        JOIN formulas f ON fi.formula_id = f.formula_id GROUP BY 1""").fetchall())


def test_finance_counts_on_what_was_posted_never_on_the_budget():
    """Budgets name an account and a cost centre too; what the accounts of a division are is what was posted."""
    _, sql, _ = _run("finance", {"intent": "breakdown", "measures": ["number_of_accounts"],
                                 "group_by": ["cost_centre"]})
    assert "journal_lines" in sql and "budgets" not in sql


def test_a_counted_document_keeps_what_it_leaves_out_by_default():
    """A booking count leaves cancelled, duplicate and voided bookings out; a technician's bookings too, listed or
    counted, and the answer says so."""
    con, _ = _domain("home_services")
    logical, sql, got = _run("home_services", {"intent": "breakdown", "measures": ["number_of_bookings"],
                                               "group_by": ["technician"]})
    by_hand = con.execute("""
        SELECT t.technician_name, COUNT(DISTINCT l.booking_number) FROM booking_lines l
        JOIN bookings b ON l.booking_number = b.booking_number JOIN technicians t ON l.technician_id = t.technician_id
        WHERE b.status_id NOT IN (4, 6, 7) GROUP BY 1""").fetchall()
    assert _by_member(got) == sorted(by_hand)
    assert any("left out (a default filter)" in n for n in logical.notes)
    tech = by_hand[0][0]
    one = [{"field": "technician.name", "op": "eq", "values": [tech]}]
    _, _, listed = _run("home_services", {"intent": "list", "group_by": ["booking.booking_number"], "filters": one})
    assert len(listed) == by_hand[0][1]
    # Asked to include them, they are counted, and said so.
    logical, _, got = _run("home_services", {"intent": "breakdown", "measures": ["number_of_bookings"],
                                             "group_by": ["technician"], "include_left_out": True})
    every = con.execute("""SELECT t.technician_name, COUNT(DISTINCT l.booking_number) FROM booking_lines l
                           JOIN technicians t ON l.technician_id = t.technician_id GROUP BY 1""").fetchall()
    assert _by_member(got) == sorted(every)
    assert any("as asked" in n for n in logical.notes)


def test_parallel_rows_are_named_when_one_is_counted():
    """A bank's loan applications, loans and cards each name a customer and a product: the answer counts on one
    and says the others hold their own."""
    logical, _, _ = _run("bank", {"intent": "breakdown", "measures": ["number_of_customers"], "group_by": ["product"]})
    said = " ".join(logical.notes)
    assert "loan application rows" in said and "name both too" in said


# Every way of asking, on one multi-step case: the customers of a product category, through their order lines.
LINES = """FROM order_lines o JOIN products p ON o.product_id = p.product_id
           JOIN categories c ON p.category_id = c.category_id JOIN calendar k ON o.order_date_key = k.date_key"""
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
IN_2025 = "k.full_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'"


def _per_category(where: str = "TRUE") -> dict[str, int]:
    con, _ = _domain("retail")
    return dict(con.execute(f"SELECT c.category_name, COUNT(DISTINCT o.customer_id) {LINES} WHERE {where} "
                            "GROUP BY 1").fetchall())


def test_one_member_and_a_period():
    counts, in_2025 = _per_category(), _per_category(IN_2025)
    name = sorted(counts)[0]
    one = [{"field": "category.name", "op": "eq", "values": [name]}]
    _, _, got = _run("retail", {"intent": "value", "measures": ["number_of_customers"], "filters": one})
    assert got[0][-1] == counts[name]
    _, _, got = _run("retail", {"intent": "value", "measures": ["number_of_customers"], "filters": one,
                                "time": Y2025})
    assert got[0][-1] == in_2025[name]
    _, _, got = _run("retail", {"intent": "breakdown", "measures": ["number_of_customers"],
                                "group_by": ["category"], "time": Y2025})
    assert dict(_by_member(got)) == in_2025


def test_top_n_and_more_than_n():
    counts = _per_category(IN_2025)
    _, _, got = _run("retail", {"intent": "rank", "measures": ["number_of_customers"], "group_by": ["category"],
                                "sort": [{"by": "number_of_customers", "desc": True}], "limit": 3, "time": Y2025})
    assert [r[-1] for r in got] == sorted(counts.values(), reverse=True)[:3]
    cut = sorted(counts.values())[len(counts) // 2]
    _, _, got = _run("retail", {"intent": "breakdown", "measures": ["number_of_customers"], "group_by": ["category"],
                                "time": Y2025,
                                "filters": [{"field": "number_of_customers", "op": "gt", "values": [cut],
                                             "total": True}]})
    assert dict(_by_member(got)) == {k: v for k, v in counts.items() if v > cut}


def test_on_average_and_how_many_have_at_least_n():
    counts = _per_category()
    _, _, got = _run("retail", {"intent": "value", "measures": ["number_of_customers"],
                                "per": {"of": "category", "summary": "avg"}})
    assert got[0][0] == pytest.approx(sum(counts.values()) / len(counts)) and got[0][-1] == len(counts)
    least = sorted(counts.values())[len(counts) // 2]
    _, _, got = _run("retail", {"intent": "value", "measures": ["number_of_customers"],
                                "per": {"of": "category", "summary": "count", "op": "gte", "values": [least]}})
    assert got[0][0] == sum(v >= least for v in counts.values())


def test_by_month():
    con, _ = _domain("retail")
    by_hand = {(str(a)[:7], b): n for a, b, n in con.execute(
        f"""SELECT DATE_TRUNC('month', k.full_date), c.category_name, COUNT(DISTINCT o.customer_id) {LINES}
            WHERE {IN_2025} GROUP BY 1, 2""").fetchall()}
    _, _, got = _run("retail", {"intent": "trend", "measures": ["number_of_customers"], "group_by": ["category"],
                                "time": {"grain": "month", **Y2025}})
    assert {(str(r[0])[:7], r[1]): r[-1] for r in got if r[1] is not None} == by_hand


def test_compared_with_the_year_before_and_as_a_share():
    now, before = _per_category(IN_2025), _per_category(
        "k.full_date BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'")
    logical, _, got = _run("retail", {"intent": "compare", "measures": ["number_of_customers"],
                                      "group_by": ["category"],
                                      "time": {**Y2025, "compare": {"kind": "previous_period"}}})
    names = [c.name for c in logical.columns] if hasattr(logical, "columns") else None
    for row in got:
        if row[0] is None:
            continue
        assert now[row[0]] in row and before[row[0]] in row, (names, row)
    _, _, got = _run("retail", {"intent": "share", "measures": ["number_of_customers"], "group_by": ["category"],
                                "time": Y2025})
    assert {r[0]: r[1] for r in got if r[0] is not None} == now


def test_listed_members_are_the_ones_counted():
    con, _ = _domain("retail")
    name = sorted(_per_category())[0]
    one = [{"field": "category.name", "op": "eq", "values": [name]}]
    by_hand = {r[0] for r in con.execute(
        f"SELECT DISTINCT o.customer_id {LINES} WHERE c.category_name = ?", [name]).fetchall()}
    _, sql, got = _run("retail", {"intent": "list", "group_by": ["customer"], "filters": one})
    assert "order_lines" in sql and len(got) == len(by_hand)
    in_2025 = {r[0] for r in con.execute(
        f"SELECT DISTINCT o.customer_id {LINES} WHERE c.category_name = ? AND {IN_2025}", [name]).fetchall()}
    _, _, got = _run("retail", {"intent": "list", "group_by": ["customer"], "filters": one, "time": Y2025})
    assert len(got) == len(in_2025)


def test_two_groupings_on_both_sides():
    con, _ = _domain("compounding_pharmacy")
    _, _, got = _run("compounding_pharmacy", {"intent": "breakdown", "measures": ["number_of_patients"],
                                              "group_by": ["prescriber.specialty", "formula.dosage_form"]})
    by_hand = con.execute("""
        SELECT d.specialty, f.dosage_form, COUNT(DISTINCT p.patient_id) FROM prescriptions p
        JOIN prescribers d ON p.prescriber_npi = d.prescriber_npi JOIN formulas f ON p.formula_id = f.formula_id
        GROUP BY 1, 2""").fetchall()
    assert sorted((r[0], r[1], r[-1]) for r in got if r[0] is not None) == sorted(by_hand)


def test_a_reader_is_never_counted_on_rows_they_may_not_see():
    _, model = _domain("compounding_pharmacy")
    hidden = {k for k in model.tables if k.endswith(".prescriptions")}
    allowed = set(model.tables) - hidden
    with pytest.raises(ResolveError):
        _run("compounding_pharmacy", {"intent": "breakdown", "measures": ["number_of_patients"],
                                      "group_by": ["prescriber"]}, allowed=allowed)
    _, model = _domain("retail")
    allowed = {k for k in model.tables if not k.endswith(".order_lines")}
    try:
        _, sql, _ = _run("retail", {"intent": "breakdown", "measures": ["number_of_customers"],
                                    "group_by": ["category"]}, allowed=allowed)
    except ResolveError:
        return
    assert "order_lines" not in sql
    # Nor reaches the grouping through a table they may not see (a product's category, for the order lines).
    allowed = {k for k in model.tables if not k.endswith(".products")}
    try:
        _, sql, _ = _run("retail", {"intent": "breakdown", "measures": ["number_of_customers"],
                                    "group_by": ["category"]}, allowed=allowed)
    except ResolveError:
        return
    assert "products" not in sql


@pytest.mark.parametrize("name,measure,by,why", [
    ("home_services", "number_of_plans", "technician", "is not linked to"),
    ("compounding_pharmacy", "number_of_quality_tests", "prescriber", "is not linked to"),
    ("bank", "number_of_transactions", "product", "is not linked to"),
    ("retail", "number_of_regions", "customer", "cannot split"),
    ("finance", "number_of_divisions", "cost_centre", "cannot split"),
    ("compounding_pharmacy", "number_of_payers", "patient", "cannot split"),
])
def test_what_no_rows_tie_together_or_sits_under_the_list_is_still_refused(name, measure, by, why):
    with pytest.raises(ResolveError, match=why):
        _run(name, {"intent": "breakdown", "measures": [measure], "group_by": [by]})


def test_what_the_list_reaches_itself_still_counts_the_list():
    con, _ = _domain("retail")
    _, sql, got = _run("retail", {"intent": "breakdown", "measures": ["number_of_customers"], "group_by": ["customer.segment"]})
    assert "order_lines" not in sql
    assert sum(r[-1] for r in got) == con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]


def test_in_the_chat_a_technicians_customers_are_answered_and_said():
    con, model = _domain("home_services")

    def ai(stable: str, tail: str) -> str:
        return json.dumps({"kind": "query", "intent": "breakdown", "measures": ["number_of_customers"],
                           "group_by": ["technician"]})

    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=ai, index=MemberIndex(), today=TODAY)
    payload = answer_question("how many customers did each technician serve", services, Session())
    assert payload.get("clarify") is None and payload["data"]["rows"]
    assert "booking line" in payload["answer"]["headline"]


def test_across_a_bridge_on_a_public_warehouse():
    """A film is in several categories (a bridge): the customers of a category are the distinct customers renting
    its films, each once."""
    from evals.core2 import public

    if "sakila" not in public.cached():
        pytest.skip("the public DVD rental warehouse is not cached (python -m evals.core2.public fetch)")
    built, model = learn(public.build("sakila"), "descriptive")
    customers = next(m.slug for m in model.measures.values() if m.slug == "number_of_customers")
    category = next(e.slug for e in model.entities.values() if e.table.endswith(".category"))
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "breakdown", "measures": [customers],
                                           "group_by": [category]}), model, Context(today=TODAY))
    rows = DuckDBWarehouse(built.con).query(compile_query(logical, model, "duckdb").sql).rows
    by_hand = built.con.execute("""
        SELECT c.name, COUNT(DISTINCT r.customer_id) FROM rental r JOIN inventory i ON r.inventory_id = i.inventory_id
        JOIN film_category fc ON i.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id
        GROUP BY 1""").fetchall()
    assert _by_member(rows) == sorted(by_hand)
