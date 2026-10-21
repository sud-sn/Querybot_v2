"""A rate of what came back over what went out, and the sales with no customer as their own line.

Found with a real model (gpt-4o) on a sample workbook:

* "Of the units each store sold in 2025, what % came back as returns?" was planned as a rate of the units sold
  whose condition is on the returns ("a return quantity above 0"): rows of another table that point at the sales
  lines, so the plan was refused or read as each store's share of the units. It is the units returned over the
  units sold, each added up on its own rows. Both are a "Quantity": the answer said "2,634 quantity and 65
  quantity", and named the rate "Quantity ÷ Quantity".
* "What share of 2025 net sales came from each customer segment? Put the sales with no customer on file as their
  own line" was planned with the rows a table leaves out by default put back (the cancelled lines): every figure
  and share came out larger than the business's own. The line with no customer was "Unknown", like a segment.
* Two customers who share a name were counted as one ("across 5 customers" for 6).

Every answer goes through the whole answer path, the AI stood in by the plan it wrote, and is compared with SQL
written by hand on an invented shop's warehouse (tests/shop_data.py).

Synthetic data only.
"""

from __future__ import annotations

import pytest

from tests import shop_data
from tests.shop_data import ask, headline, rows

IN_2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}
WHERE_2025 = "o.status <> 'Cancelled' AND o.order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'"
RETURN_RATE = ("Of the units each store sold in 2025, what % came back as returns? Show units sold, units returned "
               "and the return rate")


@pytest.fixture(scope="module")
def shop():
    return shop_data.build()


def _returned_by_store(shop) -> dict[str, tuple[float, float]]:
    """Each store's units sold in 2025 and units returned in 2025, by the return's own date (a reading the
    yardstick accepts beside the sales of 2025 and what came back of them)."""
    found = shop[0].execute(f"""
        SELECT s.store_name, SUM(o.quantity), (SELECT SUM(r.quantity) FROM returns r JOIN order_lines l
             ON l.order_id = r.order_id AND l.line = r.line JOIN stores t USING (store_id)
             WHERE t.store_name = s.store_name AND r.return_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31')
        FROM order_lines o JOIN stores s USING (store_id) WHERE {WHERE_2025} GROUP BY 1""").fetchall()
    return {store: (float(sold), float(back or 0)) for store, sold, back in found}


def test_the_share_of_units_that_came_back_is_units_returned_over_units_sold(shop):
    # As the AI wrote it: a rate of the units sold, its condition on the returns' quantity.
    payload = ask(shop, RETURN_RATE, {
        "intent": "breakdown", "measures": ["quantity", "return_quantity"], "group_by": ["store.name"],
        "time": IN_2025,
        "rates": [{"name": "Return rate", "measure": "quantity",
                   "where": [{"field": "return_quantity", "op": "gt", "values": [0]}]}],
        "asked": [{"words": "units sold", "kind": "figure", "plan": "quantity"},
                  {"words": "units returned", "kind": "figure", "plan": "return_quantity"},
                  {"words": "what % came back as returns", "kind": "figure", "plan": "rates"}]})
    expected = _returned_by_store(shop)
    got = {r["store_name"]: r for r in rows(payload)}
    assert set(got) == set(expected)
    rate = next(k for k in rows(payload)[0] if "return_rate" in k)
    returned = next(k for k in rows(payload)[0] if k != "quantity" and "quantity" in k and "rate" not in k)
    for store, (sold, back) in expected.items():
        assert float(got[store]["quantity"]) == pytest.approx(sold), store
        assert float(got[store][returned]) == pytest.approx(back), store
        assert float(got[store][rate]) == pytest.approx(100 * back / sold, abs=1e-4), store
    labels = payload["data"]["header_labels"]
    # Two figures named "Quantity", told apart by their table's name.
    assert labels["quantity"] == "Order line quantity" and labels[returned] == "Return quantity", labels
    assert not payload.get("left_out"), payload.get("left_out")


def test_a_rate_of_the_rows_own_condition_stays_a_rate(shop):
    # "What share of the lines were cancelled": its condition is on the lines themselves, never another table.
    payload = ask(shop, "what share of order lines were cancelled in 2025, by store", {
        "intent": "breakdown", "measures": ["number_of_order_lines"], "group_by": ["store.name"], "time": IN_2025,
        "include_left_out": True,
        "rates": [{"name": "Cancellation rate", "measure": "number_of_order_lines",
                   "where": [{"field": "order_line.status", "op": "eq", "values": ["Cancelled"]}]}]})
    expected = dict(shop[0].execute(
        "SELECT s.store_name, 100.0 * COUNT(*) FILTER (WHERE o.status = 'Cancelled') / COUNT(*) FROM order_lines o "
        "JOIN stores s USING (store_id) WHERE o.order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31' "
        "GROUP BY 1").fetchall())
    rate = next(k for k in rows(payload)[0] if "cancellation" in k)
    for r in rows(payload):
        assert float(r[rate]) == pytest.approx(float(expected[r["store_name"]]), abs=1e-4), r["store_name"]


NO_CUSTOMER = ("What share of 2025 net sales came from each customer segment? Put the sales with no customer on file "
               "as their own line")


def test_the_sales_with_no_customer_are_their_own_line_and_cancelled_lines_stay_out(shop):
    # As the AI wrote it: the cancelled lines put back to make room for the line with no customer.
    payload = ask(shop, NO_CUSTOMER, {"intent": "share", "measures": ["net_amount"], "group_by": ["customer.segment"],
                                      "time": IN_2025, "include_left_out": True,
                                      "asked": [{"words": "sales with no customer on file as their own line",
                                                 "plan": "include_left_out"}]})
    expected = dict(shop[0].execute(
        f"SELECT COALESCE(c.segment, '(none)'), SUM(o.net_amount) FROM order_lines o LEFT JOIN customers c "
        f"USING (customer_id) WHERE {WHERE_2025} GROUP BY 1").fetchall())
    whole = sum(float(v) for v in expected.values())
    got = {r["customer_segment"]: r for r in rows(payload)}
    assert set(got) == {"Retail", "Business", "No customer recorded"}, set(got)
    for segment, net in expected.items():
        row = got["No customer recorded" if segment == "(none)" else segment]
        assert float(row["net_amount"]) == pytest.approx(float(net)), segment
        assert float(row["net_amount_share"]) == pytest.approx(100 * float(net) / whole, abs=1e-3), segment


def test_cancelled_lines_asked_for_are_counted(shop):
    payload = ask(shop, "net sales by segment in 2025, including cancelled orders",
                  {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer.segment"],
                   "time": IN_2025, "include_left_out": True})
    (everything,), = shop[0].execute("SELECT SUM(net_amount) FROM order_lines WHERE order_date BETWEEN "
                                     "DATE '2025-01-01' AND DATE '2025-12-31'").fetchall()
    assert sum(float(r["net_amount"]) for r in rows(payload)) == pytest.approx(float(everything))


def test_customers_who_share_a_name_are_counted_apart(shop):
    payload = ask(shop, "net sales by customer in 2025", {"intent": "breakdown", "measures": ["net_amount"],
                                                          "group_by": ["customer.name"], "time": IN_2025})
    (customers,), = shop[0].execute(f"SELECT COUNT(DISTINCT customer_id) FROM order_lines o WHERE {WHERE_2025}"
                                    ).fetchall()
    assert customers == len(shop_data.CUSTOMERS) == 6
    assert f"across {customers} customers" in headline(payload), headline(payload)


@pytest.mark.parametrize("value,none", [(None, True), ("", True), ("Unknown", True), ("No customer recorded", True),
                                        ("No supplier recorded", True), ("No Frills", False),
                                        ("No Frills Grocery", False), ("Ari Lane", False), (0, False)])
def test_the_rows_with_no_member_are_told_by_their_name(value, none):
    from core2.answer.builder import no_member

    assert no_member(value) is none, value


def test_a_rate_of_sales_by_a_condition_on_another_tables_units_is_never_units_over_dollars(shop):
    # "What % of net sales came from lines later returned", as the AI wrote it: a rate of the net amount whose
    # condition is on the returns' units. Read as returned units over dollars, it said 1.3% for about 9%.
    payload = ask(shop, "What % of 2025 net sales by store came from lines that were later returned?", {
        "intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"], "time": IN_2025,
        "rates": [{"name": "Share of sales returned", "measure": "net_amount",
                   "where": [{"field": "return_quantity", "op": "gt", "values": [0]}]}]})
    hand = dict(shop[0].execute(
        f"SELECT s.store_name, 100 * SUM(o.net_amount) FILTER (WHERE r.order_id IS NOT NULL) / SUM(o.net_amount) "
        f"FROM order_lines o JOIN stores s USING (store_id) LEFT JOIN returns r ON r.order_id = o.order_id "
        f"AND r.line = o.line WHERE {WHERE_2025} GROUP BY 1").fetchall())
    for r in rows(payload):
        for key in (k for k in r if "share" in k or "rate" in k):
            assert float(r[key]) == pytest.approx(float(hand[r["store_name"]]), abs=0.05), (key, r)
    assert rows(payload) or "could not" in headline(payload).lower(), headline(payload)


@pytest.mark.parametrize("question", ["Chiffre d'affaires des commandes annulées par magasin en 2025",
                                      "Net sales on voided orders by store in 2025"])
def test_the_rows_left_out_by_default_are_counted_when_the_question_asks_for_them(shop, question):
    payload = ask(shop, question, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"],
                                   "time": IN_2025, "include_left_out": True,
                                   "filters": [{"field": "order_line.status", "op": "eq", "values": ["Cancelled"]}]})
    (cancelled,), = shop[0].execute("SELECT SUM(net_amount) FROM order_lines WHERE status = 'Cancelled' AND "
                                    "order_date BETWEEN DATE '2025-01-01' AND DATE '2025-12-31'").fetchall()
    assert sum(float(r["net_amount"]) for r in rows(payload)) == pytest.approx(float(cancelled)), headline(payload)


def test_the_sales_with_no_customer_never_lead_a_series_by_segment(shop):
    # Found with gpt-4o: "No customer recorded led in 4 of the 6 months, among the 5 segments".
    payload = ask(shop, "average net sales per line by customer segment, month by month in 2025",
                  {"intent": "trend", "measures": ["net_amount"], "group_by": ["customer.segment"],
                   "time": {**IN_2025, "grain": "month"},
                   "derived": [{"name": "Net per line", "op": "ratio", "measures": ["net_amount",
                                                                                    "number_of_order_lines"]}]})
    said = headline(payload)
    assert "No customer recorded led" not in said and "among the 2 segments" in said, said


def _customer_link(model):
    """The order lines' link to the customers."""
    return next(j for j in model.joins.values() if model.tables[j.to_table].name == "customers"
                and model.tables[j.from_table].name == "order_lines")


BY_SEGMENT = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer.segment"], "time": IN_2025}


def _segments(shop, model) -> set:
    con, _, index = shop
    return {r["customer_segment"] for r in rows(ask((con, model, index), "net sales by customer segment in 2025",
                                                   BY_SEGMENT))}


def test_the_line_with_no_customer_needs_every_number_on_the_way_measured_by_learn(shop):
    # As learned, every customer number on the lines names a customer, measured on every line.
    assert "No customer recorded" in _segments(shop, shop[1])
    # Found in review: a link whose rows were not all measured may leave a line whose number names no customer on
    # file, so its empty rows are "Unknown", never "No customer recorded".
    from core2.model.schema import ColumnFilter

    current = shop[1].model_copy(deep=True)       # the admin keeps only some customers: the rest find no match
    link = _customer_link(current)
    segment = next(c.key for c in current.columns.values() if c.table == link.to_table and c.name == "segment")
    link.conditions = [ColumnFilter(column=segment, op="eq", values=["Retail"])]
    brought = shop[1].model_copy(deep=True)       # brought over from today's setup: no measure of Learn's
    _customer_link(brought).evidence = []
    sampled = shop[1].model_copy(deep=True)       # measured on a sample of a large table
    for e in _customer_link(sampled).evidence:
        if e.kind == "test_run":
            e.data = {**e.data, "rows": e.data["rows"] // 2}
    for model in (current, brought, sampled):
        found = _segments(shop, model)
        assert "No customer recorded" not in found and "Unknown" in found, found


def test_the_empty_rows_of_a_second_link_to_a_table_are_named_by_that_link(shop):
    # Found in review: grouped by one link's member and another link's attribute, the empty rows were said to have
    # no customer, though each has its bill-to customer: the walk followed the first link to the table.
    import datetime as dt

    import duckdb

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.plan.values import build_index
    from core2.warehouse.runner import DuckDBWarehouse

    con = duckdb.connect()
    con.execute("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name VARCHAR NOT NULL, "
                "segment VARCHAR NOT NULL)")
    con.executemany("INSERT INTO customers VALUES (?,?,?)",
                    [(1, "Ann", "Retail"), (2, "Bob", "Business"), (3, "Cat", "Retail"), (4, "Dan", "Business")])
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE NOT NULL, bill_to_customer_id "
                "INTEGER NOT NULL, ship_to_customer_id INTEGER, amount DECIMAL(12,2))")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
        (i + 1, dt.date(2025, 1, 1) + dt.timedelta(days=i % 365), 1 + i % 4,
         None if i % 10 == 0 else (99 if i % 10 == 1 else 1 + (i + 1) % 4), 10 + i % 7) for i in range(400)])
    warehouse = DuckDBWarehouse(con)
    declared = [{"constraint_name": f"fk_{c}", "parent_table": "orders", "parent_col": c, "ref_table": "customers",
                 "ref_col": "customer_id", "ordinal": 1, "enforced": False}
                for c in ("bill_to_customer_id", "ship_to_customer_id")]
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=declared), client_id="t-two-links",
                        options=BuildOptions(workers=1))

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    payload = ask((con, model, build_index(model, fetch)), "sales by bill-to customer and ship-to segment in 2025", {
        "intent": "breakdown", "measures": ["amount"], "group_by": ["customer.name", "customer.segment"],
        "via": {"customer.name": "role:Bill to customer", "customer.segment": "role:Ship to customer"},
        "time": IN_2025})
    segments = {r["customer_segment"] for r in rows(payload)}
    # Some ship-to numbers name no customer on file: those rows are not known to have none.
    assert "No customer recorded" not in segments and "Unknown" in segments, segments


@pytest.mark.parametrize("question", [
    "What share of 2025 net sales came from each customer segment, including the sales with no customer on file "
    "as their own line?",
    "What share of 2025 net sales came from each customer segment? Put the sales with no customer on file as a "
    "line too",
    "Net sales by segment in 2025 excluding the Business segment",
])
def test_ordinary_words_never_put_back_the_rows_left_out(shop, question):
    # Found in review: "including", "too" and "excluding" kept the AI's "put back what is left out by default", so
    # the cancelled lines were counted.
    payload = ask(shop, question, {**BY_SEGMENT, "include_left_out": True})
    (kept,), = shop[0].execute(f"SELECT SUM(o.net_amount) FROM order_lines o WHERE {WHERE_2025}").fetchall()
    assert sum(float(r["net_amount"]) for r in rows(payload)) <= float(kept) + 0.01, headline(payload)


def test_a_word_of_the_status_columns_name_never_puts_back_the_rows_left_out(shop):
    # With the status column named "Order status", "orders" anywhere kept the cancelled lines.
    from core2.plan.ir import Plan
    from core2.plan.planner import left_out_unasked

    model = shop[1].model_copy(deep=True)
    status = next(c for c in model.columns.values() if model.tables[c.table].name == "order_lines"
                  and c.name == "status")
    status.business_name = "Order status"
    plan = Plan.model_validate({"kind": "query", **BY_SEGMENT, "include_left_out": True})
    assert not left_out_unasked(plan, model, "net sales of the orders by segment in 2025", None).include_left_out
    assert left_out_unasked(plan, model, "net sales by order status in 2025", None).include_left_out


@pytest.fixture(scope="module")
def coded():
    """Orders whose status is a number with a table of statuses, the cancelled ones left out by default."""
    import datetime as dt

    import duckdb

    from core2.bootstrap.build import BuildOptions, build_model
    from core2.bootstrap.inventory import from_duckdb
    from core2.model.schema import ColumnFilter
    from core2.plan.values import build_index
    from core2.warehouse.runner import DuckDBWarehouse

    con = duckdb.connect()
    con.execute("CREATE TABLE stores (store_id INTEGER PRIMARY KEY, store_name VARCHAR NOT NULL)")
    con.executemany("INSERT INTO stores VALUES (?,?)", [(1, "North"), (2, "South")])
    con.execute("CREATE TABLE order_statuses (status_id INTEGER PRIMARY KEY, status_name VARCHAR NOT NULL)")
    con.executemany("INSERT INTO order_statuses VALUES (?,?)", [(1, "Completed"), (2, "Cancelled"), (3, "Pending")])
    con.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, order_date DATE NOT NULL, store_id INTEGER NOT "
                "NULL REFERENCES stores(store_id), status_id INTEGER NOT NULL REFERENCES order_statuses(status_id), "
                "amount DECIMAL(12,2))")
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
        (i + 1, dt.date(2025, 1, 1) + dt.timedelta(days=i % 365), 1 + i % 2, 2 if i % 9 == 0 else (3 if i % 13 == 0
                                                                                                  else 1), 10 + i % 5)
        for i in range(500)])
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-coded", options=BuildOptions(workers=1))
    orders = next(k for k, t in model.tables.items() if t.name == "orders")
    if not model.tables[orders].default_filters:
        status = next(k for k, c in model.columns.items() if c.table == orders and c.name == "status_id")
        model.tables[orders].default_filters = [ColumnFilter(column=status, op="not_in", values=[2],
                                                             shown=["Cancelled"])]
        model.tables[orders].readers_may_include = True

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'"{model.tables[column.table].name}"').rows]

    return con, model, build_index(model, fetch)


@pytest.mark.parametrize("question", ["Chiffre d'affaires des commandes annulées par magasin en 2025",
                                      "Net sales on voided orders by store in 2025"])
def test_rows_left_out_by_a_status_code_are_counted_when_the_question_filters_on_its_name(coded, question):
    # Found in review: the status a table leaves out was a code whose name is in a table of statuses, and a filter on
    # that name was not seen as asking for those rows: "no rows match".
    con, model, _ = coded
    name = next(s for s in model.attributes if s.endswith(".name") and "status" in s)
    payload = ask(coded, question, {"intent": "breakdown", "measures": ["amount"], "group_by": ["store.name"],
                                    "include_left_out": True, "time": IN_2025,
                                    "filters": [{"field": name, "op": "eq", "values": ["Cancelled"]}]})
    (cancelled,), = con.execute("SELECT SUM(amount) FROM orders WHERE status_id = 2").fetchall()
    assert sum(float(r["amount"]) for r in rows(payload)) == pytest.approx(float(cancelled)), headline(payload)


REFUNDED = (f"SELECT s.store_name, 100 * SUM(o.net_amount) FILTER (WHERE r.order_id IS NOT NULL) / SUM(o.net_amount) "
            f"FROM order_lines o JOIN stores s USING (store_id) LEFT JOIN returns r ON r.order_id = o.order_id "
            f"AND r.line = o.line WHERE {WHERE_2025} GROUP BY 1")


@pytest.mark.parametrize("question", ["What % of 2025 net sales by store came from lines that were later refunded?",
                                      "What share of 2025 net sales by store had a refund?"])
def test_the_share_of_sales_from_refunded_lines_is_never_refunds_over_sales(shop, question):
    # Found in review: "refunded" read as what came back, and "Refund amount" and "Net amount" share their last
    # word, so the share of sales from refunded lines (about 9%) was refunds over sales (about 4%).
    payload = ask(shop, question, {
        "intent": "breakdown", "measures": ["net_amount"], "group_by": ["store.name"], "time": IN_2025,
        "rates": [{"name": "Share refunded", "measure": "net_amount",
                   "where": [{"field": "refund_amount", "op": "gt", "values": [0]}]}]})
    hand = dict(shop[0].execute(REFUNDED).fetchall())
    for r in rows(payload):
        for key in (k for k in r if "share" in k or "rate" in k or "refund" in k):
            assert float(r[key]) == pytest.approx(float(hand[r["store_name"]]), abs=0.05), (key, r)
    assert rows(payload) or "could not" in headline(payload).lower(), headline(payload)


def test_units_returned_over_units_sold_whatever_the_figures_are_called(shop):
    # Found in review: named "Units sold" and "Units returned", the figures no longer shared a last word, and the
    # return rate the feature was built for was refused again.
    con, model, index = shop
    renamed = model.model_copy(deep=True)
    renamed.measures[next(k for k, m in renamed.measures.items() if m.slug == "quantity")].business_name = "Units sold"
    renamed.measures[next(k for k, m in renamed.measures.items() if m.slug == "return_quantity")].business_name = \
        "Units returned"
    payload = ask((con, renamed, index), RETURN_RATE, {
        "intent": "breakdown", "measures": ["quantity"], "group_by": ["store.name"], "time": IN_2025,
        "rates": [{"name": "Return rate", "measure": "quantity",
                   "where": [{"field": "return_quantity", "op": "gt", "values": [0]}]}]})
    expected = _returned_by_store(shop)
    rate = next((k for k in (rows(payload) or [{}])[0] if "return_rate" in k), None)
    assert rate is not None, headline(payload)
    for r in rows(payload):
        sold, back = expected[r["store_name"]]
        assert float(r[rate]) == pytest.approx(100 * back / sold, abs=1e-4), r
