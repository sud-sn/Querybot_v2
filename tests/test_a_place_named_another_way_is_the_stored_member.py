"""A member named another way than the data writes it narrows to the member the data keeps.

Readers write places in full ("Oregon", "United States") where the data keeps codes ("OR", "USA"): the filter
compared the reader's words with the stored ones and found nothing, answering "no value" for Oregon, and an
exclusion ("outside the United States") silently left nothing out. A place written another way is now the
member the data keeps for it, in a grouping whose members are places of that kind (never "Canada" for
California's "CA"); an exclusion of a member the data does not have says it changed nothing; and two names
left unquoted are quoted together by one click.

An invented shop with sales by state and country.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse

STATES = ["OR", "CA", "WA", "NV", "TX"]
COUNTRIES = ["USA", "Canada", "Germany", "Mexico"]
PLANS = ["Starter", "Growth", "Scale"]


@pytest.fixture(scope="module")
def shop():
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (sale_id INTEGER, sale_date DATE, state VARCHAR, country VARCHAR, "
                "plan_name VARCHAR, amount DECIMAL(10,2))")
    rows = [(i, dt.date(2025, 1, 1) + dt.timedelta(days=i % 360), STATES[i % 5], COUNTRIES[i % 4], PLANS[i % 3],
             10 + i % 13) for i in range(1, 1201)]
    con.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)", rows)
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse), client_id="t-places", options=BuildOptions(workers=1))
    index = MemberIndex()
    for slug, members in (("sale.state", STATES), ("sale.country", COUNTRIES), ("sale.plan_name", PLANS)):
        assert slug in model.attributes, sorted(model.attributes)
        index.add(slug, members)
    return con, model, index


def _ask(shop, question, plan):
    con, model, index = shop
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=lambda s, t: json.dumps({"kind": "query", **plan}), index=index,
                        today=dt.date(2026, 6, 1))
    return answer_question(question, services, Session())


Y25 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


def test_a_state_written_out_is_its_code():
    index = MemberIndex()
    index.add("customer.state", STATES)
    index.add("order.status", ["CA", "OP", "CL"])
    assert index.stored("customer.state", "Oregon") == "OR"
    assert index.stored("customer.state", "california") == "CA"
    assert index.stored("customer.state", "Canada") is None        # a country is no state
    assert index.stored("order.status", "California") is None      # a status code is no place


def test_a_country_written_another_way_is_the_stored_one():
    index = MemberIndex()
    index.add("supplier.country", COUNTRIES)
    assert index.stored("supplier.country", "United States") == "USA"
    assert index.stored("supplier.country", "U.S.") == "USA"
    assert index.stored("supplier.country", "germany") == "Germany"


def test_sales_in_oregon_are_the_sales_stored_as_or(shop):
    con = shop[0]
    payload = _ask(shop, 'Sales in "Oregon" in 2025', {"intent": "value", "measures": ["amount"], "time": Y25,
                                                         "filters": [{"field": "sale.state", "op": "eq",
                                                                      "values": ["Oregon"]}]})
    want = con.execute("SELECT SUM(amount) FROM sales WHERE state = 'OR'").fetchone()[0]
    assert payload["data"]["rows"][0]["amount"] == float(want)


def test_leaving_out_a_country_the_data_writes_otherwise_leaves_it_out(shop):
    con = shop[0]
    payload = _ask(shop, 'Sales outside the "United States" in 2025', {
        "intent": "value", "measures": ["amount"], "time": Y25,
        "filters": [{"field": "sale.country", "op": "not_in", "values": ["United States"]}]})
    want = con.execute("SELECT SUM(amount) FROM sales WHERE country <> 'USA'").fetchone()[0]
    assert payload["data"]["rows"][0]["amount"] == float(want)


def test_leaving_out_a_member_there_is_none_of_says_it_changed_nothing(shop):
    payload = _ask(shop, 'Sales outside "Atlantis" in 2025', {
        "intent": "value", "measures": ["amount"], "time": Y25,
        "filters": [{"field": "sale.country", "op": "not_in", "values": ["Atlantis"]}]})
    assert any('"Atlantis"' in c and "changed nothing" in c for c in payload["coverage_caveats"])


def test_two_names_left_unquoted_are_quoted_together_by_one_click(shop):
    # Lone words in lower case: words of the question, until the reader quotes them.
    question = "How many sales on starter in canada in 2025?"
    payload = _ask(shop, question, {"intent": "value", "measures": ["number_of_sales"], "time": Y25, "filters": [
        {"field": "sale.plan_name", "op": "eq", "values": ["Starter"]},
        {"field": "sale.country", "op": "eq", "values": ["Canada"]}]})
    first = payload["follow_up_suggestions"][0]
    assert first["question"] == 'How many sales on "starter" in "canada" in 2025?'
    assert first["label"] == 'Only "Starter" and "Canada"'


def test_two_names_written_as_names_narrow_at_once(shop):
    question = "How many sales on Starter in Canada in 2025?"
    payload = _ask(shop, question, {"intent": "value", "measures": ["number_of_sales"], "time": Y25, "filters": [
        {"field": "sale.plan_name", "op": "eq", "values": ["Starter"]},
        {"field": "sale.country", "op": "eq", "values": ["Canada"]}]})
    assert {f["field"] for f in payload["plan"]["filters"]} == {"sale.plan_name", "sale.country"}
    labels = [c["label"] for c in payload["follow_up_suggestions"]]
    assert 'Not only "Starter"' in labels and 'Not only "Canada"' in labels
