"""Admin → Data → Knowledge base and Dates edit the new core's fields and dates.

The Knowledge base tab opened today's knowledge-base files, which the new core only
imported from, and the Dates tab today's date roles. Now both are the new core's own
pages. A table says what one row is; a field has its name, what it means, the words people
also use, names for its codes ("C" is Cancelled) and whether it is shown, hidden or
sensitive. A table's dates can be named, one chosen as the date it is counted by, one the
build did not find added (checked against the data first), and the fiscal year's first
month set. Each change is an admin decision: it holds from the next answer and through
the next Learn.

Each save starts at the page's own route and ends at a question resolved, run or put to
the AI with it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from unittest.mock import patch

import pytest

from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


def _page(retail, tmp_path, monkeypatch, model=None):
    built, learned = retail
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_knowledge, core2_relationships, core2_routes, routes
    from core2.warehouse.runner import DuckDBWarehouse

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "portal")
    store.update_client_meta(account, client_name="Retail (invented)")
    store.save_core2_model(account, None, (model or learned).model_dump_json())
    app = FastAPI()
    app.include_router(routes.router)
    return app, account, store, DuckDBWarehouse(built.con), [
        patch.object(routes, "_is_auth", return_value=True), patch.object(core2_routes, "_is_auth", return_value=True),
        patch.object(core2_relationships, "_is_auth", return_value=True),
        patch.object(core2_relationships, "_warehouse", lambda *a: DuckDBWarehouse(built.con)),
        patch.object(core2_knowledge, "_is_auth", return_value=True),
        patch.object(core2_knowledge, "_warehouse", lambda *a: DuckDBWarehouse(built.con))]


@pytest.fixture
def page(retail, tmp_path, monkeypatch):
    from contextlib import ExitStack

    from starlette.testclient import TestClient

    app, account, store, warehouse, patches = _page(retail, tmp_path, monkeypatch)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield TestClient(app), account, store, warehouse


def _model(account):
    from core2.bootstrap.service import load_model

    return load_model(account, None)


def _column(model, table: str, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(f".{table}") and c.name == name)


def _table(model, name: str) -> str:
    return next(k for k in model.tables if k.endswith(f".{name}"))


def _plan(plan: dict):
    from core2.plan.ir import Plan

    return Plan.model_validate({"kind": "query", **plan})


def _resolved(model, plan: dict):
    from core2.resolve.resolver import Context, resolve

    return resolve(_plan(plan), model, Context(today=TODAY))


def _answer(model, warehouse, plan: dict) -> list[dict]:
    from core2.compile.compiler import compile_query

    result = warehouse.query(compile_query(_resolved(model, plan), model, warehouse.dialect).sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


def _kb(html: str) -> dict:
    start = html.index('<script type="application/json" id="kb-data">') + len('<script type="application/json" id="kb-data">')
    return json.loads(html[start:html.index("</script>", start)])["table"]


# ── the pages ────────────────────────────────────────────────────────────────

def test_the_tabs_open_the_new_cores_fields_and_dates(page):
    client, account, store, _ = page
    html = client.get(f"/admin/clients/{account}/knowledge").text
    table = _kb(html)
    assert table["name"] == "Order line" and table["grain"] == "one row per order line"
    status = next(f for f in table["fields"] if f["physical"] == "status_code")
    assert status["namable"] and [v["code"] for v in status["values"]] == ["C", "D", "O", "S"]
    assert next(f for f in table["fields"] if f["physical"] == "ship_date_key")["kind"] == "date"
    # The tables are listed by kind, the events first.
    import re

    assert re.findall(r'<div class="kb-grp">([^<]+)</div>', html) == ["Events", "Things you group by", "Dates"]
    customers = _table(_model(account), "customers")
    other = _kb(client.get(f"/admin/clients/{account}/knowledge", params={"table": customers}).text)
    assert other["name"] == "Customer" and any(f["physical"] == "segment" for f in other["fields"])

    dates = client.get(f"/admin/clients/{account}/dates").text
    assert "Order date" in dates and "Ship date" in dates and "Fiscal year starts in" in dates
    order_date = _column(_model(account), "order_lines", "order_date_key")
    assert f'value="{order_date}" id="dt-r-{order_date}"\n          checked' in dates
    # The load stamp is listed but can never be the date a table is counted by.
    loaded = _column(_model(account), "order_lines", "loaded_at")
    assert f'value="{loaded}" id="dt-r-{loaded}"\n           disabled' in dates
    # Today's pages are kept for support under Settings → Diagnostics.
    diagnostics = client.get(f"/admin/clients/{account}/diagnostics").text
    assert f"/admin/clients/{account}/kb" in diagnostics and f"/admin/clients/{account}/date-roles" in diagnostics


# ── fields ───────────────────────────────────────────────────────────────────

def test_named_values_are_what_answers_show_and_what_filters_read(page):
    client, account, store, warehouse = page
    status = _column(_model(account), "order_lines", "status_code")
    r = client.post(f"/admin/clients/{account}/knowledge/api/field",
                    json={"column": status, "value_names": {"C": "Cancelled", "D": "Delivered", "O": "Open", "S": "Shipped"}})
    assert r.json()["ok"] and r.json()["changed"] == ["value_names"]
    assert {v["code"]: v["name"] for v in r.json()["field"]["values"]}["C"] == "Cancelled"
    model = _model(account)

    rows = _answer(model, warehouse, {"measures": ["number_of_order_lines"], "group_by": ["order_line.status_code"]})
    assert sorted(r["order_line_status_code"] for r in rows) == ["Cancelled", "Delivered", "Open", "Shipped"]
    # A reader's word filters on the stored code, and the answer says it the reader's way.
    by_name = {"measures": ["number_of_order_lines"],
               "filters": [{"field": "order_line.status_code", "op": "eq", "values": ["cancelled"]}]}
    by_code = {"measures": ["number_of_order_lines"],
               "filters": [{"field": "order_line.status_code", "op": "eq", "values": ["C"]}]}
    cancelled = _answer(model, warehouse, by_name)
    assert cancelled == _answer(model, warehouse, by_code) and list(cancelled[0].values())[0] == 1919
    for plan in (by_name, by_code):
        logical = _resolved(model, plan)
        assert [c.values for c in logical.conditions if c.kind == "member"] == [["Cancelled"]]
    listed = _answer(model, warehouse, {"group_by": ["order_line.status_code"], "intent": "list"})
    assert [list(r.values())[0] for r in listed] == ["Cancelled", "Delivered", "Open", "Shipped"]

    # The AI reads the names beside the codes, and a question's word is found as the stored code.
    from core2.plan.catalog import catalog_text
    from core2.plan.values import MemberIndex

    assert "Cancelled (C), Delivered (D), Open (O), Shipped (S)" in catalog_text(model)
    index = MemberIndex()
    index.add("order_line.status_code", ["C", "D", "O", "S"])
    found = index.named(model).match("how many cancelled order lines")
    assert [(m.text, m.value) for m in found] == [("cancelled", "C")]
    assert index.named(model).stored("order_line.status_code", "Shipped") == "S"


def test_a_value_name_is_one_of_the_fields_values_and_tells_them_apart(page):
    client, account, store, _ = page
    status = _column(_model(account), "order_lines", "status_code")
    r = client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": status, "value_names": {"X": "Lost"}})
    assert r.status_code == 400 and "not one of this field's values" in r.json()["error"]
    r = client.post(f"/admin/clients/{account}/knowledge/api/field",
                    json={"column": status, "value_names": {"C": "Closed", "S": "closed"}})
    assert r.status_code == 400 and "same name" in r.json()["error"]
    # A field with too many values to name has none to offer.
    net = _column(_model(account), "order_lines", "net_amount")
    r = client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": net, "value_names": {"1": "One"}})
    assert r.status_code == 400
    assert _model(account).columns[status].value_names == {}


def test_a_sensitive_field_is_never_grouped_listed_or_filtered_by(page):
    client, account, store, warehouse = page
    from core2.plan.catalog import catalog_text
    from core2.plan.values import listable
    from core2.resolve.resolver import ResolveError

    segment = _column(_model(account), "customers", "segment")
    assert "customer.segment" in listable(_model(account))
    r = client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": segment, "shown": "sensitive"})
    assert r.json()["ok"] and r.json()["field"]["shown"] == "sensitive"
    model = _model(account)
    for plan in ({"measures": ["net_amount"], "group_by": ["customer.segment"]},
                 {"measures": ["net_amount"], "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]},
                 {"group_by": ["customer.segment"], "intent": "list"},
                 {"group_by": ["customer.name"], "intent": "list",
                  "filters": [{"field": "customer.segment", "op": "eq", "values": ["Retail"]}]}):
        with pytest.raises(ResolveError) as refused:
            _resolved(model, plan)
        assert refused.value.kind == "sensitive" and "Segment is marked sensitive" in refused.value.message
    # The AI knows the field is there, never its values; the member names index leaves it out.
    line = next(x for x in catalog_text(model).splitlines() if x.strip().startswith("- customer.segment"))
    assert "sensitive: never shown" in line and "Retail" not in line
    assert "customer.segment" not in listable(model)
    # Counted inside a metric it still is: only showing it is refused.
    assert _answer(model, warehouse, {"measures": ["net_amount"], **{"time": H1}})

    # Shown again, it answers again.
    client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": segment, "shown": "shown"})
    assert len(_answer(_model(account), warehouse, {"measures": ["net_amount"], "group_by": ["customer.segment"]})) == 3


def test_a_hidden_field_is_not_offered_at_all(page):
    client, account, store, _ = page
    from core2.plan.catalog import catalog_text
    from core2.resolve.resolver import ResolveError

    city = _column(_model(account), "customers", "city")
    client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": city, "shown": "hidden"})
    model = _model(account)
    with pytest.raises(ResolveError) as refused:
        _resolved(model, {"measures": ["net_amount"], "group_by": ["customer.city"]})
    assert refused.value.kind == "unknown" and "customer.city" not in refused.value.options
    assert "customer.city" not in catalog_text(model)
    r = client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": city, "shown": "secret"})
    assert r.status_code == 400


def test_a_fields_name_words_and_meaning_reach_the_ai(page):
    client, account, store, _ = page
    from core2.plan.catalog import catalog_text

    status = _column(_model(account), "order_lines", "status_code")
    r = client.post(f"/admin/clients/{account}/knowledge/api/field", json={
        "column": status, "name": "Order state", "description": "Where the order line is in its life.",
        "synonyms": ["order status", " order status ", "state"]})
    assert sorted(r.json()["changed"]) == ["business_name", "description", "synonyms"]
    model = _model(account)
    assert model.attributes["order_line.status_code"].business_name == "Order state"
    line = next(x for x in catalog_text(model).splitlines() if x.strip().startswith("- order_line.status_code"))
    assert "Order state (Where the order line is in its life.)" in line and "also called: order status, state" in line
    # Saving the same again changes nothing.
    again = client.post(f"/admin/clients/{account}/knowledge/api/field", json={
        "column": status, "name": "Order state", "synonyms": ["order status", "state"]})
    assert again.json()["changed"] == []
    assert client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": status, "name": " "}).status_code == 400


def test_what_one_row_is_is_what_the_ai_reads(page):
    client, account, store, _ = page
    from core2.plan.catalog import catalog_text

    table = _table(_model(account), "order_lines")
    r = client.post(f"/admin/clients/{account}/knowledge/api/table",
                    json={"table": table, "grain": "One product on one customer order."})
    assert r.json() == {"ok": True, "grain": "One product on one customer order."}
    assert "## Order line (One product on one customer order.)" in catalog_text(_model(account))
    assert client.post(f"/admin/clients/{account}/knowledge/api/table", json={"table": table, "grain": ""}).status_code == 400


# ── dates ────────────────────────────────────────────────────────────────────

def test_the_date_a_table_is_counted_by_moves_its_metrics(page):
    client, account, store, warehouse = page
    model = _model(account)
    table = _table(model, "order_lines")
    ship = _column(model, "order_lines", "ship_date_key")
    delivered = _column(model, "order_lines", "delivered_date_key")
    # A default chosen another way before (What QueryBot learned) does not compete with this page's choice.
    from core2.model.overrides import target

    store.set_core2_override(account, None, target("date_role", delivered), "is_default", True)
    r = client.post(f"/admin/clients/{account}/dates/api/default", json={"table": table, "date": ship})
    assert r.json()["ok"]
    model = _model(account)
    assert model.tables[table].default_date == ship and model.date_roles[ship].is_default
    assert not model.date_roles[delivered].is_default
    logical = _resolved(model, {"measures": ["net_amount"], "time": H1})
    assert logical.parts[0].date.role.key == ship
    assert _answer(model, warehouse, {"measures": ["net_amount"], "time": H1}) != _answer(
        model, warehouse, {"measures": ["net_amount"], "time": {**H1, "date": "order_date"}})
    loaded = _column(model, "order_lines", "loaded_at")
    r = client.post(f"/admin/clients/{account}/dates/api/default", json={"table": table, "date": loaded})
    assert r.status_code == 400 and "when the row was written" in r.json()["error"]
    other = _column(model, "returns", "return_date_key")
    assert client.post(f"/admin/clients/{account}/dates/api/default", json={"table": table, "date": other}).status_code == 400


def test_a_metric_whose_own_date_an_admin_chose_keeps_it(page):
    client, account, store, _ = page
    from core2.model.overrides import target

    model = _model(account)
    table = _table(model, "order_lines")
    order_date, ship = _column(model, "order_lines", "order_date_key"), _column(model, "order_lines", "ship_date_key")
    delivered = _column(model, "order_lines", "delivered_date_key")
    by_slug = {m.slug: m.key for m in model.measures.values()}
    # One metric an admin set to the table's own default date, deliberately; one to another date.
    store.set_core2_override(account, None, target("measure", by_slug["quantity"]), "default_date", order_date)
    store.set_core2_override(account, None, target("measure", by_slug["cost_amount"]), "default_date", delivered)
    # The Relationships page sets the table's default: the metrics that followed it follow it still.
    assert client.post(f"/admin/clients/{account}/relationships/save-table",
                       json={"table": table, "default_date": ship}).json()["ok"]
    after = {m.slug: m.default_date for m in _model(account).measures.values() if m.table == table}
    assert after["net_amount"] == ship and after["number_of_orders"] == ship
    assert after["quantity"] == order_date and after["cost_amount"] == delivered


def test_a_date_is_named_and_given_the_words_people_use(page):
    client, account, store, _ = page
    from core2.plan.catalog import catalog_text

    ship = _column(_model(account), "order_lines", "ship_date_key")
    r = client.post(f"/admin/clients/{account}/dates/api/date",
                    json={"date": ship, "name": "Dispatch date", "synonyms": ["shipped", "dispatched"]})
    assert r.json() == {"ok": True, "changed": ["name", "synonyms"]}
    line = next(x for x in catalog_text(_model(account)).splitlines() if x.startswith("- ship_date |"))
    assert "Dispatch date" in line and "also called: dispatched, shipped" in line
    assert client.post(f"/admin/clients/{account}/dates/api/date", json={"date": ship, "name": ""}).status_code == 400


@pytest.fixture
def missed(retail, tmp_path, monkeypatch):
    """The retail model as a Learn that did not find the ship date: neither the date nor its calendar link."""
    from contextlib import ExitStack

    from starlette.testclient import TestClient

    _, learned = retail
    model = learned.model_copy(deep=True)
    ship = next(k for k in model.date_roles if k.endswith("order_lines.ship_date_key"))
    link = model.date_roles.pop(ship).calendar_join
    model.joins.pop(link)
    app, account, store, warehouse, patches = _page(retail, tmp_path, monkeypatch, model)
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield TestClient(app), account, store, warehouse, ship


def test_a_date_the_build_missed_is_checked_and_added(missed):
    client, account, store, warehouse, ship = missed
    from core2.model import knowledge
    from core2.resolve.resolver import ResolveError

    model = _model(account)
    table = _table(model, "order_lines")
    offered = [c["physical"] for c in knowledge.candidates(model, table)]
    assert offered == ["ship_date_key"]           # a line number is not offered as a date
    with pytest.raises(ResolveError):
        _resolved(model, {"measures": ["net_amount"], "time": {**H1, "date": "ship_date"}})

    checked = client.post(f"/admin/clients/{account}/dates/api/check", json={"column": ship}).json()
    assert checked["ok"] and checked["mode"] == "calendar" and checked["granularity"] == "day"
    assert checked["words"] == "Joins the calendar · 94% of rows have one · values from Jan 2023 to Jun 2026"
    bad = client.post(f"/admin/clients/{account}/dates/api/check",
                      json={"column": _column(model, "order_lines", "line_number")})
    assert bad.status_code == 400

    r = client.post(f"/admin/clients/{account}/dates/api/add",
                    json={"column": ship, "name": "Ship date", "synonyms": ["shipped", "dispatched"]})
    assert r.json() == {"ok": True, "date": ship}
    model = _model(account)
    role = model.date_roles[ship]
    assert (role.name, role.slug, role.calendar, role.provenance) == ("Ship date", "ship_date", _table(model, "calendar"),
                                                                     "admin")
    assert model.joins[role.calendar_join].to_calendar and not role.is_default
    assert role.synonyms == {"en": ["shipped", "dispatched"]}
    rows = _answer(model, warehouse, {"measures": ["net_amount"], "time": {**H1, "date": "ship_date"}})
    assert rows and rows != _answer(model, warehouse, {"measures": ["net_amount"], "time": H1})
    # Added once: the column is no longer offered, and the check says so.
    assert knowledge.candidates(model, table) == []
    assert client.post(f"/admin/clients/{account}/dates/api/check", json={"column": ship}).status_code == 400


def test_a_date_is_added_only_as_checked_on_the_server(missed):
    client, account, store, warehouse, ship = missed
    r = client.post(f"/admin/clients/{account}/dates/api/add", json={"column": ship, "name": ""})
    assert r.status_code == 400 and "name" in r.json()["error"]
    loaded = _column(_model(account), "order_lines", "loaded_at")
    r = client.post(f"/admin/clients/{account}/dates/api/add", json={"column": loaded, "name": "Load date"})
    assert r.status_code == 400
    assert ship not in _model(account).date_roles


def test_the_fiscal_year_starts_in_the_month_chosen(page):
    client, account, store, _ = page
    from core2.model.knowledge import fiscal_words

    assert client.post(f"/admin/clients/{account}/dates/api/fiscal", json={"month": 7}).json() == {"ok": True, "month": 7}
    model = _model(account)
    assert model.settings.fiscal_year_start_month == 7
    logical = _resolved(model, {"measures": ["net_amount"], "time": {"window": {"kind": "this", "unit": "year",
                                                                              "fiscal": True}}})
    assert logical.window.start == dt.date(2025, 7, 1)
    assert client.post(f"/admin/clients/{account}/dates/api/fiscal", json={"month": 13}).status_code == 400
    assert fiscal_words(4, dt.date(2026, 10, 9)).startswith("FY2027 runs from April 2026 to March 2027.")
    assert fiscal_words(1, dt.date(2026, 10, 9)) == "The fiscal year is the calendar year."


def test_an_entity_whose_name_field_is_hidden_is_still_grouped_by_but_never_when_sensitive(page):
    client, account, store, warehouse = page
    from core2.resolve.resolver import ResolveError

    name = _column(_model(account), "customers", "customer_name")
    client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": name, "shown": "hidden"})
    plan = {"measures": ["net_amount"], "group_by": ["customer"], "limit": 5}
    assert len(_answer(_model(account), warehouse, plan)) == 5
    with pytest.raises(ResolveError):
        _resolved(_model(account), {"measures": ["net_amount"], "group_by": ["customer.name"]})
    client.post(f"/admin/clients/{account}/knowledge/api/field", json={"column": name, "shown": "sensitive"})
    with pytest.raises(ResolveError) as refused:
        _resolved(_model(account), plan)
    assert refused.value.kind == "sensitive"
