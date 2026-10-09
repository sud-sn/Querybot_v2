"""Admin → Data → Metrics edits the new core's metrics.

The Metrics tab opened today's metric registry, which the new core only imported from.
Now it lists the new core's own metrics: how each is counted, the tables it adds up, its
default date, how often it was asked and where it came from. An admin adds one in their
own words (the workspace's AI writes the formula) or as a formula over any table's fields,
changes a learned one, or hides one; each is checked against the data first and saved as
an admin decision, so it answers from the next question and holds through the next Learn.

Each save starts at the page's own route and ends at a question answered with it.
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


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")


@pytest.fixture
def page(retail, tmp_path, monkeypatch):
    built, model = retail
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_metrics, core2_relationships, core2_routes, routes
    from core2.warehouse.runner import DuckDBWarehouse

    store.init_db()
    account = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account, "Retail (invented)")
    store.save_core2_model(account, None, model.model_dump_json())
    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(core2_routes, "_is_auth", return_value=True), \
            patch.object(core2_relationships, "_is_auth", return_value=True), \
            patch.object(core2_metrics, "_is_auth", return_value=True), \
            patch.object(core2_metrics, "_warehouse", lambda *a: DuckDBWarehouse(built.con)):
        yield TestClient(app), account, store, DuckDBWarehouse(built.con)


def _model(account):
    from core2.bootstrap.service import load_model

    return load_model(account, None)


def _measure(model, slug: str):
    return next(m for m in model.measures.values() if m.slug == slug)


def _column(model, table: str, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(f".{table}") and c.name == name)


def _value(model, warehouse, slug: str) -> float:
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": [slug]}), model,
                      Context(today=TODAY))
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql)
    name = next(c.name for c in compiled.columns if c.role == "measure")
    return float(dict(zip(result.columns, result.rows[0]))[name])


def _json(html: str, element_id: str) -> dict:
    start = html.index(f'<script type="application/json" id="{element_id}">') + len(
        f'<script type="application/json" id="{element_id}">')
    return json.loads(html[start:html.index("</script>", start)])


def test_the_tab_lists_the_new_cores_metrics_and_how_often_each_was_asked(page):
    client, account, store, _ = page
    store.log_core2_answer(account, user_id="u1", mode="core2", question="net sales by month", status="answered",
                           headline="", sql="", row_count=1, plan={"measures": ["net_amount"]}, duration_ms=1,
                           model_version=1)
    html = client.get(f"/admin/clients/{account}/measures").text
    assert "Net amount" in html and "Learned" in html and "Add a metric" in html
    row = html[html.index('aria-label="Edit Net amount"'):]
    row = row[:row.index("</tr>")]
    assert 'data-label="Asked, 30 days">1<' in row and "Order line" in row
    nav = client.get(f"/admin/clients/{account}/learned").text
    assert f'href="/admin/clients/{account}/measures"' in nav


def test_a_workspace_not_learned_yet_is_sent_to_learn(page):
    client, _, store, _ = page
    store.upsert_client("acct-new", "New")
    assert "QueryBot has not learned this database yet" in client.get("/admin/clients/acct-new/measures").text
    assert client.get("/admin/clients/acct-new/measures/edit", follow_redirects=False).status_code == 303


def test_a_metric_written_as_a_formula_over_two_tables_is_saved_and_answers(page):
    client, account, store, warehouse = page
    body = {"formula": "SUM([Order line · Net amount]) - SUM([Return · Refund amount])",
            "name": "Sales after refunds", "synonyms": ["net of returns"], "format": "currency"}
    read = client.post(f"/admin/clients/{account}/measures/api/read", json=body).json()
    assert read["ok"] and read["tables"] == ["Order line", "Return"]
    assert [p["text"] for p in read["parts"]] == ["Net amount", "−", "Refund amount"]
    saved = client.post(f"/admin/clients/{account}/measures/api/save", json=body).json()
    assert saved == {"ok": True, "key": "admin:sales_after_refunds"}
    model = _model(account)
    added = model.measures["admin:sales_after_refunds"]
    assert added.business_name == "Sales after refunds" and added.synonyms == {"en": ["net of returns"]}
    (sales,) = warehouse.query("select sum(net_amount) from order_lines").rows[0]
    (refunds,) = warehouse.query("select sum(refund_amount) from returns").rows[0]
    assert _value(model, warehouse, "sales_after_refunds") == pytest.approx(float(sales) - float(refunds))
    # Edited again, it keeps its key and is defined anew.
    again = client.post(f"/admin/clients/{account}/measures/api/save",
                        json={**body, "key": "admin:sales_after_refunds", "formula": "SUM([Return · Refund amount])"})
    assert again.json()["key"] == "admin:sales_after_refunds"
    assert _value(_model(account), warehouse, "sales_after_refunds") == pytest.approx(float(refunds))


def test_a_learned_metric_changed_keeps_its_key_and_its_condition_is_used(page):
    client, account, store, warehouse = page
    model = _model(account)
    net = _measure(model, "net_amount")
    status = _column(model, "order_lines", "status_code")
    body = {"key": net.key, "formula": "SUM([Order line · Net amount])", "name": "Net sales",
            "conditions": [{"column": status, "op": "ne", "values": ["C"]}]}
    assert client.post(f"/admin/clients/{account}/measures/api/save", json=body).json() == {"ok": True,
                                                                                          "key": net.key}
    after = _model(account)
    changed = after.measures[net.key]
    assert changed.business_name == "Net sales" and changed.provenance == "admin"
    (kept,) = warehouse.query("select sum(net_amount) from order_lines where status_code <> 'C' "
                              "or status_code is null").rows[0]
    assert _value(after, warehouse, net.slug) == pytest.approx(float(kept))
    html = client.get(f"/admin/clients/{account}/measures").text
    row = html[html.index('aria-label="Edit Net sales"'):]
    assert "Changed by an admin" in row[:row.index("</tr>")]


@pytest.mark.parametrize("body,said", [
    ({"formula": "SUM([Order line · Net amount])", "name": "Net amount"}, "There is already a metric called Net amount"),
    ({"formula": "SUM([Order line · Net amount])", "name": ""}, "Give the metric the name"),
    ({"formula": "[Order line · Net amount]", "name": "Raw"}, "is a field: say how to add it up"),
    ({"formula": "SUM([Order line · Net amount])", "name": "Odd",
      "conditions": [{"column": "LINE", "op": "eq", "values": ["C"]}]}, "holds numbers: C is not one"),
    ({"formula": "SUM([Order line · Net amount])", "name": "Odd",
      "conditions": [{"column": "LINE", "op": "gt", "values": []}]}, "a value"),
])
def test_a_metric_that_cannot_be_one_is_refused_in_words(page, body, said):
    client, account, store, _ = page
    model = _model(account)
    for c in body.get("conditions", []):
        c["column"] = _column(model, "order_lines", "line_number")
    before = set(model.measures)
    r = client.post(f"/admin/clients/{account}/measures/api/save", json=body)
    assert r.status_code == 400 and said in r.json()["error"]
    assert set(_model(account).measures) == before


def test_a_draft_is_hidden_from_questions_until_it_is_saved_as_a_metric(page):
    client, account, store, _ = page
    body = {"formula": "AVG([Order line · Net amount])", "name": "Average line", "draft": True}
    key = client.post(f"/admin/clients/{account}/measures/api/save", json=body).json()["key"]
    assert _model(account).measures[key].hidden is True
    client.post(f"/admin/clients/{account}/measures/api/save", json={**body, "key": key, "draft": False})
    assert _model(account).measures[key].hidden is False


def test_hide_and_show_a_learned_metric(page):
    client, account, store, _ = page
    key = _measure(_model(account), "net_amount").key
    assert client.post(f"/admin/clients/{account}/measures/api/hide", json={"key": key, "hidden": True}).json()["ok"]
    assert _model(account).measures[key].hidden is True
    from core2.plan.catalog import catalog_text

    assert "| Net amount |" not in catalog_text(_model(account))
    client.post(f"/admin/clients/{account}/measures/api/hide", json={"key": key, "hidden": False})
    assert _model(account).measures[key].hidden is False


def test_the_check_runs_the_draft_on_the_data(page):
    client, account, store, warehouse = page
    status = _column(_model(account), "order_lines", "status_code")
    checked = client.post(f"/admin/clients/{account}/measures/api/check", json={
        "formula": "SUM([Order line · Net amount])", "name": "Completed sales",
        "conditions": [{"column": status, "op": "ne", "values": ["C"]}]}).json()
    assert checked["ok"] and checked["problem"] == "" and len(checked["months"]) == 6
    last = checked["months"][-1]["period"]
    start, end = int(last.replace("-", "") + "01"), int(last.replace("-", "") + "31")
    (june,) = warehouse.query(f"select sum(net_amount) from order_lines where (status_code <> 'C' or status_code is "
                              f"null) and order_date_key between {start} and {end}").rows[0]
    assert checked["months"][-1]["value"] == pytest.approx(float(june))


def test_described_in_words_the_workspaces_ai_writes_it_and_it_is_read_like_a_typed_one(page):
    client, account, store, _ = page
    asked = {}

    def writer(account_id, client_row, *, description):
        def complete(stable, tail):
            asked.update(stable=stable, tail=tail)
            return json.dumps({"formula": "sum(net_amount) - sum([Return · Refund amount])",
                               "conditions": [{"field": "[Order line · Status code]", "op": "ne", "values": ["C"]}],
                               "name": "Kept sales", "synonyms": ["sales kept"], "format": "currency",
                               "description": "Sales less refunds, cancelled lines left out.",
                               "assumptions": ["C means cancelled"], "question": ""})
        return complete

    with patch("core2.bootstrap.ai.metric_writer", writer):
        written = client.post(f"/admin/clients/{account}/measures/api/describe",
                              json={"description": "sales less refunds, without cancelled lines"}).json()
    assert written["ok"] and written["problem"] == ""
    assert written["formula"] == "SUM([Order line · Net amount]) - SUM([Return · Refund amount])"
    assert written["conditions"] == [{"column": _column(_model(account), "order_lines", "status_code"),
                                      "op": "ne", "values": ["C"]}]
    assert "[Return · Refund amount]" in asked["stable"] and "without cancelled lines" in asked["tail"]
    assert client.post(f"/admin/clients/{account}/measures/api/describe", json={"description": " "}).status_code == 400


def test_the_editor_offers_every_tables_fields_the_metrics_and_the_functions(page):
    client, account, store, _ = page
    key = _measure(_model(account), "net_amount").key
    editor = _json(client.get(f"/admin/clients/{account}/measures/edit?key={key}").text, "me-data")
    refs = {f["ref"] for f in editor["fields"]}
    assert {"Order line · Net amount", "Return · Refund amount", "Customer · Segment"} <= refs
    assert "Net amount" not in {m["ref"] for m in editor["metrics"]}          # not itself
    assert {"SUM", "COUNT"} <= {f["name"] for f in editor["functions"]}
    assert editor["state"]["formula"] == "SUM([Order line · Net amount])" and editor["state"]["key"] == key
    assert client.get(f"/admin/clients/{account}/measures/edit?key=nothing",
                      follow_redirects=False).status_code == 303


def test_an_answer_does_not_say_rows_are_counted_that_the_metrics_own_condition_leaves_out(page):
    client, account, store, _ = page
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    model = _model(account)
    net = _measure(model, "net_amount")
    plan = Plan.model_validate({"kind": "query", "intent": "value", "measures": [net.slug]})
    said = "Includes rows whose status code is C"
    assert any(said in n for n in resolve(plan, model, Context(today=TODAY)).notes)
    status = _column(model, "order_lines", "status_code")
    client.post(f"/admin/clients/{account}/measures/api/save", json={
        "key": net.key, "formula": "SUM([Order line · Net amount])", "name": "Net amount",
        "conditions": [{"column": status, "op": "ne", "values": ["C"]}]})
    assert not any(said in n for n in resolve(plan, _model(account), Context(today=TODAY)).notes)
