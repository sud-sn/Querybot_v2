"""Admin → Data → Relationships edits the new core's links and the rows each table leaves out.

The Relationships tab opened today's entity graph, which the new core only imported
from. Now it is the new core's own page: the learned links drawn around one table and
listed in full, each with its columns, how many rows it matched and its condition; a
link's condition, columns and name can be changed (or a link added, or turned off), a
table's name, kind, default date and always-leave-out rows set, each checked against
the data first. A change is an admin decision: it holds from the next answer and
through the next Learn, and a learned link that is changed is turned off rather than
edited, so the next Learn cannot bring it back.

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
H1 = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}}


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

    from admin import core2_relationships, core2_routes, routes
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
            patch.object(core2_relationships, "_warehouse", lambda *a: DuckDBWarehouse(built.con)):
        yield TestClient(app), account, store, DuckDBWarehouse(built.con)


def _model(store, account):
    from core2.bootstrap.service import load_model

    return load_model(account, None)


def _column(model, table: str, name: str) -> str:
    return next(c.key for c in model.columns.values() if c.table.endswith(f".{table}") and c.name == name)


def _link(model, source: str, target: str):
    return next(j for j in model.joins.values() if j.from_table.endswith(f".{source}") and j.to_table.endswith(f".{target}"))


def _answer(model, warehouse, plan: dict) -> list[dict]:
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    logical = resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))
    result = warehouse.query(compile_query(logical, model, warehouse.dialect).sql)
    return [dict(zip(result.columns, row)) for row in result.rows]


def _data(html: str) -> dict:
    start = html.index('<script type="application/json" id="rel-data">') + len('<script type="application/json" id="rel-data">')
    return json.loads(html[start:html.index("</script>", start)])


def test_the_tab_opens_the_new_cores_links_and_tables(page):
    client, account, store, _ = page
    html = client.get(f"/admin/clients/{account}/relationships").text
    data = _data(html)
    names = {(j["from_name"], j["to_name"]) for j in data["joins"]}
    assert ("Order line", "Customer") in names and ("Return", "Order line") in names
    assert {"Order line", "Customer", "Return"} <= {t["name"] for t in data["tables"]}
    assert f'href="/admin/clients/{account}/relationships"' in client.get(f"/admin/clients/{account}/learned").text


def test_a_workspace_not_learned_yet_is_sent_to_learn(page):
    client, _, store, _ = page
    store.upsert_client("acct-new", "New")
    html = client.get("/admin/clients/acct-new/relationships").text
    assert "QueryBot has not learned this database yet" in html and "rel-data" not in html


def test_a_condition_saved_on_a_link_is_used_by_the_next_answer(page):
    client, account, store, warehouse = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    body = {"key": link.key, "from": link.from_table, "to": link.to_table,
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)],
            "conditions": [{"column": segment, "op": "eq", "values": ["Wholesale"]}]}
    checked = client.post(f"/admin/clients/{account}/relationships/check-link", json=body).json()
    assert checked["ok"] and 0 < checked["match"] < 1 and checked["without"]["match"] == 1.0
    assert client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json() == {"ok": True,
                                                                                                  "key": link.key}
    rows = _answer(_model(store, account), warehouse, {"intent": "breakdown", "measures": ["net_amount"],
                                                       "group_by": ["customer.segment"], "time": H1})
    assert {r["customer_segment"] for r in rows} == {"Wholesale", None}


def test_a_link_changed_to_other_columns_turns_the_learned_one_off(page):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "returns", "customers")
    body = {"key": link.key, "from": link.from_table, "to": link.to_table, "role": "Returned by",
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)]}
    saved = client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json()
    assert saved["ok"] and saved["key"] != link.key
    after = _model(store, account)
    assert after.joins[link.key].trust == "rejected"
    assert after.joins[saved["key"]].trust == "admin" and after.joins[saved["key"]].role == "Returned by"
    # Taken back out, the admin's own link goes and the learned one stays off until turned on.
    assert client.post(f"/admin/clients/{account}/relationships/turn-off-link", json={"key": saved["key"]}).json()["ok"]
    assert saved["key"] not in _model(store, account).joins
    assert client.post(f"/admin/clients/{account}/relationships/turn-on-link", json={"key": link.key}).json()["ok"]
    assert _model(store, account).joins[link.key].trust == "admin"


def test_a_new_link_is_added_and_one_that_exists_is_refused(page):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "order_lines", "products")
    pairs = [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)]
    refused = client.post(f"/admin/clients/{account}/relationships/save-link",
                          json={"from": link.from_table, "to": link.to_table, "pairs": pairs})
    assert refused.status_code == 400 and "already linked" in refused.json()["error"]


@pytest.mark.parametrize("change,said", [
    ({"pairs": []}, "at least one column"),
    ({"to": "no.such.table"}, "Choose the two tables"),
    ({"conditions": [{"column": "COLUMN_OF_FROM", "op": "eq", "values": ["C"]}]}, "choose one of its columns"),
    ({"conditions": [{"column": "SEGMENT", "op": "gt", "values": []}]}, "a value"),
    ({"conditions": [{"column": "CUSTOMER_ID", "op": "eq", "values": ["abc"]}]}, "holds numbers: abc is not one"),
])
def test_a_link_that_cannot_be_one_is_refused_in_words(page, change, said):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "order_lines", "customers")
    keys = {"COLUMN_OF_FROM": _column(model, "order_lines", "status_code"),
            "SEGMENT": _column(model, "customers", "segment"), "CUSTOMER_ID": _column(model, "customers", "customer_id")}
    body = {"key": link.key, "from": link.from_table, "to": link.to_table,
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)], **change}
    for c in body.get("conditions", []):
        c["column"] = keys[c["column"]]
    r = client.post(f"/admin/clients/{account}/relationships/save-link", json=body)
    assert r.status_code == 400 and said in r.json()["error"]
    assert _model(store, account).joins[link.key].conditions == []


def test_rows_a_table_always_leaves_out_are_checked_saved_and_left_out(page):
    client, account, store, warehouse = page
    model = _model(store, account)
    table, status = _link(model, "order_lines", "customers").from_table, _column(model, "order_lines", "status_code")
    body = {"table": table, "name": "Order line", "kind": "fact",
            "leaves_out": [{"column": status, "op": "ne", "values": ["C"]}]}
    checked = client.post(f"/admin/clients/{account}/relationships/check-table", json=body).json()
    (cancelled,) = warehouse.query("select count(*) from order_lines where status_code = 'C'").rows[0]
    assert (checked["left_out"], checked["measure"]) == (cancelled, "Net amount")
    # The amount is the one left out of the table's latest year, as an answer for that year would show it.
    role = model.date_roles[model.tables[table].default_date]
    assert checked["year"] == str(role.last.year)
    column, year = model.columns[role.column].name, role.last.year
    in_the_year = (f"{column} between {year}0101 and {year}1231" if role.calendar  # a yyyymmdd key into the calendar
                   else f"extract(year from {column}) = {year}")
    (in_year,) = warehouse.query(f"select sum(net_amount) from order_lines where status_code = 'C' "
                                 f"and {in_the_year}").rows[0]
    assert checked["amount"] == pytest.approx(float(in_year))
    assert checked["says"] == "Order line: rows where Status code is C are left out (a default filter)."
    assert client.post(f"/admin/clients/{account}/relationships/save-table", json=body).json()["ok"]
    (row,) = _answer(_model(store, account), warehouse, {"intent": "value", "measures": ["net_amount"]})
    (kept,) = warehouse.query("select sum(net_amount) from order_lines where status_code <> 'C' "
                              "or status_code is null").rows[0]
    assert float(row["net_amount"]) == pytest.approx(float(kept))


def test_a_table_rule_with_a_value_its_column_cannot_hold_is_refused(page):
    client, account, store, _ = page
    model = _model(store, account)
    table = _link(model, "order_lines", "customers").from_table
    line = _column(model, "order_lines", "line_number")
    r = client.post(f"/admin/clients/{account}/relationships/save-table",
                    json={"table": table, "leaves_out": [{"column": line, "op": "eq", "values": ["C"]}]})
    assert r.status_code == 400 and "holds numbers" in r.json()["error"]
    assert _model(store, account).tables[table].default_filters == []


def test_a_table_renamed_and_given_another_default_date(page):
    client, account, store, _ = page
    model = _model(store, account)
    table = _link(model, "order_lines", "customers").from_table
    ship = next(r.key for r in model.date_roles.values() if r.table == table and "ship" in r.name.lower())
    assert client.post(f"/admin/clients/{account}/relationships/save-table",
                       json={"table": table, "name": "Sales line", "default_date": ship}).json()["ok"]
    after = _model(store, account)
    assert after.tables[table].business_name == "Sales line" and after.tables[table].default_date == ship
    another_tables_date = next(r.key for r in after.date_roles.values() if r.table != table)
    r = client.post(f"/admin/clients/{account}/relationships/save-table",
                    json={"table": table, "default_date": another_tables_date})
    assert r.status_code == 400 and "this table's own dates" in r.json()["error"]


def _resolved(model, plan: dict):
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    return resolve(Plan.model_validate({"kind": "query", **plan}), model, Context(today=TODAY))


def test_a_join_that_keeps_matched_rows_only_leaves_the_others_out_and_says_so(page):
    client, account, store, warehouse = page
    model = _model(store, account)
    link, segment = _link(model, "order_lines", "customers"), _column(model, "customers", "segment")
    body = {"key": link.key, "from": link.from_table, "to": link.to_table,
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)],
            "conditions": [{"column": segment, "op": "eq", "values": ["Wholesale"]}], "keep_unmatched": False}
    assert client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json()["ok"]
    after = _model(store, account)
    assert after.joins[link.key].keep_unmatched is False
    plan = {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["customer.segment"], "time": H1}
    rows = _answer(after, warehouse, plan)
    assert {r["customer_segment"] for r in rows} == {"Wholesale"}
    assert any("rows with no customer are left out" in n for n in _resolved(after, plan).notes)
    # Kept again, the rows come back as Unknown.
    assert client.post(f"/admin/clients/{account}/relationships/save-link",
                       json={**body, "keep_unmatched": True}).json()["ok"]
    assert {r["customer_segment"] for r in _answer(_model(store, account), warehouse, plan)} == {"Wholesale", None}


def test_rows_left_out_come_back_when_a_reader_asks_and_the_admin_lets_them(page):
    client, account, store, warehouse = page
    model = _model(store, account)
    table, status = _link(model, "order_lines", "customers").from_table, _column(model, "order_lines", "status_code")
    rules = [{"column": status, "op": "ne", "values": ["C"]}]
    assert client.post(f"/admin/clients/{account}/relationships/save-table",
                       json={"table": table, "leaves_out": rules, "readers_may_include": True}).json()["ok"]
    (everything,) = warehouse.query("select sum(net_amount) from order_lines").rows[0]
    asked = {"intent": "value", "measures": ["net_amount"], "include_left_out": True}
    (row,) = _answer(_model(store, account), warehouse, asked)
    assert float(row["net_amount"]) == pytest.approx(float(everything))
    assert "Including the order line rows left out by default (Status code is C), as asked." in \
        _resolved(_model(store, account), asked).notes
    # The admin says readers may not: the rows stay out even when asked.
    assert client.post(f"/admin/clients/{account}/relationships/save-table",
                       json={"table": table, "leaves_out": rules, "readers_may_include": False}).json()["ok"]
    after = _model(store, account)
    assert after.tables[table].readers_may_include is False
    (row,) = _answer(after, warehouse, asked)
    (kept,) = warehouse.query("select sum(net_amount) from order_lines where status_code <> 'C' "
                              "or status_code is null").rows[0]
    assert float(row["net_amount"]) == pytest.approx(float(kept))


def test_the_planner_is_told_which_rows_a_reader_may_ask_for(page):
    from core2.plan.catalog import catalog_text

    client, account, store, _ = page
    model = _model(store, account)
    table, status = _link(model, "order_lines", "customers").from_table, _column(model, "order_lines", "status_code")
    rules = [{"column": status, "op": "ne", "values": ["C"]}]
    client.post(f"/admin/clients/{account}/relationships/save-table", json={"table": table, "leaves_out": rules})
    assert "leaves out rows unless asked: Status code is C" in catalog_text(_model(store, account))
    client.post(f"/admin/clients/{account}/relationships/save-table",
                json={"table": table, "leaves_out": rules, "readers_may_include": False})
    assert "always leaves out: Status code is C" in catalog_text(_model(store, account))


def test_check_all_matches_every_join_in_use_again(page):
    client, account, store, warehouse = page
    off = _link(_model(store, account), "returns", "customers").key
    assert client.post(f"/admin/clients/{account}/relationships/turn-off-link", json={"key": off}).json()["ok"]
    model = _model(store, account)
    checked = client.post(f"/admin/clients/{account}/relationships/check-all", json={}).json()
    assert checked["ok"]
    in_use = {j.key for j in model.joins.values() if j.trust != "rejected"}
    assert set(checked["joins"]) == in_use and off not in in_use
    link = _link(model, "order_lines", "customers")
    (matched,) = warehouse.query("select avg(case when c.customer_id is null then 0.0 else 1.0 end) from order_lines o "
                                 "left join customers c on o.customer_id = c.customer_id "
                                 "where o.customer_id is not null").rows[0]
    assert checked["joins"][link.key]["match"] == pytest.approx(float(matched), abs=1e-4)
    assert checked["joins"][link.key]["twice"] == 0


def test_the_page_carries_what_its_editor_needs(page):
    client, account, store, _ = page
    data = _data(client.get(f"/admin/clients/{account}/relationships").text)
    join = next(j for j in data["joins"] if (j["from_name"], j["to_name"]) == ("Order line", "Customer"))
    assert join["keep_unmatched"] is True and join["pairs"] and join["status"] in ("Confirmed", "Suggested")
    table = next(t for t in data["tables"] if t["name"] == "Order line")
    assert table["readers_may_include"] is True and table["dates"] and table["columns"]
    assert data["counts"]["all"] == sum(1 for j in data["joins"] if j["status"] != "Turned off")


def test_a_join_said_to_be_one_to_one_is_saved_so(page):
    client, account, store, _ = page
    model = _model(store, account)
    link = _link(model, "order_lines", "customers")
    body = {"key": link.key, "from": link.from_table, "to": link.to_table, "cardinality": "one_to_one",
            "pairs": [{"from": f, "to": t} for f, t in zip(link.from_columns, link.to_columns)]}
    assert client.post(f"/admin/clients/{account}/relationships/save-link", json=body).json() == {"ok": True,
                                                                                                  "key": link.key}
    assert _model(store, account).joins[link.key].cardinality == "one_to_one"
    # Only the two a question can walk: anything else is read as many to one.
    client.post(f"/admin/clients/{account}/relationships/save-link", json={**body, "cardinality": "many_to_many"})
    assert _model(store, account).joins[link.key].cardinality == "many_to_one"
