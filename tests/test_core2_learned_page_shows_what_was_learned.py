"""The "What QueryBot learned" page shows what a build learned, and an admin's
decision on it takes effect and can be undone.

The build runs the way production runs it -- discovery's _schema.json read
from the workspace's schema folder, the warehouse studied through a connection,
the model stored as a version -- with a DuckDB copy of the synthetic retail
warehouse standing in for the connection. Nothing is handed from the build to
the page by the test: the page must find what the build stored.
"""

from __future__ import annotations

import asyncio
import html
import json
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from starlette.requests import Request

from core2.warehouse.runner import DuckDBWarehouse
from evals.core2.domains import retail
from evals.core2.framework import materialize

ACCOUNT = "acct-core2-learned"


class _Connection(DuckDBWarehouse):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _request(path: str, fields: dict | None = None, method: str = "POST", query: str = "") -> Request:
    body = urlencode(fields or {}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({
        "type": "http", "method": method, "path": path, "root_path": "", "scheme": "http",
        "query_string": query.encode(), "server": ("testserver", 80), "client": ("127.0.0.1", 1),
        "headers": [(b"content-type", b"application/x-www-form-urlencoded"),
                    (b"content-length", str(len(body)).encode())],
    }, receive)


def _schema_file(built, folder) -> None:
    """What discovery writes for this warehouse, in the shape it writes it."""
    con = built.con
    out: dict = {}
    for logical, physical in built.tables.items():
        columns = con.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            f"WHERE table_name = '{physical}' ORDER BY ordinal_position").fetchall()
        out[f"MEMORY.MAIN.{physical}"] = {
            "columns": [{"name": c, "type": t, "nullable": n == "YES", "comment": ""} for c, t, n in columns],
            "pk_columns": built.declared_pks.get(physical, []),
            "row_count": con.execute(f'SELECT COUNT(*) FROM "{physical}"').fetchone()[0],
            "schema": "main", "database": "memory",
        }
    out["__db_fk_constraints__"] = built.declared_fks
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "_schema.json").write_text(json.dumps(out, default=str))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    import store.crypto

    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    store.upsert_client(ACCOUNT, "web")
    store.save_compliance_profile(ACCOUNT, mode="standard")
    db_id = store.save_db_config("azure_sql", "retail", {"server": "s", "database": "d", "user": "u",
                                                          "password": "p"})
    store.update_client_meta(ACCOUNT, db_config_id=db_id)
    built = materialize(retail.build(), "descriptive")
    schema_dir = tmp_path / "clients" / ACCOUNT / "schema"
    store.update_client_state(ACCOUNT, "READY", {"schema_dir": str(schema_dir)})
    from core2.warehouse import querybot

    monkeypatch.setattr(querybot, "QueryBotWarehouse", lambda db_type, credentials: _Connection(built.con))
    return store, built, schema_dir


def _page(query: str = "") -> str:
    from admin import core2_routes

    with patch.object(core2_routes, "_is_auth", return_value=True):
        response = asyncio.run(core2_routes.learned_page(
            _request(f"/admin/clients/{ACCOUNT}/learned", method="GET", query=query), ACCOUNT))
    return html.unescape(response.body.decode())


def _post(handler, path: str, fields: dict):
    from admin import core2_routes

    with patch.object(core2_routes, "_is_auth", return_value=True):
        return asyncio.run(handler(_request(path, fields), ACCOUNT, **fields))


def test_before_any_build_the_page_says_what_to_do(workspace):
    page = _page()
    assert "has not studied this database yet" in page and "Learn this database" in page


def test_a_build_without_discovery_says_so(workspace):
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    assert "Run discovery first" in _page()


def test_what_a_build_learns_is_what_the_page_shows(workspace):
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    assert store.latest_core2_build(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])["status"] == "done"
    page = _page()
    # The learned findings, in business words, with their evidence.
    assert "Order date" in page and "Ship date" in page and "Delivered date" in page
    assert "Home store" in page
    assert "Net amount" in page and "taken at the end of each period" not in page.split("Net amount")[1][:200]
    assert "rows are stamped when they are loaded" in page
    assert "Fiscal years start in month 4" in page
    assert "is the same (38) on every row" in page
    assert "Version 1" in page


def test_a_decision_applies_at_once_and_can_be_undone(workspace):
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes
    from core2.bootstrap.service import load_model
    from core2.model.overrides import target

    core2_routes._run_build(ACCOUNT)
    db_id = store.get_client(ACCOUNT)["db_config_id"]
    model = load_model(ACCOUNT, db_id)
    ship = next(r for r in model.date_roles.values() if r.name == "Ship date")
    order = next(r for r in model.date_roles.values() if r.name == "Order date")
    assert order.is_default and not ship.is_default

    path = f"/admin/clients/{ACCOUNT}/learned/decide"
    response = _post(core2_routes.learned_decide, path,
                     {"target": target("date_role", ship.key), "field": "is_default", "value": "true", "note": ""})
    assert response.status_code == 303 and "saved=decision" in response.headers["location"]
    model = load_model(ACCOUNT, db_id)
    assert model.date_roles[ship.key].is_default and not model.date_roles[order.key].is_default
    assert model.tables[ship.table].default_date == ship.key
    assert "Your decisions" in _page()

    # A rebuild keeps the decision.
    core2_routes._run_build(ACCOUNT)
    assert load_model(ACCOUNT, db_id).date_roles[ship.key].is_default

    response = _post(core2_routes.learned_undo, f"/admin/clients/{ACCOUNT}/learned/undo",
                     {"target": target("date_role", ship.key), "field": "is_default"})
    assert response.status_code == 303
    model = load_model(ACCOUNT, db_id)
    assert model.date_roles[order.key].is_default and not model.date_roles[ship.key].is_default


def test_cancelled_rows_are_left_out_from_the_page_and_answers_follow(workspace):
    """The note on an answer ("an admin can leave them out by default") names a button that exists (E3)."""
    import re

    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes
    from core2.bootstrap.service import load_model
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    core2_routes._run_build(ACCOUNT)
    with patch.object(core2_routes, "_is_auth", return_value=True):
        raw = asyncio.run(core2_routes.learned_page(
            _request(f"/admin/clients/{ACCOUNT}/learned", method="GET"), ACCOUNT)).body.decode()
    form = next(f for f in re.findall(r"<form[^>]*learned/decide.*?</form>", raw, re.S) if "Leave out C" in f)
    # What a browser sends: the attribute values as written in the page, unescaped.
    fields = {k: html.unescape(v) for k, v in re.findall(r'name="(target|field|value|note)" value="([^"]*)"', form)}
    response = _post(core2_routes.learned_decide, f"/admin/clients/{ACCOUNT}/learned/decide", fields)
    assert response.status_code == 303 and "saved=decision" in response.headers["location"]
    assert "Left out: C (your decision)" in _page()

    model = load_model(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": ["net_amount"]}),
                      model, Context(today=__import__("datetime").date(2026, 6, 15)))
    assert not any("an admin can leave them out" in n for n in logical.notes)
    got = DuckDBWarehouse(built.con).query(compile_query(logical, model, "duckdb").sql).rows[0][0]
    want = built.con.execute("SELECT SUM(net_amount) FROM order_lines WHERE status_code <> 'C'").fetchone()[0]
    assert float(got) == pytest.approx(float(want))


def test_a_measure_is_renamed_given_words_and_told_how_it_adds_up_from_the_page(workspace):
    import re

    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes
    from core2.bootstrap.service import load_model
    from core2.plan.catalog import catalog_text

    core2_routes._run_build(ACCOUNT)
    with patch.object(core2_routes, "_is_auth", return_value=True):
        raw = asyncio.run(core2_routes.learned_page(
            _request(f"/admin/clients/{ACCOUNT}/learned", method="GET"), ACCOUNT)).body.decode()
    form = next(f for f in re.findall(r"<form[^>]*learned/measure.*?</form>", raw, re.S)
                if 'value="Net amount"' in f)
    target = html.unescape(re.search(r'name="target" value="([^"]*)"', form).group(1))
    fields = {"target": target, "name": "Revenue", "synonyms": "sales, turnover , sales", "adds_up": "last"}
    response = _post(core2_routes.learned_measure, f"/admin/clients/{ACCOUNT}/learned/measure", fields)
    assert response.status_code == 303 and "saved=decision" in response.headers["location"]

    model = load_model(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    measure = model.measures[target.partition(":")[2]]
    assert measure.business_name == "Revenue" and measure.synonyms == {"en": ["sales", "turnover"]}
    assert (measure.additivity, measure.time_aggregation) == ("semi_additive", "last")
    assert "also called: sales, turnover" in catalog_text(model)
    page = _page()
    assert "Also called: sales, turnover" in page and "Your decisions" in page

    # Emptying the words takes them away; the other decisions stand.
    _post(core2_routes.learned_measure, f"/admin/clients/{ACCOUNT}/learned/measure",
          {"target": target, "name": "Revenue", "synonyms": " ", "adds_up": "last"})
    measure = load_model(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"]).measures[target.partition(":")[2]]
    assert measure.synonyms == {"en": []} and measure.business_name == "Revenue"


def test_a_field_the_data_decides_cannot_be_set_from_the_page(workspace):
    from admin import core2_routes

    response = _post(core2_routes.learned_decide, f"/admin/clients/{ACCOUNT}/learned/decide",
                     {"target": "join:x", "field": "match_rate", "value": "1", "note": ""})
    assert "error=" in response.headers["location"]
