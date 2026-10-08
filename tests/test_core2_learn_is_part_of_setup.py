"""Learning the database is setup's last step, and says when discovery has moved on.

* The setup page ends with "Learn the database for the new core": not learned
  yet, learned (which version, who answers the portal), or out of date.
* Discovery run again with other tables, columns or masking marks Learn out of
  date, on the setup page and on the learned page, until QueryBot learns again.
  A model learned before masking was part of the fingerprint is not called out
  of date when nothing changed.
* A column masked after Learn keeps its values out at once: they are not put
  before the AI, and its members are not matched, before QueryBot learns again.
* The fiscal year is set on the learned page when the data has no fiscal
  calendar: questions use it, and "use what the data says" undoes it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import html
import json
from unittest.mock import patch

import pytest

from tests.test_core2_learned_page_shows_what_was_learned import (
    ACCOUNT,
    _page,
    _post,
    _request,
    _schema_file,
    workspace,  # noqa: F401 - the fixture
)

BEHIND = "Discovery has run since QueryBot last learned"


@pytest.fixture
def learned(workspace, monkeypatch):  # noqa: F811
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    from admin import core2_routes
    from core2 import service

    monkeypatch.setattr(service, "_INDEXES", type(service._INDEXES)())
    core2_routes._run_build(ACCOUNT)
    return store, built, schema_dir


def _setup_page() -> str:
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        response = asyncio.run(routes.client_setup_page(
            _request(f"/admin/clients/{ACCOUNT}/setup", method="GET"), ACCOUNT))
    return html.unescape(response.body.decode())


def _rediscover(schema_dir, change) -> None:
    """Discovery runs again and writes what it found now."""
    path = schema_dir / "_schema.json"
    schema = json.loads(path.read_text())
    change(schema)
    path.write_text(json.dumps(schema))


def _customers(schema) -> dict:
    return next(v for k, v in schema.items() if k.lower().endswith(".customers"))


def test_setup_ends_with_learning_the_database(workspace):  # noqa: F811
    store, built, schema_dir = workspace
    page = _setup_page()
    assert "Learn the database for the new core" in page and "Not learned yet" in page
    assert "Learn this database</button>" not in page, "nothing to learn from before discovery"
    _schema_file(built, schema_dir)
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    page = _setup_page()
    assert "Learned: version 1" in page and "Out of date" not in page
    assert "today's pipeline; choose the new core" in page
    store.set_query_engine(ACCOUNT, "core2")
    assert "answered by the new core" in " ".join(_setup_page().split())


def test_discovery_run_again_marks_learn_out_of_date_until_it_learns_again(learned):
    store, built, schema_dir = learned
    assert BEHIND not in _page() and "Out of date" not in _setup_page()
    _rediscover(schema_dir, lambda s: _customers(s)["columns"].append(
        {"name": "loyalty_tier", "type": "VARCHAR", "nullable": True, "comment": ""}))
    page = _setup_page()
    assert BEHIND in _page() and "Out of date" in page and "Learn again</button>" in page
    from admin import core2_routes

    core2_routes._run_build(ACCOUNT)
    assert BEHIND not in _page() and "Learned: version 2" in _setup_page()


def test_a_model_learned_before_masking_was_counted_is_not_out_of_date_for_nothing(learned):
    store, built, schema_dir = learned
    import hashlib

    from core2.bootstrap.inventory import from_schema_json
    from core2.bootstrap.service import source_hash

    inventory = from_schema_json(json.loads((schema_dir / "_schema.json").read_text()), "duckdb")
    before = sorted((k, [(c.name, c.raw_type) for c in t.columns]) for k, t in inventory.tables.items())
    assert source_hash(inventory) == hashlib.sha256(json.dumps(before).encode()).hexdigest()[:16]


def _ask(store, built, monkeypatch, question: str) -> str:
    import core.compliance.governed_query as gq
    import core2.bootstrap.ai as ai
    import sqlglot
    from core2 import service

    def run_query(credentials, db_type, sql, max_rows=200):
        cursor = built.con.cursor()
        cursor.execute(sqlglot.transpile(sql, read="tsql", write="duckdb")[0])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchmany(max_rows)]

    monkeypatch.setattr(gq, "run_query", run_query)
    seen: list[str] = []

    def planner(*args, **kwargs):
        def complete(stable, tail):
            seen.append(stable + tail)
            return json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"]})
        return complete

    monkeypatch.setattr(ai, "workspace_planner", planner)
    service.portal_answer(ACCOUNT, question, {"id": 1, "role": "admin"}, session_key=question)
    return seen[0]


def test_a_column_masked_after_learn_keeps_its_values_out_at_once(learned, monkeypatch):
    store, built, schema_dir = learned
    code = str(built.con.execute('SELECT MIN(customer_code) FROM "customers"').fetchone()[0])
    assert "VALUE MATCHES (" in _ask(store, built, monkeypatch, f"net sales for {code}")

    # The admin masks the customer codes and discovery runs again; QueryBot has not learned again.
    _rediscover(schema_dir, lambda s: _customers(s).update(masked_fields=["customer_code"], mask_mode="selective"))
    assert BEHIND in _page()
    prompt = _ask(store, built, monkeypatch, f"net sales for {code} again")
    assert "VALUE MATCHES (" not in prompt and prompt.count(code) == 1
    from core2.bootstrap.service import answering_model
    from core2.plan.values import listable

    model = answering_model(ACCOUNT, store.get_client(ACCOUNT))
    assert "customer.code" not in listable(model)
    assert not model.columns[model.attributes["customer.code"].column].values_allowed


def test_the_fiscal_year_set_on_the_learned_page_is_what_questions_use(learned):
    store, *_ = learned
    from admin import core2_routes
    from core2.bootstrap.service import load_model
    from core2.plan.catalog import catalog_text
    from core2.plan.ir import Window
    from core2.resolve.time import resolve_window

    db_id = store.get_client(ACCOUNT)["db_config_id"]
    # The retail warehouse's calendar table says its fiscal years start in April.
    assert load_model(ACCOUNT, db_id).settings.fiscal_year_start_month == 4
    page = " ".join(_page().split())
    assert "Fiscal years start in April: FY2026 runs from April 2025 to March 2026. (Found in the calendar table.)" in page

    _post(core2_routes.learned_decide, f"/admin/clients/{ACCOUNT}/learned/decide",
          {"target": "settings:workspace", "field": "fiscal_year_start_month", "value": "7", "note": ""})
    model = load_model(ACCOUNT, db_id)
    assert model.settings.fiscal_year_start_month == 7
    assert "The fiscal year starts in July" in catalog_text(model)
    this_year = resolve_window(Window(kind="this", unit="year", fiscal=True), today=dt.date(2026, 10, 8),
                               fiscal_start=model.settings.fiscal_year_start_month)
    assert this_year.start == dt.date(2026, 7, 1)
    page = " ".join(_page().split())
    assert "Fiscal years start in July: FY2026 runs from July 2025 to June 2026. (Your decision.)" in page

    _post(core2_routes.learned_decide, f"/admin/clients/{ACCOUNT}/learned/decide",
          {"target": "settings:workspace", "field": "fiscal_year_start_month", "value": "13", "note": ""})
    assert load_model(ACCOUNT, db_id).settings.fiscal_year_start_month == 4, "a month that is not one is refused"

    _post(core2_routes.learned_undo, f"/admin/clients/{ACCOUNT}/learned/undo",
          {"target": "settings:workspace", "field": "fiscal_year_start_month"})
    assert load_model(ACCOUNT, db_id).settings.fiscal_year_start_month == 4
    assert "(Your decision.)" not in _page()


def test_without_a_fiscal_calendar_the_page_says_the_calendar_year_is_used(learned):
    store, *_ = learned
    from core2.bootstrap.service import load_model
    from core2.model.view import learned_view

    model = load_model(ACCOUNT, store.get_client(ACCOUNT)["db_config_id"])
    model.calendars.clear()
    model.settings.fiscal_year_start_month = None
    fiscal = learned_view(model)["fiscal"]
    assert fiscal == {"month": None, "found": None}
