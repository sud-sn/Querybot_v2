"""What an admin already decided in today's setup is what the new core uses (DESIGN §12.4).

Links confirmed or rejected, the default date of a fact table, metrics built in
today's metric builder, descriptions, display names, synonyms and confirmed
meanings come over as overrides authored "import". Names are matched by the
warehouse's spelling and anything that matches nothing is reported, not guessed;
a metric written as SQL is reported, never pasted. A metric that comes over is
answered by the new core with the warehouse's numbers. An admin's decision on
the new core's own page outranks an import, and a decision dropped from today's
setup disappears at the next import.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.compile.compiler import compile_query
from core2.model.imports import AUTHOR, Legacy, decisions, write
from core2.model.overrides import apply_overrides
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)

ENTITIES = [{"entity_name": "OrderLine", "schema_name": "MAIN", "table_name": "ORDER_LINES"},
            {"entity_name": "Customer", "schema_name": "MAIN", "table_name": "CUSTOMERS"},
            {"entity_name": "Store", "schema_name": "MAIN", "table_name": "STORES"},
            {"entity_name": "Ghost", "schema_name": "MAIN", "table_name": "NO_SUCH_TABLE"}]
METRICS = [
    {"id": 1, "name": "Revenue", "synonyms": "sales, turnover", "result_format": "currency", "is_active": 1,
     "metric_status": "published", "base_table": "MAIN.ORDER_LINES",
     "metric_builder_config": json.dumps({"enabled": True, "mode": "aggregate", "aggregation": "SUM",
                                          "measure": "NET_AMOUNT", "filters": []})},
    {"id": 2, "name": "Discounted sales", "synonyms": "", "result_format": "currency", "is_active": 1,
     "metric_status": "validated", "base_table": "MAIN.ORDER_LINES",
     "metric_builder_config": json.dumps({"enabled": True, "mode": "aggregate", "aggregation": "SUM",
                                          "measure": "NET_AMOUNT",
                                          "filters": [{"field": "DISCOUNT_AMOUNT", "operator": "greater_than",
                                                       "value": "0"}]})},
    {"id": 3, "name": "Discount rate", "synonyms": "", "result_format": "percentage", "is_active": 1,
     "metric_status": "published", "base_table": "MAIN.ORDER_LINES",
     "metric_builder_config": json.dumps({"enabled": True, "mode": "ratio",
                                          "numerator": {"aggregation": "SUM", "measure": "DISCOUNT_AMOUNT"},
                                          "denominator": {"aggregation": "SUM", "measure": "GROSS_AMOUNT"}})},
    {"id": 4, "name": "Hand written", "result_format": "number", "is_active": 1, "metric_status": "published",
     "base_table": "MAIN.ORDER_LINES", "sql_template": "SELECT SUM(x) FROM y", "metric_builder_config": ""},
    {"id": 6, "name": "Delivered sales", "synonyms": "", "result_format": "currency", "is_active": 1,
     "metric_status": "published", "base_table": "MAIN.ORDER_LINES",
     "metric_builder_config": json.dumps({"enabled": True, "mode": "aggregate", "aggregation": "SUM",
                                          "measure": "NET_AMOUNT",
                                          "filters": [{"field": "STATUS_CODE", "operator": "equals",
                                                       "value": "D"}]})},
    {"id": 5, "name": "Draft one", "result_format": "number", "is_active": 1, "metric_status": "draft",
     "base_table": "MAIN.ORDER_LINES",
     "metric_builder_config": json.dumps({"enabled": True, "mode": "aggregate", "aggregation": "SUM",
                                          "measure": "QUANTITY"})},
]


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _legacy(model, **kw) -> Legacy:
    links = [
        {"from_entity": "OrderLine", "to_entity": "Customer", "from_column": "CUSTOMER_ID",
         "to_column": "CUSTOMER_ID", "status": "confirmed", "is_active": 1},
        {"from_entity": "Customer", "to_entity": "Store", "from_column": "HOME_STORE_ID", "to_column": "STORE_ID",
         "status": "rejected", "is_active": 0},
        {"from_entity": "OrderLine", "to_entity": "Ghost", "from_column": "STORE_ID", "to_column": "ID",
         "status": "confirmed", "is_active": 1},
        {"from_entity": "OrderLine", "to_entity": "Store", "from_column": "STORE_ID", "to_column": "STORE_ID",
         "status": "suggested", "is_active": 1},
    ]
    ship = next(r for r in model.date_roles.values() if r.slug == "ship_date")
    delivered = next(r for r in model.date_roles.values() if r.slug == "delivered_date")
    base = dict(
        entities=ENTITIES, relationships=links, metrics=METRICS,
        date_contexts=[{"metric_id": 6, "is_active": 1, "is_default": 1, "fact_table": "MAIN.ORDER_LINES",
                        "fact_column": model.columns[delivered.column].name.upper()}],
        table_descriptions={"MAIN.ORDER_LINES": {"table_name": "MAIN.ORDER_LINES",
                                                 "description": "One row per order line, net of discounts.",
                                                 "column_synonym_map": {"NET_AMOUNT": ["net sales"]}}},
        properties=[{"entity_name": "Customer", "column_name": "SEGMENT", "role": "dimension",
                     "display_name": "Customer segment", "synonyms": "tier"},
                    {"entity_name": "Customer", "column_name": "CUSTOMER_NAME", "role": "dimension",
                     "display_name": "Client name", "synonyms": "client"},
                    {"entity_name": "OrderLine", "column_name": "LOADED_AT", "role": "ignore"}],
        # A column's reading lists the columns it was told apart from: they keep their own names.
        meanings=[{"scope": "column", "subject": "STATUS_CODE", "status": "confirmed",
                   "decided_reading": "Order line status", "decided_synonyms": ["line state"],
                   "found_in": ["ORDER_LINES.STATUS_CODE", "ORDER_LINES.CURRENCY_CODE"]},
                  {"scope": "table", "subject": "ORDER_LINES", "status": "confirmed",
                   "decided_reading": "Sales order lines", "found_in": ["ORDER_LINES"]},
                  {"scope": "code", "subject": "DLY", "status": "confirmed", "decided_reading": "daily",
                   "found_in": []},
                  {"scope": "column", "subject": "CURRENCY_CODE", "status": "suggested",
                   "reading": "Money", "found_in": ["ORDER_LINES.CURRENCY_CODE"]}],
        semantic_model={"date_roles": [{"fact_table": "MAIN.ORDER_LINES",
                                        "fact_column": model.columns[ship.column].name.upper(),
                                        "status": "approved", "is_default": True}]},
    )
    base.update(kw)
    return Legacy(**base)


def _apply(model, report):
    copy = model.model_copy(deep=True)
    notes = apply_overrides(copy, [{"object_key": d.object_key, "field": d.field, "value": d.value}
                                   for d in report.decisions])
    return copy, notes


def _by(report, field):
    return {d.object_key: d.value for d in report.decisions if d.field == field}


def test_links_come_over_confirmed_rejected_or_added_and_a_suggestion_is_left_to_the_data(retail):
    _, model = retail
    report = decisions(model, _legacy(model))
    trust = _by(report, "trust")
    to_customers = next(j for j in model.joins.values() if j.from_table.endswith("order_lines")
                        and j.to_table.endswith("customers"))
    assert trust[f"join:{to_customers.key}"] == "admin"
    home = [j for j in model.joins.values() if j.from_table.endswith("customers") and j.to_table.endswith("stores")]
    assert all(trust.get(f"join:{j.key}") == "rejected" for j in home)
    to_stores = next(j for j in model.joins.values() if j.from_table.endswith("order_lines")
                     and j.to_table.endswith("stores"))
    assert f"join:{to_stores.key}" not in trust                  # "suggested": the data's own evidence decides
    assert any("OrderLine -> Ghost" in m for m in report.missed)   # a table nobody learned: reported, not guessed


def _value(con, model, slug, start, end):
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": [slug],
                                           "time": {"window": {"kind": "between", "start": start, "end": end}}}),
                      model, Context(today=TODAY))
    return DuckDBWarehouse(con).query(compile_query(logical, model, "duckdb").sql).rows[0][0]


def _reference(con, what, date_column, where, start, end):
    return con.execute(f"SELECT {what} FROM order_lines o JOIN calendar c ON c.date_key = o.{date_column} "
                       f"WHERE {where} c.full_date >= DATE '{start}' AND c.full_date <= DATE '{end}'").fetchone()[0]


def test_a_metric_comes_over_as_a_measure_the_new_core_answers_with_the_warehouses_numbers(retail):
    con, model = retail
    report = decisions(model, _legacy(model))
    applied, notes = _apply(model, report)
    assert not notes, notes
    revenue = next(m for m in applied.measures.values() if m.slug == "net_amount")
    assert revenue.business_name == "Revenue"                    # matched, not duplicated: the admin's name
    assert {"sales", "turnover", "Net amount", "net sales"} <= set(revenue.synonyms["en"])
    assert sum(1 for m in applied.measures.values() if m.business_name == "Revenue") == 1

    # No date of its own: counted by the table's default, which today's setup made the ship date.
    discounted = next(m for m in applied.measures.values() if m.business_name == "Discounted sales")
    assert discounted.kind == "import" and discounted.default_date is None
    got = _value(con, applied, discounted.slug, "2026-01-01", "2026-01-31")
    want = _reference(con, "SUM(o.net_amount)", "ship_date_key", "o.discount_amount > 0 AND", "2026-01-01",
                      "2026-01-31")
    by_order_date = _reference(con, "SUM(o.net_amount)", "order_date_key", "o.discount_amount > 0 AND",
                               "2026-01-01", "2026-01-31")
    assert float(got) == pytest.approx(float(want)) and float(want) != pytest.approx(float(by_order_date))

    # Bound to its own date today (the delivered date): that binding stands.
    delivered = next(m for m in applied.measures.values() if m.business_name == "Delivered sales")
    assert applied.date_roles[delivered.default_date].slug == "delivered_date"
    got = _value(con, applied, delivered.slug, "2026-01-01", "2026-01-31")
    want = _reference(con, "SUM(o.net_amount)", "delivered_date_key", "o.status_code = 'D' AND", "2026-01-01",
                      "2026-01-31")
    by_ship_date = _reference(con, "SUM(o.net_amount)", "ship_date_key", "o.status_code = 'D' AND",
                              "2026-01-01", "2026-01-31")
    assert float(got) == pytest.approx(float(want)) and float(want) != pytest.approx(float(by_ship_date))

    rate = next(m for m in applied.measures.values() if m.business_name == "Discount rate")
    assert rate.format == "percent" and rate.additivity == "non_additive"
    got = _value(con, applied, rate.slug, "2025-01-01", "2025-12-31")
    want = _reference(con, "100.0 * SUM(o.discount_amount) / SUM(o.gross_amount)", "ship_date_key", "",
                      "2025-01-01", "2025-12-31")
    assert float(got) == pytest.approx(float(want))


def test_a_metric_written_as_a_query_or_still_a_draft_does_not_come_over(retail):
    _, model = retail
    report = decisions(model, _legacy(model))
    assert any("Hand written" in m and "it uses SELECT" in m for m in report.missed)   # the reason, in words
    said = json.dumps([d.value for d in report.decisions])      # neither added nor naming a measure
    assert "Draft one" not in said and "Hand written" not in said


def test_names_descriptions_meanings_and_the_default_date_come_over(retail):
    _, model = retail
    report = decisions(model, _legacy(model))
    applied, _ = _apply(model, report)
    lines = next(t for t in applied.tables.values() if t.name == "order_lines")
    assert lines.description == "One row per order line, net of discounts."
    assert applied.attributes["customer.segment"].business_name == "Customer segment"
    assert "tier" in applied.attributes["customer.segment"].synonyms["en"]
    loaded = next(c for c in applied.columns.values() if c.table == lines.key and c.name == "loaded_at")
    assert loaded.hidden
    assert lines.business_name == "Sales order lines"

    def shown_as(name):
        column = next(c for c in applied.columns.values() if c.table == lines.key and c.name == name)
        return [a for a in applied.attributes.values() if a.column == column.key] or [column]

    assert [o.business_name for o in shown_as("status_code")] == ["Order line status"]
    assert "line state" in shown_as("status_code")[0].synonyms.get("en", [])
    assert "Order line status" not in [o.business_name for o in shown_as("currency_code")]
    assert "Money" not in [o.business_name for o in shown_as("currency_code")]   # only suggested
    assert any("codes inside names" in m for m in report.missed)
    customer = next(e for e in applied.entities.values() if e.table.endswith("customers"))
    assert customer.business_name == model.entities[customer.slug].business_name   # a column's name is not its
    assert "client" in customer.synonyms["en"]                                        # entity's; its words are
    ship = next(r for r in applied.date_roles.values() if r.slug == "ship_date")
    assert ship.is_default and lines.default_date == ship.key


def test_an_admins_own_decision_outranks_an_import_and_a_dropped_one_disappears(retail, tmp_path, monkeypatch):
    import store
    import store.crypto

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.setattr(store.crypto, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    _, model = retail
    report = decisions(model, _legacy(model))
    revenue = next(m for m in model.measures.values() if m.slug == "net_amount")
    store.set_core2_override("acct", 1, f"measure:{revenue.key}", "business_name", "Net sales (finance)",
                             author="admin")
    write("acct", 1, report)
    kept = {(o["object_key"], o["field"]): o for o in store.list_core2_overrides("acct", 1)}
    assert kept[(f"measure:{revenue.key}", "business_name")]["value"] == "Net sales (finance)"
    assert kept[(f"measure:{revenue.key}", "business_name")]["author"] == "admin"
    assert any(o["author"] == AUTHOR for o in kept.values())
    # Today's setup drops a metric: the next import removes it.
    write("acct", 1, decisions(model, _legacy(model, metrics=[m for m in METRICS if m["id"] != 2])))
    names = [o["value"].get("business_name") for o in store.list_core2_overrides("acct", 1)
             if o["field"] == "define" and isinstance(o["value"], dict)]
    assert "Discounted sales" not in names and {"Discount rate", "Delivered sales"} <= set(names)


# ── from today's stores, through a build, to the page and the answers ──────

from tests.test_core2_learned_page_shows_what_was_learned import (  # noqa: E402
    ACCOUNT, _page, _post, _request, _schema_file, workspace)

__all__ = ["workspace"]


def _reject(rel_id: int) -> None:
    """The way today's admin rejects a link (the graph page's button)."""
    import asyncio
    from unittest.mock import patch

    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True):
        asyncio.run(routes.graph_reject_rel(_request("/x"), ACCOUNT, rel_id))


def _seed_todays_setup(store, tmp_path) -> None:
    for entity, table in (("OrderLine", "order_lines"), ("Customer", "customers"), ("Store", "stores")):
        store.save_entity(ACCOUNT, entity, table, schema_name="main")
    store.save_relationship(ACCOUNT, "OrderLine", "Customer", "customer_id", "customer_id", status="confirmed")
    _reject(store.save_relationship(ACCOUNT, "Customer", "Store", "home_store_id", "store_id", status="suggested"))
    store.save_metric(ACCOUNT, {
        "name": "Delivered sales", "synonyms": "delivered revenue", "result_format": "currency",
        "base_table": "main.order_lines",
        "sql_template": "SELECT SUM(net_amount) FROM main.order_lines WHERE status_code = 'D'",
        "metric_builder_config": json.dumps({"enabled": True, "mode": "aggregate", "aggregation": "SUM",
                                             "measure": "net_amount", "filters": [
                                                 {"field": "status_code", "operator": "equals", "value": "D"}]})})
    store.save_table_description(ACCOUNT, "main.order_lines", description="One row per order line.",
                                 column_synonyms="NET_AMOUNT = turnover")
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "_semantic_model.json").write_text(json.dumps({"date_roles": [
        {"fact_table": "main.order_lines", "fact_column": "SHIP_DATE_KEY", "status": "approved", "is_default": True},
        {"fact_table": "main.order_lines", "fact_column": "ORDER_DATE_KEY", "status": "approved",
         "is_default": False}]}))
    state = json.loads(store.get_client(ACCOUNT)["state_data"])
    store.update_client_state(ACCOUNT, "READY", {**state, "kb_dir": str(kb)})


def test_a_build_brings_todays_setup_over_and_answers_by_it(workspace, tmp_path):
    store, built, schema_dir = workspace
    _schema_file(built, schema_dir)
    _seed_todays_setup(store, tmp_path)
    from admin import core2_routes
    from core2.bootstrap.service import load_model

    core2_routes._run_build(ACCOUNT)
    db_id = store.get_client(ACCOUNT)["db_config_id"]
    model = load_model(ACCOUNT, db_id)
    lines = next(t for t in model.tables.values() if t.name == "order_lines")
    assert model.date_roles[lines.default_date].slug == "ship_date"
    assert lines.description == "One row per order line."
    assert "turnover" in next(m for m in model.measures.values() if m.slug == "net_amount").synonyms["en"]
    home = [j for j in model.joins.values() if j.from_table.endswith("customers") and j.to_table.endswith("stores")]
    assert home and all(j.trust == "rejected" for j in home)
    delivered = next(m for m in model.measures.values() if m.business_name == "Delivered sales")
    got = _value(built.con, model, delivered.slug, "2026-01-01", "2026-01-31")
    want = _reference(built.con, "SUM(o.net_amount)", "ship_date_key", "o.status_code = 'D' AND", "2026-01-01",
                      "2026-01-31")
    assert float(got) == pytest.approx(float(want))

    page = _page()
    assert "Brought over from today's setup" in page and "the metric Delivered sales from today's setup" in page
    assert "metrics added" in page and "links rejected" in page and "default dates" in page

    # Today's setup changes (the metric is deleted); "Bring them over again" follows it.
    with __import__("store.db", fromlist=["get_db"]).get_db() as conn:
        conn.execute("UPDATE metric_registry SET is_active=0, metric_status='deprecated' WHERE account_id=?",
                     (ACCOUNT,))
    response = _post(core2_routes.learned_import, f"/admin/clients/{ACCOUNT}/learned/import", {})
    assert response.status_code == 303 and "saved=imported" in response.headers["location"]
    assert not any(m.business_name == "Delivered sales" for m in load_model(ACCOUNT, db_id).measures.values())
