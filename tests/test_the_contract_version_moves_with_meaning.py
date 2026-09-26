"""
The answer contract's version moves when meaning changes, and only then.

contract_version stamps every answer, gates the reuse of a planned SQL, and
tells a dashboard whether its cached rows still hold. It was an md5 of the
whole compiled body, so it moved on things that are not meaning: a metric's
usage counter (bumped by every answered question), a table dragged on the
graph canvas, a join's profiling timestamps, a Graph Chat proposal nobody had
accepted, and the model's drift record, rewritten on every KB build. It did
not move on meaning the contract never carried: a confirmed business meaning,
a table description or column term, a subject area, a vocabulary pack or the
client's vocabulary overlay.

Three readers of the published version read columns the compiler state has
never had (active_contract_version, published_version), so dashboards served
a cache from before any change in meaning and agent runs were stamped with
nothing; and a rollback moved the published pointer while answers went on
reading the contract file it never rewrote.

These compile real contracts from a scratch store and a synthetic model.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _table(key, *columns):
    return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
            "pk_columns": [key], "row_count": 10, "comment": "", "schema": "MART", "database": "WH"}


SCHEMA = {
    "WH.MART.SALES_FACT": _table("SALES_KEY", ("SALES_KEY", "bigint"), ("CUSTOMER_SK", "int"),
                                 ("NET_AMT", "decimal(18,2)")),
    "WH.MART.CUSTOMER_DIM": _table("CUSTOMER_SK", ("CUSTOMER_SK", "int"), ("CUSTOMER_NAME", "varchar(80)")),
}


@pytest.fixture
def tenant(tmp_path, monkeypatch):
    import store
    from core import vocab_packs
    from core.semantic_model import write_semantic_model

    monkeypatch.setattr(vocab_packs, "_CLIENTS_DIR", tmp_path / "clients")
    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    kb = tempfile.mkdtemp()
    with open(os.path.join(kb, "_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(SCHEMA, handle)
    store.update_client_state(account_id, "READY", {"schema_dir": kb, "kb_dir": kb})
    write_semantic_model(schema_dir=kb, kb_dir=kb, account_id=account_id)
    store.save_metric(account_id, {"name": "Net Sales", "formula_type": "expression", "sql_template": "SUM(NET_AMT)",
                                   "base_table": "MART.SALES_FACT", "synonyms": "revenue"})
    store.save_entity(account_id=account_id, entity_name="Customer", table_name="CUSTOMER_DIM", schema_name="MART",
                      pos_x=100.0, pos_y=100.0)
    return account_id, kb


def _version(tenant):
    from core.semantic_contract import compile_contract

    account_id, kb = tenant
    return compile_contract(account_id, kb)["meta"]["contract_version"]


# ── What is not meaning ──────────────────────────────────────────────────────

class TestRecordKeepingDoesNotMoveTheVersion:

    def test_answering_questions(self, tenant):
        import store

        before = _version(tenant)
        store.increment_metric_usage(tenant[0], ["Net Sales"])
        assert _version(tenant) == before

    def test_dragging_a_table_on_the_graph_canvas(self, tenant):
        import store

        before = _version(tenant)
        store.save_entity(account_id=tenant[0], entity_name="Customer", table_name="CUSTOMER_DIM",
                          schema_name="MART", pos_x=640.0, pos_y=380.0, color="#7C3AED")
        assert _version(tenant) == before

    def test_a_graph_chat_proposal_nobody_has_accepted(self, tenant):
        import store

        before = _version(tenant)
        entity = store.get_entity(tenant[0], "Customer")
        store.create_graph_change_proposal(tenant[0], "set_entity_filter", "entity", "Customer",
                                           dict(entity), dict(entity, entity_filter="IS_TEST = 0"))
        assert _version(tenant) == before

    def test_a_later_kb_build_that_changed_nothing(self, tenant):
        # Every build stamps the model's drift record with the time it ran.
        from core.semantic_model import MODEL_JSON

        before = _version(tenant)
        path = os.path.join(tenant[1], MODEL_JSON)
        with open(path, encoding="utf-8") as handle:
            model = json.load(handle)
        model["_last_drift"]["recorded_at"] = "2030-01-01T00:00:00+00:00"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(model, handle)
        assert _version(tenant) == before


# ── What is meaning ──────────────────────────────────────────────────────────

class TestMeaningMovesTheVersion:

    def test_a_metric_s_formula(self, tenant):
        import store

        before = _version(tenant)
        metric = store.list_metrics(tenant[0])[0]
        store.update_metric(metric["id"], {"sql_template": "SUM(NET_AMT) - 0"})
        assert _version(tenant) != before

    def test_a_confirmed_business_meaning(self, tenant):
        import store

        store.save_business_meanings(tenant[0], [{"scope": "code", "subject": "WHS", "reading": "warehouse",
                                                   "synonyms": [], "rule": "values", "confidence": 90}])
        before = _version(tenant)
        store.decide_business_meaning(tenant[0], "code", "WHS", "confirmed", reading="warehouse")
        assert _version(tenant) != before

    def test_a_table_description(self, tenant):
        import store

        before = _version(tenant)
        store.save_table_description(tenant[0], "MART.SALES_FACT", description="One row per invoice line")
        assert _version(tenant) != before

    def test_a_subject_area(self, tenant):
        import store

        before = _version(tenant)
        store.save_domain(tenant[0], "Sales", tables=["MART.SALES_FACT"], synonyms="selling")
        assert _version(tenant) != before

    def test_the_vocabulary_packs_in_force(self, tenant):
        import store

        before = _version(tenant)
        store.update_client_meta(tenant[0], erp_packs=json.dumps(["wholesale_distribution"]))
        assert _version(tenant) != before

    def test_the_client_s_vocabulary_overlay(self, tenant):
        from core import vocab_packs

        before = _version(tenant)
        overlay = vocab_packs._CLIENTS_DIR / tenant[0] / "vocab.json"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text(json.dumps({"abbreviations": {"WHS": "warehouse"}}), encoding="utf-8")
        assert _version(tenant) != before


# ── Readers of the published version ─────────────────────────────────────────

def _publish(account_id, version):
    import store

    store.save_semantic_contract_version(account_id, {"meta": {"contract_version": version}}, status="active")
    store.publish_semantic_contract_version(account_id, version)


class TestReadersFollowThePublishedVersion:

    def test_a_dashboard_does_not_serve_rows_cached_before_a_change_in_meaning(self, tenant, monkeypatch):
        import core.dashboard_refresh as refresh

        account_id, _ = tenant
        _publish(account_id, "after-the-change")
        source = {"id": 7, "dashboard_id": 3, "account_id": account_id, "user_id": 11, "db_config_id": 5,
                  "sql_query": "SELECT 1 AS n", "semantic_contract_version": "before-the-change"}
        monkeypatch.setattr(refresh.store, "get_source_cache", lambda *a, **k: {
            "rows": [{"n": 1}], "policy_version": 0, "contract_version": "before-the-change", "refreshed_at": ""})
        monkeypatch.setattr(refresh.store, "save_source_cache", lambda *a, **k: None)
        monkeypatch.setattr(refresh.store, "get_db_config", lambda db_id: {"db_type": "azure_sql", "credentials": {}})
        monkeypatch.setattr(refresh.store, "get_allowed_tables", lambda user: None)
        monkeypatch.setattr("core.compliance.policy_engine.resolve_context", lambda *a, **k: object())
        monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", lambda *a, **k: SimpleNamespace(
            rows=[{"n": 2}], sql=a[2], decision=SimpleNamespace(cache_ttl_seconds=600)))
        result = refresh.execute_dashboard_source(source, {"id": 11, "account_id": account_id, "is_active": 1})
        assert (result.from_cache, result.rows) == (False, [{"n": 2}])

    def test_an_agent_run_is_stamped_with_the_published_version(self, tenant):
        import store
        from core.agent_runtime import AgentRunSession

        account_id, _ = tenant
        _publish(account_id, "published-v")
        user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com",
                                       password="a-password-they-chose", role="admin")
        session = AgentRunSession.start(account_id=account_id, portal_user={"id": user_id, "account_id": account_id,
                                                                           "role": "admin"},
                                        external_thread_id="thread-1", objective="net sales by customer")
        run = store.get_agent_run(account_id=account_id, portal_user_id=user_id, run_id=session.run_id)
        assert run["contract_version"] == "published-v"

    def test_a_rollback_changes_the_contract_answers_read(self, tenant):
        import asyncio

        import store
        import admin.routes as routes
        from core.semantic_contract import load_contract, write_contract

        account_id, kb = tenant
        first = write_contract(account_id, kb)["meta"]["contract_version"]
        store.save_table_description(account_id, "MART.SALES_FACT", description="One row per invoice line")
        second = write_contract(account_id, kb)["meta"]["contract_version"]
        assert second != first and load_contract(kb)["meta"]["contract_version"] == second
        with patch.object(routes, "_is_auth", return_value=True):
            asyncio.run(routes.model_health_rollback_contract_version(MagicMock(), account_id, first))
        assert load_contract(kb)["meta"]["contract_version"] == first
