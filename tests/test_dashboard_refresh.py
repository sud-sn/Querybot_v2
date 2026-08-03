from datetime import date
from decimal import Decimal
from types import SimpleNamespace
import uuid

import core.dashboard_refresh as refresh
import store


SOURCE = {
    "id": 7,
    "dashboard_id": 3,
    "account_id": "tenant-a",
    "user_id": 11,
    "db_config_id": 5,
    "sql_query": "SELECT region, SUM(revenue) AS revenue FROM sales GROUP BY region",
    "semantic_contract_version": "contract-2",
}
OWNER = {"id": 11, "account_id": "tenant-a", "is_active": 1}
VIEWER = {"id": 12, "account_id": "tenant-a", "is_active": 1}


def _common_mocks(monkeypatch):
    monkeypatch.setattr(refresh.store, "get_compliance_profile", lambda account: {"active_policy_version": 4})
    monkeypatch.setattr(refresh.store, "get_semantic_compiler_state", lambda account: {"active_contract_version": "contract-2"})
    monkeypatch.setattr(refresh.store, "get_db_config", lambda db_id: {"db_type": "azure_sql", "credentials": {}})
    monkeypatch.setattr(refresh.store, "get_client_state", lambda account: {"schema_dir": ""})
    monkeypatch.setattr(refresh.store, "get_allowed_tables", lambda user: {"sales"})
    monkeypatch.setattr("core.schema.load_known_tables", lambda path: {"sales": {}})
    monkeypatch.setattr("core.schema.load_schema_columns", lambda path: {"sales": ["region", "revenue"]})
    monkeypatch.setattr("core.compliance.policy_engine.resolve_context", lambda *args, **kwargs: object())


def test_owner_can_use_matching_encrypted_cache(monkeypatch):
    _common_mocks(monkeypatch)
    monkeypatch.setattr(refresh.store, "get_source_cache", lambda *args: {
        "rows": [{"region": "North", "revenue": 50}],
        "policy_version": 4,
        "contract_version": "contract-2",
        "refreshed_at": "2026-08-03 10:00:00",
    })
    result = refresh.execute_dashboard_source(SOURCE, OWNER)
    assert result.from_cache is True
    assert result.rows[0]["revenue"] == 50


def test_team_viewer_executes_live_under_their_acl_and_filters(monkeypatch):
    _common_mocks(monkeypatch)
    calls = {}

    def governed(*args, **kwargs):
        calls["sql"] = args[2]
        calls["allowed_tables"] = kwargs["allowed_tables"]
        return SimpleNamespace(
            rows=[{"region": "North", "revenue": 50}],
            sql=args[2],
            decision=SimpleNamespace(cache_ttl_seconds=600),
        )

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
    monkeypatch.setattr(refresh.store, "save_source_cache", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("viewer result must not be cached as owner")))
    result = refresh.execute_dashboard_source(
        SOURCE,
        VIEWER,
        filters=[{"field": "region", "operator": "equals"}],
        filter_values={"region": "North"},
    )
    assert result.from_cache is False
    assert result.applied_filters == ("region",)
    assert "qb_dashboard_source" in calls["sql"]
    assert calls["allowed_tables"] == {"sales"}


def test_owner_decimal_and_date_rows_are_json_safe_before_cache(monkeypatch):
    _common_mocks(monkeypatch)
    monkeypatch.setattr(refresh.store, "get_source_cache", lambda *args: None)
    captured = {}

    def governed(*args, **kwargs):
        return SimpleNamespace(
            rows=[{"period": date(2026, 1, 1), "revenue": Decimal("52677.25")}],
            sql=args[2], decision=SimpleNamespace(cache_ttl_seconds=600),
        )

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
    monkeypatch.setattr(
        refresh.store, "save_source_cache",
        lambda source, rows, **kwargs: captured.setdefault("rows", rows),
    )
    result = refresh.execute_dashboard_source(SOURCE, OWNER)
    assert result.rows == [{"period": "2026-01-01", "revenue": 52677.25}]
    assert captured["rows"] == result.rows


def test_dashboard_runs_real_policy_engine_and_masks_before_release(monkeypatch):
    """Only the external DB call is stubbed; policy, lineage and masking are real."""
    store.init_db()
    account_id = f"dashboard-policy-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account_id, "portal")
    store.save_compliance_profile(
        account_id,
        mode="regulated",
        industry="healthcare_pharmacy",
        policy_pack_key="healthcare_pharmacy_v1",
        enforcement_mode="enforce",
        active_policy_version=1,
    )
    store.save_classification(
        account_id, "DBO.CUSTOMERS", "CUSTOMER_NAME",
        sensitivity="RESTRICTED", identifiability="DIRECT",
        tags=["PII"], mask_strategy="redact", reviewed=True,
    )
    store.replace_policy_rules(account_id, 1, [
        {
            "name": f"Mask PII on {action}", "subject_type": "role",
            "subject_id": "analyst", "resource_type": "classification",
            "resource_pattern": "PII", "action": action, "effect": "mask",
            "mask_strategy": "redact", "cache_ttl_seconds": 0,
        }
        for action in ("query_execution", "result_release")
    ])
    store.replace_purposes(account_id, [{
        "purpose_key": "analytics", "name": "Analytics",
        "default_for_roles": ["analyst"],
        "permissions": {"PII": ["query_execution", "result_release"]},
    }])
    source = {
        **SOURCE, "account_id": account_id,
        "sql_query": "SELECT CUSTOMER_NAME FROM DBO.CUSTOMERS",
    }
    viewer = {"id": 424242, "account_id": account_id, "role": "analyst", "is_active": 1}
    monkeypatch.setattr(refresh.store, "get_db_config", lambda _id: {"db_type": "azure_sql", "credentials": {}})
    monkeypatch.setattr(refresh.store, "get_client_state", lambda _account: {"schema_dir": "test-schema"})
    monkeypatch.setattr(refresh.store, "get_allowed_tables", lambda _viewer: {"DBO.CUSTOMERS"})
    monkeypatch.setattr("core.schema.load_known_tables", lambda _path: {"DBO.CUSTOMERS"})
    monkeypatch.setattr("core.schema.load_schema_columns", lambda _path: {"DBO.CUSTOMERS": {"CUSTOMER_NAME": "varchar"}})
    monkeypatch.setattr(
        "core.compliance.governed_query.run_query",
        lambda *_args, **_kwargs: [{"CUSTOMER_NAME": "Alice Smith"}],
    )
    try:
        result = refresh.execute_dashboard_source(source, viewer, allow_cache=False)
        assert result.rows[0]["CUSTOMER_NAME"] != "Alice Smith"
        assert "REDACT" in result.rows[0]["CUSTOMER_NAME"].upper()
    finally:
        with store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id=?", (account_id,))
