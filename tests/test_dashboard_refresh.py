from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import core.dashboard_refresh as refresh


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
