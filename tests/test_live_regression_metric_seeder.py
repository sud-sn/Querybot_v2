import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "fixtures"
    / "azure_sql_live_regression"
    / "04_seed_querybot_metrics.py"
)


def _load_seeder():
    spec = importlib.util.spec_from_file_location("live_regression_metric_seeder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_metric_seeder_is_idempotent_and_binds_invoice_date(monkeypatch):
    seeder = _load_seeder()
    calls = {"saved": [], "updated": [], "contexts": []}
    monkeypatch.setattr(
        seeder.store,
        "list_metrics",
        lambda account_id, active_only=False: [{"id": 41, "name": "Revenue"}],
    )
    monkeypatch.setattr(
        seeder.store,
        "update_metric",
        lambda metric_id, definition, **kwargs: calls["updated"].append(
            (metric_id, definition, kwargs)
        ),
    )

    next_ids = iter((42, 43))

    def save_metric(account_id, definition, **kwargs):
        metric_id = next(next_ids)
        calls["saved"].append((account_id, definition, kwargs, metric_id))
        return metric_id

    monkeypatch.setattr(seeder.store, "save_metric", save_metric)
    monkeypatch.setattr(
        seeder.store,
        "save_metric_date_context",
        lambda account_id, binding: calls["contexts"].append((account_id, binding)) or 1,
    )
    monkeypatch.setattr(
        seeder.store,
        "get_metric",
        lambda metric_id: {"id": metric_id, "metric_status": "validated"},
    )

    results = seeder.seed("Test_Az")

    assert [name for name, _, _ in results] == [
        "Revenue", "Gross Sales", "Discount Amount",
    ]
    assert len(calls["updated"]) == 1
    assert calls["updated"][0][0] == 41
    assert len(calls["saved"]) == 2
    assert len(calls["contexts"]) == 3
    for account_id, binding in calls["contexts"]:
        assert account_id == "Test_Az"
        assert binding["fact_table"] == "QBOT_LIVE_TEST.F_SALES_INVOICE"
        assert binding["fact_column"] == "INVOICE_DATE_SK"
        assert binding["dimension_table"] == "QBOT_LIVE_TEST.D_DATE"
        assert binding["dimension_key"] == "DATE_SK"
        assert binding["date_value_column"] == "FULL_DATE"
        assert binding["is_default"] is True
