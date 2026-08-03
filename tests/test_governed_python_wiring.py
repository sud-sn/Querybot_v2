from pathlib import Path
from unittest.mock import patch

from admin.routes import _client_health_score
from gateway.webhooks import _ANALYSIS_WORK_INTENT_RE, _CUSTOM_PYTHON_INTENT_RE


ROOT = Path(__file__).resolve().parents[1]


def test_custom_python_intents_do_not_fall_through_to_sql_generation():
    assert _CUSTOM_PYTHON_INTENT_RE.search("use Python to calculate a 3 month rolling average")
    assert _CUSTOM_PYTHON_INTENT_RE.search("run this python: result = []")
    assert _CUSTOM_PYTHON_INTENT_RE.search("```python\nresult = []\n```")

    webhooks = (ROOT / "gateway/webhooks.py").read_text("utf-8")
    dispatch = webhooks[webhooks.index("if _ANALYSIS_WORK_INTENT_RE.search"):]
    assert "_CUSTOM_PYTHON_INTENT_RE.search(text)" in dispatch[:300]
    assert "_run_analysis_work(text)" in dispatch[:500]


def test_natural_latest_successful_result_phrasing_routes_to_analysis():
    assert _ANALYSIS_WORK_INTENT_RE.search(
        "Analyze the latest successful result deeply for outliers and profile the numeric distribution."
    )
    assert _ANALYSIS_WORK_INTENT_RE.search("profile my previous result")


def test_admin_and_portal_governed_python_proof_is_wired():
    admin = (ROOT / "admin/templates/client_detail.html").read_text("utf-8")
    route = (ROOT / "admin/routes.py").read_text("utf-8")
    portal = (ROOT / "portal/templates/portal_chat.html").read_text("utf-8")
    schema = (ROOT / "store/db.py").read_text("utf-8")

    assert "Enable governed Python analysis" in admin
    assert "Allow users to submit governed Python source" in admin
    assert "Recent analysis tasks" in admin
    assert "SQL quality gate" in admin
    assert "enable_python_analysis" in route
    assert "allow_user_python" in route
    assert "sql_accuracy_target_pct" in schema
    assert "agent-analysis" in admin
    assert "eval_rate >= eval_target" in route
    assert "No database query" in portal


def test_admin_readiness_requires_latest_golden_suite_to_meet_target():
    client = {
        "account_id": "quality-tenant",
        "state": "READY",
        "state_data": '{"kb_tables":["dbo.Sales"],"masking_config":{}}',
        "db_config_id": 1,
        "sql_accuracy_target_pct": 85,
    }
    with (
        patch("admin.routes.store.get_db_config", return_value={"id": 1}),
        patch("admin.routes.store.list_users", return_value=[{"id": 1}]),
        patch("admin.routes.store.count_semantic_field_feedback", return_value=0),
        patch("admin.routes.store.get_query_stats", return_value={"total": 10, "succeeded": 10}),
        patch("admin.routes.store.latest_eval_run", return_value={"total_cases": 20, "passed_cases": 16}),
    ):
        below = _client_health_score("quality-tenant", client)
    eval_component = next(item for item in below["components"] if item["key"] == "evals")
    assert eval_component["ok"] is False
    assert "80.0%" in eval_component["hint"]

    with (
        patch("admin.routes.store.get_db_config", return_value={"id": 1}),
        patch("admin.routes.store.list_users", return_value=[{"id": 1}]),
        patch("admin.routes.store.count_semantic_field_feedback", return_value=0),
        patch("admin.routes.store.get_query_stats", return_value={"total": 10, "succeeded": 10}),
        patch("admin.routes.store.latest_eval_run", return_value={"total_cases": 20, "passed_cases": 17}),
    ):
        at_target = _client_health_score("quality-tenant", client)
    eval_component = next(item for item in at_target["components"] if item["key"] == "evals")
    assert eval_component["ok"] is True
