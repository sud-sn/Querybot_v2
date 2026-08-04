from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway.web_adapter import WebAdapter
from gateway.webhooks import (
    _DASHBOARD_ADD_INTENT_RE,
    _DASHBOARD_ADD_QUERY_RE,
    _DASHBOARD_CHART_TYPE_RE,
    _DASHBOARD_CREATE_INTENT_RE,
    _DASHBOARD_FILTER_INTENT_RE,
    _DASHBOARD_PUBLISH_INTENT_RE,
    _DASHBOARD_RENAME_INTENT_RE,
    _DASHBOARD_ROLLBACK_INTENT_RE,
    _DASHBOARD_SCHEDULE_INTENT_RE,
    _DASHBOARD_SHARE_INTENT_RE,
    _DASHBOARD_TAB_INTENT_RE,
    _dashboard_request_tail,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text",
    [
        "create a dashboard from this result",
        "build me a dashboard showing monthly revenue",
        "make my dashboard with top products",
    ],
)
def test_dashboard_creation_intent(text):
    assert _DASHBOARD_CREATE_INTENT_RE.search(text)


def test_dashboard_refinement_intents_are_explicit():
    assert _DASHBOARD_ADD_INTENT_RE.search("add this to my dashboard")
    assert _DASHBOARD_RENAME_INTENT_RE.search("rename the dashboard to Executive view")
    assert _DASHBOARD_PUBLISH_INTENT_RE.search("publish this dashboard")
    add_query = _DASHBOARD_ADD_QUERY_RE.search(
        "add a chart showing weekly orders by status to my dashboard"
    )
    assert add_query and add_query.group("question") == "weekly orders by status"
    assert _DASHBOARD_CHART_TYPE_RE.search("change the dashboard chart to line")
    assert not _DASHBOARD_CREATE_INTENT_RE.search("my dashboard shows a different total")


def test_dashboard_question_tail_is_routed_to_governed_query():
    assert _dashboard_request_tail(
        "create a dashboard showing monthly revenue by region"
    ) == "monthly revenue by region"
    assert _dashboard_request_tail("create a dashboard from this result") == ""


def test_dashboard_work_controls_are_parsed_without_trailing_scope_words():
    filter_match = _DASHBOARD_FILTER_INTENT_RE.search(
        "add a filter for region to my dashboard"
    )
    tab_match = _DASHBOARD_TAB_INTENT_RE.search(
        "add a tab called Products to my dashboard"
    )
    assert filter_match and filter_match.group("field") == "region"
    assert tab_match and tab_match.group("tab") == "Products"
    assert _DASHBOARD_SCHEDULE_INTENT_RE.search("refresh this dashboard daily")
    assert _DASHBOARD_SHARE_INTENT_RE.search("share this dashboard with the team")
    assert _DASHBOARD_ROLLBACK_INTENT_RE.search("restore dashboard to version 2")


def _adapter() -> WebAdapter:
    adapter = WebAdapter(AsyncMock(), "tenant-a", "7", "thread-1")
    adapter.last_result = {
        "rows": [{"month": "2026-01", "revenue": 1250}],
        "question": "Monthly revenue",
        "sql": "SELECT month, SUM(revenue) AS revenue FROM sales GROUP BY month",
        "db_cfg": {"id": 11, "credentials": {"secret": "not-persisted"}},
    }
    return adapter


def test_materialize_dashboard_reuses_governed_sql_without_rows():
    adapter = _adapter()
    response = {
        "type": "assistant_response",
        "answer": {"headline": "Monthly revenue"},
        "chart": {"chart_type": "line", "title": "Revenue trend"},
    }
    created = {
        "id": 31, "name": "Revenue dashboard", "status": "draft", "version": 1
    }
    updated = {**created, "version": 2}
    with (
        patch("store.dashboard_store.create_dashboard", return_value=created),
        patch("store.dashboard_store.create_data_source", return_value={"id": 52}),
        patch("store.dashboard_store.add_chart", return_value=True),
        patch("store.dashboard_store.get_dashboard", return_value=updated),
        patch("store.user_store.pin_chart", return_value=44) as pin,
    ):
        artifact = adapter.materialize_dashboard(response, name="Revenue dashboard")

    assert artifact["id"] == 31
    assert artifact["chart_id"] == 44
    assert artifact["data_source_id"] == 52
    assert artifact["item_type"] == "line"
    kwargs = pin.call_args.kwargs
    assert kwargs["sql_query"].startswith("SELECT month")
    assert "rows" not in kwargs
    assert "credentials" not in kwargs
    assert kwargs["dashboard_id"] == 31
    assert "rows" not in kwargs["display_config"]


def test_materialize_dashboard_uses_authenticated_portal_identity():
    adapter = WebAdapter(
        AsyncMock(), "tenant-a", "web_7", "thread-1", portal_user_id=7
    )
    adapter.last_result = {
        "rows": [{"revenue": 1250}],
        "question": "Revenue",
        "sql": "SELECT SUM(revenue) AS revenue FROM sales",
        "db_cfg": {"id": 11},
    }
    created = {"id": 31, "name": "Revenue", "status": "draft", "version": 1}
    with (
        patch("store.dashboard_store.create_dashboard", return_value=created) as create,
        patch("store.dashboard_store.create_data_source", return_value={"id": 52}),
        patch("store.dashboard_store.add_chart", return_value=True),
        patch("store.dashboard_store.get_dashboard", return_value=created),
        patch("store.user_store.pin_chart", return_value=44) as pin,
    ):
        adapter.materialize_dashboard(name="Revenue")

    assert create.call_args.args[1] == 7
    assert pin.call_args.kwargs["user_id"] == 7


def test_dashboard_rejects_synthetic_identity_without_authenticated_owner():
    adapter = WebAdapter(AsyncMock(), "tenant-a", "web_7", "thread-1")
    adapter.last_result = {
        "rows": [{"revenue": 1250}],
        "question": "Revenue",
        "sql": "SELECT SUM(revenue) AS revenue FROM sales",
        "db_cfg": {"id": 11},
    }
    with pytest.raises(ValueError, match="Authenticated portal user identity"):
        adapter.materialize_dashboard(name="Revenue")


def test_materialize_requires_a_real_executed_result():
    adapter = _adapter()
    adapter.last_result = None
    with pytest.raises(ValueError, match="Run the business question first"):
        adapter.materialize_dashboard({"type": "assistant_response"})


def test_portal_has_ana_style_split_workspace_contract():
    chat = (ROOT / "portal/templates/portal_chat.html").read_text(encoding="utf-8")
    assert 'id="artifactPane"' in chat
    assert "renderArtifactPreview(msg)" in chat
    assert "data-open-artifact" in chat
    assert "Add to dashboard" in chat
    assert "dashboardPickerBackdrop" in chat
    assert "/portal/api/dashboards" in chat
    assert "assistant_dashboard" in chat
    assert "DASHBOARD_ID" in chat
    assert "citation-strip" in chat
    assert "@media(max-width:820px)" in chat

    dashboard = (ROOT / "portal/templates/portal_dashboard.html").read_text(encoding="utf-8")
    assert "Revision history" in dashboard
    assert "Data sources" in dashboard
    assert "sortDashboardTable" in dashboard


def test_dashboard_store_is_owner_and_tenant_scoped():
    source = (ROOT / "store/dashboard_store.py").read_text(encoding="utf-8")
    assert "WHERE id=? AND user_id=? AND account_id=?" in source
    assert "WHERE dashboard_id=? AND user_id=?" in source
    assert "result rows" in source.lower()
