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


@pytest.mark.parametrize(
    "text",
    [
        "add this to my dashboard",
        "add this chart to the dashboard",
        "put that visual on this dashboard",
        "save the result in my dashboard",
        "place this KPI on the dashboard",
    ],
)
def test_dashboard_add_intent_accepts_natural_result_references(text):
    assert _DASHBOARD_ADD_INTENT_RE.search(text)


def test_dashboard_refinement_intents_are_explicit():
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

    # Rendered, not grepped: these labels live in the message catalogue now,
    # and the catalogue is injected into the page as JSON -- so a source grep
    # would pass whether or not the drawers render.
    import sys
    sys.path.insert(0, str(ROOT / "tests"))
    from dashboard_render import CHART, render, visible

    dashboard_markup = visible(render(charts=[CHART]))
    assert "Revision history" in dashboard_markup
    assert "Data sources" in dashboard_markup
    dashboard = (ROOT / "portal/templates/portal_dashboard.html").read_text(encoding="utf-8")
    assert "sortDashboardTable" in dashboard


# ── The tenant boundary, executed ────────────────────────────────────────────
#
# This was three greps over store/dashboard_store.py. None of them could detect
# the loss of the scope they were named for:
#
#   assert "result rows" in source.lower()
#       matches the module docstring. Behaviourally it can never fail.
#   assert "WHERE id=? AND user_id=? AND account_id=?" in source
#   assert "WHERE dashboard_id=? AND user_id=?" in source
#       those fragments occur 20 and 11 times. Any ONE query can drop
#       `account_id` and the assertion still matches the other 19.
#
# Proven, not assumed: deleting `AND account_id=?` from get_dashboard_for_view
# -- the only tenant boundary behind GET /portal/dashboard?dashboard_id=N --
# left all 75 tests in this file and test_dashboard_page.py passing.
#
# So: build two real tenants and ask each reader for the other's data.


def _two_tenants():
    """Two clients, two users, one published team dashboard owned by alpha.

    Uses the convention already in tests/test_dashboard_page.py -- a temp DB
    via setdefault plus random account ids -- rather than reloading `store`,
    which rebinds sys.modules and breaks unrelated tests in the same run.
    """
    import os
    import tempfile

    os.environ.setdefault("QUERYBOT_DB_PATH",
                          os.path.join(tempfile.mkdtemp(), "dash.db"))
    import store
    store.init_db()

    alpha = f"acct{os.urandom(4).hex()}"
    beta = f"acct{os.urandom(4).hex()}"
    store.upsert_client(alpha, "T")
    store.upsert_client(beta, "T")
    alpha_user, _ = store.create_user(alpha, "Ann", f"{os.urandom(4).hex()}@x.com")
    beta_user, _ = store.create_user(beta, "Bob", f"{os.urandom(4).hex()}@x.com")

    board = store.create_dashboard(alpha, alpha_user, "th", "Alpha board",
                                   visibility="team")
    store.publish_dashboard(int(board["id"]), alpha_user, alpha)
    config_id = store.save_db_config(
        "azure_sql", "cfg", {"server": "s", "user": "u", "password": "p"})
    source = store.create_data_source(
        int(board["id"]), alpha_user, alpha, name="src", question="q",
        sql_query="SELECT 1", db_config_id=config_id)
    return store, alpha, alpha_user, beta, beta_user, int(board["id"]), int(source["id"])


def _cross_tenant_readers(store, board_id, source_id, alpha_user, beta):
    """Every reader that takes an account_id, asked for alpha's data as beta.

    Alpha's OWN user id is passed with beta's account id deliberately: the
    tenant scope has to hold even when the caller knows the right user, which
    is the case a stolen or guessed id produces.
    """
    return {
        "get_dashboard":
            lambda: store.get_dashboard(board_id, alpha_user, beta),
        "get_dashboard_for_view":
            lambda: store.get_dashboard_for_view(board_id, alpha_user, beta),
        "latest_dashboard_for_thread":
            lambda: store.latest_dashboard_for_thread(beta, alpha_user, "th"),
        "list_dashboards":
            lambda: store.list_dashboards(beta, alpha_user),
        "list_editable_dashboards":
            lambda: store.list_editable_dashboards(beta, alpha_user),
        "list_data_sources":
            lambda: store.list_data_sources(board_id, alpha_user, beta),
        "list_data_sources_for_view":
            lambda: store.list_data_sources_for_view(board_id, alpha_user, beta),
        "get_data_source":
            lambda: store.get_data_source(source_id, alpha_user, beta),
        "list_dashboard_versions":
            lambda: store.list_dashboard_versions(board_id, alpha_user, beta),
        "publish_dashboard":
            lambda: store.publish_dashboard(board_id, alpha_user, beta),
        "rename_dashboard":
            lambda: store.rename_dashboard(board_id, alpha_user, beta, "pwned"),
        "mark_dashboard_draft":
            lambda: store.mark_dashboard_draft(board_id, alpha_user, beta),
    }


def test_no_dashboard_reader_crosses_the_tenant_boundary():
    store, alpha, alpha_user, beta, beta_user, board_id, source_id = _two_tenants()

    leaked = []
    for name, call in _cross_tenant_readers(
            store, board_id, source_id, alpha_user, beta).items():
        got = call()
        if got:
            leaked.append(f"{name} returned {got!r}")
    assert not leaked, (
        "these read across the tenant boundary:\n  " + "\n  ".join(leaked))


def test_and_the_owning_tenant_can_still_read_all_of_it():
    """The control.

    Without it, every assertion above is satisfied by a store that returns
    nothing to anybody -- which is exactly what a botched scoping fix looks
    like, and it would ship as a green suite.
    """
    store, alpha, alpha_user, beta, beta_user, board_id, source_id = _two_tenants()

    assert store.get_dashboard(board_id, alpha_user, alpha)
    assert store.get_dashboard_for_view(board_id, alpha_user, alpha)
    assert store.latest_dashboard_for_thread(alpha, alpha_user, "th")
    assert store.list_dashboards(alpha, alpha_user)
    assert store.list_editable_dashboards(alpha, alpha_user)
    assert store.list_data_sources(board_id, alpha_user, alpha)
    assert store.list_data_sources_for_view(board_id, alpha_user, alpha)
    assert store.get_data_source(source_id, alpha_user, alpha)
    assert store.list_dashboard_versions(board_id, alpha_user, alpha)


def test_a_published_team_dashboard_is_shared_inside_its_tenant_only():
    """get_dashboard_for_view is the one reader that deliberately serves a
    NON-owner -- that is what publishing to a team means -- so it is the one
    whose account_id clause is doing all the work. A second user inside alpha
    sees the board; beta's user does not, with the same arguments otherwise.
    """
    import os

    store, alpha, alpha_user, beta, beta_user, board_id, _ = _two_tenants()
    colleague, _ = store.create_user(alpha, "Cy", f"{os.urandom(4).hex()}@x.com")

    seen = store.get_dashboard_for_view(board_id, colleague, alpha)
    assert seen, "a published team dashboard must be visible to the team"
    assert seen["can_edit"] == 0, "a viewer is not an editor"

    assert store.get_dashboard_for_view(board_id, beta_user, beta) is None
    assert store.get_dashboard_for_view(board_id, colleague, beta) is None
