import uuid

import pytest

import store


@pytest.fixture()
def dashboard_users():
    store.init_db()
    account_id = f"dashboard-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account_id, "portal")
    owner_id, _ = store.create_user(
        account_id, "Dashboard Owner", f"owner-{uuid.uuid4().hex[:8]}@test.com"
    )
    viewer_id, _ = store.create_user(
        account_id, "Dashboard Viewer", f"viewer-{uuid.uuid4().hex[:8]}@test.com"
    )
    try:
        yield account_id, owner_id, viewer_id
    finally:
        with store.get_db() as conn:
            conn.execute("DELETE FROM dashboard_subscription WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM pinned_chart WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM dashboard_data_source WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM dashboard_artifact_version WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM dashboard_artifact WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM portal_user WHERE account_id=?", (account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (account_id,))


def _chart(account_id: str, user_id: int, chart_type: str, title: str) -> int:
    return store.pin_chart(
        user_id=user_id,
        account_id=account_id,
        title=title,
        question=title,
        sql_query="SELECT 1 AS value",
        chart_type=chart_type,
        db_config_id=None,
    )


def test_kpis_are_laid_out_before_other_visuals(dashboard_users):
    account_id, owner_id, _ = dashboard_users
    dashboard = store.create_dashboard(account_id, owner_id, "thread-a", "Executive")
    bar_id = _chart(account_id, owner_id, "bar", "Revenue by state")
    assert store.add_chart_to_dashboard(dashboard["id"], bar_id, owner_id, account_id)
    kpi_id = _chart(account_id, owner_id, "kpi", "Total revenue")
    assert store.add_chart_to_dashboard(dashboard["id"], kpi_id, owner_id, account_id)

    charts = store.list_dashboard_charts(dashboard["id"], owner_id)
    assert [chart["id"] for chart in charts] == [kpi_id, bar_id]
    assert charts[0]["grid_y"] == 0
    assert charts[0]["grid_w"] == 3
    assert charts[1]["grid_y"] >= charts[0]["grid_h"]


def test_manual_layout_is_locked_when_new_kpi_is_added(dashboard_users):
    account_id, owner_id, _ = dashboard_users
    dashboard = store.create_dashboard(account_id, owner_id, "thread-b", "Operations")
    chart_id = _chart(account_id, owner_id, "bar", "Orders")
    store.add_chart_to_dashboard(dashboard["id"], chart_id, owner_id, account_id)
    assert store.update_dashboard_layouts(
        dashboard["id"], owner_id, account_id,
        [{"chart_id": chart_id, "x": 0, "y": 9, "w": 12, "h": 6, "position": 1}],
    )
    kpi_id = _chart(account_id, owner_id, "kpi", "Order count")
    store.add_chart_to_dashboard(dashboard["id"], kpi_id, owner_id, account_id)

    charts = {chart["id"]: chart for chart in store.list_dashboard_charts(dashboard["id"], owner_id)}
    assert (charts[chart_id]["grid_x"], charts[chart_id]["grid_y"]) == (0, 9)
    assert (charts[chart_id]["grid_w"], charts[chart_id]["grid_h"]) == (12, 6)
    assert charts[chart_id]["layout_locked"] == 1
    assert charts[kpi_id]["grid_y"] == 0


def test_layout_update_is_dashboard_owner_and_tenant_scoped(dashboard_users):
    account_id, owner_id, viewer_id = dashboard_users
    dashboard = store.create_dashboard(account_id, owner_id, "thread-c", "Private")
    chart_id = _chart(account_id, owner_id, "line", "Trend")
    store.add_chart_to_dashboard(dashboard["id"], chart_id, owner_id, account_id)
    assert not store.update_dashboard_layouts(
        dashboard["id"], viewer_id, account_id,
        [{"chart_id": chart_id, "x": 6, "y": 6, "w": 6, "h": 5}],
    )


def test_changing_visual_to_kpi_reflows_it_to_top(dashboard_users):
    account_id, owner_id, _ = dashboard_users
    dashboard = store.create_dashboard(account_id, owner_id, "thread-kpi", "Metrics")
    first_id = _chart(account_id, owner_id, "bar", "Breakdown")
    second_id = _chart(account_id, owner_id, "bar", "Single value")
    store.add_chart_to_dashboard(dashboard["id"], first_id, owner_id, account_id)
    store.add_chart_to_dashboard(dashboard["id"], second_id, owner_id, account_id)
    assert store.update_dashboard_chart(
        dashboard["id"], second_id, owner_id, account_id, chart_type="kpi"
    )
    charts = store.list_dashboard_charts(dashboard["id"], owner_id)
    assert charts[0]["id"] == second_id
    assert charts[0]["grid_y"] == 0
    assert charts[0]["grid_w"] == 3


def test_legacy_pins_migrate_once_into_named_dashboard(dashboard_users):
    account_id, owner_id, _ = dashboard_users
    chart_id = _chart(account_id, owner_id, "bar", "Legacy chart")
    migrated = store.migrate_legacy_charts(account_id, owner_id)
    assert migrated and migrated["name"] == "My Dashboard"
    charts = store.list_dashboard_charts(migrated["id"], owner_id)
    assert [chart["id"] for chart in charts] == [chart_id]
    assert store.migrate_legacy_charts(account_id, owner_id) is None


def test_dashboard_subscription_requires_view_access_and_is_user_scoped(dashboard_users):
    account_id, owner_id, viewer_id = dashboard_users
    dashboard = store.create_dashboard(
        account_id, owner_id, "thread-d", "Team dashboard", visibility="team"
    )
    store.publish_dashboard(dashboard["id"], owner_id, account_id)
    subscription = store.subscribe_dashboard(
        dashboard["id"], viewer_id, account_id, cadence="weekly"
    )
    assert subscription["cadence"] == "weekly"
    assert store.get_dashboard_subscription(dashboard["id"], viewer_id, account_id)
    assert store.unsubscribe_dashboard(dashboard["id"], viewer_id, account_id)
    assert store.get_dashboard_subscription(dashboard["id"], viewer_id, account_id) is None

    private = store.create_dashboard(account_id, owner_id, "thread-e", "Owner only")
    with pytest.raises(PermissionError):
        store.subscribe_dashboard(private["id"], viewer_id, account_id)
