"""
tests/dashboard_render.py

Render portal_dashboard.html for a test, with its own default fixtures.

Not a test module. The environment, the language context and visible() live in
tests/portal_render.py, which every portal page shares; what stays here is the
dashboard's own context -- the artifact, the chart and the two usage meters.
"""

from __future__ import annotations

from tests.portal_render import Req as _Req, visible


def render(*, charts=None, artifact=None, library=None, lang=None,
           allowed_tables=("DW.SALES",)) -> str:
    from tests.portal_render import render as _render

    default_artifact = {
        "id": 5, "name": "Pharmacy performance", "description": "",
        "status": "published", "visibility": "team", "version": 3,
        "can_edit": 1, "refresh_schedule": "daily", "thread_id": "t",
        "last_refreshed_at": "", "tabs_json": "", "filters_json": "",
        "created_at": "", "updated_at": "", "published_at": "",
        "user_id": 1, "account_id": "a",
    }
    artifact = default_artifact if artifact is None else artifact
    return _render(
        "portal_dashboard.html", lang=lang, path="/portal/dashboard",
        user={"id": 1, "name": "Ada Lovelace", "account_id": "a",
              "role": "analyst", "group_name": "Analysts"},
        client={"client_name": "Acme"},
        charts=list(charts or []),
        dashboard_artifact=artifact or None,
        dashboards=library if library is not None else [],
        dashboard_filters=[], dashboard_sources=[],
        dashboard_tabs=["Overview"], dashboard_versions=[],
        dashboard_subscription=None, selected_tab="Overview", welcome=False,
        allowed_tables=list(allowed_tables), group_tables=[], monthly_count=3,
        query_status={"blocked": False, "warning": "", "limit_label": "500",
                      "limit_pct": 1, "remaining_label": "497", "used_label": "3"},
        token_status={"unlimited": False, "limit_pct": 10, "limit_label": "1M",
                      "remaining_label": "900K", "used_label": "100K",
                      "limit": 1, "remaining": 1, "total_tokens": 1},
    )


CHART = {
    "id": 101, "title": "Revenue by region", "question": "revenue by region",
    "chart_type": "bar", "color_palette": "default", "row_count": 12,
    "grid_x": 0, "grid_y": 0, "grid_w": 6, "grid_h": 5, "sort_enabled": False,
    "from_cache": False, "cache_refreshed_at": "", "chart_json": '{"type":"bar"}',
    "error": None, "error_next_step": "", "kpi": None, "kpi_display": "",
    "table_columns": [], "table_rows": [], "table_column_formats": {},
    "table_truncated": False, "table_shown": 0, "filter_warnings": [],
    "dashboard_tab": "Overview",
}
