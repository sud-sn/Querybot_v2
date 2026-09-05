"""
tests/dashboard_render.py

Render portal_dashboard.html for a test.

Not a test module. It exists because the page's English labels moved into the
message catalogue, and the catalogue is injected into the page as JSON -- so
`"Query limit" in template_source` is now true whatever the page renders, and
every assertion of that shape became a tautology at the same moment.

The fix is to look at what the page actually produces. The language context
comes from portal.routes._language_context, the PRODUCTION context processor,
so a fixture here cannot drift from what a browser is served.
"""

from __future__ import annotations

import re


class _URL:
    path = "/portal/dashboard"


class _Req:
    url = _URL()
    query_params: dict = {}

    def __init__(self, lang=None):
        self.cookies = {"qb_lang": lang} if lang else {}
        self.headers = {}


def visible(markup: str) -> str:
    """The markup with <script> blocks removed -- see the module docstring."""
    return re.sub(r"<script\b.*?</script>", " ", markup, flags=re.S | re.I)


def render(*, charts=None, artifact=None, library=None, lang=None,
           allowed_tables=("DW.SALES",)) -> str:
    from jinja2 import ChainableUndefined

    import portal.routes as pr

    default_artifact = {
        "id": 5, "name": "Pharmacy performance", "description": "",
        "status": "published", "visibility": "team", "version": 3,
        "can_edit": 1, "refresh_schedule": "daily", "thread_id": "t",
        "last_refreshed_at": "", "tabs_json": "", "filters_json": "",
        "created_at": "", "updated_at": "", "published_at": "",
        "user_id": 1, "account_id": "a",
    }
    artifact = default_artifact if artifact is None else artifact
    env = pr.templates.env
    previous, env.undefined = env.undefined, ChainableUndefined
    try:
        return env.get_template("portal_dashboard.html").render(
            request=_Req(lang),
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
            **pr._language_context(_Req(lang)),
        )
    finally:
        env.undefined = previous


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
