"""
tests/chat_render.py

Render portal_chat.html for a test, with its own default fixtures.

Not a test module. The environment, the language context and visible() live in
tests/portal_render.py, which every portal page shares; what stays here is the
chat page's own context -- the client, the suggestions, the schema list and the
two usage meters.
"""

from __future__ import annotations

from tests.portal_render import visible  # noqa: F401  (re-exported for callers)


def render(*, lang=None, enabled=True, suggestions=None, schemas=None,
           client_name="Acme") -> str:
    from tests.portal_render import render as _render

    return _render(
        "portal_chat.html", lang=lang, path="/portal/chat",
        user={"id": 1, "name": "Ada Lovelace", "account_id": "a",
              "role": "analyst", "group_name": "Analysts"},
        client={"client_name": client_name, "chat_ui_enabled": 1},
        enabled=enabled,
        suggestions=list(suggestions if suggestions is not None else [
            {"question": "revenue by region", "fqn": "DW.SALES"},
            {"question": "margin by product", "fqn": "DW.SALES"},
        ]),
        available_schemas=list(schemas if schemas is not None else [
            {"name": "HR", "table_count": 3}, {"name": "FIN", "table_count": 1},
        ]),
        query_status={"blocked": False, "warning": "", "limit_label": "500",
                      "limit_pct": 1, "remaining_label": "497", "used_label": "3",
                      "message": "497 queries remaining this month."},
        token_usage={"total_label": "1M", "input_label": "600K",
                     "output_label": "400K"},
        selected_dashboard=None,
    )
