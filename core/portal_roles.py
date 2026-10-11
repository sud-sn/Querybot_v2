"""
core/portal_roles.py

What a reader may do in the portal, by their portal role.

    viewer   opens the dashboards shared with them, uses their filters, follows them
    asker    also asks questions in the chat, saves answers to dashboards of their own (never shared),
             suggests metrics and fixes, exports answers
    creator  also has dashboards made from a request, changes them by talking, shares and publishes them

Which tables and rows a reader sees is a separate matter (their group, their own extra tables, row rules,
masking): a role never widens it. Every account made before roles is a creator, so nobody loses anything on
the upgrade.

Every surface asks ``can(user, action)``: the pages, their APIs, the chat socket and the chat platforms. A
refusal says why in the reader's language (``refusal``).
"""

from __future__ import annotations

from typing import Any

VIEWER, ASKER, CREATOR = "viewer", "asker", "creator"
PORTAL_ROLES = (VIEWER, ASKER, CREATOR)
_RANK = {VIEWER: 1, ASKER: 2, CREATOR: 3}

# The least role each action needs.
ACTIONS: dict[str, str] = {
    "view_dashboards": VIEWER,      # dashboards shared with them, and their own
    "follow": VIEWER,               # follow a dashboard, get its updates
    "settings": VIEWER,             # their own profile, password, chart style
    "ask": ASKER,                   # the chat, the data guide, history, answer feedback
    "save_answer": ASKER,           # add an answer to a dashboard of their own
    "own_dashboards": ASKER,        # make and arrange dashboards of their own
    "suggest": ASKER,               # suggest a metric or a fix to the admins
    "export": ASKER,                # download an answer's rows
    "create_dashboard": CREATOR,    # a dashboard made from a request
    "talk_edit": CREATOR,           # change a dashboard by talking
    "share": CREATOR,               # share or publish a dashboard, look people up to share with
    "reports": CREATOR,             # scheduled reports
}

_REFUSAL = {
    "ask": "portal.role.no_chat",
    "create_dashboard": "portal.role.no_dashboard_request",
    "talk_edit": "portal.role.no_dashboard_talk",
    "share": "portal.role.no_share",
}


def portal_role(user: dict[str, Any] | None) -> str:
    """The reader's portal role; an account that has none (made before roles) is a creator."""
    value = str((user or {}).get("portal_role") or "").strip().lower()
    return value if value in PORTAL_ROLES else CREATOR


def can(user: dict[str, Any] | None, action: str) -> bool:
    if not user:
        return False
    need = ACTIONS.get(action)
    if need is None:
        raise ValueError(f"Unknown portal action: {action!r}")
    return _RANK[portal_role(user)] >= _RANK[need]


def refusal(action: str, lang: str | None = None) -> str:
    """Why the reader cannot, in their language."""
    from core.i18n import t

    return t(_REFUSAL.get(action, "portal.role.not_allowed"), lang=lang)


def home_path(user: dict[str, Any] | None) -> str:
    """Where a reader lands: the chat for those who ask, their dashboards for a viewer."""
    return "/portal/chat" if can(user, "ask") else "/portal/dashboard"
