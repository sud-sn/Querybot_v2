"""How people reach a workspace: the web portal, always, and one chat app, chosen on the workspace's Settings page.

A workspace was created with a chat platform chosen up front (or "Web portal only"), and the platform's credentials
were typed on a separate Platforms page, then picked from a list (Teams only); a Slack or Zoom workspace received
messages only when its own account ID happened to be the chat's team ID. Now a workspace is created with none: the
web portal is always on, and its Settings page says how people sign in to it. Choosing Microsoft Teams, Slack or
Zoom there shows the credentials that app needs, the ID whose messages come to this workspace, and the address to
paste into the app, on the same page. The credentials are kept encrypted as a platform configuration of the
workspace's own (store.save_platform); a secret is never shown back, and left empty it is kept.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse

import store
from admin.routes import _is_auth, router

log = logging.getLogger("querybot.admin")

# Each chat app: the credentials it needs (store.PLATFORM_FIELDS, in this order), where to find each one, the field
# naming whose messages come to this workspace, and where the app is given Solviz's address.
CHANNELS: dict[str, dict[str, Any]] = {
    "teams": {
        "label": "Microsoft Teams",
        "endpoint": "webhook/teams",
        "paste": "the Azure Bot's Configuration, as its Messaging endpoint",
        "fields": [
            {"name": "app_id", "label": "Microsoft App ID", "hint": "Azure Bot → Configuration"},
            {"name": "app_password", "label": "App password", "secret": True,
             "hint": "A client secret of the bot's app registration (Certificates & secrets)"},
            {"name": "tenant_id", "label": "Directory (tenant) ID", "route": True,
             "hint": "Your Microsoft Entra tenant: messages from it come to this workspace"},
        ],
    },
    "slack": {
        "label": "Slack",
        "endpoint": "webhook/slack",
        "paste": "the Slack app's Event Subscriptions, as its Request URL",
        "fields": [
            {"name": "bot_token", "label": "Bot User OAuth Token", "secret": True,
             "hint": "OAuth & Permissions; it starts with xoxb-"},
            {"name": "signing_secret", "label": "Signing secret", "secret": True,
             "hint": "Basic Information → App Credentials"},
            {"name": "app_id", "label": "App ID", "hint": "Basic Information → App Credentials"},
            {"name": "team_id", "label": "Slack workspace ID", "route": True,
             "hint": "Starts with T (in the workspace's About page): messages from it come to this workspace"},
        ],
    },
    "zoom": {
        "label": "Zoom Team Chat",
        "endpoint": "webhook/zoom",
        "paste": "the Zoom app's Team Chat Subscription, as its Bot endpoint URL",
        "fields": [
            {"name": "client_id", "label": "Client ID", "hint": "The Zoom app's App Credentials"},
            {"name": "client_secret", "label": "Client secret", "secret": True,
             "hint": "The Zoom app's App Credentials"},
            {"name": "bot_jid", "label": "Bot JID", "hint": "Team Chat Subscription; it ends with @xmpp.zoom.us"},
            {"name": "webhook_secret", "label": "Secret token", "secret": True,
             "hint": "The Zoom app's Access page"},
            {"name": "account_id", "label": "Zoom account ID", "route": True,
             "hint": "Your Zoom account (Account Profile): messages from it come to this workspace"},
        ],
    },
}


def _route_field(kind: str) -> str:
    return next(f["name"] for f in CHANNELS[kind]["fields"] if f.get("route"))


def channel_of(client: dict[str, Any]) -> dict[str, Any]:
    """The workspace's chat app as its Settings page shows it: which one, the values it may show back (never a
    secret: only whether one is saved), and the ID its messages come from."""
    kind = str(client.get("platform_type") or "")
    platform = None
    if kind in CHANNELS and client.get("platform_config_id"):
        platform = store.get_platform(int(client["platform_config_id"]))
        if not platform or platform.get("platform_type") != kind:
            platform = None
    if platform is None:
        return {"kind": "", "shown": {}, "saved": set()}
    credentials = platform.get("credentials") or {}
    secrets = {f["name"] for f in CHANNELS[kind]["fields"] if f.get("secret")}
    values = {k: v for k, v in credentials.items() if k not in secrets}
    route = _route_field(kind)
    values[route] = str((client.get("teams_tenant_id") if kind == "teams" else client.get("chat_team_id"))
                        or credentials.get(route) or "")
    return {"kind": kind, "shown": values, "saved": {k for k in secrets if credentials.get(k)},
            "active": bool(platform.get("is_active", True))}


def _shared(platform_id: int, account_id: str) -> bool:
    """Does another workspace use this chat app's credentials?"""
    return any(c.get("platform_config_id") == platform_id and c["account_id"] != account_id
               for c in store.list_clients())


def _back(account_id: str, **query: str) -> RedirectResponse:
    tail = "&".join(f"{k}={quote(v)}" for k, v in query.items())
    return RedirectResponse(f"/admin/clients/{quote(account_id)}/settings?{tail}#channels", status_code=303)


@router.post("/clients/{account_id}/channel")
async def channel_save(request: Request, account_id: str):
    """Save how people reach the workspace: no chat app (the web portal alone), or one with its credentials."""
    if not _is_auth(request):
        return RedirectResponse("/admin/login", status_code=303)
    client = store.get_client(account_id)
    if not client:
        return RedirectResponse("/admin/clients", status_code=303)
    form = await request.form()
    kind = str(form.get("chat_app") or "").strip()
    if not kind:
        # The platform type stays as it is: older databases allow only zoom, teams or slack there, and with no
        # chat app's credentials assigned (platform_config_id) it names none (channel_of).
        store.update_client_meta(account_id, platform_config_id=0, teams_tenant_id="", chat_team_id="",
                                 portal_only=1, chat_ui_enabled=1)
        log.info("Admin: workspace %s is reached through the web portal alone", account_id)
        return _back(account_id, saved="channel")
    if kind not in CHANNELS:
        return _back(account_id, channel_error="Choose Microsoft Teams, Slack or Zoom, or no chat app.")
    spec = CHANNELS[kind]
    current = channel_of(client)
    kept = {}
    if current["kind"] == kind:
        kept = (store.get_platform(int(client["platform_config_id"])) or {}).get("credentials") or {}
    values: dict[str, str] = {}
    missing: list[str] = []
    for field in spec["fields"]:
        typed = str(form.get(field["name"]) or "").strip()
        value = typed or (str(kept.get(field["name"]) or "") if field.get("secret") else "")
        if not value:
            missing.append(field["label"])
        values[field["name"]] = value
    if missing:
        return _back(account_id, channel=kind, channel_error=f"{spec['label']} needs: {', '.join(missing)}.")
    route = _route_field(kind)
    team = values[route]
    other = (store.get_client_by_teams_tenant_id(team) if kind == "teams"
             else store.get_client_by_chat_team(kind, team))
    if other and other["account_id"] != account_id:
        return _back(account_id, channel=kind, channel_error=(
            f"Messages from {team} already go to the workspace {other.get('client_name') or other['account_id']}."))
    credentials = {k: v for k, v in values.items() if k in store.PLATFORM_FIELDS[kind]}
    platform_id = int(client["platform_config_id"]) if current["kind"] == kind else None
    if platform_id and _shared(platform_id, account_id):
        platform_id = None       # another workspace's too (picked on the Platforms page): this one gets its own
    name = f"{client.get('client_name') or account_id} · {spec['label']}"
    try:
        platform_id = store.save_platform(kind, name, credentials, platform_id)
        store.update_client_meta(account_id, platform_type=kind, platform_config_id=int(platform_id),
                                 teams_tenant_id=team if kind == "teams" else "",
                                 chat_team_id="" if kind == "teams" else team, portal_only=0, chat_ui_enabled=1)
    except ValueError as exc:
        return _back(account_id, channel=kind, channel_error=str(exc))
    log.info("Admin: workspace %s is reached through %s too", account_id, spec["label"])
    return _back(account_id, saved="channel")
