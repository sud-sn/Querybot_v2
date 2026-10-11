"""
core/tile_alerts.py

Alerts on a dashboard's number tiles: "tell me when Net amount is above $200K", "... falls more than 10% on the
period before".

Anyone who can open a dashboard sets one for themselves from the tile's bell; the dashboard's owner, when they may
share it, also for people and groups of the workspace (saving shares the dashboard with them). Checked every hour or
every day by core/notification_scheduler.py, for each person it tells as that person's own access draws the tile --
their tables, row rules, masking -- so nobody is told a number they could not see on the dashboard.

    fires        once, when the line is crossed for a person: a notice in the app, shown at once if they are on a page,
                 and an email too when it was asked for
    comes back   once, when it is no longer crossed
    the tile     its bell shows who is told, red while it is crossed for the reader; the alert's dialog lists what it
                 told them lately
    paused       the tile or the dashboard is gone, or its maker can no longer open it: paused, saying why

The change conditions read the tile's own comparison with the period before (its ▲/▼ line): a tile without one is
never crossed by them.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Optional

import store
from store import tile_alert_store as alerts

log = logging.getLogger("querybot.tile_alerts")


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


# ── Reading a tile ───────────────────────────────────────────────────────────

def _percent(text) -> Optional[float]:
    """"+4.6%" or "-1.2 pts" as a number; None when the tile has no comparison."""
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text or ""))
    return float(found.group(0).replace(",", "")) if found else None


def reading(drawn: dict) -> dict:
    """What a drawn number tile says: {"value", "change" (% on the period before, or points for a rate), "vs",
    "display", "format"}; value None when it is not a number tile or could not be drawn."""
    kpi = drawn.get("kpi") if not drawn.get("error") else None
    if not isinstance(kpi, dict):
        return {"value": None, "change": None, "vs": "", "display": "", "format": ""}
    try:
        value = float(kpi.get("value"))
    except (TypeError, ValueError):
        value = None
    change = kpi.get("change") if isinstance(kpi.get("change"), dict) else {}
    moved = None
    if change.get("direction") == "flat":
        moved = 0.0
    elif change:
        moved = _percent(change.get("pct")) if change.get("pct") else _percent(change.get("change"))
    return {"value": value, "change": moved, "vs": str(change.get("vs") or ""),
            "display": str(drawn.get("kpi_display") or kpi.get("value") or ""), "format": str(kpi.get("format") or "")}


def crossed(condition: str, threshold: float, read: dict) -> Optional[bool]:
    """Whether the line is crossed; None when the tile cannot say (no number, or no comparison for a change)."""
    if condition in ("above", "below"):
        if read["value"] is None:
            return None
        return read["value"] > threshold if condition == "above" else read["value"] < threshold
    moved = read["change"]
    if moved is None:
        return None
    if condition == "rise":
        return moved > threshold
    if condition == "fall":
        return moved < -threshold
    return abs(moved) > threshold


def threshold_text(alert: dict, fmt: str, lang: str) -> str:
    from core.i18n import activate_language, deactivate_language
    from core.response_builder import _format_number

    if alert["condition"] in ("rise", "fall", "move"):
        return f"{float(alert['threshold']):g}%"
    token = activate_language(lang)
    try:
        return str(_format_number(float(alert["threshold"]), fmt or None) or alert["threshold"])
    finally:
        deactivate_language(token)


def describe(alert: dict, fmt: str, lang: str) -> str:
    """"is above $200,000.00", "falls more than 10% on the period before", in the reader's language."""
    from core.i18n import t

    return t(f"alert.when.{alert['condition']}", lang=lang, threshold=threshold_text(alert, fmt, lang))


# ── Saving one ───────────────────────────────────────────────────────────────

def read_form(payload: dict, user: dict, *, may_tell_others: bool) -> dict | str:
    """The alert's fields from what its dialog sent, checked; or what is wrong, as a message id (alert.error.*)."""
    condition = str(payload.get("condition") or "")
    if condition not in alerts.CONDITIONS:
        return "alert.error.condition"
    try:
        threshold = float(str(payload.get("threshold")).replace(",", ""))
    except (TypeError, ValueError):
        return "alert.error.threshold"
    if threshold != threshold or abs(threshold) == float("inf"):
        return "alert.error.threshold"
    if condition in ("rise", "fall", "move") and threshold <= 0:
        return "alert.error.percent"
    try:
        every = int(payload.get("every") or 60)
    except (TypeError, ValueError):
        every = 60
    recipients = []
    if may_tell_others:
        for item in payload.get("recipients") or []:
            if not isinstance(item, dict):
                continue
            try:
                subject_id = int(item.get("id") or 0)
            except (TypeError, ValueError):
                continue
            if item.get("type") == "user":
                person = store.get_user(subject_id)
                if person and person.get("account_id") == user["account_id"] and person.get("is_active") \
                        and int(person["id"]) != int(user["id"]):
                    recipients.append(("user", subject_id))
            elif item.get("type") == "group":
                group = store.get_group(subject_id)
                if group and group.get("account_id") == user["account_id"]:
                    recipients.append(("group", subject_id))
    return {"condition": condition, "threshold": threshold, "check_minutes": every if every in alerts.EVERY else 60,
            "by_email": bool(payload.get("by_email")),
            "recipients": list(dict.fromkeys(recipients))[:alerts.MAX_SUBJECTS]}


def save(user: dict, dashboard: dict, chart_id: int, fields: dict) -> int:
    """Keep an alert, first checked within the minute; share the dashboard with whoever else it tells."""
    recipients = fields.pop("recipients")
    alert_id = alerts.create_alert(user["account_id"], int(dashboard["id"]), int(chart_id), int(user["id"]),
                                   next_check_at=_stamp(_utc_now()), recipients=recipients, **fields)
    for subject_type, subject_id in recipients:
        try:
            store.add_dashboard_share(int(dashboard["id"]), int(user["id"]), user["account_id"], subject_type,
                                      int(subject_id))
        except Exception as exc:  # noqa: BLE001 - the alert stands; whoever cannot open the dashboard is not told
            log.warning("Alert %s: could not share dashboard %s with %s %s: %s", alert_id, dashboard.get("id"),
                        subject_type, subject_id, exc)
    return alert_id


# ── Checking ─────────────────────────────────────────────────────────────────

def run_due_tile_alerts(now: Optional[dt.datetime] = None) -> int:
    """Check every alert that is due. How many were checked."""
    now = now or _utc_now()
    checked = 0
    for alert in alerts.due_alerts(_stamp(now)):
        upcoming = _stamp(now + dt.timedelta(minutes=int(alert.get("check_minutes") or 60)))
        if not alerts.claim_check(int(alert["id"]), str(alert["next_check_at"]), upcoming):
            continue                                     # another scheduler has it
        try:
            check(alert)
            checked += 1
        except Exception:  # noqa: BLE001 - one alert failing must not stop the others
            log.exception("Tile alert %s could not be checked", alert.get("id"))
    return checked


def _tile_for(alert: dict, person: dict) -> Optional[dict]:
    charts = store.list_dashboard_charts_for_view(int(alert["dashboard_id"]), int(person["id"]),
                                                  str(alert["account_id"]))
    return next((c for c in charts if int(c["id"]) == int(alert["chart_id"])), None)


def _draw(chart: dict, person: dict) -> dict:
    from core.i18n import activate_language, deactivate_language
    from portal.routes import _refresh_chart

    db_cfg = store.get_db_config(chart["db_config_id"]) if chart.get("db_config_id") else None
    token = activate_language(str(person.get("lang") or "en"))
    try:
        return _refresh_chart(chart, db_cfg, person)
    finally:
        deactivate_language(token)


def check(alert: dict) -> dict:
    """Check one alert for everyone it tells; fire or clear it for each. {user id: what happened}."""
    owner = store.get_user(int(alert["owner_id"]))
    if not owner or not owner.get("is_active"):
        alerts.set_alert_active(int(alert["id"]), False, reason="Its maker can no longer sign in.")
        return {}
    if _tile_for(alert, owner) is None:
        alerts.set_alert_active(int(alert["id"]), False, reason="The tile is no longer on a dashboard its maker can open.")
        return {}
    board = store.get_dashboard_for_view(int(alert["dashboard_id"]), int(owner["id"]), str(alert["account_id"])) or {}
    from store import email_store

    done = {}
    for person in alerts.alert_people(alert):
        if person.get("email") and email_store.opted_out(str(person["email"]), "alert", str(alert["id"])):
            done[int(person["id"])] = "stopped"          # "Stop telling me", or the link in an email
            continue
        chart = _tile_for(alert, person)
        if chart is None:
            done[int(person["id"])] = "cannot open it"
            continue
        read = reading(_draw(chart, person))
        now_crossed = crossed(str(alert["condition"]), float(alert["threshold"]), read)
        state = alerts.get_state(int(alert["id"]), int(person["id"]))
        was = (state or {}).get("state") or alerts.OK
        if now_crossed is None:
            done[int(person["id"])] = "no reading"
            continue
        if now_crossed and was != alerts.FIRING:
            _tell(alert, person, owner, board, chart, alerts.FIRED, read)
            alerts.put_state(int(alert["id"]), int(person["id"]), state=alerts.FIRING, value=read["value"], changed=True)
            done[int(person["id"])] = alerts.FIRED
        elif not now_crossed and was == alerts.FIRING:
            _tell(alert, person, owner, board, chart, alerts.CLEARED, read)
            alerts.put_state(int(alert["id"]), int(person["id"]), state=alerts.OK, value=read["value"], changed=True)
            done[int(person["id"])] = alerts.CLEARED
        else:
            alerts.put_state(int(alert["id"]), int(person["id"]), state=was, value=read["value"], changed=state is None)
            done[int(person["id"])] = "unchanged"
    return done


def _tell(alert: dict, person: dict, owner: dict, board: dict, chart: dict, kind: str, read: dict) -> None:
    """A notice in the app, and an email when the alert asked for one, in the person's language."""
    from core.i18n import t
    from core.notices import tell

    lang = str(person.get("lang") or "en")
    tile = str(chart.get("title") or "")
    change = f"{read['change']:+g}%" if read["change"] is not None else ""
    if kind == alerts.FIRED:
        title = t(f"alert.fired.{alert['condition']}", lang=lang, tile=tile, value=read["display"],
                  threshold=threshold_text(alert, read["format"], lang), change=change, vs=read["vs"])
    else:
        title = t("alert.cleared", lang=lang, tile=tile, value=read["display"])
    body = t("alert.body", lang=lang, dashboard=board.get("name") or "", when=describe(alert, read["format"], lang))
    link = f"/portal/dashboard?dashboard_id={int(alert['dashboard_id'])}"
    event_id = alerts.add_event(int(alert["id"]), int(person["id"]), kind, value=read["value"], display=read["display"])
    tell(str(alert["account_id"]), int(person["id"]), "tile_alert", title, body, link)
    if not int(alert.get("by_email") or 0) or not person.get("email"):
        return
    from core.mail import links, outbox, render

    base = outbox.portal_base()
    address = str(person["email"])
    ref = str(alert["id"])
    stop = links.unsubscribe_url(base, address, "alert", ref)
    why = t("email.alert.why", lang=lang, owner=owner.get("name") or "", tile=tile)
    rows = [(tile, read["display"])]
    if read["change"] is not None and read["vs"]:
        rows.append((t("ui.kpi.vs", lang=lang, period=read["vs"]), change))
    button = (t("email.open_dashboard", lang=lang), base + link)
    outbox.queue(kind="alert", to=address, subject=title, account_id=str(alert["account_id"]), user_id=int(person["id"]),
                 ref=ref, opt_out_kind="alert", unsubscribe_url=stop, dedupe_key=f"alert:{alert['id']}:{event_id}",
                 text=render.text(title, [body], rows=rows, button=button, why=why, unsubscribe=stop, lang=lang),
                 html=render.page(title, [body], rows=rows, button=button, why=why, unsubscribe=stop, lang=lang))
