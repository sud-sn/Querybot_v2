"""
core/profile.py

A reader's own page (portal Settings): what they may change about themselves, and everything that tells them.

    profile        name, job title, language, time zone and start page, each checked here; the time zone is the one a
                   new schedule starts from, the start page where signing in lands (one they may still open, or
                   their workspace's own)
    photo          whatever image was uploaded is re-drawn as a PNG square of PHOTO_SIZE pixels: nothing of the
                   original file (its metadata, its location, a script hidden in it) is kept
    sign-ins       each one read as "Chrome on Windows", with its address and when
    notifications  every follow, schedule and alert that tells the reader, in one list, each with its on and off:
                   their own schedules and alerts pause and resume (for everyone they tell); one someone else made
                   for them stops or starts telling them alone
"""

from __future__ import annotations

import io
import re
import zoneinfo
from functools import lru_cache
from typing import Optional

import store
from store import profile_store

PHOTO_SIZE = 256
PHOTO_MAX_BYTES = 8 * 1024 * 1024
PHOTO_MAX_PIXELS = 40_000_000
PHOTO_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
NAME_MAX = 120
JOB_MAX = 120


@lru_cache(maxsize=1)
def zones() -> tuple[str, ...]:
    """The time zones a reader may pick: the IANA names, without the legacy aliases ("US/Eastern", "Etc/GMT+5")."""
    names = {z for z in zoneinfo.available_timezones()
             if "/" in z and not z.startswith(("Etc/", "SystemV/", "US/")) and z not in ("Factory", "localtime")}
    return tuple(["UTC"] + sorted(names))


def valid_zone(name: str) -> bool:
    return str(name or "") in zones()


# ── Start page ───────────────────────────────────────────────────────────────

def start_choices(user: dict) -> list[dict]:
    """Where signing in may land this reader: [{"value", "label_key" | "label"}]. "" is their workspace's own."""
    from core.portal_roles import can

    choices = [{"value": "", "label_key": "ui.profile.start.default"}]
    client = store.get_client(str(user.get("account_id") or "")) or {}
    if client.get("chat_ui_enabled") and can(user, "ask"):
        choices.append({"value": "home", "label_key": "ui.profile.start.home"})
        choices.append({"value": "chat", "label_key": "ui.profile.start.chat"})
    choices.append({"value": "dashboards", "label_key": "ui.profile.start.dashboards"})
    for board in store.list_dashboards(str(user["account_id"]), int(user["id"])):
        choices.append({"value": f"dashboard:{int(board['id'])}", "label": str(board.get("name") or "")})
    return choices


def start_path(user: dict | None) -> str:
    """The page this reader chose to land on, when they may still open it; "" for their workspace's own."""
    if not user or not user.get("id"):
        return ""
    try:
        chosen = profile_store.get_profile(int(user["id"]))["start_page"]
    except Exception:  # noqa: BLE001 - no profile table yet: the workspace's own
        return ""
    if not chosen:
        return ""
    from core.portal_roles import can

    if chosen in ("chat", "home"):
        client = store.get_client(str(user.get("account_id") or "")) or {}
        return f"/portal/{chosen}" if client.get("chat_ui_enabled") and can(user, "ask") else ""
    if chosen == "dashboards":
        return "/portal/dashboard"
    found = re.fullmatch(r"dashboard:(\d+)", chosen)
    if found and store.get_dashboard_for_view(int(found.group(1)), int(user["id"]), str(user["account_id"])):
        return f"/portal/dashboard?dashboard_id={int(found.group(1))}"
    return ""


# ── The profile form ─────────────────────────────────────────────────────────

def read_form(form, user: dict) -> dict | str:
    """The profile's fields from its form, checked; or what is wrong, as a message id (ui.profile.error.*)."""
    name = " ".join(str(form.get("name") or "").split())
    if not name:
        return "ui.profile.error.name"
    if len(name) > NAME_MAX:
        return "ui.profile.error.name_long"
    job = " ".join(str(form.get("job_title") or "").split())
    if len(job) > JOB_MAX:
        return "ui.profile.error.job_long"
    zone = str(form.get("timezone") or "").strip()
    if zone and not valid_zone(zone):
        return "ui.profile.error.timezone"
    start = str(form.get("start_page") or "").strip()
    if start not in {c["value"] for c in start_choices(user)}:
        return "ui.profile.error.start"
    lang = str(form.get("lang") or "").strip()
    return {"name": name, "job_title": job, "timezone": zone, "start_page": start,
            "lang": store.normalise_language(lang) if lang else ""}


# ── The photo ────────────────────────────────────────────────────────────────

def photo_png(data: bytes) -> bytes:
    """The uploaded image re-drawn as a PHOTO_SIZE square PNG (the middle of it when it is not square), turned the
    way its camera meant. ValueError with a message id (ui.profile.error.photo_*) when it is not one we take."""
    from PIL import Image, ImageOps

    if not data:
        raise ValueError("ui.profile.error.photo_none")
    if len(data) > PHOTO_MAX_BYTES:
        raise ValueError("ui.profile.error.photo_large")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            kind = probe.format
            width, height = probe.size
            probe.verify()
    except Exception:  # noqa: BLE001 - anything Pillow cannot read is not a photo
        raise ValueError("ui.profile.error.photo_kind") from None
    if kind not in PHOTO_FORMATS:
        raise ValueError("ui.profile.error.photo_kind")
    if width * height > PHOTO_MAX_PIXELS or not width or not height:
        raise ValueError("ui.profile.error.photo_large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(0)
            picture = ImageOps.exif_transpose(image)
            picture = picture.convert("RGBA")
    except Exception:  # noqa: BLE001
        raise ValueError("ui.profile.error.photo_kind") from None
    side = min(picture.size)
    left, top = (picture.width - side) // 2, (picture.height - side) // 2
    square = picture.crop((left, top, left + side, top + side)).resize((PHOTO_SIZE, PHOTO_SIZE), Image.LANCZOS)
    out = io.BytesIO()
    square.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ── Sign-ins ─────────────────────────────────────────────────────────────────

_BROWSERS = (("Edg/", "Edge"), ("OPR/", "Opera"), ("Firefox/", "Firefox"), ("Chrome/", "Chrome"),
             ("CriOS/", "Chrome"), ("FxiOS/", "Firefox"), ("Safari/", "Safari"))
_SYSTEMS = (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android"), ("Windows", "Windows"),
            ("Mac OS X", "macOS"), ("Macintosh", "macOS"), ("CrOS", "ChromeOS"), ("Linux", "Linux"))


def device(agent: str) -> tuple[str, str]:
    """("Chrome", "Windows") from a browser's User-Agent; "" for what is not recognised."""
    agent = str(agent or "")
    browser = next((name for mark, name in _BROWSERS if mark in agent), "")
    system = next((name for mark, name in _SYSTEMS if mark in agent), "")
    return browser, system


def sign_ins(user: dict, lang: str) -> list[dict]:
    """The reader's latest sign-ins, newest first, each when (in their time zone), on what and from where."""
    import datetime as dt

    from core.dashboard_snapshot import local_time
    from core.i18n import t

    zone = profile_store.get_profile(int(user["id"]))["timezone"] or "UTC"
    found = []
    for row in profile_store.recent_sign_ins(int(user["id"])):
        browser, system = device(row.get("agent") or "")
        if browser and system:
            what = t("ui.profile.device", lang=lang, browser=browser, system=system)
        else:
            what = browser or system or t("ui.profile.device_unknown", lang=lang)
        try:
            moment = dt.datetime.fromisoformat(str(row["at"])[:19])
            when = local_time(moment, zone, lang)
        except (TypeError, ValueError):
            when = str(row.get("at") or "")
        found.append({"when": when, "device": what, "ip": str(row.get("ip") or "")})
    return found


# ── Everything that tells the reader ─────────────────────────────────────────

FOLLOW, SCHEDULE, SCHEDULE_FOR_ME, ALERT, ALERT_FOR_ME = ("follow", "schedule", "schedule-for-me", "alert",
                                                          "alert-for-me")
KINDS = (FOLLOW, SCHEDULE, SCHEDULE_FOR_ME, ALERT, ALERT_FOR_ME)


def _stopped(user: dict, kind: str, ref: int) -> bool:
    from store import email_store

    return bool(user.get("email")) and email_store.opted_out(str(user["email"]), kind, str(ref))


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def notifications(user: dict, lang: str) -> list[dict]:
    """[{"kind", "id", "dashboard", "dashboard_id", "tile", "what", "from", "on", "note"}], follows first, then
    schedules, then alerts; within each, the reader's own before those made for them."""
    from core import dashboard_schedule, tile_alerts
    from core.i18n import t
    from store.db import get_db

    account_id, user_id = str(user["account_id"]), int(user["id"])
    found: list[dict] = []
    with get_db() as conn:
        follows = conn.execute(
            """SELECT s.*, d.name AS dashboard FROM dashboard_subscription s
                 JOIN dashboard_artifact d ON d.id=s.dashboard_id AND d.account_id=s.account_id
                WHERE s.account_id=? AND s.user_id=? ORDER BY d.name, s.id""", (account_id, user_id)).fetchall()
        schedules = conn.execute(
            """SELECT s.*, d.name AS dashboard, o.name AS owner_name FROM dashboard_schedule s
                 JOIN dashboard_artifact d ON d.id=s.dashboard_id AND d.account_id=s.account_id
                 JOIN portal_user o ON o.id=s.owner_id
                WHERE s.account_id=? AND (s.owner_id=?
                      OR EXISTS (SELECT 1 FROM dashboard_schedule_recipient r WHERE r.schedule_id=s.id
                                    AND ((r.subject_type='user' AND r.subject_id=?)
                                      OR (r.subject_type='group' AND r.subject_id=(SELECT group_id FROM portal_user
                                                                                   WHERE id=?)))))
                ORDER BY CASE WHEN s.owner_id=? THEN 0 ELSE 1 END, d.name, s.id""",
            (account_id, user_id, user_id, user_id, user_id)).fetchall()
        alerts = conn.execute(
            """SELECT a.*, d.name AS dashboard, c.title AS tile, o.name AS owner_name FROM tile_alert a
                 JOIN dashboard_artifact d ON d.id=a.dashboard_id AND d.account_id=a.account_id
                 JOIN pinned_chart c ON c.id=a.chart_id
                 JOIN portal_user o ON o.id=a.owner_id
                WHERE a.account_id=? AND (a.owner_id=?
                      OR EXISTS (SELECT 1 FROM tile_alert_recipient r WHERE r.alert_id=a.id
                                    AND ((r.subject_type='user' AND r.subject_id=?)
                                      OR (r.subject_type='group' AND r.subject_id=(SELECT group_id FROM portal_user
                                                                                   WHERE id=?)))))
                ORDER BY CASE WHEN a.owner_id=? THEN 0 ELSE 1 END, d.name, a.id""",
            (account_id, user_id, user_id, user_id, user_id)).fetchall()
    for row in follows:
        follow = dict(row)
        cadence = t(f"ui.enum.cadence.{follow['cadence']}", lang=lang)
        found.append({"kind": FOLLOW, "id": int(follow["dashboard_id"]), "dashboard": follow["dashboard"],
                      "dashboard_id": int(follow["dashboard_id"]), "tile": "",
                      "what": t("ui.profile.n.follow_email" if follow["channel"] == "email" else "ui.profile.n.follow",
                                lang=lang, cadence=cadence),
                      "from": "", "on": follow["status"] == "active", "note": ""})
    for row in schedules:
        schedule = dict(row)
        mine = int(schedule["owner_id"]) == user_id
        if not mine and not store.get_dashboard_for_view(int(schedule["dashboard_id"]), user_id, account_id):
            continue
        found.append({"kind": SCHEDULE if mine else SCHEDULE_FOR_ME, "id": int(schedule["id"]),
                      "dashboard": schedule["dashboard"], "dashboard_id": int(schedule["dashboard_id"]), "tile": "",
                      "what": _first_upper(dashboard_schedule.describe(schedule, lang)),
                      "from": "" if mine else str(schedule.get("owner_name") or ""),
                      "on": bool(schedule["is_active"]) if mine else not _stopped(user, "schedule", schedule["id"]),
                      "note": "" if mine or schedule["is_active"] else t("ui.profile.n.paused_by_owner", lang=lang)})
    for row in alerts:
        alert = dict(row)
        mine = int(alert["owner_id"]) == user_id
        if not mine and not store.get_dashboard_for_view(int(alert["dashboard_id"]), user_id, account_id):
            continue
        found.append({"kind": ALERT if mine else ALERT_FOR_ME, "id": int(alert["id"]),
                      "dashboard": alert["dashboard"], "dashboard_id": int(alert["dashboard_id"]),
                      "tile": str(alert.get("tile") or ""),
                      "what": t("ui.profile.n.alert", lang=lang,
                                when=tile_alerts.describe(alert, alert.get("value_format") or "", lang)),
                      "from": "" if mine else str(alert.get("owner_name") or ""),
                      "on": bool(alert["is_active"]) if mine else not _stopped(user, "alert", alert["id"]),
                      "note": "" if mine or alert["is_active"] else t("ui.profile.n.paused_by_owner", lang=lang)})
    return found


def switch(user: dict, kind: str, item_id: int, on: bool) -> Optional[str]:
    """Turn one of the reader's notifications on or off; None when done, else a message id saying why not."""
    import datetime as dt

    from core import dashboard_schedule
    from core.portal_roles import can
    from store import email_store, schedule_store, tile_alert_store

    account_id, user_id = str(user["account_id"]), int(user["id"])
    item = next((n for n in notifications(user, "en") if n["kind"] == kind and n["id"] == int(item_id)), None)
    if item is None:
        return "ui.profile.n.error.not_found"
    if kind == FOLLOW:
        store.set_follow_status(int(item_id), user_id, account_id, "active" if on else "paused")
    elif kind == SCHEDULE:
        if on:
            if not can(user, "reports"):
                return "ui.profile.n.error.role"
            dashboard_schedule.resume(schedule_store.get_schedule(int(item_id)))
        else:
            schedule_store.set_schedule_active(int(item_id), False)
    elif kind == ALERT:
        if on:
            tile_alert_store.set_alert_active(
                int(item_id), True, next_check_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        else:
            tile_alert_store.set_alert_active(int(item_id), False)
    else:
        if not user.get("email"):
            return "ui.profile.n.error.not_found"
        opt_kind = "schedule" if kind == SCHEDULE_FOR_ME else "alert"
        if on:
            email_store.opt_back_in(str(user["email"]), opt_kind, str(int(item_id)))
        else:
            email_store.opt_out(str(user["email"]), opt_kind, str(int(item_id)))
    return None
