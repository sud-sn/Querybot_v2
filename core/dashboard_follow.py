"""Dashboard follows delivered (store/dashboard_store.py, dashboard_subscription).

A reader who follows a dashboard daily, weekly or monthly is told at that pace: a notice kept on their
Notifications page and shown at once on a page of theirs that is open (core/notices.py). It names the
dashboard's headline numbers -- its first number tiles, drawn as this reader is allowed to see them -- in their
own language, with a link. Someone who can no longer open the dashboard stops following it and is sent nothing.
Run every minute by core/notification_scheduler.py; the first update comes one period after following.
"""

from __future__ import annotations

import datetime as dt
import logging

import store

log = logging.getLogger("querybot.dashboard_follow")

PACE = {"daily": dt.timedelta(days=1), "weekly": dt.timedelta(days=7), "monthly": dt.timedelta(days=30)}

# The number tiles an update names, in the dashboard's order.
HEADLINES = 3


def _when(text) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(text or "")[:19].replace("T", " "))
    except ValueError:
        return None


def run_due_follows(now: dt.datetime | None = None) -> int:
    """Send every update that is due. How many were sent."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    sent = 0
    for follow in store.list_active_follows():
        previous = follow.get("last_sent_at")
        since = _when(previous or follow.get("created_at"))
        if since is None or now - since < PACE.get(str(follow.get("cadence")), PACE["daily"]):
            continue
        if not store.claim_follow(int(follow["id"]), previous, stamp):
            continue                                    # another scheduler sent it
        try:
            if _deliver(follow):
                sent += 1
        except Exception as exc:  # noqa: BLE001 - one follow failing must not stop the others
            store.release_follow(int(follow["id"]), stamp, previous)
            log.warning("Dashboard follow %s: the update was not sent and is tried again: %s", follow.get("id"), exc)
    return sent


def _deliver(follow: dict) -> bool:
    from core.i18n import t
    from core.notices import tell

    account_id, user_id = str(follow["account_id"]), int(follow["user_id"])
    dashboard_id = int(follow["dashboard_id"])
    user = store.get_user(user_id)
    board = store.get_dashboard_for_view(dashboard_id, user_id, account_id) if user else None
    if not board:
        store.unsubscribe_dashboard(dashboard_id, user_id, account_id)
        log.info("Dashboard follow %s ended: user %s can no longer open dashboard %s",
                 follow.get("id"), user_id, dashboard_id)
        return False
    lang = str(follow.get("lang") or "en")
    numbers = _headlines(board, user, lang)
    cadence = str(follow.get("cadence") or "daily")
    title = t("notice.follow.title", lang=lang, name=board.get("name") or "",
              cadence=t(f"notice.follow.cadence.{cadence}", lang=lang))
    body = " · ".join(f"{name}: {value}" for name, value in numbers) or t("notice.follow.no_numbers", lang=lang)
    tell(account_id, user_id, "dashboard_follow", title, body, f"/portal/dashboard?dashboard_id={dashboard_id}")
    return True


def _headlines(board: dict, user: dict, lang: str) -> list[tuple[str, str]]:
    """The dashboard's first number tiles, drawn now as ``user`` sees them: (name, value). A tile that cannot be
    drawn for them is left out."""
    from core.i18n import activate_language, deactivate_language
    from portal.routes import _refresh_chart

    charts = store.list_dashboard_charts_for_view(int(board["id"]), int(user["id"]), str(user["account_id"]))
    first = next((c for c in charts if c.get("db_config_id")), None)
    fallback = store.get_db_config(first["db_config_id"]) if first else None
    found: list[tuple[str, str]] = []
    token = activate_language(lang)                     # numbers written as the reader writes them
    try:
        for chart in [c for c in charts if str(c.get("chart_type") or "") == "kpi"][:HEADLINES]:
            db_cfg = store.get_db_config(chart["db_config_id"]) if chart.get("db_config_id") else fallback
            try:
                drawn = _refresh_chart(chart, db_cfg, user)
            except Exception as exc:  # noqa: BLE001 - the update goes out with the numbers that could be drawn
                log.warning("Dashboard %s tile %s not drawn for a follow: %s", board.get("id"), chart.get("id"), exc)
                continue
            value = drawn.get("kpi_display") or (drawn.get("kpi") or {}).get("value")
            if drawn.get("error") or value in (None, ""):
                continue
            found.append((str(drawn.get("title") or chart.get("title") or ""), str(value)))
    finally:
        deactivate_language(token)
    return found
