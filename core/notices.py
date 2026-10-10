"""Tell a portal reader something: kept as a notice (store/notice_store.py) and, when a page of theirs is
open, shown at once (core/portal_notifications.py).

Kept first, so the reader is told whether or not they are online: the Notifications page lists it. The live
copy is extra; a failure to send it never loses the notice.
"""

from __future__ import annotations

import logging

import store

log = logging.getLogger("querybot.notices")


def _payload(notice_id: int, account_id: str, user_id: int, kind: str, title: str, body: str, link: str) -> dict:
    return {"type": "notice", "id": notice_id, "kind": kind, "title": title, "body": body, "link": link,
            "unread": store.unread_notice_count(account_id, user_id)}


def tell(account_id: str, user_id: int, kind: str, title: str, body: str = "", link: str = "") -> int:
    """From code with no event loop of its own (a scheduled job). The notice's id."""
    notice_id = store.add_notice(account_id, user_id, kind, title, body, link)
    try:
        from core.portal_notifications import portal_notification_hub

        portal_notification_hub.deliver_from_thread(
            user_id, _payload(notice_id, account_id, user_id, kind, title, body, link))
    except Exception as exc:  # noqa: BLE001 - kept; the live copy is extra
        log.warning("Notice %s kept but not shown live: %s", notice_id, exc)
    return notice_id


async def tell_async(account_id: str, user_id: int, kind: str, title: str, body: str = "", link: str = "") -> int:
    """From a route (the server's own loop). The notice's id."""
    notice_id = store.add_notice(account_id, user_id, kind, title, body, link)
    try:
        from core.portal_notifications import portal_notification_hub

        await portal_notification_hub.deliver(user_id, _payload(notice_id, account_id, user_id, kind, title, body,
                                                                link))
    except Exception as exc:  # noqa: BLE001 - kept; the live copy is extra
        log.warning("Notice %s kept but not shown live: %s", notice_id, exc)
    return notice_id
