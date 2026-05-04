"""Realtime notifications for authenticated portal users."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("querybot.portal.notifications")


class PortalNotificationHub:
    def __init__(self) -> None:
        self._by_user: dict[int, set[WebSocket]] = defaultdict(set)
        self._by_account: dict[str, set[WebSocket]] = defaultdict(set)
        self._meta: dict[WebSocket, tuple[str, int]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, *, account_id: str, user_id: int) -> None:
        await websocket.accept()
        async with self._lock:
            self._by_user[int(user_id)].add(websocket)
            self._by_account[str(account_id)].add(websocket)
            self._meta[websocket] = (str(account_id), int(user_id))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            meta = self._meta.pop(websocket, None)
            if not meta:
                return
            account_id, user_id = meta
            self._by_user.get(user_id, set()).discard(websocket)
            self._by_account.get(account_id, set()).discard(websocket)

    async def broadcast_to_user(self, user_id: int, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._by_user.get(int(user_id), set()))
        await self._broadcast(targets, payload)

    async def broadcast_to_account(self, account_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._by_account.get(str(account_id), set()))
        await self._broadcast(targets, payload)

    async def _broadcast(self, targets: list[WebSocket], payload: dict[str, Any]) -> None:
        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                log.debug("Dropping stale portal notification socket: %s", exc)
                stale.append(websocket)
        for websocket in stale:
            await self.disconnect(websocket)


portal_notification_hub = PortalNotificationHub()


async def notify_portal_semantic_feedback_changed(
    *,
    account_id: str,
    portal_user_id: int | None,
    feedback_id: int,
    status: str,
    table_fqn: str,
    column_name: str,
    suggested_meaning: str = "",
    suggested_use_case: str = "",
    admin_note: str = "",
) -> None:
    payload = {
        "type": "semantic_feedback_reviewed",
        "account_id": account_id,
        "feedback_id": feedback_id,
        "status": status,
        "table_fqn": table_fqn,
        "column_name": column_name,
        "suggested_meaning": suggested_meaning,
        "suggested_use_case": suggested_use_case,
        "admin_note": admin_note,
    }
    if portal_user_id:
        await portal_notification_hub.broadcast_to_user(int(portal_user_id), payload)
    else:
        await portal_notification_hub.broadcast_to_account(account_id, payload)
