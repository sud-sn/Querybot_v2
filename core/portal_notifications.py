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
        # The server's own loop, where the sockets live: a scheduled job runs in a worker thread with a loop
        # of its own, and a socket written to from another loop fails (and was then dropped as stale).
        self._loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket, *, account_id: str, user_id: int) -> None:
        self._loop = asyncio.get_running_loop()
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

    async def broadcast_to_user(self, user_id: int, payload: dict[str, Any]) -> int:
        """Send to every page the user has open. How many received it."""
        async with self._lock:
            targets = list(self._by_user.get(int(user_id), set()))
        return await self._broadcast(targets, payload)

    async def deliver(self, user_id: int, payload: dict[str, Any]) -> int:
        """``broadcast_to_user`` from any loop: on the server's own, directly; from another (a worker
        thread's), handed to the server's loop and awaited. 0 when no page is open (or no loop yet)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return 0
        if asyncio.get_running_loop() is loop:
            return await self.broadcast_to_user(user_id, payload)
        future = asyncio.run_coroutine_threadsafe(self.broadcast_to_user(user_id, payload), loop)
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=10)

    def deliver_from_thread(self, user_id: int, payload: dict[str, Any]) -> int:
        """``deliver`` for code with no loop of its own (a scheduled job's worker thread)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return 0
        try:
            return asyncio.run_coroutine_threadsafe(self.broadcast_to_user(user_id, payload), loop).result(timeout=10)
        except Exception as exc:  # noqa: BLE001 - a notice is kept either way; the live copy is extra
            log.warning("Live notice to user %s was not sent: %s", user_id, exc)
            return 0

    async def broadcast_to_account(self, account_id: str, payload: dict[str, Any]) -> int:
        async with self._lock:
            targets = list(self._by_account.get(str(account_id), set()))
        return await self._broadcast(targets, payload)

    async def _broadcast(self, targets: list[WebSocket], payload: dict[str, Any]) -> int:
        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                log.debug("Dropping stale portal notification socket: %s", exc)
                stale.append(websocket)
        for websocket in stale:
            await self.disconnect(websocket)
        return len(targets) - len(stale)


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
    suggested_synonyms: str = "",
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
        "suggested_synonyms": suggested_synonyms,
        "admin_note": admin_note,
    }
    if portal_user_id:
        await portal_notification_hub.broadcast_to_user(int(portal_user_id), payload)
    else:
        await portal_notification_hub.broadcast_to_account(account_id, payload)
