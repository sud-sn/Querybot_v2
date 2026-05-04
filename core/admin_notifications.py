"""Realtime admin notifications.

The hub is intentionally small and dependency-free: it gives instant updates
for the normal single-VM deployment, while the admin UI also polls the summary
endpoint as a reconciliation fallback after reconnects or process restarts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

import store

log = logging.getLogger("querybot.admin.notifications")


class AdminNotificationHub:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections)
        if not targets:
            return

        stale: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                log.debug("Dropping stale admin notification socket: %s", exc)
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections.discard(websocket)


admin_notification_hub = AdminNotificationHub()


def semantic_feedback_summary() -> dict[str, Any]:
    return store.semantic_feedback_pending_summary()


async def notify_semantic_feedback_changed(
    *,
    account_id: str,
    action: str,
    feedback_id: int | None = None,
) -> None:
    summary = semantic_feedback_summary()
    client = store.get_client(account_id) or {}
    await admin_notification_hub.broadcast({
        "type": "semantic_feedback_pending",
        "action": action,
        "account_id": account_id,
        "client_name": client.get("client_name") or account_id,
        "feedback_id": feedback_id,
        "summary": summary,
    })


async def notify_kb_build_changed(
    *,
    account_id: str,
    status: str,
    progress: dict[str, Any] | None = None,
) -> None:
    client = store.get_client(account_id) or {}
    await admin_notification_hub.broadcast({
        "type": "kb_build_progress",
        "account_id": account_id,
        "client_name": client.get("client_name") or account_id,
        "status": status,
        "progress": progress or {},
    })
