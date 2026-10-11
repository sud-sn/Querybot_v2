"""Realtime admin notifications.

The hub is intentionally small and dependency-free: it gives instant updates
for the normal single-VM deployment, while the admin UI also polls the summary
endpoint as a reconciliation fallback after reconnects or process restarts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from fastapi import WebSocket

import store

log = logging.getLogger("querybot.admin.notifications")


Opens = Optional[Callable[[str], bool]]


def scoped(payload: dict[str, Any], opens: Opens) -> dict[str, Any] | None:
    """What one admin is told: nothing about a workspace they do not look after (``opens``: None for every one).

    A notice about another workspace is not sent at all, and a summary across workspaces keeps only theirs."""
    if opens is None:
        return payload
    account_id = payload.get("account_id")
    if account_id and not opens(str(account_id)):
        return None
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("clients"), list):
        payload = {**payload, "summary": scoped_summary(summary, opens)}
    return payload


def scoped_summary(summary: dict[str, Any], opens: Opens) -> dict[str, Any]:
    if opens is None:
        return summary
    clients = [c for c in summary.get("clients") or [] if opens(str(c.get("account_id") or ""))]
    return {**summary, "clients": clients, "total": sum(int(c.get("pending") or 0) for c in clients)}


class AdminNotificationHub:
    def __init__(self) -> None:
        # Each socket with what its admin may hear about (admin/access.py): None for a super admin.
        self._connections: dict[WebSocket, Opens] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, opens: Opens = None) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = opens

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(websocket, None)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections.items())
        if not targets:
            return

        stale: list[WebSocket] = []
        for websocket, opens in targets:
            told = scoped(payload, opens)
            if told is None:
                continue
            try:
                await websocket.send_json(told)
            except Exception as exc:
                log.debug("Dropping stale admin notification socket: %s", exc)
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections.pop(websocket, None)


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


async def notify_core2_request_changed(*, account_id: str, action: str, request_id: int | None = None) -> None:
    """A reader's request was sent or decided: the admin's Requests count follows."""
    client = store.get_client(account_id) or {}
    await admin_notification_hub.broadcast({
        "type": "core2_request",
        "action": action,
        "account_id": account_id,
        "client_name": client.get("client_name") or account_id,
        "request_id": request_id,
        "waiting": store.count_core2_requests(account_id).get("waiting", 0),
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


async def notify_graph_ready(
    *,
    account_id: str,
    summary: dict[str, Any] | None = None,
) -> None:
    """Broadcast after the post-KB-build entity graph sync so the admin sees
    a 'semantic model ready for review' toast with suggestion counts."""
    client = store.get_client(account_id) or {}
    await admin_notification_hub.broadcast({
        "type": "graph_ready",
        "account_id": account_id,
        "client_name": client.get("client_name") or account_id,
        "summary": summary or {},
    })
