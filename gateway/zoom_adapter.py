"""
gateway/zoom_adapter.py

Zoom Team Chat adapter.
Wraps all Zoom-specific webhook verification, payload parsing,
and message/file sending behind the PlatformAdapter interface.
"""

from __future__ import annotations
import hashlib
import hmac
import json
import logging
import tempfile
import os
from typing import Optional

import httpx

from gateway.base import PlatformAdapter, PlatformEvent

log = logging.getLogger("gateway.zoom")

_TOKEN_URL   = "https://zoom.us/oauth/token"
_MESSAGE_URL = "https://api.zoom.us/v2/im/chat/messages"
_FILE_URL    = "https://file.zoom.us/v2/im/chat/messages/files"


class ZoomAdapter(PlatformAdapter):

    platform_type = "zoom"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self._client_id     = credentials["client_id"]
        self._client_secret = credentials["client_secret"]
        self._bot_jid       = credentials["bot_jid"]
        self._webhook_secret = credentials["webhook_secret"]

    # ── Signature verification ────────────────────────────────────────────────

    async def verify_request(self, body: bytes, headers: dict) -> bool:
        timestamp = headers.get("x-zm-request-timestamp", "")
        signature = headers.get("x-zm-signature", "")
        msg      = f"v0:{timestamp}:{body.decode()}"
        expected = "v0=" + hmac.new(
            self._webhook_secret.encode(), msg.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    # ── Parse incoming payload ────────────────────────────────────────────────

    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        payload = json.loads(body)
        event_type = payload.get("event", "")

        # URL validation challenge — handled by the webhook router directly
        if event_type == "endpoint.url_validation":
            return None

        if event_type != "bot_notification":
            return None

        p = payload.get("payload", {})
        cmd = p.get("cmd", "").strip()
        if not cmd:
            return None

        return PlatformEvent(
            account_id = p.get("accountId", ""),
            user_id    = p.get("userId", ""),
            channel_id = p.get("toJid", ""),
            text       = cmd,
            platform   = "zoom",
            raw        = payload,
        )

    def handle_challenge(self, body: bytes) -> Optional[dict]:
        """
        Called by the webhook router for endpoint.url_validation events.
        Returns the JSON response dict Zoom expects, or None.
        """
        payload = json.loads(body)
        if payload.get("event") == "endpoint.url_validation":
            plain = payload["payload"]["plainToken"]
            enc   = hmac.new(
                self._webhook_secret.encode(), plain.encode(), hashlib.sha256
            ).hexdigest()
            return {"plainToken": plain, "encryptedToken": enc}
        return None

    # ── OAuth token ───────────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        _TOKEN_URL,
                        params={"grant_type": "client_credentials"},
                        auth=(self._client_id, self._client_secret),
                        timeout=10,
                    )
                    resp.raise_for_status()
                    return resp.json()["access_token"]
            except httpx.HTTPStatusError as e:
                if attempt == 0 and e.response.status_code >= 500:
                    log.warning("Zoom token fetch retrying: %s", e)
                    continue
                log.error("Zoom token fetch failed: %s", e.response.text)
                raise
        raise RuntimeError("Zoom token fetch failed after retries")

    # ── Send message ──────────────────────────────────────────────────────────

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        token = await self._get_token()
        payload = {
            "robot_jid":           self._bot_jid,
            "to_jid":              event.channel_id,
            "user_jid":            event.channel_id,
            "account_id":          event.account_id,
            "is_markdown_support": True,
            "content": {
                "head": {"text": "Query Bot"},
                "body": [{"type": "message", "text": text}],
            },
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _MESSAGE_URL,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                log.error("Zoom send_message %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        log.info("Zoom message sent to %s", event.channel_id)

    # ── Upload file ───────────────────────────────────────────────────────────

    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        token = await self._get_token()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            async with httpx.AsyncClient() as client:
                with open(tmp_path, "rb") as f:
                    resp = await client.post(
                        _FILE_URL,
                        headers={"Authorization": f"Bearer {token}"},
                        data={
                            "robot_jid":  self._bot_jid,
                            "to_jid":     event.channel_id,
                            "account_id": event.account_id,
                        },
                        files={"file": (filename, f, mime_type)},
                        timeout=20,
                    )
                resp.raise_for_status()
                log.info("Zoom file uploaded (%d bytes)", len(file_bytes))
        except Exception as e:
            log.warning("Zoom file upload failed: %s", e)
        finally:
            os.unlink(tmp_path)
