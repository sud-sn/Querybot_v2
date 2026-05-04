"""
gateway/slack_adapter.py

Slack adapter using Slack's Events API.

Incoming:  Slack sends signed JSON payloads to your endpoint
Outgoing:  Slack Web API (chat.postMessage, files.upload)
Auth:      HMAC-SHA256 over request body using Signing Secret

Credentials required:
  bot_token      — xoxb-... Bot OAuth token (for sending messages)
  signing_secret — for verifying incoming requests
  app_id         — Slack App ID
"""

from __future__ import annotations
import hashlib
import hmac
import json
import logging
import time
from typing import Optional

import httpx

from gateway.base import PlatformAdapter, PlatformEvent

log = logging.getLogger("gateway.slack")

_SLACK_API    = "https://slack.com/api"
_POST_MESSAGE = f"{_SLACK_API}/chat.postMessage"
_FILE_UPLOAD  = f"{_SLACK_API}/files.getUploadURLExternal"
_FILE_COMPLETE = f"{_SLACK_API}/files.completeUploadExternal"


class SlackAdapter(PlatformAdapter):

    platform_type = "slack"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self._bot_token     = credentials["bot_token"]
        self._signing_secret = credentials["signing_secret"]
        self._app_id        = credentials.get("app_id", "")

    # ── Signature verification ────────────────────────────────────────────────

    async def verify_request(self, body: bytes, headers: dict) -> bool:
        timestamp = headers.get("x-slack-request-timestamp", "")
        signature = headers.get("x-slack-signature", "")

        # Reject requests older than 5 minutes (replay attack prevention)
        try:
            if abs(time.time() - float(timestamp)) > 300:
                log.warning("Slack: request timestamp too old")
                return False
        except (ValueError, TypeError):
            return False

        base_str = f"v0:{timestamp}:{body.decode()}"
        expected = "v0=" + hmac.new(
            self._signing_secret.encode(),
            base_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    # ── Parse incoming event ──────────────────────────────────────────────────

    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        try:
            payload = json.loads(body)
        except Exception:
            return None

        # URL verification challenge from Slack
        if payload.get("type") == "url_verification":
            return None  # handled by router directly

        if payload.get("type") != "event_callback":
            return None

        event = payload.get("event", {})

        # Only handle plain messages (not bot messages, edits, etc.)
        if event.get("type") != "message":
            return None
        if event.get("bot_id") or event.get("subtype"):
            return None

        text = (event.get("text") or "").strip()
        # Strip bot mention if present  <@UBOT123> query text
        if text.startswith("<@"):
            text = text.split(">", 1)[-1].strip()

        if not text:
            return None

        return PlatformEvent(
            account_id = payload.get("team_id", ""),
            user_id    = event.get("user", ""),
            channel_id = event.get("channel", ""),
            text       = text,
            platform   = "slack",
            raw        = payload,
        )

    def handle_challenge(self, body: bytes) -> Optional[dict]:
        """Return the Slack URL verification challenge response if applicable."""
        try:
            payload = json.loads(body)
        except Exception:
            return None
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge")}
        return None

    # ── Send message ──────────────────────────────────────────────────────────

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _POST_MESSAGE,
                headers={
                    "Authorization": f"Bearer {self._bot_token}",
                    "Content-Type":  "application/json",
                },
                json={
                    "channel":  event.channel_id,
                    "text":     text,
                    "mrkdwn":   True,   # enable Slack markdown
                },
                timeout=10,
            )
            data = resp.json()
            if not data.get("ok"):
                log.error("Slack send_message error: %s", data.get("error"))
            else:
                log.info("Slack message sent to %s", event.channel_id)

    # ── Upload file ───────────────────────────────────────────────────────────

    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        """
        Upload using Slack's two-step external upload API (recommended for >1MB).
        Step 1: getUploadURLExternal → get an upload URL
        Step 2: PUT the file bytes
        Step 3: completeUploadExternal → publish to channel
        """
        headers = {
            "Authorization": f"Bearer {self._bot_token}",
        }
        async with httpx.AsyncClient() as client:
            # Step 1: request upload URL
            r1 = await client.get(
                _FILE_UPLOAD,
                headers=headers,
                params={
                    "filename": filename,
                    "length":   len(file_bytes),
                },
                timeout=10,
            )
            d1 = r1.json()
            if not d1.get("ok"):
                log.warning("Slack upload URL failed: %s", d1.get("error"))
                return

            upload_url = d1["upload_url"]
            file_id    = d1["file_id"]

            # Step 2: PUT the bytes
            r2 = await client.put(
                upload_url,
                content=file_bytes,
                headers={"Content-Type": mime_type},
                timeout=30,
            )
            if r2.status_code not in (200, 204):
                log.warning("Slack file PUT failed: %s", r2.status_code)
                return

            # Step 3: complete the upload and post to channel
            r3 = await client.post(
                _FILE_COMPLETE,
                headers=headers,
                json={
                    "files":           [{"id": file_id, "title": filename}],
                    "channel_id":      event.channel_id,
                },
                timeout=10,
            )
            d3 = r3.json()
            if not d3.get("ok"):
                log.warning("Slack upload complete failed: %s", d3.get("error"))
            else:
                log.info("Slack chart uploaded (%d bytes)", len(file_bytes))
