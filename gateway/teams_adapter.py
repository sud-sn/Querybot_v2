"""
gateway/teams_adapter.py

Microsoft Teams adapter via the Bot Framework.

Incoming:  Teams sends Activity JSON to your endpoint
Outgoing:  POST to the Activity's serviceUrl with a reply Activity
Auth:      JWT token from Microsoft identity platform, verified via HMAC on App Password
           OR via the Bot Framework connector's JWT public keys

Feature parity with the portal's internal chat UI:
  • send_message       — plain text with markdown
  • upload_file        — chart PNG as inline base64 in an Adaptive Card image
  • send_status        — typing indicator for live progress feedback
  • send_clarification_prompt — Adaptive Card with Action.Submit buttons for
                                each clarification option (no free-text reply needed)
  • parse_event        — handles BOTH text messages AND Adaptive Card submits,
                         so tapping a clarification button feeds back into the
                         same dispatch path as a typed reply

Credentials required:
  app_id       — Azure Bot App ID (also called MicrosoftAppId)
  app_password — Azure Bot App Password (client secret)
  tenant_id    — Azure AD tenant ("common" for multi-tenant bots)
"""

from __future__ import annotations
import json
import logging
from typing import Optional

import httpx

from gateway.base import PlatformAdapter, PlatformEvent

log = logging.getLogger("gateway.teams")

_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
_SCOPE     = "https://api.botframework.com/.default"


class TeamsAdapter(PlatformAdapter):

    platform_type = "teams"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self._app_id       = credentials["app_id"]
        self._app_password = credentials["app_password"]
        self._tenant_id    = credentials.get("tenant_id", "common")

    # ── Signature / auth verification ─────────────────────────────────────────
    # Teams uses JWT Bearer tokens on incoming requests.
    # Full verification requires fetching Microsoft's JWKS and validating the
    # token signature, audience, and issuer.
    # For initial integration we validate the token is present and structurally
    # valid; full JWKS verification is a hardening step.

    async def verify_request(self, body: bytes, headers: dict) -> bool:
        auth_header = headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            log.warning("Teams: missing Bearer token")
            return False
        # Structural check — full JWT validation recommended for production
        parts = auth_header[7:].split(".")
        if len(parts) != 3:
            log.warning("Teams: malformed JWT")
            return False
        return True

    # ── Parse incoming Activity ───────────────────────────────────────────────

    def parse_event(self, body: bytes, headers: dict) -> Optional[PlatformEvent]:
        try:
            activity = json.loads(body)
        except Exception:
            return None

        # Only handle message activities (ignore typing, conversationUpdate, etc.)
        if activity.get("type") != "message":
            return None

        # Two shapes of "message" arrive from Teams:
        #   1. A typed user message → activity.text is the text
        #   2. An Adaptive Card Action.Submit → activity.value carries the submitted
        #      data (e.g. {"option_id": "opt_1", "label": "late pickups"}) and
        #      activity.text is empty
        text = (activity.get("text") or "").strip()
        value = activity.get("value")

        if not text and isinstance(value, dict):
            # Card submit path. The clarification card we send below stashes the
            # option's human-readable label under "label" and the option_id
            # under "option_id". The core dispatcher matches on label text, so
            # sending back the label is what the resolver expects.
            label = (value.get("label") or value.get("option_id") or "").strip()
            if not label:
                return None
            text = label

        # Teams sometimes prefixes messages with a bot mention like "@BotName "
        # Strip the mention if present
        if "<at>" in text:
            import re
            text = re.sub(r"<at>[^<]*</at>", "", text).strip()

        if not text:
            return None

        channel_data = activity.get("channelData", {})
        tenant_id    = (
            channel_data.get("tenant", {}).get("id")
            or activity.get("conversation", {}).get("tenantId", "")
        )
        account_id = tenant_id

        return PlatformEvent(
            account_id = account_id,
            user_id    = activity.get("from", {}).get("id", ""),
            channel_id = json.dumps({
                "service_url":    activity.get("serviceUrl", ""),
                "conversation_id": activity.get("conversation", {}).get("id", ""),
                "activity_id":    activity.get("id", ""),
            }),
            text     = text,
            platform = "teams",
            raw      = activity,
        )

    # ── Get Bot Framework token ───────────────────────────────────────────────

    async def _get_token(self) -> str:
        url = _TOKEN_URL.format(tenant=self._tenant_id)
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, data={
                "grant_type":    "client_credentials",
                "client_id":     self._app_id,
                "client_secret": self._app_password,
                "scope":         _SCOPE,
            }, timeout=10)
            resp.raise_for_status()
            return resp.json()["access_token"]

    # ── Send message ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_result_card(text: str) -> dict | None:
        """
        Convert a query result message (table or single-value) to an Adaptive Card.
        Returns None for plain messages (errors, help, status, clarification prompts).
        """
        import re

        lines      = text.split("\n")
        has_table  = any("|" in l for l in lines)
        has_single = "━" in text  # single-value box uses ━━━ separator

        if not has_table and not has_single:
            return None

        body = []

        def _clean(s: str) -> str:
            """Strip markdown bold/italic markers."""
            s = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", s)
            s = re.sub(r"_([^_]+)_", r"\1", s)
            return s.strip()

        # ── Single-value result ────────────────────────────────────────────────
        if has_single and not has_table:
            header_q  = ""
            col_label = ""
            value_str = ""
            duration  = ""

            for line in lines:
                s = line.strip()
                if not s or re.match(r"^[━\-─\s]+$", s):
                    continue
                m_dur  = re.match(r"^_(.+)_$", s)
                m_bold = re.match(r"^\*\*?(.+?)\*\*?$", s)
                if m_dur:
                    duration = m_dur.group(1)
                elif m_bold:
                    val = m_bold.group(1)
                    if not header_q:
                        header_q = val
                    elif not value_str:
                        value_str = val
                elif not col_label and "Confidence" not in s and not s.startswith("-"):
                    col_label = s

            if header_q:
                body.append({
                    "type": "TextBlock", "text": header_q,
                    "wrap": True, "weight": "Bolder", "size": "Medium",
                })
            body.append({
                "type": "Container",
                "style": "emphasis",
                "spacing": "Medium",
                "items": [
                    {
                        "type": "TextBlock",
                        "text": col_label.replace("_", " ").title() if col_label else "Result",
                        "size": "Small", "color": "Accent", "weight": "Bolder",
                    },
                    {
                        "type": "TextBlock",
                        "text": value_str or col_label,
                        "size": "ExtraLarge", "weight": "Bolder", "spacing": "None",
                    },
                ],
            })
            if duration:
                body.append({
                    "type": "TextBlock", "text": duration,
                    "size": "Small", "isSubtle": True, "spacing": "Small",
                })

        # ── Multi-row table result ────────────────────────────────────────────
        elif has_table:
            pre_lines, table_lines, post_lines = [], [], []
            section = "pre"
            for line in lines:
                if section == "pre":
                    if "|" in line:
                        section = "table"
                        table_lines.append(line)
                    else:
                        pre_lines.append(line)
                elif section == "table":
                    if "|" in line or re.match(r"^[\-+\s]+$", line):
                        table_lines.append(line)
                    else:
                        section = "post"
                        post_lines.append(line)
                else:
                    post_lines.append(line)

            # Render pre-table header lines
            for line in pre_lines:
                s = line.strip()
                if not s or re.match(r"^[-─━\s]+$", s):
                    continue
                clean = _clean(s)
                is_bold = s.startswith("*")
                is_rows = bool(re.match(r"^\d+\s+rows?$", clean, re.I))
                body.append({
                    "type": "TextBlock", "text": clean, "wrap": True,
                    "weight": "Bolder" if (is_bold or is_rows) else "Default",
                    "color": "Accent" if is_rows else "Default",
                    "spacing": "Small",
                })

            # Parse table headers + data rows
            headers     = []
            parsed_rows = []
            for line in table_lines:
                if not line.strip():
                    continue
                cells = [c.strip() for c in line.split("|")]
                # remove empty outer items from split
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                if not cells:
                    continue
                # separator line?
                if all(re.match(r"^[-+]+$", c) for c in cells):
                    continue
                if not headers:
                    headers = cells
                else:
                    parsed_rows.append(cells)

            if headers:
                num_cols = len(headers)

                def _col(text_val, weight="Default", color="Default", is_num=False, is_first=False):
                    return {
                        "type":  "Column",
                        "width": "auto" if (is_num or not is_first) else "stretch",
                        "items": [{
                            "type":    "TextBlock",
                            "text":    str(text_val),
                            "weight":  weight,
                            "color":   color,
                            "size":    "Small",
                            "wrap":    not is_num,
                            "spacing": "None",
                        }],
                    }

                # Header ColumnSet
                body.append({
                    "type": "ColumnSet",
                    "separator": True,
                    "spacing": "Medium",
                    "columns": (
                        [_col("#", weight="Bolder", color="Accent", is_num=True)]
                        + [_col(h, weight="Bolder", is_first=(i == 0))
                           for i, h in enumerate(headers)]
                    ),
                })

                # Data rows
                for idx, row in enumerate(parsed_rows, 1):
                    cells = [row[i] if i < len(row) else "" for i in range(num_cols)]
                    body.append({
                        "type": "ColumnSet",
                        "spacing": "Small",
                        "columns": (
                            [_col(idx, color="Accent", is_num=True)]
                            + [_col(c, is_first=(i == 0)) for i, c in enumerate(cells)]
                        ),
                    })

            # Post-table: confidence summary + duration
            conf_summary = ""
            duration     = ""
            for line in post_lines:
                s = line.strip()
                if not s:
                    continue
                m_dur = re.match(r"^_(.+)_$", s)
                if m_dur:
                    duration = m_dur.group(1)
                elif "Confidence:" in s and not conf_summary:
                    m = re.match(r"(Confidence:\s*[\w ]+\(\d+/\d+\))", s)
                    conf_summary = m.group(1) if m else ""

            footer = " · ".join(x for x in [conf_summary, duration] if x)
            if footer:
                body.append({
                    "type": "TextBlock", "text": footer,
                    "size": "Small", "isSubtle": True, "wrap": True, "spacing": "Small",
                })

        if not body:
            return None

        return {
            "type":    "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.3",
            "body":    body,
        }

    async def send_message(self, event: PlatformEvent, text: str) -> None:
        channel_info    = json.loads(event.channel_id)
        service_url     = channel_info["service_url"]
        conversation_id = channel_info["conversation_id"]
        reply_url = (
            f"{service_url.rstrip('/')}/v3/conversations/"
            f"{conversation_id}/activities"
        )
        token = await self._get_token()

        card = self._build_result_card(text)
        if card:
            activity = {
                "type": "message",
                "attachments": [{
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content":     card,
                }],
            }
        else:
            # Plain messages (errors, help, status, clarification) — send as markdown
            activity = {
                "type":       "message",
                "text":       text,
                "textFormat": "markdown",
            }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                reply_url,
                headers={"Authorization": f"Bearer {token}"},
                json=activity,
                timeout=15,
            )
            if resp.status_code not in (200, 201):
                log.error("Teams send_message %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        log.info("Teams message sent to conversation %s", conversation_id)

    # ── Upload file ───────────────────────────────────────────────────────────

    async def upload_file(
        self,
        event: PlatformEvent,
        file_bytes: bytes,
        filename: str,
        mime_type: str = "image/png",
    ) -> None:
        """
        Teams bot file upload requires either:
          a) Inline base64 image in an Adaptive Card (works without extra permissions)
          b) Bot Framework file consent flow (requires Files.ReadWrite scope)

        We use option (a) — embed the chart as a base64 image in a card.
        This works without any extra Azure AD permissions.
        """
        import base64
        channel_info    = json.loads(event.channel_id)
        service_url     = channel_info["service_url"]
        conversation_id = channel_info["conversation_id"]
        reply_url = (
            f"{service_url.rstrip('/')}/v3/conversations/"
            f"{conversation_id}/activities"
        )
        token = await self._get_token()
        b64 = base64.b64encode(file_bytes).decode()

        # Adaptive Card with inline image
        card = {
            "type":    "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.3",
            "body": [{
                "type":  "Image",
                "url":   f"data:{mime_type};base64,{b64}",
                "altText": filename,
                "size":  "stretch",
            }],
        }
        activity = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content":     card,
            }],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                reply_url,
                headers={"Authorization": f"Bearer {token}"},
                json=activity,
                timeout=20,
            )
            if resp.status_code not in (200, 201):
                log.warning("Teams file upload %s: %s", resp.status_code, resp.text)
        log.info("Teams chart sent as Adaptive Card (%d bytes)", len(file_bytes))

    # ── send_status ───────────────────────────────────────────────────────────
    # Matches web_adapter.send_status. Teams doesn't have a true "progress"
    # stream, but it DOES have a typing indicator — which is the right
    # affordance here: it tells the user the bot is working without cluttering
    # the chat with transient status text that would stay in the transcript.
    # We send `typing` on every stage call; Teams auto-clears it when the next
    # message arrives.

    async def send_status(
        self,
        event: PlatformEvent,
        stage: str,
        label: str,
        detail: str = "",
    ) -> None:
        channel_info    = json.loads(event.channel_id)
        service_url     = channel_info["service_url"]
        conversation_id = channel_info["conversation_id"]
        reply_url = (
            f"{service_url.rstrip('/')}/v3/conversations/"
            f"{conversation_id}/activities"
        )
        try:
            token = await self._get_token()
        except Exception as e:
            log.debug("Teams send_status: token fetch failed: %s", e)
            return
        activity = {"type": "typing"}
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    reply_url,
                    headers={"Authorization": f"Bearer {token}"},
                    json=activity,
                    timeout=5,
                )
        except Exception as e:
            # Typing indicator is best-effort; never block the query on it.
            log.debug("Teams typing indicator failed for stage=%s: %s", stage, e)

    # ── send_clarification_prompt ─────────────────────────────────────────────
    # Matches web_adapter.send_clarification_prompt. Renders an Adaptive Card
    # with one Action.Submit button per clarification option. When the user
    # taps a button, Teams posts a `message` activity with the `value` field
    # populated (and no `text`) — parse_event handles that shape by reading
    # value.label and feeding it back into the same dispatcher path as a
    # typed reply. So clarification flow is consistent across Teams buttons,
    # Teams typed replies, and portal clicks.

    async def send_clarification_prompt(
        self,
        event: PlatformEvent,
        question: str,
        options: list[dict],
        pending_id: str | None = None,
    ) -> None:
        # Build one Action.Submit per option. The `data` payload is what comes
        # back as activity.value when the button is tapped. We include both
        # `option_id` (machine-readable) and `label` (human-readable); the
        # parser will feed `label` back through the clarification resolver so
        # the existing "exact → substring → 2-token overlap" match path still
        # works uniformly.
        actions = []
        for opt in (options or [])[:5]:  # Teams practical cap; resolver matches any
            label = (opt.get("label") or opt.get("value") or "").strip()
            if not label:
                continue
            actions.append({
                "type":  "Action.Submit",
                "title": label[:60],  # card button label truncation, UI-only
                "data": {
                    "option_id": opt.get("id") or opt.get("_term_id") or "",
                    "label":     label,
                    "pending_id": pending_id or "",
                },
            })

        # Fall back to plain text if no usable options — avoids sending an
        # empty card AND avoids the unnecessary token fetch + POST.
        if not actions:
            await self.send_message(event,
                f"❓ {question}\n\n_(Reply in plain language and I'll continue.)_"
            )
            return

        channel_info    = json.loads(event.channel_id)
        service_url     = channel_info["service_url"]
        conversation_id = channel_info["conversation_id"]
        reply_url = (
            f"{service_url.rstrip('/')}/v3/conversations/"
            f"{conversation_id}/activities"
        )
        token = await self._get_token()

        card = {
            "type":    "AdaptiveCard",
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "version": "1.3",
            "body": [
                {
                    "type":   "TextBlock",
                    "text":   "I need a bit more context",
                    "weight": "Bolder",
                    "size":   "Medium",
                    "color":  "Accent",
                    "wrap":   True,
                },
                {
                    "type":   "TextBlock",
                    "text":   question,
                    "wrap":   True,
                    "spacing": "Small",
                },
                {
                    "type":     "TextBlock",
                    "text":     "Pick an option below, or type your own clarification.",
                    "isSubtle": True,
                    "size":     "Small",
                    "spacing":  "Small",
                    "wrap":     True,
                },
            ],
            "actions": actions,
        }
        activity = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content":     card,
            }],
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                reply_url,
                headers={"Authorization": f"Bearer {token}"},
                json=activity,
                timeout=15,
            )
            if resp.status_code not in (200, 201):
                log.error("Teams send_clarification_prompt %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        log.info("Teams clarification prompt sent (%d options)", len(actions))

    # ── send_chart ────────────────────────────────────────────────────────────
    # Matches web_adapter.send_chart. Accepts a chart_payload dict produced by
    # core.chart.build_chart_payload() and renders it as a PNG via the existing
    # matplotlib pipeline, then hands off to upload_file so Teams displays it
    # inside an Adaptive Card. This keeps the rendering pipeline identical to
    # the one the portal uses — same colors, formatting, fallback behavior.

    async def send_chart(self, event: PlatformEvent, chart: dict) -> None:
        try:
            from core.chart import generate_chart
        except Exception as e:
            log.debug("Teams send_chart: renderer unavailable (%s) — skipping", e)
            return

        rows       = chart.get("rows") or []
        chart_type = chart.get("chart_type") or "bar"
        title      = chart.get("title") or "Results"
        if not rows:
            log.debug("Teams send_chart: empty rows, nothing to render")
            return

        try:
            png_bytes = generate_chart(rows, chart_type, title)
        except Exception as e:
            log.warning("Teams send_chart: render failed: %s", e)
            return
        if not png_bytes:
            return

        filename = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in title)[:40] + ".png"
        await self.upload_file(event, png_bytes, filename, "image/png")
