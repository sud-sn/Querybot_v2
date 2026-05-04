"""
core/webhook_dedup.py  —  Webhook idempotency (Fix #8)

Zoom, Slack and Teams all deliver webhooks at-least-once. A retry that
lands while the first delivery is still processing creates exactly the
bug that breaks clarifications:

   T+0s  : Zoom delivers message → dispatch saves pending clarification
   T+1s  : Zoom retries the SAME message (it didn't get a 200 fast enough)
   T+1s  : Dispatch sees pending, but now processes the duplicate as if
           it were a second, fresh message. Two LLM calls, two replies.

We deduplicate BEFORE dispatch on a short-lived (platform, account, user,
message-identifier) key. The cache is in-process — fine for a single
worker, which is what the current systemd unit runs. For multi-worker
deployments move to Redis.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("querybot.webhook_dedup")

# TTL for dedup entries. Webhook platforms retry within ~30–60 seconds
# when they don't see a 200. 120s gives us a comfortable margin.
_DEDUP_TTL_SECONDS = 120
_DEDUP_MAX_ENTRIES = 10_000

# key → first-seen monotonic timestamp
_SEEN: dict[str, float] = {}


def _gc_if_needed() -> None:
    if len(_SEEN) <= _DEDUP_MAX_ENTRIES:
        return
    now = time.monotonic()
    stale = [k for k, ts in _SEEN.items() if now - ts > _DEDUP_TTL_SECONDS]
    for k in stale:
        _SEEN.pop(k, None)


def _extract_message_id(event) -> Optional[str]:
    """
    Pull the platform's own message ID out of event.raw when available.

    Zoom  : payload.payload.message_id (when present); fall back to timestamp.
    Slack : event.event_id from the outer wrapper (stable per delivery).
    Teams : activity.id on the Bot Framework activity.

    Returns None if no stable identifier can be extracted.
    """
    raw = getattr(event, "raw", None) or {}

    platform = getattr(event, "platform", "")
    if platform == "zoom":
        p = raw.get("payload") or {}
        mid = p.get("message_id") or p.get("messageId")
        if mid:
            return str(mid)
        ts = p.get("timestamp") or raw.get("event_ts") or ""
        return str(ts) if ts else None

    if platform == "slack":
        # Slack sets event_id at the outer envelope; it's idempotent per
        # delivery attempt for the same originating event.
        eid = raw.get("event_id")
        if eid:
            return str(eid)
        inner = raw.get("event") or {}
        ts = inner.get("ts") or inner.get("event_ts")
        return str(ts) if ts else None

    if platform == "teams":
        aid = raw.get("id")
        return str(aid) if aid else None

    return None


def _key(event) -> str:
    mid = _extract_message_id(event)
    platform = getattr(event, "platform", "unknown")
    account = getattr(event, "account_id", "") or ""
    user = getattr(event, "user_id", "") or ""
    if mid:
        return f"{platform}:{account}:{user}:{mid}"
    # No stable ID → fall back to content hash. This is weaker (two genuine
    # identical messages from the same user within TTL will collide) but
    # better than nothing for platforms that don't provide message IDs.
    text = (getattr(event, "text", "") or "")[:200]
    return f"{platform}:{account}:{user}:content:{hash(text)}"


def is_duplicate_event(event) -> bool:
    """
    Return True iff we've seen this event within the TTL window.

    Call this BEFORE dispatch. If True, the caller should short-circuit
    and return 200 to the platform without invoking downstream handlers.
    """
    k = _key(event)
    now = time.monotonic()
    prev = _SEEN.get(k)
    if prev is not None and (now - prev) <= _DEDUP_TTL_SECONDS:
        log.info(
            "Duplicate webhook dropped: platform=%s key=%s age=%.1fs",
            getattr(event, "platform", "?"), k, now - prev,
        )
        return True
    return False


def remember_event(event) -> None:
    """
    Record that we are now processing this event. Call AFTER the duplicate
    check returns False, before dispatch. Ensures a near-simultaneous
    retry is blocked even while the first delivery is still in flight.
    """
    _gc_if_needed()
    _SEEN[_key(event)] = time.monotonic()


def forget_event(event) -> None:
    """
    Drop the dedup entry for this event. Useful in tests.
    Production code should leave entries to expire naturally.
    """
    _SEEN.pop(_key(event), None)


def _reset_for_tests() -> None:
    _SEEN.clear()
