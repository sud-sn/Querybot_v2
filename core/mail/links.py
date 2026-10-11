"""
core/mail/links.py

Links in an email that work without signing in: "stop these emails" (unsubscribe). Each is signed with a key
kept in this server's settings, so a link cannot be made up or changed to stop someone else's email, and it
keeps working after a restart.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Optional

import store

_KEY = "email_link_secret"


def _secret() -> bytes:
    value = str(store.get_system(_KEY, "") or "")
    if not value:
        value = secrets.token_urlsafe(32)
        if not store.claim_system_key(_KEY, value):       # another process made one first
            value = str(store.get_system(_KEY, "") or value)
    return value.encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def sign(payload: dict) -> str:
    body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    mac = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:18])
    return f"{body}.{mac}"


def read(token: str) -> Optional[dict]:
    try:
        body, mac = str(token or "").split(".", 1)
        expected = _b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest()[:18])
        if not hmac.compare_digest(mac, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - anything unreadable is no link
        return None


def unsubscribe_token(address: str, kind: str, ref: str = "") -> str:
    return sign({"a": str(address).strip().lower(), "k": kind, "r": str(ref or "")})


def unsubscribe_url(base: str, address: str, kind: str, ref: str = "") -> str:
    return f"{base.rstrip('/')}/portal/unsubscribe/{unsubscribe_token(address, kind, ref)}"
