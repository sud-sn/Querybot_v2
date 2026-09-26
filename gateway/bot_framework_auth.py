"""
gateway/bot_framework_auth.py

Whether a request to the Teams webhook really comes from Microsoft's Bot
Connector, for this bot.

The Teams adapter used to check only that the Authorization header held three
dot-separated parts. Any such header was accepted, the forged activity was
answered as the user it named, and the reply -- with the bot's own Bot
Framework access token attached -- was posted to whatever serviceUrl the
forged body gave.

Now the bearer token is checked as the Bot Framework specifies
(learn.microsoft.com, "Authentication with the Bot Connector API", connector
to bot): an RS256 signature by a key from Microsoft's published key set,
issuer https://api.botframework.com, audience this bot's app id, not expired
and already valid within five minutes' clock skew, a serviceUrl claim equal
to the activity's -- which is what stops a reply being sent elsewhere -- and
a signing key endorsed for the activity's channel. Anything else is refused,
including the case where the key set cannot be fetched.
"""

from __future__ import annotations

import base64
import json
import logging
import time

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

log = logging.getLogger("gateway.teams.auth")

OPENID_CONFIGURATION = "https://login.botframework.com/v1/.well-known/openidconfiguration"
ISSUER = "https://api.botframework.com"
CLOCK_SKEW_SECONDS = 5 * 60
KEYS_MAX_AGE_SECONDS = 24 * 3600
# An unknown key id fetches the set again -- Microsoft rotates keys -- but not
# more often than this, so a stream of made-up key ids cannot make every
# request a call to Microsoft.
REFETCH_AFTER_SECONDS = 5 * 60

_keys: dict[str, dict] = {}
_keys_fetched_at = 0.0


async def _fetch_keys() -> dict[str, dict]:
    """Microsoft's current signing keys, by key id."""
    async with httpx.AsyncClient(timeout=10) as client:
        configuration = (await client.get(OPENID_CONFIGURATION)).raise_for_status().json()
        key_set = (await client.get(configuration["jwks_uri"])).raise_for_status().json()
    return {key["kid"]: key for key in key_set.get("keys", []) if key.get("kid")}


async def _signing_key(kid: str) -> dict | None:
    """The key with this id, from a key set at most a day old."""
    global _keys, _keys_fetched_at
    age = time.time() - _keys_fetched_at
    if age > KEYS_MAX_AGE_SECONDS or (kid not in _keys and age > REFETCH_AFTER_SECONDS):
        _keys, _keys_fetched_at = await _fetch_keys(), time.time()
    return _keys.get(kid)


def _b64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _public_key(jwk: dict) -> rsa.RSAPublicKey:
    n = int.from_bytes(_b64(jwk["n"]), "big")
    e = int.from_bytes(_b64(jwk["e"]), "big")
    return rsa.RSAPublicNumbers(e, n).public_key()


async def refusal(authorization: str, activity: dict, app_id: str, now: float | None = None) -> str:
    """Why this request is not from the Bot Connector for this bot; "" when it is."""
    if not authorization.startswith("Bearer "):
        return "no bearer token"
    parts = authorization[len("Bearer "):].strip().split(".")
    if len(parts) != 3:
        return "not a signed token"
    try:
        header, claims, signature = json.loads(_b64(parts[0])), json.loads(_b64(parts[1])), _b64(parts[2])
    except (ValueError, TypeError):
        return "not a signed token"
    if not isinstance(header, dict) or not isinstance(claims, dict):
        return "not a signed token"
    if header.get("alg") != "RS256":
        return "not signed with RS256"
    try:
        key = await _signing_key(str(header.get("kid") or ""))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("Teams: could not fetch the Bot Framework signing keys: %s", exc)
        return "signing keys unavailable"
    if key is None:
        return "signed by a key Microsoft does not publish"
    try:
        _public_key(key).verify(signature, f"{parts[0]}.{parts[1]}".encode(), padding.PKCS1v15(), hashes.SHA256())
    except (InvalidSignature, ValueError, KeyError):
        return "signature does not verify"
    if claims.get("iss") != ISSUER:
        return "issued by someone else"
    audience = claims.get("aud")
    if not app_id or (audience != app_id and not (isinstance(audience, list) and app_id in audience)):
        return "meant for another bot"
    now = time.time() if now is None else now
    expires, not_before = claims.get("exp"), claims.get("nbf")
    if not isinstance(expires, (int, float)) or now > expires + CLOCK_SKEW_SECONDS:
        return "expired"
    if isinstance(not_before, (int, float)) and now + CLOCK_SKEW_SECONDS < not_before:
        return "not valid yet"
    if not activity.get("serviceUrl") or claims.get("serviceUrl") != activity.get("serviceUrl"):
        return "serviceUrl differs from the token's"
    if activity.get("channelId") not in (key.get("endorsements") or []):
        return "key not endorsed for this channel"
    return ""
