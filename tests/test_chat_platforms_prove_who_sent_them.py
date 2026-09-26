"""
A message from Teams or Zoom is acted on only when the platform proves it sent it.

The Teams webhook accepted any Authorization header with three dot-separated
parts. A forged activity was answered as whichever user it named -- with that
user's data access -- and the reply, carrying this bot's own Bot Framework
token, was posted to the serviceUrl in the forged body: the answer and the
token both went wherever the forger said. Zoom's signature was checked, but
never its timestamp, so a captured request could be replayed forever.

Now the Teams token is verified as the Bot Framework specifies
(gateway/bot_framework_auth.py): signature by a published Microsoft key,
issuer, audience, expiry with five minutes' skew, the serviceUrl claim equal to
the activity's, and the key endorsed for the channel. A Zoom request older than
five minutes is refused, as Slack's already was.

The key set is served from a key generated here, at the one boundary that
fetches Microsoft's (bot_framework_auth._fetch_keys).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from gateway import bot_framework_auth as auth

APP_ID = "11111111-2222-3333-4444-555555555555"
SERVICE_URL = "https://smba.trafficmanager.net/emea/"
MICROSOFT = rsa.generate_private_key(public_exponent=65537, key_size=2048)
SOMEONE_ELSE = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _jwk(private_key, kid="k1", endorsements=("msteams", "skype")):
    numbers = private_key.public_key().public_numbers()
    return {"kty": "RSA", "kid": kid, "use": "sig", "endorsements": list(endorsements),
            "n": _b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
            "e": _b64(numbers.e.to_bytes(3, "big"))}


def _token(signer=MICROSOFT, kid="k1", alg="RS256", **overrides):
    now = int(time.time())
    claims = {"iss": auth.ISSUER, "aud": APP_ID, "exp": now + 3600, "nbf": now - 60,
              "serviceUrl": SERVICE_URL, **overrides}
    head = _b64(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = signer.sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"Bearer {head}.{body}.{_b64(signature)}"


def _activity(**overrides):
    return {"type": "message", "text": "total sales", "channelId": "msteams", "serviceUrl": SERVICE_URL,
            "from": {"id": "29:user", "aadObjectId": "aad-user"},
            "conversation": {"id": "conv-1", "tenantId": "tenant-1"},
            "channelData": {"tenant": {"id": "tenant-1"}}, **overrides}


@pytest.fixture
def published(monkeypatch):
    """Microsoft's key set, as this test serves it; counts the fetches."""
    keys = {"k1": _jwk(MICROSOFT)}
    fetches = []

    async def fetch():
        fetches.append(1)
        return dict(keys)

    monkeypatch.setattr(auth, "_fetch_keys", fetch)
    monkeypatch.setattr(auth, "_keys", {})
    monkeypatch.setattr(auth, "_keys_fetched_at", 0.0)
    return keys, fetches


def _refusal(authorization, activity=None, app_id=APP_ID, now=None):
    return asyncio.run(auth.refusal(authorization, activity or _activity(), app_id, now=now))


class TestTheTeamsToken:

    def test_a_token_the_bot_connector_signed_for_this_bot_is_accepted(self, published):
        assert _refusal(_token()) == ""

    @pytest.mark.parametrize("authorization", [
        "Bearer a.b.c",                 # what the old check accepted
        "Bearer not-a-token",
        "",
        "Basic dXNlcjpwYXNz",
    ])
    def test_a_header_that_is_not_a_signed_token_is_refused(self, published, authorization):
        assert _refusal(authorization) != ""

    def test_a_token_signed_by_anyone_else_is_refused(self, published):
        assert _refusal(_token(signer=SOMEONE_ELSE)) == "signature does not verify"

    @pytest.mark.parametrize("alg", ["none", "HS256"])
    def test_only_rs256_is_accepted(self, published, alg):
        assert _refusal(_token(alg=alg)) == "not signed with RS256"

    # Times are offsets from when the test runs, not from when it was
    # collected: a full suite reaches this file minutes after collection.
    @pytest.mark.parametrize("claims, why", [
        (lambda now: {"iss": "https://evil.example"}, "issued by someone else"),
        (lambda now: {"aud": "another-bot"}, "meant for another bot"),
        (lambda now: {"exp": now - 301}, "expired"),
        (lambda now: {"nbf": now + 301}, "not valid yet"),
        (lambda now: {"serviceUrl": "https://evil.example/"}, "serviceUrl differs from the token's"),
    ])
    def test_each_claim_is_checked(self, published, claims, why):
        assert _refusal(_token(**claims(int(time.time())))) == why

    def test_five_minutes_of_clock_skew_are_allowed(self, published):
        assert _refusal(_token(exp=int(time.time()) - 200, nbf=int(time.time()) + 200)) == ""

    def test_a_reply_address_other_than_the_signed_one_is_refused(self, published):
        # The forged body names somewhere else to send the answer and the token.
        assert _refusal(_token(), _activity(serviceUrl="https://evil.example/")) != ""
        assert _refusal(_token(), _activity(serviceUrl="")) != ""

    def test_a_key_not_endorsed_for_the_channel_is_refused(self, published):
        keys, _fetches = published
        keys["k1"] = _jwk(MICROSOFT, endorsements=("skype",))
        assert _refusal(_token()) == "key not endorsed for this channel"

    def test_an_unknown_key_refetches_the_set_but_not_on_every_request(self, published):
        _keys, fetches = published
        assert _refusal(_token(kid="k9")) == "signed by a key Microsoft does not publish"
        assert _refusal(_token(kid="k8")) != ""
        assert len(fetches) == 1

    def test_a_rotated_key_is_found_once_the_set_is_old_enough_to_refetch(self, published):
        keys, fetches = published
        assert _refusal(_token()) == ""
        keys["k2"] = _jwk(SOMEONE_ELSE, kid="k2")
        auth._keys_fetched_at -= auth.REFETCH_AFTER_SECONDS + 1
        assert _refusal(_token(signer=SOMEONE_ELSE, kid="k2")) == ""
        assert len(fetches) == 2

    def test_when_the_keys_cannot_be_fetched_nothing_is_accepted(self, monkeypatch):
        import httpx

        async def unreachable():
            raise httpx.ConnectError("no route")

        monkeypatch.setattr(auth, "_fetch_keys", unreachable)
        monkeypatch.setattr(auth, "_keys", {})
        monkeypatch.setattr(auth, "_keys_fetched_at", 0.0)
        assert _refusal(_token()) == "signing keys unavailable"

    def test_an_app_id_is_required(self, published):
        assert _refusal(_token(aud=""), app_id="") == "meant for another bot"


class TestTheTeamsWebhook:

    def _post(self, authorization, activity):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from gateway import webhooks
        from gateway.teams_adapter import TeamsAdapter

        adapter = TeamsAdapter({"app_id": APP_ID, "app_password": "secret", "tenant_id": "common"})
        app = FastAPI()
        app.include_router(webhooks.router)
        with patch.object(webhooks, "_load_teams_adapter", return_value=adapter), \
                patch.object(webhooks, "dispatch", new=AsyncMock()) as dispatch, \
                patch.object(TeamsAdapter, "send_status", new=AsyncMock()) as typing, \
                patch.object(webhooks, "is_duplicate_event", return_value=False), \
                patch.object(webhooks, "remember_event"):
            response = TestClient(app).post("/webhook/teams", content=json.dumps(activity).encode(),
                                            headers={"authorization": authorization,
                                                     "content-type": "application/json"})
        return response, dispatch, typing

    def test_a_forged_message_is_refused_and_nothing_is_sent_anywhere(self, published):
        forged = _activity(serviceUrl="https://evil.example/")
        response, dispatch, typing = self._post("Bearer a.b.c", forged)
        assert response.status_code == 401
        assert not dispatch.called and not typing.called

    def test_a_genuine_message_is_answered(self, published):
        response, dispatch, _typing = self._post(_token(), _activity())
        assert response.status_code == 200 and dispatch.called


class TestZoom:

    SECRET = "zoom-webhook-secret"

    def _adapter(self):
        from gateway.zoom_adapter import ZoomAdapter

        return ZoomAdapter({"client_id": "id", "client_secret": "secret", "bot_jid": "jid",
                            "webhook_secret": self.SECRET, "account_id": "acct"})

    def _headers(self, body, timestamp):
        signature = hmac.new(self.SECRET.encode(), f"v0:{timestamp}:{body.decode()}".encode(),
                             hashlib.sha256).hexdigest()
        return {"x-zm-request-timestamp": str(timestamp), "x-zm-signature": f"v0={signature}"}

    @pytest.mark.parametrize("scale", [1, 1000], ids=["seconds", "milliseconds"])
    def test_a_fresh_signed_request_is_accepted(self, scale):
        body = b'{"event":"bot_notification"}'
        timestamp = int(time.time() * scale)
        assert asyncio.run(self._adapter().verify_request(body, self._headers(body, timestamp)))

    def test_a_replayed_request_is_refused_though_its_signature_is_good(self):
        body = b'{"event":"bot_notification"}'
        old = int(time.time()) - 600
        assert not asyncio.run(self._adapter().verify_request(body, self._headers(body, old)))

    def test_a_request_without_a_timestamp_is_refused(self):
        body = b'{"event":"bot_notification"}'
        headers = self._headers(body, "")
        assert not asyncio.run(self._adapter().verify_request(body, headers))
