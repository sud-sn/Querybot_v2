# -*- coding: utf-8 -*-
"""tests/test_secrets_never_fall_back_to_a_public_default.py

Anyone who had read this file knew the admin session secret.

Three call sites signed something security-relevant with a secret that, if
an operator forgot one environment variable, fell back to a fixed string
checked into this public repository:

    admin/routes.py    _session_secret()    "change-me-in-production"
    portal/routes.py   _session_secret()    "change-me-in-production"
    core/compliance/
      result_guard.py  _pseudonym_secret()  "querybot-development-pseudonym-
                                              secret"

"The operator forgot to set an env var" and "an attacker who has read this
file" ended in the exact same key. Enforcement was a startup log.warning; the
server started either way (main.py's startup hook). Anyone who knew the
literal could forge an admin or portal session cookie, or recompute the HMAC
that produces a PII pseudonym and match it against candidate values, for any
deployment that missed one environment variable.

core/process_secrets.env_secret_or_random replaces the literal with a secret
generated once, randomly, and held for the life of the process. Not a hard
startup failure -- that would break every dev/test deployment that has never
needed cross-restart session persistence, for a property most of them do not
need -- but a secret nobody outside this running process can know, rather
than one checked into source control with no expiry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import unittest
from unittest.mock import patch

OLD_SESSION_LITERAL = "change-me-in-production"
OLD_PSEUDONYM_LITERAL = "querybot-development-pseudonym-secret"


def _cleared_env(*names):
    """Context manager: these env vars are guaranteed unset for the block."""
    return patch.dict(os.environ, {n: "" for n in names}, clear=False)


class _FakeRequest:
    """The one attribute _is_auth actually reads -- not a mock of the
    function under test, a stand-in for the Request object it's handed."""

    def __init__(self, cookie_value: str | None):
        self.cookies = {"querybot_session": cookie_value} if cookie_value else {}


class EnvSecretOrRandomTests(unittest.TestCase):
    """Unit tests on the shared helper in isolation."""

    def setUp(self):
        import core.process_secrets as ps

        self._backup = dict(ps._generated)
        ps._generated.clear()

    def tearDown(self):
        import core.process_secrets as ps

        ps._generated.clear()
        ps._generated.update(self._backup)

    def test_an_explicitly_set_env_var_wins_over_generation(self):
        from core.process_secrets import env_secret_or_random

        with patch.dict(os.environ, {"MY_SECRET": "operator-chosen-value"}):
            self.assertEqual(
                env_secret_or_random("MY_SECRET", purpose="test_explicit"),
                "operator-chosen-value",
            )

    def test_the_precedence_order_is_first_set_wins(self):
        """Both set, to DIFFERENT values -- the case that actually exercises
        order. A test where only one of the two is ever set (the earlier
        version of this test) passes identically whichever order the
        function checks them in, since there is only one value to find
        either way."""
        from core.process_secrets import env_secret_or_random

        with patch.dict(
            os.environ,
            {"FIRST_VAR": "first-value", "SECOND_VAR": "second-value"},
        ):
            self.assertEqual(
                env_secret_or_random("FIRST_VAR", "SECOND_VAR", purpose="test_order"),
                "first-value",
            )

    def test_nothing_set_generates_a_secret_not_a_fixed_string(self):
        from core.process_secrets import env_secret_or_random

        with _cleared_env("UNSET_A", "UNSET_B"):
            generated = env_secret_or_random("UNSET_A", "UNSET_B", purpose="test_gen")
        self.assertTrue(generated)
        self.assertGreaterEqual(len(generated), 32)
        self.assertNotIn(generated, (OLD_SESSION_LITERAL, OLD_PSEUDONYM_LITERAL))

    def test_the_generated_secret_is_memoized_within_one_process(self):
        from core.process_secrets import env_secret_or_random

        with _cleared_env("UNSET_A"):
            first = env_secret_or_random("UNSET_A", purpose="test_memo")
            second = env_secret_or_random("UNSET_A", purpose="test_memo")
        self.assertEqual(first, second)

    def test_two_purposes_never_share_a_generated_secret(self):
        """The specific coincidence being closed: admin and portal used to
        collapse onto the SAME literal when both were unconfigured. Distinct
        purposes must never share a generated value even though both hit the
        same "nothing is set" branch."""
        from core.process_secrets import env_secret_or_random

        with _cleared_env("UNSET_A", "UNSET_B"):
            a = env_secret_or_random("UNSET_A", purpose="purpose_one")
            b = env_secret_or_random("UNSET_B", purpose="purpose_two")
        self.assertNotEqual(a, b)

    def test_two_generation_rounds_for_the_same_purpose_differ_across_a_restart(self):
        """Simulates a process restart by clearing the cache directly --
        the actual boundary a real restart crosses (a fresh Python process
        starts with an empty _generated dict)."""
        import core.process_secrets as ps

        with _cleared_env("UNSET_A"):
            first = ps.env_secret_or_random("UNSET_A", purpose="test_restart")
            ps._generated.clear()
            second = ps.env_secret_or_random("UNSET_A", purpose="test_restart")
        self.assertNotEqual(first, second)

    def test_missing_secret_is_logged_at_warning(self):
        from core.process_secrets import env_secret_or_random

        with _cleared_env("UNSET_LOGGED"), self.assertLogs(
            "querybot.process_secrets", level="WARNING"
        ) as captured:
            env_secret_or_random("UNSET_LOGGED", purpose="test_log")
        self.assertTrue(captured.output)
        self.assertIn("UNSET_LOGGED", " ".join(captured.output))

    def test_the_log_never_contains_the_generated_secret_itself(self):
        """The log line explains what's missing, not what was generated in
        its place -- printing the secret would be a strange way to keep it
        secret."""
        from core.process_secrets import env_secret_or_random

        with _cleared_env("UNSET_C"), self.assertLogs(
            "querybot.process_secrets", level="WARNING"
        ) as captured:
            generated = env_secret_or_random("UNSET_C", purpose="test_no_leak")
        self.assertNotIn(generated, " ".join(captured.output))


class AdminSessionSecretTests(unittest.TestCase):
    """Executes the real sign/verify round trip in admin/routes.py."""

    def setUp(self):
        import core.process_secrets as ps

        self._backup = dict(ps._generated)
        ps._generated.clear()

    def tearDown(self):
        import core.process_secrets as ps

        ps._generated.clear()
        ps._generated.update(self._backup)

    def test_a_session_signed_and_verified_with_no_env_vars_set_still_works(self):
        import admin.routes as routes

        with _cleared_env("ADMIN_SESSION_SECRET", "SESSION_SECRET"):
            cookie = routes._sign_admin_session()
            self.assertTrue(routes._is_auth(_FakeRequest(cookie)))

    def test_the_old_public_literal_can_no_longer_forge_a_session(self):
        """The actual vulnerability. Anyone who had read this file's history
        knew "change-me-in-production" was the fallback -- if it still were,
        a cookie forged with it would authenticate."""
        import admin.routes as routes

        payload = b"admin"
        forged_sig = hmac.new(
            OLD_SESSION_LITERAL.encode(), payload, hashlib.sha256
        ).hexdigest()
        forged_token = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        forged_cookie = f"{forged_token}.{forged_sig}"

        with _cleared_env("ADMIN_SESSION_SECRET", "SESSION_SECRET"):
            self.assertFalse(routes._is_auth(_FakeRequest(forged_cookie)))

    def test_a_cookie_from_before_a_restart_no_longer_authenticates(self):
        """No env var configured means no persistence guarantee -- this is
        the accepted tradeoff, not a bug, and is worth pinning down as the
        expected behaviour rather than discovering it by accident."""
        import core.process_secrets as ps
        import admin.routes as routes

        with _cleared_env("ADMIN_SESSION_SECRET", "SESSION_SECRET"):
            cookie = routes._sign_admin_session()
            ps._generated.clear()  # simulate a process restart
            self.assertFalse(routes._is_auth(_FakeRequest(cookie)))

    def test_an_explicit_secret_still_authenticates_across_a_restart(self):
        """The property real production deployments rely on: SET the env var
        and sessions keep working exactly as they always did."""
        import core.process_secrets as ps
        import admin.routes as routes

        with patch.dict(os.environ, {"ADMIN_SESSION_SECRET": "a-real-persisted-secret"}):
            cookie = routes._sign_admin_session()
            ps._generated.clear()
            self.assertTrue(routes._is_auth(_FakeRequest(cookie)))


class PortalSessionSecretTests(unittest.TestCase):
    """Same shape, the portal cookie path."""

    def setUp(self):
        import core.process_secrets as ps

        self._backup = dict(ps._generated)
        ps._generated.clear()

    def tearDown(self):
        import core.process_secrets as ps

        ps._generated.clear()
        ps._generated.update(self._backup)

    def test_a_session_signed_and_verified_with_no_env_vars_set_still_works(self):
        import portal.routes as routes

        with _cleared_env("PORTAL_SESSION_SECRET", "SESSION_SECRET"):
            cookie = routes._sign_session_value(42)
            self.assertEqual(routes._read_session_value(cookie), 42)

    def test_the_old_public_literal_can_no_longer_forge_a_session(self):
        import portal.routes as routes

        payload = b"42"
        forged_sig = hmac.new(
            OLD_SESSION_LITERAL.encode(), payload, hashlib.sha256
        ).hexdigest()
        forged_token = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        forged_cookie = f"{forged_token}.{forged_sig}"

        with _cleared_env("PORTAL_SESSION_SECRET", "SESSION_SECRET"):
            self.assertIsNone(routes._read_session_value(forged_cookie))

    def test_admin_and_portal_no_longer_share_a_secret_by_accident(self):
        """Pre-fix, both fell back to the identical literal, so an admin
        cookie's signature and a portal cookie's signature used the same key
        whenever neither app-specific var nor SESSION_SECRET was set. Never a
        deliberate design decision -- just two placeholder strings that
        happened to be spelled the same.

        Checked by comparing the two secrets directly, not by round-tripping
        a forged cookie between the two verifiers: an admin cookie's payload
        (b"admin") can never parse as portal's expected integer user id, so
        _read_session_value would return None on that mismatch ALONE, whether
        or not the underlying secrets actually differ -- a first version of
        this test passed for that unrelated reason even with a mutation that
        made the two purposes identical.
        """
        import admin.routes as admin_routes
        import portal.routes as portal_routes

        with _cleared_env(
            "ADMIN_SESSION_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET"
        ):
            admin_secret = admin_routes._session_secret()
            portal_secret = portal_routes._session_secret()
        self.assertNotEqual(admin_secret, portal_secret)


class PseudonymSecretTests(unittest.TestCase):
    """core/compliance/result_guard.py's PII pseudonym HMAC key."""

    def setUp(self):
        import core.process_secrets as ps

        self._backup = dict(ps._generated)
        ps._generated.clear()

    def tearDown(self):
        import core.process_secrets as ps

        ps._generated.clear()
        ps._generated.update(self._backup)

    def test_the_pseudonym_still_generates_with_no_env_vars_set(self):
        from core.compliance.result_guard import _mask

        with _cleared_env(
            "PII_PSEUDONYM_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET"
        ):
            alias = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                          source="PHYSICIAN_NAME")
        self.assertTrue(alias.startswith("Dr. "))
        self.assertNotIn("Alice", alias)

    def test_the_old_public_literal_can_no_longer_reproduce_the_pseudonym(self):
        """The actual vulnerability. Recomputing HMAC(old literal, payload)
        for a candidate name and matching it against the displayed alias is
        exactly how the fallback used to be defeated."""
        from core.compliance.result_guard import _mask

        with _cleared_env(
            "PII_PSEUDONYM_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET"
        ):
            real_alias = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                               source="PHYSICIAN_NAME")

        payload = "acct-1|PHYSICIAN_NAME|Dr. Alice Real".encode("utf-8")
        forged_digest = hmac.new(
            OLD_PSEUDONYM_LITERAL.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest().upper()
        forged_initial = chr(ord("A") + (int(forged_digest[:2], 16) % 26))
        forged_code = forged_digest[2:5]
        forged_alias = f"Dr. {forged_initial}-{forged_code}"

        self.assertNotEqual(real_alias, forged_alias)

    def test_the_same_value_still_produces_the_same_pseudonym_within_a_process(self):
        """No regression: the whole point of a pseudonym is that it is STABLE
        for one value within one account -- this must survive the fix."""
        from core.compliance.result_guard import _mask

        with _cleared_env(
            "PII_PSEUDONYM_SECRET", "PORTAL_SESSION_SECRET", "SESSION_SECRET"
        ):
            first = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                         source="PHYSICIAN_NAME")
            second = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                          source="PHYSICIAN_NAME")
        self.assertEqual(first, second)

    def test_an_explicit_secret_reproduces_the_same_pseudonym_across_a_restart(self):
        import core.process_secrets as ps
        from core.compliance.result_guard import _mask

        with patch.dict(os.environ, {"PII_PSEUDONYM_SECRET": "a-real-persisted-secret"}):
            first = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                         source="PHYSICIAN_NAME")
            ps._generated.clear()
            second = _mask("Dr. Alice Real", "safe_alias", "acct-1",
                          source="PHYSICIAN_NAME")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
