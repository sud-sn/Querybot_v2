"""Egress posture — core/compliance/egress.py and its enforcement in core/llm.py.

The product's strongest governance claim is "nothing leaves your network", and
a claim like that is worth exactly as much as the thing that enforces it.
These tests execute the real predicate and the real funnel.

Two things are asserted hardest:

  * a workspace that has DECLARED airgapped can never reach a hosted endpoint,
    whatever a caller passes — because the check lives at ``llm_complete``,
    where there is no keyword to forget; and
  * a workspace that has declared nothing keeps working, because the strict
    answer here blocks rather than filters and a new client has no compliance
    profile until an admin makes one.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.compliance.egress import (  # noqa: E402
    AIRGAPPED,
    CLOUD,
    DEFAULT_POSTURE,
    FAIL_CLOSED_POSTURE,
    POSTURES,
    PRIVATE,
    describe,
    egress_posture,
    normalise_posture,
    posture_is_declared,
    provider_allowed,
)


def _arun(coro):
    return asyncio.run(coro)


def _profile(**values):
    """Patch the store so an account has (or lacks) a profile."""
    exists = values.pop("_exists", True)
    return (
        patch("store.compliance_profile_exists", return_value=exists),
        patch("store.get_compliance_profile", return_value=dict(values)),
    )


class TestReadingThePosture(unittest.TestCase):

    def test_a_declared_posture_is_returned(self):
        for declared in POSTURES:
            with _profile(egress_posture=declared)[0], \
                 _profile(egress_posture=declared)[1]:
                self.assertEqual(egress_posture("acct"), declared)

    def test_an_undeclared_posture_reads_as_the_default(self):
        a, b = _profile(egress_posture="")
        with a, b:
            self.assertEqual(egress_posture("acct"), DEFAULT_POSTURE)
            self.assertFalse(posture_is_declared("acct"))

    def test_a_workspace_with_no_profile_at_all_still_works(self):
        # A client created after the upgrade backfill has no profile until an
        # admin completes compliance setup. Treating that as air-gapped would
        # mean a new workspace cannot generate SQL at all -- an outage, not a
        # safe default.
        a, b = _profile(_exists=False)
        with a, b:
            self.assertEqual(egress_posture("brand-new"), DEFAULT_POSTURE)
            self.assertFalse(posture_is_declared("brand-new"))
            allowed, _ = provider_allowed("brand-new", "anthropic")
            self.assertTrue(allowed)

    def test_an_unrecognised_stored_value_fails_closed(self):
        a, b = _profile(egress_posture="whatever-someone-typed")
        with a, b:
            self.assertEqual(egress_posture("acct"), FAIL_CLOSED_POSTURE)

    def test_a_broken_lookup_fails_closed(self):
        with patch("store.compliance_profile_exists",
                   side_effect=RuntimeError("db gone")):
            self.assertEqual(egress_posture("acct"), FAIL_CLOSED_POSTURE)
            self.assertFalse(posture_is_declared("acct"))

    def test_normalisation_rejects_anything_not_a_posture(self):
        self.assertEqual(normalise_posture("CLOUD"), CLOUD)
        self.assertEqual(normalise_posture("  airgapped "), AIRGAPPED)
        self.assertEqual(normalise_posture("nonsense"), FAIL_CLOSED_POSTURE)
        self.assertEqual(normalise_posture(""), "")
        self.assertEqual(normalise_posture(None), "")

    def test_fail_closed_is_the_strictest_posture(self):
        self.assertEqual(FAIL_CLOSED_POSTURE, AIRGAPPED)
        self.assertNotEqual(DEFAULT_POSTURE, FAIL_CLOSED_POSTURE)


class TestWhichProvidersEachPostureAllows(unittest.TestCase):

    def _allowed(self, posture, provider):
        a, b = _profile(egress_posture=posture)
        with a, b:
            return provider_allowed("acct", provider)

    def test_airgapped_permits_only_a_local_model(self):
        for hosted in ("anthropic", "openai", "azure_openai"):
            allowed, reason = self._allowed(AIRGAPPED, hosted)
            self.assertFalse(allowed, hosted)
            self.assertIn("airgapped", reason)
            self.assertIn(hosted, reason)
        self.assertTrue(self._allowed(AIRGAPPED, "local")[0])

    def test_cloud_permits_every_provider(self):
        for provider in ("anthropic", "openai", "azure_openai", "local"):
            self.assertTrue(self._allowed(CLOUD, provider)[0], provider)

    def test_private_excludes_the_plain_openai_endpoint(self):
        self.assertFalse(self._allowed(PRIVATE, "openai")[0])
        self.assertTrue(self._allowed(PRIVATE, "azure_openai")[0])
        self.assertTrue(self._allowed(PRIVATE, "local")[0])

    def test_an_unknown_provider_is_refused_under_every_posture(self):
        for posture in POSTURES:
            self.assertFalse(self._allowed(posture, "some-new-vendor")[0], posture)

    def test_a_refusal_says_what_would_have_been_permitted(self):
        _, reason = self._allowed(AIRGAPPED, "openai")
        self.assertIn("permitted: local", reason)


class TestTheFunnelEnforcesIt(unittest.TestCase):
    """llm_complete is where the guarantee actually lives.

    resolve_provider checks too, but a caller can bypass it by passing a
    provider directly. The funnel cannot be bypassed, because there is no
    argument to omit — it reads the account from the ambient audit scope the
    pipeline already opens.
    """

    def _call(self, provider="openai"):
        from core.llm import llm_complete
        from core.llm_audit import llm_audit_scope

        with llm_audit_scope(account_id="acct-air", question="q",
                             enabled=True, request_id="r1",
                             component="sql_generation"):
            return _arun(llm_complete(
                system="s", user="u", provider=provider,
                model="m", api_key="k",
            ))

    def test_a_declared_airgapped_workspace_cannot_reach_a_hosted_model(self):
        from core.llm import EgressPostureError

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("core.llm._openai_complete",
                         new_callable=AsyncMock) as hosted:
            with self.assertRaises(EgressPostureError):
                self._call("openai")
        # Not merely "an error was raised" — the provider was never reached.
        hosted.assert_not_called()

    def test_the_refusal_writes_a_proof_row(self):
        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.log_llm_call") as log_call, \
                patch("core.llm._openai_complete", new_callable=AsyncMock):
            try:
                self._call("openai")
            except Exception:
                pass
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "blocked")
        self.assertIn("egress", log_call.call_args.kwargs["payload_preview_sanitized"])

    def test_the_same_workspace_may_use_its_local_model(self):
        # The negative test above proves nothing unless the permitted path
        # actually completes.
        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("core.llm._local_complete", new_callable=AsyncMock,
                         return_value=("answer", 10, 5)) as local:
            text, tin, tout = self._call("local")
        local.assert_awaited_once()
        self.assertEqual((text, tin, tout), ("answer", 10, 5))

    def test_a_cloud_workspace_is_unaffected(self):
        a, b = _profile(egress_posture=CLOUD)
        with a, b, patch("core.llm._openai_complete", new_callable=AsyncMock,
                         return_value=("answer", 1, 1)) as hosted:
            self._call("openai")
        hosted.assert_awaited_once()

    def test_a_call_with_no_account_in_scope_is_not_blocked(self):
        # Evals and KB jobs complete outside a per-tenant scope. There is no
        # tenant whose posture could apply, so there is nothing to enforce —
        # and blocking them would be enforcing a posture nobody declared.
        from core.llm import llm_complete

        with patch("core.llm._openai_complete", new_callable=AsyncMock,
                   return_value=("answer", 1, 1)) as hosted:
            _arun(llm_complete(system="s", user="u", provider="openai",
                               model="m", api_key="k"))
        hosted.assert_awaited_once()


class TestResolveProviderRefusesEarly(unittest.TestCase):

    def test_an_airgapped_workspace_pointed_at_a_hosted_provider_is_told_so(self):
        from core.llm import EgressPostureError, resolve_provider

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.get_all_system", return_value={
            "default_llm_provider": "anthropic", "anthropic_api_key": "k",
        }):
            with self.assertRaises(EgressPostureError) as caught:
                resolve_provider({"account_id": "acct-air"})
        self.assertIn("airgapped", str(caught.exception))
        self.assertIn("Admin", str(caught.exception))

    def test_the_same_workspace_resolves_its_local_model(self):
        from core.llm import resolve_provider

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.get_all_system", return_value={
            "default_llm_provider": "local",
            "local_llm_base_url": "http://localhost:11434/v1",
            "local_llm_model": "llama3.1:8b",
        }):
            provider, model, api_key, extra = resolve_provider({"account_id": "acct-air"})
        self.assertEqual(provider, "local")
        self.assertEqual(model, "llama3.1:8b")
        self.assertEqual(extra["azure_endpoint"], "http://localhost:11434/v1")

    def test_a_local_provider_needs_an_address_not_a_key(self):
        # Ollama ignores the key entirely; demanding one would make the
        # posture that protects the most also the hardest to configure.
        from core.llm import resolve_provider

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.get_all_system", return_value={
            "default_llm_provider": "local",
            "local_llm_base_url": "http://localhost:11434/v1",
        }):
            provider, model, api_key, _ = resolve_provider({"account_id": "a"})
        self.assertEqual(provider, "local")
        self.assertEqual(api_key, "")
        self.assertTrue(model)

    def test_a_local_provider_with_no_address_says_what_to_do(self):
        from core.llm import resolve_provider

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.get_all_system", return_value={
            "default_llm_provider": "local",
        }):
            with self.assertRaises(RuntimeError) as caught:
                resolve_provider({"account_id": "a"})
        self.assertIn("base URL", str(caught.exception))


class TestTheLocalProvider(unittest.TestCase):

    def test_a_local_completion_is_audited_like_any_other(self):
        from core.llm import llm_complete
        from core.llm_audit import llm_audit_scope

        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b, patch("store.log_llm_call") as log_call, \
                patch("core.llm._local_complete", new_callable=AsyncMock,
                      return_value=("answer", 12, 4)):
            with llm_audit_scope(account_id="acct-air", question="q",
                                 enabled=True, request_id="r1",
                                 component="sql_generation"):
                _arun(llm_complete(system="s", user="u", provider="local",
                                   model="llama3.1:8b", api_key=""))
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "success")
        self.assertEqual(log_call.call_args.kwargs["llm_provider"], "local")

    def test_the_base_url_reaches_the_completion(self):
        from core.llm import llm_complete

        with patch("core.llm._local_complete", new_callable=AsyncMock,
                   return_value=("answer", 1, 1)) as local:
            _arun(llm_complete(
                system="s", user="u", provider="local", model="m", api_key="",
                azure_endpoint="http://box:8000/v1",
            ))
        self.assertIn("http://box:8000/v1", local.await_args.args)

    def test_a_server_that_reports_no_token_usage_still_answers(self):
        # llama.cpp and some Ollama builds omit the usage block. An answer
        # that arrived must not be thrown away because the server did not say
        # what it cost.
        from unittest.mock import MagicMock

        import core.llm as llm

        message = MagicMock(content="the answer")
        choice = MagicMock(message=message, finish_reason="stop")
        response = MagicMock(choices=[choice], usage=None)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        with patch("core.llm._get_local_client", return_value=client):
            text, tin, tout = _arun(llm._local_complete(
                "s", "u", "m", "", 100, "http://localhost:11434/v1"))
        self.assertEqual(text, "the answer")
        self.assertEqual((tin, tout), (0, 0))

    def test_a_missing_base_url_is_a_legible_error_not_a_connection_failure(self):
        import core.llm as llm

        with self.assertRaises(RuntimeError) as caught:
            llm._get_local_client("", "")
        self.assertIn("base URL", str(caught.exception))


class TestDescribeForTheProofPack(unittest.TestCase):

    def test_an_airgapped_workspace_asserts_no_external_egress(self):
        a, b = _profile(egress_posture=AIRGAPPED)
        with a, b:
            described = describe("acct")
        self.assertEqual(described["posture"], AIRGAPPED)
        self.assertTrue(described["declared"])
        self.assertFalse(described["external_egress"])
        self.assertEqual(described["permitted_providers"], ["local"])

    def test_an_undeclared_workspace_says_so(self):
        a, b = _profile(egress_posture="")
        with a, b:
            described = describe("acct")
        self.assertEqual(described["posture"], CLOUD)
        self.assertFalse(described["declared"])
        self.assertTrue(described["external_egress"])

    def test_embeddings_are_reported_as_local_in_every_posture(self):
        # core.vector_store runs SentenceTransformer in-process, so retrieval
        # text never leaves regardless of posture. It is worth stating,
        # because an auditor asked about "the AI" means all of it.
        for posture in POSTURES:
            a, b = _profile(egress_posture=posture)
            with a, b:
                self.assertTrue(describe("acct")["embeddings_local"], posture)


if __name__ == "__main__":
    unittest.main()


# ══════════════════════════════════════════════════════════════════════════════
# The admin surface
# ══════════════════════════════════════════════════════════════════════════════

class TestDeclaringThePostureFromAdmin(unittest.TestCase):
    """Write API to read API with nothing handed in between.

    The posture is only worth having if the form that sets it and the
    predicate that enforces it agree about where it is stored. Every test
    here posts the real route and then asks ``egress_posture`` — nothing is
    passed from one to the other by the test.
    """

    def setUp(self):
        import uuid

        import store
        store.init_db()
        self.account_id = f"acct-egress-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    @staticmethod
    def _post(form_data):
        from unittest.mock import MagicMock

        from starlette.datastructures import FormData

        req = MagicMock()
        req.query_params = {}

        async def _form():
            return FormData(form_data)

        req.form = _form
        return req

    def _save(self, posture):
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.compliance_save_egress(
                self._post({"egress_posture": posture}), self.account_id))

    def test_a_posture_saved_in_admin_is_the_posture_that_is_enforced(self):
        result = self._save("airgapped")
        self.assertEqual(result.status_code, 303)
        self.assertIn("saved=egress", result.headers["location"])
        # Nothing handed in below this line — the predicate must find it.
        self.assertEqual(egress_posture(self.account_id), AIRGAPPED)
        self.assertTrue(posture_is_declared(self.account_id))
        self.assertFalse(provider_allowed(self.account_id, "openai")[0])

    def test_changing_it_back_takes_effect(self):
        self._save("airgapped")
        self._save("cloud")
        self.assertEqual(egress_posture(self.account_id), CLOUD)
        self.assertTrue(provider_allowed(self.account_id, "openai")[0])

    def test_an_invalid_posture_is_refused_and_changes_nothing(self):
        self._save("airgapped")
        import admin.routes as routes
        with patch.object(routes, "_is_auth", return_value=True):
            result = _arun(routes.compliance_save_egress(
                self._post({"egress_posture": "whatever"}), self.account_id))
        self.assertIn("error=egress_posture", result.headers["location"])
        self.assertEqual(egress_posture(self.account_id), AIRGAPPED)

    def test_declaring_a_posture_does_not_reset_the_policy_lifecycle(self):
        # Its own route precisely so that changing where a model runs does not
        # invalidate an approved policy version, which the industry form does.
        import store
        store.save_compliance_profile(
            self.account_id, mode="regulated", industry="banking",
            lifecycle_state="ACTIVE", enforcement_mode="enforce",
            active_policy_version=7,
        )
        self._save("airgapped")
        profile = store.get_compliance_profile(self.account_id)
        self.assertEqual(profile["lifecycle_state"], "ACTIVE")
        self.assertEqual(profile["enforcement_mode"], "enforce")
        self.assertEqual(profile["active_policy_version"], 7)
        self.assertEqual(profile["egress_posture"], "airgapped")

    def test_an_unauthenticated_post_is_refused(self):
        import admin.routes as routes
        from fastapi import HTTPException
        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(HTTPException):
                _arun(routes.compliance_save_egress(
                    self._post({"egress_posture": "airgapped"}), self.account_id))
        self.assertFalse(posture_is_declared(self.account_id))


# The admin System form writes credentials through Fernet, and by the time a
# full-suite run reaches this module the process has been through a dozen test
# modules that repoint QUERYBOT_KEY_FILE at their own mkdtemp() and re-import
# store. The result is that a value written here comes back as InvalidToken
# inside the same test -- with the key file, the database, and every module
# binding verifiably identical. Rather than assert around that, the round-trip
# runs in a clean interpreter: same real route, same real resolve_provider,
# nothing handed between them, and no inherited module state.
_ROUNDTRIP_SCRIPT = r"""
import json, os, sys, tempfile, asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, sys.argv[1])
work = tempfile.mkdtemp(prefix="qb-roundtrip-")
os.environ["DB_PATH"] = os.environ["QUERYBOT_DB_PATH"] = os.path.join(work, "s.db")
os.environ["QUERYBOT_KEY_FILE"] = os.path.join(work, "key")

import store
import admin.routes as routes
store.init_db()

fields = json.loads(sys.argv[2])
req = MagicMock()
req.query_params = {}
with patch.object(routes, "_is_auth", return_value=True):
    asyncio.run(routes.system_save(
        req,
        anthropic_key="", openai_key="", azure_openai_key="",
        azure_openai_endpoint="", azure_api_version="",
        azure_query_deployment="", azure_kb_deployment="",
        default_provider="local", default_model="", kb_model="",
        database_backend="", database_url="", save_section="credentials",
        **fields,
    ))

# Nothing handed across this line -- resolve_provider must find what the route
# wrote, through the real system-key allow-list and the real cipher.
from core.llm import _default_model, resolve_provider
provider, model, api_key, extra = resolve_provider({"account_id": ""})
_p, kb_model, _k, _e = resolve_provider({"account_id": ""}, purpose="kb")
print(json.dumps({
    "provider": provider, "model": model, "api_key": api_key,
    "base_url": extra.get("azure_endpoint"), "kb_model": kb_model,
    "stored_model": store.get_all_system().get("local_llm_model"),
    "default_fast": _default_model("local", "fast"),
}))
"""


def _roundtrip(**fields):
    """Save through the admin route and read back through resolve_provider."""
    import json
    import subprocess

    defaults = {
        "local_base_url": "http://gpu-box:8000/v1",
        "local_api_key": "",
        "local_model": "qwen2.5:14b",
        "local_kb_model": "qwen2.5:32b",
    }
    defaults.update(fields)
    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _ROUNDTRIP_SCRIPT, root, json.dumps(defaults)],
        capture_output=True, text=True, cwd=root, timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"round-trip subprocess failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestTheLocalModelSettingsRoundTrip(unittest.TestCase):
    """The admin System form to resolve_provider, with nothing in between.

    This is the write-to-read test for the local provider: if the form field
    names, the system-key allow-list and resolve_provider's lookups ever stop
    agreeing, every one of these fails. It is the shape that catches an
    identity mismatch, which is exactly what happened here the first time --
    ``set_system`` rejected the four new keys because they were not on the
    allow-list, and nothing else would have noticed.
    """

    def test_settings_saved_in_admin_are_what_resolve_provider_reads(self):
        out = _roundtrip()
        self.assertEqual(out["provider"], "local")
        self.assertEqual(out["model"], "qwen2.5:14b")
        self.assertEqual(out["base_url"], "http://gpu-box:8000/v1")
        self.assertEqual(out["kb_model"], "qwen2.5:32b")

    def test_a_local_provider_needs_no_api_key_to_resolve(self):
        # The hosted providers raise "No API key configured"; a local runtime
        # ignores the key entirely, and demanding one would make the posture
        # that protects the most also the hardest to configure.
        out = _roundtrip(local_api_key="")
        self.assertEqual(out["provider"], "local")
        self.assertEqual(out["api_key"], "")

    def test_clearing_the_model_name_falls_back_to_the_dropdown(self):
        # Blank means "use the dropdown", not "keep the old value" -- a free
        # text field that cannot be emptied is a field that traps a typo.
        out = _roundtrip(local_model="")
        self.assertEqual(out["stored_model"], "")
        self.assertEqual(out["model"], out["default_fast"])

    def test_a_different_base_url_is_the_one_that_comes_back(self):
        out = _roundtrip(local_base_url="http://gpu-7:9000/v1")
        self.assertEqual(out["base_url"], "http://gpu-7:9000/v1")


class TestOneUnreadableSettingDoesNotTakeDownTheRest(unittest.TestCase):
    """get_all_system used to raise on the first row it could not decrypt.

    A key rotation, a restored backup or a half-migrated deployment then took
    out every setting at once: the admin System page and resolve_provider both
    died on a cryptography stack trace that named no key, so nobody could tell
    which value to re-enter. An unreadable row is now skipped and logged.
    """

    def setUp(self):
        import store
        store.init_db()
        store.set_system("openai_api_key", "sk-readable")
        store.set_system("default_llm_provider", "openai")

    def test_a_row_that_will_not_decrypt_is_skipped_not_fatal(self):
        import store
        import store.config_store as cs

        real = cs.decrypt

        def _selective(blob):
            value = real(blob)
            if value == "sk-readable":
                raise ValueError("wrong key for this row")
            return value

        with patch.object(cs, "decrypt", _selective):
            settings = store.get_all_system()

        self.assertNotIn("openai_api_key", settings)
        self.assertEqual(settings.get("default_llm_provider"), "openai")

    def test_a_single_unreadable_key_reads_as_its_default(self):
        import store
        import store.config_store as cs

        with patch.object(cs, "decrypt", side_effect=ValueError("wrong key")):
            self.assertEqual(store.get_system("openai_api_key", "fallback"), "fallback")

    def test_a_readable_setting_still_comes_back(self):
        # The skipping above is only safe if the normal path is untouched.
        import store
        self.assertEqual(store.get_system("openai_api_key"), "sk-readable")
        self.assertEqual(store.get_all_system()["openai_api_key"], "sk-readable")


# ── Proof of refusal ──────────────────────────────────────────────────────────

class TestARefusalIsRecordedEvenWithCallLoggingOff(unittest.TestCase):
    """The evidence of refusal is the point of the clause it serves.

    record_llm_blocked used to share record_llm_call's `enabled` gate, which
    is the client's "enable LLM audit" toggle — and that column defaults to 0.
    So on a default workspace an air-gapped refusal blocked correctly and
    recorded nothing, and the proof pack's refusals section, which reads
    exactly these rows, reported "0 model calls were refused" for a tenant
    that had refused some.

    That gate is about the volume of CALL logging. A refusal is not a call: it
    carries no prompt, no payload and no data, and it is rare by construction.
    """

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-refusal-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-ref-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _blocked_rows(self):
        import store

        with store.get_db() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT status, component, payload_preview_sanitized "
                "FROM llm_call_log WHERE account_id=? AND status='blocked'",
                (self.account_id,),
            ).fetchall()]

    def _refuse(self, *, enabled):
        from core.llm_audit import llm_audit_scope, record_llm_blocked

        with llm_audit_scope(
            account_id=self.account_id, question="what is revenue",
            enabled=enabled, request_id="r1", question_id="q1",
            component="sql_generation",
        ):
            record_llm_blocked("llm_complete", "egress posture: airgapped")

    def test_the_refusal_is_recorded_with_the_audit_toggle_off(self):
        self._refuse(enabled=False)
        rows = self._blocked_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "blocked")
        self.assertIn("airgapped", rows[0]["payload_preview_sanitized"])

    def test_it_is_still_recorded_with_the_toggle_on(self):
        self._refuse(enabled=True)
        self.assertEqual(len(self._blocked_rows()), 1)

    def test_the_proof_pack_counts_it_either_way(self):
        # The read side, from the write side, with nothing passed between.
        from core.compliance.proof_pack import refusals_section

        self._refuse(enabled=False)
        section = refusals_section(self.account_id, 30)
        self.assertEqual(section["total"], 1)
        self.assertIn("1 model calls were refused", section["statement"])

    def test_with_no_account_in_scope_nothing_is_written(self):
        """The remaining gate, and it is quieter than the database's.

        llm_call_log has a foreign key to client, so a refusal attributed to a
        made-up account is refused by SQLite anyway — the guard is defence in
        depth over that. What it adds is silence: without it, every scope-less
        call site raises an IntegrityError into this function's fail-open
        handler and logs a warning, and a steady drip of "audit write failed"
        is exactly the noise that hides a real audit failure.
        """
        import store

        from core.llm_audit import record_llm_blocked

        with self.assertNoLogs("querybot.llm_audit", level="WARNING"):
            record_llm_blocked("llm_complete", "no scope here")

        # And for ANY account, not just this one: a refusal filed under an
        # invented id would have the pack reporting a refusal that belongs to
        # nobody.
        with store.get_db() as conn:
            everywhere = conn.execute(
                "SELECT account_id FROM llm_call_log WHERE status='blocked'"
            ).fetchall()
        self.assertEqual([dict(r) for r in everywhere], [])
