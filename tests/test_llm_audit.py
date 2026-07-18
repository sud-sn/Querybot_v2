import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.llm import llm_complete
from core.llm_audit import llm_audit_scope, sanitize_payload_preview


class LlmAuditTests(unittest.TestCase):
    def test_sanitize_payload_preview_masks_literals_and_identifiers(self):
        preview = sanitize_payload_preview(
            'Return JSON with {"email":"alice@example.com"} and id "1234567890".',
            "User asked about employee 'Alice Adams' and phone +1 555 123 4567.",
        )

        self.assertNotIn("alice@example.com", preview)
        self.assertNotIn("Alice Adams", preview)
        self.assertNotIn("1234567890", preview)
        self.assertNotIn("+1 555 123 4567", preview)
        self.assertIn("[SYSTEM]", preview)
        self.assertIn("[USER]", preview)

    def test_llm_complete_logs_success_when_scope_enabled(self):
        async def _run():
            with patch("core.llm._openai_complete", new=AsyncMock(return_value=("SELECT 1", 11, 7))), \
                 patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_1",
                    question="show revenue by month",
                    enabled=True,
                    request_id="req123",
                    component="sql_generation",
                ):
                    result = await llm_complete(
                        system="System prompt",
                        user="User prompt",
                        provider="openai",
                        model="gpt-4o",
                        api_key="test-key",
                    )
                return result, log_call

        result, log_call = asyncio.run(_run())
        self.assertEqual(result, ("SELECT 1", 11, 7))
        log_call.assert_called_once()
        kwargs = log_call.call_args.kwargs
        self.assertEqual(kwargs["account_id"], "acct_1")
        self.assertEqual(kwargs["request_id"], "req123")
        self.assertEqual(kwargs["component"], "sql_generation")
        self.assertEqual(kwargs["llm_model"], "gpt-4o")
        self.assertEqual(kwargs["status"], "success")
        self.assertGreater(kwargs["prompt_chars"], 0)

    def test_llm_complete_logs_error_when_provider_fails(self):
        async def _run():
            with patch("core.llm._openai_complete", new=AsyncMock(side_effect=RuntimeError("provider down"))), \
                 patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_2",
                    question="show headcount",
                    enabled=True,
                    request_id="req999",
                    component="analysis",
                ):
                    with self.assertRaises(RuntimeError):
                        await llm_complete(
                            system="System prompt",
                            user='Show rows for "Bob Brown"',
                            provider="openai",
                            model="gpt-4o-mini",
                            api_key="test-key",
                        )
                return log_call

        log_call = asyncio.run(_run())
        log_call.assert_called_once()
        kwargs = log_call.call_args.kwargs
        self.assertEqual(kwargs["status"], "error")
        self.assertIn("provider down", kwargs["error_msg"])

    def test_llm_complete_records_response_hash_and_sanitized_preview(self):
        # Response capture: the audit row must prove what came BACK, not just
        # what was sent — SHA-256 of the exact response text + a sanitized
        # preview (PII masked by the same rules as the prompt side).
        response_text = "SELECT name FROM t WHERE email = 'carol@example.com'"

        async def _run():
            with patch("core.llm._openai_complete", new=AsyncMock(return_value=(response_text, 5, 9))), \
                 patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_r",
                    question="who signed up",
                    enabled=True,
                    request_id="req555",
                    component="sql_generation",
                ):
                    await llm_complete(
                        system="System prompt",
                        user="User prompt",
                        provider="openai",
                        model="gpt-4o",
                        api_key="test-key",
                    )
                return log_call

        log_call = asyncio.run(_run())
        kwargs = log_call.call_args.kwargs
        import hashlib
        self.assertEqual(
            kwargs["response_hash"],
            hashlib.sha256(response_text.encode()).hexdigest(),
        )
        self.assertEqual(kwargs["response_chars"], len(response_text))
        self.assertNotIn("carol@example.com", kwargs["response_preview_sanitized"])
        self.assertIn("SELECT", kwargs["response_preview_sanitized"])

    def test_error_rows_have_empty_response_fields(self):
        async def _run():
            with patch("core.llm._openai_complete", new=AsyncMock(side_effect=RuntimeError("down"))), \
                 patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_e",
                    question="q",
                    enabled=True,
                    request_id="req556",
                    component="sql_generation",
                ):
                    with self.assertRaises(RuntimeError):
                        await llm_complete(
                            system="s", user="u",
                            provider="openai", model="gpt-4o", api_key="k",
                        )
                return log_call

        log_call = asyncio.run(_run())
        kwargs = log_call.call_args.kwargs
        self.assertEqual(kwargs["response_hash"], "")
        self.assertEqual(kwargs["response_chars"], 0)
        self.assertEqual(kwargs["response_preview_sanitized"], "")

    def test_llm_complete_skips_logging_when_scope_disabled(self):
        async def _run():
            with patch("core.llm._openai_complete", new=AsyncMock(return_value=("ok", 1, 1))), \
                 patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_3",
                    question="show headcount",
                    enabled=False,
                    request_id="req000",
                    component="sql_generation",
                ):
                    await llm_complete(
                        system="System prompt",
                        user="User prompt",
                        provider="openai",
                        model="gpt-4o-mini",
                        api_key="test-key",
                    )
                return log_call

        log_call = asyncio.run(_run())
        log_call.assert_not_called()


class ResponseCaptureRoundTripTests(unittest.TestCase):
    """Real DB round-trip: the migration added the response columns and
    log_llm_call persists them so get_recent_llm_calls (SELECT *) surfaces
    them in the admin audit panel."""

    def test_response_fields_persist_and_read_back(self):
        import uuid
        import store
        store.init_db()
        acct = f"acct-resp-{uuid.uuid4().hex[:8]}"
        store.upsert_client(acct, "portal")
        try:
            store.log_llm_call(
                account_id=acct, request_id="r1", question="q",
                component="sql_generation", llm_provider="azure_openai",
                llm_model="gpt-4o", status="success",
                payload_hash="p" * 64, payload_preview_sanitized="[SYSTEM]\nx",
                prompt_chars=10,
                response_hash="a" * 64,
                response_preview_sanitized="SELECT [number] FROM T",
                response_chars=42,
            )
            groups = store.get_recent_llm_calls(acct, limit=5)
            self.assertEqual(len(groups), 1)
            call = groups[0]["calls"][0]
            self.assertEqual(call["response_hash"], "a" * 64)
            self.assertEqual(call["response_preview_sanitized"], "SELECT [number] FROM T")
            self.assertEqual(call["response_chars"], 42)
        finally:
            with store.get_db() as conn:
                conn.execute("DELETE FROM llm_call_log WHERE account_id=?", (acct,))
                conn.execute("DELETE FROM client WHERE account_id=?", (acct,))


class ResponseCapturePanelWiringTests(unittest.TestCase):
    def test_audit_panel_renders_response_preview_and_hashes(self):
        src = (ROOT / "admin/templates/client_detail.html").read_text(encoding="utf-8")
        self.assertIn("response_preview_sanitized", src)
        self.assertIn("Sanitized response preview", src)
        self.assertIn("call.response_hash", src)
        self.assertIn("call.payload_hash", src)


if __name__ == "__main__":
    unittest.main()
