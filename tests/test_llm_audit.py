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


if __name__ == "__main__":
    unittest.main()
