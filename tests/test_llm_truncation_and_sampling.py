"""
tests/test_llm_truncation_and_sampling.py

Two defects at the provider boundary in core/llm.py.

1. A truncated completion was accepted as a finished one. `stop_reason` and
   `finish_reason` appeared nowhere in the repository, so when a model ran into
   the token ceiling the partial text was returned exactly like a complete
   response. For generated SQL that is the dangerous case: the tail is lost --
   a WHERE, the back half of a GROUP BY -- and what remains can still parse, so
   it reaches the validator looking like a legitimate query.

2. `temperature` was sent to Anthropic unconditionally. Claude models from the
   Opus 4.7 / Sonnet 5 generation onward reject sampling parameters with a 400
   rather than ignoring them, so the very first call to any of them failed and
   the model could not be selected at all.

Every assertion here fails against the pre-fix code. Nothing is asserted about
source text: each test drives the real provider adapters with a fake client and
checks what they return, raise, or send.
"""

import unittest
from unittest.mock import patch

from core.llm import (
    LLMTruncatedError,
    _anthropic_accepts_temperature,
    _anthropic_temperature_rejected,
    llm_complete,
)


# -- Fakes shaped like the two vendor SDKs -----------------------------------

class _Block:
    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, a, b, anthropic_style):
        if anthropic_style:
            self.input_tokens, self.output_tokens = a, b
        else:
            self.prompt_tokens, self.completion_tokens = a, b


class _AnthropicResponse:
    def __init__(self, text, stop_reason):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self.usage = _Usage(11, 7, anthropic_style=True)


class _Choice:
    def __init__(self, text, finish_reason):
        self.message = type("M", (), {"content": text})()
        self.finish_reason = finish_reason


class _OpenAIResponse:
    def __init__(self, text, finish_reason):
        self.choices = [_Choice(text, finish_reason)]
        self.usage = _Usage(11, 7, anthropic_style=False)


class _BadRequestError(Exception):
    """Shaped like the SDK's 400 for a parameter the model does not accept."""

    status_code = 400


class _FakeAnthropicClient:
    def __init__(self, text="SELECT 1 FROM t WHERE", stop_reason="end_turn",
                 reject_temperature=False):
        self.calls = []
        self._text = text
        self._stop_reason = stop_reason
        self._reject_temperature = reject_temperature
        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer._reject_temperature and "temperature" in kwargs:
                    raise _BadRequestError(
                        "Unexpected value(s) `temperature` for the model"
                    )
                return _AnthropicResponse(outer._text, outer._stop_reason)

        self.messages = _Messages()


class _FakeOpenAIClient:
    def __init__(self, text="SELECT 1 FROM t WHERE", finish_reason="stop"):
        self.calls = []
        self._text = text
        self._finish_reason = finish_reason
        outer = self

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                return _OpenAIResponse(outer._text, outer._finish_reason)

        self.chat = type("C", (), {"completions": _Completions()})()


def _anthropic(client):
    return patch("core.llm._get_anthropic_client", lambda _key: client)


def _openai(client):
    return patch("core.llm._get_openai_client", lambda _key: client)


def _azure(client):
    return patch("core.llm._get_azure_client", lambda *_a, **_k: client)


async def _ask(provider="anthropic", **kwargs):
    return await llm_complete(
        "system", "user", provider, kwargs.pop("model", "claude-sonnet-4-6"),
        "key", max_tokens=64, azure_endpoint="https://example.invalid",
        **kwargs,
    )


class TruncatedCompletionsAreRefused(unittest.IsolatedAsyncioTestCase):
    """A partial response must not be handed back as a finished one."""

    async def test_anthropic_truncation_raises(self):
        client = _FakeAnthropicClient(stop_reason="max_tokens")
        with _anthropic(client):
            with self.assertRaises(LLMTruncatedError):
                await _ask()

    async def test_openai_truncation_raises(self):
        client = _FakeOpenAIClient(finish_reason="length")
        with _openai(client):
            with self.assertRaises(LLMTruncatedError):
                await _ask(provider="openai", model="gpt-4o")

    async def test_azure_truncation_raises(self):
        client = _FakeOpenAIClient(finish_reason="length")
        with _azure(client):
            with self.assertRaises(LLMTruncatedError):
                await _ask(provider="azure_openai", model="gpt-4o-deployment")

    async def test_complete_response_is_returned_untouched(self):
        """The check must not fire on a normal completion."""
        client = _FakeAnthropicClient(text="SELECT 1", stop_reason="end_turn")
        with _anthropic(client):
            text, tok_in, tok_out = await _ask()
        self.assertEqual((text, tok_in, tok_out), ("SELECT 1", 11, 7))

    async def test_partial_text_is_carried_on_the_error(self):
        """So an opted-in caller need not pay for the completion twice."""
        client = _FakeAnthropicClient(text="half an ans", stop_reason="max_tokens")
        with _anthropic(client):
            try:
                await _ask()
            except LLMTruncatedError as exc:
                self.assertEqual(exc.text, "half an ans")
                self.assertEqual((exc.input_tokens, exc.output_tokens), (11, 7))
            else:
                self.fail("expected LLMTruncatedError")


class OptedInCallersStillGetTheirPartialText(unittest.IsolatedAsyncioTestCase):
    """Prose callers keep half a sentence; it beats no sentence."""

    async def test_allow_truncated_returns_the_partial_response(self):
        client = _FakeAnthropicClient(text="Sales rose sharply in",
                                      stop_reason="max_tokens")
        with _anthropic(client):
            text, tok_in, tok_out = await _ask(allow_truncated=True)
        self.assertEqual(text, "Sales rose sharply in")
        self.assertEqual((tok_in, tok_out), (11, 7))

    async def test_a_kept_truncation_is_audited_as_truncated_not_success(self):
        client = _FakeAnthropicClient(text="cut off", stop_reason="max_tokens")
        with _anthropic(client), patch("core.llm.record_llm_call") as rec:
            await _ask(allow_truncated=True)
        statuses = [c.kwargs.get("status") for c in rec.call_args_list]
        self.assertIn("truncated", statuses)
        self.assertNotIn("success", statuses)

    async def test_a_refused_truncation_is_audited_as_an_error(self):
        client = _FakeAnthropicClient(stop_reason="max_tokens")
        with _anthropic(client), patch("core.llm.record_llm_call") as rec:
            with self.assertRaises(LLMTruncatedError):
                await _ask()
        statuses = [c.kwargs.get("status") for c in rec.call_args_list]
        self.assertEqual(statuses, ["error"])


class AnthropicSamplingParameters(unittest.IsolatedAsyncioTestCase):
    """Sending `temperature` to a current Claude model is a 400, not a no-op."""

    def setUp(self):
        saved = set(_anthropic_temperature_rejected)
        _anthropic_temperature_rejected.clear()
        self.addCleanup(_anthropic_temperature_rejected.update, saved)
        self.addCleanup(_anthropic_temperature_rejected.clear)

    def test_known_models_are_classified_correctly(self):
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-7",
                      "claude-opus-4-8", "claude-opus-5-20260101"):
            self.assertFalse(
                _anthropic_accepts_temperature(model),
                f"{model} rejects sampling parameters",
            )
        for model in ("claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5"):
            self.assertTrue(
                _anthropic_accepts_temperature(model),
                f"{model} still accepts temperature, and 1.0 is not what SQL wants",
            )

    async def test_temperature_is_omitted_for_a_model_that_rejects_it(self):
        client = _FakeAnthropicClient()
        with _anthropic(client):
            await _ask(model="claude-opus-5", temperature=0.7)
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("temperature", client.calls[0])

    async def test_temperature_is_still_sent_where_it_is_accepted(self):
        client = _FakeAnthropicClient()
        with _anthropic(client):
            await _ask(model="claude-sonnet-4-6", temperature=0.2)
        self.assertEqual(client.calls[0].get("temperature"), 0.2)

    async def test_an_unknown_model_learns_from_its_first_rejection(self):
        """A hardcoded list goes stale; the rejection itself does not."""
        client = _FakeAnthropicClient(reject_temperature=True)
        with _anthropic(client):
            text, _, _ = await _ask(model="claude-something-new")
        self.assertEqual(text, "SELECT 1 FROM t WHERE")
        self.assertEqual(len(client.calls), 2, "one rejected call, then one without")
        self.assertIn("temperature", client.calls[0])
        self.assertNotIn("temperature", client.calls[1])

        # And the lesson sticks, so it costs one call per model, not per question.
        client.calls.clear()
        with _anthropic(client):
            await _ask(model="claude-something-new")
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("temperature", client.calls[0])

    async def test_an_unrelated_bad_request_is_not_mistaken_for_one(self):
        client = _FakeAnthropicClient()

        async def _boom(**kwargs):
            client.calls.append(kwargs)
            raise _BadRequestError("credit balance is too low")

        client.messages.create = _boom
        with _anthropic(client):
            with self.assertRaises(RuntimeError) as ctx:
                await _ask(model="claude-sonnet-4-6")
        self.assertIn("credit balance", str(ctx.exception))
        self.assertEqual(len(client.calls), 1, "must not retry an unrelated 400")


if __name__ == "__main__":
    unittest.main()
