# -*- coding: utf-8 -*-
"""A completion the content filter cut in half was returned as a finished one.

An OpenAI-shaped chat completion says why it stopped. ``finish_reason ==
"length"`` means the token ceiling, and all three providers here raised
``LLMTruncatedError`` for it -- because "a truncated completion is not a shorter
answer, it is a different one" is a principle this codebase already holds.
``finish_reason == "content_filter"`` means a policy stopped it, and none of the
three checked for it. The partial text fell through the length check and was
RETURNED, and ``llm_complete`` recorded it in the audit trail as
``status="success"``.

Azure OpenAI applies a content filter to every deployment by default, so this
was live from the first question. Measured on the real code path before the fix:

    finish_reason="content_filter", content="SELECT SUM(NET"
      -> ('SELECT SUM(NET', 100, 20)          a clean success
    finish_reason="content_filter", content=None
      -> ('', 100, 20)                        a clean success, empty
    choices=[]
      -> IndexError: list index out of range  a bare Python error

For SQL the validator catches the first case and the turn degrades to a refusal.
For the model-written narrative there is no validator, so a half-sentence went to
the reader as the answer. And the third case -- which is what Azure returns when
a PROMPT is refused and the deployment is configured for asynchronous filtering
-- reached the reader as ``IndexError``, not as this module's own "check your
endpoint URL, API key, and deployment name".

Three copies of these eight lines existed, one per provider, and they had
drifted: the local path had learned to guard a missing ``usage`` block and the
other two had not. They are now one function, ``_read_chat_completion``, so the
guards are the same for all three and a fourth provider inherits them.

``LLMContentFilteredError`` is deliberately not a subclass of
``LLMTruncatedError``. ``allow_truncated`` is a caller accepting the ceiling IT
chose; a filter is a policy the caller never saw. And ``_kb_complete`` retries a
truncation at a HIGHER ceiling, which for a filtered document is one more
refusal.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import sys
import unittest
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import core.llm as llm  # noqa: E402
from core.knowledge import _kb_complete, kb_doc_token_budget  # noqa: E402
from core.llm import (  # noqa: E402
    LLMContentFilteredError,
    LLMTruncatedError,
    is_single_request_rejection,
    llm_complete,
)
from core.llm_audit import llm_audit_scope  # noqa: E402

ENDPOINT = "https://emco.openai.azure.com"
API_VERSION = "2024-02-01"


# ── an OpenAI-shaped response, in the shapes a provider really sends ─────────

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason):
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 100
    completion_tokens = 20


class _Response:
    def __init__(self, choices, usage=_Usage()):
        self.choices = choices
        self.usage = usage


class _Client:
    """The provider SDK's client, returning one canned response."""

    def __init__(self, response):
        self._response = response

    @property
    def chat(self):
        outer = self

        class _Chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    return outer._response

        return _Chat


def _install(response, provider="azure"):
    """Put a stub client in the real cache, keyed the way the real code keys it."""
    timeout, retries = llm._llm_timeout_seconds(), llm._llm_max_retries()
    if provider == "azure":
        key = ("azure", "key", ENDPOINT, API_VERSION, timeout, retries)
    elif provider == "openai":
        key = ("openai", "key", timeout, retries)
    else:
        key = ("local", ENDPOINT, "key", timeout, retries)
    llm._llm_client_cache[key] = _Client(response)
    return key


async def _azure_async(response, max_tokens=768):
    """The real Azure completion against a canned response."""
    key = _install(response, "azure")
    try:
        return await llm._azure_openai_complete(
            "sys", "user", "emco-prod", "key", max_tokens,
            ENDPOINT, API_VERSION, 0.0)
    finally:
        llm._llm_client_cache.pop(key, None)


def _azure(response, max_tokens=768):
    """Same call, for the tests that are not already inside a loop."""
    return asyncio.run(_azure_async(response, max_tokens))


class TestAFilteredCompletionIsRefused:

    def test_partial_text_is_no_longer_returned_as_an_answer(self):
        """The defect, on the exact shape that produced it."""
        with pytest.raises(LLMContentFilteredError) as caught:
            _azure(_Response([_Choice("SELECT SUM(NET", "content_filter")]))
        assert caught.value.text == "SELECT SUM(NET"
        assert caught.value.input_tokens == 100
        assert caught.value.output_tokens == 20

    def test_a_filtered_completion_with_no_text_is_refused_too(self):
        """This one returned ('', 100, 20) -- an empty string a caller would
        treat as "the model had nothing to say"."""
        with pytest.raises(LLMContentFilteredError):
            _azure(_Response([_Choice(None, "content_filter")]))

    def test_it_is_not_a_truncation(self):
        """Not a subclass, and the reason is behavioural, not taxonomic:
        allow_truncated must not swallow it and _kb_complete must not retry it
        at a higher ceiling."""
        error = LLMContentFilteredError("filtered")
        assert not isinstance(error, LLMTruncatedError)
        assert isinstance(error, RuntimeError)

    def test_the_message_says_a_filter_stopped_it(self):
        with pytest.raises(LLMContentFilteredError) as caught:
            _azure(_Response([_Choice("part", "content_filter")]))
        assert "content filter" in str(caught.value)

    def test_a_finished_completion_is_still_just_returned(self):
        assert _azure(_Response([_Choice("SELECT 1", "stop")])) == ("SELECT 1", 100, 20)

    def test_and_a_ceiling_truncation_still_raises_its_own_error(self):
        with pytest.raises(LLMTruncatedError):
            _azure(_Response([_Choice("SELECT 1 FRO", "length")]))


class TestAResponseWithNoCompletion:

    def test_an_empty_choices_list_is_not_an_index_error(self):
        """`IndexError: list index out of range` is what a reader used to get.
        It says nothing about what to check."""
        with pytest.raises(RuntimeError) as caught:
            _azure(_Response([]))
        assert not isinstance(caught.value, IndexError)
        assert "no completion" in str(caught.value)

    def test_the_message_names_the_two_things_that_cause_it(self):
        with pytest.raises(RuntimeError) as caught:
            _azure(_Response([]))
        message = str(caught.value).lower()
        assert "content filter" in message
        assert "proxy" in message

    def test_a_response_with_no_choices_attribute_at_all(self):
        """A proxy in front of the provider that returns a different JSON shape
        gives the SDK nothing to parse into choices."""
        with pytest.raises(RuntimeError) as caught:
            _azure(_Response(None))
        assert "no completion" in str(caught.value)

    def test_a_missing_usage_block_does_not_throw_the_answer_away(self):
        """The local path had already learned this; the other two had not, and
        one of the three copies knowing something is why there is now one."""
        assert _azure(
            _Response([_Choice("SELECT 1", "stop")], usage=None)
        ) == ("SELECT 1", 0, 0)


class TestAllThreeProvidersReadTheResponseTheSameWay(unittest.TestCase):
    """The point of consolidating. Before this, the same response got three
    different treatments depending on which provider the workspace used."""

    CASES = {
        "filtered": ([_Choice("half", "content_filter")], LLMContentFilteredError),
        "truncated": ([_Choice("half", "length")], LLMTruncatedError),
        "no choices": ([], RuntimeError),
    }

    def _call(self, provider, choices):
        response = _Response(choices)
        if provider == "azure":
            key = _install(response, "azure")
            coro = llm._azure_openai_complete(
                "sys", "user", "m", "key", 768, ENDPOINT, API_VERSION, 0.0)
        elif provider == "openai":
            key = _install(response, "openai")
            coro = llm._openai_complete("sys", "user", "m", "key", 768, 0.0)
        else:
            key = _install(response, "local")
            coro = llm._local_complete(
                "sys", "user", "m", "key", 768, ENDPOINT, 0.0)
        try:
            return asyncio.run(coro)
        finally:
            llm._llm_client_cache.pop(key, None)

    def test_each_provider_raises_the_same_thing(self):
        for label, (choices, expected) in self.CASES.items():
            for provider in ("azure", "openai", "local"):
                with self.subTest(case=label, provider=provider):
                    with self.assertRaises(expected):
                        self._call(provider, choices)

    def test_and_each_returns_the_same_thing_on_success(self):
        for provider in ("azure", "openai", "local"):
            with self.subTest(provider=provider):
                self.assertEqual(
                    self._call(provider, [_Choice(" SELECT 1 ", "stop")]),
                    ("SELECT 1", 100, 20),
                )


class TestTheAuditTrailSaysWhatHappened(unittest.TestCase):
    """A filtered answer recorded as a success is the row an admin cannot find
    when a reader reports an answer that stops mid-sentence."""

    def _logged(self, side_effect, **complete_kwargs):
        async def run():
            with patch("core.llm._openai_complete", new=AsyncMock(side_effect=side_effect)), \
                    patch("store.log_llm_call") as log_call:
                with llm_audit_scope(
                    account_id="acct_1",
                    question="net sales last month",
                    enabled=True,
                    request_id="req123",
                    component="sql_generation",
                ):
                    try:
                        await llm_complete(
                            system="System prompt", user="User prompt",
                            provider="openai", model="gpt-4o", api_key="key",
                            **complete_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        return log_call, exc
                    return log_call, None

        return asyncio.run(run())

    def test_a_filtered_call_is_not_recorded_as_a_success(self):
        log_call, raised = self._logged(
            LLMContentFilteredError("stopped by a content filter", text="half"))
        self.assertIsInstance(raised, LLMContentFilteredError)
        log_call.assert_called_once()
        self.assertEqual(log_call.call_args.kwargs["status"], "content_filtered")

    def test_the_row_keeps_the_partial_text_and_the_reason(self):
        log_call, _raised = self._logged(
            LLMContentFilteredError("stopped by a content filter", text="half"))
        kwargs = log_call.call_args.kwargs
        self.assertIn("content filter", kwargs["error_msg"])
        self.assertEqual(kwargs["account_id"], "acct_1")

    def test_allow_truncated_does_not_accept_a_filtered_answer(self):
        """That flag is the caller accepting the ceiling IT set. A caller that
        says "a cut-off answer is fine" has not said "an answer a policy
        rewrote is fine" -- it does not know the policy exists."""
        log_call, raised = self._logged(
            LLMContentFilteredError("stopped by a content filter", text="half"),
            allow_truncated=True)
        self.assertIsInstance(raised, LLMContentFilteredError)
        self.assertEqual(log_call.call_args.kwargs["status"], "content_filtered")

    def test_a_kept_truncation_is_still_kept(self):
        """The neighbouring behaviour, so this change is not quietly a
        regression of the truncation contract."""
        log_call, raised = self._logged(
            LLMTruncatedError("truncated", text="SELECT 1", input_tokens=5,
                              output_tokens=2),
            allow_truncated=True)
        self.assertIsNone(raised)
        self.assertEqual(log_call.call_args.kwargs["status"], "truncated")


class TestWhatTheRestOfTheProductDoesWithIt(unittest.IsolatedAsyncioTestCase):

    async def test_a_filtered_kb_document_skips_its_table(self):
        """It must classify as a per-request rejection, or one filtered table
        takes the whole knowledge-base build down with it.

        The error comes from the REAL path, not from a message written here: the
        predicate reads the text, so a test that composes its own text proves
        the predicate matches that text and nothing about what the product
        emits. Asserting on a hand-written message is how the two drift.
        """
        with pytest.raises(LLMContentFilteredError) as caught:
            await _azure_async(_Response([_Choice("SELECT SUM(NET", "content_filter")]))
        assert is_single_request_rejection(caught.value)

    async def test_and_the_build_records_it_rather_than_raising(self):
        raised = None
        try:
            await _azure_async(_Response([_Choice("## half", "content_filter")]))
        except LLMContentFilteredError as exc:
            raised = exc

        async def filtered(*args, **kwargs):
            raise raised

        with patch("core.llm.llm_complete", side_effect=filtered):
            text, reason = await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "azure_openai",
                "emco-prod", "key", max_tokens=kb_doc_token_budget(121),
                azure_endpoint=ENDPOINT, azure_api_version=API_VERSION)
        self.assertIsNone(text)
        self.assertIn("ContentFiltered", reason)

    async def test_a_filtered_narrative_does_not_kill_the_turn(self):
        """The reader still gets their numbers. core/result_conversation.py
        already fails open on any LLM failure, and this must stay one -- raising
        instead of returning a half sentence is only an improvement if the card
        still renders."""
        from core import result_conversation

        calls: list[int] = []

        async def filtered(*args, **kwargs):
            calls.append(1)
            raise LLMContentFilteredError(
                "stopped by a content filter", text="Net sales rose by")

        async def answered(*args, **kwargs):
            calls.append(1)
            return "Quebec is the only region in this result.", 10, 5

        # The result-chat feature is gated on the account's compliance posture,
        # and an account with no profile is treated as regulated -- so without
        # this the model is never asked at all and "" proves nothing. Found by
        # asserting the clean-answer case first, which is why it is here.
        import core.compliance.policy_engine as policy_engine

        async def ask(side_effect):
            with patch("core.llm.llm_complete", side_effect=side_effect), \
                    patch.object(policy_engine, "result_llm_features_allowed",
                                 lambda _account: True):
                return await result_conversation.converse_about_result(
                    "what does this show?",
                    rows=[{"REGION": "QC", "NET_SALES": 100}],
                    result_question="net sales by region",
                    account_id="acct_1",
                    provider="azure_openai",
                    model="emco-prod",
                    api_key="key",
                    azure_endpoint=ENDPOINT,
                    azure_api_version=API_VERSION,
                )

        # Non-empty on a clean answer, so "" below is the fail-open path and not
        # an early return this fixture happens to trip.
        self.assertTrue(await ask(answered))
        self.assertEqual(len(calls), 1, "the model was never asked")

        calls.clear()
        self.assertEqual(await ask(filtered), "")
        self.assertEqual(len(calls), 1, "the filtered call never happened")


class TestTheLogIsTheOnlySignal(unittest.TestCase):
    """A fail-open narrative means the log is the only place this shows up."""

    def test_a_filtered_completion_logs_at_warning(self):
        with self.assertLogs("querybot.llm", level=logging.WARNING) as captured:
            with self.assertRaises(LLMContentFilteredError):
                _azure(_Response([_Choice("half an answer", "content_filter")]))
        blob = "\n".join(captured.output)
        self.assertIn("content filter", blob)
        self.assertIn("emco-prod", blob)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
