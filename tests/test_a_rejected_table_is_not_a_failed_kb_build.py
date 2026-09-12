# -*- coding: utf-8 -*-
"""A 400 on the widest fact still took the whole knowledge-base build down.

An earlier fix stopped a TRUNCATED document from aborting the build: the ceiling
is now derived from the table's width, a short estimate is retried once at the
cap, and a document that still will not finish skips that one table while the
rest of the build completes. It closed the trigger it was written for and left
every other trigger open, because ``_kb_complete`` caught only
``LLMTruncatedError`` and ``build_kb``'s per-table loop has no ``try`` of its
own (AST-confirmed: the loop spans the three completion calls and contains no
handler).

On Azure OpenAI that gap is not theoretical, and it is the same wide fact that
finds it. GPT-4o allows 16,384 output tokens from version 2024-08-06 but only
4,096 on 2024-05-13, and the budget here scales with the column count:

    kb_doc_token_budget(40)   ->  4,096     narrow tables, fine either way
    kb_doc_token_budget(121)  ->  7,984     EMCO's invoice fact
    kb_doc_token_budget(210)  -> 12,256     the measured wide case

So on a 2024-05-13 deployment every table past 40 columns is rejected with
``400 max_tokens must be less than or equal to 4096`` -- and the narrow ones in
the same build would all have succeeded. Before this, the first wide table
raised, the exception left the loop, every remaining table went undocumented,
and ``_build_failures.json`` -- written AFTER the loop -- was never written at
all. The admin got a failed build, a raw Azure error blob as its only
explanation, and no record of which table caused it.

Azure's default content filter is the same shape: what trips it is the text of
one document.

What must NOT be contained is the other half of the rule, and it is why this is
a predicate rather than a bare ``except Exception``. A rejected key, a
deployment that does not exist, a dead endpoint, a rate limit still in force a
second later: those reject the next table identically. Swallowing them produces
a build that reports success having written nothing, and a knowledge base with
holes answers questions about those tables from the schema alone -- quietly, for
as long as nobody rebuilds it. That is worse than a failed build, which at least
tells someone.

The tests drive the real ``build_kb`` over a real schema directory with the
provider as the only boundary, and assert on the documents on disk and the
failure report -- not on ``_kb_complete``'s return value, because the defect was
never in that function. It was in what the exception did on its way out.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from core.knowledge import (
    _kb_complete,
    kb_doc_token_budget,
)
from core.llm import EgressPostureError, is_single_request_rejection

WIDE = ("CUS_ORD_IVC_FCT", 121)      # EMCO's invoice fact, as measured
NARROW = ("CUS_RTN_FCT", 40)
TINY = ("DT_DMS", 12)

AZURE_KWARGS = {
    "azure_endpoint": "https://emco.openai.azure.com",
    "azure_api_version": "2024-02-01",
}


def azure_error(message: str) -> RuntimeError:
    """An error shaped like the one _azure_openai_complete actually raises.

    It wraps the SDK exception in a RuntimeError ``from e``, so the typed error
    survives only as ``__cause__`` and the status code only in the text. Any
    classification has to cope with that, and a test that raised a bare
    exception of a convenient type would not be testing the real shape.
    """
    try:
        raise ValueError(f"Error code: {message}")
    except ValueError as sdk_error:
        try:
            raise RuntimeError(f"Azure OpenAI error: {sdk_error}") from sdk_error
        except RuntimeError as wrapped:
            return wrapped


# Real Azure rejection texts. Pinned as literals: the predicate reads the
# message, so the message is the contract, and generating these from the
# implementation would make the test agree with any mistake in it.
CEILING_TOO_HIGH = azure_error(
    "400 - {'error': {'code': 'BadRequest', 'message': 'max_tokens must be "
    "less than or equal to 4096'}}")
CONTENT_FILTERED = azure_error(
    "400 - {'error': {'code': 'content_filter', 'message': 'The response was "
    "filtered due to the prompt triggering Azure OpenAI's content management "
    "policy'}}")
CONTEXT_TOO_LONG = azure_error(
    "400 - {'error': {'code': 'context_length_exceeded', 'message': \"This "
    "model's maximum context length is 128000 tokens\"}}")

KEY_REJECTED = azure_error("401 - Access denied due to invalid subscription key")
NO_DEPLOYMENT = azure_error(
    "404 - {'error': {'code': 'DeploymentNotFound', 'message': 'The API "
    "deployment for this resource does not exist'}}")
RATE_LIMITED = azure_error(
    "429 - Requests to the ChatCompletions_Create Operation under Azure OpenAI "
    "API have exceeded token rate limit of your current quota")
SERVER_ERROR = azure_error("500 - InternalServerError")


class TestWhichRejectionsBelongToOneRequest:
    """The predicate, on the message shapes Azure actually sends."""

    @pytest.mark.parametrize("exc,why", [
        (CEILING_TOO_HIGH, "the ceiling came from this table's width"),
        (CONTENT_FILTERED, "the filter reacted to this document's text"),
        (CONTEXT_TOO_LONG, "this prompt is the one that is too long"),
    ])
    def test_these_are_about_this_request(self, exc, why):
        assert is_single_request_rejection(exc), why

    @pytest.mark.parametrize("exc,why", [
        (KEY_REJECTED, "the next table is rejected identically"),
        (NO_DEPLOYMENT, "nothing will work until an admin fixes it"),
        (RATE_LIMITED, "still in force a second later, and a holed KB is worse"),
        (SERVER_ERROR, "no reason to think the next call differs"),
        (EgressPostureError("this workspace forbids that provider"),
         "a posture refusal applies to the whole build"),
        (RuntimeError("Azure OpenAI error: connection reset"), "unrecognised"),
    ])
    def test_these_are_about_the_configuration(self, exc, why):
        assert not is_single_request_rejection(exc), why

    def test_a_missing_max_tokens_is_not_a_ceiling_that_was_too_high(self):
        """The word appears in both directions. Only an upper bound is
        per-request; "max_tokens is required" is a caller bug on every call."""
        assert not is_single_request_rejection(
            azure_error("400 - {'error': {'message': 'max_tokens is required'}}"))

    def test_the_real_wrapper_carries_the_rejection_text(self):
        """The predicate reads the exception's message and nothing else. That is
        only safe because _azure_openai_complete interpolates the provider's own
        error into the RuntimeError it raises.

        Executed against the real function rather than reasoned about: reading
        __cause__ as well was tried and removed for changing no outcome, so this
        is the assertion that keeps the simplification honest. A wrapper changed
        to raise a bare "Azure OpenAI error" would fail here, not silently make
        every 400 unclassifiable.
        """
        from core import llm

        class _Stub:
            @property
            def chat(self):
                class _Chat:
                    class completions:
                        @staticmethod
                        async def create(**kwargs):
                            raise ValueError(
                                "Error code: 400 - {'error': {'code': "
                                "'BadRequest', 'message': 'max_tokens must be "
                                "less than or equal to 4096'}}")
                return _Chat

        key = ("azure", "key", "https://emco.openai.azure.com", "2024-02-01",
               llm._llm_timeout_seconds(), llm._llm_max_retries())
        llm._llm_client_cache[key] = _Stub()
        try:
            with pytest.raises(RuntimeError) as caught:
                asyncio.run(llm._azure_openai_complete(
                    "sys", "user", "emco-prod", "key", 7_984,
                    "https://emco.openai.azure.com", "2024-02-01", 0.0))
        finally:
            llm._llm_client_cache.pop(key, None)

        assert is_single_request_rejection(caught.value)


class TestOneCallContainsOrPropagates(unittest.IsolatedAsyncioTestCase):
    """_kb_complete, executed, against each error shape."""

    async def _complete(self, exc, *, columns=121):
        calls: list[int] = []

        async def fake(system, user, provider, model, api_key,
                       max_tokens=1024, **kw):
            calls.append(max_tokens)
            raise exc

        with patch("core.llm.llm_complete", side_effect=fake):
            outcome = await _kb_complete(
                f"document for {WIDE[0]}", "sys", "user", "azure_openai",
                "emco-prod", "key",
                max_tokens=kb_doc_token_budget(columns), **AZURE_KWARGS)
        return outcome, calls

    async def test_a_ceiling_rejection_skips_the_label_and_says_why(self):
        (text, reason), calls = await self._complete(CEILING_TOO_HIGH)
        self.assertIsNone(text)
        self.assertIn("max_tokens", reason)
        self.assertIn("4096", reason)

    async def test_and_it_is_not_retried_higher(self):
        """Retrying at the cap is right for a ceiling that was too LOW. For one
        that was already too high it spends a second call to be refused the same
        way -- and 16,000 is further over the limit than 7,984 was."""
        _outcome, calls = await self._complete(CEILING_TOO_HIGH)
        self.assertEqual(calls, [7_984])

    async def test_a_filtered_document_skips_its_label_too(self):
        (text, reason), calls = await self._complete(CONTENT_FILTERED)
        self.assertIsNone(text)
        self.assertIn("content_filter", reason)
        self.assertEqual(len(calls), 1)

    async def test_a_reason_is_bounded_so_it_can_go_in_a_json_report(self):
        (_text, reason), _calls = await self._complete(
            azure_error("400 - content_filter " + "x" * 5_000))
        self.assertLessEqual(len(reason), 340)

    async def test_a_rejected_key_still_propagates(self):
        with self.assertRaises(RuntimeError):
            await self._complete(KEY_REJECTED)

    async def test_a_rate_limit_still_propagates(self):
        """Containing this is how a build finishes with thirty-five holes in
        it. A failed build an admin can retry is the better outcome."""
        with self.assertRaises(RuntimeError):
            await self._complete(RATE_LIMITED)

    async def test_an_egress_refusal_still_propagates(self):
        with self.assertRaises(EgressPostureError):
            await self._complete(
                EgressPostureError("this workspace forbids that provider"))


def schema_md(table: str, column_count: int) -> str:
    """A schema document in the shape core/schema.py writes one."""
    rows = "\n".join(
        f"| `COL_{index:03d}_AMT` | decimal(18) | No |  |"
        for index in range(column_count)
    )
    return (
        f"# EMDW_DMART.{table}\n\n"
        f"**Type:** BASE TABLE  **Schema:** EMDW_DMART\n\n"
        f"**SQL table name:** `EMDW_DMART.{table}`\n\n"
        f"**Row count:** 9200000  **Scale:** Large\n\n"
        f"## Columns\n\n"
        f"| Column | Type | Nullable | Distinct Values |\n"
        f"|--------|------|:--------:|-----------------|\n{rows}\n"
    )


class TestTheWholeBuild(unittest.TestCase):
    """build_kb over a real schema directory, provider as the only boundary."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="qb-kb-reject-")
        self.schema_dir = os.path.join(self.root, "schema")
        self.kb_dir = os.path.join(self.root, "kb")
        for path in (self.schema_dir, self.kb_dir):
            os.makedirs(path)
        for table, columns in (WIDE, NARROW, TINY):
            with open(os.path.join(self.schema_dir, f"{table}.md"), "w") as handle:
                handle.write(schema_md(table, columns))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @staticmethod
    def _stage(user: str) -> str:
        if "Generate the complete Knowledge Base document" in user:
            return "table_doc"
        if "## Overview" in user:
            return "query_examples"
        return "business_vocab"

    def build(self, *, reject_above: int | None = None, error=CEILING_TOO_HIGH):
        """Run the real build. `reject_above` is the deployment's output cap:
        any call asking for more than it is refused, exactly as Azure would."""
        calls: list[tuple[str, str, int]] = []

        async def fake(system, user, provider, model, api_key,
                       max_tokens=1024, **kw):
            stage = self._stage(user)
            table = "" if stage == "business_vocab" else next(
                (name for name, _ in (WIDE, NARROW, TINY) if name in user), "")
            calls.append((stage, table, max_tokens))
            if reject_above is not None and max_tokens > reject_above:
                raise error
            return "## Overview\nfinished\n", 100, 200

        from core import knowledge

        with patch("core.llm.llm_complete", side_effect=fake), \
                patch.object(knowledge, "_embed_kb_files_qdrant",
                             return_value=None):
            count = asyncio.run(knowledge.build_kb(
                schema_dir=self.schema_dir, kb_dir=self.kb_dir,
                chroma_dir="acct", business_desc="a plumbing distributor",
                provider="azure_openai", model="emco-prod", api_key="key",
                extra_kwargs=dict(AZURE_KWARGS), account_id="acct"))
        return count, calls

    def docs(self):
        return sorted(name for name in os.listdir(self.kb_dir)
                      if name.endswith("_kb.md") and not name.startswith("_"))

    def failures(self):
        path = os.path.join(self.kb_dir, "_build_failures.json")
        if not os.path.exists(path):
            return []
        with open(path) as handle:
            return json.load(handle)

    # ── the defect ───────────────────────────────────────────────────────────

    def test_a_wide_fact_is_not_the_whole_build(self):
        """A gpt-4o 2024-05-13 deployment: 4,096 output tokens. The 121-column
        fact asks for 7,984 and is refused; the 40- and 12-column tables ask for
        4,096 and are fine. Before this, all three went undocumented."""
        count, _calls = self.build(reject_above=4_096)
        self.assertEqual(self.docs(), ["CUS_RTN_FCT_kb.md", "DT_DMS_kb.md"])
        self.assertEqual(count, 2)

    def test_the_skipped_table_is_named_in_the_failure_report(self):
        """Written after the loop, so an escaping exception used to skip it --
        leaving the admin with no record of which table caused the failure."""
        self.build(reject_above=4_096)
        reported = self.failures()
        self.assertTrue(reported, "no _build_failures.json was written")
        wide = [row for row in reported if row["table"] == WIDE[0]]
        self.assertTrue(wide, reported)
        self.assertEqual(wide[0]["columns"], 121)
        self.assertEqual(wide[0]["max_tokens"], 7_984)

    def test_and_the_report_says_what_actually_happened(self):
        """The reason used to be the constant "output truncated at the token
        ceiling", which for a 400 is simply untrue -- and untrue in the one
        place an admin goes to find out."""
        self.build(reject_above=4_096)
        reason = next(row["reason"] for row in self.failures()
                      if row["table"] == WIDE[0])
        self.assertIn("max_tokens", reason)
        self.assertNotIn("truncated", reason)

    def test_a_content_filter_on_one_table_costs_that_table_only(self):
        count, _calls = self.build(reject_above=4_096, error=CONTENT_FILTERED)
        self.assertEqual(count, 2)
        self.assertEqual(self.docs(), ["CUS_RTN_FCT_kb.md", "DT_DMS_kb.md"])
        reason = next(row["reason"] for row in self.failures()
                      if row["table"] == WIDE[0])
        self.assertIn("content_filter", reason)

    def test_the_narrow_tables_keep_their_worked_examples(self):
        """A build that only half-happened is worth little. The tables that
        succeeded must be fully built, not just documented."""
        self.build(reject_above=4_096)
        examples = sorted(name for name in os.listdir(self.kb_dir)
                          if name.endswith("_queries.md"))
        self.assertEqual(examples, ["CUS_RTN_FCT_queries.md", "DT_DMS_queries.md"])

    # ── what must still fail loudly ──────────────────────────────────────────

    def test_a_rejected_key_fails_the_build_rather_than_emptying_it(self):
        with self.assertRaises(RuntimeError):
            self.build(reject_above=0, error=KEY_REJECTED)

    def test_a_rate_limit_fails_the_build(self):
        with self.assertRaises(RuntimeError):
            self.build(reject_above=0, error=RATE_LIMITED)

    def test_a_deployment_that_does_not_exist_fails_the_build(self):
        with self.assertRaises(RuntimeError):
            self.build(reject_above=0, error=NO_DEPLOYMENT)

    # ── and the unaffected case is untouched ─────────────────────────────────

    def test_a_deployment_with_room_builds_everything(self):
        """gpt-4o 2024-08-06 onwards: 16,384 output tokens, so nothing here is
        refused and there is no failure report at all."""
        count, _calls = self.build(reject_above=16_384)
        self.assertEqual(count, 3)
        self.assertEqual(self.docs(), ["CUS_ORD_IVC_FCT_kb.md",
                                       "CUS_RTN_FCT_kb.md", "DT_DMS_kb.md"])
        self.assertEqual(self.failures(), [])

    def test_the_wide_fact_asks_for_what_its_width_needs(self):
        """The number that makes this an Azure problem at all."""
        _count, calls = self.build(reject_above=16_384)
        asked = [tokens for stage, table, tokens in calls
                 if stage == "table_doc" and table == WIDE[0]]
        self.assertEqual(asked, [7_984])
        self.assertGreater(7_984, 4_096, "under the 2024-05-13 cap there is no bug")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
