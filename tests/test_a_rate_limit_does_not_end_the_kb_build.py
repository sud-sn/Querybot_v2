# -*- coding: utf-8 -*-
"""A quota window, hit at table three of fifteen, failed the whole build.

Azure OpenAI admits a request against a tokens-per-minute quota using the
``max_tokens`` the request ASKED for, not the tokens it ends up spending. The
knowledge-base build is therefore the workload in this product most likely to
exhaust one: two calls per table, issued back to back with no pacing, each
asking for a ceiling derived from the table's width -- 7,984 for EMCO's
121-column invoice fact, up to 16,000 for anything wider.

The SDK client is built with ``max_retries=1``, deliberately: a question in the
chat window has a reader watching a spinner, and the retry count multiplies the
180-second timeout into the real worst-case wait. So one 429 that outlived that
single retry propagated out of ``_kb_complete``, out of ``build_kb``'s per-table
loop -- which has no handler of its own -- and failed the build at whichever
table happened to cross the line. Every table after it went undocumented, and
``_build_failures.json``, written after the loop, was never written.

A rate limit is the one error here that gets better by waiting, and that is
exactly what made it awkward: it is neither of ``is_single_request_rejection``'s
two cases. It is not about this one document -- the next table fails the same
way, so containing it and carrying on would produce a knowledge base full of
holes. But unlike a rejected key it stops being true. So it gets its own
predicate and its own treatment: wait, and try the SAME call again.

Three waits of 5, 15 and 30 seconds, budgeted per LABEL rather than per attempt.
Azure's quota refills on a one-minute window, so a build that crossed the line
early in one is usually through after the first wait. Bounded rather than
exponential-forever because a provider that is simply unreachable must still
fail the build in under a minute -- waiting is only correct while there is a
quota to wait for. And the waits poll the stop event, because ``build_kb``
checks it only between tables and an admin's Stop button that does nothing for
thirty seconds is a broken button.

The query path deliberately does NOT do any of this. Asserted below, because
the tempting next step after this commit is to "make it consistent".
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.knowledge as knowledge  # noqa: E402
import core.llm as llm  # noqa: E402
from core.knowledge import (  # noqa: E402
    _KB_DOC_MAX_TOKENS,
    _KB_RATE_LIMIT_WAITS,
    _kb_complete,
    kb_doc_token_budget,
)
from core.llm import is_rate_limited, is_single_request_rejection  # noqa: E402

AZURE_KWARGS = {
    "azure_endpoint": "https://emco.openai.azure.com",
    "azure_api_version": "2024-02-01",
}

# Real provider text, pinned as literals: the predicate reads the message, so
# the message is the contract.
def _wrapped(message: str) -> RuntimeError:
    try:
        raise ValueError(f"Error code: {message}")
    except ValueError as sdk_error:
        try:
            raise RuntimeError(f"Azure OpenAI error: {sdk_error}") from sdk_error
        except RuntimeError as wrapped:
            return wrapped


RATE_LIMITED = _wrapped(
    "429 - Requests to the ChatCompletions_Create Operation under Azure OpenAI "
    "API have exceeded token rate limit of your current quota. Please retry "
    "after 13 seconds.")
QUOTA_EXCEEDED = _wrapped(
    "429 - {'error': {'code': '429', 'message': 'You exceeded your current "
    "quota, please check your plan and billing details.'}}")
# A gateway in front of the resource that passes the status through without the
# provider's body. Nothing in it but the code, which is why the code is checked.
RATE_LIMITED_TERSE = _wrapped("429 - Too Many Requests")
RATE_LIMITED_BARE = _wrapped("429")
KEY_REJECTED = _wrapped("401 - Access denied due to invalid subscription key")
NO_DEPLOYMENT = _wrapped("404 - {'error': {'code': 'DeploymentNotFound'}}")
CEILING_TOO_HIGH = _wrapped(
    "400 - {'error': {'message': 'max_tokens must be less than or equal to 4096'}}")
SERVER_ERROR = _wrapped("500 - InternalServerError")

# Same shape as the real constant, fast enough to execute. The real values are
# asserted for shape in TestThePatienceIsBounded.
FAST_WAITS = (0.01, 0.02, 0.03)


class TestWhichErrorsGetBetterByWaiting:

    @pytest.mark.parametrize("exc", [RATE_LIMITED, QUOTA_EXCEEDED,
                                     RATE_LIMITED_TERSE, RATE_LIMITED_BARE])
    def test_a_rate_limit_does(self, exc):
        assert is_rate_limited(exc)

    @pytest.mark.parametrize("exc", [KEY_REJECTED, NO_DEPLOYMENT,
                                     CEILING_TOO_HIGH, SERVER_ERROR])
    def test_nothing_else_does(self, exc):
        assert not is_rate_limited(exc)

    def test_a_rate_limit_is_not_a_per_request_rejection(self):
        """The two predicates must not both claim it. Containing a rate limit
        per table is how a build finishes with thirty-five holes in it."""
        assert not is_single_request_rejection(RATE_LIMITED)
        assert not is_single_request_rejection(QUOTA_EXCEEDED)

    def test_and_a_per_request_rejection_is_not_a_rate_limit(self):
        assert not is_rate_limited(CEILING_TOO_HIGH)
        assert is_single_request_rejection(CEILING_TOO_HIGH)


class _Provider:
    """A provider that rate-limits the first `refusals` calls, then answers."""

    def __init__(self, refusals: int, error=RATE_LIMITED):
        self.refusals = refusals
        self.error = error
        self.ceilings: list[int] = []

    def install(self):
        async def complete(system, user, model, api_key, max_tokens,
                           endpoint, api_version, temperature=0.0):
            self.ceilings.append(max_tokens)
            if len(self.ceilings) <= self.refusals:
                raise self.error
            return "## Overview\nfinished\n", 100, 200
        return patch.object(llm, "_azure_openai_complete", complete)


class TestOneCompletionWaitsItOut(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self._patch = patch.object(knowledge, "_KB_RATE_LIMIT_WAITS", FAST_WAITS)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    async def _complete(self, provider, *, columns=121, stop_event=None):
        with provider.install():
            return await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "azure_openai",
                "emco-prod", "key", max_tokens=kb_doc_token_budget(columns),
                stop_event=stop_event, **AZURE_KWARGS)

    async def test_one_rate_limit_costs_a_wait_not_the_document(self):
        provider = _Provider(refusals=1)
        text, reason = await self._complete(provider)
        self.assertEqual(text, "## Overview\nfinished\n")
        self.assertEqual(reason, "")
        self.assertEqual(len(provider.ceilings), 2)

    async def test_the_retry_asks_for_the_same_ceiling(self):
        """A rate limit says nothing about the ceiling. Advancing to the cap
        would spend the TRUNCATION retry on a quota problem, so a document that
        then genuinely needed more room would have no attempt left."""
        provider = _Provider(refusals=1)
        await self._complete(provider)
        self.assertEqual(provider.ceilings, [7_984, 7_984])

    async def test_and_a_truncation_after_that_still_gets_the_cap(self):
        """The two retries are independent budgets: riding out a quota window
        must not cost the room a wide table needs."""
        seen: list[int] = []

        async def rate_limited_then_truncated(system, user, model, api_key,
                                              max_tokens, endpoint,
                                              api_version, temperature=0.0):
            seen.append(max_tokens)
            if len(seen) == 1:
                raise RATE_LIMITED
            if max_tokens < _KB_DOC_MAX_TOKENS:
                raise llm.LLMTruncatedError("truncated", text="## partial")
            return "## Overview\nfinished\n", 100, 200

        with patch.object(llm, "_azure_openai_complete",
                          rate_limited_then_truncated):
            text, reason = await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "azure_openai",
                "emco-prod", "key", max_tokens=kb_doc_token_budget(121),
                **AZURE_KWARGS)
        self.assertEqual(text, "## Overview\nfinished\n")
        self.assertEqual(seen, [7_984, 7_984, _KB_DOC_MAX_TOKENS])

    async def test_the_patience_runs_out(self):
        """A provider that is simply unreachable must not hold the build."""
        provider = _Provider(refusals=99)
        with self.assertRaises(RuntimeError):
            await self._complete(provider)
        self.assertEqual(len(provider.ceilings), len(FAST_WAITS) + 1)

    async def test_the_budget_is_per_label_not_per_ceiling(self):
        """Otherwise a two-ceiling label gets twice the patience of a one-ceiling
        one, for no reason anybody chose."""
        provider = _Provider(refusals=99)
        with self.assertRaises(RuntimeError):
            await self._complete(provider, columns=121)
        at_cap = _Provider(refusals=99)
        with self.assertRaises(RuntimeError):
            # A label already at the cap has only one ceiling to try.
            with at_cap.install():
                await _kb_complete(
                    "vocabulary", "sys", "user", "azure_openai", "emco-prod",
                    "key", max_tokens=_KB_DOC_MAX_TOKENS, **AZURE_KWARGS)
        self.assertEqual(len(provider.ceilings), len(at_cap.ceilings))

    async def test_stop_during_a_wait_gives_the_build_back_at_once(self):
        stop = threading.Event()
        stop.set()
        provider = _Provider(refusals=99)
        started = time.time()
        with self.assertRaises(RuntimeError):
            await self._complete(provider, stop_event=stop)
        self.assertEqual(len(provider.ceilings), 1, "it retried after Stop")
        self.assertLess(time.time() - started, 1.0)

    async def test_a_wait_is_polled_rather_than_slept_through(self):
        """build_kb checks the stop event only between tables, so a wait that
        ignored it would leave the Stop button dead for its whole length."""
        stop = threading.Event()

        async def stop_shortly():
            await asyncio.sleep(0.02)
            stop.set()

        with patch.object(knowledge, "_KB_RATE_LIMIT_WAITS", (3.0, 3.0, 3.0)):
            provider = _Provider(refusals=99)
            started = time.time()
            waiter = asyncio.create_task(stop_shortly())
            with self.assertRaises(RuntimeError):
                await self._complete(provider, stop_event=stop)
            await waiter
        elapsed = time.time() - started
        self.assertLess(elapsed, 2.5, f"the wait ignored the stop event ({elapsed:.1f}s)")


class TestThePatienceIsBounded:
    """The real constant, not the fast one the tests run with."""

    def test_three_waits_covering_about_a_minute(self):
        assert len(_KB_RATE_LIMIT_WAITS) == 3
        assert 45 <= sum(_KB_RATE_LIMIT_WAITS) <= 75

    def test_they_get_longer(self):
        assert list(_KB_RATE_LIMIT_WAITS) == sorted(_KB_RATE_LIMIT_WAITS)

    def test_the_first_is_short_enough_to_be_worth_trying(self):
        assert _KB_RATE_LIMIT_WAITS[0] <= 10


def schema_md(table: str, column_count: int) -> str:
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


TABLES = (("CUS_ORD_IVC_FCT", 121), ("CUS_RTN_FCT", 40), ("DT_DMS", 12))


class TestTheWholeBuild(unittest.TestCase):
    """build_kb over a real schema directory, provider as the only boundary."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="qb-kb-429-")
        self.schema_dir = os.path.join(self.root, "schema")
        self.kb_dir = os.path.join(self.root, "kb")
        for path in (self.schema_dir, self.kb_dir):
            os.makedirs(path)
        for table, columns in TABLES:
            with open(os.path.join(self.schema_dir, f"{table}.md"), "w") as handle:
                handle.write(schema_md(table, columns))
        self._patch = patch.object(knowledge, "_KB_RATE_LIMIT_WAITS", FAST_WAITS)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, *, refuse_calls=()):
        """`refuse_calls` are the 1-based call numbers that get a 429."""
        calls: list[int] = []

        async def fake(system, user, provider, model, api_key,
                       max_tokens=1024, **kw):
            calls.append(max_tokens)
            if len(calls) in refuse_calls:
                raise RATE_LIMITED
            return "## Overview\nfinished\n", 100, 200

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

    def test_a_quota_window_mid_build_no_longer_ends_it(self):
        """The defect: one 429 at table two, and tables two and three went
        undocumented along with the failure report."""
        count, _calls = self.build(refuse_calls=(3, 4))
        self.assertEqual(count, 3)
        self.assertEqual(self.docs(), ["CUS_ORD_IVC_FCT_kb.md",
                                       "CUS_RTN_FCT_kb.md", "DT_DMS_kb.md"])

    def test_and_nothing_is_recorded_as_a_failure(self):
        """A table that arrived after a wait is not a degraded table."""
        self.build(refuse_calls=(3, 4))
        self.assertEqual(self.failures(), [])

    def test_a_quota_that_never_clears_still_fails_the_build(self):
        """Waiting is only right while there is a quota to wait for. A holed
        knowledge base reported as a success is the worse outcome."""
        with self.assertRaises(RuntimeError):
            self.build(refuse_calls=tuple(range(1, 200)))

    def test_the_first_table_is_retried_rather_than_skipped(self):
        count, calls = self.build(refuse_calls=(1,))
        self.assertEqual(count, 3)
        # The vocabulary call is first, and it asked twice for the same budget.
        self.assertEqual(calls[0], calls[1])


class TestTheQueryPathStillDoesNotWait(unittest.IsolatedAsyncioTestCase):
    """A reader is watching a spinner. Thirty seconds of silence is worse than
    a refusal they can act on, and the SDK's own single retry already covers a
    momentary spike."""

    async def test_a_rate_limit_reaches_the_caller_immediately(self):
        calls: list[int] = []

        async def rate_limited(system, user, model, api_key, max_tokens,
                               endpoint, api_version, temperature=0.0):
            calls.append(max_tokens)
            raise RATE_LIMITED

        started = time.time()
        with patch.object(llm, "_azure_openai_complete", rate_limited):
            with self.assertRaises(RuntimeError):
                await llm.llm_complete(
                    "sys", "user", "azure_openai", "emco-prod", "key",
                    max_tokens=768, **AZURE_KWARGS)
        self.assertEqual(len(calls), 1, "the query path retried a rate limit")
        self.assertLess(time.time() - started, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
