# -*- coding: utf-8 -*-
"""One knowledge-base request that times out skips its table, not the build.

Each model call has a time limit (180 seconds by default). A wide table's
document can take long enough to reach it, and a timeout was neither of the two
things the build knew how to contain -- a rejection of that one request, or a
rate limit to wait out -- so it propagated out of build_kb's loop and failed
the whole build at whichever table was slowest. Every table after it went
undocumented, and _build_failures.json, written after the loop, was not
written at all.

Now a timeout skips that one document -- the table, or its worked examples --
and says so in the failure report. Three in a row end the build: one slow
document is that document's problem, but three is a provider that is not
answering, and waiting out every remaining table's timeout would only make a
failed build slower. An answer in between starts the count again.

build_kb runs over a real schema directory with the provider as the only
boundary; the timeout is the real Azure wrapper's, built from the SDK's own
exception.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.knowledge as knowledge  # noqa: E402
import core.llm as llm  # noqa: E402
from core.knowledge import _kb_complete  # noqa: E402
from core.llm import is_timed_out  # noqa: E402

ENDPOINT = "https://acme.openai.azure.com"
API_VERSION = "2024-02-01"
AZURE_KWARGS = {"azure_endpoint": ENDPOINT, "azure_api_version": API_VERSION}
REQUEST = httpx.Request("POST", f"{ENDPOINT}/openai/deployments/prod/chat/completions")


def _through_the_azure_wrapper(sdk_error) -> Exception:
    """What core/llm.py raises when the SDK raises `sdk_error`."""
    create = AsyncMock(side_effect=sdk_error)
    client = type("C", (), {"chat": type("Chat", (), {
        "completions": type("Completions", (), {"create": create})()})()})()
    key = ("azure", "key", ENDPOINT, API_VERSION, llm._llm_timeout_seconds(), llm._llm_max_retries())
    llm._llm_client_cache[key] = client
    try:
        asyncio.run(llm._azure_openai_complete(
            "sys", "user", "prod", "key", 512, ENDPOINT, API_VERSION, 0.0))
    except Exception as exc:  # noqa: BLE001
        return exc
    finally:
        llm._llm_client_cache.pop(key, None)
    raise AssertionError("the call returned")


TIMED_OUT = _through_the_azure_wrapper(openai.APITimeoutError(request=REQUEST))
KEY_REJECTED = _through_the_azure_wrapper(openai.AuthenticationError(
    "Error code: 401 - invalid subscription key",
    response=httpx.Response(401, request=REQUEST, json={}), body={}))
RATE_LIMITED = _through_the_azure_wrapper(openai.RateLimitError(
    "Error code: 429 - token rate limit exceeded",
    response=httpx.Response(429, request=REQUEST, json={}), body={}))


class TestWhatATimeoutIs:

    def test_the_azure_wrappers_timeout(self):
        assert is_timed_out(TIMED_OUT)

    def test_anthropics_as_its_wrapper_words_it(self):
        sdk = anthropic.APITimeoutError(request=REQUEST)
        assert is_timed_out(RuntimeError(f"Anthropic API error: {sdk}"))

    def test_pythons_own(self):
        assert is_timed_out(TimeoutError())

    @pytest.mark.parametrize("exc", [KEY_REJECTED, RATE_LIMITED], ids=["rejected key", "rate limit"])
    def test_not_anything_else(self, exc):
        assert not is_timed_out(exc)


class TestOneRequest:

    def test_a_timeout_skips_that_document_and_says_why(self):
        async def run():
            with patch("core.llm.llm_complete", AsyncMock(side_effect=TIMED_OUT)):
                return await _kb_complete("document for ITM_BAL_DLY_FCT", "sys", "user",
                                          "azure_openai", "prod", "key", max_tokens=4288, **AZURE_KWARGS)

        text, reason = asyncio.run(run())
        assert text is None
        assert reason.startswith("timed out: RuntimeError: Azure OpenAI error: Request timed out.")

    def test_a_rejected_key_still_ends_it(self):
        async def run():
            with patch("core.llm.llm_complete", AsyncMock(side_effect=KEY_REJECTED)):
                await _kb_complete("document for ITM_BAL_DLY_FCT", "sys", "user",
                                   "azure_openai", "prod", "key", max_tokens=4288, **AZURE_KWARGS)

        with pytest.raises(RuntimeError, match="401"):
            asyncio.run(run())


def _schema_md(table: str, column_count: int) -> str:
    rows = "\n".join(f"| `COL_{i:03d}_QTY` | decimal(18) | No |  |" for i in range(column_count))
    return (f"# MART.{table}\n\n**Type:** BASE TABLE  **Schema:** MART\n\n"
            f"**SQL table name:** `MART.{table}`\n\n**Row count:** 5800  **Scale:** Medium\n\n"
            f"## Columns\n\n| Column | Type | Nullable | Distinct Values |\n"
            f"|--------|------|:--------:|-----------------|\n{rows}\n")


# Built in name order: call 1 is the business vocabulary, then each table's
# document and its worked examples -- ITM_BAL_DLY_FCT 2 and 3, ITM_BAL_PRD_FCT
# 4 and 5, WHS_DMS 6 and 7.
TABLES = (("ITM_BAL_DLY_FCT", 44), ("ITM_BAL_PRD_FCT", 67), ("WHS_DMS", 9))


class TestTheWholeBuild:

    @pytest.fixture(autouse=True)
    def dirs(self):
        self.root = tempfile.mkdtemp(prefix="qb-kb-timeout-")
        self.schema_dir = os.path.join(self.root, "schema")
        self.kb_dir = os.path.join(self.root, "kb")
        for path in (self.schema_dir, self.kb_dir):
            os.makedirs(path)
        for table, columns in TABLES:
            with open(os.path.join(self.schema_dir, f"{table}.md"), "w") as handle:
                handle.write(_schema_md(table, columns))
        yield
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self, *, time_out_calls=()):
        """`time_out_calls` are the 1-based calls that time out."""
        calls: list[int] = []

        async def provider(system, user, provider, model, api_key, max_tokens=1024, **kw):
            calls.append(max_tokens)
            if len(calls) in time_out_calls:
                raise TIMED_OUT
            return "## Overview\nfinished\n", 100, 200

        with patch("core.llm.llm_complete", side_effect=provider), \
                patch.object(knowledge, "_embed_kb_files_qdrant", return_value=None):
            return asyncio.run(knowledge.build_kb(
                schema_dir=self.schema_dir, kb_dir=self.kb_dir, chroma_dir="acct",
                business_desc="a plumbing distributor", provider="azure_openai", model="prod",
                api_key="key", extra_kwargs=dict(AZURE_KWARGS), account_id="acct"))

    def docs(self):
        return sorted(n for n in os.listdir(self.kb_dir) if n.endswith("_kb.md") and not n.startswith("_"))

    def failures(self):
        path = os.path.join(self.kb_dir, "_build_failures.json")
        if not os.path.exists(path):
            return []
        with open(path) as handle:
            return [(f["table"], f["stage"], f["reason"].split(":")[0]) for f in json.load(handle)]

    def test_a_table_whose_document_times_out_is_skipped_and_the_rest_are_built(self):
        count = self.build(time_out_calls=(2,))
        assert count == 2
        assert self.docs() == ["ITM_BAL_PRD_FCT_kb.md", "WHS_DMS_kb.md"]
        assert self.failures() == [("ITM_BAL_DLY_FCT", "table_doc", "timed out")]

    def test_worked_examples_that_time_out_leave_the_document(self):
        count = self.build(time_out_calls=(3,))
        assert count == 3
        assert "ITM_BAL_DLY_FCT_kb.md" in self.docs()
        assert self.failures() == [("ITM_BAL_DLY_FCT", "query_examples", "timed out")]

    def test_the_vocabulary_timing_out_does_not_stop_the_tables(self):
        count = self.build(time_out_calls=(1,))
        assert count == 3
        assert not os.path.exists(os.path.join(self.kb_dir, "_business_kb.md"))

    @pytest.mark.parametrize("calls", [(1, 2, 3), (3, 4, 5)],
                             ids=["vocabulary and two documents", "examples and two documents"])
    def test_three_in_a_row_end_the_build(self, calls):
        with pytest.raises(RuntimeError, match="3 knowledge-base requests in a row timed out") as caught:
            self.build(time_out_calls=calls)
        assert "QUERYBOT_LLM_TIMEOUT_SECONDS" in str(caught.value)

    def test_an_answer_between_them_starts_the_count_again(self):
        count = self.build(time_out_calls=(2, 4, 6))
        assert count == 2
        assert self.failures() == [
            ("ITM_BAL_DLY_FCT", "table_doc", "timed out"),
            ("ITM_BAL_PRD_FCT", "query_examples", "timed out"),
            ("WHS_DMS", "query_examples", "timed out"),
        ]
