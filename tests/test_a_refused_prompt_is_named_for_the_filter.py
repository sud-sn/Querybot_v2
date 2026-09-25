# -*- coding: utf-8 -*-
"""A prompt Azure's content filter refuses is reported as that, not as a setup fault.

Azure OpenAI answers a prompt its content filter refuses with HTTP 400, code
"content_filter", inner code "ResponsibleAIPolicyViolation". The Azure wrapper
(core/llm.py, _azure_openai_complete) turned every exception into "Azure OpenAI
error: ...  Check your endpoint URL, API key, and deployment name in Admin →
System" -- for a refused prompt, a rate limit and a timeout alike. The reader
saw "⚠️ AI error:" and the provider's raw JSON; the audit recorded "error"; an
admin went to check a key that was fine.

Now a refused prompt raises the content-filter error the filtered completion
already raises: the audit records "content_filtered", the knowledge-base build
still skips just that table, and the reader is told, in their language, that
the AI service's filter blocked the request. The configuration hint is kept for
the errors it fits -- a rejected key, a deployment or path that does not exist,
an endpoint that cannot be reached -- and dropped from a rate limit and a
timeout.

The SDK's client is replaced at the boundary with one that raises the exact
exceptions the openai SDK raises; everything in core/llm.py runs for real.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

# The store is conftest's scratch database for the run (tests/conftest.py);
# nothing here repoints or re-imports it, so modules imported before this one
# keep the same store object as every test after it.
import store  # noqa: E402

import core.llm as llm  # noqa: E402
from core.knowledge import _kb_complete  # noqa: E402
from core.llm import (  # noqa: E402
    LLMContentFilteredError,
    is_rate_limited,
    is_single_request_rejection,
    llm_complete,
)
from core.llm_audit import llm_audit_scope  # noqa: E402

ENDPOINT = "https://acme.openai.azure.com"
API_VERSION = "2024-02-01"
DEPLOYMENT = "gpt-4o-prod"
HINT = "Check your endpoint URL, API key, and deployment name"
URL = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}"


def _status_error(cls, status: int, body: dict):
    """An openai SDK status error, built as the SDK builds it from a response."""
    request = httpx.Request("POST", URL)
    response = httpx.Response(status, request=request, json={"error": body})
    return cls(f"Error code: {status} - {{'error': {body}}}", response=response, body=body)


def refused_prompt():
    return _status_error(openai.BadRequestError, 400, {
        "message": "The response was filtered due to the prompt triggering Azure OpenAI's content "
                   "management policy. Please modify your prompt and retry.",
        "type": None, "param": "prompt", "code": "content_filter", "status": 400,
        "innererror": {"code": "ResponsibleAIPolicyViolation",
                       "content_filter_result": {"violence": {"filtered": True, "severity": "medium"}}},
    })


def rejected_key():
    return _status_error(openai.AuthenticationError, 401, {
        "code": "401", "message": "Access denied due to invalid subscription key or wrong API endpoint."})


def missing_deployment():
    return _status_error(openai.NotFoundError, 404, {
        "code": "DeploymentNotFound", "message": "The API deployment for this resource does not exist."})


def rate_limited():
    return _status_error(openai.RateLimitError, 429, {
        "code": "429", "message": "Requests to the ChatCompletions_Create Operation have exceeded the "
                                  "token rate limit. Please retry after 20 seconds."})


def timed_out():
    return openai.APITimeoutError(request=httpx.Request("POST", URL))


def unreachable():
    return openai.APIConnectionError(message="Connection error.", request=httpx.Request("POST", URL))


class _Client:
    """The SDK's client: its one call raises what Azure answered."""

    def __init__(self, error):
        create = AsyncMock(side_effect=error)
        self.chat = type("Chat", (), {"completions": type("Completions", (), {"create": create})()})()


@contextlib.contextmanager
def _azure_answers(error):
    """Put the client in the real cache, keyed as the real code keys it."""
    key = ("azure", "key", ENDPOINT, API_VERSION, llm._llm_timeout_seconds(), llm._llm_max_retries())
    llm._llm_client_cache[key] = _Client(error)
    try:
        yield
    finally:
        llm._llm_client_cache.pop(key, None)


def _raised(error) -> BaseException:
    with _azure_answers(error):
        try:
            asyncio.run(llm._azure_openai_complete(
                "sys", "user", DEPLOYMENT, "key", 512, ENDPOINT, API_VERSION, 0.0))
        except Exception as exc:  # noqa: BLE001
            return exc
    raise AssertionError("the call returned")


class TestARefusedPrompt:

    def test_is_the_content_filter_error(self):
        raised = _raised(refused_prompt())
        assert isinstance(raised, LLMContentFilteredError)
        assert "content filter refused the prompt" in str(raised)
        assert DEPLOYMENT in str(raised)

    def test_does_not_send_the_admin_to_check_the_setup(self):
        assert HINT not in str(_raised(refused_prompt()))

    def test_is_recorded_as_filtered_in_the_audit(self):
        async def run():
            with _azure_answers(refused_prompt()), patch("store.log_llm_call") as log_call:
                with llm_audit_scope(account_id="acct_1", question="stock by warehouse", enabled=True,
                                     request_id="req1", component="sql_generation"):
                    with pytest.raises(LLMContentFilteredError):
                        await llm_complete(
                            system="sys", user="user", provider="azure_openai", model=DEPLOYMENT,
                            api_key="key", azure_endpoint=ENDPOINT, azure_api_version=API_VERSION)
            return log_call

        log_call = asyncio.run(run())
        log_call.assert_called_once()
        assert log_call.call_args.kwargs["status"] == "content_filtered"

    def test_the_knowledge_base_build_still_skips_only_that_table(self):
        async def run():
            with _azure_answers(refused_prompt()):
                return await _kb_complete(
                    "document for ITM_BAL_DLY_FCT", "sys", "user", "azure_openai", DEPLOYMENT, "key",
                    max_tokens=4096, azure_endpoint=ENDPOINT, azure_api_version=API_VERSION)

        text, reason = asyncio.run(run())
        assert text is None
        assert "ContentFiltered" in reason
        assert is_single_request_rejection(_raised(refused_prompt()))


class TestTheSetupHintIsKeptWhereItFits:

    @pytest.mark.parametrize("error", [rejected_key, missing_deployment, unreachable],
                             ids=["rejected key", "missing deployment", "unreachable endpoint"])
    def test_a_configuration_error_still_says_what_to_check(self, error):
        raised = _raised(error())
        assert type(raised) is RuntimeError
        assert HINT in str(raised)

    @pytest.mark.parametrize("error", [rate_limited, timed_out], ids=["rate limit", "timeout"])
    def test_a_rate_limit_or_a_timeout_does_not(self, error):
        raised = _raised(error())
        assert type(raised) is RuntimeError
        assert HINT not in str(raised)
        assert str(raised).startswith("Azure OpenAI error: ")

    def test_a_rate_limit_is_still_read_as_one(self):
        """The KB build waits and retries on it; the provider's own text,
        with its 429, is still in the message it reads."""
        assert is_rate_limited(_raised(rate_limited()))


class TestTheReaderIsTold:
    """_handle_query_impl end to end, the model and the warehouse stubbed at
    their boundaries: what the reader sees when the model call fails."""

    def _answer(self, error, lang: str) -> list[str]:
        import json

        import core.query_pipeline as qp
        from core.graph_autopopulate import auto_populate_from_schema
        from gateway.base import PlatformEvent

        sent: list[str] = []
        store.init_db()
        account = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account, "Test Ltd")
        tmp = tempfile.mkdtemp()

        def table(own_key, *columns):
            return {"columns": [{"name": n, "type": t, "nullable": False, "comment": ""} for n, t in columns],
                    "pk_columns": [own_key], "row_count": 1000, "comment": "", "schema": "MART", "database": "WH"}

        schema = {
            "WH.MART.ITM_BAL_DLY_FCT": table("ITM_BAL_DLY_FCT_KEY", ("ITM_BAL_DLY_FCT_KEY", "bigint"),
                                             ("WHS_DMS_KEY", "int"), ("ON_HND_QTY", "decimal(18,4)")),
            "WH.MART.WHS_DMS": table("WHS_DMS_KEY", ("WHS_DMS_KEY", "int"), ("WHS_DSC", "nvarchar")),
        }
        with open(os.path.join(tmp, "_schema.json"), "w", encoding="utf-8") as fh:
            json.dump(schema, fh)
        store.update_client_state(account, "READY", {"schema_dir": tmp})
        auto_populate_from_schema(account, tmp)
        store.save_metric(account, {
            "name": "Stock On Hand", "formula_type": "expression", "sql_template": "SUM(ON_HND_QTY)",
            "base_table": "MART.ITM_BAL_DLY_FCT", "synonyms": "stock on hand, stock",
            "description": "Units on hand",
        })
        known = {fqn.split(".", 1)[1] for fqn in schema}
        columns = {fqn.split(".", 1)[1]: {c["name"]: c["type"] for c in meta["columns"]}
                   for fqn, meta in schema.items()}

        class Retriever:
            last_retrieval_weak = False
            last_retrieval_unscored = False

            def retrieve(self, question, n=8, allowed_tables=None):
                return ["# MART.ITM_BAL_DLY_FCT\n| `WHS_DMS_KEY` | int |\n| `ON_HND_QTY` | decimal |"]

            def retrieve_fact_patterns(self, question, n=2, allowed_tables=None):
                return []

            def _is_global(self, doc):
                return False

        async def model(*args, **kwargs):
            raise error

        class Adapter:
            platform, session_id, thread_id, last_result_id = "portal", f"{account}:portal:u1", "t1", None

            async def send_message(self, event, text, **kwargs):
                sent.append(str(text))

            async def send_typing(self, *args, **kwargs):
                return None

            def add_to_history(self, **kwargs):
                return None

            def cache_result(self, *args, **kwargs):
                return None

        question = "stock on hand by warehouse"
        with contextlib.ExitStack() as stack:
            for mock in (
                patch.object(qp, "get_state", return_value={"state": "READY", "schema_dir": tmp, "kb_dir": tmp}),
                patch.object(qp, "get_client_db", return_value={
                    "db_type": "azure_sql", "credentials": {}, "name": "db", "id": 1}),
                patch.object(qp.store, "get_client", return_value={
                    "account_id": account, "state": "READY", "name": "T"}),
                patch.object(qp, "load_known_tables", return_value=known),
                patch.object(qp, "load_schema_columns", return_value=columns),
                patch.object(qp, "load_retriever", return_value=Retriever()),
                patch.object(qp, "llm_complete", model),
                patch.object(qp, "resolve_provider", return_value=("azure_openai", DEPLOYMENT, "key", {})),
                patch.object(qp, "_log_q", lambda *args, **kwargs: None),
                patch.object(qp, "retrieve_similar_examples", return_value=[]),
            ):
                stack.enter_context(mock)
            asyncio.run(qp._handle_query_impl(
                account, PlatformEvent(account, "u1", "c1", question, "portal"), Adapter(), question,
                {"id": 1, "role": "admin", "email": "u@x.com", "name": "U", "group_name": None, "lang": lang},
            ))
        return sent

    @pytest.mark.parametrize("lang, words", [
        ("en", "The AI service's content filter blocked this request"),
        ("fr", "Le filtre de contenu du service d'IA a bloqué cette demande"),
    ])
    def test_that_the_filter_blocked_it_in_their_language(self, lang, words):
        sent = self._answer(LLMContentFilteredError(
            f"Azure OpenAI's content filter refused the prompt for {DEPLOYMENT}: {refused_prompt()}"), lang)
        assert any(words in text for text in sent), sent
        assert not any("ResponsibleAIPolicyViolation" in text or "AI error" in text for text in sent), sent

    def test_any_other_failure_is_still_an_ai_error_with_its_detail(self):
        sent = self._answer(RuntimeError("Azure OpenAI error: Error code: 429 - rate limit"), "en")
        assert any(text.startswith("⚠️ AI error: Azure OpenAI error: Error code: 429") for text in sent), sent
