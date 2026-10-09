"""The workspace's own AI model, as the bootstrap's labeler.

The call goes through QueryBot's model layer exactly as every other call does:
the provider and model the workspace is configured with, temperature 0, inside
an audit scope that records what left (tables and columns named, no rows).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any, cast


def workspace_labeler(account_id: str, client: dict[str, Any], *, tables: list[str],
                      columns: list[str]) -> Callable[[str, str], str]:
    from core.llm import Provider, llm_complete, resolve_provider
    from core.llm_audit import llm_audit_scope

    name, model, api_key, extra = resolve_provider(client, purpose="kb")
    provider = cast(Provider, name)
    build_id = f"core2-labels-{uuid.uuid4().hex[:12]}"

    def complete(system: str, user: str) -> str:
        async def call() -> str:
            with llm_audit_scope(account_id=account_id, question="core2: name the tables QueryBot learned",
                                 enabled=bool(client.get("enable_llm_audit")), request_id=build_id,
                                 question_id=build_id, component="core2_labels",
                                 egress={"tables": tables, "columns": columns}):
                text, _, _ = await llm_complete(system, user, provider, model, api_key, max_tokens=6000,
                                                temperature=0.0, **extra)
                return text

        return asyncio.run(call())

    return complete


def workspace_planner(account_id: str, client: dict[str, Any], *, question: str,
                      question_id: str = "") -> Callable[[str, str], str]:
    """The workspace's AI as core2's planner: the question's provider, temperature 0, audited.

    The stable half (rules, schema, catalog) is marked as the cached prefix; only
    the question and its tail change from call to call. The calls carry the question's
    own id (the one its query_log row has), so what they cost is that question's cost.
    """
    from core.llm import Provider, llm_complete, resolve_provider
    from core.llm_audit import llm_audit_scope
    from core.prompt_cache import CachedPrompt

    name, model, api_key, extra = resolve_provider(client, purpose="query")
    provider = cast(Provider, name)
    request_id = f"core2-plan-{uuid.uuid4().hex[:12]}"

    def complete(stable: str, tail: str) -> str:
        async def call() -> str:
            with llm_audit_scope(account_id=account_id, question=question,
                                 enabled=bool(client.get("enable_llm_audit")), request_id=request_id,
                                 question_id=question_id or request_id, component="core2_planner",
                                 egress={"question": question}):
                text, _, _ = await llm_complete(CachedPrompt(stable=stable, volatile="Answer with one JSON object."),
                                                tail, provider, model, api_key, max_tokens=1500, temperature=0.0,
                                                **extra)
                return text

        return asyncio.run(call())

    return complete


def metric_writer(account_id: str, client: dict[str, Any], *, description: str) -> Callable[[str, str], str]:
    """The workspace's AI writing a metric from an admin's words: the model's fields as the
    cached prefix, the words after it; audited as a metric authoring call, and costed as one."""
    from core.llm import Provider, llm_complete, resolve_provider
    from core.llm_audit import llm_audit_scope
    from core.prompt_cache import CachedPrompt

    name, model, api_key, extra = resolve_provider(client, purpose="query")
    provider = cast(Provider, name)
    request_id = f"core2-metric-{uuid.uuid4().hex[:12]}"

    def complete(stable: str, tail: str) -> str:
        async def call() -> str:
            with llm_audit_scope(account_id=account_id, question=description,
                                 enabled=bool(client.get("enable_llm_audit")), request_id=request_id,
                                 question_id=request_id, component="core2_metric_authoring",
                                 egress={"question": description}):
                text, _, _ = await llm_complete(CachedPrompt(stable=stable, volatile="Answer with one JSON object."),
                                                tail, provider, model, api_key, max_tokens=1200, temperature=0.0,
                                                **extra)
                return text

        return asyncio.run(call())

    return complete
