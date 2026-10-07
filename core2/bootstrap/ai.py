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
