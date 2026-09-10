"""
core/result_conversation.py
───────────────────────────
Answering the reader when they talk to a result instead of commanding it.

The result card understands seven operations — filter, aggregate, sort, limit,
exclude, chart, compare — and until now that was the whole of what it could
hear. Everything else was met with the same message: "I could not answer that.
The current result only has these columns…". That included the most ordinary
question anyone asks about a table on screen ("what is the total?"), the most
ordinary follow-up ("why is that?"), the most ordinary courtesy ("thanks"),
and the most ordinary confusion ("what does this column mean?").

None of those are commands, and three of the four are not queries either: they
are answerable from the result the reader is already looking at. This module
answers them, and it is reached only after both the governed cache engine and
the production-database fallback have declined — so a question that can be
answered with data still is, and this never competes with a real query.

What the model receives
───────────────────────
A statistical brief of the result, exactly as core/insight.py builds it for
every other narrative, and never the row set. The brief is not value-free —
core/insight.py says so at length — so the reply carries its own provenance
note rather than borrowing the governed cache engine's stronger claim.

A regulated tenant reaches no model at all: result_llm_features_allowed
refuses, the refusal is recorded, and the caller falls back to the
deterministic hint it would have sent anyway.
"""

from __future__ import annotations

import logging

log = logging.getLogger("querybot.result_conversation")

# Long enough for a paragraph, short enough that nothing here turns into an
# essay about a four-row table.
_MAX_TOKENS = 400


async def converse_about_result(
    question: str,
    *,
    rows: list[dict],
    result_question: str,
    account_id: str,
    provider: str,
    model: str,
    api_key: str,
    sql: str = "",
    history: list[dict] | None = None,
    business_context: str = "",
    **extra_kwargs,
) -> str:
    """Prose answering `question` about `rows`, or "" if it cannot be answered.

    An empty string is the caller's signal to send the deterministic hint
    instead: a regulated tenant, a provider failure, or a model that returned
    nothing. It is never an error the reader sees, because by the time this
    runs the reader is owed an answer of some kind either way.
    """
    if not question.strip() or not rows:
        return ""

    from core.compliance.policy_engine import result_llm_features_allowed

    if not result_llm_features_allowed(account_id):
        from core.llm_audit import record_llm_blocked
        record_llm_blocked(
            "result_conversation",
            "Conversational result reply blocked — regulated tenant, LLM "
            "never received a summary of the result.",
        )
        return ""

    from core.insight import (
        build_action_contract,
        build_insight_prompt_from_contract,
        compute_data_brief,
        parse_insight_response,
    )
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_component
    from core.response_builder import summarize_result_context

    try:
        context = summarize_result_context(rows, result_question, sql=sql)
        brief = compute_data_brief(
            rows,
            result_question,
            result_scope=context.get("result_scope"),
            context=context,
        )
        contract = build_action_contract(
            "converse", result_question, brief, follow_up=question,
        )
        system, user_msg = build_insight_prompt_from_contract(
            contract,
            follow_up=question,
            business_context=business_context,
            history=history,
        )
        with llm_audit_component("result_conversation"):
            raw, _tok_in, _tok_out = await llm_complete(
                system, user_msg, provider, model, api_key,
                max_tokens=_MAX_TOKENS, temperature=0.3,
                # Prose a person reads. A reply that stops mid-sentence is
                # worth more than the "I could not answer that" it replaces,
                # and nothing downstream parses it for a governed decision.
                allow_truncated=True,
            )
    except Exception as exc:
        # The reader gets the deterministic hint, which is what they would
        # have got anyway. Logged at warning because "always broken" and
        # "working perfectly" are indistinguishable from outside here.
        log.warning("Conversational result reply failed: %s", exc)
        return ""

    parsed = parse_insight_response(raw)
    # The prompt asks for prose, so the whole reply lands in `body`. A model
    # that reached for the card format anyway still has its headline kept
    # rather than dropped.
    headline = str(parsed.get("headline") or "").strip()
    body = str(parsed.get("body") or "").strip()
    if headline and headline not in body:
        return f"{headline} {body}".strip()
    return body or headline
