"""A model that says more before its plan still gets to finish the plan.

The planner's call allowed 1,500 output tokens: enough for a model that writes a plan in about 250. Another wrote two
to three times that (a line of reasoning, then the JSON), and on a bank test pack one plan in about three hundred
reached the ceiling: the reader got an error card instead of an answer. The ceiling is only what may be written,
never what is paid for, so a plan of a few thousand tokens is let through.

The model layer is stood in for: it writes a plan of the length given, and cuts it off where a real provider would.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def wordy(monkeypatch):
    from core import llm

    asked = {}

    def write(tokens: int):
        async def llm_complete(system, user, provider, model, api_key, *, max_tokens, temperature=0.0, **_):
            asked["max_tokens"] = max_tokens
            if tokens > max_tokens:
                raise llm.LLMTruncatedError(f"truncated at the {max_tokens}-token ceiling")
            return "I read this as a total by region. " + '{"kind": "query"}', 100, tokens

        monkeypatch.setattr(llm, "llm_complete", llm_complete)
        monkeypatch.setattr(llm, "resolve_provider", lambda client, purpose="query": ("anthropic", "a-model", "k", {}))
        return asked

    return write


@pytest.mark.parametrize("tokens", [250, 1800, 3000])
def test_a_plan_written_at_length_is_read_whole(wordy, tokens):
    from core2.bootstrap.ai import workspace_planner

    asked = wordy(tokens)
    text = workspace_planner("acct-x", {}, question="Sales by region")("rules and catalog", "the question")
    assert text.endswith('{"kind": "query"}'), "the plan came back whole"
    assert asked["max_tokens"] >= tokens


def test_a_metric_written_at_length_is_read_whole(wordy):
    from core2.bootstrap.ai import metric_writer

    wordy(2500)
    assert metric_writer("acct-x", {}, description="net sales")("fields", "words").endswith("}")
