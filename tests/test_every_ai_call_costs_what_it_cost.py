"""Every AI call is recorded with its tokens as the provider billed them, at the model's own price.

A question's cost was ``tokens_in x input + tokens_out x output`` for its first SQL call
only, with one rate pair per model:

- cached input was never priced (Anthropic reports reads and writes apart from
  ``input_tokens``; Azure counts cached tokens inside ``prompt_tokens`` at a discount);
- the new core recorded 0 tokens and $0 for every question, and Learn nothing;
- every model the table did not know (Azure deployment names, local models) was
  charged gpt-4o's rates; several built-in rates were stale.

Now every call through ``llm_complete`` is one row in llm_usage, tied to the question
its audit scope names, and priced from the price table (core/llm_prices.py). A model
with no price records no cost rather than a guessed one. Run against a fresh store with
fake provider clients: the provider's response is the only thing invented.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from core.llm_prices import BUILT_IN, Price, Usage, cost_of, usage_from_anthropic, usage_from_openai


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from admin import routes
    from core import llm_prices

    # System settings are encrypted: under a key of the test's own, whichever store
    # module an earlier test left bound.
    for module in {id(m): m for m in (store, routes.store)}.values():
        for namespace in (module.crypto.__dict__, module.config_store.decrypt.__globals__):
            monkeypatch.setitem(namespace, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    llm_prices.forget_cached_prices()
    yield store
    llm_prices.forget_cached_prices()


def _account(store) -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


# ── Reading what the provider billed ──────────────────────────────────────────

def test_anthropic_reads_and_writes_are_apart_from_input():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=9000,
                            cache_creation_input_tokens=0, cache_creation=None)
    assert usage_from_anthropic(usage) == Usage(input=1000, cached_input=9000, output=500)


def test_anthropic_writes_split_by_ttl_when_the_response_says_so():
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                            cache_creation_input_tokens=3000,
                            cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=3000))
    assert usage_from_anthropic(usage).cache_write_1h == 3000


def test_azure_cached_tokens_are_inside_the_prompt():
    usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=100,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=1024),
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    assert usage_from_openai(usage) == Usage(input=176, cached_input=1024, output=100)


def test_a_server_that_sends_no_details_gives_plain_counts():
    assert usage_from_openai(SimpleNamespace(prompt_tokens=50, completion_tokens=7)) == Usage(input=50, output=7)


# ── Pricing ───────────────────────────────────────────────────────────────────

def test_cached_input_and_cache_writes_are_priced_at_their_own_rates():
    sonnet = BUILT_IN[("anthropic", "claude-sonnet-4-6")]
    usage = Usage(input=1000, cached_input=9000, cache_write_1h=2000, output=500)
    assert cost_of(usage, sonnet) == pytest.approx((1000 * 3 + 9000 * 0.30 + 2000 * 6 + 500 * 15) / 1e6)


def test_the_built_in_prices_are_todays():
    assert BUILT_IN[("anthropic", "claude-opus-4-5")].input == 5.00, "it was charged at 15"
    assert BUILT_IN[("anthropic", "claude-haiku-4-5")].output == 5.00, "it was charged at 4"


def test_a_long_prompt_moves_to_the_higher_rate():
    haiku = BUILT_IN[("anthropic", "claude-haiku-5-5")]
    assert cost_of(Usage(input=200_000, output=1000), haiku) == pytest.approx((200_000 * 0.50 + 1000 * 2.50) / 1e6)


def test_a_model_with_no_price_costs_nothing_known(fresh_store):
    from core.llm_prices import price_for

    assert price_for("azure_openai", "prod-4o").price is None
    assert price_for("openai", "gpt-4o").price is None
    assert cost_of(Usage(input=10, output=10), None) is None


def test_a_local_model_is_free(fresh_store):
    from core.llm_prices import price_for

    assert cost_of(Usage(input=10_000, output=10_000), price_for("local", "llama3.1:8b").price) == 0.0


def test_an_azure_deployment_is_priced_as_the_model_behind_it(fresh_store):
    from core.llm_prices import forget_cached_prices, price_for

    fresh_store.set_system("azure_deployment_models", json.dumps({"prod-4o": "gpt-4o"}))
    fresh_store.set_system("azure_deployment_type", "data_zone")
    fresh_store.save_llm_price(provider="azure_openai", model="gpt-4o", deployment_type="data_zone",
                               input=3.0, cached_input=1.5, output=12.0)
    forget_cached_prices()
    priced = price_for("azure_openai", "prod-4o")
    assert (priced.model, priced.deployment_type) == ("gpt-4o", "data_zone")
    assert cost_of(Usage(input=1000, cached_input=1000, output=1000), priced.price) == pytest.approx(16.5 / 1000)


def test_the_admins_price_wins_over_the_built_in_one(fresh_store):
    from core.llm_prices import forget_cached_prices, price_for

    fresh_store.save_llm_price(provider="anthropic", model="claude-sonnet-4-6", input=2.0, output=10.0)
    forget_cached_prices()
    assert price_for("anthropic", "claude-sonnet-4-6").price == Price(
        input=2.0, output=10.0, source="Set by an admin", checked=price_for("anthropic", "claude-sonnet-4-6").price.checked)


# ── Recorded through the real call ───────────────────────────────────────────

def _anthropic_reply(monkeypatch, *, stop="end_turn", cached=9000):
    from core import llm

    resp = SimpleNamespace(content=[SimpleNamespace(text="answer")], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=cached,
                                                 cache_creation_input_tokens=0, cache_creation=None))

    async def create(**kwargs):
        return resp

    monkeypatch.setattr(llm, "_get_anthropic_client", lambda key: SimpleNamespace(messages=SimpleNamespace(create=create)))


def _azure_reply(monkeypatch):
    from core import llm

    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=100,
                                                 prompt_tokens_details=SimpleNamespace(cached_tokens=1024)))

    async def create(**kwargs):
        return resp

    monkeypatch.setattr(llm, "_get_azure_client",
                        lambda *a: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))


def _ask(account_id, question_id, provider="anthropic", model="claude-sonnet-4-6", component="sql", **kw):
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_scope

    async def run():
        with llm_audit_scope(account_id=account_id, question="q", enabled=False, question_id=question_id,
                             component=component):
            return await llm_complete("system", "user", provider, model, "key", **kw)
    return asyncio.run(run())


def test_a_call_is_recorded_and_its_question_costs_it(fresh_store, monkeypatch):
    _anthropic_reply(monkeypatch)
    account = _account(fresh_store)
    _ask(account, "q-1")
    used = fresh_store.question_usage("q-1")
    assert (used["calls"], used["input_tokens"], used["cached_input_tokens"], used["output_tokens"]) == (1, 1000, 9000, 500)
    assert used["cost_usd"] == pytest.approx((1000 * 3 + 9000 * 0.30 + 500 * 15) / 1e6)

    fresh_store.log_query(account, "q", "SELECT 1", question_id="q-1", llm_provider="anthropic",
                          llm_model="claude-sonnet-4-6", tokens_in=1000, tokens_out=500)
    row = fresh_store.get_recent_queries(account)[0]
    assert row["cost_source"] == "usage" and row["cost_usd"] == pytest.approx(used["cost_usd"])


def test_calls_after_the_answer_are_the_questions_too(fresh_store, monkeypatch):
    _anthropic_reply(monkeypatch)
    account = _account(fresh_store)
    _ask(account, "q-2")
    fresh_store.log_query(account, "q", "SELECT 1", question_id="q-2")
    _ask(account, "q-2", component="narrative")
    row = fresh_store.get_recent_queries(account)[0]
    assert row["usage"]["calls"] == 2
    assert {s["component"] for s in row["usage"]["steps"]} == {"sql", "narrative"}


def test_totals_count_each_call_once(fresh_store, monkeypatch):
    _anthropic_reply(monkeypatch)
    account = _account(fresh_store)
    _ask(account, "q-3")
    fresh_store.log_query(account, "q", "SELECT 1", question_id="q-3")
    one = fresh_store.question_usage("q-3")["cost_usd"]
    # an old row, written before calls were recorded, keeps its estimate
    fresh_store.log_query(account, "old", "SELECT 1", llm_provider="anthropic", llm_model="claude-sonnet-4-6",
                          tokens_in=1_000_000, tokens_out=0)
    stats = fresh_store.get_query_stats(account)
    assert stats["total_cost_usd"] == pytest.approx(one + 3.0)
    assert fresh_store.get_monthly_token_status(account)["total_tokens"] == 1_000_000 + 10_500


def test_an_azure_call_is_priced_with_its_cached_tokens(fresh_store, monkeypatch):
    from core.llm_prices import forget_cached_prices

    _azure_reply(monkeypatch)
    account = _account(fresh_store)
    fresh_store.set_system("azure_deployment_models", json.dumps({"prod-4o": "gpt-4o"}))
    fresh_store.save_llm_price(provider="azure_openai", model="gpt-4o", deployment_type="global",
                               input=2.0, cached_input=1.0, output=8.0)
    forget_cached_prices()
    _ask(account, "q-4", provider="azure_openai", model="prod-4o", azure_endpoint="https://x.openai.azure.com")
    used = fresh_store.question_usage("q-4")
    assert (used["input_tokens"], used["cached_input_tokens"]) == (176, 1024)
    assert used["cost_usd"] == pytest.approx((176 * 2 + 1024 * 1 + 100 * 8) / 1e6)


def test_a_call_with_no_price_is_counted_and_says_so(fresh_store, monkeypatch):
    _azure_reply(monkeypatch)
    account = _account(fresh_store)
    _ask(account, "q-5", provider="azure_openai", model="unpriced-deployment", azure_endpoint="https://x.openai.azure.com")
    used = fresh_store.question_usage("q-5")
    assert used["calls"] == 1 and used["unpriced_calls"] == 1 and used["cost_usd"] == 0.0
    assert fresh_store.get_query_stats(account)["unpriced_calls"] == 1


def test_a_cut_off_answer_was_still_billed(fresh_store, monkeypatch):
    from core.llm import LLMTruncatedError

    _anthropic_reply(monkeypatch, stop="max_tokens")
    account = _account(fresh_store)
    with pytest.raises(LLMTruncatedError):
        _ask(account, "q-6")
    assert fresh_store.question_usage("q-6")["calls"] == 1


def test_the_new_cores_planner_calls_carry_the_questions_id(monkeypatch):
    from core import llm, llm_audit
    from core2.bootstrap import ai

    seen = []

    async def complete(*args, **kwargs):
        seen.append(llm_audit.get_current_llm_audit_scope()["question_id"])
        return "{}", 1, 1

    monkeypatch.setattr(llm, "llm_complete", complete)
    monkeypatch.setattr(llm, "resolve_provider", lambda client, purpose="query": ("anthropic", "m", "k", {}))
    ai.workspace_planner("acct", {}, question="q", question_id="c2-abc")("stable", "tail")
    assert seen == ["c2-abc"]
