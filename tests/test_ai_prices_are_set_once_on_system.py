"""Model prices are set once, on Admin → System, and every page shows what the calls cost.

Prices lived on each workspace's Billing page as one input/output pair per model, with
a stale built-in list behind them, and an Azure deployment was priced by its own name
(so as nothing, then as gpt-4o's fallback). Now:

- System → AI prices lists the built-in list prices and the admin's own, with where each
  came from; the admin adds or changes one there, for every workspace;
- the models that answered with no price are named, with a button to set the price, and
  setting it costs the calls they already made (a priced call keeps its cost: a new
  price counts from now on);
- fetching Azure deployments remembers which model each runs, and the admin can say so
  by hand and pick the deployment type, so the calls are priced as the model behind them;
- the question log shows each question's cost with its calls, tokens and steps, and the
  Learned page what the last Learn cost.

The admin's routes run for real against a fresh store; only the providers' responses
are invented.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest


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


@pytest.fixture
def admin(fresh_store):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import core2_routes, routes

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True), patch.object(core2_routes, "_is_auth", return_value=True):
        yield TestClient(app)


def _account(store) -> str:
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _reply(monkeypatch, provider: str):
    from core import llm

    if provider == "azure_openai":
        resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=100,
                                                     prompt_tokens_details=SimpleNamespace(cached_tokens=1024)))

        async def create(**kwargs):
            return resp

        monkeypatch.setattr(llm, "_get_azure_client",
                            lambda *a: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    else:
        resp = SimpleNamespace(content=[SimpleNamespace(text="answer")], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=9000,
                                                     cache_creation_input_tokens=0, cache_creation=None))

        async def create(**kwargs):
            return resp

        monkeypatch.setattr(llm, "_get_anthropic_client",
                            lambda key: SimpleNamespace(messages=SimpleNamespace(create=create)))


def _call(account_id, question_id, provider, model, component="sql_generation"):
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_scope

    extra = {"azure_endpoint": "https://x.openai.azure.com"} if provider == "azure_openai" else {}

    async def run():
        with llm_audit_scope(account_id=account_id, question="q", enabled=False, question_id=question_id,
                             component=component):
            return await llm_complete("system", "user", provider, model, "key", **extra)
    return asyncio.run(run())


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def _prices_card(html: str) -> str:
    start = html.index('id="ai-prices"')
    return html[start:html.index("<!-- ── Services", start)]


# ── Setting a price on System ────────────────────────────────────────────────

def test_an_azure_deployment_with_no_price_is_named_and_pricing_it_costs_its_calls(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "azure_openai")
    account = _account(fresh_store)
    _call(account, "q-az", "azure_openai", "prod-4o")
    assert fresh_store.question_usage("q-az")["unpriced_calls"] == 1

    card = _text(_prices_card(admin.get("/admin/system").text))
    assert "No price set for 1 model used in the last 30 days" in card
    assert "prod-4o · Azure OpenAI · Global — 1 call" in card

    # The deployment runs gpt-4o, in a Data Zone deployment.
    r = admin.post("/admin/system/azure-pricing", data={"deployment_type": "data_zone", "deployment": ["prod-4o"],
                                                         "deployment_model": ["gpt-4o"]}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].endswith("saved=azure-pricing#ai-prices")
    card = _text(_prices_card(admin.get("/admin/system").text))
    assert "gpt-4o · Azure OpenAI · Data Zone — 1 call" in card, "the call is now asked about as the model it ran"

    r = admin.post("/admin/system/prices/save", data={
        "provider": "azure_openai", "model": "gpt-4o", "deployment_type": "data_zone",
        "input_rate": "2.75", "cached_rate": "1.375", "output_rate": "11"}, follow_redirects=False)
    assert "saved=prices" in r.headers["location"] and "costed=1" in r.headers["location"]

    used = fresh_store.question_usage("q-az")
    assert used["unpriced_calls"] == 0
    assert used["cost_usd"] == pytest.approx((176 * 2.75 + 1024 * 1.375 + 100 * 11) / 1e6)
    page = admin.get("/admin/system?saved=prices&costed=1").text
    assert "1 call made before the price was set is now costed too." in _text(page)
    assert "No price set" not in _text(_prices_card(page))


def test_a_priced_call_keeps_its_cost_and_a_new_price_counts_from_now_on(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "anthropic")
    account = _account(fresh_store)
    _call(account, "q-old", "anthropic", "claude-sonnet-4-6")
    before = fresh_store.question_usage("q-old")["cost_usd"]
    assert before == pytest.approx((1000 * 3 + 9000 * 0.30 + 500 * 15) / 1e6)

    admin.post("/admin/system/prices/save", data={"provider": "anthropic", "model": "claude-sonnet-4-6",
                                                  "input_rate": "1", "cached_rate": "0.1", "output_rate": "5"})
    _call(account, "q-new", "anthropic", "claude-sonnet-4-6")
    assert fresh_store.question_usage("q-old")["cost_usd"] == pytest.approx(before)
    assert fresh_store.question_usage("q-new")["cost_usd"] == pytest.approx((1000 * 1 + 9000 * 0.1 + 500 * 5) / 1e6)


def test_the_list_shows_where_each_price_came_from_and_an_override_can_go_back(fresh_store, admin):
    card = _text(_prices_card(admin.get("/admin/system").text))
    assert "claude-opus-4-5 Anthropic $5.00 $0.50 $6.25 $10.00 $25.00 Anthropic API list price checked" in card
    assert "Prompts over 100,000 tokens: $0.50 in, $2.50 out" in card, "Haiku 5.5's long-prompt rate is shown"

    admin.post("/admin/system/prices/save", data={"provider": "anthropic", "model": "claude-opus-4-5",
                                                  "input_rate": "4", "output_rate": "20"})
    card = _text(_prices_card(admin.get("/admin/system").text))
    assert "claude-opus-4-5 Anthropic $4.00 as input as input as input $20.00 Set by an admin" in card
    assert "Back to list price" in card

    admin.post("/admin/system/prices/delete", data={"provider": "anthropic", "model": "claude-opus-4-5"})
    from core.llm_prices import price_for

    assert price_for("anthropic", "claude-opus-4-5").price.input == 5.00


@pytest.mark.parametrize("field,value", [("input_rate", ""), ("input_rate", "abc"), ("input_rate", "-1"),
                                         ("output_rate", "nan"), ("output_rate", "inf"), ("cached_rate", "20000")])
def test_a_price_that_is_not_a_plain_number_is_refused(fresh_store, admin, field, value):
    data = {"provider": "openai", "model": "gpt-4o", "input_rate": "2.5", "output_rate": "10", field: value}
    r = admin.post("/admin/system/prices/save", data=data, follow_redirects=False)
    assert "error=" in r.headers["location"]
    assert fresh_store.list_llm_prices() == []


def test_a_provider_or_model_name_that_is_not_one_is_refused(fresh_store, admin):
    for data in ({"provider": "nobody", "model": "m"}, {"provider": "openai", "model": ""},
                 {"provider": "openai", "model": "x" * 129}):
        r = admin.post("/admin/system/prices/save", data={**data, "input_rate": "1", "output_rate": "1"},
                       follow_redirects=False)
        assert "error=" in r.headers["location"]
    assert fresh_store.list_llm_prices() == []


def test_fetching_azure_deployments_remembers_the_model_each_runs(fresh_store, admin):
    import httpx

    fresh_store.set_system("azure_openai_endpoint", "https://acme.openai.azure.com")
    fresh_store.set_system("azure_openai_api_key", "key")

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            body = {"data": [{"id": "prod-4o", "model": "gpt-4o", "status": "succeeded"},
                             {"id": "cheap", "model": "gpt-4o-mini", "status": "succeeded"}]}
            return SimpleNamespace(status_code=200, is_success=True, headers={}, text=json.dumps(body),
                                   json=lambda: body)

    with patch.object(httpx, "AsyncClient", _Client):
        assert admin.get("/admin/system/azure-deployments").json()["ok"]
    assert json.loads(fresh_store.get_system("azure_deployment_models")) == {"cheap": "gpt-4o-mini", "prod-4o": "gpt-4o"}
    from core.llm_prices import price_for

    assert price_for("azure_openai", "cheap").model == "gpt-4o-mini"
    page = admin.get("/admin/system").text
    assert 'name="deployment" value="prod-4o"' in page and 'name="deployment_model" value="gpt-4o"' in page


def test_a_deployment_typed_by_hand_is_mapped_and_a_cleared_one_is_forgotten(fresh_store, admin):
    admin.post("/admin/system/azure-pricing", data={"deployment_type": "global", "new_deployment": "hand-made",
                                                    "new_deployment_model": "gpt-4.1"})
    assert json.loads(fresh_store.get_system("azure_deployment_models")) == {"hand-made": "gpt-4.1"}
    admin.post("/admin/system/azure-pricing", data={"deployment_type": "nonsense", "deployment": ["hand-made"],
                                                    "deployment_model": [""]})
    assert json.loads(fresh_store.get_system("azure_deployment_models")) == {}
    assert fresh_store.get_system("azure_deployment_type") == "global", "an unknown type is not saved"


# ── Where the cost is shown ──────────────────────────────────────────────────

def test_the_question_log_shows_each_questions_cost_with_its_steps(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "anthropic")
    account = _account(fresh_store)
    _call(account, "q-1", "anthropic", "claude-sonnet-4-6", component="sql_generation")
    _call(account, "q-1", "anthropic", "claude-sonnet-4-6", component="analysis")
    fresh_store.log_query(account, "net sales by store", "SELECT 1", question_id="q-1", llm_provider="anthropic",
                          llm_model="claude-sonnet-4-6")
    fresh_store.log_query(account, "an old question", "SELECT 1", llm_provider="anthropic",
                          llm_model="claude-sonnet-4-6", tokens_in=1000, tokens_out=0)
    fresh_store.log_query(account, "a cached answer", "SELECT 1", question_id="q-none", llm_provider="anthropic")

    html = admin.get(f"/admin/clients/{account}/queries").text
    cost = html.index('class="q-cost-pop"')
    cell = _text(html[cost:html.index('class="q-cost-note"', cost)])
    one = (1000 * 3 + 9000 * 0.30 + 500 * 15) / 1e6
    assert f"${2 * one:.4f}" in cell
    assert "2 AI calls" in cell
    assert "Sent 20,000 tokens: 2,000 new, 18,000 read from the cache" in cell
    assert "Writing the query · claude-sonnet-4-6" in cell and "Writing the answer · claude-sonnet-4-6" in cell
    assert "~$0.0030" in _text(html), "an old row shows its estimate as one"
    assert "Answered without asking the model" in html


def test_a_question_with_an_unpriced_call_says_so(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "azure_openai")
    account = _account(fresh_store)
    _call(account, "q-u", "azure_openai", "prod-4o")
    fresh_store.log_query(account, "q", "SELECT 1", question_id="q-u", llm_provider="azure_openai")
    html = admin.get(f"/admin/clients/{account}/queries").text
    assert "1 call was made with a model that has no price, so it is not in this cost." in _text(html)


def test_the_billing_page_shows_usage_and_sends_prices_to_system(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "azure_openai")
    account = _account(fresh_store)
    _call(account, "q-b", "azure_openai", "prod-4o")
    fresh_store.log_query(account, "q", "SELECT 1", question_id="q-b", llm_provider="azure_openai")
    html = admin.get(f"/admin/clients/{account}/billing").text
    assert "1 AI call was made with a model that has no price set" in _text(html)
    assert 'href="/admin/system#ai-prices"' in html
    assert "billing/pricing/save" not in html


def test_the_learned_page_shows_what_learning_cost(fresh_store, admin, monkeypatch):
    _reply(monkeypatch, "anthropic")
    account = _account(fresh_store)
    _call(account, "core2-labels-old", "anthropic", "claude-sonnet-4-6", component="core2_labels")
    with fresh_store.get_db() as conn:  # an earlier Learn's call
        conn.execute("UPDATE llm_usage SET created_at = '2000-01-01 00:00:00' WHERE question_id = 'core2-labels-old'")
    started = fresh_store.start_core2_build(account, None)
    _call(account, "core2-labels-x", "anthropic", "claude-sonnet-4-6", component="core2_labels")
    _call(account, "q-same-time", "anthropic", "claude-sonnet-4-6", component="core2_planner")  # a question meanwhile
    fresh_store.finish_core2_build(account, None, started, status="ready", version=1)
    html = _text(admin.get(f"/admin/clients/{account}/learned").text)
    one = (1000 * 3 + 9000 * 0.30 + 500 * 15) / 1e6
    assert f"made 1 AI call (10,500 tokens), costing ${one:.4f}." in html
