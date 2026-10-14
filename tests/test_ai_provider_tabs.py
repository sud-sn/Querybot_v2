"""Admin → System: one tab per AI provider, each with its own key and only its own models.

The AI section was three steps: pick a default provider and save it, then the
matching credential panel appeared, then the models. The query and KB model
dropdowns listed every provider's models and hid the others with a script, so a
Claude id could still be posted as the OpenAI model, and the Claude list was a
generation old. Now:

- a tab each for Azure OpenAI, Anthropic (Claude), OpenAI (GPT) and a local model,
  each saying whether it is connected and which one answers new questions; the
  page opens on that one;
- a tab holds its provider's key and models, drawn on the server from one
  catalogue, so its selects can only list that provider's models; the server
  refuses a model saved under the wrong provider all the same;
- saving a tab never touches another provider's settings, and a provider is used
  for new questions only once it has a key and models;
- Anthropic and OpenAI can load the models the saved key can use, through a route
  that reads the key on the server and answers in plain words.

The admin's routes run for real against a fresh store; the providers' HTTP
answers are invented at the httpx boundary.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

PROVIDERS = ("azure_openai", "anthropic", "openai", "local")


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from admin import routes

    # Settings are encrypted: under a key of the test's own, whichever store module
    # an earlier test left bound.
    for module in {id(m): m for m in (store, routes.store)}.values():
        for namespace in (module.crypto.__dict__, module.config_store.decrypt.__globals__):
            monkeypatch.setitem(namespace, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    yield store


@pytest.fixture
def admin(fresh_store):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import routes

    app = FastAPI()
    app.include_router(routes.router)
    with patch.object(routes, "_is_auth", return_value=True):
        yield TestClient(app)


def _panels(html: str) -> dict[str, str]:
    """Each provider's tab panel, by its id."""
    found = dict(re.findall(r'<section[^>]*\bid="ai-panel-([a-z_]+)"[^>]*>(.*?)</section>', html, re.S))
    assert sorted(found) == sorted(PROVIDERS), sorted(found)
    return found


def _panel_tags(html: str) -> dict[str, str]:
    """Each provider's tab panel's opening tag."""
    return {p: tag for tag, p in re.findall(r'(<section[^>]*\bid="ai-panel-([a-z_]+)"[^>]*>)', html)}


def _options(panel: str, name: str) -> list[tuple[str, str, bool]]:
    """(value, text, selected) of the select called ``name`` in a panel."""
    select = re.search(rf'<select[^>]*\bname="{name}"[^>]*>(.*?)</select>', panel, re.S)
    assert select, f"no select {name}"
    return [(v, re.sub(r"\s+", " ", t).strip(), "selected" in attrs)
            for v, attrs, t in re.findall(r'<option value="([^"]*)"([^>]*)>(.*?)</option>', select.group(1), re.S)]


def _tabs(html: str) -> dict[str, str]:
    """Each tab button's markup, by provider."""
    return {p: tag for tag, p in re.findall(r'(<button[^>]*role="tab"[^>]*data-tab="([a-z_]+)"[^>]*>)', html)}


def _saved(response) -> dict:
    assert response.status_code == 303, response.text[:300]
    return {k: v[0] for k, v in parse_qs(urlparse(response.headers["location"]).query).items()}


# ── One catalogue ────────────────────────────────────────────────────────────

def test_each_provider_lists_only_its_own_models():
    from admin import ai_providers as ai

    claude = {m.id for m in ai.models_for("anthropic")}
    gpt = {m.id for m in ai.models_for("openai")}
    assert {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"} <= claude
    assert {"claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5"} <= claude
    assert {"gpt-4o", "gpt-4o-mini"} <= gpt
    assert all(i.startswith("claude-") for i in claude)
    assert not any(i.startswith("claude-") for i in gpt)
    assert ai.models_for("azure_openai") == [], "Azure lists the resource's deployments, not a catalogue"
    older = {m.id for m in ai.models_for("anthropic") if m.older}
    assert older == {"claude-sonnet-4-6", "claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5"}


def test_the_lists_other_pages_use_come_from_the_same_catalogue():
    from admin import ai_providers as ai
    from admin import routes

    for purpose, legacy in (("query", routes.QUERY_MODELS), ("kb", routes.KB_MODELS)):
        assert all(len(row) == 3 for row in legacy), "workspace and setup pages unpack (provider, model, label)"
        listed = {(p, m) for p, m, _label in legacy}
        for provider in ("anthropic", "openai"):
            assert {(provider, m.id) for m in ai.models_for(provider, purpose)} <= listed
        for provider, model in listed:
            assert ai.belongs(provider, model), (provider, model)


@pytest.mark.parametrize("provider, model, ok", [
    ("anthropic", "claude-sonnet-5-5", True),
    ("anthropic", "claude-3-5-sonnet-20241022", True),   # a model the key listed, not in the catalogue
    ("anthropic", "gpt-4o", False),
    ("openai", "gpt-4o", True),
    ("openai", "o3-mini", True),
    ("openai", "claude-sonnet-5-5", False),
    ("openai", "text-embedding-3-small", False),
    ("openai", "gpt-4o-realtime-preview", False),
])
def test_a_model_belongs_to_one_provider(provider, model, ok):
    from admin import ai_providers as ai

    assert ai.belongs(provider, model) is ok


# ── The page ─────────────────────────────────────────────────────────────────

def test_each_tab_lists_only_its_own_providers_models(fresh_store, admin):
    fresh_store.set_system("anthropic_api_key", "sk-ant-0123456789abcd")
    fresh_store.set_system("openai_api_key", "sk-proj-0123456789wxyz")
    html = admin.get("/admin/system").text
    panels = _panels(html)
    from admin import ai_providers as ai

    claude = {m.id for m in ai.models_for("anthropic")}
    gpt = {m.id for m in ai.models_for("openai")}
    for name in ("query_model", "kb_model"):
        anthropic = {v for v, _t, _s in _options(panels["anthropic"], name)}
        openai = {v for v, _t, _s in _options(panels["openai"], name)}
        assert anthropic == claude, name
        assert openai == gpt, name
    every_model = claude | gpt
    for provider in ("azure_openai", "local"):
        shown = set(re.findall(r'<option value="([^"]*)"', panels[provider]))
        assert not shown & every_model, (provider, shown & every_model)
    # Nowhere else on the page does a select mix them, to be hidden by a script.
    assert not re.search(r"<option[^>]*data-provider=", html)
    assert set(re.findall(r'<select[^>]*\bname="((?:query|kb)_model)"', html)) == {"query_model", "kb_model"}
    assert len(re.findall(r'<select[^>]*\bname="(?:query|kb)_model"', html)) == 4, "two per hosted tab"


def test_the_page_opens_on_the_provider_in_use_and_says_which_are_connected(fresh_store, admin):
    fresh_store.set_system("default_llm_provider", "openai")
    fresh_store.set_system("openai_api_key", "sk-proj-0123456789wxyz")
    html = admin.get("/admin/system").text
    tabs, panels = _tabs(html), _panel_tags(html)
    assert list(tabs) == ["azure_openai", "anthropic", "openai", "local"]
    assert 'role="tablist"' in html
    for provider, tag in tabs.items():
        assert f'aria-selected="{"true" if provider == "openai" else "false"}"' in tag, provider
        assert f'aria-controls="ai-panel-{provider}"' in tag
        assert bool(re.search(r"\shidden[\s>]", panels[provider])) is (provider != "openai"), provider
        assert 'role="tabpanel"' in panels[provider]
    texts = {p: re.sub(r"<[^>]+>|\s+", " ", block) for p, block in
             re.findall(r'data-tab="([a-z_]+)"[^>]*>(.*?)</button>', html, re.S)}
    assert "In use" in texts["openai"] and "In use" not in texts["anthropic"]
    assert "Connected" in texts["openai"] and "Not connected" in texts["anthropic"]
    # A tab asked for by the page's own redirect wins.
    again = admin.get("/admin/system?tab=anthropic").text
    assert 'aria-selected="true"' in _tabs(again)["anthropic"]


def test_a_saved_older_model_still_shows_selected(fresh_store, admin):
    fresh_store.set_system("default_llm_provider", "anthropic")
    fresh_store.set_system("default_llm_model", "claude-sonnet-4-6")
    fresh_store.set_system("kb_llm_model", "claude-opus-4-5")
    panel = _panels(admin.get("/admin/system").text)["anthropic"]
    query = {v: (t, s) for v, t, s in _options(panel, "query_model")}
    kb = {v: (t, s) for v, t, s in _options(panel, "kb_model")}
    assert query["claude-sonnet-4-6"][1] and "older" in query["claude-sonnet-4-6"][0].lower()
    assert kb["claude-opus-4-5"][1]
    assert [v for v, (_t, s) in query.items() if s] == ["claude-sonnet-4-6"]


def test_a_saved_model_the_catalogue_does_not_list_stays_selected(fresh_store, admin):
    fresh_store.set_system("anthropic_query_model", "claude-3-5-sonnet-20241022")
    panel = _panels(admin.get("/admin/system?tab=anthropic").text)["anthropic"]
    query = _options(panel, "query_model")
    assert [v for v, _t, s in query if s] == ["claude-3-5-sonnet-20241022"]


def test_keys_stay_masked_on_the_page(fresh_store, admin):
    fresh_store.set_system("anthropic_api_key", "sk-ant-api03-SECRETSECRET-abcd")
    fresh_store.set_system("azure_openai_api_key", "azure-SECRETSECRET-wxyz")
    html = admin.get("/admin/system").text
    assert "SECRETSECRET" not in html
    assert "abcd" in html and "wxyz" in html


# ── Saving a tab ─────────────────────────────────────────────────────────────

def test_a_model_saved_under_the_wrong_provider_is_refused(fresh_store, admin):
    r = admin.post("/admin/system", data={"provider": "openai", "openai_key": "sk-proj-new",
                                          "query_model": "claude-sonnet-5-5", "kb_model": "gpt-4o"},
                   follow_redirects=False)
    answer = _saved(r)
    assert "claude-sonnet-5-5" in answer["error"] and "OpenAI" in answer["error"]
    assert answer["tab"] == "openai"
    assert fresh_store.get_system("openai_query_model") == ""
    assert fresh_store.get_system("openai_api_key") == "", "a refused save saves nothing"
    page = admin.get(r.headers["location"]).text
    assert "claude-sonnet-5-5" in page and "OpenAI" in page


def test_saving_one_providers_key_keeps_the_others(fresh_store, admin):
    fresh_store.set_system("anthropic_api_key", "sk-ant-kept")
    fresh_store.set_system("azure_openai_api_key", "azure-kept")
    fresh_store.set_system("azure_openai_endpoint", "https://kept.openai.azure.com")
    r = admin.post("/admin/system", data={
        "provider": "openai", "openai_key": "sk-proj-new", "query_model": "gpt-4o-mini", "kb_model": "gpt-4o",
        # Another tab's fields posted alongside are not this tab's to save.
        "anthropic_key": "sk-ant-intruder", "azure_openai_endpoint": "https://other.example.com"},
        follow_redirects=False)
    assert _saved(r)["saved"] == "ai"
    assert fresh_store.get_system("openai_api_key") == "sk-proj-new"
    assert fresh_store.get_system("openai_query_model") == "gpt-4o-mini"
    assert fresh_store.get_system("anthropic_api_key") == "sk-ant-kept"
    assert fresh_store.get_system("azure_openai_api_key") == "azure-kept"
    assert fresh_store.get_system("azure_openai_endpoint") == "https://kept.openai.azure.com"
    # A blank key, or the masked one the page shows, keeps the saved key.
    for shown in ("", "•••••••••••-new"):
        admin.post("/admin/system", data={"provider": "openai", "openai_key": shown,
                                          "query_model": "gpt-4o", "kb_model": "gpt-4o"})
        assert fresh_store.get_system("openai_api_key") == "sk-proj-new"


def test_saving_a_provider_that_is_not_in_use_leaves_new_questions_alone(fresh_store, admin):
    fresh_store.set_system("default_llm_provider", "anthropic")
    fresh_store.set_system("anthropic_api_key", "sk-ant-kept")
    fresh_store.set_system("default_llm_model", "claude-sonnet-5-5")
    admin.post("/admin/system", data={"provider": "openai", "openai_key": "sk-proj-new",
                                      "query_model": "gpt-4o", "kb_model": "gpt-4o"})
    assert fresh_store.get_system("default_llm_provider") == "anthropic"
    assert fresh_store.get_system("default_llm_model") == "claude-sonnet-5-5"


def test_a_provider_without_a_key_is_not_used_for_new_questions(fresh_store, admin):
    fresh_store.set_system("default_llm_provider", "anthropic")
    r = admin.post("/admin/system", data={"provider": "openai", "action": "use", "openai_key": "",
                                          "query_model": "gpt-4o", "kb_model": "gpt-4o"},
                   follow_redirects=False)
    answer = _saved(r)
    assert "API key" in answer["error"]
    assert fresh_store.get_system("default_llm_provider") == "anthropic"
    r = admin.post("/admin/system", data={"provider": "azure_openai", "action": "use",
                                          "azure_openai_key": "azure-key",
                                          "azure_openai_endpoint": "https://acme.openai.azure.com"},
                   follow_redirects=False)
    assert "deployment" in _saved(r)["error"]
    assert fresh_store.get_system("default_llm_provider") == "anthropic"


def test_a_provider_made_the_default_is_what_new_questions_use(fresh_store, admin):
    r = admin.post("/admin/system", data={"provider": "openai", "action": "use", "openai_key": "sk-proj-new",
                                          "query_model": "gpt-4o-mini", "kb_model": "gpt-4o"},
                   follow_redirects=False)
    assert _saved(r) == {"saved": "ai-use", "tab": "openai"}
    # Nothing handed across: the runtime resolver finds what the tab saved.
    from core.llm import resolve_provider

    assert resolve_provider({"account_id": ""})[:3] == ("openai", "gpt-4o-mini", "sk-proj-new")
    assert resolve_provider({"account_id": ""}, purpose="kb")[1] == "gpt-4o"
    # A workspace's own model still comes first.
    assert resolve_provider({"account_id": "", "llm_model": "gpt-4.1"})[1] == "gpt-4.1"
    # Saving the tab in use changes what new questions use.
    admin.post("/admin/system", data={"provider": "openai", "query_model": "gpt-4.1", "kb_model": "gpt-4o"})
    assert resolve_provider({"account_id": ""})[1] == "gpt-4.1"
    page = admin.get("/admin/system").text
    assert 'aria-selected="true"' in _tabs(page)["openai"]


def test_azure_deployments_become_the_models_new_questions_use(fresh_store, admin):
    admin.post("/admin/system", data={"provider": "azure_openai", "action": "use", "azure_openai_key": "azure-key",
                                      "azure_openai_endpoint": "https://acme.openai.azure.com",
                                      "azure_api_version": "2024-10-21", "azure_query_deployment": "prod-4o",
                                      "azure_kb_deployment": "prod-4o-kb"})
    from core.llm import resolve_provider

    provider, model, key, extra = resolve_provider({"account_id": ""})
    assert (provider, model, key) == ("azure_openai", "prod-4o", "azure-key")
    assert extra["azure_api_version"] == "2024-10-21"
    assert resolve_provider({"account_id": ""}, purpose="kb")[1] == "prod-4o-kb"
    panel = _panels(admin.get("/admin/system").text)["azure_openai"]
    assert "prod-4o" in panel and "prod-4o-kb" in panel


# ── Loading the models a key can use ─────────────────────────────────────────

class _Http:
    """httpx.AsyncClient at the boundary: one canned answer, every request kept."""

    def __init__(self, status=200, body=None, raises=None):
        self.status, self.body, self.raises = status, body, raises
        self.requests: list[dict] = []
        self.made: list[dict] = []

    def client(self):
        http = self

        class _Response:
            status_code = http.status
            is_success = 200 <= http.status < 300
            headers: dict = {}
            text = json.dumps(http.body)

            @staticmethod
            def json():
                return http.body

        class _Client:
            def __init__(self, *a, **kw):
                http.made.append(kw)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, **kw):
                http.requests.append({"url": url, **kw})
                if http.raises:
                    raise http.raises
                return _Response()

        return _Client


def _load(admin, provider, http, **params):
    import httpx

    with patch.object(httpx, "AsyncClient", http.client()):
        r = admin.get("/admin/system/provider-models", params={"provider": provider, **params})
    return r


OPENAI_LIST = {"object": "list", "data": [
    {"id": "gpt-4o", "created": 1715367049},
    {"id": "text-embedding-3-small", "created": 1705948997},
    {"id": "o3-mini", "created": 1737146383},
    {"id": "dall-e-3", "created": 1698785189},
    {"id": "chatgpt-4o-latest", "created": 1723515131},
    {"id": "whisper-1", "created": 1677532384},
    {"id": "gpt-4o-realtime-preview", "created": 1727659998},
    {"id": "omni-moderation-latest", "created": 1731689265},
    {"id": "gpt-4o-mini-tts", "created": 1742403959},
]}


def test_openai_lists_only_chat_models_newest_first_with_the_saved_key(fresh_store, admin):
    fresh_store.set_system("openai_api_key", "sk-proj-stored")
    http = _Http(body=OPENAI_LIST)
    r = _load(admin, "openai", http, key="sk-from-the-browser")
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["models"]] == ["o3-mini", "chatgpt-4o-latest", "gpt-4o"]
    assert r.json()["ok"] is True
    (sent,) = http.requests
    assert sent["url"] == "https://api.openai.com/v1/models"
    assert sent["headers"]["Authorization"] == "Bearer sk-proj-stored"
    assert "sk-from-the-browser" not in json.dumps(http.requests)
    assert 0 < float(http.made[0]["timeout"]) <= 30


def test_anthropic_lists_its_models_by_name_with_the_saved_key(fresh_store, admin):
    fresh_store.set_system("anthropic_api_key", "sk-ant-stored")
    http = _Http(body={"data": [
        {"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": "2026-08-01T00:00:00Z"},
        {"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-09-01T00:00:00Z"},
    ], "has_more": False})
    r = _load(admin, "anthropic", http)
    assert r.json() == {"ok": True, "models": [
        {"id": "claude-opus-5-5", "label": "Claude Opus 5.5 (claude-opus-5-5)"},
        {"id": "claude-haiku-5-5", "label": "Claude Haiku 5.5 (claude-haiku-5-5)"}]}
    (sent,) = http.requests
    assert sent["url"].startswith("https://api.anthropic.com/v1/models")
    assert sent["headers"]["x-api-key"] == "sk-ant-stored"
    assert sent["headers"]["anthropic-version"] == "2023-06-01"


def test_no_saved_key_is_said_without_calling_out(fresh_store, admin):
    http = _Http(body=OPENAI_LIST)
    r = _load(admin, "openai", http)
    assert r.json()["ok"] is False and "No OpenAI API key is saved" in r.json()["error"]
    assert http.requests == []
    assert _load(admin, "azure_openai", http).json()["ok"] is False


def test_a_timeout_or_a_refusal_is_a_plain_message(fresh_store, admin):
    import httpx

    fresh_store.set_system("openai_api_key", "sk-proj-stored-9f8e")
    r = _load(admin, "openai", _Http(raises=httpx.ReadTimeout("timed out")))
    assert r.json()["ok"] is False and "took too long" in r.json()["error"]
    r = _load(admin, "openai", _Http(raises=httpx.ConnectError("dns failure at line 12")))
    assert r.json()["ok"] is False and "Could not reach api.openai.com" in r.json()["error"]
    refused = _Http(status=401, body={"error": {"message": "Incorrect API key provided: sk-proj-***9f8e"}})
    error = _load(admin, "openai", refused).json()["error"]
    assert "refused" in error and "9f8e" not in error and "Traceback" not in error
    error = _load(admin, "openai", _Http(status=500, body="<html>oops</html>")).json()["error"]
    assert "500" in error and "oops" not in error


def test_the_model_list_needs_an_admin(fresh_store):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from admin import access, routes

    app = FastAPI()
    app.include_router(routes.router)
    assert TestClient(app).get("/admin/system/provider-models?provider=openai").status_code == 401
    # The same rule as the Azure deployments route: super admins only.
    assert access.rule_for("/system/provider-models") == access.rule_for("/system/azure-deployments") \
        == (access.SUPER, False, "")


# ── The page's own script ───────────────────────────────────────────────────

def _js(fn: str, *args):
    dukpy = pytest.importorskip("dukpy")
    from pathlib import Path

    from tests.js_lift import function as lift

    source = (Path(__file__).resolve().parents[1] / "admin" / "templates" / "system.html").read_text(encoding="utf-8")
    call = f"{fn}(" + ", ".join(f"dukpy['a{i}']" for i in range(len(args))) + ");"
    return dukpy.evaljs(lift(source, f"function {fn}(") + "\n" + call, **{f"a{i}": a for i, a in enumerate(args)})


def test_a_loaded_list_keeps_the_saved_model_selected():
    loaded = [{"id": "gpt-4.1", "label": "gpt-4.1"}, {"id": "gpt-4o", "label": "gpt-4o"}]
    assert _js("aiModelOptions", loaded, "gpt-4o") == [
        {"value": "gpt-4.1", "label": "gpt-4.1", "selected": False},
        {"value": "gpt-4o", "label": "gpt-4o", "selected": True}]
    kept = _js("aiModelOptions", loaded, "gpt-4o-2024-05-13")
    assert kept[-1] == {"value": "gpt-4o-2024-05-13", "label": "gpt-4o-2024-05-13 (saved)", "selected": True}
    assert [o["selected"] for o in kept[:-1]] == [False, False]


def test_saving_one_tab_warns_about_another_tabs_unsaved_changes():
    labels = {"anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT)", "local": "Local model"}
    assert _js("aiUnsavedElsewhere", "openai", {"anthropic": True, "openai": True, "local": False}, labels) \
        == ["Anthropic (Claude)"]
    assert _js("aiUnsavedElsewhere", "openai", {"openai": True}, labels) == []


# ── The other pages that list models ────────────────────────────────────────

def test_a_workspace_model_the_lists_no_longer_name_still_shows_selected(fresh_store, admin):
    fresh_store.upsert_client("acct-tabs", "Test Ltd")
    fresh_store.update_client_meta("acct-tabs", llm_model="gpt-35-turbo")
    html = admin.get("/admin/clients/acct-tabs/settings").text
    assert re.search(r'<option value="gpt-35-turbo" selected>', html)
    fresh_store.update_client_meta("acct-tabs", llm_model="claude-sonnet-5-5")
    html = admin.get("/admin/clients/acct-tabs/settings").text
    assert re.search(r'<option value="claude-sonnet-5-5" selected>', html)
    assert "(saved)" not in html


def test_first_run_setup_offers_the_current_models_first(fresh_store, admin):
    html = admin.get("/admin/setup").text
    for name, model in (("default_model", "claude-sonnet-5-5"), ("kb_model", "claude-opus-5-5")):
        options = re.search(rf'<select name="{name}">(.*?)</select>', html, re.S).group(1)
        assert re.findall(r'<option value="([^"]+)" selected>', options) == [model], name
