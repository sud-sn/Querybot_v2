"""The AI providers Admin → System offers, and the models each one lists.

One catalogue, provider -> models, read by every page that offers a model: the
System page's provider tabs, first-run setup and a workspace's model override
(admin/routes.py builds QUERY_MODELS and KB_MODELS from it). A tab's selects are
drawn on the server from its own provider's entries, and a save is checked
against the same entries, so a Claude model cannot be saved as the OpenAI one
whatever the browser posts.

Azure OpenAI has no entries: Azure routes on the deployment names the tenant
chose, so its tab lists the resource's deployments (GET /admin/system/azure-deployments).
A local server serves whatever its operator pulled, so its names are typed; the
local entries are only examples.

What the system settings hold, and what reads them:
  default_llm_provider          the provider new questions use (core.llm.resolve_provider)
  default_llm_model, kb_llm_model
                                that provider's query and KB models, copied from its tab
                                when it is made the default or its tab is saved while in use
  anthropic_query_model, anthropic_kb_model, openai_query_model, openai_kb_model
                                each hosted provider's own choice, kept while another is in use
  azure_query_deployment_name, azure_kb_deployment_name, local_llm_model, local_kb_model
                                Azure's and the local server's own, read by the runtime directly
A workspace's own provider and model still come before all of these.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("querybot.admin")

PROVIDERS = ("azure_openai", "anthropic", "openai", "local")       # the tabs, in order
LABELS = {
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI (GPT)",
    "local": "Local model",
}
HOSTED = ("anthropic", "openai")      # chosen from a list of model ids, with a key of their own
PURPOSES = ("query", "kb")


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    note: str = ""               # what it is good for, shown after the name
    older: bool = False          # kept so a saved one still shows selected
    purposes: tuple[str, ...] = PURPOSES

    def label(self) -> str:
        return f"{self.name} — {self.note}" if self.note else self.name


_CATALOGUE: dict[str, tuple[Model, ...]] = {
    "anthropic": (
        Model("claude-sonnet-5-5", "Claude Sonnet 5.5", "recommended for questions"),
        Model("claude-opus-5-5", "Claude Opus 5.5", "recommended for knowledge-base builds"),
        Model("claude-haiku-5-5", "Claude Haiku 5.5", "fast"),
        Model("claude-fable-5-1", "Claude Fable 5.1"),
        Model("claude-sonnet-4-6", "Claude Sonnet 4.6", "older", older=True),
        Model("claude-opus-4-6", "Claude Opus 4.6", "older", older=True),
        Model("claude-opus-4-5", "Claude Opus 4.5", "older", older=True),
        Model("claude-haiku-4-5", "Claude Haiku 4.5", "older", older=True),
    ),
    # A short list that works without asking OpenAI; "Load the models this key can use"
    # replaces it with the account's own.
    "openai": (
        Model("gpt-4o", "GPT-4o", "recommended"),
        Model("gpt-4.1", "GPT-4.1"),
        Model("gpt-4o-mini", "GPT-4o mini", "fast, lowest cost"),
    ),
    "local": (
        Model("llama3.1:8b", "Llama 3.1 8B", purposes=("query",)),
        Model("qwen2.5:14b", "Qwen 2.5 14B", purposes=("query",)),
        Model("mistral-small", "Mistral Small", purposes=("query",)),
        Model("llama3.1:70b", "Llama 3.1 70B", purposes=("kb",)),
        Model("qwen2.5:32b", "Qwen 2.5 32B", purposes=("kb",)),
    ),
}

RECOMMENDED = {
    "anthropic": {"query": "claude-sonnet-5-5", "kb": "claude-opus-5-5"},
    "openai": {"query": "gpt-4o", "kb": "gpt-4o"},
    "local": {"query": "llama3.1:8b", "kb": "llama3.1:70b"},
}

# The settings each hosted provider keeps its own models under.
OWN_MODEL_KEYS = {(p, purpose): f"{p}_{purpose}_model" for p in HOSTED for purpose in PURPOSES}

# A model id each provider's API takes. OpenAI's list also holds embeddings, images,
# speech and moderation models, and chat-like ones the chat completions call does not
# take; only chat models are offered.
_FAMILY = {
    "anthropic": re.compile(r"^claude-[a-z0-9][a-z0-9.\-]*$"),
    "openai": re.compile(r"^(?:gpt-|chatgpt-|o\d)[a-z0-9.\-]*$"),
}
_OPENAI_NOT_CHAT = re.compile(r"-(?:audio|realtime|transcribe|tts|image|instruct|codex|search)\b|-pro\b")


def models_for(provider: str, purpose: str = "query") -> list[Model]:
    """The catalogue's models for ``provider`` and ``purpose`` ("query" or "kb"), newest first."""
    return [m for m in _CATALOGUE.get(provider, ()) if purpose in m.purposes]


def is_chat_model(provider: str, model_id: str) -> bool:
    """Whether ``model_id`` is one of ``provider``'s chat models, by the shape of its id."""
    family = _FAMILY.get(provider)
    if family is None or not family.match(model_id or ""):
        return False
    return not (provider == "openai" and _OPENAI_NOT_CHAT.search(model_id))


def belongs(provider: str, model_id: str) -> bool:
    """Whether ``model_id`` is one of ``provider``'s: in its catalogue, or a chat model id of its API."""
    model_id = (model_id or "").strip()
    if not model_id:
        return False
    if any(m.id == model_id for m in _CATALOGUE.get(provider, ())):
        return True
    return is_chat_model(provider, model_id)


def model_problem(provider: str, model_id: str, purpose: str) -> str:
    """Why ``model_id`` cannot be saved as ``provider``'s ``purpose`` model, in plain words; "" when it can."""
    what = "knowledge-base model" if purpose == "kb" else "question model"
    if not model_id:
        return f"Pick a {what} for {LABELS[provider]}."
    if len(model_id) > 128 or not model_id.isprintable():
        return f"That {what} name is not one {LABELS[provider]} uses."
    if not belongs(provider, model_id):
        return (f"“{model_id}” is not an {LABELS[provider]} model. "
                f"Pick the {what} from the {LABELS[provider]} list.")
    return ""


def legacy_choices(purpose: str) -> list[tuple[str, str, str]]:
    """(provider, model id, label) for the pages that list every provider's models in one select."""
    suffix = {"anthropic": "", "openai": " (OpenAI)", "local": " (local)"}
    return [(p, m.id, m.label() + suffix[p]) for p in ("anthropic", "openai", "local") for m in models_for(p, purpose)]


# ── What a tab shows ─────────────────────────────────────────────────────────

def in_use_provider(cfg: dict) -> str:
    """The provider new questions use: the saved default, or Anthropic, as the runtime reads it."""
    saved = (cfg.get("default_llm_provider") or "").strip()
    return saved if saved in PROVIDERS else "anthropic"


def connected(provider: str, cfg: dict) -> bool:
    """Whether the provider's own credentials are saved (a local server needs only its address)."""
    if provider == "azure_openai":
        return bool(cfg.get("azure_openai_api_key") and cfg.get("azure_openai_endpoint"))
    if provider == "local":
        return bool((cfg.get("local_llm_base_url") or "").strip())
    return bool(cfg.get(f"{provider}_api_key"))


def chosen_model(provider: str, purpose: str, cfg: dict) -> str:
    """The model a tab shows for ``purpose``: the one that runs if this provider is in use, else its own."""
    if provider == "azure_openai":
        return (cfg.get(f"azure_{purpose}_deployment_name") or "").strip()
    if provider == "local":
        return (cfg.get("local_kb_model" if purpose == "kb" else "local_llm_model") or "").strip()
    running = (cfg.get("kb_llm_model" if purpose == "kb" else "default_llm_model") or "").strip()
    if in_use_provider(cfg) == provider and belongs(provider, running):
        return running
    own = (cfg.get(OWN_MODEL_KEYS[(provider, purpose)]) or "").strip()
    return own if belongs(provider, own) else RECOMMENDED[provider][purpose]


def missing_for_use(provider: str, cfg: dict) -> str:
    """What the provider still needs before it can answer new questions, in plain words; "" when ready."""
    if provider == "azure_openai":
        if not cfg.get("azure_openai_api_key"):
            return "save its API key first"
        if not cfg.get("azure_openai_endpoint"):
            return "save its endpoint first"
        if not (cfg.get("azure_query_deployment_name") and cfg.get("azure_kb_deployment_name")):
            return "pick a deployment for questions and one for knowledge-base builds first"
        return ""
    if provider == "local":
        return "" if connected("local", cfg) else "save the model server's address first"
    if not cfg.get(f"{provider}_api_key"):
        return "save its API key first"
    if not all(belongs(provider, cfg.get(OWN_MODEL_KEYS[(provider, p)]) or "") for p in PURPOSES):
        return "pick its models first"
    return ""


def default_models(provider: str, cfg: dict) -> dict[str, str]:
    """The default_llm_model / kb_llm_model a provider in use sets: its own choice, from its tab.

    An Azure deployment left blank is not copied: the runtime falls back to the saved
    model name for it (core.llm.azure_deployment_name), so emptying that would stop a
    resource that relied on it. A blank local name is copied: it means the runtime's own
    default name, as the local tab says.
    """
    if provider == "azure_openai":
        found = {"default_llm_model": cfg.get("azure_query_deployment_name") or "",
                 "kb_llm_model": cfg.get("azure_kb_deployment_name") or ""}
        return {k: v for k, v in found.items() if v}
    if provider == "local":
        return {"default_llm_model": cfg.get("local_llm_model") or "", "kb_llm_model": cfg.get("local_kb_model") or ""}
    return {"default_llm_model": cfg.get(OWN_MODEL_KEYS[(provider, "query")]) or "",
            "kb_llm_model": cfg.get(OWN_MODEL_KEYS[(provider, "kb")]) or ""}


def _options(provider: str, purpose: str, chosen: str) -> list[dict]:
    """A hosted provider's select: its current models, then its older ones, and a saved one it does not list."""
    rows = [{"id": m.id, "label": m.label(), "older": m.older, "selected": m.id == chosen}
            for m in models_for(provider, purpose)]
    if chosen and not any(r["selected"] for r in rows):
        rows.append({"id": chosen, "label": f"{chosen} (saved)", "older": False, "selected": True})
    return rows


def provider_tabs(cfg: dict) -> list[dict]:
    """One entry per tab, for the System page. Holds no credential, only whether one is saved."""
    using = in_use_provider(cfg)
    tabs = []
    for provider in PROVIDERS:
        tab = {"key": provider, "label": LABELS[provider], "connected": connected(provider, cfg),
               "in_use": provider == using, "missing": missing_for_use(provider, cfg)}
        for purpose in PURPOSES:
            chosen = chosen_model(provider, purpose, cfg)
            tab[purpose] = chosen
            if provider in HOSTED:
                tab[f"{purpose}_options"] = _options(provider, purpose, chosen)
            elif provider == "local":
                tab[f"{purpose}_example"] = RECOMMENDED["local"][purpose]
        tabs.append(tab)
    return tabs


def in_use_summary(cfg: dict) -> dict:
    """What new questions use, as the runtime will read it, and anything wrong with it."""
    from core.llm import _default_model    # the runtime's own last resort, so the line says what runs

    provider = in_use_provider(cfg)
    summary = {"provider": provider, "label": LABELS[provider], "query": "", "kb": "", "problem": ""}
    if provider in HOSTED:
        for purpose, key, tier in (("query", "default_llm_model", "fast"), ("kb", "kb_llm_model", "high")):
            running = (cfg.get(key) or "").strip()
            summary[purpose] = running or _default_model(provider, tier)
            if running and not belongs(provider, running):
                summary["problem"] = (f"New questions are set to use “{running}”, which is not an "
                                      f"{LABELS[provider]} model. Pick one on its tab and save.")
    elif provider == "local":
        summary["query"] = (chosen_model("local", "query", cfg) or (cfg.get("default_llm_model") or "").strip()
                            or _default_model("local", "fast"))
        summary["kb"] = (chosen_model("local", "kb", cfg) or chosen_model("local", "query", cfg)
                         or (cfg.get("kb_llm_model") or "").strip() or _default_model("local", "high"))
    else:
        summary["query"] = chosen_model(provider, "query", cfg) or (cfg.get("default_llm_model") or "").strip()
        summary["kb"] = chosen_model(provider, "kb", cfg) or summary["query"]
    if not connected(provider, cfg):
        summary["problem"] = summary["problem"] or f"{LABELS[provider]} is not connected: {missing_for_use(provider, cfg)}."
    return summary


# ── The models a saved key can use ───────────────────────────────────────────

LIST_TIMEOUT = 12.0     # seconds; the page waits on it


class ModelListError(Exception):
    """A plain sentence for the page: never a provider's body, which can quote part of the key."""


_LISTS = {
    "openai": {"host": "api.openai.com", "url": "https://api.openai.com/v1/models", "params": {}},
    "anthropic": {"host": "api.anthropic.com", "url": "https://api.anthropic.com/v1/models",
                  "params": {"limit": 1000}},
}


def _headers(provider: str, key: str) -> dict:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}


def _chat_models(provider: str, body: dict) -> list[dict]:
    rows = body.get("data") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        raise ValueError("no data list")
    found = []
    for row in rows:
        model_id = str((row or {}).get("id") or "").strip() if isinstance(row, dict) else ""
        if not is_chat_model(provider, model_id):
            continue
        if provider == "anthropic":
            name = str(row.get("display_name") or "").strip()
            found.append((str(row.get("created_at") or ""), model_id, f"{name} ({model_id})" if name else model_id))
        else:
            created = row.get("created")
            found.append((created if isinstance(created, (int, float)) else 0, model_id, model_id))
    found.sort(key=lambda r: (r[0], r[1]), reverse=True)          # newest first
    return [{"id": model_id, "label": label} for _when, model_id, label in found]


async def account_models(provider: str, cfg: dict) -> list[dict]:
    """The chat models the saved key for ``provider`` can use, newest first, as [{id, label}].

    Reads the key from the saved settings only; raises ModelListError with a sentence for the page.
    """
    if provider not in _LISTS:
        raise ModelListError("Models can be loaded for Anthropic and OpenAI. Azure lists its deployments.")
    spec, label = _LISTS[provider], LABELS[provider].split(" (")[0]
    key = (cfg.get(f"{provider}_api_key") or "").strip()
    if not key:
        raise ModelListError(f"No {label} API key is saved yet. Add it on this tab and save first.")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=LIST_TIMEOUT) as client:
            resp = await client.get(spec["url"], headers=_headers(provider, key), params=spec["params"])
    except httpx.TimeoutException:
        raise ModelListError(f"{spec['host']} took too long to answer. Try again in a moment.") from None
    except Exception as exc:  # noqa: BLE001 - the page gets a sentence; the log says what failed
        log.warning("Listing %s models failed: %s", provider, type(exc).__name__)
        raise ModelListError(f"Could not reach {spec['host']}. Check that this server can reach the internet.") from None

    if resp.status_code in (401, 403):
        raise ModelListError(f"{label} refused the saved key ({resp.status_code}). "
                             "Check the key, save it again, then load the models.")
    if resp.status_code == 429:
        raise ModelListError(f"{label} is limiting requests from this key (429). Try again in a minute.")
    if not resp.is_success:
        raise ModelListError(f"{spec['host']} answered HTTP {resp.status_code}. Try again later.")
    try:
        return _chat_models(provider, resp.json())
    except Exception as exc:  # noqa: BLE001 - a proxy's page on a 200 is not a model list
        log.warning("The %s model list could not be read: %s", provider, type(exc).__name__)
        raise ModelListError(f"{spec['host']} sent an answer that is not a model list.") from None
