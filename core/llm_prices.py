"""What one AI call cost: its tokens, as the provider counted them, at the model's own prices.

Cost used to be ``tokens_in * input + tokens_out * output`` with one global rate pair per
model, and that was wrong in four ways:

- Cached input was never priced. Anthropic reports cached reads and cache writes apart
  from ``input_tokens`` (which is only the uncached rest), and bills them at 0.1x and
  1.25x/2x the input price; Azure and OpenAI count cached tokens inside
  ``prompt_tokens`` and bill them at a discount.
- An unknown model (every Azure deployment name, every local model) was charged
  gpt-4o's rates, so a cost appeared for calls nobody priced.
- Several built-in rates were stale.
- Only the first SQL call of a question was counted.

Here a call's usage is read in full from the provider's response (:class:`Usage`), and
priced from the price table (:func:`price_for`): the admin's own rows first, then the
built-in rows below. A model with no price costs ``None`` -- shown as "no price set",
never as a guess. A model running on the workspace's own server costs nothing.

The built-in prices are the providers' published list prices, each with where it came
from and when it was checked. Only prices that were checked are built in: a price that
could not be confirmed is left for the admin (or the price list update) to set.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("querybot.llm_prices")

PER_MILLION = 1_000_000

# Deployment types an Azure OpenAI resource can serve a model under. Azure prices each
# differently (Global is the reference; Data Zone and Regional carry an uplift).
AZURE_DEPLOYMENT_TYPES = ("global", "data_zone", "regional")


@dataclass(frozen=True)
class Usage:
    """One call's tokens, split the way providers bill them.

    ``input`` is the uncached prompt; ``cached_input`` the prompt read from the cache;
    ``cache_write_5m`` / ``cache_write_1h`` the prompt written to it (Anthropic only);
    ``output`` everything generated, reasoning included (``reasoning`` is that part,
    kept for display only: it is already inside ``output``).
    """

    input: int = 0
    cached_input: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def prompt(self) -> int:
        return self.input + self.cached_input + self.cache_write_5m + self.cache_write_1h

    @property
    def total(self) -> int:
        return self.prompt + self.output


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_from_anthropic(usage: Any, *, default_ttl: str = "5m") -> Usage:
    """Anthropic's ``usage``: ``input_tokens`` is the uncached rest; reads and writes are apart.

    The write is split by TTL when the response says so (``usage.cache_creation``);
    otherwise it is counted at the TTL the request asked for.
    """
    if usage is None:
        return Usage()
    write_5m = write_1h = 0
    split = getattr(usage, "cache_creation", None)
    if split is not None:
        write_5m = _int(getattr(split, "ephemeral_5m_input_tokens", 0))
        write_1h = _int(getattr(split, "ephemeral_1h_input_tokens", 0))
    if not (write_5m or write_1h):
        written = _int(getattr(usage, "cache_creation_input_tokens", 0))
        if default_ttl == "1h":
            write_1h = written
        else:
            write_5m = written
    return Usage(input=_int(getattr(usage, "input_tokens", 0)),
                 cached_input=_int(getattr(usage, "cache_read_input_tokens", 0)),
                 cache_write_5m=write_5m, cache_write_1h=write_1h,
                 output=_int(getattr(usage, "output_tokens", 0)))


def usage_from_openai(usage: Any) -> Usage:
    """Azure's and OpenAI's ``usage``: ``prompt_tokens`` includes the cached part.

    Cached tokens are ``prompt_tokens_details.cached_tokens``; reasoning tokens (o-series,
    gpt-5) are ``completion_tokens_details.reasoning_tokens`` and already inside
    ``completion_tokens``. A runtime that sends no details (most local servers) gives
    plain prompt and completion counts.
    """
    if usage is None:
        return Usage()
    prompt = _int(getattr(usage, "prompt_tokens", 0))
    details = getattr(usage, "prompt_tokens_details", None)
    cached = min(_int(getattr(details, "cached_tokens", 0)) if details is not None else 0, prompt)
    out_details = getattr(usage, "completion_tokens_details", None)
    reasoning = _int(getattr(out_details, "reasoning_tokens", 0)) if out_details is not None else 0
    return Usage(input=prompt - cached, cached_input=cached, output=_int(getattr(usage, "completion_tokens", 0)),
                 reasoning=reasoning)


@dataclass(frozen=True)
class Price:
    """USD per million tokens. ``None`` for a part means "billed as plain input"."""

    input: float
    output: float
    cached_input: float | None = None
    cache_write_5m: float | None = None
    cache_write_1h: float | None = None
    # A higher rate once a prompt passes a size (Claude Haiku 5.5 above 100K tokens).
    long_prompt_over: int = 0
    long_input: float | None = None
    long_output: float | None = None
    source: str = ""
    checked: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def cost_of(usage: Usage, price: Price | None) -> float | None:
    """The call's cost in USD, or None when there is no price for its model."""
    if price is None:
        return None
    rate_in, rate_out = price.input, price.output
    if price.long_prompt_over and usage.prompt > price.long_prompt_over:
        scale = (price.long_input / price.input) if price.long_input and price.input else 1.0
        rate_in = price.long_input if price.long_input is not None else rate_in
        rate_out = price.long_output if price.long_output is not None else rate_out
    else:
        scale = 1.0
    cached = price.cached_input * scale if price.cached_input is not None else rate_in
    write_5m = price.cache_write_5m * scale if price.cache_write_5m is not None else rate_in
    write_1h = price.cache_write_1h * scale if price.cache_write_1h is not None else rate_in
    total = (usage.input * rate_in + usage.cached_input * cached + usage.cache_write_5m * write_5m
             + usage.cache_write_1h * write_1h + usage.output * rate_out)
    return total / PER_MILLION


# ── Built-in prices ──────────────────────────────────────────────────────────
# Anthropic's published API list prices (first-party; the same on Microsoft Foundry).
# Cache reads are 0.1x the input price (0.05x on Claude Opus 5.5); cache writes 1.25x
# for the 5-minute TTL and 2x for the 1-hour TTL.
_ANTHROPIC_SOURCE = "Anthropic API list price"
_CHECKED = "2026-10-09"


def _anthropic(inp: float, out: float, cached: float, **extra: Any) -> Price:
    return Price(input=inp, output=out, cached_input=cached, cache_write_5m=inp * 1.25, cache_write_1h=inp * 2,
                 source=_ANTHROPIC_SOURCE, checked=_CHECKED, **extra)


BUILT_IN: dict[tuple[str, str], Price] = {
    ("anthropic", "claude-opus-5-5"): _anthropic(4.00, 20.00, 0.20),
    ("anthropic", "claude-opus-5"): _anthropic(5.00, 25.00, 0.50),
    ("anthropic", "claude-opus-4-8"): _anthropic(5.00, 25.00, 0.50),
    ("anthropic", "claude-opus-4-7"): _anthropic(5.00, 25.00, 0.50),
    ("anthropic", "claude-opus-4-6"): _anthropic(5.00, 25.00, 0.50),
    ("anthropic", "claude-opus-4-5"): _anthropic(5.00, 25.00, 0.50),
    ("anthropic", "claude-sonnet-5-5"): _anthropic(2.00, 10.00, 0.20),
    ("anthropic", "claude-sonnet-5"): _anthropic(2.00, 10.00, 0.20),
    ("anthropic", "claude-sonnet-4-6"): _anthropic(3.00, 15.00, 0.30),
    ("anthropic", "claude-haiku-5-5"): _anthropic(0.10, 0.50, 0.01, long_prompt_over=100_000,
                                                  long_input=0.50, long_output=2.50),
    ("anthropic", "claude-haiku-4-5"): _anthropic(1.00, 5.00, 0.10),
}

FREE = Price(input=0.0, output=0.0, cached_input=0.0, cache_write_5m=0.0, cache_write_1h=0.0,
             source="Runs on your own server", checked="")


# ── The price table: the admin's rows over the built-in ones ─────────────────
_CACHE_SECONDS = 30.0
_cache: dict[str, Any] = {"at": 0.0, "rows": {}, "system": {}}


def _refresh() -> None:
    if time.monotonic() - _cache["at"] < _CACHE_SECONDS:
        return
    rows: dict[tuple[str, str, str], Price] = {}
    system: dict[str, Any] = {}
    try:
        import store

        for row in store.list_llm_prices():
            rows[(row["provider"], row["model"], row.get("deployment_type") or "")] = Price(
                input=float(row["input"]), output=float(row["output"]),
                cached_input=_maybe(row.get("cached_input")), cache_write_5m=_maybe(row.get("cache_write_5m")),
                cache_write_1h=_maybe(row.get("cache_write_1h")), source=row.get("source") or "Set by an admin",
                checked=str(row.get("updated_at") or "")[:10])
        system = {"azure_deployment_models": store.get_system("azure_deployment_models", ""),
                  "azure_deployment_type": store.get_system("azure_deployment_type", "global")}
    except Exception as exc:  # noqa: BLE001 - a missing table on first start prices nothing, loudly
        log.warning("LLM prices could not be read: %s", exc)
    _cache.update(at=time.monotonic(), rows=rows, system=system)


def forget_cached_prices() -> None:
    """After an admin saves a price or a deployment's model: read them again on the next call."""
    _cache["at"] = 0.0


def _maybe(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def azure_deployment_model(deployment: str) -> str:
    """The model an Azure deployment runs, as the admin's resource listed it.

    Azure calls are made by deployment name ("prod-4o"); the price belongs to the model
    behind it ("gpt-4o"). The map is saved when the admin fetches the deployments. A
    deployment named after its model needs no map.
    """
    _refresh()
    raw = _cache["system"].get("azure_deployment_models") or ""
    try:
        mapping = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
    except ValueError:
        mapping = {}
    return str(mapping.get(deployment) or deployment or "")


def azure_deployment_type() -> str:
    _refresh()
    value = str(_cache["system"].get("azure_deployment_type") or "global")
    return value if value in AZURE_DEPLOYMENT_TYPES else "global"


@dataclass(frozen=True)
class PricedModel:
    """Which price a call was charged at: the model behind it, and how it was deployed."""

    provider: str
    model: str
    deployment_type: str = ""
    price: Price | None = field(default=None, compare=False)


def price_for(provider: str, model: str) -> PricedModel:
    """The price of a call to ``model`` through ``provider``; ``price`` is None when none is set."""
    if provider == "local":
        return PricedModel(provider, model, "", FREE)
    _refresh()
    rows = _cache["rows"]
    if provider == "azure_openai":
        underlying, dtype = azure_deployment_model(model), azure_deployment_type()
        price = rows.get((provider, underlying, dtype)) or rows.get((provider, underlying, ""))
        return PricedModel(provider, underlying, dtype, price)
    price = rows.get((provider, model, "")) or BUILT_IN.get((provider, model))
    return PricedModel(provider, model, "", price)


def all_prices() -> list[dict[str, Any]]:
    """Every price the table holds, the admin's and the built-in, for the prices page."""
    _refresh()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for (provider, model, dtype), price in sorted(_cache["rows"].items()):
        out.append({"provider": provider, "model": model, "deployment_type": dtype, "own": True,
                    "over_built_in": not dtype and (provider, model) in BUILT_IN, **price.as_dict()})
        seen.add((provider, model, dtype))
    for (provider, model), price in sorted(BUILT_IN.items()):
        if (provider, model, "") not in seen:
            out.append({"provider": provider, "model": model, "deployment_type": "", "own": False,
                        "over_built_in": False, **price.as_dict()})
    return out
