"""
core/prompt_cache.py

A cache breakpoint for the system prompt.

Nothing in this codebase cached anything: `cache_control` appeared zero times,
so every question re-sent and re-paid for the same rules and the same knowledge
base. The obstacle was never the API, it was the shape of the prompt --
`build_sql_system_prompt` gates rules on the question, so the bytes change from
one question to the next and a prefix that changes cannot be cached.

The fix is not to stop gating rules; the gating is a quality win worth keeping.
It is to move the breakpoint, so the part that never varies for an account
comes first and everything the question touches comes after it.

    [ universal rules + full account knowledge base ]   <- stable, cached
    [ gated rules + resolved plan + session context ]   <- varies, not cached

This module owns that split and how it reaches each provider. Anthropic takes
an explicit `cache_control` breakpoint. OpenAI and Azure OpenAI cache prefixes
automatically with no parameter to set, so for them the same ordering is what
does the work and the two halves are simply concatenated -- which is why
`CachedPrompt` has to behave like the plain string it replaces.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger("querybot.prompt_cache")

# Anthropic's minimum cacheable prefix varies by model: 512 tokens on Opus 5,
# 1,024 on Sonnet, 4,096 on the Haiku and Opus 4.5 generation.
#
# This was 4,096 -- the largest of them -- on the reasoning that clearing every
# model's floor avoids a stale model table. That was the wrong trade. The query
# path defaults to `claude-sonnet-4-6` (_default_model in core/llm.py), whose
# real minimum is 1,024, so a 4,096 floor declined to cache prefixes that would
# have cached perfectly well -- turning the conservative choice into the very
# silent no-op it was meant to prevent, four times over.
#
# 1,024 covers Sonnet and Opus. Below a model's own floor the API does not
# error, it just caches nothing -- so `anthropic_system_blocks` logs every
# decision instead, and a prefix that never pays shows up in the log rather
# than in the bill.
_MIN_CACHEABLE_TOKENS = 1024

# Deliberately crude, and deliberately an under-estimate of tokens per char, so
# a block only qualifies when it clears the minimum with room to spare. Getting
# this wrong in the safe direction costs a cache miss; wrong the other way costs
# a breakpoint that never pays.
_CHARS_PER_TOKEN = 4
MIN_CACHEABLE_CHARS = _MIN_CACHEABLE_TOKENS * _CHARS_PER_TOKEN


@dataclass(frozen=True)
class CachedPrompt:
    """A system prompt split at a cache breakpoint.

    ``stable`` is byte-identical for every question on an account; ``volatile``
    is everything a question changes. Anywhere a plain string is still expected
    -- the audit trail, the token counter, a provider without explicit caching
    -- ``str(prompt)`` gives back exactly the prompt that would have been sent
    without the split.
    """

    stable: str
    volatile: str = ""

    @property
    def text(self) -> str:
        if not self.volatile:
            return self.stable
        if not self.stable:
            return self.volatile
        return f"{self.stable}\n\n{self.volatile}"

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.text)

    @property
    def cacheable(self) -> bool:
        """Is the stable half worth a breakpoint on any supported model?

        A split with nothing after it is not worth one either: the whole prompt
        would be the prefix, and marking it changes nothing about what varies.
        """
        return bool(self.volatile) and len(self.stable) >= MIN_CACHEABLE_CHARS


def as_prompt_text(system: str | CachedPrompt) -> str:
    """The prompt as one string, for callers with no notion of a breakpoint."""
    return system.text if isinstance(system, CachedPrompt) else str(system or "")


def anthropic_system_blocks(system: str | CachedPrompt) -> str | list[dict]:
    """Shape the system prompt for the Anthropic Messages API.

    Returns the plain string unless there is a breakpoint worth taking, so the
    request is unchanged from today in every case where caching would not have
    happened anyway.

    The 1-hour TTL rather than the 5-minute default is deliberate: the thing
    being cached is an account's rules and knowledge base, which is warm across
    a working session and stone cold between them.
    """
    if not isinstance(system, CachedPrompt):
        return as_prompt_text(system)
    if not system.cacheable:
        # Said out loud. Declining silently is how a breakpoint ends up never
        # firing for a whole release while the code reads as though caching is
        # on -- there is no error from the API either way.
        log.warning(
            "No cache breakpoint taken: the stable block is %d chars, below the "
            "%d needed (%d tokens). The prompt is sent unsplit and nothing is "
            "cached.",
            len(system.stable), MIN_CACHEABLE_CHARS, _MIN_CACHEABLE_TOKENS,
        )
        return as_prompt_text(system)
    return [
        {
            "type": "text",
            "text": system.stable,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        {"type": "text", "text": system.volatile},
    ]


def cache_usage(usage: object) -> dict[str, int]:
    """Read the cache counters off a response, when the SDK reports them.

    These are the only honest answer to "is the breakpoint actually working".
    A prefix that is still varying reads as a permanent zero here rather than
    as an error, which is precisely why it has to be logged rather than
    assumed.
    """
    out: dict[str, int] = {}
    for field in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        try:
            value = int(getattr(usage, field, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            out[field] = value
    return out


# ── The switch ───────────────────────────────────────────────────────────────
# Off by default, and deliberately so. Preloading changes which documents the
# model sees and the order it sees them in, on the one prompt in this product
# that decides whether an answer is right. That earns an eval run against the
# golden questions on a real account before it becomes the default, not an
# assurance that the reordering looked harmless:
#
#     QUERYBOT_PROMPT_CACHE=on python -m evals.run --client <id> --cases <file> --generate --execute
#
# The failure mode of leaving it off is the latency that exists today. The
# failure mode of turning it on untested is a wrong number, so the asymmetry
# picks the default.
_TRUTHY = {"1", "true", "on", "yes"}
_logged_state: bool | None = None


def prompt_cache_enabled() -> bool:
    """Is the preload-and-cache path on for this process?"""
    global _logged_state
    enabled = os.getenv("QUERYBOT_PROMPT_CACHE", "").strip().lower() in _TRUTHY
    if enabled != _logged_state:
        _logged_state = enabled
        log.info(
            "Prompt cache %s: the knowledge base is %s",
            "ON" if enabled else "off",
            "preloaded whole and cached" if enabled
            else "retrieved per question (set QUERYBOT_PROMPT_CACHE=on to change)",
        )
    return enabled
