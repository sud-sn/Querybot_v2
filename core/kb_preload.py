"""
core/kb_preload.py

Hand the model the whole knowledge base instead of searching for part of it.

Retrieval in this pipeline does not do what its name suggests. The tables that
actually reach the prompt are supplied by `guarantee_table_coverage`, which
looks them up deterministically from the entity graph; the semantic search runs
first, costs a dense pass plus a BM25 rebuild plus a cross-encoder re-rank, and
then has its answer overridden. On a live account that stage was measured at
~12.9 s a question, scoring 0.0002 against its own 0.05 floor -- the
cross-encoder judging correctly that a markdown column table is not a prose
passage answering a business question, which is a true statement and a useless
one for choosing a table.

An account's knowledge base is small enough to simply send. Fourteen tables at
the 9,000-character per-document cap is ~126,000 characters, about what the
prompt is already allowed to carry. So this is not a trade of completeness for
latency: the model sees every table it is permitted to see, the search stage
disappears, and the block is identical for every question -- which is what
makes it cacheable.

Two things are load-bearing here:

- **The allow-list is the boundary.** Only tables the caller has already
  resolved as permitted are fetched. Preloading is a change to how much of the
  permitted knowledge base is sent, never to which knowledge base it is.
- **The order is fixed.** A prefix that reorders itself between questions
  caches nothing, and would fail silently rather than loudly, so documents are
  assembled in sorted order regardless of how the caller supplies them.
"""

from __future__ import annotations

import logging
import time

from core.pipeline_helpers import (
    _PROMPT_CONTEXT_CHAR_CAP,
    _clamp_kb_doc,
    _clamp_prompt_context,
)

log = logging.getLogger("querybot.kb_preload")

# A knowledge base changes only when an admin rebuilds it. Five minutes bounds
# how long a rebuild can go unnoticed by a session that is already running;
# `invalidate_account_kb` closes the gap when the rebuild reports in.
_TTL_SECONDS = 300

# Keyed by account AND permitted-table set, not by account alone. Two users of
# the same account with different grants see different blocks, and a
# single-entry-per-account cache would have them evicting each other's on every
# question -- the preload would still be correct and would never be reused.
# (account_id, permitted tables) -> (built_at, context, tables)
_cache: dict[tuple[str, frozenset[str]], tuple[float, str, tuple[str, ...]]] = {}

# The knowledge base is the cache prefix, so it must leave room for the rules
# and the question-specific context that follow it. 85% of the prompt cap keeps
# the split from being decided by a truncation.
_PREFIX_SHARE = 0.85


def invalidate_account_kb(account_id: str) -> None:
    """Drop every preloaded block for this account after a rebuild changed it.

    Every grant combination has to go, not just the one that happens to be
    cached first -- a user whose grants differ from the rebuilder's would
    otherwise keep reading the old schema for the rest of the TTL.
    """
    account_id = str(account_id or "")
    stale = [key for key in _cache if key[0] == account_id]
    for key in stale:
        _cache.pop(key, None)
    if stale:
        log.info(
            "Preloaded KB dropped for %s after a rebuild (%d grant sets)",
            account_id, len(stale),
        )


def preload_account_kb(
    account_id: str,
    allowed_tables: set[str] | list[str] | None,
    *,
    cap: int = 0,
) -> tuple[str, list[str]]:
    """Assemble every permitted table's KB document into one stable block.

    Returns the block and the tables that made it in. An empty block means no
    document could be fetched for any permitted table, which is a real
    condition -- an account whose KB has never been built -- and the caller
    must fall back to retrieval rather than send the model nothing.
    """
    account_id = str(account_id or "")
    tables = sorted({str(t).strip().upper() for t in (allowed_tables or []) if str(t).strip()})
    if not account_id or not tables:
        return "", []

    key = (account_id, frozenset(tables))
    cached = _cache.get(key)
    if cached and (time.time() - cached[0]) < _TTL_SECONDS:
        return cached[1], list(cached[2])

    from core.vector_store import fetch_docs_for_fqn

    cap = cap or int(_PROMPT_CONTEXT_CHAR_CAP * _PREFIX_SHARE)
    docs: list[str] = []
    included: list[str] = []
    missing: list[str] = []
    budget_hit = False

    for fqn in tables:
        try:
            content = fetch_docs_for_fqn(account_id, fqn, sections=None)
        except Exception as exc:
            log.warning("Preload could not fetch %s for %s: %s", fqn, account_id, exc)
            content = None
        if not content:
            missing.append(fqn)
            continue
        doc = _clamp_kb_doc(content)
        # Stop before the assembled block would be truncated mid-document. A
        # tail-truncated prefix still caches, but it caches a document cut in
        # half, and the model cannot tell that is what it is reading.
        projected = sum(len(d) for d in docs) + len(doc) + (len(docs) * 7)
        if docs and projected > cap:
            budget_hit = True
            break
        docs.append(doc)
        included.append(fqn)

    if not docs:
        log.warning(
            "Preload found no KB document for any of the %d permitted tables "
            "on %s -- falling back to retrieval.", len(tables), account_id,
        )
        return "", []

    context = _clamp_prompt_context("\n\n---\n\n".join(docs), cap=cap)
    _cache[key] = (time.time(), context, tuple(included))

    log.info(
        "Preloaded KB for %s: %d/%d tables, %d chars%s%s",
        account_id, len(included), len(tables), len(context),
        f", {len(missing)} without a document" if missing else "",
        f", stopped at the {cap}-char prefix budget" if budget_hit else "",
    )
    if budget_hit:
        # Naming them matters: this is the list of tables the model cannot see
        # this question, and it is the first thing to check when an answer
        # picks the wrong table on a wide account.
        log.warning(
            "Preload budget reached on %s -- not sent: %s",
            account_id, ", ".join(t for t in tables if t not in set(included)),
        )
    return context, included
