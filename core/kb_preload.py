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

An account's knowledge base is usually small enough to simply send, and the
search stage then disappears along with its latency.

**It does not always fit, and the arithmetic is worth stating plainly.** The
per-document cap is 9,000 characters and the prefix budget is 85% of the
120,000-character prompt cap, so about 11 full-size documents fit -- not the 14
an earlier version of this docstring claimed. On an account whose documents run
near the cap, tables past the budget are skipped and named in a WARNING. When
that happens the premise of preloading is not being met: the model is seeing an
arbitrary subset chosen by sort order rather than by relevance, which is worse
than retrieval, not better. Treat that warning as a signal to raise
QUERYBOT_PROMPT_CONTEXT_CHAR_CAP or trim the documents.

Three things are load-bearing here:

- **The allow-list is the boundary.** Only tables the caller has already
  resolved as permitted are fetched. Preloading is a change to how much of the
  permitted knowledge base is sent, never to which knowledge base it is.
- **The order is fixed.** A prefix that reorders itself between questions
  caches nothing, and would fail silently rather than loudly, so documents are
  assembled in sorted order regardless of how the caller supplies them.
- **The account-wide documents come too.** The join map and business
  vocabulary are indexed under fqn "_global", which `fetch_docs_for_fqn`
  cannot reach; retrieval pinned them ahead of every ranked document, so
  replacing retrieval means fetching them explicitly.
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

    from core.vector_store import fetch_docs_for_fqn, fetch_global_docs

    cap = cap or int(_PROMPT_CONTEXT_CHAR_CAP * _PREFIX_SHARE)
    docs: list[str] = []
    included: list[str] = []
    missing: list[str] = []
    dropped: list[str] = []

    # The account-wide documents first, exactly where retrieval put them:
    # `_is_global` pins them ahead of the ranked table docs. They carry the
    # join map and the business vocabulary, they are not per-table, and
    # `fetch_docs_for_fqn` cannot see them at all -- so without this the
    # preload would send a knowledge base with the join map missing.
    try:
        docs.extend(_clamp_kb_doc(d) for d in fetch_global_docs(account_id) if d)
    except Exception as exc:
        log.warning("Preload could not fetch global docs for %s: %s", account_id, exc)

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
        # Skip, never stop. Documents are assembled in a fixed order so the
        # prefix is byte-stable, and an early `break` turned that fixed order
        # into a selection rule: one oversized table near the front of the
        # alphabet dropped every table behind it, however small. Skipping the
        # one that does not fit keeps the rest.
        projected = sum(len(d) for d in docs) + len(doc) + (len(docs) * 7)
        if projected > cap:
            # Including the first one. An earlier `docs and` here waved the
            # first document through unconditionally to guarantee the block was
            # never empty -- which meant a single oversized document was
            # admitted over the budget and then tail-truncated by
            # _clamp_prompt_context, cutting it in half, which is the one thing
            # this check exists to prevent. If nothing fits, `included` stays
            # empty and the caller falls back to retrieval, which is the honest
            # outcome: preloading cannot work for this account at this cap.
            dropped.append(fqn)
            continue
        docs.append(doc)
        included.append(fqn)

    if not included:
        # Global documents alone are not a knowledge base -- they name no
        # columns. Fall back to retrieval rather than prompt the model with a
        # join map and no tables.
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
        f", {len(dropped)} over the {cap}-char prefix budget" if dropped else "",
    )
    if dropped:
        # Naming them matters twice over. This is the list of tables the model
        # cannot see, so it is the first thing to check when an answer picks
        # the wrong table on a wide account -- and it means the premise of
        # preloading ("the model sees everything") is not being met, which is
        # a reason to raise QUERYBOT_PROMPT_CONTEXT_CHAR_CAP or trim the KB
        # documents, not something to leave running quietly.
        log.warning(
            "Preload budget reached on %s -- %d table(s) NOT sent to the model: %s",
            account_id, len(dropped), ", ".join(dropped),
        )
    return context, included
