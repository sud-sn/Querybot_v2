"""How well-curated a table is, and what that should do to retrieval.

KB retrieval already does the hard part — dense HNSW, BM25, reciprocal-rank
fusion, a cross-encoder reranker and a relevance floor that flags weak
retrieval instead of confidently returning the wrong table. The one thing it
cannot express is *curation status*: a table an admin has described, given
synonyms, modelled as an entity and assigned column roles ranks level with a
raw one the discovery pass found yesterday, at equal relevance.

It should not. A described table answers better because the model is told
what it means, and the effort someone spent describing it is a signal about
which table the business actually uses.

**The weight never crosses the relevance floor.** It is applied to the
ordering key only, after the reranker, and the score the floor reads is left
untouched. A well-curated irrelevant table therefore cannot be promoted past
a relevant one — it can only win a tie among documents that were all relevant
to begin with. That distinction is the whole reason this lives here rather
than inside the reranker.

The score is also the input to the modelling backlog: a table that keeps
being retrieved with a low curation score is the next thing worth describing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger("querybot.curation_weight")

# The most a fully-curated table's ordering key may be lifted. Small on
# purpose: enough to break a tie between two plausible tables, never enough
# to reorder a clear relevance gap. The cross-encoder separates relevant from
# irrelevant by an order of magnitude in sigmoid space, so 15% cannot cross
# that; it can only settle the cases where the reranker was already unsure.
MAX_BOOST = 0.15

# What counts as curation, and how much each is worth. A description is the
# single most useful thing an admin can add, because it is what the model
# actually reads; synonyms matter next because they decide whether the
# table is FOUND at all. Weights sum to 1.0 so a fully-curated table scores 1.
SIGNAL_WEIGHTS: dict[str, float] = {
    "described": 0.40,
    "synonyms": 0.25,
    "column_synonyms": 0.15,
    "modelled_as_entity": 0.12,
    "column_roles_assigned": 0.08,
}

# A description shorter than this is a placeholder, not documentation.
# "Customer table" tells the model nothing it could not read off the name.
MIN_DESCRIPTION_CHARS = 25

# Curation changes when an admin edits it, which is rare, and the lookup is
# several queries. Cached per account for the same reason the BM25 corpus is.
_CACHE_TTL_SEC = 300
_cache: dict[str, tuple[float, dict[str, float]]] = {}


def _normalise(table: str) -> str:
    """Uppercase, with the delimiters a catalogue writes stripped off.

    A bare ``.strip().upper()`` left ``[SALES].[ORDERS]`` intact, and the
    stores hold the same table as ``SALES.ORDERS`` -- so on every workspace
    whose KB header uses the bracketed form (which is what core/schema.py
    writes when no database is configured) every lookup missed and the boost
    was silently 1.0. Nothing logged and nothing failed.
    """
    parts = [
        part.strip().strip('"').strip("'").strip("[]").strip()
        for part in str(table or "").split(".")
    ]
    return ".".join(part for part in parts if part).upper()


def score_for_table(scores: dict[str, float], fqn: str) -> float:
    """This table's curation score, whatever spelling its document used.

    The indexed name comes from the document's H1: ``DB.SCHEMA.TABLE`` where a
    database is configured, ``[SCHEMA].[TABLE]`` where one is not. The signals
    are keyed on what the stores hold, which is the bare or schema-qualified
    name -- the entity graph has no database column at all. Matching the whole
    string only meant a three-part header never matched a two-part signal, so
    a table modelled as an entity and given column roles scored zero unless it
    also happened to carry a description row.

    Progressively shorter suffixes, most-qualified first, so a genuine
    ``DB.SALES.ORDERS`` entry still wins over a bare ``ORDERS``.
    """
    name = _normalise(fqn)
    if not name:
        return 0.0
    parts = name.split(".")
    for start in range(len(parts)):
        candidate = ".".join(parts[start:])
        if candidate in scores:
            return scores[candidate]
    return 0.0


def signals_for(account_id: str) -> dict[str, dict[str, bool]]:
    """Which curation signals each table in this workspace carries.

    Returns ``{TABLE_FQN: {signal: bool}}`` with tables keyed uppercase. Any
    store failure yields an empty mapping and a warning: retrieval must keep
    working without curation, and silently returning "nothing is curated"
    would look identical to "curation is switched off".
    """
    import store

    signals: dict[str, dict[str, bool]] = {}

    def _entry(table: str) -> dict[str, bool]:
        return signals.setdefault(_normalise(table), {
            key: False for key in SIGNAL_WEIGHTS
        })

    try:
        descriptions = store.list_table_descriptions(account_id) or {}
    except Exception as exc:
        log.warning("Table descriptions unavailable for %s: %s", account_id, exc)
        descriptions = {}
    for table, row in descriptions.items():
        entry = _entry(table)
        text = str((row or {}).get("description") or "").strip()
        entry["described"] = len(text) >= MIN_DESCRIPTION_CHARS
        entry["synonyms"] = bool(str((row or {}).get("synonyms") or "").strip())
        entry["column_synonyms"] = bool(
            str((row or {}).get("column_synonyms") or "").strip())

    try:
        entities = store.list_entities(account_id) or []
    except Exception as exc:
        log.warning("Entity graph unavailable for %s: %s", account_id, exc)
        entities = []

    roles_by_entity: dict[str, bool] = {}
    try:
        with store.get_db() as conn:
            for row in conn.execute(
                "SELECT DISTINCT entity_name FROM entity_properties "
                "WHERE account_id=? AND synonyms <> ''", (account_id,),
            ).fetchall():
                roles_by_entity[str(row["entity_name"])] = True
    except Exception as exc:
        log.warning("Entity properties unavailable for %s: %s", account_id, exc)

    for entity in entities:
        table = (entity or {}).get("table_name") or ""
        if not table:
            continue
        schema = str((entity or {}).get("schema_name") or "").strip()
        # An entity names a bare table; retrieval keys on the fully-qualified
        # name. Record both so whichever form the payload carries matches.
        for key in filter(None, (table, f"{schema}.{table}" if schema else "")):
            entry = _entry(key)
            entry["modelled_as_entity"] = True
            if roles_by_entity.get(str(entity.get("entity_name") or "")):
                entry["column_roles_assigned"] = True

    return signals


def score_from_signals(signals: dict[str, bool]) -> float:
    """One table's curation score, 0.0 to 1.0."""
    return round(sum(
        weight for key, weight in SIGNAL_WEIGHTS.items() if signals.get(key)
    ), 4)


def scores_for(account_id: str, *, use_cache: bool = True) -> dict[str, float]:
    """Curation score per table for this workspace, keyed uppercase."""
    now = time.time()
    if use_cache:
        cached = _cache.get(account_id)
        if cached and cached[0] > now:
            return cached[1]
    scores = {
        table: score_from_signals(flags)
        for table, flags in signals_for(account_id).items()
    }
    _cache[account_id] = (now + _CACHE_TTL_SEC, scores)
    return scores


def invalidate(account_id: str = "") -> None:
    """Drop the cache after an admin edits curation.

    Without an account, drops everything — used by tests and by any operation
    that rebuilds a workspace wholesale.
    """
    if account_id:
        _cache.pop(account_id, None)
    else:
        _cache.clear()


def ordering_key(hit: dict[str, Any], scores: dict[str, float]) -> float:
    """The value a candidate is sorted by once curation is taken into account.

    Reads ``_rerank_score`` and leaves it alone. A candidate the reranker did
    not score (the model was unavailable) is returned unchanged rather than
    boosted from nothing — curation is a tie-breaker between relevance
    judgements, not a substitute for one.
    """
    score = hit.get("_rerank_score")
    if score is None:
        return 0.0
    curation = score_for_table(scores, hit.get("fqn") or "")
    return float(score) * (1.0 + MAX_BOOST * curation)


def apply(hits: list[dict], scores: dict[str, float]) -> list[dict]:
    """Reorder relevant candidates by relevance-and-curation.

    Stamps ``_curation_score`` for the trace, so a "why did it pick that
    table" question can be answered from the record rather than re-derived.
    Candidates the reranker did not score keep their existing order at the
    end, exactly as they would without this step.
    """
    if not hits or not scores:
        return hits
    scored, unscored = [], []
    for hit in hits:
        if hit.get("_rerank_score") is None:
            unscored.append(hit)
            continue
        hit["_curation_score"] = score_for_table(scores, hit.get("fqn") or "")
        scored.append(hit)
    scored.sort(key=lambda h: ordering_key(h, scores), reverse=True)
    return scored + unscored
