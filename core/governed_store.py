"""
core/governed_store.py

Qdrant-backed governed example store.

Collection: querybot_governed  (separate from querybot_kb -- legacy stays read-only)

Purpose
-------
Holds human-approved learning candidates for use as few-shot examples at
query time.  The separation from querybot_kb is intentional:

  querybot_kb       -- KB docs + legacy auto-harvested examples.  Read-only
                       for the governed loop.  Never written by the learning
                       pipeline once enable_feedback_collection=1.
  querybot_governed -- Admin-approved candidates only.  Written exclusively by
                       the learning loop (approve action + backfill utility).
                       Read alongside querybot_kb at query time (dual retrieval).

Payload schema (one point per approved candidate)
--------------------------------------------------
  candidate_id  str  -- primary key linking to learning_candidate row
  account_id    str  -- tenant key
  question      str  -- question text (embedded for semantic retrieval)
  sql           str  -- SQL (original or admin-corrected)
  source        str  -- "auto" | "admin_correction" | "pre_governed"
  final_score   int  -- quality score at time of approval (0-100)
  doc_type      str  -- always "governed_example"
  ts            int  -- Unix timestamp of embedding

ACL note
--------
Governed examples are tenant-isolated by account_id only.  Table-level ACL
filtering is intentionally omitted because:
  (a) The admin has explicitly ratified each example.
  (b) Examples are used as few-shot SQL patterns, not raw data results.
  (c) The main SQL generation still enforces table-level ACL independently.

Point IDs
---------
IDs are deterministic: MD5 of "governed::{candidate_id}" formatted as a UUID.
This makes upsert idempotent -- re-approving (e.g. after a SQL correction)
safely overwrites the previous embedding with no orphaned points.
"""

from __future__ import annotations

import hashlib
import logging
import time

log = logging.getLogger("querybot.governed_store")

_GOVERNED_COLLECTION = "querybot_governed"
_VECTOR_DIM          = 384    # all-MiniLM-L6-v2 output dimension
_EF_SEARCH           = 128    # HNSW ef at search time


# ── Shared Qdrant client + embedding model (re-uses vector_store singletons) ──

def _qdrant():
    """
    Re-use the shared Qdrant client from core.vector_store.
    This avoids creating a second TCP connection to the same instance.
    """
    from core.vector_store import _qdrant as _vs_qdrant
    return _vs_qdrant()


def _embed(texts: list[str]) -> list[list[float]]:
    """Re-use the shared sentence-transformer embedder from core.vector_store."""
    from core.vector_store import _embed as _vs_embed
    return _vs_embed(texts)


# ── Deterministic point ID ────────────────────────────────────────────────────

def _governed_point_id(candidate_id: str) -> str:
    """
    UUID-format deterministic Qdrant point ID for a governed example.

    Using the candidate_id as the seed means:
    - upsert == update (idempotent re-approve)
    - delete needs only the candidate_id (no qdrant_id lookup required)
    """
    digest = hashlib.md5(f"governed::{candidate_id}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


# ── Collection setup ──────────────────────────────────────────────────────────

def _ensure_governed_collection(client) -> None:
    """
    Create querybot_governed if it doesn't exist, then ensure all required
    payload indexes are present.  Fully idempotent.
    """
    from qdrant_client.models import (
        Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    )
    existing = {c.name for c in client.get_collections().collections}
    if _GOVERNED_COLLECTION not in existing:
        client.create_collection(
            collection_name=_GOVERNED_COLLECTION,
            vectors_config=VectorParams(
                size=_VECTOR_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=64,
                    full_scan_threshold=10_000,
                ),
            ),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=500),
        )
        log.info("Qdrant collection '%s' created", _GOVERNED_COLLECTION)

    # Payload indexes -- idempotent; Qdrant ignores duplicate creation.
    for field in ("account_id", "doc_type", "source", "candidate_id"):
        try:
            client.create_payload_index(
                collection_name=_GOVERNED_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass   # already exists or Qdrant version doesn't support it


# ══════════════════════════════════════════════════════════════════════════════
# Write-side API
# ══════════════════════════════════════════════════════════════════════════════

def upsert_governed_example(
    candidate_id: str,
    account_id: str,
    question: str,
    sql: str,
    source: str = "auto",
    final_score: int = 0,
    schema_scope: str = "",
) -> str:
    """
    Embed and upsert a governed example into querybot_governed.

    Idempotent -- the point ID is deterministic from candidate_id so
    calling this twice (e.g. after a corrected SQL update + re-approve)
    safely overwrites the previous embedding without leaving orphaned points.

    schema_scope is stored in the payload so retrieve_governed_examples can
    filter by schema — preventing cross-schema example leakage.  An empty
    string means "unscoped / valid for all schemas" (used for pre-governed
    examples loaded before schema isolation was introduced).

    Returns the Qdrant point ID string so the caller can store it in the
    learning_candidate.qdrant_id column for observability.
    Returns "" when the inputs are empty (skipped without error).
    """
    from qdrant_client.models import PointStruct

    if not question.strip() or not sql.strip():
        log.warning(
            "upsert_governed_example: empty question or SQL for candidate %s -- skipped",
            candidate_id[:12] if candidate_id else "?",
        )
        return ""

    client   = _qdrant()
    _ensure_governed_collection(client)

    point_id = _governed_point_id(candidate_id)
    vector   = _embed([question])[0]

    client.upsert(
        collection_name=_GOVERNED_COLLECTION,
        points=[PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "candidate_id": candidate_id,
                "account_id":   account_id,
                "question":     question,
                "sql":          sql,
                "source":       source,
                "final_score":  final_score,
                "schema_scope": schema_scope,
                "doc_type":     "governed_example",
                "ts":           int(time.time()),
            },
        )],
    )
    log.info(
        "governed_store: upserted candidate %s for account %s (source=%s score=%d schema=%s)",
        candidate_id[:12], account_id, source, final_score, schema_scope or "all",
    )
    return point_id


def delete_governed_example(candidate_id: str, account_id: str = "") -> None:
    """
    Remove a governed example from querybot_governed.

    Called when a candidate is revoked.  No-op (with a warning) if the point
    doesn't exist or Qdrant is unavailable.

    account_id is used for logging only.
    """
    from qdrant_client.models import PointIdsList

    point_id = _governed_point_id(candidate_id)
    try:
        _qdrant().delete(
            collection_name=_GOVERNED_COLLECTION,
            points_selector=PointIdsList(points=[point_id]),
        )
        log.info(
            "governed_store: deleted candidate %s (account=%s)",
            candidate_id[:12], account_id or "?",
        )
    except Exception as exc:
        log.warning(
            "governed_store: delete failed for candidate %s -- %s",
            candidate_id[:12] if candidate_id else "?", exc,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Read-side API
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_governed_examples(
    account_id: str,
    question: str,
    n: int = 3,
    allowed_tables: set[str] | None = None,   # reserved -- see ACL note in module docstring
    schema_scope: str = "",
) -> list[dict]:
    """
    Semantic search over querybot_governed for the given tenant.

    Returns up to n results as dicts with keys:
      question  str  -- the original question
      sql       str  -- the approved SQL (original or corrected)
      table     str  -- always "" (governed examples aren't FQN-tagged)
      source    str  -- "auto" | "admin_correction" | "pre_governed"

    Returns [] when:
      - The collection doesn't exist yet (fresh install)
      - No governed examples exist for this account
      - Qdrant is unavailable (graceful degradation -- caller gets legacy only)

    schema_scope: when non-empty, only examples whose schema_scope matches OR
    is unscoped (empty / absent) are returned.  This prevents an approved
    example from schema A from being injected into a query scoped to schema B.
    When empty (default), no schema filtering is applied — all examples for
    the account are considered.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField

    # Guard: check collection exists before querying so we never emit noisy
    # "collection not found" errors on fresh deployments.
    try:
        client   = _qdrant()
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        log.warning("governed_store: can't list collections -- %s", exc)
        return []

    if _GOVERNED_COLLECTION not in existing:
        return []

    # Base filter: always scope to account + doc_type.
    # When a schema_scope is requested, add a should clause that matches:
    #   (a) examples explicitly tagged with this schema
    #   (b) examples tagged as unscoped ("")   — valid across all schemas
    #   (c) legacy examples with no schema_scope field in payload
    # This prevents cross-schema leakage while preserving backward compat.
    must_conditions: list = [
        FieldCondition(key="account_id", match=MatchValue(value=account_id)),
        FieldCondition(key="doc_type",   match=MatchValue(value="governed_example")),
    ]
    should_conditions: list = []
    if schema_scope:
        should_conditions = [
            FieldCondition(key="schema_scope", match=MatchValue(value=schema_scope)),
            FieldCondition(key="schema_scope", match=MatchValue(value="")),
            IsNullCondition(is_null=PayloadField(key="schema_scope")),
        ]

    _filter = Filter(
        must=must_conditions,
        should=should_conditions if should_conditions else None,
    )

    count = client.count(
        collection_name=_GOVERNED_COLLECTION,
        count_filter=_filter,
        exact=False,
    ).count
    if count == 0:
        return []

    try:
        vector = _embed([question])[0]
        hits = client.query_points(
            collection_name=_GOVERNED_COLLECTION,
            query=vector,
            query_filter=_filter,
            limit=n,
            with_payload=True,
            search_params={"hnsw_ef": _EF_SEARCH},
        ).points
    except Exception as exc:
        log.warning("governed_store: retrieve failed for %s -- %s", account_id, exc)
        return []

    results = []
    for h in hits:
        q = (h.payload or {}).get("question", "")
        s = (h.payload or {}).get("sql", "")
        if q and s:
            results.append({
                "question": q,
                "sql":      s,
                "table":    "",
                "source":   (h.payload or {}).get("source", "auto"),
            })
    return results


def get_governed_count(account_id: str) -> int:
    """
    Return the approximate number of governed examples for an account.

    Used for admin dashboard badges and health checks.
    Returns 0 on any error (non-raising).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        client   = _qdrant()
        existing = {c.name for c in client.get_collections().collections}
        if _GOVERNED_COLLECTION not in existing:
            return 0
        return client.count(
            collection_name=_GOVERNED_COLLECTION,
            count_filter=Filter(must=[
                FieldCondition(key="account_id", match=MatchValue(value=account_id)),
            ]),
            exact=False,
        ).count
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# Backfill utility
# ══════════════════════════════════════════════════════════════════════════════

def backfill_approved_candidates(account_id: str) -> int:
    """
    Embed all approved learning_candidate rows into querybot_governed.

    Idempotent -- upsert overwrites existing points so safe to re-run after
    a SQL correction or a bulk import.  Useful for:
      - First-time provisioning when candidates were approved before governed
        collection existed
      - Recovery after a Qdrant index rebuild

    Returns the number of candidates successfully upserted.
    """
    from store.learning_store import list_candidates

    approved = list_candidates(account_id, status="approved", limit=9999)
    if not approved:
        log.info("governed_store: no approved candidates to backfill for %s", account_id)
        return 0

    count = 0
    for c in approved:
        # Prefer corrected SQL over original SQL
        sql = c.get("corrected_sql") or c.get("sql_text", "")
        if not sql:
            continue
        try:
            upsert_governed_example(
                candidate_id=c["candidate_id"],
                account_id=account_id,
                question=c.get("question_text", ""),
                sql=sql,
                source=c.get("source", "auto"),
                final_score=int(c.get("final_score") or 0),
                schema_scope=c.get("schema_scope", ""),
            )
            count += 1
        except Exception as exc:
            cid = (c.get("candidate_id") or "?")[:12]
            log.warning("governed_store: backfill failed for %s -- %s", cid, exc)

    log.info(
        "governed_store: backfilled %d/%d approved candidates for %s",
        count, len(approved), account_id,
    )
    return count
