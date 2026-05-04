"""
core/vector_store.py

Qdrant-backed vector store.  Drop-in replacement for every ChromaDB call in
the codebase.  A single Qdrant collection ("querybot_kb") holds both KB docs
and validated examples for all clients.  Multi-tenant isolation, multi-schema
ACL filtering, and concurrent writes are all handled correctly.

Collection: querybot_kb
  Each point represents one chunk (KB doc or validated example) and carries
  a full payload:

    account_id   str   — tenant key
    fqn          str   — fully-qualified table name  DB.SCHEMA.TABLE  (upper)
                          "_global" for join-map / business-vocab docs
    doc_type     str   — "kb" | "queries" | "example" | "global"
    content      str   — the text that was embedded (stored for prompt injection)
    table_name   str   — bare table name
    schema_name  str   — schema name (e.g. "HR")
    database     str   — database name (e.g. "CHATBOT_DB")
    source_file  str   — original filename if from a KB .md file
    question     str   — original question text (examples only)
    sql          str   — SQL string (examples only)

Environment variables
    QDRANT_URL   default http://localhost:6333
    QDRANT_API_KEY  optional, for cloud instances
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger("querybot.vector_store")

_COLLECTION   = "querybot_kb"
_EMBED_MODEL  = "all-MiniLM-L6-v2"
_VECTOR_DIM   = 384           # MiniLM-L6-v2 output dimension
_EF_SEARCH    = 128           # HNSW ef at search time — higher = better recall
_EF_CONSTRUCT = 64            # HNSW ef at index build time
_HNSW_M       = 16            # HNSW connectivity


# ── Singleton Qdrant client + embedding model ─────────────────────────────────

_qdrant_client = None
_embed_model   = None


def _qdrant() -> "QdrantClient":
    """Lazy singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        url     = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _qdrant_client = QdrantClient(url=url, api_key=api_key, timeout=30)
        _ensure_collection(_qdrant_client)
        log.info("Qdrant client connected: %s", url)
    return _qdrant_client


def _embedder():
    """Lazy singleton SentenceTransformer model."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(_EMBED_MODEL)
        log.info("Embedding model loaded: %s", _EMBED_MODEL)
    return _embed_model


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts → list of float vectors."""
    model = _embedder()
    return model.encode(texts, show_progress_bar=False).tolist()


def _ensure_collection(client) -> None:
    """Create the collection if it doesn't exist (idempotent)."""
    from qdrant_client.models import (
        Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    )
    existing = {c.name for c in client.get_collections().collections}
    if _COLLECTION not in existing:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(
                size=_VECTOR_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=_HNSW_M,
                    ef_construct=_EF_CONSTRUCT,
                    full_scan_threshold=10_000,
                ),
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=1_000,  # start HNSW after 1k points
            ),
        )
        # Payload indexes for fast filtering without full scan
        for field in ("account_id", "fqn", "doc_type"):
            client.create_payload_index(
                collection_name=_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        log.info("Qdrant collection '%s' created", _COLLECTION)


def _point_id(account_id: str, fqn: str, doc_type: str, extra: str = "") -> str:
    """
    Deterministic point ID so upserts are idempotent.
    Qdrant accepts UUID-format strings as IDs.
    """
    raw = f"{account_id}::{fqn}::{doc_type}::{extra}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    # Format as UUID
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _parse_fqn(fqn: str) -> tuple[str, str, str]:
    """
    Parse DB.SCHEMA.TABLE → (database, schema, table).
    Two-part SCHEMA.TABLE → ("", schema, table).
    Bare TABLE → ("", "", table).
    """
    parts = fqn.upper().split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    return "", "", parts[0]


# ══════════════════════════════════════════════════════════════════════════════
# KB document embedding
# ══════════════════════════════════════════════════════════════════════════════

def upsert_kb_file(
    account_id: str,
    fqn: str,                 # DB.SCHEMA.TABLE  (uppercase)
    doc_type: str,            # "kb" | "queries" | "global"
    content: str,
    source_file: str = "",
) -> None:
    """
    Embed and upsert a single KB document into Qdrant.

    Idempotent — running the KB build twice does not create duplicates because
    the point ID is deterministic.  Old versions are replaced in-place.
    """
    if not content.strip():
        return
    from qdrant_client.models import PointStruct

    db, schema, table = _parse_fqn(fqn) if fqn != "_global" else ("", "", "")
    vector = _embed([content])[0]
    point  = PointStruct(
        id=_point_id(account_id, fqn, doc_type),
        vector=vector,
        payload={
            "account_id":  account_id,
            "fqn":         fqn,
            "doc_type":    doc_type,
            "content":     content,
            "database":    db,
            "schema_name": schema,
            "table_name":  table,
            "source_file": source_file,
            "question":    "",
            "sql":         "",
            "ts":          int(time.time()),
        },
    )
    _qdrant().upsert(collection_name=_COLLECTION, points=[point])
    log.debug("Upserted %s / %s / %s", account_id, fqn, doc_type)


def upsert_kb_directory(
    account_id: str,
    kb_dir: str,
) -> int:
    """
    Embed all *.md files in kb_dir and upsert into Qdrant.

    File naming convention (matches existing KB build output):
      TABLENAME_kb.md      → doc_type "kb"
      TABLENAME_queries.md → doc_type "queries"
      _join_map.md         → doc_type "global", fqn "_global"
      _business_vocab.md   → doc_type "global", fqn "_global"

    Returns the number of documents upserted.
    """
    from qdrant_client.models import PointStruct

    path = Path(kb_dir)
    if not path.exists():
        log.warning("upsert_kb_directory: %s does not exist", kb_dir)
        return 0

    md_files = sorted(path.glob("*.md"))
    if not md_files:
        return 0

    # Build all points first, then batch-upsert
    points: list[PointStruct] = []
    db, schema = "", ""  # extracted from fqn stored inside first non-global doc

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue

        stem = md_file.stem  # e.g. "FACT_RXFILL_kb" or "_join_map"

        if stem.startswith("_"):
            fqn      = "_global"
            doc_type = "global"
        elif stem.endswith("_queries"):
            raw_name = stem[:-len("_queries")]
            # Try to reconstruct fqn from the schema embed stored in the MD header
            fqn      = _fqn_from_md_header(content) or raw_name.upper()
            doc_type = "queries"
        elif stem.endswith("_kb"):
            raw_name = stem[:-len("_kb")]
            fqn      = _fqn_from_md_header(content) or raw_name.upper()
            doc_type = "kb"
        else:
            fqn      = _fqn_from_md_header(content) or stem.upper()
            doc_type = "kb"

        db_p, sch_p, tbl_p = _parse_fqn(fqn) if fqn != "_global" else ("", "", "")
        vector = _embed([content])[0]
        points.append(PointStruct(
            id=_point_id(account_id, fqn, doc_type),
            vector=vector,
            payload={
                "account_id":  account_id,
                "fqn":         fqn,
                "doc_type":    doc_type,
                "content":     content,
                "database":    db_p,
                "schema_name": sch_p,
                "table_name":  tbl_p,
                "source_file": md_file.name,
                "question":    "",
                "sql":         "",
                "ts":          int(time.time()),
            },
        ))

    if not points:
        return 0

    # Batch upsert in chunks of 100 to avoid timeout on large KBs
    chunk = 100
    for i in range(0, len(points), chunk):
        _qdrant().upsert(collection_name=_COLLECTION, points=points[i:i+chunk])

    log.info("Qdrant: upserted %d KB docs for %s", len(points), account_id)
    return len(points)


def _fqn_from_md_header(content: str) -> str | None:
    """
    Extract the FQN from the first heading of a KB markdown file.
    The schema writer uses  # DB.SCHEMA.TABLE  as the first line.
    Falls back to None if not found.
    """
    for line in content.splitlines()[:5]:
        line = line.strip().lstrip("#").strip()
        if "." in line and not " " in line:
            return line.upper()
        # e.g. "# SCHEMA.TABLE — description"
        m = re.match(r"([A-Z0-9_]+\.[A-Z0-9_]+(?:\.[A-Z0-9_]+)?)", line.upper())
        if m:
            return m.group(1)
    return None


def delete_kb_for_client(account_id: str) -> None:
    """
    Delete all KB points for an account.  Called before a full KB rebuild
    so stale docs from tables that were removed don't linger.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    _qdrant().delete(
        collection_name=_COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(key="account_id", match=MatchValue(value=account_id))
        ]),
    )
    log.info("Qdrant: deleted all KB points for %s", account_id)


def re_embed_single_file(
    account_id: str,
    kb_dir: str,
    filename: str,
) -> None:
    """
    Re-embed a single KB file after admin inline edits.
    Replaces the existing point in-place (upsert is idempotent on point ID).
    """
    path = Path(kb_dir) / filename
    if not path.exists():
        log.warning("re_embed_single_file: %s not found", path)
        return
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    stem    = path.stem
    if stem.startswith("_"):
        fqn, doc_type = "_global", "global"
    elif stem.endswith("_queries"):
        raw  = stem[:-len("_queries")]
        fqn  = _fqn_from_md_header(content) or raw.upper()
        doc_type = "queries"
    else:
        raw  = stem[:-len("_kb")] if stem.endswith("_kb") else stem
        fqn  = _fqn_from_md_header(content) or raw.upper()
        doc_type = "kb"

    upsert_kb_file(account_id, fqn, doc_type, content, source_file=filename)
    log.info("Re-embedded %s for %s", filename, account_id)


# ══════════════════════════════════════════════════════════════════════════════
# Example embedding
# ══════════════════════════════════════════════════════════════════════════════

def upsert_examples(
    account_id: str,
    examples: list[tuple[str, str, str]],   # (question, sql, fqn)
) -> None:
    """
    Embed validated (question, sql, fqn) examples into Qdrant.
    The question is embedded; sql and fqn travel as payload.
    Idempotent — question hash used as ID so duplicates are overwritten.
    """
    from qdrant_client.models import PointStruct
    if not examples:
        return

    questions = [q for q, _, _ in examples]
    vectors   = _embed(questions)

    points = []
    for (question, sql, fqn), vector in zip(examples, vectors):
        fqn_upper    = fqn.upper() if fqn else "_global"
        db_p, sch_p, tbl_p = _parse_fqn(fqn_upper)
        points.append(PointStruct(
            id=_point_id(account_id, fqn_upper, "example", question),
            vector=vector,
            payload={
                "account_id":  account_id,
                "fqn":         fqn_upper,
                "doc_type":    "example",
                "content":     question,
                "database":    db_p,
                "schema_name": sch_p,
                "table_name":  tbl_p,
                "source_file": "",
                "question":    question,
                "sql":         sql,
                "ts":          int(time.time()),
            },
        ))

    _qdrant().upsert(collection_name=_COLLECTION, points=points)
    log.info("Qdrant: upserted %d examples for %s", len(points), account_id)


# ══════════════════════════════════════════════════════════════════════════════
# KB Retriever — mirrors the KBRetriever interface in core/knowledge.py
# ══════════════════════════════════════════════════════════════════════════════

class QdrantKBRetriever:
    """
    Semantic search over the Qdrant KB collection.

    Public interface is identical to KBRetriever in core/knowledge.py so that
    main.py can swap between backends by changing only the import / construction.

    ACL filtering is applied inside the HNSW search — not as a post-filter —
    so performance is constant regardless of how many tables exist in the KB.
    """

    def __init__(self, account_id: str):
        self._account_id = account_id
        self._client     = _qdrant()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _count(self) -> int:
        r = self._client.count(
            collection_name=_COLLECTION,
            count_filter=self._account_filter(),
            exact=False,
        )
        return r.count

    def _account_filter(self, allowed_fqns: list[str] | None = None):
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        must = [FieldCondition(key="account_id", match=MatchValue(value=self._account_id))]
        if allowed_fqns is not None:
            # Always include global docs (_join_map, _business_vocab)
            # alongside allowed table docs.
            fqn_filter = allowed_fqns + ["_global"]
            must.append(FieldCondition(key="fqn", match=MatchAny(any=fqn_filter)))
        return Filter(must=must)

    def _search(
        self,
        query: str,
        n: int,
        allowed_fqns: list[str] | None,
        doc_types: list[str] | None = None,
    ) -> list[dict]:
        """Core search — returns list of payload dicts sorted by score."""
        from qdrant_client.models import FieldCondition, MatchAny, Filter

        count = self._count()
        if count == 0:
            return []

        must_clauses = list(self._account_filter(allowed_fqns).must)
        if doc_types:
            must_clauses.append(
                FieldCondition(key="doc_type", match=MatchAny(any=doc_types))
            )

        hits = self._client.query_points(
            collection_name=_COLLECTION,
            query=_embed([query])[0],
            query_filter=Filter(must=must_clauses),
            limit=n,
            with_payload=True,
            search_params={"hnsw_ef": _EF_SEARCH},
        ).points
        return [hit.payload for hit in hits]

    @staticmethod
    def _is_global(doc: str) -> bool:
        """Return True if this is a global (join map / business vocab) doc."""
        low = doc[:200].lower()
        return (
            "join" in low or
            "business vocabulary" in low or
            "cross-table" in low
        )

    # ── Public API (mirrors KBRetriever) ─────────────────────────────────────

    def retrieve(
        self,
        question: str,
        n: int = 6,
        allowed_tables: set[str] | None = None,
    ) -> list[str]:
        """
        Return the top-n most relevant KB documents for the question.

        allowed_tables: set of fully-qualified table names the user may access.
            None → admin / unrestricted — all tables returned.
            Empty set → return only global docs (zero table access).
        """
        allowed_fqns = (
            [t.upper() for t in allowed_tables]
            if allowed_tables is not None else None
        )
        hits = self._search(question, n, allowed_fqns, doc_types=["kb", "queries", "global"])
        return [h["content"] for h in hits if h.get("content")]

    def retrieve_fact_patterns(
        self,
        question: str,
        n: int = 2,
        allowed_tables: set[str] | None = None,
    ) -> list[str]:
        """
        Retrieve Stage-2 query-pattern docs that contain JOIN examples.
        Used when the question has grouping/aggregation keywords.
        """
        allowed_fqns = (
            [t.upper() for t in allowed_tables]
            if allowed_tables is not None else None
        )
        # Bias the query toward join/aggregation patterns
        fact_query = f"{question} join aggregation group by"
        hits = self._search(
            fact_query, n * 3, allowed_fqns, doc_types=["queries", "global"]
        )
        fact_docs = []
        for h in hits:
            content = h.get("content", "")
            has_join = "JOIN" in content.upper()
            is_query_doc = h.get("doc_type") == "queries"
            if (has_join or is_query_doc) and len(fact_docs) < n:
                fact_docs.append(content)
        return fact_docs


# ══════════════════════════════════════════════════════════════════════════════
# Example retrieval — mirrors retrieve_similar_examples in core/examples.py
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar_examples(
    account_id: str,
    question: str,
    n: int = 3,
    allowed_tables: set[str] | None = None,
) -> list[dict]:
    """
    Return the top-n most semantically similar validated examples.

    Each result: {"question": str, "sql": str, "table": str}
    Returns [] if no examples exist yet.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    count = _qdrant().count(
        collection_name=_COLLECTION,
        count_filter=Filter(must=[
            FieldCondition(key="account_id", match=MatchValue(value=account_id)),
            FieldCondition(key="doc_type",   match=MatchValue(value="example")),
        ]),
        exact=False,
    ).count
    if count == 0:
        return []

    must = [
        FieldCondition(key="account_id", match=MatchValue(value=account_id)),
        FieldCondition(key="doc_type",   match=MatchValue(value="example")),
    ]
    if allowed_tables is not None:
        must.append(
            FieldCondition(key="fqn", match=MatchAny(any=[t.upper() for t in allowed_tables]))
        )

    try:
        hits = _qdrant().query_points(
            collection_name=_COLLECTION,
            query=_embed([question])[0],
            query_filter=Filter(must=must),
            limit=n,
            with_payload=True,
            search_params={"hnsw_ef": _EF_SEARCH},
        ).points
    except Exception as e:
        log.error("Example retrieval failed: %s", e)
        return []

    return [
        {
            "question": h.payload.get("question", ""),
            "sql":      h.payload.get("sql", ""),
            "table":    h.payload.get("table_name", ""),
        }
        for h in hits
        if h.payload.get("sql")
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Convenience factory — mirrors load_retriever in core/knowledge.py
# ══════════════════════════════════════════════════════════════════════════════

def load_retriever(account_id: str) -> QdrantKBRetriever:
    """
    Return a QdrantKBRetriever for the given account.

    The account_id replaces the chroma_dir parameter — Qdrant is a shared
    service so no per-client directory path is needed.
    """
    return QdrantKBRetriever(account_id)
