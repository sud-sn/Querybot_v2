"""
core/vector_store.py

Qdrant-backed vector store.  Drop-in replacement for every ChromaDB call in
the codebase.  A single Qdrant collection ("querybot_kb") holds both KB docs
and validated examples for all clients.  Multi-tenant isolation, multi-schema
ACL filtering, and concurrent writes are all handled correctly.

Collection: querybot_kb
  Each point represents one chunk (KB section, query pattern, or example) and
  carries a full payload:

    account_id   str   — tenant key
    fqn          str   — fully-qualified table name  DB.SCHEMA.TABLE  (upper)
                          "_global" for join-map / business-vocab docs
    doc_type     str   — "kb" | "queries" | "example" | "global"
    section_type str   — section slug for KB chunks; "full" for whole-doc points
                          one of: overview | key_metrics | always_exclude |
                          columns | patterns | join_keys | synonyms | full
    content      str   — the text that was embedded (stored for prompt injection)
    table_name   str   — bare table name
    schema_name  str   — schema name (e.g. "HR")
    database     str   — database name (e.g. "CHATBOT_DB")
    source_file  str   — original filename if from a KB .md file
    question     str   — original question text (examples only)
    sql          str   — SQL string (examples only)

Chunking strategy (KB docs only):
  doc_type="kb"      → split into per-section points (7 per table).
                        Each section has its own embedding vector so BM25
                        and dense search find the right *part* of the KB,
                        not just the right table.
  doc_type="queries" → whole-doc embedding (SQL pattern docs already focused)
  doc_type="global"  → whole-doc (join-map / business-vocab — cross-table)
  doc_type="example" → one point per validated example (question embedded)

Environment variables
    QDRANT_URL          default http://localhost:6333
    QDRANT_API_KEY      optional, for cloud instances
    QUERYBOT_RERANK     set to "false" to disable cross-encoder re-ranking
                        (useful when latency is more important than precision)

Retrieval pipeline (per query):
  1. Dense HNSW search (all-MiniLM-L6-v2, cosine) — top n×4 candidates
  2. BM25 keyword search (rank_bm25) over in-memory corpus — top n×4 candidates
     BM25 corpus is cached per (account_id, allowed_fqns) with a 5-min TTL
     and invalidated whenever the KB is rebuilt or a file is re-embedded.
  3. Reciprocal Rank Fusion (RRF, k=60) — merges + deduplicates both lists
  4. Cross-encoder re-ranking (ms-marco-MiniLM-L6-v2) — top 20 candidates
     re-scored with full (query, document) attention, ~5 ms per pair
  5. Return top-n documents

Graceful degradation:
  - rank_bm25 not installed → dense-only (no BM25)
  - sentence-transformers cross-encoder unavailable → RRF order kept
  - Any step failure → falls back to the previous step's output
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

# ── Hybrid retrieval tuning ───────────────────────────────────────────────────
_RRF_K            = 60    # RRF damping constant — standard value
_RERANK_POOL      = 20    # cross-encoder sees top-N candidates from RRF
_BM25_TTL_SEC     = 300   # BM25 corpus cache TTL (5 minutes)
_RERANK_ENABLED   = os.getenv("QUERYBOT_RERANK", "true").lower() != "false"

# ── BM25 corpus cache ─────────────────────────────────────────────────────────
# Key:   "{account_id}:{fqns_str}:{dtypes_str}"
# Value: (expires_at: float, BM25Okapi, docs: list[dict])
_bm25_index_cache: dict[str, tuple[float, object, list[dict]]] = {}

# ── Cross-encoder lazy singleton ──────────────────────────────────────────────
_cross_encoder_singleton = None  # None = not yet loaded; False = load failed


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


# ══════════════════════════════════════════════════════════════════════════════
# Section-level chunking helpers
# ══════════════════════════════════════════════════════════════════════════════

# Map normalised ## heading text → section_type slug stored in payload
_SECTION_SLUG: dict[str, str] = {
    "overview":              "overview",
    "key metrics":           "key_metrics",
    "always exclude":        "always_exclude",
    "columns":               "columns",
    "common query patterns": "patterns",
    "join keys":             "join_keys",
    "business synonyms":     "synonyms",
}

# Canonical display order when reassembling a full doc from section chunks
_SECTION_ORDER: list[str] = [
    "overview", "key_metrics", "always_exclude",
    "columns", "patterns", "join_keys", "synonyms",
    "preamble", "full",
]


def _section_sort_key(section_type: str) -> int:
    """Return an integer rank for canonical section ordering."""
    try:
        return _SECTION_ORDER.index(section_type)
    except ValueError:
        return len(_SECTION_ORDER)   # unknown sections sort last


def _extract_fqn_header(content: str) -> str | None:
    """
    Return the top-level heading line  "# DB.SCHEMA.TABLE"  from a KB doc, or
    None if not found.  Used to prepend the FQN context to every section chunk.
    """
    for line in content.splitlines()[:8]:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            return stripped
    return None


def _split_kb_sections(fqn_header: str, content: str) -> list[tuple[str, str]]:
    """
    Split a KB markdown document into (section_type, chunk_text) pairs.

    Each chunk starts with the FQN header line so that:
      • `_fqn_from_md_header()` and `table_coverage._fqn_from_doc()` still work
      • The LLM always sees which table a section belongs to

    Rules:
      • Split only on  "## " lines (not "### " sub-headers)
      • Unknown headings get a normalised slug derived from the heading text
      • If no ## sections exist → return [("full", content)] (whole-doc fallback)
      • Empty sections are skipped
    """
    lines       = content.splitlines()
    sections:   list[tuple[str, list[str]]] = []   # (slug, lines)
    cur_slug:   str | None = None
    cur_lines:  list[str]  = []

    for line in lines:
        stripped = line.strip()
        # Detect a section boundary:  "## Heading" but NOT "### Sub-heading"
        if (stripped.startswith("## ")
                and not stripped.startswith("### ")
                and len(stripped) > 3):
            # Save previous section
            if cur_slug is not None:
                sections.append((cur_slug, cur_lines))
            # Determine slug for the new section
            heading_raw = stripped[3:].strip().lower()
            slug = None
            for key, val in _SECTION_SLUG.items():
                if heading_raw.startswith(key):
                    slug = val
                    break
            if slug is None:
                # Non-standard heading — normalise to a slug
                slug = re.sub(r"[^a-z0-9]+", "_", heading_raw).strip("_") or "section"
            cur_slug  = slug
            cur_lines = [line]
        else:
            cur_lines.append(line)

    # Flush the last section
    if cur_slug is not None:
        sections.append((cur_slug, cur_lines))

    if not sections:
        # No ## headings found — store as a single "full" chunk
        return [("full", content)]

    result: list[tuple[str, str]] = []
    for slug, sec_lines in sections:
        body = "\n".join(sec_lines).strip()
        if not body:
            continue
        # Prepend the FQN header so retrieval context is always complete
        chunk = f"{fqn_header}\n\n{body}"
        result.append((slug, chunk))

    return result if result else [("full", content)]


def _reconstruct_full_doc(fqn: str, section_payloads: list[dict]) -> str | None:
    """
    Rebuild a coherent KB document from a list of section payloads.

    Sections are ordered canonically (overview → columns → synonyms …).
    The FQN header appears once at the top; repeated headers inside section
    chunks are stripped to avoid redundancy.
    """
    # Sort sections in canonical display order
    ordered = sorted(section_payloads, key=lambda p: _section_sort_key(p.get("section_type", "full")))

    fqn_header: str | None = None
    bodies: list[str] = []

    for payload in ordered:
        content = (payload.get("content") or "").strip()
        if not content:
            continue
        lines = content.splitlines()
        # First line of each chunk is the repeated "# FQN" header
        if lines and lines[0].strip().startswith("#") and not lines[0].strip().startswith("##"):
            if fqn_header is None:
                fqn_header = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        else:
            body = content
        if body:
            bodies.append(body)

    if not bodies:
        return None
    header = fqn_header or f"# {fqn.upper()}"
    return header + "\n\n" + "\n\n".join(bodies)


def _group_chunks_by_table(hits: list[dict]) -> list[str]:
    """
    Merge section-level chunks from the same table into one coherent doc string.

    Table ordering is determined by the rank of each table's best-scoring chunk
    (first chunk that appeared in the re-ranked list).  Within each table,
    sections are assembled in canonical order regardless of retrieval rank.

    Global docs (fqn="_global", join maps, business vocab) are passed through
    unchanged and returned first so  `retriever._is_global()`  still works.
    """
    from collections import defaultdict

    global_docs:      list[str]                     = []
    table_first_rank: dict[str, int]                = {}
    table_chunks:     dict[str, list[dict]]         = defaultdict(list)

    for rank, payload in enumerate(hits):
        fqn     = payload.get("fqn", "")
        content = payload.get("content", "")

        if fqn == "_global" or not fqn:
            if content:
                global_docs.append(content)
            continue

        if fqn not in table_first_rank:
            table_first_rank[fqn] = rank
        table_chunks[fqn].append(payload)

    # Order tables by their first-seen rank (best-scoring chunk wins)
    ordered_fqns = sorted(table_first_rank, key=lambda f: table_first_rank[f])

    result: list[str] = list(global_docs)   # globals first

    for fqn in ordered_fqns:
        chunks = table_chunks[fqn]

        # Check whether these are section chunks or legacy whole-doc payloads
        has_sections = any(
            c.get("section_type") not in ("full", None)
            for c in chunks
        )

        if has_sections:
            merged = _reconstruct_full_doc(fqn, chunks)
            if merged:
                result.append(merged)
        else:
            # Whole-doc payload (queries doc or legacy KB) — take the best one
            best = max(chunks, key=lambda c: _section_sort_key(c.get("section_type", "full")))
            content = best.get("content", "")
            if content:
                result.append(content)

    return result


def _delete_points_for_fqn_doctype(account_id: str, fqn: str, doc_type: str) -> None:
    """
    Delete all Qdrant points for a specific (account_id, fqn, doc_type) triple.

    Called before re-embedding a single KB file so stale section chunks from
    the old version don't accumulate (section count can change after edits).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        _qdrant().delete(
            collection_name=_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="account_id", match=MatchValue(value=account_id)),
                FieldCondition(key="fqn",        match=MatchValue(value=fqn)),
                FieldCondition(key="doc_type",   match=MatchValue(value=doc_type)),
            ]),
        )
        log.debug("Deleted old points: %s / %s / %s", account_id, fqn, doc_type)
    except Exception as exc:
        log.warning("_delete_points_for_fqn_doctype failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# BM25 + RRF + Cross-encoder helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bm25_tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric chars, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return [t for t in text.split() if t]


def _invalidate_bm25_cache(account_id: str) -> None:
    """Remove all BM25 cache entries for this account (called after every upsert)."""
    stale = [k for k in list(_bm25_index_cache) if k.startswith(account_id + ":")]
    for k in stale:
        _bm25_index_cache.pop(k, None)
    if stale:
        log.debug("BM25 cache invalidated for %s (%d entries removed)", account_id, len(stale))


def _get_bm25(
    account_id: str,
    allowed_fqns: list[str] | None,
    doc_types: list[str] | None = None,
) -> tuple[object, list[dict]] | None:
    """
    Return (BM25Okapi, docs) for the given account + ACL scope.

    Scrolls Qdrant to build the corpus if the cache is cold/expired.
    Returns None if rank_bm25 is not installed or the corpus is empty.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None

    # Cache key encodes the full ACL scope — different roles → separate indexes
    fqns_str   = "|".join(sorted(allowed_fqns)) if allowed_fqns is not None else "*"
    dtypes_str = "|".join(sorted(doc_types))    if doc_types      else "*"
    cache_key  = f"{account_id}:{fqns_str}:{dtypes_str}"

    entry = _bm25_index_cache.get(cache_key)
    if entry and entry[0] > time.time():
        return entry[1], entry[2]          # cache hit

    # Build corpus via Qdrant scroll (no vector transfer needed)
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    must = [FieldCondition(key="account_id", match=MatchValue(value=account_id))]
    if allowed_fqns is not None:
        must.append(
            FieldCondition(key="fqn", match=MatchAny(any=allowed_fqns + ["_global"]))
        )
    if doc_types:
        must.append(FieldCondition(key="doc_type", match=MatchAny(any=doc_types)))

    client = _qdrant()
    docs: list[dict] = []
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        docs.extend(r.payload for r in results if r.payload)
        if next_offset is None:
            break
        offset = next_offset

    if not docs:
        return None

    corpus = [_bm25_tokenize(d.get("content", "")) for d in docs]
    try:
        bm25 = BM25Okapi(corpus)
    except Exception as exc:
        log.warning("BM25 index build failed: %s", exc)
        return None

    _bm25_index_cache[cache_key] = (time.time() + _BM25_TTL_SEC, bm25, docs)
    log.debug("BM25 index built: %d docs for %s", len(docs), account_id)
    return bm25, docs


def _bm25_search(
    bm25: object,
    docs: list[dict],
    query: str,
    n: int,
    doc_types: list[str] | None = None,
) -> list[dict]:
    """Score corpus with BM25, apply optional doc_type filter, return top-n."""
    tokens = _bm25_tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results: list[dict] = []
    for idx, score in indexed:
        if score <= 0:
            break
        doc = docs[idx]
        if doc_types and doc.get("doc_type") not in doc_types:
            continue
        results.append(doc)
        if len(results) >= n:
            break
    return results


def _rrf_fuse(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k: int = _RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — merge and deduplicate two ranked lists.

    score(doc) = Σ  1 / (k + rank_in_list)

    Deduplication key: payload["id"] when present, else MD5 of first 512
    chars of content.
    """
    def _key(doc: dict) -> str:
        return str(doc.get("id") or hashlib.md5(
            (doc.get("content") or "").encode()[:512]
        ).hexdigest())

    scores: dict[str, float]  = {}
    docs_by_key: dict[str, dict] = {}

    for rank, doc in enumerate(dense_hits, start=1):
        key = _key(doc)
        scores[key]      = scores.get(key, 0.0) + 1.0 / (k + rank)
        docs_by_key[key] = doc

    for rank, doc in enumerate(bm25_hits, start=1):
        key = _key(doc)
        scores[key]      = scores.get(key, 0.0) + 1.0 / (k + rank)
        docs_by_key[key] = doc

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs_by_key[key] for key, _ in ordered]


def _get_cross_encoder():
    """
    Lazy singleton cross-encoder (ms-marco-MiniLM-L6-v2, ~22 MB).

    Returns the model, or None if unavailable / disabled.
    """
    global _cross_encoder_singleton
    if _cross_encoder_singleton is None and _RERANK_ENABLED:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            _cross_encoder_singleton = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L6-v2",
                max_length=512,
            )
            log.info("Cross-encoder loaded: ms-marco-MiniLM-L6-v2")
        except Exception as exc:
            log.warning("Cross-encoder unavailable, RRF order will be used: %s", exc)
            _cross_encoder_singleton = False    # sentinel — don't retry
    # False means load failed; return None to signal "skip re-rank"
    return _cross_encoder_singleton if _cross_encoder_singleton else None


def _rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Re-score candidates with the cross-encoder and return top_n.
    Falls back to input order (RRF order) if the model is unavailable.
    """
    if not candidates:
        return candidates
    model = _get_cross_encoder()
    if model is None:
        return candidates[:top_n]
    try:
        pairs  = [(query, d.get("content", "")[:512]) for d in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]
    except Exception as exc:
        log.warning("Re-ranking failed, falling back to RRF order: %s", exc)
        return candidates[:top_n]


def _ensure_collection(client) -> None:
    """
    Create the collection if it doesn't exist, then ensure all required
    payload indexes exist (idempotent — Qdrant ignores duplicate index creation).
    """
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
        log.info("Qdrant collection '%s' created", _COLLECTION)

    # Ensure all payload indexes exist — called unconditionally so that
    # section_type (added for chunking) is indexed on existing deployments too.
    # create_payload_index is idempotent: no error if the index already exists.
    for field in ("account_id", "fqn", "doc_type", "section_type"):
        try:
            client.create_payload_index(
                collection_name=_COLLECTION,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass   # already exists or Qdrant version doesn't support it


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

def _build_kb_section_points(
    account_id: str,
    fqn: str,
    content: str,
    source_file: str,
) -> list:
    """
    Build Qdrant PointStruct objects for every section of a KB overview doc.

    Returns a list of PointStructs ready for batch upsert.
    Each point carries the full FQN header so retrieval context is always
    complete, plus a `section_type` field for targeted filtering.
    """
    from qdrant_client.models import PointStruct

    db, schema, table = _parse_fqn(fqn) if fqn != "_global" else ("", "", "")
    fqn_header = _extract_fqn_header(content) or f"# {fqn}"
    sections   = _split_kb_sections(fqn_header, content)

    # Embed all section chunks in one batched call (one model.encode call)
    texts   = [chunk for _, chunk in sections]
    vectors = _embed(texts)

    now = int(time.time())
    points = []
    for (section_type, chunk), vector in zip(sections, vectors):
        if not chunk.strip():
            continue
        points.append(PointStruct(
            id=_point_id(account_id, fqn, "kb", extra=f"section_{section_type}"),
            vector=vector,
            payload={
                "account_id":   account_id,
                "fqn":          fqn,
                "doc_type":     "kb",
                "section_type": section_type,
                "content":      chunk,
                "database":     db,
                "schema_name":  schema,
                "table_name":   table,
                "source_file":  source_file,
                "question":     "",
                "sql":          "",
                "ts":           now,
            },
        ))
    return points


def _build_whole_doc_point(
    account_id: str,
    fqn: str,
    doc_type: str,
    content: str,
    source_file: str,
) -> object:
    """
    Build a single PointStruct for a whole-doc embedding (queries / global).
    `section_type` is set to "full" for forward-compatibility filtering.
    """
    from qdrant_client.models import PointStruct

    db, schema, table = _parse_fqn(fqn) if fqn != "_global" else ("", "", "")
    vector = _embed([content])[0]
    return PointStruct(
        id=_point_id(account_id, fqn, doc_type),
        vector=vector,
        payload={
            "account_id":   account_id,
            "fqn":          fqn,
            "doc_type":     doc_type,
            "section_type": "full",
            "content":      content,
            "database":     db,
            "schema_name":  schema,
            "table_name":   table,
            "source_file":  source_file,
            "question":     "",
            "sql":          "",
            "ts":           int(time.time()),
        },
    )


def upsert_kb_file(
    account_id: str,
    fqn: str,                 # DB.SCHEMA.TABLE  (uppercase)
    doc_type: str,            # "kb" | "queries" | "global"
    content: str,
    source_file: str = "",
) -> None:
    """
    Embed and upsert a KB document into Qdrant.

    doc_type="kb"  → section-level chunking: one point per ## section.
                     Point IDs are deterministic per (account, fqn, section)
                     so running a KB build twice is idempotent.
    doc_type="queries" / "global" → single whole-doc point (unchanged).
    """
    if not content.strip():
        return

    if doc_type == "kb":
        points = _build_kb_section_points(account_id, fqn, content, source_file)
        if points:
            _qdrant().upsert(collection_name=_COLLECTION, points=points)
            log.debug(
                "Upserted %d section chunk(s) for %s / %s",
                len(points), account_id, fqn,
            )
    else:
        point = _build_whole_doc_point(account_id, fqn, doc_type, content, source_file)
        _qdrant().upsert(collection_name=_COLLECTION, points=[point])
        log.debug("Upserted whole-doc %s / %s / %s", account_id, fqn, doc_type)

    _invalidate_bm25_cache(account_id)


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

    # Collect all points (sections + whole-docs) then batch-upsert.
    # KB docs ("*_kb.md") are split into per-section points;
    # query-pattern and global docs remain as single whole-doc points.
    all_points: list = []
    files_processed = 0

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
            fqn      = _fqn_from_md_header(content) or raw_name.upper()
            doc_type = "queries"
        elif stem.endswith("_kb"):
            raw_name = stem[:-len("_kb")]
            fqn      = _fqn_from_md_header(content) or raw_name.upper()
            doc_type = "kb"
        else:
            fqn      = _fqn_from_md_header(content) or stem.upper()
            doc_type = "kb"

        if doc_type == "kb":
            # Section-level chunking — may produce up to 7 points per file
            all_points.extend(
                _build_kb_section_points(account_id, fqn, content, md_file.name)
            )
        else:
            # Whole-doc embedding for queries and global docs
            all_points.append(
                _build_whole_doc_point(account_id, fqn, doc_type, content, md_file.name)
            )
        files_processed += 1

    if not all_points:
        return 0

    # Batch upsert in chunks of 100 to avoid timeout on large KBs
    batch = 100
    for i in range(0, len(all_points), batch):
        _qdrant().upsert(collection_name=_COLLECTION, points=all_points[i:i+batch])

    _invalidate_bm25_cache(account_id)
    log.info(
        "Qdrant: upserted %d points (%d KB chunks + whole-docs) from %d files for %s",
        len(all_points), len(all_points), files_processed, account_id,
    )
    return files_processed


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
    Re-embed a single KB file after admin inline edits (semantic layer patch,
    manual KB edit via admin UI).

    For KB overview files (doc_type="kb"):
      Deletes ALL existing section points for this (account_id, fqn, "kb")
      before re-upserting.  This ensures removed or renamed ## sections don't
      leave stale chunks in Qdrant — the point-ID scheme is deterministic per
      section slug, so a section that's renamed would otherwise persist.

    For query-pattern and global files:
      Point ID is deterministic; upsert replaces the old point in-place.
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
        raw      = stem[:-len("_queries")]
        fqn      = _fqn_from_md_header(content) or raw.upper()
        doc_type = "queries"
    else:
        raw      = stem[:-len("_kb")] if stem.endswith("_kb") else stem
        fqn      = _fqn_from_md_header(content) or raw.upper()
        doc_type = "kb"

    if doc_type == "kb":
        # Delete stale section chunks before re-inserting so that renamed or
        # removed sections don't survive as ghost points in the index.
        _delete_points_for_fqn_doctype(account_id, fqn, "kb")

    upsert_kb_file(account_id, fqn, doc_type, content, source_file=filename)
    # upsert_kb_file already invalidates the BM25 cache; the explicit call
    # below is kept for clarity (re_embed_single_file = full re-index).
    _invalidate_bm25_cache(account_id)
    log.info("Re-embedded %s for %s (doc_type=%s)", filename, account_id, doc_type)


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

    def _hybrid_search(
        self,
        query: str,
        n: int,
        allowed_fqns: list[str] | None,
        doc_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Full hybrid pipeline:
          1. Dense HNSW search  — top n×4 candidates
          2. BM25 keyword search — top n×4 candidates (skipped if rank_bm25 absent)
          3. RRF fusion          — deduplicate + merge (skipped without BM25)
          4. Cross-encoder re-rank on top _RERANK_POOL candidates
          5. Return top-n

        Each step gracefully degrades to the previous step's output on failure.
        """
        pool = max(n * 4, _RERANK_POOL)

        # Step 1 — dense HNSW
        dense_hits = self._search(query, pool, allowed_fqns, doc_types)

        # Step 2 — BM25
        bm25_result = _get_bm25(self._account_id, allowed_fqns, doc_types)
        if bm25_result is None:
            # rank_bm25 not available or corpus empty — dense-only path
            candidates = dense_hits
        else:
            bm25_obj, bm25_docs = bm25_result
            bm25_hits = _bm25_search(bm25_obj, bm25_docs, query, pool, doc_types)

            # Step 3 — RRF fusion
            candidates = _rrf_fuse(dense_hits, bm25_hits)

        # Step 4 — cross-encoder re-rank on top _RERANK_POOL candidates
        reranked = _rerank(query, candidates[:_RERANK_POOL], top_n=n)

        # Fill up to n if re-ranker returned fewer (e.g. model unavailable)
        if len(reranked) < n:
            seen_ids = {id(d) for d in reranked}
            for d in candidates[_RERANK_POOL:]:
                if id(d) not in seen_ids:
                    reranked.append(d)
                if len(reranked) >= n:
                    break

        return reranked[:n]

    # ── Public API (mirrors KBRetriever) ─────────────────────────────────────

    def retrieve(
        self,
        question: str,
        n: int = 6,
        allowed_tables: set[str] | None = None,
    ) -> list[str]:
        """
        Return the top-n most relevant KB documents for the question.

        With section chunking, the retriever fetches n×4 section-level candidates
        from the hybrid pipeline, then _group_chunks_by_table() merges sections
        from the same table into coherent per-table docs before returning.
        The result is a list of full KB doc strings (one per table + global docs).

        allowed_tables: set of fully-qualified table names the user may access.
            None → admin / unrestricted — all tables returned.
            Empty set → return only global docs (zero table access).
        """
        allowed_fqns = (
            [t.upper() for t in allowed_tables]
            if allowed_tables is not None else None
        )
        # Retrieve a larger pool so grouping has enough sections to work with
        # (n*4 candidates → typically covers n different tables after merging)
        hits = self._hybrid_search(
            question, n, allowed_fqns, doc_types=["kb", "queries", "global"]
        )
        # Merge section chunks from the same table, preserve global docs
        grouped = _group_chunks_by_table(hits)
        return [doc for doc in grouped if doc.strip()]

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
        hits = self._hybrid_search(
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
# Direct FQN fetch — used by the table coverage guarantee
# ══════════════════════════════════════════════════════════════════════════════

def fetch_docs_for_fqn(account_id: str, fqn: str) -> str | None:
    """
    Fetch and reconstruct the full KB document for a specific table FQN.

    Used by core.table_coverage.guarantee_table_coverage() to gap-fill tables
    that dense + BM25 retrieval missed.  This is a deterministic filter-only
    fetch — never a semantic search — so it always targets the right table.

    Chunked KB docs (doc_type="kb", section_type != "full"):
      All section chunks for the FQN are fetched, sorted in canonical order,
      and reassembled into one coherent markdown string.

    Whole-doc / legacy KB docs (section_type="full"):
      Returned as-is.

    FQN variant matching:
      Tries DB.SCHEMA.TABLE, SCHEMA.TABLE, and bare TABLE so a mismatch in
      qualification level never causes a miss.

    Returns None when no document exists for this FQN.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    fqn_upper = fqn.upper().strip()
    parts     = fqn_upper.split(".")
    fqn_variants: list[str] = [fqn_upper]
    if len(parts) >= 2:
        fqn_variants.append(parts[-1])                    # bare table name
        fqn_variants.append(f"{parts[-2]}.{parts[-1]}")   # SCHEMA.TABLE
    fqn_variants = list(dict.fromkeys(fqn_variants))      # deduplicate

    must = [
        FieldCondition(key="account_id", match=MatchValue(value=account_id)),
        FieldCondition(key="fqn",        match=MatchAny(any=fqn_variants)),
        FieldCondition(key="doc_type",   match=MatchAny(any=["kb", "queries"])),
    ]

    try:
        # Fetch up to 10 — handles up to 7 KB sections + 1 queries doc + buffer
        results, _ = _qdrant().scroll(
            collection_name=_COLLECTION,
            scroll_filter=Filter(must=must),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        log.warning("fetch_docs_for_fqn: Qdrant scroll failed for %s — %s", fqn, exc)
        return None

    if not results:
        return None

    # Separate KB section chunks from whole-doc / queries payloads
    kb_section_payloads = [
        r.payload for r in results
        if r.payload
        and r.payload.get("doc_type") == "kb"
        and r.payload.get("section_type", "full") != "full"
    ]
    kb_whole_payloads = [
        r.payload for r in results
        if r.payload
        and r.payload.get("doc_type") == "kb"
        and r.payload.get("section_type", "full") == "full"
    ]
    other_payloads = [
        r.payload for r in results
        if r.payload and r.payload.get("doc_type") != "kb"
    ]

    # Priority 1 — chunked KB (new format): reconstruct from sections
    if kb_section_payloads:
        return _reconstruct_full_doc(fqn_upper, kb_section_payloads)

    # Priority 2 — whole-doc KB (legacy format or no ## sections found)
    for p in kb_whole_payloads:
        content = p.get("content") or ""
        if content.strip():
            return content

    # Priority 3 — fallback to queries doc (SQL patterns still useful)
    for p in other_payloads:
        content = p.get("content") or ""
        if content.strip():
            return content

    return None


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
