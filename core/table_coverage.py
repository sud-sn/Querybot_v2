"""
core/table_coverage.py

Table Coverage Guarantee — ensures the LLM prompt contains at least one
KB document for every table the entity graph resolver determined is needed
for the current query.

─────────────────────────────────────────────────────────────────────────────
Problem
─────────────────────────────────────────────────────────────────────────────
Dense + BM25 hybrid retrieval ranks documents by similarity to the question.
For multi-table queries the retriever often returns several docs for the
most-prominent table (the one most mentioned in the question) but zero docs
for secondary JOIN tables — even though the graph resolver's JOIN skeleton
requires them.  The LLM then sees no column definitions for those tables,
hallucinate column names, and emits CANNOT_GENERATE.

Example:
  Question: "Show me the top 10 patients by total dispensed quantity"
  Graph detected: ["Patient", "Prescription", "Drug"]
  RAG returned:   docs for FACT_RXFILL (prescription) only
  Gap:            DIM_PATIENT and DIM_DRUG have no KB doc in context
  Without fix:    LLM guesses patient_id, drug_name → CANNOT_GENERATE
  With fix:       DIM_PATIENT_kb.md + DIM_DRUG_kb.md injected → correct SQL

─────────────────────────────────────────────────────────────────────────────
Solution
─────────────────────────────────────────────────────────────────────────────
After RAG retrieval, compare:
  required_fqns — tables the entity graph identified as needed (from detected
                  entities resolved to their table_name / schema_name / database)
  covered_fqns  — tables whose KB doc is already in the retrieved context
                  (parsed from the "# DB.SCHEMA.TABLE" heading of each doc)

For each gap, issue a direct Qdrant filter fetch (not a semantic search) to
pull that table's KB overview document and return it for prompt injection.

─────────────────────────────────────────────────────────────────────────────
Design rules
─────────────────────────────────────────────────────────────────────────────
  • Maximum MAX_GAP_FILL (default 3) gap-fill docs per query — never bloats
    the prompt; the cap prioritises the most critical missing tables.
  • Deterministic fetch, not semantic search — the right table, every time.
  • ACL respected: FQNs outside rag_filter are silently skipped.
  • Graceful degradation: every fetch is wrapped in try/except; a gap-fill
    failure never blocks SQL generation.
  • No new Python packages required — uses qdrant_client (already in requirements).

─────────────────────────────────────────────────────────────────────────────
Usage (main.py, after graph resolver runs)
─────────────────────────────────────────────────────────────────────────────
    from core.table_coverage import build_required_fqns, guarantee_table_coverage

    _required_fqns = build_required_fqns(_graph_ctx, _full_graph)
    _gap_docs = guarantee_table_coverage(
        account_id   = account_id,
        required_fqns= _required_fqns,
        retrieved_docs= relevant_kbs,          # docs already in context
        rag_filter   = rag_filter,             # ACL scope (None = admin)
        max_fill     = 3,
    )
    if _gap_docs:
        context_with_terms += "\\n\\n---\\n\\n" + "\\n\\n---\\n\\n".join(_gap_docs)
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("querybot.table_coverage")

# Hard cap on injected gap-fill docs per query
MAX_GAP_FILL = 3

# Regex for a dotted identifier that starts a KB doc heading line.
# Matches 2- or 3-part FQNs: SCHEMA.TABLE  or  DB.SCHEMA.TABLE
_FQN_TOKEN_RE = re.compile(
    r"^([A-Z0-9_]+(?:\.[A-Z0-9_]+){1,2})(?:\s|$)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# FQN helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fqn_from_doc(text: str) -> str | None:
    """
    Extract the fully-qualified table name from the first heading of a KB
    markdown document.  KB files use  # DB.SCHEMA.TABLE  as their first line.

    Returns the FQN in UPPER CASE, or None if no dotted identifier is found.
    """
    for line in (text or "").splitlines()[:8]:
        stripped = line.strip().lstrip("#").strip()
        if not stripped or "." not in stripped:
            continue
        first_token = stripped.split()[0]
        if "." in first_token:
            m = _FQN_TOKEN_RE.match(first_token)
            if m:
                return m.group(1).upper()
    return None


def _fqn_variants(fqn: str) -> frozenset[str]:
    """
    Return all sub-name variants of an FQN for cross-level matching.

    Handles the fact that the entity graph may produce a 2-part FQN
    (SCHEMA.TABLE) while Qdrant may have stored a 3-part FQN (DB.SCHEMA.TABLE),
    or vice versa.  Matching on any variant is sufficient.

    "MYDB.HR.EMPLOYEES" → {"MYDB.HR.EMPLOYEES", "HR.EMPLOYEES", "EMPLOYEES"}
    "HR.EMPLOYEES"       → {"HR.EMPLOYEES", "EMPLOYEES"}
    "EMPLOYEES"          → {"EMPLOYEES"}
    """
    upper = fqn.upper().strip()
    parts = upper.split(".")
    variants: set[str] = {upper}
    if len(parts) >= 2:
        variants.add(parts[-1])                           # bare table name
        variants.add(f"{parts[-2]}.{parts[-1]}")          # SCHEMA.TABLE
    return frozenset(variants)


def _covered_variants(docs: list[str]) -> frozenset[str]:
    """
    Scan a list of KB doc strings and return the union of all FQN variants
    that are already represented.  Global docs (join map, business vocab)
    have no dotted FQN header and contribute nothing — intentionally.
    """
    result: set[str] = set()
    for doc in docs:
        fqn = _fqn_from_doc(doc)
        if fqn:
            result |= _fqn_variants(fqn)
    return frozenset(result)


def _fqn_allowed(fqn: str, rag_filter: set[str] | None) -> bool:
    """
    Return True when the FQN is within the current ACL scope.
    None means admin (unrestricted) — always allowed.
    """
    if rag_filter is None:
        return True
    allowed_upper = {f.upper() for f in rag_filter}
    return bool(_fqn_variants(fqn) & allowed_upper)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def build_required_fqns(graph_ctx: dict, full_graph: dict) -> set[str]:
    """
    Derive the set of required table FQNs from the entity graph resolver output.

    Parameters
    ----------
    graph_ctx  : return value of core.graph_resolver.resolve_for_question()
    full_graph : raw graph dict from store.get_full_graph()

    Returns an empty set when:
      - graph_ctx["enabled"] is False (no graph configured / no entities detected)
      - No entity has a table_name configured in the admin panel

    Expansion rule — FACT table injection:
      If the detected entities are ALL dimension-type (no fact entity was
      matched by the question), the resolver won't include the fact table
      in `detected` even though any aggregate query (revenue, count, sum)
      needs it.  We therefore expand the required set by one level: for
      every detected entity, also include entities that are directly
      connected to it in the graph relationships where the *other* entity
      has entity_type="fact".  This guarantees the fact table's KB doc
      (with its revenue/metric columns) is always in the prompt context
      for aggregate queries.
    """
    if not graph_ctx.get("enabled"):
        return set()

    entities_map = {
        e["entity_name"]: e
        for e in (full_graph.get("entities") or [])
    }
    relationships = full_graph.get("relationships") or []

    def _fqn_for(ent: dict) -> str | None:
        tbl = (ent.get("table_name") or "").strip().upper()
        sch = (ent.get("schema_name") or "").strip().upper()
        db  = (ent.get("database") or "").strip().upper()
        if not tbl:
            return None
        if db and sch:
            return f"{db}.{sch}.{tbl}"
        if sch:
            return f"{sch}.{tbl}"
        return tbl

    detected = set(graph_ctx.get("detected", []))
    required: set[str] = set()

    for ent_name in detected:
        ent = entities_map.get(ent_name, {})
        fqn = _fqn_for(ent)
        if not fqn:
            log.debug(
                "build_required_fqns: entity %r has no table_name configured — skipped",
                ent_name,
            )
            continue
        required.add(fqn)

    # ── Expansion: add directly-connected FACT tables if none detected ────────
    # Queries like "revenue by prescriber" hit DIM entities only; the fact
    # table that holds the revenue column is connected via JOIN relationships
    # but never shows up in `detected`.  Include it so the LLM sees its columns.
    detected_types = {
        entities_map.get(n, {}).get("entity_type", "dimension").lower()
        for n in detected
    }
    all_facts_detected = all(t == "fact" for t in detected_types) if detected_types else True
    if not all_facts_detected:          # at least one DIM detected, no fact
        fact_names_added: set[str] = set()
        for rel in relationships:
            from_e = rel.get("from_entity", "")
            to_e   = rel.get("to_entity", "")
            # A relationship where one side is detected and the other is a fact
            for candidate in (from_e, to_e):
                partner = to_e if candidate == from_e else from_e
                if partner in detected:
                    candidate_ent = entities_map.get(candidate, {})
                    if candidate_ent.get("entity_type", "").lower() == "fact":
                        fqn = _fqn_for(candidate_ent)
                        if fqn and candidate not in fact_names_added:
                            required.add(fqn)
                            fact_names_added.add(candidate)
                            log.debug(
                                "build_required_fqns: expanding to fact entity %r "
                                "(%s) — connected to detected %r",
                                candidate, fqn, partner,
                            )
        if fact_names_added:
            log.info(
                "build_required_fqns: injected %d fact table(s) via relationship "
                "expansion: %s",
                len(fact_names_added), sorted(fact_names_added),
            )

    return required


def guarantee_table_coverage(
    account_id: str,
    required_fqns: set[str],
    retrieved_docs: list[str],
    rag_filter: set[str] | None,
    max_fill: int = MAX_GAP_FILL,
) -> list[str]:
    """
    Check which required tables are missing from retrieved_docs and return
    KB doc strings for each gap.

    Parameters
    ----------
    account_id    : tenant key
    required_fqns : tables the entity graph determined are needed for this query
    retrieved_docs: KB docs already in the prompt context (pinned + table_kbs)
    rag_filter    : ACL-allowed FQN set (None = admin / unrestricted)
    max_fill      : hard cap on injected gap-fill docs (default 3)

    Returns
    -------
    List of KB doc content strings (may be empty).  Each element is one full
    KB markdown document ready to concatenate into the prompt with "---" separators.
    """
    if not required_fqns:
        return []

    covered = _covered_variants(retrieved_docs)

    # ── Find gaps ─────────────────────────────────────────────────────────────
    missing: list[str] = []
    for fqn in sorted(required_fqns):                 # sorted for stable log output
        if _fqn_variants(fqn) & covered:
            log.debug("Table coverage: %s already covered — OK", fqn)
        elif not _fqn_allowed(fqn, rag_filter):
            log.debug("Table coverage: %s required but outside ACL — skipped", fqn)
        else:
            missing.append(fqn)

    if not missing:
        log.debug(
            "Table coverage OK — all %d required table(s) present in context",
            len(required_fqns),
        )
        return []

    log.info(
        "Table coverage gap — required=%s  covered_sample=%s  missing=%s",
        sorted(required_fqns),
        sorted(list(covered)[:8]),
        missing,
    )

    # ── Fetch gap-fill docs from Qdrant ───────────────────────────────────────
    from core.vector_store import fetch_docs_for_fqn

    gap_docs: list[str] = []
    for fqn in missing[:max_fill]:
        try:
            content = fetch_docs_for_fqn(account_id, fqn)
            if content:
                gap_docs.append(content)
                log.info(
                    "Table coverage: gap-filled %-44s  (%d chars)",
                    fqn, len(content),
                )
            else:
                log.warning(
                    "Table coverage: no KB doc found in Qdrant for %s — "
                    "run a KB rebuild to index this table",
                    fqn,
                )
        except Exception as exc:
            log.warning("Table coverage: fetch failed for %s — %s", fqn, exc)

    return gap_docs
