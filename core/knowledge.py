"""
core/knowledge.py

v8 — Two-stage KB generation with DataPilot-style format.

Stage 1 — Per-table KB:
  DataPilot format: Overview / Key Metrics / Always Exclude /
  Columns / Common Query Patterns / Join Keys / Business Synonyms.
  NEEDS CONTEXT flags for ambiguous columns.
  Distinct values from schema discovery grounded into every section.
  max_tokens=4096 so all 7 sections always fit.

Stage 2 — Query translation document:
  Second LLM call per table using the Stage 1 KB as input.
  Generates 10+ natural-language-question → SQL pairs from the
  actual column names and distinct values in the KB.
  Time-relative queries use MAX(date_col) not system clock.

Business vocab KB:
  Covers all tables. Column names explicitly grounded.

ChromaDB:
  Collection name: kb_store (≥3 chars — ChromaDB requirement).
  Embedding runs in ThreadPoolExecutor — event loop never blocked.
  re_embed_file uses safe upsert pattern.
"""

import asyncio
import logging
import re
from pathlib import Path

log = logging.getLogger("querybot.knowledge")

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_COLLECTION_NAME = "kb_store"


def _inject_deterministic_synonyms(kb_text: str, table_cols: list[str]) -> str:
    """
    Post-process a Stage-1 KB document to guarantee every ERP column that has
    a known entry in ERP_COLUMN_DICT appears in the ## Business Synonyms section.

    The LLM may paraphrase, reorder, or omit synonyms.  This function patches
    the generated markdown so downstream semantic matching always has the
    canonical synonym rows regardless of LLM output quality.

    Only adds rows that are genuinely missing — never duplicates existing ones.
    """
    from core.erp_column_dict import ERP_COLUMN_DICT

    lines = kb_text.splitlines()

    # ── Step 1: find which columns already have synonym rows ─────────────────
    in_synonyms = False
    covered_cols: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_synonyms:
                break
            in_synonyms = stripped.lower().startswith("## business synonyms")
            continue
        if in_synonyms and stripped.startswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                col = cells[1].strip("` ").upper()
                if col:
                    covered_cols.add(col)

    # ── Step 2: build missing rows ────────────────────────────────────────────
    missing_rows: list[str] = []
    for col in table_cols:
        col_upper = col.upper()
        if col_upper in covered_cols:
            continue
        entry = ERP_COLUMN_DICT.get(col_upper)
        if not entry:
            continue
        label, synonyms = entry
        for syn in synonyms[:4]:            # top 4 synonyms per column
            missing_rows.append(f"| {syn} | `{col}` | ERP code — {label} |")

    if not missing_rows:
        return kb_text                      # nothing to add

    # ── Step 3: insert rows just before the next ## section (or at EOF) ───────
    result: list[str] = []
    in_synonyms = False
    inserted = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_synonyms and not inserted:
                result.extend(missing_rows)
                inserted = True
            in_synonyms = stripped.lower().startswith("## business synonyms")
        result.append(line)

    if not inserted:                        # Business Synonyms was the last section
        result.extend(missing_rows)

    log.debug(
        "Injected %d deterministic synonym rows (%d columns)",
        len(missing_rows),
        len({r.split("|")[2].strip().strip("`").upper() for r in missing_rows}),
    )
    return "\n".join(result)


def _parse_schema_descriptions(business_desc: str) -> tuple[str, dict[str, str]]:
    """
    Parse a business description that may contain per-schema sections.

    Format (produced by the multi-schema UI):
        Overall context sentence.

        [PROFITABILITY]: Contains cost and order profitability data.
        [ORDERS]: Contains customer order and billing data.

    Returns:
        (overall_desc, {schema_name_upper: description, ...})

    If no [SCHEMA]: blocks are found, returns (full_text, {}) so single-schema
    descriptions work unchanged.
    """
    overall_parts: list[str] = []
    schema_map: dict[str, str] = {}

    for line in (business_desc or "").splitlines():
        m = re.match(r"^\[([A-Za-z0-9_$#]+)\]:\s*(.+)$", line.strip())
        if m:
            schema_map[m.group(1).upper()] = m.group(2).strip()
        else:
            overall_parts.append(line)

    overall = "\n".join(overall_parts).strip()
    return overall, schema_map


def _build_table_business_desc(business_desc: str, schema_name: str) -> str:
    """
    Build the business context string to inject into a per-table KB prompt.
    If per-schema descriptions are present, combines overall + schema-specific.
    """
    overall, schema_map = _parse_schema_descriptions(business_desc)
    schema_key = (schema_name or "").upper()
    schema_specific = schema_map.get(schema_key, "")

    if schema_specific:
        parts = []
        if overall:
            parts.append(overall)
        parts.append(f"This table belongs to the [{schema_key}] schema: {schema_specific}")
        return "\n".join(parts)
    # No per-schema blocks — return as-is (single-schema or plain description)
    return business_desc or ""


# ══════════════════════════════════════════════════════════════════════════════
# KB generation — public entry point
# ══════════════════════════════════════════════════════════════════════════════

def _format_approved_metrics_for_kb(account_id: str, table_name: str) -> str:
    """
    Return a formatted block of approved metric formulas from the metric registry
    that reference this table. Injected into the KB generation prompt so the LLM
    writes the CORRECT formula into ## Key Metrics — preventing the KB from
    documenting a nearby column that contradicts the approved formula.

    Returns empty string when no metrics match or metric_registry is unavailable.
    """
    try:
        from store.config_store import list_metrics
        metrics = list_metrics(account_id)
    except Exception:
        return ""
    if not metrics:
        return ""

    table_upper = table_name.upper()
    relevant: list[str] = []
    for m in metrics:
        formula_type = (m.get("formula_type") or "query").lower()
        if formula_type != "expression":
            continue
        sql = (m.get("sql_template") or "").strip()
        if not sql:
            continue
        # Match if the metric's base_table or required_columns reference this table
        base = (m.get("base_table") or "").upper()
        req  = (m.get("required_columns") or "").upper()
        # Include if table is explicitly set, or no table restriction (apply to all)
        if base and table_upper not in base and table_upper.split(".")[-1] not in base:
            continue
        name     = m.get("name", "metric")
        synonyms = m.get("synonyms", "")
        relevant.append(
            f"- **{name}** (synonyms: {synonyms})\n"
            f"  APPROVED FORMULA: `{sql}`\n"
            f"  CRITICAL: Use EXACTLY this formula in ## Key Metrics and ## Common Query Patterns.\n"
            f"  Do NOT document a nearby column (e.g. CUS_IVC_LIN_AMT) as the {name} measure.\n"
            f"  Required columns that MUST appear in the SELECT: "
            f"{m.get('required_columns', 'see formula above')}"
        )

    if not relevant:
        return ""

    lines = [
        "ADMIN-APPROVED METRIC FORMULAS FOR THIS TABLE:",
        "The following metrics have been pre-approved by the administrator.",
        "You MUST use these exact formulas in the ## Key Metrics section.",
        "If the schema contains a similar but different column, still use the approved formula — not the similar column.",
        "",
    ]
    lines.extend(relevant)
    return "\n".join(lines)


async def build_kb(
    schema_dir: str,
    kb_dir: str,
    chroma_dir: str,
    business_desc: str,
    provider: str,
    model: str,
    api_key: str,
    extra_kwargs: dict | None = None,
    progress_callback=None,
    stop_event=None,
    account_id: str = "",
    db_type: str = "azure_sql",
) -> int:
    """
    Two-stage KB generation for every table discovered in schema_dir.

    Stage 1: DataPilot-style KB document per table (grounded to schema).
    Stage 2: Question-SQL translation document from Stage 1 output.
    Business vocab: Cross-table synonym and metric mapping.

    Returns number of tables processed.
    Embedding runs in ThreadPoolExecutor so the admin UI stays responsive.
    """
    from core.llm import (
        llm_complete, build_kb_system_prompt,
        build_kb_query_prompt, build_biz_vocab_prompt,
    )
    from core.llm_audit import llm_audit_component
    from core.erp_column_dict import get_erp_hints
    from core.naming_convention import get_naming_hints, build_naming_convention_doc
    from core.schema_enrichment import (
        format_column_reference_for_vocab,
        format_schema_intelligence,
    )
    from core.semantic_model import write_semantic_model

    import hashlib as _hashlib
    import json as _json

    schema_path = Path(schema_dir)
    kb_path     = Path(kb_dir)
    kb_path.mkdir(parents=True, exist_ok=True)

    # Load persisted schema hashes for partial-rebuild skipping
    _hash_file = kb_path / "_kb_hashes.json"
    _schema_hashes: dict[str, str] = {}
    if _hash_file.exists():
        try:
            _schema_hashes = _json.loads(_hash_file.read_text(encoding="utf-8"))
        except Exception:
            _schema_hashes = {}

    # Write the naming convention reference as a global KB doc so it is
    # retrieved at query time for any question — structural grammar rules
    # apply across all tables.
    naming_conv_doc = build_naming_convention_doc()
    (kb_path / "_naming_convention.md").write_text(naming_conv_doc, encoding="utf-8")
    log.info("Naming convention KB doc written (%d chars)", len(naming_conv_doc))

    kw       = extra_kwargs or {}
    md_files = sorted(f for f in schema_path.glob("*.md")
                      if not f.name.startswith("_"))

    if not md_files:
        raise RuntimeError(
            f"No schema MD files found in {schema_dir}. "
            "Run schema discovery first."
        )

    try:
        semantic_model = write_semantic_model(
            schema_dir=schema_dir,
            kb_dir=kb_dir,
            business_desc=business_desc,
            account_id=account_id or chroma_dir,
        )
        log.info(
            "Structured semantic model written (%d tables, %d relationships)",
            len(semantic_model.get("tables") or []),
            len(semantic_model.get("relationships") or []),
        )
    except Exception as exc:
        log.warning("Structured semantic model generation failed: %s", exc)

    # Base system prompt used for business vocab (no per-table ERP hints needed there)
    system_base = build_kb_system_prompt()

    async def _progress(**payload):
        if not progress_callback:
            return
        result = progress_callback(payload)
        if asyncio.iscoroutine(result):
            await result

    # ── Load the auto-discovered join map once, so every per-table KB prompt
    # can see the real FK edges for its neighbours. Without this the LLM
    # invents join keys; with it, the KB "Join Keys" section is grounded in
    # column names that actually exist in the database.
    join_map_path = Path(schema_dir) / "_join_map.md"
    join_map_text = ""
    if join_map_path.exists():
        join_map_text = join_map_path.read_text(encoding="utf-8")

    def _slice_join_map(table: str) -> str:
        """Return only the join-map lines that reference this specific table."""
        if not join_map_text:
            return ""
        upper = table.upper()
        keep: list[str] = []
        for line in join_map_text.splitlines():
            if upper in line.upper():
                keep.append(line)
        return "\n".join(keep).strip()

    # ── Build column reference for business vocab + per-table grounding ───────
    table_names       = [f.stem for f in md_files]
    column_ref_lines  = []

    for md_file in md_files:
        schema_md = md_file.read_text(encoding="utf-8")
        cols = []
        for line in schema_md.splitlines():
            stripped = line.strip()
            if stripped.startswith("| `"):
                m = re.search(r"\| `([A-Za-z0-9_]+)`", stripped)
                if m:
                    cols.append(m.group(1))
        if cols:
            column_ref_lines.append(
                format_column_reference_for_vocab(
                    md_file.stem,
                    cols,
                    schema_md=schema_md,
                )
            )

    column_reference = "\n".join(column_ref_lines)

    # ── Business vocabulary KB ────────────────────────────────────────────────
    biz_user = build_biz_vocab_prompt(table_names, column_reference, business_desc)
    with llm_audit_component("kb_business_vocab"):
        biz_text, _, _ = await llm_complete(
            system_base, biz_user, provider, model, api_key, max_tokens=3000, **kw
        )
    (kb_path / "_business_kb.md").write_text(biz_text, encoding="utf-8")
    log.info("Business vocab KB written (%d tables grounded)", len(table_names))
    await _progress(
        phase="building",
        step="Business vocabulary ready",
        current=0,
        total=len(md_files),
        percent=0,
        current_table="",
    )

    # ── Per-table KB — two stages ─────────────────────────────────────────────
    processed = 0
    for md_file in md_files:
        # Check for stop signal before each table
        if stop_event is not None and stop_event.is_set():
            log.info("KB build stopped by user after %d tables", processed)
            break
        schema_md = md_file.read_text(encoding="utf-8")
        table_name = md_file.stem

        # ── Hash-based partial rebuild: skip if schema unchanged ─────────────
        _schema_hash = _hashlib.sha256(schema_md.encode()).hexdigest()
        _kb_file  = kb_path / f"{table_name}_kb.md"
        _qry_file = kb_path / f"{table_name}_queries.md"
        if (
            _schema_hashes.get(table_name) == _schema_hash
            and _kb_file.exists()
            and _qry_file.exists()
        ):
            log.info("KB partial rebuild: skipping %s — schema unchanged", table_name)
            processed += 1
            await _progress(
                phase="building",
                step=f"Skipped {table_name} (unchanged)",
                current=processed,
                total=len(md_files),
                percent=round(processed / len(md_files) * 100),
                current_table=table_name,
            )
            continue

        # Extract the SQL table name from the schema file header.
        # _az_md writes: **SQL table name:** `[SCHEMA].[TABLE]`
        # This is the exact 2-part name the LLM should use in generated SQL.
        # Fall back to the file stem if the anchor line isn't present (legacy files).
        sql_table_name = table_name  # default: filename stem
        for line in schema_md.splitlines()[:10]:
            if "SQL table name:" in line:
                # Extract the backtick-quoted name: `[SCHEMA].[TABLE]`
                m = re.search(r"`([^`]+)`", line)
                if m:
                    sql_table_name = m.group(1)
                break

        # Extract column list from this specific table for grounding
        table_cols = []
        for line in schema_md.splitlines():
            stripped = line.strip()
            if stripped.startswith("| `"):
                m = re.search(r"\| `([A-Za-z0-9_]+)`", stripped)
                if m:
                    table_cols.append(m.group(1))
        col_list = ", ".join(table_cols)

        # Build a table-specific system prompt that includes ERP short-code hints
        # when the table contains cryptic M3/JDE column names.
        erp_hints = get_erp_hints(table_cols)
        # Naming convention hints cover structural patterns (_DMS_KEY, _AMT, _PCT,
        # AZ_ audit prefixes, table type from suffix) — complementary to ERP hints.
        naming_hints = get_naming_hints(table_cols, table_name)
        system = build_kb_system_prompt(erp_hints=erp_hints, naming_hints=naming_hints)
        if erp_hints:
            log.info("ERP hints injected for %s (%d matched codes)", table_name, erp_hints.count("\n") + 1)
        if naming_hints:
            log.info("Naming convention hints injected for %s", table_name)
        schema_intelligence = format_schema_intelligence(table_name, table_cols, schema_md=schema_md)

        # ── Stage 1: DataPilot-style KB document ──────────────────────────────
        join_slice = _slice_join_map(table_name)
        join_block = (
            f"\nAuto-discovered FK relationships touching this table "
            f"(use these EXACT column names in the Join Keys section — "
            f"do not invent new ones):\n{join_slice}\n"
            if join_slice else ""
        )

        # Extract schema name from sql_table_name (e.g. [profitability].[TABLE] → profitability)
        _schema_part = ""
        _sql_parts = sql_table_name.strip("[]").replace("[", "").replace("]", "").split(".")
        if len(_sql_parts) >= 2:
            _schema_part = _sql_parts[-2]
        table_biz_desc = _build_table_business_desc(business_desc, _schema_part)

        # Inject approved metric formulas from the metric registry so the KB
        # documents the EXACT approved formula — not a nearby similar column.
        approved_metrics_block = _format_approved_metrics_for_kb(
            account_id or chroma_dir, table_name
        )

        stage1_user = (
            f"Business description:\n{table_biz_desc}\n\n"
            f"Table schema for: {table_name}\n"
            f"Exact column names in this table (ONLY use these): {col_list}\n\n"
            f"Full schema with distinct values:\n{schema_md}\n"
            f"\n{schema_intelligence}\n"
            + (f"\n{approved_metrics_block}\n" if approved_metrics_block else "")
            + f"{join_block}\n"
            "Generate the complete Knowledge Base document for this table "
            "following all 7 sections in the format specified. "
            "Use the distinct values from the schema for every filter example. "
            "Ground the Join Keys section in the auto-discovered FK list above — "
            "never invent join columns. If the list is empty, write "
            "'No foreign-key relationships detected.' in that section. "
            "Use the deterministic schema intelligence block to explain cryptic ERP, "
            "abbreviated, date-key, dimension-key, status/filter, and measure-candidate fields. "
            "If ADMIN-APPROVED METRIC FORMULAS are listed above, use those EXACT formulas "
            "in ## Key Metrics — do not substitute nearby columns. "
            "Do not promote candidate metrics to official metrics unless the evidence is strong "
            "or the business description explicitly supports it. "
            "Mark any column whose business rule is unclear with [NEEDS CONTEXT]."
        )
        with llm_audit_component("kb_table_doc"):
            kb_text, _, _ = await llm_complete(
                system, stage1_user, provider, model, api_key,
                max_tokens=4096, **kw
            )

        # Fix 4: guarantee ERP synonym rows are present regardless of LLM output
        kb_text = _inject_deterministic_synonyms(kb_text, table_cols)

        (kb_path / f"{table_name}_kb.md").write_text(kb_text, encoding="utf-8")
        log.info("Stage 1 KB written for %s (%d cols)", table_name, len(table_cols))

        # ── Stage 2: Question-SQL translation from actual KB content ──────────
        # Fix 3: pass the join-map slice so the LLM generates real cross-table
        # Q&A examples using verified join paths instead of single-table queries.
        query_user = build_kb_query_prompt(
            sql_table_name, kb_text, table_biz_desc,
            related_tables=join_slice,
            db_type=db_type,
        )
        with llm_audit_component("kb_query_examples"):
            query_text, _, _ = await llm_complete(
                system, query_user, provider, model, api_key,
                max_tokens=3000, **kw
            )
        (kb_path / f"{table_name}_queries.md").write_text(query_text, encoding="utf-8")
        log.info("Stage 2 query patterns written for %s", table_name)

        # Record hash after successful generation
        _schema_hashes[table_name] = _schema_hash

        processed += 1
        await _progress(
            phase="building",
            step=f"Completed {table_name}",
            current=processed,
            total=len(md_files),
            percent=round(processed / len(md_files) * 100),
            current_table=table_name,
        )

    # ── Persist schema hashes for next partial-rebuild check ──────────────────
    try:
        _hash_file.write_text(_json.dumps(_schema_hashes, indent=2), encoding="utf-8")
    except Exception as _he:
        log.warning("KB: could not write hash file: %s", _he)

    # ── Copy join map from schema discovery ───────────────────────────────────
    import shutil as _shutil
    join_map_src = Path(schema_dir) / "_join_map.md"
    join_map_dst = kb_path / "_join_map.md"
    if join_map_src.exists():
        _shutil.copy2(join_map_src, join_map_dst)
        log.info("Join map copied to KB dir")

    # ── Embed everything into Qdrant ──────────────────────────────────────────
    # Delete stale points for this client first so tables removed from the
    # selection don't linger in the vector index.
    # Run in ThreadPoolExecutor — CPU-heavy embedding must not block the event loop.
    await _progress(
        phase="embedding",
        step="Storing Knowledge Base in vector index",
        current=len(md_files),
        total=len(md_files),
        percent=100,
        current_table="",
    )
    loop = asyncio.get_event_loop()
    account_id_for_embed = chroma_dir  # caller passes account_id as chroma_dir for compat
    await loop.run_in_executor(
        None, _embed_kb_files_qdrant, kb_path, account_id_for_embed
    )
    log.info("Embedded all KB files into Qdrant for account %s", account_id_for_embed)
    await _progress(
        phase="embedding",
        step="Knowledge Base stored in vector index",
        current=len(md_files),
        total=len(md_files),
        percent=100,
        current_table="",
    )

    # ── Step 2: Validate Stage 2 query patterns against real DB ──────────────
    # Import here to avoid circular imports
    log.info("Stage 2 validation will be triggered by main.py after KB build")

    # ── Build suggestion cache ───────────────────────────────────────────────
    # Parse all *_queries.md files and write suggested_questions.json so the
    # portal chat UI has suggestions available immediately after KB build.
    try:
        await _progress(
            phase="suggestions",
            step="Preparing chat suggestions",
            current=len(md_files),
            total=len(md_files),
            percent=100,
            current_table="",
        )
        from core.suggestions import build_suggestion_cache
        q_count = build_suggestion_cache(kb_dir)
        log.info("Suggestion cache built: %d questions for %s", q_count, kb_dir)
    except Exception as _sug_err:
        log.debug("Suggestion cache build failed (non-critical): %s", _sug_err)

    return processed


# ══════════════════════════════════════════════════════════════════════════════
# Qdrant embedding (replaces ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_kb_files_qdrant(kb_path: Path, account_id: str) -> None:
    """
    Embed all KB markdown files into Qdrant.
    Synchronous — always call via run_in_executor.
    Deletes existing KB points for the account first, then re-upserts,
    so a KB rebuild after removing tables doesn't leave stale vectors.
    """
    from core.vector_store import delete_kb_for_client, upsert_kb_directory
    delete_kb_for_client(account_id)
    count = upsert_kb_directory(account_id, str(kb_path))
    log.info("Qdrant: embedded %d KB docs for account %s", count, account_id)


# ══════════════════════════════════════════════════════════════════════════════
# RAG retrieval
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# RAG retrieval — thin wrappers around core/vector_store.py
# ══════════════════════════════════════════════════════════════════════════════

class KBRetriever:
    """
    Semantic search over the Qdrant KB collection.

    This class preserves the same public interface as the old ChromaDB
    KBRetriever so main.py needs no changes.  Under the hood it delegates
    to QdrantKBRetriever in core/vector_store.py.
    """

    def __init__(self, account_id: str):
        from core.vector_store import QdrantKBRetriever
        self._retriever = QdrantKBRetriever(account_id)

    def _is_global(self, doc: str) -> bool:
        return self._retriever._is_global(doc)

    def retrieve(
        self,
        question: str,
        n: int = 6,
        allowed_tables: set[str] | None = None,
    ) -> list[str]:
        return self._retriever.retrieve(question, n=n, allowed_tables=allowed_tables)

    def retrieve_fact_patterns(
        self,
        question: str,
        n: int = 2,
        allowed_tables: set[str] | None = None,
    ) -> list[str]:
        return self._retriever.retrieve_fact_patterns(
            question, n=n, allowed_tables=allowed_tables
        )


def load_retriever(account_id: str) -> KBRetriever:
    """
    Return a KBRetriever for the given account_id.

    The parameter was previously chroma_dir (a filesystem path).
    It is now account_id — Qdrant is a shared service, no per-client path needed.
    Callers that pass a chroma_dir path like "clients/xyz/chroma" will have
    the account_id extracted from the second path component for backwards compat.
    """
    # Handle legacy callers that still pass a filesystem path
    if "/" in str(account_id) or "\\" in str(account_id):
        parts = Path(account_id).parts
        # "clients/ACCOUNT_ID/chroma" → parts[1]
        if len(parts) >= 2:
            account_id = parts[1]
    return KBRetriever(account_id)


def re_embed_file(kb_dir: str, account_id_or_chroma_dir: str, filename: str) -> None:
    """
    Re-embed a single KB file after admin inline edits.

    The second parameter was previously chroma_dir but is now account_id.
    Legacy filesystem paths are handled the same way as in load_retriever.
    """
    from core.vector_store import re_embed_single_file

    account_id = account_id_or_chroma_dir
    if "/" in str(account_id) or "\\" in str(account_id):
        parts = Path(account_id).parts
        if len(parts) >= 2:
            account_id = parts[1]

    re_embed_single_file(account_id, kb_dir, filename)
    log.info("Re-embedded %s for account %s", filename, account_id)
