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


# ══════════════════════════════════════════════════════════════════════════════
# KB generation — public entry point
# ══════════════════════════════════════════════════════════════════════════════

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

    schema_path = Path(schema_dir)
    kb_path     = Path(kb_dir)
    kb_path.mkdir(parents=True, exist_ok=True)

    kw       = extra_kwargs or {}
    md_files = sorted(f for f in schema_path.glob("*.md")
                      if not f.name.startswith("_"))

    if not md_files:
        raise RuntimeError(
            f"No schema MD files found in {schema_dir}. "
            "Run schema discovery first."
        )

    system = build_kb_system_prompt()

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
            column_ref_lines.append(f"  {md_file.stem}: {', '.join(cols)}")

    column_reference = "\n".join(column_ref_lines)

    # ── Business vocabulary KB ────────────────────────────────────────────────
    biz_user = build_biz_vocab_prompt(table_names, column_reference, business_desc)
    with llm_audit_component("kb_business_vocab"):
        biz_text, _, _ = await llm_complete(
            system, biz_user, provider, model, api_key, max_tokens=3000, **kw
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
        schema_md = md_file.read_text(encoding="utf-8")
        table_name = md_file.stem

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

        # ── Stage 1: DataPilot-style KB document ──────────────────────────────
        join_slice = _slice_join_map(table_name)
        join_block = (
            f"\nAuto-discovered FK relationships touching this table "
            f"(use these EXACT column names in the Join Keys section — "
            f"do not invent new ones):\n{join_slice}\n"
            if join_slice else ""
        )

        stage1_user = (
            f"Business description:\n{business_desc}\n\n"
            f"Table schema for: {table_name}\n"
            f"Exact column names in this table (ONLY use these): {col_list}\n\n"
            f"Full schema with distinct values:\n{schema_md}\n"
            f"{join_block}\n"
            "Generate the complete Knowledge Base document for this table "
            "following all 7 sections in the format specified. "
            "Use the distinct values from the schema for every filter example. "
            "Ground the Join Keys section in the auto-discovered FK list above — "
            "never invent join columns. If the list is empty, write "
            "'No foreign-key relationships detected.' in that section. "
            "Mark any column whose business rule is unclear with [NEEDS CONTEXT]."
        )
        with llm_audit_component("kb_table_doc"):
            kb_text, _, _ = await llm_complete(
                system, stage1_user, provider, model, api_key,
                max_tokens=4096, **kw
            )
        (kb_path / f"{table_name}_kb.md").write_text(kb_text, encoding="utf-8")
        log.info("Stage 1 KB written for %s (%d cols)", table_name, len(table_cols))

        # ── Stage 2: Question-SQL translation from actual KB content ──────────
        # Pass the SQL-formatted table name (e.g. [SCHEMA].[TABLE] for Azure SQL)
        # so the LLM generates examples with the correct table name format.
        query_user = build_kb_query_prompt(sql_table_name, kb_text, business_desc)
        with llm_audit_component("kb_query_examples"):
            query_text, _, _ = await llm_complete(
                system, query_user, provider, model, api_key,
                max_tokens=3000, **kw
            )
        (kb_path / f"{table_name}_queries.md").write_text(query_text, encoding="utf-8")
        log.info("Stage 2 query patterns written for %s", table_name)

        processed += 1
        await _progress(
            phase="building",
            step=f"Completed {table_name}",
            current=processed,
            total=len(md_files),
            percent=round(processed / len(md_files) * 100),
            current_table=table_name,
        )

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
