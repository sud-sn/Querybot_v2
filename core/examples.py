"""
core/examples.py

Validated SQL examples retriever — Step 2 of the TextQL/Vanna approach.

After KB generation, Stage 2 query patterns are executed against the real
database. Pairs that succeed are stored in validated_examples (SQLite) and
also embedded into a separate ChromaDB collection for semantic retrieval.

At query time, the top-3 most similar validated examples are retrieved and
injected into the SQL generation prompt as few-shot examples. This is the
single biggest reliability improvement — the LLM learns from proof, not prose.

Step 4 (query log harvesting) feeds back into the same store — every
successful user query becomes a validated example automatically.
"""

import logging
from pathlib import Path

log = logging.getLogger("querybot.examples")

_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
_EXAMPLES_COLL     = "validated_examples"   # separate collection from kb_store


def _fqn_from_kb_header(content: str) -> str:
    """Extract DB.SCHEMA.TABLE or SCHEMA.TABLE from the first markdown heading."""
    import re
    for line in content.splitlines()[:6]:
        stripped = line.strip().lstrip("#").strip()
        match = re.match(
            r"^([A-Z0-9_]+\.[A-Z0-9_]+(?:\.[A-Z0-9_]+)?)(?:\s|$)",
            stripped.upper(),
        )
        if match:
            return match.group(1)
    return ""


def _table_name_for_query_file(qfile: Path) -> str:
    """Prefer the FQN from the sibling *_kb.md header for validated examples."""
    bare_name = qfile.stem.replace("_queries", "").upper()
    kb_file = qfile.with_name(qfile.name.replace("_queries.md", "_kb.md"))
    if kb_file.exists():
        try:
            fqn = _fqn_from_kb_header(kb_file.read_text(encoding="utf-8", errors="replace"))
            if fqn:
                return fqn
        except Exception:
            pass
    return bare_name


# ══════════════════════════════════════════════════════════════════════════════
# Validate Stage 2 patterns against real DB
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_store_examples(
    account_id: str,
    queries_dir: str,
    credentials: dict,
    db_type: str,
    chroma_dir: str,   # kept for API compat — account_id used instead
) -> int:
    """
    Parse all *_queries.md files from Stage 2, execute each SQL with LIMIT/TOP 1
    against the real database, keep only those that succeed, store them in
    SQLite and embed into Qdrant.

    Uses a SINGLE connection for the entire validation batch — not one per query.
    This prevents the Snowflake connection flood (110 connections for 11 tables).

    Returns number of validated examples stored.
    """
    import store

    queries_path = Path(queries_dir)
    if not queries_path.exists():
        log.warning("Queries dir not found: %s", queries_dir)
        return 0

    # Collect all pairs first — one pass through files
    all_pairs: list[tuple[str, str, str]] = []  # (question, sql, table_name)
    for qfile in sorted(queries_path.glob("*_queries.md")):
        table_name = _table_name_for_query_file(qfile)
        content    = qfile.read_text(encoding="utf-8")
        pairs      = _parse_query_pairs(content)
        for question, sql in pairs:
            all_pairs.append((question, sql, table_name))

    if not all_pairs:
        log.info("No query pairs found to validate")
        return 0

    log.info("Validating %d query patterns against real DB (single connection)...",
             len(all_pairs))

    # Open ONE connection and run all validations through it
    validated: list[tuple[str, str, str]] = []
    conn = _open_connection(credentials, db_type)
    if conn is None:
        log.warning("Could not open DB connection for validation — skipping")
        return 0

    try:
        for question, sql, table_name in all_pairs:
            test_sql = _add_row_cap(sql, db_type, n=1)
            try:
                _execute_on_connection(conn, db_type, test_sql)
                store.save_validated_example(account_id, question, sql, table_name)
                validated.append((question, sql, table_name))
                log.info("Validated: %s", question[:60])
            except Exception as e:
                log.debug("Failed: '%s' — %s", question[:50], str(e)[:80])
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Embed all validated examples into Qdrant
    if validated:
        from core.vector_store import upsert_examples
        upsert_examples(account_id, validated)
        log.info("Stored and embedded %d/%d validated examples for %s",
                 len(validated), len(all_pairs), account_id)

    return len(validated)


def _open_connection(credentials: dict, db_type: str):
    """Open a single reusable DB connection for batch validation.
    
    Uses the same credential key names as core/schema.py — the credentials
    dict comes from db_config.credentials_encrypted which stores keys
    matching the DB_REQUIRED_FIELDS definitions in config_store.py.
    """
    try:
        if db_type == "snowflake":
            from core.schema import _sf_connect
            return _sf_connect(credentials)
        elif db_type == "oracle":
            from core.schema import _ora_connect
            return _ora_connect(credentials)
        elif db_type == "azure_sql":
            from core.schema import _az_connect
            return _az_connect(credentials)
    except Exception as e:
        log.error("Failed to open validation connection: %s", e)
        return None


def _execute_on_connection(conn, db_type: str, sql: str) -> None:
    """Execute SQL on an existing open connection. Raises on failure."""
    if db_type == "snowflake":
        import snowflake.connector
        cur = conn.cursor(snowflake.connector.DictCursor)
    else:
        cur = conn.cursor()
    try:
        cur.execute(sql)
        cur.fetchmany(1)   # consume result to confirm query ran
    finally:
        cur.close()


def _parse_query_pairs(content: str) -> list[tuple[str, str]]:
    """Extract (question, sql) pairs from *_queries.md format."""
    import re
    pairs   = []
    lines   = content.splitlines()
    q, sql  = None, []
    in_sql  = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("Q:"):
            # Save previous pair if complete
            if q and sql:
                pairs.append((q, "\n".join(sql).strip()))
            q     = stripped[2:].strip()
            sql   = []
            in_sql = False
        elif stripped.upper().startswith("SQL:"):
            sql_text = stripped[4:].strip()
            if sql_text:
                sql = [sql_text]
            in_sql = True
        elif in_sql and stripped:
            # Continuation of multi-line SQL
            if stripped.startswith("Q:"):
                if q and sql:
                    pairs.append((q, "\n".join(sql).strip()))
                q     = stripped[2:].strip()
                sql   = []
                in_sql = False
            elif not stripped.startswith("#"):
                sql.append(stripped)

    # Last pair
    if q and sql:
        pairs.append((q, "\n".join(sql).strip()))

    return [(q, s) for q, s in pairs if q and s and len(s) > 10]


def _add_row_cap(sql: str, db_type: str, n: int = 1) -> str:
    """
    Add a minimal row cap to a SQL statement for validation testing.
    We want to test if the SQL is valid, not retrieve real data.
    """
    s = sql.strip().rstrip(";")
    u = s.upper()

    if db_type == "snowflake":
        if "LIMIT" not in u:
            return f"{s} LIMIT {n}"
        return sql

    elif db_type == "oracle":
        if "FETCH FIRST" not in u and "ROWNUM" not in u:
            return f"{s} FETCH FIRST {n} ROWS ONLY"
        return sql

    elif db_type == "azure_sql":
        # For Azure SQL, wrap in a subquery with TOP if SELECT TOP not present
        if "SELECT TOP" not in u and "TOP(" not in u:
            # Replace first SELECT with SELECT TOP N
            import re
            return re.sub(r"(?i)^(\s*SELECT\s+)", f"SELECT TOP {n} ", s, count=1)
        return sql

    return sql


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB embedding for validated examples
# ══════════════════════════════════════════════════════════════════════════════

def _embed_examples(validated: list[tuple[str, str, str]], chroma_dir: str) -> None:
    """Embed validated (question, sql, table) into a separate ChromaDB collection."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef     = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=chroma_dir)

    # Get or create — safe for incremental updates
    try:
        col = client.get_collection(_EXAMPLES_COLL, embedding_function=ef)
    except Exception:
        col = client.create_collection(_EXAMPLES_COLL, embedding_function=ef)

    # Build documents: embed the question (used for retrieval)
    # Store the SQL as metadata so we can retrieve it
    docs, ids, metas = [], [], []
    for i, (question, sql, table_name) in enumerate(validated):
        doc_id = f"ex_{abs(hash(question)) % 10**9}"
        docs.append(question)
        ids.append(doc_id)
        metas.append({"sql": sql, "table": table_name,
                      "question": question})

    if docs:
        # Upsert — safe on re-runs
        try:
            col.upsert(documents=docs, ids=ids, metadatas=metas)
        except Exception as e:
            log.error("Failed to embed examples: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval at query time
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar_examples(
    question: str,
    chroma_dir: str,
    n: int = 3,
    allowed_tables: set[str] | None = None,
) -> list[dict]:
    """
    Return the top-n most semantically similar validated examples to the question.
    Each result: {"question": str, "sql": str, "table": str}
    Returns empty list if no examples exist yet.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    try:
        ef     = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=chroma_dir)
        col    = client.get_collection(_EXAMPLES_COLL, embedding_function=ef)
    except Exception:
        return []  # Collection does not exist yet — no examples

    total = col.count()
    if total == 0:
        return []

    try:
        results = col.query(
            query_texts=[question],
            n_results=min(n * 2, total),
            include=["documents", "metadatas"],
        )
    except Exception as e:
        log.error("Example retrieval failed: %s", e)
        return []

    metas = results.get("metadatas", [[]])[0]
    examples = []
    for meta in metas:
        if allowed_tables is not None:
            tbl = (meta.get("table") or "").upper()
            if tbl and tbl not in allowed_tables:
                continue
        examples.append({
            "question": meta.get("question", ""),
            "sql":      meta.get("sql", ""),
            "table":    meta.get("table", ""),
        })
        if len(examples) >= n:
            break

    return examples


def format_examples_for_prompt(examples: list[dict]) -> str:
    """
    Format validated examples as few-shot context for the SQL generation prompt.
    Returns empty string if no examples.
    """
    if not examples:
        return ""

    lines = [
        "VERIFIED EXAMPLES — These question→SQL pairs have been tested against "
        "this exact database. Use them as a guide for SQL syntax and patterns ONLY.\n"
        "CRITICAL: Examples show SQL structure only. The GROUP BY dimension, SELECT columns, "
        "aliases, and filters MUST be derived from the current question — never copied from "
        "the example. Each question determines its own dimension and output columns.\n"
    ]
    for ex in examples:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL: {ex['sql']}\n")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Query log harvesting
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar_examples(
    question: str,
    account_id_or_chroma_dir: str,
    n: int = 3,
    allowed_tables: set[str] | None = None,
) -> list[dict]:
    """
    Return the top-n most semantically similar validated examples to the question.
    Each result: {"question": str, "sql": str, "table": str}

    Dual-collection retrieval (Day 5-7 -- governed learning loop):
      1. querybot_governed (admin-approved candidates) -- interleaved first
      2. querybot_kb (legacy auto-harvested examples)  -- fill remaining slots

    Governed examples are prioritised because they have been explicitly ratified
    by a human reviewer.  Results are deduplicated by question text (case-insensitive)
    so the same Q/SQL pair is never injected into the prompt twice.

    Graceful degradation:
      - If querybot_governed is empty or unreachable, returns only legacy examples.
      - If querybot_kb is empty or unreachable, returns only governed examples.
      - Both empty -> [].

    The second parameter was previously chroma_dir (a filesystem path).
    It is now account_id. Legacy filesystem paths are handled gracefully.
    """
    from core.vector_store import retrieve_similar_examples as _qs_retrieve

    account_id = account_id_or_chroma_dir
    if "/" in str(account_id) or "\\" in str(account_id):
        parts = Path(account_id).parts
        if len(parts) >= 2:
            account_id = parts[1]

    # 1. Legacy examples (querybot_kb, doc_type="example")
    legacy: list[dict] = []
    try:
        legacy = _qs_retrieve(account_id, question, n=n, allowed_tables=allowed_tables)
    except Exception as exc:
        log.debug("legacy example retrieval skipped (non-fatal): %s", exc)

    # 2. Governed examples (querybot_governed) -- best-effort
    governed: list[dict] = []
    try:
        from core.governed_store import retrieve_governed_examples
        governed = retrieve_governed_examples(
            account_id, question, n=n, allowed_tables=allowed_tables,
        )
    except Exception as exc:
        log.debug("governed example retrieval skipped (non-fatal): %s", exc)

    if not governed:
        return legacy[:n]
    if not legacy:
        return governed[:n]

    # Interleave: governed first (human-approved), then legacy, deduplicate by question
    seen: set[str] = set()
    merged: list[dict] = []
    for ex in governed + legacy:
        q_key = (ex.get("question") or "").strip().lower()
        if q_key and q_key not in seen:
            seen.add(q_key)
            merged.append(ex)
        if len(merged) >= n:
            break

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Query log harvesting
# ══════════════════════════════════════════════════════════════════════════════

def harvest_and_embed(account_id: str, chroma_dir: str = "") -> int:  # chroma_dir unused
    """
    Harvest successful queries from query_log into validated_examples,
    then re-embed into Qdrant. Run nightly or on-demand.
    Returns number of new examples added.
    """
    import store
    from core.vector_store import upsert_examples

    added = store.harvest_successful_queries(account_id, days_back=30)
    if added > 0:
        all_examples = store.get_validated_examples(account_id)
        validated = [
            (ex["question"], ex["sql_query"], ex.get("table_name", ""))
            for ex in all_examples
        ]
        if validated:
            upsert_examples(account_id, validated)
        log.info("Harvested %d new examples from query log for %s", added, account_id)

    return added
