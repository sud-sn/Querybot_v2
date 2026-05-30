"""
core/schema.py

Schema discovery for Snowflake, Oracle, and Azure SQL.

v7 additions:
  - Distinct value scan for categorical columns — LLM sees ALL possible values
    for short-text columns (Status, Type, Department, Gender etc.) not just 5
    sample rows. This fixes CANNOT_GENERATE for business concept queries.
  - Auto join-map generation — writes _join_map.md documenting every FK
    relationship so the LLM always knows how to JOIN across tables.
  - Synthetic PII samples unchanged.
"""

import json
import logging
from pathlib import Path

from core.synthetic import generate_synthetic_sample, should_use_synthetic

log = logging.getLogger("querybot.schema")

# ── Categorical column detection ───────────────────────────────────────────────
# Columns whose name contains any of these words are scanned for distinct values
_CATEGORICAL_NAME_HINTS = {
    "status", "type", "flag", "gender", "sex", "category", "class",
    "department", "dept", "division", "group", "team", "role", "level",
    "grade", "rank", "priority", "severity", "state", "phase", "stage",
    "mode", "method", "reason", "code", "indicator", "yn", "active",
    "enabled", "attrition", "punch", "shift", "position", "title",
    "country", "region", "zone", "area", "branch", "location",
}

# Max distinct values to scan — above this we skip (high-cardinality columns)
_MAX_DISTINCT = 30

# Column data types treated as potentially categorical
_CATEGORICAL_TYPES = {
    "varchar", "varchar2", "nvarchar", "char", "nchar",
    "text", "string", "character varying",
}


def _is_categorical(col_name: str, col_type: str) -> bool:
    """Return True if this column is likely categorical."""
    name_lower  = col_name.lower()
    type_lower  = (col_type or "").lower().split("(")[0].strip()
    name_match  = any(h in name_lower for h in _CATEGORICAL_NAME_HINTS)
    type_match  = any(t in type_lower for t in _CATEGORICAL_TYPES)
    return name_match and type_match


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def _clean_identifier_part(value: str) -> str:
    """Normalize a table/schema/database identifier for selection matching."""
    return str(value or "").strip().strip('"').strip("'").strip("[]").upper()


def _allowed_table_match(
    allowed: set[str] | None,
    table_name: str,
    schema: str | None = None,
    database: str | None = None,
) -> bool:
    """
    Return True when a discovered table is included in the selected table set.

    Supports both legacy bare names ("ORDERS") and schema-aware selections
    ("SALES.ORDERS" or "CRM.SALES.ORDERS"). The admin schema browser stores
    fully-qualified names so tables from non-default schemas are not dropped
    during file generation.
    """
    if allowed is None:
        return True

    normalized = {_clean_identifier_part(x) for x in allowed if str(x).strip()}
    table = _clean_identifier_part(table_name)
    schema_part = _clean_identifier_part(schema) if schema else ""
    db_part = _clean_identifier_part(database) if database else ""

    candidates = {table}
    if schema_part:
        candidates.add(f"{schema_part}.{table}")
    if db_part and schema_part:
        candidates.add(f"{db_part}.{schema_part}.{table}")
    return bool(candidates & normalized)


def _allowed_has_qualified_refs(allowed: set[str] | None) -> bool:
    return bool(allowed and any("." in str(x) for x in allowed))


def _apply_masking(
    fetch_fn,
    col_defs: list[dict],
    mode: str,
    explicit_fields: set[str],
    table_name: str,
    seed_key: str = "",
    allow_unmasked: bool = False,
) -> tuple[list[dict], set[str], dict[str, str], bool]:
    """
    Centralised masking helper shared by all three DB backends.

    Returns (sample_rows, masked_field_names, replacement_map, synthetic_fallback_used).

    replacement_map is {field_name: strategy_name} for each masked field,
    e.g. {"EMAIL": "email", "FIRST_NAME": "first_name"}.

    mode values:
      "none"      — read real rows, no masking
      "all"       — read real rows, mask every field
      "selective" — read real rows, mask only explicit_fields
      "auto"      — read real rows, mask PII-detected fields (default)

    seed_key: when non-empty, masking is deterministic (HMAC-based) so the
    same real value always maps to the same fake across all tables — preserving
    FK consistency. Typically set to account_id by the caller.

    If the DB read fails, falls back to generate_synthetic_sample() and
    returns synthetic_fallback_used=True.
    """
    from core.masking import (
        detect_sensitive_columns, mask_rows, get_strategy_map,
        scan_values_for_pii, scrub_unmasked_free_text,
    )

    # ── Guard: real, unmasked egress must be an explicit, logged opt-in ───────
    # An admin setting mode="none" would ship raw production rows to the LLM.
    # Unless the client has explicitly allowed it, downgrade to "auto" so PII
    # detection still runs. (B4)
    if mode == "none" and not allow_unmasked:
        log.warning(
            "schema: mode='none' requested for %s without allow_unmasked_kb — "
            "downgrading to 'auto' to prevent raw PII egress", table_name,
        )
        mode = "auto"

    # Determine which fields to mask based on mode
    if mode == "none":
        masked_fields: set[str] = set()
    elif mode == "all":
        masked_fields = {c["name"] for c in col_defs}
    elif mode == "selective":
        masked_fields = explicit_fields or set()
    else:  # "auto"
        detected = detect_sensitive_columns(col_defs)
        masked_fields = set(detected.keys())

    # Fetch real rows
    try:
        rows = fetch_fn()
    except Exception:
        rows = []

    if not rows:
        # Fallback: fully synthetic rows (DB unreachable or empty table)
        log.info("schema: synthetic fallback for %s (DB read returned 0 rows)", table_name)
        return generate_synthetic_sample(col_defs, seed=seed_key), set(), {}, True

    # ── Value-based PII detection (auto mode only) ────────────────────────────
    # Scans actual sample values for emails, SSNs, credit cards, etc. that
    # name-based detection would miss (e.g. a column called FIELD_01 containing
    # email addresses).  Only runs in auto mode — other modes either mask
    # everything (all), nothing (none), or exactly what the admin specified
    # (selective), so adding extra columns would be surprising.
    value_strategy_overrides: dict[str, str] = {}
    if mode == "auto":
        value_hits = scan_values_for_pii(rows, col_defs)
        for col_name, hit in value_hits.items():
            if col_name not in masked_fields:
                masked_fields.add(col_name)
                log.info(
                    "schema: value-scan auto-masked %r as %s (%.0f%% confidence)",
                    col_name, hit["pii_type"], hit["confidence"] * 100,
                )
            # Override takes effect even if the column was already name-detected,
            # since the value-scan result is more specific (e.g. a column named
            # CONTACT_INFO that actually contains emails → use "email" strategy
            # instead of the generic "text_mask" fallback).
            value_strategy_overrides[col_name] = hit["strategy"]

    # Apply masking
    replacement_map: dict[str, str] = {}
    if masked_fields:
        rows = mask_rows(
            rows, masked_fields, col_defs,
            seed_key=seed_key,
            strategy_overrides=value_strategy_overrides or None,
        )
        replacement_map = get_strategy_map(masked_fields, col_defs)
        # Patch replacement_map so value-scan-detected strategies are reflected
        # in the egress log and admin UI (not just the generic name-based label).
        for col_name, strategy in value_strategy_overrides.items():
            if col_name in masked_fields:
                replacement_map[col_name] = strategy
        log.info(
            "schema: masked %d field(s) in %s (mode=%s, value_scan=%d)",
            len(masked_fields), table_name, mode, len(value_strategy_overrides),
        )

    # Final safety net: scrub embedded PII from any narrative string column that
    # was NOT explicitly masked above (defense-in-depth). (B2)
    rows = scrub_unmasked_free_text(rows, col_defs, masked_fields)

    return rows, masked_fields, replacement_map, False


def _resolve_masking_fields(
    col_defs: list[dict],
    mode: str,
    explicit_fields: set[str],
) -> set[str]:
    """Return columns that must not expose raw values in samples or distinct lists."""
    from core.masking import detect_sensitive_columns

    if mode == "none":
        return set()
    if mode == "all":
        return {c["name"] for c in col_defs}
    if mode == "selective":
        return explicit_fields or set()
    detected = detect_sensitive_columns(col_defs)
    return set(detected.keys())


def _sf_fetch_sample(cur, schema: str, name: str) -> list[dict]:
    cur.execute(f'SELECT * FROM "{schema}"."{name}" LIMIT 5')
    return [dict(r) for r in cur.fetchall()]


def _safe_table_file_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return stem.strip("._-") or "table"


def discover_and_write(
    credentials: dict,
    db_type: str,
    output_dir: str,
    allowed_tables: set[str] | None = None,
    masking_config: dict | None = None,
    seed_key: str = "",
) -> int:
    """
    Discover schema from the customer DB and write one .md file per table.

    allowed_tables  — optional set of table FQNs (DB.SCHEMA.TABLE, uppercase).
    masking_config  — optional dict keyed by table FQN (uppercase):
                      {
                        "DB.SCHEMA.TABLE": {
                          "mode": "none"|"all"|"selective"|"auto",
                          "masked_fields": ["ColA", "ColB"]   # for "selective"
                        }
                      }
                      Absent FQN → default "auto" (detect PII cols by name pattern).
    seed_key        — when non-empty, masking uses HMAC-based deterministic
                      pseudonyms so FK relationships are preserved across tables.
                      Pass account_id from the calling route.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    allowed  = {t.upper() for t in allowed_tables} if allowed_tables is not None else None
    mc       = {k.upper(): v for k, v in (masking_config or {}).items()}
    if db_type == "snowflake":
        return _discover_snowflake(credentials, out, allowed, mc, seed_key=seed_key)
    elif db_type == "oracle":
        return _discover_oracle(credentials, out, allowed, mc, seed_key=seed_key)
    elif db_type == "azure_sql":
        return _discover_azure_sql(credentials, out, allowed, mc, seed_key=seed_key)
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}")


def test_connection(credentials: dict, db_type: str) -> dict:
    """
    Open a read-only smoke-test connection and run a tiny metadata query.

    Used by the admin "Test connection" button. It intentionally does not
    persist credentials or discover schema files.
    """
    if db_type == "snowflake":
        # Cap login_timeout at 15 s for test path — Snowflake default is 60 s
        smoke_cfg = dict(credentials)
        smoke_cfg["login_timeout"] = min(int(smoke_cfg.get("login_timeout", 15)), 15)
        smoke_cfg["network_timeout"] = min(int(smoke_cfg.get("network_timeout", 15)), 15)
        conn = _sf_connect(smoke_cfg, max_retries=1)
        try:
            cur = conn.cursor()
            cur.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_USER()")
            row = cur.fetchone()
            return {
                "database": str(row[0] or ""),
                "schema": str(row[1] or ""),
                "user": str(row[2] or ""),
            }
        finally:
            conn.close()
    elif db_type == "oracle":
        conn = _ora_connect(credentials, max_retries=1)
        try:
            cur = conn.cursor()
            cur.execute("SELECT SYS_CONTEXT('USERENV','CURRENT_SCHEMA'), USER FROM DUAL")
            row = cur.fetchone()
            return {
                "schema": str(row[0] or ""),
                "user": str(row[1] or ""),
            }
        finally:
            conn.close()
    elif db_type == "azure_sql":
        smoke_cfg = dict(credentials)
        # Cap the ODBC-level connection timeout at 15 s for the test path.
        # The default is 60 s, which causes the route's asyncio.wait_for to
        # fire before the driver has given up — producing a false "timeout"
        # even when credentials are wrong and the error would have come back
        # in under a second with a shorter timeout.
        smoke_cfg["login_timeout"] = min(int(smoke_cfg.get("login_timeout", 15)), 15)
        try:
            conn = _az_connect(smoke_cfg, max_retries=1)
        except Exception as e:
            # Re-raise with a clean, human-readable message so the route can
            # surface it directly instead of the cryptic ODBC error string.
            err_str = str(e)
            if "Login timeout expired" in err_str or "HYT00" in err_str:
                raise TimeoutError(
                    "Connection timed out after 15 s. Check the server name, "
                    "firewall rules (is the VM's IP allowed?), and that the "
                    "database is not paused."
                )
            if "Login failed" in err_str or "28000" in err_str:
                raise PermissionError(
                    f"Login failed — check the username and password. ({e})"
                )
            raise
        try:
            cur = conn.cursor()
            cur.execute("SELECT DB_NAME(), SCHEMA_NAME(), SUSER_SNAME()")
            row = cur.fetchone()
            return {
                "database": str(row[0] or ""),
                "schema": str(row[1] or ""),
                "user": str(row[2] or ""),
            }
        finally:
            conn.close()
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}")


def run_query(credentials: dict, db_type: str, sql: str, max_rows: int = 200) -> list[dict]:
    if db_type == "snowflake":
        return _run_snowflake(credentials, sql, max_rows)
    elif db_type == "oracle":
        return _run_oracle(credentials, sql, max_rows)
    elif db_type == "azure_sql":
        return _run_azure_sql(credentials, sql, max_rows)
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}")


def load_known_tables(schema_dir: str) -> set[str]:
    """
    Load the set of known table identifiers from _schema.json.

    Since schema.py now stores keys as 3-part FQNs (DB.SCHEMA.TABLE),
    but the SQL validator receives bare table names or 2-part names from
    sqlglot's parser (node.name), we expand every FQN key into all its
    name variants so the validator's membership test works regardless of
    how the LLM qualified the table name in the generated SQL.

    Example: MYDB.HR.EMPLOYEES → also adds
      HR.EMPLOYEES  and  EMPLOYEES
    """
    p = Path(schema_dir) / "_schema.json"
    if not p.exists():
        return set()
    result: set[str] = set()
    for key in json.loads(p.read_text(encoding="utf-8")):
        upper = key.upper()
        result.add(upper)                        # full FQN
        parts = upper.split(".")
        if len(parts) >= 2:
            result.add(parts[-1])                # bare table name
            result.add(".".join(parts[-2:]))     # schema.table
    return result


def load_schema_columns(schema_dir: str) -> dict[str, dict[str, str]]:
    """
    Load table -> column metadata from _schema.json.

    Keys are expanded the same way as load_known_tables(): full FQN,
    schema.table, and bare table name. Values map UPPER column name to the
    original column type string. If multiple schemas share a bare table name,
    the bare entry is the union of those columns; qualified keys stay exact.
    """
    p = Path(schema_dir) / "_schema.json"
    if not p.exists():
        return {}

    try:
        master = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: dict[str, dict[str, str]] = {}
    for key, info in master.items():
        upper = str(key).upper()
        parts = upper.split(".")
        variants = [upper]
        if len(parts) >= 2:
            variants.extend([parts[-1], ".".join(parts[-2:])])

        cols: dict[str, str] = {}
        for col in (info or {}).get("columns", []) or []:
            name = (
                col.get("name")
                or col.get("COLUMN_NAME")
                or col.get("column_name")
                or ""
            )
            if not name:
                continue
            cols[str(name).upper()] = str(
                col.get("type")
                or col.get("DATA_TYPE")
                or col.get("data_type")
                or ""
            )

        for variant in variants:
            result.setdefault(variant, {}).update(cols)
    return result


def load_schema_json(schema_dir: str) -> dict:
    p = Path(schema_dir) / "_schema.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def list_schema_files(schema_dir: str) -> list[str]:
    d = Path(schema_dir)
    if not d.exists():
        return []
    return sorted(f.name for f in d.glob("*.md"))


# ══════════════════════════════════════════════════════════════════════════════
# Join map generation
# ══════════════════════════════════════════════════════════════════════════════

def _build_join_map(master: dict) -> str:
    """
    Analyse all discovered tables and auto-detect FK relationships by matching
    column names across tables. Writes a human-readable join map the LLM can
    use to construct multi-table JOINs without guessing.
    """
    lines = [
        "# Cross-Table Join Map",
        "",
        "Use this document to determine how tables relate to each other.",
        "When a question requires data from multiple tables, use these join paths.",
        "",
        "## Detected Relationships",
        "",
    ]

    # Build index: column_name_upper → list of tables that have it
    col_to_tables: dict[str, list[str]] = {}
    for tbl_name, tbl_info in master.items():
        for col in tbl_info.get("columns", []):
            cname = col["name"].upper()
            col_to_tables.setdefault(cname, []).append(tbl_name)

    # Find columns shared across 2+ tables — these are likely join keys
    join_pairs_seen = set()
    relationships = []

    for col_name, tables in col_to_tables.items():
        if len(tables) < 2:
            continue
        # Skip very generic columns that are not real FKs
        if col_name in {"ID", "NAME", "STATUS", "TYPE", "CODE", "DATE",
                        "CREATED_AT", "UPDATED_AT", "DESCRIPTION", "NOTES"}:
            continue

        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                t1, t2 = sorted([tables[i], tables[j]])
                key = f"{t1}|{t2}|{col_name}"
                if key in join_pairs_seen:
                    continue
                join_pairs_seen.add(key)

                # Determine which is likely the fact and which the dimension
                t1_cols = [c["name"] for c in master[t1].get("columns", [])]
                t2_cols = [c["name"] for c in master[t2].get("columns", [])]

                relationships.append({
                    "left":  t1,
                    "right": t2,
                    "key":   col_name,
                    "left_cols":  t1_cols,
                    "right_cols": t2_cols,
                })

    if not relationships:
        lines.append("No shared key columns detected automatically.")
        lines.append("")
        lines.append("Tip: Join columns typically end in _ID, _NO, _CODE, or _NUM.")
        return "\n".join(lines)

    for rel in relationships:
        left_extra  = [c for c in rel["right_cols"] if c not in rel["left_cols"]][:6]
        lines.append(f"### {rel['left']} ↔ {rel['right']}")
        lines.append(f"**Join column:** `{rel['key']}`")
        lines.append(f"```sql")
        lines.append(f"JOIN [{rel['right']}] ON [{rel['left']}].{rel['key']} = [{rel['right']}].{rel['key']}")
        lines.append(f"```")
        if left_extra:
            lines.append(
                f"Joining gets from **{rel['right']}**: "
                + ", ".join(f"`{c}`" for c in left_extra)
            )
        lines.append("")

    lines.append("## Usage guidance")
    lines.append("")
    lines.append("- Always qualify column names with table names in multi-table queries")
    lines.append("- Verify join columns exist in both tables before using")
    lines.append("- Use LEFT JOIN when the right-side record may not exist")
    lines.append("- Use INNER JOIN when you only want rows with matches in both tables")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Entity graph auto-generation from discovered schema
# ══════════════════════════════════════════════════════════════════════════════

_FACT_PREFIXES   = {"fact_", "fct_"}
_FACT_SUFFIXES   = {"_fact", "_fct"}
_BRIDGE_PREFIXES = {"bridge_", "brg_", "xref_", "map_", "rel_"}
_BRIDGE_SUFFIXES = {"_bridge", "_brg", "_xref", "_map"}

# Colors per entity type
_ENTITY_COLORS = {
    "fact":      "#C0392B",   # red — fact tables stand out
    "dimension": "#4F86C6",   # blue
    "bridge":    "#9B59B6",   # purple
}

# Generic columns that are NOT reliable join keys
_GENERIC_COLS = {
    "ID", "NAME", "STATUS", "TYPE", "CODE", "DATE",
    "CREATED_AT", "UPDATED_AT", "DESCRIPTION", "NOTES",
    "CREATED_BY", "UPDATED_BY", "IS_ACTIVE", "ACTIVE",
}


def _infer_entity_type(table_name: str) -> str:
    n = table_name.lower()
    if any(n.startswith(p) for p in _FACT_PREFIXES) or any(n.endswith(s) for s in _FACT_SUFFIXES):
        return "fact"
    if any(n.startswith(p) for p in _BRIDGE_PREFIXES) or any(n.endswith(s) for s in _BRIDGE_SUFFIXES):
        return "bridge"
    return "dimension"


def _infer_pk_column(table_name: str, columns: list[dict]) -> str:
    """Heuristic: find the most likely primary-key column for a table."""
    bare = table_name.split(".")[-1].upper()
    col_names_upper = [c["name"].upper() for c in columns]

    # 1. <TABLE>_ID pattern (e.g. CUSTOMER_ID in DIM_CUSTOMER)
    for suffix in ("_ID", "_KEY", "_NO", "_NUM", "_CODE"):
        candidate = bare.lstrip("DIM_").lstrip("FACT_").lstrip("FCT_") + suffix
        # try the full bare name too
        for check in (bare + suffix, candidate):
            if check in col_names_upper:
                idx = col_names_upper.index(check)
                return columns[idx]["name"]

    # 2. Column literally named "ID"
    if "ID" in col_names_upper:
        return columns[col_names_upper.index("ID")]["name"]

    # 3. First column whose name ends with _ID
    for col in columns:
        if col["name"].upper().endswith("_ID"):
            return col["name"]

    # 4. Fallback: first column
    return columns[0]["name"] if columns else ""


def _display_name_from_table(table_name: str) -> str:
    """DIM_CUSTOMER → Customer, FACT_DAILY_SALES → Daily Sales"""
    bare = table_name.split(".")[-1]
    for prefix in ("DIM_", "FACT_", "FCT_", "BRIDGE_", "BRG_", "XREF_", "MAP_", "REL_", "STG_", "VW_", "V_"):
        if bare.upper().startswith(prefix):
            bare = bare[len(prefix):]
            break
    return " ".join(w.capitalize() for w in bare.replace("_", " ").split())


def build_entity_graph_from_schema(schema_dir: str) -> dict:
    """
    Parse _schema.json to auto-generate a starter entity graph.

    Returns:
        {
          "entities":      [{"entity_name", "table_name", "schema_name",
                             "pk_column", "display_name", "entity_type",
                             "color", "pos_x", "pos_y"}],
          "relationships": [{"from_entity", "to_entity",
                             "from_column", "to_column"}],
        }

    Both sets are annotated with confidence_score=75 / status='suggested'
    so the admin sees them as drafts requiring confirmation.
    """
    schema_json = Path(schema_dir) / "_schema.json"
    if not schema_json.exists():
        return {"entities": [], "relationships": []}

    master: dict = json.loads(schema_json.read_text(encoding="utf-8"))
    if not master:
        return {"entities": [], "relationships": []}

    # ── Build entity list ────────────────────────────────────────────────────
    entities: list[dict] = []
    # Maps FQN key → bare entity_name for relationship building
    fqn_to_entity: dict[str, str] = {}

    for idx, (fqn, tbl_info) in enumerate(master.items()):
        parts = fqn.split(".")
        bare_table  = parts[-1]
        schema_part = parts[-2] if len(parts) >= 2 else ""

        entity_name = bare_table          # unique within account, short
        entity_type = _infer_entity_type(bare_table)
        columns     = tbl_info.get("columns", [])
        pk_col      = _infer_pk_column(bare_table, columns)
        display     = _display_name_from_table(bare_table)
        color       = _ENTITY_COLORS.get(entity_type, "#4F86C6")

        # Arrange in a simple grid (facts centred, dims around)
        col_pos = idx % 5
        row_pos = idx // 5
        pos_x   = 80 + col_pos * 220
        pos_y   = 80 + row_pos * 180

        fqn_to_entity[fqn] = entity_name
        entities.append({
            "entity_name":      entity_name,
            "table_name":       bare_table,
            "schema_name":      schema_part,
            "pk_column":        pk_col,
            "display_name":     display,
            "entity_type":      entity_type,
            "color":            color,
            "pos_x":            pos_x,
            "pos_y":            pos_y,
            "confidence_score": 75,
            "status":           "suggested",
        })

    # ── Build relationship list (reuse shared-column logic from _build_join_map) ─
    col_to_fqns: dict[str, list[str]] = {}
    for fqn, tbl_info in master.items():
        for col in tbl_info.get("columns", []):
            cname = col["name"].upper()
            col_to_fqns.setdefault(cname, []).append(fqn)

    seen_pairs: set[str] = set()
    relationships: list[dict] = []

    entity_names_set = {e["entity_name"] for e in entities}

    for col_name, fqns in col_to_fqns.items():
        if len(fqns) < 2 or col_name in _GENERIC_COLS:
            continue
        for i in range(len(fqns)):
            for j in range(i + 1, len(fqns)):
                f1 = fqns[i]
                f2 = fqns[j]
                e1 = fqn_to_entity.get(f1, f1.split(".")[-1])
                e2 = fqn_to_entity.get(f2, f2.split(".")[-1])
                if e1 not in entity_names_set or e2 not in entity_names_set:
                    continue

                # Prefer fact as from_entity (FK owner)
                t1_type = _infer_entity_type(e1)
                t2_type = _infer_entity_type(e2)
                if t2_type == "fact" and t1_type != "fact":
                    e1, e2 = e2, e1   # swap so fact is from_entity

                pair_key = f"{min(e1,e2)}|{max(e1,e2)}|{col_name}"
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Resolve actual case of the column from master
                actual_col = col_name   # uppercase; resolver handles quoting
                for col in master[f1].get("columns", []):
                    if col["name"].upper() == col_name:
                        actual_col = col["name"]
                        break

                relationships.append({
                    "from_entity":      e1,
                    "to_entity":        e2,
                    "from_column":      actual_col,
                    "to_column":        actual_col,
                    "relationship_type": "many_to_one",
                    "join_type":        "INNER",
                    "confidence_score": 70,
                    "status":           "suggested",
                })

    return {"entities": entities, "relationships": relationships}


# ══════════════════════════════════════════════════════════════════════════════
# Snowflake
# ══════════════════════════════════════════════════════════════════════════════

def _sf_connect(cfg: dict, max_retries: int = 3):
    """Snowflake connection with retry for transient network failures."""
    import snowflake.connector
    import time as _time

    allowed_keys = {
        "account", "user", "password", "warehouse", "database", "schema", "role",
        "authenticator", "login_timeout", "network_timeout",
        "client_session_keep_alive",
    }
    connect_cfg = {k: v for k, v in cfg.items() if v and k in allowed_keys}
    last_err = None
    for attempt in range(max_retries):
        try:
            conn = snowflake.connector.connect(**connect_cfg)
            if attempt > 0:
                log.info("Snowflake connected on retry %d/%d", attempt + 1, max_retries)
            return conn
        except snowflake.connector.errors.OperationalError as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            log.warning(
                "Snowflake connection attempt %d/%d failed — retrying in %ds: %s",
                attempt + 1, max_retries, wait, str(e)[:120],
            )
            _time.sleep(wait)
        except Exception:
            # Auth errors, config errors — fail fast, don't retry
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Snowflake connection failed")


def _sf_distinct(cur, schema: str, name: str, col_name: str) -> list[str]:
    """Get distinct values for a categorical column in Snowflake."""
    try:
        cur.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{schema}"."{name}" '
            f'WHERE "{col_name}" IS NOT NULL LIMIT {_MAX_DISTINCT + 1}'
        )
        vals = [str(r[0]) for r in cur.fetchall()]
        if len(vals) > _MAX_DISTINCT:
            return []  # Too many — not actually categorical
        return sorted(vals)
    except Exception:
        return []


def _discover_snowflake(cfg: dict, out: Path, allowed: set[str] | None = None, mc: dict | None = None, seed_key: str = "") -> int:
    import snowflake.connector
    conn = _sf_connect(cfg)
    master = {}
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        schema_filter = "" if allowed is not None else "AND TABLE_SCHEMA = CURRENT_SCHEMA()"
        cur.execute(f"""
            SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE, ROW_COUNT, COMMENT
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE IN ('BASE TABLE','VIEW')
              {schema_filter}
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        tables = [dict(r) for r in cur.fetchall()]

        selected_tables = []
        for tbl in tables:
            name = tbl["TABLE_NAME"]
            schema = tbl.get("TABLE_SCHEMA") or cfg.get("schema") or "PUBLIC"
            database = tbl.get("TABLE_CATALOG") or cfg.get("database") or ""
            if not _allowed_table_match(allowed, name, schema, database):
                log.debug("Snowflake: skipping %s (not in selected tables)", name)
                continue
            selected_tables.append(tbl)

        name_counts = {}
        for tbl in selected_tables:
            key_name = _clean_identifier_part(tbl["TABLE_NAME"])
            name_counts[key_name] = name_counts.get(key_name, 0) + 1

        for tbl in selected_tables:
            name = tbl["TABLE_NAME"]
            schema = tbl.get("TABLE_SCHEMA") or cfg.get("schema") or "PUBLIC"
            database = tbl.get("TABLE_CATALOG") or cfg.get("database") or ""
            normalized_name = _clean_identifier_part(name)
            table_key = name if name_counts.get(normalized_name, 0) <= 1 else f"{schema}.{name}"
            file_stem = name if table_key == name else f"{schema}__{name}"
            cur.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                       CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{name}'
                ORDER BY ORDINAL_POSITION
            """)
            columns = [dict(r) for r in cur.fetchall()]

            # ── Masking resolution ────────────────────────────────────────
            _sf_db   = (database or cfg.get("database", "")).upper()
            _fqn     = f"{_sf_db}.{schema}.{name}".upper()
            _tbl_cfg = (mc or {}).get(_fqn, {})
            _mode    = _tbl_cfg.get("mode", "auto")

            col_defs = [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"]} for c in columns]
            _pre_masked_fields = _resolve_masking_fields(
                col_defs, _mode, set(_tbl_cfg.get("masked_fields", []))
            )

            # Distinct value scan (always for real-row tables)
            distinct_map = {}
            if _mode != "all":  # "all" masking makes distinct values pointless
                plain_cur = conn.cursor()
                for col in columns:
                    cname, ctype = col["COLUMN_NAME"], col["DATA_TYPE"]
                    if cname in _pre_masked_fields:
                        continue
                    if _is_categorical(cname, ctype):
                        vals = _sf_distinct(plain_cur, schema, name, cname)
                        if vals:
                            distinct_map[cname] = vals

            # Read real rows then mask, or fall back to fully synthetic
            _sf_schema, _sf_name = schema, name  # capture for lambda
            sample, _masked_set, _replacement_map, _synthetic_used = _apply_masking(
                fetch_fn=lambda: _sf_fetch_sample(cur, _sf_schema, _sf_name),
                col_defs=col_defs,
                mode=_mode,
                explicit_fields=set(_tbl_cfg.get("masked_fields", [])),
                table_name=name,
                seed_key=seed_key,
                allow_unmasked=bool(_tbl_cfg.get("allow_unmasked_kb", False)),
            )

            # Strip raw distinct values for any column that is being masked.
            # Without this, real PII (names, emails, etc.) leaks into the KB
            # markdown even when the column itself is masked in the sample rows.
            _masked_upper = {f.upper() for f in _masked_set}
            distinct_map = {k: v for k, v in distinct_map.items()
                            if k.upper() not in _masked_upper}

            (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                _sf_md(name, tbl, columns, sample, distinct_map), encoding="utf-8"
            )
            master[table_key] = {
                "columns": [
                    {"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                     "nullable": c["IS_NULLABLE"] == "YES",
                     "comment": c.get("COMMENT") or ""}
                    for c in columns
                ],
                "row_count":           tbl.get("ROW_COUNT"),
                "comment":             tbl.get("COMMENT") or "",
                "schema":              schema,
                "database":            database,
                "fields_sent":         [c["COLUMN_NAME"] for c in columns],
                "row_count_sent":      len(sample),
                "masked_fields":       sorted(_masked_set),
                "mask_mode":           _mode,
                "mask_replacement_map": _replacement_map,
                "synthetic_used":      _synthetic_used,
            }
            log.info("Snowflake: discovered %s (%d cols, %d categorical)",
                     name, len(columns), len(distinct_map))
    finally:
        conn.close()
    _write_schema_json(out, master)
    _write_join_map(out, master)
    return len(master)


def _run_snowflake(cfg: dict, sql: str, max_rows: int = 200) -> list[dict]:
    import snowflake.connector
    conn = _sf_connect(cfg)
    try:
        cur = conn.cursor(snowflake.connector.DictCursor)
        cur.execute(sql)
        return [dict(r) for r in cur.fetchmany(max_rows)]
    finally:
        conn.close()


def _sf_md(name, meta, columns, sample, distinct_map: dict) -> str:
    lines = [f"# {name}"]
    if meta.get("COMMENT"):
        lines.append(f"\n{meta['COMMENT']}")
    lines.append(f"\n**Type:** {meta.get('TABLE_TYPE','TABLE')}  ")
    lines.append(f"**Approximate row count:** {meta.get('ROW_COUNT','unknown')}")
    lines.append("\n## Columns\n")
    lines.append("| Column | Type | Nullable | Notes | Distinct Values |")
    lines.append("|--------|------|:--------:|-------|-----------------|")
    for c in columns:
        nullable = "Yes" if c.get("IS_NULLABLE") == "YES" else "No"
        comment  = (c.get("COMMENT") or "").replace("|", "\\|")
        col_type = c["DATA_TYPE"]
        if c.get("CHARACTER_MAXIMUM_LENGTH"):
            col_type += f"({c['CHARACTER_MAXIMUM_LENGTH']})"
        elif c.get("NUMERIC_PRECISION"):
            col_type += f"({c['NUMERIC_PRECISION']})"
        cname = c["COLUMN_NAME"]
        dist  = ", ".join(f"'{v}'" for v in distinct_map.get(cname, []))
        lines.append(f"| `{cname}` | {col_type} | {nullable} | {comment} | {dist} |")
    _append_sample(lines, sample)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Oracle
# ══════════════════════════════════════════════════════════════════════════════

def _ora_connect(cfg: dict, max_retries: int = 3):
    """Oracle connection with retry for transient network failures."""
    import oracledb
    import time as _time

    last_err = None
    for attempt in range(max_retries):
        try:
            conn = oracledb.connect(
                user=cfg["user"], password=cfg["password"], dsn=cfg["dsn"]
            )
            if attempt > 0:
                log.info("Oracle connected on retry %d/%d", attempt + 1, max_retries)
            return conn
        except oracledb.OperationalError as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            log.warning(
                "Oracle connection attempt %d/%d failed — retrying in %ds: %s",
                attempt + 1, max_retries, wait, str(e)[:120],
            )
            _time.sleep(wait)
        except Exception:
            # Auth errors — fail fast
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Oracle connection failed")


def _ora_distinct(cur, owner: str, name: str, col_name: str) -> list[str]:
    try:
        cur.execute(
            f'SELECT DISTINCT "{col_name}" FROM "{owner}"."{name}" '
            f'WHERE "{col_name}" IS NOT NULL AND ROWNUM <= {_MAX_DISTINCT + 1}'
        )
        vals = [str(r[0]) for r in cur.fetchall()]
        return sorted(vals) if len(vals) <= _MAX_DISTINCT else []
    except Exception:
        return []


def _discover_oracle(cfg: dict, out: Path, allowed: set[str] | None = None, mc: dict | None = None, seed_key: str = "") -> int:
    owner = (cfg.get("schema") or cfg["user"]).upper()
    conn  = _ora_connect(cfg)
    master = {}
    try:
        cur = conn.cursor()
        if allowed is None:
            cur.execute("""
            SELECT t.OWNER, t.TABLE_NAME, 'BASE TABLE' AS TABLE_TYPE, t.NUM_ROWS, tc.COMMENTS
            FROM ALL_TABLES t
            LEFT JOIN ALL_TAB_COMMENTS tc ON tc.OWNER=t.OWNER AND tc.TABLE_NAME=t.TABLE_NAME
            WHERE t.OWNER = :owner ORDER BY t.TABLE_NAME
            """, owner=owner)
        else:
            cur.execute("""
            SELECT t.OWNER, t.TABLE_NAME, 'BASE TABLE' AS TABLE_TYPE, t.NUM_ROWS, tc.COMMENTS
            FROM ALL_TABLES t
            LEFT JOIN ALL_TAB_COMMENTS tc ON tc.OWNER=t.OWNER AND tc.TABLE_NAME=t.TABLE_NAME
            ORDER BY t.OWNER, t.TABLE_NAME
            """)
        tables = [{"OWNER": r[0], "TABLE_NAME": r[1], "TABLE_TYPE": r[2],
                   "ROW_COUNT": r[3], "COMMENT": r[4]} for r in cur.fetchall()]
        if allowed is None:
            cur.execute("""
            SELECT v.OWNER, v.VIEW_NAME, 'VIEW' AS TABLE_TYPE, NULL, vc.COMMENTS
            FROM ALL_VIEWS v
            LEFT JOIN ALL_TAB_COMMENTS vc ON vc.OWNER=v.OWNER AND vc.TABLE_NAME=v.VIEW_NAME
            WHERE v.OWNER = :owner ORDER BY v.VIEW_NAME
            """, owner=owner)
        else:
            cur.execute("""
            SELECT v.OWNER, v.VIEW_NAME, 'VIEW' AS TABLE_TYPE, NULL, vc.COMMENTS
            FROM ALL_VIEWS v
            LEFT JOIN ALL_TAB_COMMENTS vc ON vc.OWNER=v.OWNER AND vc.TABLE_NAME=v.VIEW_NAME
            ORDER BY v.OWNER, v.VIEW_NAME
            """)
        tables += [{"OWNER": r[0], "TABLE_NAME": r[1], "TABLE_TYPE": r[2],
                    "ROW_COUNT": r[3], "COMMENT": r[4]} for r in cur.fetchall()]

        selected_tables = []
        for tbl in tables:
            name = tbl["TABLE_NAME"]
            tbl_owner = (tbl.get("OWNER") or owner).upper()
            if not _allowed_table_match(allowed, name, tbl_owner):
                log.debug("Oracle: skipping %s (not in selected tables)", name)
                continue
            selected_tables.append(tbl)

        name_counts = {}
        for tbl in selected_tables:
            key_name = _clean_identifier_part(tbl["TABLE_NAME"])
            name_counts[key_name] = name_counts.get(key_name, 0) + 1

        for tbl in selected_tables:
            name = tbl["TABLE_NAME"]
            tbl_owner = (tbl.get("OWNER") or owner).upper()
            normalized_name = _clean_identifier_part(name)
            table_key = name if name_counts.get(normalized_name, 0) <= 1 else f"{tbl_owner}.{name}"
            file_stem = name if table_key == name else f"{tbl_owner}__{name}"
            cur.execute("""
                SELECT c.COLUMN_NAME, c.DATA_TYPE, c.NULLABLE,
                       c.DATA_LENGTH, c.DATA_PRECISION, cc.COMMENTS
                FROM ALL_TAB_COLUMNS c
                LEFT JOIN ALL_COL_COMMENTS cc ON cc.OWNER=c.OWNER
                    AND cc.TABLE_NAME=c.TABLE_NAME AND cc.COLUMN_NAME=c.COLUMN_NAME
                WHERE c.OWNER=:owner AND c.TABLE_NAME=:tname ORDER BY c.COLUMN_ID
            """, owner=tbl_owner, tname=name)
            columns = [{"COLUMN_NAME": r[0], "DATA_TYPE": r[1],
                        "IS_NULLABLE": "YES" if r[2] == "Y" else "NO",
                        "DATA_LENGTH": r[3], "DATA_PRECISION": r[4],
                        "COMMENT": r[5]} for r in cur.fetchall()]

            # ── Masking resolution ────────────────────────────────────────
            _fqn     = f"{tbl_owner}.{name}".upper()
            _tbl_cfg = (mc or {}).get(_fqn, {})
            _mode    = _tbl_cfg.get("mode", "auto")

            col_defs = [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"]} for c in columns]
            _pre_masked_fields = _resolve_masking_fields(
                col_defs, _mode, set(_tbl_cfg.get("masked_fields", []))
            )

            distinct_map = {}
            if _mode != "all":
                for col in columns:
                    cname = col["COLUMN_NAME"]
                    if cname in _pre_masked_fields:
                        continue
                    if _is_categorical(cname, col["DATA_TYPE"]):
                        vals = _ora_distinct(cur, tbl_owner, name, cname)
                        if vals:
                            distinct_map[cname] = vals

            def _ora_fetch():
                try:
                    cur.execute(f'SELECT * FROM "{tbl_owner}"."{name}" FETCH FIRST 5 ROWS ONLY')
                    col_names = [d[0] for d in cur.description]
                    return [dict(zip(col_names, row)) for row in cur.fetchall()]
                except Exception:
                    return []

            sample, _masked_set, _replacement_map, _synthetic_used = _apply_masking(
                fetch_fn=_ora_fetch,
                col_defs=col_defs,
                mode=_mode,
                explicit_fields=set(_tbl_cfg.get("masked_fields", [])),
                table_name=name,
                seed_key=seed_key,
                allow_unmasked=bool(_tbl_cfg.get("allow_unmasked_kb", False)),
            )

            # Strip raw distinct values for masked columns
            _masked_upper = {f.upper() for f in _masked_set}
            distinct_map = {k: v for k, v in distinct_map.items()
                            if k.upper() not in _masked_upper}

            (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                _ora_md(name, tbl, columns, sample, tbl_owner, distinct_map), encoding="utf-8"
            )
            master[table_key] = {
                "columns": [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                              "nullable": c["IS_NULLABLE"] == "YES",
                              "comment": c.get("COMMENT") or ""}
                             for c in columns],
                "row_count":           tbl.get("ROW_COUNT"),
                "comment":             tbl.get("COMMENT") or "",
                "owner":               tbl_owner,
                "fields_sent":         [c["COLUMN_NAME"] for c in columns],
                "row_count_sent":      len(sample),
                "masked_fields":       sorted(_masked_set),
                "mask_mode":           _mode,
                "mask_replacement_map": _replacement_map,
                "synthetic_used":      _synthetic_used,
            }
    finally:
        conn.close()
    _write_schema_json(out, master)
    _write_join_map(out, master)
    return len(master)


def _run_oracle(cfg: dict, sql: str, max_rows: int = 200) -> list[dict]:
    conn = _ora_connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, row)) for row in cur.fetchmany(max_rows)]
    finally:
        conn.close()


def _ora_md(name, meta, columns, sample, owner, distinct_map: dict) -> str:
    lines = [f"# {owner}.{name}"]
    if meta.get("COMMENT"):
        lines.append(f"\n{meta['COMMENT']}")
    lines.append(f"\n**Type:** {meta.get('TABLE_TYPE','TABLE')}  **Owner:** {owner}  ")
    lines.append(f"**Approx row count:** {meta.get('ROW_COUNT','unknown')}")
    lines.append("\n## Columns\n")
    lines.append("| Column | Type | Nullable | Notes | Distinct Values |")
    lines.append("|--------|------|:--------:|-------|-----------------|")
    for c in columns:
        nullable = "Yes" if c["IS_NULLABLE"] == "YES" else "No"
        comment  = (c.get("COMMENT") or "").replace("|", "\\|")
        col_type = c["DATA_TYPE"]
        if c.get("DATA_PRECISION"):
            col_type += f"({c['DATA_PRECISION']})"
        elif c.get("DATA_LENGTH") and c["DATA_TYPE"] in ("VARCHAR2", "CHAR", "NVARCHAR2"):
            col_type += f"({c['DATA_LENGTH']})"
        cname = c["COLUMN_NAME"]
        dist  = ", ".join(f"'{v}'" for v in distinct_map.get(cname, []))
        lines.append(f"| `{cname}` | {col_type} | {nullable} | {comment} | {dist} |")
    _append_sample(lines, sample)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Azure SQL
# ══════════════════════════════════════════════════════════════════════════════

def _az_connect(cfg: dict, max_retries: int = 4):
    """
    Open Azure SQL connection with retry + backoff.

    database in cfg is now optional.  When omitted the driver connects to the
    login's default database on the server — useful for server-level enumeration
    when multiple databases need to be discovered.
    """
    import pyodbc
    import time as _time
    driver        = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    login_timeout = int(cfg.get("login_timeout", 60))
    db_part       = f"DATABASE={cfg['database']};" if cfg.get("database") else ""
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={cfg['server']};"
        f"{db_part}"
        f"UID={cfg['user']};"
        f"PWD={cfg['password']};"
        f"Encrypt=yes;TrustServerCertificate=no;"
        f"Connection Timeout={login_timeout};"
    )

    # Transient error codes worth retrying — 40613 is Azure SQL auto-pause
    _TRANSIENT_CODES = {"HYT00", "HYT01", "08001", "08S01", "08003", "08004", "40613"}
    # Wait times per retry attempt (seconds) — longer for 40613 cold start
    _WAIT_TIMES = [0, 5, 20, 30]

    last_err = None
    for attempt in range(max_retries):
        try:
            conn = pyodbc.connect(conn_str, timeout=login_timeout)
            if attempt > 0:
                log.info("Azure SQL connected on retry %d/%d", attempt + 1, max_retries)
            return conn
        except pyodbc.Error as e:
            last_err = e
            sqlstate = e.args[0] if e.args else ""
            if sqlstate not in _TRANSIENT_CODES:
                raise  # non-transient — bad credentials, wrong server etc.
            if attempt == max_retries - 1:
                raise  # exhausted all retries
            wait = _WAIT_TIMES[attempt + 1] if attempt + 1 < len(_WAIT_TIMES) else 30
            if sqlstate == "40613":
                log.warning(
                    "Azure SQL auto-pause detected (40613) — waiting %ds for DB to resume "
                    "(attempt %d/%d). Disable auto-pause in Azure Portal to avoid this delay.",
                    wait, attempt + 1, max_retries,
                )
            else:
                log.warning(
                    "Azure SQL transient error %s (attempt %d/%d) — retrying in %ds",
                    sqlstate, attempt + 1, max_retries, wait,
                )
            _time.sleep(wait)
    if last_err:
        raise last_err
    raise RuntimeError("Azure SQL connection failed — no error captured")


def _az_distinct(cur, schema: str, name: str, col_name: str) -> list[str]:
    try:
        cur.execute(
            f"SELECT DISTINCT TOP {_MAX_DISTINCT + 1} [{col_name}] "
            f"FROM [{schema}].[{name}] WHERE [{col_name}] IS NOT NULL"
        )
        vals = [str(r[0]) for r in cur.fetchall()]
        return sorted(vals) if len(vals) <= _MAX_DISTINCT else []
    except Exception:
        return []


def _discover_azure_sql(cfg: dict, out: Path, allowed: set[str] | None = None, mc: dict | None = None, seed_key: str = "") -> int:
    """
    Discover Azure SQL schema and write one .md per table.

    Single-DB mode (database specified): one connection, writes md files for
    all selected tables across all schemas in that database.

    Multi-DB mode (database NOT specified): enumerates sys.databases, then
    reconnects to each database separately.  Azure SQL Database (cloud) does
    not support cross-database three-part references like
    [db].INFORMATION_SCHEMA.TABLES (error 40515), so a new connection is
    opened per database — no prefix syntax used.
    """
    master: dict = {}

    def _process_db(db_cfg: dict) -> tuple[str, int]:
        """Connect to one database, write .md files, return (db_name, count)."""
        conn = _az_connect(db_cfg)
        try:
            cur = conn.cursor()
            # Get actual DB name (handles case mismatches)
            try:
                cur.execute("SELECT DB_NAME()")
                row = cur.fetchone()
                db_upper = (row[0] if row else db_cfg.get("database", "")).upper()
            except Exception:
                db_upper = db_cfg.get("database", "").upper()

            cur.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                FROM   INFORMATION_SCHEMA.TABLES
                WHERE  TABLE_TYPE IN ('BASE TABLE','VIEW')
                ORDER  BY TABLE_SCHEMA, TABLE_NAME
            """)
            all_rows = [(r[0], r[1], r[2]) for r in cur.fetchall()]

            # Filter to allowed tables
            selected = [
                (schema, name, ttype)
                for schema, name, ttype in all_rows
                if _allowed_table_match(allowed, name, schema, db_upper)
            ]

            if not selected:
                return db_upper, 0

            # Detect duplicate table names across schemas
            name_counts: dict[str, int] = {}
            for schema, name, _ in selected:
                key = _clean_identifier_part(name)
                name_counts[key] = name_counts.get(key, 0) + 1

            written = 0
            for schema, name, _ in selected:
                normalized_name = _clean_identifier_part(name)
                table_key = f"{db_upper}.{schema.upper()}.{name}"
                file_stem = (
                    name if name_counts.get(normalized_name, 0) <= 1
                    else f"{schema}__{name}"
                )

                try:
                    cur.execute("""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE,
                               CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION
                        FROM   INFORMATION_SCHEMA.COLUMNS
                        WHERE  TABLE_SCHEMA=? AND TABLE_NAME=?
                        ORDER  BY ORDINAL_POSITION
                    """, schema, name)
                    columns = [
                        {"COLUMN_NAME": r[0], "DATA_TYPE": r[1], "IS_NULLABLE": r[2],
                         "CHARACTER_MAXIMUM_LENGTH": r[3], "NUMERIC_PRECISION": r[4]}
                        for r in cur.fetchall()
                    ]
                except Exception as e:
                    log.warning("AzSQL: cannot read columns for %s.%s: %s", schema, name, e)
                    continue

                # ── Masking resolution ────────────────────────────────────
                _fqn     = f"{db_upper}.{schema}.{name}".upper()
                _tbl_cfg = (mc or {}).get(_fqn, {})
                _mode    = _tbl_cfg.get("mode", "auto")

                col_defs = [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"]}
                            for c in columns]
                _pre_masked_fields = _resolve_masking_fields(
                    col_defs, _mode, set(_tbl_cfg.get("masked_fields", []))
                )

                distinct_map: dict[str, list[str]] = {}
                if _mode != "all":
                    for col in columns:
                        cname = col["COLUMN_NAME"]
                        if cname in _pre_masked_fields:
                            continue
                        if _is_categorical(cname, col["DATA_TYPE"]):
                            vals = _az_distinct(cur, schema, name, cname)
                            if vals:
                                distinct_map[cname] = vals
                                log.info("AzSQL distinct values for %s.%s: %s",
                                         name, cname, vals)

                def _az_fetch():
                    try:
                        cur.execute(f"SELECT TOP 5 * FROM [{schema}].[{name}]")
                        col_names = [d[0] for d in cur.description]
                        return [dict(zip(col_names, r)) for r in cur.fetchall()]
                    except Exception:
                        return []

                sample, _masked_set, _replacement_map, _synthetic_used = _apply_masking(
                    fetch_fn=_az_fetch,
                    col_defs=col_defs,
                    mode=_mode,
                    explicit_fields=set(_tbl_cfg.get("masked_fields", [])),
                    table_name=name,
                    seed_key=seed_key,
                    allow_unmasked=bool(_tbl_cfg.get("allow_unmasked_kb", False)),
                )

                # Strip raw distinct values for masked columns
                _masked_upper = {f.upper() for f in _masked_set}
                distinct_map = {k: v for k, v in distinct_map.items()
                                if k.upper() not in _masked_upper}

                tbl_meta = {"TABLE_SCHEMA": schema, "TABLE_NAME": name,
                            "TABLE_TYPE": "BASE TABLE", "TABLE_CATALOG": db_upper}
                (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                    _az_md(name, tbl_meta, columns, sample, schema,
                           distinct_map, database=db_upper),
                    encoding="utf-8",
                )
                master[table_key] = {
                    "columns": [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                                 "nullable": c["IS_NULLABLE"] == "YES", "comment": ""}
                                for c in columns],
                    "row_count":           None,
                    "comment":             "",
                    "schema":              schema,
                    "database":            db_upper,
                    "fields_sent":         [c["COLUMN_NAME"] for c in columns],
                    "row_count_sent":      len(sample),
                    "masked_fields":       sorted(_masked_set),
                    "mask_mode":           _mode,
                    "mask_replacement_map": _replacement_map,
                    "synthetic_used":      _synthetic_used,
                }
                log.info("Azure SQL: discovered [%s].[%s].[%s] (%d cols, %d categorical)",
                         db_upper, schema, name, len(columns), len(distinct_map))
                written += 1

            return db_upper, written
        finally:
            conn.close()

    if cfg.get("database"):
        # ── Single-DB mode ────────────────────────────────────────────────────
        _process_db(cfg)
    else:
        # ── Multi-DB mode: enumerate then reconnect per database ──────────────
        # First connection (no DB specified) just to enumerate sys.databases
        enum_cfg = dict(cfg)
        conn0 = _az_connect(enum_cfg)
        try:
            cur0 = conn0.cursor()
            try:
                cur0.execute("""
                    SELECT name FROM sys.databases
                    WHERE  name NOT IN ('master','tempdb','model','msdb')
                      AND  state_desc = 'ONLINE'
                    ORDER  BY name
                """)
                db_names = [r[0] for r in cur0.fetchall()]
            except Exception as e:
                log.warning("AzSQL: sys.databases enumeration failed: %s — "
                            "using current DB", e)
                try:
                    cur0.execute("SELECT DB_NAME()")
                    row = cur0.fetchone()
                    db_names = [row[0]] if row and row[0] else []
                except Exception:
                    db_names = []
        finally:
            conn0.close()

        if not db_names:
            log.warning("AzSQL multi-DB: no user databases found")
        else:
            log.info("AzSQL multi-DB: discovered %d databases: %s",
                     len(db_names), ", ".join(db_names))
            for db in db_names:
                per_cfg = dict(cfg)
                per_cfg["database"] = db
                try:
                    _process_db(per_cfg)
                except Exception as e:
                    log.warning("AzSQL: skipping DB %s — %s", db, e)

    _write_schema_json(out, master)
    _write_join_map(out, master)
    return len(master)


def _run_azure_sql(cfg: dict, sql: str, max_rows: int = 200) -> list[dict]:
    conn = _az_connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, row)) for row in cur.fetchmany(max_rows)]
    finally:
        conn.close()


def _az_md(name, meta, columns, sample, schema, distinct_map: dict,
           database: str = "") -> str:
    # Header uses full 3-part FQN for identification purposes (KB keys, Qdrant)
    # but the SQL table name anchor uses 2-part [SCHEMA].[TABLE] because
    # Azure SQL Database (cloud) does not support 3-part names in DML queries.
    if database:
        fqn = f"{database}.{schema}.{name}"
        header = f"# {fqn}"
    else:
        fqn = f"{schema}.{name}"
        header = f"# [{schema}].[{name}]"

    lines = [header]
    lines.append(
        f"\n**Type:** {meta.get('TABLE_TYPE','TABLE')}  "
        f"**Schema:** {schema}"
        + (f"  **Database:** {database}" if database else "")
    )
    if schema:
        # 2-part name for actual SQL — Azure SQL Database only supports [SCHEMA].[TABLE]
        lines.append(
            f"\n**SQL table name:** `[{schema}].[{name}]`  "
            f"— always use this exact two-part name in generated SQL."
        )
    lines.append("\n## Columns\n")
    lines.append("| Column | Type | Nullable | Distinct Values |")
    lines.append("|--------|------|:--------:|-----------------|")
    for c in columns:
        nullable = "Yes" if c["IS_NULLABLE"] == "YES" else "No"
        col_type = c["DATA_TYPE"]
        if c.get("CHARACTER_MAXIMUM_LENGTH"):
            col_type += f"({c['CHARACTER_MAXIMUM_LENGTH']})"
        elif c.get("NUMERIC_PRECISION"):
            col_type += f"({c['NUMERIC_PRECISION']})"
        cname = c["COLUMN_NAME"]
        dist  = ", ".join(f"'{v}'" for v in distinct_map.get(cname, []))
        lines.append(f"| `{cname}` | {col_type} | {nullable} | {dist} |")
    _append_sample(lines, sample)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _append_sample(lines: list, sample: list) -> None:
    if not sample:
        return
    lines.append("\n## Sample data\n")
    headers = list(sample[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in sample:
        vals = [str(row.get(h, ""))[:40].replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(vals) + " |")


def _write_schema_json(out: Path, master: dict) -> None:
    (out / "_schema.json").write_text(
        json.dumps(master, indent=2, default=str), encoding="utf-8"
    )


def _write_join_map(out: Path, master: dict) -> None:
    """Write auto-detected cross-table join relationships to _join_map.md."""
    content = _build_join_map(master)
    (out / "_join_map.md").write_text(content, encoding="utf-8")
    log.info("Join map written to %s/_join_map.md", out)
