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

# ERP/M3 raw short-code columns that are categorically meaningful regardless of
# data type.  These use numeric or short-text codes whose possible values (e.g.
# CRST=0 open / 4=credit hold / 9=closed) are critical for KB filter conditions.
# They are scanned for distinct values unconditionally.
_ERP_CATEGORICAL_CODES = {
    # AR / financial status
    "CRST", "TEPY", "PYTP", "PYCD", "TECD", "CLST", "RMST", "IIST",
    "IVTP", "ARCD", "IVCL", "TRCD",
    # Order / line status and type
    "ORST", "ORTP", "LTYP",
    # Item / product classification
    "ITGR", "ITTY", "ITCL",
    # Organisation / routing
    "CUCL", "CUTP", "CSCD", "CUCD", "SDST",
    "DIVI", "WHLO", "FACI", "CONO",
    # Sales
    "SMCD", "AGNT",
    # Discount / pricing system
    "DISY", "PRMO",
}


def _is_categorical(col_name: str, col_type: str) -> bool:
    """Return True if this column is likely categorical.

    Extended to also return True for known ERP/M3 raw short-code columns whose
    distinct values are business-critical (status codes, type codes, etc.) but
    whose names don't contain keywords like 'status' or 'type'.
    """
    # ERP categorical codes — scan regardless of data type
    if col_name.upper() in _ERP_CATEGORICAL_CODES:
        return True
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
        master = _normalize_schema(json.loads(p.read_text(encoding="utf-8")))
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


def _normalize_schema(master: dict) -> dict:
    """Old _schema.json stored columns as a plain list; new format wraps in {"columns": [...]}. Normalise to the new format."""
    return {k: ({"columns": v} if isinstance(v, list) else v) for k, v in master.items()}


def load_schema_json(schema_dir: str) -> dict:
    p = Path(schema_dir) / "_schema.json"
    return _normalize_schema(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else {}


def list_schema_files(schema_dir: str) -> list[str]:
    d = Path(schema_dir)
    if not d.exists():
        return []
    return sorted(f.name for f in d.glob("*.md"))


# ══════════════════════════════════════════════════════════════════════════════
# Join map generation
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_pk_columns_azure(cur) -> dict[str, list[str]]:
    """Return {TABLE_NAME_UPPER: [pk_col, ...]} from sys.key_constraints (Azure SQL / SQL Server)."""
    pk_map: dict[str, list[str]] = {}
    try:
        cur.execute("""
            SELECT t.name, c.name
            FROM   sys.key_constraints kc
            JOIN   sys.index_columns   ic ON kc.parent_object_id = ic.object_id
                                         AND kc.unique_index_id  = ic.index_id
            JOIN   sys.tables          t  ON kc.parent_object_id = t.object_id
            JOIN   sys.columns         c  ON ic.object_id        = c.object_id
                                         AND ic.column_id        = c.column_id
            WHERE  kc.type = 'PK'
            ORDER BY t.name, ic.key_ordinal
        """)
        for tbl, col in cur.fetchall():
            pk_map.setdefault(tbl.upper(), []).append(col)
    except Exception as exc:
        log.debug("AzSQL: PK constraint query skipped: %s", exc)
    return pk_map


def _fetch_pk_columns_snowflake(cur, schema: str) -> dict[str, list[str]]:
    """Return {TABLE_NAME_UPPER: [pk_col, ...]} from INFORMATION_SCHEMA (Snowflake)."""
    pk_map: dict[str, list[str]] = {}
    try:
        cur.execute(f"""
            SELECT tc.table_name, kcu.column_name
            FROM   information_schema.table_constraints tc
            JOIN   information_schema.key_column_usage  kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            WHERE  tc.constraint_type = 'PRIMARY KEY'
               AND UPPER(tc.table_schema) = UPPER('{schema}')
            ORDER BY tc.table_name, kcu.ordinal_position
        """)
        for tbl, col in cur.fetchall():
            pk_map.setdefault(tbl.upper(), []).append(col)
    except Exception as exc:
        log.debug("Snowflake: PK constraint query skipped: %s", exc)
    return pk_map


def _fetch_pk_columns_oracle(cur, owner: str) -> dict[str, list[str]]:
    """Return {TABLE_NAME_UPPER: [pk_col, ...]} from ALL_CONSTRAINTS (Oracle)."""
    pk_map: dict[str, list[str]] = {}
    try:
        cur.execute("""
            SELECT acc.TABLE_NAME, acc.COLUMN_NAME
            FROM   ALL_CONSTRAINTS  ac
            JOIN   ALL_CONS_COLUMNS acc
                ON ac.CONSTRAINT_NAME = acc.CONSTRAINT_NAME
               AND ac.OWNER           = acc.OWNER
            WHERE  ac.CONSTRAINT_TYPE = 'P'
               AND ac.OWNER = :owner
            ORDER BY acc.TABLE_NAME, acc.POSITION
        """, owner=owner)
        for tbl, col in cur.fetchall():
            pk_map.setdefault(tbl.upper(), []).append(col)
    except Exception as exc:
        log.debug("Oracle: PK constraint query skipped: %s", exc)
    return pk_map


def _fetch_fk_constraints_azure(cur) -> list[dict]:
    """Query sys.foreign_keys for DB-enforced FK relationships (Azure SQL / SQL Server)."""
    try:
        cur.execute("""
            SELECT tp.name, cp.name, tr.name, cr.name
            FROM   sys.foreign_keys          fk
            JOIN   sys.foreign_key_columns   fkc ON fk.object_id           = fkc.constraint_object_id
            JOIN   sys.tables                tp  ON fkc.parent_object_id    = tp.object_id
            JOIN   sys.columns               cp  ON fkc.parent_object_id    = cp.object_id
                                                AND fkc.parent_column_id    = cp.column_id
            JOIN   sys.tables                tr  ON fkc.referenced_object_id = tr.object_id
            JOIN   sys.columns               cr  ON fkc.referenced_object_id = cr.object_id
                                                AND fkc.referenced_column_id = cr.column_id
            ORDER BY tp.name, cp.name
        """)
        return [
            {"parent_table": r[0], "parent_col": r[1], "ref_table": r[2], "ref_col": r[3]}
            for r in cur.fetchall()
        ]
    except Exception as exc:
        log.debug("AzSQL: FK constraint query skipped: %s", exc)
        return []


def _fetch_fk_constraints_snowflake(cur, schema: str) -> list[dict]:
    """Query INFORMATION_SCHEMA for Snowflake FK constraints (may be unenforced but declared)."""
    try:
        cur.execute(f"""
            SELECT kcu.table_name, kcu.column_name, ccu.table_name, ccu.column_name
            FROM   information_schema.table_constraints      tc
            JOIN   information_schema.key_column_usage       kcu
                ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema    = kcu.table_schema
            JOIN   information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
            JOIN   information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name   = ccu.constraint_name
               AND rc.unique_constraint_schema = ccu.table_schema
            WHERE  tc.constraint_type = 'FOREIGN KEY'
               AND UPPER(tc.table_schema) = UPPER('{schema}')
            ORDER BY kcu.table_name, kcu.ordinal_position
        """)
        return [
            {"parent_table": r[0], "parent_col": r[1], "ref_table": r[2], "ref_col": r[3]}
            for r in cur.fetchall()
        ]
    except Exception as exc:
        log.debug("Snowflake: FK constraint query skipped: %s", exc)
        return []


def _fetch_fk_constraints_oracle(cur, owner: str) -> list[dict]:
    """Query ALL_CONSTRAINTS for Oracle FK relationships."""
    try:
        cur.execute("""
            SELECT a.TABLE_NAME, a.COLUMN_NAME, b.TABLE_NAME, b.COLUMN_NAME
            FROM   ALL_CONS_COLUMNS a
            JOIN   ALL_CONSTRAINTS  ac
                ON a.CONSTRAINT_NAME = ac.CONSTRAINT_NAME AND a.OWNER = ac.OWNER
            JOIN   ALL_CONSTRAINTS  pk
                ON ac.R_CONSTRAINT_NAME = pk.CONSTRAINT_NAME AND ac.R_OWNER = pk.OWNER
            JOIN   ALL_CONS_COLUMNS b
                ON b.CONSTRAINT_NAME = pk.CONSTRAINT_NAME AND b.OWNER = pk.OWNER
               AND b.POSITION = a.POSITION
            WHERE  ac.CONSTRAINT_TYPE = 'R'
               AND a.OWNER = :owner
            ORDER BY a.TABLE_NAME, a.POSITION
        """, owner=owner)
        return [
            {"parent_table": r[0], "parent_col": r[1], "ref_table": r[2], "ref_col": r[3]}
            for r in cur.fetchall()
        ]
    except Exception as exc:
        log.debug("Oracle: FK constraint query skipped: %s", exc)
        return []


def _build_join_map(master: dict) -> str:
    """
    Analyse all discovered tables and auto-detect FK relationships by matching
    column names across tables. Also injects ERP↔DMS alias joins where a
    DMS fact column (e.g. CUS_ORD_NUM) is the semantic equivalent of an ERP raw
    code column (e.g. ORNO) — these would never be detected by shared-name
    matching alone.
    """
    from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table
    from core.schema_enrichment import KNOWN_JOIN_EQUIVALENTS

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
    # Skip special metadata keys (e.g. __db_fk_constraints__ which is a list, not a table dict)
    col_to_tables: dict[str, list[str]] = {}
    for tbl_name, tbl_info in master.items():
        if not isinstance(tbl_info, dict):
            continue
        for col in tbl_info.get("columns", []):
            cname = col["name"].upper()
            col_to_tables.setdefault(cname, []).append(tbl_name)

    join_pairs_seen: set[str] = set()
    relationships: list[dict] = []

    # ── Pass 0: DB-enforced FK constraints (authoritative, highest priority) ──
    # These come from sys.foreign_keys / INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS
    # / ALL_CONSTRAINTS and were stored during schema discovery.  They are always
    # correct and take precedence over the heuristic passes below.
    db_fk_lines: list[str] = []
    for fk in master.get("__db_fk_constraints__", []):
        pt  = fk["parent_table"]
        pc  = fk["parent_col"]
        rt  = fk["ref_table"]
        rc  = fk["ref_col"]
        # Register this pair so Pass 1 shared-name detection doesn't duplicate it
        for _key in (
            f"{min(pt, rt)}|{max(pt, rt)}|{pc}={rc}",
            f"{min(pt, rt)}|{max(pt, rt)}|{pc}={pc}",
        ):
            join_pairs_seen.add(_key)
        db_fk_lines.append(f"### {pt} → {rt}  *(DB-enforced FK)*")
        db_fk_lines.append(
            f"**Join:** `{pt}.{pc}` = `{rt}.{rc}` "
            f"— declared foreign key; `{pt}` is the many-side, `{rt}` is the one-side (parent)."
        )
        db_fk_lines.append("```sql")
        db_fk_lines.append(f"-- many-to-one: keep all {pt} rows even with no {rt} match")
        db_fk_lines.append(f"LEFT JOIN [{rt}] ON [{pt}].[{pc}] = [{rt}].[{rc}]")
        db_fk_lines.append("```")
        db_fk_lines.append("")

    # ── Pass 1: shared column name joins ──────────────────────────────────────
    for col_name, tables in col_to_tables.items():
        if len(tables) < 2:
            continue
        # Skip generic / audit columns that are not meaningful join keys
        if col_name in {
            "ID", "NAME", "STATUS", "TYPE", "CODE", "DATE",
            "CREATED_AT", "UPDATED_AT", "DESCRIPTION", "NOTES",
            # DMS/Azure audit columns present in every fact table — not join keys
            "AZ_EXT_ID", "AZ_LST_UPD_TS", "AZ_LST_UPD_USR", "AZ_LST_UPD_DT",
            "DEL_REC_IND", "DEL_ORD_REC_IND", "DEL_SOP_REC_IND", "DEL_IVC_REC_IND",
            "UNT_OF_MSR",
        }:
            continue

        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                t1, t2 = sorted([tables[i], tables[j]])
                key = f"{t1}|{t2}|{col_name}={col_name}"
                if key in join_pairs_seen:
                    continue
                join_pairs_seen.add(key)
                t1_cols = [c["name"] for c in master[t1].get("columns", [])]
                t2_cols = [c["name"] for c in master[t2].get("columns", [])]
                relationships.append({
                    "left": t1, "right": t2,
                    "left_col": col_name, "right_col": col_name,
                    "alias": False,
                    "left_cols": t1_cols, "right_cols": t2_cols,
                })

    # ── Pass 2: ERP↔DMS alias joins (different column names, same concept) ────
    # KNOWN_JOIN_EQUIVALENTS e.g. {"CUS_ORD_NUM": ["ORNO"], "DLV_NUM": ["DLIX"]}
    # Finds tables that have the DMS column on one side and the ERP code on the other.
    alias_lines: list[str] = []
    for dms_col, erp_codes in KNOWN_JOIN_EQUIVALENTS.items():
        dms_tables = col_to_tables.get(dms_col.upper(), [])
        for erp_code in erp_codes:
            erp_tables = col_to_tables.get(erp_code.upper(), [])
            for dms_tbl in dms_tables:
                for erp_tbl in erp_tables:
                    if dms_tbl == erp_tbl:
                        continue
                    t1, t2 = sorted([dms_tbl, erp_tbl])
                    key = f"{t1}|{t2}|{dms_col}={erp_code}"
                    if key in join_pairs_seen:
                        continue
                    join_pairs_seen.add(key)
                    alias_lines.append(
                        f"### {dms_tbl} ↔ {erp_tbl}  *(ERP alias join)*"
                    )
                    alias_lines.append(
                        f"**Join:** `{dms_tbl}.{dms_col}` = `{erp_tbl}.{erp_code}` "
                        f"— these columns hold the same business key under different names."
                    )
                    alias_lines.append("```sql")
                    alias_lines.append(
                        f"JOIN [{erp_tbl}] ON [{dms_tbl}].{dms_col} = [{erp_tbl}].{erp_code}"
                    )
                    alias_lines.append("```")
                    alias_lines.append("")

    # ── Pass 3: role-playing date dimension joins ───────────────────────────
    # A fact can have many date keys pointing to the same physical date table.
    # Each key has a different business role: order date, invoice date, receipt
    # date, delivery date, etc.  Shared-name detection cannot find these because
    # fact keys are named CUS_IVC_DT_DMS_KEY while the date dimension key is
    # usually DATE_DMS_KEY.
    date_role_lines: list[str] = []
    date_dims: list[tuple[str, str]] = []
    for tbl_name, tbl_info in master.items():
        if not isinstance(tbl_info, dict):
            continue
        cols = tbl_info.get("columns", [])
        if is_date_dimension_table(tbl_name, cols):
            pk = find_date_dimension_key(cols)
            if pk:
                date_dims.append((tbl_name, pk))

    if date_dims:
        for fact_tbl, tbl_info in master.items():
            if not isinstance(tbl_info, dict):
                continue
            fact_cols = [c.get("name", "") for c in tbl_info.get("columns", [])]
            for fact_col in fact_cols:
                role = detect_date_role(fact_col)
                if not role:
                    continue
                for date_tbl, date_pk in date_dims:
                    if fact_tbl == date_tbl:
                        continue
                    key = f"{fact_tbl}|{date_tbl}|{fact_col}={date_pk}|{role.key}"
                    if key in join_pairs_seen:
                        continue
                    join_pairs_seen.add(key)
                    date_role_lines.append(
                        f"### {fact_tbl} → {date_tbl}  *(date role: {role.label})*"
                    )
                    date_role_lines.append(
                        f"**Join:** `{fact_tbl}.{fact_col}` = `{date_tbl}.{date_pk}` "
                        f"— use this for business questions about **{role.label.lower()}**."
                    )
                    if role.synonyms:
                        date_role_lines.append(
                            "**Business terms:** " + ", ".join(f"`{s}`" for s in role.synonyms[:6])
                        )
                    date_role_lines.append("```sql")
                    date_role_lines.append(
                        f"JOIN [{date_tbl}] AS [{role.key}] ON [{fact_tbl}].{fact_col} = [{role.key}].{date_pk}"
                    )
                    date_role_lines.append("```")
                    date_role_lines.append("")

    if not db_fk_lines and not relationships and not alias_lines and not date_role_lines:
        lines.append("No shared key columns detected automatically.")
        lines.append("")
        lines.append("Tip: Join columns typically end in _ID, _NO, _CODE, or _NUM.")
        return "\n".join(lines)

    if db_fk_lines:
        lines.append("## DB-Enforced FK Constraints")
        lines.append("")
        lines.append(
            "These relationships are declared as foreign key constraints in the database. "
            "Use them with highest confidence — the join keys and cardinality are authoritative. "
            "The many-side (child) should be joined to the one-side (parent) with LEFT JOIN "
            "when you want to keep all child rows even if the parent is missing."
        )
        lines.append("")
        lines.extend(db_fk_lines)

    # Count how many tables each shared column appears in (to detect dimension keys)
    _col_table_count = {col: len(tbls) for col, tbls in col_to_tables.items()}

    for rel in relationships:
        left_extra = [c for c in rel["right_cols"] if c not in rel["left_cols"]][:6]
        col = rel["left_col"]
        _is_dim_key = col.upper().endswith("_DMS_KEY") and _col_table_count.get(col.upper(), 0) >= 3
        lines.append(f"### {rel['left']} ↔ {rel['right']}")
        if _is_dim_key:
            lines.append(
                f"> **Note:** `{col}` is a dimension surrogate key shared across "
                f"{_col_table_count[col.upper()]} tables. Both tables reference the same "
                f"dimension — do NOT join these two fact tables directly on this key. "
                f"Instead, join each to its respective dimension table."
            )
        lines.append(f"**Join column:** `{col}`")
        lines.append("```sql")
        lines.append(
            f"JOIN [{rel['right']}] ON [{rel['left']}].{col}"
            f" = [{rel['right']}].{rel['right_col']}"
        )
        lines.append("```")
        if left_extra:
            lines.append(
                f"Joining gets from **{rel['right']}**: "
                + ", ".join(f"`{c}`" for c in left_extra)
            )
        lines.append("")

    if alias_lines:
        lines.append("## ERP ↔ DMS Alias Joins")
        lines.append("")
        lines.append(
            "These joins connect ERP raw-code tables (OOLINE, OSBSTD, FIFO_BI_SAL_MGP_EXT) "
            "to DMS fact tables (CUS_ORD_IVC_FCT etc.) via semantically equivalent "
            "columns that have *different names* in each table."
        )
        lines.append("")
        lines.extend(alias_lines)

    if date_role_lines:
        lines.append("## Role-Playing Date Dimension Joins")
        lines.append("")
        lines.append(
            "These joins map multiple fact-table date keys to the same physical date "
            "dimension. Choose the join whose business role matches the user's wording. "
            "If the user asks only for a generic month/date and multiple roles are valid, "
            "use the approved metric default time role if configured; otherwise ask for clarification."
        )
        lines.append("")
        lines.extend(date_role_lines)

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
_FACT_CONTAINS   = {"_fact_"}
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
    # Common ERP / DW audit columns (exact names)
    "ROW_ID", "ROW_NUM", "SEQ_NUM", "SORT_ORDER",
    "INSERT_DATE", "UPDATE_DATE", "INSERT_TS", "UPDATE_TS",
    "INSERT_BY", "UPDATE_BY", "LOAD_DATE", "LOAD_TS",
    "ETL_BATCH_ID", "BATCH_ID", "JOB_ID", "PROCESS_ID",
    "RECORD_SOURCE", "SOURCE_SYSTEM", "DATA_SOURCE",
    "IS_DELETED", "IS_CURRENT", "IS_VALID", "IS_PROCESSED",
    "EFF_DATE", "EXP_DATE", "START_DATE", "END_DATE",
    "VALID_FROM", "VALID_TO", "EFFECTIVE_DATE", "EXPIRY_DATE",
}

# Prefix-based audit / ETL column patterns (uppercase, checked with startswith)
# These are injected by pipeline tools (ADF, Informatica, Talend, dbt, etc.)
# and appear in nearly every table — joining on them is always wrong.
_AUDIT_COL_PREFIXES = (
    "AZ_",        # Azure Data Factory audit columns (AZ_LST_UPD_TS, AZ_EXT_ID, …)
    "ETL_",       # ETL tool metadata columns
    "DW_",        # Data warehouse housekeeping columns
    "STG_",       # Staging metadata
    "META_",      # Generic metadata prefix
    "SYS_",       # System-generated columns
    "CDC_",       # Change data capture columns
    "DBT_",       # dbt model metadata
    "FIVETRAN_",  # Fivetran connector columns
    "STITCH_",    # Stitch data columns
)

# Suffix-based audit column patterns
_AUDIT_COL_SUFFIXES = (
    "_UPD_TS", "_UPD_DT", "_UPD_DTM",   # update timestamps
    "_INS_TS", "_INS_DT", "_INS_DTM",   # insert timestamps
    "_CRT_TS", "_CRT_DT", "_CRT_DTM",   # create timestamps
    "_LOAD_TS", "_LOAD_DT",              # load timestamps
    "_HASH",                             # row hash / checksum
    "_CHECKSUM",
    "_ROW_VERSION",
    "_BATCH_KEY",
)

# High-prevalence threshold: if a column appears in more than this fraction of
# all tables it is almost certainly an audit/housekeeping column, not a FK.
_AUDIT_PREVALENCE_THRESHOLD = 0.35


def _is_audit_column(col_upper: str, total_tables: int, col_table_count: int) -> bool:
    """Return True if col_upper looks like an audit/ETL column that should not be
    used as a join key in the auto-generated entity graph."""
    if col_upper in _GENERIC_COLS:
        return True
    for prefix in _AUDIT_COL_PREFIXES:
        if col_upper.startswith(prefix):
            return True
    for suffix in _AUDIT_COL_SUFFIXES:
        if col_upper.endswith(suffix):
            return True
    # High-prevalence heuristic: column in >35% of all tables → audit column
    if total_tables > 0 and col_table_count / total_tables > _AUDIT_PREVALENCE_THRESHOLD:
        return True
    return False


# ── Smart FK-to-entity resolution ────────────────────────────────────────────
# Surrogate key suffixes tried longest-first so _DMS_KEY beats _KEY
_SURROGATE_SUFFIXES = (
    "_DMS_KEY", "_KEY", "_FK", "_ID", "_SK",
    "_CD", "_CODE", "_NO", "_NBR", "_NUM",
)

# Contact / demographic attributes that are never FK columns
_DESCRIPTIVE_COLS = frozenset({
    "NAME", "FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "FULL_NAME",
    "DESCRIPTION", "DESC", "NOTES", "NOTE", "COMMENT", "COMMENTS",
    "REMARK", "REMARKS",
    "EMAIL", "PHONE", "FAX", "MOBILE", "TELEPHONE",
    "ADDRESS", "ADDRESS1", "ADDRESS2", "STREET", "SUBURB",
    "CITY", "STATE", "COUNTRY", "ZIP", "ZIPCODE", "POSTCODE", "POSTAL_CODE",
    "GENDER", "WEBSITE", "URL", "DOB",
})


def _is_surrogate_key_col(col_name: str) -> bool:
    """Return True if col_name looks like a surrogate / FK key column."""
    u = col_name.upper()
    if u in _DESCRIPTIVE_COLS or u in _GENERIC_COLS:
        return False
    for pfx in _AUDIT_COL_PREFIXES:
        if u.startswith(pfx):
            return False
    return any(u.endswith(sfx) for sfx in _SURROGATE_SUFFIXES)


def _col_to_target_entity(
    col_upper: str,
    entity_upper_map: dict[str, str],
) -> str | None:
    """
    Derive the target entity from a FK column name by stripping the suffix
    and fuzzy-matching the stem to a known entity.

    WHS_DMS_KEY  → strip _DMS_KEY → WHS  → try WHS, WHS_DMS → WHS_DMS ✓
    Patient_ID   → strip _ID      → PATIENT → try PATIENT, DIM_PATIENT → DIM_Patient ✓
    BRAND_DMS_KEY→ strip _DMS_KEY → BRAND → BRAND_DMS ✓
    """
    for sfx in _SURROGATE_SUFFIXES:
        if col_upper.endswith(sfx):
            stem = col_upper[:-len(sfx)]
            for candidate in (
                stem,
                stem + "_DMS",
                "DIM_" + stem,
                "FACT_" + stem,
                stem + "_DIM",
            ):
                if candidate in entity_upper_map:
                    return entity_upper_map[candidate]
            break  # only strip the longest matching suffix
    return None


def _infer_entity_type(table_name: str) -> str:
    n = table_name.lower()
    if (any(n.startswith(p) for p in _FACT_PREFIXES)
            or any(n.endswith(s) for s in _FACT_SUFFIXES)
            or any(seg in n for seg in _FACT_CONTAINS)):
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

    master: dict = _normalize_schema(json.loads(schema_json.read_text(encoding="utf-8")))
    if not master:
        return {"entities": [], "relationships": []}
    from core.date_roles import detect_date_role, find_date_dimension_key, is_date_dimension_table

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
        # Prefer DB-authoritative PK; fall back to name-pattern heuristic
        _db_pks = tbl_info.get("pk_columns", [])
        pk_col  = _db_pks[0] if _db_pks else _infer_pk_column(bare_table, columns)
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
            "generated_by":     "heuristic",
            "reason":           f"Table name pattern classified {bare_table} as {entity_type}",
        })

    # ── Add role-playing date entities ──────────────────────────────────────
    # Example: one physical DIM_DATE table is represented as "Order Date",
    # "Invoice Date", "Delivery Date", etc. so the resolver can choose the
    # correct FK when the user asks for a specific business date.
    date_dims: list[tuple[str, str, str, str]] = []  # fqn, table, schema, pk
    for fqn, tbl_info in master.items():
        cols = tbl_info.get("columns", [])
        if not is_date_dimension_table(fqn, cols):
            continue
        parts = fqn.split(".")
        date_table = parts[-1]
        date_schema = parts[-2] if len(parts) >= 2 else ""
        date_pk = find_date_dimension_key(cols)
        if date_pk:
            date_dims.append((fqn, date_table, date_schema, date_pk))

    role_entity_names: set[str] = {e["entity_name"] for e in entities}
    role_date_relationships: list[dict] = []
    if date_dims:
        for fqn, tbl_info in master.items():
            fact_entity = fqn_to_entity.get(fqn, fqn.split(".")[-1])
            if _infer_entity_type(fact_entity) != "fact":
                continue
            for col in tbl_info.get("columns", []):
                fact_col = col.get("name", "")
                role = detect_date_role(fact_col)
                if not role:
                    continue
                date_fqn, date_table, date_schema, date_pk = date_dims[0]
                entity_name = role.label
                if entity_name not in role_entity_names:
                    pos_idx = len(role_entity_names)
                    entities.append({
                        "entity_name":      entity_name,
                        "table_name":       date_table,
                        "schema_name":      date_schema,
                        "pk_column":        date_pk,
                        "display_name":     role.label,
                        "description":      (
                            f"Role-playing date dimension: {date_table} used as "
                            f"{role.label}."
                        ),
                        "entity_type":      "dimension",
                        "color":            "#7C3AED",
                        "pos_x":            120 + (pos_idx % 5) * 220,
                        "pos_y":            520 + (pos_idx // 5) * 160,
                        "confidence_score": 88,
                        "status":           "suggested",
                        "generated_by":     "heuristic",
                        "reason":           f"Date role '{role.label}' detected from column {fact_col}",
                    })
                    role_entity_names.add(entity_name)
                role_date_relationships.append({
                    "from_entity":       fact_entity,
                    "to_entity":         entity_name,
                    "from_column":       fact_col,
                    "to_column":         date_pk,
                    "relationship_type": "many_to_one",
                    "join_type":         "LEFT",
                    "label":             role.label,
                    "confidence_score":  88,
                    "status":            "suggested",
                    "generated_by":      "heuristic",
                    "reason":            f"Date role: {fact_col} → {date_table}.{date_pk}",
                })

    # ── Build relationships via FK column-to-entity resolution ──────────────────
    # Star-schema rules:
    #   fact  → dimension  ✓  (standard star edge)
    #   dim   → dimension  ✓  (snowflake edge — dim has FK to sub-dim)
    #   fact  → fact        ✗  never (shared dim-keys ≠ a join between facts)
    #   dim   → fact        ✗  wrong direction
    #
    # One relationship per (from_entity, to_entity) pair — highest-confidence
    # column match wins.

    entity_upper_map: dict[str, str] = {
        e["entity_name"].upper(): e["entity_name"] for e in entities
    }
    entity_type_map: dict[str, str] = {
        e["entity_name"]: e["entity_type"] for e in entities
    }
    pk_map: dict[str, str] = {
        e["entity_name"]: e["pk_column"]
        for e in entities if e.get("pk_column")
    }

    # best_rel[(from, to)] = (confidence, rel_dict)
    best_rel: dict[tuple[str, str], tuple[int, dict]] = {}

    for fqn, tbl_info in master.items():
        e_name = fqn_to_entity.get(fqn, fqn.split(".")[-1])
        e_type = entity_type_map.get(e_name)
        if not e_type:
            continue

        for col in tbl_info.get("columns", []):
            col_name  = col.get("name", "")
            col_upper = col_name.upper()

            if not _is_surrogate_key_col(col_name):
                continue

            target = _col_to_target_entity(col_upper, entity_upper_map)
            if not target or target == e_name:
                continue

            t_type = entity_type_map.get(target)
            if not t_type:
                continue

            # Enforce type rules
            if e_type == "fact" and t_type == "fact":
                continue          # never join fact → fact
            if e_type == "dimension" and t_type == "fact":
                continue          # wrong direction

            # Confidence: higher when column name closely mirrors target table name
            stem = col_upper
            for sfx in _SURROGATE_SUFFIXES:
                if col_upper.endswith(sfx):
                    stem = col_upper[:-len(sfx)]
                    break
            tgt_upper = target.upper()
            if stem == tgt_upper or stem + "_DMS" == tgt_upper:
                confidence = 95   # WHS_DMS_KEY → WHS_DMS  (exact)
            elif tgt_upper.replace("DIM_", "") == stem or tgt_upper in stem:
                confidence = 88   # Patient_ID → DIM_Patient  (prefix-stripped)
            else:
                confidence = 78   # valid key match, less precise stem

            to_col   = pk_map.get(target) or col_name
            pair     = (e_name, target)
            cur_best = best_rel.get(pair, (0, None))[0]
            if confidence > cur_best:
                best_rel[pair] = (confidence, {
                    "from_entity":       e_name,
                    "to_entity":         target,
                    "from_column":       col_name,
                    "to_column":         to_col,
                    "relationship_type": "many_to_one",
                    "join_type":         "INNER" if e_type == "fact" else "LEFT",
                    "confidence_score":  confidence,
                    "status":            "suggested",
                    "generated_by":      "heuristic",
                    "reason":            f"FK column {col_name} resolves to {target}",
                })

    # Merge: date-role relationships take precedence; add heuristic ones after
    date_pairs = {
        (r["from_entity"], r["to_entity"]) for r in role_date_relationships
    }
    relationships: list[dict] = list(role_date_relationships)
    for pair, (_, rel) in best_rel.items():
        if pair not in date_pairs:
            relationships.append(rel)

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

        # Fetch PK columns for all selected tables before the per-table loop
        _sf_pk_schema = cfg.get("schema") or "PUBLIC"
        _sf_pk_cur    = conn.cursor()
        pk_map = _fetch_pk_columns_snowflake(_sf_pk_cur, _sf_pk_schema)
        if pk_map:
            log.info("Snowflake: PK columns discovered for %d tables", len(pk_map))

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

            _pk_cols = pk_map.get(name.upper(), [])
            (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                _sf_md(name, tbl, columns, sample, distinct_map, pk_columns=_pk_cols),
                encoding="utf-8",
            )
            master[table_key] = {
                "columns": [
                    {"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                     "nullable": c["IS_NULLABLE"] == "YES",
                     "comment": c.get("COMMENT") or ""}
                    for c in columns
                ],
                "pk_columns":          _pk_cols,
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

        # ── Fetch DB-declared FK constraints ─────────────────────────────────
        _sf_schema = cfg.get("schema") or "PUBLIC"
        _sf_fk_cur = conn.cursor()
        _sf_fks = _fetch_fk_constraints_snowflake(_sf_fk_cur, _sf_schema)
        if _sf_fks:
            master["__db_fk_constraints__"] = _sf_fks
            log.info("Snowflake: %d DB-declared FK relationships discovered", len(_sf_fks))
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


def _sf_md(name, meta, columns, sample, distinct_map: dict,
           pk_columns: list | None = None) -> str:
    lines = [f"# {name}"]
    if meta.get("COMMENT"):
        lines.append(f"\n{meta['COMMENT']}")
    lines.append(f"\n**Type:** {meta.get('TABLE_TYPE','TABLE')}  ")
    lines.append(f"**Approximate row count:** {meta.get('ROW_COUNT','unknown')}")
    _pk_set = {c.upper() for c in (pk_columns or [])}
    if _pk_set:
        lines.append(f"\n**Primary Key:** {', '.join(f'`{c}`' for c in (pk_columns or []))}")
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
        label = f"`{cname}` **[PK]**" if cname.upper() in _pk_set else f"`{cname}`"
        lines.append(f"| {label} | {col_type} | {nullable} | {comment} | {dist} |")
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

        # Fetch PK columns for all selected tables before the per-table loop
        pk_map = _fetch_pk_columns_oracle(cur, owner)
        if pk_map:
            log.info("Oracle: PK columns discovered for %d tables", len(pk_map))

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

            _pk_cols = pk_map.get(name.upper(), [])
            (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                _ora_md(name, tbl, columns, sample, tbl_owner, distinct_map,
                        pk_columns=_pk_cols),
                encoding="utf-8",
            )
            master[table_key] = {
                "columns": [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                              "nullable": c["IS_NULLABLE"] == "YES",
                              "comment": c.get("COMMENT") or ""}
                             for c in columns],
                "pk_columns":          _pk_cols,
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

        # ── Fetch DB-enforced FK constraints ──────────────────────────────────
        _ora_fks = _fetch_fk_constraints_oracle(cur, owner)
        if _ora_fks:
            master["__db_fk_constraints__"] = _ora_fks
            log.info("Oracle: %d DB-enforced FK relationships discovered", len(_ora_fks))
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


def _ora_md(name, meta, columns, sample, owner, distinct_map: dict,
            pk_columns: list | None = None) -> str:
    lines = [f"# {owner}.{name}"]
    if meta.get("COMMENT"):
        lines.append(f"\n{meta['COMMENT']}")
    lines.append(f"\n**Type:** {meta.get('TABLE_TYPE','TABLE')}  **Owner:** {owner}  ")
    lines.append(f"**Approx row count:** {meta.get('ROW_COUNT','unknown')}")
    _pk_set = {c.upper() for c in (pk_columns or [])}
    if _pk_set:
        lines.append(f"\n**Primary Key:** {', '.join(f'`{c}`' for c in (pk_columns or []))}")
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
        label = f"`{cname}` **[PK]**" if cname.upper() in _pk_set else f"`{cname}`"
        lines.append(f"| {label} | {col_type} | {nullable} | {comment} | {dist} |")
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

            # Fetch PK columns for all tables in one query (before the per-table loop)
            pk_map = _fetch_pk_columns_azure(cur)
            if pk_map:
                log.info("AzSQL: PK columns discovered for %d tables in %s", len(pk_map), db_upper)

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

                # ── Per-column stats: null%, min, max (numeric/date cols) ───────
                _STAT_NUMERIC = {"int", "bigint", "smallint", "tinyint", "decimal",
                                 "numeric", "float", "real", "money", "smallmoney"}
                _STAT_DATE    = {"date", "datetime", "datetime2", "smalldatetime",
                                 "datetimeoffset"}
                stat_cols = [
                    c for c in columns
                    if c["DATA_TYPE"].lower() in (_STAT_NUMERIC | _STAT_DATE)
                    and c["COLUMN_NAME"] not in _pre_masked_fields
                ]
                stats_map: dict[str, dict] = {}
                row_count: int | None = None
                if stat_cols:
                    try:
                        select_parts = ["COUNT(*) AS _total"]
                        for sc in stat_cols:
                            cn = sc["COLUMN_NAME"]
                            select_parts.append(
                                f"SUM(CASE WHEN [{cn}] IS NULL THEN 1 ELSE 0 END)"
                            )
                            select_parts.append(f"MIN([{cn}])")
                            select_parts.append(f"MAX([{cn}])")
                        cur.execute(
                            f"SELECT {', '.join(select_parts)} "
                            f"FROM [{schema}].[{name}]"
                        )
                        srow = cur.fetchone()
                        if srow:
                            row_count = srow[0]
                            _total = srow[0] or 1
                            _idx = 1
                            for sc in stat_cols:
                                cn = sc["COLUMN_NAME"]
                                _null_count = srow[_idx]
                                _mn = srow[_idx + 1]
                                _mx = srow[_idx + 2]
                                _idx += 3
                                stats_map[cn] = {
                                    "null_pct": round(
                                        (_null_count or 0) / _total * 100, 1
                                    ),
                                    "min": str(_mn) if _mn is not None else None,
                                    "max": str(_mx) if _mx is not None else None,
                                }
                    except Exception as _se:
                        log.debug("AzSQL: stats query failed for %s.%s: %s",
                                  schema, name, _se)
                else:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM [{schema}].[{name}]")
                        _rc = cur.fetchone()
                        if _rc:
                            row_count = _rc[0]
                    except Exception:
                        pass

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

                _pk_cols = pk_map.get(name.upper(), [])
                tbl_meta = {"TABLE_SCHEMA": schema, "TABLE_NAME": name,
                            "TABLE_TYPE": "BASE TABLE", "TABLE_CATALOG": db_upper}
                (out / f"{_safe_table_file_stem(file_stem)}.md").write_text(
                    _az_md(name, tbl_meta, columns, sample, schema,
                           distinct_map, database=db_upper,
                           stats_map=stats_map, row_count=row_count,
                           pk_columns=_pk_cols),
                    encoding="utf-8",
                )
                master[table_key] = {
                    "columns": [{"name": c["COLUMN_NAME"], "type": c["DATA_TYPE"],
                                 "nullable": c["IS_NULLABLE"] == "YES", "comment": ""}
                                for c in columns],
                    "pk_columns":          _pk_cols,
                    "row_count":           row_count,
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

            # ── Fetch DB-enforced FK constraints for this database ────────────
            _az_fks = _fetch_fk_constraints_azure(cur)
            if _az_fks:
                existing = master.get("__db_fk_constraints__", [])
                master["__db_fk_constraints__"] = existing + _az_fks
                log.info("AzSQL: %d DB-enforced FK relationships discovered in %s",
                         len(_az_fks), db_upper)

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
           database: str = "", stats_map: dict | None = None,
           row_count: int | None = None, pk_columns: list | None = None) -> str:
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
        if database:
            # SQL Server / Managed Instance: requires 3-part name db.schema.table
            lines.append(
                f"\n**SQL table name:** `{database}.{schema}.{name}`  "
                f"— always use this exact three-part name in generated SQL."
            )
        else:
            # Azure SQL Database (no database component): 2-part [SCHEMA].[TABLE] only
            lines.append(
                f"\n**SQL table name:** `[{schema}].[{name}]`  "
                f"— always use this exact two-part name in generated SQL."
            )
    if row_count is not None:
        if row_count >= 1_000_000:
            _rc_display = f"{row_count / 1_000_000:.1f}M"
            _scale_label = "Large"
        elif row_count >= 100_000:
            _rc_display = f"{row_count:,}"
            _scale_label = "Medium"
        else:
            _rc_display = f"{row_count:,}"
            _scale_label = "Small"
        lines.append(
            f"\n**Row count:** {_rc_display}  **Scale:** {_scale_label}"
        )
    _pk_set = {c.upper() for c in (pk_columns or [])}
    if _pk_set:
        lines.append(f"\n**Primary Key:** {', '.join(f'`{c}`' for c in (pk_columns or []))}")
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
        label = f"`{cname}` **[PK]**" if cname.upper() in _pk_set else f"`{cname}`"
        lines.append(f"| {label} | {col_type} | {nullable} | {dist} |")

    # ── Column Statistics section (numeric/date columns only) ─────────────────
    if stats_map:
        lines.append("\n## Column Statistics\n")
        lines.append("| Column | Null % | Min | Max |")
        lines.append("|--------|-------:|-----|-----|")
        for cname, st in stats_map.items():
            null_pct = f"{st['null_pct']}%" if st.get("null_pct") is not None else ""
            mn = st.get("min") or ""
            mx = st.get("max") or ""
            lines.append(f"| `{cname}` | {null_pct} | {mn} | {mx} |")

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
