"""
core/schema_discovery.py

Discovers the full catalogue of databases → schemas → tables from a client
database and returns a structured dict.  Used by the admin panel "Browse
schemas" UI — NOT called on the user query path.

Returned structure:
  {
    "<DATABASE>": {
      "<SCHEMA>": {
        "tables": ["TABLE_A", "TABLE_B", ...],
        "views":  ["VIEW_X", ...]
      }
    }
  }

Key design decisions:
  • Reuses the existing _sf_connect / _ora_connect / _az_connect helpers from
    core/schema.py so we don't duplicate retry / error-handling logic.
  • Read-only — only queries INFORMATION_SCHEMA (or equivalents).  No DDL, no DML.
  • Short timeout: this runs interactively from the admin panel, so we bound it
    to 30 s and surface a clear error instead of hanging the UI.
  • Does NOT write any files — that is discover_and_write()'s job.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("querybot.schema_discovery")


# ── Public entry point ────────────────────────────────────────────────────────

async def discover_schema_tree(
    db_type: str,
    credentials: dict,
    timeout_seconds: int = 30,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """
    Return the full database → schema → {tables, views} tree for the given
    database connection.

    Runs the blocking connector call in a thread pool so the FastAPI event
    loop isn't blocked.  Raises on connection errors or timeout.
    """
    loop = asyncio.get_event_loop()

    if db_type == "snowflake":
        fn = _discover_snowflake
    elif db_type == "oracle":
        fn = _discover_oracle
    elif db_type == "azure_sql":
        fn = _discover_azure_sql
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}")

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, fn, credentials),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Schema discovery timed out after {timeout_seconds}s — "
            f"check that the DB is reachable and credentials are correct."
        )
    return result


# ── Snowflake ─────────────────────────────────────────────────────────────────

def _discover_snowflake(cfg: dict) -> dict:
    """
    Query Snowflake's INFORMATION_SCHEMA across all databases the connected
    role can see.

    Snowflake security model:
      - The role in the credentials determines what's visible.
      - We iterate every accessible database, switch to it, and query
        INFORMATION_SCHEMA.TABLES.  We catch PermissionError per database so
        one locked DB doesn't abort the whole discovery.

    Returns:  { DATABASE: { SCHEMA: { "tables": [...], "views": [...] } } }
    """
    from core.schema import _sf_connect  # reuse retry + error handling

    conn = _sf_connect(cfg)
    tree: dict[str, Any] = {}

    try:
        import snowflake.connector
        cur = conn.cursor(snowflake.connector.DictCursor)

        # List all databases the role can see
        cur.execute("SHOW DATABASES")
        databases = [row["name"] for row in cur.fetchall()]

        # If the credentials specify a database, only return that one — avoids
        # hammering every DB in a large account.
        if cfg.get("database"):
            target = cfg["database"].upper()
            databases = [d for d in databases if d.upper() == target] or databases[:1]

        for db in databases:
            try:
                # We need to USE the database to query its INFORMATION_SCHEMA
                cur.execute(f'USE DATABASE "{db}"')
                cur.execute("""
                    SELECT TABLE_SCHEMA,
                           TABLE_NAME,
                           TABLE_TYPE
                    FROM   INFORMATION_SCHEMA.TABLES
                    WHERE  TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                      AND  TABLE_SCHEMA NOT IN (
                               'INFORMATION_SCHEMA', 'ACCOUNT_USAGE',
                               'SNOWFLAKE', 'PUBLIC'
                           )
                    ORDER  BY TABLE_SCHEMA, TABLE_TYPE DESC, TABLE_NAME
                """)
                rows = cur.fetchall()
            except Exception as e:
                log.warning("Snowflake: cannot read DB %s — %s", db, e)
                continue

            if not rows:
                continue

            db_entry: dict[str, Any] = {}
            for row in rows:
                schema = row["TABLE_SCHEMA"]
                name   = row["TABLE_NAME"]
                kind   = row["TABLE_TYPE"]     # 'BASE TABLE' or 'VIEW'
                if schema not in db_entry:
                    db_entry[schema] = {"tables": [], "views": []}
                if kind == "VIEW":
                    db_entry[schema]["views"].append(name)
                else:
                    db_entry[schema]["tables"].append(name)

            if db_entry:
                tree[db] = db_entry
                log.info(
                    "Snowflake discovery: %s → %d schemas, %d total objects",
                    db,
                    len(db_entry),
                    sum(len(v["tables"]) + len(v["views"]) for v in db_entry.values()),
                )
    finally:
        conn.close()

    return tree


# ── Oracle ────────────────────────────────────────────────────────────────────

def _discover_oracle(cfg: dict) -> dict:
    """
    Oracle doesn't have multiple "databases" in the same sense as Snowflake —
    a single connection is already scoped to one database (CDB/PDB).  Schemas
    map to Oracle "owners" (users).

    We query ALL_TABLES and ALL_VIEWS for the objects visible to the
    connected user, excluding Oracle system schemas.

    Returns:  { "<DATABASE>": { OWNER: { "tables": [...], "views": [...] } } }
    where <DATABASE> is derived from cfg["dsn"] or cfg["service_name"].
    """
    from core.schema import _ora_connect

    conn = _ora_connect(cfg)
    # Label the top-level key with whatever identifies this DB in the creds
    db_label = (cfg.get("service_name") or cfg.get("dsn") or "ORACLE_DB").upper()
    tree: dict[str, Any] = {}

    _SYSTEM_SCHEMAS = {
        "SYS", "SYSTEM", "OUTLN", "DBA", "DBSNMP", "SYSMAN", "WMSYS",
        "EXFSYS", "CTXSYS", "ORDSYS", "MDSYS", "OLAPSYS", "APEX_PUBLIC_USER",
        "APPQOSSYS", "GSMADMIN_INTERNAL", "XDB", "ANONYMOUS",
    }

    try:
        cur = conn.cursor()

        # Tables
        cur.execute("""
            SELECT OWNER, TABLE_NAME
            FROM   ALL_TABLES
            ORDER  BY OWNER, TABLE_NAME
        """)
        table_rows = cur.fetchall()

        # Views
        cur.execute("""
            SELECT OWNER, VIEW_NAME
            FROM   ALL_VIEWS
            ORDER  BY OWNER, VIEW_NAME
        """)
        view_rows = cur.fetchall()
    finally:
        conn.close()

    db_entry: dict[str, Any] = {}

    for owner, table_name in table_rows:
        if owner.upper() in _SYSTEM_SCHEMAS:
            continue
        if owner not in db_entry:
            db_entry[owner] = {"tables": [], "views": []}
        db_entry[owner]["tables"].append(table_name)

    for owner, view_name in view_rows:
        if owner.upper() in _SYSTEM_SCHEMAS:
            continue
        if owner not in db_entry:
            db_entry[owner] = {"tables": [], "views": []}
        db_entry[owner]["views"].append(view_name)

    if db_entry:
        tree[db_label] = db_entry
        log.info(
            "Oracle discovery: %d schemas, %d total objects",
            len(db_entry),
            sum(len(v["tables"]) + len(v["views"]) for v in db_entry.values()),
        )

    return tree


# ── Azure SQL ─────────────────────────────────────────────────────────────────

def _discover_azure_sql(cfg: dict) -> dict:
    """
    Azure SQL / SQL Server schema discovery.

    Single-DB mode (database in cfg): one connection, query INFORMATION_SCHEMA.

    Multi-DB mode (database NOT in cfg):
      1. Connect without a database to enumerate sys.databases.
      2. For EACH database, open a SEPARATE connection to that specific database
         and query its INFORMATION_SCHEMA directly.
      Azure SQL Database (cloud) does NOT support three-part cross-database
      references like [db].INFORMATION_SCHEMA.TABLES (error 40515), so we
      must reconnect per database — no prefix syntax is used.

    Returns:  { DATABASE_NAME: { SCHEMA_NAME: { "tables": [...], "views": [...] } } }
    """
    from core.schema import _az_connect

    interactive_cfg = dict(cfg)
    interactive_cfg["login_timeout"] = min(
        int(interactive_cfg.get("login_timeout", 15)), 15
    )

    _SYSTEM_SCHEMAS = {"sys", "INFORMATION_SCHEMA", "guest", "db_owner"}
    tree: dict[str, Any] = {}

    def _query_single_db(db_cfg: dict) -> tuple[str, dict]:
        """Connect to one database and return (db_name, schema_entry)."""
        conn = _az_connect(db_cfg, max_retries=1)
        try:
            cur = conn.cursor()
            try:
                cur.execute("SELECT DB_NAME()")
                row = cur.fetchone()
                db_name = row[0] if row else db_cfg.get("database", "")
            except Exception:
                db_name = db_cfg.get("database", "")

            cur.execute("""
                SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                FROM   INFORMATION_SCHEMA.TABLES
                WHERE  TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                ORDER  BY TABLE_SCHEMA, TABLE_TYPE, TABLE_NAME
            """)
            rows = cur.fetchall()
        finally:
            conn.close()

        db_entry: dict[str, Any] = {}
        for schema, name, kind in rows:
            if schema in _SYSTEM_SCHEMAS:
                continue
            if schema not in db_entry:
                db_entry[schema] = {"tables": [], "views": []}
            if "VIEW" in kind.upper():
                db_entry[schema]["views"].append(name)
            else:
                db_entry[schema]["tables"].append(name)

        return db_name, db_entry

    if cfg.get("database"):
        # ── Single-DB mode ────────────────────────────────────────────────────
        db_name, db_entry = _query_single_db(interactive_cfg)
        if db_entry:
            tree[db_name.upper()] = db_entry
            log.info("AzSQL discovery: %s → %d schemas, %d objects",
                     db_name.upper(), len(db_entry),
                     sum(len(v["tables"]) + len(v["views"])
                         for v in db_entry.values()))
        return tree

    # ── Multi-DB mode ─────────────────────────────────────────────────────────
    # Step 1: connect without a database to enumerate sys.databases
    conn = _az_connect(interactive_cfg, max_retries=1)
    try:
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT name FROM sys.databases
                WHERE  name NOT IN ('master','tempdb','model','msdb')
                  AND  state_desc = 'ONLINE'
                ORDER  BY name
            """)
            db_names = [r[0] for r in cur.fetchall()]
        except Exception as e:
            log.warning("AzSQL: sys.databases failed: %s — trying current DB", e)
            try:
                cur.execute("SELECT DB_NAME()")
                row = cur.fetchone()
                db_names = [row[0]] if row and row[0] else []
            except Exception:
                db_names = []
    finally:
        conn.close()

    if not db_names:
        log.warning("AzSQL multi-DB: no user databases found on server")
        return tree

    log.info("AzSQL multi-DB: found %d databases: %s",
             len(db_names), ", ".join(db_names))

    # Step 2: reconnect to EACH database separately — Azure SQL Database cloud
    # does not support cross-database queries from a single connection.
    for db in db_names:
        per_db_cfg = dict(interactive_cfg)
        per_db_cfg["database"] = db
        try:
            db_name, db_entry = _query_single_db(per_db_cfg)
            if db_entry:
                tree[db_name.upper()] = db_entry
                log.info("AzSQL discovery: %s → %d schemas, %d objects",
                         db_name.upper(), len(db_entry),
                         sum(len(v["tables"]) + len(v["views"])
                             for v in db_entry.values()))
        except Exception as e:
            log.warning("AzSQL: cannot connect to DB %s: %s", db, e)
            continue

    return tree


# ── TTL cache ─────────────────────────────────────────────────────────────────
# We cache discovery results in memory per db_config_id for 24 hours so the
# admin panel doesn't re-query Snowflake on every page load.  The cache is
# busted manually via bust_cache(db_config_id) after a DB config change.

import time as _time
_cache: dict[int, tuple[float, dict]] = {}   # db_config_id → (expires_at, tree)
_CACHE_TTL = 24 * 3600                       # 24 hours


def get_cached_tree(db_config_id: int) -> dict | None:
    """Return the cached schema tree, or None if absent / expired."""
    entry = _cache.get(db_config_id)
    if not entry:
        return None
    expires_at, tree = entry
    if _time.monotonic() > expires_at:
        del _cache[db_config_id]
        return None
    return tree


def set_cached_tree(db_config_id: int, tree: dict) -> None:
    """Store a schema tree in the cache."""
    _cache[db_config_id] = (_time.monotonic() + _CACHE_TTL, tree)


def bust_cache(db_config_id: int) -> None:
    """Invalidate the cached tree for a DB config (call after cred changes)."""
    _cache.pop(db_config_id, None)
