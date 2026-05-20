"""
Admin-only export of QueryBot logs into a configured production database.

The application continues writing logs to local SQLite first. When enabled on
an admin database connection, this module provisions a log schema/tables in the
target database and copies new local rows into those tables on demand or daily.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import store
from store.db import get_db

from core.schema import _az_connect, _ora_connect, _sf_connect

log = logging.getLogger("querybot.log_export")

QUERY_TABLE = "QUERY_LOG"
LLM_TABLE = "LLM_CALL_LOG"
EGRESS_TABLE = "KB_DATA_EGRESS_LOG"
DEFAULT_LOG_SCHEMA = "LOGS"
DEFAULT_EXPORT_TIME = "02:00"
_IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")

QUERY_COLUMNS = [
    "SOURCE_ID", "ACCOUNT_ID", "PORTAL_USER_ID", "ZOOM_USER_ID", "QUESTION",
    "SQL_GENERATED", "ROW_COUNT", "SUCCESS", "ERROR_MSG", "LLM_PROVIDER",
    "LLM_MODEL", "TOKENS_IN", "TOKENS_OUT", "COST_USD", "DURATION_MS",
    "CREATED_AT",
]

LLM_COLUMNS = [
    "SOURCE_ID", "ACCOUNT_ID", "QUESTION_ID", "REQUEST_ID", "QUESTION",
    "COMPONENT", "LLM_PROVIDER", "LLM_MODEL", "STATUS", "PAYLOAD_HASH",
    "PAYLOAD_PREVIEW_SANITIZED", "PROMPT_CHARS", "ERROR_MSG", "CREATED_AT",
]

EGRESS_COLUMNS = [
    "SOURCE_ID", "ACCOUNT_ID", "OPERATION", "DB_TYPE",
    "DATABASE_NAME", "SCHEMA_NAME", "TABLE_NAME",
    "COLUMN_COUNT", "SAMPLE_MODE", "DISTINCT_COL_COUNT",
    "TRIGGERED_BY", "CREATED_AT",
    "FIELDS_SENT", "ROW_COUNT_SENT", "MASKED_FIELDS", "MASK_MODE",
    "MASK_REPLACEMENT_MAP",
]


def is_log_export_enabled(credentials: dict) -> bool:
    return str(credentials.get("log_export_enabled", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def get_export_state(db_config_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM external_log_export_state WHERE db_config_id = ?",
            (db_config_id,),
        ).fetchone()
    return dict(row) if row else None


def provision_external_log_store(db_cfg: dict, *, record_state: bool = True) -> dict:
    """Create the target log schema/tables for one saved DB config."""
    db_id = int(db_cfg.get("id") or 0)
    try:
        conn, db_type, schema = _connect_for_export(db_cfg)
        try:
            cur = conn.cursor()
            if db_type == "snowflake":
                _provision_snowflake(cur, schema)
            elif db_type == "azure_sql":
                _provision_azure_sql(cur, schema)
            elif db_type == "oracle":
                _provision_oracle(cur, schema)
            else:
                raise ValueError(f"Unsupported database type for log export: {db_type}")
            _commit(conn)
        finally:
            _close(conn)

        result = {
            "schema": schema,
            "query_table": f"{schema}.{QUERY_TABLE}",
            "llm_table": f"{schema}.{LLM_TABLE}",
        }
        if record_state and db_id:
            _write_export_state(
                db_id,
                status="provisioned",
                message=f"Log tables ready in {schema}",
            )
        return result
    except Exception as exc:
        if record_state and db_id:
            _write_export_state(db_id, status="failed", message=str(exc))
        raise


def reset_egress_and_sync(db_cfg: dict, *, limit: int = 10000) -> dict:
    """
    Truncate the external KB_DATA_EGRESS_LOG table and re-export all local
    egress rows from scratch.  Use this when the external table has a higher
    max SOURCE_ID than the local SQLite (e.g. after a server reinstall).
    """
    db_id = int(db_cfg.get("id") or 0)
    started_at = _now_utc()
    if db_id:
        _write_export_state(db_id, status="running",
                            message="Reset egress + re-sync running", started_at=started_at)
    try:
        provision_external_log_store(db_cfg, record_state=False)
        conn, db_type, schema = _connect_for_export(db_cfg)
        try:
            cur = conn.cursor()
            # Truncate external egress table so we can re-insert from local
            if db_type == "snowflake":
                cur.execute(f'TRUNCATE TABLE "{schema}"."{EGRESS_TABLE}"')
            elif db_type == "azure_sql":
                cur.execute(f"TRUNCATE TABLE [{schema}].[{EGRESS_TABLE}]")
            else:  # oracle
                cur.execute(f'TRUNCATE TABLE "{schema}"."{EGRESS_TABLE}"')
            _commit(conn)
            # Now re-export all local rows (watermark = 0)
            egress_rows = _fetch_egress_rows_after(0, limit)
            _insert_rows(cur, db_type, schema, EGRESS_TABLE, EGRESS_COLUMNS, egress_rows)
            _commit(conn)
        finally:
            _close(conn)

        result = {
            "schema": schema,
            "egress_count": len(egress_rows),
            "reset": True,
        }
        if db_id:
            _write_export_state(
                db_id,
                status="success",
                message=f"Egress reset + re-exported {len(egress_rows)} rows",
                started_at=started_at,
                finished_at=_now_utc(),
                run_date=datetime.now().date().isoformat(),
                egress_count=len(egress_rows),
                last_egress_id=max([int(r[0]) for r in egress_rows if r[0] is not None] or [0]),
            )
        return result
    except Exception as exc:
        if db_id:
            _write_export_state(db_id, status="failed", message=str(exc),
                                started_at=started_at, finished_at=_now_utc(),
                                run_date=datetime.now().date().isoformat())
        raise


def reset_all_and_sync(db_cfg: dict, *, limit: int = 10000) -> dict:
    """
    Truncate ALL three external log tables (QUERY_LOG, LLM_CALL_LOG,
    KB_DATA_EGRESS_LOG) and re-export every local row from scratch.
    Use this after a server reinstall where the external DB has higher
    SOURCE_IDs than the fresh local SQLite.
    """
    db_id = int(db_cfg.get("id") or 0)
    started_at = _now_utc()
    if db_id:
        _write_export_state(db_id, status="running",
                            message="Full reset + re-sync running", started_at=started_at)
    try:
        provision_external_log_store(db_cfg, record_state=False)
        conn, db_type, schema = _connect_for_export(db_cfg)
        try:
            cur = conn.cursor()
            # Truncate all three tables
            for tbl in (QUERY_TABLE, LLM_TABLE, EGRESS_TABLE):
                if db_type == "snowflake":
                    cur.execute(f'TRUNCATE TABLE "{schema}"."{tbl}"')
                elif db_type == "azure_sql":
                    cur.execute(f"TRUNCATE TABLE [{schema}].[{tbl}]")
                else:  # oracle
                    cur.execute(f'TRUNCATE TABLE "{schema}"."{tbl}"')
            _commit(conn)

            # Re-export all local rows from id = 0
            query_rows  = _fetch_query_rows_after(0, limit)
            llm_rows    = _fetch_llm_rows_after(0, limit)
            egress_rows = _fetch_egress_rows_after(0, limit)
            _insert_rows(cur, db_type, schema, QUERY_TABLE,  QUERY_COLUMNS,  query_rows)
            _insert_rows(cur, db_type, schema, LLM_TABLE,    LLM_COLUMNS,    llm_rows)
            _insert_rows(cur, db_type, schema, EGRESS_TABLE, EGRESS_COLUMNS, egress_rows)
            _commit(conn)
        finally:
            _close(conn)

        result = {
            "schema":        schema,
            "query_count":   len(query_rows),
            "llm_count":     len(llm_rows),
            "egress_count":  len(egress_rows),
            "reset":         True,
        }
        if db_id:
            _write_export_state(
                db_id,
                status="success",
                message=(
                    f"Full reset: re-exported {len(query_rows)} query, "
                    f"{len(llm_rows)} LLM, {len(egress_rows)} egress rows"
                ),
                started_at=started_at,
                finished_at=_now_utc(),
                run_date=datetime.now().date().isoformat(),
                query_count=len(query_rows),
                llm_count=len(llm_rows),
                last_query_id=max([int(r[0]) for r in query_rows  if r[0] is not None] or [0]),
                last_llm_id=max([int(r[0])   for r in llm_rows    if r[0] is not None] or [0]),
                egress_count=len(egress_rows),
                last_egress_id=max([int(r[0]) for r in egress_rows if r[0] is not None] or [0]),
            )
        return result
    except Exception as exc:
        if db_id:
            _write_export_state(db_id, status="failed", message=str(exc),
                                started_at=started_at, finished_at=_now_utc(),
                                run_date=datetime.now().date().isoformat())
        raise


def sync_external_logs(db_cfg: dict, *, limit: int = 10000) -> dict:
    """
    Provision, then copy new local query/LLM logs to the external log tables.
    Uses SOURCE_ID from local SQLite and only exports rows higher than the
    target table's current max SOURCE_ID.

    If the external table's max SOURCE_ID is HIGHER than the local max id
    (can happen after a server reinstall with a fresh SQLite), the watermark
    is clamped to the local max so new local rows are still exported.
    """
    db_id = int(db_cfg.get("id") or 0)
    started_at = _now_utc()
    if db_id:
        _write_export_state(db_id, status="running", message="Export running", started_at=started_at)

    try:
        provision_external_log_store(db_cfg, record_state=False)
        conn, db_type, schema = _connect_for_export(db_cfg)
        try:
            cur = conn.cursor()
            query_last_id  = _target_max_source_id(cur, db_type, schema, QUERY_TABLE)
            llm_last_id    = _target_max_source_id(cur, db_type, schema, LLM_TABLE)
            egress_last_id = _target_max_source_id(cur, db_type, schema, EGRESS_TABLE)

            # Guard: if external max > local max, clamp to avoid perpetual 0-row exports.
            # The rows that exist only in the external table (from a previous install) stay;
            # we start exporting from just below the local min to catch all current rows.
            with get_db() as _lconn:
                _lrow = _lconn.execute(
                    "SELECT COALESCE(MIN(id),0), COALESCE(MAX(id),0) FROM kb_data_egress_log"
                ).fetchone()
                local_egress_min = int(_lrow[0] or 0)
                local_egress_max = int(_lrow[1] or 0)

            if egress_last_id > local_egress_max and local_egress_max > 0:
                log.warning(
                    "sync_external_logs: external egress max (%d) > local max (%d) — "
                    "clamping watermark to %d so local rows are exported. "
                    "Use Reset Egress & Re-sync to fully refresh the external table.",
                    egress_last_id, local_egress_max, local_egress_min - 1,
                )
                egress_last_id = max(0, local_egress_min - 1)

            query_rows  = _fetch_query_rows_after(query_last_id, limit)
            llm_rows    = _fetch_llm_rows_after(llm_last_id, limit)
            egress_rows = _fetch_egress_rows_after(egress_last_id, limit)
            _insert_rows(cur, db_type, schema, QUERY_TABLE,  QUERY_COLUMNS,  query_rows)
            _insert_rows(cur, db_type, schema, LLM_TABLE,    LLM_COLUMNS,    llm_rows)
            _insert_rows(cur, db_type, schema, EGRESS_TABLE, EGRESS_COLUMNS, egress_rows)
            _commit(conn)
        finally:
            _close(conn)

        max_query_id  = max([query_last_id]  + [int(r[0]) for r in query_rows  if r[0] is not None])
        max_llm_id    = max([llm_last_id]    + [int(r[0]) for r in llm_rows    if r[0] is not None])
        max_egress_id = max([egress_last_id] + [int(r[0]) for r in egress_rows if r[0] is not None])
        result = {
            "schema": schema,
            "query_count":  len(query_rows),
            "llm_count":    len(llm_rows),
            "egress_count": len(egress_rows),
            "last_query_id":  max_query_id,
            "last_llm_id":    max_llm_id,
            "last_egress_id": max_egress_id,
        }
        if db_id:
            _write_export_state(
                db_id,
                status="success",
                message=(
                    f"Exported {len(query_rows)} query, "
                    f"{len(llm_rows)} LLM, "
                    f"{len(egress_rows)} egress rows"
                ),
                started_at=started_at,
                finished_at=_now_utc(),
                run_date=datetime.now().date().isoformat(),
                query_count=len(query_rows),
                llm_count=len(llm_rows),
                last_query_id=max_query_id,
                last_llm_id=max_llm_id,
                egress_count=len(egress_rows),
                last_egress_id=max_egress_id,
            )
        return result
    except Exception as exc:
        if db_id:
            _write_export_state(
                db_id,
                status="failed",
                message=str(exc),
                started_at=started_at,
                finished_at=_now_utc(),
                run_date=datetime.now().date().isoformat(),
            )
        raise


def run_due_exports_once(now: datetime | None = None) -> list[dict]:
    """Run enabled exports whose configured daily server time has passed."""
    now = now or datetime.now()
    results: list[dict] = []
    for db_cfg in store.list_db_configs():
        creds = db_cfg.get("credentials", {})
        if not is_log_export_enabled(creds):
            continue
        state = get_export_state(int(db_cfg["id"]))
        if not _is_due(creds, state, now):
            continue
        try:
            result = sync_external_logs(db_cfg)
            results.append({"db_config_id": db_cfg["id"], "status": "success", **result})
        except Exception as exc:
            log.warning("Scheduled log export failed for db_config_id=%s: %s", db_cfg["id"], exc)
            results.append({"db_config_id": db_cfg["id"], "status": "failed", "message": str(exc)})
    return results


async def scheduled_log_export_loop(poll_seconds: int = 60) -> None:
    while True:
        try:
            await asyncio.to_thread(run_due_exports_once)
        except Exception as exc:
            log.warning("Scheduled log export loop error: %s", exc)
        await asyncio.sleep(poll_seconds)


def _connect_for_export(db_cfg: dict):
    db_type = db_cfg["db_type"]
    creds = db_cfg.get("credentials", {})
    schema = _log_schema(db_type, creds)
    db_creds = {k: v for k, v in creds.items() if not k.startswith("log_")}
    if db_type == "snowflake":
        return _sf_connect(db_creds), db_type, schema
    if db_type == "azure_sql":
        return _az_connect(db_creds), db_type, schema
    if db_type == "oracle":
        return _ora_connect(db_creds), db_type, schema
    raise ValueError(f"Unsupported database type for log export: {db_type}")


def _log_schema(db_type: str, credentials: dict) -> str:
    default = credentials.get("user") if db_type == "oracle" else DEFAULT_LOG_SCHEMA
    raw = credentials.get("log_schema") or default or DEFAULT_LOG_SCHEMA
    return _safe_ident(raw, DEFAULT_LOG_SCHEMA)


def _safe_ident(value: Any, default: str = DEFAULT_LOG_SCHEMA) -> str:
    candidate = str(value or default).strip()
    if not _IDENT_RE.match(candidate):
        raise ValueError(
            "Log schema/table names may contain only letters, numbers, and underscores, "
            "and must start with a letter."
        )
    return candidate.upper()


def _provision_snowflake(cur, schema: str) -> None:
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."{QUERY_TABLE}" (
            SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
            ACCOUNT_ID VARCHAR,
            PORTAL_USER_ID NUMBER,
            ZOOM_USER_ID VARCHAR,
            QUESTION VARCHAR,
            SQL_GENERATED VARCHAR,
            ROW_COUNT NUMBER,
            SUCCESS BOOLEAN,
            ERROR_MSG VARCHAR,
            LLM_PROVIDER VARCHAR,
            LLM_MODEL VARCHAR,
            TOKENS_IN NUMBER,
            TOKENS_OUT NUMBER,
            COST_USD FLOAT,
            DURATION_MS NUMBER,
            CREATED_AT VARCHAR,
            EXPORTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."{LLM_TABLE}" (
            SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
            ACCOUNT_ID VARCHAR,
            QUESTION_ID VARCHAR,
            REQUEST_ID VARCHAR,
            QUESTION VARCHAR,
            COMPONENT VARCHAR,
            LLM_PROVIDER VARCHAR,
            LLM_MODEL VARCHAR,
            STATUS VARCHAR,
            PAYLOAD_HASH VARCHAR,
            PAYLOAD_PREVIEW_SANITIZED VARCHAR,
            PROMPT_CHARS NUMBER,
            ERROR_MSG VARCHAR,
            CREATED_AT VARCHAR,
            EXPORTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."{EGRESS_TABLE}" (
            SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
            ACCOUNT_ID VARCHAR,
            OPERATION VARCHAR,
            DB_TYPE VARCHAR,
            DATABASE_NAME VARCHAR,
            SCHEMA_NAME VARCHAR,
            TABLE_NAME VARCHAR,
            COLUMN_COUNT NUMBER,
            SAMPLE_MODE VARCHAR,
            DISTINCT_COL_COUNT NUMBER,
            TRIGGERED_BY VARCHAR,
            CREATED_AT VARCHAR,
            FIELDS_SENT VARCHAR,
            ROW_COUNT_SENT NUMBER,
            MASKED_FIELDS VARCHAR,
            MASK_MODE VARCHAR,
            MASK_REPLACEMENT_MAP VARCHAR,
            EXPORTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    # Add columns to existing tables that were provisioned before this migration
    for col, typedef in (
        ("FIELDS_SENT",          "VARCHAR"),
        ("ROW_COUNT_SENT",       "NUMBER"),
        ("MASKED_FIELDS",        "VARCHAR"),
        ("MASK_MODE",            "VARCHAR"),
        ("MASK_REPLACEMENT_MAP", "VARCHAR"),
    ):
        try:
            cur.execute(
                f'ALTER TABLE "{schema}"."{EGRESS_TABLE}" ADD COLUMN IF NOT EXISTS {col} {typedef}'
            )
        except Exception:
            pass  # column already exists or DB doesn't support IF NOT EXISTS


def _provision_azure_sql(cur, schema: str) -> None:
    cur.execute(
        f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'{schema}') "
        f"EXEC(N'CREATE SCHEMA [{schema}]')"
    )
    cur.execute(f"""
        IF OBJECT_ID(N'[{schema}].[{QUERY_TABLE}]', N'U') IS NULL
        BEGIN
            CREATE TABLE [{schema}].[{QUERY_TABLE}] (
                SOURCE_ID BIGINT NOT NULL PRIMARY KEY,
                ACCOUNT_ID NVARCHAR(255) NULL,
                PORTAL_USER_ID BIGINT NULL,
                ZOOM_USER_ID NVARCHAR(255) NULL,
                QUESTION NVARCHAR(MAX) NULL,
                SQL_GENERATED NVARCHAR(MAX) NULL,
                ROW_COUNT BIGINT NULL,
                SUCCESS BIT NULL,
                ERROR_MSG NVARCHAR(MAX) NULL,
                LLM_PROVIDER NVARCHAR(100) NULL,
                LLM_MODEL NVARCHAR(200) NULL,
                TOKENS_IN BIGINT NULL,
                TOKENS_OUT BIGINT NULL,
                COST_USD DECIMAL(18,8) NULL,
                DURATION_MS BIGINT NULL,
                CREATED_AT NVARCHAR(40) NULL,
                EXPORTED_AT DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            )
        END
    """)
    cur.execute(f"""
        IF OBJECT_ID(N'[{schema}].[{LLM_TABLE}]', N'U') IS NULL
        BEGIN
            CREATE TABLE [{schema}].[{LLM_TABLE}] (
                SOURCE_ID BIGINT NOT NULL PRIMARY KEY,
                ACCOUNT_ID NVARCHAR(255) NULL,
                QUESTION_ID NVARCHAR(255) NULL,
                REQUEST_ID NVARCHAR(255) NULL,
                QUESTION NVARCHAR(MAX) NULL,
                COMPONENT NVARCHAR(100) NULL,
                LLM_PROVIDER NVARCHAR(100) NULL,
                LLM_MODEL NVARCHAR(200) NULL,
                STATUS NVARCHAR(50) NULL,
                PAYLOAD_HASH NVARCHAR(128) NULL,
                PAYLOAD_PREVIEW_SANITIZED NVARCHAR(MAX) NULL,
                PROMPT_CHARS BIGINT NULL,
                ERROR_MSG NVARCHAR(MAX) NULL,
                CREATED_AT NVARCHAR(40) NULL,
                EXPORTED_AT DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            )
        END
    """)
    cur.execute(f"""
        IF OBJECT_ID(N'[{schema}].[{EGRESS_TABLE}]', N'U') IS NULL
        BEGIN
            CREATE TABLE [{schema}].[{EGRESS_TABLE}] (
                SOURCE_ID BIGINT NOT NULL PRIMARY KEY,
                ACCOUNT_ID NVARCHAR(255) NULL,
                OPERATION NVARCHAR(50) NULL,
                DB_TYPE NVARCHAR(50) NULL,
                DATABASE_NAME NVARCHAR(255) NULL,
                SCHEMA_NAME NVARCHAR(255) NULL,
                TABLE_NAME NVARCHAR(255) NULL,
                COLUMN_COUNT BIGINT NULL,
                SAMPLE_MODE NVARCHAR(50) NULL,
                DISTINCT_COL_COUNT BIGINT NULL,
                TRIGGERED_BY NVARCHAR(100) NULL,
                CREATED_AT NVARCHAR(40) NULL,
                FIELDS_SENT NVARCHAR(MAX) NULL,
                ROW_COUNT_SENT BIGINT NULL,
                MASKED_FIELDS NVARCHAR(MAX) NULL,
                MASK_MODE NVARCHAR(50) NULL,
                MASK_REPLACEMENT_MAP NVARCHAR(MAX) NULL,
                EXPORTED_AT DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            )
        END
    """)
    # Add columns to existing tables — use INFORMATION_SCHEMA instead of
    # COL_LENGTH because COL_LENGTH does not reliably handle bracketed names.
    for col, typedef in (
        ("FIELDS_SENT",          "NVARCHAR(MAX) NULL"),
        ("ROW_COUNT_SENT",       "BIGINT NULL"),
        ("MASKED_FIELDS",        "NVARCHAR(MAX) NULL"),
        ("MASK_MODE",            "NVARCHAR(50) NULL"),
        ("MASK_REPLACEMENT_MAP", "NVARCHAR(MAX) NULL"),
    ):
        try:
            cur.execute(f"""
                IF NOT EXISTS (
                    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = N'{schema}'
                      AND TABLE_NAME   = N'{EGRESS_TABLE}'
                      AND COLUMN_NAME  = N'{col}'
                )
                ALTER TABLE [{schema}].[{EGRESS_TABLE}] ADD [{col}] {typedef}
            """)
        except Exception as _col_exc:
            log.debug("Azure SQL add column %s skipped: %s", col, _col_exc)


def _provision_oracle(cur, schema: str) -> None:
    try:
        cur.execute("SELECT COUNT(*) FROM ALL_USERS WHERE USERNAME = :owner", {"owner": schema})
        exists = int(cur.fetchone()[0] or 0)
    except Exception:
        exists = 1
    if not exists:
        raise ValueError(
            f"Oracle schema '{schema}' does not exist. "
            f"Create the Oracle user/schema first with:\n"
            f"  CREATE USER {schema} IDENTIFIED BY <password>;\n"
            f"  GRANT CONNECT, RESOURCE TO {schema};\n"
            f"Or set log_schema to the name of an existing Oracle user "
            f"(the connected user is always safe to use)."
        )
    if not _oracle_table_exists(cur, schema, EGRESS_TABLE):
        cur.execute(f"""
            CREATE TABLE "{schema}"."{EGRESS_TABLE}" (
                SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
                ACCOUNT_ID VARCHAR2(255),
                OPERATION VARCHAR2(50),
                DB_TYPE VARCHAR2(50),
                DATABASE_NAME VARCHAR2(255),
                SCHEMA_NAME VARCHAR2(255),
                TABLE_NAME VARCHAR2(255),
                COLUMN_COUNT NUMBER,
                SAMPLE_MODE VARCHAR2(50),
                DISTINCT_COL_COUNT NUMBER,
                TRIGGERED_BY VARCHAR2(100),
                CREATED_AT VARCHAR2(40),
                FIELDS_SENT CLOB,
                ROW_COUNT_SENT NUMBER,
                MASKED_FIELDS CLOB,
                MASK_MODE VARCHAR2(50),
                MASK_REPLACEMENT_MAP CLOB,
                EXPORTED_AT TIMESTAMP DEFAULT SYSTIMESTAMP
            )
        """)
    else:
        # Add columns to existing tables; Oracle raises ORA-01430 if column exists
        for col, typedef in (
            ("FIELDS_SENT",          "CLOB"),
            ("ROW_COUNT_SENT",       "NUMBER"),
            ("MASKED_FIELDS",        "CLOB"),
            ("MASK_MODE",            "VARCHAR2(50)"),
            ("MASK_REPLACEMENT_MAP", "CLOB"),
        ):
            try:
                cur.execute(f'ALTER TABLE "{schema}"."{EGRESS_TABLE}" ADD ("{col}" {typedef})')
            except Exception:
                pass  # ORA-01430: column already exists
    if not _oracle_table_exists(cur, schema, QUERY_TABLE):
        cur.execute(f"""
            CREATE TABLE "{schema}"."{QUERY_TABLE}" (
                SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
                ACCOUNT_ID VARCHAR2(255),
                PORTAL_USER_ID NUMBER,
                ZOOM_USER_ID VARCHAR2(255),
                QUESTION CLOB,
                SQL_GENERATED CLOB,
                ROW_COUNT NUMBER,
                SUCCESS NUMBER(1),
                ERROR_MSG CLOB,
                LLM_PROVIDER VARCHAR2(100),
                LLM_MODEL VARCHAR2(200),
                TOKENS_IN NUMBER,
                TOKENS_OUT NUMBER,
                COST_USD NUMBER(18,8),
                DURATION_MS NUMBER,
                CREATED_AT VARCHAR2(40),
                EXPORTED_AT TIMESTAMP DEFAULT SYSTIMESTAMP
            )
        """)
    if not _oracle_table_exists(cur, schema, LLM_TABLE):
        cur.execute(f"""
            CREATE TABLE "{schema}"."{LLM_TABLE}" (
                SOURCE_ID NUMBER NOT NULL PRIMARY KEY,
                ACCOUNT_ID VARCHAR2(255),
                QUESTION_ID VARCHAR2(255),
                REQUEST_ID VARCHAR2(255),
                QUESTION CLOB,
                COMPONENT VARCHAR2(100),
                LLM_PROVIDER VARCHAR2(100),
                LLM_MODEL VARCHAR2(200),
                STATUS VARCHAR2(50),
                PAYLOAD_HASH VARCHAR2(128),
                PAYLOAD_PREVIEW_SANITIZED CLOB,
                PROMPT_CHARS NUMBER,
                ERROR_MSG CLOB,
                CREATED_AT VARCHAR2(40),
                EXPORTED_AT TIMESTAMP DEFAULT SYSTIMESTAMP
            )
        """)


def _oracle_table_exists(cur, schema: str, table: str) -> bool:
    cur.execute(
        "SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = :owner AND TABLE_NAME = :table_name",
        {"owner": schema, "table_name": table},
    )
    return int(cur.fetchone()[0] or 0) > 0


def _target_max_source_id(cur, db_type: str, schema: str, table: str) -> int:
    if db_type == "snowflake":
        cur.execute(f'SELECT COALESCE(MAX(SOURCE_ID), 0) FROM "{schema}"."{table}"')
    elif db_type == "azure_sql":
        cur.execute(f"SELECT COALESCE(MAX(SOURCE_ID), 0) FROM [{schema}].[{table}]")
    elif db_type == "oracle":
        cur.execute(f'SELECT NVL(MAX(SOURCE_ID), 0) FROM "{schema}"."{table}"')
    else:
        raise ValueError(f"Unsupported database type for log export: {db_type}")
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def _fetch_query_rows_after(last_id: int, limit: int) -> list[tuple]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, account_id, portal_user_id, zoom_user_id, question,
                   sql_generated, row_count, success, error_msg, llm_provider,
                   llm_model, tokens_in, tokens_out, cost_usd, duration_ms,
                   created_at
              FROM query_log
             WHERE id > ?
             ORDER BY id
             LIMIT ?
        """, (int(last_id), int(limit))).fetchall()
    return [tuple(r) for r in rows]


def _fetch_llm_rows_after(last_id: int, limit: int) -> list[tuple]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, account_id, question_id, request_id, question, component,
                   llm_provider, llm_model, status, payload_hash,
                   payload_preview_sanitized, prompt_chars, error_msg, created_at
              FROM llm_call_log
             WHERE id > ?
             ORDER BY id
             LIMIT ?
        """, (int(last_id), int(limit))).fetchall()
    return [tuple(r) for r in rows]


def _fetch_egress_rows_after(last_id: int, limit: int) -> list[tuple]:
    with get_db() as conn:
        # Detect which optional columns exist so we can handle DBs that haven't
        # had every migration applied yet — fall back to safe defaults for each.
        existing_cols = {
            row[1] for row in
            conn.execute("PRAGMA table_info(kb_data_egress_log)").fetchall()
        }
        def _col(name: str, fallback: str) -> str:
            return f"COALESCE({name}, {fallback})" if name in existing_cols else fallback

        sql = f"""
            SELECT id, account_id, operation, db_type,
                   database_name, schema_name, table_name,
                   column_count, sample_mode, distinct_col_count,
                   triggered_by, created_at,
                   {_col('fields_sent',          "'[]'")} AS fields_sent,
                   {_col('row_count_sent',        '0')}   AS row_count_sent,
                   {_col('masked_fields',         "'[]'")} AS masked_fields,
                   {_col('mask_mode',             "'none'")} AS mask_mode,
                   {_col('mask_replacement_map',  "'{}'")} AS mask_replacement_map
              FROM kb_data_egress_log
             WHERE id > ?
             ORDER BY id
             LIMIT ?
        """
        rows = conn.execute(sql, (int(last_id), int(limit))).fetchall()

    local_total = 0
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) FROM kb_data_egress_log").fetchone()
        local_total = row[0] if row else 0

    log.info(
        "egress fetch: last_id=%d  found=%d  local_total=%d",
        last_id, len(rows), local_total,
    )
    return [tuple(r) for r in rows]


def diagnose_external_log_store(db_cfg: dict) -> dict:
    """
    Return a diagnostic dict without modifying anything:
      local_egress_count     — rows in local kb_data_egress_log
      local_egress_max_id    — highest local id
      external_table_exists  — whether KB_DATA_EGRESS_LOG exists in external DB
      external_egress_count  — rows in external table (None if table missing)
      external_egress_max_id — max SOURCE_ID in external table (None if missing)
      missing_columns        — columns expected by EGRESS_COLUMNS but absent
      error                  — error string if connection/query failed, else None
    """
    result: dict = {
        "local_egress_count":    0,
        "local_egress_max_id":   0,
        "external_table_exists": False,
        "external_egress_count": None,
        "external_egress_max_id": None,
        "missing_columns":       [],
        "error":                 None,
    }

    # Local counts
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(id),0) FROM kb_data_egress_log"
            ).fetchone()
            result["local_egress_count"]  = int(row[0] or 0)
            result["local_egress_max_id"] = int(row[1] or 0)
    except Exception as exc:
        result["error"] = f"Local DB read failed: {exc}"
        return result

    # External DB
    try:
        conn, db_type, schema = _connect_for_export(db_cfg)
        try:
            cur = conn.cursor()

            # Does the table exist?
            if db_type == "azure_sql":
                cur.execute(
                    f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
                    f"WHERE TABLE_SCHEMA=N'{schema}' AND TABLE_NAME=N'{EGRESS_TABLE}'"
                )
            elif db_type == "snowflake":
                cur.execute(
                    f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
                    f"WHERE TABLE_SCHEMA='{schema}' AND TABLE_NAME='{EGRESS_TABLE}'"
                )
            else:  # oracle
                cur.execute(
                    "SELECT COUNT(*) FROM ALL_TABLES "
                    "WHERE OWNER=:s AND TABLE_NAME=:t",
                    {"s": schema, "t": EGRESS_TABLE},
                )
            tbl_exists = int((cur.fetchone() or [0])[0] or 0) > 0
            result["external_table_exists"] = tbl_exists

            if tbl_exists:
                # Row / id counts
                result["external_egress_max_id"]  = _target_max_source_id(cur, db_type, schema, EGRESS_TABLE)
                if db_type == "azure_sql":
                    cur.execute(f"SELECT COUNT(*) FROM [{schema}].[{EGRESS_TABLE}]")
                elif db_type == "snowflake":
                    cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{EGRESS_TABLE}"')
                else:
                    cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{EGRESS_TABLE}"')
                result["external_egress_count"] = int((cur.fetchone() or [0])[0] or 0)

                # Column check
                if db_type == "azure_sql":
                    cur.execute(
                        f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                        f"WHERE TABLE_SCHEMA=N'{schema}' AND TABLE_NAME=N'{EGRESS_TABLE}'"
                    )
                elif db_type == "snowflake":
                    cur.execute(
                        f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                        f"WHERE TABLE_SCHEMA='{schema}' AND TABLE_NAME='{EGRESS_TABLE}'"
                    )
                else:
                    cur.execute(
                        "SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS "
                        "WHERE OWNER=:s AND TABLE_NAME=:t",
                        {"s": schema, "t": EGRESS_TABLE},
                    )
                existing = {r[0].upper() for r in cur.fetchall()}
                result["missing_columns"] = [
                    c for c in EGRESS_COLUMNS if c not in existing
                ]
        finally:
            _close(conn)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def _insert_rows(cur, db_type: str, schema: str, table: str, columns: list[str], rows: list[tuple]) -> None:
    if not rows:
        return
    column_sql = ", ".join(columns)
    if db_type == "snowflake":
        placeholders = ", ".join(["%s"] * len(columns))
        sql = f'INSERT INTO "{schema}"."{table}" ({column_sql}) VALUES ({placeholders})'
        for row in rows:
            cur.execute(sql, row)
    elif db_type == "azure_sql":
        placeholders  = ", ".join(["?"] * len(columns))
        bracketed_cols = ", ".join(f"[{c}]" for c in columns)
        sql = f"INSERT INTO [{schema}].[{table}] ({bracketed_cols}) VALUES ({placeholders})"
        for row in rows:
            try:
                cur.execute(sql, row)
            except Exception as _row_exc:
                err_str = str(_row_exc)
                if "duplicate" in err_str.lower() or "primary key" in err_str.lower() or "2627" in err_str or "2601" in err_str:
                    log.debug("Azure SQL insert skip duplicate SOURCE_ID in %s: %s", table, err_str[:120])
                else:
                    log.warning("Azure SQL insert error in %s: %s", table, err_str[:200])
                    raise
    elif db_type == "oracle":
        bind_sql = ", ".join([f":{col}" for col in columns])
        sql = f'INSERT INTO "{schema}"."{table}" ({column_sql}) VALUES ({bind_sql})'
        for row in rows:
            cur.execute(sql, {col: value for col, value in zip(columns, row)})
    else:
        raise ValueError(f"Unsupported database type for log export: {db_type}")


def _is_due(credentials: dict, state: dict | None, now: datetime) -> bool:
    hour, minute = _parse_time(credentials.get("log_export_time") or DEFAULT_EXPORT_TIME)
    if now.hour < hour or (now.hour == hour and now.minute < minute):
        return False
    today = now.date().isoformat()
    return not state or state.get("last_run_date") != today


def _parse_time(value: str) -> tuple[int, int]:
    try:
        hour_s, minute_s = str(value or DEFAULT_EXPORT_TIME).split(":", 1)
        hour = max(0, min(23, int(hour_s)))
        minute = max(0, min(59, int(minute_s)))
        return hour, minute
    except Exception:
        return 2, 0


def _write_export_state(
    db_config_id: int,
    *,
    status: str,
    message: str = "",
    started_at: str = "",
    finished_at: str = "",
    run_date: str = "",
    query_count: int = 0,
    llm_count: int = 0,
    last_query_id: int = 0,
    last_llm_id: int = 0,
    egress_count: int = 0,
    last_egress_id: int = 0,
) -> None:
    message = (message or "")[:500]
    with get_db() as conn:
        conn.execute("""
            INSERT INTO external_log_export_state
                (db_config_id, last_run_date, last_started_at, last_finished_at,
                 last_status, last_message, last_query_id, last_llm_id,
                 last_query_count, last_llm_count, last_egress_id, last_egress_count,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(db_config_id) DO UPDATE SET
                last_run_date    = COALESCE(NULLIF(excluded.last_run_date, ''),    external_log_export_state.last_run_date),
                last_started_at  = COALESCE(NULLIF(excluded.last_started_at, ''), external_log_export_state.last_started_at),
                last_finished_at = COALESCE(NULLIF(excluded.last_finished_at,''), external_log_export_state.last_finished_at),
                last_status      = excluded.last_status,
                last_message     = excluded.last_message,
                last_query_id    = CASE WHEN excluded.last_query_id  > 0 THEN excluded.last_query_id  ELSE external_log_export_state.last_query_id  END,
                last_llm_id      = CASE WHEN excluded.last_llm_id    > 0 THEN excluded.last_llm_id    ELSE external_log_export_state.last_llm_id    END,
                last_egress_id   = CASE WHEN excluded.last_egress_id > 0 THEN excluded.last_egress_id ELSE external_log_export_state.last_egress_id END,
                last_query_count  = excluded.last_query_count,
                last_llm_count    = excluded.last_llm_count,
                last_egress_count = excluded.last_egress_count,
                updated_at        = excluded.updated_at
        """, (
            int(db_config_id), run_date, started_at, finished_at, status, message,
            int(last_query_id), int(last_llm_id),
            int(query_count), int(llm_count),
            int(last_egress_id), int(egress_count),
        ))


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _commit(conn) -> None:
    if hasattr(conn, "commit"):
        conn.commit()


def _close(conn) -> None:
    if hasattr(conn, "close"):
        conn.close()
