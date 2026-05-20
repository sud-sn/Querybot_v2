"""
store/db.py

SQLite schema and connection management.

Location: ./data/querybot.db  (override with DB_PATH env var)

Tables:
  system_config   — global settings (API keys, models, admin password)
  platform_config — chat platform credentials (Zoom / Teams / Slack)
  db_config       — database connection credentials (Snowflake / Oracle / Azure SQL)
  client          — one row per connected workspace/tenant
  query_log       — every query executed (usage tracking + billing)
"""

import os
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger("querybot.db")

DB_PATH = Path(os.getenv("DB_PATH", "data/querybot.db"))

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Global system settings ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_config (
    key             TEXT PRIMARY KEY,
    value_encrypted TEXT NOT NULL,
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- ── Chat platform credentials ─────────────────────────────────────────────────
-- platform_type: zoom | teams | slack | web (web = portal-only, no chat platform)
CREATE TABLE IF NOT EXISTS platform_config (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_type         TEXT    NOT NULL CHECK(platform_type IN ('zoom','teams','slack')),
    name                  TEXT    NOT NULL,
    is_active             INTEGER NOT NULL DEFAULT 1,
    credentials_encrypted TEXT    NOT NULL,
    created_at            TEXT    DEFAULT (datetime('now')),
    updated_at            TEXT    DEFAULT (datetime('now'))
);

-- ── Database connection credentials ──────────────────────────────────────────
-- db_type: snowflake | oracle | azure_sql
CREATE TABLE IF NOT EXISTS db_config (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT    NOT NULL,
    db_type               TEXT    NOT NULL CHECK(db_type IN ('snowflake','oracle','azure_sql')),
    credentials_encrypted TEXT    NOT NULL,
    created_at            TEXT    DEFAULT (datetime('now')),
    updated_at            TEXT    DEFAULT (datetime('now'))
);

-- ── Client / tenant registry ──────────────────────────────────────────────────
-- account_id : Zoom accountId / Teams tenantId / Slack team_id
-- state      : NEW | SCHEMA_READY | KB_BUILDING | READY
-- llm_provider: anthropic | openai  (per-client override, NULL = system default)
CREATE TABLE IF NOT EXISTS client (
    account_id          TEXT    PRIMARY KEY,
    client_name         TEXT    NOT NULL DEFAULT '',
    platform_type       TEXT    NOT NULL,
    platform_config_id  INTEGER REFERENCES platform_config(id) ON DELETE SET NULL,
    db_config_id        INTEGER REFERENCES db_config(id)       ON DELETE SET NULL,
    state               TEXT    NOT NULL DEFAULT 'NEW',
    state_data          TEXT    DEFAULT '{}',
    business_desc       TEXT    DEFAULT '',
    llm_provider        TEXT    DEFAULT NULL,
    llm_model           TEXT    DEFAULT NULL,
    query_limit_monthly INTEGER DEFAULT 500,
    token_limit_monthly INTEGER DEFAULT 0,
    chat_ui_enabled     INTEGER NOT NULL DEFAULT 0,
    enable_llm_audit    INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    DEFAULT (datetime('now')),
    updated_at          TEXT    DEFAULT (datetime('now'))
);

-- ── Query log (usage tracking + billing) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id    TEXT    REFERENCES client(account_id) ON DELETE CASCADE,
    portal_user_id INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    zoom_user_id  TEXT    DEFAULT '',
    question      TEXT,
    sql_generated TEXT,
    row_count     INTEGER DEFAULT 0,
    success       INTEGER NOT NULL DEFAULT 1,
    error_msg     TEXT    DEFAULT '',
    llm_provider  TEXT    DEFAULT '',
    llm_model     TEXT    DEFAULT '',
    tokens_in     INTEGER DEFAULT 0,
    tokens_out    INTEGER DEFAULT 0,
    cost_usd      REAL    DEFAULT 0.0,
    duration_ms   INTEGER DEFAULT 0,
    created_at    TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_call_log (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id                TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    question_id               TEXT    NOT NULL DEFAULT '',
    request_id                TEXT    NOT NULL,
    question                  TEXT    DEFAULT '',
    component                 TEXT    NOT NULL DEFAULT 'general',
    llm_provider              TEXT    DEFAULT '',
    llm_model                 TEXT    DEFAULT '',
    status                    TEXT    NOT NULL DEFAULT 'success',
    payload_hash              TEXT    DEFAULT '',
    payload_preview_sanitized TEXT    DEFAULT '',
    prompt_chars              INTEGER DEFAULT 0,
    error_msg                 TEXT    DEFAULT '',
    created_at                TEXT    DEFAULT (datetime('now'))
);


-- ── User groups (per client) ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS external_log_export_state (
    db_config_id     INTEGER PRIMARY KEY REFERENCES db_config(id) ON DELETE CASCADE,
    last_run_date    TEXT    DEFAULT '',
    last_started_at  TEXT    DEFAULT '',
    last_finished_at TEXT    DEFAULT '',
    last_status      TEXT    DEFAULT '',
    last_message     TEXT    DEFAULT '',
    last_query_id    INTEGER DEFAULT 0,
    last_llm_id      INTEGER DEFAULT 0,
    last_query_count INTEGER DEFAULT 0,
    last_llm_count   INTEGER DEFAULT 0,
    updated_at       TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_group (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- ── Group table access (which tables a group can query) ───────────────────────
CREATE TABLE IF NOT EXISTS group_table_access (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id   INTEGER NOT NULL REFERENCES user_group(id) ON DELETE CASCADE,
    account_id TEXT    NOT NULL,
    table_name TEXT    NOT NULL,
    UNIQUE(group_id, table_name)
);

-- ── Portal users ──────────────────────────────────────────────────────────────
-- zoom_user_id : the userId from Zoom webhook payload (links chat identity)
-- is_temp_pw   : 1 = must change password on next portal login
-- role         : admin | analyst (admin sees all tables, analyst sees group tables)
CREATE TABLE IF NOT EXISTS portal_user (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    group_id        INTEGER REFERENCES user_group(id) ON DELETE SET NULL,
    name            TEXT    NOT NULL,
    email           TEXT    NOT NULL,
    password_hash   TEXT    NOT NULL,
    zoom_user_id    TEXT    DEFAULT NULL,
    role            TEXT    NOT NULL DEFAULT 'analyst' CHECK(role IN ('admin','analyst')),
    is_temp_pw      INTEGER NOT NULL DEFAULT 1,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, email)
);

-- ── Individual table overrides (on top of group access) ───────────────────────
CREATE TABLE IF NOT EXISTS user_table_access (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    account_id TEXT    NOT NULL,
    table_name TEXT    NOT NULL,
    UNIQUE(user_id, table_name)
);

-- ── Registration tokens (one-time link sent in chat) ─────────────────────────
CREATE TABLE IF NOT EXISTS registration_token (
    token      TEXT PRIMARY KEY,
    account_id TEXT NOT NULL,
    zoom_user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── Pinned charts (user portal dashboard) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS pinned_chart (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    account_id   TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    question     TEXT    NOT NULL,
    sql_query    TEXT    NOT NULL,
    chart_type   TEXT    NOT NULL DEFAULT 'bar',
    db_config_id INTEGER REFERENCES db_config(id) ON DELETE SET NULL,
    position     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    DEFAULT (datetime('now')),
    last_refreshed TEXT  DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_portal_user_account  ON portal_user(account_id);
CREATE INDEX IF NOT EXISTS idx_portal_user_zoom     ON portal_user(zoom_user_id);
CREATE INDEX IF NOT EXISTS idx_group_account        ON user_group(account_id);
CREATE INDEX IF NOT EXISTS idx_reg_token            ON registration_token(token);
CREATE INDEX IF NOT EXISTS idx_pinned_user          ON pinned_chart(user_id);

CREATE INDEX IF NOT EXISTS idx_query_log_account ON query_log(account_id);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_account ON llm_call_log(account_id);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_request ON llm_call_log(request_id);
CREATE INDEX IF NOT EXISTS idx_llm_call_log_created ON llm_call_log(created_at);
CREATE INDEX IF NOT EXISTS idx_external_log_export_status ON external_log_export_state(last_status);
CREATE INDEX IF NOT EXISTS idx_client_platform   ON client(platform_type);
CREATE INDEX IF NOT EXISTS idx_client_state      ON client(state);

-- ── Pin tokens (short-lived, for chart pinning from chat) ────────────────────
CREATE TABLE IF NOT EXISTS pin_token (
    token        TEXT PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    account_id   TEXT NOT NULL,
    question     TEXT NOT NULL,
    sql_query    TEXT NOT NULL,
    chart_type   TEXT NOT NULL,
    db_config_id INTEGER NOT NULL,
    expires_at   TEXT NOT NULL,
    created_at   TEXT DEFAULT (datetime('now'))
);

-- ── Metric registry (deterministic SQL for known metrics) ────────────────────
CREATE TABLE IF NOT EXISTS metric_registry (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    name         TEXT    NOT NULL,
    synonyms     TEXT    NOT NULL DEFAULT '',
    sql_template TEXT    NOT NULL,
    description  TEXT    DEFAULT '',
    formula_type TEXT    NOT NULL DEFAULT 'query',
    result_format TEXT   NOT NULL DEFAULT 'number',
    required_columns TEXT DEFAULT '',
    allowed_dimensions TEXT DEFAULT '',
    example_questions TEXT DEFAULT '',
    grain        TEXT    DEFAULT '',
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now'))
);

-- ── Validated examples (proven question→SQL pairs for few-shot) ──────────────
CREATE TABLE IF NOT EXISTS validated_examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    sql_query   TEXT    NOT NULL,
    table_name  TEXT    DEFAULT '',
    source      TEXT    DEFAULT 'kb_stage2',
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- ── Pending clarification (short-lived, per-user clarification state) ────────
CREATE TABLE IF NOT EXISTS pending_clarification (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id    TEXT    NOT NULL,
    zoom_user_id  TEXT    NOT NULL,
    original_q    TEXT    NOT NULL,
    context       TEXT    NOT NULL DEFAULT '',
    expires_at    TEXT    NOT NULL,
    created_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, zoom_user_id)
);

-- ── Business semantic layer (business_term glossary) ─────────────────────────
-- Canonical business terms for an account. Queried at clarification time
-- and injected into SQL prompts when matched. Populated either by admin
-- CRUD or auto-extracted from KB "Business Synonyms" sections.
--
-- kind: 'dimension' | 'metric' | 'filter' | 'entity'
--   - dimension: groupable concept (department, region, product)
--   - metric:    measurable expression (revenue, absenteeism, churn)
--   - filter:    a predicate (active customer, late employee)
--   - entity:    a noun that maps to a table (prescriber → PRESCRIBER table)
-- canonical_expression: the SQL fragment the term compiles to.
--   For metrics: a full aggregate expression (SUM/COUNT/AVG).
--   For filters: a WHERE predicate.
--   For dimensions: a column reference (optionally with CASE grouping).
--   For entities: a table name.
-- requires_clarification: 1 = always ask user to pick from options, 0 = use directly.
-- clarification_options: JSON array of {label, expression, definition} when
--   the term has multiple valid interpretations. Empty string if N/A.
CREATE TABLE IF NOT EXISTS business_term (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id             TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    term                   TEXT    NOT NULL,
    kind                   TEXT    NOT NULL DEFAULT 'metric'
                           CHECK(kind IN ('dimension','metric','filter','entity')),
    canonical_expression   TEXT    NOT NULL DEFAULT '',
    tables_involved        TEXT    DEFAULT '',
    grain                  TEXT    DEFAULT '',
    aliases                TEXT    DEFAULT '',
    definition             TEXT    DEFAULT '',
    requires_clarification INTEGER NOT NULL DEFAULT 0,
    clarification_options  TEXT    DEFAULT '',
    source                 TEXT    DEFAULT 'manual'
                           CHECK(source IN ('manual','kb_extracted','metric_registry')),
    is_active              INTEGER NOT NULL DEFAULT 1,
    created_at             TEXT    DEFAULT (datetime('now')),
    updated_at             TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, term)
);

CREATE INDEX IF NOT EXISTS idx_metric_registry_account ON metric_registry(account_id);
CREATE INDEX IF NOT EXISTS idx_validated_examples_account ON validated_examples(account_id);
CREATE INDEX IF NOT EXISTS idx_business_term_account ON business_term(account_id, is_active);

-- User-submitted semantic layer corrections. These never update KB content
-- directly; admins review and approve/reject them separately.
CREATE TABLE IF NOT EXISTS semantic_field_feedback (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id           TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    portal_user_id       INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    table_fqn            TEXT    NOT NULL,
    schema_name          TEXT    DEFAULT '',
    table_name           TEXT    NOT NULL,
    column_name          TEXT    NOT NULL,
    current_meaning      TEXT    DEFAULT '',
    current_use_case     TEXT    DEFAULT '',
    suggested_meaning    TEXT    DEFAULT '',
    suggested_use_case   TEXT    DEFAULT '',
    user_comment         TEXT    DEFAULT '',
    confidence_score     INTEGER DEFAULT 0,
    status               TEXT    NOT NULL DEFAULT 'pending'
                         CHECK(status IN ('pending','approved','rejected')),
    admin_note           TEXT    DEFAULT '',
    created_at           TEXT    DEFAULT (datetime('now')),
    reviewed_at          TEXT    DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_semantic_feedback_account_status
    ON semantic_field_feedback(account_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_semantic_feedback_field
    ON semantic_field_feedback(account_id, table_fqn, column_name, status);

-- ── LLM pricing (editable per-model cost rates, USD per 1M tokens) ───────────
-- Seeded from hardcoded defaults on first startup.
-- calculate_cost() reads this table first; falls back to defaults for
-- any model not yet present so new models never silently use $0 rates.
CREATE TABLE IF NOT EXISTS llm_pricing (
    model       TEXT PRIMARY KEY,
    tokens_in   REAL NOT NULL,      -- USD per 1M input tokens
    tokens_out  REAL NOT NULL,      -- USD per 1M output tokens
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- ── KB data egress log ────────────────────────────────────────────────────────
-- Records every table processed during schema discovery and KB build so
-- administrators and clients have full transparency over what data was
-- sent to the LLM and what was protected by the synthetic sample guard.
-- One row per table per operation. Persists across KB rebuilds (append-only).
CREATE TABLE IF NOT EXISTS kb_data_egress_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT    NOT NULL,
    operation       TEXT    NOT NULL,   -- 'discovery' | 'kb_build'
    db_type         TEXT    NOT NULL,   -- 'azure_sql' | 'snowflake' | 'oracle'
    database_name   TEXT    NOT NULL DEFAULT '',
    schema_name     TEXT    NOT NULL DEFAULT '',
    table_name      TEXT    NOT NULL,
    column_count    INTEGER NOT NULL DEFAULT 0,
    -- sample_mode: 'synthetic' | 'real' | 'none'
    -- 'synthetic'  = generated fake rows — NO real data reached the LLM
    -- 'real'       = 5 actual rows sent to LLM during KB generation
    -- 'none'       = no sample rows included (schema discovery only)
    sample_mode     TEXT    NOT NULL DEFAULT 'none',
    -- distinct_col_count: number of categorical columns where distinct
    -- production values were scanned and embedded in the KB context
    distinct_col_count INTEGER NOT NULL DEFAULT 0,
    triggered_by    TEXT    NOT NULL DEFAULT 'admin',   -- 'admin' | 'api' | 'system'
    -- v19: fields sent/masked detail
    fields_sent     TEXT    NOT NULL DEFAULT '[]',
    row_count_sent  INTEGER NOT NULL DEFAULT 0,
    masked_fields   TEXT    NOT NULL DEFAULT '[]',
    mask_mode       TEXT    NOT NULL DEFAULT 'none',
    -- v21: per-field replacement strategy map {field: strategy_name}
    mask_replacement_map TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kb_egress_account_op
    ON kb_data_egress_log(account_id, operation, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kb_egress_account_table
    ON kb_data_egress_log(account_id, table_name);

-- ── Entity graph (structured join map) ───────────────────────────────────────
-- Stores the business object model that drives deterministic SQL JOIN resolution.
-- Each entity maps a business concept (Customer, Prescription) to a DB table.
CREATE TABLE IF NOT EXISTS entity_graph (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    entity_name  TEXT    NOT NULL,          -- e.g. "Customer"
    table_name   TEXT    NOT NULL,          -- e.g. "DIM_CUSTOMER"
    schema_name  TEXT    NOT NULL DEFAULT '',
    pk_column    TEXT    NOT NULL DEFAULT '', -- primary / unique key column
    display_name TEXT    NOT NULL DEFAULT '',
    description  TEXT    NOT NULL DEFAULT '',
    entity_type  TEXT    NOT NULL DEFAULT 'dimension', -- fact | dimension | bridge
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_entity_graph_account
    ON entity_graph(account_id, is_active);

-- ── Entity relationships (join edges) ─────────────────────────────────────────
-- Each row defines one JOIN edge between two entities.
-- from_entity holds the FK column; to_entity holds the referenced PK column.
CREATE TABLE IF NOT EXISTS entity_relationships (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id        TEXT    NOT NULL,
    from_entity       TEXT    NOT NULL,   -- entity that owns the FK
    to_entity         TEXT    NOT NULL,   -- entity being joined to
    from_column       TEXT    NOT NULL,   -- FK column in from_entity table
    to_column         TEXT    NOT NULL,   -- PK/referenced column in to_entity table
    relationship_type TEXT    NOT NULL DEFAULT 'many_to_one', -- many_to_one|one_to_one
    join_type         TEXT    NOT NULL DEFAULT 'INNER',       -- INNER|LEFT
    label             TEXT    NOT NULL DEFAULT '',            -- "places", "prescribes"
    is_active         INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT    DEFAULT (datetime('now')),
    join_conditions   TEXT    NOT NULL DEFAULT '[]'           -- JSON: [{from_col,to_col},...] extra join conditions
);
CREATE INDEX IF NOT EXISTS idx_entity_rel_account
    ON entity_relationships(account_id, is_active);

-- ── Entity properties (column roles) ──────────────────────────────────────────
-- Classifies each column of an entity as metric, dimension, filter, date, etc.
-- Drives smarter SELECT-clause generation and synonym resolution.
CREATE TABLE IF NOT EXISTS entity_properties (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    entity_name  TEXT    NOT NULL,
    column_name  TEXT    NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'dimension', -- metric|dimension|filter|date|identifier|ignore
    display_name TEXT    NOT NULL DEFAULT '',
    synonyms     TEXT    NOT NULL DEFAULT '',          -- comma-separated alternative names
    UNIQUE(account_id, entity_name, column_name)
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager: yields connection, commits on clean exit, rolls back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create all tables if they don't exist. Safe to call on every startup.
    Also runs lightweight migrations to add new columns to existing tables
    so upgrades never require manual ALTER TABLE commands.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(_SCHEMA)
        # Indexes that aren't worth coupling to the _SCHEMA string — idempotent.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_querylog_account_created "
            "ON query_log(account_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_querylog_portal_user "
            "ON query_log(portal_user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_llmcall_account_created "
            "ON llm_call_log(account_id, created_at DESC)"
        )
    _run_migrations()
    log.info("Database initialised at %s", DB_PATH)


def _run_migrations() -> None:
    """
    Idempotent column migrations — adds new columns to existing tables.
    Each migration checks if the column exists before attempting ALTER TABLE.
    Safe to run on every startup — skips columns that already exist.
    """
    migrations = [
        # v10: chat UI toggle per client
        ("client", "chat_ui_enabled", "INTEGER NOT NULL DEFAULT 0"),
        # v11: per-client LLM audit toggle
        ("client", "enable_llm_audit", "INTEGER NOT NULL DEFAULT 0"),
        # v12: parent question grouping on audit log
        ("llm_call_log", "question_id", "TEXT NOT NULL DEFAULT ''"),
        # v13: web-portal-only clients (no chat platform required)
        ("client", "portal_only", "INTEGER NOT NULL DEFAULT 0"),
        # v14: optional monthly token budget per client; 0 = unlimited
        ("client", "token_limit_monthly", "INTEGER DEFAULT 0"),
        # v15: metric registry formula builder metadata
        ("metric_registry", "formula_type", "TEXT NOT NULL DEFAULT 'query'"),
        ("metric_registry", "result_format", "TEXT NOT NULL DEFAULT 'number'"),
        ("metric_registry", "required_columns", "TEXT DEFAULT ''"),
        ("metric_registry", "allowed_dimensions", "TEXT DEFAULT ''"),
        ("metric_registry", "example_questions", "TEXT DEFAULT ''"),
        ("metric_registry", "grain", "TEXT DEFAULT ''"),
        # v16: egress log watermark on external export state
        ("external_log_export_state", "last_egress_id",    "INTEGER DEFAULT 0"),
        ("external_log_export_state", "last_egress_count", "INTEGER DEFAULT 0"),
        # v17: visual canvas positions and color for entity graph nodes
        ("entity_graph", "pos_x",  "REAL DEFAULT 120"),
        ("entity_graph", "pos_y",  "REAL DEFAULT 120"),
        ("entity_graph", "color",  "TEXT DEFAULT '#4F86C6'"),
        # v18: LLM suggestion confidence + confirmation status
        ("entity_graph",         "confidence_score", "INTEGER DEFAULT 100"),
        ("entity_graph",         "status",           "TEXT DEFAULT 'confirmed'"),
        ("entity_relationships", "confidence_score", "INTEGER DEFAULT 100"),
        ("entity_relationships", "status",           "TEXT DEFAULT 'confirmed'"),
        ("entity_properties",    "confidence_score", "INTEGER DEFAULT 100"),
        ("entity_properties",    "status",           "TEXT DEFAULT 'confirmed'"),
        # v19: egress log — per-table detail of what actually reached the LLM
        ("kb_data_egress_log", "fields_sent",    "TEXT DEFAULT '[]'"),
        ("kb_data_egress_log", "row_count_sent", "INTEGER DEFAULT 0"),
        ("kb_data_egress_log", "masked_fields",  "TEXT DEFAULT '[]'"),
        ("kb_data_egress_log", "mask_mode",      "TEXT DEFAULT 'none'"),
        # v20: compound (multi-column) join conditions
        ("entity_relationships", "join_conditions", "TEXT NOT NULL DEFAULT '[]'"),
        # v21: per-field masking replacement strategy map
        ("kb_data_egress_log", "mask_replacement_map", "TEXT NOT NULL DEFAULT '{}'"),
    ]
    with get_db() as conn:
        _ensure_llm_call_log_table(conn)
        _ensure_external_log_export_state_table(conn)
        _ensure_semantic_field_feedback_table(conn)
        for table, column, col_def in migrations:
            try:
                existing = [
                    row[1] for row in
                    conn.execute(f"PRAGMA table_info({table})").fetchall()
                ]
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
                    )
                    log.info("Migration: added %s.%s", table, column)
            except Exception as e:
                log.debug("Migration skip %s.%s: %s", table, column, e)
        # Seed llm_pricing from hardcoded defaults — only inserts rows that
        # don't already exist so admin edits are never overwritten on restart.
        try:
            from store.config_store import LLM_COST_RATES
            for model, rates in LLM_COST_RATES.items():
                conn.execute(
                    """INSERT OR IGNORE INTO llm_pricing (model, tokens_in, tokens_out)
                       VALUES (?, ?, ?)""",
                    (model, rates["in"], rates["out"]),
                )
        except Exception as e:
            log.debug("llm_pricing seed skipped: %s", e)


def _ensure_llm_call_log_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS llm_call_log (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id                TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            question_id               TEXT    NOT NULL DEFAULT '',
            request_id                TEXT    NOT NULL,
            question                  TEXT    DEFAULT '',
            component                 TEXT    NOT NULL DEFAULT 'general',
            llm_provider              TEXT    DEFAULT '',
            llm_model                 TEXT    DEFAULT '',
            status                    TEXT    NOT NULL DEFAULT 'success',
            payload_hash              TEXT    DEFAULT '',
            payload_preview_sanitized TEXT    DEFAULT '',
            prompt_chars              INTEGER DEFAULT 0,
            error_msg                 TEXT    DEFAULT '',
            created_at                TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_account  ON llm_call_log(account_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_question ON llm_call_log(question_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_request  ON llm_call_log(request_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_created  ON llm_call_log(created_at);
        """
    )


def _ensure_semantic_field_feedback_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS semantic_field_feedback (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id           TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id       INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
            table_fqn            TEXT    NOT NULL,
            schema_name          TEXT    DEFAULT '',
            table_name           TEXT    NOT NULL,
            column_name          TEXT    NOT NULL,
            current_meaning      TEXT    DEFAULT '',
            current_use_case     TEXT    DEFAULT '',
            suggested_meaning    TEXT    DEFAULT '',
            suggested_use_case   TEXT    DEFAULT '',
            user_comment         TEXT    DEFAULT '',
            confidence_score     INTEGER DEFAULT 0,
            status               TEXT    NOT NULL DEFAULT 'pending'
                                 CHECK(status IN ('pending','approved','rejected')),
            admin_note           TEXT    DEFAULT '',
            created_at           TEXT    DEFAULT (datetime('now')),
            reviewed_at          TEXT    DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_feedback_account_status
            ON semantic_field_feedback(account_id, status, created_at);
        CREATE INDEX IF NOT EXISTS idx_semantic_feedback_field
            ON semantic_field_feedback(account_id, table_fqn, column_name, status);
        """
    )


def _ensure_external_log_export_state_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS external_log_export_state (
            db_config_id     INTEGER PRIMARY KEY REFERENCES db_config(id) ON DELETE CASCADE,
            last_run_date    TEXT    DEFAULT '',
            last_started_at  TEXT    DEFAULT '',
            last_finished_at TEXT    DEFAULT '',
            last_status      TEXT    DEFAULT '',
            last_message     TEXT    DEFAULT '',
            last_query_id    INTEGER DEFAULT 0,
            last_llm_id      INTEGER DEFAULT 0,
            last_query_count INTEGER DEFAULT 0,
            last_llm_count   INTEGER DEFAULT 0,
            updated_at       TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_external_log_export_status
            ON external_log_export_state(last_status);
        """
    )
