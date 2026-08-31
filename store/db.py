"""
store/db.py

Database schema, migrations, and connection management for QueryBot v2.

Connection is provided by store.database (SQLite by default, PostgreSQL when
DATABASE_URL env var is set). All caller code uses the same import:

    from store.db import get_db
    with get_db() as conn:
        conn.execute(...)

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

# Re-export the unified adapter so all callers (from store.db import get_db)
# continue to work without any changes.
from store.database import (  # noqa: F401
    get_connection,
    get_db,
    get_table_columns,
    is_postgres,
    DB_PATH,
    DATABASE_URL,
)

log = logging.getLogger("querybot.db")

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
    enable_python_analysis INTEGER NOT NULL DEFAULT 0,
    allow_user_python   INTEGER NOT NULL DEFAULT 0,
    sql_accuracy_target_pct INTEGER NOT NULL DEFAULT 85,
    created_at          TEXT    DEFAULT (datetime('now')),
    updated_at          TEXT    DEFAULT (datetime('now'))
);

-- ── Query log (usage tracking + billing) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id         TEXT    REFERENCES client(account_id) ON DELETE CASCADE,
    portal_user_id     INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    zoom_user_id       TEXT    DEFAULT '',
    question           TEXT,
    sql_generated      TEXT,
    row_count          INTEGER DEFAULT 0,
    success            INTEGER NOT NULL DEFAULT 1,
    error_msg          TEXT    DEFAULT '',
    llm_provider       TEXT    DEFAULT '',
    llm_model          TEXT    DEFAULT '',
    tokens_in          INTEGER DEFAULT 0,
    tokens_out         INTEGER DEFAULT 0,
    cost_usd           REAL    DEFAULT 0.0,
    duration_ms        INTEGER DEFAULT 0,
    question_id        TEXT    DEFAULT '',
    parent_question_id TEXT    DEFAULT '',
    created_at         TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS answer_trace (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id             TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    portal_user_id          INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    platform_user_id        TEXT    DEFAULT '',
    session_id              TEXT    DEFAULT '',
    question_id             TEXT    DEFAULT '',
    parent_question_id      TEXT    DEFAULT '',
    question_text_sanitized TEXT    DEFAULT '',
    request_source          TEXT    DEFAULT '',
    route                   TEXT    DEFAULT '',
    selected_schema         TEXT    DEFAULT '',
    allowed_tables_snapshot TEXT    DEFAULT '[]',
    retrieved_kb_chunk_ids  TEXT    DEFAULT '[]',
    retrieved_kb_scores     TEXT    DEFAULT '[]',
    llm_provider            TEXT    DEFAULT '',
    llm_model               TEXT    DEFAULT '',
    prompt_tokens           INTEGER DEFAULT 0,
    completion_tokens       INTEGER DEFAULT 0,
    generated_sql           TEXT    DEFAULT '',
    sql_validation_status   TEXT    DEFAULT '',
    sql_validation_error    TEXT    DEFAULT '',
    db_type                 TEXT    DEFAULT '',
    query_row_count         INTEGER DEFAULT 0,
    query_duration_ms       INTEGER DEFAULT 0,
    answer_type             TEXT    DEFAULT '',
    final_answer_summary    TEXT    DEFAULT '',
    error_message           TEXT    DEFAULT '',
    status                  TEXT    DEFAULT 'started',
    created_at              TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS answer_trace_step (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id       INTEGER NOT NULL REFERENCES answer_trace(id) ON DELETE CASCADE,
    step_order     INTEGER NOT NULL DEFAULT 0,
    step_name      TEXT    NOT NULL,
    input_summary  TEXT    DEFAULT '',
    output_summary TEXT    DEFAULT '',
    duration_ms    INTEGER DEFAULT 0,
    status         TEXT    DEFAULT 'success',
    metadata_json  TEXT    DEFAULT '{}',
    created_at     TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_run (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    schema_name     TEXT    DEFAULT '',
    case_file       TEXT    DEFAULT '',
    total_cases     INTEGER DEFAULT 0,
    passed_cases    INTEGER DEFAULT 0,
    avg_score       REAL    DEFAULT 0.0,
    status          TEXT    DEFAULT 'completed',
    report_path     TEXT    DEFAULT '',
    created_at      TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS eval_case_result (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id        INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    case_id            TEXT    NOT NULL,
    question           TEXT    NOT NULL,
    score              REAL    DEFAULT 0.0,
    passed             INTEGER DEFAULT 0,
    generated_sql      TEXT    DEFAULT '',
    validation_status  TEXT    DEFAULT '',
    validation_error   TEXT    DEFAULT '',
    execution_status   TEXT    DEFAULT '',
    row_count          INTEGER DEFAULT 0,
    failures_json      TEXT    DEFAULT '[]',
    created_at         TEXT    DEFAULT (datetime('now'))
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
    response_hash             TEXT    DEFAULT '',
    response_preview_sanitized TEXT   DEFAULT '',
    response_chars            INTEGER DEFAULT 0,
    error_msg                 TEXT    DEFAULT '',
    -- Structured statement of WHAT went to the model on this call: tables,
    -- columns, content categories, and a computed values_sent flag. The
    -- sanitized preview is good forensics but cannot answer "were any data
    -- values sent?" without a human reading it; this can. See core/llm_audit.py.
    egress_manifest           TEXT    NOT NULL DEFAULT '',
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

-- ── Pending platform users (admin-approval flow, no link required) ──────────
-- platform_type    : "teams" | "zoom" | "slack"
-- platform_user_id : raw user ID from the platform (Teams: from.id, etc.)
-- display_name     : display name from the platform activity
-- conversation_ref : JSON blob with service_url + conversation_id for proactive msg
-- status           : pending | approved | rejected
CREATE TABLE IF NOT EXISTS pending_platform_user (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id         TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    platform_type      TEXT    NOT NULL DEFAULT 'teams',
    platform_user_id   TEXT    NOT NULL,
    display_name       TEXT    NOT NULL DEFAULT '',
    conversation_ref   TEXT    NOT NULL DEFAULT '{}',
    status             TEXT    NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','approved','rejected')),
    portal_user_id     INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    reviewer_id        TEXT    DEFAULT NULL,
    reviewer_note      TEXT    DEFAULT '',
    created_at         TEXT    DEFAULT (datetime('now')),
    reviewed_at        TEXT    DEFAULT NULL,
    UNIQUE(account_id, platform_user_id)
);
CREATE INDEX IF NOT EXISTS idx_pending_platform_user_account ON pending_platform_user(account_id, status);

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
    grid_x       INTEGER NOT NULL DEFAULT 0,
    grid_y       INTEGER NOT NULL DEFAULT 0,
    grid_w       INTEGER NOT NULL DEFAULT 6,
    grid_h       INTEGER NOT NULL DEFAULT 5,
    created_at   TEXT    DEFAULT (datetime('now')),
    last_refreshed TEXT  DEFAULT (datetime('now'))
);

-- Named dashboard drafts created conversationally in the portal. Result rows
-- are never stored here; items reference governed pinned SQL instead.
CREATE TABLE IF NOT EXISTS dashboard_artifact (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    thread_id    TEXT    NOT NULL DEFAULT 'default',
    name         TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'draft',
    visibility   TEXT    NOT NULL DEFAULT 'personal',
    refresh_schedule TEXT NOT NULL DEFAULT 'manual',
    filters_json TEXT    NOT NULL DEFAULT '[]',
    tabs_json    TEXT    NOT NULL DEFAULT '["Overview"]',
    version      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now')),
    published_at TEXT    DEFAULT NULL
);

-- Reusable governed dashboard sources. SQL is retained for live refresh, but
-- result rows and database credentials are deliberately never stored here.
CREATE TABLE IF NOT EXISTS dashboard_data_source (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dashboard_id INTEGER NOT NULL REFERENCES dashboard_artifact(id) ON DELETE CASCADE,
    account_id   TEXT    NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    source_type  TEXT    NOT NULL DEFAULT 'governed_sql',
    question     TEXT    NOT NULL DEFAULT '',
    sql_query    TEXT    NOT NULL,
    db_config_id INTEGER REFERENCES db_config(id) ON DELETE SET NULL,
    semantic_contract_version TEXT NOT NULL DEFAULT '',
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now'))
);

-- Encrypted, policy-scoped materialization for scheduled dashboard refresh.
-- The token contains only rows already released through governed_query for
-- this exact user. Credentials and unprotected source rows are never cached.
CREATE TABLE IF NOT EXISTS dashboard_source_cache (
    source_id     INTEGER PRIMARY KEY REFERENCES dashboard_data_source(id) ON DELETE CASCADE,
    dashboard_id  INTEGER NOT NULL REFERENCES dashboard_artifact(id) ON DELETE CASCADE,
    account_id    TEXT    NOT NULL,
    user_id       INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    rows_encrypted TEXT   NOT NULL,
    row_count     INTEGER NOT NULL DEFAULT 0,
    policy_version INTEGER NOT NULL DEFAULT 0,
    contract_version TEXT NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'ready',
    error_message TEXT    NOT NULL DEFAULT '',
    refreshed_at  TEXT    DEFAULT (datetime('now')),
    expires_at    TEXT    NOT NULL
);

-- Presentation-only checkpoints. Snapshots contain dashboard controls and
-- chart/source references, never rows or credentials.
CREATE TABLE IF NOT EXISTS dashboard_artifact_version (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dashboard_id INTEGER NOT NULL REFERENCES dashboard_artifact(id) ON DELETE CASCADE,
    account_id   TEXT    NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    change_summary TEXT  NOT NULL DEFAULT '',
    snapshot_json TEXT   NOT NULL DEFAULT '{}',
    created_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(dashboard_id, version)
);

-- Users follow named dashboards, not individual charts.  A subscription is
-- deliberately separate from refresh_schedule: refresh controls when the
-- live dashboard data is recomputed, while this row records who follows the
-- artifact and how they want to be notified.
CREATE TABLE IF NOT EXISTS dashboard_subscription (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dashboard_id INTEGER NOT NULL REFERENCES dashboard_artifact(id) ON DELETE CASCADE,
    account_id   TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    cadence      TEXT    NOT NULL DEFAULT 'daily'
                         CHECK(cadence IN ('daily','weekly','monthly')),
    channel      TEXT    NOT NULL DEFAULT 'in_app'
                         CHECK(channel IN ('in_app','email')),
    status       TEXT    NOT NULL DEFAULT 'active'
                         CHECK(status IN ('active','paused')),
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(dashboard_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_portal_user_account  ON portal_user(account_id);
CREATE INDEX IF NOT EXISTS idx_portal_user_zoom     ON portal_user(zoom_user_id);
CREATE INDEX IF NOT EXISTS idx_group_account        ON user_group(account_id);
CREATE INDEX IF NOT EXISTS idx_reg_token            ON registration_token(token);
CREATE INDEX IF NOT EXISTS idx_pinned_user          ON pinned_chart(user_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_owner      ON dashboard_artifact(account_id, user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_thread     ON dashboard_artifact(account_id, user_id, thread_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_source     ON dashboard_data_source(dashboard_id, user_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_cache_due  ON dashboard_source_cache(account_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_dashboard_version    ON dashboard_artifact_version(dashboard_id, version DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_subscription ON dashboard_subscription(account_id, user_id, status);

CREATE INDEX IF NOT EXISTS idx_query_log_account ON query_log(account_id);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_answer_trace_account ON answer_trace(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_answer_trace_question ON answer_trace(question_id);
CREATE INDEX IF NOT EXISTS idx_answer_trace_step_trace ON answer_trace_step(trace_id, step_order);
CREATE INDEX IF NOT EXISTS idx_eval_run_account ON eval_run(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_case_run ON eval_case_result(eval_run_id);
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
    metric_builder_config TEXT DEFAULT '',
    example_questions TEXT DEFAULT '',
    grain        TEXT    DEFAULT '',
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    DEFAULT (datetime('now')),
    updated_at   TEXT    DEFAULT (datetime('now'))
);

-- Governed metric/context -> role-playing date bindings. A metric can use
-- different business dates for different analytical contexts while keeping
-- one measure expression (for example Revenue by invoice date vs inventory
-- sales by accounting date).
CREATE TABLE IF NOT EXISTS metric_date_context (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    metric_id       INTEGER NOT NULL REFERENCES metric_registry(id) ON DELETE CASCADE,
    context_name    TEXT    NOT NULL,
    aliases         TEXT    NOT NULL DEFAULT '',
    date_role       TEXT    NOT NULL,
    fact_table      TEXT    NOT NULL,
    fact_column     TEXT    NOT NULL,
    dimension_table TEXT    NOT NULL,
    dimension_key   TEXT    NOT NULL,
    date_value_column TEXT  NOT NULL DEFAULT '',
    date_key_type   TEXT    NOT NULL DEFAULT 'surrogate_fk',
    is_default      INTEGER NOT NULL DEFAULT 0,
    priority        INTEGER NOT NULL DEFAULT 50,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    DEFAULT (datetime('now')),
    updated_at      TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, metric_id, context_name)
);

-- ── Reports (named, multi-metric, admin-defined) ──────────────────────────────
-- A report is an admin-defined named group of metric_registry entries a user
-- can ask about live in chat ("what's today's report?") or subscribe to on a
-- schedule. Access to each metric is still checked per-user at run time via
-- metric_registry.base_table + store.get_allowed_tables() — the report itself
-- grants no extra access.
CREATE TABLE IF NOT EXISTS report (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id  TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    is_default  INTEGER NOT NULL DEFAULT 0,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    DEFAULT (datetime('now')),
    updated_at  TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, name)
);

-- Many-to-many: which metrics belong to a report, and in what display order.
CREATE TABLE IF NOT EXISTS report_metric (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL REFERENCES report(id) ON DELETE CASCADE,
    metric_id   INTEGER NOT NULL REFERENCES metric_registry(id) ON DELETE CASCADE,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    UNIQUE(report_id, metric_id)
);

-- A user's subscription to a report's scheduled digest.
-- cadence: 'daily' | 'weekly'. day_of_week: 0=Monday..6=Sunday, only used
-- for 'weekly'. hour: 0-23, server local time (matches log_export's own
-- server-local time convention).
CREATE TABLE IF NOT EXISTS report_subscription (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    user_id      INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    report_id    INTEGER NOT NULL REFERENCES report(id) ON DELETE CASCADE,
    cadence      TEXT    NOT NULL DEFAULT 'daily' CHECK(cadence IN ('daily','weekly')),
    day_of_week  INTEGER NOT NULL DEFAULT 0,
    hour         INTEGER NOT NULL DEFAULT 8,
    last_sent    TEXT    DEFAULT NULL,
    status       TEXT    NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused')),
    created_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(user_id, report_id)
);
CREATE INDEX IF NOT EXISTS idx_report_account ON report(account_id);
CREATE INDEX IF NOT EXISTS idx_report_metric_report ON report_metric(report_id);
CREATE INDEX IF NOT EXISTS idx_report_subscription_account ON report_subscription(account_id, status);

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
CREATE INDEX IF NOT EXISTS idx_metric_date_context_account
    ON metric_date_context(account_id, metric_id, is_active);
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
    -- v38: correlation id linking this egress row to the llm_call_log rows of
    -- the same KB build / discovery run. Without it the two logs could only be
    -- matched by table name and timestamp, by hand.
    request_id      TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kb_egress_account_op
    ON kb_data_egress_log(account_id, operation, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kb_egress_account_table
    ON kb_data_egress_log(account_id, table_name);

-- ── Per-table business description ───────────────────────────────────────────
-- What this table is FOR, in the admin's words, for the tables they selected.
--
-- Before this, core/knowledge._build_table_business_desc had two levels: one
-- description for the whole client, plus optional per-SCHEMA blocks. Every
-- table in a schema therefore received byte-identical business context in its
-- KB prompt, so the one sentence worth writing about a specific table -- "this
-- is the month-end stock snapshot, one row per item per warehouse" -- had
-- nowhere to go.
--
-- Keyed on the table rather than on entity_graph.entity_name deliberately: a
-- selected table does not necessarily have a graph entity, and the admin
-- selects tables long before the graph exists.
--
-- `synonyms` is separate from `description` and is NOT prose. Source
-- resolution never reads descriptions or KB text -- it matches on names,
-- labels and synonyms (core/source_resolution._table_aliases) -- so prose
-- alone improves the KB and changes nothing about which table gets picked.
-- These are the terms that do.
CREATE TABLE IF NOT EXISTS table_description (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT    NOT NULL,
    table_name   TEXT    NOT NULL,          -- as selected: SCHEMA.TABLE
    description  TEXT    NOT NULL DEFAULT '',
    synonyms     TEXT    NOT NULL DEFAULT '',  -- comma-separated business terms
    updated_by   TEXT    NOT NULL DEFAULT '',
    updated_at   TEXT    DEFAULT (datetime('now')),
    UNIQUE(account_id, table_name)
);
CREATE INDEX IF NOT EXISTS idx_table_description_account
    ON table_description(account_id);

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
    ,relationship_key TEXT    NOT NULL DEFAULT ''              -- stable multi-edge identity
    ,constraint_name  TEXT    NOT NULL DEFAULT ''              -- source DB constraint, when present
    ,source_enforced  INTEGER NOT NULL DEFAULT 0               -- DB guarantees referential integrity
    ,optionality      TEXT    NOT NULL DEFAULT 'unknown'       -- required|optional|unknown
    ,match_rate       REAL    NOT NULL DEFAULT -1
    ,orphan_rate      REAL    NOT NULL DEFAULT -1
    ,null_fk_rate     REAL    NOT NULL DEFAULT -1
    ,fanout_ratio     REAL    NOT NULL DEFAULT -1
    ,last_profiled_at TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_entity_rel_account
    ON entity_relationships(account_id, is_active);

-- ── Entity properties (column roles) ────────────────��─────────────────────────
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

-- Point-in-time snapshots of the whole entity graph (entities +
-- relationships + properties as one JSON blob). Written on demand from the
-- graph Tools menu and automatically before destructive operations
-- (import, restore), so a bad import is always one restore away from undone.
CREATE TABLE IF NOT EXISTS graph_version (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id    TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    label         TEXT    NOT NULL DEFAULT '',
    created_by    TEXT    NOT NULL DEFAULT 'admin',
    snapshot_json TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    DEFAULT (datetime('now'))
);

-- Conversational graph mutations that must not alter confirmed semantics
-- until an administrator reviews the before/after diff.
CREATE TABLE IF NOT EXISTS graph_change_proposal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id      TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    action          TEXT    NOT NULL,
    target_kind     TEXT    NOT NULL,
    target_id       TEXT    NOT NULL DEFAULT '',
    before_json     TEXT    NOT NULL DEFAULT '{}',
    payload_json    TEXT    NOT NULL DEFAULT '{}',
    status          TEXT    NOT NULL DEFAULT 'pending',
    confidence_score INTEGER NOT NULL DEFAULT 0,
    generated_by    TEXT    NOT NULL DEFAULT 'chat',
    reason          TEXT    NOT NULL DEFAULT '',
    reviewed_by     TEXT    NOT NULL DEFAULT '',
    reviewed_at     TEXT    DEFAULT NULL,
    created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_graph_change_proposal_account
    ON graph_change_proposal(account_id, status, created_at);

-- ── Metric proposals — a metric someone asked for but nobody has approved ─────
-- Same lifecycle as graph_change_proposal, deliberately its OWN table rather
-- than a graft onto it: that one's target_kind/target_id are consumed by
-- graph-specific accept logic and its rows are surfaced inside get_full_graph,
-- so metrics riding along would pollute the graph payload and its review UI.
--
-- payload_json is a metric_registry-shaped dict. validation_json and dryrun_json
-- carry the evidence gathered when the proposal was made — the parse result and
-- the live probe — so a reviewer sees whether it actually runs without having to
-- re-test it. dryrun_json never includes the probe's scalar value; that is real
-- data which never crossed the compliance boundary.
--
-- requested_by_user_id is the portal user who asked. reviewed_by is always
-- 'admin': admin auth is a single shared password with no identity, the same
-- known limitation graph_change_proposal already carries.
CREATE TABLE IF NOT EXISTS metric_proposal (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id           TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    action               TEXT    NOT NULL DEFAULT 'create_metric',
    target_metric_id     INTEGER DEFAULT NULL,
    before_json          TEXT    NOT NULL DEFAULT '{}',
    payload_json         TEXT    NOT NULL DEFAULT '{}',
    status               TEXT    NOT NULL DEFAULT 'pending',
    confidence_score     INTEGER NOT NULL DEFAULT 0,
    generated_by         TEXT    NOT NULL DEFAULT 'portal_chat',
    requested_by_user_id INTEGER DEFAULT NULL,
    request_text         TEXT    NOT NULL DEFAULT '',
    source_question      TEXT    NOT NULL DEFAULT '',
    validation_json      TEXT    NOT NULL DEFAULT '{}',
    dryrun_json          TEXT    NOT NULL DEFAULT '{}',
    reason               TEXT    NOT NULL DEFAULT '',
    reviewed_by          TEXT    NOT NULL DEFAULT '',
    reviewed_at          TEXT    DEFAULT NULL,
    review_note          TEXT    NOT NULL DEFAULT '',
    created_at           TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_metric_proposal_account
    ON metric_proposal(account_id, status, created_at);

-- ── Session metric drafts — logic a user composed, live only in their thread ──
-- The first of the two tracks: a user describes a calculation, it runs and
-- answers their question immediately, and it stays usable for follow-ups in the
-- same thread. It is NEVER written to metric_registry, so it changes nobody
-- else's answers; becoming shared requires a metric_proposal and a human.
--
-- Not conversation_state (its docstring forbids values and it expires in 30
-- minutes) and not pending_clarification_thread (5 minutes, one row per thread,
-- single-choice). A draft needs its own lifetime and its own shape.
--
-- source_tables is re-checked against the user's ACL on EVERY read, so a table
-- revoked mid-thread kills the draft rather than leaving a stale grant behind.
CREATE TABLE IF NOT EXISTS session_metric_draft (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id            TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    session_id            TEXT    NOT NULL,
    portal_user_id        INTEGER NOT NULL,
    name                  TEXT    NOT NULL,
    synonyms              TEXT    NOT NULL DEFAULT '',
    description           TEXT    NOT NULL DEFAULT '',
    sql_template          TEXT    NOT NULL,
    formula_type          TEXT    NOT NULL DEFAULT 'expression',
    result_format         TEXT    NOT NULL DEFAULT 'number',
    base_table            TEXT    NOT NULL DEFAULT '',
    required_columns      TEXT    NOT NULL DEFAULT '',
    allowed_dimensions    TEXT    NOT NULL DEFAULT '',
    default_time_column   TEXT    NOT NULL DEFAULT '',
    metric_builder_config TEXT    NOT NULL DEFAULT '',
    source_tables         TEXT    NOT NULL DEFAULT '[]',
    validation_json       TEXT    NOT NULL DEFAULT '{}',
    dryrun_json           TEXT    NOT NULL DEFAULT '{}',
    confidence            REAL    NOT NULL DEFAULT 0.0,
    source_question       TEXT    NOT NULL DEFAULT '',
    origin                TEXT    NOT NULL DEFAULT 'portal_chat',
    status                TEXT    NOT NULL DEFAULT 'active',
    proposal_id           INTEGER DEFAULT NULL,
    created_at            TEXT    DEFAULT (datetime('now')),
    expires_at            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_metric_draft_thread
    ON session_metric_draft(account_id, session_id, status, expires_at);

-- =============================================================================
-- Learning loop tables (v30)
-- =============================================================================

-- ── Answer feedback — thumbs-up / down on every completed answer ──────────────
-- question_id  : = audit_request_id (public key, exposed in trust block)
-- rating       : 1 = up, -1 = down
-- reason_code  : wrong_metric | wrong_dimension | wrong_filter | wrong_join |
--                wrong_data | incomplete | confusing | expected_data_missing | other
-- question_text / sql_text: denormalized from trace for fast admin queue display
CREATE TABLE IF NOT EXISTS answer_feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id   TEXT    NOT NULL,
    user_id       INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
    account_id    TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    schema_scope  TEXT    NOT NULL DEFAULT '',
    rating        INTEGER NOT NULL CHECK(rating IN (-1, 1)),
    reason_code   TEXT    NOT NULL DEFAULT '',
    comment       TEXT    NOT NULL DEFAULT '',
    question_text TEXT    NOT NULL DEFAULT '',
    sql_text      TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    DEFAULT (datetime('now')),
    updated_at    TEXT    DEFAULT (datetime('now')),
    UNIQUE(question_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_answer_feedback_question
    ON answer_feedback(question_id);
CREATE INDEX IF NOT EXISTS idx_answer_feedback_account_status
    ON answer_feedback(account_id, rating, created_at DESC);

-- ── Learning candidate — one row per answer considered for memory ─────────────
-- candidate_type  : positive | review | negative | regression
-- status          : pending_review | approved | rejected | known_failure | revoked
-- source          : auto | admin_correction | pre_governed
-- technical_score : sum of dimension scores (0–100, no feedback)
-- feedback_delta  : net feedback adjustment (+10 or -30)
-- final_score     : technical_score + feedback_delta (clamped to 0+)
-- evidence        : JSON breakdown of how each dimension scored
CREATE TABLE IF NOT EXISTS learning_candidate (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id           TEXT    NOT NULL UNIQUE,
    origin_question_id     TEXT    NOT NULL DEFAULT '',
    account_id             TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    schema_scope           TEXT    NOT NULL DEFAULT '',
    candidate_type         TEXT    NOT NULL DEFAULT 'review'
                           CHECK(candidate_type IN ('positive','review','negative','regression')),
    question_text          TEXT    NOT NULL DEFAULT '',
    sql_text               TEXT    NOT NULL DEFAULT '',
    corrected_sql          TEXT    NOT NULL DEFAULT '',
    technical_score        INTEGER NOT NULL DEFAULT 0,
    feedback_delta         INTEGER NOT NULL DEFAULT 0,
    final_score            INTEGER NOT NULL DEFAULT 0,
    evidence               TEXT    NOT NULL DEFAULT '{}',
    positive_vote_count    INTEGER NOT NULL DEFAULT 0,
    negative_vote_count    INTEGER NOT NULL DEFAULT 0,
    status                 TEXT    NOT NULL DEFAULT 'pending_review'
                           CHECK(status IN ('pending_review','approved','rejected','known_failure','revoked')),
    source                 TEXT    NOT NULL DEFAULT 'auto'
                           CHECK(source IN ('auto','admin_correction','pre_governed')),
    semantic_model_version TEXT    NOT NULL DEFAULT '',
    metric_version         TEXT    NOT NULL DEFAULT '',
    schema_version         TEXT    NOT NULL DEFAULT '',
    qdrant_id              TEXT    NOT NULL DEFAULT '',
    reviewer_id            TEXT    NOT NULL DEFAULT '',
    reviewer_note          TEXT    NOT NULL DEFAULT '',
    reviewed_at            TEXT    DEFAULT NULL,
    promoted_at            TEXT    DEFAULT NULL,
    created_at             TEXT    DEFAULT (datetime('now')),
    updated_at             TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_learning_candidate_account_status
    ON learning_candidate(account_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_learning_candidate_question
    ON learning_candidate(origin_question_id);
CREATE INDEX IF NOT EXISTS idx_learning_candidate_type
    ON learning_candidate(account_id, candidate_type, status);

-- ── Recommendation event — tracks suggestion impressions and outcomes ──────────
-- event_type : displayed | clicked | executed | successful | dismissed
-- suggestion_source : genie | learned | static
CREATE TABLE IF NOT EXISTS recommendation_event (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT    NOT NULL DEFAULT '',
    user_id            INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
    account_id         TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
    event_type         TEXT    NOT NULL
                       CHECK(event_type IN ('displayed','clicked','executed','successful','dismissed')),
    suggestion_text    TEXT    NOT NULL DEFAULT '',
    suggestion_source  TEXT    NOT NULL DEFAULT '',
    result_question_id TEXT    NOT NULL DEFAULT '',
    created_at         TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_recommendation_event_account
    ON recommendation_event(account_id, event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recommendation_event_session
    ON recommendation_event(session_id);
"""


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
    _migrate_legacy_examples_to_candidates()
    _backfill_compliance_profiles()
    if is_postgres():
        log.info("Database initialised (PostgreSQL: %s)", DATABASE_URL.split("@")[-1])
    else:
        log.info("Database initialised (SQLite: %s)", DB_PATH)


def _backfill_compliance_profiles() -> None:
    """
    Give every pre-existing client an explicit `standard` compliance profile.

    core.compliance.policy_engine.is_regulated fails closed when an account has
    no compliance_profile row, so that a workspace which never went through
    compliance setup cannot silently run with the loosest posture. That is only
    a safe rule once "no row" means genuinely new — before this backfill it also
    meant "created before compliance profiles existed", and treating those as
    regulated would abruptly disable result narration for every legacy tenant.

    Writes the same values get_compliance_profile already synthesizes, so the
    effective posture of an existing client does not change; only the ability to
    distinguish legacy from unprovisioned does. Idempotent — inserts nothing for
    accounts that already have a row.
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT c.account_id FROM client c "
                "LEFT JOIN compliance_profile p ON p.account_id = c.account_id "
                "WHERE p.account_id IS NULL"
            ).fetchall()
            if not rows:
                return
            for row in rows:
                conn.execute(
                    "INSERT INTO compliance_profile "
                    "(account_id, industry, mode, enforcement_mode, lifecycle_state) "
                    "VALUES (?, 'standard', 'standard', 'shadow', 'DRAFT')",
                    (row[0],),
                )
        log.info(
            "Migration: backfilled %d standard compliance profile(s) for "
            "pre-existing clients", len(rows)
        )
    except Exception as exc:  # noqa: BLE001 — never block startup on a backfill
        log.warning("Compliance profile backfill skipped: %s", exc)


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
        ("metric_registry", "metric_builder_config", "TEXT DEFAULT ''"),
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
        # v22: metric registry — category/domain and default time column
        ("metric_registry", "category",             "TEXT DEFAULT ''"),
        ("metric_registry", "default_time_column",  "TEXT DEFAULT ''"),
        # v23: WHERE conditions always applied to this join
        ("entity_relationships", "where_clause", "TEXT NOT NULL DEFAULT ''"),
        # v24: static entity-level filter applied whenever this table is joined
        ("entity_graph", "entity_filter", "TEXT NOT NULL DEFAULT ''"),
        # v25: user-chosen colour palette for each pinned chart
        ("pinned_chart", "color_palette", "TEXT NOT NULL DEFAULT 'default'"),
        # v25b: persisted dashboard layout for draggable/resizable chart cards
        ("pinned_chart", "grid_x", "INTEGER NOT NULL DEFAULT 0"),
        ("pinned_chart", "grid_y", "INTEGER NOT NULL DEFAULT 0"),
        ("pinned_chart", "grid_w", "INTEGER NOT NULL DEFAULT 6"),
        ("pinned_chart", "grid_h", "INTEGER NOT NULL DEFAULT 5"),
        ("pinned_chart", "dashboard_id", "INTEGER DEFAULT NULL"),
        ("pinned_chart", "display_config", "TEXT NOT NULL DEFAULT '{}'"),
        # v41: Ana-style reusable dashboard artifacts and presentation controls.
        ("pinned_chart", "data_source_id", "INTEGER DEFAULT NULL"),
        ("pinned_chart", "dashboard_tab", "TEXT NOT NULL DEFAULT 'Overview'"),
        ("pinned_chart", "sort_enabled", "INTEGER NOT NULL DEFAULT 1"),
        ("pinned_chart", "layout_locked", "INTEGER NOT NULL DEFAULT 0"),
        ("dashboard_artifact", "visibility", "TEXT NOT NULL DEFAULT 'personal'"),
        ("dashboard_artifact", "refresh_schedule", "TEXT NOT NULL DEFAULT 'manual'"),
        ("dashboard_artifact", "filters_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("dashboard_artifact", "tabs_json", "TEXT NOT NULL DEFAULT '[\"Overview\"]'"),
        ("dashboard_artifact", "last_refreshed_at", "TEXT DEFAULT NULL"),
        # v26: link drill-down result_chat queries to their parent main query
        ("query_log", "question_id",        "TEXT DEFAULT ''"),
        ("query_log", "parent_question_id", "TEXT DEFAULT ''"),
        # v27: semantic layer — metric registry enrichment
        # metric_status: draft | validated | published | deprecated
        # Default 'published' so all existing metrics stay live without change.
        ("metric_registry", "metric_status",      "TEXT NOT NULL DEFAULT 'published'"),
        ("metric_registry", "base_entity",        "TEXT DEFAULT ''"),
        ("metric_registry", "base_table",         "TEXT DEFAULT ''"),
        ("metric_registry", "formula_ast",        "TEXT DEFAULT '{}'"),
        ("metric_registry", "dependencies",       "TEXT DEFAULT '[]'"),
        ("metric_registry", "last_validated_at",  "TEXT DEFAULT ''"),
        ("metric_registry", "validation_errors",  "TEXT DEFAULT '[]'"),
        ("metric_registry", "usage_count",        "INTEGER DEFAULT 0"),
        ("metric_registry", "eval_coverage",      "INTEGER DEFAULT 0"),
        ("metric_registry", "owner",              "TEXT DEFAULT ''"),
        ("metric_registry", "version",            "INTEGER DEFAULT 1"),
        # v28: entity graph / relationships / properties — suggestion audit trail
        ("entity_graph",         "generated_by",       "TEXT DEFAULT 'manual'"),
        ("entity_graph",         "model",               "TEXT DEFAULT ''"),
        ("entity_graph",         "reason",              "TEXT DEFAULT ''"),
        ("entity_graph",         "reviewed_by",         "TEXT DEFAULT ''"),
        ("entity_graph",         "reviewed_at",         "TEXT DEFAULT ''"),
        ("entity_graph",         "rejected_reason",     "TEXT DEFAULT ''"),
        ("entity_relationships", "generated_by",        "TEXT DEFAULT 'manual'"),
        ("entity_relationships", "model",               "TEXT DEFAULT ''"),
        ("entity_relationships", "reason",              "TEXT DEFAULT ''"),
        ("entity_relationships", "reviewed_by",         "TEXT DEFAULT ''"),
        ("entity_relationships", "reviewed_at",         "TEXT DEFAULT ''"),
        ("entity_relationships", "rejected_reason",     "TEXT DEFAULT ''"),
        # v28b: relationship join validation results
        ("entity_relationships", "validation_status",   "TEXT DEFAULT 'untested'"),
        ("entity_relationships", "validated_at",        "TEXT DEFAULT ''"),
        ("entity_relationships", "row_count_estimate",  "INTEGER DEFAULT -1"),
        ("entity_relationships", "join_multiplicity",   "TEXT DEFAULT ''"),
        # v40: governed, multi-edge relationship planning
        ("entity_relationships", "relationship_key", "TEXT NOT NULL DEFAULT ''"),
        ("entity_relationships", "constraint_name",  "TEXT NOT NULL DEFAULT ''"),
        ("entity_relationships", "source_enforced",  "INTEGER NOT NULL DEFAULT 0"),
        ("entity_relationships", "optionality",      "TEXT NOT NULL DEFAULT 'unknown'"),
        ("entity_relationships", "match_rate",       "REAL NOT NULL DEFAULT -1"),
        ("entity_relationships", "orphan_rate",      "REAL NOT NULL DEFAULT -1"),
        ("entity_relationships", "null_fk_rate",     "REAL NOT NULL DEFAULT -1"),
        ("entity_relationships", "fanout_ratio",     "REAL NOT NULL DEFAULT -1"),
        ("entity_relationships", "last_profiled_at", "TEXT NOT NULL DEFAULT ''"),
        ("entity_properties",    "generated_by",        "TEXT DEFAULT 'manual'"),
        ("entity_properties",    "model",               "TEXT DEFAULT ''"),
        ("entity_properties",    "reason",              "TEXT DEFAULT ''"),
        ("entity_properties",    "reviewed_by",         "TEXT DEFAULT ''"),
        ("entity_properties",    "reviewed_at",         "TEXT DEFAULT ''"),
        # v29: answer_trace — metric + compiler traceability
        ("answer_trace", "metric_id",                  "INTEGER DEFAULT NULL"),
        ("answer_trace", "metric_version_at_query",    "INTEGER DEFAULT 0"),
        ("answer_trace", "used_deterministic_compiler","INTEGER DEFAULT 0"),
        ("answer_trace", "compiler_confidence",        "REAL DEFAULT 0.0"),
        # v32: regulated execution stores protected (never raw) rows and policy version.
        ("answer_trace", "result_rows",                "TEXT NOT NULL DEFAULT '[]'"),
        ("answer_trace", "policy_version_at_query",    "INTEGER NOT NULL DEFAULT 0"),
        # v30: self-learning loop — feature flags on client
        ("client", "enable_feedback_collection", "INTEGER NOT NULL DEFAULT 0"),
        ("client", "enable_learned_retrieval",   "INTEGER NOT NULL DEFAULT 0"),
        ("client", "enable_genie_suggestions",   "INTEGER NOT NULL DEFAULT 0"),
        # Governed custom Python is opt-in per tenant. Pasted source is a
        # separate, stricter permission from metadata-only generated plans.
        ("client", "enable_python_analysis", "INTEGER NOT NULL DEFAULT 0"),
        ("client", "allow_user_python",       "INTEGER NOT NULL DEFAULT 0"),
        ("client", "sql_accuracy_target_pct", "INTEGER NOT NULL DEFAULT 85"),
        # v31: entity graph — allow unreviewed (status='suggested') entities and
        # joins to feed SQL generation. Default 1 preserves existing behaviour;
        # set 0 to enforce admin review before the graph affects queries.
        ("client", "graph_use_suggested",        "INTEGER NOT NULL DEFAULT 1"),
        # v30: validated_examples governance columns
        ("validated_examples", "approval_status",          "TEXT NOT NULL DEFAULT 'legacy'"),
        ("validated_examples", "candidate_id",             "TEXT NOT NULL DEFAULT ''"),
        ("validated_examples", "quality_score",            "INTEGER NOT NULL DEFAULT 0"),
        ("validated_examples", "schema_scope",             "TEXT NOT NULL DEFAULT ''"),
        ("validated_examples", "semantic_model_version",   "TEXT NOT NULL DEFAULT ''"),
        # v32: per-client ERP terminology packs — JSON array of pack ids from
        # packs/ (e.g. ["generic_star_schema"]). Empty = builtin vocab only.
        ("client", "erp_packs",                  "TEXT NOT NULL DEFAULT '[]'"),
        # v33: portal users can suggest explicit business terms/synonyms for a
        # field alongside meaning/use case — comma-joined text, parsed the
        # same way as the admin Edit Field form's synonyms textarea.
        ("semantic_field_feedback", "suggested_synonyms", "TEXT NOT NULL DEFAULT ''"),
        # v34: compiled semantic contract — every answer and eval run is
        # stamped with the contract version it ran under so quality can be
        # correlated with exact semantic states.
        ("answer_trace", "contract_version",       "TEXT NOT NULL DEFAULT ''"),
        ("learning_candidate", "contract_version", "TEXT NOT NULL DEFAULT ''"),
        # v34: eval runs record what approval triggered them and whether the
        # pass rate regressed vs the previous run of the same case file.
        ("eval_run", "trigger_label",    "TEXT NOT NULL DEFAULT ''"),
        ("eval_run", "contract_version", "TEXT NOT NULL DEFAULT ''"),
        ("eval_run", "prev_pass_rate",   "REAL DEFAULT NULL"),
        ("eval_run", "regressed",        "INTEGER NOT NULL DEFAULT 0"),
        # Response capture: audit rows previously only proved what was SENT
        # to the LLM (payload hash + preview) — these prove what came BACK.
        ("llm_call_log", "response_hash",              "TEXT DEFAULT ''"),
        ("llm_call_log", "response_preview_sanitized", "TEXT DEFAULT ''"),
        ("llm_call_log", "response_chars",             "INTEGER DEFAULT 0"),
        # v35: per-client Teams tenant mapping — lets the shared Teams bot
        # registration route an incoming Azure AD tenant to the ONE client it
        # belongs to, instead of the "exactly one configured client" auto-map
        # heuristic that breaks as soon as a second client is configured.
        ("client", "teams_tenant_id", "TEXT NOT NULL DEFAULT ''"),
        # Admin query-log/overview surfacing: the validator/execution failure
        # classification code, so the admin UI can render the same
        # business-readable translate_failure() message chat gets instead of
        # a raw technical string, and tag the failure with its exact scenario.
        ("query_log", "error_code", "TEXT NOT NULL DEFAULT ''"),
        # v36: session boundary tracking for personalized greeting.
        # NULL = user has never sent a message; updated to now() on every active
        # message. touch_user_activity() compares this to detect a new session.
        ("portal_user", "last_active_at", "TEXT DEFAULT NULL"),
        # v37: self-service reports — nullable creator FK. NULL = admin-created
        # (existing rows unaffected). Attribution/edit-scoping only -- does not
        # gate visibility; user-created reports join the same account-wide pool
        # as admin-created ones (askable/subscribable by anyone in the account).
        ("report", "created_by_user_id", "INTEGER DEFAULT NULL"),
        # v38: per-call data-egress manifest — the structured answer to "what
        # went to the model on this call, and did any data values go with it".
        ("llm_call_log", "egress_manifest", "TEXT NOT NULL DEFAULT ''"),
        ("kb_data_egress_log", "request_id", "TEXT NOT NULL DEFAULT ''"),
    ]
    with get_db() as conn:
        _ensure_llm_call_log_table(conn)
        _ensure_answer_trace_tables(conn)
        _ensure_agent_runtime_tables(conn)
        _ensure_graph_change_proposal_table(conn)
        _ensure_eval_tables(conn)
        _ensure_external_log_export_state_table(conn)
        _ensure_semantic_field_feedback_table(conn)
        _ensure_metric_version_table(conn)
        _ensure_metric_test_table(conn)
        _ensure_learning_loop_tables(conn)
        _ensure_compliance_tables(conn)
        _ensure_semantic_compiler_tables(conn)
        for table, column, col_def in migrations:
            try:
                # SAVEPOINT per migration: in PostgreSQL a failed statement
                # aborts the whole transaction.  Rolling back to a savepoint
                # recovers cleanly so subsequent migrations still run.
                # SQLite supports SAVEPOINTs too, so this is safe for both.
                conn.execute("SAVEPOINT sp_migration")
                existing = get_table_columns(conn, table)
                if column not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
                    )
                    log.info("Migration: added %s.%s", table, column)
                conn.execute("RELEASE SAVEPOINT sp_migration")
            except Exception as e:
                try:
                    conn.execute("ROLLBACK TO SAVEPOINT sp_migration")
                except Exception:
                    pass
                log.debug("Migration skip %s.%s: %s", table, column, e)
        # Post-migration indexes — created after column migrations so the
        # referenced columns are guaranteed to exist.
        _post_migration_indexes(conn)
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


def _ensure_graph_change_proposal_table(conn: sqlite3.Connection) -> None:
    """Create the review-first queue used by conversational graph edits."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_change_proposal (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id       TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            action           TEXT NOT NULL,
            target_kind      TEXT NOT NULL,
            target_id        TEXT NOT NULL DEFAULT '',
            before_json      TEXT NOT NULL DEFAULT '{}',
            payload_json     TEXT NOT NULL DEFAULT '{}',
            status           TEXT NOT NULL DEFAULT 'pending',
            confidence_score INTEGER NOT NULL DEFAULT 0,
            generated_by     TEXT NOT NULL DEFAULT 'chat',
            reason           TEXT NOT NULL DEFAULT '',
            reviewed_by      TEXT NOT NULL DEFAULT '',
            reviewed_at      TEXT DEFAULT NULL,
            created_at       TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_graph_change_proposal_account
            ON graph_change_proposal(account_id, status, created_at);
        """
    )


def _ensure_semantic_compiler_tables(conn: sqlite3.Connection) -> None:
    """Create the tenant-scoped semantic compiler control-plane tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS business_date_anchor (
            account_id     TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            fact_table     TEXT NOT NULL,
            fact_column    TEXT NOT NULL,
            anchor_value   TEXT NOT NULL DEFAULT '',
            date_column    TEXT NOT NULL DEFAULT '',
            source         TEXT NOT NULL DEFAULT '',
            probe_ms       INTEGER NOT NULL DEFAULT 0,
            resolved_at    TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (account_id, fact_table, fact_column)
        );

        CREATE TABLE IF NOT EXISTS semantic_compiler_state (
            account_id       TEXT PRIMARY KEY REFERENCES client(account_id) ON DELETE CASCADE,
            mode             TEXT NOT NULL DEFAULT 'shadow'
                             CHECK(mode IN ('off','shadow','enforce')),
            publish_mode     TEXT NOT NULL DEFAULT 'auto_publish_clean'
                             CHECK(publish_mode IN ('auto_publish_clean','explicit_publish')),
            baseline_version TEXT NOT NULL DEFAULT '',
            active_version   TEXT NOT NULL DEFAULT '',
            draft_version    TEXT NOT NULL DEFAULT '',
            updated_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS semantic_compile_run (
            run_id                   TEXT PRIMARY KEY,
            account_id               TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            trigger_label            TEXT NOT NULL DEFAULT '',
            initiated_by             TEXT NOT NULL DEFAULT '',
            mode                     TEXT NOT NULL DEFAULT 'shadow',
            status                   TEXT NOT NULL DEFAULT 'running',
            base_version             TEXT NOT NULL DEFAULT '',
            draft_version            TEXT NOT NULL DEFAULT '',
            published_version        TEXT NOT NULL DEFAULT '',
            error_count              INTEGER NOT NULL DEFAULT 0,
            warning_count            INTEGER NOT NULL DEFAULT 0,
            info_count               INTEGER NOT NULL DEFAULT 0,
            source_fingerprints_json TEXT NOT NULL DEFAULT '{}',
            message                  TEXT NOT NULL DEFAULT '',
            started_at               TEXT DEFAULT (datetime('now')),
            completed_at             TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_compile_run_account
            ON semantic_compile_run(account_id, started_at DESC);

        CREATE TABLE IF NOT EXISTS semantic_contract_version (
            account_id               TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            version                  TEXT NOT NULL,
            status                   TEXT NOT NULL DEFAULT 'draft'
                                     CHECK(status IN ('draft','active','superseded','rejected')),
            contract_json            TEXT NOT NULL,
            source_fingerprints_json TEXT NOT NULL DEFAULT '{}',
            compile_run_id           TEXT NOT NULL DEFAULT '',
            created_by               TEXT NOT NULL DEFAULT '',
            created_at               TEXT DEFAULT (datetime('now')),
            published_at             TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (account_id, version)
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_contract_version_status
            ON semantic_contract_version(account_id, status, created_at DESC);

        CREATE TABLE IF NOT EXISTS semantic_conflict (
            conflict_id       TEXT PRIMARY KEY,
            compile_run_id    TEXT NOT NULL REFERENCES semantic_compile_run(run_id) ON DELETE CASCADE,
            account_id        TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            conflict_key      TEXT NOT NULL,
            code              TEXT NOT NULL,
            severity          TEXT NOT NULL DEFAULT 'ERROR'
                              CHECK(severity IN ('ERROR','WARNING','INFO','STALE')),
            object_type       TEXT NOT NULL DEFAULT '',
            object_id         TEXT NOT NULL DEFAULT '',
            schema_name       TEXT NOT NULL DEFAULT '',
            table_name        TEXT NOT NULL DEFAULT '',
            origin            TEXT NOT NULL DEFAULT '',
            message           TEXT NOT NULL,
            evidence_json     TEXT NOT NULL DEFAULT '{}',
            suggestions_json  TEXT NOT NULL DEFAULT '[]',
            status            TEXT NOT NULL DEFAULT 'open'
                              CHECK(status IN ('open','resolved','acknowledged','dismissed')),
            resolution_note   TEXT NOT NULL DEFAULT '',
            resolved_by       TEXT NOT NULL DEFAULT '',
            created_at        TEXT DEFAULT (datetime('now')),
            resolved_at       TEXT NOT NULL DEFAULT '',
            UNIQUE(compile_run_id, conflict_key)
        );
        CREATE INDEX IF NOT EXISTS idx_semantic_conflict_account
            ON semantic_conflict(account_id, status, severity, created_at DESC);
        """
    )


def _ensure_compliance_tables(conn: sqlite3.Connection) -> None:
    """Create the normalized regulated-industry policy and audit schema."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS compliance_profile (
            account_id                 TEXT PRIMARY KEY REFERENCES client(account_id) ON DELETE CASCADE,
            mode                       TEXT NOT NULL DEFAULT 'standard',
            industry                   TEXT NOT NULL DEFAULT 'standard',
            jurisdictions_json         TEXT NOT NULL DEFAULT '[]',
            frameworks_json            TEXT NOT NULL DEFAULT '[]',
            policy_pack_key            TEXT NOT NULL DEFAULT '',
            policy_pack_version        TEXT NOT NULL DEFAULT '',
            lifecycle_state            TEXT NOT NULL DEFAULT 'DRAFT',
            enforcement_mode           TEXT NOT NULL DEFAULT 'shadow',
            active_policy_version       INTEGER NOT NULL DEFAULT 0,
            identity_control            TEXT NOT NULL DEFAULT 'password',
            managed_secrets_enabled     INTEGER NOT NULL DEFAULT 0,
            immutable_audit_enabled     INTEGER NOT NULL DEFAULT 0,
            external_audit_destination TEXT NOT NULL DEFAULT '',
            activated_by                TEXT NOT NULL DEFAULT '',
            activated_at                TEXT DEFAULT NULL,
            invalidated_at              TEXT DEFAULT NULL,
            invalidated_reason          TEXT NOT NULL DEFAULT '',
            created_at                  TEXT DEFAULT (datetime('now')),
            updated_at                  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS compliance_policy_version (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            version         INTEGER NOT NULL,
            status          TEXT NOT NULL DEFAULT 'draft',
            snapshot_json   TEXT NOT NULL DEFAULT '{}',
            change_summary  TEXT NOT NULL DEFAULT '',
            created_by      TEXT NOT NULL DEFAULT '',
            activated_by    TEXT NOT NULL DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now')),
            activated_at    TEXT DEFAULT NULL,
            UNIQUE(account_id, version)
        );

        CREATE TABLE IF NOT EXISTS data_asset_classification (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            table_fqn       TEXT NOT NULL,
            column_name     TEXT NOT NULL,
            sensitivity     TEXT NOT NULL DEFAULT 'INTERNAL',
            identifiability TEXT NOT NULL DEFAULT 'NONE',
            confidence      REAL NOT NULL DEFAULT 0.0,
            source          TEXT NOT NULL DEFAULT 'auto',
            reviewed        INTEGER NOT NULL DEFAULT 0,
            reviewed_by     TEXT NOT NULL DEFAULT '',
            reviewed_at     TEXT DEFAULT NULL,
            mask_strategy   TEXT NOT NULL DEFAULT 'redact',
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(account_id, table_fqn, column_name)
        );

        CREATE TABLE IF NOT EXISTS data_classification_tag (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER NOT NULL REFERENCES data_asset_classification(id) ON DELETE CASCADE,
            tag               TEXT NOT NULL,
            UNIQUE(classification_id, tag)
        );

        CREATE TABLE IF NOT EXISTS policy_rule (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id        TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            policy_version    INTEGER NOT NULL DEFAULT 1,
            name              TEXT NOT NULL DEFAULT '',
            subject_type      TEXT NOT NULL DEFAULT 'role',
            subject_id        TEXT NOT NULL DEFAULT 'analyst',
            resource_type     TEXT NOT NULL DEFAULT 'classification',
            resource_pattern  TEXT NOT NULL DEFAULT '*',
            action            TEXT NOT NULL DEFAULT 'query_execution',
            effect            TEXT NOT NULL DEFAULT 'deny',
            mask_strategy     TEXT NOT NULL DEFAULT '',
            aggregate_only    INTEGER NOT NULL DEFAULT 0,
            export_allowed    INTEGER NOT NULL DEFAULT 0,
            cache_ttl_seconds INTEGER NOT NULL DEFAULT 0,
            mandatory         INTEGER NOT NULL DEFAULT 0,
            enabled           INTEGER NOT NULL DEFAULT 1,
            created_at        TEXT DEFAULT (datetime('now')),
            updated_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS row_policy (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id       TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            policy_version   INTEGER NOT NULL DEFAULT 1,
            name             TEXT NOT NULL DEFAULT '',
            subject_type     TEXT NOT NULL DEFAULT 'role',
            subject_id       TEXT NOT NULL DEFAULT 'analyst',
            table_fqn        TEXT NOT NULL,
            condition_json   TEXT NOT NULL DEFAULT '{}',
            enabled          INTEGER NOT NULL DEFAULT 1,
            created_at       TEXT DEFAULT (datetime('now')),
            updated_at       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS purpose_registry (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id           TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            purpose_key          TEXT NOT NULL,
            name                 TEXT NOT NULL,
            description          TEXT NOT NULL DEFAULT '',
            legal_basis_ref      TEXT NOT NULL DEFAULT '',
            default_for_roles    TEXT NOT NULL DEFAULT '[]',
            requires_prompt      INTEGER NOT NULL DEFAULT 0,
            enabled              INTEGER NOT NULL DEFAULT 1,
            created_at           TEXT DEFAULT (datetime('now')),
            updated_at           TEXT DEFAULT (datetime('now')),
            UNIQUE(account_id, purpose_key)
        );

        CREATE TABLE IF NOT EXISTS purpose_permission (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            purpose_id      INTEGER NOT NULL REFERENCES purpose_registry(id) ON DELETE CASCADE,
            classification  TEXT NOT NULL,
            action          TEXT NOT NULL,
            effect          TEXT NOT NULL DEFAULT 'deny',
            UNIQUE(purpose_id, classification, action)
        );

        CREATE TABLE IF NOT EXISTS provider_agreement (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            provider        TEXT NOT NULL,
            agreement_type  TEXT NOT NULL,
            frameworks_json TEXT NOT NULL DEFAULT '[]',
            artifact_ref    TEXT NOT NULL DEFAULT '',
            artifact_hash   TEXT NOT NULL DEFAULT '',
            signed_at       TEXT DEFAULT NULL,
            expires_at      TEXT DEFAULT NULL,
            enabled         INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS user_attestation (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id       TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id   TEXT NOT NULL,
            attestation_type TEXT NOT NULL DEFAULT 'confidentiality',
            document_ref     TEXT NOT NULL DEFAULT '',
            granted_by       TEXT NOT NULL DEFAULT '',
            granted_at       TEXT DEFAULT (datetime('now')),
            revoked_at       TEXT DEFAULT NULL,
            revoked_by       TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS break_glass_grant (
            id                TEXT PRIMARY KEY,
            account_id        TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            user_id           TEXT NOT NULL,
            incident_ref      TEXT NOT NULL,
            reason            TEXT NOT NULL,
            resource_json     TEXT NOT NULL DEFAULT '[]',
            action_json       TEXT NOT NULL DEFAULT '[]',
            expires_at        TEXT NOT NULL,
            revoked_at        TEXT DEFAULT NULL,
            created_by        TEXT NOT NULL DEFAULT '',
            created_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS compliance_assessment_run (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            policy_version  INTEGER NOT NULL DEFAULT 0,
            status          TEXT NOT NULL DEFAULT 'running',
            critical_failed INTEGER NOT NULL DEFAULT 0,
            high_failed     INTEGER NOT NULL DEFAULT 0,
            passed_count    INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now')),
            completed_at    TEXT DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS compliance_assessment_result (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER NOT NULL REFERENCES compliance_assessment_run(id) ON DELETE CASCADE,
            control_key     TEXT NOT NULL,
            severity        TEXT NOT NULL,
            status          TEXT NOT NULL,
            message         TEXT NOT NULL DEFAULT '',
            remediation     TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS policy_decision_log (
            id              TEXT PRIMARY KEY,
            account_id      TEXT NOT NULL,
            user_id         TEXT NOT NULL DEFAULT '',
            action          TEXT NOT NULL,
            purpose_id      TEXT NOT NULL DEFAULT '',
            channel         TEXT NOT NULL DEFAULT '',
            allowed         INTEGER NOT NULL DEFAULT 0,
            reason_code     TEXT NOT NULL DEFAULT '',
            resource_json   TEXT NOT NULL DEFAULT '[]',
            obligation_json TEXT NOT NULL DEFAULT '{}',
            policy_version  INTEGER NOT NULL DEFAULT 0,
            previous_hash   TEXT NOT NULL DEFAULT '',
            record_hash     TEXT NOT NULL DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS export_event (
            id              TEXT PRIMARY KEY,
            account_id      TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            user_id         TEXT NOT NULL DEFAULT '',
            trace_id        TEXT NOT NULL DEFAULT '',
            policy_version  INTEGER NOT NULL DEFAULT 0,
            purpose_id      TEXT NOT NULL DEFAULT '',
            format          TEXT NOT NULL DEFAULT 'csv',
            row_count       INTEGER NOT NULL DEFAULT 0,
            columns_json    TEXT NOT NULL DEFAULT '[]',
            fingerprint     TEXT NOT NULL DEFAULT '',
            status          TEXT NOT NULL DEFAULT 'created',
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_classification_account_table
            ON data_asset_classification(account_id, table_fqn);
        CREATE INDEX IF NOT EXISTS idx_policy_rule_account_version
            ON policy_rule(account_id, policy_version, enabled);
        CREATE INDEX IF NOT EXISTS idx_row_policy_account_table
            ON row_policy(account_id, table_fqn, enabled);
        CREATE INDEX IF NOT EXISTS idx_policy_decision_account_created
            ON policy_decision_log(account_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_assessment_account_created
            ON compliance_assessment_run(account_id, created_at DESC);
        """
    )


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
            response_hash             TEXT    DEFAULT '',
            response_preview_sanitized TEXT   DEFAULT '',
            response_chars            INTEGER DEFAULT 0,
            error_msg                 TEXT    DEFAULT '',
            egress_manifest           TEXT    NOT NULL DEFAULT '',
            created_at                TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_account  ON llm_call_log(account_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_question ON llm_call_log(question_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_request  ON llm_call_log(request_id);
        CREATE INDEX IF NOT EXISTS idx_llm_call_log_created  ON llm_call_log(created_at);
        """
    )


def _ensure_agent_runtime_tables(conn: sqlite3.Connection) -> None:
    """Create the durable, tenant-scoped read-only agent runtime tables.

    The agent layer records orchestration state only. SQL, semantic resolution,
    validation, compliance decisions, and execution continue to live in the
    existing governed query pipeline and answer-trace tables.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_thread (
            id                 TEXT PRIMARY KEY,
            account_id         TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id     INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            external_thread_id TEXT NOT NULL,
            title              TEXT NOT NULL DEFAULT '',
            status             TEXT NOT NULL DEFAULT 'active',
            created_at         TEXT DEFAULT (datetime('now')),
            updated_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(account_id, portal_user_id, external_thread_id)
        );

        CREATE TABLE IF NOT EXISTS agent_run (
            id                      TEXT PRIMARY KEY,
            thread_id               TEXT NOT NULL REFERENCES agent_thread(id) ON DELETE CASCADE,
            account_id              TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id          INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            objective_sanitized     TEXT NOT NULL DEFAULT '',
            status                  TEXT NOT NULL DEFAULT 'queued',
            current_stage           TEXT NOT NULL DEFAULT '',
            max_tool_calls          INTEGER NOT NULL DEFAULT 5,
            tool_calls_used         INTEGER NOT NULL DEFAULT 0,
            selected_schema         TEXT NOT NULL DEFAULT '',
            purpose_id              TEXT NOT NULL DEFAULT '',
            allowed_tables_snapshot TEXT NOT NULL DEFAULT '[]',
            policy_version          INTEGER NOT NULL DEFAULT 0,
            contract_version        TEXT NOT NULL DEFAULT '',
            answer_trace_id         INTEGER REFERENCES answer_trace(id) ON DELETE SET NULL,
            outcome_type            TEXT NOT NULL DEFAULT '',
            error_code              TEXT NOT NULL DEFAULT '',
            error_message           TEXT NOT NULL DEFAULT '',
            started_at              TEXT DEFAULT NULL,
            completed_at            TEXT DEFAULT NULL,
            created_at              TEXT DEFAULT (datetime('now')),
            updated_at              TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS agent_message (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id         TEXT NOT NULL REFERENCES agent_thread(id) ON DELETE CASCADE,
            run_id            TEXT REFERENCES agent_run(id) ON DELETE SET NULL,
            account_id        TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id    INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            role              TEXT NOT NULL,
            message_type      TEXT NOT NULL DEFAULT 'text',
            content_sanitized TEXT NOT NULL DEFAULT '',
            metadata_json     TEXT NOT NULL DEFAULT '{}',
            created_at        TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS agent_step (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id             TEXT NOT NULL REFERENCES agent_run(id) ON DELETE CASCADE,
            account_id         TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id     INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            step_index         INTEGER NOT NULL,
            tool_name          TEXT NOT NULL,
            label              TEXT NOT NULL DEFAULT '',
            detail             TEXT NOT NULL DEFAULT '',
            status             TEXT NOT NULL DEFAULT 'pending',
            answer_trace_id    INTEGER REFERENCES answer_trace(id) ON DELETE SET NULL,
            policy_reason_code TEXT NOT NULL DEFAULT '',
            metadata_json      TEXT NOT NULL DEFAULT '{}',
            started_at         TEXT DEFAULT NULL,
            completed_at       TEXT DEFAULT NULL,
            created_at         TEXT DEFAULT (datetime('now')),
            updated_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(run_id, step_index)
        );

        CREATE TABLE IF NOT EXISTS agent_subtask (
            id                 TEXT PRIMARY KEY,
            parent_run_id      TEXT NOT NULL REFERENCES agent_run(id) ON DELETE CASCADE,
            account_id         TEXT NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id     INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            objective_sanitized TEXT NOT NULL DEFAULT '',
            tool_name          TEXT NOT NULL DEFAULT '',
            status             TEXT NOT NULL DEFAULT 'queued',
            result_summary     TEXT NOT NULL DEFAULT '',
            metadata_json      TEXT NOT NULL DEFAULT '{}',
            started_at         TEXT DEFAULT NULL,
            completed_at       TEXT DEFAULT NULL,
            created_at         TEXT DEFAULT (datetime('now')),
            updated_at         TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_agent_thread_owner
            ON agent_thread(account_id, portal_user_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_run_owner
            ON agent_run(account_id, portal_user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_run_thread
            ON agent_run(thread_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_agent_message_thread
            ON agent_message(thread_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_agent_step_run
            ON agent_step(run_id, step_index);
        CREATE INDEX IF NOT EXISTS idx_agent_subtask_run
            ON agent_subtask(parent_run_id, status, created_at);
        """
    )


def _ensure_answer_trace_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS answer_trace (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id             TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            portal_user_id          INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
            platform_user_id        TEXT    DEFAULT '',
            session_id              TEXT    DEFAULT '',
            question_id             TEXT    DEFAULT '',
            parent_question_id      TEXT    DEFAULT '',
            question_text_sanitized TEXT    DEFAULT '',
            request_source          TEXT    DEFAULT '',
            route                   TEXT    DEFAULT '',
            selected_schema         TEXT    DEFAULT '',
            allowed_tables_snapshot TEXT    DEFAULT '[]',
            retrieved_kb_chunk_ids  TEXT    DEFAULT '[]',
            retrieved_kb_scores     TEXT    DEFAULT '[]',
            llm_provider            TEXT    DEFAULT '',
            llm_model               TEXT    DEFAULT '',
            prompt_tokens           INTEGER DEFAULT 0,
            completion_tokens       INTEGER DEFAULT 0,
            generated_sql           TEXT    DEFAULT '',
            sql_validation_status   TEXT    DEFAULT '',
            sql_validation_error    TEXT    DEFAULT '',
            db_type                 TEXT    DEFAULT '',
            query_row_count         INTEGER DEFAULT 0,
            query_duration_ms       INTEGER DEFAULT 0,
            answer_type             TEXT    DEFAULT '',
            final_answer_summary    TEXT    DEFAULT '',
            error_message           TEXT    DEFAULT '',
            status                  TEXT    DEFAULT 'started',
            created_at              TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS answer_trace_step (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id       INTEGER NOT NULL REFERENCES answer_trace(id) ON DELETE CASCADE,
            step_order     INTEGER NOT NULL DEFAULT 0,
            step_name      TEXT    NOT NULL,
            input_summary  TEXT    DEFAULT '',
            output_summary TEXT    DEFAULT '',
            duration_ms    INTEGER DEFAULT 0,
            status         TEXT    DEFAULT 'success',
            metadata_json  TEXT    DEFAULT '{}',
            created_at     TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_answer_trace_account
            ON answer_trace(account_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_answer_trace_question
            ON answer_trace(question_id);
        CREATE INDEX IF NOT EXISTS idx_answer_trace_step_trace
            ON answer_trace_step(trace_id, step_order);
        """
    )


def _ensure_eval_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS eval_run (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id      TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            schema_name     TEXT    DEFAULT '',
            case_file       TEXT    DEFAULT '',
            total_cases     INTEGER DEFAULT 0,
            passed_cases    INTEGER DEFAULT 0,
            avg_score       REAL    DEFAULT 0.0,
            status          TEXT    DEFAULT 'completed',
            report_path     TEXT    DEFAULT '',
            created_at      TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS eval_case_result (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_run_id        INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
            case_id            TEXT    NOT NULL,
            question           TEXT    NOT NULL,
            score              REAL    DEFAULT 0.0,
            passed             INTEGER DEFAULT 0,
            generated_sql      TEXT    DEFAULT '',
            validation_status  TEXT    DEFAULT '',
            validation_error   TEXT    DEFAULT '',
            execution_status   TEXT    DEFAULT '',
            row_count          INTEGER DEFAULT 0,
            failures_json      TEXT    DEFAULT '[]',
            created_at         TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_eval_run_account
            ON eval_run(account_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_eval_case_run
            ON eval_case_result(eval_run_id);
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


def _post_migration_indexes(conn: sqlite3.Connection) -> None:
    """
    Indexes that reference columns added by migrations (v27+).
    Must run after the migration loop, not inside _SCHEMA.
    """
    idxs = [
        "CREATE INDEX IF NOT EXISTS idx_metric_registry_status "
        "ON metric_registry(account_id, metric_status)",
        "CREATE INDEX IF NOT EXISTS idx_answer_trace_metric "
        "ON answer_trace(metric_id)",
        "CREATE INDEX IF NOT EXISTS idx_entity_rel_validation "
        "ON entity_relationships(account_id, validation_status)",
        "CREATE INDEX IF NOT EXISTS idx_entity_rel_identity "
        "ON entity_relationships(account_id, relationship_key)",
    ]
    for ddl in idxs:
        try:
            conn.execute(ddl)
        except Exception as e:
            log.debug("Post-migration index skipped: %s", e)

    # v31: learning_candidate idempotency — one row per (account, question).
    # First deduplicate any existing rows (keep most recent per group), then
    # create the partial unique index.  Both steps are wrapped individually so
    # a transient error on one does not block the other.
    try:
        conn.execute(
            """
            DELETE FROM learning_candidate
            WHERE origin_question_id != ''
            AND id NOT IN (
                SELECT MAX(id)
                FROM   learning_candidate
                WHERE  origin_question_id != ''
                GROUP  BY account_id, origin_question_id
            )
            """
        )
    except Exception as e:
        log.debug("learning_candidate dedup skipped: %s", e)

    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_learning_candidate_origin_unique "
            "ON learning_candidate(account_id, origin_question_id) "
            "WHERE origin_question_id != ''"
        )
    except Exception as e:
        log.debug("learning_candidate unique index skipped: %s", e)

    # v35: one Teams tenant maps to exactly one client. Partial index so
    # every client with an empty teams_tenant_id (the default) is exempt.
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_client_teams_tenant_id "
            "ON client(teams_tenant_id) WHERE teams_tenant_id != ''"
        )
    except Exception as e:
        log.debug("client teams_tenant_id unique index skipped: %s", e)


def _ensure_metric_version_table(conn: sqlite3.Connection) -> None:
    """
    Versioned snapshot of every metric save.
    Each time a metric is updated, a new row is inserted here so the full
    edit history is preserved. answer_trace.metric_version_at_query links
    a query back to the exact metric definition that was active at the time.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metric_version (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_id    INTEGER NOT NULL REFERENCES metric_registry(id) ON DELETE CASCADE,
            account_id   TEXT    NOT NULL,
            version      INTEGER NOT NULL DEFAULT 1,
            name         TEXT    NOT NULL DEFAULT '',
            synonyms     TEXT    DEFAULT '',
            sql_template TEXT    NOT NULL DEFAULT '',
            formula_type TEXT    DEFAULT 'query',
            formula_ast  TEXT    DEFAULT '{}',
            description  TEXT    DEFAULT '',
            base_entity  TEXT    DEFAULT '',
            base_table   TEXT    DEFAULT '',
            metric_status TEXT   DEFAULT 'published',
            changed_by   TEXT    DEFAULT '',
            change_note  TEXT    DEFAULT '',
            created_at   TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_metric_version_metric
            ON metric_version(metric_id, version DESC);
        CREATE INDEX IF NOT EXISTS idx_metric_version_account
            ON metric_version(account_id, created_at DESC);
        """
    )


def _ensure_learning_loop_tables(conn: sqlite3.Connection) -> None:
    """
    Create the self-learning loop tables if they don't exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS answer_feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id   TEXT    NOT NULL,
            user_id       INTEGER NOT NULL REFERENCES portal_user(id) ON DELETE CASCADE,
            account_id    TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            schema_scope  TEXT    NOT NULL DEFAULT '',
            rating        INTEGER NOT NULL CHECK(rating IN (-1, 1)),
            reason_code   TEXT    NOT NULL DEFAULT '',
            comment       TEXT    NOT NULL DEFAULT '',
            question_text TEXT    NOT NULL DEFAULT '',
            sql_text      TEXT    NOT NULL DEFAULT '',
            created_at    TEXT    DEFAULT (datetime('now')),
            updated_at    TEXT    DEFAULT (datetime('now')),
            UNIQUE(question_id, user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_answer_feedback_question
            ON answer_feedback(question_id);
        CREATE INDEX IF NOT EXISTS idx_answer_feedback_account_status
            ON answer_feedback(account_id, rating, created_at DESC);

        CREATE TABLE IF NOT EXISTS learning_candidate (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id           TEXT    NOT NULL UNIQUE,
            origin_question_id     TEXT    NOT NULL DEFAULT '',
            account_id             TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            schema_scope           TEXT    NOT NULL DEFAULT '',
            candidate_type         TEXT    NOT NULL DEFAULT 'review'
                                   CHECK(candidate_type IN ('positive','review','negative','regression')),
            question_text          TEXT    NOT NULL DEFAULT '',
            sql_text               TEXT    NOT NULL DEFAULT '',
            corrected_sql          TEXT    NOT NULL DEFAULT '',
            technical_score        INTEGER NOT NULL DEFAULT 0,
            feedback_delta         INTEGER NOT NULL DEFAULT 0,
            final_score            INTEGER NOT NULL DEFAULT 0,
            evidence               TEXT    NOT NULL DEFAULT '{}',
            positive_vote_count    INTEGER NOT NULL DEFAULT 0,
            negative_vote_count    INTEGER NOT NULL DEFAULT 0,
            status                 TEXT    NOT NULL DEFAULT 'pending_review'
                                   CHECK(status IN ('pending_review','approved','rejected','known_failure','revoked')),
            source                 TEXT    NOT NULL DEFAULT 'auto'
                                   CHECK(source IN ('auto','admin_correction','pre_governed')),
            semantic_model_version TEXT    NOT NULL DEFAULT '',
            metric_version         TEXT    NOT NULL DEFAULT '',
            schema_version         TEXT    NOT NULL DEFAULT '',
            qdrant_id              TEXT    NOT NULL DEFAULT '',
            reviewer_id            TEXT    NOT NULL DEFAULT '',
            reviewer_note          TEXT    NOT NULL DEFAULT '',
            reviewed_at            TEXT    DEFAULT NULL,
            promoted_at            TEXT    DEFAULT NULL,
            created_at             TEXT    DEFAULT (datetime('now')),
            updated_at             TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_learning_candidate_account_status
            ON learning_candidate(account_id, status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_learning_candidate_question
            ON learning_candidate(origin_question_id);
        CREATE INDEX IF NOT EXISTS idx_learning_candidate_type
            ON learning_candidate(account_id, candidate_type, status);

        CREATE TABLE IF NOT EXISTS recommendation_event (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id         TEXT    NOT NULL DEFAULT '',
            user_id            INTEGER REFERENCES portal_user(id) ON DELETE SET NULL,
            account_id         TEXT    NOT NULL REFERENCES client(account_id) ON DELETE CASCADE,
            event_type         TEXT    NOT NULL
                               CHECK(event_type IN ('displayed','clicked','executed','successful','dismissed')),
            suggestion_text    TEXT    NOT NULL DEFAULT '',
            suggestion_source  TEXT    NOT NULL DEFAULT '',
            result_question_id TEXT    NOT NULL DEFAULT '',
            created_at         TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_recommendation_event_account
            ON recommendation_event(account_id, event_type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_recommendation_event_session
            ON recommendation_event(session_id);
        """
    )


def _ensure_metric_test_table(conn: sqlite3.Connection) -> None:
    """
    Golden-question test cases bound to a specific metric.
    Every published metric should have at least one test row.
    Eval runs execute these and store pass/fail + error details.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metric_test (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_id             INTEGER NOT NULL REFERENCES metric_registry(id) ON DELETE CASCADE,
            account_id            TEXT    NOT NULL,
            question              TEXT    NOT NULL,
            expected_sql_pattern  TEXT    DEFAULT '',
            expected_result_shape TEXT    DEFAULT '{}',
            required_tables       TEXT    DEFAULT '[]',
            required_dimensions   TEXT    DEFAULT '[]',
            forbidden_columns     TEXT    DEFAULT '[]',
            last_run_status       TEXT    DEFAULT ''
                                  CHECK(last_run_status IN ('','pass','fail')),
            last_run_at           TEXT    DEFAULT '',
            last_run_error        TEXT    DEFAULT '',
            created_at            TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_metric_test_metric
            ON metric_test(metric_id);
        CREATE INDEX IF NOT EXISTS idx_metric_test_account
            ON metric_test(account_id);
        """
    )


def _migrate_legacy_examples_to_candidates() -> None:
    """
    One-time migration: convert all existing validated_examples that still carry
    approval_status='legacy' into learning_candidate rows with:
      status='pending_review', source='pre_governed', technical_score=85.

    Idempotent — already-migrated rows are skipped.

    IMPORTANT (B7 = dual-collection): the legacy Qdrant collection is left
    completely untouched.  Legacy examples continue to be served from it until
    admins approve them through the new governed collection.  Existing retrieval
    quality is preserved during the transition.
    """
    import uuid as _uuid
    migrated = 0
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, account_id, question, sql_query, source, created_at
                FROM   validated_examples
                WHERE  approval_status = 'legacy'
                  AND  candidate_id    = ''
                """
            ).fetchall()

            for row in rows:
                cid = _uuid.uuid4().hex[:12]
                conn.execute(
                    """
                    INSERT OR IGNORE INTO learning_candidate (
                        candidate_id, origin_question_id, account_id,
                        candidate_type, question_text, sql_text,
                        technical_score, final_score,
                        evidence, status, source, created_at
                    ) VALUES (?, '', ?, 'positive', ?, ?, 85, 85,
                              '{"note":"migrated_from_validated_examples"}',
                              'pending_review', 'pre_governed', ?)
                    """,
                    (cid, row["account_id"], row["question"],
                     row["sql_query"], row["created_at"]),
                )
                conn.execute(
                    "UPDATE validated_examples SET candidate_id=? WHERE id=?",
                    (cid, row["id"]),
                )
                migrated += 1

    except Exception as exc:
        log.warning("Legacy-examples migration skipped: %s", exc)
        return

    if migrated:
        log.info(
            "Learning loop: migrated %d validated_examples → learning_candidate "
            "(pending_review, pre_governed).  Legacy Qdrant collection untouched.",
            migrated,
        )
