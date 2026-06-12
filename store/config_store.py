"""
store/config_store.py

All read/write operations against the SQLite config store.
Every credential field is stored encrypted — callers get plain dicts.
"""

import json
import logging
from typing import Optional
from datetime import datetime, timezone

from store.db import get_db, get_table_columns
from store.crypto import encrypt, decrypt, decrypt_json

log = logging.getLogger("querybot.store")

# ── LLM cost rates (USD per 1M tokens) ───────────────────────────────────────
# These are the FALLBACK defaults used when a model is not yet in the
# llm_pricing SQLite table.  calculate_cost() reads the DB first — so admins
# can update rates live from the billing page without touching code or
# restarting the service.  Only add new models here; the DB rows take
# precedence once seeded.
LLM_COST_RATES: dict[str, dict] = {
    "claude-sonnet-4-6":  {"in": 3.00,  "out": 15.00},
    "claude-opus-4-5":    {"in": 15.00, "out": 75.00},
    "claude-haiku-4-5":   {"in": 0.80,  "out": 4.00},
    "gpt-4o":             {"in": 5.00,  "out": 15.00},
    "gpt-4o-mini":        {"in": 0.15,  "out": 0.60},
}

# In-process cache so every LLM call doesn't hit SQLite for pricing.
# Invalidated whenever save_pricing() writes a new rate.
_pricing_cache: dict[str, dict] | None = None


def _load_pricing_cache() -> dict[str, dict]:
    """Read all rows from llm_pricing into a model→rates dict."""
    result: dict[str, dict] = {}
    try:
        with get_db() as conn:
            for row in conn.execute(
                "SELECT model, tokens_in, tokens_out FROM llm_pricing"
            ).fetchall():
                result[row[0]] = {"in": float(row[1]), "out": float(row[2])}
    except Exception:
        pass  # table may not exist yet on very first startup before init_db
    return result


def get_all_pricing() -> list[dict]:
    """
    Return all known models with their current effective rates and source.
    Merges DB rows (source='db') with any hardcoded defaults not yet in DB
    (source='default').  Used by the billing page to show the editable table.
    """
    db_rates = _load_pricing_cache()
    rows: list[dict] = []
    seen: set[str] = set()
    # DB rows first — these are the authoritative rates
    try:
        with get_db() as conn:
            for row in conn.execute(
                "SELECT model, tokens_in, tokens_out, updated_at FROM llm_pricing ORDER BY model"
            ).fetchall():
                rows.append({
                    "model":      row[0],
                    "tokens_in":  float(row[1]),
                    "tokens_out": float(row[2]),
                    "updated_at": row[3] or "",
                    "source":     "db",
                })
                seen.add(row[0])
    except Exception:
        pass
    # Fallback defaults for any model not yet in DB
    for model, rates in sorted(LLM_COST_RATES.items()):
        if model not in seen:
            rows.append({
                "model":      model,
                "tokens_in":  rates["in"],
                "tokens_out": rates["out"],
                "updated_at": "",
                "source":     "default",
            })
    return rows


def save_pricing(model: str, tokens_in: float, tokens_out: float) -> None:
    """Upsert a model's rates into llm_pricing and invalidate the cache."""
    global _pricing_cache
    with get_db() as conn:
        conn.execute(
            """INSERT INTO llm_pricing (model, tokens_in, tokens_out, updated_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(model) DO UPDATE SET
                   tokens_in  = excluded.tokens_in,
                   tokens_out = excluded.tokens_out,
                   updated_at = excluded.updated_at""",
            (model.strip(), float(tokens_in), float(tokens_out)),
        )
    _pricing_cache = None  # force reload on next calculate_cost call
    log.info("llm_pricing updated: model=%s in=%.4f out=%.4f", model, tokens_in, tokens_out)


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """
    Compute USD cost for an LLM call.
    Reads rates from llm_pricing SQLite table (admin-editable) first;
    falls back to LLM_COST_RATES hardcoded defaults for unknown models.
    Falls back to gpt-4o rates if the model is completely unknown.
    """
    global _pricing_cache
    if _pricing_cache is None:
        _pricing_cache = _load_pricing_cache()
    rates = (
        _pricing_cache.get(model)
        or LLM_COST_RATES.get(model)
        or {"in": 5.00, "out": 15.00}   # safe fallback: gpt-4o price
    )
    return (tokens_in * rates["in"] + tokens_out * rates["out"]) / 1_000_000


# ══════════════════════════════════════════════════════════════════════════════
# System config
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_KEYS = {
    "anthropic_api_key",
    "openai_api_key",
    # Azure OpenAI (optional — use instead of openai if you have Azure credits)
    "azure_openai_api_key",
    "azure_openai_endpoint",    # https://yourresource.openai.azure.com
    "azure_openai_api_version", # e.g. 2024-02-01
    "default_llm_provider",     # "anthropic" | "openai" | "azure_openai"
    "default_llm_model",        # model name or Azure deployment name
    "kb_llm_model",             # used once for KB generation
    "admin_password_hash",
}


def set_system(key: str, value: str) -> None:
    if key not in SYSTEM_KEYS:
        raise ValueError(f"Unknown system key: {key!r}")
    with get_db() as conn:
        conn.execute("""
            INSERT INTO system_config (key, value_encrypted, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value_encrypted = excluded.value_encrypted,
                updated_at      = excluded.updated_at
        """, (key, encrypt(value)))


def get_system(key: str, default: str = "") -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT value_encrypted FROM system_config WHERE key = ?", (key,)
        ).fetchone()
    return decrypt(row["value_encrypted"]) if row else default


def get_all_system() -> dict[str, str]:
    with get_db() as conn:
        rows = conn.execute("SELECT key, value_encrypted FROM system_config").fetchall()
    return {r["key"]: decrypt(r["value_encrypted"]) for r in rows}


# ══════════════════════════════════════════════════════════════════════════════
# Platform config
# ══════════════════════════════════════════════════════════════════════════════

PLATFORM_FIELDS: dict[str, list[str]] = {
    "zoom":  ["client_id", "client_secret", "bot_jid", "webhook_secret"],
    "teams": ["app_id", "app_password", "tenant_id"],
    "slack": ["bot_token", "signing_secret", "app_id"],
}

PLATFORM_LABELS = {"zoom": "Zoom Team Chat", "teams": "Microsoft Teams", "slack": "Slack"}


def save_platform(
    platform_type: str, name: str, credentials: dict,
    platform_id: Optional[int] = None,
) -> int:
    _validate_fields(platform_type, credentials, PLATFORM_FIELDS)
    enc = encrypt(credentials)
    with get_db() as conn:
        if platform_id:
            conn.execute("""
                UPDATE platform_config
                SET name=?, platform_type=?, credentials_encrypted=?, updated_at=datetime('now')
                WHERE id=?
            """, (name, platform_type, enc, platform_id))
            return platform_id
        cur = conn.execute("""
            INSERT INTO platform_config (platform_type, name, is_active, credentials_encrypted)
            VALUES (?, ?, 1, ?)
        """, (platform_type, name, enc))
        log.info("Created platform_config id=%d (%s)", cur.lastrowid, platform_type)
        return cur.lastrowid


def get_platform(platform_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM platform_config WHERE id=?", (platform_id,)).fetchone()
    return _platform_row(row) if row else None


def list_platforms(platform_type: Optional[str] = None) -> list[dict]:
    with get_db() as conn:
        if platform_type:
            rows = conn.execute(
                "SELECT * FROM platform_config WHERE platform_type=? ORDER BY id",
                (platform_type,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM platform_config ORDER BY platform_type, id"
            ).fetchall()
    return [_platform_row(r) for r in rows]


def delete_platform(platform_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM platform_config WHERE id=?", (platform_id,))


def _platform_row(row) -> dict:
    d = dict(row)
    d["credentials"] = decrypt_json(d.pop("credentials_encrypted"))
    d["label"] = PLATFORM_LABELS.get(d["platform_type"], d["platform_type"])
    return d


# ══════════════════════════════════════════════════════════════════════════════
# DB config
# ══════════════════════════════════════════════════════════════════════════════

DB_REQUIRED_FIELDS: dict[str, list[str]] = {
    "snowflake": ["account", "user", "password", "warehouse"],
    "oracle":    ["user", "password", "dsn"],
    "azure_sql": ["server", "user", "password"],   # database is optional — omit to browse all DBs on the server
}

DB_DIALECT: dict[str, str] = {
    "snowflake": "snowflake",
    "oracle":    "oracle",
    "azure_sql": "tsql",
}

DB_LABEL: dict[str, str] = {
    "snowflake": "Snowflake",
    "oracle":    "Oracle",
    "azure_sql": "Azure SQL",
}


def save_db_config(
    db_type: str, name: str, credentials: dict,
    db_id: Optional[int] = None,
) -> int:
    _validate_fields(db_type, credentials, DB_REQUIRED_FIELDS)
    enc = encrypt(credentials)
    with get_db() as conn:
        if db_id:
            conn.execute("""
                UPDATE db_config
                SET name=?, db_type=?, credentials_encrypted=?, updated_at=datetime('now')
                WHERE id=?
            """, (name, db_type, enc, db_id))
            return db_id
        cur = conn.execute("""
            INSERT INTO db_config (name, db_type, credentials_encrypted)
            VALUES (?, ?, ?)
        """, (name, db_type, enc))
        log.info("Created db_config id=%d (%s)", cur.lastrowid, db_type)
        return cur.lastrowid


def get_db_config(db_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM db_config WHERE id=?", (db_id,)).fetchone()
    return _db_row(row) if row else None


def list_db_configs() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM db_config ORDER BY db_type, id").fetchall()
    return [_db_row(r) for r in rows]


def delete_db_config(db_id: int) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM db_config WHERE id=?", (db_id,))


def _db_row(row) -> dict:
    d = dict(row)
    d["credentials"] = decrypt_json(d.pop("credentials_encrypted"))
    d["dialect"] = DB_DIALECT.get(d["db_type"], "snowflake")
    d["label"]   = DB_LABEL.get(d["db_type"], d["db_type"])
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Client registry
# ══════════════════════════════════════════════════════════════════════════════

def upsert_client(account_id: str, platform_type: str) -> None:
    """Register a new client or touch updated_at if already exists."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO client (account_id, platform_type)
            VALUES (?, ?)
            ON CONFLICT(account_id) DO UPDATE SET updated_at = datetime('now')
        """, (account_id, platform_type))


def get_client(account_id: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM client WHERE account_id=?", (account_id,)).fetchone()
    return dict(row) if row else None


def get_client_state(account_id: str) -> dict:
    """Return the parsed state_data dict for a client (empty dict if not found)."""
    client = get_client(account_id)
    if not client:
        return {}
    return json.loads(client.get("state_data") or "{}")


def list_clients(search: Optional[str] = None) -> list[dict]:
    with get_db() as conn:
        if search:
            rows = conn.execute("""
                SELECT * FROM client
                WHERE client_name LIKE ? OR account_id LIKE ?
                ORDER BY created_at DESC
            """, (f"%{search}%", f"%{search}%")).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM client ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def update_client_state(
    account_id: str,
    state: str,
    state_data: Optional[dict] = None,
    business_desc: Optional[str] = None,
) -> None:
    with get_db() as conn:
        if business_desc is not None:
            conn.execute("""
                UPDATE client SET state=?, state_data=?, business_desc=?, updated_at=datetime('now')
                WHERE account_id=?
            """, (state, json.dumps(state_data or {}), business_desc, account_id))
        else:
            conn.execute("""
                UPDATE client SET state=?, state_data=?, updated_at=datetime('now')
                WHERE account_id=?
            """, (state, json.dumps(state_data or {}), account_id))


def update_client_meta(
    account_id: str,
    client_name: Optional[str] = None,
    db_config_id: Optional[int] = None,
    platform_config_id: Optional[int] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    query_limit_monthly: Optional[int] = None,
    token_limit_monthly: Optional[int] = None,
    chat_ui_enabled: Optional[int] = None,
    enable_llm_audit: Optional[int] = None,
    portal_only: Optional[int] = None,
    enable_feedback_collection: Optional[int] = None,
    graph_use_suggested: Optional[int] = None,
) -> None:
    """Update one or more metadata fields on a client row."""
    fields, params = [], []
    if client_name is not None:
        fields.append("client_name = ?"); params.append(client_name)
    if db_config_id is not None:
        fields.append("db_config_id = ?"); params.append(db_config_id)
    if platform_config_id is not None:
        fields.append("platform_config_id = ?"); params.append(platform_config_id)
    if llm_provider is not None:
        fields.append("llm_provider = ?"); params.append(llm_provider)
    if llm_model is not None:
        fields.append("llm_model = ?"); params.append(llm_model)
    if query_limit_monthly is not None:
        fields.append("query_limit_monthly = ?"); params.append(query_limit_monthly)
    if token_limit_monthly is not None:
        fields.append("token_limit_monthly = ?"); params.append(token_limit_monthly)
    if chat_ui_enabled is not None:
        fields.append("chat_ui_enabled = ?"); params.append(chat_ui_enabled)
    if enable_llm_audit is not None:
        fields.append("enable_llm_audit = ?"); params.append(enable_llm_audit)
    if portal_only is not None:
        fields.append("portal_only = ?"); params.append(portal_only)
    if enable_feedback_collection is not None:
        fields.append("enable_feedback_collection = ?"); params.append(enable_feedback_collection)
    if graph_use_suggested is not None:
        fields.append("graph_use_suggested = ?"); params.append(graph_use_suggested)
    if not fields:
        return
    fields.append("updated_at = datetime('now')")
    params.append(account_id)
    with get_db() as conn:
        conn.execute(f"UPDATE client SET {', '.join(fields)} WHERE account_id = ?", params)


def delete_client(account_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM client WHERE account_id=?", (account_id,))


def get_monthly_query_count(account_id: str) -> int:
    """Queries run this calendar month for this client."""
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS cnt FROM query_log
            WHERE account_id = ?
              AND success = 1
              AND SUBSTRING(created_at, 1, 7) = ?
        """, (account_id, current_month)).fetchone()
    return row["cnt"] if row else 0


# ══════════════════════════════════════════════════════════════════════════════
# Query log
# ══════════════════════════════════════════════════════════════════════════════

def get_monthly_token_usage(
    account_id: str,
    portal_user_id: Optional[int] = None,
) -> dict:
    """Token usage for the current calendar month.

    When portal_user_id is supplied, the result is scoped to that portal user.
    This keeps the user chat KPI personal while still reusing the same query_log
    fields that power admin billing.
    """
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    where = [
        "account_id = ?",
        "SUBSTRING(created_at, 1, 7) = ?",
    ]
    params: list = [account_id, current_month]
    if portal_user_id is not None:
        where.append("portal_user_id = ?")
        params.append(portal_user_id)

    with get_db() as conn:
        row = conn.execute(f"""
            SELECT
                COUNT(*)                       AS query_count,
                COALESCE(SUM(tokens_in), 0)    AS tokens_in,
                COALESCE(SUM(tokens_out), 0)   AS tokens_out,
                COALESCE(SUM(cost_usd), 0.0)   AS cost_usd
            FROM query_log
            WHERE {' AND '.join(where)}
        """, tuple(params)).fetchone()

    data = dict(row) if row else {}
    tokens_in = int(data.get("tokens_in") or 0)
    tokens_out = int(data.get("tokens_out") or 0)
    data["tokens_in"] = tokens_in
    data["tokens_out"] = tokens_out
    data["total_tokens"] = tokens_in + tokens_out
    data["query_count"] = int(data.get("query_count") or 0)
    data["cost_usd"] = float(data.get("cost_usd") or 0.0)
    return data


def get_monthly_token_status(
    account_id: str,
    portal_user_id: Optional[int] = None,
) -> dict:
    usage = get_monthly_token_usage(account_id, portal_user_id)
    client = get_client(account_id) or {}
    limit = int(client.get("token_limit_monthly") or 0)
    used = int(usage.get("total_tokens") or 0)
    remaining = None if limit <= 0 else max(limit - used, 0)
    pct = 0 if limit <= 0 else min(round(used / limit * 100), 100)
    usage.update({
        "limit": limit,
        "remaining": remaining,
        "limit_pct": pct,
        "unlimited": limit <= 0,
    })
    return usage


def log_query(
    account_id: str,
    question: str,
    sql_generated: str,
    row_count: int = 0,
    success: bool = True,
    error_msg: str = "",
    llm_provider: str = "",
    llm_model: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    duration_ms: int = 0,
    portal_user_id: Optional[int] = None,
    zoom_user_id: str = "",
    question_id: str = "",
    parent_question_id: str = "",
) -> None:
    cost = calculate_cost(llm_model, tokens_in, tokens_out)
    with get_db() as conn:
        conn.execute("""
            INSERT INTO query_log
                (account_id, portal_user_id, zoom_user_id,
                 question, sql_generated, row_count,
                 success, error_msg, llm_provider, llm_model,
                 tokens_in, tokens_out, cost_usd, duration_ms,
                 question_id, parent_question_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (account_id, portal_user_id, zoom_user_id or "",
              question, sql_generated, row_count,
              1 if success else 0, error_msg, llm_provider, llm_model,
              tokens_in, tokens_out, cost, duration_ms,
              question_id or "", parent_question_id or ""))


def get_query_stats(account_id: Optional[str] = None, month: Optional[str] = None) -> dict:
    """
    Summary stats — optionally filtered by client and/or month (YYYY-MM).
    Returns dict with keys: total, succeeded, total_tokens_in, total_tokens_out,
    total_cost_usd, avg_duration_ms.
    """
    where_clauses, params = [], []
    if account_id:
        where_clauses.append("account_id = ?"); params.append(account_id)
    if month:
        where_clauses.append("SUBSTRING(created_at, 1, 7) = ?"); params.append(month)
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with get_db() as conn:
        row = conn.execute(f"""
            SELECT
                COUNT(*)                              AS total,
                SUM(CASE WHEN success=1 THEN 1 END)  AS succeeded,
                SUM(tokens_in)                        AS total_tokens_in,
                SUM(tokens_out)                       AS total_tokens_out,
                SUM(cost_usd)                         AS total_cost_usd,
                AVG(duration_ms)                      AS avg_duration_ms
            FROM query_log {where}
        """, params).fetchone()
    return dict(row) if row else {}


def get_suggestions(account_id: str, q: str, limit: int = 8) -> list[dict]:
    """Return autocomplete suggestions for a partial chat query string.

    Sources (ranked highest → lowest):
      1. Recent successful query_log questions  (score 120 prefix / 70 contains)
      2. Metric names + synonyms + example_questions (score 100 prefix / 60 contains)
      3. Entity display names                   (score 90 prefix / 50 contains)
      4. Business terms + aliases               (score 80 prefix / 40 contains)

    Returns a list of dicts: {text: str, kind: "history"|"metric"|"entity"|"term"}
    """
    import re as _re
    q = (q or "").strip()
    if len(q) < 2:
        return []
    q_lower = q.lower()
    candidates: list[tuple[int, str, str]] = []  # (score, text, kind)

    with get_db() as conn:
        # 1 — Recent successful questions (most valuable: real questions that worked)
        rows = conn.execute(
            """SELECT DISTINCT question FROM query_log
                WHERE account_id=? AND success=1 AND question IS NOT NULL
                ORDER BY created_at DESC LIMIT 300""",
            (account_id,),
        ).fetchall()
        for r in rows:
            text = (r["question"] or "").strip()
            if not text:
                continue
            tl = text.lower()
            if q_lower in tl:
                candidates.append((120 if tl.startswith(q_lower) else 70, text, "history"))

        # 2 — Metric names, synonyms, example_questions
        rows = conn.execute(
            "SELECT name, synonyms, example_questions FROM metric_registry WHERE account_id=? AND is_active=1",
            (account_id,),
        ).fetchall()
        for r in rows:
            for field in (r["name"], r["synonyms"], r["example_questions"]):
                if not field:
                    continue
                for part in _re.split(r"[,;\n]+", str(field)):
                    text = part.strip()
                    if len(text) < 3:
                        continue
                    tl = text.lower()
                    if q_lower in tl:
                        candidates.append((100 if tl.startswith(q_lower) else 60, text, "metric"))

        # 3 — Entity names
        rows = conn.execute(
            "SELECT entity_name, display_name FROM entity_graph WHERE account_id=? AND is_active=1",
            (account_id,),
        ).fetchall()
        for r in rows:
            for field in (r["display_name"], r["entity_name"]):
                text = (field or "").strip()
                if len(text) < 2:
                    continue
                tl = text.lower()
                if q_lower in tl:
                    candidates.append((90 if tl.startswith(q_lower) else 50, text, "entity"))

        # 4 — Business terms and aliases
        rows = conn.execute(
            "SELECT term, aliases FROM business_term WHERE account_id=? AND is_active=1",
            (account_id,),
        ).fetchall()
        for r in rows:
            for field in (r["term"], r["aliases"]):
                if not field:
                    continue
                for part in _re.split(r"[,;\n]+", str(field)):
                    text = part.strip()
                    if len(text) < 2:
                        continue
                    tl = text.lower()
                    if q_lower in tl:
                        candidates.append((80 if tl.startswith(q_lower) else 40, text, "term"))

    # Deduplicate (case-insensitive), rank, cap
    seen: set[str] = set()
    result: list[dict] = []
    for score, text, kind in sorted(candidates, key=lambda x: -x[0]):
        key = text.lower()
        if key not in seen:
            seen.add(key)
            result.append({"text": text, "kind": kind})
        if len(result) >= limit:
            break
    return result


def get_recent_queries(account_id: str, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("""
            SELECT q.*,
                   COALESCE(u.name, '') AS user_name,
                   COALESCE(u.email, '') AS user_email
              FROM query_log q
              LEFT JOIN portal_user u ON u.id = q.portal_user_id
             WHERE q.account_id = ?
             ORDER BY q.created_at DESC
             LIMIT ?
        """, (account_id, limit)).fetchall()
    return [dict(r) for r in rows]


def log_llm_call(
    account_id: str,
    request_id: str,
    question: str,
    component: str,
    llm_provider: str,
    llm_model: str,
    status: str,
    payload_hash: str,
    payload_preview_sanitized: str,
    prompt_chars: int = 0,
    error_msg: str = "",
    question_id: str = "",
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO llm_call_log
                (account_id, question_id, request_id, question, component,
                 llm_provider, llm_model, status, payload_hash,
                 payload_preview_sanitized, prompt_chars, error_msg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                question_id or request_id,  # fall back to request_id so existing rows are never blank
                request_id,
                question,
                component,
                llm_provider,
                llm_model,
                status,
                payload_hash,
                payload_preview_sanitized,
                prompt_chars,
                error_msg,
            ),
        )


def get_recent_llm_calls(
    account_id: str,
    limit: int = 100,
    component: str = "",
    status: str = "",
) -> list[dict]:
    """
    Return llm_call_log rows for a client grouped by question_id so that
    all LLM calls for one user question (initial SQL, drilldown, follow-up
    analysis) appear as a single logical entry with child rows nested inside.

    Each dict has:
      question_id  — stable ID for the original question
      question     — the user's question text (sanitized)
      first_call   — created_at of the first LLM call in this group
      call_count   — total LLM calls in this group
      components   — comma-separated list of distinct component names
      total_chars  — sum of prompt_chars across all calls in this group
      any_error    — 1 if any call in this group had status='error'
      calls        — list of individual row dicts (id, component, model,
                     status, prompt_chars, payload_preview_sanitized,
                     error_msg, created_at, request_id)

    Filtered by component or status if provided — both apply to individual
    calls; a group is included if ANY of its calls match the filter.
    """
    # Build the per-call WHERE clause
    where = ["account_id = ?"]
    params: list = [account_id]
    if component:
        where.append("component = ?")
        params.append(component)
    if status:
        where.append("status = ?")
        params.append(status)

    where_clause = " AND ".join(where)

    with get_db() as conn:
        # Fetch flat matching rows, ordered oldest-first within each group
        rows = conn.execute(
            f"""
            SELECT *
              FROM llm_call_log
             WHERE {where_clause}
             ORDER BY question_id, created_at ASC, id ASC
            """,
            tuple(params),
        ).fetchall()

    if not rows:
        return []

    # Group into question buckets, preserving newest-first question order
    from collections import defaultdict, OrderedDict
    groups: dict[str, list] = OrderedDict()
    for row in rows:
        r = dict(row)
        qid = r.get("question_id") or r["request_id"]
        if qid not in groups:
            groups[qid] = []
        groups[qid].append(r)

    result = []
    for qid, calls in groups.items():
        first = calls[0]
        components_seen = []
        for c in calls:
            if c["component"] not in components_seen:
                components_seen.append(c["component"])
        result.append({
            "question_id":  qid,
            "question":     first.get("question", ""),
            "first_call":   first["created_at"],
            "last_call":    calls[-1]["created_at"],
            "call_count":   len(calls),
            "components":   ", ".join(components_seen),
            "total_chars":  sum(c.get("prompt_chars", 0) for c in calls),
            "any_error":    1 if any(c["status"] == "error" for c in calls) else 0,
            "calls":        calls,
        })

    # Sort groups newest-first (by first_call timestamp descending)
    result.sort(key=lambda g: g["first_call"], reverse=True)

    # Apply limit at the group level
    return result[:limit]


def purge_old_llm_calls(retention_days: int = 30) -> int:
    """
    Delete llm_call_log rows older than retention_days. Returns the number
    of rows deleted. Called on startup and safe to call repeatedly.
    """
    if retention_days <= 0:
        return 0
    from datetime import timedelta
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=int(retention_days))
    ).strftime("%Y-%m-%dT%H:%M:%S")
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM llm_call_log WHERE created_at < ?",
            (cutoff,),
        )
        return cur.rowcount or 0


def get_monthly_breakdown(account_id: str) -> list[dict]:
    """Daily query counts + cost for the current month — for billing export."""
    current_month = datetime.now(timezone.utc).strftime("%Y-%m")
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                SUBSTRING(created_at, 1, 10)      AS date,
                COUNT(*)                          AS total_queries,
                SUM(CASE WHEN success=1 THEN 1 END) AS successful,
                SUM(tokens_in)                    AS tokens_in,
                SUM(tokens_out)                   AS tokens_out,
                SUM(cost_usd)                     AS cost_usd
            FROM query_log
            WHERE account_id = ?
              AND SUBSTRING(created_at, 1, 7) = ?
            GROUP BY date ORDER BY date
        """, (account_id, current_month)).fetchall()
    return [dict(r) for r in rows]


# ── Internal helper ───────────────────────────────────────────────────────────

def _validate_fields(type_key: str, credentials: dict, field_map: dict) -> None:
    required = field_map.get(type_key)
    if required is None:
        raise ValueError(f"Unknown type: {type_key!r}")
    missing = [f for f in required if not credentials.get(f)]
    if missing:
        raise ValueError(f"Missing required fields for {type_key}: {', '.join(missing)}")


def load_schema_tables(schema_dir: str) -> set[str]:
    """Return set of uppercase table names from _schema.json."""
    import json
    from pathlib import Path
    p = Path(schema_dir) / "_schema.json"
    if not p.exists():
        return set()
    return {t.upper() for t in json.loads(p.read_text(encoding="utf-8"))}


# ══════════════════════════════════════════════════════════════════════════════
# Metric registry — Step 3
# ══════════════════════════════════════════════════════════════════════════════

def _metric_choice(value: str, allowed: set[str], default: str) -> str:
    value = (value or "").strip().lower()
    return value if value in allowed else default


def _ensure_metric_registry_schema(conn) -> None:
    conn.execute("""
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
            category     TEXT    DEFAULT '',
            default_time_column TEXT DEFAULT '',
            is_active    INTEGER NOT NULL DEFAULT 1,
            created_at   TEXT    DEFAULT (datetime('now')),
            updated_at   TEXT    DEFAULT (datetime('now'))
        )
    """)
    existing = set(get_table_columns(conn, "metric_registry"))
    columns = {
        "formula_type": "TEXT NOT NULL DEFAULT 'query'",
        "result_format": "TEXT NOT NULL DEFAULT 'number'",
        "required_columns": "TEXT DEFAULT ''",
        "allowed_dimensions": "TEXT DEFAULT ''",
        "metric_builder_config": "TEXT DEFAULT ''",
        "example_questions": "TEXT DEFAULT ''",
        "grain": "TEXT DEFAULT ''",
        "category": "TEXT DEFAULT ''",
        "default_time_column": "TEXT DEFAULT ''",
    }
    for column, definition in columns.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE metric_registry ADD COLUMN {column} {definition}")


def save_metric(account_id: str, metric: dict, *, db_type: str = "azure_sql") -> int:
    """
    Save a metric definition to the metric_registry table.

    Runs backend validation before saving. Sets metric_status to 'validated'
    when the formula passes all checks, or 'draft' when it fails.
    Writes validation_errors and last_validated_at on every save.

    Returns the new metric id.
    """
    import json as _json
    from store.db import get_db
    from core.metric_validator import validate_metric, derive_metric_status, load_schema_columns

    formula_type = _metric_choice(
        metric.get("formula_type", "query"), {"query", "expression"}, "query"
    )
    result_format = _metric_choice(
        metric.get("result_format", "number"),
        {"number", "currency", "percentage", "date", "text"},
        "number",
    )

    schema_columns = load_schema_columns(account_id)
    vr = validate_metric(metric, db_type=db_type, schema_columns=schema_columns)
    metric_status = derive_metric_status(vr, formula_type)

    with get_db() as conn:
        _ensure_metric_registry_schema(conn)
        cur = conn.execute("""
            INSERT INTO metric_registry
                (account_id, name, synonyms, sql_template, description,
                 formula_type, result_format, required_columns,
                 allowed_dimensions, metric_builder_config, example_questions, grain,
                 category, default_time_column,
                 base_entity, base_table,
                 formula_ast, metric_status, validation_errors,
                 last_validated_at, version, owner)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), 1, ?)
        """, (
            account_id,
            metric["name"],
            metric.get("synonyms", ""),
            metric["sql_template"],
            metric.get("description", ""),
            formula_type,
            result_format,
            metric.get("required_columns", ""),
            metric.get("allowed_dimensions", ""),
            metric.get("metric_builder_config", ""),
            metric.get("example_questions", ""),
            metric.get("grain", ""),
            metric.get("category", ""),
            metric.get("default_time_column", ""),
            metric.get("base_entity", ""),
            metric.get("base_table", ""),
            _json.dumps(vr.formula_ast),
            metric_status,
            _json.dumps(vr.errors),
            metric.get("owner", ""),
        ))
        return cur.lastrowid


def list_metrics(account_id: str, active_only: bool = True) -> list[dict]:
    from store.db import get_db
    with get_db() as conn:
        _ensure_metric_registry_schema(conn)
        if active_only:
            rows = conn.execute(
                "SELECT * FROM metric_registry WHERE account_id=? AND is_active=1 ORDER BY name",
                (account_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM metric_registry WHERE account_id=? ORDER BY is_active DESC, name",
                (account_id,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_metric(metric_id: int) -> dict | None:
    from store.db import get_db
    with get_db() as conn:
        _ensure_metric_registry_schema(conn)
        row = conn.execute(
            "SELECT * FROM metric_registry WHERE id=?", (metric_id,)
        ).fetchone()
    return dict(row) if row else None


def update_metric(
    metric_id: int,
    updates: dict,
    *,
    account_id: str = "",
    db_type: str = "azure_sql",
) -> None:
    """
    Update a metric. Enforces account_id ownership and re-runs validation
    whenever sql_template or formula_type changes.
    """
    import json as _json
    from store.db import get_db
    from core.metric_validator import validate_metric, derive_metric_status, load_schema_columns

    if "formula_type" in updates:
        updates["formula_type"] = _metric_choice(
            updates.get("formula_type", "query"), {"query", "expression"}, "query"
        )
    if "result_format" in updates:
        updates["result_format"] = _metric_choice(
            updates.get("result_format", "number"),
            {"number", "currency", "percentage", "date", "text"},
            "number",
        )

    # Re-validate whenever the formula changes
    formula_fields = {"sql_template", "formula_type", "required_columns",
                      "base_table", "base_entity", "metric_builder_config"}
    needs_revalidation = bool(formula_fields & set(updates.keys()))
    if needs_revalidation:
        existing = get_metric(metric_id) or {}
        merged   = {**existing, **updates}
        schema_columns = load_schema_columns(account_id) if account_id else None
        vr = validate_metric(merged, db_type=db_type, schema_columns=schema_columns)
        updates["metric_status"]     = derive_metric_status(vr, merged.get("formula_type", "query"))
        updates["formula_ast"]       = _json.dumps(vr.formula_ast)
        updates["validation_errors"] = _json.dumps(vr.errors)
        updates["last_validated_at"] = "datetime('now')"  # handled below as literal

    fields, params = [], []
    for key in (
        "name", "synonyms", "sql_template", "description", "formula_type",
        "result_format", "required_columns", "allowed_dimensions",
        "metric_builder_config", "example_questions", "grain", "category", "default_time_column",
        "is_active", "base_entity", "base_table", "metric_status",
        "formula_ast", "validation_errors", "owner",
    ):
        if key in updates:
            fields.append(f"{key}=?")
            params.append(updates[key])

    # Increment version on every content-changing update
    if needs_revalidation:
        fields.append("version = COALESCE(version, 0) + 1")
        fields.append("last_validated_at = datetime('now')")

    if not fields:
        return
    fields.append("updated_at = datetime('now')")

    # Enforce account_id ownership in WHERE clause when provided
    if account_id:
        params.extend([metric_id, account_id])
        where = "id=? AND account_id=?"
    else:
        params.append(metric_id)
        where = "id=?"

    with get_db() as conn:
        _ensure_metric_registry_schema(conn)
        conn.execute(
            f"UPDATE metric_registry SET {','.join(fields)} WHERE {where}", params
        )


def deprecate_metric(metric_id: int, account_id: str) -> None:
    """
    Soft-delete a metric by setting status to 'deprecated' and is_active=0.
    Preserves the row and all version history. Preferred over hard delete.
    """
    from store.db import get_db
    with get_db() as conn:
        conn.execute(
            "UPDATE metric_registry SET metric_status='deprecated', is_active=0, "
            "updated_at=datetime('now') WHERE id=? AND account_id=?",
            (metric_id, account_id),
        )


def delete_metric(metric_id: int, account_id: str = "") -> None:
    """
    Hard-delete a metric. Prefer deprecate_metric() unless the metric was
    created by mistake. Enforces account_id ownership when provided.
    """
    from store.db import get_db
    with get_db() as conn:
        if account_id:
            conn.execute(
                "DELETE FROM metric_registry WHERE id=? AND account_id=?",
                (metric_id, account_id),
            )
        else:
            conn.execute("DELETE FROM metric_registry WHERE id=?", (metric_id,))


def _metric_phrases(metric: dict) -> list[str]:
    import re
    raw = [
        metric.get("name", ""),
        metric.get("synonyms", ""),
        metric.get("description", ""),
        metric.get("example_questions", ""),
    ]
    phrases: list[str] = []
    for value in raw:
        for item in re.split(r"[,;\n]+", str(value or "")):
            item = item.strip().lower().replace("_", " ")
            if item:
                phrases.append(item)
    return phrases


def _metric_tokens(text: str) -> set[str]:
    import re
    stop = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "how",
        "in", "is", "me", "of", "on", "or", "per", "show", "the", "to",
        "total", "what", "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower().replace("_", " "))
        if len(token) > 2 and token not in stop
    }


def _score_metric_for_question(metric: dict, question: str) -> int:
    q = (question or "").lower().replace("_", " ")
    q_tokens = _metric_tokens(q)
    if not q_tokens:
        return 0

    score = 0
    for phrase in _metric_phrases(metric):
        phrase_tokens = _metric_tokens(phrase)
        if phrase and phrase in q:
            score += 8
        overlap = q_tokens & phrase_tokens
        if overlap:
            score += len(overlap) * 3

    metadata_tokens = _metric_tokens(" ".join([
        metric.get("required_columns", ""),
        metric.get("allowed_dimensions", ""),
        metric.get("grain", ""),
    ]))
    score += len(q_tokens & metadata_tokens)
    return score


def list_metric_formula_context(account_id: str, question: str, limit: int = 6) -> list[dict]:
    """
    Return active metric definitions relevant to a user question.

    Unlike match_metric(), this does not skip grouped questions. The caller
    injects these approved formulas into the SQL prompt so grouped requests
    such as "profit percentage by customer" can use the trusted calculation.
    """
    scored: list[tuple[int, dict]] = []
    for metric in list_metrics(account_id):
        score = _score_metric_for_question(metric, question)
        if score > 0:
            metric = dict(metric)
            metric["_score"] = score
            scored.append((score, metric))
    scored.sort(key=lambda item: (-item[0], item[1].get("name", "")))
    return [metric for _, metric in scored[: max(1, int(limit or 6))]]


def match_metric(account_id: str, question: str) -> dict | None:
    """
    Check if the question matches any trusted full-SQL metric by synonym lookup.
    Returns the metric dict if matched, None otherwise.
    Case-insensitive. Matches whole words only to avoid false positives.

    Skips metric registry when the question contains grouping/dimension keywords
    (by, per, grouped by, breakdown, each) — those need the LLM to generate a
    GROUP BY query, not a simple aggregate from the stored template SQL.
    """
    import re
    q_lower = question.lower()

    # Skip metric registry for grouping/dimension questions
    # These need LLM to generate GROUP BY — stored SQL won't have it
    grouping_patterns = [
        r"\bby\s+\w",        # "by prescriber", "by month", "by category"
        r"\bper\s+\w",       # "per doctor", "per department"
        r"\bgrouped\s+by\b", # "grouped by"
        r"\bbreakdown\b",     # "breakdown"
        r"\bsplit\s+by\b",   # "split by"
        r"\beach\s+\w",      # "each prescriber", "each department"
        r"\bfor\s+each\b",   # "for each"
    ]
    for pat in grouping_patterns:
        if re.search(pat, q_lower):
            return None  # Fall through to LLM for GROUP BY queries

    for metric in list_metrics(account_id):
        if (metric.get("formula_type") or "query").lower() != "query":
            continue
        if not (metric.get("sql_template") or "").lstrip().upper().startswith("SELECT"):
            continue
        synonyms = [s.strip().lower() for s in metric["synonyms"].split(",") if s.strip()]
        synonyms.append(metric["name"].lower())
        for syn in synonyms:
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, q_lower):
                return metric
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Validated examples store — Step 2
# ══════════════════════════════════════════════════════════════════════════════

def save_validated_example(account_id: str, question: str, sql: str, table_name: str = "") -> None:
    """Store a validated (question → SQL) pair for RAG retrieval."""
    from store.db import get_db
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validated_examples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id  TEXT    NOT NULL,
                question    TEXT    NOT NULL,
                sql_query   TEXT    NOT NULL,
                table_name  TEXT    DEFAULT '',
                source      TEXT    DEFAULT 'kb_stage2',
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        # Avoid exact duplicates
        existing = conn.execute(
            "SELECT id FROM validated_examples WHERE account_id=? AND question=?",
            (account_id, question)
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO validated_examples (account_id, question, sql_query, table_name, source)
                VALUES (?, ?, ?, ?, ?)
            """, (account_id, question, sql, table_name, "kb_stage2"))


def get_validated_examples(account_id: str, limit: int = 200) -> list[dict]:
    from store.db import get_db
    with get_db() as conn:
        try:
            rows = conn.execute("""
                SELECT * FROM validated_examples
                WHERE account_id=?
                ORDER BY created_at DESC LIMIT ?
            """, (account_id, limit)).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


def harvest_successful_queries(account_id: str, days_back: int = 30) -> int:
    """
    Step 4 — Query log harvesting.
    Copy successful queries from query_log into validated_examples.
    Returns number of new examples added.
    """
    from store.db import get_db
    added = 0
    with get_db() as conn:
        rows = conn.execute("""
            SELECT question, sql_generated FROM query_log
            WHERE account_id=?
              AND success=1
              AND sql_generated != ''
              AND question != ''
              AND created_at >= datetime('now', ?)
            ORDER BY created_at DESC
        """, (account_id, f"-{days_back} days")).fetchall()

        for row in rows:
            existing = conn.execute(
                "SELECT id FROM validated_examples WHERE account_id=? AND question=?",
                (account_id, row["question"])
            ).fetchone()
            if not existing:
                try:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS validated_examples (
                            id          INTEGER PRIMARY KEY AUTOINCREMENT,
                            account_id  TEXT    NOT NULL,
                            question    TEXT    NOT NULL,
                            sql_query   TEXT    NOT NULL,
                            table_name  TEXT    DEFAULT '',
                            source      TEXT    DEFAULT 'query_log',
                            created_at  TEXT    DEFAULT (datetime('now'))
                        )
                    """)
                    conn.execute("""
                        INSERT INTO validated_examples
                            (account_id, question, sql_query, source)
                        VALUES (?, ?, ?, 'query_log')
                    """, (account_id, row["question"], row["sql_generated"]))
                    added += 1
                except Exception:
                    pass
    return added


# ══════════════════════════════════════════════════════════════════════════════
# KB data egress log
# Records which tables were processed during schema discovery and KB build,
# and whether real or synthetic data was sent to the LLM.
# Append-only — existing rows are never updated.
# ══════════════════════════════════════════════════════════════════════════════

def log_kb_egress(
    account_id: str,
    operation: str,           # 'discovery' | 'kb_build'
    db_type: str,
    table_name: str,
    sample_mode: str,         # 'masked' | 'real' | 'synthetic' | 'none'
    database_name: str = "",
    schema_name: str = "",
    column_count: int = 0,
    distinct_col_count: int = 0,
    triggered_by: str = "admin",
    fields_sent: list | None = None,
    row_count_sent: int = 0,
    masked_fields: list | None = None,
    mask_mode: str = "none",
    mask_replacement_map: dict | None = None,
) -> None:
    """
    Write one row to kb_data_egress_log for a single processed table.

    sample_mode values:
      'masked'     — real rows fetched, PII fields replaced locally before sending
      'real'       — actual production rows sent unchanged (admin explicit opt-in)
      'synthetic'  — fully fake rows (fallback: DB unreachable or admin forced)
      'none'       — schema-only, no sample rows sent

    fields_sent          — every column name present in the LLM prompt
    row_count_sent       — number of sample rows sent (0 for synthetic / schema-only)
    masked_fields        — subset of fields_sent that had masking applied
    mask_mode            — 'none' | 'all' | 'selective' | 'auto'
    mask_replacement_map — {field_name: strategy_name} for every masked field
                           e.g. {"EMAIL": "email", "FIRST_NAME": "first_name"}
    """
    import json as _json
    # Compliance alert: real, unmasked production rows leaving for the LLM is a
    # high-signal event — surface it loudly so the audit row is paired with an
    # operational alert, not just a silent DB write. (B4)
    if sample_mode == "real":
        log.critical(
            "EGRESS ALERT: real unmasked sample rows sent to LLM — "
            "account=%s table=%s op=%s rows=%d cols=%d triggered_by=%s",
            account_id, table_name, operation, row_count_sent,
            column_count, triggered_by,
        )
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO kb_data_egress_log
                    (account_id, operation, db_type,
                     database_name, schema_name, table_name,
                     column_count, sample_mode, distinct_col_count, triggered_by,
                     fields_sent, row_count_sent, masked_fields, mask_mode,
                     mask_replacement_map)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                account_id, operation, db_type,
                database_name or "", schema_name or "", table_name,
                column_count, sample_mode, distinct_col_count, triggered_by,
                _json.dumps(fields_sent or []),
                row_count_sent,
                _json.dumps(masked_fields or []),
                mask_mode,
                _json.dumps(mask_replacement_map or {}),
            ))
    except Exception as exc:
        log.warning("kb_data_egress_log write failed for %s/%s: %s",
                    account_id, table_name, exc)


def update_egress_masking(
    account_id: str,
    masking_config: dict,
) -> int:
    """
    Immediately reflect a saved masking config in the most recent egress log row
    for each configured table. Updates sample_mode → 'masked', masked_fields, and
    mask_mode so the admin egress panel shows the change without waiting for KB rebuild.

    masking_config keys are FQN strings (SCHEMA.TABLE or TABLE), values are
    dicts with 'mode' and 'masked_fields'.

    Returns the number of rows updated.
    """
    import json as _json
    updated = 0
    try:
        # Resolve masking strategy names for each field so we can store the
        # replacement map even when updating without re-running discovery.
        from core.masking import get_strategy_map as _get_strategy_map

        with get_db() as conn:
            for fqn, cfg in masking_config.items():
                if not isinstance(cfg, dict):
                    continue
                mode = cfg.get("mode", "selective")
                mf   = cfg.get("masked_fields") or []
                # Derive table_name from the last segment of the FQN
                table_name = fqn.split(".")[-1].upper()
                sample_mode = "masked" if (mode != "none" and mf) else "real"
                # Build replacement map: field → strategy (no col_defs available
                # here, so we use name-only detection via empty col_defs list —
                # fields without a matching PII pattern get "text_mask" fallback)
                replacement_map = _get_strategy_map(set(mf), [])
                conn.execute("""
                    UPDATE kb_data_egress_log
                    SET sample_mode          = ?,
                        masked_fields        = ?,
                        mask_mode            = ?,
                        mask_replacement_map = ?
                    WHERE id = (
                        SELECT id FROM kb_data_egress_log
                        WHERE account_id = ?
                          AND UPPER(table_name) = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                """, (
                    sample_mode,
                    _json.dumps(mf),
                    mode,
                    _json.dumps(replacement_map),
                    account_id,
                    table_name,
                ))
                updated += conn.execute("SELECT changes()").fetchone()[0]
    except Exception as exc:
        log.warning("update_egress_masking failed for %s: %s", account_id, exc)
    return updated


def list_kb_egress(
    account_id: str,
    operation: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """
    Return recent egress log rows for a client, newest first.
    Optionally filtered to a specific operation ('discovery' or 'kb_build').
    """
    params: list = [account_id]
    where = "WHERE account_id = ?"
    if operation:
        where += " AND operation = ?"
        params.append(operation)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM kb_data_egress_log {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def get_kb_egress_summary(account_id: str) -> dict:
    """
    Return a summary dict for the admin client setup page:
      last_discovery_at     — ISO timestamp of the most recent discovery run
      last_kb_build_at      — ISO timestamp of the most recent KB build
      total_tables_discovery — total tables ever processed in discovery
      total_tables_kb_build  — total tables ever processed in KB build
      real_sample_count      — tables where real rows reached the LLM
      synthetic_sample_count — tables where synthetic rows were used
      discovery_rows         — list[dict] of the most recent discovery run (newest first)
      kb_build_rows          — list[dict] of the most recent KB build run (newest first)
    """
    with get_db() as conn:
        agg = conn.execute("""
            SELECT
                MAX(CASE WHEN operation='discovery' THEN created_at END) AS last_discovery_at,
                MAX(CASE WHEN operation='kb_build'  THEN created_at END) AS last_kb_build_at,
                SUM(CASE WHEN operation='discovery' THEN 1 ELSE 0 END)   AS total_tables_discovery,
                SUM(CASE WHEN operation='kb_build'  THEN 1 ELSE 0 END)   AS total_tables_kb_build,
                SUM(CASE WHEN sample_mode='real'      THEN 1 ELSE 0 END) AS real_sample_count,
                SUM(CASE WHEN sample_mode='synthetic' THEN 1 ELSE 0 END) AS synthetic_sample_count,
                SUM(CASE WHEN sample_mode='masked'    THEN 1 ELSE 0 END) AS masked_sample_count
            FROM kb_data_egress_log
            WHERE account_id = ?
        """, (account_id,)).fetchone()

    if agg:
        raw = dict(agg)
        # SQLite SUM/MAX return NULL for empty sets — normalise to 0/""
        summary = {
            "last_discovery_at":     raw.get("last_discovery_at") or "",
            "last_kb_build_at":      raw.get("last_kb_build_at")  or "",
            "total_tables_discovery":int(raw.get("total_tables_discovery") or 0),
            "total_tables_kb_build": int(raw.get("total_tables_kb_build")  or 0),
            "real_sample_count":     int(raw.get("real_sample_count")      or 0),
            "synthetic_sample_count":int(raw.get("synthetic_sample_count") or 0),
            "masked_sample_count":   int(raw.get("masked_sample_count")    or 0),
        }
    else:
        summary = {
            "last_discovery_at": "", "last_kb_build_at": "",
            "total_tables_discovery": 0, "total_tables_kb_build": 0,
            "real_sample_count": 0, "synthetic_sample_count": 0,
            "masked_sample_count": 0,
        }

    # Fetch rows from the latest discovery run (identified by its timestamp bucket)
    latest_discovery = summary.get("last_discovery_at") or ""
    latest_kb_build  = summary.get("last_kb_build_at")  or ""

    disc_rows: list[dict] = []
    kb_rows:   list[dict] = []

    if latest_discovery:
        # Rows within 5 minutes of the latest discovery timestamp = same run
        with get_db() as conn:
            disc_rows = [dict(r) for r in conn.execute("""
                SELECT * FROM kb_data_egress_log
                WHERE account_id = ?
                  AND operation = 'discovery'
                  AND datetime(created_at) >= datetime(?, '-5 minutes')
                ORDER BY created_at DESC
            """, (account_id, latest_discovery)).fetchall()]

    if latest_kb_build:
        with get_db() as conn:
            kb_rows = [dict(r) for r in conn.execute("""
                SELECT * FROM kb_data_egress_log
                WHERE account_id = ?
                  AND operation = 'kb_build'
                  AND datetime(created_at) >= datetime(?, '-5 minutes')
                ORDER BY created_at DESC
            """, (account_id, latest_kb_build)).fetchall()]

    import json as _json

    def _parse_row(row: dict) -> dict:
        for key, dest, default in (
            ("fields_sent",          "fields_sent_list",      "[]"),
            ("masked_fields",        "masked_fields_list",    "[]"),
            ("mask_replacement_map", "replacement_map_parsed", "{}"),
        ):
            raw = row.get(key) or default
            try:
                row[dest] = _json.loads(raw) if isinstance(raw, str) else (raw or ([] if default == "[]" else {}))
            except Exception:
                row[dest] = [] if default == "[]" else {}
        return row

    summary["discovery_rows"] = [_parse_row(r) for r in disc_rows]
    summary["kb_build_rows"]  = [_parse_row(r) for r in kb_rows]
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Entity graph — CRUD for entities, relationships, and properties
# ══════════════════════════════════════════════════════════════════════════════

def save_entity(
    account_id: str,
    entity_name: str,
    table_name: str,
    schema_name: str = "",
    pk_column: str = "",
    display_name: str = "",
    description: str = "",
    entity_type: str = "dimension",
    is_active: int = 1,
    pos_x: float = 120,
    pos_y: float = 120,
    color: str = "#4F86C6",
    confidence_score: int = 100,
    status: str = "confirmed",
    entity_filter: str = "",
    generated_by: str = "manual",
    reason: str = "",
) -> int:
    """Upsert an entity. Returns its id.
    status: 'suggested' = LLM prediction awaiting admin review
            'confirmed' = admin-approved, feeds into SQL generation
    entity_filter: static SQL WHERE conditions always applied when this table is joined
    generated_by/reason: provenance evidence ('manual' | 'heuristic' | 'llm') and a
        human-readable explanation of why this entity was suggested. Written on
        INSERT only — updates never overwrite the original provenance.
    """
    with get_db() as conn:
        conn.execute("""
            INSERT INTO entity_graph
                (account_id, entity_name, table_name, schema_name, pk_column,
                 display_name, description, entity_type, is_active,
                 pos_x, pos_y, color, confidence_score, status, entity_filter,
                 generated_by, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, entity_name) DO UPDATE SET
                table_name       = excluded.table_name,
                schema_name      = excluded.schema_name,
                pk_column        = excluded.pk_column,
                display_name     = excluded.display_name,
                description      = excluded.description,
                entity_type      = excluded.entity_type,
                is_active        = excluded.is_active,
                pos_x            = excluded.pos_x,
                pos_y            = excluded.pos_y,
                color            = excluded.color,
                confidence_score = excluded.confidence_score,
                status           = excluded.status,
                entity_filter    = excluded.entity_filter
        """, (account_id, entity_name, table_name, schema_name, pk_column,
              display_name, description, entity_type, is_active,
              pos_x, pos_y, color, confidence_score, status, entity_filter,
              generated_by, reason))
        row = conn.execute(
            "SELECT id FROM entity_graph WHERE account_id=? AND entity_name=?",
            (account_id, entity_name)
        ).fetchone()
    return row["id"] if row else -1


def list_entities(account_id: str, active_only: bool = True) -> list[dict]:
    where = "WHERE account_id = ?" + (" AND is_active = 1" if active_only else "")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM entity_graph {where} ORDER BY entity_type DESC, entity_name",
            (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_entity(account_id: str, entity_name: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM entity_graph WHERE account_id=? AND entity_name=?",
            (account_id, entity_name)
        ).fetchone()
    return dict(row) if row else None


def delete_entity(account_id: str, entity_name: str) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM entity_graph WHERE account_id=? AND entity_name=?",
            (account_id, entity_name)
        )
        conn.execute(
            "DELETE FROM entity_relationships WHERE account_id=? AND (from_entity=? OR to_entity=?)",
            (account_id, entity_name, entity_name)
        )
        conn.execute(
            "DELETE FROM entity_properties WHERE account_id=? AND entity_name=?",
            (account_id, entity_name)
        )


def _split_table_ref(ref: str, fallback_schema: str = "") -> tuple[str, str]:
    parts = [
        p.strip().strip('"`[]').upper()
        for p in str(ref or "").split(".")
        if p.strip().strip('"`[]')
    ]
    table = parts[-1] if parts else ""
    schema = (
        str(fallback_schema or "").strip().strip('"`[]').upper()
        or (parts[-2] if len(parts) >= 2 else "")
    )
    return schema, table


def prune_entity_graph_to_tables(account_id: str, table_refs: list[str] | set[str] | tuple[str, ...]) -> dict:
    """
    Remove graph entities that no longer belong to the current discovered schema.

    KB/schema rebuilds are full replacements, but the entity graph is persisted
    separately because admins can edit it. This keeps admin edits for tables
    still present in the latest _schema.json while deleting stale entities,
    relationships, and properties for tables dropped from the KB scope.
    """
    allowed_refs = [str(t or "").strip() for t in (table_refs or []) if str(t or "").strip()]
    if not allowed_refs:
        return {"entities_removed": 0, "relationships_removed": 0, "properties_removed": 0}

    allowed_exact = {_split_table_ref(ref) for ref in allowed_refs}
    allowed_bare = {table for _schema, table in allowed_exact if table}
    allowed_exact = {pair for pair in allowed_exact if pair[1]}

    with get_db() as conn:
        rows = conn.execute(
            "SELECT entity_name, table_name, schema_name FROM entity_graph WHERE account_id=?",
            (account_id,),
        ).fetchall()
        stale_names: list[str] = []
        for row in rows:
            ent = dict(row)
            schema, table = _split_table_ref(ent.get("table_name", ""), ent.get("schema_name", ""))
            if not table:
                continue
            if schema:
                keep = (schema, table) in allowed_exact
            else:
                # Legacy/manual graph rows may have only a bare table name.
                keep = table in allowed_bare
            if not keep:
                stale_names.append(ent["entity_name"])

        if not stale_names:
            return {"entities_removed": 0, "relationships_removed": 0, "properties_removed": 0}

        placeholders = ",".join("?" for _ in stale_names)
        rel_removed = conn.execute(
            f"""
            DELETE FROM entity_relationships
             WHERE account_id=?
               AND (from_entity IN ({placeholders}) OR to_entity IN ({placeholders}))
            """,
            (account_id, *stale_names, *stale_names),
        ).rowcount
        prop_removed = conn.execute(
            f"DELETE FROM entity_properties WHERE account_id=? AND entity_name IN ({placeholders})",
            (account_id, *stale_names),
        ).rowcount
        ent_removed = conn.execute(
            f"DELETE FROM entity_graph WHERE account_id=? AND entity_name IN ({placeholders})",
            (account_id, *stale_names),
        ).rowcount

    return {
        "entities_removed": max(ent_removed, 0),
        "relationships_removed": max(rel_removed, 0),
        "properties_removed": max(prop_removed, 0),
    }


def save_relationship(
    account_id: str,
    from_entity: str,
    to_entity: str,
    from_column: str,
    to_column: str,
    relationship_type: str = "many_to_one",
    confidence_score: int = 100,
    status: str = "confirmed",
    join_type: str = "INNER",
    label: str = "",
    rel_id: int = 0,
    join_conditions: list | None = None,
    where_clause: str = "",
    generated_by: str = "manual",
    reason: str = "",
) -> int:
    """Insert or update a relationship edge. Returns its id.
    generated_by/reason: provenance evidence ('manual' | 'heuristic' | 'llm') and a
    human-readable explanation (e.g. "Shared column CUSTOMER_ID"). Written on
    INSERT only — edits never overwrite the original provenance.
    """
    import json as _json
    jc_json = _json.dumps(join_conditions or [])
    with get_db() as conn:
        if rel_id:
            conn.execute("""
                UPDATE entity_relationships SET
                    from_entity=?, to_entity=?, from_column=?, to_column=?,
                    relationship_type=?, join_type=?, label=?, join_conditions=?,
                    where_clause=?,
                    validation_status='untested',
                    validated_at='',
                    row_count_estimate=-1,
                    join_multiplicity=''
                WHERE id=? AND account_id=?
            """, (from_entity, to_entity, from_column, to_column,
                  relationship_type, join_type, label, jc_json,
                  where_clause, rel_id, account_id))
            return rel_id
        conn.execute("""
            INSERT INTO entity_relationships
                (account_id, from_entity, to_entity, from_column, to_column,
                 relationship_type, join_type, label, is_active,
                 confidence_score, status, join_conditions, where_clause,
                 generated_by, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
        """, (account_id, from_entity, to_entity, from_column, to_column,
              relationship_type, join_type, label, confidence_score, status,
              jc_json, where_clause, generated_by, reason))
        row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
    return row["id"] if row else -1


def get_relationship(account_id: str, rel_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM entity_relationships WHERE account_id=? AND id=?",
            (account_id, rel_id),
        ).fetchone()
    return dict(row) if row else None


def list_relationships(account_id: str, active_only: bool = True) -> list[dict]:
    where = "WHERE account_id = ?" + (" AND is_active = 1" if active_only else "")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM entity_relationships {where} ORDER BY from_entity, to_entity",
            (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def update_relationship_validation(
    account_id: str,
    rel_id: int,
    validation_status: str,
    *,
    row_count_estimate: int = -1,
    join_multiplicity: str = "",
) -> None:
    allowed = {"untested", "valid", "warning", "broken"}
    status_value = validation_status if validation_status in allowed else "untested"
    with get_db() as conn:
        conn.execute("""
            UPDATE entity_relationships
               SET validation_status=?,
                   validated_at=datetime('now'),
                   row_count_estimate=?,
                   join_multiplicity=?
             WHERE account_id=? AND id=?
        """, (
            status_value,
            int(row_count_estimate),
            join_multiplicity or "",
            account_id,
            rel_id,
        ))


def delete_relationship(account_id: str, rel_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM entity_relationships WHERE id=? AND account_id=?",
            (rel_id, account_id)
        )


def save_entity_property(
    account_id: str,
    entity_name: str,
    column_name: str,
    role: str = "dimension",
    display_name: str = "",
    synonyms: str = "",
    confidence_score: int = 100,
    status: str = "confirmed",
) -> None:
    """Save a column role mapping.
    status: 'suggested' = LLM prediction, 'confirmed' = admin-approved.
    Confirmed fields are synced to the semantic layer business_term table.
    """
    with get_db() as conn:
        conn.execute("""
            INSERT INTO entity_properties
                (account_id, entity_name, column_name, role,
                 display_name, synonyms, confidence_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_id, entity_name, column_name) DO UPDATE SET
                role             = excluded.role,
                display_name     = excluded.display_name,
                synonyms         = excluded.synonyms,
                confidence_score = excluded.confidence_score,
                status           = excluded.status
        """, (account_id, entity_name, column_name, role,
              display_name, synonyms, confidence_score, status))




def confirm_entity_property(
    account_id: str,
    entity_name: str,
    column_name: str,
) -> None:
    """
    Admin confirms a field role — sets status='confirmed', confidence_score=100,
    and writes an 'approved' row into semantic_field_feedback so the semantic
    layer immediately reflects the admin-verified meaning.
    This short-circuits the user-portal feedback loop: instead of waiting
    for a user to notice a wrong synonym and submit feedback, the admin
    resolves it here before any user query runs.
    """
    with get_db() as conn:
        conn.execute("""
            UPDATE entity_properties
               SET status='confirmed', confidence_score=100
             WHERE account_id=? AND entity_name=? AND column_name=?
        """, (account_id, entity_name, column_name))
        # Read the current property to get display_name / synonyms for sync
        row = conn.execute("""
            SELECT * FROM entity_properties
             WHERE account_id=? AND entity_name=? AND column_name=?
        """, (account_id, entity_name, column_name)).fetchone()

    if not row:
        return
    prop = dict(row)

    # Sync to semantic layer: upsert business_term
    if prop.get("display_name"):
        try:
            save_term(
                account_id   = account_id,
                term         = prop["display_name"].strip(),
                column_name  = column_name,
                table_hint   = entity_name,
                is_active    = 1,
                source       = "entity_graph",
            )
        except Exception:
            pass

    # Write to semantic_field_feedback as admin-approved at 100%
    # This makes the field show as 100% confirmed in the user portal too
    try:
        with get_db() as conn:
            # Get entity table name for table_fqn
            ent_row = conn.execute(
                "SELECT table_name, schema_name FROM entity_graph "
                "WHERE account_id=? AND entity_name=?",
                (account_id, entity_name)
            ).fetchone()
            if ent_row:
                ent = dict(ent_row)
                tbl_fqn = (
                    f"{ent['schema_name']}.{ent['table_name']}"
                    if ent.get("schema_name")
                    else ent["table_name"]
                )
                conn.execute("""
                    INSERT INTO semantic_field_feedback
                        (account_id, table_fqn, table_name, schema_name,
                         column_name, current_meaning, suggested_meaning,
                         confidence_score, status, reviewed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 100, 'approved', datetime('now'))
                    ON CONFLICT DO NOTHING
                """, (
                    account_id, tbl_fqn,
                    ent["table_name"], ent.get("schema_name",""),
                    column_name,
                    prop.get("display_name",""),
                    prop.get("synonyms",""),
                ))
    except Exception:
        pass  # semantic_field_feedback sync is best-effort

def list_entity_properties(account_id: str, entity_name: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM entity_properties WHERE account_id=? AND entity_name=? ORDER BY column_name",
            (account_id, entity_name)
        ).fetchall()
    return [dict(r) for r in rows]


def list_all_entity_properties(account_id: str) -> list[dict]:
    """All column-role properties for an account (every entity), for the resolver."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM entity_properties WHERE account_id=? ORDER BY entity_name, column_name",
            (account_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_full_graph(account_id: str) -> dict:
    """Return the complete graph as {entities, relationships, properties} for the resolver."""
    return {
        "entities":      list_entities(account_id, active_only=True),
        "relationships": list_relationships(account_id, active_only=True),
        "properties":    list_all_entity_properties(account_id),
    }
