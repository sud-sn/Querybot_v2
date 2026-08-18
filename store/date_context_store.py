"""Governed metric and business-context date bindings."""

from __future__ import annotations

import logging

from store.db import get_db, get_table_columns

log = logging.getLogger("querybot.store.date_context")


def _ensure_table(conn) -> None:
    conn.executescript(
        """
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
        CREATE INDEX IF NOT EXISTS idx_metric_date_context_account
            ON metric_date_context(account_id, metric_id, is_active);
        """
    )
    if "date_value_column" not in set(get_table_columns(conn, "metric_date_context")):
        conn.execute(
            "ALTER TABLE metric_date_context "
            "ADD COLUMN date_value_column TEXT NOT NULL DEFAULT ''"
        )
    if "date_key_type" not in set(get_table_columns(conn, "metric_date_context")):
        conn.execute(
            "ALTER TABLE metric_date_context "
            "ADD COLUMN date_key_type TEXT NOT NULL DEFAULT 'surrogate_fk'"
        )


def save_metric_date_context(account_id: str, binding: dict) -> int:
    """Create or replace one governed context binding for a metric."""
    metric_id = int(binding.get("metric_id") or 0)
    context_name = str(binding.get("context_name") or "").strip()
    required = {
        "metric_id": metric_id,
        "context_name": context_name,
        "date_role": str(binding.get("date_role") or "").strip(),
        "fact_table": str(binding.get("fact_table") or "").strip(),
        "fact_column": str(binding.get("fact_column") or "").strip(),
        "dimension_table": str(binding.get("dimension_table") or "").strip(),
        "dimension_key": str(binding.get("dimension_key") or "").strip(),
        "date_value_column": str(binding.get("date_value_column") or "").strip(),
        "date_key_type": str(binding.get("date_key_type") or "surrogate_fk").strip(),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Missing date context fields: " + ", ".join(missing))

    with get_db() as conn:
        _ensure_table(conn)
        metric = conn.execute(
            "SELECT id FROM metric_registry WHERE id=? AND account_id=? AND is_active=1",
            (metric_id, account_id),
        ).fetchone()
        if not metric:
            raise ValueError("Metric does not belong to this client or is inactive.")

        if int(binding.get("is_default") or 0):
            conn.execute(
                "UPDATE metric_date_context SET is_default=0, updated_at=datetime('now') "
                "WHERE account_id=? AND metric_id=?",
                (account_id, metric_id),
            )

        existing = conn.execute(
            "SELECT id FROM metric_date_context "
            "WHERE account_id=? AND metric_id=? AND lower(context_name)=lower(?)",
            (account_id, metric_id, context_name),
        ).fetchone()
        values = (
            str(binding.get("aliases") or "").strip(),
            required["date_role"],
            required["fact_table"],
            required["fact_column"],
            required["dimension_table"],
            required["dimension_key"],
            required["date_value_column"],
            required["date_key_type"],
            1 if binding.get("is_default") else 0,
            max(0, min(100, int(binding.get("priority") or 50))),
            1 if binding.get("is_active", True) else 0,
        )
        if existing:
            conn.execute(
                """
                UPDATE metric_date_context
                   SET aliases=?, date_role=?, fact_table=?, fact_column=?,
                       dimension_table=?, dimension_key=?, date_value_column=?, date_key_type=?, is_default=?,
                       priority=?, is_active=?, updated_at=datetime('now')
                 WHERE id=? AND account_id=?
                """,
                (*values, int(existing["id"]), account_id),
            )
            return int(existing["id"])

        cur = conn.execute(
            """
            INSERT INTO metric_date_context (
                account_id, metric_id, context_name, aliases, date_role,
                fact_table, fact_column, dimension_table, dimension_key,
                date_value_column, date_key_type, is_default, priority, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (account_id, metric_id, context_name, *values),
        )
        return int(cur.lastrowid)


def list_metric_date_contexts(
    account_id: str,
    *,
    metric_ids: list[int] | None = None,
    active_only: bool = True,
) -> list[dict]:
    with get_db() as conn:
        _ensure_table(conn)
        where = ["b.account_id=?"]
        params: list = [account_id]
        if active_only:
            where.append("b.is_active=1")
        clean_ids = sorted({int(value) for value in (metric_ids or []) if int(value or 0) > 0})
        if clean_ids:
            placeholders = ",".join("?" for _ in clean_ids)
            where.append(f"b.metric_id IN ({placeholders})")
            params.extend(clean_ids)
        rows = conn.execute(
            f"""
            SELECT b.*, m.name AS metric_name
              FROM metric_date_context b
              JOIN metric_registry m ON m.id=b.metric_id AND m.account_id=b.account_id
             WHERE {' AND '.join(where)}
             ORDER BY m.name, b.is_default DESC, b.priority DESC, b.context_name
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_metric_date_context(binding_id: int, account_id: str) -> dict | None:
    with get_db() as conn:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT * FROM metric_date_context WHERE id=? AND account_id=?",
            (binding_id, account_id),
        ).fetchone()
    return dict(row) if row else None


def delete_metric_date_context(binding_id: int, account_id: str) -> bool:
    with get_db() as conn:
        _ensure_table(conn)
        cur = conn.execute(
            "DELETE FROM metric_date_context WHERE id=? AND account_id=?",
            (binding_id, account_id),
        )
        return bool(cur.rowcount)


# ══════════════════════════════════════════════════════════════════════════════
# Business date anchor — durable across restarts
# ══════════════════════════════════════════════════════════════════════════════
#
# "The latest business date present in this fact" is a property of the WAREHOUSE
# LOAD, not of a process. Holding it only in memory meant every restart threw it
# away and the next relative-date question paid the full probe again — 800s on a
# 9.2M-row unindexed fact on a starved instance, with the user watching a
# spinner and no explanation.
#
# Persisting it makes the probe run once per load rather than once per process,
# which is what allows a slow warehouse to be handed to a client for testing.

def load_business_date_anchor(
    account_id: str, fact_table: str, fact_column: str,
) -> dict:
    """Return the stored anchor for this fact+key, or {} when none is stored."""
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT anchor_value, date_column, source, resolved_at
                     FROM business_date_anchor
                    WHERE account_id=? AND fact_table=? AND fact_column=?""",
                (str(account_id), str(fact_table).upper(), str(fact_column).upper()),
            ).fetchone()
    except Exception as exc:                                  # pragma: no cover
        log.warning(
            "Business date anchor read failed for %s (%s.%s): %s — the probe "
            "will run again, which is slow but correct",
            account_id, fact_table, fact_column, exc,
        )
        return {}
    if not row or not row["anchor_value"]:
        return {}
    return {
        "value": row["anchor_value"],
        "fact_table": fact_table,
        "fact_column": fact_column,
        "date_column": row["date_column"] or "",
        "source": row["source"] or "probed_from_fact_rows",
        "resolved_at": row["resolved_at"] or "",
        "restored_from": "store",
    }


def save_business_date_anchor(
    account_id: str, fact_table: str, fact_column: str, anchor: dict,
) -> None:
    """Persist a resolved anchor so a restart does not re-probe the warehouse."""
    value = str((anchor or {}).get("value") or "").strip()
    if not value:
        return
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO business_date_anchor
                       (account_id, fact_table, fact_column, anchor_value,
                        date_column, source, probe_ms, resolved_at)
                   VALUES (?,?,?,?,?,?,?, datetime('now'))
                   ON CONFLICT(account_id, fact_table, fact_column) DO UPDATE SET
                       anchor_value = excluded.anchor_value,
                       date_column  = excluded.date_column,
                       source       = excluded.source,
                       probe_ms     = excluded.probe_ms,
                       resolved_at  = excluded.resolved_at""",
                (
                    str(account_id), str(fact_table).upper(), str(fact_column).upper(),
                    value, str(anchor.get("date_column") or ""),
                    str(anchor.get("source") or "probed_from_fact_rows"),
                    int(anchor.get("probe_ms") or 0),
                ),
            )
    except Exception as exc:                                  # pragma: no cover
        # Loud: a silent failure here reintroduces the 800s probe on every
        # restart, and nothing else would ever report it.
        log.warning(
            "Business date anchor write failed for %s (%s.%s): %s — the anchor "
            "will not survive a restart",
            account_id, fact_table, fact_column, exc,
        )


def clear_business_date_anchor(account_id: str, fact_table: str = "") -> int:
    """Forget stored anchors — after a warehouse reload, or from the admin UI."""
    try:
        with get_db() as conn:
            if fact_table:
                cur = conn.execute(
                    "DELETE FROM business_date_anchor WHERE account_id=? AND fact_table=?",
                    (str(account_id), str(fact_table).upper()),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM business_date_anchor WHERE account_id=?",
                    (str(account_id),),
                )
            return int(cur.rowcount or 0)
    except Exception as exc:                                  # pragma: no cover
        log.warning("Business date anchor clear failed for %s: %s", account_id, exc)
        return 0
