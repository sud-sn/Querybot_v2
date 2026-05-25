"""
Relationship validation for the entity graph.

This module validates joins in two layers:
1. Structural validation against the discovered _schema.json.
2. Optional live DB probe that executes a bounded COUNT query.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("querybot.relationship_validator")

STATUS_UNTESTED = "untested"
STATUS_VALID = "valid"
STATUS_WARNING = "warning"
STATUS_BROKEN = "broken"


@dataclass
class RelationshipValidationResult:
    relationship_id: int
    status: str
    message: str
    checked_by: str = "schema"
    probe_sql: str = ""
    row_count_estimate: int = -1
    join_multiplicity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "status": self.status,
            "message": self.message,
            "checked_by": self.checked_by,
            "probe_sql": self.probe_sql,
            "row_count_estimate": self.row_count_estimate,
            "join_multiplicity": self.join_multiplicity,
        }


def validate_relationship(
    account_id: str,
    rel_id: int,
    *,
    execute: bool = False,
    timeout_seconds: int = 20,
) -> RelationshipValidationResult:
    import store

    rel = store.get_relationship(account_id, rel_id)
    if not rel:
        return RelationshipValidationResult(
            rel_id,
            STATUS_BROKEN,
            "Relationship was not found.",
        )

    entities = {e["entity_name"]: e for e in store.list_entities(account_id, active_only=False)}
    from_ent = entities.get(rel.get("from_entity", ""))
    to_ent = entities.get(rel.get("to_entity", ""))
    if not from_ent or not to_ent:
        return RelationshipValidationResult(
            rel_id,
            STATUS_BROKEN,
            "Relationship references an entity that no longer exists.",
        )

    schema_check = _validate_against_schema(account_id, rel, from_ent, to_ent)
    if schema_check.status == STATUS_BROKEN or not execute:
        return schema_check

    try:
        return _execute_probe(account_id, rel, from_ent, to_ent, timeout_seconds=timeout_seconds)
    except Exception as exc:
        log.warning("relationship live validation failed for %s/%s: %s", account_id, rel_id, exc)
        return RelationshipValidationResult(
            rel_id,
            STATUS_WARNING,
            f"Schema looks valid, but the live DB probe failed: {exc}",
            checked_by="schema",
            probe_sql=schema_check.probe_sql,
        )


def _validate_against_schema(
    account_id: str,
    rel: dict,
    from_ent: dict,
    to_ent: dict,
) -> RelationshipValidationResult:
    from core.graph_health import _load_schema, _resolve_fqn

    rel_id = int(rel.get("id") or 0)
    schema_columns = _load_schema(account_id)
    if not schema_columns:
        return RelationshipValidationResult(
            rel_id,
            STATUS_WARNING,
            "No discovered schema is available, so the join could not be checked against columns.",
        )

    schema_map: dict[str, dict] = {}
    for fqn, cols in schema_columns.items():
        schema_map[fqn.lower()] = {
            "columns": {c.lower() for c in cols},
            "original_fqn": fqn,
        }

    from_key = _resolve_fqn(from_ent.get("schema_name", ""), from_ent.get("table_name", ""), schema_map)
    to_key = _resolve_fqn(to_ent.get("schema_name", ""), to_ent.get("table_name", ""), schema_map)
    if not from_key:
        return RelationshipValidationResult(
            rel_id,
            STATUS_BROKEN,
            f"From entity table '{_display_table(from_ent)}' was not found in discovered schema.",
        )
    if not to_key:
        return RelationshipValidationResult(
            rel_id,
            STATUS_BROKEN,
            f"To entity table '{_display_table(to_ent)}' was not found in discovered schema.",
        )

    missing: list[str] = []
    pairs = _join_pairs(rel)
    for from_col, to_col in pairs:
        if from_col.lower() not in schema_map[from_key]["columns"]:
            missing.append(f"{from_ent['entity_name']}.{from_col}")
        if to_col.lower() not in schema_map[to_key]["columns"]:
            missing.append(f"{to_ent['entity_name']}.{to_col}")

    if missing:
        return RelationshipValidationResult(
            rel_id,
            STATUS_BROKEN,
            "Join column missing from discovered schema: " + ", ".join(missing),
        )

    probe_sql = build_probe_sql("azure_sql", rel, from_ent, to_ent)
    return RelationshipValidationResult(
        rel_id,
        STATUS_VALID,
        "Join structure is valid against the discovered schema.",
        checked_by="schema",
        probe_sql=probe_sql,
    )


def build_probe_sql(db_type: str, rel: dict, from_ent: dict, to_ent: dict) -> str:
    left_table = _quote_table(from_ent.get("schema_name", ""), from_ent.get("table_name", ""), db_type)
    right_table = _quote_table(to_ent.get("schema_name", ""), to_ent.get("table_name", ""), db_type)
    pairs = _join_pairs(rel)
    on_sql = " AND ".join(
        f"l.{_quote_col(left, db_type)} = r.{_quote_col(right, db_type)}"
        for left, right in pairs
    )
    join_type = (rel.get("join_type") or "INNER").upper()
    if join_type not in {"INNER", "LEFT"}:
        join_type = "INNER"

    if db_type == "azure_sql":
        return f"SELECT COUNT_BIG(1) AS row_count FROM {left_table} l {join_type} JOIN {right_table} r ON {on_sql}"
    return f"SELECT COUNT(*) AS row_count FROM {left_table} l {join_type} JOIN {right_table} r ON {on_sql}"


def _execute_probe(
    account_id: str,
    rel: dict,
    from_ent: dict,
    to_ent: dict,
    *,
    timeout_seconds: int,
) -> RelationshipValidationResult:
    import concurrent.futures
    import store
    from core.schema import _az_connect, _ora_connect, _sf_connect

    client = store.get_client(account_id)
    db_cfg_id = client.get("db_config_id") if client else None
    raw_cfg = store.get_db_config(db_cfg_id) if db_cfg_id else None
    if not raw_cfg:
        return RelationshipValidationResult(
            int(rel.get("id") or 0),
            STATUS_WARNING,
            "Schema is valid, but no database connection is assigned for a live probe.",
            checked_by="schema",
        )

    db_type = raw_cfg.get("db_type", "azure_sql")
    creds = raw_cfg.get("credentials", {})
    sql = build_probe_sql(db_type, rel, from_ent, to_ent)

    def _run() -> int:
        if db_type == "azure_sql":
            conn = _az_connect({**creds, "login_timeout": min(timeout_seconds, 20)}, max_retries=1)
        elif db_type == "snowflake":
            conn = _sf_connect(creds, max_retries=1)
        else:
            conn = _ora_connect(creds, max_retries=1)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0] if row else 0)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_run)
        row_count = future.result(timeout=timeout_seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if row_count <= 0:
        return RelationshipValidationResult(
            int(rel.get("id") or 0),
            STATUS_WARNING,
            "Join executed successfully but returned zero rows. Check whether this join is logically correct.",
            checked_by="db",
            probe_sql=sql,
            row_count_estimate=row_count,
            join_multiplicity="zero_match",
        )

    return RelationshipValidationResult(
        int(rel.get("id") or 0),
        STATUS_VALID,
        f"Join executed successfully and matched approximately {row_count} rows.",
        checked_by="db",
        probe_sql=sql,
        row_count_estimate=row_count,
        join_multiplicity="matched",
    )


def _join_pairs(rel: dict) -> list[tuple[str, str]]:
    pairs = [
        (
            str(rel.get("from_column") or "").strip(),
            str(rel.get("to_column") or "").strip(),
        )
    ]
    raw_extra = rel.get("join_conditions") or "[]"
    try:
        extra = json.loads(raw_extra) if isinstance(raw_extra, str) else raw_extra
    except Exception:
        extra = []
    for cond in extra or []:
        if not isinstance(cond, dict):
            continue
        left = str(cond.get("from_col") or "").strip()
        right = str(cond.get("to_col") or "").strip()
        if left and right and (left, right) not in pairs:
            pairs.append((left, right))
    return [(left, right) for left, right in pairs if left and right]


def _display_table(ent: dict) -> str:
    schema = (ent.get("schema_name") or "").strip()
    table = (ent.get("table_name") or "").strip()
    return f"{schema}.{table}" if schema else table


def _quote_table(schema_name: str, table_name: str, db_type: str) -> str:
    schema = (schema_name or "").strip()
    table = (table_name or "").strip()
    if db_type == "azure_sql":
        return f"[{schema}].[{table}]" if schema else f"[{table}]"
    if db_type == "snowflake":
        return f'"{schema}"."{table}"' if schema else f'"{table}"'
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _quote_col(column: str, db_type: str) -> str:
    col = (column or "").strip()
    if db_type == "azure_sql":
        return f"[{col}]"
    return f'"{col}"'
