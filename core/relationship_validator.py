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
    match_rate: float = -1.0
    orphan_rate: float = -1.0
    null_fk_rate: float = -1.0
    fanout_ratio: float = -1.0
    target_rows: int = -1
    target_distinct_keys: int = -1
    target_duplicate_keys: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "status": self.status,
            "message": self.message,
            "checked_by": self.checked_by,
            "probe_sql": self.probe_sql,
            "row_count_estimate": self.row_count_estimate,
            "join_multiplicity": self.join_multiplicity,
            "match_rate": self.match_rate,
            "orphan_rate": self.orphan_rate,
            "null_fk_rate": self.null_fk_rate,
            "fanout_ratio": self.fanout_ratio,
            "target_rows": self.target_rows,
            "target_distinct_keys": self.target_distinct_keys,
            "target_duplicate_keys": self.target_duplicate_keys,
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

    # Resolve db_type once here so both schema check and live probe use the
    # correct dialect for table/column quoting and COUNT syntax.
    client = store.get_client(account_id)
    db_cfg_id = client.get("db_config_id") if client else None
    raw_cfg = store.get_db_config(db_cfg_id) if db_cfg_id else None
    db_type = (raw_cfg or {}).get("db_type", "azure_sql")

    schema_check = _validate_against_schema(account_id, rel, from_ent, to_ent, db_type=db_type)
    if schema_check.status == STATUS_BROKEN or not execute:
        return schema_check

    try:
        return _execute_probe(account_id, rel, from_ent, to_ent,
                              db_type=db_type, raw_cfg=raw_cfg,
                              timeout_seconds=timeout_seconds)
    except Exception as exc:
        log.warning("relationship live validation failed for %s/%s: %s", account_id, rel_id, exc)
        return RelationshipValidationResult(
            rel_id,
            STATUS_WARNING,
            f"Schema looks valid, but the live DB probe failed: {exc}",
            checked_by="schema",
            probe_sql=schema_check.probe_sql,
        )


_PROFILE_GIVE_UP_AFTER = 3

# Confirmed by the data: at least this share of the probed keys found their
# dimension row, and none found two.
_CONFIRMED_MATCH_RATE = 99.0


def data_confirms(rel: dict) -> bool:
    """Whether the warehouse's own rows vouch for this join.

    The line an admin can accept without reading every join: the profile ran
    against the data (not only the schema), nearly every key found its row,
    no key found two, and the dimension repeats no key. Name-based confidence
    ("Accept all >= 85%") says the columns are spelled alike; this says the
    rows agree.
    """
    def number(key: str) -> float:
        try:
            value = rel.get(key)
            return float(value) if value is not None else -1.0
        except (TypeError, ValueError):
            return -1.0

    return (
        str(rel.get("validation_status") or "") == STATUS_VALID
        and str(rel.get("join_multiplicity") or "") == "one_to_one_or_many_to_one"
        and number("match_rate") >= _CONFIRMED_MATCH_RATE
        and 0.0 <= number("fanout_ratio") <= 1.01
    )


def profile_suggested_relationships(
    account_id: str, *, limit: int = 100, timeout_seconds: int = 20,
) -> dict[str, int]:
    """Ask the warehouse about every suggested join it has not been asked about.

    Discovery proposes joins from names; only the data says whether a key
    matches, how often it is empty, whether it fans out. The admin's
    validate-all asks on request; this asks after every discovery, so the
    resolver -- which prefers a valid edge, penalises a warning and never takes
    a broken or zero-match one -- has evidence to rank on. An unconfirmed INNER
    join whose key is empty or matches nothing on some rows becomes LEFT:
    INNER dropped those rows from every total grouped by that dimension.

    A probe that could not run (no connection, a timeout) leaves the edge
    untested rather than marking it down, and three of them end the run: a
    timed-out probe's query is still running in the warehouse, and the rest
    would only pile more onto it. Confirmed and manual edges are the admin's
    and are not profiled here. Returns what happened, by outcome.
    """
    import store

    pending = [
        rel for rel in store.list_relationships(account_id, active_only=True)
        if (rel.get("status") or "suggested") == "suggested"
        and (rel.get("generated_by") or "heuristic") != "manual"
        and (rel.get("validation_status") or "untested") == "untested"
    ]
    summary = {"valid": 0, "warning": 0, "broken": 0, "made_left": 0, "not_profiled": 0}
    if len(pending) > limit:
        log.warning(
            "Join profiling for %s: %d suggested joins, profiling the first %d; "
            "the rest stay untested until validate-all",
            account_id, len(pending), limit,
        )
    unreachable = 0
    for position, rel in enumerate(pending[:limit]):
        if unreachable >= _PROFILE_GIVE_UP_AFTER:
            log.warning(
                "Join profiling for %s stopped: %d probes could not run; "
                "%d joins left untested", account_id, unreachable,
                len(pending[:limit]) - position,
            )
            break
        rel_id = int(rel["id"])
        try:
            result = validate_relationship(
                account_id, rel_id, execute=True, timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            # The type as well: some (a credential that no longer decrypts)
            # carry no message at all.
            log.warning("Join profiling for %s/%s failed: %s: %s",
                        account_id, rel_id, type(exc).__name__, exc)
            summary["not_profiled"] += 1
            unreachable += 1
            continue
        if result.checked_by != "db" and result.status != STATUS_BROKEN:
            summary["not_profiled"] += 1
            unreachable += 1
            continue
        store.update_relationship_validation(
            account_id, rel_id, result.status,
            row_count_estimate=result.row_count_estimate,
            join_multiplicity=result.join_multiplicity,
            match_rate=result.match_rate, orphan_rate=result.orphan_rate,
            null_fk_rate=result.null_fk_rate, fanout_ratio=result.fanout_ratio,
        )
        summary[result.status] = summary.get(result.status, 0) + 1
        # A join that matches nothing is not made LEFT: it is not a join, and
        # its verdict already keeps the resolver off it.
        if (
            result.checked_by == "db"
            and result.join_multiplicity != "zero_match"
            and str(rel.get("join_type") or "").upper() == "INNER"
            and (result.null_fk_rate > 0 or result.orphan_rate > 0)
            and store.set_suggested_join_type(
                account_id, rel_id, "LEFT", optionality="optional",
                reason=(
                    f"Profiled: {result.null_fk_rate:.2f}% of keys empty and "
                    f"{result.orphan_rate:.2f}% matching nothing; a LEFT join keeps "
                    "those rows in every total."
                ),
            )
        ):
            summary["made_left"] += 1
    return summary


def _validate_against_schema(
    account_id: str,
    rel: dict,
    from_ent: dict,
    to_ent: dict,
    *,
    db_type: str = "azure_sql",
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

    probe_sql = build_probe_sql(db_type, rel, from_ent, to_ent)
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


# A join is profiled on a bounded slice of its many side. The probe used to
# read the whole fact table eight times over -- EXISTS, NOT EXISTS and a full
# join among them -- so on a warehouse-sized fact it ran past its timeout,
# three timeouts ended the profiling run, and every join stayed "untested":
# the resolver had no evidence to rank on and the admin none to approve on.
# The dimension side (its rows, distinct and duplicate keys) is still read
# whole: it is the small side, and a duplicate key anywhere in it fans every
# total out.
PROFILE_SAMPLE_ROWS = 100_000


def build_profile_sql(
    db_type: str, rel: dict, from_ent: dict, to_ent: dict,
    *, sample_rows: int | None = None,
) -> str:
    """Build a read-only join quality profile for one relationship.

    The probe measures source-key nulls, matched source rows, orphans, and
    join fanout. These signals let the resolver avoid technically valid but
    operationally poor join paths.

    One pass over the first `sample_rows` rows of the source table, joined to
    the target's keys grouped once: the counts come from the same rows, which
    a query reading the sample once per count could not promise.
    """
    left_table = _quote_table(from_ent.get("schema_name", ""), from_ent.get("table_name", ""), db_type)
    right_table = _quote_table(to_ent.get("schema_name", ""), to_ent.get("table_name", ""), db_type)
    pairs = _join_pairs(rel)
    count_fn = "COUNT_BIG(1)" if db_type == "azure_sql" else "COUNT(*)"
    left_keys = ", ".join(f"{_quote_col(left, db_type)} AS k{i}" for i, (left, _) in enumerate(pairs))
    right_keys = ", ".join(_quote_col(right, db_type) for _, right in pairs)
    right_named = ", ".join(f"{_quote_col(right, db_type)} AS k{i}" for i, (_, right) in enumerate(pairs))
    rows = int(sample_rows or PROFILE_SAMPLE_ROWS)
    if db_type == "azure_sql":
        sample = f"SELECT TOP ({rows}) {left_keys} FROM {left_table}"
    elif db_type == "snowflake":
        sample = f"SELECT {left_keys} FROM {left_table} LIMIT {rows}"
    else:
        sample = f"SELECT {left_keys} FROM {left_table} FETCH FIRST {rows} ROWS ONLY"
    on_sql = " AND ".join(f"s.k{i} = t.k{i}" for i in range(len(pairs)))
    non_null_sql = " AND ".join(f"s.k{i} IS NOT NULL" for i in range(len(pairs)))
    # The target's statistics sit outside the aggregate on purpose: Oracle
    # refuses a scalar subquery beside an aggregate without a GROUP BY.
    return f"""
WITH s AS (
  {sample}
), t AS (
  SELECT {right_named}, {count_fn} AS n FROM {right_table} GROUP BY {right_keys}
), p AS (
  SELECT
    {count_fn} AS left_rows,
    SUM(CASE WHEN {non_null_sql} THEN 1 ELSE 0 END) AS non_null_fk_rows,
    SUM(CASE WHEN t.n IS NOT NULL THEN 1 ELSE 0 END) AS matched_left_rows,
    SUM(CASE WHEN {non_null_sql} AND t.n IS NULL THEN 1 ELSE 0 END) AS orphan_rows,
    SUM(COALESCE(t.n, 0)) AS join_rows
  FROM s LEFT JOIN t ON {on_sql}
)
SELECT
  p.left_rows, p.non_null_fk_rows, p.matched_left_rows, p.orphan_rows, p.join_rows,
  (SELECT {count_fn} FROM {right_table}) AS target_rows,
  (SELECT {count_fn} FROM t) AS target_distinct_keys,
  (SELECT {count_fn} FROM t WHERE t.n > 1) AS target_duplicate_keys
FROM p
""".strip()


def build_match_exists_sql(db_type: str, rel: dict, from_ent: dict, to_ent: dict) -> str:
    """Does ANY source row match? Stops at the first one that does."""
    left_table = _quote_table(from_ent.get("schema_name", ""), from_ent.get("table_name", ""), db_type)
    right_table = _quote_table(to_ent.get("schema_name", ""), to_ent.get("table_name", ""), db_type)
    on_sql = " AND ".join(
        f"l.{_quote_col(left, db_type)} = r.{_quote_col(right, db_type)}"
        for left, right in _join_pairs(rel)
    )
    exists = f"EXISTS (SELECT 1 FROM {right_table} r WHERE {on_sql})"
    if db_type == "azure_sql":
        return f"SELECT TOP (1) 1 FROM {left_table} l WHERE {exists}"
    if db_type == "snowflake":
        return f"SELECT 1 FROM {left_table} l WHERE {exists} LIMIT 1"
    return f"SELECT 1 FROM {left_table} l WHERE {exists} FETCH FIRST 1 ROWS ONLY"


def run_probe(
    db_type: str,
    raw_cfg: dict,
    sql: str,
    *,
    timeout_seconds: int = 20,
    max_rows: int = 200,
) -> list[tuple]:
    """Run one read-only probe on a saved warehouse connection and return its
    rows (at most `max_rows`). The connection is opened for the probe and
    closed after it. A probe still running after `timeout_seconds` raises
    TimeoutError; its query may go on running in the warehouse."""
    import concurrent.futures
    from core.schema import _az_connect, _ora_connect, _sf_connect

    creds = raw_cfg.get("credentials", {})

    def _run() -> list[tuple]:
        if db_type == "azure_sql":
            conn = _az_connect({**creds, "login_timeout": min(timeout_seconds, 20)}, max_retries=1)
        elif db_type == "snowflake":
            conn = _sf_connect(creds, max_retries=1)
        else:
            conn = _ora_connect(creds, max_retries=1)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            return [tuple(row) for row in cur.fetchmany(max_rows) or []]
        finally:
            try:
                conn.close()
            except Exception:
                pass

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_run).result(timeout=timeout_seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _execute_probe(
    account_id: str,
    rel: dict,
    from_ent: dict,
    to_ent: dict,
    *,
    db_type: str = "azure_sql",
    raw_cfg: dict | None = None,
    timeout_seconds: int = 20,
) -> RelationshipValidationResult:
    if not raw_cfg:
        return RelationshipValidationResult(
            int(rel.get("id") or 0),
            STATUS_WARNING,
            "Schema is valid, but no database connection is assigned for a live probe.",
            checked_by="schema",
        )

    sql = build_profile_sql(db_type, rel, from_ent, to_ent)
    rows = run_probe(db_type, raw_cfg, sql, timeout_seconds=timeout_seconds, max_rows=1)
    (left_rows, non_null_rows, matched_rows, orphan_rows, join_rows,
     target_rows, target_distinct_keys, target_duplicate_keys) = (
        tuple(int(value or 0) for value in rows[0][:8]) if rows else (0,) * 8
    )

    # None of the sampled rows matched, and the sample did not reach the end
    # of the table. That is a verdict about those rows only, while "matches
    # nothing" keeps the resolver off a join for good -- so the whole table is
    # asked first, with a query that stops at its first match: quick exactly
    # when the join is sound.
    if matched_rows <= 0 and left_rows >= PROFILE_SAMPLE_ROWS:
        found = run_probe(db_type, raw_cfg, build_match_exists_sql(db_type, rel, from_ent, to_ent),
                          timeout_seconds=timeout_seconds, max_rows=1)
        if found:
            return RelationshipValidationResult(
                int(rel.get("id") or 0),
                STATUS_UNTESTED,
                f"None of the first {PROFILE_SAMPLE_ROWS:,} rows matched, but rows further "
                "in do: those rows are not representative, so the join is left for review.",
                checked_by="db",
                probe_sql=sql,
            )

    match_rate = round((matched_rows / non_null_rows) * 100.0, 2) if non_null_rows else 0.0
    orphan_rate = round((orphan_rows / non_null_rows) * 100.0, 2) if non_null_rows else 0.0
    null_fk_rate = round(((left_rows - non_null_rows) / left_rows) * 100.0, 2) if left_rows else 0.0
    fanout_ratio = round(join_rows / matched_rows, 3) if matched_rows else 0.0
    multiplicity = (
        "zero_match" if matched_rows <= 0
        else "one_to_many_or_many_to_many" if target_duplicate_keys > 0 or fanout_ratio > 1.01
        else "one_to_one_or_many_to_one"
    )

    if matched_rows <= 0:
        return RelationshipValidationResult(
            int(rel.get("id") or 0),
            STATUS_WARNING,
            "Join executed successfully but returned zero rows. Check whether this join is logically correct.",
            checked_by="db",
            probe_sql=sql,
            row_count_estimate=join_rows,
            join_multiplicity=multiplicity,
            match_rate=match_rate,
            orphan_rate=orphan_rate,
            null_fk_rate=null_fk_rate,
            fanout_ratio=fanout_ratio,
            target_rows=target_rows,
            target_distinct_keys=target_distinct_keys,
            target_duplicate_keys=target_duplicate_keys,
        )

    status = STATUS_WARNING if orphan_rate > 5.0 or fanout_ratio > 1.01 or target_duplicate_keys > 0 else STATUS_VALID
    quality_note = (
        f"Matched {match_rate:.2f}% of non-null source keys; "
        f"orphans {orphan_rate:.2f}%, null keys {null_fk_rate:.2f}%, "
        f"fanout {fanout_ratio:.3f}x; target duplicate keys {target_duplicate_keys}."
    )
    if left_rows >= PROFILE_SAMPLE_ROWS:
        quality_note += f" Measured on the first {left_rows:,} source rows."

    return RelationshipValidationResult(
        int(rel.get("id") or 0),
        status,
        f"Join profile completed. {quality_note}",
        checked_by="db",
        probe_sql=sql,
        row_count_estimate=join_rows,
        join_multiplicity=multiplicity,
        match_rate=match_rate,
        orphan_rate=orphan_rate,
        null_fk_rate=null_fk_rate,
        fanout_ratio=fanout_ratio,
        target_rows=target_rows,
        target_distinct_keys=target_distinct_keys,
        target_duplicate_keys=target_duplicate_keys,
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
