"""Resolve "the latest business date that actually has data" once, not per question.

Relative-date questions ("revenue for the last 2 days") must anchor on the data
rather than the clock: a warehouse loaded to Friday should answer Friday's
window on Monday, and a calendar dimension full of future rows must not anchor
to a date with no facts. Commit 975214b made that a governed rule, and it is the
right rule.

The cost is that every such question asked the database a genuinely expensive
question first — "what is the newest date present in this fact?" — which no SQL
shape can answer without reading the fact. On a large fact with no index on the
date key that alone exhausted the statement timeout.

But the answer changes only when the warehouse loads. So resolve it once, cache
it per (account, fact, date key) with a TTL, and let the compiled SQL carry a
literal window instead of a subquery. Every later question in the TTL window
becomes a plain filtered aggregate with no MAX over the fact at all.

This is not a relaxation of the governed rule: the value still comes from the
fact's own rows, it is stamped with its provenance and probe time so a trace can
show where it came from, and the TTL bounds how stale it can be. The clock is
still never the anchor.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Any, Callable

log = logging.getLogger("querybot.date_anchor")

# How long a resolved anchor stays usable. A warehouse that loads hourly is well
# served by 15 minutes; set QUERYBOT_DATE_ANCHOR_TTL_SECONDS=0 to disable the
# cache entirely and probe on every question.
_DEFAULT_TTL_SECONDS = 900

_lock = threading.Lock()
_cache: dict[tuple[str, str, str], dict[str, Any]] = {}


def _ttl_seconds() -> int:
    try:
        return max(0, int(os.getenv("QUERYBOT_DATE_ANCHOR_TTL_SECONDS", "")
                          or _DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_SECONDS


def _identity(value: Any) -> str:
    parts = [
        part.strip().strip('[]"`').upper()
        for part in str(value or "").split(".")
        if part.strip().strip('[]"`')
    ]
    return ".".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")


def anchor_key(account_id: str, policy: dict | None) -> tuple[str, str, str]:
    """Cache identity: one anchor per account, fact table and governed date key."""
    policy = policy or {}
    return (
        str(account_id or ""),
        _identity(policy.get("fact_table") or policy.get("anchor_table")),
        str(policy.get("fact_column") or "").strip().strip('[]"`').upper(),
    )


def _quote_table(table: str, db_type: str) -> str:
    parts = [
        part.strip().strip('[]"`')
        for part in str(table or "").split(".")
        if part.strip().strip('[]"`')
    ]
    if not parts:
        return ""
    if db_type == "azure_sql":
        return ".".join(f"[{part}]" for part in parts)
    if db_type in {"snowflake", "oracle"}:
        return ".".join(f'"{part}"' for part in parts)
    return ".".join(parts)


def _quote_column(name: str, db_type: str) -> str:
    clean = str(name or "").strip().strip('[]"`')
    if db_type == "azure_sql":
        return f"[{clean}]"
    if db_type in {"snowflake", "oracle"}:
        return f'"{clean}"'
    return clean


def build_anchor_probe_sql(policy: dict | None, db_type: str = "azure_sql") -> str:
    """The cheapest correct query for "latest governed date present in the fact".

    Surrogate-key roles read the date value from the dimension and prove the row
    exists in the fact with a semi-join, so the fact is reached through its key
    rather than joined row by row. A fact-native date needs no join at all.

    Returns "" when the policy is not physically complete enough to probe.
    """
    policy = policy or {}
    fact = str(policy.get("fact_table") or policy.get("anchor_table") or "")
    fact_key = str(policy.get("fact_column") or "")
    date_column = str(policy.get("date_column") or "")
    dimension = str(policy.get("dimension_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    if not fact or not date_column:
        return ""

    fact_sql = _quote_table(fact, db_type)
    if dimension and dimension_key and fact_key:
        return (
            f"SELECT MAX(anchor_date.{_quote_column(date_column, db_type)}) "
            f"AS max_business_date\n"
            f"FROM {_quote_table(dimension, db_type)} AS anchor_date\n"
            f"WHERE EXISTS (\n"
            f"    SELECT 1 FROM {fact_sql} AS anchor_fact\n"
            f"    WHERE anchor_fact.{_quote_column(fact_key, db_type)} = "
            f"anchor_date.{_quote_column(dimension_key, db_type)}\n"
            f")"
        )
    return (
        f"SELECT MAX(anchor_fact.{_quote_column(date_column, db_type)}) "
        f"AS max_business_date\n"
        f"FROM {fact_sql} AS anchor_fact"
    )


def _coerce_anchor(value: Any) -> str:
    """Render a probed value as an ISO date literal, or "" if it is not one."""
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(text[: len(fmt) + 2].strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def cached_anchor(account_id: str, policy: dict | None) -> dict[str, Any]:
    """Return a live cached anchor for this fact+key, or {} when there is none."""
    ttl = _ttl_seconds()
    if not ttl:
        return {}
    key = anchor_key(account_id, policy)
    if not key[1] or not key[2]:
        return {}
    with _lock:
        entry = _cache.get(key)
        if not entry:
            return {}
        if entry["expires_at"] <= time.time():
            _cache.pop(key, None)
            return {}
        return dict(entry["anchor"])


def remember_anchor(account_id: str, policy: dict | None, anchor: dict) -> None:
    ttl = _ttl_seconds()
    if not ttl or not (anchor or {}).get("value"):
        return
    key = anchor_key(account_id, policy)
    if not key[1] or not key[2]:
        return
    with _lock:
        _cache[key] = {"anchor": dict(anchor), "expires_at": time.time() + ttl}


def clear_cache(account_id: str | None = None) -> None:
    """Drop cached anchors — for tests, and after a data-refresh signal."""
    with _lock:
        if account_id is None:
            _cache.clear()
            return
        for key in [key for key in _cache if key[0] == str(account_id)]:
            _cache.pop(key, None)


def resolve_business_anchor(
    account_id: str,
    policy: dict | None,
    db_type: str,
    run_probe: Callable[[str], Any],
) -> dict[str, Any]:
    """Resolve the governed anchor for one temporal policy.

    ``run_probe`` receives one read-only SQL string and returns its rows. The
    probe is a single-value MAX over a governed date — no row data leaves the
    database — which is the same class of diagnostic read core/date_coverage.py
    already performs outside the heavier governed-query path.

    Returns {} when the anchor cannot be established; callers must then keep
    their existing data-relative anchor subquery rather than guess.
    """
    policy = policy or {}
    if str(policy.get("anchor_policy") or "") != "latest_available":
        return {}

    hit = cached_anchor(account_id, policy)
    if hit:
        return {**hit, "cached": True}

    sql = build_anchor_probe_sql(policy, db_type)
    if not sql:
        return {}

    started = time.time()
    try:
        rows = run_probe(sql) or []
    except Exception as exc:
        log.warning(
            "Business-date anchor probe failed for %s on %s.%s: %s — keeping the "
            "in-query anchor",
            account_id, policy.get("fact_table"), policy.get("fact_column"), exc,
        )
        return {}

    raw = None
    if rows:
        first = rows[0]
        if isinstance(first, dict):
            raw = next(iter(first.values()), None) if len(first) == 1 else (
                first.get("max_business_date")
                or first.get("MAX_BUSINESS_DATE")
                or next(iter(first.values()), None)
            )
        elif isinstance(first, (list, tuple)):
            raw = first[0] if first else None
        else:
            raw = first

    value = _coerce_anchor(raw)
    if not value:
        log.warning(
            "Business-date anchor probe for %s returned no usable date (%r) — "
            "keeping the in-query anchor", account_id, raw,
        )
        return {}

    anchor = {
        "value": value,
        "fact_table": str(policy.get("fact_table") or policy.get("anchor_table") or ""),
        "fact_column": str(policy.get("fact_column") or ""),
        "date_column": str(policy.get("date_column") or ""),
        "business_role": str(policy.get("business_role") or ""),
        "source": "probed_from_fact_rows",
        "probe_ms": int((time.time() - started) * 1000),
        "probed_at": int(time.time()),
        "ttl_seconds": _ttl_seconds(),
        "cached": False,
    }
    remember_anchor(account_id, policy, anchor)
    log.info(
        "Business-date anchor resolved for %s: %s = %s (probe %d ms, cached %d s)",
        account_id, anchor["fact_table"], value, anchor["probe_ms"],
        anchor["ttl_seconds"],
    )
    return anchor


def anchor_for_policy(
    resolved: dict[str, Any] | None, policy: dict | None
) -> dict[str, Any]:
    """Return the resolved anchor when it belongs to this exact policy.

    Guards the compiler and the validator against applying an anchor probed for
    one fact/date key to a different one.
    """
    resolved = resolved or {}
    policy = policy or {}
    if not resolved.get("value"):
        return {}
    same_fact = _identity(resolved.get("fact_table")) == _identity(
        policy.get("fact_table") or policy.get("anchor_table")
    )
    same_key = (
        str(resolved.get("fact_column") or "").strip().strip('[]"`').upper()
        == str(policy.get("fact_column") or "").strip().strip('[]"`').upper()
    )
    return dict(resolved) if same_fact and same_key else {}
