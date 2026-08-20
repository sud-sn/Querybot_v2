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
from datetime import date, datetime, timezone
from typing import Any, Callable

log = logging.getLogger("querybot.date_anchor")

# How long a resolved anchor stays usable. A warehouse that loads hourly is well
# served by 15 minutes; set QUERYBOT_DATE_ANCHOR_TTL_SECONDS=0 to disable the
# cache entirely and probe on every question.
_DEFAULT_TTL_SECONDS = 900

# How long to remember that a probe FAILED. On a starved warehouse the probe can
# exhaust the statement timeout every time; without this, every question repeats
# a two-minute failure that is already known to fail. Kept much shorter than the
# success TTL so a warehouse that recovers is picked up quickly.
_DEFAULT_FAILURE_TTL_SECONDS = 300

# How old a PERSISTED anchor may be before it must be re-probed. The in-memory
# TTL bounds staleness within a process; without this the durable row bounded
# nothing at all, because every in-memory expiry fell through to the store and
# re-armed the TTL from the same value. The warehouse would be probed exactly
# once, ever, and after the client's overnight reload every relative-date
# question would keep answering against the pre-reload date, silently excluding
# every newly loaded row.
#
# ONE HOUR, not one day. A day was chosen to make a nightly-loading warehouse
# re-probe once per load, and it is wrong for the same reason the unbounded
# version was: it assumes data only ever moves FORWARD, on a schedule we guessed.
# It does not. A reload that replaces the range (a restore, a re-point at another
# environment, a corrected extract) leaves every relative-date question answering
# against the old range, confidently and with no way for the user to tell.
# Measured on the live path, a stored anchor 23.9h old was served with zero
# probes; the answer was a day's worth of data out of date and looked identical
# to a correct one.
#
# The cost of the tighter bound is one indexed MAX per account per hour, since
# the 900s in-memory TTL still absorbs everything inside an hour. That is the
# right trade: the probe is a single-value read over a governed date column, and
# a wrong answer costs more than a cheap query.
#
# Set QUERYBOT_DATE_ANCHOR_MAX_AGE_SECONDS to tune it. NOTE THE ASYMMETRY WITH
# THE TTL KNOB: 0 here means "keep a stored anchor indefinitely", while 0 on
# QUERYBOT_DATE_ANCHOR_TTL_SECONDS means "never cache, probe every question".
# Opposite meanings for the same value on two adjacent settings, so reach for
# invalidate_anchor() rather than a 0 you have to remember the direction of.
_DEFAULT_MAX_AGE_SECONDS = 3600

_lock = threading.Lock()
_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
_failures: dict[tuple[str, str, str], float] = {}


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


def build_key_order_check_sql(policy: dict | None, db_type: str = "azure_sql") -> str:
    """Does the date dimension's KEY order match its DATE order?

    The expensive probe exists because a surrogate key carries no guarantee of
    date ordering -- key 4067 need not be later than 4066 -- so proving "the
    newest date that actually has rows" needs a semi-join across the whole fact.

    But that guarantee is a property of the DIMENSION, which is small, and it is
    cheap to establish once: count the rows whose date is earlier than the row
    before them in key order. Zero means the key is monotonic in date, and the
    anchor can then be read as MAX(fact.key) -- a single-column aggregate with
    no join at all -- rather than a semi-join over millions of rows.

    On the EMCO mart DT_DMS_KEY is 20250417 for 2025-04-17, so this returns 0
    and the cheap path applies. On a warehouse with genuinely arbitrary
    surrogates it returns non-zero and the semi-join is used, unchanged.

    A NULL date is counted as a violation. LAG comparisons silently skip NULLs
    -- `NULL < prev_d` is UNKNOWN, and the NULL also becomes the next row's
    prev_d -- so a single NULL-dated member erases the one comparison that
    spans it and hides any inversion across that gap. Unknown/N-A members are
    a standard date-dimension pattern, and the consequence of trusting a
    blinded check is severe: the cheap probe returns a date that is NOT the
    latest with fact rows, stamped as probed from the fact and persisted. So
    any NULL makes this dimension ineligible for the cheap path.
    """
    policy = policy or {}
    dimension = str(policy.get("dimension_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    date_column = str(policy.get("date_column") or "")
    if not (dimension and dimension_key and date_column):
        return ""
    key_sql = _quote_column(dimension_key, db_type)
    date_sql = _quote_column(date_column, db_type)
    return "\n".join([
        "SELECT COUNT(*) AS out_of_order",
        "FROM (",
        f"    SELECT {date_sql} AS d,",
        f"           LAG({date_sql}) OVER (ORDER BY {key_sql}) AS prev_d",
        f"    FROM {_quote_table(dimension, db_type)}",
        ") AS ordered",
        "WHERE d IS NULL OR d < prev_d",
    ])


def build_cheap_anchor_probe_sql(policy: dict | None, db_type: str = "azure_sql") -> str:
    """Anchor probe for a dimension whose key is monotonic in date.

    Reads MAX of the fact's OWN key -- one column, no join, no EXISTS -- then
    translates it through the dimension's primary key, which is a seek.

    Only valid when build_key_order_check_sql has returned 0 for this policy.
    """
    policy = policy or {}
    fact = str(policy.get("fact_table") or policy.get("anchor_table") or "")
    fact_key = str(policy.get("fact_column") or "")
    dimension = str(policy.get("dimension_table") or "")
    dimension_key = str(policy.get("dimension_key") or "")
    date_column = str(policy.get("date_column") or "")
    if not (fact and fact_key and dimension and dimension_key and date_column):
        return ""
    return "\n".join([
        f"SELECT MAX(anchor_date.{_quote_column(date_column, db_type)}) AS max_business_date",
        f"FROM {_quote_table(dimension, db_type)} AS anchor_date",
        f"WHERE anchor_date.{_quote_column(dimension_key, db_type)} = (",
        f"    SELECT MAX(anchor_fact.{_quote_column(fact_key, db_type)})",
        f"    FROM {_quote_table(fact, db_type)} AS anchor_fact",
        ")",
    ])


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
    # The in-memory entry may not outlive the persisted max-age bound. Raising
    # the TTL for performance — a perfectly reasonable thing to do on a
    # warehouse that loads once a day — used to switch off the ONLY staleness
    # check there is, because a live in-memory hit returns before the stored
    # anchor's age is ever examined. The two knobs answer different questions
    # ("how often may we probe" and "how stale may an answer be"), and the
    # second has to hold whichever cache serves the request.
    lifetime = ttl
    max_age = _max_age_seconds()
    if max_age:
        age = _anchor_age_seconds(anchor.get("resolved_at")) or 0.0
        lifetime = min(ttl, max(0.0, max_age - age))
        if lifetime <= 0:
            return
    with _lock:
        _cache[key] = {"anchor": dict(anchor), "expires_at": time.time() + lifetime}


def _failure_ttl_seconds() -> int:
    try:
        return max(0, int(os.getenv("QUERYBOT_DATE_ANCHOR_FAILURE_TTL_SECONDS", "")
                          or _DEFAULT_FAILURE_TTL_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_FAILURE_TTL_SECONDS


def _remember_failure(key: tuple[str, str, str]) -> None:
    ttl = _failure_ttl_seconds()
    if not ttl:
        return
    with _lock:
        _failures[key] = time.time() + ttl


def clear_cache(account_id: str | None = None, *, persistent: bool = False) -> None:
    """Drop cached anchors and failures — for tests, and after a data refresh.

    ``persistent`` also forgets the durable copy. Without it a "refresh the
    business date" action would clear memory, read the same value straight back
    out of the store, and appear to do nothing.
    """
    if persistent and account_id is not None:
        try:
            import store
            removed = store.clear_business_date_anchor(account_id)
            if removed:
                log.info(
                    "Stored business-date anchor cleared for %s (%d row(s)) — the "
                    "next relative-date question will re-probe the warehouse",
                    account_id, removed,
                )
        except Exception as exc:
            log.warning(
                "Stored business-date anchor could not be cleared for %s: %s — "
                "memory was cleared but the stale value will be restored",
                account_id, exc,
            )
    with _lock:
        if account_id is None:
            _cache.clear()
            _failures.clear()
            return
        for key in [key for key in _cache if key[0] == str(account_id)]:
            _cache.pop(key, None)
        for key in [key for key in _failures if key[0] == str(account_id)]:
            _failures.pop(key, None)


def _max_age_seconds() -> int:
    try:
        return max(0, int(os.getenv("QUERYBOT_DATE_ANCHOR_MAX_AGE_SECONDS", "")
                          or _DEFAULT_MAX_AGE_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SECONDS


def _anchor_age_seconds(resolved_at: Any) -> float | None:
    """Seconds since a stored anchor was probed, or None if unreadable."""
    raw = str(resolved_at or "").strip().replace("Z", "+00:00")
    if not raw:
        return None
    for text in (raw, raw.replace(" ", "T")):
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            continue
        if stamp.tzinfo is not None:
            stamp = stamp.replace(tzinfo=None)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return max(0.0, (now - stamp).total_seconds())
    return None


def _stored_anchor(account_id: str, policy: dict | None) -> dict[str, Any]:
    """Read a previously resolved anchor from the durable store.

    The in-memory cache dies with the process. On a slow warehouse that meant
    every restart, redeploy or crash re-ran an 800-second probe while a user
    watched a spinner. The latest loaded date is a property of the WAREHOUSE,
    not of a process, so it belongs in the database.
    """
    fact = str((policy or {}).get("fact_table") or (policy or {}).get("anchor_table") or "")
    column = str((policy or {}).get("fact_column") or "")
    if not account_id or not fact or not column:
        return {}
    try:
        import store
        stored = store.load_business_date_anchor(account_id, fact, column) or {}
    except Exception as exc:
        log.debug("Stored anchor unavailable for %s: %s", account_id, exc)
        return {}
    if not stored.get("value"):
        return {}

    max_age = _max_age_seconds()
    if not max_age:
        return stored
    age = _anchor_age_seconds(stored.get("resolved_at"))
    if age is None:
        # An unreadable timestamp cannot be shown to be fresh. Re-probe rather
        # than serve a value of unknown age as though it were current.
        log.warning(
            "Stored business-date anchor for %s (%s.%s) has an unreadable "
            "resolved_at (%r) — re-probing",
            account_id, fact, column, stored.get("resolved_at"),
        )
        return {}
    if age > max_age:
        log.info(
            "Stored business-date anchor for %s (%s.%s) is %.0fh old (max %.0fh) "
            "— re-probing so a warehouse reload is picked up",
            account_id, fact, column, age / 3600.0, max_age / 3600.0,
        )
        return {}
    return stored


def _persist_anchor(account_id: str, policy: dict | None, anchor: dict) -> None:
    fact = str((policy or {}).get("fact_table") or (policy or {}).get("anchor_table") or "")
    column = str((policy or {}).get("fact_column") or "")
    if not account_id or not fact or not column:
        return
    try:
        import store
        store.save_business_date_anchor(account_id, fact, column, anchor)
    except Exception as exc:
        log.warning(
            "Business-date anchor could not be persisted for %s (%s.%s): %s — it "
            "will be re-probed after the next restart",
            account_id, fact, column, exc,
        )


def _first_scalar(rows: Any) -> Any:
    """First column of the first row, whatever shape the driver returned."""
    if not rows:
        return None
    first = rows[0]
    if isinstance(first, dict):
        return next(iter(first.values()), None)
    if isinstance(first, (list, tuple)):
        return first[0] if first else None
    return first


def _select_probe_sql(
    account_id: str,
    policy: dict | None,
    db_type: str,
    run_probe: Callable[[str], Any],
) -> str:
    """Choose the cheapest probe this dimension's key ordering permits.

    The semi-join form is always correct and always expensive: it reads the
    whole fact to prove which dates have rows. When the dimension's key is
    monotonic in its date -- which a YYYYMMDD-style key always is -- MAX of the
    fact's own key identifies the same row, and that is a single-column
    aggregate with no join.

    The ordering check runs against the date dimension only, which is small.
    Anything unexpected falls back to the semi-join, so a wrong guess here
    can cost time but never correctness.
    """
    ordering_sql = build_key_order_check_sql(policy, db_type)
    if ordering_sql:
        try:
            out_of_order = _first_scalar(run_probe(ordering_sql))
            if out_of_order is not None and int(out_of_order) == 0:
                cheap = build_cheap_anchor_probe_sql(policy, db_type)
                if cheap:
                    log.info(
                        "Business-date anchor for %s will use the cheap probe: "
                        "%s is monotonic in %s, so MAX(%s) identifies the latest "
                        "loaded row without a semi-join over the fact",
                        account_id, (policy or {}).get("dimension_key"),
                        (policy or {}).get("date_column"),
                        (policy or {}).get("fact_column"),
                    )
                    return cheap
            else:
                log.info(
                    "Business-date anchor for %s must use the semi-join probe: "
                    "%s rows in %s are out of date order, so the surrogate key "
                    "cannot identify the latest loaded date",
                    account_id, out_of_order, (policy or {}).get("dimension_table"),
                )
        except Exception as exc:
            log.info(
                "Key-order check unavailable for %s (%s) — using the semi-join probe",
                account_id, exc,
            )
    return build_anchor_probe_sql(policy, db_type)


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

    # Durable store next: a restart must not repeat the probe.
    stored = _stored_anchor(account_id, policy)
    if stored.get("value"):
        stored.setdefault("business_role", str(policy.get("business_role") or ""))
        remember_anchor(account_id, policy, stored)
        log.info(
            "Business-date anchor restored from store for %s: %s = %s (resolved %s) "
            "— no probe needed",
            account_id, stored.get("fact_table"), stored.get("value"),
            stored.get("resolved_at") or "unknown",
        )
        return {**stored, "cached": True}

    key = anchor_key(account_id, policy)
    with _lock:
        failed_until = _failures.get(key, 0.0)
    if failed_until > time.time():
        # Already known to fail. Retrying costs the full statement timeout again
        # and yields the same nothing, so skip straight to the in-query anchor.
        log.info(
            "Business-date anchor probe skipped for %s on %s.%s — a recent probe "
            "failed and is still in the negative cache",
            account_id, policy.get("fact_table"), policy.get("fact_column"),
        )
        return {}

    sql = _select_probe_sql(account_id, policy, db_type, run_probe)
    if not sql:
        return {}

    started = time.time()
    try:
        rows = run_probe(sql) or []
    except Exception as exc:
        _remember_failure(key)
        log.warning(
            "Business-date anchor probe failed for %s on %s.%s after %d ms: %s — "
            "keeping the in-query anchor and not retrying for %d s",
            account_id, policy.get("fact_table"), policy.get("fact_column"),
            int((time.time() - started) * 1000), exc, _failure_ttl_seconds(),
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
        _remember_failure(key)
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
    _persist_anchor(account_id, policy, anchor)
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
