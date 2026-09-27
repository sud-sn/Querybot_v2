"""A period a metric's own table does not keep is read from the table that does.

A business often keeps one measure twice: day by day for the recent past, and
at each month's end for its history -- the daily snapshot is kept for months,
the month-end one for years. Its administrator registers both, named alike:
"Units in Stock" and "Month-End Units in Stock", "Allocated quantity" and
"Month-end allocated quantity". A reader asking for "units in stock at the end
of 2024" uses the first name; the daily snapshot starts long after 2024, so the
answer was empty -- or declined -- where the business keeps the figure.

A metric's sibling is a metric on another table whose name is the same once
the words that say how often it is kept are left out (month-end, monthly,
daily, end of month, fin de mois, mensuel...). Where a question names a period,
the metric's own table keeps no figure of it by its governed date, and a
sibling's table does, the sibling answers in its place and the reader is told.
A figure, not a row: a month-end snapshot whose stock column is empty for the
period keeps no stock for it, however many rows it has. A period no table
keeps is still no data; a period the metric's own table keeps is
answered from it, as before; and where either cannot be told -- no governed
date, a probe that fails -- nothing changes.

The probe is one MAX over the governed date inside the period's bounds, of
the rows where the metric's own columns hold a value, run through the caller's
governed executor, so row policies and table access apply to it as to the
answer; it is cached per workspace, table, date, period, columns and row scope
for as long as a business-date anchor is.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from typing import Any, Callable

log = logging.getLogger("querybot.period_siblings")

# How often a figure is kept, in the words a metric's name says it with.
_KEPT_RE = re.compile(
    r"\b(?:as\s+(?:at|of)\s+|at\s+)?(?:the\s+)?"
    r"(?:(?:day|week|month|quarter|year|period)\s+end"
    r"|end\s+of\s+(?:the\s+)?(?:day|week|month|quarter|year|period))\b"
    r"|\b(?:eod|eow|eom|eoq|eoy|eop)\b"
    r"|\b(?:daily|weekly|monthly|quarterly|yearly|annual)\b"
    r"|\b(?:(?:en|de|a\s+la)\s+)?fin\s+(?:de\s+|du\s+|d\s+)?(?:la\s+)?"
    r"(?:journee|jour|semaine|mois|trimestre|annee|periode|exercice)\b"
    r"|\b(?:quotidien|journalier|hebdomadaire|mensuel|trimestriel|annuel)(?:le|ne|e)?s?\b"
)


def _words(text: Any) -> str:
    folded = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


def measured(name: Any) -> str:
    """A metric's name less the words that say how often it is kept."""
    return " ".join(_KEPT_RE.sub(" ", _words(name)).split())


def _table_key(table: Any) -> tuple[str, ...]:
    parts = [part.strip().strip('[]"`').upper() for part in str(table or "").split(".") if part.strip()]
    return tuple(parts[-2:])


def _same_table(left: Any, right: Any) -> bool:
    a, b = _table_key(left), _table_key(right)
    if not a or not b:
        return False
    if len(a) == len(b):
        return a == b
    return a[-1] == b[-1]


def period_siblings(metric: dict, metrics: list[dict]) -> list[dict]:
    """The metrics kept for the same measure on another table: named alike,
    once the words that say how often each is kept are left out."""
    name = str(metric.get("name") or "")
    measure = measured(name)
    if not measure:
        return []
    return [
        other for other in metrics or []
        if str(other.get("name") or "").casefold() != name.casefold()
        and other.get("base_table") and metric.get("base_table")
        and not _same_table(other.get("base_table"), metric.get("base_table"))
        and measured(other.get("name")) == measure
    ]


def governed_date(metric: dict, date_roles: list[dict], bindings: list[dict] | None = None) -> dict:
    """The date a metric's own table is read by: the date its administrator
    bound to it, else its table's everyday date, else the table's one approved
    date. {} when none is settled."""
    table = metric.get("base_table")
    metric_id = int(metric.get("id") or 0)
    name = str(metric.get("name") or "").casefold()
    own = [
        binding for binding in bindings or []
        if (metric_id and int(binding.get("metric_id") or 0) == metric_id)
        or (name and str(binding.get("metric_name") or "").casefold() == name)
    ]
    if own:
        return dict(sorted(own, key=lambda binding: -int(binding.get("is_default") or 0))[0])
    approved = [
        role for role in date_roles or []
        if str(role.get("status") or "") == "approved" and _same_table(role.get("fact_table"), table)
    ]
    everyday = [role for role in approved if role.get("is_default")]
    if everyday:
        return dict(everyday[0])
    return dict(approved[0]) if len(approved) == 1 else {}


def build_period_rows_probe_sql(
    date: dict, start: str, end: str, db_type: str = "azure_sql", measures: tuple[str, ...] = (),
) -> str:
    """The newest governed date inside [start, end) on which the table holds a
    value in one of ``measures`` (any row, with none named), or "" when the
    date is not physically complete enough to probe."""
    from core.contextual_dates import format_date_value_expression
    from core.date_anchor import _quote_column, _quote_table, _table_alias
    from core.date_roles import normalize_date_key_type
    from core.pipeline_helpers import _date_literal

    fact = str(date.get("fact_table") or "")
    fact_column = str(date.get("fact_column") or "")
    date_column = str(date.get("date_value_column") or "")
    low, high = _date_literal(start, db_type), _date_literal(end, db_type)
    if not (fact and fact_column and low and high):
        return ""
    key_type = normalize_date_key_type(str(date.get("date_key_type") or ""))
    dimension = str(date.get("dimension_table") or "")
    dimension_key = str(date.get("dimension_key") or "")
    fact_sql = _table_alias(_quote_table(fact, db_type), "period_rows", db_type)
    if key_type == "surrogate_fk":
        if not (dimension and dimension_key and date_column):
            return ""
        ref = f"period_date.{_quote_column(date_column, db_type)}"
        source = (f"{fact_sql}\nJOIN {_table_alias(_quote_table(dimension, db_type), 'period_date', db_type)} "
                  f"ON period_rows.{_quote_column(fact_column, db_type)} = "
                  f"period_date.{_quote_column(dimension_key, db_type)}")
    else:
        ref = format_date_value_expression(
            "period_rows", _quote_column(date_column or fact_column, db_type), key_type, db_type)
        source = fact_sql
    valued = " OR ".join(f"period_rows.{_quote_column(column, db_type)} IS NOT NULL"
                         for column in sorted(set(measures)))
    return (f"SELECT MAX({ref}) AS last_date_in_period\nFROM {source}\n"
            f"WHERE {ref} >= {low}\n  AND {ref} < {high}" + (f"\n  AND ({valued})" if valued else ""))


_DEFAULT_TTL_SECONDS = 900
_lock = threading.Lock()
_cache: dict[tuple, tuple[float, bool]] = {}


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def keeps_period(
    account_id: str,
    date: dict,
    period: dict,
    run_probe: Callable[[str], Any],
    db_type: str = "azure_sql",
    scope: str = "",
    measures: tuple[str, ...] = (),
) -> bool | None:
    """Whether the table holds a value of ``measures`` in the period by this
    date: True, False, or None when it cannot be told (nothing to probe, or
    the probe failed)."""
    start, end = str(period.get("start") or ""), str(period.get("end") or "")
    sql = build_period_rows_probe_sql(date, start, end, db_type, measures)
    if not sql:
        return None
    key = (str(account_id), _table_key(date.get("fact_table")), str(date.get("fact_column") or "").upper(),
           start, end, tuple(sorted({str(column).upper() for column in measures})), str(scope or ""))
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > time.time():
            return hit[1]
    try:
        rows = run_probe(sql) or []
    except Exception as exc:
        log.warning("Period probe failed for %s on %s.%s: %s -- the metric is read as asked",
                    account_id, date.get("fact_table"), date.get("fact_column"), exc)
        return None
    first = rows[0] if rows else None
    if isinstance(first, dict):
        value = next(iter(first.values()), None)
    elif isinstance(first, (list, tuple)):
        value = first[0] if first else None
    else:
        value = first
    kept = value is not None
    with _lock:
        _cache[key] = (time.time() + _DEFAULT_TTL_SECONDS, kept)
    return kept


def sibling_for_period(
    metric: dict,
    metrics: list[dict],
    period: dict,
    date_roles: list[dict],
    bindings: list[dict] | None,
    keeps: Callable[[dict, dict], bool | None],
) -> dict:
    """The sibling to read ``period`` from, when ``metric``'s own table keeps
    no figure of it in the period and the sibling's table keeps one of the
    sibling: {"metric", "date"}; else {}. ``keeps(date, metric)`` answers it."""
    if not (period.get("start") and period.get("end")):
        return {}
    siblings = period_siblings(metric, metrics)
    if not siblings:
        return {}
    own = governed_date(metric, date_roles, bindings)
    if not own or keeps(own, metric) is not False:
        return {}
    for sibling in siblings:
        date = governed_date(sibling, date_roles, bindings)
        if date and keeps(date, sibling):
            return {"metric": sibling, "date": date}
    return {}


def read_as(sibling: dict, metric: dict) -> dict:
    """The sibling, carrying the words the reader asked for the metric with,
    so the question that named the metric resolves to it."""
    words = [str(sibling.get("synonyms") or ""), str(metric.get("name") or ""), str(metric.get("synonyms") or "")]
    return {**sibling, "synonyms": ", ".join(word for word in words if word.strip())}
