"""
core/date_coverage.py

Date-range coverage-gap detection: compares a requested "last N days"-style
window against how many distinct calendar days actually have data in that
window. Exists to catch the case a single-aggregate answer ("net revenue for
the last 7 days") otherwise hides completely -- the returned rows carry no
date column at all to inspect, so the gap can only be found by asking the
database directly, not by looking at what already came back.

Runs two small, read-only diagnostic queries via core.schema.run_query,
bypassing the heavier governed_query pipeline for the same reason
core/pipeline_helpers.py::_count_tables_for_zero_row already does: these are
aggregate counts (COUNT/MIN), never raw regulated row data. Best-effort
throughout -- any failure, missing policy field, or adequate coverage returns
None rather than altering or blocking the main answer.

Deliberately no LLM call: this is exact date arithmetic on a database's own
clock, and a model has no business rephrasing a fact this precise.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from core.pipeline_helpers import _quote_table_for_count
from core.schema import run_query

log = logging.getLogger("querybot.date_coverage")

_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}
# Defense-in-depth before interpolating a resolved identifier into a SQL
# string -- these come from admin-configured date-role metadata, not raw
# user input, but there's no reason not to guard the shape anyway (mirrors
# core/report_engine.py's identical _SAFE_IDENT_RE).
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9_$]+(\.[A-Za-z0-9_$]+)*$")


@dataclass(frozen=True)
class CoverageGap:
    requested_days: int
    actual_days: int
    message: str


def _window_to_days(amount, unit) -> int:
    try:
        amount_int = int(amount)
    except (TypeError, ValueError):
        return 0
    per_unit = _UNIT_DAYS.get(str(unit or "").strip().lower())
    if not per_unit or amount_int <= 0:
        return 0
    return amount_int * per_unit


def _parse_date_value(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "")).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _run_scalar(credentials: dict, db_type: str, sql: str) -> object | None:
    rows = run_query(credentials, db_type, sql, max_rows=1)
    if not rows:
        return None
    return next(iter(rows[0].values()), None)


def check_date_coverage(
    db_cfg: dict,
    policy: dict,
    db_type: str,
    metric_name: str = "",
) -> CoverageGap | None:
    """Compare the requested window (``policy['amount']``/``policy['unit']``)
    against how many distinct calendar days actually have data in that
    window, using the same fact/dimension resolution the main query's date
    anchor already uses (core.contextual_dates.format_required_anchor).
    """
    from core.contextual_dates import format_required_anchor

    # Day-coverage diagnostics count distinct days. A monthly encoded snapshot
    # is valid for month-level analysis but cannot be evaluated by this check
    # without producing a false "missing days" warning.
    if str(policy.get("temporal_grain") or "").lower() not in {"", "day"}:
        return None

    requested_days = _window_to_days(policy.get("amount"), policy.get("unit"))
    if requested_days <= 0:
        return None

    fact_table = str(policy.get("fact_table") or "")
    fact_column = str(policy.get("fact_column") or "")
    surrogate = str(policy.get("date_key_type") or "") == "surrogate_fk"

    if surrogate:
        dimension_table = str(policy.get("dimension_table") or policy.get("date_table") or "")
        dimension_key = str(policy.get("dimension_key") or "")
        date_column = str(policy.get("date_column") or "")
        idents = [fact_table, fact_column, dimension_table, dimension_key, date_column]
        if not all(idents) or not all(_SAFE_IDENT_RE.match(v) for v in idents):
            return None
    else:
        date_table = str(policy.get("date_table") or fact_table)
        date_column = str(policy.get("date_column") or fact_column)
        idents = [date_table, date_column]
        if not all(idents) or not all(_SAFE_IDENT_RE.match(v) for v in idents):
            return None

    anchor_expr = format_required_anchor(policy, db_type)
    if not anchor_expr:
        return None

    credentials = (db_cfg or {}).get("credentials") or {}

    try:
        anchor_value = _run_scalar(
            credentials, db_type, f"SELECT {anchor_expr} AS AnchorDate",
        )
        anchor_date = _parse_date_value(anchor_value)
        if anchor_date is None:
            return None

        window_start = anchor_date - timedelta(days=requested_days - 1)
        start_iso, end_iso = window_start.isoformat(), anchor_date.isoformat()

        if surrogate:
            quoted_dim = _quote_table_for_count(dimension_table, db_type)
            quoted_fact = _quote_table_for_count(fact_table, db_type)
            coverage_sql = (
                f"SELECT COUNT(DISTINCT d.{date_column}) AS DaysWithData "
                f"FROM {quoted_dim} d JOIN {quoted_fact} f "
                f"ON f.{fact_column} = d.{dimension_key} "
                f"WHERE d.{date_column} >= '{start_iso}' AND d.{date_column} <= '{end_iso}'"
            )
        else:
            quoted_table = _quote_table_for_count(date_table, db_type)
            coverage_sql = (
                f"SELECT COUNT(DISTINCT {date_column}) AS DaysWithData "
                f"FROM {quoted_table} "
                f"WHERE {date_column} >= '{start_iso}' AND {date_column} <= '{end_iso}'"
            )
        actual_value = _run_scalar(credentials, db_type, coverage_sql)
        actual_days = int(actual_value) if actual_value is not None else 0
    except Exception as exc:
        log.debug("Date coverage check skipped: %s", exc)
        return None

    # Small tolerance: today's row may be partial without being a real gap.
    if actual_days >= requested_days - 1:
        return None

    role = str(policy.get("business_role") or "business date").strip().lower()
    if role.endswith(" date"):
        role = role[:-5].strip()
    date_label = f"{role} date" if role else "business date"
    if actual_days != 1:
        date_label += "s"
    metric_subject = (
        f"{str(metric_name).strip()} records"
        if str(metric_name).strip()
        else "Records"
    )
    requested_amount = int(policy.get("amount") or requested_days)
    requested_unit = str(policy.get("unit") or "day").strip().lower() or "day"
    return CoverageGap(
        requested_days=requested_days,
        actual_days=actual_days,
        message=(
            f"{metric_subject} were available on {actual_days} distinct "
            f"{date_label} within the requested {requested_amount}-{requested_unit} "
            "period. The result reflects those available records."
        ),
    )
