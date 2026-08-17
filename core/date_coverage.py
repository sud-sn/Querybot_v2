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
    metric_active_days: int | None = None


def _safe_metric_formula(value: object) -> str:
    """Return a safe approved aggregate expression for coverage probing.

    Metric formulas have already passed the Metric Registry validator.  This
    extra, deliberately conservative gate keeps the best-effort diagnostic
    query from accepting query-shaped SQL or non-aggregate row expressions.
    Complex or qualified formulas simply fall back to row-date coverage.
    """
    formula = str(value or "").strip().rstrip(";")
    if not formula:
        return ""
    if re.search(r";|--|/\*|\*/|\b(?:select|from|with|insert|update|delete|merge|drop|alter|create)\b", formula, re.I):
        return ""
    if not re.search(r"\b(?:sum|avg|min|max|count)\s*\(", formula, re.I):
        return ""
    # Table-qualified formulas need the registry's full join/alias plan and
    # cannot safely be transplanted into this small diagnostic query.
    if re.search(r"(?:\]|\"|`|[A-Za-z0-9_$])\s*\.\s*(?:\[|\"|`|[A-Za-z_$])", formula):
        return ""
    return formula


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
    metric_formula: str = "",
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

        coverage_from = ""
        coverage_where = ""
        coverage_date_ref = ""
        if surrogate:
            quoted_dim = _quote_table_for_count(dimension_table, db_type)
            quoted_fact = _quote_table_for_count(fact_table, db_type)
            coverage_from = (
                f"{quoted_dim} d JOIN {quoted_fact} f "
                f"ON f.{fact_column} = d.{dimension_key}"
            )
            coverage_date_ref = f"d.{date_column}"
            coverage_where = (
                f"{coverage_date_ref} >= '{start_iso}' "
                f"AND {coverage_date_ref} <= '{end_iso}'"
            )
            coverage_sql = (
                f"SELECT COUNT(DISTINCT {coverage_date_ref}) AS DaysWithData "
                f"FROM {coverage_from} WHERE {coverage_where}"
            )
        else:
            quoted_table = _quote_table_for_count(date_table, db_type)
            coverage_from = quoted_table
            coverage_date_ref = date_column
            coverage_where = (
                f"{coverage_date_ref} >= '{start_iso}' "
                f"AND {coverage_date_ref} <= '{end_iso}'"
            )
            coverage_sql = (
                f"SELECT COUNT(DISTINCT {coverage_date_ref}) AS DaysWithData "
                f"FROM {coverage_from} WHERE {coverage_where}"
            )
        actual_value = _run_scalar(credentials, db_type, coverage_sql)
        actual_days = int(actual_value) if actual_value is not None else 0

        metric_active_days: int | None = None
        approved_formula = _safe_metric_formula(metric_formula)
        if approved_formula:
            metric_activity_sql = (
                "SELECT COUNT(*) AS DaysWithMetricData FROM ("
                f"SELECT {coverage_date_ref} AS MetricDate "
                f"FROM {coverage_from} WHERE {coverage_where} "
                f"GROUP BY {coverage_date_ref} "
                f"HAVING ({approved_formula}) IS NOT NULL "
                f"AND ({approved_formula}) <> 0"
                ") metric_activity"
            )
            try:
                metric_value = _run_scalar(credentials, db_type, metric_activity_sql)
                metric_active_days = int(metric_value) if metric_value is not None else 0
            except Exception as exc:
                # This extra probe is advisory.  Formula shapes that need more
                # joins or dialect-specific handling must not erase the basic
                # row-date coverage result that was already obtained above.
                log.debug("Metric activity coverage skipped: %s", exc)
                metric_active_days = None
    except Exception as exc:
        log.debug("Date coverage check skipped: %s", exc)
        return None

    # A partial current day is still represented by one distinct date, so an
    # off-by-one tolerance here hides a real coverage gap.  Report every
    # window where fewer distinct business dates were observed than requested;
    # the result remains valid, but the user can see that the period is sparse.
    if actual_days >= requested_days and (
        metric_active_days is None or metric_active_days >= requested_days
    ):
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
    if (
        requested_unit == "day"
        and metric_active_days is not None
        and metric_active_days < min(actual_days, requested_days)
    ):
        available_label = "day" if metric_active_days == 1 else "days"
        metric_label = str(metric_name).strip() or "The selected metric"
        message = (
            f"You asked for the last {requested_amount} days. Records existed "
            f"on {actual_days} {date_label}, but {metric_label} was nonzero on "
            f"only {metric_active_days} {available_label}. The result reflects "
            "the available metric values."
        )
    elif requested_unit == "day":
        available_label = "day" if actual_days == 1 else "days"
        message = (
            f"You asked for the last {requested_amount} days, but "
            f"{metric_subject.lower()} were found on only {actual_days} "
            f"{date_label} ({actual_days} {available_label} with data). "
            "The result reflects the available data."
        )
    else:
        message = (
            f"{metric_subject} were available on {actual_days} distinct "
            f"{date_label} within the requested {requested_amount}-{requested_unit} "
            "period. The result reflects those available records."
        )

    return CoverageGap(
        requested_days=requested_days,
        actual_days=actual_days,
        message=message,
        metric_active_days=metric_active_days,
    )
