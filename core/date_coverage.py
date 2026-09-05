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

from core.i18n import grain_label as _grain, plural as _plural, t as _t
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
    # The newest business date the data actually holds, as this check read it
    # from the database on this request. It was being read and discarded: the
    # window was derived from it and then it was never mentioned again, so a
    # user shown "the result reflects the available data" was not told which
    # dates that meant.
    observed_through: str = ""


def _metric_subject(metric_text: str, *, lead: bool) -> str:
    """What the missing records are called, in the position it appears in.

    English lowercased the whole phrase mid-sentence, which also lowercased
    the metric's own name -- an "EBITDA" metric was reported as "ebitda
    records". The name is the tenant's, so it keeps its case here and only the
    frame around it changes; French needs its article at the head of a
    sentence and none in the middle, which is why the position is a parameter
    rather than a call to ``.lower()`` on a translated string.
    """
    kind = "named" if metric_text else "generic"
    return _t(f"caveat.dates.subject.{kind}{'_lead' if lead else ''}",
              metric=metric_text)


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

    # Day-coverage diagnostics count distinct days. A period-grained source is
    # valid for month-level analysis but cannot be evaluated by a distinct-day
    # count without producing a false "missing days" warning — so the COUNT is
    # skipped for it. The anchor read is not: which period the data actually
    # runs through is exactly what a "this month" question needs to know, and
    # returning early here skipped that too.
    counts_days = str(policy.get("temporal_grain") or "").lower() in {"", "day"}

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

        if not counts_days:
            # No day count to make, but the read already happened and the
            # answer is about a period the user has not been told the identity
            # of. Say which date the data runs through and stop.
            grain = str(policy.get("temporal_grain") or "period").strip().lower()
            return CoverageGap(
                requested_days=requested_days,
                actual_days=0,
                observed_through=end_iso,
                message=_t(
                    "caveat.dates.grain_recorded",
                    grain=_grain(grain, 1), through=end_iso,
                ),
            )

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
                log.info(
                    "Metric-activity coverage probe unavailable for %r (%s) — "
                    "falling back to row-date coverage, which is still reported",
                    metric_name or "the selected metric", exc,
                )
                metric_active_days = None
    except Exception as exc:
        # This is the only live read of the true current data date on the
        # answer path. Swallowed at debug, a failed probe was indistinguishable
        # from a confirmed-fresh answer: both produce no caveat at all.
        log.warning(
            "Date coverage check FAILED for %s.%s (%s) — this answer carries no "
            "freshness caveat, and that is because the check could not run, not "
            "because the window is covered",
            policy.get("fact_table"), policy.get("fact_column"), exc,
            exc_info=True,
        )
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
    # "business" is this module's own default word for the date, so it is copy
    # and translates. Any other role is the tenant's own term -- "invoice",
    # "posting" -- and is interpolated into the frame rather than looked up,
    # because it is data. The plural is the catalogue's decision either way:
    # `+= "s"` is English, and French takes the singular at zero as well as
    # one.
    label_stem = (
        "caveat.dates.business_date" if role in ("", "business")
        else "caveat.dates.role_date"
    )
    date_label = _plural(label_stem, actual_days, role=role)
    metric_text = str(metric_name).strip()
    requested_amount = int(policy.get("amount") or requested_days)
    requested_unit = str(policy.get("unit") or "day").strip().lower() or "day"
    if (
        requested_unit == "day"
        and metric_active_days is not None
        and metric_active_days < min(actual_days, requested_days)
    ):
        message = _t(
            "caveat.dates.metric_sparse",
            requested=requested_amount,
            actual=actual_days,
            date_label=date_label,
            metric=metric_text or _t("caveat.dates.default_metric"),
            active=metric_active_days,
            active_label=_grain("day", metric_active_days),
            through=end_iso,
        )
    elif requested_unit == "day":
        message = _t(
            "caveat.dates.days_sparse",
            requested=requested_amount,
            subject=_metric_subject(metric_text, lead=False),
            actual=actual_days,
            date_label=date_label,
            active_label=_grain("day", actual_days),
            through=end_iso,
        )
    else:
        message = _t(
            "caveat.dates.period_sparse",
            subject_lead=_metric_subject(metric_text, lead=True),
            actual=actual_days,
            date_label=date_label,
            requested=requested_amount,
            units=_grain(requested_unit, requested_amount),
            through=end_iso,
        )

    return CoverageGap(
        requested_days=requested_days,
        actual_days=actual_days,
        message=message,
        metric_active_days=metric_active_days,
        observed_through=end_iso,
    )
