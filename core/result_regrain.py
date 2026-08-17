"""Re-grain an answered question over its own governed business date.

"What was my revenue for the past 5 days?" returns one aggregate. The natural
follow-up — "provide the trend" — is neither of the two things the pipeline knew
how to do:

  * It is not a question about the returned rows. A single total cannot be
    un-aggregated, so the governed result cache has nothing to give and the
    metadata planner correctly reports "unsupported".
  * It is not a new question either. Re-deriving the metric, the business date
    and the window from three words loses all three: metric scoping finds no
    metric in "provide the trend", so the date role has no fact to bind to and
    the answer comes back wrong or not at all.

It is the SAME query at a finer grain. So compile it that way: take the parent's
already-validated SQL, add the parent's own governed business date to the SELECT
list and the GROUP BY, order by it chronologically, and change nothing else.
The WHERE clause — and therefore the window the user asked about — is preserved
byte for byte, so the daily series always sums back to the parent's total.

No LLM is involved, and the date can only ever be the role the parent answer was
already governed by, never a newly guessed one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.contextual_dates import (
    format_date_value_expression,
    format_period_bucket_expression,
)
from core.date_roles import normalize_date_key_type

log = logging.getLogger("querybot.result_regrain")

_GRAIN_WORDS = {
    "day": "day", "days": "day", "daily": "day", "date": "day", "dates": "day",
    "week": "week", "weeks": "week", "weekly": "week",
    "month": "month", "months": "month", "monthly": "month",
    "quarter": "quarter", "quarters": "quarter", "quarterly": "quarter",
    "year": "year", "years": "year", "yearly": "year", "annual": "year",
    "annually": "year",
}

_GRAIN_ORDER = {"day": 0, "week": 1, "month": 2, "quarter": 3, "year": 4}

# Filler that may wrap the request without changing it.
_FILLER = frozenset({
    "please", "can", "you", "could", "just", "now", "also", "and", "ok", "okay",
    "thanks", "thank", "a", "an", "the", "me", "us", "it", "its", "this", "that",
    "these", "those", "as", "for", "of", "in", "on", "to", "with", "chart",
    "graph", "plot", "view", "form", "instead", "too", "well", "same",
})

# Verbs that ask for a rendering of what was already answered.
_ASK_VERBS = frozenset({
    "provide", "show", "give", "display", "plot", "draw", "chart", "graph",
    "render", "break", "split", "group", "see", "get", "make", "visualize",
    "visualise", "trend",
})

# The whole utterance must reduce to one of these, after filler is dropped.
_REGRAIN_PATTERNS = tuple(
    re.compile(rf"^(?:{body})$") for body in (
        # "trend", "the trend", "daily trend", "trend over time", "trend by day"
        r"(?:daily|weekly|monthly|quarterly|yearly|annual)?\s*"
        r"trend(?:line|s)?(?:\s+(?:over\s+time|by\s+\w+|per\s+\w+))?",
        r"time\s*series(?:\s+(?:by|per)\s+\w+)?",
        r"over\s+time",
        r"(?:by|per|each|every)\s+(?:day|date|week|month|quarter|year)",
        r"(?:day|date|week|month|quarter|year)\s*wise",
        r"(?:daily|weekly|monthly|quarterly|yearly|annual)"
        r"(?:\s+(?:breakdown|split|numbers|figures|values|totals|trend))?",
        r"break\s*down(?:\s+(?:by|per)\s+(?:day|date|week|month|quarter|year))?",
        r"(?:group|split)(?:\s+(?:by|per)\s+(?:day|date|week|month|quarter|year))",
        r"(?:trend|breakdown|series)\s+(?:by|per)\s+"
        r"(?:day|date|week|month|quarter|year)",
    )
)


@dataclass(frozen=True)
class RegrainRequest:
    """A request to re-grain the parent answer over its governed date."""

    grain: str          # "" means "inherit the parent's own grain"
    matched_phrase: str


def parse_trend_regrain_request(text: str) -> RegrainRequest | None:
    """Whether a follow-up asks for the parent answer at a finer time grain.

    Deliberately narrow, and anchored for the same reason the clarification
    rejection detector is: a message that names any new business content
    ("show the trend of orders", "revenue trend by warehouse") is a new
    question, and must reach the full pipeline rather than being silently
    answered from the previous question's SQL.
    """
    tokens = [
        token for token in re.split(r"[^a-z0-9]+", str(text or "").casefold())
        if token
    ]
    if not tokens or len(tokens) > 8:
        return None

    stripped = [token for token in tokens if token not in _FILLER]
    if not stripped:
        return None

    # A leading ask-verb is optional ("provide the trend" / "trend"), but some
    # verbs are part of the request itself ("break down by day"), so try the
    # phrase both with and without it rather than guessing which kind it is.
    candidates = [" ".join(stripped)]
    if len(stripped) > 1 and stripped[0] in _ASK_VERBS:
        candidates.append(" ".join(stripped[1:]))

    phrase = next(
        (
            candidate for candidate in candidates
            if any(pattern.match(candidate) for pattern in _REGRAIN_PATTERNS)
        ),
        "",
    )
    if not phrase:
        return None

    grain = ""
    for token in stripped:
        candidate = _GRAIN_WORDS.get(token, "")
        # An explicitly named grain wins; the finest one if several appear.
        if candidate and (
            not grain or _GRAIN_ORDER[candidate] < _GRAIN_ORDER[grain]
        ):
            grain = candidate
    return RegrainRequest(grain=grain, matched_phrase=phrase)


def temporal_policy_from_plan(semantic_plan: dict | None) -> dict:
    """Return the governed date this answer was actually built on.

    Prefers the compiled temporal policy (which carries the anchor contract),
    then the semantic plan's own contextual-date field. Both come from the
    parent answer, so the follow-up can never move to a different date role.
    """
    plan = semantic_plan or {}
    for policy in plan.get("temporal_policies") or []:
        if not isinstance(policy, dict):
            continue
        if policy.get("fact_table") and policy.get("fact_column"):
            return dict(policy)

    for field in plan.get("fields") or []:
        if not isinstance(field, dict):
            continue
        if str(field.get("role") or "") != "contextual_date":
            continue
        source_table = str(field.get("source_table") or "")
        source_column = str(field.get("source_key_column") or "")
        if not source_table or not source_column:
            continue
        field_table = str(field.get("table") or "")
        key_type = normalize_date_key_type(field.get("date_key_type"))
        is_surrogate = key_type == "surrogate_fk" and field_table != source_table
        return {
            "fact_table": source_table,
            "fact_column": source_column,
            "dimension_table": field_table if is_surrogate else "",
            "dimension_key": "",
            "date_column": str(field.get("column") or ""),
            "date_key_type": key_type,
            "temporal_grain": str(field.get("temporal_grain") or ""),
            "business_role": str(field.get("term") or ""),
            "calendar_attributes": dict(field.get("calendar_attributes") or {}),
        }
    return {}


def _dialect(db_type: str) -> str:
    value = str(db_type or "").strip().lower()
    return {
        "azure_sql": "tsql", "sql_server": "tsql", "mssql": "tsql",
        "postgresql": "postgres",
    }.get(value, value or "tsql")


def _table_parts(value: Any) -> tuple[str, ...]:
    return tuple(
        part.lower()
        for part in re.findall(r"[A-Za-z0-9_$#]+", str(value or ""))
        if part
    )


def _table_matches(actual: Any, expected: Any) -> bool:
    left = _table_parts(actual)
    right = _table_parts(expected)
    if not left or not right:
        return False
    if len(right) == 1:
        return left[-1] == right[-1]
    width = min(len(left), len(right))
    return width >= 2 and left[-width:] == right[-width:]


def resolve_regrain_grain(request: RegrainRequest, policy: dict | None) -> str:
    """Pick the period the trend is bucketed by.

    An explicitly named grain wins. Otherwise inherit the grain the governed
    date role itself is stored at, so a month-grain role is never sliced into
    days it does not have.
    """
    policy = policy or {}
    if request.grain:
        role_grain = str(policy.get("temporal_grain") or "").lower()
        if role_grain in _GRAIN_ORDER and (
            _GRAIN_ORDER[request.grain] < _GRAIN_ORDER[role_grain]
        ):
            # The user asked finer than the data is stored — honor the data.
            return role_grain
        return request.grain
    role_grain = str(policy.get("temporal_grain") or "").lower()
    if role_grain in _GRAIN_ORDER:
        return role_grain
    return "day"


def build_regrain_sql(
    parent_sql: str,
    policy: dict,
    grain: str,
    db_type: str = "azure_sql",
) -> tuple[str, str]:
    """Add the governed date to the parent SQL's SELECT, GROUP BY and ORDER BY.

    Returns ``(sql, "")`` on success, or ``("", reason)`` when the parent query
    cannot be re-grained deterministically — the caller then falls back to the
    normal pipeline rather than guessing.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except Exception:                                   # pragma: no cover
        return "", "sqlglot unavailable"

    if not parent_sql:
        return "", "no parent SQL"
    fact_table = str((policy or {}).get("fact_table") or "")
    fact_column = str((policy or {}).get("fact_column") or "")
    if not fact_table or not fact_column:
        return "", "no governed date on the parent answer"

    dialect = _dialect(db_type)
    try:
        tree = sqlglot.parse_one(str(parent_sql).strip().rstrip(";"), read=dialect)
    except Exception as exc:
        return "", f"parent SQL did not parse ({exc})"
    if not isinstance(tree, exp.Select):
        return "", "parent SQL is not a plain SELECT"

    # A re-grain only means something for an aggregated answer.
    has_aggregate = any(
        expression.find(exp.AggFunc) is not None for expression in tree.expressions
    )
    if not has_aggregate and tree.args.get("group") is None:
        return "", "parent answer is not aggregated"

    # TOP/LIMIT was applied to the parent's coarser rows. Re-graining underneath
    # it would silently return the first N days of a series instead of the
    # ranked rows the user is looking at.
    if tree.args.get("limit") is not None or tree.args.get("top") is not None:
        return "", "parent answer is row-limited"

    root_tables = [
        table for table in tree.find_all(exp.Table)
        if table.find_ancestor(exp.Select) is tree
    ]

    dimension_table = str((policy or {}).get("dimension_table") or "")
    date_column = str((policy or {}).get("date_column") or "")
    key_type = normalize_date_key_type((policy or {}).get("date_key_type"))

    if dimension_table and date_column:
        # Role-playing surrogate date: the governed dimension join is already in
        # the parent SQL (the validator requires it for any date filter), so
        # reuse that exact alias — never add a second join to the same role.
        date_table = next(
            (table for table in root_tables
             if _table_matches(
                 ".".join(
                     str(part) for part in (table.catalog, table.db, table.name)
                     if str(part or "").strip()
                 ),
                 dimension_table,
             )),
            None,
        )
        if date_table is None:
            return "", "governed date dimension is not joined in the parent SQL"
        alias = str(date_table.alias_or_name or "")
        column = date_column
        # The dimension's date value column is a real calendar date; the
        # surrogate encoding describes the FACT key, not this column.
        value_key_type = "native_date"
    else:
        source_table = next(
            (table for table in root_tables
             if _table_matches(
                 ".".join(
                     str(part) for part in (table.catalog, table.db, table.name)
                     if str(part or "").strip()
                 ),
                 fact_table,
             )),
            None,
        )
        if source_table is None:
            return "", "governed fact table is not in the parent SQL"
        alias = str(source_table.alias_or_name or "")
        column = date_column or fact_column
        value_key_type = key_type

    if not column:
        return "", "governed date has no calendar column"

    # Already a trend — the date is in the output, so there is nothing to add.
    for projection in tree.expressions:
        for col in projection.find_all(exp.Column):
            if str(col.name or "").casefold() == column.casefold():
                return "", "parent answer already reports this date"

    date_ref = format_date_value_expression(alias, column, value_key_type, db_type)
    bucket_sql = format_period_bucket_expression(
        date_ref,
        grain,
        db_type,
        role_alias=alias,
        calendar_attributes=(policy or {}).get("calendar_attributes") or {},
    )
    try:
        bucket_expr = sqlglot.parse_one(bucket_sql, read=dialect)
    except Exception as exc:
        return "", f"period expression did not compile ({exc})"

    label = str((policy or {}).get("business_role") or "").strip() or "Date"
    label = re.sub(r"[_\s]+", " ", label).strip().title()
    if grain != "day":
        label = f"{label} ({grain.title()})"

    # The period leads the SELECT list: a trend reads left-to-right in time, and
    # chart selection treats the first column as the category axis.
    tree.set(
        "expressions",
        [exp.alias_(bucket_expr.copy(), label, quoted=True), *tree.expressions],
    )
    tree.group_by(bucket_expr.copy(), append=True, copy=False)
    # A trend reads chronologically. Any ranking the parent applied stays, but
    # after the period, so each period's rows remain ordered as before.
    existing_order = tree.args.get("order")
    tree.set("order", None)
    tree.order_by(bucket_expr.copy(), append=False, copy=False)
    if existing_order is not None:
        for ordered in existing_order.expressions:
            tree.order_by(ordered.copy(), append=True, copy=False)

    try:
        return tree.sql(dialect=dialect), ""
    except Exception as exc:                            # pragma: no cover
        return "", f"re-grained SQL did not render ({exc})"


def regrain_question_text(parent_question: str, grain: str) -> str:
    """Restate the parent question at the requested grain.

    Used for the answer's own title, and as the fallback question when the
    deterministic rewrite cannot be compiled — either way the metric, the
    business date and the window come from the parent question rather than
    from three words of follow-up.
    """
    base = re.sub(r"[?\s]+$", "", str(parent_question or "").strip())
    if not base:
        return f"Trend by {grain}"
    suffix = {
        "day": "by day", "week": "by week", "month": "by month",
        "quarter": "by quarter", "year": "by year",
    }.get(grain, "by day")
    if re.search(rf"\b{re.escape(suffix)}\b", base, re.I):
        return base
    return f"{base} {suffix}"
