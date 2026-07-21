"""Metadata-only planning for conversational cached-result analysis.

The planner may see a sanitized user intent and a manifest of column names,
types, formats, and row count. It must never see cached rows, sample values,
source SQL, statistics, or locally bound filter literals. The model returns a
small JSON AST which is validated and compiled into the deterministic
``ResultCommand`` execution path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.result_commands import ResultCommand, resolve_result_column


_RESULT_CONTEXT_RE = re.compile(
    r"\b(?:this|the|these|current|previous|cached|returned)\s+"
    r"(?:result|results|data|dataset|rows?|table)\b",
    re.IGNORECASE,
)
_ANALYTIC_INTENT_RE = re.compile(
    r"\b(?:filter|where|group|summari[sz]e|aggregate|total|sum|average|avg|"
    r"mean|count|minimum|min|maximum|max|percentage|percent|contribution|"
    r"ratio|divided|sort|rank|top|bottom|greater|less|above|below|contains|"
    r"starts?\s+with|ends?\s+with)\b",
    re.IGNORECASE,
)
_MASKED_MARKERS = {"redacted", "masked", "hidden", "protected", "restricted"}
_ALLOWED_OPERATIONS = {
    "filter", "aggregate", "contribution", "ratio", "profit_percentage",
    "sort", "keep_top",
}
_ALLOWED_AGGREGATIONS = {"sum", "avg", "count", "min", "max"}
_ALLOWED_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "starts_with", "ends_with"}
_ALLOWED_DIRECTIONS = {"asc", "desc"}
_PLAN_KEYS = {
    "operation", "dimension", "metric", "aggregation", "operator",
    "value_ref", "numerator", "denominator", "direction", "limit_ref",
}
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_GUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)
_PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")


@dataclass(frozen=True)
class PlannerInput:
    sanitized_question: str
    system_prompt: str
    user_prompt: str
    bindings: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerResult:
    ok: bool
    command: ResultCommand | None = None
    reason: str = ""
    binding_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def is_metadata_result_question(text: str) -> bool:
    """Return True only for analytical wording explicitly tied to a result."""
    value = str(text or "")
    return bool(_RESULT_CONTEXT_RE.search(value) and _ANALYTIC_INTENT_RE.search(value))


def strip_result_context(text: str) -> str:
    """Remove cache-reference filler before a governed source-query fallback."""
    cleaned = _RESULT_CONTEXT_RE.sub("the data", str(text or ""))
    cleaned = re.sub(r"\b(?:using|from|based\s+on)\s+the\s+data\s*,?", "", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip(" ,")


def build_planner_input(question: str, snapshot: dict) -> PlannerInput:
    """Build a zero-row planner prompt and retain literal bindings locally."""
    schema = _metadata_schema(snapshot)
    sanitized, bindings = _sanitize_question(question, snapshot, schema)
    manifest = {
        "row_count": int(snapshot.get("row_count") or len(snapshot.get("rows") or [])),
        "columns": schema,
    }
    system_prompt = (
        "You plan analysis over a session-local result table. Return exactly one JSON object, "
        "with no markdown and no SQL. You receive metadata only: column names, types, formats, "
        "row count, and a sanitized request. Never invent a column or literal.\n\n"
        "Allowed operation values: filter, aggregate, contribution, ratio, "
        "profit_percentage, sort, keep_top.\n"
        "Allowed keys: operation, dimension, metric, aggregation, operator, value_ref, "
        "numerator, denominator, direction, limit_ref.\n"
        "Use exact column names from the manifest. aggregation must be sum, avg, count, min, "
        "or max. operator must be eq, ne, gt, gte, lt, lte, contains, starts_with, or "
        "ends_with. direction must be asc or desc. A filter literal must be represented only "
        "by a VALUE_REF_n token already present in the request. A keep_top limit must use a "
        "VALUE_REF_n token. Do not output a raw value or a numeric limit. If the request cannot "
        "be expressed exactly, return {\"operation\":\"unsupported\"}."
    )
    user_prompt = json.dumps(
        {"request": sanitized, "result_metadata": manifest},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return PlannerInput(
        sanitized_question=sanitized,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        bindings=bindings,
        metadata={
            "metadata_only": True,
            "row_count_disclosed": manifest["row_count"],
            "column_count_disclosed": len(schema),
            "rows_sent_to_llm": 0,
            "sample_values_sent_to_llm": 0,
            "source_sql_sent_to_llm": False,
            "literal_binding_count": len(bindings),
        },
    )


async def plan_result_command(
    question: str,
    snapshot: dict,
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
) -> PlannerResult:
    """Ask a model for a constrained AST, then compile it locally."""
    planner_input = build_planner_input(question, snapshot)
    try:
        raw, _, _ = await complete(
            system=planner_input.system_prompt,
            user=planner_input.user_prompt,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        return PlannerResult(
            False,
            reason="The metadata planner was unavailable.",
            binding_count=len(planner_input.bindings),
            metadata=planner_input.metadata,
        )
    command, reason = compile_planner_response(
        raw,
        snapshot,
        planner_input.bindings,
    )
    return PlannerResult(
        command is not None,
        command=command,
        reason=reason,
        binding_count=len(planner_input.bindings),
        metadata=planner_input.metadata,
    )


def compile_planner_response(
    raw_response: str,
    snapshot: dict,
    bindings: dict[str, Any],
) -> tuple[ResultCommand | None, str]:
    """Validate a model JSON plan and compile it into a local command."""
    plan = _parse_json_object(raw_response)
    if plan is None:
        return None, "The metadata planner did not return valid JSON."
    if set(plan) - _PLAN_KEYS:
        return None, "The metadata planner returned unsupported fields."
    operation = str(plan.get("operation") or "").strip().lower()
    if operation == "unsupported":
        return None, "The cached result cannot safely answer that request."
    if operation not in _ALLOWED_OPERATIONS:
        return None, "The metadata planner returned an unsupported operation."

    columns = _column_lookup(snapshot)

    def column(key: str, *, required: bool = True) -> str:
        value = str(plan.get(key) or "").strip()
        if not value and not required:
            return ""
        exact = columns.get(_normalise(value), "")
        if exact:
            return exact
        resolved, _ = resolve_result_column(list(snapshot.get("rows") or []), value)
        return resolved

    if operation == "filter":
        target = column("metric") or column("dimension")
        operator = str(plan.get("operator") or "").strip().lower()
        value_ref = str(plan.get("value_ref") or "").strip()
        if not target or operator not in _ALLOWED_OPERATORS or value_ref not in bindings:
            return None, "The filter plan was not grounded to local metadata and bindings."
        if _normalise(bindings[value_ref]) in _MASKED_MARKERS:
            return None, "A generic masked value cannot be used as a local filter."
        return ResultCommand(
            "filter",
            target_text=target,
            operator=operator,
            value_text=str(bindings[value_ref]),
        ), ""

    if operation == "aggregate":
        dimension = column("dimension")
        metric_text = str(plan.get("metric") or "").strip()
        metric = "*" if metric_text == "*" else column("metric")
        aggregation = str(plan.get("aggregation") or "").strip().lower()
        if (
            not dimension
            or not metric
            or aggregation not in _ALLOWED_AGGREGATIONS
            or (metric == "*" and aggregation != "count")
        ):
            return None, "The aggregate plan was not grounded to local metadata."
        return ResultCommand(
            "aggregate",
            dimension_text=dimension,
            metric_text=metric,
            aggregation=aggregation,
        ), ""

    if operation == "contribution":
        dimension, metric = column("dimension"), column("metric")
        if not dimension or not metric:
            return None, "The contribution plan was not grounded to local metadata."
        return ResultCommand("contribution", dimension_text=dimension, metric_text=metric), ""

    if operation == "ratio":
        numerator, denominator = column("numerator"), column("denominator")
        if not numerator or not denominator or numerator == denominator:
            return None, "The ratio plan was not grounded to two local columns."
        return ResultCommand(
            "ratio",
            numerator_text=numerator,
            denominator_text=denominator,
        ), ""

    if operation == "profit_percentage":
        return ResultCommand("profit_percentage"), ""

    if operation == "sort":
        target = column("metric") or column("dimension")
        direction = str(plan.get("direction") or "desc").strip().lower()
        if not target or direction not in _ALLOWED_DIRECTIONS:
            return None, "The sort plan was not grounded to local metadata."
        return ResultCommand("sort", target_text=target, direction=direction), ""

    limit_ref = str(plan.get("limit_ref") or "").strip()
    if limit_ref not in bindings:
        return None, "The row limit was not locally bound."
    try:
        limit = int(str(bindings[limit_ref]).replace(",", ""))
    except (TypeError, ValueError):
        return None, "The row limit was not numeric."
    metric = column("metric", required=False)
    return ResultCommand(
        "keep_top",
        limit=max(1, min(limit, 1000)),
        metric_text=metric,
    ), ""


def _metadata_schema(snapshot: dict) -> list[dict[str, str]]:
    formats = {
        _normalise(name): str(value or "number").lower()
        for name, value in (snapshot.get("column_formats") or {}).items()
    }
    output: list[dict[str, str]] = []
    for item in snapshot.get("schema") or []:
        name = str((item or {}).get("name") or "").strip()
        if not name:
            continue
        dtype = str((item or {}).get("type") or "TEXT").upper()
        default_format = "text" if dtype in {"TEXT", "VARCHAR", "STRING"} else "number"
        output.append({
            "name": name,
            "type": dtype,
            "format": formats.get(_normalise(name), default_format),
        })
    return output


def _column_lookup(snapshot: dict) -> dict[str, str]:
    return {
        _normalise(item.get("name")): str(item.get("name"))
        for item in _metadata_schema(snapshot)
        if item.get("name")
    }


def _sanitize_question(
    question: str,
    snapshot: dict,
    schema: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    text = str(question or "")
    bindings: dict[str, Any] = {}
    column_names = {_normalise(item["name"]) for item in schema}

    candidates: list[tuple[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in snapshot.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if value is None or isinstance(value, (dict, list, tuple, set, bool)):
                continue
            display = str(value).strip()
            if len(display) < 2 or _normalise(display) in _MASKED_MARKERS:
                continue
            key = (type(value).__name__, display.casefold())
            if key not in seen and _normalise(display) not in column_names:
                seen.add(key)
                candidates.append((display, value))
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    def bind(display: str, actual: Any) -> str:
        for ref, value in bindings.items():
            if type(value) is type(actual) and value == actual:
                return ref
        ref = f"VALUE_REF_{len(bindings) + 1}"
        bindings[ref] = actual
        return ref

    for display, actual in candidates:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(display)}(?![A-Za-z0-9_])", re.I)
        if pattern.search(text):
            text = pattern.sub(lambda _match, d=display, a=actual: bind(d, a), text)

    # Protect common identifiers even when the requested value is not in the
    # current slice. Binding remains local and is never logged or serialized.
    text = _EMAIL_RE.sub(lambda match: bind(match.group(0), match.group(0)), text)
    text = _GUID_RE.sub(lambda match: bind(match.group(0), match.group(0)), text)
    text = _PHONE_RE.sub(lambda match: bind(match.group(0), match.group(0)), text)

    # Bind remaining quoted literals and standalone numbers locally. This
    # includes limits such as "top 5" and literals not present in the rows.
    quoted_re = re.compile(r"(['\"])([^'\"\n]{1,160})\1")
    text = quoted_re.sub(lambda match: bind(match.group(2), match.group(2)), text)

    number_re = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_])")

    def bind_number(match: re.Match) -> str:
        raw = match.group(0)
        cleaned = raw.replace(",", "")
        actual: Any = float(cleaned) if "." in cleaned else int(cleaned)
        return bind(raw, actual)

    text = number_re.sub(bind_number, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, bindings


def _parse_json_object(raw_response: str) -> dict | None:
    raw = str(raw_response or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
