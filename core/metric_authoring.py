"""Turn "revenue per active customer" into a metric, without the model writing SQL.

The model fills structured slots -- an aggregation, a column reference, a
filter -- and ``core.metric_builder`` compiles those into a formula locally.
That is the whole safety argument. The existing AI-import route
(``admin/routes.py``'s ``/metrics/api/ai-import``) accepts a ``row_expression``
SQL string straight from the model; here there is no key in the allow-list that
can carry SQL text, so there is nothing to sanitise and nothing to get wrong.

Tables and columns are bound to ``TABLE_REF_n`` / ``COL_REF_n`` tokens. The
model can reference them and cannot invent them: a name it makes up matches no
binding and the plan is refused. Same discipline as
``core.report_planner``'s ``METRIC_REF_n`` and stronger than the entity-graph
chat's raw-name-plus-manifest-check, because there is no raw name to check.

The prompt carries table and column NAMES only -- never a row, never a value --
so no ``_VALUE_BEARING_MARKERS`` entry is required in the egress manifest.
Anyone adding a sample value to this prompt must add one.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger("querybot.metric_authoring")

_PLAN_KEYS = {
    "operation", "name", "synonyms", "description", "result_format", "mode",
    "aggregation", "measure_ref", "filters", "numerator", "denominator",
    "base_table_ref", "dimension_refs", "time_column_ref", "confidence",
}
_SIDE_KEYS = {"aggregation", "measure_ref", "filters"}
_FILTER_KEYS = {"field_ref", "operator", "value"}
_MODES = {"aggregate", "ratio"}
_AGGREGATIONS = {"SUM", "AVG", "COUNT", "MIN", "MAX", "COUNT_DISTINCT"}
_OPERATORS = {
    "equals", "not_equals", "greater_than", "less_than",
    "greater_or_equal", "less_or_equal", "contains", "in", "not_null",
}
_RESULT_FORMATS = {"number", "currency", "percentage", "date", "text"}

# Fail closed. A response with missing or malformed confidence is not evidence
# of high confidence -- same rule as report_planner and graph_commands.
_DEFAULT_CONFIDENCE = 0.0

_REF_RE = re.compile(r"^(?:TABLE|COL)_REF_\d+$")


@dataclass(frozen=True)
class MetricDraft:
    name: str
    sql_template: str
    required_columns: str
    base_table: str
    metric_builder_config: str
    source_tables: tuple[str, ...]
    synonyms: str = ""
    description: str = ""
    result_format: str = "number"
    formula_type: str = "expression"
    allowed_dimensions: str = ""
    default_time_column: str = ""
    confidence: float = _DEFAULT_CONFIDENCE

    def as_metric(self) -> dict[str, Any]:
        """metric_registry-shaped, so every downstream consumer treats it the
        same as an approved metric."""
        return {
            "name": self.name,
            "synonyms": self.synonyms,
            "description": self.description,
            "sql_template": self.sql_template,
            "formula_type": self.formula_type,
            "result_format": self.result_format,
            "required_columns": self.required_columns,
            "allowed_dimensions": self.allowed_dimensions,
            "default_time_column": self.default_time_column,
            "metric_builder_config": self.metric_builder_config,
            "base_table": self.base_table,
        }


@dataclass(frozen=True)
class MetricPlanInput:
    system_prompt: str
    user_prompt: str
    tables: dict[str, str] = field(default_factory=dict)   # TABLE_REF_n -> fqn
    columns: dict[str, tuple[str, str]] = field(default_factory=dict)  # COL_REF_n -> (fqn, column)


def build_metric_plan_input(
    text: str,
    schema_manifest: dict[str, list[str]],
    existing_metric_names: list[str] | None = None,
    history: list[dict] | None = None,
) -> MetricPlanInput:
    """``schema_manifest``: {table_fqn: [column, ...]}, ALREADY ACL-filtered.

    The caller must filter before calling this, so a table the user cannot see
    is never even offered as an option -- the same ordering the report builder
    enforces and its wiring test asserts.
    """
    tables: dict[str, str] = {}
    columns: dict[str, tuple[str, str]] = {}
    lines: list[str] = []
    for fqn, cols in (schema_manifest or {}).items():
        table_ref = f"TABLE_REF_{len(tables) + 1}"
        tables[table_ref] = fqn
        column_refs = []
        for column in cols or []:
            col_ref = f"COL_REF_{len(columns) + 1}"
            columns[col_ref] = (fqn, str(column))
            column_refs.append(f"{col_ref}={column}")
        lines.append(f"{table_ref} ({fqn}): " + ", ".join(column_refs))
    manifest = "\n".join(lines)[:12000]

    taken = ", ".join(sorted(set(existing_metric_names or []))[:60])
    system_prompt = (
        "You turn a plain-language description of a business calculation into a "
        "structured metric definition. Return exactly one JSON object, no markdown, "
        "no explanation.\n\n"
        "You NEVER write SQL. You choose an aggregation and reference columns by "
        "their COL_REF_n token; the formula is compiled from your choices. A name "
        "you invent that is not a token in the manifest will be rejected.\n\n"
        "Fields:\n"
        '  operation      "define_metric" or "unsupported"\n'
        '  name           short business name, Title Case\n'
        '  synonyms       comma-separated alternative phrasings, optional\n'
        '  description    one sentence, optional\n'
        '  result_format  number | currency | percentage | date | text\n'
        '  mode           "aggregate" for a total, "ratio" for an X-per-Y\n'
        '  base_table_ref TABLE_REF_n the calculation is anchored on\n'
        "  For mode=aggregate: aggregation (SUM|AVG|COUNT|MIN|MAX|COUNT_DISTINCT), "
        "measure_ref (COL_REF_n), filters\n"
        "  For mode=ratio: numerator and denominator, each an object with "
        "aggregation, measure_ref and optional filters\n"
        '  filters        list of {"field_ref": COL_REF_n, "operator": '
        f'{sorted(_OPERATORS)}, "value": "..."}}\n'
        "  dimension_refs COL_REF_n columns this may be broken down by, optional\n"
        "  time_column_ref COL_REF_n the date column, optional\n"
        "  confidence     0.0-1.0, how sure you are this matches the request\n\n"
        'Return {"operation":"unsupported"} if the request does not describe a '
        "calculation, or if the columns needed are not in the manifest.\n\n"
        f"MANIFEST\n{manifest}\n"
        + (f"\nMetric names already in use (do not reuse): {taken}\n" if taken else "")
    )

    turns = ""
    for turn in (history or [])[-8:]:
        role = str((turn or {}).get("role") or "")
        content = str((turn or {}).get("content") or "")[:500]
        if role in {"user", "assistant"} and content:
            turns += f"{role}: {content}\n"
    user_prompt = (turns + f"user: {text}") if turns else str(text or "")
    return MetricPlanInput(system_prompt, user_prompt, tables, columns)


async def parse_metric_plan(
    text: str,
    schema_manifest: dict[str, list[str]],
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
    *,
    existing_metric_names: list[str] | None = None,
    history: list[dict] | None = None,
    db_type: str = "azure_sql",
) -> tuple[MetricDraft | None, str]:
    """Ask a model for a constrained plan, then compile it locally."""
    plan_input = build_metric_plan_input(text, schema_manifest, existing_metric_names, history)
    try:
        raw, _, _ = await complete(
            system=plan_input.system_prompt,
            user=plan_input.user_prompt,
            temperature=0.0,
            max_tokens=700,
        )
    except Exception as exc:
        log.info("Metric planner unavailable: %s", exc)
        return None, "The metric planner was unavailable."
    return compile_metric_plan_response(raw, plan_input, db_type=db_type)


def _resolve_column(ref: Any, plan_input: MetricPlanInput) -> tuple[str, str] | None:
    key = str(ref or "").strip()
    if not _REF_RE.match(key):
        return None
    return plan_input.columns.get(key)


def _compile_filters(raw: Any, plan_input: MetricPlanInput) -> tuple[list[dict], list[str], str]:
    """Return (filters, tables_touched, error)."""
    filters: list[dict] = []
    tables: list[str] = []
    if raw is None:
        return filters, tables, ""
    if not isinstance(raw, list):
        return [], [], "The metric planner returned malformed filters."
    for item in raw:
        if not isinstance(item, dict) or set(item) - _FILTER_KEYS:
            return [], [], "The metric planner returned an unsupported filter."
        resolved = _resolve_column(item.get("field_ref"), plan_input)
        if not resolved:
            return [], [], f'"{item.get("field_ref")}" is not a column available to you.'
        operator = str(item.get("operator") or "equals").strip().lower()
        if operator not in _OPERATORS:
            return [], [], f'"{operator}" is not a supported filter operator.'
        table, column = resolved
        tables.append(table)
        filters.append({
            "field": column, "operator": operator, "value": str(item.get("value") or ""),
        })
    return filters, tables, ""


def _compile_side(raw: Any, label: str, plan_input: MetricPlanInput) -> tuple[dict | None, list[str], str]:
    if not isinstance(raw, dict) or set(raw) - _SIDE_KEYS:
        return None, [], f"The metric planner returned an unsupported {label}."
    aggregation = str(raw.get("aggregation") or "").strip().upper()
    if aggregation not in _AGGREGATIONS:
        return None, [], f'"{aggregation}" is not a supported {label} aggregation.'
    resolved = _resolve_column(raw.get("measure_ref"), plan_input)
    if not resolved:
        return None, [], f'The {label} references a column that is not available to you.'
    table, column = resolved
    filters, filter_tables, error = _compile_filters(raw.get("filters"), plan_input)
    if error:
        return None, [], error
    return (
        {"aggregation": aggregation, "measure": column, "filters": filters},
        [table, *filter_tables],
        "",
    )


def compile_metric_plan_response(
    raw_response: str,
    plan_input: MetricPlanInput,
    *,
    db_type: str = "azure_sql",
) -> tuple[MetricDraft | None, str]:
    """Validate a model plan against the local bindings and compile it.

    Never trusts a table or column the model names: every reference must match
    a token this process issued.
    """
    plan = _parse_json_object(raw_response)
    if plan is None:
        return None, "The metric planner did not return valid JSON."
    if set(plan) - _PLAN_KEYS:
        return None, "The metric planner returned unsupported fields."

    operation = str(plan.get("operation") or "").strip().lower()
    if operation == "unsupported":
        return None, "That calculation could not be built from the data available to you."
    if operation != "define_metric":
        return None, "The metric planner returned an unsupported operation."

    name = str(plan.get("name") or "").strip()
    if not name:
        return None, "The metric planner did not name the metric."

    mode = str(plan.get("mode") or "aggregate").strip().lower()
    if mode not in _MODES:
        return None, f'"{mode}" is not a supported metric shape.'

    tables_touched: list[str] = []
    if mode == "ratio":
        numerator, num_tables, error = _compile_side(plan.get("numerator"), "numerator", plan_input)
        if error:
            return None, error
        denominator, den_tables, error = _compile_side(plan.get("denominator"), "denominator", plan_input)
        if error:
            return None, error
        builder_config = {
            "enabled": True, "mode": "ratio",
            "numerator": numerator, "denominator": denominator,
        }
        tables_touched = num_tables + den_tables
    else:
        side, side_tables, error = _compile_side(
            {
                "aggregation": plan.get("aggregation"),
                "measure_ref": plan.get("measure_ref"),
                "filters": plan.get("filters"),
            },
            "measure", plan_input,
        )
        if error:
            return None, error
        builder_config = {
            "enabled": True, "mode": "aggregate",
            "aggregation": side["aggregation"], "measure": side["measure"],
            "filters": side["filters"],
        }
        tables_touched = side_tables

    # Compiled locally from the structured plan. This is the step that means no
    # SQL the model produced can reach the formula, because it never produced any.
    try:
        from core.metric_builder import compile_metric_builder_config

        compiled = compile_metric_builder_config(builder_config, db_type)
    except ValueError as exc:
        return None, f"That calculation could not be compiled: {exc}"
    if compiled is None:
        return None, "That calculation could not be compiled."

    # A ${MetricName} reference resolves against list_metrics, which cannot see
    # a session draft -- it would raise inside a try/except and ship the
    # unresolved literal into the prompt. Refuse rather than compose.
    if "${" in compiled.formula:
        return None, "A composed metric cannot reference another metric."

    base_table = ""
    base_ref = str(plan.get("base_table_ref") or "").strip()
    if base_ref in plan_input.tables:
        base_table = plan_input.tables[base_ref]
    elif tables_touched:
        base_table = tables_touched[0]

    dimensions: list[str] = []
    for ref in (plan.get("dimension_refs") or []) if isinstance(plan.get("dimension_refs"), list) else []:
        resolved = _resolve_column(ref, plan_input)
        if resolved:
            dimensions.append(resolved[1])
            tables_touched.append(resolved[0])

    time_column = ""
    resolved_time = _resolve_column(plan.get("time_column_ref"), plan_input)
    if resolved_time:
        time_column = resolved_time[1]
        tables_touched.append(resolved_time[0])

    result_format = str(plan.get("result_format") or "number").strip().lower()
    if result_format not in _RESULT_FORMATS:
        result_format = "number"

    return MetricDraft(
        name=name[:120],
        sql_template=compiled.formula,
        required_columns=", ".join(compiled.required_columns),
        base_table=base_table,
        metric_builder_config=compiled.config_json,
        source_tables=tuple(dict.fromkeys(t for t in tables_touched if t)),
        synonyms=str(plan.get("synonyms") or "").strip()[:300],
        description=str(plan.get("description") or "").strip()[:500],
        result_format=result_format,
        allowed_dimensions=", ".join(dict.fromkeys(dimensions)),
        default_time_column=time_column,
        confidence=_extract_confidence(plan),
    ), ""


def schema_columns_for_draft(
    draft: MetricDraft, schema_columns: dict[str, list[str]],
) -> dict[str, list[str]]:
    """The columns a multi-table metric is allowed to reference.

    ``core.metric_validator.validate_metric`` checks ``required_columns`` against
    ``base_table`` alone. That is right for a single-table metric and wrong for a
    ratio spanning a fact and a dimension -- which is precisely the shape this
    feature exists to compose. "Revenue per active customer" takes the amount
    from the invoice fact and the active flag from the customer master, and the
    flag is quite correctly not on the fact.

    So the base_table entry is widened to the union of the tables the draft
    itself declares, making the check mean "these columns exist in the tables
    this metric uses". The union is bounded by the draft's own source_tables,
    every one of which came from a COL_REF binding, so this cannot admit a
    column from a table the user was never offered.

    Whether the join between those tables is correct is a different question,
    and it is the live dry run that answers it.
    """
    widened = dict(schema_columns or {})
    if len(draft.source_tables) <= 1 or not draft.base_table:
        return widened

    def _resolve(table: str) -> list[str]:
        target = str(table or "").upper()
        for fqn, columns in (schema_columns or {}).items():
            fqn_upper = str(fqn).upper()
            if fqn_upper == target or fqn_upper.endswith("." + target.split(".")[-1]):
                return list(columns or [])
        return []

    union: list[str] = []
    for table in draft.source_tables:
        for column in _resolve(table):
            if column not in union:
                union.append(column)
    if union:
        widened[draft.base_table] = union
    return widened


def _extract_confidence(plan: dict) -> float:
    raw = plan.get("confidence")
    if raw is None:
        return _DEFAULT_CONFIDENCE
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if value != value:  # NaN
        return _DEFAULT_CONFIDENCE
    return max(0.0, min(1.0, value))


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


# ── Deterministic paste form ────────────────────────────────────────────────
# An administrator who already knows the answer should not have to wait for a
# model to rediscover it. `Net Revenue = SUM(AMOUNT) - SUM(DISCOUNT) FROM
# DW.SALES_FACT` is unambiguous, so it is parsed here, with no LLM call, no
# token spend, and no chance of an invented column.
#
# The output is the SAME MetricDraft the model path produces, so everything
# downstream -- validation, dry run, the proposal record, source_tables and the
# ACL re-check that rides on them -- is identical either way. A second draft
# shape would be a second set of governance holes.

_EXPLICIT_RE = re.compile(
    r"^\s*(?P<name>[^=]{1,120}?)\s*=\s*(?P<expr>.+?)\s+FROM\s+(?P<table>[\w\.\[\]`\"]+)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_AGG_CALL_RE = re.compile(
    r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(\s*(DISTINCT\s+)?([A-Za-z_][\w]*)\s*\)",
    re.IGNORECASE,
)
# Everything an expression may contain BESIDES aggregate calls. Anything else
# -- a subselect, a semicolon, a second FROM, a function we have not vetted --
# means this is not the simple paste form and must not be waved through.
_EXPR_RESIDUE_RE = re.compile(r"^[\s\d\+\-\*/\(\)\.,]*$")


def parse_explicit_metric_definition(
    text: str,
    schema_manifest: dict[str, list[str]],
    *,
    db_type: str = "azure_sql",
) -> tuple[MetricDraft | None, str]:
    """Parse ``name = <aggregates> FROM <table>`` without an LLM.

    Returns ``(None, "")`` when the text simply is not this shape -- that is not
    an error, it means "hand this to the planner". A non-empty reason means the
    text LOOKED like the paste form and was rejected, which the caller should
    show rather than silently retrying with the model.

    ``schema_manifest`` must already be ACL-filtered: the table has to be one
    the caller was offered, and every column has to exist on it. That is what
    stops a pasted definition reaching a table the admin cannot see.
    """
    raw = str(text or "").strip()
    if not raw or "=" not in raw:
        return None, ""
    match = _EXPLICIT_RE.match(raw)
    if not match:
        return None, ""

    name = " ".join(match.group("name").split())
    expr = " ".join(match.group("expr").split())
    table_raw = match.group("table").strip().strip("[]`\"")

    # Resolve the table against the manifest, accepting an unambiguous trailing
    # qualification the way the SQL validator and the ACL layer already do.
    by_full = {str(t).casefold(): t for t in (schema_manifest or {})}
    resolved = by_full.get(table_raw.casefold(), "")
    if not resolved:
        bare = table_raw.split(".")[-1].casefold()
        hits = [t for t in (schema_manifest or {}) if str(t).split(".")[-1].casefold() == bare]
        if len(hits) == 1:
            resolved = hits[0]
        elif len(hits) > 1:
            return None, f'"{table_raw}" is ambiguous — it matches {len(hits)} tables.'
    if not resolved:
        return None, f'"{table_raw}" is not a table you have access to.'

    calls = _AGG_CALL_RE.findall(expr)
    if not calls:
        return None, ""      # no aggregate at all: not this form, try the planner

    # Whatever is left once the aggregate calls are removed must be arithmetic.
    if not _EXPR_RESIDUE_RE.match(_AGG_CALL_RE.sub("", expr)):
        return None, (
            "That formula has parts I will not accept from a paste. Describe it "
            "in words instead and I will compose it."
        )

    available = {str(c).casefold(): str(c) for c in (schema_manifest.get(resolved) or [])}
    columns: list[str] = []
    for _agg, _distinct, column in calls:
        actual = available.get(column.casefold())
        if not actual:
            return None, f'"{column}" is not a column on {resolved}.'
        if actual not in columns:
            columns.append(actual)

    # Rebuild the expression from the RESOLVED column names rather than reusing
    # the pasted text, so casing is canonical and nothing unparsed survives.
    def _rebuild(m: re.Match[str]) -> str:
        agg = m.group(1).upper()
        distinct = "DISTINCT " if m.group(2) else ""
        return f"{agg}({distinct}{available[m.group(3).casefold()]})"

    formula = _AGG_CALL_RE.sub(_rebuild, expr)

    return MetricDraft(
        name=name[:120],
        sql_template=formula,
        required_columns=", ".join(columns),
        base_table=resolved,
        # No builder config: this is a hand-written expression, and claiming a
        # structured config it does not have would make the metric editor show
        # a builder that cannot round-trip it.
        metric_builder_config="",
        source_tables=(resolved,),
        # Deterministic parse of an explicit statement. There is no model
        # guessing here, so there is nothing to be less than certain about.
        confidence=1.0,
    ), ""
