"""Metric selection helpers for multi-schema questions.

The metric registry is account-wide, but SQL generation needs a narrower view:
when a question mentions a schema-specific dimension/entity, only metrics from
that same schema should be enforced.  This module keeps that scoping logic
small and deterministic so generic names like "Revenue" do not override a more
specific schema-local metric such as "Total Revenue USD".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "each", "find", "for", "from",
    "give", "how", "in", "is", "me", "my", "of", "on", "or", "per", "show",
    "the", "to", "total", "what", "with",
}


@dataclass
class MetricScopeResult:
    metrics: list[dict[str, Any]]
    ambiguous: bool = False
    options: list[str] | None = None
    reason: str = ""
    context_schemas: set[str] | None = None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", _norm(text))
        if len(tok) > 1 and tok not in _STOP_WORDS
    }


def _table_schema(table: str) -> str:
    parts = re.split(r"[.\[\]\"]+", (table or "").upper())
    parts = [p for p in parts if p]
    return parts[-2] if len(parts) >= 2 else ""



def _metric_phrases(metric: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for key in ("name", "synonyms", "example_questions"):
        value = str(metric.get(key) or "")
        for part in re.split(r"[,;\n]+", value):
            part = _norm(part)
            if part:
                phrases.append(part)
    return list(dict.fromkeys(phrases))


def _phrase_score(metric: dict[str, Any], question: str) -> int:
    q = _norm(question)
    q_tokens = _tokens(question)
    if not q_tokens:
        return 0
    best = 0
    for phrase in _metric_phrases(metric):
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens:
            continue
        score = len(q_tokens & phrase_tokens) * 10
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q):
            score += 100 + len(phrase_tokens) * 12
        best = max(best, score)
    metadata_tokens = _tokens(" ".join([
        str(metric.get("required_columns") or ""),
        str(metric.get("allowed_dimensions") or ""),
        str(metric.get("grain") or ""),
        str(metric.get("category") or ""),
    ]))
    return best + len(q_tokens & metadata_tokens)


def _split_required_columns(raw: str) -> set[str]:
    ignore = {
        "AS", "CASE", "CAST", "COALESCE", "ELSE", "END", "ISNULL", "NULLIF",
        "SUM", "THEN", "WHEN",
    }
    cols: set[str] = set()
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw or ""):
        token_u = token.upper()
        if token_u not in ignore:
            cols.add(token_u)
    return cols


def _sql_tables(sql: str) -> set[str]:
    tables: set[str] = set()
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+)){0,2})",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql or ""):
        raw = re.sub(r"\s+", "", match.group(1))
        raw = raw.replace("[", "").replace("]", "").replace('"', "")
        if raw:
            tables.add(raw.upper())
    return tables


def metric_source_tables(metric: dict[str, Any], table_columns: dict[str, dict[str, str]] | None) -> set[str]:
    """Infer metric source tables from base_table, SQL, and required columns."""
    tables: set[str] = set()
    base_table = str(metric.get("base_table") or "").strip()
    if base_table:
        tables.add(base_table.upper())
    tables.update(_sql_tables(str(metric.get("sql_template") or "")))

    required = _split_required_columns(str(metric.get("required_columns") or ""))
    if table_columns and required:
        for table, cols in table_columns.items():
            col_names = {str(c).upper() for c in (cols or {})}
            if required & col_names:
                tables.add(str(table).upper())
    return tables


def metric_source_schemas(
    metric: dict[str, Any],
    table_columns: dict[str, dict[str, str]] | None,
    entity_schema_map: dict[str, str] | None = None,
) -> set[str]:
    """Return the set of schema names this metric's source tables belong to.

    Resolution order (first match wins):
    1. base_entity → entity_schema_map lookup (most reliable: the entity graph
       always has schema_name; metrics created through the UI have base_entity)
    2. base_table / sql_template table parsing (works when the admin set an FQN
       like PHARMACY.FACT_PRESCRIPTION_FILL in base_table)
    3. required_columns matched against all_columns from _schema.json (fallback
       when base_table is a bare name and entity_schema_map is not available)
    """
    # ── Path 1: base_entity → entity graph schema (preferred) ─────────────────
    if entity_schema_map:
        base_entity = str(metric.get("base_entity") or "").strip()
        if base_entity:
            schema = entity_schema_map.get(base_entity, "")
            if schema:
                return {schema.upper()}

    # ── Path 2 & 3: infer from table references / column matching ─────────────
    return {
        schema
        for schema in (_table_schema(t) for t in metric_source_tables(metric, table_columns))
        if schema
    }


def _schemas_from_graph(graph_context: dict[str, Any] | None, graph: dict[str, Any] | None) -> set[str]:
    if not graph_context or not graph_context.get("enabled") or not graph:
        return set()
    detected = {str(e) for e in graph_context.get("detected") or []}
    if graph_context.get("anchor"):
        detected.add(str(graph_context.get("anchor")))
    schemas: set[str] = set()
    for entity in graph.get("entities") or []:
        if entity.get("entity_name") not in detected:
            continue
        schema = str(entity.get("schema_name") or "").upper().strip()
        if schema:
            schemas.add(schema)
    return schemas


def _schemas_from_semantic_plan(semantic_plan: dict[str, Any] | None) -> set[str]:
    schemas: set[str] = set()
    for field in (semantic_plan or {}).get("fields") or []:
        schema = _table_schema(str(field.get("table") or ""))
        if schema:
            schemas.add(schema)
    return schemas


def _is_generic_metric_question(question: str) -> bool:
    q_tokens = _tokens(question)
    return bool(q_tokens) and q_tokens <= {"revenue", "sales", "amount", "charge"}


def resolve_metric_scope(
    metrics: list[dict[str, Any]],
    question: str,
    table_columns: dict[str, dict[str, str]] | None,
    *,
    selected_schema: str = "",
    graph_context: dict[str, Any] | None = None,
    graph: dict[str, Any] | None = None,
    semantic_plan: dict[str, Any] | None = None,
    entity_schema_map: dict[str, str] | None = None,
    limit: int = 6,
) -> MetricScopeResult:
    """Return the metrics that should be visible/enforced for this question.

    Parameters
    ----------
    entity_schema_map : Optional dict mapping entity_name → schema_name (UPPER).
        Built from the full entity graph.  When provided, ``base_entity`` on each
        metric is used as the primary schema lookup — bypassing fragile
        ``base_table`` bare-name parsing and ``_schema.json`` column matching.
        Pass ``{e["entity_name"]: e["schema_name"].upper() for e in graph entities}``.
    """
    selected_schema = (selected_schema or "").upper().strip()
    context_schemas = {selected_schema} if selected_schema else set()
    context_schemas.update(_schemas_from_graph(graph_context, graph))
    context_schemas.update(_schemas_from_semantic_plan(semantic_plan))

    scored: list[tuple[int, dict[str, Any], set[str]]] = []
    for metric in metrics or []:
        score = _phrase_score(metric, question)
        if score <= 0:
            continue
        schemas = metric_source_schemas(metric, table_columns, entity_schema_map)
        if context_schemas and schemas and not (schemas & context_schemas):
            continue
        if context_schemas and schemas:
            score += 80
        if schemas:
            metric = dict(metric)
            metric["_source_schemas"] = ",".join(sorted(schemas))
        scored.append((score, metric, schemas))

    scored.sort(key=lambda item: (-item[0], item[1].get("name", "")))
    if not scored:
        return MetricScopeResult(metrics=[], context_schemas=context_schemas)

    schema_sets = [schemas for _, _, schemas in scored if schemas]
    has_cross_schema_choices = len({s for schemas in schema_sets for s in schemas}) > 1
    if not context_schemas and has_cross_schema_choices and _is_generic_metric_question(question):
        return MetricScopeResult(
            metrics=[],
            ambiguous=True,
            options=[m.get("name", "") for _, m, _ in scored[:4] if m.get("name")],
            reason="Multiple revenue-like metrics exist across schemas.",
            context_schemas=context_schemas,
        )

    # Keep only the best metric when it is clearly ahead or schema context exists.
    # The window of 8 points (one token overlap = 10pts, exact-phrase bonus ≥100pts)
    # is intentionally tight: it admits a close second candidate so the LLM can
    # pick between them, but does NOT admit a clearly weaker metric.
    best_score = scored[0][0]
    _TIE_WINDOW = 8
    chosen = [m for score, m, _ in scored if score >= best_score - _TIE_WINDOW]
    return MetricScopeResult(
        metrics=chosen[: max(1, int(limit or 6))],
        context_schemas=context_schemas,
    )
