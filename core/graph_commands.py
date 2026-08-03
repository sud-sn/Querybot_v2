"""Metadata-only conversational parsing for Entity Graph edits.

The LLM may see ONLY table and column names (the same schema metadata
already exposed read-only via admin/routes.py's graph_api_schema_tables/
graph_api_columns) -- there is no row data in scope for this feature at
all. The model returns a small JSON AST which is validated against the
real schema manifest and compiled into a GraphEditCommand; a table or
column name it names is trusted only when it matches the manifest exactly.

Resolving a validated table reference into an existing (or newly
suggested) entity_graph row is the caller's job (admin/routes.py), not
this module's -- this module only ever talks about tables/columns, never
entities, keeping the parse step pure and side-effect free.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_ALLOWED_ACTIONS = {"register_entity", "create_join", "update_join"}
_ALLOWED_JOIN_TYPES = {"INNER", "LEFT", "RIGHT", "FULL"}
_ALLOWED_REL_TYPES = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
_PLAN_KEYS = {
    "action", "table_name", "schema_name", "entity_name",
    "from_table", "to_table", "from_column", "to_column",
    "join_type", "relationship_type", "where_clause", "confidence",
}
_MIN_CONFIDENCE = 0.0
_MAX_CONFIDENCE = 1.0
# Fail closed: a response missing/malformed confidence is not evidence of
# high confidence -- mirrors core/result_planner.py's same discipline so a
# non-compliant response never silently reads as "certain".
_DEFAULT_CONFIDENCE = _MIN_CONFIDENCE


@dataclass(frozen=True)
class GraphEditCommand:
    action: str
    table_name: str = ""
    schema_name: str = ""
    entity_name: str = ""
    from_table: str = ""
    to_table: str = ""
    from_column: str = ""
    to_column: str = ""
    join_type: str = "LEFT"
    relationship_type: str = "many_to_one"
    where_clause: str = ""
    confidence: float = _DEFAULT_CONFIDENCE


@dataclass(frozen=True)
class GraphCommandInput:
    system_prompt: str
    user_prompt: str


def parse_explicit_graph_commands(
    text: str,
    schema_manifest: dict[str, list[str]],
) -> tuple[list[GraphEditCommand], str]:
    """Parse schema-validated, paste-friendly graph mappings without an LLM.

    Supports multiple lines such as ``F_ORDER is a fact`` and
    ``F_ORDER.CUSTOMER_ID -> D_CUSTOMER.CUSTOMER_ID``. Unqualified table
    names are accepted only when unique in the discovered schema.
    """
    raw = str(text or "").strip()
    if not raw:
        return [], ""
    table_by_norm = {_normalise(table): table for table in schema_manifest}
    bare_tables: dict[str, list[str]] = {}
    for table in schema_manifest:
        bare_tables.setdefault(_normalise(table.split(".")[-1]), []).append(table)

    def resolve_table(value: str) -> str:
        cleaned = str(value or "").strip().strip("[]`\"")
        exact = table_by_norm.get(_normalise(cleaned), "")
        if exact:
            return exact
        # Discovered manifests can be catalog.schema.table while an admin
        # naturally pastes schema.table. Resolve any unambiguous trailing
        # qualification, just as the SQL validator and ACL layer do.
        wanted_parts = [
            part.strip().strip("[]`\"").casefold()
            for part in cleaned.split(".") if part.strip()
        ]
        qualified_matches = []
        if len(wanted_parts) >= 2:
            for table in schema_manifest:
                table_parts = [
                    part.strip().strip("[]`\"").casefold()
                    for part in str(table).split(".") if part.strip()
                ]
                if len(table_parts) >= len(wanted_parts) and table_parts[-len(wanted_parts):] == wanted_parts:
                    qualified_matches.append(table)
        if len(qualified_matches) == 1:
            return qualified_matches[0]
        matches = bare_tables.get(_normalise(cleaned), [])
        return matches[0] if len(matches) == 1 else ""

    def split_field(value: str) -> tuple[str, str]:
        parts = [part.strip().strip("[]`\"") for part in value.strip().split(".")]
        if len(parts) < 2:
            return "", ""
        table = resolve_table(".".join(parts[:-1])) or resolve_table(parts[-2])
        columns = {_normalise(col): col for col in schema_manifest.get(table, [])}
        return table, columns.get(_normalise(parts[-1]), "")

    commands: list[GraphEditCommand] = []
    explicit_seen = False
    chunks = [part.strip(" \t-*\u2022") for part in re.split(r"[\r\n;]+", raw) if part.strip()]
    for chunk in chunks:
        join_match = re.search(
            r"([\[\]`\"A-Za-z0-9_.]+)\s*(?:->|=>|=|\bjoins?\s+(?:to\s+)?|\bmaps?\s+(?:to\s+)?)\s*"
            r"([\[\]`\"A-Za-z0-9_.]+)", chunk, flags=re.I,
        )
        if join_match and "." in join_match.group(1) and "." in join_match.group(2):
            explicit_seen = True
            from_table, from_column = split_field(join_match.group(1))
            to_table, to_column = split_field(join_match.group(2))
            if not all((from_table, from_column, to_table, to_column)):
                return [], f'Could not match this mapping to the discovered schema: "{chunk}".'
            join_type_match = re.search(r"\b(inner|left|right|full)\b", chunk, re.I)
            rel_match = re.search(
                r"\b(one[_ -]to[_ -]one|one[_ -]to[_ -]many|many[_ -]to[_ -]one|many[_ -]to[_ -]many)\b",
                chunk, re.I,
            )
            commands.append(GraphEditCommand(
                action="create_join", from_table=from_table, to_table=to_table,
                from_column=from_column, to_column=to_column,
                join_type=join_type_match.group(1).upper() if join_type_match else "LEFT",
                relationship_type=(re.sub(r"[ -]", "_", rel_match.group(1).lower()) if rel_match else "many_to_one"),
                confidence=1.0,
            ))
            continue

        type_match = re.search(
            r"(?:change\s+)?([\[\]`\"A-Za-z0-9_.]+)\s+(?:from\s+\w+\s+)?(?:is|as|to|=)\s+(?:an?\s+)?"
            r"(fact|dimension|bridge|date(?:[_ -]?role)?)\b",
            chunk, flags=re.I,
        )
        if type_match:
            explicit_seen = True
            table = resolve_table(type_match.group(1))
            if not table:
                return [], f'Could not find table "{type_match.group(1)}" in the discovered schema.'
            entity_type = type_match.group(2).lower().replace(" ", "_").replace("-", "_")
            commands.append(GraphEditCommand(
                action="register_entity", table_name=table,
                entity_name=table.split(".")[-1],
                where_clause=f"entity_type:{entity_type}", confidence=1.0,
            ))

    return (commands, "") if commands else ([], "" if not explicit_seen else "No valid graph mappings were found.")


def build_graph_command_input(
    text: str,
    schema_manifest: dict[str, list[str]],
) -> GraphCommandInput:
    """schema_manifest: {table_fqn: [column_name, ...]} -- table/column
    names only, sourced from the same _schema.json the read-only
    graph_api_schema_tables/graph_api_columns routes already expose."""
    manifest_lines = [
        f"{table}: {', '.join(columns)}"
        for table, columns in sorted(schema_manifest.items())
    ]
    manifest_text = "\n".join(manifest_lines)[:6000]

    system_prompt = (
        "You translate a plain-language description of a table mapping or a join "
        "into exactly one JSON object describing a graph edit. Return JSON only, "
        "no markdown, no explanation. You receive ONLY table and column names -- "
        "never row data. Use exact table and column names from the schema manifest "
        "below; never invent one.\n\n"
        "Allowed action values: register_entity, create_join, update_join.\n"
        "Allowed keys: action, table_name, schema_name, entity_name, from_table, "
        "to_table, from_column, to_column, join_type, relationship_type, "
        "where_clause, confidence.\n"
        "register_entity requires table_name (optionally schema_name, "
        "entity_name). create_join/update_join require from_table, to_table, "
        "from_column, to_column. join_type must be INNER, LEFT, RIGHT, or FULL "
        "(default LEFT). relationship_type must be one_to_one, one_to_many, "
        "many_to_one, or many_to_many (default many_to_one). "
        "If the request cannot be expressed exactly against the schema below, "
        "return {\"action\":\"unsupported\"}.\n\n"
        "Always include \"confidence\": a number from 0.0 to 1.0 rating how "
        "certain you are this mapping is what the user meant, given only the "
        "schema manifest and their text. Use a high value (0.9-1.0) only when "
        "every table/column reference is unambiguous.\n\n"
        f"Schema manifest (table: columns):\n{manifest_text}"
    )
    user_prompt = json.dumps(
        {"request": str(text or "").strip()},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return GraphCommandInput(system_prompt=system_prompt, user_prompt=user_prompt)


async def parse_graph_command(
    text: str,
    schema_manifest: dict[str, list[str]],
    complete: Callable[..., Awaitable[tuple[str, int, int]]],
) -> tuple[GraphEditCommand | None, str]:
    """Ask a model for a constrained AST, then compile+validate it locally."""
    command_input = build_graph_command_input(text, schema_manifest)
    try:
        raw, _, _ = await complete(
            system=command_input.system_prompt,
            user=command_input.user_prompt,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        return None, "The graph command parser was unavailable."
    return compile_graph_command_response(raw, schema_manifest)


def compile_graph_command_response(
    raw_response: str,
    schema_manifest: dict[str, list[str]],
) -> tuple[GraphEditCommand | None, str]:
    """Validate a model JSON plan against the real schema manifest and
    compile it into a local command. Never trusts a table/column name the
    model returns unless it matches the manifest exactly."""
    plan = _parse_json_object(raw_response)
    if plan is None:
        return None, "The graph command parser did not return valid JSON."
    if set(plan) - _PLAN_KEYS:
        return None, "The graph command parser returned unsupported fields."

    action = str(plan.get("action") or "").strip().lower()
    if action == "unsupported":
        return None, "That mapping could not be matched to the current schema."
    if action not in _ALLOWED_ACTIONS:
        return None, "The graph command parser returned an unsupported action."

    table_lookup = {_normalise(t): t for t in schema_manifest}
    column_lookup = {
        _normalise(t): {_normalise(c): c for c in cols}
        for t, cols in schema_manifest.items()
    }
    confidence = _extract_confidence(plan)

    def resolve_table(key: str) -> str:
        value = str(plan.get(key) or "").strip()
        return table_lookup.get(_normalise(value), "")

    def resolve_column(table: str, key: str) -> str:
        value = str(plan.get(key) or "").strip()
        cols = column_lookup.get(_normalise(table), {})
        return cols.get(_normalise(value), "")

    if action == "register_entity":
        table = resolve_table("table_name")
        if not table:
            return None, "The table named was not found in the current schema."
        entity_name = str(plan.get("entity_name") or "").strip() or table.split(".")[-1]
        schema_name = str(plan.get("schema_name") or "").strip()
        return GraphEditCommand(
            action="register_entity",
            table_name=table,
            schema_name=schema_name,
            entity_name=entity_name,
            confidence=confidence,
        ), ""

    # create_join / update_join
    from_table = resolve_table("from_table")
    to_table = resolve_table("to_table")
    if not from_table or not to_table:
        return None, "One or both tables in the join were not found in the current schema."
    from_column = resolve_column(from_table, "from_column")
    to_column = resolve_column(to_table, "to_column")
    if not from_column or not to_column:
        return None, "One or both join columns were not found on their tables."
    join_type = str(plan.get("join_type") or "LEFT").strip().upper()
    if join_type not in _ALLOWED_JOIN_TYPES:
        join_type = "LEFT"
    relationship_type = str(plan.get("relationship_type") or "many_to_one").strip().lower()
    if relationship_type not in _ALLOWED_REL_TYPES:
        relationship_type = "many_to_one"
    where_clause = str(plan.get("where_clause") or "").strip()
    return GraphEditCommand(
        action=action,
        from_table=from_table,
        to_table=to_table,
        from_column=from_column,
        to_column=to_column,
        join_type=join_type,
        relationship_type=relationship_type,
        where_clause=where_clause,
        confidence=confidence,
    ), ""


def _extract_confidence(plan: dict) -> float:
    raw_value = plan.get("confidence")
    if raw_value is None:
        return _DEFAULT_CONFIDENCE
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return _DEFAULT_CONFIDENCE
    if value != value:  # NaN guard
        return _DEFAULT_CONFIDENCE
    return max(_MIN_CONFIDENCE, min(_MAX_CONFIDENCE, value))


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
