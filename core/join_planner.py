"""Production join planning contracts for governed analytical SQL.

This module does not generate SQL.  It compiles a deterministic, auditable
plan that the SQL generator and validator must follow.  In particular, it
never authorizes a raw fact-to-fact join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable


def _bare(value: str) -> str:
    return str(value or "").strip().strip("[]\"`").split(".")[-1].upper()


def _role(entity: dict[str, Any]) -> str:
    role = str(entity.get("entity_type") or "dimension").strip().lower()
    return role if role in {"fact", "dimension", "bridge"} else "dimension"


def _table_matches(entity: dict[str, Any], table: str) -> bool:
    return _bare(entity.get("table_name", "")) == _bare(table)


@dataclass
class JoinPlan:
    status: str
    anchor_fact: str = ""
    output_grain: str = "question_requested_grain"
    fact_entities: list[str] = field(default_factory=list)
    dimension_entities: list[str] = field(default_factory=list)
    bridge_entities: list[str] = field(default_factory=list)
    required_edge_ids: list[Any] = field(default_factory=list)
    isolated_fact_plans: list[dict[str, Any]] = field(default_factory=list)
    common_dimensions: list[str] = field(default_factory=list)
    bridge_allocations: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "status": self.status,
            "anchor_fact": self.anchor_fact,
            "output_grain": self.output_grain,
            "fact_entities": self.fact_entities,
            "dimension_entities": self.dimension_entities,
            "bridge_entities": self.bridge_entities,
            "required_edge_ids": self.required_edge_ids,
            "isolated_fact_plans": self.isolated_fact_plans,
            "common_dimensions": self.common_dimensions,
            "bridge_allocations": self.bridge_allocations,
            "warnings": self.warnings,
            "reason": self.reason,
            "raw_fact_to_fact_allowed": False,
        }


def choose_fact_anchor(
    entities: Iterable[dict[str, Any]],
    detected: Iterable[str],
    *,
    metric_formula_tables: Iterable[str] = (),
    authoritative_fact_tables: Iterable[str] = (),
) -> str:
    """Choose the fact anchor from governed evidence, never list order alone."""
    entity_map = {str(entity.get("entity_name") or ""): entity for entity in entities}
    detected_names = [name for name in detected if name in entity_map]
    facts = [name for name in detected_names if _role(entity_map[name]) == "fact"]
    if not facts:
        return detected_names[0] if detected_names else ""

    metric_tables = {_bare(value) for value in metric_formula_tables or ()}
    authoritative = {_bare(value) for value in authoritative_fact_tables or ()}
    ranked: list[tuple[int, int, str]] = []
    for index, name in enumerate(facts):
        entity = entity_map[name]
        score = 0
        if any(_table_matches(entity, table) for table in metric_tables):
            score += 1000
        if any(_table_matches(entity, table) for table in authoritative):
            score += 500
        if str(entity.get("status") or "confirmed").lower() == "confirmed":
            score += 100
        if str(entity.get("generated_by") or "").lower() == "manual":
            score += 40
        try:
            score += int(entity.get("confidence_score") or 0)
        except (TypeError, ValueError):
            pass
        ranked.append((score, -index, name))
    ranked.sort(reverse=True)
    return ranked[0][2]


def relationship_is_admissible(
    relationship: dict[str, Any],
    entities: dict[str, dict[str, Any]],
) -> tuple[bool, str]:
    """Validate the semantic orientation of an edge.

    Relationships are stored FK-owner -> referenced-table.  Facts may point
    to dimensions; dimensions may point to parent dimensions (snowflake);
    bridges may point to their endpoints.  Direct fact-to-fact edges and
    unmodelled many-to-many edges are fail-closed.
    """
    source = entities.get(str(relationship.get("from_entity") or ""), {})
    target = entities.get(str(relationship.get("to_entity") or ""), {})
    source_role, target_role = _role(source), _role(target)
    rel_type = str(relationship.get("relationship_type") or "many_to_one").lower()
    validation = str(relationship.get("validation_status") or "untested").lower()
    status = str(relationship.get("status") or "confirmed").lower()

    if validation in {"broken", "zero_match"}:
        return False, "relationship validation is broken"
    if status == "rejected":
        return False, "relationship was rejected"
    if source_role == "fact" and target_role == "fact":
        return False, "raw fact-to-fact joins are prohibited"
    if rel_type == "many_to_many" and source_role != "bridge" and target_role != "bridge":
        return False, "many-to-many relationship requires an explicit bridge"
    if source_role == "dimension" and target_role == "fact":
        return False, "dimension-to-fact FK direction is not a star/snowflake edge"
    if source_role in {"fact", "dimension", "bridge"} and target_role in {"dimension", "bridge"}:
        return True, "admissible star/snowflake edge"
    return False, f"unsupported {source_role}-to-{target_role} relationship"


def traversal_is_admissible(
    relationship: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    *,
    direction: str,
    anti_join: bool = False,
) -> tuple[bool, str]:
    """Validate one traversal direction from a selected analytical anchor."""
    source_role = _role(entities.get(str(relationship.get("from_entity") or ""), {}))
    target_role = _role(entities.get(str(relationship.get("to_entity") or ""), {}))
    if anti_join and source_role == "fact" and target_role == "fact":
        return True, "governed anti-join existence path between business events"
    allowed, reason = relationship_is_admissible(relationship, entities)
    if not allowed:
        return allowed, reason
    if direction == "forward":
        return True, "FK owner to referenced member"

    if source_role == "fact":
        if anti_join and target_role == "dimension":
            return True, "anti-join retains dimension members and tests for missing facts"
        return False, "backward traversal from a dimension into raw fact rows is prohibited"
    if source_role == "bridge":
        return True, "explicit bridge traversal"
    if source_role == "dimension" and target_role == "dimension":
        return False, "backward snowflake traversal can multiply parent members"
    return False, "unsupported backward traversal"


def compile_join_plan(
    graph: dict[str, Any],
    detected: Iterable[str],
    join_path: Iterable[dict[str, Any]],
    *,
    metric_formula_tables: Iterable[str] = (),
    authoritative_fact_tables: Iterable[str] = (),
    anti_join: bool = False,
) -> dict[str, Any]:
    """Compile a single-fact or isolated multi-fact execution contract."""
    entities_list = list(graph.get("entities") or [])
    entity_map = {str(entity.get("entity_name") or ""): entity for entity in entities_list}
    detected_names = [str(name) for name in detected if str(name) in entity_map]
    facts = [name for name in detected_names if _role(entity_map[name]) == "fact"]
    dimensions = [name for name in detected_names if _role(entity_map[name]) == "dimension"]
    bridges = [name for name in detected_names if _role(entity_map[name]) == "bridge"]
    bridge_allocations: list[dict[str, str]] = []
    for bridge_name in bridges:
        bridge_entity = entity_map.get(bridge_name, {})
        physical_table = str(bridge_entity.get("table_name") or "").strip()
        schema_name = str(
            bridge_entity.get("schema_name") or bridge_entity.get("schema") or ""
        ).strip()
        if physical_table and "." not in physical_table and schema_name:
            physical_table = f"{schema_name}.{physical_table}"
        for prop in graph.get("properties") or []:
            if str(prop.get("entity_name") or "") != bridge_name:
                continue
            if str(prop.get("status") or "confirmed").casefold() != "confirmed":
                continue
            evidence = " ".join(
                str(prop.get(key) or "")
                for key in ("column_name", "display_name", "business_meaning", "synonyms")
            ).casefold()
            # Do not accept a generic physical "weight" field (for example
            # product net weight) as an allocation. Allocation weight is still
            # detected by its allocation/share wording.
            if not re.search(r"\b(?:allocat\w*|share|ratio|pct|percent\w*|proportion)\b", evidence):
                continue
            column = str(prop.get("column_name") or "").strip()
            if column:
                bridge_allocations.append({
                    "entity": bridge_name,
                    "table": physical_table,
                    "column": column,
                })
    anchor = choose_fact_anchor(
        entities_list,
        detected_names,
        metric_formula_tables=metric_formula_tables,
        authoritative_fact_tables=authoritative_fact_tables,
    )

    path = list(join_path or [])
    invalid_edges: list[str] = []
    edge_ids: list[Any] = []
    for edge in path:
        direction = str(edge.get("_direction") or "forward")
        anti_fact_edge = (
            anti_join
            and str(edge.get("from_entity") or "") in facts
            and str(edge.get("to_entity") or "") in facts
        )
        isolated_secondary_fact = (
            len(facts) > 1
            and direction == "backward"
            and str(edge.get("from_entity") or "") in facts
        )
        if anti_fact_edge:
            allowed, reason = True, "governed anti-join fact existence edge"
        elif isolated_secondary_fact:
            allowed, reason = True, "secondary fact is isolated before combination"
        else:
            allowed, reason = traversal_is_admissible(
                edge,
                entity_map,
                direction=direction,
            )
        if not allowed:
            invalid_edges.append(
                f"{edge.get('from_entity')} -> {edge.get('to_entity')}: {reason}"
            )
        identity = edge.get("id") or edge.get("relationship_key")
        if identity is not None:
            edge_ids.append(identity)

    if invalid_edges:
        return JoinPlan(
            status="blocked",
            anchor_fact=anchor,
            fact_entities=facts,
            dimension_entities=dimensions,
            bridge_entities=bridges,
            bridge_allocations=bridge_allocations,
            required_edge_ids=edge_ids,
            warnings=invalid_edges,
            reason="One or more graph edges violate the governed star/snowflake contract.",
        ).to_dict()

    if len(facts) > 1 and not anti_join:
        touched_by_fact: dict[str, set[str]] = {fact: set() for fact in facts}
        for edge in path:
            source, target = str(edge.get("from_entity") or ""), str(edge.get("to_entity") or "")
            if source in touched_by_fact and target in dimensions:
                touched_by_fact[source].add(target)
            if target in touched_by_fact and source in dimensions:
                touched_by_fact[target].add(source)
        sets = [values for values in touched_by_fact.values() if values]
        common = sorted(set.intersection(*sets)) if len(sets) == len(facts) and sets else []
        isolated = [
            {
                "fact_entity": fact,
                "aggregate_before_join": True,
                "group_by": common or dimensions,
                "physical_fact_must_appear_in_own_cte": True,
            }
            for fact in facts
        ]
        return JoinPlan(
            status="requires_isolated_aggregation",
            anchor_fact=anchor,
            fact_entities=facts,
            dimension_entities=dimensions,
            bridge_entities=bridges,
            bridge_allocations=bridge_allocations,
            required_edge_ids=edge_ids,
            isolated_fact_plans=isolated,
            common_dimensions=common,
            warnings=[
                "Aggregate each fact independently to the common output grain; join only the aggregated result sets.",
                "Never join physical fact rows to each other, directly or through a shared dimension.",
            ],
            reason="Multiple facts were requested and require conformed-dimension isolation.",
        ).to_dict()

    if anti_join and len(facts) > 1:
        warnings = [
            "This fact-to-fact path is authorized only for missing-record existence logic; do not aggregate measures across both facts.",
        ]
    elif bridges:
        warnings = ([
            "Bridge traversal can multiply rows; every additive measure must use the governed allocation column.",
        ] if bridge_allocations else [
            "Bridge traversal has no governed allocation column; additive measures are blocked until one is mapped. Distinct-key counts remain allowed.",
        ])
    else:
        warnings = []

    return JoinPlan(
        status="selected" if anchor else "clarification_required",
        anchor_fact=anchor,
        fact_entities=facts,
        dimension_entities=dimensions,
        bridge_entities=bridges,
        bridge_allocations=bridge_allocations,
        required_edge_ids=edge_ids,
        warnings=warnings,
        reason=(
            "A deterministic fact anchor and governed star/snowflake path were selected."
            if anchor else "No unambiguous analytical anchor could be selected."
        ),
    ).to_dict()
