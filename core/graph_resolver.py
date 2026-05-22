"""
core/graph_resolver.py

Structured entity graph resolver — the deterministic JOIN engine.

The graph resolver takes the user's question, identifies which business
entities are referenced (Customer, Drug, Prescription…), finds the shortest
JOIN path through the entity relationship graph, and returns a pre-built
SQL JOIN skeleton to inject into the LLM prompt.

The LLM then only needs to:
  1. Write the SELECT clause (metrics + dimensions)
  2. Add WHERE / HAVING / ORDER BY / GROUP BY

It no longer needs to guess table names, FK columns, or JOIN directions —
the graph resolver provides those deterministically.

Integration points
──────────────────
  main.py       → calls resolve_for_question() before build_sql_system_prompt()
  core/llm.py   → build_sql_system_prompt() accepts graph_context param
  admin/routes.py → CRUD routes use store CRUD functions
"""

from __future__ import annotations

import re
import logging
from collections import deque
from typing import Optional

log = logging.getLogger("querybot.graph_resolver")


# ══════════════════════════════════════════════════════════════════════════════
# Table quoting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _quote_table(table: str, schema: str, db_type: str) -> str:
    """Return properly quoted [schema].[table] / "schema"."table" reference."""
    if db_type == "azure_sql":
        if schema:
            return f"[{schema}].[{table}]"
        return f"[{table}]"
    elif db_type == "oracle":
        if schema:
            return f'"{schema.upper()}"."{table.upper()}"'
        return f'"{table.upper()}"'
    else:  # snowflake
        if schema:
            return f'"{schema}"."{table}"'
        return f'"{table}"'


def _quote_col(col: str, db_type: str) -> str:
    if db_type == "azure_sql":
        return f"[{col}]"
    return f'"{col}"'


# ══════════════════════════════════════════════════════════════════════════════
# Entity detection
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def detect_entities(question: str, graph: dict) -> list[str]:
    """
    Score each entity against the question and return those that match.

    Scoring (no LLM call required):
      entity_name exact word match     → +10
      display_name word match          → +8
      table_name substring match       → +5
      column synonym/display_name match → +3

    Returns entity names sorted by score descending, threshold ≥ 3.
    """
    q = _normalize(question)
    scores: dict[str, int] = {}

    entities   = graph.get("entities", [])

    for ent in entities:
        name  = ent["entity_name"]
        score = 0

        # entity name — check each word of the name appears in question
        norm_name = _normalize(name)
        for word in norm_name.split():
            if len(word) >= 3 and re.search(r"\b" + re.escape(word) + r"\b", q):
                score += 10

        # display name
        disp = _normalize(ent.get("display_name") or name)
        for word in disp.split():
            if len(word) >= 3 and re.search(r"\b" + re.escape(word) + r"\b", q):
                score += 8

        # table name substring
        tbl = _normalize(ent.get("table_name", ""))
        for word in tbl.split("_"):
            if len(word) >= 4 and word in q:
                score += 5

        if score >= 3:
            scores[name] = score

    # Always include at least one fact entity if none scored
    if not scores:
        for ent in entities:
            if ent.get("entity_type") == "fact":
                scores[ent["entity_name"]] = 1
                break

    return sorted(scores, key=lambda n: scores[n], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# BFS pathfinder
# ══════════════════════════════════════════════════════════════════════════════

def _build_adjacency(relationships: list[dict]) -> dict[str, list[dict]]:
    """Build undirected adjacency list from relationship edges."""
    adj: dict[str, list[dict]] = {}
    for rel in relationships:
        f, t = rel["from_entity"], rel["to_entity"]
        adj.setdefault(f, []).append({**rel, "_direction": "forward"})
        adj.setdefault(t, []).append({**rel, "_direction": "backward"})
    return adj


def find_join_path(entity_names: list[str], graph: dict) -> list[dict]:
    """
    BFS from the anchor entity (fact table preferred) through relationship
    edges to reach all other entities. Returns an ordered list of JOIN steps:

      [{"from_entity", "to_entity", "from_column", "to_column",
        "join_type", "relationship_type", "_direction"}]

    Empty list if the graph has no entities / relationships.
    """
    if not entity_names or len(entity_names) < 2:
        return []

    entities_map = {e["entity_name"]: e for e in graph.get("entities", [])}
    rels = graph.get("relationships", [])
    adj  = _build_adjacency(rels)

    # Prefer starting from a fact entity
    anchor = entity_names[0]
    for name in entity_names:
        if entities_map.get(name, {}).get("entity_type") == "fact":
            anchor = name
            break

    targets = [n for n in entity_names if n != anchor]
    visited_edges: list[dict] = []
    visited_nodes = {anchor}

    # BFS to reach each target
    for target in targets:
        if target in visited_nodes:
            continue
        # BFS from any visited node to target
        found = False
        for start_node in list(visited_nodes):
            queue: deque = deque([(start_node, [])])
            seen: set = {start_node}
            while queue:
                node, path = queue.popleft()
                if node == target:
                    for step in path:
                        if step not in visited_edges:
                            visited_edges.append(step)
                    visited_nodes.update(n for step in path
                                         for n in [step["from_entity"], step["to_entity"]])
                    found = True
                    break
                for edge in adj.get(node, []):
                    neighbour = (edge["to_entity"] if edge["_direction"] == "forward"
                                 else edge["from_entity"])
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append((neighbour, path + [edge]))
            if found:
                break

        if not found:
            log.debug("Graph resolver: no path from %s to %s", visited_nodes, target)

    return visited_edges


# ══════════════════════════════════════════════════════════════════════════════
# JOIN skeleton builder
# ══════════════════════════════════════════════════════════════════════════════

def build_join_skeleton(
    join_path: list[dict],
    entities_map: dict[str, dict],
    anchor_entity: str,
    db_type: str,
) -> str:
    """
    Convert an ordered list of JOIN steps into a SQL FROM + JOIN clause.

    Returns something like:
      FROM [dbo].[FACT_RXFILL] f
      INNER JOIN [dbo].[DIM_CUSTOMER] c ON f.[CustomerID] = c.[CustomerID]
      LEFT  JOIN [dbo].[DIM_DRUG]     d ON f.[DrugCode]   = d.[DrugCode]
    """
    if not join_path:
        return ""

    # Assign short aliases
    aliases: dict[str, str] = {}
    used: set[str] = set()

    def _alias(entity_name: str) -> str:
        if entity_name in aliases:
            return aliases[entity_name]
        base = re.sub(r"[^a-z]", "", entity_name.lower())[:3] or "t"
        alias = base
        i = 2
        while alias in used:
            alias = base + str(i)
            i += 1
        used.add(alias)
        aliases[entity_name] = alias
        return alias

    anchor_ent  = entities_map.get(anchor_entity, {})
    _anchor_tbl_name = (anchor_ent.get("table_name") or "").strip()
    if not _anchor_tbl_name:
        log.warning(
            "Graph resolver: anchor entity %r has no table_name configured — "
            "JOIN skeleton disabled. Fix the entity's table_name in the Entity Graph admin.",
            anchor_entity,
        )
        return ""
    anchor_tbl  = _quote_table(
        _anchor_tbl_name,
        anchor_ent.get("schema_name", ""),
        db_type,
    )
    anchor_alias = _alias(anchor_entity)
    lines = [f"FROM {anchor_tbl} {anchor_alias}"]

    seen_nodes = {anchor_entity}
    for step in join_path:
        fwd = step["_direction"] == "forward"
        from_ent = step["from_entity"]
        to_ent   = step["to_entity"]
        from_col = step["from_column"]
        to_col   = step["to_column"]
        jtype    = step.get("join_type", "INNER").upper()

        # Determine which end is new
        new_ent  = to_ent if from_ent in seen_nodes else from_ent
        old_ent  = from_ent if from_ent in seen_nodes else to_ent

        if new_ent in seen_nodes:
            continue  # already joined

        new_meta = entities_map.get(new_ent, {})
        _new_tbl_name = (new_meta.get("table_name") or "").strip()
        if not _new_tbl_name:
            log.warning(
                "Graph resolver: entity %r has no table_name configured — "
                "skipping this JOIN. Fix the entity's table_name in the Entity Graph admin.",
                new_ent,
            )
            seen_nodes.add(new_ent)   # mark visited so we don't loop over it again
            continue
        new_tbl  = _quote_table(
            _new_tbl_name,
            new_meta.get("schema_name", ""),
            db_type,
        )
        new_alias = _alias(new_ent)
        old_alias = aliases.get(old_ent, old_ent[:3].lower())

        # Resolve ON columns depending on direction
        if from_ent == old_ent:
            # forward: old_ent.from_col = new_ent.to_col
            on_clause = (
                f"{old_alias}.{_quote_col(from_col, db_type)} = "
                f"{new_alias}.{_quote_col(to_col, db_type)}"
            )
        else:
            # backward: new_ent.from_col = old_ent.to_col
            on_clause = (
                f"{new_alias}.{_quote_col(from_col, db_type)} = "
                f"{old_alias}.{_quote_col(to_col, db_type)}"
            )

        # Append WHERE conditions stored on the relationship (uses entity.col notation)
        where_sql = (step.get("where_clause") or "").strip()
        if where_sql:
            # Replace entity name prefixes with SQL aliases (aliases built so far)
            for ent_name, a in aliases.items():
                where_sql = where_sql.replace(f"{ent_name}.", f"{a}.")
            on_clause += f" AND {where_sql}"

        lines.append(f"{jtype:5} JOIN {new_tbl} {new_alias} ON {on_clause}")
        seen_nodes.add(new_ent)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def resolve_for_question(
    question: str,
    account_id: str,
    db_type: str,
    graph: Optional[dict] = None,
) -> dict:
    """
    Main entry point called from main.py before SQL generation.

    Returns:
      {
        "enabled":        bool   — False if graph is empty or no entities detected
        "detected":       list   — entity names detected in the question
        "join_skeleton":  str    — SQL FROM + JOIN clause ready for injection
        "anchor":         str    — the root entity (usually the fact table)
        "entity_count":   int    — number of entities in the graph
      }
    """
    empty = {
        "enabled": False, "detected": [], "join_skeleton": "",
        "anchor": "", "entity_count": 0,
    }

    if graph is None:
        try:
            import store
            graph = store.get_full_graph(account_id)
        except Exception as exc:
            log.debug("Graph load failed: %s", exc)
            return empty

    entities   = graph.get("entities", [])
    rels       = graph.get("relationships", [])
    entity_count = len(entities)

    if not entities:
        return empty

    detected = detect_entities(question, graph)
    log.debug("Graph: question=%r detected=%s", question[:60], detected)

    if not detected:
        return {**empty, "entity_count": entity_count}

    if len(detected) < 2 or not rels:
        # Single entity — still useful: tells the LLM which table to anchor on
        ent = {e["entity_name"]: e for e in entities}.get(detected[0], {})
        _single_tbl_name = (ent.get("table_name") or "").strip()
        if not _single_tbl_name:
            log.warning(
                "Graph resolver: single entity %r has no table_name configured — "
                "graph injection disabled. Fix the entity's table_name in the Entity Graph admin.",
                detected[0],
            )
            return {**empty, "entity_count": entity_count}
        anchor_tbl = _quote_table(
            _single_tbl_name,
            ent.get("schema_name", ""),
            db_type,
        )
        alias = re.sub(r"[^a-z]", "", detected[0].lower())[:3] or "t"
        skeleton = f"FROM {anchor_tbl} {alias}"
        return {
            "enabled": True, "detected": detected,
            "join_skeleton": skeleton, "anchor": detected[0],
            "entity_count": entity_count,
        }

    entities_map = {e["entity_name"]: e for e in entities}
    join_path    = find_join_path(detected, graph)

    # Determine anchor (fact table if possible)
    anchor = detected[0]
    for name in detected:
        if entities_map.get(name, {}).get("entity_type") == "fact":
            anchor = name
            break

    skeleton = build_join_skeleton(join_path, entities_map, anchor, db_type)

    return {
        "enabled":       bool(skeleton),
        "detected":      detected,
        "join_skeleton": skeleton,
        "anchor":        anchor,
        "entity_count":  entity_count,
    }
