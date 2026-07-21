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

import heapq
import itertools
import json
import re
import logging
from typing import Optional

from core.date_roles import question_has_temporal_intent, relationship_matches_date_role

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


def _split_and_conditions(text: str) -> list[str]:
    """
    Split a WHERE/filter expression into its top-level AND conditions.

    A naive re.split(r"\\bAND\\b") corrupts BETWEEN x AND y (yielding a dangling
    "y" condition) and breaks apart parenthesised OR groups. Skip the AND that
    completes a BETWEEN, and any AND nested inside parentheses.
    """
    parts: list[str] = []
    depth = 0
    pending_between = 0
    start = 0
    for m in re.finditer(r"\(|\)|\bBETWEEN\b|\bAND\b", text or "", re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth = max(0, depth - 1)
        elif tok.upper() == "BETWEEN":
            if depth == 0:
                pending_between += 1
        else:  # AND
            if depth > 0:
                continue
            if pending_between:
                pending_between -= 1
                continue
            piece = text[start:m.start()].strip()
            if piece:
                parts.append(piece)
            start = m.end()
    tail = (text or "")[start:].strip()
    if tail:
        parts.append(tail)
    return parts


# ══════════════════════════════════════════════════════════════════════════════
# Entity detection
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in _normalize(text).split():
        if len(tok) < 3:
            continue
        out.add(tok)
        if tok.endswith("ies") and len(tok) > 4:
            out.add(tok[:-3] + "y")
        if tok.endswith("es") and len(tok) > 4:
            out.add(tok[:-2])
        if tok.endswith("s") and len(tok) > 3:
            out.add(tok[:-1])
    return out


def _phrase_in_question(phrase: str, q_tokens: set[str], q_norm: str) -> bool:
    phrase_norm = _normalize(phrase)
    if not phrase_norm:
        return False
    words = [w for w in phrase_norm.split() if len(w) >= 3]
    if not words:
        return False
    if len(words) == 1:
        word = words[0]
        return word in q_tokens or (word + "s") in q_tokens
    return all((w in q_tokens or (w + "s") in q_tokens) for w in words) or phrase_norm in q_norm


def _table_candidates_from_required(required_tables: list[str] | set[str] | None) -> set[str]:
    candidates: set[str] = set()
    for raw in required_tables or []:
        text = str(raw).strip().strip("[]\"`")
        if not text:
            continue
        parts = [p.strip("[]\"`").upper() for p in text.split(".") if p.strip("[]\"`")]
        for part in parts:
            candidates.add(part)
        if len(parts) >= 2:
            candidates.add(".".join(parts[-2:]))
    return candidates


def detect_entities(
    question: str,
    graph: dict,
    required_tables: list[str] | set[str] | None = None,
    required_entities: list[str] | set[str] | None = None,
) -> list[str]:
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
    q_tokens = _tokens(question)
    has_temporal_intent = question_has_temporal_intent(question)
    scores: dict[str, int] = {}

    entities   = graph.get("entities", [])
    required_table_candidates = _table_candidates_from_required(required_tables)
    required_entity_names = {str(e) for e in (required_entities or [])}

    props_by_entity: dict[str, list[dict]] = {}
    for prop in graph.get("properties", []) or []:
        props_by_entity.setdefault(prop.get("entity_name", ""), []).append(prop)

    relationship_role_scores: dict[str, int] = {}
    for rel in graph.get("relationships", []) or []:
        label = rel.get("label") or ""
        if not label:
            continue
        if _phrase_in_question(label, q_tokens, q) or relationship_matches_date_role(question, label):
            # Relationship labels represent the business role of the edge. For
            # role-playing dates, the to_entity is the specific date role
            # ("Invoice Date", "Delivery Date"), while from_entity is the fact.
            relationship_role_scores[rel.get("to_entity", "")] = (
                relationship_role_scores.get(rel.get("to_entity", ""), 0) + 35
            )
            relationship_role_scores[rel.get("from_entity", "")] = (
                relationship_role_scores.get(rel.get("from_entity", ""), 0) + 8
            )

    for ent in entities:
        name  = ent["entity_name"]
        score = relationship_role_scores.get(name, 0)
        strong_score = score   # relationship-label matches are a strong signal

        # entity name — check each word of the name appears in question
        if name in required_entity_names:
            score += 100
            strong_score += 100

        norm_name = _normalize(name)
        display_name = _normalize(ent.get("display_name") or name)
        is_date_role_entity = (
            "date" in norm_name.split()
            or "date" in display_name.split()
        )
        if is_date_role_entity and not has_temporal_intent and name not in required_entity_names:
            continue

        for word in norm_name.split():
            if len(word) >= 3 and _phrase_in_question(word, q_tokens, q):
                score += 10
                strong_score += 10

        # display name
        if _phrase_in_question(display_name, q_tokens, q):
            score += 8
            strong_score += 8

        # table name substring
        table_name = (ent.get("table_name") or "").upper()
        schema_name = (ent.get("schema_name") or "").upper()
        if table_name in required_table_candidates or f"{schema_name}.{table_name}" in required_table_candidates:
            score += 90
            strong_score += 90
        tbl = _normalize(table_name)
        for word in tbl.split():
            if len(word) >= 4 and _phrase_in_question(word, q_tokens, q):
                score += 5
                strong_score += 5

        # Column/property matches are the weakest signal available: a single
        # generic one-word synonym ("total", "amount", "cost", "date"…) can
        # coincidentally match almost any fact table's column list, which is
        # what pulls unrelated fact tables into a join and produces silent
        # fan-out (row-count multiplication) once the SQL is generated. A
        # multi-word phrase ("net revenue", "on hand quantity") is far more
        # specific and safe to treat as a real match on its own; a lone
        # single-word hit is not — it only counts once several independent
        # single-word hits corroborate each other.
        #
        # "Specific" must be measured the same way _phrase_in_question()
        # itself measures a match: by SIGNIFICANT (len >= 3) words, not by a
        # raw split. Every FK/PK identifier column normalizes to two raw
        # tokens ("PHARMACY_ID" -> "pharmacy id") but the trailing "id" is
        # too short to count as a real word — the match collapses to the
        # single bare word "pharmacy", exactly the generic-word case this
        # gate exists to block. Using the raw split count here let literally
        # every fact table sharing a dimension FK column match any question
        # naming that dimension, independent of actual relevance.
        weak_score = 0
        has_specific_property_hit = False
        for prop in props_by_entity.get(name, []):
            for field in ("column_name", "display_name", "synonyms", "description"):
                raw = prop.get(field) or ""
                phrases = [raw]
                if field == "synonyms":
                    phrases = [p.strip() for p in raw.split(",")]
                for phrase in phrases:
                    if not phrase or not _phrase_in_question(phrase, q_tokens, q):
                        continue
                    significant_words = [w for w in _normalize(phrase).split() if len(w) >= 3]
                    if len(significant_words) >= 2:
                        weak_score += 3
                        has_specific_property_hit = True
                    else:
                        weak_score += 1
        score += weak_score

        # A FACT table must earn its place via a real signal, not purely
        # coincidental overlap on a generic column-name word — otherwise one
        # common word pulls unrelated fact tables into the join skeleton and
        # the SQL generator fans out across them (multiplying row counts
        # before aggregation, inflating sums/counts by orders of magnitude
        # with no error). Dimension/bridge tables keep the looser bar since
        # joining an extra dimension doesn't multiply fact rows.
        entity_type = (ent.get("entity_type") or "").lower()
        if (
            entity_type == "fact"
            and name not in required_entity_names
            and strong_score < 3
            and not has_specific_property_hit
            and weak_score < 6
        ):
            continue

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
        if (rel.get("validation_status") or "").lower() == "broken":
            continue
        f, t = rel["from_entity"], rel["to_entity"]
        adj.setdefault(f, []).append({**rel, "_direction": "forward"})
        adj.setdefault(t, []).append({**rel, "_direction": "backward"})
    return adj


def _edge_weight(rel: dict) -> float:
    """Return semantic risk cost for one graph edge (lower is safer)."""
    provenance = (rel.get("generated_by") or "heuristic").lower()
    status = (rel.get("status") or "confirmed").lower()
    validation = (rel.get("validation_status") or "untested").lower()
    multiplicity = (rel.get("join_multiplicity") or "").lower()

    if validation == "broken" or multiplicity == "zero_match":
        return float("inf")

    if provenance == "manual" and status == "confirmed":
        cost = 0.5
    elif provenance == "db_fk" or bool(rel.get("source_enforced")):
        cost = 1.0
    elif provenance == "db_declared_fk":
        cost = 2.0
    elif provenance in {"date_role", "semantic"}:
        cost = 3.0
    elif provenance == "llm":
        cost = 9.0
    else:
        cost = 6.0

    if status == "suggested":
        cost += 1.5
    if validation == "valid":
        cost -= 0.5
    elif validation == "warning":
        cost += 4.0
    else:
        cost += 1.0

    confidence = max(0.0, min(100.0, float(rel.get("confidence_score") or 0)))
    cost += (100.0 - confidence) / 25.0
    if (rel.get("relationship_type") or "").lower() == "many_to_many":
        cost += 10.0

    try:
        fanout = float(rel.get("fanout_ratio") or -1)
    except (TypeError, ValueError):
        fanout = -1.0
    try:
        orphan = float(rel.get("orphan_rate") or -1)
    except (TypeError, ValueError):
        orphan = -1.0
    if fanout > 1.05:
        cost += min(8.0, (fanout - 1.0) * 2.0)
    if orphan > 20.0:
        cost += 4.0
    return max(0.1, cost)


def find_join_path(entity_names: list[str], graph: dict, prefer_fact_anchor: bool = True) -> list[dict]:
    """
    Weighted shortest path from the anchor entity (fact table preferred)
    through relationship edges to reach all other entities. Returns an ordered
    list of JOIN steps. Edge weights prefer governed/validated DB constraints
    and penalize untested heuristics, fanout, warnings, and many-to-many joins.

      [{"from_entity", "to_entity", "from_column", "to_column",
        "join_type", "relationship_type", "_direction"}]

    Empty list if the graph has no entities / relationships.
    """
    if not entity_names or len(entity_names) < 2:
        return []

    entities_map = {e["entity_name"]: e for e in graph.get("entities", [])}
    rels = graph.get("relationships", [])
    adj  = _build_adjacency(rels)

    # Prefer starting from a fact entity for normal analytics. Anti-join mode
    # can disable this so the source/parent entity remains the anchor.
    anchor = entity_names[0]
    if prefer_fact_anchor:
        for name in entity_names:
            if entities_map.get(name, {}).get("entity_type") == "fact":
                anchor = name
                break

    targets = [n for n in entity_names if n != anchor]
    visited_edges: list[dict] = []
    visited_nodes = {anchor}

    # Grow a minimum-risk join tree to reach each target.
    sequence = itertools.count()
    for target in targets:
        if target in visited_nodes:
            continue
        queue: list[tuple[float, int, str, list[dict]]] = [
            (0.0, next(sequence), start, []) for start in sorted(visited_nodes)
        ]
        heapq.heapify(queue)
        best_cost: dict[str, float] = {start: 0.0 for start in visited_nodes}
        chosen_path: list[dict] | None = None

        while queue:
            cost, _, node, path = heapq.heappop(queue)
            if cost > best_cost.get(node, float("inf")):
                continue
            if node == target:
                chosen_path = path
                break
            for edge in adj.get(node, []):
                weight = _edge_weight(edge)
                if weight == float("inf"):
                    continue
                neighbour = (
                    edge["to_entity"] if edge["_direction"] == "forward"
                    else edge["from_entity"]
                )
                new_cost = cost + weight
                if new_cost < best_cost.get(neighbour, float("inf")):
                    best_cost[neighbour] = new_cost
                    heapq.heappush(
                        queue,
                        (new_cost, next(sequence), neighbour, path + [edge]),
                    )

        if chosen_path is None:
            log.debug("Graph resolver: no path from %s to %s", visited_nodes, target)
            continue
        for step in chosen_path:
            edge_identity = step.get("id") or step.get("relationship_key") or (
                step.get("from_entity"), step.get("to_entity"),
                step.get("from_column"), step.get("to_column"),
            )
            if not any(
                (existing.get("id") or existing.get("relationship_key") or (
                    existing.get("from_entity"), existing.get("to_entity"),
                    existing.get("from_column"), existing.get("to_column"),
                )) == edge_identity
                for existing in visited_edges
            ):
                visited_edges.append(step)
        visited_nodes.update(
            node_name for step in chosen_path
            for node_name in [step["from_entity"], step["to_entity"]]
        )

    return visited_edges


def _multi_fact_fanout_risk(join_path: list[dict], entities_map: dict[str, dict]) -> list[str]:
    """
    Return the FACT-table entity names in this join skeleton that are only
    reachable through a shared dimension key — never a direct fact-to-fact
    or fact-to-bridge edge.

    This is the classic analytics fan-out antipattern: two unrelated fact
    tables (e.g. inventory snapshots and purchase receipts) sharing a common
    dimension key (e.g. PHARMACY_ID) get INNER JOINed together with no other
    constraint, multiplying rows before any SUM()/COUNT() runs — silently
    inflating the result by orders of magnitude with no error and no empty
    result set to signal something went wrong. A single fact table anchoring
    the query is always safe; risk only exists once 2+ facts are joined in.
    """
    fact_names = {
        name for name, ent in entities_map.items()
        if (ent.get("entity_type") or "").lower() == "fact"
    }
    if len(fact_names) < 2:
        return []

    joined_facts: set[str] = set()
    fact_to_fact_linked: set[str] = set()
    for step in join_path:
        f, t = step.get("from_entity", ""), step.get("to_entity", "")
        if f in fact_names:
            joined_facts.add(f)
        if t in fact_names:
            joined_facts.add(t)
        if f in fact_names and t in fact_names:
            fact_to_fact_linked.add(f)
            fact_to_fact_linked.add(t)

    if len(joined_facts) < 2:
        return []
    return sorted(joined_facts - fact_to_fact_linked)


# ══════════════════════════════════════════════════════════════════════════════
# JOIN skeleton builder
# ══════════════════════════════════════════════════════════════════════════════

def build_join_skeleton(
    join_path: list[dict],
    entities_map: dict[str, dict],
    anchor_entity: str,
    db_type: str,
    anti_join: bool = False,
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
    global_where: list[str] = []   # conditions on anchor/already-joined tables → SQL WHERE

    for step in join_path:
        fwd = step["_direction"] == "forward"
        from_ent = step["from_entity"]
        to_ent   = step["to_entity"]
        from_col = step["from_column"]
        to_col   = step["to_column"]
        jtype    = "LEFT" if anti_join else step.get("join_type", "INNER").upper()

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
            on_clause = (
                f"{old_alias}.{_quote_col(from_col, db_type)} = "
                f"{new_alias}.{_quote_col(to_col, db_type)}"
            )
        else:
            on_clause = (
                f"{new_alias}.{_quote_col(from_col, db_type)} = "
                f"{old_alias}.{_quote_col(to_col, db_type)}"
            )

        raw_conditions = step.get("join_conditions") or []
        if isinstance(raw_conditions, str):
            try:
                raw_conditions = json.loads(raw_conditions)
            except Exception:
                raw_conditions = []
        for condition in raw_conditions or []:
            if not isinstance(condition, dict):
                continue
            extra_from = str(condition.get("from_col") or "").strip()
            extra_to = str(condition.get("to_col") or "").strip()
            if not extra_from or not extra_to:
                continue
            if from_ent == old_ent:
                on_clause += (
                    f" AND {old_alias}.{_quote_col(extra_from, db_type)} = "
                    f"{new_alias}.{_quote_col(extra_to, db_type)}"
                )
            else:
                on_clause += (
                    f" AND {new_alias}.{_quote_col(extra_from, db_type)} = "
                    f"{old_alias}.{_quote_col(extra_to, db_type)}"
                )

        # ── WHERE conditions stored on this relationship ──────────────────
        # After alias substitution, split individual AND parts and route them:
        #   • Condition starts with new_alias.  → stays in JOIN ON  (dim-side filter;
        #     keeping it in ON preserves LEFT JOIN semantics correctly)
        #   • Condition references anchor or already-joined table → SQL WHERE clause
        #     (putting an anchor/fact-table filter in an ON clause of a LEFT JOIN
        #      would NOT exclude rows — it must be a hard WHERE filter instead)
        where_sql = (step.get("where_clause") or "").strip()
        if where_sql:
            # Substitute entity names → SQL aliases
            for ent_name, a in aliases.items():
                where_sql = where_sql.replace(f"{ent_name}.", f"{a}.")
            # Split on top-level AND (BETWEEN- and paren-aware) per condition
            parts = _split_and_conditions(where_sql)
            for part in parts:
                if part.upper().startswith(f"{new_alias.upper()}."):
                    # condition on the newly-joined dim → goes into ON clause
                    on_clause += f" AND {part}"
                else:
                    # condition on fact/anchor or previously-joined table → SQL WHERE
                    global_where.append(part)

        # ── Entity-level static filter (always applies to this table) ─────
        # entity_filter is stored on the entity itself, not on a specific join.
        # It always targets new_ent's columns, so it always goes into the ON clause.
        # e.g. "DIM_Patient.status = 'Active'" → joined as "AND pat.status = 'Active'"
        entity_filter_sql = (new_meta.get("entity_filter") or "").strip()
        if entity_filter_sql:
            ef = entity_filter_sql
            # Substitute entity name prefix → alias (e.g. "DIM_Patient." → "pat.")
            for ent_name, a in aliases.items():
                ef = ef.replace(f"{ent_name}.", f"{a}.")
            # If user wrote bare column names (no table prefix), prepend new alias
            ef_parts = _split_and_conditions(ef)
            for part in ef_parts:
                # If part already has an alias prefix, keep as-is; else prepend new_alias
                if "." not in part.split()[0]:
                    on_clause += f" AND {new_alias}.{part}"
                else:
                    on_clause += f" AND {part}"

        lines.append(f"{jtype:5} JOIN {new_tbl} {new_alias} ON {on_clause}")
        seen_nodes.add(new_ent)

    # ── Apply entity_filter for the anchor table → SQL WHERE ─────────────────
    # Anchor is in the FROM clause, not in a JOIN ON, so its entity_filter
    # must go into the SQL WHERE clause.
    anchor_meta = entities_map.get(anchor_entity, {})
    anchor_filter_sql = (anchor_meta.get("entity_filter") or "").strip()
    if anchor_filter_sql:
        anchor_alias = aliases.get(anchor_entity, anchor_entity[:3].lower())
        af = anchor_filter_sql
        for ent_name, a in aliases.items():
            af = af.replace(f"{ent_name}.", f"{a}.")
        af_parts = _split_and_conditions(af)
        for part in af_parts:
            if "." not in part.split()[0]:
                global_where.append(f"{anchor_alias}.{part}")
            else:
                global_where.append(part)

    # Append static WHERE clause for fact/anchor-table conditions
    if global_where:
        lines.append("WHERE " + "\n  AND ".join(global_where))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def _is_confirmed_row(row: dict) -> bool:
    """Legacy rows have status='' or no status column — treat those as confirmed.
    Only explicit status='suggested' marks an unreviewed auto-generated row."""
    return (row.get("status") or "confirmed") != "suggested"


def _confirmed_subgraph(graph: dict) -> dict:
    """Filter a full graph down to admin-confirmed entities and relationships.
    Relationships survive only when both endpoints are confirmed entities."""
    ents  = [e for e in graph.get("entities", []) if _is_confirmed_row(e)]
    names = {e["entity_name"] for e in ents}
    rels  = [
        r for r in graph.get("relationships", [])
        if _is_confirmed_row(r)
        and r.get("from_entity") in names
        and r.get("to_entity") in names
    ]
    props = [
        p for p in graph.get("properties", []) or []
        if p.get("entity_name") in names
    ]
    return {"entities": ents, "relationships": rels, "properties": props}


def _client_allows_suggested(account_id: str) -> bool:
    """Per-client toggle: may unreviewed (suggested) graph rows feed SQL generation?
    Defaults to True (existing behaviour) when the client row is unavailable."""
    try:
        import store
        client = store.get_client(account_id) or {}
        val = client.get("graph_use_suggested")
        return True if val is None else bool(int(val))
    except Exception:
        return True


def _resolve_on_graph(
    question: str,
    db_type: str,
    graph: dict,
    intent: Optional[dict],
    required_entities,
    metric_formula_tables,
) -> dict:
    """Run detection + pathfinding + skeleton build against one graph snapshot."""
    entities     = graph.get("entities", [])
    rels         = graph.get("relationships", [])
    entity_count = len(entities)
    empty = {
        "enabled": False, "detected": [], "join_skeleton": "",
        "anchor": "", "entity_count": entity_count, "entities": entities,
    }

    if not entities:
        return empty

    anti_join = bool((intent or {}).get("wants_missing_records"))
    detected = detect_entities(
        question,
        graph,
        required_tables=metric_formula_tables,
        required_entities=required_entities,
    )
    log.debug("Graph: question=%r detected=%s", question[:60], detected)

    if not detected:
        return empty

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
            return empty
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
            "entities": entities,
            "properties": graph.get("properties") or [],
        }

    entities_map = {e["entity_name"]: e for e in entities}
    join_path    = find_join_path(detected, graph, prefer_fact_anchor=not anti_join)

    # Determine anchor (fact table if possible)
    anchor = detected[0]
    if not anti_join:
        for name in detected:
            if entities_map.get(name, {}).get("entity_type") == "fact":
                anchor = name
                break

    skeleton = build_join_skeleton(join_path, entities_map, anchor, db_type, anti_join=anti_join)
    fanout_risk_facts = _multi_fact_fanout_risk(join_path, entities_map)
    resolved_edges = []
    for edge in join_path:
        from_meta = entities_map.get(edge.get("from_entity", ""), {})
        to_meta = entities_map.get(edge.get("to_entity", ""), {})
        raw_extra = edge.get("join_conditions") or []
        if isinstance(raw_extra, str):
            try:
                raw_extra = json.loads(raw_extra)
            except Exception:
                raw_extra = []
        conditions = [[edge.get("from_column", ""), edge.get("to_column", "")]]
        conditions.extend([
            [condition.get("from_col", ""), condition.get("to_col", "")]
            for condition in raw_extra or [] if isinstance(condition, dict)
        ])
        resolved_edges.append({
            "id": edge.get("id"),
            "relationship_key": edge.get("relationship_key", ""),
            "from_entity": edge.get("from_entity", ""),
            "to_entity": edge.get("to_entity", ""),
            "from_table": from_meta.get("table_name", ""),
            "from_schema": from_meta.get("schema_name", ""),
            "to_table": to_meta.get("table_name", ""),
            "to_schema": to_meta.get("schema_name", ""),
            "conditions": conditions,
            "join_type": "LEFT" if anti_join else (edge.get("join_type") or "INNER").upper(),
            "provenance": edge.get("generated_by", ""),
            "validation_status": edge.get("validation_status", "untested"),
            "weight": round(_edge_weight(edge), 3),
        })

    return {
        "enabled":       bool(skeleton),
        "detected":      detected,
        "join_skeleton": skeleton,
        "anchor":        anchor,
        "entity_count":  entity_count,
        "anti_join":     anti_join,
        "entities":      entities,
        "edge_ids":      [edge["id"] for edge in resolved_edges if edge.get("id") is not None],
        "resolved_edges": resolved_edges,
        "fanout_risk_facts": fanout_risk_facts,
        "properties":    graph.get("properties") or [],
    }


def resolve_for_question(
    question: str,
    account_id: str,
    db_type: str,
    graph: Optional[dict] = None,
    intent: Optional[dict] = None,
    required_entities: Optional[list[str] | set[str]] = None,
    metric_formula_tables: Optional[list[str] | set[str]] = None,
    use_suggested: Optional[bool] = None,
) -> dict:
    """
    Main entry point called before SQL generation.

    Resolution order:
      1. Admin-confirmed entities/relationships only (status != 'suggested').
      2. If that yields nothing useful AND the client allows it
         (client.graph_use_suggested, default on), fall back to the full
         graph including unreviewed suggestions — logged when it happens.

    use_suggested: override the per-client toggle (None = read client row).

    Returns:
      {
        "enabled":        bool   — False if graph is empty or no entities detected
        "detected":       list   — entity names detected in the question
        "join_skeleton":  str    — SQL FROM + JOIN clause ready for injection
        "anchor":         str    — the root entity (usually the fact table)
        "entity_count":   int    — number of entities in the graph
        "graph_scope":    str    — 'confirmed' | 'suggested_fallback' (when enabled)
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

    entities     = graph.get("entities", [])
    entity_count = len(entities)
    if not entities:
        return empty

    confirmed = _confirmed_subgraph(graph)
    has_suggested = (
        len(confirmed["entities"]) < entity_count
        or len(confirmed["relationships"]) < len(graph.get("relationships", []))
    )

    confirmed_result: dict = {}
    if confirmed["entities"]:
        confirmed_result = _resolve_on_graph(
            question, db_type, confirmed, intent,
            required_entities, metric_formula_tables,
        )
        # A confirmed multi-entity join — or a graph with nothing suggested —
        # is final. A single-entity anchor may just be the weak fact fallback,
        # so give suggested content a chance to produce a real join below.
        if confirmed_result.get("enabled") and (
            len(confirmed_result.get("detected", [])) >= 2 or not has_suggested
        ):
            confirmed_result["entity_count"] = entity_count
            confirmed_result["graph_scope"] = "confirmed"
            return confirmed_result

    if has_suggested:
        allow = use_suggested if use_suggested is not None else _client_allows_suggested(account_id)
        if allow:
            full_result = _resolve_on_graph(
                question, db_type, graph, intent,
                required_entities, metric_formula_tables,
            )
            # Prefer the suggested-inclusive result only when it adds value
            # (a join the confirmed graph could not produce).
            if full_result.get("enabled") and (
                not confirmed_result.get("enabled")
                or len(full_result.get("detected", [])) > len(confirmed_result.get("detected", []))
            ):
                log.info(
                    "Graph resolver: using unreviewed suggestions for %r "
                    "(confirmed graph had no full join path — review the Entity Graph "
                    "or set client.graph_use_suggested=0 to disable this fallback)",
                    question[:60],
                )
                full_result["graph_scope"] = "suggested_fallback"
                return full_result
        else:
            log.info(
                "Graph resolver: suggested-only graph content skipped for %r "
                "(client.graph_use_suggested=0)", question[:60],
            )

    if confirmed_result.get("enabled"):
        confirmed_result["entity_count"] = entity_count
        confirmed_result["graph_scope"] = "confirmed"
        return confirmed_result

    return {**empty, "entity_count": entity_count}
