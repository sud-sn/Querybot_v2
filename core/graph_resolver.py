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

from core.date_roles import (
    DATE_DIMENSION_TABLE_HINTS,
    DATE_ROLES,
    date_role_terms,
    normalize_date_role_text,
    question_has_temporal_intent,
    relationship_matches_date_role,
)
from core.relationship_identity import physical_relationship_signature

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


# Generic temporal vocabulary. These words say *when* a user wants data, never
# *which* governed business date answers the question. Every role-playing date
# entity in a tenant shares them, so a match on one of these alone identifies
# no particular role.
_GENERIC_TEMPORAL_WORDS = frozenset({
    "date", "dates", "day", "days", "daily", "week", "weeks", "weekly",
    "month", "months", "monthly", "quarter", "quarters", "quarterly",
    "year", "years", "yearly", "period", "periods", "time", "times",
    "timeline", "calendar", "last", "latest", "recent", "previous", "prior",
    "current", "next", "today", "yesterday", "tomorrow", "available",
    "earliest", "trend", "trends", "when", "range", "ago",
    "ytd", "mtd", "qtd", "yoy", "mom", "wow",
})

# Words that mark an entity as a business-date concept rather than an ordinary
# business dimension. Kept deliberately narrow: an ordinary entity (Customer,
# Warehouse, Invoice) does not carry one of these as a standalone word.
_DATE_ENTITY_MARKERS = frozenset({
    "date", "dates", "dt", "day", "week", "month", "quarter", "year",
    "period", "calendar",
})


def _collapse(text: str) -> str:
    """Normalize and collapse whitespace so phrase matching is contiguous."""
    return " ".join(_normalize(text).split())


def is_date_role_entity(entity: dict) -> bool:
    """Whether a graph entity represents a (role-playing) business date.

    Role-playing date dimensions are the one entity class whose *name* is made
    almost entirely of generic temporal words, which is what makes loose
    per-word matching pick them at random. Recognized from tenant metadata
    only — name/display-name markers plus the shared date-dimension table
    hints — never from a hardcoded business name.
    """
    words = set(_normalize(entity.get("entity_name") or "").split())
    words |= set(_normalize(entity.get("display_name") or "").split())
    if words & _DATE_ENTITY_MARKERS:
        return True
    table = re.sub(
        r"[^A-Z0-9]+", "_",
        str(entity.get("table_name") or "").split(".")[-1].upper(),
    )
    if table in DATE_DIMENSION_TABLE_HINTS:
        return True
    return any(hint in table for hint in ("DIM_DATE", "DATE_DIM", "CALENDAR"))


def _role_phrase_variants(phrase: str) -> list[str]:
    """Business phrases that authoritatively name one date role.

    A phrase qualifies only when it still carries a non-generic business word:
    "invoice date" and "modified date" name a role, bare "date" or "last date"
    name every role in the tenant and therefore none of them.

    Leading generic modifiers are also dropped ("Last Modified Date" →
    "modified date") so a user who types the business half of the role name is
    understood, while the business word itself stays mandatory.
    """
    words = _collapse(phrase).split()
    if not words:
        return []
    variants: list[str] = []
    for start in range(len(words)):
        candidate = words[start:]
        if not any(word not in _GENERIC_TEMPORAL_WORDS for word in candidate):
            continue
        variants.append(" ".join(candidate))
        # Only leading *generic* modifiers may be dropped; removing a business
        # word would turn "Invoice Date" into "date" and match everything.
        if words[start] not in _GENERIC_TEMPORAL_WORDS:
            break
    return variants


def date_role_entity_phrases(graph: dict) -> dict[str, set[str]]:
    """Registered business phrases naming each date-role entity in the graph.

    Sources are tenant metadata plus the builtin business-date vocabulary:
    the entity's own name and display name, the labels of relationships that
    target it (the business role of a role-playing edge), its properties'
    display names and synonyms, and the registered synonyms of any builtin
    date role that edge label maps onto ("invoiced on", "billing date").
    """
    phrases: dict[str, set[str]] = {}
    date_entities = {
        str(entity.get("entity_name") or ""): entity
        for entity in graph.get("entities") or []
        if entity.get("entity_name") and is_date_role_entity(entity)
    }
    if not date_entities:
        return {}

    for name, entity in date_entities.items():
        bucket = phrases.setdefault(name, set())
        for raw in (name, entity.get("display_name") or ""):
            bucket.update(_role_phrase_variants(raw))

    for rel in graph.get("relationships") or []:
        target = str(rel.get("to_entity") or "")
        if target not in date_entities:
            # Role-playing edges are stored fact → date, but tolerate either
            # direction so a reversed row still contributes its business role.
            target = str(rel.get("from_entity") or "")
            if target not in date_entities:
                continue
        bucket = phrases.setdefault(target, set())
        for raw in (rel.get("label") or "", rel.get("description") or ""):
            bucket.update(_role_phrase_variants(raw))
            label_norm = normalize_date_role_text(raw)
            if not label_norm:
                continue
            for role in DATE_ROLES:
                if label_norm in {
                    normalize_date_role_text(role.label),
                    role.key.replace("_", " "),
                }:
                    for term in date_role_terms(role):
                        bucket.update(_role_phrase_variants(term))

    for prop in graph.get("properties") or []:
        owner = str(prop.get("entity_name") or "")
        if owner not in date_entities:
            continue
        bucket = phrases.setdefault(owner, set())
        for field in ("display_name", "synonyms"):
            raw = prop.get(field) or ""
            for piece in (raw.split(",") if field == "synonyms" else [raw]):
                bucket.update(_role_phrase_variants(piece))
    return phrases


def question_names_date_role(question: str, phrases: set[str] | frozenset[str]) -> bool:
    """Whether the question authoritatively names this date role.

    The complete business phrase must appear contiguously at word boundaries.
    A trailing plural is tolerated ("by invoice dates"); a single word from the
    role name is never enough, which is the whole point of this gate.
    """
    q = _collapse(question)
    if not q:
        return False
    for phrase in phrases or ():
        words = phrase.split()
        if not words:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(
            rf"{re.escape(word)}s?" for word in words
        ) + r"(?!\w)"
        if re.search(pattern, q):
            return True
    return False


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


def entity_name_for_table(graph: dict, table_ref: str) -> str:
    """Map a qualified ("SCHEMA.TABLE") or bare table name to its graph
    entity_name, so callers holding a table name from another source (e.g.
    the structured semantic model's date_roles) can force that table into
    entity-graph resolution via required_entities.

    Returns "" when no entity matches — callers should treat that as
    "nothing to force," not an error; a table that hasn't been discovered
    into the graph yet simply can't be targeted this way.
    """
    table_ref = (table_ref or "").strip()
    if not table_ref:
        return ""
    parts = table_ref.split(".")
    bare = parts[-1].upper()
    schema = parts[-2].upper() if len(parts) >= 2 else ""
    for ent in graph.get("entities", []) or []:
        if (ent.get("table_name") or "").upper() != bare:
            continue
        ent_schema = (ent.get("schema_name") or "").upper()
        if schema and ent_schema and ent_schema != schema:
            continue
        return ent.get("entity_name", "")
    return ""


def date_role_entity_for_binding(graph: dict, binding: dict) -> str:
    """Resolve which date-role entity a governed date binding points at.

    Several role-playing date entities normally share one physical date
    dimension, so ``entity_name_for_table`` alone returns whichever role
    happens to come first — that is how an approved Invoice Date binding could
    force "Last Modified Date" into the required entities. Resolve by physical
    edge first (the binding names the exact fact column and dimension key),
    then by the role's own business name, and only fall back to the table when
    exactly one entity owns it.

    Returns "" when nothing matches; callers treat that as "nothing to force".
    """
    binding = binding or {}
    fact_table = str(binding.get("fact_table") or "")
    dim_table = str(binding.get("dimension_table") or binding.get("date_table") or "")
    fact_col = str(binding.get("fact_column") or "").strip().strip('[]"`').upper()
    dim_col = str(binding.get("dimension_key") or "").strip().strip('[]"`').upper()

    entities = graph.get("entities") or []
    by_name = {str(e.get("entity_name") or ""): e for e in entities if e.get("entity_name")}
    fact_entity = entity_name_for_table(graph, fact_table) if fact_table else ""

    # 1. The exact physical edge the binding describes.
    if fact_entity and fact_col and dim_col:
        for rel in graph.get("relationships") or []:
            left = str(rel.get("from_entity") or "")
            right = str(rel.get("to_entity") or "")
            from_col = str(rel.get("from_column") or "").strip().strip('[]"`').upper()
            to_col = str(rel.get("to_column") or "").strip().strip('[]"`').upper()
            if left == fact_entity and from_col == fact_col and to_col == dim_col:
                if is_date_role_entity(by_name.get(right, {})):
                    return right
            if right == fact_entity and to_col == fact_col and from_col == dim_col:
                if is_date_role_entity(by_name.get(left, {})):
                    return left

    # 2. The date entity whose business name is this role.
    role_names = {
        _collapse(str(binding.get(key) or "").replace("_", " "))
        for key in ("context_name", "date_role", "name", "business_role")
    } - {""}
    if role_names:
        for name, entity in by_name.items():
            if not is_date_role_entity(entity):
                continue
            if dim_table and not _same_table_ref(entity, dim_table):
                continue
            candidates = {_collapse(name), _collapse(str(entity.get("display_name") or ""))}
            if candidates & role_names:
                return name

    # 3. A single entity owning the dimension table is unambiguous.
    if dim_table:
        owners = [
            name for name, entity in by_name.items()
            if _same_table_ref(entity, dim_table)
        ]
        if len(owners) == 1:
            return owners[0]
    return ""


def _same_table_ref(entity: dict, table_ref: str) -> bool:
    parts = [p.strip().strip('[]"`').upper() for p in str(table_ref or "").split(".") if p.strip()]
    if not parts:
        return False
    bare = parts[-1]
    schema = parts[-2] if len(parts) >= 2 else ""
    if str(entity.get("table_name") or "").strip().strip('[]"`').upper() != bare:
        return False
    ent_schema = str(entity.get("schema_name") or "").strip().strip('[]"`').upper()
    return not (schema and ent_schema and schema != ent_schema)


def infer_connected_default_date_fact(
    graph: dict,
    *,
    requested_entities: list[str] | set[str] | None,
    requested_tables: list[str] | set[str] | None,
    candidate_fact_tables: list[str] | set[str] | None,
    excluded_tables: list[str] | set[str] | None = None,
    allow_suggested: bool = True,
) -> dict:
    """Infer the uniquely connected fact that owns a default business date.

    This is used only for generic temporal questions whose business wording
    identifies dimensions but not a fact, for example "patient count by state
    today". Candidates are restricted to facts with an approved default date
    role supplied by the caller. The graph remains authoritative: broken and
    zero-match edges are rejected by the normal risk-weighted pathfinder, and
    paths that travel through a different fact are rejected to avoid fan-out.

    A close tie is returned as ambiguous rather than guessed.
    """
    entities = graph.get("entities") or []
    by_name = {str(entity.get("entity_name") or ""): entity for entity in entities}

    def table_key(value: str) -> str:
        parts = [
            part.strip().strip("[]\"`").upper()
            for part in str(value or "").split(".")
            if part.strip().strip("[]\"`")
        ]
        return ".".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")

    def entity_table(entity: dict) -> str:
        table = str(entity.get("table_name") or "")
        schema = str(entity.get("schema_name") or "")
        return table_key(f"{schema}.{table}" if schema else table)

    excluded = {table_key(table) for table in (excluded_tables or []) if table}
    requested = {str(name) for name in (requested_entities or []) if str(name)}
    for table in requested_tables or []:
        name = entity_name_for_table(graph, str(table))
        if name:
            requested.add(name)

    # Only business dimensions should drive fact inference. Date dimensions
    # and already detected facts are consequences of temporal planning, not
    # evidence that a particular business event was requested.
    targets = {
        name for name in requested
        if name in by_name
        and str(by_name[name].get("entity_type") or "").lower() not in {"fact", "bridge"}
        and entity_table(by_name[name]) not in excluded
    }
    if not targets:
        return {"status": "none", "reason": "no requested business dimensions"}

    candidate_entities: dict[str, str] = {}
    for table in candidate_fact_tables or []:
        name = entity_name_for_table(graph, str(table))
        entity = by_name.get(name, {})
        if name and str(entity.get("entity_type") or "").lower() == "fact":
            candidate_entities[table_key(str(table))] = name
    if not candidate_entities:
        return {"status": "none", "reason": "no default-date fact candidates in graph"}

    confirmed = _confirmed_subgraph(graph)
    graph_variants = [("confirmed", confirmed)]
    if allow_suggested and (
        len(confirmed.get("entities") or []) < len(entities)
        or len(confirmed.get("relationships") or []) < len(graph.get("relationships") or [])
    ):
        graph_variants.append(("suggested_fallback", graph))

    ranked: list[dict] = []
    for fact_table, fact_entity in candidate_entities.items():
        for graph_scope, candidate_graph in graph_variants:
            candidate_names = {
                str(entity.get("entity_name") or "")
                for entity in candidate_graph.get("entities") or []
            }
            if fact_entity not in candidate_names or not targets.issubset(candidate_names):
                continue
            path = find_join_path(
                [fact_entity, *sorted(targets)],
                candidate_graph,
                prefer_fact_anchor=True,
            )
            path_nodes = {fact_entity}
            for edge in path:
                path_nodes.add(str(edge.get("from_entity") or ""))
                path_nodes.add(str(edge.get("to_entity") or ""))
            if not targets.issubset(path_nodes):
                continue
            traversed_other_facts = {
                name for name in path_nodes
                if name != fact_entity
                and str(by_name.get(name, {}).get("entity_type") or "").lower() == "fact"
            }
            if traversed_other_facts:
                continue
            unique_edges: dict[object, dict] = {}
            for edge in path:
                identity = edge.get("id") or edge.get("relationship_key") or (
                    edge.get("from_entity"), edge.get("to_entity"),
                    edge.get("from_column"), edge.get("to_column"),
                )
                unique_edges[identity] = edge
            ranked.append({
                "fact_table": fact_table,
                "fact_entity": fact_entity,
                "graph_scope": graph_scope,
                "cost": round(sum(_edge_weight(edge) for edge in unique_edges.values()), 4),
                "hops": len(unique_edges),
                "target_entities": sorted(targets),
                "edge_ids": [
                    edge.get("id") for edge in unique_edges.values()
                    if edge.get("id") is not None
                ],
            })
            # A confirmed path always wins over a suggested fallback for the
            # same fact, so there is no reason to evaluate that fact again.
            break

    if not ranked:
        return {"status": "none", "reason": "no safe graph path to a default-date fact"}

    ranked.sort(key=lambda item: (
        0 if item["graph_scope"] == "confirmed" else 1,
        item["cost"], item["hops"], item["fact_table"],
    ))
    best = ranked[0]
    best_scope = best["graph_scope"]
    close = [
        item for item in ranked
        if item["graph_scope"] == best_scope
        and abs(float(item["cost"]) - float(best["cost"])) <= 1.0
        and abs(int(item["hops"]) - int(best["hops"])) <= 1
    ]
    if len(close) > 1:
        return {
            "status": "ambiguous",
            "reason": "multiple default-date facts are equally connected",
            "candidates": close,
        }
    return {
        "status": "selected",
        "reason": "unique default-date fact connected to requested dimensions",
        **best,
    }


def detect_entities(
    question: str,
    graph: dict,
    required_tables: list[str] | set[str] | None = None,
    required_entities: list[str] | set[str] | None = None,
    authoritative_fact_tables: list[str] | set[str] | None = None,
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
    scores: dict[str, int] = {}

    entities   = graph.get("entities", [])
    required_table_candidates = _table_candidates_from_required(required_tables)
    authoritative_fact_candidates = _table_candidates_from_required(authoritative_fact_tables)
    required_entity_names = {str(e) for e in (required_entities or [])}
    date_role_phrases = date_role_entity_phrases(graph)

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

        # ── Role-playing date entities: authoritative signals only ───────────
        # Every business date in a tenant ("Invoice Date", "Order Date",
        # "Last Modified Date") is named almost entirely out of generic
        # temporal words, and several of them share one physical date
        # dimension. The loose scoring below therefore selected them by
        # coincidence: "revenue for the last 2 days" matched "Last Modified
        # Date" on the single word "last", which forced an audit date into the
        # join skeleton beside the metric's own approved date. Two date edges
        # into the same dimension then either return no rows or provoke an
        # irrelevant clarification.
        #
        # So a date role must be named, not guessed. Lexical selection requires
        # the complete registered business phrase (or a registered synonym);
        # every other authoritative source — the metric's approved default date
        # role, a date role selected earlier in the thread, a prior
        # clarification, the compiled semantic plan — arrives through
        # required_entities, which is honored above and below.
        if is_date_role_entity(ent):
            if name in required_entity_names:
                scores[name] = score
                continue
            if not question_names_date_role(q, date_role_phrases.get(name, set())):
                continue
            # Named explicitly. Score on the role itself, never on the shared
            # date-dimension table name or its generic calendar columns.
            score += 10
            if score >= 3:
                scores[name] = score
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
            and authoritative_fact_candidates
            and table_name not in authoritative_fact_candidates
            and f"{schema_name}.{table_name}" not in authoritative_fact_candidates
        ):
            # The final pre-generation pass may carry an approved semantic,
            # metric, or date-role fact scope. Do not let a weaker lexical
            # hit introduce another fact and force a contradictory join path.
            continue
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


def _path_identity(path: list[dict]) -> tuple:
    """Canonical physical identity, independent of persisted row IDs.

    Schema discovery and Date Role promotion can represent the same physical
    FK more than once with different database IDs.  IDs are useful for
    replaying a selected option, but they must not make one logical join look
    like two equally governed paths.
    """
    return tuple(physical_relationship_signature(edge) for edge in path)


def _edge_selection_id(edge: dict):
    """Stable JSON-safe identifier used to resume a real path choice."""
    if edge.get("id") is not None:
        return edge.get("id")
    if edge.get("relationship_key"):
        return edge.get("relationship_key")
    return repr(physical_relationship_signature(edge))


def _path_selection_ids(path: list[dict]) -> list:
    return [_edge_selection_id(edge) for edge in path]


def _selected_path_matches(path: list[dict], selected_edge_ids) -> bool:
    selected = [str(value) for value in (selected_edge_ids or []) if value is not None]
    actual = [str(value) for value in _path_selection_ids(path)]
    return bool(selected) and selected == actual


def _path_label(path: list[dict], entities: list[dict] | None = None) -> str:
    """A business-readable name for one join path.

    The previous form joined every edge label with " -> ", which produced two
    kinds of unusable option: role-playing edges that share a label repeated
    themselves ("Confirmed Delivery Date -> Confirmed Delivery Date"), and
    unlabelled edges fell back to raw physical table names
    ("CUS_ORD_IVC_FCT to ITM_DMS -> PCH_ORD_RCT_FCT to ITM_DMS"). Neither tells
    a business user what they are choosing between.

    So: prefer the edge's business role, then the entities' display names, never
    a bare physical name when a display name exists; collapse consecutive
    repeats; and describe intermediate hops with "via".
    """
    display_by_entity = {
        str(entity.get("entity_name") or ""): str(
            entity.get("display_name") or entity.get("entity_name") or ""
        ).strip()
        for entity in entities or []
        if entity.get("entity_name")
    }

    def _name(entity_name: Any) -> str:
        key = str(entity_name or "")
        return display_by_entity.get(key) or key

    labels: list[str] = []
    for edge in path:
        label = str(edge.get("label") or "").strip()
        if not label:
            label = str(edge.get("business_role") or "").strip()
        if not label:
            label = f"{_name(edge.get('from_entity'))} via {_name(edge.get('to_entity'))}"
        # Role-playing edges through one dimension repeat the same business
        # role; saying it twice adds nothing.
        if labels and labels[-1].casefold() == label.casefold():
            continue
        labels.append(label)
    return " via ".join(labels)


def _question_edge_weight(edge: dict, question: str) -> float:
    """Prefer a confirmed business role explicitly named by the user."""
    weight = _edge_weight(edge)
    if not question or weight == float("inf"):
        return weight
    q_norm = _normalize(question)
    q_tokens = _tokens(question)
    phrases = [
        str(edge.get("label") or "").strip(),
        str(edge.get("business_role") or "").strip(),
    ]
    if any(
        phrase and _phrase_in_question(phrase, q_tokens, q_norm)
        for phrase in phrases
    ):
        weight -= 0.75
    # Operational audit timestamps are valid graph metadata but are almost
    # never the business event requested by an analytical question.  Keep
    # them available when explicitly named while preventing them from tying
    # with Invoice/Order/Shipment paths for unrelated questions.
    edge_role = _normalize(" ".join(phrases))
    audit_phrases = (
        "last modified", "last updated", "modified date", "modified time",
        "update date", "updated date", "audit date", "audit time",
        "load date", "load time", "etl date", "etl time",
    )
    if any(phrase in edge_role for phrase in audit_phrases) and not any(
        phrase in q_norm for phrase in audit_phrases
    ):
        weight += 12.0
    return max(0.1, weight)


def find_join_path_with_diagnostics(
    entity_names: list[str],
    graph: dict,
    prefer_fact_anchor: bool = True,
    *,
    anti_join: bool = False,
    question: str = "",
    selected_edge_ids=None,
) -> tuple[list[dict], dict]:
    """
    Weighted shortest path from the anchor entity (fact table preferred)
    through relationship edges to reach all other entities. Returns an ordered
    list of JOIN steps. Edge weights prefer governed/validated DB constraints
    and penalize untested heuristics, fanout, warnings, and many-to-many joins.

      [{"from_entity", "to_entity", "from_column", "to_column",
        "join_type", "relationship_type", "_direction"}]

    Returns ``(path, diagnostics)``. Diagnostics identify unreachable entities
    and equally safe alternative paths so callers can fail closed or ask a
    business-role clarification instead of silently selecting one.
    """
    diagnostics = {"unreachable": [], "ambiguous_targets": []}
    if not entity_names or len(entity_names) < 2:
        return [], diagnostics

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
        queue: list[tuple[float, int, str, list[dict], frozenset[str]]] = [
            (0.0, next(sequence), start, [], frozenset({start}))
            for start in sorted(visited_nodes)
        ]
        heapq.heapify(queue)
        best_cost: dict[str, float] = {start: 0.0 for start in visited_nodes}
        candidates: list[tuple[float, list[dict]]] = []
        candidate_ids: set[tuple] = set()

        while queue:
            cost, _, node, path, path_nodes = heapq.heappop(queue)
            if cost > best_cost.get(node, float("inf")) + 0.25:
                continue
            if node == target:
                identity = _path_identity(path)
                if identity not in candidate_ids:
                    candidates.append((cost, path))
                    candidate_ids.add(identity)
                if len(candidates) >= 2:
                    break
                continue
            for edge in adj.get(node, []):
                from core.join_planner import traversal_is_admissible
                neighbour = (
                    edge["to_entity"] if edge["_direction"] == "forward"
                    else edge["from_entity"]
                )
                secondary_fact_target = (
                    edge["_direction"] == "backward"
                    and neighbour in entity_names
                    and str(entities_map.get(neighbour, {}).get("entity_type") or "").lower() == "fact"
                )
                admissible, _ = traversal_is_admissible(
                    edge,
                    entities_map,
                    direction=edge["_direction"],
                    anti_join=anti_join,
                )
                if not admissible and not secondary_fact_target:
                    continue
                if neighbour in path_nodes:
                    continue
                weight = _question_edge_weight(edge, question)
                if weight == float("inf"):
                    continue
                new_cost = cost + weight
                if new_cost <= best_cost.get(neighbour, float("inf")) + 0.25:
                    best_cost[neighbour] = min(
                        new_cost, best_cost.get(neighbour, float("inf"))
                    )
                    heapq.heappush(
                        queue,
                        (
                            new_cost, next(sequence), neighbour, path + [edge],
                            path_nodes | {neighbour},
                        ),
                    )

        if not candidates:
            log.debug("Graph resolver: no path from %s to %s", visited_nodes, target)
            diagnostics["unreachable"].append(target)
            continue
        candidates.sort(
            key=lambda item: (item[0], len(item[1]), repr(_path_identity(item[1])))
        )
        selected_candidate = next(
            (
                candidate for candidate in candidates
                if _selected_path_matches(candidate[1], selected_edge_ids)
            ),
            None,
        )
        chosen_cost, chosen_path = selected_candidate or candidates[0]
        if (
            selected_candidate is None
            and len(candidates) > 1
            and candidates[1][0] - candidates[0][0] <= 0.25
        ):
            _graph_entities = graph.get("entities") or []
            _path_options: list[dict] = []
            for _, path in candidates[:2]:
                label = _path_label(path, _graph_entities)
                # Two paths a user cannot tell apart are not a choice. When the
                # business labels collapse to the same text the paths mean the
                # same thing to them, so offer it once and let the deterministic
                # ranking pick — asking twice with identical buttons was the
                # defect, not the ambiguity.
                if any(
                    option["label"].casefold() == label.casefold()
                    for option in _path_options
                ):
                    continue
                _path_options.append({
                    "label": label,
                    "value": label,
                    "edge_ids": _path_selection_ids(path),
                })
            if len(_path_options) > 1:
                diagnostics["ambiguous_targets"].append({
                    "target": target,
                    "options": _path_options,
                })
            else:
                log.info(
                    "Graph: %d equally-ranked paths to %s are not "
                    "business-distinguishable (%s) — using the top-ranked path "
                    "instead of asking an unanswerable question",
                    len(candidates[:2]), target,
                    _path_options[0]["label"] if _path_options else "unlabelled",
                )
        for step in chosen_path:
            edge_identity = physical_relationship_signature(step)
            if not any(
                physical_relationship_signature(existing) == edge_identity
                for existing in visited_edges
            ):
                visited_edges.append(step)
        visited_nodes.update(
            node_name for step in chosen_path
            for node_name in [step["from_entity"], step["to_entity"]]
        )

    return visited_edges, diagnostics


def find_join_path(
    entity_names: list[str],
    graph: dict,
    prefer_fact_anchor: bool = True,
    *,
    anti_join: bool = False,
) -> list[dict]:
    """Backward-compatible path-only interface used by existing callers."""
    path, _ = find_join_path_with_diagnostics(
        entity_names,
        graph,
        prefer_fact_anchor=prefer_fact_anchor,
        anti_join=anti_join,
    )
    return path


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
    for step in join_path:
        f, t = step.get("from_entity", ""), step.get("to_entity", "")
        if f in fact_names:
            joined_facts.add(f)
        if t in fact_names:
            joined_facts.add(t)

    if len(joined_facts) < 2:
        return []
    # A direct fact-to-fact edge is even more dangerous than a shared-
    # dimension path.  Join Planner V2 never exempts it.
    return sorted(joined_facts)


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
    """Return whether a row is executable governance.

    Legacy rows have no status and pre-date graph review, so they remain
    executable for backwards compatibility. Every explicit non-confirmed
    state (including ``suggested`` and ``rejected``) is non-executable.
    """
    return str(row.get("status") or "").strip().lower() in {"", "confirmed"}


def _is_reviewable_row(row: dict) -> bool:
    """Return whether a row may appear in an explicit admin preview.

    Rejected rows must never re-enter planning, even when an administrator
    explicitly previews suggested graph content.
    """
    return str(row.get("status") or "").strip().lower() in {
        "", "confirmed", "suggested",
    }


def _filter_subgraph(graph: dict, predicate) -> dict:
    """Filter graph rows and enforce endpoint/property ownership integrity."""
    ents = [e for e in graph.get("entities", []) if predicate(e)]
    names = {e["entity_name"] for e in ents}
    rels = [
        r for r in graph.get("relationships", [])
        if predicate(r)
        and r.get("from_entity") in names
        and r.get("to_entity") in names
    ]
    props = [
        p for p in graph.get("properties", []) or []
        if predicate(p) and p.get("entity_name") in names
    ]
    return {"entities": ents, "relationships": rels, "properties": props}


def _confirmed_subgraph(graph: dict) -> dict:
    """Filter a full graph down to admin-confirmed entities and relationships.
    Relationships survive only when both endpoints are confirmed entities."""
    return _filter_subgraph(graph, _is_confirmed_row)


def _reviewable_subgraph(graph: dict) -> dict:
    """Filter rejected rows out of the admin suggested-content preview."""
    return _filter_subgraph(graph, _is_reviewable_row)


def _client_allows_suggested(account_id: str) -> bool:
    """Per-client toggle: may unreviewed (suggested) graph rows feed SQL generation?
    Defaults to False. Unreviewed relationships are useful onboarding evidence,
    but they must not silently become executable join plans for a new tenant.
    Clients can still opt in explicitly while their graph is being curated."""
    try:
        import store
        client = store.get_client(account_id) or {}
        val = client.get("graph_use_suggested")
        return False if val is None else bool(int(val))
    except Exception:
        return False


def _anti_join_anchor(question: str, detected: list[str], entities_map: dict[str, dict]) -> str:
    """Choose the records being retained for a missing-record question.

    For wording such as ``orders without fills`` or ``customers with no
    invoices`` the left side is the business entity before the missing marker.
    This lexical choice is tenant agnostic and only ranks entities already
    grounded by the client's entity graph.
    """
    normalized = _normalize(question)
    marker = re.search(
        r"\b(?:without|with no|have no|has no|having no|missing|not matched|unmatched)\b",
        normalized,
    )
    prefix = normalized[: marker.start()] if marker else normalized
    prefix_tokens = _tokens(prefix)
    generic = {"fact", "dim", "dimension", "bridge", "table", "data", "rx"}
    ranked: list[tuple[int, int, str]] = []
    for index, name in enumerate(detected):
        entity = entities_map.get(name, {})
        phrases = [
            str(entity.get("display_name") or ""),
            str(name or ""),
            str(entity.get("table_name") or ""),
        ]
        score = 0
        for phrase in phrases:
            words = [
                word for word in _tokens(phrase)
                if len(word) >= 3 and word not in generic
            ]
            if not words:
                continue
            matched = sum(1 for word in words if word in prefix_tokens)
            score = max(score, matched * 10 + (5 if matched == len(words) else 0))
        ranked.append((score, -index, name))
    ranked.sort(reverse=True)
    return ranked[0][2] if ranked else (detected[0] if detected else "")


def _resolve_on_graph(
    question: str,
    db_type: str,
    graph: dict,
    intent: Optional[dict],
    required_entities,
    metric_formula_tables,
    authoritative_fact_tables,
    selected_edge_ids=None,
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

    def _unresolved_result(
        detected_entities: list[str],
        *,
        status: str,
        reason: str,
        options: list[dict] | None = None,
        missing_entities: list[str] | None = None,
    ) -> dict:
        join_plan = {
            "enabled": True,
            "status": status,
            "reason": reason,
            "clarification_options": options or [],
            "missing_entities": missing_entities or [],
            "edge_ids": [],
            "steps": [],
        }
        return {
            **empty,
            "enabled": True,
            "detected": detected_entities,
            "join_plan": join_plan,
            "planning_status": status,
            "clarification_options": options or [],
            "missing_entities": missing_entities or [],
            "reason": reason,
            "properties": graph.get("properties") or [],
        }

    anti_join = bool((intent or {}).get("wants_missing_records"))
    detected = detect_entities(
        question,
        graph,
        required_tables=metric_formula_tables,
        required_entities=required_entities,
        authoritative_fact_tables=authoritative_fact_tables,
    )
    log.debug("Graph: question=%r detected=%s", question[:60], detected)

    if not detected:
        return empty

    if len(detected) < 2:
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
        from core.join_planner import compile_join_plan
        join_plan = compile_join_plan(
            graph, detected, [],
            metric_formula_tables=metric_formula_tables or (),
            authoritative_fact_tables=authoritative_fact_tables or (),
            anti_join=anti_join,
        )
        return {
            "enabled": True, "detected": detected,
            "join_skeleton": skeleton, "anchor": detected[0],
            "entity_count": entity_count,
            "entities": entities,
            "properties": graph.get("properties") or [],
            "join_plan": join_plan,
            "planning_status": join_plan.get("status", "selected"),
        }

    if not rels:
        # Multiple governed entities were requested, but no governed
        # relationship connects them. A single-table fallback would silently
        # discard part of the question and falsely report a resolved plan.
        return {
            **empty,
            "detected": detected,
            "planning_status": "blocked",
            "missing_entities": detected[1:],
            "reason": (
                "No confirmed relationship connects the requested business entities. "
                "An administrator must confirm the required entity-graph join."
            ),
        }

    entities_map = {e["entity_name"]: e for e in entities}
    from core.join_planner import choose_fact_anchor, compile_join_plan
    preferred_anchor = choose_fact_anchor(
        entities,
        detected,
        metric_formula_tables=metric_formula_tables or (),
        authoritative_fact_tables=authoritative_fact_tables or (),
    )
    ordered_detected = list(detected)
    if preferred_anchor in ordered_detected:
        ordered_detected.remove(preferred_anchor)
        ordered_detected.insert(0, preferred_anchor)
    join_path, path_diagnostics = find_join_path_with_diagnostics(
        ordered_detected,
        graph,
        prefer_fact_anchor=not anti_join,
        anti_join=anti_join,
        question=question,
        selected_edge_ids=selected_edge_ids,
    )

    # ── Near-tied paths are ranked, not escalated ────────────────────────────
    # find_join_path_with_diagnostics has ALREADY chosen the lowest risk-weighted
    # path (and honours selected_edge_ids when the user picked one). Discarding
    # that answer to ask the user was wrong twice over: a business user cannot
    # know which physical join path to take, and the question was asked per
    # AMBIGUOUS TARGET — so answering it only resolved the first, and the next
    # target dead-ended with the same message and no way forward.
    #
    # A near-tie is not an absence of evidence. The ranking already weighs
    # confirmed over suggested, validated over untested, and penalises fan-out
    # and broken edges; ties break on a stable ordering, so the same question
    # resolves the same way every time. Take that path, record what was chosen
    # and what it was chosen over, and let the caller disclose it.
    #
    # Genuinely unresolvable cases below (nothing reachable, no anchor, raw
    # fact-to-fact) still block — those are absences of governance, not ties.
    ambiguity_note: dict = {}
    if path_diagnostics.get("ambiguous_targets"):
        ambiguity = path_diagnostics["ambiguous_targets"][0]
        options = list(ambiguity.get("options") or [])
        chosen_label = str((options[0] or {}).get("label") or "") if options else ""
        alternatives = [
            str(option.get("label") or "") for option in options[1:] if option
        ]
        ambiguity_note = {
            "target": ambiguity.get("target"),
            "chosen": chosen_label,
            "alternatives": alternatives,
            "options": options,
        }
        log.info(
            "Graph: %d near-tied paths reach %s — using the top-ranked "
            "relationship %r over %s rather than asking the user",
            len(options), ambiguity.get("target"),
            chosen_label or "(unlabelled)", alternatives or "no alternative",
        )

    reached_entities = {
        entity_name
        for edge in join_path
        for entity_name in (edge.get("from_entity"), edge.get("to_entity"))
        if entity_name
    }
    reached_entities.add(ordered_detected[0])
    missing_entities = [
        entity_name for entity_name in detected if entity_name not in reached_entities
    ]
    missing_entities.extend(
        entity_name
        for entity_name in (path_diagnostics.get("unreachable") or [])
        if entity_name not in missing_entities
    )
    if missing_entities:
        return _unresolved_result(
            detected,
            status="blocked",
            reason=(
                "The confirmed entity graph cannot reach every business entity "
                f"required by this question: {', '.join(missing_entities)}."
            ),
            missing_entities=missing_entities,
        )

    # Determine anchor (fact table if possible)
    anchor = detected[0]
    if anti_join:
        anchor = _anti_join_anchor(question, detected, entities_map)
    else:
        anchor = preferred_anchor or anchor

    skeleton = build_join_skeleton(join_path, entities_map, anchor, db_type, anti_join=anti_join)
    fanout_risk_facts = _multi_fact_fanout_risk(join_path, entities_map)
    join_plan = compile_join_plan(
        graph,
        detected,
        join_path,
        metric_formula_tables=metric_formula_tables or (),
        authoritative_fact_tables=authoritative_fact_tables or (),
        anti_join=anti_join,
    )
    planning_status = join_plan.get("status", "selected")
    if planning_status in {"blocked", "requires_isolated_aggregation", "clarification_required"}:
        # Never hand a raw multi-fact/invalid skeleton to the LLM.  The
        # structured plan below supplies the safe isolated-aggregation
        # contract instead.
        skeleton = ""
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
            "status": edge.get("status", ""),
            "cardinality": edge.get("cardinality", ""),
            "fanout_ratio": edge.get("fanout_ratio"),
            "orphan_rate": edge.get("orphan_rate"),
            "many_to_many": bool(edge.get("many_to_many")),
            "direction": edge.get("_direction", "forward"),
            "weight": round(_edge_weight(edge), 3),
        })

    return {
        "enabled":       bool(skeleton) or planning_status in {
            "blocked", "requires_isolated_aggregation", "clarification_required",
        },
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
        "join_plan":     join_plan,
        "planning_status": planning_status,
        # Present only when near-tied paths were ranked rather than escalated.
        # The caller discloses it; it never blocks the answer.
        "ranked_relationship": ambiguity_note,
    }


def resolve_for_question(
    question: str,
    account_id: str,
    db_type: str,
    graph: Optional[dict] = None,
    intent: Optional[dict] = None,
    required_entities: Optional[list[str] | set[str]] = None,
    metric_formula_tables: Optional[list[str] | set[str]] = None,
    authoritative_fact_tables: Optional[list[str] | set[str]] = None,
    use_suggested: Optional[bool] = None,
    selected_edge_ids: Optional[list] = None,
) -> dict:
    """
    Main entry point called before SQL generation.

    Resolution order:
      1. Admin-confirmed (or legacy status-less) rows only.
      2. Admin diagnostics may explicitly preview suggested rows. Rejected
         rows never participate in either normal or diagnostic planning.

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
    reviewable = _reviewable_subgraph(graph)
    has_suggested = any(
        str(row.get("status") or "").strip().lower() == "suggested"
        for row in [
            *(reviewable.get("entities") or []),
            *(reviewable.get("relationships") or []),
        ]
    )

    confirmed_result: dict = {}
    if confirmed["entities"]:
        confirmed_result = _resolve_on_graph(
            question, db_type, confirmed, intent,
            required_entities, metric_formula_tables, authoritative_fact_tables,
            selected_edge_ids,
        )
        # A confirmed multi-entity join — or a graph with nothing suggested —
        # is final. A confirmed single-entity anchor is retained below unless
        # graph-review tooling explicitly forces suggested relationships.
        if confirmed_result.get("enabled") and (
            len(confirmed_result.get("detected", [])) >= 2 or not has_suggested
        ):
            confirmed_result["entity_count"] = entity_count
            confirmed_result["graph_scope"] = "confirmed"
            return confirmed_result

    if has_suggested:
        # Suggested rows are review evidence, not executable governance.  The
        # explicit override is reserved for admin diagnostics and previews;
        # normal SQL generation passes None and can use confirmed rows only.
        allow = use_suggested is True
        # A confirmed anchor is an executable governed plan. Do not replace it
        # merely because suggested rows can produce a larger join: "more
        # detected entities" is not evidence that the extra tables are needed,
        # and was the source of silent aggregate fan-out. The explicit function
        # override remains available to graph-review tooling.
        force_suggested = use_suggested is True
        if allow and (force_suggested or not confirmed_result.get("enabled")):
            full_result = _resolve_on_graph(
                question, db_type, reviewable, intent,
                required_entities, metric_formula_tables, authoritative_fact_tables,
                selected_edge_ids,
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
                "(review-only; no explicit diagnostic override)", question[:60],
            )

    if confirmed_result.get("enabled"):
        confirmed_result["entity_count"] = entity_count
        confirmed_result["graph_scope"] = "confirmed"
        return confirmed_result

    result = {**empty, "entity_count": entity_count}
    if has_suggested:
        # Preserve enough current graph state for consumers to distinguish
        # "there is no graph" from "the graph exists but is still review-only".
        # In particular, cached SQL created before governance was tightened
        # must not be reused merely because executable detection is empty.
        result.update({
            "entities": entities,
            "graph_scope": "review_only",
            "review_only": True,
        })
    return result
