"""
core/graph_autopopulate.py — KB-aware entity graph auto-population.

After every KB build the entity graph is synchronised automatically:

  1. prune    — drop graph rows for tables no longer in the schema scope
  2. populate — heuristic entities/relationships (provenance-tagged,
                status='suggested', insert-only — admin rows untouched)
  3. enrich   — harvest the KB markdown the build just paid for:
                ## Overview     → entity description
                ## Business Synonyms → entity_properties synonyms
                ## Key Metrics  → entity_properties metric roles
  4. layout   — fact-centric auto-layout for suggested (unreviewed) nodes;
                admin-confirmed nodes never move

The same heuristics also run after schema discovery (admin/routes.py
delegates here) so the graph exists even before the first KB build.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import store

log = logging.getLogger("querybot.graph_autopopulate")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — heuristic population (moved from admin/routes.py)
# ══════════════════════════════════════════════════════════════════════════════

def auto_populate_from_schema(account_id: str, schema_dir: str) -> tuple[int, int]:
    """
    Parse _schema.json and insert auto-detected entities/relationships with
    status='suggested'. Only inserts rows that don't already exist — never
    overwrites confirmed admin edits.

    Returns (entities_added, relationships_added).
    """
    from core.schema import build_entity_graph_from_schema

    graph_data = build_entity_graph_from_schema(schema_dir)
    if not graph_data["entities"]:
        return 0, 0

    existing_entities = {e["entity_name"] for e in store.list_entities(account_id, active_only=False)}
    ent_added = 0
    for ent in graph_data["entities"]:
        if ent["entity_name"] in existing_entities:
            continue
        store.save_entity(
            account_id       = account_id,
            entity_name      = ent["entity_name"],
            table_name       = ent["table_name"],
            schema_name      = ent.get("schema_name", ""),
            pk_column        = ent.get("pk_column", ""),
            display_name     = ent.get("display_name", ""),
            description      = ent.get("description", ""),
            entity_type      = ent.get("entity_type", "dimension"),
            color            = ent.get("color", "#4F86C6"),
            pos_x            = ent.get("pos_x", 120),
            pos_y            = ent.get("pos_y", 120),
            confidence_score = ent.get("confidence_score", 75),
            status           = ent.get("status", "suggested"),
            generated_by     = ent.get("generated_by", "heuristic"),
            reason           = ent.get("reason", ""),
        )
        ent_added += 1

    rel_added = 0
    all_entities = {e["entity_name"] for e in store.list_entities(account_id, active_only=False)}
    for rel in graph_data["relationships"]:
        # Both entities must exist before we can add the relationship
        if rel["from_entity"] not in all_entities or rel["to_entity"] not in all_entities:
            continue
        # upsert enforces one-relationship-per-table-pair and never touches
        # confirmed/manual rows
        store.upsert_relationship_by_pair(
            account_id        = account_id,
            from_entity       = rel["from_entity"],
            to_entity         = rel["to_entity"],
            from_column       = rel["from_column"],
            to_column         = rel["to_column"],
            relationship_type = rel.get("relationship_type", "many_to_one"),
            join_type         = rel.get("join_type", "INNER"),
            label             = rel.get("label", ""),
            confidence_score  = rel.get("confidence_score", 70),
            status            = rel.get("status", "suggested"),
            generated_by      = rel.get("generated_by", "heuristic"),
            reason            = rel.get("reason", ""),
        )
        rel_added += 1

    return ent_added, rel_added


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — KB harvest enrichment
# ══════════════════════════════════════════════════════════════════════════════

def _extract_overview(content: str) -> str:
    """First sentence/paragraph of the ## Overview section of a Stage 1 KB doc."""
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if in_section:
                break
            in_section = stripped.lower().startswith("## overview")
            continue
        if in_section and stripped:
            return stripped.lstrip("-* ").strip()
    return ""


_PLAIN_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")


def enrich_graph_from_kb(account_id: str, kb_dir: str) -> dict:
    """
    Harvest Stage 1 KB markdown into the entity graph.

    - Entity descriptions: filled from ## Overview, only when the entity is
      still 'suggested' AND its description is empty.
    - entity_properties: synonyms from ## Business Synonyms, metric roles
      from ## Key Metrics. Admin-confirmed properties are never touched;
      suggested ones refresh on every build.

    Returns {"descriptions": n, "properties": n}.
    """
    from store.semantic_store import _extract_synonyms, _extract_key_metrics

    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return {"descriptions": 0, "properties": 0}

    entities = store.list_entities(account_id, active_only=False)
    by_table: dict[str, list[dict]] = {}
    for e in entities:
        by_table.setdefault((e.get("table_name") or "").upper(), []).append(e)

    desc_count = 0
    prop_count = 0

    for md_file in sorted(kb_path.glob("*_kb.md")):
        table = md_file.stem[:-3].upper() if md_file.stem.endswith("_kb") else md_file.stem.upper()
        matches = by_table.get(table) or []
        if not matches:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # ── Descriptions ────────────────────────────────────────────────────
        overview = _extract_overview(content)
        if overview:
            with store.get_db() as conn:
                for e in matches:
                    if (e.get("status") or "") != "suggested":
                        continue
                    if (e.get("description") or "").strip():
                        continue
                    conn.execute(
                        "UPDATE entity_graph SET description=? "
                        "WHERE account_id=? AND entity_name=? "
                        "AND (description IS NULL OR description='') AND status='suggested'",
                        (overview[:500], account_id, e["entity_name"]),
                    )
                    desc_count += 1

        # ── Column properties (synonyms + metric roles) ────────────────────
        # Attach to the base entity for the table: prefer the entity literally
        # named after the table over role-playing aliases of it.
        target = next(
            (e for e in matches if e["entity_name"].upper() == table),
            matches[0],
        )
        ename = target["entity_name"]
        existing = {
            (p.get("column_name") or "").upper(): p
            for p in store.list_entity_properties(account_id, ename)
        }

        # Merge both sections into one plan per column so a column that is
        # both a metric and a synonym target gets a single combined save.
        plan: dict[str, dict] = {}

        for syn in _extract_synonyms(content, table):
            col = (syn.get("column") or "").strip("`").strip()
            col = col.split(".")[-1]
            if not _PLAIN_IDENT.fullmatch(col):
                continue
            terms = [syn.get("term") or ""]
            terms += [a for a in (syn.get("aliases") or "").split(",")]
            terms = sorted({t.strip().lower() for t in terms if t.strip()})
            if not terms:
                continue
            entry = plan.setdefault(col, {"role": "", "display_name": "", "synonyms": set()})
            entry["synonyms"].update(terms)

        for met in _extract_key_metrics(content, table):
            expr = (met.get("expression") or "").strip()
            if not _PLAIN_IDENT.fullmatch(expr):
                continue  # only plain column references — skip formulas
            entry = plan.setdefault(expr, {"role": "", "display_name": "", "synonyms": set()})
            entry["role"] = "metric"
            if not entry["display_name"]:
                entry["display_name"] = (met.get("term") or "").strip().title()

        for col, entry in plan.items():
            prev = existing.get(col.upper())
            if prev and (prev.get("status") or "") == "confirmed":
                continue  # admin-confirmed — never touch
            merged_syns = set(entry["synonyms"])
            if prev and (prev.get("synonyms") or "").strip():
                merged_syns.update(
                    s.strip().lower() for s in prev["synonyms"].split(",") if s.strip()
                )
            store.save_entity_property(
                account_id       = account_id,
                entity_name      = ename,
                column_name      = col,
                role             = entry["role"] or (prev or {}).get("role") or "dimension",
                display_name     = entry["display_name"] or (prev or {}).get("display_name") or "",
                synonyms         = ", ".join(sorted(merged_syns)),
                confidence_score = 80,
                status           = "suggested",
                generated_by     = "kb_harvest",
                reason           = f"Harvested from {md_file.name}",
            )
            prop_count += 1

    return {"descriptions": desc_count, "properties": prop_count}


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — fact-centric auto-layout
# ══════════════════════════════════════════════════════════════════════════════

def auto_layout_graph(account_id: str) -> int:
    """
    Position entities on the canvas: facts in a centre row, each dimension on a
    ring around the fact it joins to, orphans in a grid below. Only entities
    with status='suggested' are moved — admin-arranged (confirmed) nodes stay
    exactly where the admin left them.

    Returns the number of entities repositioned.
    """
    entities = store.list_entities(account_id, active_only=False)
    if not entities:
        return 0
    rels = store.list_relationships(account_id, active_only=False)

    movable = {e["entity_name"] for e in entities if (e.get("status") or "") == "suggested"}
    if not movable:
        return 0

    names = {e["entity_name"] for e in entities}
    adj: dict[str, set[str]] = {}
    for r in rels:
        a, b = r.get("from_entity"), r.get("to_entity")
        if a in names and b in names:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

    facts = [e for e in entities if e.get("entity_type") == "fact"]
    dims  = [e for e in entities if e.get("entity_type") != "fact"]
    fact_names = {f["entity_name"] for f in facts}

    FACT_Y       = 420.0
    FACT_SPACING = 560.0
    BASE_RADIUS  = 250.0

    pos: dict[str, tuple[float, float]] = {}

    # Facts: confirmed facts anchor where the admin put them; suggested facts
    # flow left-to-right in a centre row.
    x_cursor = 200.0
    for f in facts:
        name = f["entity_name"]
        if name in movable:
            pos[name] = (x_cursor, FACT_Y)
        else:
            pos[name] = (float(f.get("pos_x") or x_cursor), float(f.get("pos_y") or FACT_Y))
        x_cursor = max(x_cursor, pos[name][0]) + FACT_SPACING

    # Ring slots per fact (count first so angles distribute evenly)
    ring_total: dict[str, int] = {}
    for d in dims:
        if d["entity_name"] not in movable:
            continue
        owners = [n for n in sorted(adj.get(d["entity_name"], ())) if n in fact_names]
        if owners:
            ring_total[owners[0]] = ring_total.get(owners[0], 0) + 1

    ring_index: dict[str, int] = {}
    orphan_i = 0
    max_fact_x = max((p[0] for p in pos.values()), default=200.0)

    for d in dims:
        name = d["entity_name"]
        if name not in movable:
            pos[name] = (float(d.get("pos_x") or 120), float(d.get("pos_y") or 120))
            continue
        owners = [n for n in sorted(adj.get(name, ())) if n in fact_names]
        if owners:
            owner = owners[0]
            i = ring_index.get(owner, 0)
            n = max(ring_total.get(owner, 1), 1)
            ring_index[owner] = i + 1
            radius = BASE_RADIUS + (60.0 if n > 8 else 0.0)
            angle = (i / n) * 2.0 * math.pi - math.pi / 2.0
            ox, oy = pos.get(owner, (600.0, FACT_Y))
            pos[name] = (ox + radius * math.cos(angle), oy + radius * math.sin(angle))
        else:
            # Orphans (no join to any fact): grid below the fact row
            pos[name] = (
                200.0 + (orphan_i % 6) * 240.0,
                FACT_Y + BASE_RADIUS + 260.0 + (orphan_i // 6) * 170.0,
            )
            orphan_i += 1

    moved = 0
    with store.get_db() as conn:
        for name in movable:
            if name not in pos:
                continue
            x, y = pos[name]
            conn.execute(
                "UPDATE entity_graph SET pos_x=?, pos_y=? WHERE account_id=? AND entity_name=?",
                (round(x, 1), round(y, 1), account_id, name),
            )
            moved += 1
    log.info("Auto-layout positioned %d suggested entities for %s (%d facts)",
             moved, account_id, len(facts))
    return moved


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — runs after every KB build
# ══════════════════════════════════════════════════════════════════════════════

def sync_graph_after_kb_build(account_id: str, schema_dir: str, kb_dir: str) -> dict:
    """
    Full post-KB-build semantic model sync: prune → populate → enrich → layout.
    Each step is independent — a failure in one is logged and the rest proceed.

    Returns a summary dict for the admin notification.
    """
    summary = {
        "entities_added":        0,
        "relationships_added":   0,
        "descriptions_enriched": 0,
        "properties_enriched":   0,
        "entities_positioned":   0,
        "entities_total":        0,
        "relationships_total":   0,
        "pending_review":        0,
    }

    # 1. Prune to the current schema scope
    try:
        schema_path = Path(schema_dir) / "_schema.json"
        if schema_path.exists():
            scope = list(json.loads(schema_path.read_text(encoding="utf-8")).keys())
            if scope:
                pruned = store.prune_entity_graph_to_tables(account_id, scope)
                if any(pruned.values()):
                    log.info("Graph pruned to KB scope for %s: %s", account_id, pruned)
    except Exception as exc:
        log.warning("Graph prune failed for %s: %s", account_id, exc)

    # 2. Heuristic population
    try:
        ent, rel = auto_populate_from_schema(account_id, schema_dir)
        summary["entities_added"] = ent
        summary["relationships_added"] = rel
    except Exception as exc:
        log.warning("Graph heuristic populate failed for %s: %s", account_id, exc)

    # 3. KB enrichment
    try:
        enriched = enrich_graph_from_kb(account_id, kb_dir)
        summary["descriptions_enriched"] = enriched["descriptions"]
        summary["properties_enriched"]   = enriched["properties"]
    except Exception as exc:
        log.warning("Graph KB enrichment failed for %s: %s", account_id, exc)

    # 4. Auto-layout
    try:
        summary["entities_positioned"] = auto_layout_graph(account_id)
    except Exception as exc:
        log.warning("Graph auto-layout failed for %s: %s", account_id, exc)

    try:
        entities = store.list_entities(account_id, active_only=False)
        rels     = store.list_relationships(account_id, active_only=False)
        summary["entities_total"]      = len(entities)
        summary["relationships_total"] = len(rels)
        summary["pending_review"] = (
            sum(1 for e in entities if (e.get("status") or "") == "suggested")
            + sum(1 for r in rels if (r.get("status") or "") == "suggested")
        )
    except Exception:
        pass

    log.info("Graph sync after KB build for %s: %s", account_id, summary)
    return summary
