"""
core/table_description_author.py

Propose a business description and the terms that go with it, for one table.

The panel that collects these shipped empty and stayed empty -- one table
described out of fourteen -- because writing fourteen paragraphs cold is the
kind of task nobody gets to. The model has better inputs for it than the admin
does anyway: it can see the column names, the table's role and grain, what it
joins to, and the KB document already written about it. "One row per item per
warehouse per period" is derivable from the grain; a person would have to go
and look it up.

NOTHING HERE TAKES EFFECT. A proposal is written to the suggestion columns and
waits for a human, exactly as a metric proposal does. That is not politeness:
table terms decide which table a question reaches and column terms decide which
measure it resolves, so an auto-applied term moves answers silently -- the same
failure class as every bug this module was written after.

The model is given names and structure only. No sampled values are sent, so
this needs no value-egress budget and is safe for a regulated tenant.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("querybot.table_description_author")

_MAX_COLUMNS = 60
_MAX_JOINS = 12

_SYSTEM = """You document tables in a business intelligence semantic layer.

Given one table's structure, produce THREE things:

1. description — two or three sentences a business person would recognise.
   Say what one row IS (the grain), and any rule about how the figures may be
   used. Snapshot facts matter most: if the table holds a balance measured at
   a point in time, say so and say it must not be summed across periods.

2. synonyms — words a business user would use for the TABLE itself, when
   asking a question. These route a question to this table.

3. column_terms — for MEASURE columns only, the phrases a user would say to
   ask for that figure. These are what let a question be answered rather than
   merely routed. Prefer the phrasing a person uses out loud ("inventory
   value") over the column name.

RULES
- Use only the evidence given. Never invent a column that is not listed.
- Terms must be things a user would SAY, not restatements of the column name.
- Do not propose terms for key columns, foreign keys, or audit timestamps.
- A generic word on its own ("value", "amount", "total") is useless as a term
  because every table has one; qualify it ("inventory value").
- If the evidence is too thin to be useful, return empty lists rather than
  guessing. A wrong term sends questions to the wrong number.

Return ONLY minified JSON:
{"description": "...", "synonyms": ["..."], "column_terms": {"COLUMN": ["..."]}}"""


def _clean_terms(raw: Any, limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in (raw or []) if isinstance(raw, (list, tuple)) else []:
        term = " ".join(str(item).split()).strip().strip(",;")
        if not term or len(term) > 60:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out[:limit]


def build_evidence(
    table_fqn: str,
    columns: list[dict],
    *,
    entity_type: str = "",
    fact_type: str = "",
    grain: str = "",
    joins: list[dict] | None = None,
    kb_excerpt: str = "",
) -> str:
    """Assemble the structural facts the model is allowed to reason from."""
    lines = [f"TABLE: {table_fqn}"]
    if entity_type:
        lines.append(f"ROLE: {entity_type}" + (f" ({fact_type})" if fact_type else ""))
    if grain:
        lines.append(f"GRAIN: {grain}")

    lines.append("\nCOLUMNS:")
    for col in (columns or [])[:_MAX_COLUMNS]:
        name = str(col.get("name") or col.get("column") or "").strip()
        if not name:
            continue
        ctype = str(col.get("type") or "").strip()
        # The aggregation verdict is the single most useful hint for a snapshot
        # table, and it is the thing the description most needs to state.
        aggregation = str(col.get("aggregation") or col.get("aggregation_semantics") or "").strip()
        bits = [f"  {name}"]
        if ctype:
            bits.append(f"({ctype})")
        if aggregation:
            bits.append(f"[{aggregation}]")
        lines.append(" ".join(bits))

    if joins:
        lines.append("\nJOINS TO:")
        for join in joins[:_MAX_JOINS]:
            target = str(join.get("to_entity") or join.get("to_table") or "").strip()
            if target:
                lines.append(f"  {target}")

    if kb_excerpt:
        lines.append("\nEXISTING KNOWLEDGE BASE NOTES:\n" + kb_excerpt[:1200])
    return "\n".join(lines)


def parse_proposal(raw: str, known_columns: set[str]) -> dict[str, Any]:
    """Parse the model's reply, discarding anything not grounded in the schema.

    A term on a column that does not exist cannot help and can only confuse a
    later reader, so unknown columns are dropped rather than stored and
    puzzled over.
    """
    text = str(raw or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {"description": "", "synonyms": [], "column_terms": {}}
    try:
        data = json.loads(match.group(0))
    except Exception:
        log.debug("table description proposal was not JSON: %r", text[:200])
        return {"description": "", "synonyms": [], "column_terms": {}}

    upper = {c.upper() for c in known_columns}
    column_terms: dict[str, list[str]] = {}
    for column, terms in (data.get("column_terms") or {}).items():
        name = re.sub(r"[^A-Za-z0-9_]", "", str(column)).upper()
        if name not in upper:
            log.debug("dropping proposed terms for unknown column %r", column)
            continue
        cleaned = _clean_terms(terms)
        if cleaned:
            column_terms[name] = cleaned

    description = " ".join(str(data.get("description") or "").split())[:1000]
    return {
        "description": description,
        "synonyms": _clean_terms(data.get("synonyms")),
        "column_terms": column_terms,
    }


def format_column_terms(column_terms: dict[str, list[str]]) -> str:
    """Render as the `COLUMN = term, term` form the admin edits."""
    return "\n".join(
        f"{column} = {', '.join(terms)}"
        for column, terms in sorted((column_terms or {}).items())
        if terms
    )
