"""Business names from the AI: the one step of learning that reads like a person.

The AI is shown, per table, what the data already settled (kind, grain, keys,
joins, dates, measures and how they add up) plus each column's profile, and the
most common values of short code and name columns only where those may be
shown. It returns names, descriptions and synonyms, and nothing else: it cannot
make a key of a non-unique column, a measure of a code, or change how anything
adds up. Whatever it does not return keeps the reading of its name.

The caller provides ``complete(system, user) -> text``; production wraps the
workspace's configured model with its audit scope, tests pass a stub.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from core2.bootstrap import names
from core2.model.schema import SemanticModel

log = logging.getLogger("querybot.core2")

Complete = Callable[[str, str], str]

SYSTEM = """You name the parts of a company's data warehouse in plain business words.

For each table you receive what was already established from the data itself: what kind of table it is,
what one row means, its keys, joins, dates and measures, and a profile of every column (how many distinct
values, how many are empty, ranges, and for short code columns their most common values when allowed).
Names may be abbreviations, codes or meaningless (C07); use the profile and the context to read them.

Reply with one JSON object, no markdown, in exactly this shape:
{"tables": {"<table>": {
   "name": "short business name, singular (Invoice line)",
   "description": "one sentence: what the table records",
   "grain": "one row per ...",
   "columns": {"<column>": {"name": "business name", "description": "one sentence",
                            "synonyms_en": ["..."], "synonyms_fr": ["..."], "unit": "currency code or unit, or empty"}}
}}}

Rules: keep every name short (1 to 4 words); never invent facts the profile does not support; when unsure of a
column, give the most likely reading and say "probably" in its description; do not rename keys you cannot read
(leave them out); synonyms are words business people use for the same thing, at most 4 per language."""


def table_brief(model: SemanticModel, table_key: str) -> dict[str, Any]:
    """What the AI sees about one table: established facts and profiles, values only where allowed."""
    table = model.tables[table_key]
    joins = [f"{model.columns[j.from_columns[0]].name} -> {model.tables[j.to_table].name}"
             + (f" (role: {j.role})" if j.role else "")
             for j in model.joins.values() if j.from_table == table_key]
    dates = [f"{model.columns[r.column].name}: {r.kind}{', default' if r.is_default else ''}"
             for r in model.date_roles.values() if r.table == table_key]
    measures = []
    for m in model.measures.values():
        if m.table != table_key:
            continue
        column_key = getattr(m.expr, "column", None)
        measures.append(f"{getattr(m.expr, 'agg', '')}({model.columns[column_key].name if column_key else '*'}) "
                        f"{m.additivity.replace('_', '-')}")
    columns = []
    for key in table.columns:
        column = model.columns[key]
        p = column.profile
        entry: dict[str, Any] = {"column": column.name, "type": column.data_type, "role": column.role}
        if p:
            entry["distinct"] = p.distinct
            if p.rows:
                entry["empty"] = f"{1 - p.non_null / p.rows:.0%}"
            if column.data_type in ("integer", "decimal", "float", "date", "timestamp") and p.min is not None:
                entry["range"] = [p.min, p.max]
            if column.values_allowed and p.top and column.data_type in ("text", "integer", "boolean"):
                entry["common_values"] = [t.value for t in p.top[:8]]
            if p.avg_len:
                entry["avg_length"] = round(p.avg_len, 1)
        columns.append(entry)
    return {"table": table.name, "kind": table.kind, "rows": table.row_count, "grain": table.grain_text,
            "joins": joins, "dates": dates, "measures": measures, "columns": columns}


def _parse(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        value = json.loads(cleaned[start:end + 1]) if 0 <= start < end else {}
    return value if isinstance(value, dict) else {}


def _short(text: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def label(model: SemanticModel, complete: Complete, *, batch: int = 4) -> list[str]:
    """Ask for names, table by table (``batch`` tables per call); returns notes on what failed."""
    notes: list[str] = []
    keys = [k for k, t in model.tables.items() if not t.hidden]
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        briefs = [table_brief(model, k) for k in chunk]
        user = "Tables:\n" + json.dumps(briefs, ensure_ascii=False, default=str)
        try:
            answer = _parse(complete(SYSTEM, user))
        except Exception as exc:  # noqa: BLE001 - names stay as read from the data; the page says so
            log.warning("core2 labels: the AI could not name %s: %s", [model.tables[k].name for k in chunk], exc)
            notes.append(f"Names for {len(chunk)} table(s) are read from their column names: the AI call failed.")
            continue
        found = answer.get("tables")
        named: dict[str, Any] = found if isinstance(found, dict) else {}
        for key in chunk:
            _apply(model, key, named.get(model.tables[key].name) or {})
    return notes


def _apply(model: SemanticModel, table_key: str, reply: dict[str, Any]) -> None:
    if not isinstance(reply, dict):
        return
    table = model.tables[table_key]
    if reply.get("name") and table.provenance != "admin":
        table.business_name = _short(reply["name"], 60)
        table.provenance = "ai"
    if reply.get("description"):
        table.description = _short(reply["description"], 300)
    if reply.get("grain"):
        table.grain_text = _short(reply["grain"], 120)
    by_name = {model.columns[k].name: model.columns[k] for k in table.columns}
    found = reply.get("columns")
    columns: dict[str, Any] = found if isinstance(found, dict) else {}
    for name, info in columns.items():
        column = by_name.get(name)
        if column is None or not isinstance(info, dict) or column.provenance == "admin":
            continue
        if info.get("name"):
            column.business_name = _short(info["name"], 60)
            column.provenance = "ai"
        if info.get("description"):
            column.description = _short(info["description"], 300)
        synonyms = {lang: [_short(s, 40) for s in (info.get(f"synonyms_{lang}") or [])[:4] if s]
                    for lang in ("en", "fr")}
        column.synonyms = {lang: words for lang, words in synonyms.items() if words}
        if info.get("unit") and not column.unit:
            column.unit = _short(info["unit"], 12)
    # Measures, date roles and entities take their names from the columns they rest on.
    for measure in model.measures.values():
        column_key = getattr(measure.expr, "column", None)
        if measure.table != table_key or measure.provenance == "admin":
            continue
        agg = getattr(measure.expr, "agg", "")
        if column_key and agg in ("sum", "avg") and model.columns[column_key].provenance == "ai":
            measure.business_name = model.columns[column_key].business_name
            measure.synonyms = dict(model.columns[column_key].synonyms)
        elif agg == "count" and table.provenance == "ai":
            measure.business_name = f"Number of {names.plural(table.business_name.lower())}"
        elif column_key and agg == "count_distinct" and model.columns[column_key].provenance == "ai":
            thing = re.sub(r"\s+(number|no\.?|id|key|code|reference)$", "", model.columns[column_key].business_name,
                           flags=re.I)
            measure.business_name = f"Number of {names.plural(thing.lower())}"
    for role in model.date_roles.values():
        column = model.columns[role.column]
        if role.table == table_key and role.provenance != "admin" and column.provenance == "ai" and not (
                role.calendar_join and model.joins[role.calendar_join].role):
            role.name = column.business_name
    for entity in model.entities.values():
        if entity.table == table_key and entity.provenance != "admin" and table.provenance == "ai":
            entity.business_name = table.business_name
