"""The "What QueryBot learned" page, as plain data (the template only lays it out).

Everything is said in business words with its evidence beside it: why a table is
a fact, why a date is the default, how a measure adds up, how well a join
matched. Raw keys appear only where an admin decision has to name its object.
"""

from __future__ import annotations

import json
from typing import Any

from core2.model.overrides import target
from core2.model.schema import SemanticModel

_KIND_ORDER = {"fact": 0, "snapshot": 1, "dimension": 2, "bridge": 3, "calendar": 4, "other": 5}
KIND_WORDS = {
    "fact": "Events", "snapshot": "Snapshots", "dimension": "Things you group by", "bridge": "Links",
    "calendar": "Calendar", "other": "Other tables",
}
ADDS_UP = {
    ("additive", None): "summed",
    ("semi_additive", "last"): "taken at the end of each period",
    ("semi_additive", "average"): "averaged over time",
    ("non_additive", None): "averaged, never summed",
}
# How an admin may say a total adds up over time (a choice on the page): summed, or a level.
ADDS_UP_CHOICES = {"summed": ("additive", None), "last": ("semi_additive", "last"),
                   "average": ("semi_additive", "average")}
TRUST_WORDS = {"admin": "approved by an admin", "verified": "verified by the data", "declared": "declared by the database",
               "proposed": "proposed: used only when nothing better exists", "rejected": "rejected"}


def _evidence(items: list[Any]) -> list[str]:
    return [e.detail for e in items if getattr(e, "detail", "") and getattr(e, "kind", "") != "base"]


def _adds_up(m: Any) -> str:
    agg = getattr(m.expr, "agg", None)
    if agg == "count":
        return "counted"
    if agg == "count_distinct":
        return "each counted once"
    return ADDS_UP.get((m.additivity, m.time_aggregation), m.additivity.replace("_", " "))


def _unit(m: Any, model: SemanticModel) -> str:
    if m.unit:
        return m.unit
    if m.unit_column:
        flagged = next((q for q in model.quality if q.kind == "unit_mix" and q.object == getattr(m.expr, "column", "")), None)
        units = flagged.data.get("units") if flagged else None
        return "mixed: " + ", ".join(units) if units else "see " + model.columns[m.unit_column].name
    return ""


def learned_view(model: SemanticModel) -> dict[str, Any]:
    tables = sorted(model.tables.values(), key=lambda t: (_KIND_ORDER.get(t.kind, 9), t.business_name.lower()))
    joins_from: dict[str, list[dict[str, Any]]] = {}
    joins: list[dict[str, Any]] = []
    for j in sorted(model.joins.values(), key=lambda j: (j.from_table, j.key)):
        row = {
            "key": j.key, "target": target("join", j.key),
            "from": model.tables[j.from_table].business_name,
            "from_column": model.columns[j.from_columns[0]].name,
            "to": model.tables[j.to_table].business_name, "to_column": model.columns[j.to_columns[0]].name,
            "role": j.role or "", "trust": j.trust, "trust_words": TRUST_WORDS.get(j.trust, j.trust),
            "match": f"{j.match_rate:.1%}", "to_calendar": j.to_calendar, "evidence": _evidence(j.evidence),
        }
        joins.append(row)
        joins_from.setdefault(j.from_table, []).append(row)

    table_rows = []
    for t in tables:
        roles = sorted((r for r in model.date_roles.values() if r.table == t.key),
                       key=lambda r: (not r.is_default, -r.score))
        measures = sorted((m for m in model.measures.values() if m.table == t.key), key=lambda m: m.business_name)
        table_rows.append({
            "key": t.key, "target": target("table", t.key), "name": t.business_name, "physical": t.name,
            "kind": t.kind, "kind_words": KIND_WORDS.get(t.kind, t.kind), "rows": t.row_count, "grain": t.grain_text,
            "evidence": _evidence(t.evidence),
            "dates": [{
                "key": r.key, "target": target("date_role", r.key), "name": r.name, "column": model.columns[r.column].name,
                "kind": r.kind, "default": r.is_default, "coverage": f"{r.coverage:.0%}",
                "range": f"{r.first or '?'} to {r.last or '?'}" if (r.first or r.last) else "",
                "evidence": _evidence(r.evidence), "status": r.status,
            } for r in roles],
            "measures": [{
                "key": m.key, "target": target("measure", m.key), "name": m.business_name,
                "adds_up": _adds_up(m), "synonyms": ", ".join(sorted({w for ws in m.synonyms.values() for w in ws})),
                "adds_up_value": next((k for k, v in ADDS_UP_CHOICES.items()
                                       if v == (m.additivity, m.time_aggregation)), ""),
                "adds_up_options": [(k, ADDS_UP[v]) for k, v in ADDS_UP_CHOICES.items()]
                if getattr(m.expr, "agg", None) == "sum" else [],
                "format": m.format, "unit": _unit(m, model), "status": m.status, "evidence": _evidence(m.evidence),
            } for m in measures],
            "joins": joins_from.get(t.key, []),
            "columns": [{
                "key": c.key, "target": target("column", c.key), "name": c.name, "business_name": c.business_name,
                "role": c.role, "type": c.data_type,
                "filled": f"{(c.profile.non_null / c.profile.rows):.0%}" if c.profile and c.profile.rows else "",
                "distinct": c.profile.distinct if c.profile else 0,
            } for c in (model.columns[k] for k in t.columns if k in model.columns)],
        })

    calendars = [{
        "table": model.tables[c.table].business_name, "physical": model.tables[c.table].name,
        "date": model.columns[c.date_column].name if c.date_column else "",
        "key": model.columns[c.key_column].name if c.key_column else "",
        "grain": c.grain,
        "range": f"{c.first_date} to {c.last_date}" if c.grain == "day" else
                 f"{c.first_date:%b %Y} to {c.last_date:%b %Y}" if c.first_date and c.last_date else "",
        "attributes": [(a.replace("_", " "), model.columns[col].name) for a, col in sorted(c.attributes.items())],
        "fiscal": (f"Fiscal years start in month {c.fiscal_year_start_month}, named by the year they "
                   f"{c.fiscal_year_named_by}") if c.fiscal_year_start_month else "",
        "placeholders": ", ".join(str(p) for p in c.placeholders),
    } for c in model.calendars.values()]

    kinds: dict[str, int] = {}
    for t in tables:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    return {
        "version": model.version, "built_at": model.built_at,
        "summary": {
            "tables": len(tables), "kinds": [(KIND_WORDS.get(k, k), n) for k, n in
                                             sorted(kinds.items(), key=lambda kv: _KIND_ORDER.get(kv[0], 9))],
            "joins": len(joins), "joins_trusted": sum(1 for j in model.joins.values()
                                                      if j.trust in ("admin", "verified", "declared")),
            "dates": len(model.date_roles), "measures": len(model.measures), "quality": len(model.quality),
            "review": len(model.review),
        },
        "tables": table_rows,
        "joins": joins,
        "calendars": calendars,
        "quality": [{"message": q.message, "severity": q.severity, "kind": q.kind.replace("_", " ")}
                    for q in sorted(model.quality, key=lambda q: (q.severity != "warning", q.kind))],
        "review": [{"question": r.question, "choice": r.choice_made, "alternatives": r.alternatives,
                    "evidence": _evidence(r.evidence)[:4], "object": r.object, **_leave_out(model, r.key, r.object)}
                   for r in model.review],
        "notes": list(model.notes),
    }


def _leave_out(model: SemanticModel, key: str, column: str) -> dict[str, Any]:
    """For a status column with cancel-like codes: the decision that leaves those rows out of the table's
    totals (a default filter, undone like any decision), or what was already decided."""
    if not key.startswith("default_filter:") or column not in model.columns:
        return {}
    flag = next((q for q in model.quality if q.kind == "status_column" and q.object == column), None)
    codes = [str(c) for c in (flag.data.get("cancel_like") or [])] if flag else []
    table = model.tables[model.columns[column].table]
    if any(f.column == column for f in table.default_filters):
        decided = next(f for f in table.default_filters if f.column == column)
        return {"decided": f"Left out: {', '.join(map(str, decided.values))}"}
    if not codes:
        return {}
    filters = [f.model_dump(mode="json") for f in table.default_filters]
    filters.append({"column": column, "op": "not_in", "values": codes})
    return {"action": {"target": target("table", table.key), "field": "default_filters",
                       "value": json.dumps(filters), "label": f"Leave out {', '.join(codes)}",
                       "note": f"{table.business_name}: rows whose {model.columns[column].business_name.lower()} "
                               f"is {' or '.join(codes)} are left out of totals"}}
