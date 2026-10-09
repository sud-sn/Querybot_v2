"""Readers' requests: what they may suggest about the data, and what accepting it changes.

A reader sees what the data covers (the Data guide): each subject and what one row of it
is, its metrics (what each means, how it is counted, the date it is counted by, the other
names people use), what they break down by and what their codes mean ("C" is Cancelled),
and the dates: the admin's Knowledge base, as the reader may see it. Any of it can be
suggested a change: a better meaning, another name people use, a name for a code, another
date to count a metric by, or how a metric should be counted (words for the admin, who
changes the formula). A metric that is not there can be described.

A suggestion is checked here against the model as the reader may see it (their tables,
nothing hidden or sensitive), and turned into the admin decisions accepting it writes
(core2.model.overrides): one per field, so an admin can accept it as sent or edit it first,
and every decision can be undone. Other names are added to the words already there when
the request is accepted, never written over them.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from core2.answer.describe import _measure_sentence, _Reader, _span
from core2.model.schema import Attribute, DateRole, Entity, Measure, SemanticModel

MAX_NAMES = 10
MAX_CODE_NAME = 60
MAX_MEANING = 500
MAX_NOTE = 1000
MAX_EXAMPLE = 200
BREAKDOWNS_SHOWN = 40


class RequestError(ValueError):
    """A suggestion that cannot be sent as it is; the message is the reader's."""


def _words(synonyms: dict[str, list[str]]) -> list[str]:
    return list(dict.fromkeys(w for ws in synonyms.values() for w in ws if w))


def _visible_column(model: SemanticModel, column_key: str | None) -> bool:
    column = model.columns.get(column_key or "")
    return column is not None and not column.hidden and column.sensitivity == "none"


def _codes(model: SemanticModel, column_key: str | None) -> list[dict[str, Any]]:
    """A field's codes and what each means ("C" is Cancelled), when its values are codes a reader may read:
    a text field of few values, all known, short (or named by the admin already)."""
    from core2.model.knowledge import namable

    column = model.columns.get(column_key or "")
    if column is None or column.hidden or column.sensitivity != "none":
        return []
    if any(column.key in (e.code_column, e.label_column, *e.key_columns) for e in model.entities.values()):
        return []            # a store's code is the store: it is named by the store's name, not here
    values = namable(column)
    if not values or not (column.value_names or all(len(v["code"]) <= 4 for v in values)):
        return []
    return [{"code": v["code"], "name": v["name"]} for v in values]


# ── what the reader sees ─────────────────────────────────────────────────────

def reader_view(model: SemanticModel, *, today: dt.date, allowed: set[str] | None = None,
                values: bool = True) -> dict[str, Any]:
    """The data as this reader may ask about it, laid out to be read and suggested on."""
    reader = _Reader(model, allowed, values, today)
    subjects = []
    for table in reader.subjects():
        dates = reader.dates(table)
        first = dates[0] if dates else None
        metrics = []
        for m in reader.measures(table):
            wanted = m.default_date or model.tables[table].default_date
            date = next((r for r in dates if r.key == wanted), first)
            metrics.append({
                "key": m.key, "name": m.business_name, "meaning": m.description,
                "how": _measure_sentence(reader, m), "date_key": date.key if date else "",
                "date": date.name if date else "", "also": _words(m.synonyms),
                "by_admin": m.provenance == "admin" and m.kind in ("metric", "import"),
                "format": m.format})
        breakdowns = []
        for e in reader.breakdowns(table):
            label = model.columns.get(e.label_column or "")
            breakdowns.append({"kind": "entity", "key": e.slug, "name": e.business_name,
                               "meaning": label.description if label is not None else "", "also": _words(e.synonyms)})
        reached = {t for t in model.tables if reader.may(t) and reader.steps(table, t) is not None}
        for a in sorted(model.attributes.values(), key=lambda a: a.business_name.casefold()):
            column = model.columns.get(a.column)
            if column is None or column.table not in reached or not _visible_column(model, a.column):
                continue
            entity = model.entities.get(a.entity or "")
            if entity is not None and entity.label_column == a.column:
                continue        # the entity itself, listed above
            breakdowns.append({"kind": "attribute", "key": a.slug, "name": a.business_name,
                               "meaning": column.description, "also": _words(a.synonyms),
                               # Codes are member values: shown only where this reader may see values.
                               "codes": _codes(model, a.column) if reader.values else []})
        seen: set[str] = set()
        unique = []
        for b in breakdowns:
            if b["name"].casefold() not in seen:
                seen.add(b["name"].casefold())
                unique.append(b)
        subjects.append({
            "key": table, "name": reader.name(table), "grain": model.tables[table].grain_text or "",
            "meaning": model.tables[table].description or "",
            "span": _span(first.first, first.last, first.granularity) if first else "",
            "metrics": metrics, "breakdowns": unique[:BREAKDOWNS_SHOWN], "more": max(0, len(unique) - BREAKDOWNS_SHOWN),
            "dates": [{"key": r.key, "name": r.name, "default": r.is_default, "also": _words(r.synonyms),
                       "span": _span(r.first, r.last, r.granularity)} for r in dates]})
    return {"subjects": subjects}


# ── a suggestion, as the admin decisions it would write ──────────────────────

def _target(model: SemanticModel, reader: _Reader, kind: str, key: str) -> Measure | Attribute | Entity | DateRole:
    found: Any = {"measure": model.measures, "attribute": model.attributes, "entity": model.entities,
                  "date_role": model.date_roles}.get(kind, {}).get(key)
    if isinstance(found, Measure) and not found.hidden and found.slug and reader.may(found.table):
        return found
    if isinstance(found, Attribute) and _visible_column(model, found.column) \
            and reader.may(model.columns[found.column].table):
        return found
    if isinstance(found, Entity) and reader.may(found.table):
        return found
    if isinstance(found, DateRole) and found.kind != "audit" and reader.may(found.table):
        return found
    raise RequestError("That is not in the data you can ask about any more.")


def _name_of(thing: Any) -> str:
    return getattr(thing, "business_name", "") or getattr(thing, "name", "") or getattr(thing, "key", "")


def split_names(text: str | list[str]) -> list[str]:
    """Other names as a reader types them: one per comma or line."""
    items = text if isinstance(text, list) else re.split(r"[,;\n]", str(text or ""))
    return list(dict.fromkeys(" ".join(str(w).split())[:60] for w in items if str(w).strip()))


def _named_elsewhere(model: SemanticModel, word: str, thing: Any) -> str:
    """Another metric or field already called (or also called) ``word``: the admin is told before accepting."""
    folded = word.casefold()
    for other in [*model.measures.values(), *model.attributes.values(), *model.entities.values()]:
        if other is thing or getattr(other, "hidden", False):
            continue
        if folded == _name_of(other).casefold() or folded in (w.casefold() for w in _words(other.synonyms)):
            return _name_of(other)
    return ""


def proposal(model: SemanticModel, *, kind: str, key: str, meaning: str = "", names: str | list[str] = "",
             date: str = "", note: str = "", codes: dict[str, Any] | None = None, allowed: set[str] | None = None,
             today: dt.date | None = None) -> tuple[str, list[dict[str, Any]]]:
    """(the thing's name, the changes) a suggestion asks for, or RequestError with what to fix."""
    reader = _Reader(model, allowed, False, today or dt.date.today())
    thing = _target(model, reader, kind, key)
    name = _name_of(thing)
    changes: list[dict[str, Any]] = []

    meaning = " ".join(str(meaning or "").split())
    if meaning:
        if len(meaning) > MAX_MEANING:
            raise RequestError(f"Keep what it means under {MAX_MEANING} characters.")
        where = _meaning_of(model, thing)
        if where is None:
            raise RequestError(f"What {name} means cannot be changed here; say it in a note instead.")
        object_key, now = where
        if meaning != now:
            changes.append({"object_key": object_key, "field": "description", "label": "What it means",
                            "now": now, "value": meaning})

    words = split_names(names)
    if len(words) > MAX_NAMES:
        raise RequestError(f"Suggest at most {MAX_NAMES} other names at a time.")
    have = {w.casefold() for w in [name, *_words(thing.synonyms)]}
    added = [w for w in words if w.casefold() not in have]
    if added:
        changes.append({"object_key": _object_key(thing), "field": "synonyms", "label": "Other names people use",
                        "now": _words(thing.synonyms), "value": added, "merge": True,
                        "clashes": {w: c for w in added if (c := _named_elsewhere(model, w, thing))}})

    if date:
        if not isinstance(thing, Measure):
            raise RequestError("Only a metric is counted by a date.")
        role = model.date_roles.get(date)
        if role is None or role.table != thing.table or role.kind == "audit":
            raise RequestError(f"{name} cannot be counted by that date.")
        current = thing.default_date or model.tables[thing.table].default_date or ""
        if date != current:
            now_role = model.date_roles.get(current)
            changes.append({"object_key": _object_key(thing), "field": "default_date", "label": "Counted by",
                            "now": now_role.name if now_role else "", "value": date, "shown": role.name})

    if codes:
        changes.extend(_code_names(model, thing, name, codes))

    note = " ".join(str(note or "").split())
    if len(note) > MAX_NOTE:
        raise RequestError(f"Keep the note under {MAX_NOTE} characters.")
    if not changes and not note:
        raise RequestError("Nothing to send: change what it means, add a name people use, or say what is wrong.")
    return name, changes


def _code_names(model: SemanticModel, thing: Any, name: str, codes: Any) -> list[dict[str, Any]]:
    """Names for a field's codes, as the change accepting them writes: only codes it has, only names that
    differ from today's, and no two codes named alike (a reader could not tell them apart)."""
    column_key = thing.column if isinstance(thing, Attribute) else None
    known = {c["code"] for c in _codes(model, column_key)}
    if not known:
        raise RequestError(f"{name} has no codes to name.")
    if not isinstance(codes, dict):
        raise RequestError("Name the codes as code and name.")
    given = {str(code): " ".join(str(text or "").split()) for code, text in codes.items()}
    given = {code: text for code, text in given.items() if text}
    if any(len(text) > MAX_CODE_NAME for text in given.values()):
        raise RequestError(f"Keep each code's name under {MAX_CODE_NAME} characters.")
    stray = sorted(set(given) - known)
    if stray:
        raise RequestError(f"{', '.join(stray[:3])} is not one of the codes of {name}.")
    column = model.columns[column_key or ""]
    new = {code: text for code, text in given.items() if column.value_names.get(code) != text}
    if not new:
        return []
    named = {**column.value_names, **new}
    if len({text.casefold() for text in named.values()}) < len(named):
        raise RequestError("Two codes would have the same name: readers could not tell them apart.")
    return [{"object_key": f"column:{column.key}", "field": "value_names", "label": "What its codes mean",
             "now": dict(column.value_names), "value": new, "merge": True}]


def _object_key(thing: Any) -> str:
    """The key its decisions are stored under: a metric and a date by their key, a field and an entity by slug."""
    kind = {Measure: "measure", Attribute: "attribute", Entity: "entity", DateRole: "date_role"}[type(thing)]
    return f"{kind}:{thing.slug if isinstance(thing, (Attribute, Entity)) else thing.key}"


def _meaning_of(model: SemanticModel, thing: Any) -> tuple[str, str] | None:
    """Where what a thing means is kept, and what it says now: a metric's own, a field's column's."""
    if isinstance(thing, Measure):
        return f"measure:{thing.key}", thing.description
    column_key = thing.column if isinstance(thing, Attribute) else thing.label_column if isinstance(thing, Entity) \
        else None
    column = model.columns.get(column_key or "")
    return (f"column:{column.key}", column.description) if column is not None else None


def to_write(model: SemanticModel, change: dict[str, Any], *, lang: str = "en") -> tuple[str, str, Any]:
    """The decision one change writes now, against the model as it is now: added names join today's words."""
    object_key, field, value = change["object_key"], change["field"], change["value"]
    if field == "value_names":
        # Named codes join today's names; a code named since keeps the name suggested here.
        column = model.columns.get(object_key.partition(":")[2])
        if column is None:
            raise RequestError("It is not in the data any more.")
        named = {**column.value_names, **{str(k): str(v) for k, v in dict(value).items()}}
        if len({text.casefold() for text in named.values()}) < len(named):
            raise RequestError("Two codes would have the same name: readers could not tell them apart.")
        return object_key, field, named
    if change.get("merge"):
        kind, _, key = object_key.partition(":")
        thing: Any = {"measure": model.measures, "attribute": model.attributes, "entity": model.entities,
                      "date_role": model.date_roles}.get(kind, {}).get(key)
        if thing is None:
            raise RequestError("It is not in the data any more.")
        merged = {k: list(v) for k, v in thing.synonyms.items()}
        have = {w.casefold() for w in _words(thing.synonyms)}
        merged[lang] = [*merged.get(lang, []), *(w for w in split_names(value) if w.casefold() not in have)]
        return object_key, field, merged
    return object_key, field, value
