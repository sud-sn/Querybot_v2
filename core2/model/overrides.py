"""Admin decisions, applied on top of every model build.

An override is (object key, field, value). It is stored once and applied after
every build, so a rebuild never loses a decision. An override whose object no
longer exists (a dropped column) is reported, not silently discarded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from core2.model.schema import AggExpr, Join, Measure, SemanticModel

# What an admin may change, per kind of object. Everything else is the data's call.
ALLOWED = {
    "table": {"business_name", "description", "kind", "default_date", "hidden", "default_filters"},
    "column": {"business_name", "description", "role", "format", "unit", "synonyms", "hidden", "sensitivity"},
    "join": {"trust", "role"},
    "date_role": {"name", "kind", "is_default", "synonyms"},
    "measure": {"business_name", "description", "synonyms", "additivity", "time_aggregation", "format", "hidden",
                "filters", "default_date"},
    "entity": {"business_name", "label_column", "code_column", "synonyms"},
    "attribute": {"business_name", "synonyms"},
}
# A metric or a link that today's setup has and the build did not find is added
# whole ("define"), then decided on like any other object.
DEFINABLE: dict[str, type[Measure] | type[Join]] = {"measure": Measure, "join": Join}


def target(kind: str, key: str) -> str:
    """The object key an override is stored under: ``kind:key`` (a date role and its column share a key)."""
    if kind not in ALLOWED:
        raise ValueError(f"no such kind of object: {kind!r}")
    return f"{kind}:{key}"


def _find(model: SemanticModel, object_key: str) -> tuple[str, BaseModel] | None:
    kind, _, key = object_key.partition(":")
    collections: dict[str, dict[str, Any]] = {
        "table": model.tables, "column": model.columns, "join": model.joins, "date_role": model.date_roles,
        "measure": model.measures, "entity": model.entities, "attribute": model.attributes}
    collection = collections.get(kind)
    if collection is None or key not in collection:
        return None
    return kind, collection[key]


def _define(model: SemanticModel, object_key: str, value: Any, notes: list[str]) -> None:
    """Add a metric or a link today's setup has; one the build found since is kept as found (and trusted)."""
    kind, _, key = object_key.partition(":")
    kind_type = DEFINABLE.get(kind)
    if kind_type is None:
        notes.append(f"A decision defines {object_key}, which cannot be added by hand.")
        return
    collection: dict[str, Any] = model.measures if kind == "measure" else model.joins
    if key in collection:
        if kind == "join":
            collection[key].trust = "admin"
        return
    try:
        obj = kind_type.model_validate(value)
    except ValidationError:
        notes.append(f"A decision adds {key}, but its definition cannot be read.")
        return
    if isinstance(obj, Join):
        known = obj.from_table in model.tables and obj.to_table in model.tables and all(
            c in model.columns for c in [*obj.from_columns, *obj.to_columns])
    else:
        known = obj.table in model.tables and all(c in model.columns for c in _columns_of(obj.expr))
        if obj.default_date not in model.date_roles:
            obj.default_date = None       # a date no longer learned: counted by its table's default
        taken = {m.slug for m in model.measures.values()}
        if obj.slug in taken:
            from core2 import ids

            obj.slug = ids.unique_slug(obj.slug, taken)
    if not known:
        notes.append(f"A decision adds {getattr(obj, 'business_name', '') or key}, but it refers to tables or "
                     "columns that are not in the data any more.")
        return
    collection[key] = obj


def _columns_of(expr: Any) -> list[str]:
    if isinstance(expr, AggExpr):
        return [c for c in [expr.column, *[f.column for f in expr.filters]] if c]
    return [c for arg in getattr(expr, "args", []) for c in _columns_of(arg)]


def apply_overrides(model: SemanticModel, overrides: list[dict[str, Any]]) -> list[str]:
    """Apply admin decisions in place; returns plain-language notes on any that could not apply."""
    notes: list[str] = []
    for item in overrides:            # what is added comes first: decisions about it follow
        if item["field"] == "define":
            _define(model, item["object_key"], item["value"], notes)
    for item in overrides:
        if item["field"] == "define":
            continue
        key, field, value = item["object_key"], item["field"], item["value"]
        found = _find(model, key)
        if found is None:
            notes.append(f"A decision about {key} no longer applies: it is not in the data any more.")
            continue
        kind, obj = found
        key = key.partition(":")[2]
        if field not in ALLOWED[kind]:
            notes.append(f"A decision sets {field} on {key}, which cannot be changed by hand.")
            continue
        if kind == "date_role" and field == "is_default" and value:
            # One default per table: choosing one clears the others.
            owner = model.date_roles[key].table
            for role in model.date_roles.values():
                if role.table == owner:
                    role.is_default = role.key == key
            model.tables[owner].default_date = key
            for measure in model.measures.values():
                if measure.table == owner and measure.provenance != "admin":
                    measure.default_date = key
        elif kind == "table" and field == "default_date":
            if value in model.date_roles and model.date_roles[value].table == key:
                for role in model.date_roles.values():
                    if role.table == key:
                        role.is_default = role.key == value
                model.tables[key].default_date = value
        else:
            # Validated through the object's own type, so a bad value is refused
            # and nested values (filters) become the model's objects.
            try:
                validated = type(obj).model_validate({**obj.model_dump(), field: value})
            except ValidationError:
                notes.append(f"A decision sets {field} on {key} to a value it cannot take ({value!r}).")
                continue
            setattr(obj, field, getattr(validated, field))
            if kind == "measure" and field == "additivity":
                assert isinstance(obj, Measure)
                _aggregate_as(obj, str(value))
        if hasattr(obj, "provenance"):
            obj.provenance = "admin"  # type: ignore[attr-defined]
        if hasattr(obj, "status"):
            rejected = (field == "trust" and value == "rejected") or (field == "hidden" and value)
            obj.status = "rejected" if rejected else "approved"  # type: ignore[attr-defined]
        if kind == "join" and field == "trust":
            obj.trust = value  # type: ignore[attr-defined]
    model.notes = [n for n in model.notes if not n.startswith("A decision")] + notes
    return notes


def _aggregate_as(measure: Measure, additivity: str) -> None:
    """A column measure follows its additivity: what adds up is summed, what does not (a price) is averaged.

    A balance (semi-additive) is still summed across things, at one point in time.
    Counts and formulas keep their own aggregation.
    """
    expr = measure.expr
    if not isinstance(expr, AggExpr) or expr.column is None or expr.agg not in ("sum", "avg"):
        return
    if expr.agg != ("avg" if additivity == "non_additive" else "sum"):
        measure.expr = AggExpr(agg="avg" if additivity == "non_additive" else "sum", column=expr.column,
                               filters=list(expr.filters))
    if additivity == "semi_additive" and measure.time_aggregation is None:
        measure.time_aggregation = "last"
