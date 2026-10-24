"""Admin decisions, applied on top of every model build.

An override is (object key, field, value). It is stored once and applied after
every build, so a rebuild never loses a decision. An override whose object no
longer exists (a dropped column) is reported, not silently discarded.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from core2.model.schema import (AggExpr, DateRole, Evidence, Join, Measure, RefExpr, SavedSegment, SemanticModel,
                                 SqlExpr)

# What an admin may change, per kind of object. Everything else is the data's call.
ALLOWED = {
    "table": {"business_name", "description", "kind", "default_date", "hidden", "default_filters",
              "readers_may_include", "grain_text"},
    "column": {"business_name", "description", "role", "format", "unit", "synonyms", "hidden", "sensitivity",
               "value_names", "personal", "age", "record_number"},
    # How many target rows a source row meets (max_fanout, to_unique, target_checked) is measured, never typed:
    # measured again when an admin changes a link's conditions (admin/core2_relationships.py).
    "join": {"trust", "role", "conditions", "keep_unmatched", "cardinality", "max_fanout", "to_unique",
             "target_checked"},
    "date_role": {"name", "kind", "is_default", "synonyms"},
    "measure": {"business_name", "description", "synonyms", "additivity", "time_aggregation", "format", "hidden",
                "filters", "default_date", "expr", "table", "better", "empty_periods"},
    "entity": {"business_name", "label_column", "code_column", "synonyms"},
    "attribute": {"business_name", "synonyms"},
    "settings": {"fiscal_year_start_month"},      # one object: settings:workspace
    # A group of members an admin named ("Regular shoppers"): always the admin's own, added whole and decided on.
    "segment": {"business_name", "description", "synonyms", "hidden", "kind"},
}
# A metric, a link or a date that today's setup or an admin has and the build did
# not find is added whole ("define"), then decided on like any other object.
DEFINABLE: dict[str, type[Measure] | type[Join] | type[DateRole] | type[SavedSegment]] = {
    "measure": Measure, "join": Join, "date_role": DateRole, "segment": SavedSegment}


def set_aside(model: SemanticModel, table_key: str, aside: bool, why: str = "") -> None:
    """A table set aside answers nothing: its figures and fields go with it, and come back with it, those it hid
    alone (a field kept from readers for its own reason stays kept)."""
    table = model.tables[table_key]
    if aside and not table.hidden:
        columns = [c.key for c in model.columns.values() if c.table == table_key and not c.hidden]
        measures = [m.key for m in model.measures.values() if m.table == table_key and not m.hidden]
        for k in columns:
            model.columns[k].hidden = True
        for k in measures:
            model.measures[k].hidden = True
        table.evidence.append(Evidence(kind="set_aside", weight=1, detail=why or "set aside by an admin",
                                       data={"columns": columns, "measures": measures}))
    elif not aside and table.hidden:
        for e in table.evidence:
            if e.kind == "set_aside":
                for k in e.data.get("columns") or []:
                    if k in model.columns:
                        model.columns[k].hidden = False
                for k in e.data.get("measures") or []:
                    if k in model.measures:
                        model.measures[k].hidden = False
        table.evidence = [e for e in table.evidence if e.kind != "set_aside"]
    table.hidden = aside


def target(kind: str, key: str) -> str:
    """The object key an override is stored under: ``kind:key`` (a date role and its column share a key)."""
    if kind not in ALLOWED:
        raise ValueError(f"no such kind of object: {kind!r}")
    return f"{kind}:{key}"


def _find(model: SemanticModel, object_key: str) -> tuple[str, BaseModel] | None:
    kind, _, key = object_key.partition(":")
    collections: dict[str, dict[str, Any]] = {
        "table": model.tables, "column": model.columns, "join": model.joins, "date_role": model.date_roles,
        "measure": model.measures, "entity": model.entities, "attribute": model.attributes,
        "segment": model.segments, "settings": {"workspace": model.settings}}
    collection = collections.get(kind)
    if collection is None or key not in collection:
        return None
    return kind, collection[key]


def _define(model: SemanticModel, object_key: str, value: Any, notes: list[str]) -> None:
    """Add a metric, a link or a date the build did not find; one the build found since is kept as found."""
    kind, _, key = object_key.partition(":")
    kind_type = DEFINABLE.get(kind)
    if kind_type is None:
        notes.append(f"A decision defines {object_key}, which cannot be added by hand.")
        return
    collection: dict[str, Any] = {"measure": model.measures, "join": model.joins, "date_role": model.date_roles,
                                  "segment": model.segments}[kind]
    if key in collection:
        if kind == "join":
            collection[key].trust = "admin"
        return
    try:
        obj = kind_type.model_validate(value)
    except ValidationError:
        notes.append(f"A decision adds {key}, but its definition cannot be read.")
        return
    if isinstance(obj, DateRole):
        known = obj.table in model.tables and obj.column in model.columns \
            and model.columns[obj.column].table == obj.table \
            and (obj.calendar is None or (obj.calendar in model.calendars and obj.calendar_join in model.joins)) \
            and (obj.quarter_column is None or (obj.quarter_column in model.columns
                                                and model.columns[obj.quarter_column].table == obj.table))
        if obj.quarter_column:
            # A year and a quarter number date the quarter, nothing finer; a year and a period number, the month.
            obj.granularity = "quarter" if obj.period_months == 3 else "month"
        obj.is_default = False      # a default is its own decision
        from core2 import ids

        taken = {r.slug for r in model.date_roles.values()}
        obj.slug = ids.unique_slug(ids.slug(obj.slug or obj.name or obj.key), taken)
    elif isinstance(obj, SavedSegment):
        # Members and activity still learned: the entity, and a measure of its rows.
        activity = next((m for m in model.measures.values() if m.slug == obj.measure), None)
        known = obj.of in model.entities and activity is not None and activity.table in model.tables
        taken = {s.slug for s in model.segments.values()}
        if obj.slug in taken:
            from core2 import ids

            obj.slug = ids.unique_slug(obj.slug, taken)
    elif isinstance(obj, Join):
        known = obj.from_table in model.tables and obj.to_table in model.tables and all(
            c in model.columns for c in [*obj.from_columns, *obj.to_columns]) and not _stray_conditions(model, obj)
    else:
        known = obj.table in model.tables and not _unknown_parts(model, obj.expr, obj.filters)
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
    if isinstance(expr, SqlExpr):
        return [*expr.columns, *[f.column for f in expr.filters]]
    return [c for arg in getattr(expr, "args", []) for c in _columns_of(arg)]


def _unknown_parts(model: SemanticModel, expr: Any, filters: list[Any]) -> list[str]:
    """What a measure's definition reads that the model does not have: columns, tables, other measures."""
    missing = [c for c in [*_columns_of(expr), *[f.column for f in filters]] if c not in model.columns]

    def walk(e: Any) -> None:
        if isinstance(e, AggExpr) and e.table and e.table not in model.tables:
            missing.append(e.table)
        if isinstance(e, RefExpr) and e.measure not in model.measures:
            missing.append(e.measure)
        for arg in getattr(e, "args", []):
            walk(arg)

    walk(expr)
    return missing


def _named_measures(value: Any) -> set[str]:
    """The measures a stored definition is made of: the keys of its references (``{"measure": key}``)."""
    named: set[str] = set()

    def walk(e: Any) -> None:
        if isinstance(e, dict):
            if set(e) == {"measure"} and isinstance(e["measure"], str):
                named.add(e["measure"])
            for v in e.values():
                walk(v)
        elif isinstance(e, list):
            for v in e:
                walk(v)

    walk(value.get("expr") if isinstance(value, dict) else None)
    return named


def _stray_conditions(model: SemanticModel, join: Join) -> list[str]:
    """Conditions of a link that are not on the table it reaches (or on no column at all)."""
    return [f.column for f in join.conditions
            if f.column not in model.columns or model.columns[f.column].table != join.to_table]


def apply_overrides(model: SemanticModel, overrides: list[dict[str, Any]]) -> list[str]:
    """Apply admin decisions in place; returns plain-language notes on any that could not apply."""
    notes: list[str] = []
    # What is added comes first, decisions about it follow; a date is added after the link to its calendar.
    # Metrics whose own date an admin chose keep it when their table's default changes.
    pinned = {item["object_key"].partition(":")[2] for item in overrides
              if item["field"] == "default_date" and item["object_key"].startswith("measure:")}
    # Links first (a date reads its calendar through one), then dates (a metric may be counted by one the admin
    # added), then metrics, then segments (each the members of a metric's rows). Decisions come back in the order
    # of their keys: a metric made of another the admin added ("Net charge-offs" of "Recoveries") waits until that
    # one is in, and only then is checked for what it reads; a segment waits for every metric.
    order = {"join": 0, "date_role": 1, "measure": 2, "segment": 3}
    defines = sorted((item for item in overrides if item["field"] == "define"),
                     key=lambda item: order.get(item["object_key"].partition(":")[0], 2))
    while defines:
        waiting = {item["object_key"].partition(":")[2] for item in defines
                   if item["object_key"].startswith("measure:")}
        ready = [item for item in defines
                 if not (item["object_key"].startswith("segment:") and waiting)
                 and not _named_measures(item["value"]) & waiting - {item["object_key"].partition(":")[2]}]
        for item in ready or defines:       # none ready: they name each other, and each is said
            _define(model, item["object_key"], item["value"], notes)
        done = {id(item) for item in ready or defines}
        defines = [item for item in defines if id(item) not in done]
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
            _default_date(model, model.date_roles[key].table, key, pinned)
        elif kind == "table" and field == "default_date":
            if value in model.date_roles and model.date_roles[value].table == key:
                _default_date(model, key, value, pinned)
        elif kind == "table" and field == "hidden":
            set_aside(model, key, bool(value))
        else:
            # Validated through the object's own type, so a bad value is refused
            # and nested values (filters) become the model's objects.
            try:
                validated = type(obj).model_validate({**obj.model_dump(), field: value})
            except ValidationError:
                notes.append(f"A decision sets {field} on {key} to a value it cannot take ({value!r}).")
                continue
            if isinstance(validated, Join) and field == "conditions" and _stray_conditions(model, validated):
                notes.append(f"A decision keeps rows of {key} by a column that is not on the table it reaches.")
                continue
            if isinstance(validated, Measure) and field in ("expr", "filters", "table") and (
                    validated.table not in model.tables or _unknown_parts(model, validated.expr, validated.filters)):
                notes.append(f"A decision changes how {key} is counted, but it reads columns or measures that are "
                             "not in the data any more.")
                continue
            setattr(obj, field, getattr(validated, field))
            if kind == "column" and field in ("business_name", "synonyms"):
                # A field is grouped and filtered by under its own name and words (the Knowledge base page).
                for attribute in model.attributes.values():
                    if attribute.column == key:
                        setattr(attribute, field, getattr(validated, field))
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
    from core2.resolve.time import fiscal_named_by, quarter_span

    fiscal = model.settings.fiscal_year_start_month
    for role in model.date_roles.values():
        if role.quarter_column and role.quarters:
            # Kept by fiscal quarter: its first and last days by the fiscal year as set now, not at Learn.
            role.first, role.last = quarter_span(role.quarters, role.fiscal, fiscal, role.period_months,
                                                 fiscal_named_by(model, fiscal))
    return notes


def _default_date(model: SemanticModel, table: str, key: str, pinned: set[str]) -> None:
    """``key`` becomes ``table``'s default date, and the date of the metrics that were on the old one.

    A metric with no date of its own (None) follows its table's default when it is answered, and is left so.
    """
    before = model.tables[table].default_date
    for role in model.date_roles.values():
        if role.table == table:
            role.is_default = role.key == key
    model.tables[table].default_date = key
    for measure in model.measures.values():
        if measure.table == table and measure.key not in pinned and measure.default_date == before:
            measure.default_date = key


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
