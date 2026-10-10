"""What the planner knows about a workspace: one stable text per model version.

The catalog is the cached prefix of every planning prompt (DESIGN §7.2): the
measures, the dates they are counted by, the things to group and filter by, and
the calendar, in the model's own slugs. It is sorted and carries nothing about
the question or the clock, so it is byte-identical for a model version and a
provider can cache it. Tables and joins are not in it: paths are the resolver's
job. Member values appear only for short lists, and never where the tenant keeps
values out of prompts or the column is sensitive.
"""

from __future__ import annotations

import re

from sqlglot import exp

from core2.model import formula
from core2.model.schema import AggExpr, Attribute, Measure, MeasureExpr, OpExpr, RefExpr, SemanticModel, SqlExpr
from core2.plan.ir import TIME_ATTRIBUTES
from core2.resolve import paths as P

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
ADDS_UP = {"additive": "adds up", "non_additive": "does not add up (averaged or recomputed)",
            "semi_additive": "a balance: taken on the last day of each period, never added over time"}


def definition(model: SemanticModel, expr: MeasureExpr, values: bool = True) -> str:
    """A measure's definition in words: "sum of Net amount where Status is not C".

    Where values stay out of prompts (``values`` false), a filter's values and a formula's
    text values are not written: "where Status is not (a value)".
    """
    if isinstance(expr, AggExpr):
        target = model.columns[expr.column].business_name if expr.column else (
            f"{model.tables[expr.table].business_name} rows" if expr.table in model.tables else "rows")
        head = {"sum": f"sum of {target}", "count": f"number of {target}",
                "count_distinct": f"number of distinct {target}", "avg": f"average of {target}",
                "min": f"smallest {target}", "max": f"largest {target}"}[expr.agg]
        if expr.filters:
            head += " where " + "; ".join(
                f"{model.columns[f.column].business_name} {f.op.replace('_', ' ')} "
                f"{', '.join(map(str, f.values)) if values else '(a value)'}"
                for f in expr.filters)
        return head
    if isinstance(expr, OpExpr):
        sign = {"ratio": "÷", "subtract": "−", "add": "+", "multiply": "×"}[expr.op]
        text = f" {sign} ".join(f"({definition(model, a, values)})" for a in expr.args)
        return f"{text} × {expr.scale:g}" if expr.scale != 1 else text
    if isinstance(expr, RefExpr):
        ref = model.measures.get(expr.measure)
        return ref.business_name if ref else expr.measure
    if isinstance(expr, SqlExpr):
        tree = formula.parse_stored(expr.sql)
        for column in list(tree.find_all(exp.Column)):
            if column.name in model.columns:      # an admin's formula names fields by their keys
                column.replace(exp.column(exp.to_identifier(model.columns[column.name].business_name
                                                            or model.columns[column.name].name, quoted=True)))
        if not values:
            for literal in list(tree.find_all(exp.Literal)):
                if literal.is_string:
                    literal.replace(exp.Literal.string("…"))
        return f"the formula {tree.sql()}"
    return ""


def left_out_words(model: SemanticModel, f, values: bool = True) -> str:
    """A rule a table leaves rows out by, as the rows it leaves out: "Status code is C"."""
    kept_as = {"ne": "is", "not_in": "is", "eq": "is not", "in": "is not", "not_null": "is empty",
               "is_null": "is filled", "gt": "is at most", "gte": "is below", "lt": "is at least", "lte": "is above"}
    named = list(getattr(f, "shown", None) or [])
    shown = (", ".join(named) if named else ", ".join(map(str, f.values))) if values else "(a value)"
    name = model.columns[f.column].business_name
    if named:   # a status kept by number: "Order status is Closed - Cancelled", not "Order status id is 8"
        name = re.sub(r"\s+(?:id|key|code|no|number|nbr|num)$", "", name, flags=re.IGNORECASE)
    word = kept_as.get(f.op)
    if word is None:
        return f"{name} not {f.op.replace('_', ' ')} {shown}"
    return f"{name} {word}" + (f" {shown}" if f.op not in ("is_null", "not_null") else "")


def left_out_note(model: SemanticModel, table_key: str, f) -> str:
    """What an answer says of a table's default filter, in the rows it left out:
    "Order line: rows where Status code is C are left out (a default filter)."
    """
    table = model.tables[table_key]
    return f"{table.business_name or table.name}: rows where {left_out_words(model, f)} are left out (a default filter)."


def _synonyms(values: dict[str, list[str]]) -> str:
    words = sorted({w for ws in values.values() for w in ws})
    return f" | also called: {', '.join(words)}" if words else ""


def _meaning(model: SemanticModel, attribute: Attribute) -> str:
    """What an admin wrote a field means (the Knowledge base page); the build's own guesses are left out."""
    column = model.columns[attribute.column]
    text = " ".join(column.description.split())
    return f" ({text[:160]})" if column.provenance == "admin" and text else ""


def _members(model: SemanticModel, attribute: Attribute, values_allowed: bool, limit: int) -> str:
    column = model.columns[attribute.column]
    p = column.profile
    count = attribute.members or (p.distinct if p else 0)
    if column.sensitivity != "none":
        return "sensitive: never shown or filtered on"
    if not values_allowed or not column.values_allowed or not p or not p.top \
            or count > limit:
        return f"{count:,} values" if count else ""
    shown = sorted(str(t.value) for t in p.top if t.value is not None)[:limit]
    names = column.value_names
    return f"{count} values: {', '.join(f'{names[v]} ({v})' if v in names else v for v in shown)}"


def catalog_text(model: SemanticModel, *, values_allowed: bool = True, list_values_up_to: int = 12) -> str:
    """The planner's view of ``model``: stable for a model version (no clock, no question)."""
    lines: list[str] = ["DATA CATALOG", "Use only these names (slugs), exactly as written.", ""]

    # What can be measured, by subject (the table the measure counts), with its dates.
    measure_tables = sorted({m.table for m in model.measures.values() if not m.hidden},
                            key=lambda t: (model.tables[t].business_name, t))
    lines.append("MEASURES (slug | name | format | how it adds up | definition)")
    for table_key in measure_tables:
        table = model.tables[table_key]
        dates = sorted((r for r in model.date_roles.values() if r.table == table_key and r.kind != "audit"),
                       key=lambda r: (not r.is_default, r.slug))
        counted = ", ".join(f"{r.slug}{' (default)' if r.is_default else ''}" for r in dates) or "no date"
        leaves = "; ".join(left_out_words(model, f, values_allowed) for f in table.default_filters
                           if f.column in model.columns)
        left_out = (f"; leaves out rows unless asked: {leaves}" if table.readers_may_include else
                    f"; always leaves out: {leaves}") if leaves else ""
        lines.append(f"## {table.business_name or table.name} ({table.grain_text or table.kind}); dates: {counted}"
                     f"{left_out}")
        for m in sorted((m for m in model.measures.values() if m.table == table_key and not m.hidden),
                        key=lambda m: m.slug):
            status = " | unconfirmed" if m.status in ("needs_review", "proposed") else ""
            unit = f" in {m.unit}" if m.unit else ""
            lines.append(f"- {m.slug} | {m.business_name} | {m.format}{unit} | {ADDS_UP.get(m.additivity, '')} | "
                         f"{definition(model, m.expr, values_allowed)}{_synonyms(m.synonyms)}{status}")
    lines.append("")

    lines.append("DATES (slug | name | what it records | data range)")
    for r in sorted((r for r in model.date_roles.values() if r.kind != "audit" and r.table in measure_tables),
                    key=lambda r: (model.tables[r.table].business_name, r.slug)):
        span = f"{r.first or '?'} to {r.last or '?'}" if (r.first or r.last) else "range unknown"
        grain = "monthly" if r.granularity == "month" else ("date and time" if r.granularity == "timestamp" else "daily")
        lines.append(f"- {r.slug} | {r.name} | {r.kind}, {grain}, on {model.tables[r.table].business_name} | "
                     f"{span}{_synonyms(r.synonyms)}")
    lines.append("")

    lines.append("GROUP AND FILTER BY (slug | name | values | reached from)")
    reach: dict[str, list[str]] = {}
    for table_key in measure_tables:
        for e in model.entities.values():
            if e.table == table_key or P.all_paths(model, table_key, e.table):
                reach.setdefault(e.table, []).append(model.tables[table_key].business_name)
    by_owner: dict[str, list[Attribute]] = {}
    for a in model.attributes.values():
        if model.columns[a.column].hidden:
            continue
        by_owner.setdefault(model.columns[a.column].table, []).append(a)
    for e in sorted(model.entities.values(), key=lambda e: e.slug):
        if e.table not in reach:
            continue
        label = model.columns[e.label_column].business_name if e.label_column else "its key"
        named_roles = sorted({r for t in measure_tables for r in P.role_names(model, t, e.table)})
        as_roles = f" | roles (name one in via): {', '.join(named_roles)}" if named_roles else ""
        lines.append(f"- {e.slug} | {e.business_name} (named by {label}) | {e.members:,} members | "
                     f"{', '.join(sorted(set(reach[e.table])))}{as_roles}{_synonyms(e.synonyms)}")
        for a in sorted(by_owner.get(e.table, []), key=lambda a: a.slug):
            if a.column == e.label_column:
                continue
            values = _members(model, a, values_allowed, list_values_up_to)
            lines.append(f"  - {a.slug} | {a.business_name}{_meaning(model, a)} | {values}{_synonyms(a.synonyms)}")
    for table_key in measure_tables:
        own = sorted(by_owner.get(table_key, []), key=lambda a: a.slug)
        if own and not any(e.table == table_key for e in model.entities.values()):
            lines.append(f"- on {model.tables[table_key].business_name} itself:")
            for a in own:
                values = _members(model, a, values_allowed, list_values_up_to)
                lines.append(f"  - {a.slug} | {a.business_name}{_meaning(model, a)} | {values}{_synonyms(a.synonyms)}")
    lines.append("")

    lines.append(f"TIME ATTRIBUTES: {', '.join(TIME_ATTRIBUTES)} (from the date a question uses)")
    fiscal = model.settings.fiscal_year_start_month
    if fiscal and fiscal != 1:
        lines.append(f"CALENDAR: weeks start on Monday. The fiscal year starts in {_MONTHS[fiscal - 1]}; "
                     "fiscal grains are fiscal_month, fiscal_quarter, fiscal_year.")
    else:
        lines.append("CALENDAR: weeks start on Monday. The fiscal year is the calendar year.")
    return "\n".join(lines) + "\n"
