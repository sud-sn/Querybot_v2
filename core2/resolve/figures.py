"""Figures no metric holds, worked out for the question that asks them (Plan.figures).

"The lowest batch yield", "the smallest and largest quantity dispensed", "average margin per fill, price charged
minus ingredient cost": no saved metric is any of them, and an answer should not wait for an admin to add one.
The planner writes each as a formula in the admin's own formula language (core2/model/authoring.py) from the names
it is given. Here it is read exactly as an admin's metric is -- parsed into the model's form, never pasted into a
query -- and refused when it reads a field the reader may not use (a personal or hidden field, a table outside
their access). It is then a measure of this question alone, counted like any other: grouped, compared over two
periods, sorted, filtered, with its table's left-out rows left out.

Reading the same plan twice gives the same measures (the same keys), so the service can read them once for the
whole answer and the resolver again for any other caller (a dashboard tile, an alert, the question map).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core2 import ids
from core2.model import authoring
from core2.model.schema import AggExpr, Measure, OpExpr, SemanticModel, SqlExpr

if TYPE_CHECKING:
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context

PREFIX = "figure:"


def worked_out(plan: "Plan", model: SemanticModel, ctx: "Context") -> tuple["Plan", SemanticModel, list[str]]:
    """``plan`` with its figures named as measures, ``model`` with them in it, and what each answer says of them.

    A figure the plan names nowhere else is one of its measures; one a derived figure, a rate or a sort uses is
    left to them. Raises ResolveError: "unknown" with the reason a formula cannot be read, "denied" when it reads
    data the reader may not use."""
    from core2.resolve.resolver import ResolveError, _readable

    if not plan.figures:
        return plan, model, []
    made: dict[str, Measure] = {}
    reads: set[str] = set()          # the fields the figures are worked out from
    notes: list[str] = []
    renamed: dict[str, str] = {}
    taken = {m.slug for m in model.measures.values() if not m.key.startswith(PREFIX)}
    for figure in plan.figures:
        try:
            read = authoring.read(_with(model, made), figure.formula)
        except authoring.AuthoringError as exc:
            raise ResolveError("unknown", f"{figure.name} could not be worked out: {exc}") from None
        if ctx.allowed_tables is not None and not set(read.tables) <= ctx.allowed_tables:
            raise ResolveError("denied", f"{figure.name} would use data you cannot ask about")
        if any(c in model.columns and (model.columns[c].hidden or model.columns[c].sensitivity != "none")
               for c in read.columns):
            raise ResolveError("denied", f"{figure.name} would read a personal or hidden field, which no figure "
                               "may use")
        reads.update(read.columns)
        from core2.model.imports import _behaviour

        additivity, time_aggregation = _behaviour(model, read.table, read.expr)
        called = _readable(figure.name)          # "average_margin_per_fill" is shown "Average margin per fill"
        base = ids.slug(figure.name, fallback="figure")
        slug = ids.unique_slug(base, taken)
        taken.add(slug)
        renamed[_plain(figure.name)] = renamed[base] = slug
        made[PREFIX + slug] = Measure(
            key=PREFIX + slug, slug=slug, business_name=called, synonyms={}, table=read.table, expr=read.expr,
            additivity=additivity, time_aggregation=time_aggregation, format=figure.format or _format(model, read),
            kind="metric", provenance="ai", status="approved", tested=False,
            description=f"Worked out for this question: {authoring.to_text(model, read.expr)}")
        notes.append(f"{called} was worked out for this question: {authoring.to_text(model, read.expr)}.")
        notes += _not_valued(model, read, called)
    extended = _with(model, made)

    def named(ref: str) -> str:
        return renamed.get(_plain(ref), renamed.get(ref, ref))

    measures = [named(s) for s in plan.measures]
    used = {*measures, *(named(s) for d in plan.derived for s in d.measures), *(named(r.measure) for r in plan.rates)}
    measures += [m.slug for m in made.values() if m.slug not in used]
    figured = {m.slug for m in made.values()}

    def part(slug: str) -> bool:
        m = next((x for x in extended.measures.values() if x.slug == slug and x.key not in made), None)
        return m is not None and isinstance(m.expr, AggExpr) and m.expr.column in reads

    if any(part(s) for s in measures):
        # "Average margin per fill, price charged minus ingredient cost minus labor cost": the fields it is made of
        # are named with it, and the figure asked for leads the answer, not the first of its parts.
        measures = [s for s in measures if s in figured] + [s for s in measures if s not in figured]
    update = {"measures": list(dict.fromkeys(measures)),
              "sort": [s.model_copy(update={"by": named(s.by)}) for s in plan.sort],
              "filters": [f.model_copy(update={"field": named(f.field)}) for f in plan.filters],
              "derived": [d.model_copy(update={"measures": [named(s) for s in d.measures]}) for d in plan.derived],
              "rates": [r.model_copy(update={"measure": named(r.measure)}) for r in plan.rates]}
    return plan.model_copy(update=update), extended, notes


def _not_valued(model: SemanticModel, read: authoring.Read, called: str) -> list[str]:
    """A figure made of a field kept on another table ("each product's list price") has nothing for the rows whose
    member is not on file, or has no value for it: said, never shown as a figure of nothing (the "Unknown" category
    at a 100% margin)."""
    out = []
    for key in sorted(read.columns):
        column = model.columns.get(key)
        if column is None or column.table == read.table:
            continue
        link = next((j for j in model.joins.values() if j.from_table == read.table and j.to_table == column.table
                     and j.trust != "rejected"), None)
        profile = column.profile
        empty = profile is not None and profile.rows and profile.non_null < profile.rows
        if link is None or not (link.orphan_rows or link.null_rate or empty):
            continue
        rows, member = model.tables[read.table].business_name, model.tables[column.table].business_name
        out.append(f"{rows}s with no {member.lower()} on file, or no {column.business_name.lower()} for it, "
                   f"count as nothing in {called.lower()}.")
    return out


def _with(model: SemanticModel, made: dict[str, Measure]) -> SemanticModel:
    if not made or all(model.measures.get(k) == m for k, m in made.items()):
        return model
    copy = model.model_copy(deep=False)
    copy.measures = {**model.measures, **made}
    return copy


def _plain(text: str) -> str:
    return ids.slug(text, fallback="")


def _format(model: SemanticModel, read: authoring.Read) -> str:
    """How a figure is shown when the plan did not say: a count as a count, a percentage of a ratio as a percent,
    else as the field it reads is shown (money as money)."""
    expr = read.expr
    if isinstance(expr, AggExpr) and expr.agg in ("count", "count_distinct"):
        return "count"
    if isinstance(expr, SqlExpr) and expr.sql.lstrip().upper().startswith("COUNT("):
        return "count"
    if isinstance(expr, OpExpr) and expr.op == "ratio":
        return "percent" if expr.scale == 100 else "number"
    for column in read.columns:
        shown = next((m.format for m in model.measures.values()
                      if isinstance(m.expr, AggExpr) and m.expr.column == column and m.format not in ("count",)),
                     None)
        if shown:
            return shown
    return "number"
