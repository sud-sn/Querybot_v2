"""Today's admin decisions, carried into the new core (DESIGN §12.4).

A workspace that is already set up holds an admin's work: links between tables
confirmed or rejected, the date each fact table is counted by, metrics built and
published, names, descriptions and synonyms. Every build reads them from today's
stores and writes them as core2 overrides authored "import", applied like any
decision (shown, and undone, on the "What QueryBot learned" page). A decision an
admin made on that page is never overwritten by an import.

Names are matched to core2's keys by the warehouse's own spelling, compared case-
insensitively: a table written as DB.SCHEMA.TABLE, SCHEMA.TABLE or TABLE, a
column by its name on that table. A name that matches nothing, or more than one
table, is reported ("not brought over"), never guessed.

Metrics come over when today's metric builder holds them in a structural form
(a total, a count or an average with filters, or a ratio of two), or as a formula
over their own table's columns (``SUM(ON_HND_QTY * ITM_CST)``), read and checked
by core2.model.formula. Anything else (a query, another table, a function a
measure does not use) is reported with its reason, never pasted: the new core
writes every query itself.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sqlglot import exp

from core2 import ids
from core2.bootstrap import names as words_of
from core2.model import formula
from core2.model.schema import AggExpr, ColumnFilter, Join, Measure, MeasureExpr, OpExpr, SemanticModel, SqlExpr

log = logging.getLogger("querybot.core2")

AUTHOR = "import"
_AGGS = {"SUM": "sum", "AVG": "avg", "COUNT": "count", "COUNT_DISTINCT": "count_distinct",
         "COUNTDISTINCT": "count_distinct", "DISTINCT_COUNT": "count_distinct", "MIN": "min", "MAX": "max"}
_OPS = {"equals": "eq", "not_equals": "ne", "greater_than": "gt", "greater_or_equal": "gte", "less_than": "lt",
        "less_or_equal": "lte", "contains": "contains", "in": "in", "not_in": "not_in", "between": "between",
        "is_null": "is_null", "is_not_null": "not_null"}
_FORMATS = {"currency": "currency", "percentage": "percent", "percent": "percent", "integer": "integer",
            "count": "count", "number": "number"}
_UNANSWERABLE = {"draft", "deprecated"}            # as today's metric_is_answerable
_EVENT_COUNTS = {"num", "nbr", "cnt", "count", "number"}


@dataclass
class Legacy:
    """Today's setup, as plain rows from QueryBot's stores."""

    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    date_contexts: list[dict] = field(default_factory=list)
    table_descriptions: dict[str, dict] = field(default_factory=dict)
    properties: list[dict] = field(default_factory=list)
    meanings: list[dict] = field(default_factory=list)
    semantic_model: dict = field(default_factory=dict)


@dataclass
class Decision:
    object_key: str
    field: str
    value: Any
    note: str


@dataclass
class Report:
    decisions: list[Decision] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def count(self, what: str) -> None:
        self.counts[what] = self.counts.get(what, 0) + 1


def _split(text: Any) -> list[str]:
    if isinstance(text, list):
        return [str(t).strip() for t in text if str(t).strip()]
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


class _Names:
    """Today's table and column names, as core2 keys."""

    def __init__(self, model: SemanticModel):
        self.model = model
        self.tables: dict[str, set[str]] = {}
        for key, t in model.tables.items():
            spellings = {key, ids.norm(t.name), ".".join(p for p in (ids.norm(t.schema_name), ids.norm(t.name)) if p)}
            for name in spellings:
                self.tables.setdefault(name, set()).add(key)

    def table(self, *names: Any) -> str | None:
        for raw in names:
            parts = [ids.norm(p) for p in re.split(r"\.", str(raw or "").replace("[", "").replace("]", "")) if p.strip()]
            for start in range(len(parts)):
                found = self.tables.get(".".join(parts[start:]))
                if found:
                    return next(iter(found)) if len(found) == 1 else None
        return None

    def column(self, table: str | None, column: Any) -> str | None:
        if not table or not str(column or "").strip():
            return None
        key = ids.column_key(table, str(column).split(".")[-1])
        return key if key in self.model.columns else None


# ── what today's setup decided ─────────────────────────────────────────────


def decisions(model: SemanticModel, legacy: Legacy) -> Report:
    report = Report()
    names = _Names(model)
    synonyms: dict[str, list[str]] = {}

    def add_synonyms(object_key: str, words: list[str]) -> None:
        if words:
            synonyms.setdefault(object_key, [])
            synonyms[object_key] += [w for w in words if w not in synonyms[object_key]]

    _links(model, legacy, names, report)
    _date_roles(model, legacy, names, report, add_synonyms)
    _metrics(model, legacy, names, report, add_synonyms)
    _descriptions(model, legacy, names, report, add_synonyms)
    for object_key, words in synonyms.items():
        report.decisions.append(Decision(object_key, "synonyms", {"en": words}, "synonyms from today's setup"))
    return report


def _entity_tables(legacy: Legacy, names: _Names) -> dict[str, str | None]:
    return {str(e.get("entity_name") or ""): names.table(
        ".".join(p for p in (str(e.get("schema_name") or ""), str(e.get("table_name") or "")) if p),
        e.get("table_name")) for e in legacy.entities}


def _links(model: SemanticModel, legacy: Legacy, names: _Names, report: Report) -> None:
    tables = _entity_tables(legacy, names)
    for rel in legacy.relationships:
        status = str(rel.get("status") or "confirmed").lower()
        active = bool(int(rel.get("is_active", 1) or 0))
        rejected = status == "rejected"   # rejecting sets the status and switches the link off
        if not rejected and (not active or status not in ("confirmed", "approved")):
            continue                      # proposed, or switched off without a decision: the data decides
        what = f"the link {rel.get('from_entity')} -> {rel.get('to_entity')}"
        ft, tt = tables.get(str(rel.get("from_entity") or "")), tables.get(str(rel.get("to_entity") or ""))
        pairs = [(rel.get("from_column"), rel.get("to_column"))]
        try:
            extra = json.loads(rel.get("join_conditions") or "[]")
        except (TypeError, ValueError):
            extra = []
        pairs += [(c.get("from_col") or c.get("from_column"), c.get("to_col") or c.get("to_column"))
                  for c in extra if isinstance(c, dict)]
        kind = str(rel.get("relationship_type") or "many_to_one").lower().replace("-", "_").replace(" ", "_")
        if kind == "one_to_many":
            # Written from the one side (a customer to their orders): the link a question follows goes from
            # the many to the one. Followed as written, it would count a customer once per order.
            ft, tt, pairs = tt, ft, [(b, a) for a, b in pairs]
        left = [names.column(ft, a) for a, _ in pairs]
        right = [names.column(tt, b) for _, b in pairs]
        if not ft or not tt or None in left or None in right:
            report.missed.append(f"{what}: its tables or columns are not in the data the new core learned")
            continue
        lefts, rights = sorted(c for c in left if c), sorted(c for c in right if c)
        found = [j for j in model.joins.values() if j.from_table == ft and j.to_table == tt
                 and sorted(j.from_columns) == lefts and sorted(j.to_columns) == rights]
        if rejected:
            for j in found:
                report.decisions.append(Decision(f"join:{j.key}", "trust", "rejected", f"{what} was rejected"))
                report.count("links rejected")
            continue
        if found:
            for j in found:
                report.decisions.append(Decision(f"join:{j.key}", "trust", "admin", f"{what} was confirmed"))
            report.count("links confirmed")
            continue
        key = ids.join_key(ft, [str(a) for a, _ in pairs], tt, [str(b) for _, b in pairs])
        rate = rel.get("match_rate")
        join = Join(key=key, from_table=ft, from_columns=[c for c in left if c], to_table=tt,
                    to_columns=[c for c in right if c],
                    # Many to many is kept, never followed: each row would be counted once per match.
                    cardinality=kind if kind in ("one_to_one", "many_to_many") else "many_to_one",
                    match_rate=float(rate) if isinstance(rate, (int, float)) and rate >= 0 else 1.0,
                    trust="admin", provenance="admin", status="approved")
        report.decisions.append(Decision(f"join:{key}", "define", join.model_dump(mode="json"),
                                         f"{what} was confirmed (the new core had not found it)"))
        report.count("links added")


def _date_roles(model: SemanticModel, legacy: Legacy, names: _Names, report: Report, add_synonyms) -> None:
    # Today's model keeps each date role twice (the whole list and each table's);
    # the Date Roles page writes both, so either copy carries the decision.
    copies = list(legacy.semantic_model.get("date_roles") or [])
    copies += [dr for t in legacy.semantic_model.get("tables") or [] for dr in t.get("date_roles") or []]
    seen: set[str] = set()
    cleared: set[str] = set()
    for dr in copies:
        if str(dr.get("status") or "").lower() != "approved":
            continue
        table = names.table(dr.get("fact_table"))
        column = names.column(table, dr.get("fact_column"))
        role = next((r for r in model.date_roles.values() if r.table == table and r.column == column), None)
        if role is None:
            report.missed.append(f"the date {dr.get('fact_table')}.{dr.get('fact_column')}: not a date the new core "
                                 "found")
            continue
        if role.key in seen:
            continue
        seen.add(role.key)
        if dr.get("default_disabled") and table:
            cleared.add(table)
        elif dr.get("is_default"):
            report.decisions.append(Decision(f"date_role:{role.key}", "is_default", True,
                                             f"{role.name} is the default date (today's setup)"))
            report.count("default dates")
        name = str(dr.get("name") or "").strip()
        if name and name != role.name:
            report.decisions.append(Decision(f"date_role:{role.key}", "name", name, "named in today's setup"))
        add_synonyms(f"date_role:{role.key}", _split(dr.get("synonyms")))
    for table in sorted(cleared):
        current = model.date_roles.get(model.tables[table].default_date or "")
        if current is not None:
            report.missed.append(f"{model.tables[table].business_name} has no default date in today's setup; the "
                                 f"new core counts it by {current.name} until you choose on this page")


def _filters(side: dict, table: str, names: _Names) -> list[ColumnFilter] | None:
    out = []
    for f in side.get("filters") or []:
        if not isinstance(f, dict) or not f.get("field"):
            continue
        column = names.column(table, f["field"])
        op = _OPS.get(str(f.get("operator") or "equals").lower())
        if column is None or op is None:
            return None                   # a filter on another table, or one the new core cannot spell
        raw = str(f.get("value") or "")
        values: list[Any] = [] if op in ("is_null", "not_null") else (
            _split(re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", raw)) if op in ("in", "not_in", "between") else [raw])
        out.append(ColumnFilter(column=column, op=op, values=values))   # type: ignore[arg-type]
    return out


def _side(side: dict, table: str, names: _Names, default_agg: str) -> AggExpr | None:
    agg = _AGGS.get(str(side.get("aggregation") or default_agg).upper())
    measure = str(side.get("measure") or "")
    column = names.column(table, measure)
    filters = _filters(side, table, names)
    if agg is None or column is None or filters is None:
        return None
    return AggExpr(agg=agg, column=column, filters=filters)   # type: ignore[arg-type]


def _metric_table(metric: dict, config: dict, names: _Names) -> str | None:
    field_names = [str(config.get("measure") or ""), str((config.get("numerator") or {}).get("measure") or "")]
    qualified = [f.rsplit(".", 1)[0] for f in field_names if f.count(".") >= 1]
    return names.table(metric.get("base_table"), *qualified)


def _metrics(model: SemanticModel, legacy: Legacy, names: _Names, report: Report, add_synonyms) -> None:
    taken = {m.slug for m in model.measures.values()}
    for metric in legacy.metrics:
        status = str(metric.get("metric_status") or "published").lower()
        if not int(metric.get("is_active", 1) or 0) or status in _UNANSWERABLE:
            continue
        name = str(metric.get("name") or "").strip()
        try:
            config = json.loads(metric.get("metric_builder_config") or "{}") or {}
        except (TypeError, ValueError):
            config = {}
        table = _metric_table(metric, config, names)
        mode = str(config.get("mode") or "aggregate").lower()
        expr: MeasureExpr | None = None
        percent = str(metric.get("result_format") or "").lower() in ("percentage", "percent")
        why = "written as a query or over another table's columns"
        if config.get("enabled") and table and mode in ("aggregate", "filtered_aggregate", "measure"):
            expr = _side(config, table, names, "SUM")
        elif config.get("enabled") and table and mode == "ratio":
            top = _side(config.get("numerator") or {}, table, names, "SUM")
            bottom = _side(config.get("denominator") or {}, table, names, "COUNT")
            if top and bottom:
                expr = OpExpr(op="ratio", args=[top, bottom], scale=100.0 if percent else 1.0)
        elif table and str(metric.get("sql_template") or "").strip():
            try:
                expr = _from_formula(model, table, str(metric["sql_template"]), percent)
            except formula.FormulaError as exc:
                why = f"its formula is not used: {exc}"
        if expr is None or table is None:
            report.missed.append(f"the metric {name}: {why}; define it in the new core from its parts")
            continue
        same = next((m for m in model.measures.values() if m.table == table
                     and m.expr.model_dump() == expr.model_dump()), None)
        words = _split(metric.get("synonyms"))
        if same is not None:
            if name and name != same.business_name:
                report.decisions.append(Decision(f"measure:{same.key}", "business_name", name,
                                                 "the metric's name in today's setup"))
                words = [same.business_name, *words]
            add_synonyms(f"measure:{same.key}", words)
            report.count("metrics matched")
            continue
        additivity, over_time = _behaviour(model, table, expr)
        slug = ids.unique_slug(ids.slug(name), taken)
        measure = Measure(
            key=f"import.metric_{metric.get('id')}", slug=slug, business_name=name,
            description=str(metric.get("description") or ""), synonyms={"en": words} if words else {}, table=table,
            expr=expr, additivity=additivity, time_aggregation=over_time,   # type: ignore[arg-type]
            format=_FORMATS.get(str(metric.get("result_format") or "number").lower(), "number"),   # type: ignore[arg-type]
            default_date=_metric_date(metric, legacy, model, names, table), kind="import", provenance="admin",
            status="approved")
        report.decisions.append(Decision(f"measure:{measure.key}", "define", measure.model_dump(mode="json"),
                                         f"the metric {name} from today's setup"))
        report.count("metrics added")


def _from_formula(model: SemanticModel, table: str, text: str, percent: bool) -> MeasureExpr:
    """A metric's formula as the measure it is: one aggregate, two combined, or a checked formula."""
    t = model.tables[table]
    by_name = {c.name.casefold(): c.key for c in model.columns.values() if c.table == table}
    read = formula.parse(text, lambda n: by_name.get(n.casefold()), lambda k: model.columns[k].name,
                         {t.name.casefold()})

    def simple(node: exp.Expr) -> AggExpr | None:
        while isinstance(node, exp.Paren):
            node = node.this
        if isinstance(node, exp.Nullif) and isinstance(node.expression, exp.Literal) \
                and node.expression.this in ("0", "0.0"):
            node = node.this          # a ratio already never divides by zero
        if not isinstance(node, formula.AGGREGATES):
            return None
        arg = node.this
        if isinstance(node, exp.Count) and isinstance(arg, exp.Star):
            return AggExpr(agg="count")
        if isinstance(node, exp.Count) and isinstance(arg, exp.Distinct) and len(arg.expressions) == 1 \
                and isinstance(arg.expressions[0], exp.Column):
            return AggExpr(agg="count_distinct", column=by_name[arg.expressions[0].name.casefold()])
        if isinstance(arg, exp.Column):
            agg = {exp.Sum: "sum", exp.Avg: "avg", exp.Count: "count", exp.Min: "min", exp.Max: "max"}[type(node)]
            return AggExpr(agg=agg, column=by_name[arg.name.casefold()])   # type: ignore[arg-type]
        return None

    tree = read.tree
    one = simple(tree)
    if one is not None:
        return one
    if isinstance(tree, (exp.Add, exp.Sub, exp.Div)):
        left, right = simple(tree.this), simple(tree.expression)
        if left is not None and right is not None:
            op = "add" if isinstance(tree, exp.Add) else "subtract" if isinstance(tree, exp.Sub) else "ratio"
            return OpExpr(op=op, args=[left, right],   # type: ignore[arg-type]
                          scale=100.0 if percent and op == "ratio" else 1.0)
    return SqlExpr(sql=read.sql, columns=read.columns)


def _behaviour(model: SemanticModel, table: str, expr: MeasureExpr) -> tuple[str, str | None]:
    """How an added metric adds up: never summed over time on a balance unless what it adds is a flow."""
    if isinstance(expr, AggExpr):
        additive, columns = expr.agg in ("sum", "count"), [expr.column] if expr.column else []
    elif isinstance(expr, OpExpr):
        additive = expr.op in ("add", "subtract") and all(
            isinstance(a, AggExpr) and a.agg in ("sum", "count") for a in expr.args)
        columns = [a.column for a in expr.args if isinstance(a, AggExpr) and a.column]
    elif isinstance(expr, SqlExpr):
        additive, columns = formula.additive(formula.parse_stored(expr.sql)), list(expr.columns)
    else:
        additive, columns = False, []
    if not additive:
        return "non_additive", None
    if model.tables[table].kind != "snapshot":
        return "additive", None
    learned = [m for m in model.measures.values()
               if m.table == table and isinstance(m.expr, AggExpr) and m.expr.column in columns]
    semi = next((m for m in learned if m.additivity == "semi_additive"), None)
    if semi is not None:
        return "semi_additive", semi.time_aggregation or "last"
    if learned and all(m.additivity == "additive" for m in learned):
        return "additive", None
    tokens = {t for c in columns for t in words_of.tokens(model.columns[c].name)}
    if tokens & (words_of.FLOW_WORDS | _EVENT_COUNTS) and not tokens & words_of.LEVEL_WORDS:
        return "additive", None      # a period's amount, or how many times something happened in it
    return "semi_additive", "last"


def _metric_date(metric: dict, legacy: Legacy, model: SemanticModel, names: _Names, table: str) -> str | None:
    """The date the metric was bound to today; without one it follows its table's default (None)."""
    bindings = sorted((b for b in legacy.date_contexts if b.get("metric_id") == metric.get("id")
                       and int(b.get("is_active", 1) or 0)),
                      key=lambda b: (-int(b.get("is_default") or 0), -int(b.get("priority") or 0)))
    candidates = [(b.get("fact_table"), b.get("fact_column")) for b in bindings]
    candidates.append((table, metric.get("default_time_column")))
    for fact_table, fact_column in candidates:
        t = names.table(fact_table) if fact_table != table else table
        column = names.column(t, fact_column)
        role = next((r for r in model.date_roles.values() if r.table == t and r.column == column), None)
        if role is not None and role.table == table:
            return role.key
    return None


def _descriptions(model: SemanticModel, legacy: Legacy, names: _Names, report: Report, add_synonyms) -> None:
    def targets(column: str) -> list[str]:
        """Where a column's words belong: its measures, the entity it names, or the attribute itself."""
        out = [f"measure:{m.key}" for m in model.measures.values()
               if isinstance(m.expr, AggExpr) and m.expr.column == column and not m.expr.filters]
        out += [f"entity:{e.slug}" for e in model.entities.values() if e.label_column == column]
        out += [f"attribute:{a.slug}" for a in model.attributes.values() if a.column == column]
        return out or [f"column:{column}"]

    for raw_name, entry in legacy.table_descriptions.items():
        table = names.table(entry.get("table_name") or raw_name)
        if table is None:
            report.missed.append(f"the description of {entry.get('table_name') or raw_name}: not a table the new "
                                 "core learned")
            continue
        text = str(entry.get("description") or "").strip()
        if text:
            report.decisions.append(Decision(f"table:{table}", "description", text, "described in today's setup"))
            report.count("descriptions")
        add_synonyms_to = entry.get("column_synonym_map") or {}
        for column_name, words in add_synonyms_to.items():
            column = names.column(table, column_name)
            if column is None:
                report.missed.append(f"synonyms of {raw_name}.{column_name}: not a column the new core learned")
                continue
            for target in targets(column):
                add_synonyms(target, _split(words))
    tables = _entity_tables(legacy, names)
    for prop in legacy.properties:
        table = tables.get(str(prop.get("entity_name") or ""))
        column = names.column(table, prop.get("column_name"))
        if column is None:
            continue
        if str(prop.get("role") or "").lower() == "ignore":
            report.decisions.append(Decision(f"column:{column}", "hidden", True, "ignored in today's setup"))
        _name_column(column, str(prop.get("display_name") or "").strip(), _split(prop.get("synonyms")),
                     "named in today's setup", targets, report, add_synonyms)
    codes = 0
    for meaning in legacy.meanings:
        if meaning.get("status") != "confirmed":
            continue
        reading = str(meaning.get("decided_reading") or meaning.get("reading") or "").strip()
        words = _split(meaning.get("decided_synonyms") or [])
        if meaning.get("scope") == "table":
            table = names.table(meaning.get("subject"))
            if table and reading:
                report.decisions.append(Decision(f"table:{table}", "business_name", reading, "a confirmed meaning"))
            continue
        if meaning.get("scope") != "column":
            codes += 1                    # a code read inside names: the new core names by its own reading
            continue
        # A column's reading applies to that column name wherever it stands; the
        # places also list the columns it was told apart from, which keep theirs.
        subject = ids.norm(meaning.get("subject"))
        for place in meaning.get("found_in") or []:
            table_name, _, column_name = str(place).rpartition(".")
            if ids.norm(column_name) != subject:
                continue
            column = names.column(names.table(table_name), column_name)
            if column is not None:
                _name_column(column, reading, words, "a confirmed meaning", targets, report, add_synonyms)
    if codes:
        report.missed.append(f"{codes} confirmed reading{'s' if codes > 1 else ''} of codes inside names: the new "
                             "core reads names its own way; rename on this page anything that reads wrong")


def _name_column(column: str, name: str, words: list[str], why: str, targets, report: Report, add_synonyms) -> None:
    """A column's business name and words, where it is grouped by or the column itself.

    The name of the thing a label column names (an entity) is not the column's:
    "Customer name" does not rename customers. Measures take their names from metrics.
    """
    for target in targets(column):
        if name and target.startswith(("attribute:", "column:")):
            report.decisions.append(Decision(target, "business_name", name, why))
        add_synonyms(target, words)


# ── reading today's stores and writing the decisions ───────────────────────


def read_legacy(account_id: str, kb_dir: str) -> Legacy:
    import store

    model: dict = {}
    path = Path(kb_dir) / "_semantic_model.json" if kb_dir else None
    if path is not None and path.exists():
        try:
            model = json.loads(path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            log.warning("core2 import: today's semantic model for %s is unreadable: %s", account_id, exc)
    return Legacy(
        entities=store.list_entities(account_id, active_only=False),
        relationships=store.list_relationships(account_id, active_only=False),
        metrics=store.list_metrics(account_id, active_only=True),
        date_contexts=store.list_metric_date_contexts(account_id),
        table_descriptions=store.list_table_descriptions(account_id),
        properties=store.list_all_entity_properties(account_id),
        meanings=store.list_business_meanings(account_id, statuses={"confirmed"}),
        semantic_model=model)


def _wanted(report: Report, existing: list[dict]) -> dict[tuple[str, str], Decision]:
    """The decisions to hold as imports: all but those on a field an admin decided on the new core's page."""
    admins = {(o["object_key"], o["field"]) for o in existing if o["author"] != AUTHOR}
    return {(d.object_key, d.field): d for d in report.decisions if (d.object_key, d.field) not in admins}


def write(account_id: str, db_id: int | None, report: Report) -> int:
    """Today's decisions as overrides authored "import"; an admin's own decision on the same field stands."""
    import store

    existing = store.list_core2_overrides(account_id, db_id)
    previous = {(o["object_key"], o["field"]) for o in existing if o["author"] == AUTHOR}
    wanted = _wanted(report, existing)
    for object_key, field_name in previous - set(wanted):
        store.delete_core2_override(account_id, db_id, object_key, field_name)
    for (object_key, field_name), d in wanted.items():
        store.set_core2_override(account_id, db_id, object_key, field_name, d.value, author=AUTHOR, note=d.note)
    return len(wanted)


def changes(account_id: str, db_id: int | None, report: Report) -> bool:
    """Would writing ``report`` change what came over from today's setup last time?"""
    import store

    existing = store.list_core2_overrides(account_id, db_id)
    held = {(o["object_key"], o["field"]): o["value"] for o in existing if o["author"] == AUTHOR}
    wanted = _wanted(report, existing)
    return held != {key: json.loads(json.dumps(d.value, default=str)) for key, d in wanted.items()}


def import_approvals(account_id: str, db_id: int | None, model: SemanticModel, kb_dir: str, *,
                     report: Report | None = None) -> dict[str, Any]:
    """Bring today's decisions over onto ``model`` (as learned); returns what came over and what did not.

    ``report`` is today's setup already read against ``model``, when the caller has it.
    """
    report = report or decisions(model, read_legacy(account_id, kb_dir))
    written = write(account_id, db_id, report)
    return {"written": written, "counts": report.counts, "missed": report.missed[:50]}
