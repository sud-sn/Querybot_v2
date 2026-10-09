"""A metric an admin writes: a formula over any table's fields, read into the model's form and back.

The metric editor offers every field of the model by the name people use, with its
table ("Order line · Net amount"), the metrics already defined, and the functions a
metric may use. What the admin writes is read here, never pasted into a query:

* ``SUM([Order line · Net amount])``: a field, added up (SUM, AVG, MIN, MAX, COUNT,
  COUNT(DISTINCT ...)); ``COUNT([Return])`` counts a table's rows;
* ``[Net amount] - [Refund amount]``: metrics, combined with + - * /;
* ``SUM([Order line · Quantity] * [Product · Unit cost])``: arithmetic on each row
  (with COALESCE, NULLIF and CASE WHEN), over one table and the tables its rows reach;
* a number only scales a calculation: ``... * 100`` for a percentage.

Fields may also be written as the warehouse spells them (``net_amount``,
``order_lines.net_amount``) when that names one field. Fields of different tables
can be combined: each table is added up on its own rows and the results combined
(core2.resolve.resolver), so no row is counted twice.
"""

from __future__ import annotations

import datetime as dt

import re
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from sqlglot import exp

from core2.model import formula
from core2.model.schema import AggExpr, Column, MeasureExpr, OpExpr, RefExpr, SemanticModel, SqlExpr

SEP = " · "
_REF = re.compile(r"\[([^\[\]]+)\]")
_AGGS = {exp.Sum: "sum", exp.Avg: "avg", exp.Count: "count", exp.Min: "min", exp.Max: "max"}
_ROW_NODES = tuple(n for n in formula._ALLOWED if n not in formula.AGGREGATES)  # noqa: SLF001 - one list of what a formula may use
_PLACEHOLDER = "qbref"

# The functions the editor suggests, as they are written.
FUNCTIONS = [
    ("SUM", "SUM(field)", "Add up a number"),
    ("COUNT", "COUNT(field)", "Count the rows that have a value"),
    ("COUNT DISTINCT", "COUNT(DISTINCT field)", "Count different values"),
    ("AVG", "AVG(field)", "Average of a number"),
    ("MIN", "MIN(field)", "The smallest value"),
    ("MAX", "MAX(field)", "The largest value"),
    ("COALESCE", "COALESCE(value, fallback)", "Use a fallback when a value is empty"),
    ("NULLIF", "NULLIF(value, 0)", "Treat a value as empty, so nothing is divided by it"),
    ("CASE", "CASE WHEN condition THEN value ELSE other END", "A value that depends on a condition"),
]


class AuthoringError(ValueError):
    """Why a formula is not a metric yet: said to the admin, in words."""


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def table_name(model: SemanticModel, table_key: str) -> str:
    t = model.tables[table_key]
    return t.business_name or t.name


def field_name(column: Column) -> str:
    return column.business_name or column.name


def field_ref(model: SemanticModel, column_key: str) -> str:
    """How a field is written in a formula: ``Order line · Net amount``."""
    column = model.columns[column_key]
    return f"{table_name(model, column.table)}{SEP}{field_name(column)}"


def _kind(column: Column) -> str:
    if column.data_type in ("integer", "decimal", "float") and column.role not in (
            "key", "foreign_key", "date_key", "period_key", "code", "identifier", "flag"):
        return "number"
    if column.data_type in ("date", "timestamp") or column.role in ("date", "date_key", "period_key"):
        return "date"
    if column.role == "flag" or column.data_type == "boolean":
        return "flag"
    return "text"


def _hint(column: Column) -> str:
    """What the field holds, from its profile: a range for numbers and dates, examples where allowed."""
    p = column.profile
    if p is None:
        return ""
    kind = _kind(column)
    if kind == "date":
        first, last = p.date_min or p.min, p.date_max or p.max
        return f"{first[:10]} to {last[:10]}" if first and last else ""
    if kind == "number" and p.min_num is not None and p.max_num is not None:
        return f"{p.min_num:,.2f} to {p.max_num:,.2f}"
    if column.values_allowed and p.top and column.sensitivity == "none":
        return "e.g. " + ", ".join(str(v.value) for v in p.top[:3] if v.value is not None)
    return ""


def _kind_words(column: Column, money: bool) -> str:
    """The kind of a field in a reader's words: money, whole number, number, text, date, yes or no."""
    kind = _kind(column)
    if kind == "number":
        if money or column.format == "currency":
            return "money"
        p = column.profile
        return "whole number" if column.data_type == "integer" or (p and p.integer_share == 1) else "number"
    return {"date": "date", "flag": "yes or no"}.get(kind, "text")


def _stats(column: Column) -> str:
    """How much of the field there is: "1,198 rows · 31 empty · from 2023-01-19 to 2026-06-14"."""
    p = column.profile
    if p is None or not p.rows:
        return ""
    out = [f"{p.rows:,} rows", f"{p.rows - p.non_null:,} empty"]
    if _kind(column) == "date" and (p.date_min or p.min) and (p.date_max or p.max):
        out.append(f"from {(p.date_min or p.min)[:10]} to {(p.date_max or p.max)[:10]}")
    return " · ".join(out)


def _examples(column: Column) -> list[str]:
    """The field's common values, to pick a condition's value from: only where values may be shown."""
    p = column.profile
    if p is None or not p.top or not column.values_allowed or column.sensitivity != "none":
        return []
    return [str(v.value) for v in p.top[:12] if v.value is not None]


def fields(model: SemanticModel, allowed_tables: set[str] | None = None) -> list[dict[str, Any]]:
    """Every field a metric may read, for the editor to suggest as the admin types."""
    # A field a money metric adds up is money.
    money = {m.expr.column for m in model.measures.values()
             if m.format == "currency" and isinstance(m.expr, AggExpr) and m.expr.column}
    out = []
    for column in model.columns.values():
        table = model.tables.get(column.table)
        if table is None or table.hidden or column.hidden or table.kind == "calendar":
            continue
        if allowed_tables is not None and column.table not in allowed_tables:
            continue
        out.append({"ref": field_ref(model, column.key), "key": column.key, "table": table_name(model, column.table),
                    "field": field_name(column), "name": column.name, "kind": _kind(column), "role": column.role,
                    "hint": _hint(column), "description": column.description, "kind_words": _kind_words(column, column.key in money),
                    "stats": _stats(column), "examples": _examples(column),
                    "sensitive": column.sensitivity != "none"})
    return sorted(out, key=lambda f: (f["kind"] != "number", f["table"].casefold(), f["field"].casefold()))


def metrics(model: SemanticModel) -> list[dict[str, Any]]:
    from core2.plan.catalog import definition

    return sorted(({"ref": m.business_name, "key": m.key, "slug": m.slug, "table": table_name(model, m.table),
                    "definition": definition(model, m.expr, False)}
                   for m in model.measures.values() if not m.hidden and m.table in model.tables),
                  key=lambda m: m["ref"].casefold())


def links(model: SemanticModel) -> list[dict[str, Any]]:
    """The tables that join, and on which columns, for the editor to say how a field from another table is reached."""
    out = []
    for j in model.joins.values():
        if j.trust == "rejected" or j.to_calendar or j.from_table not in model.tables or j.to_table not in model.tables:
            continue
        out.append({"from": table_name(model, j.from_table), "to": table_name(model, j.to_table),
                    "on": ", ".join(model.columns[c].name for c in j.from_columns if c in model.columns)})
    return out


def broken_down_by(model: SemanticModel, table_key: str, limit: int = 2) -> list[str]:
    """What a metric on this table can be broken down by: the things its rows are about, joined one step away."""
    out = []
    for j in sorted(model.joins.values(), key=lambda j: (j.trust != "admin", j.key)):
        if (j.from_table == table_key and j.trust != "rejected" and not j.to_calendar and j.to_table in model.tables
                and model.tables[j.to_table].kind == "dimension" and j.cardinality in ("many_to_one", "one_to_one")):
            name = table_name(model, j.to_table).lower()
            if name not in out:
                out.append(name)
    return out[:limit]


def tables(model: SemanticModel) -> list[dict[str, Any]]:
    return sorted(({"ref": table_name(model, t.key), "key": t.key, "kind": t.kind}
                   for t in model.tables.values() if not t.hidden and t.kind != "calendar"),
                  key=lambda t: t["ref"].casefold())


# ── reading a formula ─────────────────────────────────────────────────────────

@dataclass
class Read:
    expr: MeasureExpr
    table: str                                       # the table it is counted on first (its base)
    columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)  # every table whose rows it adds up or reads


class _Names:
    """What a reference in a formula can name: a field, a table, a metric."""

    def __init__(self, model: SemanticModel):
        self.model = model
        self.fields: dict[str, list[str]] = {}
        self.bare: dict[str, list[str]] = {}
        for c in model.columns.values():
            t = model.tables.get(c.table)
            if t is None or t.kind == "calendar":
                continue
            for table_word in {table_name(model, t.key), t.name, t.slug}:
                for field_word in {field_name(c), c.name}:
                    self.fields.setdefault(_plain(f"{table_word} {field_word}"), []).append(c.key)
            for field_word in {field_name(c), c.name}:
                self.bare.setdefault(_plain(field_word), []).append(c.key)
        self.tables = {_plain(w): t.key for t in model.tables.values() if t.kind != "calendar"
                       for w in (table_name(model, t.key), t.name, t.slug) if w}
        self.metrics = {_plain(w): m.key for m in model.measures.values()
                        for w in (m.business_name, m.slug) if w}

    def field(self, text: str) -> str | None:
        """A field by "Table · Field", "table.column", or a name only one field has."""
        if SEP.strip() in text or "." in text:
            head, _, tail = text.replace(SEP.strip(), ".").rpartition(".")
            found = list(dict.fromkeys(self.fields.get(_plain(f"{head} {tail}"), [])))
            if len(found) == 1:
                return found[0]
            if len(found) > 1:
                raise AuthoringError(f"{text} names more than one field")
        found = list(dict.fromkeys(self.bare.get(_plain(text), [])))
        if len(found) > 1:
            where = ", ".join(sorted({field_ref(self.model, k) for k in found})[:4])
            raise AuthoringError(f"{text} is a field of more than one table: write which ({where})")
        return found[0] if found else None


def read(model: SemanticModel, text: str) -> Read:
    """``text`` as a metric's definition, or :class:`AuthoringError` saying why not."""
    text = str(text or "").strip().rstrip(";")
    if not text:
        raise AuthoringError("write what to count: a field added up, such as SUM([Order line · Net amount])")
    names = _Names(model)
    # placeholder -> what it can name; which it is depends on where it is used
    # ("Net amount" is a metric on its own and a field inside SUM(...)).
    targets: dict[str, _Target] = {}

    def bracket(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        target = _Target(inner)
        try:
            target.field = names.field(inner)
        except AuthoringError as exc:
            target.problem = str(exc)
        target.metric = names.metrics.get(_plain(inner))
        target.table = names.tables.get(_plain(inner))
        if not (target.field or target.metric or target.table or target.problem):
            raise AuthoringError(f"there is no field, metric or table called {inner}")
        placeholder = f"{_PLACEHOLDER}{len(targets)}"
        targets[placeholder] = target
        return placeholder

    written = _REF.sub(bracket, text).replace("×", "*").replace("÷", "/").replace("−", "-")
    try:
        tree = sqlglot.parse_one(written)
    except sqlglot.errors.ParseError:
        raise AuthoringError("it cannot be read as a formula: check the brackets and the operators") from None
    reader = _Reader(model, names, targets)
    expr = reader.measure(tree)
    if not reader.tables:
        raise AuthoringError("it reads no field: COUNT(*) needs a table, such as COUNT([Order line])")
    base = reader.tables[0]
    return Read(expr=expr, table=base, columns=list(dict.fromkeys(reader.columns)),
                tables=list(dict.fromkeys(reader.tables)))


@dataclass
class _Target:
    """A bracketed name in a formula, and everything it could name."""

    text: str
    field: str | None = None
    metric: str | None = None
    table: str | None = None
    problem: str = ""            # why it names no single field, when it is used as one


class _Reader:
    def __init__(self, model: SemanticModel, names: _Names, targets: dict[str, _Target]):
        self.model = model
        self.names = names
        self.targets = targets
        self.columns: list[str] = []
        self.tables: list[str] = []

    def _target(self, node: exp.Expr) -> _Target | None:
        if isinstance(node, exp.Column) and not node.table and node.name in self.targets:
            return self.targets[node.name]
        return None

    def _column(self, node: exp.Column) -> str:
        """The field a column in the formula names (a reference, or the warehouse's own spelling)."""
        target = self._target(node)
        if target is not None:
            if target.field is None:
                if target.problem:
                    raise AuthoringError(target.problem)
                what = "a metric" if target.metric else "a table"
                raise AuthoringError(f"{target.text} is {what}, not a field to add up")
            key = target.field
        else:
            spelled = f"{node.table}.{node.name}" if node.table else node.name
            found = self.names.field(spelled)
            if found is None:
                raise AuthoringError(f"there is no field called {spelled}")
            key = found
        self.columns.append(key)
        self.tables.append(self.model.columns[key].table)
        return key

    # ── outside an aggregate: metrics, and how they combine ────────────────
    def measure(self, node: exp.Expr) -> MeasureExpr:
        if isinstance(node, exp.Paren):
            return self.measure(node.this)
        if isinstance(node, formula.AGGREGATES):
            return self.aggregate(node)
        if isinstance(node, exp.Column):
            target = self._target(node)
            metric_key = target.metric if target is not None else (
                None if node.table else self.names.metrics.get(_plain(node.name)))
            if metric_key is not None:
                metric = self.model.measures[metric_key]
                self.tables.append(metric.table)
                return RefExpr(measure=metric.key)
            what = f"[{target.text}]" if target is not None else node.sql()
            if target is not None and target.table and not target.field:
                raise AuthoringError(f"{target.text} is a table: count its rows with COUNT({what})")
            raise AuthoringError(f"{what} is a field: say how to add it up, such as SUM({what})")
        if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return self.combine(node)
        if isinstance(node, exp.Neg):
            raise AuthoringError("a metric cannot start with a minus: subtract it from another one")
        if isinstance(node, exp.Literal):
            raise AuthoringError("a number on its own is not a metric: it can only scale one (... * 100)")
        raise AuthoringError(f"it uses {node.key.upper()}, which a metric does not use outside SUM, COUNT or AVG")

    def combine(self, node: exp.Expr) -> MeasureExpr:
        left, right = node.this, node.expression
        number = next((n for n in (right, left) if isinstance(n, exp.Literal) and not n.is_string), None)
        if number is not None:
            other = left if number is right else right
            value = float(number.this)
            if isinstance(node, exp.Mul):
                scale = value
            elif isinstance(node, exp.Div) and number is right and value:
                scale = 1 / value
            else:
                raise AuthoringError("a number can only multiply or divide a calculation (... * 100, ... / 1000)")
            inner = self.measure(other)
            if not isinstance(inner, OpExpr):
                raise AuthoringError("a number can only scale a calculation of two values, such as a ratio")
            return OpExpr(op=inner.op, args=inner.args, scale=inner.scale * scale)
        op = {exp.Add: "add", exp.Sub: "subtract", exp.Mul: "multiply", exp.Div: "ratio"}[type(node)]
        return OpExpr(op=op, args=[self.measure(left), self.measure(right)])

    # ── an aggregate: one field, or a formula on each row ──────────────────
    def aggregate(self, node: exp.Expr) -> MeasureExpr:
        agg = _AGGS[type(node)]
        arg = node.this
        if isinstance(node, exp.Count) and isinstance(arg, exp.Star):
            raise AuthoringError("COUNT(*) needs its table: write COUNT([Order line]) to count its rows")
        if isinstance(node, exp.Count) and isinstance(arg, exp.Column):
            target = self._target(arg)
            if target is not None and target.table and not target.field:
                self.tables.append(target.table)
                return AggExpr(agg="count", table=target.table)
        if isinstance(node, exp.Count) and isinstance(arg, exp.Distinct):
            if len(arg.expressions) != 1 or not isinstance(arg.expressions[0], exp.Column):
                raise AuthoringError("COUNT(DISTINCT ...) counts the different values of one field")
            return AggExpr(agg="count_distinct", column=self._column(arg.expressions[0]))
        if isinstance(arg, exp.Column):
            return AggExpr(agg=agg, column=self._column(arg))
        return self.row_formula(node)

    def row_formula(self, node: exp.Expr) -> SqlExpr:
        """An aggregate over arithmetic on each row: its fields written by their keys."""
        tree = node.copy()
        for inner in tree.walk():
            if isinstance(inner, formula.AGGREGATES) and inner is not tree:
                raise AuthoringError("an aggregate cannot hold another one (SUM inside SUM)")
            if not isinstance(inner, (*_ROW_NODES, *formula.AGGREGATES)):
                raise AuthoringError(f"it uses {inner.key.upper()}, which a metric does not use")
        keys: list[str] = []
        for column in list(tree.find_all(exp.Column)):
            key = self._column(column)
            column.replace(exp.column(exp.to_identifier(key, quoted=True)))
            keys.append(key)
        if not keys:
            raise AuthoringError("it adds up no field")
        homes = list(dict.fromkeys(self.model.columns[k].table for k in keys))
        if len(homes) > 1:
            from core2.resolve.resolver import _reaches

            if not any(all(o == t or _reaches(self.model, t, o) for o in homes) for t in homes):
                raise AuthoringError("the fields of a calculation on each row must be on one row: "
                                     f"{', '.join(table_name(self.model, t) for t in homes)} are not linked that way")
        return SqlExpr(sql=tree.sql(), columns=list(dict.fromkeys(keys)))


# ── back to text, for the editor ─────────────────────────────────────────────

def to_text(model: SemanticModel, expr: MeasureExpr, *, top: bool = True) -> str:
    """A metric's definition as the admin writes it (``SUM([Order line · Net amount])``)."""
    if isinstance(expr, AggExpr):
        if expr.column is None:
            return f"COUNT([{table_name(model, expr.table)}])" if expr.table else "COUNT(*)"
        ref = f"[{field_ref(model, expr.column)}]"
        return f"COUNT(DISTINCT {ref})" if expr.agg == "count_distinct" else f"{expr.agg.upper()}({ref})"
    if isinstance(expr, RefExpr):
        metric = model.measures.get(expr.measure)
        return f"[{metric.business_name}]" if metric else f"[{expr.measure}]"
    if isinstance(expr, OpExpr):
        sign = {"ratio": "/", "subtract": "-", "add": "+", "multiply": "*"}[expr.op]
        text = f" {sign} ".join(to_text(model, a, top=False) for a in expr.args)
        if expr.scale != 1:
            scale = f"{expr.scale:g}" if expr.scale >= 1 else f"{1 / expr.scale:g}"
            return f"({text}) {'*' if expr.scale >= 1 else '/'} {scale}"
        return text if top else f"({text})"
    if isinstance(expr, SqlExpr):
        tree = formula.parse_stored(expr.sql)
        for column in list(tree.find_all(exp.Column)):
            if column.name in model.columns:
                column.replace(exp.column(f"{_PLACEHOLDER}_{list(model.columns).index(column.name)}"))
        text = tree.sql()
        return re.sub(rf"{_PLACEHOLDER}_(\d+)", lambda m: f"[{field_ref(model, list(model.columns)[int(m.group(1))])}]",
                      text)
    return ""


# ── checked against the data, before it is saved ──────────────────────────────

CHECK_MONTHS = 6
_DRAFT = "__draft__"


def _month_start(day: Any) -> Any:
    return day.replace(day=1)


def _months_back(day: Any, n: int) -> Any:
    month = day.month - 1 - n
    return day.replace(year=day.year + month // 12, month=month % 12 + 1, day=1)


def check(model: SemanticModel, measure: Any, warehouse: Any, today: Any) -> dict[str, Any]:
    """A draft metric run on the data: its last months by its date, and what its conditions leave out.

    ``model`` is the workspace's own copy: the draft is added to it to be resolved like any
    measure, so what is checked is what a question will run. Returns what the editor shows;
    a definition that cannot run says why (``problem``).
    """
    from core2.compile.compiler import CompileError, compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, ResolveError, _date_role_for, resolve

    draft = measure.model_copy(update={"key": _DRAFT, "slug": _DRAFT, "hidden": False})
    model.measures[_DRAFT] = draft
    out: dict[str, Any] = {"months": [], "latest": None, "rows": None, "left_out": None, "empty": None,
                           "notes": [], "problem": ""}
    plan_base = Plan.model_validate({"kind": "query", "measures": [_DRAFT]})
    role = _date_role_for(model, plan_base, draft, draft.table)
    try:
        if role is not None and role.last:
            last = role.last
            window = {"kind": "between", "start": _months_back(last, CHECK_MONTHS - 1).isoformat(),
                      "end": last.isoformat()}
            trend = Plan.model_validate({"kind": "query", "intent": "trend", "measures": [_DRAFT],
                                         "time": {"grain": "month", "window": window}})
            logical = resolve(trend, model, Context(today=today))
            compiled = compile_query(logical, model, warehouse.dialect)
            result = warehouse.query(compiled.sql, max_rows=CHECK_MONTHS + 1)
            value = next(c.name for c in compiled.columns if c.role == "measure")
            period = next((c.name for c in compiled.columns if c.role == "period"), None)
            for row in result.rows:
                record = dict(zip(result.columns, row))
                out["months"].append({"period": str(record.get(period) or "")[:7],
                                      "value": _number(record.get(value))})
            out["notes"] = [n for n in logical.notes if _DRAFT not in n]
            latest_window = {"kind": "between", "start": _month_start(last).isoformat(), "end": last.isoformat()}
            # A month the data stops part way through is said so: it is not a drop.
            month_end = _months_back(last, -1) - dt.timedelta(days=1)
            out["latest"] = {"period": _month_start(last).isoformat()[:7], "date": role.name,
                             "through": last.isoformat() if last < month_end else ""}
        else:
            single = Plan.model_validate({"kind": "query", "intent": "value", "measures": [_DRAFT]})
            logical = resolve(single, model, Context(today=today))
            compiled = compile_query(logical, model, warehouse.dialect)
            result = warehouse.query(compiled.sql, max_rows=2)
            value = next(c.name for c in compiled.columns if c.role == "measure")
            total = dict(zip(result.columns, result.rows[0])).get(value) if result.rows else None
            out["months"].append({"period": "", "value": _number(total)})
            out["notes"] = [n for n in logical.notes if _DRAFT not in n]
            latest_window = None
        out.update(_row_counts(model, draft, warehouse, today, latest_window))
    except (ResolveError, CompileError) as exc:
        out["problem"] = str(getattr(exc, "message", "") or exc)
    except Exception as exc:  # noqa: BLE001 - the warehouse's own refusal is the admin's to read
        out["problem"] = f"The data refused it: {str(exc)[:300]}"
    finally:
        model.measures.pop(_DRAFT, None)
        for key in [k for k in model.measures if k.startswith(_DRAFT)]:
            model.measures.pop(key, None)
    return out


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _row_counts(model: SemanticModel, draft: Any, warehouse: Any, today: Any,
                window: dict[str, Any] | None) -> dict[str, Any]:
    """On the draft's own table, in the latest month: rows counted, rows its conditions leave out,
    and rows with no value to add up."""
    from core2.compile.compiler import compile_query
    from core2.model.schema import Measure
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    expr = draft.expr
    conditions = list(draft.filters) + (list(expr.filters) if isinstance(expr, AggExpr) else [])
    counts = {
        f"{_DRAFT}all": AggExpr(agg="count", table=draft.table),
        f"{_DRAFT}kept": AggExpr(agg="count", table=draft.table, filters=conditions),
    }
    if isinstance(expr, AggExpr) and expr.column and expr.agg != "count":
        counts[f"{_DRAFT}valued"] = AggExpr(agg="count", column=expr.column, filters=conditions)
    for key, count in counts.items():
        model.measures[key] = Measure(key=key, slug=key, business_name=key, table=draft.table, expr=count,
                                      default_date=draft.default_date, format="count")
    plan: dict[str, Any] = {"kind": "query", "intent": "value", "measures": list(counts)}
    if window:
        plan["time"] = {"window": window}
    logical = resolve(Plan.model_validate(plan), model, Context(today=today))
    compiled = compile_query(logical, model, warehouse.dialect)
    result = warehouse.query(compiled.sql, max_rows=2)
    row = dict(zip(result.columns, result.rows[0])) if result.rows else {}
    by_key = {c.measure: row.get(c.name) for c in compiled.columns if c.role == "measure"}
    every, kept = int(_number(by_key.get(f"{_DRAFT}all")) or 0), int(_number(by_key.get(f"{_DRAFT}kept")) or 0)
    valued = by_key.get(f"{_DRAFT}valued")
    return {"rows": kept, "left_out": every - kept if conditions else 0,
            "empty": kept - int(_number(valued) or 0) if valued is not None else None,
            "empty_field": field_name(model.columns[expr.column]) if valued is not None else ""}


# ── written by the AI from the admin's words ─────────────────────────────────

_OPS = ("eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between", "is_null", "not_null", "contains",
        "starts_with")
_FORMATS = ("currency", "percent", "count", "number", "integer")


def describe_prompt(model: SemanticModel) -> str:
    """The stable half: how a metric is written, and every field and metric it may use.

    Field names, kinds and descriptions; common values only for a field whose values may be
    shown to the AI (the admin's setting, as everywhere else) and that holds nothing sensitive.
    """
    lines = [
        "You write one business metric for QueryBot's semantic layer, as JSON, from an admin's description.",
        "Use only the fields and metrics listed below, written exactly as listed, in square brackets.",
        "The formula: SUM, AVG, MIN, MAX, COUNT or COUNT(DISTINCT ...) of one field, as SUM([Table · Field]);",
        "COUNT([Table]) counts a table's rows; a metric by its name, as [Net amount]; combine them with + - * /;",
        "a number only scales a calculation (... * 100 for a percentage); arithmetic on each row goes inside one",
        "aggregate, as SUM([Table · Quantity] * [Table · Unit price]). Fields of different tables may be combined:",
        "each table is added up on its own rows.",
        "Rows to leave out are conditions, not part of the formula: each {\"field\": \"[Table · Field]\", \"op\": one of",
        f"{', '.join(_OPS)}, \"values\": [...]}}, with values as the data holds them (a code, when the field lists",
        "its codes).",
        'Reply with one JSON object: {"formula": "...", "conditions": [...], "name": "...", "synonyms": ["..."],',
        '"description": "one sentence for a business reader", "format": one of ' + ", ".join(_FORMATS) + ",",
        '"assumptions": ["what you assumed, in a few words"], "question": "what to ask the admin when the',
        'description could mean two different things, otherwise an empty string"}.',
        "",
        "FIELDS (reference | kind | what it holds):",
    ]
    for f in fields(model):
        if f["sensitive"]:
            continue
        hint = f["description"] or ""
        if f["hint"] and f["hint"].startswith("e.g."):
            hint = f"{hint} {f['hint']}".strip()
        lines.append(f"[{f['ref']}] | {f['kind']}" + (f" | {hint}" if hint else ""))
    lines += ["", "METRICS (name | how it is counted):"]
    lines += [f"[{m['ref']}] | {m['definition']}" for m in metrics(model)]
    return "\n".join(lines)


def describe(model: SemanticModel, description: str, complete: Any) -> dict[str, Any]:
    """The AI's metric for ``description``, read and checked like one the admin typed.

    Returns what the editor fills in; anything the AI wrote that cannot be read comes back
    with ``problem``, and the formula as written, for the admin to correct.
    """
    import json

    text = str(description or "").strip()
    if not text:
        return {"problem": "Say what the metric counts, in a sentence."}
    try:
        reply = complete(describe_prompt(model), f"The admin's description: {text}")
    except Exception as exc:  # noqa: BLE001 - the provider's refusal is the admin's to read
        return {"problem": f"The AI could not be reached: {str(exc)[:200]}"}
    match = re.search(r"\{.*\}", str(reply or ""), re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict) or not data.get("formula"):
        return {"problem": "The AI did not write a formula. Describe it again, or write the formula yourself."}
    out: dict[str, Any] = {
        "formula": str(data.get("formula") or ""), "name": " ".join(str(data.get("name") or "").split())[:120],
        "synonyms": [str(s).strip() for s in (data.get("synonyms") or []) if str(s).strip()][:12],
        "description": str(data.get("description") or "").strip()[:500],
        "format": data.get("format") if data.get("format") in _FORMATS else "number",
        "assumptions": [str(a) for a in (data.get("assumptions") or [])][:6],
        "question": str(data.get("question") or "").strip()[:300], "conditions": [], "problem": ""}
    try:
        written = read(model, out["formula"])
        out["formula"] = to_text(model, written.expr)
    except AuthoringError as exc:
        out["problem"] = f"The AI's formula needs a correction: {exc}"
    names = _Names(model)
    for c in data.get("conditions") or []:
        if not isinstance(c, dict):
            continue
        try:
            column = names.field(_REF.sub(lambda m: m.group(1), str(c.get("field") or "")).strip())
        except AuthoringError:
            column = None
        op = str(c.get("op") or "")
        if column is None or op not in _OPS:
            out["problem"] = out["problem"] or f"A condition the AI wrote cannot be read: {c.get('field')}"
            continue
        values = c.get("values") if isinstance(c.get("values"), list) else [c.get("values")]
        out["conditions"].append({"column": column, "op": op,
                                  "values": [v for v in values if v is not None and str(v).strip()]})
    return out


# ── the metrics page ─────────────────────────────────────────────────────────

OVER_TIME = {
    "added": ("additive", None, "Added up over time"),
    "per_period": ("non_additive", None, "Worked out per period"),
    "end": ("semi_additive", "last", "Taken at the end of each period"),
    "average": ("semi_additive", "average", "Averaged over the period"),
}
FORMAT_WORDS = {"currency": "Money", "percent": "Percentage", "count": "Count", "number": "Number",
                "integer": "Whole number"}
ADMIN_PREFIX = "admin:"


def over_time(additivity: str, time_aggregation: str | None) -> str:
    for key, (a, t, _) in OVER_TIME.items():
        if a == additivity and (t == time_aggregation or (a != "semi_additive")):
            return key
    return "added"


def source(m: Any) -> str:
    if m.hidden:
        return "Hidden"
    if m.key.startswith(ADMIN_PREFIX):
        return "Added by an admin"
    if m.provenance == "admin":
        return "Changed by an admin"
    if m.provenance == "import":
        return "From today's setup"
    return "Learned"


def counted_on(model: SemanticModel, m: Any) -> list[str]:
    """The tables a metric adds up rows of (one for most; two for sales less refunds)."""
    from core2.resolve.resolver import ResolveError, _homes

    try:
        homes = _homes(model, m.table, m.expr)
    except (ResolveError, KeyError):
        homes = {m.table}
    return sorted(table_name(model, t) for t in homes if t in model.tables)


def listing(model: SemanticModel, uses: dict[str, int]) -> list[dict[str, Any]]:
    from core2.plan.catalog import definition

    out = []
    for m in model.measures.values():
        if m.table not in model.tables:
            continue
        role = model.date_roles.get(m.default_date or model.tables[m.table].default_date or "")
        conditions = [f for f in m.filters if f.column in model.columns]
        how = definition(model, m.expr, True)
        if conditions:
            from core2.model.links import condition_text

            how += ", only rows where " + "; ".join(condition_text(model, f) for f in conditions)
        out.append({"key": m.key, "slug": m.slug, "name": m.business_name or m.slug, "how": how,
                    "from": counted_on(model, m), "date": role.name if role else "", "asked": uses.get(m.slug, 0),
                    "source": source(m), "hidden": m.hidden, "description": m.description,
                    "synonyms": sorted({w for ws in m.synonyms.values() for w in ws})})
    # The ones people ask for most first; the hidden ones last.
    return sorted(out, key=lambda r: (r["hidden"], -r["asked"], r["name"].casefold()))


def editor_state(model: SemanticModel, m: Any | None) -> dict[str, Any]:
    """A metric as the editor shows it: its formula as text, conditions, names and behaviour."""
    if m is None:
        return {"key": "", "formula": "", "conditions": [], "name": "", "synonyms": [], "description": "",
                "format": "number", "over_time": "", "default_date": "", "hidden": False, "table": ""}
    return {"key": m.key, "formula": to_text(model, m.expr),
            "conditions": [{"column": f.column, "op": f.op, "values": list(f.values)} for f in m.filters],
            "name": m.business_name, "synonyms": sorted({w for ws in m.synonyms.values() for w in ws}),
            "description": m.description, "format": m.format, "over_time": over_time(m.additivity, m.time_aggregation),
            "default_date": m.default_date or "", "hidden": m.hidden, "table": m.table}


def parts(model: SemanticModel, expr: MeasureExpr) -> list[dict[str, str]]:
    """The formula as blocks: each part (what is added up, how, on which table) and the signs between."""
    out: list[dict[str, str]] = []

    def walk(e: MeasureExpr, top: bool) -> None:
        if isinstance(e, OpExpr):
            if not top:
                out.append({"kind": "open", "text": "("})
            for i, arg in enumerate(e.args):
                if i:
                    out.append({"kind": "op", "text": {"ratio": "÷", "subtract": "−", "add": "+",
                                                       "multiply": "×"}[e.op]})
                walk(arg, False)
            if not top:
                out.append({"kind": "close", "text": ")"})
            if e.scale != 1:
                out.append({"kind": "op", "text": f"× {e.scale:g}" if e.scale >= 1 else f"÷ {1 / e.scale:g}"})
            return
        if isinstance(e, AggExpr):
            target = field_name(model.columns[e.column]) if e.column else "Rows"
            table = table_name(model, model.columns[e.column].table) if e.column else table_name(
                model, e.table) if e.table in model.tables else ""
            how = {"sum": "Sum", "avg": "Average", "count": "Count", "count_distinct": "Different values",
                   "min": "Smallest", "max": "Largest"}[e.agg]
            out.append({"kind": "part", "text": target, "how": f"{how} · {table}" if table else how})
        elif isinstance(e, RefExpr):
            metric = model.measures.get(e.measure)
            out.append({"kind": "part", "text": metric.business_name if metric else e.measure,
                        "how": f"Metric · {table_name(model, metric.table)}" if metric and metric.table in model.tables
                        else "Metric"})
        elif isinstance(e, SqlExpr):
            tables_read = sorted({table_name(model, model.columns[c].table) for c in e.columns if c in model.columns})
            out.append({"kind": "part", "text": to_text(model, e), "how": "Each row · " + ", ".join(tables_read)})

    walk(expr, True)
    return out
