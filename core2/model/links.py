"""How the tables connect, for the admin to see and change: links, their conditions, and the rows a table leaves out.

The Relationships page shows the model's links (learned, declared or an admin's), each
with its columns, how many rows it matches and any condition it keeps rows by, and each
table with the rows it always leaves out. A change is checked against the data before it
is saved, and saved as an admin decision (core2.model.overrides): a learned link changed
is turned off and the admin's link added, so the next Learn keeps the decision.

The checks run read-only queries written through the compiler's own helpers, for the
workspace's warehouse: aggregates only, no row leaves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from sqlglot import exp

from core2 import ids
from core2.model.schema import ColumnFilter, Join, SemanticModel
from core2.warehouse import dialect as D

log = logging.getLogger("querybot.core2")

KIND_WORDS = {"fact": "Fact", "snapshot": "Snapshot", "dimension": "Dimension", "bridge": "Bridge",
              "calendar": "Date", "other": "Table"}
KIND_CHOICES = {"fact": "Events (things that happen)", "snapshot": "Balances (a level at a point in time)",
                "dimension": "Things (what events are about)", "bridge": "Links between things", "other": "Other"}
CARDINALITY_WORDS = {"many_to_one": "Many to one", "one_to_one": "One to one", "one_to_many": "One to many",
                     "many_to_many": "Many to many"}
OP_WORDS = {"eq": "is", "ne": "is not", "in": "is one of", "not_in": "is not one of", "gt": "is more than",
            "gte": "is at least", "lt": "is less than", "lte": "is at most", "between": "is between",
            "is_null": "is empty", "not_null": "is filled", "contains": "contains", "starts_with": "starts with"}
_FAMILY = {"integer": "number", "decimal": "number", "float": "number", "text": "text", "date": "date",
           "timestamp": "date", "boolean": "flag"}


def _table_name(model: SemanticModel, key: str) -> str:
    t = model.tables[key]
    return t.business_name or t.name


def _column_name(model: SemanticModel, key: str) -> str:
    c = model.columns[key]
    return c.business_name or c.name


def condition_text(model: SemanticModel, f: ColumnFilter) -> str:
    column = model.columns.get(f.column)
    if column is None:
        return ""
    values = ", ".join(str(v) for v in f.values)
    words = OP_WORDS.get(f.op, f.op)
    return f"{_table_name(model, column.table)} · {_column_name(model, f.column)} {words}{' ' + values if values else ''}"


def value_problem(model: SemanticModel, f: ColumnFilter) -> str:
    """Why a condition's values cannot be compared with its column (a word against numbers), or ""."""
    import datetime as _dt

    column = model.columns[f.column]
    if f.op in ("is_null", "not_null"):
        return ""
    if f.op in ("contains", "starts_with"):
        return "" if column.data_type == "text" else f"{_column_name(model, f.column)} is not text: compare it with is."
    family = _FAMILY.get(column.data_type)
    for value in f.values:
        text = str(value).strip()
        if family == "number":
            try:
                float(text)
            except ValueError:
                return f"{_column_name(model, f.column)} holds numbers: {text} is not one."
        elif family == "date":
            try:
                _dt.date.fromisoformat(text[:10])
            except ValueError:
                return f"{_column_name(model, f.column)} holds dates: write them as 2026-01-31, not {text}."
    if f.op == "between" and len(f.values) != 2:
        return "Between takes two values: the first and the last."
    return ""


def status(j: Join) -> str:
    if j.trust == "rejected":
        return "Turned off"
    if j.trust in ("admin", "declared", "verified") or j.status == "approved":
        return "Confirmed"
    return "Suggested"


def view(model: SemanticModel) -> dict[str, Any]:
    """Everything the Relationships page draws: tables, links and the columns to pick from."""
    tables = []
    for t in sorted(model.tables.values(), key=lambda t: (t.kind == "calendar", _table_name(model, t.key).casefold())):
        if t.hidden:
            continue
        roles = [r for r in model.date_roles.values() if r.table == t.key]
        tables.append({
            "key": t.key, "name": _table_name(model, t.key), "physical": t.name, "kind": t.kind,
            "kind_words": KIND_WORDS.get(t.kind, t.kind), "rows": t.row_count, "default_date": t.default_date or "",
            "dates": [{"key": r.key, "name": r.name} for r in sorted(roles, key=lambda r: r.name)],
            "leaves_out": [{"column": f.column, "op": f.op, "values": list(f.values), "text": condition_text(model, f)}
                           for f in t.default_filters],
            "readers_may_include": t.readers_may_include,
            "measures": sum(1 for m in model.measures.values() if m.table == t.key and not m.hidden),
            "columns": [{"key": c.key, "name": _column_name(model, c.key), "physical": c.name,
                         "type": _FAMILY.get(c.data_type, "other"), "role": c.role,
                         "examples": [str(v.value) for v in (c.profile.top or [])[:6] if v.value is not None]
                         if c.profile and c.values_allowed and c.sensitivity == "none" else []}
                        for c in model.table_columns(t.key) if not c.hidden],
        })
    joins = []
    for j in sorted(model.joins.values(), key=lambda j: (_table_name(model, j.from_table).casefold(), j.key)):
        if j.from_table not in model.tables or j.to_table not in model.tables:
            continue
        joins.append({
            "key": j.key, "from": j.from_table, "to": j.to_table,
            "from_name": _table_name(model, j.from_table), "to_name": _table_name(model, j.to_table),
            "pairs": [{"from": f, "to": t, "from_name": model.columns[f].name, "to_name": model.columns[t].name}
                      for f, t in zip(j.from_columns, j.to_columns) if f in model.columns and t in model.columns],
            "cardinality": j.cardinality, "rows_words": CARDINALITY_WORDS.get(j.cardinality, j.cardinality),
            "role": j.role or "", "trust": j.trust, "status": status(j), "match": round(j.match_rate, 4),
            "to_calendar": j.to_calendar, "admin": j.provenance == "admin" and j.trust == "admin",
            "keep_unmatched": j.keep_unmatched,
            "conditions": [{"column": f.column, "op": f.op, "values": list(f.values),
                            "text": condition_text(model, f)} for f in j.conditions],
        })
    live = [j for j in joins if j["status"] != "Turned off"]
    return {"tables": tables, "joins": joins,
            "counts": {"all": len(live), "review": sum(1 for j in live if j["status"] == "Suggested"),
                       "conditions": sum(1 for j in live if j["conditions"])}}


# ── a link, as the admin edits it ────────────────────────────────────────────

@dataclass
class LinkSpec:
    from_table: str
    to_table: str
    pairs: list[tuple[str, str]]                    # (from column key, to column key)
    conditions: list[ColumnFilter] = field(default_factory=list)
    role: str = ""
    cardinality: str = "many_to_one"
    keep_unmatched: bool = True


class LinkError(ValueError):
    """Why a link cannot be saved, in words."""


def spec_from(model: SemanticModel, data: dict[str, Any]) -> LinkSpec:
    """A link as the page sends it, checked against the model."""
    from_table, to_table = str(data.get("from") or ""), str(data.get("to") or "")
    if from_table not in model.tables or to_table not in model.tables:
        raise LinkError("Choose the two tables the link joins.")
    if from_table == to_table:
        raise LinkError("A link joins two different tables.")
    pairs = []
    for pair in data.get("pairs") or []:
        f, t = str(pair.get("from") or ""), str(pair.get("to") or "")
        if model.columns.get(f) is None or model.columns[f].table != from_table:
            raise LinkError(f"Choose a column of {_table_name(model, from_table)} to match on.")
        if model.columns.get(t) is None or model.columns[t].table != to_table:
            raise LinkError(f"Choose a column of {_table_name(model, to_table)} to match on.")
        pairs.append((f, t))
    if not pairs:
        raise LinkError("A link matches at least one column of each table.")
    if len(set(pairs)) != len(pairs):
        raise LinkError("The same pair of columns is matched twice.")
    conditions = []
    for c in data.get("conditions") or []:
        try:
            f = ColumnFilter.model_validate({"column": c.get("column"), "op": c.get("op"),
                                             "values": [v for v in (c.get("values") or []) if str(v).strip()]})
        except Exception:
            raise LinkError("A condition is not complete: choose its column, how it compares and a value.") from None
        if model.columns.get(f.column) is None or model.columns[f.column].table != to_table:
            raise LinkError(f"A condition keeps rows of {_table_name(model, to_table)}: choose one of its columns.")
        if f.op not in ("is_null", "not_null") and not f.values:
            raise LinkError(f"Give the condition on {_column_name(model, f.column)} a value.")
        if value_problem(model, f):
            raise LinkError(value_problem(model, f))
        conditions.append(f)
    # Many to one or one to one: the two a question can walk without counting a row twice.
    cardinality = str(data.get("cardinality") or "many_to_one")
    if cardinality not in ("many_to_one", "one_to_one"):
        cardinality = "many_to_one"
    return LinkSpec(from_table, to_table, pairs, conditions, " ".join(str(data.get("role") or "").split())[:80],
                    cardinality, data.get("keep_unmatched") is not False)


def type_problems(model: SemanticModel, spec: LinkSpec) -> list[str]:
    out = []
    for f, t in spec.pairs:
        a, b = model.columns[f], model.columns[t]
        if {a.data_type, b.data_type} == {"text", "integer"}:
            continue      # a code kept as text ('0007') is compared with a number key (7) without its leading zeros
        if _FAMILY.get(a.data_type) != _FAMILY.get(b.data_type):
            out.append(f"{a.name} is {_FAMILY.get(a.data_type, a.data_type)} and {b.name} is "
                       f"{_FAMILY.get(b.data_type, b.data_type)}: they may never be equal.")
    return out


def key_for(model: SemanticModel, spec: LinkSpec) -> str:
    return ids.join_key(spec.from_table, [model.columns[f].name for f, _ in spec.pairs], spec.to_table,
                        [model.columns[t].name for _, t in spec.pairs], spec.role or None)


def as_join(model: SemanticModel, spec: LinkSpec, checked: dict[str, Any] | None = None) -> Join:
    checked = checked or {}
    return Join(key=key_for(model, spec), from_table=spec.from_table, to_table=spec.to_table,
                from_columns=[f for f, _ in spec.pairs], to_columns=[t for _, t in spec.pairs],
                cardinality=spec.cardinality, role=spec.role or None, trust="admin", provenance="admin",
                status="approved", conditions=spec.conditions, keep_unmatched=spec.keep_unmatched,
                match_rate=float(checked.get("match") or 0.0), to_unique=(checked.get("most") or 1) <= 1,
                max_fanout=float(checked.get("most") or 1.0),
                to_calendar=model.tables[spec.to_table].kind == "calendar")


def same_link(j: Join, spec: LinkSpec) -> bool:
    return (j.from_table == spec.from_table and j.to_table == spec.to_table
            and list(zip(j.from_columns, j.to_columns)) == spec.pairs and (j.role or "") == spec.role)


# ── checked against the data ─────────────────────────────────────────────────

def _table(model: SemanticModel, key: str, alias: str, dialect: str) -> exp.Table:
    t = model.tables[key]
    return D.table_expr(t.database, t.schema_name, t.name, dialect, alias)


def _col(model: SemanticModel, alias: str, key: str, dialect: str) -> exp.Expression:
    from core2.compile.compiler import column_sql

    return column_sql(model, key, exp.to_identifier(alias), dialect)


def _count_when(condition: exp.Expr) -> exp.Expression:
    return exp.Sum(this=exp.Case(ifs=[exp.If(this=condition, true=exp.Literal.number(1))],
                                 default=exp.Literal.number(0)))


def _link_counts(model: SemanticModel, spec: LinkSpec, warehouse: Any, conditions: list[ColumnFilter]) -> dict:
    from core2.compile.compiler import equal_keys, row_conditions

    d = warehouse.dialect
    keys = [D.ident(f"k{i}", d) for i in range(len(spec.pairs))]
    targets = exp.select(*[_col(model, "t", t, d).as_(k) for (_, t), k in zip(spec.pairs, keys)],
                         exp.Count(this=exp.Star()).as_(D.ident("n", d))).from_(_table(model, spec.to_table, "t", d))
    kept = row_conditions(model, d, "t", conditions)
    if kept:
        targets = targets.where(exp.and_(*kept))
    targets = targets.group_by(*[_col(model, "t", t, d) for _, t in spec.pairs])
    on = exp.and_(*[equal_keys(model, _col(model, "f", f, d), f, exp.column(k, table=exp.to_identifier("m")), t, d)
                    for (f, t), k in zip(spec.pairs, keys)])
    n = exp.column(D.ident("n", d), table=exp.to_identifier("m"))
    has_key = exp.and_(*[exp.not_(exp.Is(this=_col(model, "f", f, d), expression=exp.Null())) for f, _ in spec.pairs])
    query = exp.select(
        exp.Count(this=exp.Star()).as_(D.ident("rows_all", d)),
        _count_when(has_key).as_(D.ident("with_key", d)),
        _count_when(exp.not_(exp.Is(this=n.copy(), expression=exp.Null()))).as_(D.ident("matched", d)),
        _count_when(exp.GT(this=n.copy(), expression=exp.Literal.number(1))).as_(D.ident("multiplied", d)),
        exp.Sum(this=n.copy()).as_(D.ident("joined", d)),
        exp.Max(this=n.copy()).as_(D.ident("most", d)),
    ).from_(_table(model, spec.from_table, "f", d)).join(
        targets.subquery(exp.to_identifier("m")), on=on, join_type="left")
    result = warehouse.query(D.render(query, d), max_rows=2)
    row = dict(zip([c.lower() for c in result.columns], result.rows[0])) if result.rows else {}
    with_key, matched = int(row.get("with_key") or 0), int(row.get("matched") or 0)
    return {"rows": int(row.get("rows_all") or 0), "with_key": with_key, "matched": matched,
            "match": round(matched / with_key, 4) if with_key else 0.0,
            "per_row": round(float(row.get("joined") or 0) / matched, 2) if matched else 0.0,
            "twice": int(row.get("multiplied") or 0), "most": int(row.get("most") or 0)}


def check_link(model: SemanticModel, spec: LinkSpec, warehouse: Any) -> dict[str, Any]:
    """How the link matches: the share of rows that find a match, matches per row, rows counted twice;
    with conditions, the same without them, so the admin sees what the condition changes."""
    out: dict[str, Any] = {"types": type_problems(model, spec), "problem": ""}
    try:
        out.update(_link_counts(model, spec, warehouse, spec.conditions))
        if spec.conditions:
            out["without"] = _link_counts(model, spec, warehouse, [])
    except Exception as exc:  # noqa: BLE001 - the warehouse's refusal is the admin's to read
        out["problem"] = f"The data refused the check: {str(exc)[:300]}"
        return out
    words = []
    from_rows, to_rows = _table_name(model, spec.from_table).lower(), _table_name(model, spec.to_table).lower()
    out["from_rows"], out["to_rows"] = from_rows, to_rows
    if out["twice"]:
        words.append(f"{out['twice']:,} {_table_name(model, spec.from_table).lower()} rows match more than one "
                     f"{_table_name(model, spec.to_table).lower()} row: totals through this link would count them "
                     "more than once.")
    without = out.get("without")
    if without and without["twice"] and not out["twice"]:
        out["without_words"] = (f"{without['per_row']:.2f}", f"{to_rows} per {from_rows}, and totals by {to_rows} "
                                f"would be {max(without['per_row'] - 1, 0):.0%} too high.")
    if out["with_key"] and out["match"] < 0.9:
        words.append(f"{1 - out['match']:.0%} of the rows with a value find no match: they show as Unknown.")
    out["words"] = words
    return out


def answers_say(model: SemanticModel, table_key: str, rules: list[ColumnFilter]) -> str:
    """What an answer carries for these rules, word for word: "Order line: rows where Status code is C are left out
    (a default filter)."
    """
    from core2.plan.catalog import left_out_note

    return " ".join(left_out_note(model, table_key, f) for f in rules)


def _year_amount(model: SemanticModel, table_key: str, main: Any, rules: list[ColumnFilter], warehouse: Any,
                 today: Any) -> tuple[float | None, str]:
    """The main measure in the table's latest year of data, with these rules and without: what they leave out."""
    from core2.compile.compiler import compile_query
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    role = model.date_roles.get(model.tables[table_key].default_date or "")
    if role is None or role.last is None:
        return None, ""
    year = role.last.year
    plan = Plan.model_validate({"kind": "query", "measures": [main.slug], "time": {"window": {
        "kind": "between", "start": f"{year}-01-01", "end": f"{year}-12-31"}}})
    values = []
    for with_rules in (False, True):
        copy = model.model_copy(deep=True)
        copy.tables[table_key].default_filters = list(rules) if with_rules else []
        logical = resolve(plan, copy, Context(today=today))
        compiled = compile_query(logical, copy, warehouse.dialect)
        result = warehouse.query(compiled.sql, max_rows=2)
        name = next(c.name for c in compiled.columns if c.role == "measure")
        row = dict(zip(result.columns, result.rows[0])) if result.rows else {}
        values.append(float(row.get(name) or 0.0))
    return values[0] - values[1], str(year)


def check_all(model: SemanticModel, warehouse: Any) -> dict[str, Any]:
    """Every link in use, matched again on today's data (its conditions included)."""
    out: dict[str, Any] = {}
    for j in model.joins.values():
        if j.trust == "rejected" or j.from_table not in model.tables or j.to_table not in model.tables:
            continue
        spec = LinkSpec(j.from_table, j.to_table, list(zip(j.from_columns, j.to_columns)), list(j.conditions),
                        j.role or "", j.cardinality, j.keep_unmatched)
        try:
            counts = _link_counts(model, spec, warehouse, spec.conditions)
        except Exception as exc:  # noqa: BLE001 - one link the data refuses does not stop the others
            out[j.key] = {"problem": str(exc)[:200]}
            continue
        out[j.key] = {"match": counts["match"], "twice": counts["twice"]}
    return out


def check_table(model: SemanticModel, table_key: str, rules: list[ColumnFilter], warehouse: Any,
                today: Any = None) -> dict[str, Any]:
    """What a table's always-leave-out rules leave out: rows, and the amount of its main measure."""
    from core2.compile.compiler import row_conditions
    from core2.model.schema import AggExpr

    d = warehouse.dialect
    out: dict[str, Any] = {"problem": "", "rows": 0, "left_out": 0, "amount": None, "measure": "",
                           "metrics": sum(1 for m in model.measures.values() if m.table == table_key and not m.hidden)}
    if not rules:
        return out
    kept = exp.and_(*row_conditions(model, d, "t", rules))
    left_out = exp.not_(exp.Paren(this=kept))
    # The table's main amount: a money total, the net one when there is one.
    sums = [m for m in model.measures.values() if m.table == table_key and not m.hidden
            and isinstance(m.expr, AggExpr) and m.expr.agg == "sum" and m.expr.column]
    main = min(sums, key=lambda m: (m.format != "currency", "net" not in m.slug, m.key), default=None)
    selects = [exp.Count(this=exp.Star()).as_(D.ident("rows_all", d)),
               _count_when(left_out.copy()).as_(D.ident("left_out", d))]
    if main is not None:
        value = _col(model, "t", main.expr.column, d)  # type: ignore[union-attr]
        selects.append(exp.Sum(this=exp.Case(ifs=[exp.If(this=left_out.copy(), true=value)]))
                       .as_(D.ident("amount", d)))
    query = exp.select(*selects).from_(_table(model, table_key, "t", d))
    try:
        result = warehouse.query(D.render(query, d), max_rows=2)
    except Exception as exc:  # noqa: BLE001
        out["problem"] = f"The data refused the check: {str(exc)[:300]}"
        return out
    row = dict(zip([c.lower() for c in result.columns], result.rows[0])) if result.rows else {}
    out.update(rows=int(row.get("rows_all") or 0), left_out=int(row.get("left_out") or 0),
               says=answers_say(model, table_key, rules))
    if main is not None:
        out["amount"], out["year"] = float(row.get("amount") or 0.0), ""
        out["measure"], out["format"] = main.business_name, main.format
        if today is not None:
            try:
                amount, year = _year_amount(model, table_key, main, rules, warehouse, today)
            except Exception as exc:  # noqa: BLE001 - the all-time amount stands
                log.warning("core2: the amount left out in the latest year could not be read: %s", exc)
                amount, year = None, ""
            if amount is not None:
                out["amount"], out["year"] = amount, year
    return out
