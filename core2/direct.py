"""A question the plan cannot express, answered by one query the AI writes from the approved model and code checks
before it runs: the safety net (Phase 6 of the analyst plan).

The planner answers by naming the model's figures, fields and dates; code writes the SQL. Some questions need a
shape no plan holds ("the schools whose test takers outnumber their enrolment", "the share of posts by members over
65"), and were refused. For those, and only when the plan route has stopped, the AI writes the query itself:

* It is given the approved model and nothing else: the tables the reader may see, their columns (none an admin hid,
  no personal detail, nothing marked sensitive), the links between them with the columns they join on, the
  metrics' definitions, the dates, what each table leaves out by default. No member value, unless values may reach
  the AI; a name the reader wrote reaches it as its placeholder (⟨v1⟩), put back by code.
* Code checks the query before it runs (``check``): one read-only SELECT, made only of the kinds of node and the
  functions an answer needs (anything else is refused: a WITH, a lock, a table hint, a past version of a table, a
  function the database lets anyone define); only those tables and columns, every column named with its table;
  tables matched only on their approved links, by the columns the link joins, wherever they are matched (a join, a
  subquery, IN); no figure added up over rows a join repeats, through a subquery too. Then code writes each table
  as the approved one in full, read through its default filters as the compiler writes them (cancelled rows, and
  the lines of cancelled documents, stay out), and puts on the row limit.
* It runs as every answer does, through the governed executor: the reader's tables, row rules, masking and the
  decision log apply to it unchanged.
* The answer is the usual card, marked as a direct query, with how it was worked out in code's own words. The query
  is shown to creators only. Every part of the request the planner listed must be among the parts the query gives.

A workspace switch turns it on (Settings). It is off in every workspace until an admin turns it on: no workspace
starts getting queries the AI wrote without an admin choosing it. A workspace under compliance has no switch at all.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import sqlglot
import sqlglot.errors
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from core2.model.schema import Column, SemanticModel
from core2.plan.catalog import definition, left_out_words
from core2.resolve import paths as P
from core2.warehouse import dialect as D
from core2.warehouse.runner import ReadOnlyViolation, assert_read_only

log = logging.getLogger("querybot.core2")

ROW_CAP = 5000
SWITCH = "direct_queries_enabled"

DIALECT_NAMES = {"tsql": "SQL Server (T-SQL)", "snowflake": "Snowflake", "oracle": "Oracle", "duckdb": "DuckDB",
                 "postgres": "PostgreSQL", "databricks": "Databricks SQL"}

class NotSafe(ValueError):
    """The query the AI wrote breaks a rule: said why, and never run."""


def enabled(state: dict[str, Any] | None) -> bool:
    """The admin's switch for direct queries (Settings): off until an admin turns it on."""
    flag = (state or {}).get(SWITCH) if isinstance(state, dict) else None
    return flag is not None and bool(flag) and str(flag).lower() not in ("0", "false", "no", "off")


def allowed_for(account_id: str) -> bool:
    """May the workspace have direct queries at all? Not under compliance (regulated, never set up, or any mode but
    standard) until a person has reviewed this module: there Settings shows no switch, and a switch stored before is
    never read. Unknown is no."""
    try:
        from core.compliance.policy_engine import under_compliance

        return not under_compliance(account_id)
    except Exception:  # noqa: BLE001 - the compliance check failing is never a yes
        log.warning("core2: could not tell whether %s is under compliance; no direct queries", account_id,
                    exc_info=True)
        return False


# ── the approved model, as the AI may use it ─────────────────────────────────────────────────────────────────────

@dataclass
class TableView:
    key: str
    sql: str                                  # as the query writes it, in the dialect
    parts: tuple[str, ...]                    # database, schema, name, casefolded (empty parts left out)
    columns: dict[str, Column]                # casefolded column name -> column


@dataclass
class View:
    model: SemanticModel
    dialect: str
    tables: dict[str, TableView]              # table key -> view
    links: list[Any] = field(default_factory=list)

    def table_of(self, node: exp.Table) -> TableView | None:
        """The approved table a FROM or JOIN names, matched by as much of its database, schema and name as it
        writes; None for any other."""
        if not isinstance(node.this, exp.Identifier):
            return None                       # a table function (read_csv(...)), never a table
        written = tuple(p.casefold() for p in (node.catalog, node.db, node.name) if p)
        found = [t for t in self.tables.values() if t.parts[-len(written):] == written]
        return found[0] if len(found) == 1 else None


def _usable_column(c: Column) -> bool:
    """A column the safety net may read: one in the warehouse, not hidden, no personal detail (a contact, a birth
    date, an ID), nothing marked sensitive. A person's name may be read: the governed executor masks it as it does
    on every answer."""
    return not c.hidden and not c.parts and c.personal != "detail" and c.sensitivity == "none"


def approved(model: SemanticModel, dialect: str, allowed: set[str] | None = None) -> View:
    tables: dict[str, TableView] = {}
    for key, t in model.tables.items():
        if getattr(t, "hidden", False) or (allowed is not None and key not in allowed):
            continue
        # A column whose name differs from another's only by case (Snowflake's "name" and NAME) is approved under
        # neither: a name matched in any case must mean one column.
        spelled = Counter(c.name.casefold() for c in model.columns.values() if c.table == key)
        columns = {c.name.casefold(): c for c in model.columns.values()
                   if c.table == key and _usable_column(c) and spelled[c.name.casefold()] == 1}
        if not columns:
            continue
        parts = tuple(p.casefold() for p in (t.database, t.schema_name, t.name) if p)
        tables[key] = TableView(key, D.table_sql(t.database, t.schema_name, t.name, dialect), parts, columns)
    links = [j for j in model.joins.values() if P.usable(j) and j.from_table in tables and j.to_table in tables
             and all(model.columns[c].name.casefold() in tables[j.from_table].columns for c in j.from_columns)
             and all(model.columns[c].name.casefold() in tables[j.to_table].columns for c in j.to_columns)]
    return View(model, dialect, tables, links)


def schema_text(view: View, *, values_allowed: bool) -> str:
    """The approved model in the prompt: stable for a model version and a reader's tables (no clock, no question)."""
    model = view.model
    lines = ["TABLES (write each as given; one line per column: name | type | what it is)"]
    for t in sorted(view.tables.values(), key=lambda t: t.sql):
        table = model.tables[t.key]
        lines.append(f"## {t.sql} -- {table.business_name or table.name}: {table.grain_text or table.kind}"
                     + (f"; {table.description}" if table.description else ""))
        leaves = "; ".join(left_out_words(model, f, values_allowed) for f in table.default_filters
                           if f.column in model.columns)
        if leaves:
            lines.append(f"   (rows where {leaves} are left out before your query reads the table)")
        for c in sorted(t.columns.values(), key=lambda c: c.name.casefold()):
            about = c.business_name or c.name
            if c.description:
                about += f": {c.description}"
            if c.unit:
                about += f" (in {c.unit})"
            lines.append(f"- {c.name} | {c.data_type} | {about}")
    lines += ["", "LINKS (join tables only on these, with exactly these columns)"]
    for j in view.links:
        a, b = model.tables[j.from_table], model.tables[j.to_table]
        on = " AND ".join(f"{a.name}.{model.columns[f].name} = {b.name}.{model.columns[k].name}"
                          for f, k in zip(j.from_columns, j.to_columns))
        many = {"many_to_one": f"many {a.name} rows to one {b.name} row",
                "one_to_one": "one row to one row"}.get(j.cardinality, j.cardinality.replace("_", " "))
        lines.append(f"- {on}  ({many}{'; ' + j.role if j.role else ''})")
    shown = [m for m in model.measures.values() if not m.hidden and m.table in view.tables]
    if shown:
        lines += ["", "METRICS (the business's own definitions: use them when the question names one)"]
        for m in sorted(shown, key=lambda m: m.business_name or m.slug):
            lines.append(f"- {m.business_name or m.slug} ({model.tables[m.table].name}): "
                         f"{definition(model, m.expr, values_allowed)}")
    dates = [r for r in model.date_roles.values() if r.kind != "audit" and r.table in view.tables
             and r.column in model.columns and _usable_column(model.columns[r.column])]
    if dates:
        lines += ["", "DATES"]
        for r in sorted(dates, key=lambda r: (r.table, not r.is_default)):
            lines.append(f"- {r.name}: {model.tables[r.table].name}.{model.columns[r.column].name}"
                         f"{' (the default date of its table)' if r.is_default else ''}")
    return "\n".join(lines)


SYSTEM = """You write one SQL query that answers a business question from the tables below, for {dialect}.

Rules:
- One SELECT; subqueries are fine (no WITH, no UNION). Never change data, never two statements, no comments.
- Only the tables and columns listed. Give every table an alias and write every column with its alias (o.amount),
  everywhere: SELECT, WHERE, GROUP BY, HAVING, inside subqueries. An output's own name may be used in ORDER BY.
- Join tables only on the links listed, with exactly the columns they join on (JOIN ... ON); no other join. A
  subquery is matched to the query around it, or listed with IN, only on a link too.
- A subquery that joins tables adds its rows up (GROUP BY) before it is read or joined.
- Only common functions: COUNT, SUM, AVG, MIN, MAX, MEDIAN, CAST, CASE, COALESCE, NULLIF, ROUND, ABS, UPPER, LOWER,
  the length of a text, SUBSTRING, TRIM, REPLACE, CONCAT, YEAR, MONTH, DAY, EXTRACT, adding to a date, the days
  between two dates, ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD.
- Never add up (SUM, AVG, COUNT of a column) a figure of a table whose rows the joins repeat: a customer's credit
  limit summed over their orders counts it once per order. Work it out on its own table first, in a subquery, or
  count with COUNT(DISTINCT ...).
- Name each output column in plain words (AS total_sales).
- A value written like ⟨v1⟩ in the question stands for a name: write it in quotes exactly as written ('⟨v1⟩').
- Today is {today}.

Answer with one JSON object, nothing else:
{{"sql": "<the query>", "parts": [{{"words": "<words of the question>", "in": "<the output column or the condition
of the query that gives them>"}}]}}
or, when these tables cannot answer the question: {{"cannot": "<why, in a few words>"}}

{schema}
"""


@dataclass
class Written:
    sql: str
    parts: list[dict[str, str]]


def write(question: str, view: View, complete: Callable[[str, str], str], *, today: dt.date, why: str = "",
          values_allowed: bool, matched: list[str] | None = None) -> Written | str:
    """The query the AI writes for ``question`` (as the AI may see it), or why it cannot."""
    stable = SYSTEM.format(dialect=DIALECT_NAMES.get(view.dialect, view.dialect), today=today.isoformat(),
                           schema=schema_text(view, values_allowed=values_allowed))
    tail = f"QUESTION: {question}"
    if matched:
        tail += "\nNAMES IN THE QUESTION (field = value):\n" + "\n".join(f"- {m}" for m in matched)
    if why:
        tail += f"\nThe usual way could not answer it: {why}"
    text = complete(stable, tail)
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    except (TypeError, ValueError):
        data = {}
    if not isinstance(data, dict) or not (data.get("sql") or data.get("cannot")):
        return "the AI did not write a query"
    if data.get("cannot") and not data.get("sql"):
        return str(data["cannot"])[:200]
    parts = [p for p in data.get("parts") or [] if isinstance(p, dict) and p.get("words")]
    return Written(str(data["sql"]).strip().rstrip(";"), [{"words": str(p["words"]), "in": str(p.get("in") or "")}
                                                          for p in parts])


# ── the checks ───────────────────────────────────────────────────────────────────────────────────────────────────

# What a query may hold, as kinds of node: anything else is refused, whatever it is called in the dialect (a WITH,
# a UNION, a lambda, a table function, a call the database lets anyone define, a variable, a lock, a past version
# of a table). Kept to what an answer needs; a query that needs more is refused, never run.
_FUNCTIONS = tuple(getattr(exp, n) for n in (
    # adding up
    "Count", "Sum", "Avg", "Min", "Max", "Median", "Stddev", "StddevPop", "StddevSamp", "Variance", "VariancePop",
    # rows in order
    "RowNumber", "Rank", "DenseRank", "Ntile", "Lag", "Lead", "FirstValue", "LastValue", "PercentRank", "CumeDist",
    # values
    "Cast", "TryCast", "Convert", "Coalesce", "Nullif", "If", "Case", "Abs", "Round", "Floor", "Ceil", "Sqrt", "Pow",
    "Ln", "Log", "Exp", "Sign", "SafeDivide", "Greatest", "Least",
    # text
    "Upper", "Lower", "Length", "Substring", "Trim", "Concat", "ConcatWs", "Left", "Right", "Replace", "StrPosition",
    "Initcap", "Pad", "SplitPart",
    # dates
    "Extract", "DateTrunc", "TimestampTrunc", "DateDiff", "DateAdd", "DateSub", "TimestampDiff", "TimestampAdd",
    "DatetimeDiff", "DatetimeAdd", "TsOrDsAdd", "TsOrDsDiff", "TsOrDsToDate", "Year", "Quarter", "Month", "Week",
    "Day", "DayOfWeek", "DayOfMonth", "DayOfYear", "CurrentDate", "CurrentTimestamp", "TimeToStr", "StrToDate",
    "StrToTime", "DateStrToDate", "TimeStrToTime", "Date", "LastDay", "DateFromParts", "MonthsBetween", "ToChar",
) if hasattr(exp, n))
_NODES = tuple(getattr(exp, n) for n in (
    "Select", "From", "Join", "Table", "TableAlias", "Subquery", "Identifier", "Column", "Alias", "Star", "Distinct",
    "Where", "Group", "Having", "Qualify", "Order", "Ordered", "Limit", "LimitOptions", "Offset", "Fetch", "Literal",
    "Null", "Boolean", "National", "Paren", "Neg", "Not", "And", "Or", "Xor", "EQ", "NEQ", "NullSafeEQ",
    "NullSafeNEQ", "GT", "GTE", "LT", "LTE", "Is", "In", "Between", "Like", "ILike", "Escape", "Exists", "Add", "Sub",
    "Mul", "Div", "Mod", "IntDiv", "DPipe", "Window", "WindowSpec", "DataType", "DataTypeParam", "Interval", "Var",
) if hasattr(exp, n))
_ADDED_UP_SAFELY = (exp.Min, exp.Max)       # the same however often the joins repeat a row
_COMPARE = tuple(getattr(exp, n) for n in ("EQ", "NEQ", "NullSafeEQ", "NullSafeNEQ", "GT", "GTE", "LT", "LTE", "Like",
                                           "ILike") if hasattr(exp, n))
# The clauses each part of a query may carry; one with any other (FOR UPDATE, a table hint, AT(...), TOP ... PERCENT,
# DISTINCT ON, a column list after a subquery's name, NATURAL or USING joins) is refused.
_ARGS = {exp.Select: {"expressions", "from", "from_", "joins", "where", "group", "having", "qualify", "order",
                      "limit", "offset", "distinct"},
         exp.Table: {"this", "db", "catalog", "alias"}, exp.TableAlias: {"this"}, exp.Subquery: {"this", "alias"},
         exp.Join: {"this", "on", "kind", "side"}, exp.Column: {"this", "table"}, exp.Distinct: {"expressions"},
         exp.Window: {"this", "partition_by", "order", "spec", "over"},
         exp.In: {"this", "expressions", "query", "is_global"}}


@dataclass
class Checked:
    tree: exp.Expression
    tables: list[str]                         # table keys read
    joins: list[Any]                          # model links followed
    grouped: list[str]                        # column keys grouped by
    filtered: list[str]                       # column keys filtered on
    figures: list[str]                        # what is worked out, in words
    left_out: list[tuple[str, Any]] = field(default_factory=list)   # (table key, default filter) put on by code


def check(sql: str, view: View, *, cap: int = ROW_CAP, values: dict[str, object] | None = None) -> Checked:
    """``sql`` as it will run, or :class:`NotSafe` saying which rule it breaks."""
    try:
        assert_read_only(sql, view.dialect)
    except ReadOnlyViolation as exc:
        raise NotSafe(str(exc)) from None
    try:
        tree = sqlglot.parse_one(sql, read=view.dialect)
    except sqlglot.errors.ParseError:
        raise NotSafe("the query could not be read") from None
    if not isinstance(tree, exp.Select):
        raise NotSafe("write one SELECT (no WITH, no UNION)")
    _allowed_only(tree, view)
    _outputs_by_name(tree)
    found = _Found()
    for scope in traverse_scope(tree):
        _check_scope(scope, view, found)
    if not found.read:
        raise NotSafe("the query reads no table of this data")
    for column, stored in found.columns:
        column.set("this", D.ident(stored.name, view.dialect))   # as stored, whatever case the query wrote it in
    _put_values(tree, values or {})
    tree, left_out = _as_approved(tree, view)
    tree = _limited(tree, cap)
    return Checked(tree, list(dict.fromkeys(found.read)), found.followed, list(dict.fromkeys(found.grouped)),
                   list(dict.fromkeys(found.filtered)), found.figures, left_out)


def _allowed_only(tree: exp.Expression, view: View) -> None:
    """Every node one of the kinds a query may hold, with only the clauses its kind may carry."""
    for node in tree.walk():
        if isinstance(node, exp.Func) and not isinstance(node, _FUNCTIONS + _NODES):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            raise NotSafe(f"the function {name or type(node).__name__} is not allowed")
        if not isinstance(node, _FUNCTIONS + _NODES):
            raise NotSafe(f"{node.sql(dialect=view.dialect)[:60] or type(node).__name__} is not allowed here")
        for kind, allowed in _ARGS.items():
            if type(node) is kind:
                extra = [k for k, v in node.args.items() if v not in (None, False, []) and k not in allowed]
                if extra:
                    raise NotSafe(f"{node.sql(dialect=view.dialect)[:60]} uses {extra[0]}, which is not allowed")
        if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count):
            raise NotSafe("name each column; * reads columns that are not approved")
        if isinstance(node, exp.Var) and not isinstance(node.parent, (exp.Func, exp.Interval, exp.WindowSpec)):
            raise NotSafe(f"{node.sql(dialect=view.dialect)} is not allowed here")
        if isinstance(node, (exp.DataType, exp.DataTypeParam)) and \
                not isinstance(node.parent, (exp.Cast, exp.TryCast, exp.Convert, exp.DataType)):
            raise NotSafe("a type is written only in a CAST")
        if isinstance(node, exp.LimitOptions) and (node.args.get("percent") or node.args.get("with_ties")):
            raise NotSafe("a row limit is a number of rows (no PERCENT, no WITH TIES)")
        if isinstance(node, (exp.Limit, exp.Offset, exp.Fetch)):
            number = node.args.get("count") if isinstance(node, exp.Fetch) else node.expression
            if not (isinstance(number, exp.Literal) and number.is_int):
                raise NotSafe("a row limit is a whole number")
        if isinstance(node, exp.Join) and (node.args.get("kind") or "").upper() not in ("", "INNER", "OUTER"):
            raise NotSafe("join tables with JOIN ... ON on an approved link")


def _outputs_by_name(tree: exp.Expression) -> None:
    """An output named in ORDER BY or GROUP BY (ORDER BY total) is written as what it is: a name alone could read a
    column of the same name, in a warehouse that looks there first."""
    for select in list(tree.find_all(exp.Select)):
        named = {p.alias.casefold(): p.this for p in select.expressions if isinstance(p, exp.Alias)}
        for clause in ("order", "group"):
            node = select.args.get(clause)
            for column in list(node.find_all(exp.Column) if node is not None else []):
                if not column.table and column.name.casefold() in named and column.find_ancestor(exp.Select) is select:
                    column.replace(named[column.name.casefold()].copy())


@dataclass
class _Found:
    read: list[str] = field(default_factory=list)
    followed: list[Any] = field(default_factory=list)
    grouped: list[str] = field(default_factory=list)
    filtered: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    columns: list[tuple[exp.Column, Column]] = field(default_factory=list)   # each column read, as approved


def _sources(scope: Scope, view: View) -> dict[str, TableView | Scope]:
    """Each name a scope's columns may be qualified by: an approved table, or a subquery of the query."""
    out: dict[str, TableView | Scope] = {}
    for name, source in scope.sources.items():
        if isinstance(source, Scope):
            out[name.casefold()] = source
            continue
        table = view.table_of(source) if isinstance(source, exp.Table) else None
        if table is None:
            raise NotSafe(f"{source.sql(dialect=view.dialect)} is not one of the approved tables")
        out[name.casefold()] = table
    return out


def _check_scope(scope: Scope, view: View, found: _Found) -> None:
    select = scope.expression
    sources = _sources(scope, view)
    found.read.extend(s.key for s in sources.values() if isinstance(s, TableView))
    for name, source in sources.items():
        if isinstance(source, Scope) and source.expression.args.get("joins") and not _summed(source):
            raise NotSafe(f"the subquery {name} joins tables: add its rows up (GROUP BY) before reading it")

    def resolve(column: exp.Column) -> tuple[str, bool, tuple[TableView, Column] | None]:
        """Which source a column reads (its alias, whether it is this scope's own), and the approved table and
        column it reads as it is; None for a subquery's output it works out."""
        qualifier = column.table.casefold()
        if not qualifier:
            raise NotSafe(f"write the column {column.name} with its table's alias")
        source, own = sources.get(qualifier), True
        if source is None:
            source, own = _outer(scope, qualifier, view), False
        if source is None:
            raise NotSafe(f"{qualifier} is no table of the query")
        if isinstance(source, Scope):
            return qualifier, own, _traced(source, column.name, view)
        stored = source.columns.get(column.name.casefold())
        if stored is None:
            raise NotSafe(f"{column.name} is not an approved column of {source.sql}")
        found.columns.append((column, stored))
        return qualifier, own, (source, stored)

    ons = {id(n) for join in select.args.get("joins") or [] if join.args.get("on") is not None
           for n in join.args["on"].walk()}
    for column in _own(select, scope, exp.Column):
        resolve(column)

    # Joins: on an approved link, by the columns it joins.
    edges: list[tuple[str, str, Any]] = []          # (alias holding the link's from-columns, the other's, link)
    first = select.args.get("from_") or select.args.get("from")
    seen = [first.this.alias_or_name.casefold()] if first is not None else []
    for join in select.args.get("joins") or []:
        name = join.this.alias_or_name.casefold()
        on = join.args.get("on")
        if on is None:
            raise NotSafe("join tables with JOIN ... ON on an approved link")
        if any(isinstance(n, exp.Or) for n in on.walk()):
            raise NotSafe("a join's condition cannot hold OR")
        pairs: list[tuple[str, Column, str, Column]] = []
        for part in _conjuncts(on):
            sides = {resolve(c)[0] for c in part.find_all(exp.Column)}
            if len(sides) < 2:
                continue                           # a condition on one table (b.type = 'x'): a filter
            pairs.append(_linked(part, resolve, view))
        link_found = False
        used: dict[str, Any] = {}
        for a, ca, b, cb in pairs:
            link = _link(view, ca, cb)
            if link is None:
                raise NotSafe(f"{ca.name} = {cb.name} is not an approved link")
            if name in (a, b) and ({a, b} - {name}) <= set(seen):
                link_found = True
            used[link.key] = link
            if link not in found.followed:
                found.followed.append(link)
            edges.append((a, b, link) if (ca.key, cb.key) in set(zip(link.from_columns, link.to_columns))
                         else (b, a, link))
        joined = {(ca.key, cb.key) for _, ca, _, cb in pairs}
        joined |= {(t, f) for f, t in joined}
        for link in used.values():
            if not set(zip(link.from_columns, link.to_columns)) <= joined:
                raise NotSafe("a link of two columns is joined on both of them")
        if not link_found:
            raise NotSafe(f"{join.this.sql(dialect=view.dialect)} is joined on no approved link")
        seen.append(name)

    # Anywhere else, two tables are matched only on an approved link: a subquery to the query around it, a list of
    # a subquery's values (IN), two tables' columns made equal.
    for part in _own(select, scope, _COMPARE):
        if id(part) in ons:
            continue
        sides = {resolve(c)[:2] for c in part.find_all(exp.Column) if _in_scope(c, scope)}
        if len({alias for alias, _ in sides}) < 2:
            continue
        if isinstance(part, exp.EQ) and all(isinstance(x, exp.Column) for x in (part.left, part.right)):
            a, ca, b, cb = _linked(part, resolve, view)
            link = _link(view, ca, cb)
            if link is not None:
                if link not in found.followed:
                    found.followed.append(link)
                continue
        if isinstance(part, exp.EQ) or not all(own for _, own in sides):
            raise NotSafe(f"{part.sql(dialect=view.dialect)} matches two tables on no approved link")
    for listed in _own(select, scope, exp.In):
        query = listed.args.get("query")
        if query is None:
            continue
        inner = next((s for s in scope.subquery_scopes if s.expression is query.unnest()), None)
        outputs = query.unnest().expressions if isinstance(query.unnest(), exp.Select) else []
        if inner is None or len(outputs) != 1 or not isinstance(listed.this, exp.Column):
            raise NotSafe("IN takes one column of a subquery, matched to one column")
        mine, theirs = resolve(listed.this)[2], _traced(inner, outputs[0].alias_or_name, view)
        if mine is None or theirs is None or mine[1].key == theirs[1].key:
            continue                               # a value worked out, or the same column: no two tables matched
        if _link(view, mine[1], theirs[1]) is None:
            raise NotSafe(f"{listed.this.sql(dialect=view.dialect)} IN ... matches two tables on no approved link")

    # No figure added up over rows a join repeats.
    def repeated(alias: str) -> bool:
        return _repeats(alias, edges)

    for agg in _own(select, scope, exp.AggFunc):
        found.figures.append(agg.sql(dialect=view.dialect))
        if isinstance(agg, _ADDED_UP_SAFELY) or isinstance(agg, exp.Count) and isinstance(agg.this, exp.Distinct):
            continue
        if isinstance(agg, exp.Count) and isinstance(agg.this, exp.Star):
            if sources and all(repeated(a) for a in sources):
                raise NotSafe("COUNT(*) counts rows the joins repeat")
            continue
        for column in _valued(agg):
            alias, own, _ = resolve(column)
            if own and repeated(alias):
                raise NotSafe(f"{type(agg).__name__.upper()} of {column.name} adds up rows the joins repeat")

    for g in (select.args.get("group").expressions if select.args.get("group") else []):
        for c in g.find_all(exp.Column):
            stored = resolve(c)[2]
            if stored:
                found.grouped.append(stored[1].key)
    for clause in ("where", "having", "qualify"):
        node = select.args.get(clause)
        for c in (node.find_all(exp.Column) if node is not None else []):
            if _in_scope(c, scope):
                stored = resolve(c)[2]
                if stored:
                    found.filtered.append(stored[1].key)


def _own(select: exp.Expression, scope: Scope, kinds) -> list[exp.Expression]:
    """The nodes of these kinds that are this scope's own (not inside one of its subqueries)."""
    return [n for n in select.find_all(kinds) if _in_scope(n, scope)]


def _linked(part: exp.Expression, resolve, view: View) -> tuple[str, Column, str, Column]:
    """The two approved columns an equality matches, each with its table's alias in the query."""
    if not isinstance(part, exp.EQ) or not all(isinstance(x, exp.Column) for x in (part.left, part.right)):
        raise NotSafe(f"{part.sql(dialect=view.dialect)} joins tables on no approved link")
    (a, _, left), (b, _, right) = resolve(part.left), resolve(part.right)
    if left is None or right is None:
        raise NotSafe(f"{part.sql(dialect=view.dialect)} joins on a worked-out column")
    return a, left[1], b, right[1]


def _summed(scope: Scope) -> bool:
    """Are a subquery's rows added up (one per group, or one in all), not the rows its joins make?"""
    select = scope.expression
    return bool(select.args.get("group") or select.args.get("distinct")) or any(
        isinstance(n, exp.AggFunc) and not n.find_ancestor(exp.Window) and _in_scope(n, scope)
        for p in select.expressions for n in p.walk())


def _valued(agg: exp.Expression) -> list[exp.Column]:
    """The columns whose values an aggregate adds up: never one only tested in a condition (SUM(CASE WHEN u.age >
    65 THEN 1 ELSE 0 END) counts rows by the member's age, adding up no age)."""
    out = []
    for column in agg.find_all(exp.Column):
        node, tested = column, False
        while node is not None and node is not agg:
            parent = node.parent
            if isinstance(parent, exp.If) and parent.args.get("this") is node:
                tested = True
                break
            node = parent
        if not tested:
            out.append(column)
    return out


def _in_scope(node: exp.Expression, scope: Scope) -> bool:
    """Is ``node`` this scope's own (not inside one of its subqueries)?"""
    parent = node.parent
    while parent is not None and parent is not scope.expression:
        if isinstance(parent, (exp.Select, exp.Union)) and parent is not scope.expression:
            return False
        parent = parent.parent
    return True


def _outer(scope: Scope, qualifier: str, view: View) -> TableView | Scope | None:
    """A correlated subquery's column of the query around it."""
    parent = scope.parent
    while parent is not None:
        sources = _sources(parent, view)
        if qualifier in sources:
            return sources[qualifier]
        parent = parent.parent
    return None


def _traced(scope: Scope, name: str, view: View) -> tuple[TableView, Column] | None:
    """The approved column a subquery's output reads, when it reads one as it is; None when it works it out."""
    select = scope.expression
    named = [p for p in select.expressions if p.alias_or_name.casefold() == name.casefold()]
    if len(named) != 1:
        raise NotSafe(f"{name} is no column of its subquery" if not named else
                      f"two outputs of a subquery are named {name}")
    inner = named[0].this if isinstance(named[0], exp.Alias) else named[0]
    if not isinstance(inner, exp.Column):
        return None
    source = _sources(scope, view).get(inner.table.casefold())
    if isinstance(source, Scope):
        return _traced(source, inner.name, view)
    if isinstance(source, TableView) and inner.name.casefold() in source.columns:
        return source, source.columns[inner.name.casefold()]
    raise NotSafe(f"write the column {inner.name} with its table's alias")


def _conjuncts(node: exp.Expression) -> list[exp.Expression]:
    node = node.unnest() if isinstance(node, exp.Paren) else node
    if isinstance(node, exp.And):
        return [*_conjuncts(node.left), *_conjuncts(node.right)]
    return [node]


def _link(view: View, a: Column, b: Column) -> Any | None:
    for j in view.links:
        pairs = set(zip(j.from_columns, j.to_columns))
        if (a.key, b.key) in pairs or (b.key, a.key) in pairs:
            return j
    return None


def _repeats(alias: str, edges: list[tuple[str, str, Any]]) -> bool:
    """Do the joins repeat this source's rows? Each step away from it must be to a row of which it holds one (many
    orders to one customer: the order holds the link's columns); a step to rows that hold it (a customer's many
    orders) repeats it. A table linked to itself (an employee's manager) is told apart by which side holds them."""
    todo, seen = [alias], {alias}
    while todo:
        here = todo.pop()
        for holder, held, link in edges:
            if here not in (holder, held):
                continue
            there = held if here == holder else holder
            if there in seen:
                continue
            ahead = link.cardinality == "one_to_one" or link.cardinality == "many_to_one" and here == holder
            if not ahead or P.repeats(link) and here == holder:
                return True
            seen.add(there)
            todo.append(there)
    return False


def _put_values(tree: exp.Expression, values: dict[str, object]) -> None:
    """A name the AI was given as its placeholder (⟨v1⟩), back as it is stored."""
    for literal in list(tree.find_all(exp.Literal)):
        if not literal.is_string:
            continue
        text = literal.this
        if any(k in text for k in values):
            for k, v in values.items():
                text = text.replace(k, str(v))
            literal.replace(exp.Literal.string(text))
    if any("⟨" in (lit.this or "") for lit in tree.find_all(exp.Literal) if lit.is_string):
        raise NotSafe("a name the question did not hold")


def _as_approved(tree: exp.Expression, view: View) -> tuple[exp.Expression, list[tuple[str, Any]]]:
    """Each table as the approved one, written in full (the warehouse's search path never picks which ``orders``),
    read through its default filters, as the compiler writes them: the rows it leaves out by default (cancelled
    orders) stay out, and so do the lines of a document left out by default (a cancelled order's lines)."""
    from core2.compile.compiler import _Compiler, column_sql, equal_keys
    from core2.resolve.resolver import documents_left_out

    model, dialect = view.model, view.dialect
    compiler = _Compiler(None, model, dialect)
    left_out: list[tuple[str, Any]] = []

    def full(key: str, alias: exp.Identifier) -> exp.Table:
        t = model.tables[key]
        table = D.table_expr(t.database, t.schema_name, t.name, dialect)
        table.set("alias", exp.TableAlias(this=alias))
        return table

    def kept(key: str, alias: exp.Identifier) -> list[exp.Expression]:
        return [compiler.predicate(column_sql(model, f.column, alias, dialect), f.op, list(f.values), f.column)
                for f in model.tables[key].default_filters if f.column in model.columns]

    for node in list(tree.find_all(exp.Table)):
        table = view.table_of(node)
        if table is None:
            raise NotSafe(f"{node.sql(dialect=dialect)} is not one of the approved tables")
        named = (node.args["alias"].this if node.args.get("alias") else node.this).copy()
        row = exp.to_identifier("dq_row")
        conditions = kept(table.key, row)
        left_out += [(table.key, f) for f in model.tables[table.key].default_filters if f.column in model.columns]
        for head in [] if conditions else documents_left_out(model, table.key):
            link = next((j for j in model.joins.values() if j.from_table == table.key and j.to_table == head.key
                         and j.cardinality == "many_to_one" and P.usable(j)), None)
            if link is None:
                continue
            doc = exp.to_identifier("dq_head")
            keys = [equal_keys(model, column_sql(model, f, row, dialect), f, column_sql(model, t, doc, dialect), t,
                               dialect) for f, t in zip(link.from_columns, link.to_columns)]
            conditions.append(exp.Exists(this=exp.select("1").from_(full(head.key, doc))
                                         .where(exp.and_(*keys, *kept(head.key, doc)))))
            left_out += [(head.key, f) for f in head.default_filters if f.column in model.columns]
        if conditions:
            # The approved columns by name, never *: a classified workspace's executor refuses a query that reads
            # every column, and nothing past the approved ones is ever read through it.
            columns = [exp.column(D.ident(c.name, dialect), table=row.copy())
                       for c in sorted(table.columns.values(), key=lambda c: c.name)]
            inner = exp.select(*columns).from_(full(table.key, row)).where(exp.and_(*conditions))
            node.replace(exp.Subquery(this=inner, alias=exp.TableAlias(this=named)))
        else:
            node.replace(full(table.key, named))
    return tree, [f for i, f in enumerate(left_out) if f not in left_out[:i]]


def _limited(tree: exp.Expression, cap: int) -> exp.Expression:
    limit = tree.args.get("limit")
    number = (limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression) if limit else None
    if not (isinstance(number, exp.Literal) and number.is_int and int(number.this) <= cap):
        tree = tree.limit(cap)
    return tree


# ── the answer ───────────────────────────────────────────────────────────────────────────────────────────────────

# Words of a part's place that say nothing of where it is ("the output column", "the condition").
_PLAIN_WORDS = {"the", "output", "column", "columns", "condition", "query", "where", "and", "nothing", "its", "outputs",
                "select", "from", "group", "order", "join", "with", "row", "rows"}


def covers(asked: list[Any], written: Written, sql: str, columns: list[str]) -> list[str]:
    """The parts of the request the planner listed that the query does not give: each must be among the parts the
    query names, and each part it names must be an output column or a condition of the query."""
    given, missing = [], []
    plain = re.sub(r"\s+", " ", sql.casefold())
    outputs = {c.casefold() for c in columns}
    for p in written.parts:
        where = p["in"].strip().casefold()
        if where and (where in outputs or re.sub(r"\s+", " ", where) in plain or any(
                w in plain for w in re.findall(r"[a-z_][a-z0-9_]{2,}", where) if w not in _PLAIN_WORDS)):
            given.append(p["words"].casefold())
        else:
            missing.append(p["words"])         # said to be given, and nowhere in the query
    for part in asked:
        words = str(getattr(part, "words", "") or "").casefold().strip()
        if words and not any(words in g or g in words for g in given):
            missing.append(str(getattr(part, "words", "")))
    return missing


def worked_out(view: View, checked: Checked) -> str:
    """How the answer was worked out, in code's words (never the AI's): what it read, through which links, what it
    worked out, by what, and on what it filtered."""
    model = view.model
    name = lambda key: model.tables[key].business_name or model.tables[key].name  # noqa: E731
    col = lambda key: model.columns[key].business_name or model.columns[key].name  # noqa: E731
    text = "Worked out with a query written for this question and checked before it ran: from " + \
        _listed([name(t) for t in checked.tables]).lower()
    if checked.joins:
        text += ", each matched by " + _listed([
            " and ".join(f"{model.columns[f].business_name or model.columns[f].name}" for f in j.from_columns)
            for j in checked.joins]).lower()
    if checked.grouped:
        text += ", by " + _listed([col(c) for c in checked.grouped]).lower()
    if checked.filtered:
        text += ", for the rows chosen by " + _listed([col(c) for c in checked.filtered]).lower()
    return text + "."


def _listed(items: list[str]) -> str:
    items = list(dict.fromkeys(i for i in items if i))
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def answer(question: str, view: View, checked: Checked, result: Any, *, sees_query: bool, question_id: str,
           data_source: str, started: float, notes: list[str]) -> dict[str, Any]:
    """The usual answer card for the rows the query returned."""
    from core2.answer.builder import frame

    headers = [str(c) for c in result.columns]
    records = [dict(zip(headers, row)) for row in result.rows]
    numeric = {h for h in headers if records and all(isinstance(r.get(h), (int, float)) and not isinstance(r.get(h), bool)
                                                       for r in records if r.get(h) is not None)}
    labels = {h: h.replace("_", " ").strip().capitalize() for h in headers}
    formats = {h: "number" if h in numeric else "text" for h in headers}
    how = worked_out(view, checked)
    if len(records) == 1 and len(headers) == 1:
        value = records[0][headers[0]]
        headline = f"{labels[headers[0]]}: {_shown(value)}."
    elif len(records) == 1:
        headline = "; ".join(f"{labels[h]} {_shown(records[0][h])}" for h in headers) + "."
    elif not records:
        headline = "The query found no rows that match."
    else:
        headline = f"{len(records):,} rows{' (the first ' + str(ROW_CAP) + ')' if result.truncated else ''}."
    payload = frame(question, headline=headline, caveats=[how, *notes], chart=None, kpi=None, suggestions=[],
                    headers=headers, labels=labels, records=records, formats=formats,
                    badges=[{"kind": "direct_query", "text": "Direct query", "text_fr": "Requête directe"}],
                    sql=checked.tree.sql(dialect=view.dialect, pretty=True) if sees_query else "",
                    row_count=len(records), duration_ms=(time.perf_counter() - started) * 1000,
                    data_source=data_source, question_id=question_id, notes=notes, model_version=view.model.version)
    if len(records) == 1 and len(headers) == 1 and headers[0] in numeric:
        payload["kpi"] = {"value": records[0][headers[0]], "label": labels[headers[0]], "format": "number"}
    payload["direct_query"] = True
    payload["trust"]["direct_query"] = True
    payload["trust"]["worked_out"] = how
    return payload


def _shown(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1e15 else str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return "no value" if value is None else str(value)


def left_out_notes(view: View, checked: Checked) -> list[str]:
    """The rows code left out of each table read by default (a document's included), said on the answer."""
    from core2.plan.catalog import left_out_note

    return [left_out_note(view.model, key, f) for key, f in checked.left_out]
