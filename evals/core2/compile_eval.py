"""Level 2 of core2's evaluation: from a golden plan to the right numbers.

Every golden question with a reference answer carries its plan in logical names
(``order_lines.net_amount``). Here that plan is put in the learned model's own
words (its measure, attribute and date slugs), resolved, compiled for DuckDB and
run; the rows must equal the reference SQL's rows. The same logical query is
compiled for Snowflake, Azure SQL and Oracle as well, and each must read back in
its own dialect.

A plan that cannot be put in the model's words is a learning miss (level 1)
showing up here, and is reported as such, not as a compiler failure.

    python -m evals.core2.compile_eval                    # every domain with golden questions
    python -m evals.core2.compile_eval retail warehouse   # one domain, one naming style
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.compile.compiler import Compiled, compile_query
from core2.model.schema import AggExpr, SemanticModel
from core2.plan.ir import Plan
from core2.resolve.resolver import Context, Logical, ResolveError, resolve
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import Built, Domain, materialize
from evals.core2.learn_eval import _maps

GOLDEN = Path(__file__).parent / "golden"
OTHER_DIALECTS = ("snowflake", "tsql", "oracle")


class Untranslatable(Exception):
    """The golden plan names something the learned model does not have."""


@dataclass
class CaseResult:
    id: str
    status: str            # ok | not_learned | resolve | compile | run | mismatch | dialect
    detail: str = ""
    sql: str = ""
    dialect_errors: dict[str, str] = field(default_factory=dict)


def golden(domain: str) -> dict:
    path = GOLDEN / f"{domain}.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def learn(domain: Domain, style: str) -> tuple[Built, SemanticModel]:
    built = materialize(domain, style)
    warehouse = DuckDBWarehouse(built.con)
    inventory = from_duckdb(warehouse, declared_fks=built.declared_fks)
    model = build_model(warehouse, inventory, client_id=f"eval-{domain.name}", options=BuildOptions(workers=1))
    return built, model


# ── the golden plan in the model's words ───────────────────────────────────


class _Words:
    def __init__(self, model: SemanticModel, built: Built):
        self.model = model
        tables, columns = _maps(model, built)
        self.table_key = {logical: key for key, logical in tables.items()}
        self.column_key = {logical: key for key, logical in columns.items()}

    def column(self, ref: str) -> str:
        try:
            return self.column_key[ref]
        except KeyError:
            raise Untranslatable(f"no column {ref}") from None

    def measure(self, spec: dict) -> str:
        if "ratio" in spec:
            raise Untranslatable("ratios are written as derived measures")
        agg = spec["agg"]
        column = self.column(spec["column"]) if spec.get("column") else None
        table = self.model.columns[column].table if column else self.table_key.get(spec.get("table", ""), "")
        for m in sorted(self.model.measures.values(), key=lambda m: m.key):
            e = m.expr
            if m.table == table and isinstance(e, AggExpr) and e.agg == agg and e.column == column and not e.filters \
                    and not m.filters:
                return m.slug
        raise Untranslatable(f"no measure {agg}({spec.get('column') or spec.get('table')})")

    def attribute(self, ref: str) -> str:
        column = self.column(ref)
        for a in sorted(self.model.attributes.values(), key=lambda a: a.slug):
            if a.column == column:
                return a.slug
        raise Untranslatable(f"{ref} is not an attribute")

    def date(self, ref: str) -> str | None:
        column = self.column(ref)
        for r in self.model.date_roles.values():
            if r.column == column:
                return r.slug
        return None

    def entity_through(self, fk_ref: str) -> str:
        column = self.column(fk_ref)
        for j in self.model.joins.values():
            if j.from_columns == [column] and j.trust != "rejected":
                for e in self.model.entities.values():
                    if e.table == j.to_table:
                        return e.slug
        raise Untranslatable(f"nothing reached through {fk_ref}")

    def field(self, ref: str) -> str:
        return self.date(ref) or self.attribute(ref)

    def plan(self, spec: dict) -> Plan:
        measures = [self.measure(m) for m in spec.get("measures", [])]
        group_by = [g if g.startswith("time:") else self.attribute(g) for g in spec.get("group_by", [])]
        via = {self.attribute(a): self.entity_through(path[0]) for a, path in (spec.get("via") or {}).items()}
        filters = [{"field": self.field(f["column"]), "op": f["op"], "values": f.get("values", [])}
                   for f in spec.get("filters", [])]
        time = dict(spec.get("time") or {})
        if time.get("date"):
            slug = self.date(time["date"])
            if slug is None:
                raise Untranslatable(f"{time['date']} is not a date")
            time["date"] = slug
        sort = []
        for s in spec.get("sort", []):
            by = s["by"]
            if by not in ("change", "pct_change", "share", "period"):
                by = self._sort_slug(by, spec)
            sort.append({"by": by, "desc": s.get("desc", True)})
        return Plan.model_validate({"intent": spec.get("intent"), "measures": measures, "group_by": group_by,
                                    "via": via, "filters": filters, "time": time, "sort": sort,
                                    "limit": spec.get("limit")})

    def _sort_slug(self, ref: str, spec: dict) -> str:
        for m in spec.get("measures", []):
            if m.get("column") == ref:
                return self.measure(m)
        return self.attribute(ref)


# ── comparing answers ──────────────────────────────────────────────────────


def _norm(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, Decimal)):
        number = float(value)
        return float(f"{number:.9g}") if math.isfinite(number) else None
    if isinstance(value, dt.datetime):
        return value.date().isoformat() if value.time() == dt.time() else value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _key(value: Any) -> tuple:
    return (0, "") if value is None else (1, value) if isinstance(value, float) else (2, str(value))


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    return a == b


def same_rows(ref_cols: list[str], ref_rows: list[tuple], got_cols: list[str], got_rows: list[tuple],
              *, order_matters: bool) -> str:
    """'' when the answers agree; otherwise what differs. Columns are matched by their values, not their names."""
    if len(ref_rows) != len(got_rows):
        return f"{len(got_rows)} rows, expected {len(ref_rows)}"
    ref = [[_norm(v) for v in r] for r in ref_rows]
    got = [[_norm(v) for v in r] for r in got_rows]
    chosen: list[int] = []
    for j, name in enumerate(ref_cols):
        want = sorted((r[j] for r in ref), key=_key)
        candidates = [k for k in range(len(got_cols)) if k not in chosen and all(
            _close(a, b) for a, b in zip(sorted((r[k] for r in got), key=_key), want))]
        if not candidates:
            return f"no column holds {name} ({want[:3]}...)"
        same_name = [k for k in candidates if got_cols[k].casefold() == name.casefold()]
        chosen.append((same_name or candidates)[0])
    projected = [[r[k] for k in chosen] for r in got]
    if not order_matters:
        ref, projected = sorted(ref, key=lambda r: [_key(v) for v in r]), sorted(
            projected, key=lambda r: [_key(v) for v in r])
    for i, (a, b) in enumerate(zip(projected, ref)):
        if not all(_close(x, y) for x, y in zip(a, b)):
            return f"row {i + 1}: {a} != {b}"
    return ""


# ── running a case ─────────────────────────────────────────────────────────


def run_case(case: dict, words: _Words, model: SemanticModel, built: Built, today: dt.date,
             reference: DuckDBWarehouse) -> CaseResult:
    cid = case["id"]
    try:
        plan = words.plan(case["plan"])
    except Untranslatable as exc:
        return CaseResult(cid, "not_learned", str(exc))
    try:
        logical: Logical = resolve(plan, model, Context(today=today))
    except ResolveError as exc:
        return CaseResult(cid, "resolve", f"{exc.kind}: {exc.message} {exc.options or ''}")
    try:
        compiled: Compiled = compile_query(logical, model, "duckdb")
    except Exception as exc:     # noqa: BLE001 - reported per case
        return CaseResult(cid, "compile", f"{type(exc).__name__}: {exc}")
    try:
        got = DuckDBWarehouse(built.con).query(compiled.sql)
    except Exception as exc:     # noqa: BLE001
        return CaseResult(cid, "run", f"{type(exc).__name__}: {exc}", compiled.sql)
    expected = reference.query(case["reference_sql"])
    expect = case.get("expect") or {}
    diff = same_rows(expected.columns, expected.rows, got.columns, got.rows,
                     order_matters=bool(expect.get("order_matters")))
    if diff:
        return CaseResult(cid, "mismatch", diff, compiled.sql)
    if "partial_periods" in expect:
        found = sorted(d.isoformat() for d in logical.partial)
        if found != sorted(expect["partial_periods"]):
            return CaseResult(cid, "mismatch", f"partial periods {found}, expected {expect['partial_periods']}",
                              compiled.sql)
    result = CaseResult(cid, "ok", sql=compiled.sql)
    for dialect in OTHER_DIALECTS:
        try:
            compile_query(logical, model, dialect)
        except Exception as exc:     # noqa: BLE001
            result.dialect_errors[dialect] = f"{type(exc).__name__}: {exc}"
    if result.dialect_errors:
        result.status = "dialect"
        result.detail = "; ".join(f"{d}: {e}" for d, e in result.dialect_errors.items())
    return result


def evaluate(domain_name: str, style: str) -> list[CaseResult]:
    spec = golden(domain_name)
    cases = [c for c in spec.get("questions", []) if c.get("plan") and c.get("reference_sql")]
    if not cases:
        return []
    domain = domains.build(domain_name)
    built, model = learn(domain, style)
    reference = DuckDBWarehouse(materialize(domain, "descriptive").con)
    today = dt.date.fromisoformat(str(spec["today"]))
    words = _Words(model, built)
    return [run_case(c, words, model, built, today, reference) for c in cases]


def main(argv: list[str]) -> int:
    names = argv[:1] or [d for d in domains.available() if golden(d)]
    styles = argv[1:2] or ["descriptive", "warehouse", "pascal", "generic"]
    failed = False
    for name in names:
        for style in styles:
            results = evaluate(name, style)
            ok = sum(r.status == "ok" for r in results)
            print(f"{name:<14} {style:<12} {ok}/{len(results)} right")
            for r in results:
                if r.status != "ok":
                    print(f"    {r.id}: {r.status}: {r.detail}")
            # Names that carry meaning must give every answer; generic names may miss what was not learned.
            failed = failed or any(r.status not in ("ok", "not_learned" if style == "generic" else "ok")
                                   for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
