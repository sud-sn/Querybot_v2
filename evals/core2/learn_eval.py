"""Level 1 of core2's evaluation: does QueryBot learn each synthetic warehouse right?

For every domain and naming style, the bootstrap runs against the DuckDB build and
its model is graded against the domain's ground truth: table kinds, keys, joins
(found, trusted as they should be, named by role), the calendar, date roles and
defaults, measures and how they add up, entity labels, and the data-quality traps.

    python -m evals.core2.learn_eval              # every domain, every style
    python -m evals.core2.learn_eval retail generic
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import SemanticModel
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import Built, Domain, materialize
from evals.core2.naming import STYLES


@dataclass
class Score:
    domain: str
    style: str
    parts: dict[str, tuple[int, int]] = field(default_factory=dict)     # name -> (right, total)
    misses: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, miss: str = "") -> None:
        right, total = self.parts.get(name, (0, 0))
        self.parts[name] = (right + int(ok), total + 1)
        if not ok and miss:
            self.misses.append(f"{name}: {miss}")

    def rate(self, name: str) -> float:
        right, total = self.parts.get(name, (0, 0))
        return right / total if total else 1.0


def _maps(model: SemanticModel, built: Built) -> tuple[dict[str, str], dict[str, str]]:
    """Model table keys -> logical table; model column keys -> "table.column" (logical)."""
    physical_to_logical = {phys.casefold(): logical for logical, phys in built.tables.items()}
    tables = {k: physical_to_logical.get(t.name.casefold(), "?") for k, t in model.tables.items()}
    of = built.logical_of()
    columns = {k: of.get((model.tables[c.table].name.casefold(), c.name.casefold()), "?")
               for k, c in model.columns.items() if not c.parts}
    for k, c in model.columns.items():
        if c.parts:   # a name read from two columns: "customers.first_name+last_name"
            refs = [columns.get(p, "?") for p in c.parts]
            columns[k] = refs[0].split(".", 1)[0] + "." + "+".join(r.split(".", 1)[-1] for r in refs)
    return tables, columns


def score(domain: Domain, style: str, model: SemanticModel | None = None) -> tuple[Score, SemanticModel]:
    built = materialize(domain, style)
    if model is None:
        warehouse = DuckDBWarehouse(built.con)
        inventory = from_duckdb(warehouse, declared_fks=built.declared_fks)
        model = build_model(warehouse, inventory, client_id=f"eval-{domain.name}", options=BuildOptions(workers=1))
    t_of, c_of = _maps(model, built)
    truth = domain.truth
    s = Score(domain.name, style)

    for key, table in model.tables.items():
        logical = t_of[key]
        s.add("kinds", table.kind == truth.kinds.get(logical), f"{logical} is {truth.kinds.get(logical)}, learned {table.kind}")
        if logical in truth.primary_keys:
            found = sorted(c_of[c].split(".", 1)[1] for c in table.primary_key)
            s.add("keys", found == sorted(truth.primary_keys[logical]),
                  f"{logical} key is {truth.primary_keys[logical]}, learned {found}")

    learned = {(c_of[j.from_columns[0]], c_of[j.to_columns[0]]): j for j in model.joins.values()}
    expected = {(f"{j.from_table}.{j.from_column}", f"{j.to_table}.{j.to_column}"): j for j in truth.joins}
    for pair, j in expected.items():
        found = learned.get(pair)
        s.add("joins_found", found is not None, f"{pair[0]} -> {pair[1]} not found")
        if found is not None:
            trusted = found.trust == "verified"
            s.add("joins_trust", trusted == (j.trust == "verified"),
                  f"{pair[0]} -> {pair[1]} trusted {found.trust}, data supports {j.trust}")
            if j.role and style != "generic":
                s.add("join_roles", (found.role or "").casefold() == j.role.casefold(),
                      f"{pair[0]} role {found.role!r}, expected {j.role!r}")
    for pair, j in learned.items():
        # A column whose values fit two tables equally is put to an admin as a
        # choice and used by no question until answered: asking is not inventing,
        # as long as the right table is among the choices.
        asked = j.trust == "proposed" and any(e.kind == "ambiguous" for e in j.evidence) and any(
            p[0] == pair[0] and p in expected for p in learned)
        if pair not in expected and not asked:
            s.add("joins_precise", False, f"invented {pair[0]} -> {pair[1]} ({j.trust})")
        else:
            s.add("joins_precise", True)

    if truth.calendar:
        cal_key = next((k for k, logical in t_of.items() if logical == truth.calendar["table"]), None)
        cal = model.calendars.get(cal_key or "")
        s.add("calendar", cal is not None, "calendar not recognised")
        if cal:
            s.add("calendar", c_of[cal.date_column].endswith("." + truth.calendar["date"]), "wrong calendar date column")
            s.add("calendar", bool(cal.key_column) and c_of[cal.key_column].endswith("." + truth.calendar["key"]),
                  "wrong calendar key")
            for attribute, column in truth.calendar["attributes"].items():
                got = cal.attributes.get(attribute)
                s.add("calendar_columns", bool(got) and c_of[got].endswith("." + column),
                      f"calendar {attribute} ({column}) not read")
            if truth.calendar.get("fiscal_year_start_month", 1) != 1:
                s.add("calendar", cal.fiscal_year_start_month == truth.calendar["fiscal_year_start_month"],
                      f"fiscal year starts in month {truth.calendar['fiscal_year_start_month']}, "
                      f"learned {cal.fiscal_year_start_month}")

    roles = {c_of[r.column]: r for r in model.date_roles.values()}
    for d in truth.dates:
        ref = f"{d.table}.{d.column}"
        role = roles.get(ref)
        s.add("dates_found", role is not None, f"{ref} not seen as a date")
        if role is None:
            continue
        if d.kind == "audit":
            s.add("audit_dates", role.kind == "audit" and not role.is_default, f"{ref} is an audit stamp, learned {role.kind}")
        if d.default:
            s.add("default_dates", role.is_default, f"{d.table} should default to {d.column}")
        elif role.is_default:
            s.add("default_dates", False, f"{d.table} defaults to {d.column}, which it should not")

    learned_m = {}
    for m in model.measures.values():
        column = getattr(m.expr, "column", None)
        ref = c_of[column] if column else f"{t_of[m.table]}.*"
        learned_m[(ref, getattr(m.expr, "agg", ""))] = m
    for m in truth.measures:
        ref = f"{m.table}.{m.column}" if m.column else f"{m.table}.*"
        found = learned_m.get((ref, m.agg)) or next((x for (r, _), x in learned_m.items() if r == ref), None)
        s.add("measures_found", found is not None, f"{ref} ({m.agg}) not a measure")
        if found is None:
            continue
        if m.agg == "count_distinct":
            continue   # a distinct count is recomputed at every grain: its additivity is moot
        s.add("measures_additivity", found.additivity == m.additivity and (
            m.additivity != "semi_additive" or found.time_aggregation == m.time_aggregation),
            f"{ref} adds up {m.additivity}, learned {found.additivity}")
        if style != "generic":
            s.add("measures_format", found.format == m.format or {found.format, m.format} <= {"integer", "number"},
                  f"{ref} is {m.format}, learned {found.format}")
        if m.unit_column and style != "generic":
            s.add("measures_unit", bool(found.unit_column) and c_of[found.unit_column] == m.unit_column,
                  f"{ref} unit is {m.unit_column}, learned {found.unit_column and c_of[found.unit_column]}")
    not_measures = set(truth.not_measures)
    for (ref, _), m in learned_m.items():
        if ref in not_measures:
            s.add("measures_precise", False, f"{ref} learned as a measure")
        else:
            s.add("measures_precise", True)

    for table, label in truth.labels.items():
        entity = next((e for e in model.entities.values() if t_of[e.table] == table), None)
        s.add("labels", entity is not None and bool(entity.label_column) and c_of[entity.label_column] == f"{table}.{label}",
              f"{table} should be named by {label}, learned "
              f"{entity and entity.label_column and c_of[entity.label_column]}")

    flagged = {(c_of.get(q.object) or t_of.get(q.object) or q.object, q.kind) for q in model.quality}
    for q in truth.quality:
        s.add("quality", (q["object"], q["kind"]) in flagged, f"{q['kind']} on {q['object']} not flagged")
    return s, model


GATES = {
    # part: minimum share right, for the styles that carry meaningful names
    "kinds": 0.9, "keys": 0.9, "joins_found": 0.95, "joins_precise": 0.95, "joins_trust": 0.9, "join_roles": 0.8,
    "calendar": 1.0, "calendar_columns": 0.9, "dates_found": 0.95, "default_dates": 0.9, "audit_dates": 1.0,
    "measures_found": 0.9, "measures_precise": 0.9, "measures_additivity": 0.85, "measures_format": 0.8,
    "labels": 0.9, "quality": 0.7,
}
# Generic names carry no meaning: the learner must still find structure from values.
GENERIC_GATES = {"kinds": 0.75, "keys": 0.9, "joins_found": 0.8, "joins_precise": 0.95, "calendar": 1.0,
                 "calendar_columns": 0.9, "dates_found": 0.9, "audit_dates": 1.0, "measures_precise": 0.85}


def main(argv: list[str]) -> int:
    names = argv[:1] or domains.available()
    styles = argv[1:] or list(STYLES)
    failed = False
    for name in names:
        domain = domains.build(name)
        for style in styles:
            result, _ = score(domain, style)
            gates = GENERIC_GATES if style == "generic" else GATES
            row = []
            for part in GATES:
                if part not in result.parts:
                    continue
                rate = result.rate(part)
                below = part in gates and rate < gates[part]
                failed |= below
                row.append(f"{part}={rate:.0%}{'!' if below else ''}")
            print(f"{name:14} {style:12} " + " ".join(row))
            for miss in result.misses:
                print(f"    - {miss}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
