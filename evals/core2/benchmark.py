"""The accuracy benchmark: how well Learn finds joins, metrics and dates, by the two numbers agreed on.

For every domain (the gated six, the benchmark-only ones, and the public schemas whose data
is cached) and every naming style, Learn runs on the warehouse and three numbers are taken
for each of joins, metrics and dates:

* accuracy, with no human input: the share of what the truth holds that the learned model
  holds as the truth does (every invented join, and every number learned as a metric that
  is not one, counts against it);
* review share: the share of what Learn concluded that it sends to an admin to confirm;
* auto precision: the share right among what Learn concluded without sending it to review.

The targets agreed for the new core: accuracy at least 85% for each business type and naming
style, review share at most 15%, auto precision at least 95%. Nothing here gates a build: the
benchmark measures where the learner stands, phase by phase.

    python -m evals.core2.benchmark                        # every domain, every style
    python -m evals.core2.benchmark --domains networking --styles generic
    python -m evals.core2.benchmark --json report.json     # the numbers, for the release page
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from core2.bootstrap import names
from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.model.schema import SemanticModel
from core2.resolve.paths import usable
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import Built, Domain, materialize
from evals.core2.learn_eval import _maps
from evals.core2.naming import STYLES

AREAS = ("joins", "metrics", "dates")
TARGETS = {"accuracy": 0.85, "review_share": 0.15, "auto_precision": 0.95}
_TRUSTED = ("verified", "declared", "admin")


@dataclass
class Tally:
    """One area of one domain in one style."""

    right: int = 0          # truth items the model holds as the truth does
    expected: int = 0       # truth items, plus every invented one
    flagged: int = 0        # concluded, and sent to review
    concluded: int = 0      # concluded at all
    auto_right: int = 0     # concluded without review, and right
    auto: int = 0           # concluded without review (and the truth says something about)
    misses: list[str] = field(default_factory=list)

    def add(self, other: "Tally") -> None:
        for name in ("right", "expected", "flagged", "concluded", "auto_right", "auto"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def numbers(self) -> dict[str, float | None]:
        return {"accuracy": self.right / self.expected if self.expected else None,
                "review_share": self.flagged / self.concluded if self.concluded else None,
                "auto_precision": self.auto_right / self.auto if self.auto else None}


@dataclass
class Result:
    domain: str
    style: str
    areas: dict[str, Tally]


def _reviewed_objects(model: SemanticModel) -> set[str]:
    return {r.object for r in model.review}


def _column(c_of: dict[str, str], key: str) -> str:
    """The logical column name of a model column key ("?" when it maps to none)."""
    return c_of.get(key, "?").split(".", 1)[-1]


def _joins(model: SemanticModel, built: Built, t_of: dict[str, str], c_of: dict[str, str]) -> Tally:
    """A link of the truth is right when a question can follow it with no one's say: found, on the
    same column pairs, not rejected and not held back as one of several a column could mean.
    How far it is trusted is noted, not counted: a link the data matches at 97% is used either way."""
    tally = Tally()
    truth = built.domain.truth
    reviewed = _reviewed_objects(model)

    def key(from_table: str, to_table: str, pairs: Iterable[tuple[str, str]]) -> tuple:
        return from_table, to_table, tuple(sorted(pairs))

    def label(k: tuple) -> str:
        return f"{k[0]}.{'+'.join(p[0] for p in k[2])} -> {k[1]}"

    def held(j) -> bool:
        return j.key in reviewed or not usable(j)

    copies = {q["object"] for q in truth.quality if q["kind"] == "backup_copy"}
    expected = {key(j.from_table, j.to_table, j.pairs()): j for j in truth.joins}
    learned = {}
    for j in model.joins.values():
        if j.trust == "rejected" or t_of.get(j.from_table) in copies:
            continue                         # a backup copy's own links are its original's: not scored
        pairs = [(_column(c_of, f), _column(c_of, t)) for f, t in zip(j.from_columns, j.to_columns)]
        learned[key(t_of.get(j.from_table, "?"), t_of.get(j.to_table, "?"), pairs)] = j
    for k, want in expected.items():
        got = learned.get(k)
        right = got is not None and not held(got)
        tally.expected += 1
        tally.right += right
        if got is None:
            tally.misses.append(f"{label(k)}: not found")
        elif not right:
            tally.misses.append(f"{label(k)}: sent to review")
        elif (got.trust in _TRUSTED) != (want.trust == "verified"):
            tally.misses.append(f"note {label(k)}: trusted as {got.trust}, data supports {want.trust}")
    for k, got in learned.items():
        flagged = held(got)
        tally.concluded += 1
        tally.flagged += flagged
        if flagged:
            continue
        tally.auto += 1
        if k in expected:
            tally.auto_right += 1
        else:
            tally.expected += 1                  # an invented link, which questions follow
            tally.misses.append(f"invented {label(k)} ({got.trust})")
    return tally


def _metrics(model: SemanticModel, built: Built, t_of: dict[str, str], c_of: dict[str, str]) -> Tally:
    tally = Tally()
    truth = built.domain.truth
    reviewed = _reviewed_objects(model)
    learned: dict[str, list] = {}
    counts_rows: set[str] = set()          # distinct counts of a table's own key: a count of its rows
    for m in model.measures.values():
        column = getattr(m.expr, "column", None)
        ref = c_of.get(column, "?") if column else f"{t_of.get(m.table, '?')}.*"
        learned.setdefault(ref, []).append(m)
        table, _, name = ref.partition(".")
        if getattr(m.expr, "agg", None) == "count_distinct" and truth.primary_keys.get(table) == [name]:
            learned.setdefault(f"{table}.*", []).append(m)
            counts_rows.add(m.key)
    judged: dict[str, bool] = {}

    def agg(got, want) -> str | None:
        found = getattr(got.expr, "agg", None)
        return "count" if want.column is None and got.key in counts_rows else found

    def is_right(want, got) -> bool:
        if agg(got, want) != want.agg:
            return False
        if want.agg == "count_distinct" or got.key in counts_rows:
            return True                     # recounted at every grain: how it adds up is moot
        return got.additivity == want.additivity and (want.additivity != "semi_additive"
                                                      or got.time_aggregation == want.time_aggregation)

    for want in truth.measures:
        ref = f"{want.table}.{want.column}" if want.column else f"{want.table}.*"
        found = learned.get(ref, [])
        got = next((g for g in found if agg(g, want) == want.agg), found[0] if found else None)
        right = got is not None and is_right(want, got)
        tally.expected += 1
        tally.right += right
        if got is not None:
            judged[got.key] = right
        if not right:
            tally.misses.append(f"{ref}: " + ("not a metric" if got is None else
                                              f"{getattr(got.expr, 'agg', '?')} {got.additivity}, truth "
                                              f"{want.agg} {want.additivity}"))
    not_measures = set(truth.not_measures)
    seen: set[str] = set()
    for ref, found in learned.items():
        for got in found:
            if got.key in seen:
                continue
            seen.add(got.key)
            flagged = got.key in reviewed or got.status == "needs_review"
            tally.concluded += 1
            tally.flagged += flagged
            wrong_kind = ref in not_measures
            if wrong_kind and not flagged:
                tally.expected += 1
                tally.misses.append(f"{ref}: learned as a metric ({getattr(got.expr, 'agg', '?')})")
            if flagged or (got.key not in judged and not wrong_kind):
                continue                    # sent to review, or a number the truth says nothing about
            tally.auto += 1
            tally.auto_right += judged.get(got.key, False)
    return tally


def _dates(model: SemanticModel, built: Built, t_of: dict[str, str], c_of: dict[str, str]) -> Tally:
    tally = Tally()
    truth = built.domain.truth
    asked = {r.object: {r.choice_made, *r.alternatives} for r in model.review if r.key.startswith("default_date:")}
    roles = {c_of.get(r.column, "?"): r for r in model.date_roles.values()}

    def in_review(role) -> bool:
        among = asked.get(role.table)
        return among is not None and (role.is_default or names.readable(model.columns[role.column].name) in among)

    for want in truth.dates:
        ref = f"{want.table}.{want.column}"
        got = roles.get(ref)
        right = got is not None and got.kind == want.kind and got.is_default == want.default
        tally.expected += 1
        tally.right += right
        if not right:
            tally.misses.append(f"{ref}: " + ("not seen as a date" if got is None else
                                              f"learned {got.kind}{' (default)' if got.is_default else ''}, truth "
                                              f"{want.kind}{' (default)' if want.default else ''}"))
        if got is None:
            continue
        flagged = in_review(got) or got.status == "needs_review"
        tally.concluded += 1
        tally.flagged += flagged
        if not flagged:
            tally.auto += 1
            tally.auto_right += right
    return tally


def grade(model: SemanticModel, built: Built) -> dict[str, Tally]:
    """The learned model against the truth of the warehouse it was learned from."""
    t_of, c_of = _maps(model, built)
    return {"joins": _joins(model, built, t_of, c_of), "metrics": _metrics(model, built, t_of, c_of),
            "dates": _dates(model, built, t_of, c_of)}


def learn(built: Built) -> SemanticModel:
    warehouse = DuckDBWarehouse(built.con)
    inventory = from_duckdb(warehouse, declared_fks=built.declared_fks)
    return build_model(warehouse, inventory, client_id=f"bench-{built.domain.name}", options=BuildOptions(workers=1))


def score(domain: Domain, style: str) -> Result:
    built = materialize(domain, style)
    return Result(domain.name, style, grade(learn(built), built))


def _sources(names: list[str] | None) -> list[tuple[str, str]]:
    """(name, kind) of every domain to run: gated, benchmark and the cached public schemas."""
    from evals.core2 import public

    found = ([(n, "gated") for n in domains.available()] + [(n, "benchmark") for n in domains.BENCHMARK]
             + [(n, "public") for n in public.cached()])
    return [s for s in found if names is None or s[0] in names]


def _build(name: str, kind: str) -> Domain:
    from evals.core2 import public

    return public.build(name) if kind == "public" else domains.build(name)


def summarize(results: list[Result]) -> dict:
    """Totals by area: overall, per style, per domain."""
    def total(rows: list[Result]) -> dict:
        out = {}
        for area in AREAS:
            t = Tally()
            for r in rows:
                t.add(r.areas[area])
            out[area] = {**t.numbers(), "items": t.expected}
        return out

    return {"overall": total(results),
            "by_style": {s: total([r for r in results if r.style == s]) for s in dict.fromkeys(r.style for r in results)},
            "by_domain": {d: total([r for r in results if r.domain == d])
                          for d in dict.fromkeys(r.domain for r in results)}}


def _pct(value: float | None) -> str:
    return "  –  " if value is None else f"{value:5.0%}"


def report(results: list[Result], summary: dict) -> str:
    lines = ["accuracy / review share / auto precision, per area", ""]
    lines.append(f"{'domain':22} {'style':12} " + "  ".join(f"{a:^22}" for a in AREAS))
    for r in results:
        cells = []
        for area in AREAS:
            n = r.areas[area].numbers()
            cells.append(f"{_pct(n['accuracy'])} {_pct(n['review_share'])} {_pct(n['auto_precision'])}")
        lines.append(f"{r.domain:22} {r.style:12} " + "  ".join(f"{c:^22}" for c in cells))
    lines.append("")
    for title, block in (("by style", summary["by_style"]), ("by domain", summary["by_domain"])):
        lines.append(title)
        for name, areas in block.items():
            lines.append(f"  {name:20} " + "  ".join(
                f"{a}: {_pct(areas[a]['accuracy'])} {_pct(areas[a]['review_share'])} {_pct(areas[a]['auto_precision'])}"
                for a in AREAS))
    lines.append("")
    lines.append("overall: " + "  ".join(f"{a} {_pct(summary['overall'][a]['accuracy'])} accuracy, "
                                         f"{_pct(summary['overall'][a]['review_share'])} review, "
                                         f"{_pct(summary['overall'][a]['auto_precision'])} right unreviewed"
                                         for a in AREAS))
    lines.append(f"targets: accuracy >= {TARGETS['accuracy']:.0%}, review <= {TARGETS['review_share']:.0%}, "
                 f"right unreviewed >= {TARGETS['auto_precision']:.0%}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--domains", help="comma-separated domain names (default: all)")
    parser.add_argument("--styles", help="comma-separated naming styles (default: all)")
    parser.add_argument("--json", help="write the results and totals to this file")
    parser.add_argument("--misses", action="store_true", help="list every miss")
    args = parser.parse_args(argv)
    names = args.domains.split(",") if args.domains else None
    styles = args.styles.split(",") if args.styles else list(STYLES)
    results = []
    for name, kind in _sources(names):
        domain = _build(name, kind)
        for style in styles:
            result = score(domain, style)
            results.append(result)
            if args.misses:
                for area in AREAS:
                    for miss in result.areas[area].misses:
                        print(f"  [{name} {style} {area}] {miss}", file=sys.stderr)
    summary = summarize(results)
    print(report(results, summary))
    if args.json:
        with open(args.json, "w") as out:
            json.dump({"targets": TARGETS, "summary": summary,
                       "results": [{"domain": r.domain, "style": r.style,
                                    "areas": {a: {**asdict(t), **t.numbers()} for a, t in r.areas.items()}}
                                   for r in results]}, out, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
