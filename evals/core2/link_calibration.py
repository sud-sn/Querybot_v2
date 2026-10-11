"""Fit, and check, how sure Learn may say it is of a link (core2/bootstrap/link_score.py).

Every link Learn tests on the benchmark's warehouses, in every naming style, is recorded with what Learn found
about it (declared, how well the names agree, the share of rows that match, how much of the target it uses, ...)
and whether the warehouse's truth holds it. A logistic fit on the synthetic warehouses gives the weights; the public
schemas (Chinook, Northwind, Sakila), kept out of the fit, show whether a score of 90% means 9 in 10 are right.

    python -m evals.core2.link_calibration            # fit, print the weights and both reliability tables
    python -m evals.core2.link_calibration --check    # only the reliability of the weights in link_score.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from core2.bootstrap import joins as joins_module
from core2.bootstrap.build import BuildOptions, learn
from core2.bootstrap.inventory import from_duckdb
from core2.bootstrap.link_score import FEATURES, features, link_score
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import Built, materialize
from evals.core2.naming import STYLES

BINS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01)


def _tested(built: Built) -> list[tuple[joins_module.JoinFinding, bool]]:
    """Every link Learn tested on one warehouse, and whether the truth holds it. One Learn tested and then let go
    for a better one of the same column (the products, not the product categories) is marked rejected: it is
    fitted on, never shown, so the reliability tables leave it out."""
    seen: list[joins_module.JoinFinding] = []
    original = joins_module._test

    def recording(*args, **kwargs):
        found = original(*args, **kwargs)
        seen.append(found)
        return found

    joins_module._test = recording
    try:
        warehouse = DuckDBWarehouse(built.con)
        kept = {f.ident for f in learn(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                                       BuildOptions(workers=1, query_history=None)).joins}
    finally:
        joins_module._test = original
    for f in seen:
        if f.ident not in kept:
            f.trust = "rejected"
    logical = built.logical_of()

    def name(table_key: str, column: str) -> str:
        return logical.get((table_key.rsplit(".", 1)[-1].casefold(), column.casefold()), "?")

    truth = {(j.from_table, j.to_table, tuple(sorted(j.pairs()))) for j in built.domain.truth.joins}
    out = []
    for f in seen:
        froms = [name(f.from_table, c) for c in f.from_columns]
        tos = [name(f.to_table, t) for _, t in f.pairs]
        if "?" in froms or "?" in tos:
            continue
        key = (froms[0].split(".")[0], tos[0].split(".")[0],
               tuple(sorted(zip((c.split(".", 1)[1] for c in froms), (t.split(".", 1)[1] for t in tos)))))
        out.append((f, key in truth))
    return out


def collect(names: list[str], public: bool) -> list[tuple[joins_module.JoinFinding, bool]]:
    from evals.core2 import public as public_schemas

    rows = []
    for name in names:
        domain = public_schemas.build(name) if public else domains.build(name)
        for style in STYLES:
            rows += _tested(materialize(domain, style))
            print(f"  {name} {style}: {len(rows)} links so far", file=sys.stderr)
    return rows


def fit(rows: list[tuple[joins_module.JoinFinding, bool]], *, l2: float = 1.0, steps: int = 50) -> dict[str, float]:
    """Logistic regression by Newton's method, with a small penalty so no weight runs off on a perfect split."""
    x = np.array([[1.0, *(features(f)[k] for k in FEATURES)] for f, _ in rows])
    y = np.array([1.0 if right else 0.0 for _, right in rows])
    w = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        gradient = x.T @ (p - y) + penalty @ w
        hessian = x.T @ (x * (p * (1 - p))[:, None]) + penalty
        w -= np.linalg.solve(hessian, gradient)
    return {"bias": round(float(w[0]), 3), **{k: round(float(v), 3) for k, v in zip(FEATURES, w[1:])}}


def reliability(rows: list[tuple[joins_module.JoinFinding, bool]], weights: dict[str, float] | None = None) -> str:
    lines = [f"  {'score':>12} {'links':>6} {'right':>6} {'share right':>12}"]
    scored = [(link_score(f, weights), right) for f, right in rows if f.trust != "rejected"]
    for low, high in zip(BINS, BINS[1:]):
        inside = [right for s, right in scored if low <= s < high]
        if inside:
            lines.append(f"  {low:>5.0%}-{min(high, 1):>5.0%} {len(inside):>6} {sum(inside):>6} "
                         f"{sum(inside) / len(inside):>12.0%}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    from evals.core2 import public as public_schemas

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="only check the weights in link_score.py")
    args = parser.parse_args(argv)
    synthetic = collect([*domains.available(), *domains.BENCHMARK], public=False)
    held_out = collect(public_schemas.cached(), public=True)
    weights = None
    if not args.check:
        weights = fit(synthetic)
        print("weights:", weights)
    print(f"fitted on {len(synthetic)} tested links (synthetic warehouses, every style)")
    print(reliability(synthetic, weights))
    print(f"held out: {len(held_out)} tested links (public schemas, every style)")
    print(reliability(held_out, weights))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
