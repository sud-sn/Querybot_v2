"""The synthetic warehouses must be what their ground truth says they are.

core2 is graded against each domain's truth, so a truth that the data does not
support would grade a correct learner as wrong (or a wrong one as right). These
checks run the truth against the rows themselves: keys are unique, every join's
match rate agrees with the trust it is given, every date, measure and label
column exists with the shape claimed, and every golden question's reference
query runs and names only columns that exist. The benchmark-only domains are held to
the same checks: the benchmark's numbers are only as true as their truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals.core2 import domains
from evals.core2.framework import Built, Domain, materialize
from evals.core2.naming import STYLES

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "core2" / "golden"
NAMES = domains.available()
EVERY = NAMES + list(domains.BENCHMARK)
_cache: dict[str, tuple[Domain, Built]] = {}


def _descriptive(name: str) -> tuple[Domain, Built]:
    if name not in _cache:
        domain = domains.build(name)
        _cache[name] = (domain, materialize(domain, "descriptive"))
    return _cache[name]


def _column_exists(domain: Domain, ref: str) -> bool:
    table, _, column = ref.partition(".")
    try:
        return column in domain.table(table).data.columns
    except StopIteration:
        return False


def _placeholders(domain: Domain) -> list:
    cal = domain.truth.calendar or {}
    return list(cal.get("placeholders") or [])


@pytest.mark.parametrize("name", EVERY)
def test_every_style_loads_the_same_rows(name):
    domain, built = _descriptive(name)
    counts = {t.name: len(t.data) for t in domain.tables}
    for style in STYLES:
        other = built if style == "descriptive" else materialize(domain, style)
        for logical, n in counts.items():
            got = other.con.execute(f'SELECT COUNT(1) FROM "{other.t(logical)}"').fetchone()[0]
            assert got == n, (style, logical)


@pytest.mark.parametrize("name", EVERY)
def test_primary_keys_are_unique_and_complete(name):
    domain, built = _descriptive(name)
    for table, key in domain.truth.primary_keys.items():
        cols = ", ".join(f'"{c}"' for c in key)
        total, distinct, nulls = built.con.execute(
            f'SELECT COUNT(1), COUNT(DISTINCT ({cols})), SUM(CASE WHEN {" OR ".join(f"{c} IS NULL" for c in key)} '
            f'THEN 1 ELSE 0 END) FROM "{table}"').fetchone()
        assert total == distinct and not nulls, (table, key)


@pytest.mark.parametrize("name", EVERY)
def test_each_join_matches_as_often_as_its_trust_says(name):
    domain, built = _descriptive(name)
    placeholders = _placeholders(domain)
    for j in domain.truth.joins:
        for from_column, to_column in j.pairs():
            assert _column_exists(domain, f"{j.from_table}.{from_column}"), j
            assert _column_exists(domain, f"{j.to_table}.{to_column}"), j
        to_key = ", ".join(f'"{t}"' for _, t in j.pairs())
        unique = built.con.execute(
            f'SELECT COUNT(1) = COUNT(DISTINCT ({to_key})) FROM "{j.to_table}"').fetchone()[0]
        assert unique, f"{j.to_table}.{to_key} is not unique"
        if j.cast:
            # stored as different types ('00042' against 42): the same key once both are read as numbers
            kinds = {built.con.execute(f'SELECT typeof("{c}") FROM "{t}" LIMIT 1').fetchone()[0]
                     for t, c in ((j.from_table, j.from_column), (j.to_table, j.to_column))}
            assert len(kinds) == 2, (j, kinds)
            on = " AND ".join(f'TRY_CAST(t."{t}" AS BIGINT) = TRY_CAST(f."{f}" AS BIGINT)' for f, t in j.pairs())
        else:
            on = " AND ".join(f't."{t}" = f."{f}"' for f, t in j.pairs())
        skip = ""
        if j.to_table == (domain.truth.calendar or {}).get("table") and placeholders:
            skip = f' AND f."{j.from_column}" NOT IN ({", ".join(repr(p) for p in placeholders)})'
        rows, matched = built.con.execute(
            f'SELECT COUNT(1), COUNT(t."{j.to_column}") FROM "{j.from_table}" f '
            f'LEFT JOIN "{j.to_table}" t ON {on} '
            f'WHERE f."{j.from_column}" IS NOT NULL{skip}').fetchone()
        rate = matched / rows if rows else 0.0
        if j.trust == "verified":
            assert rate >= 0.99, (j, rate)
        else:
            assert 0.5 < rate < 0.99, (j, rate)


@pytest.mark.parametrize("name", EVERY)
def test_dates_measures_and_labels_point_at_real_columns(name):
    domain, built = _descriptive(name)
    truth = domain.truth
    joined = {(j.from_table, j.from_column) for j in truth.joins}
    defaults: dict[str, int] = {}
    for d in truth.dates:
        assert _column_exists(domain, f"{d.table}.{d.column}"), d
        if d.via_calendar:
            assert (d.table, d.column) in joined, f"{d} says it is a calendar key but no join says so"
        if d.default:
            defaults[d.table] = defaults.get(d.table, 0) + 1
        if d.kind == "audit":
            assert not d.default, d
    assert all(n == 1 for n in defaults.values()), defaults
    for kind_table, kind in truth.kinds.items():
        if kind in ("fact", "snapshot"):
            assert defaults.get(kind_table) == 1, f"{kind_table} has no default date in the truth"
    for m in truth.measures:
        if m.column is not None:
            assert _column_exists(domain, f"{m.table}.{m.column}"), m
        if m.unit_column:
            assert _column_exists(domain, m.unit_column), m
        if m.additivity == "semi_additive":
            assert m.time_aggregation in ("last", "average"), m
    for table, label in truth.labels.items():
        key = truth.primary_keys[table]
        assert len(key) == 1
        if "+" in label:
            # A person named in two parts: two people can share a name, so only the parts must exist.
            assert all(_column_exists(domain, f"{table}.{part}") for part in label.split("+")), label
            continue
        per_key = built.con.execute(
            f'SELECT MAX(n) FROM (SELECT COUNT(DISTINCT "{label}") n FROM "{table}" GROUP BY "{key[0]}")').fetchone()[0]
        distinct = built.con.execute(f'SELECT COUNT(DISTINCT "{label}") = COUNT(1) FROM "{table}"').fetchone()[0]
        assert per_key == 1 and distinct, f"{table}.{label} does not name one row each"
    for ref in list(truth.statuses) + truth.not_measures + list(truth.sensitive):
        assert _column_exists(domain, ref), ref


def _plan_columns(plan: dict) -> list[str]:
    refs: list[str] = []
    for m in plan.get("measures", []):
        if "column" in m:
            refs.append(m["column"])
    refs += [g for g in plan.get("group_by", []) if not g.startswith("time:")]
    refs += [f["column"] if "column" in f else f["measure"]["column"] for f in plan.get("filters", [])]
    time = plan.get("time") or {}
    if time.get("date"):
        refs.append(time["date"])
    for path in (plan.get("via") or {}).values():
        refs += path
    return refs


@pytest.mark.parametrize("name", NAMES)
def test_golden_questions_run_and_name_real_columns(name):
    path = GOLDEN / f"{name}.yaml"
    assert path.exists(), f"{name} has no golden questions"
    domain, built = _descriptive(name)
    golden = yaml.safe_load(path.read_text())
    assert golden["domain"] == name
    ids = [q["id"] for q in golden["questions"]]
    assert len(ids) == len(set(ids))
    for q in golden["questions"]:
        for turn in q.get("turns", [q]):
            for ref in _plan_columns(turn.get("plan") or {}):
                assert _column_exists(domain, ref), (q["id"], ref)
            expect = turn.get("expect") or {}
            if turn.get("reference_sql"):
                rows = built.con.execute(turn["reference_sql"]).fetchall()
                if "row_count" in expect:
                    assert len(rows) == expect["row_count"], (q["id"], len(rows))
                else:
                    assert rows, q["id"]
            else:
                assert expect.get("asks") or expect.get("declines"), q["id"]
