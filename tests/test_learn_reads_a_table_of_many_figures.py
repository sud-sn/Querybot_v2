"""Learn finishes on a table of many figures, and finds what it found before.

How a number adds up is read from a sample of rows: an amount that is a quantity times a price, an amount that is
another less a third. Each was tried on every three figures of a table, every row of the sample each time: a match
table of a public benchmark, 105 figures (betting odds, player positions), is half a million triples of 5,000 rows,
and Learn never finished it. A quick look at the first rows of every figure at once now rules out what cannot hold;
what the first rows cannot settle is read in full, as before, so nothing is found or missed that was not before.

Invented data only.
"""

from __future__ import annotations

import random

import pytest

from core2.bootstrap import arithmetic
from core2.bootstrap.arithmetic import _check_table
from core2.bootstrap.measures import MeasureFinding


def _finding(column, *, agg="sum", fmt="number"):
    return MeasureFinding(table="t", column=column, agg=agg, additivity="additive" if agg == "sum" else "non_additive",
                          format=fmt, name=column.replace("_", " ").capitalize())


def _table(rng, size, noise, *, holes=0.0, late=0, zeros=0.0):
    """Rows of a sales line: quantity, unit price, amount = quantity x price, gross, discount, net = gross - discount,
    and ``noise`` unrelated figures. ``late``: the relations hold only from that row on; ``holes``: the share of
    empty cells; ``zeros``: the share of rows whose amount is nought."""
    rows = {c: [] for c in ("quantity", "unit_price", "line_amount", "gross_amount", "discount_amount", "net_amount")}
    rows.update({f"figure_{i:02d}": [] for i in range(noise)})
    for r in range(size):
        q, p = float(rng.randint(1, 40)), round(rng.uniform(1, 300), 2)
        g = round(rng.uniform(10, 5000), 2)
        d = round(g * rng.uniform(0, 0.3), 2)
        holds = r >= late
        amount = round(q * p, 2) if holds else round(rng.uniform(1, 9000), 2)
        if rng.random() < zeros:
            amount = 0.0
        values = {"quantity": q, "unit_price": p, "line_amount": amount, "gross_amount": g, "discount_amount": d,
                  "net_amount": round(g - d, 2) if holds else round(rng.uniform(1, 5000), 2)}
        values.update({f"figure_{i:02d}": round(rng.uniform(-50, 900), 3) for i in range(noise)})
        for c, v in values.items():
            rows[c].append(None if rng.random() < holes else v)
    return rows


def _findings(rows):
    out = []
    for c in rows:
        if c == "unit_price":
            out.append(_finding(c, agg="avg", fmt="currency"))
        elif c in ("line_amount", "gross_amount", "discount_amount", "net_amount"):
            out.append(_finding(c, fmt="currency"))
        else:
            out.append(_finding(c))
    return out


def _read(found, same, derived):
    return (sorted((m.column, m.agg, m.additivity, m.format, m.time_aggregation,
                    tuple((e.kind, e.detail) for e in m.evidence)) for m in found),
            sorted(same), sorted((m.column, m.agg, m.name, m.times, m.after) for m in derived))


def _check(rows, quick=True, monkeypatch=None):
    found, derived = _findings(rows), []
    if not quick:
        monkeypatch.setattr(arithmetic._Quick, "ruled_out", lambda *a, **k: False)
    same = _check_table("t", found, rows, derived=derived)
    if not quick:
        monkeypatch.undo()
    return _read(found, same, derived)


def test_a_table_of_many_figures_reads_few_triples_in_full(monkeypatch):
    rows = _table(random.Random(7), 1000, noise=24)
    calls = [0]
    real = arithmetic._filled

    def counted(*series):
        if len(series) == 3:
            calls[0] += 1
        return real(*series)

    monkeypatch.setattr(arithmetic, "_filled", counted)
    found = _findings(rows)
    _check_table("t", found, rows, derived=[])
    of = {m.column: m for m in found}
    # The relations are still found...
    assert of["unit_price"].agg == "avg" and any(e.kind == "per_unit" for e in of["unit_price"].evidence)
    assert any(e.kind == "difference" for e in of["net_amount"].evidence)
    # ...while of the ~7,600 product triples and ~22,000 difference triples few are read row by row.
    assert calls[0] < 200, calls[0]


@pytest.mark.parametrize("case", [
    dict(size=1200, noise=8),                         # dense: the first rows settle every triple
    dict(size=2000, noise=6, late=300),               # holds only after the first rows: never found either way
    dict(size=1500, noise=6, holes=0.6),              # mostly empty: the window cannot settle many, read in full
    dict(size=150, noise=6),                          # shorter than the window: the window is every row
    dict(size=40, noise=4, holes=0.6),                # so few filled rows that nothing is read as holding
    dict(size=900, noise=6, zeros=0.5),               # half the amounts nought: read where the amount is not
])
def test_the_quick_look_finds_and_misses_what_reading_every_row_does(case, monkeypatch):
    for seed in range(4):
        rows = _table(random.Random(seed), **case)
        assert _check(rows) == _check(rows, quick=False, monkeypatch=monkeypatch), (case, seed)


def test_the_first_hundred_filled_rows_past_the_window_are_read_in_full(monkeypatch):
    """A relation whose columns are empty on the first rows: the window holds fewer than a hundred rows where all
    three hold something, so it settles nothing, and the relation is still found."""
    rows = _table(random.Random(3), 1200, noise=4)
    for c in ("quantity", "unit_price", "line_amount"):
        rows[c][:400] = [None] * 400
    found = _findings(rows)
    _check_table("t", found, rows, derived=[])
    assert {m.column: m for m in found}["unit_price"].agg == "avg"
    assert _check(rows) == _check(rows, quick=False, monkeypatch=monkeypatch)
