"""Ranges of a value ("under 100, 100-499, 500 and over"): their names, and round edges chosen from the data.

A range holds its lower edge and stops before the next: with edges [100, 500], 100 is in "100–499" (whole
numbers) or "100 to under 500" (amounts with cents), and 500 in "500 and over". Edges picked from the data are
round numbers near the values that split the rows (or members) into about equal parts, so each range holds a
fair share and reads as a reader would write it.
"""

from __future__ import annotations

import math

MOST_BANDS = 6


def _number(value: float, *, money: bool) -> str:
    if money:
        return f"${value:,.0f}" if float(value).is_integer() else f"${value:,.2f}"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}".rstrip("0").rstrip(".")


def labels(edges: list[float], *, whole: bool, unit: str = "", money: bool = False) -> list[str]:
    """Each range's name, in order: one more than the edges."""
    if not edges:
        return ["All"]
    tail = f" {unit}" if unit else ""
    out = [f"Under {_number(edges[0], money=money)}{tail}"]
    for low, high in zip(edges, edges[1:]):
        if whole and float(low).is_integer() and float(high).is_integer():
            top = high - 1
            out.append(f"{_number(low, money=money)}{tail}" if top == low else
                       f"{_number(low, money=money)}–{_number(top, money=money)}{tail}")
        else:
            out.append(f"{_number(low, money=money)} to under {_number(high, money=money)}{tail}")
    out.append(f"{_number(edges[-1], money=money)}{tail} and over")
    return out


def _round(value: float) -> float:
    """A round number near ``value``: 1, 2, 2.5 or 5 times a power of ten."""
    if value == 0:
        return 0.0
    sign = -1 if value < 0 else 1
    size = abs(value)
    power = 10 ** math.floor(math.log10(size))
    best = min((1, 2, 2.5, 5, 10), key=lambda step: abs(size - step * power))
    return sign * best * power


def edges_from(cuts: list[float], low: float, high: float, *, whole: bool) -> list[float]:
    """Round, increasing edges near ``cuts`` (values splitting the data into about equal parts), strictly inside
    ``low``..``high``; whole numbers for whole-number values."""
    out: list[float] = []
    for cut in cuts:
        if cut is None or not math.isfinite(cut):
            continue
        edge = _round(cut)
        if whole:
            edge = float(math.floor(edge + 0.5))
        if low < edge <= high and (not out or edge > out[-1]):
            out.append(edge)
    return out[:MOST_BANDS - 1]
