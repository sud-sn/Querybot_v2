"""
core/variance_bridge.py

Why a total moved between two results: a bridge from one total to the other,
and -- when both results carry a quantity beside the value -- the move split
into volume, mix and price.

Two results of the same breakdown (revenue by region this year and last; stock
value by item this month and last) say how much each member moved. They do not
say which moves matter, and a list of changes does not add up to anything the
reader can check. A bridge does: the first total, each member's move, the rest
of the members together, and the second total -- and the steps land exactly on
it. Price-volume-mix goes one level down, for a value that is a price times a
quantity: how much of the move is more (or fewer) units, how much is a shift
towards dearer or cheaper members, and how much is each member's own price.

Pure arithmetic over rows already released to the reader. No SQL, no model,
nothing leaves the process; the same rows give the same bridge every time.

Refused rather than approximated, with the reason:
  * a measure that does not add up across members -- a percentage, an average,
    a unit price (core.analysis_contract.measure_additivity);
  * a result with more than one row per member: matched on the member alone,
    one of its rows would stand for all of them.

The price-volume-mix convention, on the members present in both results (C):

    volume = (Q1(C) - Q0(C)) * P0(C)          P0(C) = V0(C) / Q0(C)
    mix    = sum_i Q1_i * P0_i - Q1(C) * P0(C)
    price  = sum_i (P1_i - P0_i) * Q1_i

so volume + mix + price = V1(C) - V0(C) exactly. A member only in the second
result is "new", one only in the first is "lost"; a member whose quantity is
zero or negative on either side has no price, and its move is reported whole
as "other". Every part is kept, so the parts always sum to the total move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How many members a bridge names before the rest become one step.
BRIDGE_ITEMS = 5


class BridgeRefused(ValueError):
    """The two results cannot carry a bridge; `reason` says why, for the reader."""

    def __init__(self, reason: str, **detail: Any):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Bridge:
    start: float
    end: float
    change: float
    # (member, move), largest moves first; `rest` is every other member's.
    steps: tuple[tuple[str, float], ...]
    rest: float
    rest_members: int
    # The rest's moves counted without their signs: two members moving +50 and
    # -50 net to nothing and are still movement the named members do not explain.
    rest_gross: float = 0.0

    @property
    def explained_share(self) -> float | None:
        """How much of the gross movement the named members account for."""
        named = sum(abs(move) for _, move in self.steps)
        gross = named + self.rest_gross
        if not gross:
            return None
        return named / gross


@dataclass(frozen=True)
class PriceVolumeMix:
    start: float
    end: float
    volume: float
    mix: float
    price: float
    new: float
    lost: float
    other: float
    # (member, price effect), largest first: which members' own prices moved.
    price_drivers: tuple[tuple[str, float], ...] = field(default=())
    # (member, mix effect), largest first: which members' weight shifted.
    mix_drivers: tuple[tuple[str, float], ...] = field(default=())

    @property
    def change(self) -> float:
        return self.end - self.start

    def parts(self) -> dict[str, float]:
        return {"volume": self.volume, "mix": self.mix, "price": self.price,
                "new": self.new, "lost": self.lost, "other": self.other}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _require_additive(column: str) -> None:
    from core.analysis_contract import measure_additivity

    aggregation, reason = measure_additivity(column)
    if aggregation == "non_additive":
        raise BridgeRefused("non_additive", column=column, why=reason)


def _by_member(rows: list[dict], member: str, columns: tuple[str, ...]) -> dict[str, tuple[float | None, ...]]:
    found: dict[str, tuple[float | None, ...]] = {}
    for row in rows:
        key = str(row.get(member))
        if key in found:
            raise BridgeRefused("not_one_row_per_member", member=member)
        found[key] = tuple(_number(row.get(column)) for column in columns)
    return found


def variance_bridge(
    before: list[dict], after: list[dict], *, member: str, value: str,
    other_value: str | None = None, items: int = BRIDGE_ITEMS,
) -> Bridge:
    """The first total, the members' moves, and the second total.

    `other_value` is the value's column in `after` when it is named
    differently there. A member missing from one side moves from (or to)
    nothing, so the steps always sum to the change.
    """
    other_value = other_value or value
    _require_additive(value)
    _require_additive(other_value)
    first = _by_member(before, member, (value,))
    second = _by_member(after, member, (other_value,))
    moves: dict[str, float] = {}
    start = end = 0.0
    counted = False
    for key in list(first) + [k for k in second if k not in first]:
        a = first.get(key, (None,))[0]
        b = second.get(key, (None,))[0]
        if a is None and b is None:
            continue
        counted = True
        start += a or 0.0
        end += b or 0.0
        moves[key] = (b or 0.0) - (a or 0.0)
    if not counted:
        raise BridgeRefused("no_values", value=value)
    ranked = sorted(moves.items(), key=lambda pair: (-abs(pair[1]), pair[0]))
    named = tuple((key, move) for key, move in ranked[:items] if move)
    rest_moves = [move for key, move in ranked if (key, move) not in named]
    return Bridge(start=start, end=end, change=end - start, steps=named,
                  rest=sum(rest_moves), rest_members=sum(1 for move in rest_moves if move),
                  rest_gross=sum(abs(move) for move in rest_moves))


def price_volume_mix(
    before: list[dict], after: list[dict], *, member: str, quantity: str, value: str,
    items: int = BRIDGE_ITEMS,
) -> PriceVolumeMix:
    """Split the value's move into volume, mix, price, new, lost and other."""
    _require_additive(value)
    _require_additive(quantity)
    first = _by_member(before, member, (quantity, value))
    second = _by_member(after, member, (quantity, value))

    start = sum(v or 0.0 for _, v in first.values())
    end = sum(v or 0.0 for _, v in second.values())
    new = sum(v or 0.0 for key, (_, v) in second.items() if key not in first)
    lost = -sum(v or 0.0 for key, (_, v) in first.items() if key not in second)

    both = [key for key in first if key in second]
    priced = [key for key in both
              if (first[key][0] or 0.0) > 0 and (second[key][0] or 0.0) > 0
              and first[key][1] is not None and second[key][1] is not None]
    other = sum(((second[key][1] or 0.0) - (first[key][1] or 0.0) for key in both if key not in priced), 0.0)

    q0 = sum(first[key][0] or 0.0 for key in priced)
    q1 = sum(second[key][0] or 0.0 for key in priced)
    v0 = sum(first[key][1] or 0.0 for key in priced)
    if not priced or not q0:
        raise BridgeRefused("no_priced_members", quantity=quantity, value=value)
    p0_all = v0 / q0
    p0 = {key: (first[key][1] or 0.0) / (first[key][0] or 1.0) for key in priced}
    p1 = {key: (second[key][1] or 0.0) / (second[key][0] or 1.0) for key in priced}

    volume = (q1 - q0) * p0_all
    mix_by = {key: ((second[key][0] or 0.0) - q1 * (first[key][0] or 0.0) / q0) * p0[key] for key in priced}
    price_by = {key: (p1[key] - p0[key]) * (second[key][0] or 0.0) for key in priced}

    def drivers(effects: dict[str, float]) -> tuple[tuple[str, float], ...]:
        ranked = sorted(effects.items(), key=lambda pair: (-abs(pair[1]), pair[0]))
        return tuple((key, effect) for key, effect in ranked[:items] if abs(effect) > 1e-9)

    return PriceVolumeMix(
        start=start, end=end, volume=volume, mix=sum(mix_by.values()), price=sum(price_by.values()),
        new=new, lost=lost, other=other, price_drivers=drivers(price_by), mix_drivers=drivers(mix_by),
    )
