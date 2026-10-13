"""How a number adds up, read from the rows themselves.

Names say a lot (UNIT_PRICE is averaged, LINE_AMOUNT summed), and say nothing when a warehouse is
named C07 and C11. The rows say more than either: a column that is another divided by a third on
every row is a price per unit or a ratio, and is never summed; a column between 0 and 1 on every
row is a rate; a "price" that rises with the quantity on the same row is what was charged, and is
summed; two metrics equal on nearly every row are one figure twice, and an admin is asked which
is meant (an allocated quantity that is the on-hand quantity again).

One sample of a table's rows is read (the first few thousand): only aggregates and these checks
leave the warehouse step, never the rows.
"""

from __future__ import annotations

import math
from itertools import combinations, permutations

from core2.bootstrap import names
from core2.bootstrap.inventory import Inventory
from core2.bootstrap.journal import attempt
from core2.bootstrap.measures import _CODE, _MONEY, _QUANTITY, _RATE, _UNIT_PRICE, MeasureFinding, _parts
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import Evidence
from core2.warehouse import dialect as D
from core2.warehouse.runner import Warehouse

SAMPLE = 5000
AGREE = 0.95        # share of rows on which a relation must hold


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, 0.005 * abs(a))


def _share(pairs: list[tuple[float, float]]) -> float:
    """The share of rows on which two values agree; a first hundred that mostly disagree end the count early."""
    if len(pairs) < 20:
        return 0.0
    head = pairs[:100]
    if sum(_close(a, b) for a, b in head) < 0.8 * len(head):
        return 0.0
    return sum(_close(a, b) for a, b in pairs) / len(pairs)


def _rows(warehouse: Warehouse, inventory: Inventory, table: str, columns: list[str],
          links: list[str]) -> tuple[dict[str, list[float | None]], dict[str, list[object]]]:
    """The sample: each measure column as numbers, each link column as it is stored."""
    t = inventory.tables[table]
    d = warehouse.dialect
    name = D.table_sql(t.database, t.schema, t.name, d)
    select = ", ".join(D.quote(c, d) for c in [*columns, *links])
    result = warehouse.query(f"SELECT {select} FROM {D.aliased(D.first_rows(name, d, rows=SAMPLE), 's', d)}")
    out: dict[str, list[float | None]] = {c: [] for c in columns}
    groups: dict[str, list[object]] = {c: [] for c in links}
    for row in result.rows:
        for c, v in zip(columns, row):
            try:
                out[c].append(None if v is None else float(v))
            except (TypeError, ValueError):
                out[c].append(None)
        for c, v in zip(links, row[len(columns):]):
            groups[c].append(v)
    return out, groups


def _filled(*series: list[float | None]) -> list[tuple[float, ...]]:
    return [vals for vals in zip(*series) if all(v is not None and math.isfinite(v) for v in vals)]  # type: ignore[misc]


def _varies(values: list[float | None]) -> bool:
    seen = {v for v in values if v is not None}
    return len(seen) > 1


def _whole(values: list[float | None]) -> bool:
    seen = [v for v in values if v is not None]
    return bool(seen) and all(float(v).is_integer() for v in seen)


def _correlation(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 20:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if not sx or not sy:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _per_unit(m: MeasureFinding, amount: str, other: str, share: float) -> None:
    m.agg, m.additivity, m.time_aggregation = "avg", "non_additive", None
    if m.format not in ("currency", "percent"):
        m.format = "currency" if names.tokens(amount) and set(names.tokens(amount)) & _MONEY else "number"
    m.evidence.append(Evidence(kind="per_unit", weight=1, detail=(
        f"{names.readable(amount)} is {m.name.lower()} times {names.readable(other).lower()} on {share:.0%} of rows: "
        f"{m.name.lower()} is a price or rate per unit, averaged, never summed")))


def check_arithmetic(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                     measures: list[MeasureFinding],
                     links: dict[str, list[str]] | None = None, *,
                     derived: list[MeasureFinding] | None = None) -> list[tuple[str, str, str, float]]:
    """Corrects how each column measure adds up from its rows; returns the pairs of measures that hold the same
    values, as (table, column, other column, share of rows). ``links`` names, per table, the columns that point
    at what a row is about (a product, an item): a price stays nearly the same for one of them. A value no
    column holds (a quantity times its unit cost) goes into ``derived``."""
    same: list[tuple[str, str, str, float]] = []
    by_table: dict[str, list[MeasureFinding]] = {}
    for m in measures:
        if m.column is not None and m.agg in ("sum", "avg"):
            by_table.setdefault(m.table, []).append(m)
    for table, found in by_table.items():
        if len(found) < 2 and not any(m.agg == "sum" for m in found):
            continue
        columns = [m.column for m in found if m.column]
        about = [c for c in (links or {}).get(table, []) if c not in columns][:3]
        sample = attempt(warehouse, f"the arithmetic check on {inventory.tables[table].name}",
                         lambda: _rows(warehouse, inventory, table, columns, about), None)  # noqa: B023
        if not sample:
            continue
        same += _check_table(table, found, *sample, derived=derived)
    return same


# A price or cost per unit by name ("Item cost", "Unit price"), as against an amount ("Line cost", "Total price").
_PER_UNIT_WORDS = {"cost", "cst", "price", "prc"}
_AMOUNT_WORDS = {"amount", "amt", "total", "ttl", "tot", "ext", "extended", "value", "val", "line", "net", "gross", "grs",
                 "sales", "revenue", "rev", "spend", "paid", "billed", "charge"}
_COUNTED = {"qty", "quantity", "units", "pieces", "pcs"}
_SOLD = {"order", "orders", "invoice", "invoices", "sale", "sales", "line", "lines", "detail", "details", "item",
         "items", "receipt", "receipts", "ticket", "tickets", "booking", "bookings"}
_ON_HAND = {"on", "hand", "hnd", "onhand", "stock", "stk", "balance", "bal", "available", "avl"}
STEADY = 0.05       # a unit cost moves less than 5% among one item's rows...
MOVING = 0.2        # ...while its quantities move by 20% or more


def _value_of_units(table: str, of: dict[str, MeasureFinding], rows: dict[str, list[float | None]],
                    groups: dict[str, list[object]], amounts: set[str]) -> MeasureFinding | None:
    """A row with a quantity and a price or cost per unit, and no amount: the value is the two multiplied on each
    row, then added up (stock value at cost, an order line's value at its unit price). The price is averaged,
    never summed. It is per unit by its name (a unit price, or a cost beside its own average), by being there
    where nothing is on hand without rising with the quantity, or by staying the same for one item while the
    item's quantities vary. Proposed only where no column holds an amount to add up, and once per table."""
    if amounts or any(m.agg == "sum" and m.additivity == "additive" and m.format == "currency" for m in of.values()):
        return None     # an amount is on the row already (a line amount, quantity x price): it is the value
    # A quantity that never varies (every line sells one track) still counts: the value is then the prices added
    # up, which only a price named per unit ("Unit price") is taken for.
    quantities = [c for c, m in of.items() if m.agg == "sum" and set(names.tokens(c)) & _COUNTED
                  and not set(names.tokens(c)) & {"avg", "average", "per"}
                  and (_varies(rows[c]) or any(v for v in rows[c] if v is not None))]
    if not quantities:
        return None
    level = [c for c in quantities if of[c].additivity == "semi_additive" and set(names.tokens(c)) & _ON_HAND]
    quantity = min(level or quantities, key=lambda c: (not _varies(rows[c]), len(names.tokens(c)), list(of).index(c)))
    steady = not _varies(rows[quantity])

    def per_unit(c: str) -> str | None:
        words = set(names.tokens(c))
        m = of[c]
        if c == quantity or m.format != "currency" or words & _AMOUNT_WORDS:
            return None
        if m.agg == "avg" and words & (_UNIT_PRICE | _RATE) and words & _MONEY:
            return "its name says it is per unit"
        if not words & _PER_UNIT_WORDS or steady:
            return None
        if any(set(names.tokens(o)) & {"avg", "average"} and set(names.tokens(o)) - {"avg", "average"} == words
               and of[o].agg == "avg" for o in of if o != c):
            return "its average beside it is a cost per unit, so it is one too"
        if not _varies(rows[c]):
            return None
        # An amount of the quantity is nought where the quantity is, and rises with it; a unit cost is there
        # whatever is on hand (an item out of stock still has a cost).
        pairs = _filled(rows[c], rows[quantity])
        empty = [value for value, units in pairs if units == 0]
        if len(empty) >= 20 and sum(1 for v in empty if v) >= 0.5 * len(empty) \
                and abs(_correlation([p[0] for p in pairs], [p[1] for p in pairs])) < 0.3:
            return "it is there where nothing is on hand, and does not rise with the quantity"
        for keys in groups.values():
            price, units = _spread_within(rows[c], keys), _spread_within(rows[quantity], keys)
            if price is not None and units is not None and price <= STEADY and units >= MOVING:
                return "it stays the same for one item while the item's quantities vary"
        return None

    priced = [(c, why) for c in of if (why := per_unit(c))]
    if not priced:
        return None
    # A value from a price that is empty where the quantity is not would be empty too: only a price filled on
    # most of the quantity's rows makes one. The cost itself comes before an average or a list price of it.
    counted = sum(1 for v in rows[quantity] if v is not None)
    usable = [(c, why) for c, why in priced if counted and len(_filled(rows[c], rows[quantity])) >= 0.5 * counted]
    for c, said in priced:
        m = of[c]
        if m.agg != "avg":
            m.agg, m.additivity, m.time_aggregation, m.review = "avg", "non_additive", None, None
            m.evidence.append(Evidence(kind="per_unit", weight=1, detail=(
                f"{m.name} is a price per unit, averaged, never summed: {said}")))
    if not usable:
        return None
    price, why = min(usable, key=lambda cw: (bool(set(names.tokens(cw[0])) & {"avg", "average", "std", "standard",
                                                                               "list", "lst"}),
                                             list(of).index(cw[0])))
    p, q = of[price], of[quantity]
    stem = " ".join(w for w in q.name.split() if w.casefold() not in {"quantity", "qty", "units", "pieces"})
    cost = bool(set(names.tokens(price)) & {"cost", "cst"})
    # At a selling price, on a line of what was sold, it is what was sold: the sales amount.
    sold = not cost and _sold(table)
    name = f"{stem} value at cost" if cost else "Sales amount" if sold else f"{stem} value at {p.name.lower()}"
    name = name.strip()
    name = name[:1].upper() + name[1:]
    found = MeasureFinding(table=table, column=quantity, agg="sum", additivity=q.additivity, format="currency",
                           name=name, time_aggregation=q.time_aggregation, times=price)
    found.evidence.append(Evidence(kind="value_of_units", weight=1, detail=(
        f"{q.name} times {p.name.lower()} on each row, added up: no column holds this value, and "
        f"{p.name.lower()} is per unit ({why})")))
    return found


def _sold(table: str) -> bool:
    """A line of what was sold, its words glued or not (LINEITEM, SALESORDERDETAIL, sales_lines)."""
    return any(w in _SOLD or any(len(s) >= 4 and (w.startswith(s) or w.endswith(s)) for s in _SOLD)
               for w in _parts(table.rsplit(".", 1)[-1]))


_GIVEN = {"discount", "disc", "dsc", "rebate", "rbt", "markdown", "allowance"}
_TAX = {"tax", "vat", "gst", "hst"}
_GROSS = {"extended", "ext", "extendedprice", "gross", "grs", "amount", "amt", "price", "prc", "sales", "sale",
          "total", "tot", "ttl", "value", "val", "line"}


def _after_rates(table: str, of: dict[str, MeasureFinding], rows: dict[str, list[float | None]],
                 value: MeasureFinding | None = None) -> list[MeasureFinding]:
    """A line with an amount and the rate of it given away (a discount between 0 and 1), and no column holding
    what was left: what was sold is the amount less that rate of it, on each row, added up (net sales:
    extended price x (1 - discount)); the discount given is the amount times the rate. A tax rate on the row
    is a tax on what was charged. Proposed once per table, from the plainest amount."""
    def rate(words: set[str]) -> str | None:
        # Named as one and between 0 and 1 on every row, whatever its format (a "unit price discount" of 0.02).
        return next((c for c, m in of.items() if m.agg == "avg" and (set(_parts(c)) | set(names.tokens(m.name))) & words
                     and (values := [v for v in rows[c] if v is not None]) and len(values) >= 20
                     and all(0.0 <= v <= 1.0 for v in values) and max(values) > 0), None)

    given, tax = rate(_GIVEN), rate(_TAX)
    if given is None and tax is None:
        return []
    amounts = [c for c, m in of.items() if m.agg == "sum" and m.additivity == "additive" and m.format == "currency"
               and _varies(rows[c]) and set(_parts(c)) & _GROSS
               and not set(_parts(c)) & (_TAKEN | _RESULT | _GIVEN | _TAX | {"unit", "per", "avg", "average", "list"})]
    if not amounts:
        # No amount on the row, only a quantity and its unit price (worked out as their product): what was sold is
        # the two less the discount of them, when the line is a sale's.
        if value is None or not value.times or given is None or not _sold(table) \
                or set(_parts(value.times)) & {"cost", "cst"}:
            return []
        q, p, d = value.column, value.times, given
        if sum(1 for row in _filled(rows[q], rows[p], rows[d]) if row[2]) < 20 or any(
                _share([(v, x * y * (1 - z)) for v, x, y, z in _filled(rows[c], rows[q], rows[p], rows[d]) if z])
                >= AGREE for c, m in of.items() if c not in (q, p, d) and m.agg == "sum"):
            return []
        net = MeasureFinding(table=table, column=q, agg="sum", additivity=value.additivity, format="currency",
                             name="Net sales", times=p, after=d, time_aggregation=value.time_aggregation)
        net.evidence.append(Evidence(kind="after_rate", weight=1, detail=(
            f"{of[q].name} times {of[p].name.lower()}, less its {of[d].name.lower()}, on each row, added up: "
            f"{of[d].name.lower()} is a rate between 0 and 1, and no column holds what was sold")))
        return [net]
    amount = min(amounts, key=lambda c: (not set(_parts(c)) & {"extended", "ext", "extendedprice", "gross", "grs"},
                                         len(_parts(c)), list(of).index(c)))
    a = of[amount]
    out: list[MeasureFinding] = []
    sold = _sold(table) or bool(set(_parts(amount)) & {"sales", "sale"})
    if given is not None:
        # Read where something was given away: elsewhere the amount and what is left are one.
        if sum(1 for row in _filled(rows[amount], rows[given]) if row[1]) < 20:
            return []
        # What is left is on the row already (a net amount beside the gross), or is the amount itself (a line total
        # of quantity x unit price x (1 - discount)): nothing to work out.
        held = any(_share([(v, g * (1 - d)) for v, g, d in _filled(rows[c], rows[amount], rows[given]) if d]) >= AGREE
                   for c, m in of.items() if c not in (amount, given) and m.agg == "sum")
        net_already = any(_share([(v, x * y * (1 - d)) for v, x, y, d in _filled(rows[amount], rows[x], rows[y],
                                                                                  rows[given]) if d]) >= AGREE
                          for x, y in combinations([c for c in of if c not in (amount, given)], 2))
        if held or net_already:
            return []
        d = of[given]
        net = MeasureFinding(table=table, column=amount, agg="sum", additivity="additive", format="currency",
                             name="Net sales" if sold else f"{a.name} after {d.name.lower()}", after=given)
        net.evidence.append(Evidence(kind="after_rate", weight=1, detail=(
            f"{a.name} less its {d.name.lower()} on each row ({a.name.lower()} x (1 - {d.name.lower()})), added "
            f"up: {d.name.lower()} is a rate between 0 and 1, and no column holds what is left")))
        off = MeasureFinding(table=table, column=amount, agg="sum", additivity="additive", format="currency",
                             name=f"{d.name} amount", times=given)
        off.evidence.append(Evidence(kind="value_of_rate", weight=1, detail=(
            f"{a.name} times {d.name.lower()} on each row, added up: what the {d.name.lower()} gave away")))
        out += [net, off]
    if tax is not None:
        t = of[tax]
        charged = f"{a.name.lower()} less {of[given].name.lower()}" if given is not None else a.name.lower()
        levied = MeasureFinding(table=table, column=amount, agg="sum", additivity="additive", format="currency",
                                name=f"{t.name} amount", after=given, times=tax)
        levied.evidence.append(Evidence(kind="value_of_rate", weight=1, detail=(
            f"{t.name} on what was charged ({charged}) on each row, added up: {t.name.lower()} is a rate between "
            "0 and 1")))
        out.append(levied)
    return out


def _check_table(table: str, found: list[MeasureFinding], rows: dict[str, list[float | None]],
                 groups: dict[str, list[object]] | None = None, *,
                 derived: list[MeasureFinding] | None = None) -> list[tuple[str, str, str, float]]:
    of = {m.column: m for m in found if m.column}
    judged: set[str] = set()

    # A product: AMOUNT = QUANTITY x PRICE. The price is the factor that is not a whole count, or the one
    # whose name says it is per unit.
    varying = [c for c in of if _varies(rows[c])]
    for amount, (x, y) in ((a, pair) for a in varying for pair in combinations([c for c in varying if c != a], 2)):
        if amount in judged:
            continue
        share = _share([(a, b * c) for a, b, c in _filled(rows[amount], rows[x], rows[y]) if a])
        if share < AGREE:
            continue
        rate = _rate_of(x, y, rows, of, groups or {})
        if rate is None:
            judged |= {amount, x, y}   # one is a price and one a quantity, which unknown: neither is a ratio
            continue
        if rate in judged:
            continue
        quantity = y if rate == x else x
        _per_unit(of[rate], amount, quantity, share)
        judged |= {rate, amount, quantity}
        if of[amount].agg != "sum":     # a "price charged" that is the line's amount: summed
            _summed(of[amount], f"it is {names.readable(rate).lower()} times {names.readable(quantity).lower()}")

    # A ratio: X = A / B, or (A - B) / A, as a fraction or a percentage. Only a column that stays within
    # what a percentage can be is tried, against at most a dozen others.
    others = [c for c in varying][:12]
    for x in of:
        values = [v for v in rows[x] if v is not None]
        if x in judged or not values or _whole(values) or max(abs(v) for v in values) > 1000:
            continue
        for a, b in permutations([c for c in others if c != x], 2):
            # Read where X holds something: a quantity that is nought on most rows is "nought divided by
            # anything" there, which says nothing about what X is (an allocated quantity, mostly empty).
            filled = [row for row in _filled(rows[x], rows[a], rows[b]) if row[0]]
            if len(filled) < 20:
                continue
            for scale in (1.0, 100.0):
                for shape, value in (("/", lambda va, vb: va / vb if vb else None),
                                     ("margin", lambda va, vb: (va - vb) / va if va else None)):
                    pairs = [(vx, scale * v) for vx, va, vb in filled if (v := value(va, vb)) is not None]
                    share = _share(pairs)
                    if share >= AGREE:
                        m = of[x]
                        m.agg, m.additivity, m.format, m.time_aggregation = "avg", "non_additive", "percent", None
                        how = (f"{names.readable(a).lower()} divided by {names.readable(b).lower()}" if shape == "/"
                               else f"{names.readable(a).lower()} less {names.readable(b).lower()}, over "
                                    f"{names.readable(a).lower()}")
                        m.evidence.append(Evidence(kind="ratio", weight=1, detail=(
                            f"{m.name} is {how} on {share:.0%} of rows: a ratio, averaged, never summed")))
                        judged.add(x)
                        break
                if x in judged:
                    break
            if x in judged:
                break

    # A difference: A = B - C on nearly every row, C mostly not nought (an amount after discounts is the gross less
    # the discount; a margin is the amount less the cost). Kept as evidence: what A is called follows from what is
    # taken away, once the names are given (name_by_difference).
    money = [c for c in varying if of[c].agg == "sum" and of[c].additivity == "additive"]
    differences: dict[tuple[str, str, str], float] = {}
    for a in money:
        for b, c in permutations([x for x in money if x != a], 2):
            filled = _filled(rows[a], rows[b], rows[c])
            # Each of the three mostly not nought: a rejected quantity, nought on most rows, is no "line amount
            # less the same amount in another currency" because the two are equal where nothing is converted.
            if not filled or any(sum(1 for row in filled if row[i]) < 0.2 * len(filled) for i in range(3)):
                continue
            share = _share([(va, vb - vc) for va, vb, vc in filled])
            if share >= AGREE:
                differences[(a, b, c)] = share

    def result(a: str, c: str) -> bool:
        """Of A = B - C and C = B - A (the same rows, read both ways), A is the result: its name says so (net,
        margin, profit) and C's does not; else C's says it is taken away (a discount, a cost, a credit) and A's
        does not; else A is listed after C, as derived columns are."""
        for words, a_is_result in ((_RESULT, True), (_TAKEN, False)):
            said_a, said_c = bool(set(names.tokens(a)) & words), bool(set(names.tokens(c)) & words)
            if said_a != said_c:
                return said_a if a_is_result else said_c
        return money.index(a) > money.index(c)

    for (a, b, c), share in differences.items():
        if (c, b, a) in differences and not result(a, c):
            continue
        of[a].evidence.append(Evidence(kind="difference", weight=0, detail=(
            f"{names.readable(a)} is {names.readable(b).lower()} less {names.readable(c).lower()} on "
            f"{share:.0%} of rows"), data={"of": b, "less": c}))

    # A rate: between 0 and 1 on every row, and not a count.
    for column, m in of.items():
        values = [v for v in rows[column] if v is not None]
        if column in judged or m.agg != "sum" or m.additivity != "additive" or len(values) < 20 or _whole(values) \
                or set(names.tokens(column)) & _QUANTITY:
            continue
        if all(0.0 <= v <= 1.0 for v in values) and max(values) > 0:
            m.agg, m.additivity, m.format, m.time_aggregation = "avg", "non_additive", "percent", None
            m.evidence.append(Evidence(kind="fraction", weight=1, detail=(
                f"{m.name} lies between 0 and 1 on every row: a rate or a share, averaged, never summed")))
            judged.add(column)

    # A "price" or "rate" by its name that rises with the quantity on its row is what was charged for it.
    quantities = [c for c in of if c not in judged and (set(names.tokens(c)) & _QUANTITY or _whole(rows[c]))]
    for column, m in of.items():
        if column in judged or m.agg != "avg" or not set(names.tokens(column)) & (_RATE | _UNIT_PRICE):
            continue
        if set(names.tokens(column)) & {"pct", "percent", "percentage", "ratio", "share", "unit", "unt", "per", "avg",
                                        "average", "score", "rating"}:
            continue
        for q in quantities:
            if q == column:
                continue
            pairs = _filled(rows[column], rows[q])
            r = _correlation([p[0] for p in pairs], [p[1] for p in pairs])
            if r >= 0.6:
                _summed(m, f"it rises with {names.readable(q).lower()} on the same row (correlation {r:.2f})")
                judged.add(column)
                break

    if derived is not None:
        products = {c for c in judged if of[c].agg == "sum" and of[c].format == "currency"}
        value = _value_of_units(table, of, rows, groups or {}, products)
        if value is not None:
            derived.append(value)
        derived += _after_rates(table, of, rows, value)

    # Two measures that are the same figure (counted where either holds something: two empty columns agree).
    same = []
    for a, b in combinations(of, 2):
        pairs = [(x, y) for x, y in _filled(rows[a], rows[b]) if x or y]
        if len(pairs) < 20:
            continue
        share = sum(x == y for x, y in pairs) / len(pairs)
        # Gross and net agree wherever nothing was discounted: a third column that is their difference
        # says they are two figures, rarely apart, not one figure read twice.
        if share >= AGREE and not any(
                _share([(va - vb, vc) for va, vb, vc in _filled(rows[a], rows[b], rows[c])]) >= AGREE
                for c in of if c not in (a, b)):
            same.append((table, a, b, share))
    return same


def _rate_of(x: str, y: str, rows: dict[str, list[float | None]], of: dict[str, MeasureFinding],
             groups: dict[str, list[object]]) -> str | None:
    """Of two factors of an amount, the one that is a price per unit (the quantity is the other): by its name, by
    being the one that is not a whole count, or by staying nearly the same for one product while quantities vary."""
    named = [c for c in (x, y) if set(names.tokens(c)) & (_RATE | _UNIT_PRICE) and not set(names.tokens(c)) & _QUANTITY]
    if len(named) == 1:
        return named[0]
    whole = [c for c in (x, y) if _whole(rows[c])]
    if len(whole) == 1:
        return y if whole[0] == x else x
    for keys in groups.values():
        sx, sy = _spread_within(rows[x], keys), _spread_within(rows[y], keys)
        if sx is not None and sy is not None and min(sx, sy) < 0.5 * max(sx, sy):
            return x if sx < sy else y
    return None


def _spread_within(values: list[float | None], keys: list[object]) -> float | None:
    """How much a column varies among rows about the same thing (one product), as a share of its level."""
    by: dict[object, list[float]] = {}
    for v, k in zip(values, keys):
        if v is not None and k is not None:
            by.setdefault(k, []).append(v)
    spreads = []
    for vs in by.values():
        if len(vs) >= 3:
            mean = sum(vs) / len(vs)
            if mean:
                spreads.append(math.sqrt(sum((v - mean) ** 2 for v in vs) / len(vs)) / abs(mean))
    return sum(spreads) / len(spreads) if len(spreads) >= 5 else None


def _summed(m: MeasureFinding, why: str) -> None:
    m.agg, m.additivity, m.time_aggregation = "sum", "additive", None
    m.evidence.append(Evidence(kind="amount", weight=1, detail=f"{m.name} is an amount, summed: {why}"))


def find_counters(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                  primary_keys: dict[str, list[str]], links: dict[str, list[str]], dated_by: dict[str, str],
                  measures: list[MeasureFinding]) -> list[tuple[str, str]]:
    """Running counters: a device's octets or errors, read again and again, that only ever grow (until the
    device restarts and they begin again from nought). The last reading is the figure, never the sum of
    readings. ``dated_by`` holds each table's default timestamp, ``links`` the columns naming what a row is about.
    Returns the (table, column) counters found, each made a measure read at its highest."""
    found: list[tuple[str, str]] = []
    for table, when in dated_by.items():
        about = links.get(table) or []
        if not about:
            continue
        t = inventory.tables[table]
        taken = {when, *about, *primary_keys.get(table, [])}
        numeric = [c.name for c in t.columns if c.name not in taken and c.data_type in ("integer", "decimal", "float")
                   and not names.opaque(c.name) and not set(names.tokens(c.name)) & _CODE
                   or c.name not in taken and c.data_type == "integer" and names.opaque(c.name)]
        if not numeric:
            continue
        d = warehouse.dialect
        source = D.aliased(D.first_rows(D.table_sql(t.database, t.schema, t.name, d), d, rows=SAMPLE * 4), "s", d)
        part, order = ", ".join(D.quote(a, d) for a in about[:1]), D.quote(when, d)
        for column in numeric:
            c = D.quote(column, d)
            sql = (f"SELECT COUNT(1), SUM(CASE WHEN v >= p THEN 1 ELSE 0 END), SUM(CASE WHEN v > p THEN 1 ELSE 0 END) "
                   f"FROM (SELECT {c} AS v, LAG({c}) OVER (PARTITION BY {part} ORDER BY {order}) AS p FROM {source}) q "
                   f"WHERE v IS NOT NULL AND p IS NOT NULL")
            result = attempt(warehouse, f"the counter check on {t.name}.{column}",
                             lambda sql=sql: warehouse.query(sql).rows[0], None)
            if not result:
                continue
            n, up, rising = (int(x or 0) for x in result)
            # Never falling, rising often enough to be counting (errors rise less often than octets), over many
            # values: a code that stays put per device is not counting anything.
            if n >= 50 and up / n >= 0.97 and rising / n >= 0.1 and profiles[table].columns[column].distinct >= 20:
                found.append((table, column))
                label = names.readable(column)
                m = next((x for x in measures if x.table == table and x.column == column), None)
                if m is None:
                    m = MeasureFinding(table=table, column=column, agg="max", additivity="non_additive",
                                       format="integer", name=label)
                    measures.append(m)
                m.agg, m.additivity, m.time_aggregation = "max", "non_additive", None
                m.evidence.append(Evidence(kind="counter", weight=1, detail=(
                    f"{label} never falls between readings of the same {names.readable(about[0]).lower()} on "
                    f"{up / n:.0%} of them: a running counter, read at its highest, never added up")))
    return found


def find_parent_figures(warehouse: Warehouse, inventory: Inventory, profiles: dict[str, TableProfile],
                        documents: dict[str, tuple[str, str]], measures: list[MeasureFinding]) -> list[tuple[str, str]]:
    """A document's figure written again on each of its lines: an order's freight on every line of the order.
    It belongs to the document, so adding it up over the lines counts it once per line. ``documents`` holds,
    for each table with a document-and-line key, its document column and the date its rows are dated by.

    Only lines of one document dated the same day are read so: the fills of a prescription come months
    apart, and a refill that dispenses what the last one did is an amount of its own, added up.
    Returns the (table, column) figures found, each averaged."""
    found: list[tuple[str, str]] = []
    for table, (doc_column, date_column) in documents.items():
        summed = [m for m in measures if m.table == table and m.column and m.agg == "sum" and m.additivity == "additive"
                  and (profiles[table].columns[m.column].distinct or 0) > 1]
        if not summed:
            continue
        t = inventory.tables[table]
        d = warehouse.dialect
        doc, day = D.quote(doc_column, d), D.quote(date_column, d)
        source = D.aliased(D.first_rows(D.table_sql(t.database, t.schema, t.name, d), d, rows=SAMPLE * 4), "s", d)
        inner = ", ".join(f"COUNT(DISTINCT {D.quote(m.column or '', d)}) AS d{i}" for i, m in enumerate(summed))
        outer = ", ".join(f"SUM(CASE WHEN d{i} <= 1 THEN 1 ELSE 0 END)" for i in range(len(summed)))
        sql = (f"SELECT COUNT(1), SUM(CASE WHEN days <= 1 THEN 1 ELSE 0 END), {outer} FROM (SELECT {doc}, "
               f"COUNT(DISTINCT {day}) AS days, {inner} FROM {source} GROUP BY {doc} HAVING COUNT(1) > 1) g")
        result = attempt(warehouse, f"the per-document check on {t.name}", lambda sql=sql: warehouse.query(sql).rows[0],
                         None)
        if not result or int(result[0] or 0) < 20:
            continue
        groups, one_day = int(result[0]), int(result[1] or 0)
        if one_day < 0.95 * groups:
            continue      # rows of the same number on different days: events of their own, not a document's lines
        for m, same in zip(summed, result[2:]):
            if int(same or 0) < 0.98 * groups:
                continue
            found.append((table, m.column or ""))
            label, owner = m.name, names.readable(doc_column).lower()
            m.agg, m.additivity = "avg", "non_additive"
            m.evidence.append(Evidence(kind="per_document", weight=1,
                                       detail=f"{label} is the same on every line of the same {owner} "
                                              f"({int(same):,} of {groups:,}): a figure of the {owner}, averaged, "
                                              "never added up over its lines"))
            m.review = (f"{label} repeats on every line of the same {owner}. Is it a figure of the {owner} "
                        f"(averaged), or does each line hold its own (added up)?")
    return found


# What is taken away says what is left: an amount less its discounts is what was sold (revenue, net sales); an
# amount less its cost is the margin (gross profit). The words people use for each, given to the one measure that
# holds them, unless another measure is already called so.
# What is left once something is taken away: a net amount, a margin, a profit, a balance.
_RESULT = {"net", "nt", "margin", "mgn", "mrg", "mrgn", "profit", "prft", "gp", "balance", "bal", "remaining"}
_TAKEN_FOR_SALES = {"discount", "disc", "dsc", "rebate", "rbt", "allowance", "promo", "promotion", "markdown"}
_TAKEN_FOR_MARGIN = {"cost", "cst", "cogs"}
_TAKEN = _TAKEN_FOR_SALES | _TAKEN_FOR_MARGIN | {"credit", "cr", "crd", "cdt", "tax", "fee", "refund", "return"}
_SALES_WORDS = {"en": ["Revenue", "Net sales", "Sales"], "fr": ["Chiffre d'affaires", "Ventes nettes", "Ventes"]}
_MARGIN_WORDS = {"en": ["Gross profit", "Profit"], "fr": ["Marge brute", "Bénéfice brut"]}


def name_by_difference(model) -> list[str]:
    """Gives the measures Learn found to be one amount less another the words people use for them (revenue, gross
    profit), each word to one measure only; returns a note for each word given."""
    def column_of(measure):
        return getattr(measure.expr, "column", None) or ""

    by_column = {(m.table, column_of(m).rsplit(".", 1)[-1].casefold()): m for m in model.measures.values()
                 if column_of(m) and getattr(m.expr, "agg", "") == "sum"}

    def said(word: str) -> set[str]:
        plain = word.casefold()
        return {m.key for m in model.measures.values()
                if not m.hidden and (m.business_name.casefold() == plain or any(
                    w.casefold() == plain for ws in m.synonyms.values() for w in ws))}

    def give(measure, words: dict[str, list[str]], why: str) -> list[str]:
        given = []
        for lang, add in words.items():
            have = list(measure.synonyms.get(lang) or [])
            for word in add:
                if said(word) - {measure.key} or any(w.casefold() == word.casefold() for w in have):
                    continue
                have.append(word)
                if lang == "en":
                    given.append(word)
            if have:
                measure.synonyms[lang] = have
        if given and why and why not in (measure.description or ""):
            measure.description = f"{measure.description} {why}".strip()
        return [f"{measure.business_name} is also called {', '.join(given)}: {why}"] if given else []

    notes: list[str] = []
    found: list[tuple[int, str, object, object, object]] = []      # (rank, role, measure, of, less)
    for m in model.measures.values():
        if m.hidden or m.provenance == "admin":
            continue
        for e in m.evidence:
            if e.kind != "difference":
                continue
            of = by_column.get((m.table, str(e.data.get("of") or "").casefold()))
            less = by_column.get((m.table, str(e.data.get("less") or "").casefold()))
            if of is None or less is None:
                continue
            taken = set(names.tokens(str(e.data.get("less") or "")))
            if taken & _TAKEN_FOR_SALES:
                found.append((len(names.tokens(column_of(m).rsplit(".", 1)[-1])), "sales", m, of, less))
            elif taken & _TAKEN_FOR_MARGIN:
                found.append((len(names.tokens(column_of(m).rsplit(".", 1)[-1])), "margin", m, of, less))
    # The plainer column first (a line amount before the same amount in another currency).
    for _, role, m, of, less in sorted(found, key=lambda x: (x[1] != "sales", x[0], x[2].key)):
        if role == "sales":
            notes += give(m, _SALES_WORDS, f"It is {of.business_name.lower()} less {less.business_name.lower()}: "
                                           "what was sold, after discounts.")
        else:
            notes += give(m, _MARGIN_WORDS, f"It is {of.business_name.lower()} less {less.business_name.lower()}.")
            # What the cost is taken from is what was sold, when nothing else is called so.
            notes += give(of, _SALES_WORDS, "")
    return notes