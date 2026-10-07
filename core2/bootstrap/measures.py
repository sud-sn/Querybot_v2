"""Table kinds, measures and how each measure adds up.

Kinds come from structure: a calendar was recognised already; a table other
tables point at, that names its members, is a dimension; keys and no measures
make a bridge; measures plus foreign keys or dates make a fact, a periodic
snapshot when its rows repeat for every period.

A measure is a number that is not a key, a code, a flag, a date or a sequence.
How it adds up is decided per measure: prices, rates, percentages and scores are
averaged; on a periodic snapshot a level (a balance, stock on hand, headcount) is
taken at the end of each period while a flow on the same table (a cost, a target)
is summed. Names are the evidence for that distinction, so a snapshot measure
with a meaningless name is treated as a level and offered for review: taking a
balance at period end is never wrong the way adding twelve month-end balances is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core2.bootstrap import names
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.dates import DateCandidate
from core2.bootstrap.inventory import Inventory, InvColumn, InvTable
from core2.bootstrap.joins import JoinFinding
from core2.bootstrap.keys import TableKeys
from core2.bootstrap.profiler import TableProfile
from core2.model.schema import ColumnProfile, Evidence

_PERCENT = {"pct", "percent", "percentage", "ratio", "share"}
_RATE = {"rate", "price", "prc", "avg", "average", "per", "score", "rating", "satisfaction", "csat", "nps"}
_UNIT_PRICE = {"unit", "unt", "standard", "std", "list", "lst"}
_MONEY = {"amount", "amt", "value", "val", "cost", "cst", "price", "prc", "revenue", "rev", "sales", "sls", "spend",
          "salary", "sal", "mrr", "arr", "budget", "bdgt", "refund", "rfd", "tax", "discount", "dsc", "fee", "charge",
          "gross", "grs", "net", "debit", "credit", "dr", "cr", "paid", "income", "expense", "profit", "margin", "mrg",
          "mgn", "usd", "cad", "eur", "invoice", "billed", "total", "target"}
_QUANTITY = {"qty", "quantity", "units", "count", "cnt", "volume", "weight", "hours", "days", "seats", "fte",
             "headcount", "hc", "number", "items", "pieces"}
_LEVEL = names.LEVEL_WORDS
_FLOW = names.FLOW_WORDS
_CODE = {"id", "key", "code", "cd", "no", "nbr", "num", "seq", "sequence", "line", "ln", "lin", "type", "typ", "ind", "indicator",
         "status", "sts", "flag", "flg", "year", "yr", "month", "mth", "day", "week", "wk", "quarter", "qtr",
         "version", "level", "lvl", "rank", "priority", "grade", "zip", "postal", "phone"}
_UNIT_WORDS = {"uom", "unit", "units", "unt", "um", "measure"}
_UNIT_VALUES = {"ea", "each", "pc", "pcs", "piece", "kg", "g", "lb", "lbs", "oz", "ft", "m", "cm", "mm", "l", "ml",
                "box", "bx", "cs", "case", "pk", "pack", "pallet", "unk", "units", "unit", "dz", "doz", "gal", "ton"}
_CURRENCIES = {"usd", "cad", "eur", "gbp", "jpy", "aud", "chf", "cny", "inr", "mxn", "brl", "sek", "nok", "dkk",
               "nzd", "zar", "sgd", "hkd"}


@dataclass
class MeasureFinding:
    table: str
    column: str | None          # None: a count of rows
    agg: str                    # sum | count | count_distinct | avg
    additivity: str
    format: str
    name: str
    time_aggregation: str | None = None
    unit_column: tuple[str, str] | None = None     # (table key, column)
    unit_values: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    review: str | None = None   # a question for an admin when the call rests on little


def _words(column: str) -> set[str]:
    return set(names.tokens(column))


def classify_tables(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
                    calendars: dict[str, CalendarFinding], joins: list[JoinFinding],
                    dates: dict[str, list[DateCandidate]]) -> dict[str, tuple[str, list[Evidence]]]:
    live = [j for j in joins if j.trust != "rejected"]
    out: dict[str, tuple[str, list[Evidence]]] = {}
    for key, table in inventory.tables.items():
        if key in calendars:
            out[key] = ("calendar", [Evidence(kind="calendar", detail="one row per day with calendar columns", weight=1)])
            continue
        outgoing = [j for j in live if j.from_table == key and not j.to_calendar]
        incoming = [j for j in live if j.to_table == key]
        numbers = _measure_columns(inventory, profiles, keys, joins, dates, key)
        labels = [c for c in keys[key].unique_columns if c not in keys[key].primary_key
                  and _label_like(table.column(c), profiles[key].columns[c])]
        roles = dates.get(key, [])
        periodic = [c for c in roles if c.periodic]
        evidence: list[Evidence] = []
        if incoming and labels:
            kind = "dimension"
            evidence.append(Evidence(kind="referenced", weight=1, detail=(
                f"{len(incoming)} join(s) point at it and it names its members ({', '.join(labels[:2])})")))
        elif numbers and periodic:
            kind = "snapshot"
            evidence.append(Evidence(kind="periodic", weight=1, detail=next(
                e.detail for e in periodic[0].evidence if e.kind == "snapshot")))
        elif not numbers and len(outgoing) >= 2 and not _any_label(table, profiles[key], keys[key]) and (
                len(keys[key].primary_key) != 1 or any(j.from_column == keys[key].primary_key[0] for j in outgoing)):
            kind = "bridge"
            evidence.append(Evidence(kind="bridge", weight=1, detail=f"links {len(outgoing)} tables and holds no measures"))
        elif numbers and (outgoing or roles):
            kind = "fact"
            evidence.append(Evidence(kind="measures", weight=1, detail=f"{len(numbers)} measure(s) and "
                                     f"{len(outgoing)} link(s) to other tables"))
        elif incoming or labels or (not numbers and len(keys[key].primary_key) == 1
                                    and _any_label(table, profiles[key], keys[key])):
            kind = "dimension"
            evidence.append(Evidence(kind="names_members", weight=0.5,
                                     detail="other tables point at it" if incoming else "it lists named things"))
        else:
            kind = "other"
        out[key] = (kind, evidence)
    return out


def _any_label(table: InvTable, profile: TableProfile, keys: TableKeys) -> bool:
    """Any column that names things, unique or not."""
    return any(c.name not in keys.primary_key and _label_like(c, profile.columns[c.name]) for c in table.columns)


def _label_like(column: InvColumn | None, p: ColumnProfile) -> bool:
    """Text that names things (varying length, words), not a document number (fixed-width codes)."""
    if column is None or column.data_type != "text":
        return False
    spread = (p.max_len or 0) - (p.min_len or 0)
    return spread >= 4 or (p.avg_len or 0) >= 12


def _measure_columns(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
                     joins: list[JoinFinding], dates: dict[str, list[DateCandidate]], key: str) -> list[str]:
    table = inventory.tables[key]
    profile = profiles[key]
    excluded = set(keys[key].primary_key) | {c for alt in keys[key].alternate_keys for c in alt}
    excluded |= {j.from_column for j in joins if j.from_table == key and j.trust != "rejected"}
    excluded |= {c.column for c in dates.get(key, [])}
    out = []
    for column in table.columns:
        if column.name in excluded or column.data_type not in ("integer", "decimal", "float"):
            continue
        p = profile.columns[column.name]
        if p.pattern in ("flag01", "yyyy", "yyyymm", "yyyymmdd") or column.name in keys[key].unique_columns:
            continue
        words = _words(column.name)
        if not names.opaque(column.name) and words & _CODE and not words & (_MONEY | _QUANTITY | _PERCENT | _RATE):
            continue
        if column.data_type == "integer" and names.opaque(column.name) and p.distinct <= 10:
            continue   # a small set of whole numbers with no name reads as a code
        out.append(column.name)
    return out


def find_measures(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
                  joins: list[JoinFinding], dates: dict[str, list[DateCandidate]],
                  kinds: dict[str, tuple[str, list[Evidence]]]) -> list[MeasureFinding]:
    out: list[MeasureFinding] = []
    reachable = _reachable_one_step(joins)
    for key, (kind, _) in kinds.items():
        if kind not in ("fact", "snapshot"):
            continue
        table = inventory.tables[key]
        profile = profiles[key]
        periodic = kind == "snapshot"
        for name in _measure_columns(inventory, profiles, keys, joins, dates, key):
            p = profile.columns[name]
            words = _words(name)
            opaque = names.opaque(name)
            label = names.readable(name)
            m = MeasureFinding(table=key, column=name, agg="sum", additivity="additive", format="number", name=label)
            in_unit_range = p.min_num is not None and p.max_num is not None and p.min_num >= 0 and p.max_num <= 1
            in_percent_range = p.min_num is not None and p.max_num is not None and p.min_num >= -100 and p.max_num <= 100
            if words & _PERCENT or ("margin" in words or "mrg" in words) and in_percent_range and "amt" not in words \
                    and "amount" not in words:
                m.agg, m.additivity, m.format = "avg", "non_additive", "percent"
                m.evidence.append(Evidence(kind="percent", weight=1, detail=f"{label} is a percentage: averaged, never summed"))
            elif words & _RATE or (words & _UNIT_PRICE and words & _MONEY):
                m.agg, m.additivity = "avg", "non_additive"
                m.format = "currency" if words & _MONEY else "number"
                m.evidence.append(Evidence(kind="rate", weight=1, detail=f"{label} is a rate or a price per unit: averaged"))
            elif in_unit_range and words & {"rate", "ratio", "share"}:
                m.agg, m.additivity, m.format = "avg", "non_additive", "percent"
            else:
                if words & _MONEY:
                    m.format = "currency"
                elif words & _QUANTITY or column_is_whole(p):
                    m.format = "integer" if column_is_whole(p) else "number"
                if periodic:
                    level = (bool(words & _LEVEL) or opaque or not words & _FLOW) and not (words & _FLOW and not words & _LEVEL)
                    if level:
                        m.additivity, m.time_aggregation = "semi_additive", "last"
                        m.evidence.append(Evidence(kind="level", weight=1,
                                                   detail=f"{label} is a level on a periodic snapshot: taken at "
                                                          "the end of each period, added across everything else"))
                        if opaque or not words & _LEVEL:
                            m.review = (f"Is {label} a balance (taken at period end) or an amount for the period "
                                        "(added up over time)?")
                    else:
                        m.evidence.append(Evidence(kind="flow", weight=1,
                                                   detail=f"{label} is an amount for each period: added up over time"))
            if m.format in ("number", "integer") and words & _QUANTITY or m.format in ("number", "integer") and opaque:
                unit = _unit_column(inventory, profiles, key, reachable, kind="unit")
                if unit:
                    m.unit_column, m.unit_values = unit
            if m.format == "currency":
                unit = _unit_column(inventory, profiles, key, {}, kind="currency")
                if unit:
                    m.unit_column, m.unit_values = unit
            out.append(m)

        # Counting rows and documents.
        noun = names.readable(" ".join(names.core_table(table.name))) if not names.opaque(table.name) else table.name
        out.append(MeasureFinding(table=key, column=None, agg="count", additivity="additive", format="count",
                                  name=f"Number of {noun.lower()} rows" if periodic else f"Number of {noun.lower()}s",
                                  evidence=[Evidence(kind="row_count", weight=1, detail="counts rows")]))
        for column in table.columns:
            p = profile.columns[column.name]
            if column.data_type not in ("text", "integer") or column.name in keys[key].unique_columns:
                continue
            if any(j.from_table == key and j.from_column == column.name and j.trust != "rejected" for j in joins):
                continue
            if not profile.rows or not (profile.rows / 1000 < p.distinct < profile.rows):
                continue
            words = _words(column.name)
            if names.opaque(column.name):
                is_document = column.data_type == "text" and (p.max_len or 0) <= 24
            else:
                identifying = {"number", "no", "nbr", "num", "id", "ref"}
                is_document = bool(words & identifying) and not words & (_CODE - identifying)
            if not is_document or column.name in {c.column for c in dates.get(key, [])}:
                continue
            thing = [w for w in names.core_column(column.name)]
            while len(thing) > 1 and thing[-1] in ("number", "no", "nbr", "num", "id", "ref", "code"):
                thing.pop()
            label = names.readable("_".join(thing)) if thing and not names.opaque(column.name) else column.name
            out.append(MeasureFinding(table=key, column=column.name, agg="count_distinct", additivity="non_additive",
                                      format="count", name=f"Number of {label.lower()}s",
                                      evidence=[Evidence(kind="document_count", weight=1,
                                                         detail=f"{column.name} repeats across rows: counted once each")]))
    return out


def column_is_whole(p: ColumnProfile) -> bool:
    return p.integer_share == 1.0


def _reachable_one_step(joins: list[JoinFinding]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for j in joins:
        if j.trust in ("verified", "declared") and not j.to_calendar:
            out.setdefault(j.from_table, []).append(j.to_table)
    return out


def _unit_column(inventory: Inventory, profiles: dict[str, TableProfile], key: str,
                 reachable: dict[str, list[str]], *, kind: str) -> tuple[tuple[str, str], list[str]] | None:
    for table_key in [key] + sorted(reachable.get(key, [])):
        table = inventory.tables[table_key]
        for column in table.columns:
            p = profiles[table_key].columns[column.name]
            if column.data_type != "text" or not p.top or p.distinct > 20 or (p.max_len or 99) > 6:
                continue
            values = {(t.value or "").strip().lower() for t in p.top}
            words = _words(column.name)
            if kind == "unit" and (words & _UNIT_WORDS or values and values <= _UNIT_VALUES):
                return (table_key, column.name), sorted(t.value or "" for t in p.top)
            if kind == "currency" and values and values <= _CURRENCIES:
                return (table_key, column.name), sorted(t.value or "" for t in p.top)
    return None
