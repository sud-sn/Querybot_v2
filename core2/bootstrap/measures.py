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

from core2.bootstrap import names, personal
from core2.bootstrap.calendar import CalendarFinding
from core2.bootstrap.dates import AUDIT_WORDS, DateCandidate
from core2.bootstrap.inventory import Inventory, InvColumn, InvTable
from core2.bootstrap.joins import JoinFinding, bare
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
_OWN_UNITS = {"day", "days", "hour", "hours", "hrs", "hr", "minute", "minutes", "mins", "min", "second", "seconds",
              "secs", "week", "weeks", "month", "months", "year", "years", "oz", "lb", "lbs", "kg", "kgs", "gram",
              "grams", "mg", "ml", "km", "miles", "pct", "percent", "mbps", "gbps", "gb", "mb", "tb", "kwh", "sqft"}
# Nouns that name an amount, not a thing counted: a table of them is counted in rows ("revenue rows").
AMOUNT_NOUNS = {"revenue", "income", "cost", "costs", "spend", "expense", "expenses", "billing", "inventory",
                 "stock", "usage", "shipping", "payroll", "budget", "finance", "consumption", "demand", "supply"}
_FLOW = names.FLOW_WORDS
_COST_ONLY = {"cost", "cst"}
_CODE = {"id", "key", "code", "cd", "no", "nbr", "num", "seq", "sequence", "line", "ln", "lin", "type", "typ", "ind", "indicator",
         "status", "sts", "flag", "flg", "year", "yr", "month", "mth", "day", "week", "wk", "quarter", "qtr",
         "version", "level", "lvl", "rank", "priority", "grade", "zip", "postal", "phone", "ref", "revision",
         "frequency", "diagnosis", "dx", "icd", "snomed", "cpt", "ndc", "loinc", "reaction", "fips", "lat", "lon",
         "lng", "latitude", "longitude"}
_UNIT_WORDS = {"uom", "unit", "units", "unt", "um", "measure"}
# What a thing can do (a port's speed, a circuit's bandwidth), not an amount of anything: never added up.
_CAPACITY = {"speed", "bandwidth", "capacity", "mbps", "gbps", "kbps", "bps", "threshold", "limit", "lmt", "rpm",
             "ghz", "mhz"}
# What a row allows or is allowed, not what it did: "refills authorized", "days allowed".
_ALLOWANCE = {"authorized", "authorised", "authd", "authzd", "allowed", "allw", "alwd", "allwd", "permitted",
              "entitled", "allowance", "quota"}
_CURRENCY_CODES = {"usd", "cad", "eur"}
# A rate between currencies (FxRate, ExchRate, a currency table's AverageRate): a number of one per the other, never
# a share.
_FX = {"fx", "exch", "exchange", "xchg", "currency", "curr", "ccy", "forex"}
# Rates that are shares of what they count (a tax rate of 0.0825, a churn rate of 0.04); a rate per unit (a kWh rate
# of 0.21 dollars, an hourly rate) is a price.
_SHARE_RATES = {"tax", "vat", "discount", "disc", "dsc", "commission", "comm", "interest", "int", "apr", "conversion",
                "conv", "churn", "retention", "return", "defect", "error", "growth", "win", "fill", "attrition",
                "turnover", "occupancy", "utilization", "utilisation", "response", "open", "click", "bounce",
                "completion", "success", "failure", "default", "delinquency", "vacancy", "absence", "cancellation",
                "cancel", "approval", "rejection", "reject", "close", "hit", "yield", "renewal", "adoption",
                "penetration", "markup", "margin", "redemption", "participation", "scrap", "shrink", "loss"}
# A price for the whole line or document, not per unit: an amount, added up (EXTENDED_PRICE, TOTAL_PRICE, PRICE_CHARGED).
_PRICE_TOTAL = {"extended", "ext", "total", "tot", "charged", "billed", "invoiced"}
# What a test or an instrument read: a value of its own, averaged, never added up (RESULT_VALUE, READING).
_READING = {"result", "reading", "measurement", "measured", "observed", "observation"}
_UNIT_VALUES = {"ea", "each", "pc", "pcs", "piece", "kg", "g", "lb", "lbs", "oz", "ft", "m", "cm", "mm", "l", "ml",
                "box", "bx", "cs", "case", "pk", "pack", "pallet", "unk", "units", "unit", "dz", "doz", "gal", "ton"}
_CURRENCIES = {"usd", "cad", "eur", "gbp", "jpy", "aud", "chf", "cny", "inr", "mxn", "brl", "sek", "nok", "dkk",
               "nzd", "zar", "sgd", "hkd"}
# An amount in the company's own currency whatever currency its document is in: AmountMST, Amount (LCY), an amount in
# the functional or reporting currency. "Base" says so too, where the database keeps amounts both ways (ExtAmtBase
# beside ExtAmtDoc); alone it may be an amount a tax is worked out on (TaxBaseAmount), in the document's currency,
# and beside a document's word (TaxBaseAmtDoc) it is.
_ONE_CURRENCY = {"functional", "func", "reporting", "rptg", "mst", "lcy", "acy"}
_IN_BASE = {"base", "bse"}
_IN_DOCUMENT = {"doc", "document", "cur", "curr", "txn", "transaction", "trans", "tran", "foreign", "fcy", "orig",
                "original"}


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
    # A value no column holds, worked out on each row: this column times another of the same row (an on-hand
    # quantity times its unit cost), then added up.
    times: str | None = None
    # ...or this column less a rate of itself (a line's price after its discount: price x (1 - discount)), then,
    # with ``times`` too, times that (the tax on what was charged).
    after: str | None = None
    # A percentage kept as a share of one (0.05 for 5%): each value times this before it is averaged, in points.
    scale: float = 1.0


# Words a name written in one case glues after another: LINENUMBER, SHIPPRIORITY, EXTENDEDPRICE, ACCTBAL.
_GLUED_TAILS = ("priority", "quantity", "balance", "percent", "number", "amount", "status", "total", "value", "price",
                "count", "cost", "rate", "flag", "mode", "type", "qty", "amt", "bal", "pct")
_KNOWN = _MONEY | _QUANTITY | _CODE | _RATE | _PERCENT | names.LEVEL_WORDS | names.FLOW_WORDS \
    | {w for reading in names.EXPANSIONS.values() for w in reading.split()}


def _apart(word: str) -> list[str]:
    """``linenumber`` -> line, number; ``acctbal`` -> acct, bal; ``departmentid`` -> department, id. A word of its
    own stays whole: discount, headcount, account."""
    if word in _KNOWN:
        return [word]
    parts = names.unglued(word)
    if len(parts) == 2:
        return parts
    for tail in _GLUED_TAILS:
        head = word[:-len(tail)]
        if word.endswith(tail) and (len(head) >= 4 or len(head) >= 3 and head in _KNOWN):
            return [head, tail]
    return [word]


def _parts(column: str) -> list[str]:
    """The column's words in order, glued words read apart."""
    return [part for word in names.tokens(column) for part in _apart(word)]


def _words(column: str) -> set[str]:
    out: set[str] = set()
    for word in names.tokens(column):
        out.add(word)
        if word not in _MONEY | _QUANTITY:
            out |= set(_apart(word))       # DEPARTMENTID, REASONCODE, LINENUMBER
    return out


# The last word that makes a number name a row rather than measure it: a line number, a revision, an id.
_NAMING_LAST = {"id", "key", "code", "cd", "no", "nbr", "num", "number", "ref", "seq", "sequence"}


_NUMBER_WORDS = {"num", "nbr", "no", "number", "cnt", "count"}
# What was counted, as a noun: a number of "delivered" is a number of deliveries.
_EVENT_NOUNS = {"delivered": "delivery", "received": "receipt", "returned": "return", "shipped": "shipment",
                "invoiced": "invoice", "ordered": "order", "sold": "sale", "issued": "issue", "transferred": "transfer",
                "adjusted": "adjustment", "rejected": "rejection", "cancelled": "cancellation"}


def _counted(column: str) -> list[str]:
    """What a "number of" column counts (NUM_OF_RCT: receipts), or nothing: a count to add up, not an identifier."""
    parts = names.tokens(column)
    for i, word in enumerate(parts[:-2]):
        if word in _NUMBER_WORDS and parts[i + 1] == "of":
            return parts[i + 2:]
    return []


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
        # A snapshot is dated by its period: the table's own (default) date must be
        # the periodic one. A secondary date that happens to repeat per customer and
        # day does not make invoices into balances, and nothing numbered as a
        # document or a line (invoice no., line no.) is a balance.
        periodic = [c for c in roles if c.periodic and c.is_default]
        # The rest of a balance's own key beside its period (a lot number) says what is counted, not a document.
        grain = set(keys[key].primary_key) if any(c.column in keys[key].primary_key for c in periodic) else set()
        documents = _document_numbers(table, {c for j in outgoing for c in j.from_columns} | grain)
        evidence: list[Evidence] = []
        if periodic and documents:
            evidence.append(Evidence(kind="documents", weight=1, detail=(
                f"{', '.join(documents[:2])} number documents or their lines: rows are transactions, not balances")))
            periodic = []
        named = _names_members(table, profiles[key], keys[key])
        # Numbered and dated, and pointed at by its own lines: a document (an order's header), whatever else it links.
        document = bool(documents) and any(c.kind == "event" and c.is_default for c in roles) and any(
            _lines_of(keys[j.from_table], j) for j in incoming)
        if incoming and (labels or named):
            # Pointed at, and naming its members: a dimension, though it carries figures and dates of its own (a
            # product's list price, an employee's hire date) and names repeat once per version of a member.
            kind = "dimension"
            evidence.append(Evidence(kind="referenced", weight=1, detail=(
                f"{len(incoming)} join(s) point at it and it names its members ({', '.join((labels or named)[:2])})")))
        elif numbers and periodic:
            kind = "snapshot"
            evidence.append(Evidence(kind="periodic", weight=1, detail=next(
                e.detail for e in periodic[0].evidence if e.kind == "snapshot")))
        elif not numbers and len(outgoing) >= 2 and not _any_label(table, profiles[key], keys[key]) and not document and (
                len(keys[key].primary_key) != 1 or any(j.from_column == keys[key].primary_key[0] for j in outgoing)):
            kind = "bridge"
            evidence.append(Evidence(kind="bridge", weight=1, detail=f"links {len(outgoing)} tables and holds no measures"))
        elif not numbers and not incoming and outgoing and any(not c.periodic for c in roles):
            # A diagnosis, a survey answer: an event of someone, on a date, with nothing to add up but itself.
            kind = "fact"
            evidence.append(Evidence(kind="events", weight=1, detail=(
                f"one dated event per row, linked to {len(outgoing)} table(s): counted, not summed")))
        elif numbers and (outgoing or roles):
            kind = "fact"
            evidence.append(Evidence(kind="measures", weight=1, detail=f"{len(numbers)} measure(s) and "
                                     f"{len(outgoing)} link(s) to other tables"))
        elif not numbers and any(c.kind == "event" and c.is_default for c in roles) and any(
                _lines_of(keys[j.from_table], j) for j in incoming):
            # An invoice its lines point at by their key (invoice and line number), or by its number beside each
            # line's own (a line's row number, its order and its line number on the order), dated by when it was
            # made, and carrying no figure itself: a document, counted, whose date its lines are dated by.
            kind = "fact"
            evidence.append(Evidence(kind="documents", weight=1, detail=(
                "its lines point at it by their own key, and it is dated: documents, counted")))
        elif incoming or labels or (not numbers and len(keys[key].primary_key) == 1
                                    and _any_label(table, profiles[key], keys[key])):
            kind = "dimension"
            evidence.append(Evidence(kind="names_members", weight=0.5,
                                     detail="other tables point at it" if incoming else "it lists named things"))
        else:
            kind = "other"
        out[key] = (kind, evidence)
    return out


def _lines_of(tkeys: TableKeys, join: JoinFinding) -> bool:
    """Do the rows pointing by ``join`` number their lines within what they point at: by part of their key (an
    invoice and its line number), or beside a line number keyed with it (SalesOrderID and LineNo, under a row number)."""
    columns = set(join.from_columns)
    return bool(tkeys.line_numbers) and (columns <= set(tkeys.primary_key) or any(
        columns < set(alt) and set(alt) & set(tkeys.line_numbers) for alt in tkeys.alternate_keys))


_NUMBERED = {"no", "num", "nbr", "number"}
# What numbers a row within its document, never the document: a line, a sequence, a revision.
_WITHIN = {"line", "ln", "lin", "seq", "sequence", "item", "itm", "position", "pos", "revision", "rev", "version",
           "renewal", "fill", "step"}


def _numbers_documents(column: str) -> bool:
    """Named as the number of a document (a sale's ticket number on the key of item and ticket, TICKET_NO), never as
    a line's or a revision's within one."""
    words = names.tokens(column)
    return bool(words) and words[-1] in _NUMBERED and not set(words) & _WITHIN


def _document_numbers(table: InvTable, links: set[str]) -> list[str]:
    """Columns that number documents or their lines (``IVC_NO``, ``IVC_LIN_NO``), links aside."""
    out = []
    for column in table.columns:
        words = names.tokens(column.name)
        if column.name in links or column.data_type not in ("text", "integer") or not words:
            continue
        if words[-1] in _NUMBERED and not set(words) & AUDIT_WORDS:
            out.append(column.name)
    return out


# Words a member's own name may carry besides the table's noun: EnglishProductName, FULL_NAME.
_OWN_NAME_EXTRA = {"name", "nm", "full", "display", "english", "en", "eng", "short", "long"}


def _names_members(table: InvTable, profile: TableProfile, keys: TableKeys) -> list[str]:
    """Columns that name the table's own members, unique or not: NAME, ResellerName or EnglishProductName filled
    on nearly every row (never a customer's name kept on an order), or a person's first and last name."""
    rows = profile.rows or 0
    subject = [] if names.opaque(table.name) else names.core_table(table.name)

    def own(column: str) -> bool:
        words = set(names.tokens(column))
        rest = words - _OWN_NAME_EXTRA
        return bool(words & {"name", "nm"}) and all(any(names.same_word(w, t) for t in subject) for w in rest)

    out = [c.name for c in table.columns
           if c.data_type == "text" and c.name not in keys.primary_key and own(c.name)
           and profile.columns[c.name].distinct >= 2 and profile.columns[c.name].non_null >= 0.9 * rows]
    texts = [c.name for c in table.columns if c.data_type == "text"]
    first = [c for c in texts if personal.is_first_name(c)]
    last = [c for c in texts if personal.is_last_name(c)]
    return out or (first[:1] + last[:1] if first and last else [])


def _any_label(table: InvTable, profile: TableProfile, keys: TableKeys) -> bool:
    """Any column that names things, unique or not."""
    return any(c.name not in keys.primary_key and _label_like(c, profile.columns[c.name]) for c in table.columns)


def _label_like(column: InvColumn | None, p: ColumnProfile) -> bool:
    """Text that names things (varying length, words), not a document number (fixed-width codes)."""
    if column is None or column.data_type != "text":
        return False
    spread = (p.max_len or 0) - (p.min_len or 0)
    # Long and all of one width (B241214-0000) is a code, however long; names vary in length.
    return spread >= 4 or (p.avg_len or 0) >= 12 and spread > 0


def _measure_columns(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
                     joins: list[JoinFinding], dates: dict[str, list[DateCandidate]], key: str) -> list[str]:
    table = inventory.tables[key]
    profile = profiles[key]
    # A second key that holds a quantity or an amount (an order and its line's quantity, unique by chance where
    # an order's lines never repeat one) does not make that column a key: the table has its own.
    excluded = set(keys[key].primary_key) | {c for alt in keys[key].alternate_keys for c in alt
                                             if not (keys[key].primary_key and _words(c) & (_MONEY | _QUANTITY))}
    excluded |= {c for j in joins if j.from_table == key and j.trust != "rejected" for c in j.from_columns}
    excluded |= {c.column for c in dates.get(key, [])} | {c.quarter_column for c in dates.get(key, []) if c.quarter_column}
    out = []
    for column in table.columns:
        if column.name in excluded or column.data_type not in ("integer", "decimal", "float"):
            continue
        p = profile.columns[column.name]
        if p.pattern in ("yyyy", "yyyymm", "yyyymmdd") or column.name in keys[key].unique_columns:
            continue
        if names.deletion_flag(column.name) and (p.pattern == "flag01" or p.distinct <= 2):
            continue   # IS_DELETED holding 1 or nothing: which rows were deleted, never an amount (Removed, a count)
        if p.pattern == "flag01" and not _counted(column.name):
            continue   # 0 and 1 is a flag, unless the name says it counts (NUM_OF_PHY_INV: none or one this month)
        words = _words(column.name)
        if not names.opaque(column.name) and words & _CODE and not words & (_MONEY | _QUANTITY | _PERCENT | _RATE) \
                and not _counted(column.name):
            continue
        last = [w for w in _parts(column.name) if not w.isdigit()][-1:] or [""]
        if last[0] in _NAMING_LAST and not _counted(column.name) and not (_parts(column.name) or [""])[0] in (
                "number", "num", "nbr", "no", "count", "cnt", "total"):
            continue   # SalesOrderLineNumber, RevisionNumber: a row's number, never added up
        if last[0] == "by" and not names.opaque(column.name):
            continue   # COMPOUNDED_BY, VERIFIED_BY: who did it, never an amount
        if words & (_CAPACITY | _ALLOWANCE) and not words & (_MONEY | _QUANTITY):
            continue   # a port's speed, the refills a prescription allows: what may be, not an amount of it
        if "per" in words and not dates.get(key):
            continue   # a recipe's grams per capsule: a ratio of the link, with no date to add it up over
        if column.data_type == "integer" and names.opaque(column.name) and p.distinct <= 10:
            continue   # a small set of whole numbers with no name reads as a code
        out.append(column.name)
    return out


def find_measures(inventory: Inventory, profiles: dict[str, TableProfile], keys: dict[str, TableKeys],
                  joins: list[JoinFinding], dates: dict[str, list[DateCandidate]],
                  kinds: dict[str, tuple[str, list[Evidence]]]) -> list[MeasureFinding]:
    out: list[MeasureFinding] = []
    reachable = _reachable_one_step(joins)
    headers = _documents_of_lines(joins, keys)
    kept_both_ways = _kept_both_ways(inventory)
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
            label = names.readable(bare(table, name))     # SS_EXT_SALES_PRICE: "Ext sales price", not "Ss ext..."
            if _counted(name):
                events = names.read_tokens(_counted(name)).lower().split()
                events[-1] = _EVENT_NOUNS.get(events[-1], events[-1])     # NUM_OF_DLV: deliveries
                label = f"Number of {names.plural(' '.join(events))}"
            m = MeasureFinding(table=key, column=name, agg="sum", additivity="additive", format="number", name=label)
            in_unit_range = p.min_num is not None and p.max_num is not None and p.min_num >= 0 and p.max_num <= 1
            # Shares of one, a margin's below nothing too (-0.2 to 0.6 is -20% to 60%).
            in_share_range = p.min_num is not None and p.max_num is not None and p.min_num >= -1 and p.max_num <= 1
            in_percent_range = p.min_num is not None and p.max_num is not None and p.min_num >= -100 and p.max_num <= 100
            # EXTENDED_PRICE, TOTAL_PRICE, PRICE_CHARGED: a line's or a document's price, an amount.
            rate = words & _RATE - ({"price", "prc"} if words & _PRICE_TOTAL else set())
            if words & _PERCENT or ("margin" in words or "mrg" in words) and in_percent_range and "amt" not in words \
                    and "amount" not in words:
                m.agg, m.additivity, m.format = "avg", "non_additive", "percent"
                m.evidence.append(Evidence(kind="percent", weight=1, detail=f"{label} is a percentage: averaged, never summed"))
                if in_share_range and not column_is_whole(p):
                    m.scale = 100.0
                    m.evidence.append(Evidence(kind="share_of_one", weight=1, detail=(
                        f"{label} holds shares of one ({p.min_num:g} to {p.max_num:g}): shown in points, 0.05 as 5%")))
            elif words & _READING and not words & (_QUANTITY | _MONEY - {"value", "val"}):
                m.agg, m.additivity = "avg", "non_additive"
                m.evidence.append(Evidence(kind="reading", weight=1,
                                           detail=f"{label} is what a test or an instrument read: averaged, never summed"))
            elif "rate" in words and words & _SHARE_RATES and in_share_range and not column_is_whole(p) \
                    and not words & _FX and not set(names.tokens(table.name)) & _FX:
                # A conversion rate of 0.031, a churn rate of 0.04: shares of one, shown as 3.1% and 4%.
                m.agg, m.additivity, m.format, m.scale = "avg", "non_additive", "percent", 100.0
                m.evidence.append(Evidence(kind="share_of_one", weight=1, detail=(
                    f"{label} holds shares of one ({p.min_num:g} to {p.max_num:g}): a rate shown in points")))
            elif rate or (words & _UNIT_PRICE and words & _MONEY and not words & _PRICE_TOTAL):
                m.agg, m.additivity = "avg", "non_additive"
                m.format = "currency" if words & _MONEY else "number"
                m.evidence.append(Evidence(kind="rate", weight=1, detail=f"{label} is a rate or a price per unit: averaged"))
            elif in_unit_range and words & {"rate", "ratio", "share"}:
                m.agg, m.additivity, m.format = "avg", "non_additive", "percent"
            else:
                if words & {"price", "prc"} and words & _PRICE_TOTAL:
                    m.evidence.append(Evidence(kind="amount", weight=1, detail=(
                        f"{label} is the price of a whole line or document, not of one unit: an amount, added up")))
                if words & _MONEY and not (words & _QUANTITY and words & _MONEY <= _CURRENCY_CODES):
                    m.format = "currency"     # PRTS_USD_QTY: parts used, a quantity, never US dollars
                elif words & _QUANTITY or column_is_whole(p):
                    m.format = "integer" if column_is_whole(p) else "number"
                if periodic:
                    # On a balance, a "cost" alone may be a unit cost or a value at cost: never summed over time
                    # on that word alone (asked about instead).
                    flow = words & _FLOW - _COST_ONLY
                    if words & {"in", "out"} and words & _QUANTITY:
                        flow = flow | {"movement"}  # UnitsIn, QTY_OUT: what moved in the period, added up
                    if _counted(name):
                        flow = flow | {"count"}     # a number of receipts in the month: events, added up
                    level = (bool(words & _LEVEL) or opaque or not flow) and not (flow and not words & _LEVEL)
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
            # A number named by its own unit ("days supply", "weight oz") is in that unit, whatever else the row holds.
            if m.format in ("number", "integer") and not words & _OWN_UNITS and (words & _QUANTITY or opaque):
                unit = _unit_column(inventory, profiles, key, reachable, kind="unit", measured=words)
                if unit:
                    m.unit_column, m.unit_values = unit
            # An amount is in its row's currency, or in its document's (a line's amount in the currency on its order's
            # header): kept apart by that code. One in the company's own currency, or naming its currency
            # (OpenAmtUSD), is one currency, whatever currency code its row or document holds.
            one_currency = words & (_ONE_CURRENCY | _CURRENCIES) or (
                kept_both_ways and words & _IN_BASE and not words & _IN_DOCUMENT)
            if m.format == "currency" and not one_currency:
                unit = _unit_column(inventory, profiles, key, headers, kind="currency")
                if unit:
                    m.unit_column, m.unit_values = unit
            out.append(m)

        # Counting rows and documents.
        noun = names.counted_noun(names.readable(" ".join(names.table_noun(table.name)))) \
            if not names.opaque(table.name) else table.name
        counted = f"{noun.lower()} rows" if periodic or noun.lower().split()[-1:] and \
            noun.lower().split()[-1] in AMOUNT_NOUNS else names.plural(noun.lower())
        out.append(MeasureFinding(table=key, column=None, agg="count", additivity="additive", format="count",
                                  name=f"Number of {counted}",
                                  evidence=[Evidence(kind="row_count", weight=1, detail="counts rows")]))
        for column in table.columns:
            p = profile.columns[column.name]
            if column.data_type not in ("text", "integer") or column.name in keys[key].unique_columns:
                continue
            if column.name in keys[key].line_numbers or not _numbers_documents(column.name) and any(
                    column.name in k[1:] for k in [keys[key].primary_key, *keys[key].alternate_keys] if len(k) > 1):
                continue   # a line or renewal number runs within its document: it counts nothing on its own
            if any(j.from_table == key and column.name in j.from_columns and j.trust != "rejected" for j in joins):
                continue
            if not profile.rows or not (profile.rows / 1000 < p.distinct < profile.rows):
                continue
            words = _words(column.name)
            if names.opaque(column.name):
                # Short text, and many values: three currency codes are a code, not invoice numbers.
                is_document = column.data_type == "text" and (p.max_len or 0) <= 24 \
                    and p.distinct >= max(20, profile.rows / 100)
            else:
                identifying = {"number", "no", "nbr", "num", "id", "ref"}
                is_document = bool(words & identifying) and not words & (_CODE - identifying) \
                    and not _counted(column.name)
            if not is_document or column.name in {c.column for c in dates.get(key, [])}:
                continue
            thing = [w for w in names.core_column(bare(table, column.name))]
            while len(thing) > 1 and thing[-1] in ("number", "no", "nbr", "num", "id", "ref", "code") \
                    and not (len(thing) == 2 and len(thing[0]) <= 3):   # "Rx number": a short word keeps its "number"
                thing.pop()
            if len(thing) == 2 and len(thing[0]) <= 3 and thing[-1] in ("no", "nbr", "num"):
                thing[-1] = "number"
            label = names.readable("_".join(thing)) if thing and not names.opaque(column.name) else column.name
            out.append(MeasureFinding(table=key, column=column.name, agg="count_distinct", additivity="non_additive",
                                      format="count", name=f"Number of {names.plural(label.lower())}",
                                      evidence=[Evidence(kind="document_count", weight=1,
                                                         detail=f"{column.name} repeats across rows: counted once each")]))
    return out


def column_is_whole(p: ColumnProfile) -> bool:
    return p.integer_share == 1.0


def _kept_both_ways(inventory: Inventory) -> bool:
    """Does the database keep amounts both in the company's currency and in the document's (ExtAmtBase beside
    ExtAmtDoc, PaidAmtBase beside PaidAmtDoc)? Then "base" names the company's currency on every table."""
    for table in inventory.tables.values():
        doc = {tuple(w for w in names.tokens(c.name) if w not in _IN_DOCUMENT) for c in table.columns
               if set(names.tokens(c.name)) & _IN_DOCUMENT}
        if any(tuple(w for w in names.tokens(c.name) if w not in _IN_BASE) in doc for c in table.columns
               if set(names.tokens(c.name)) & _IN_BASE):
            return True
    return False


def _reachable_one_step(joins: list[JoinFinding]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for j in joins:
        if j.trust in ("verified", "declared") and not j.to_calendar:
            out.setdefault(j.from_table, []).append(j.to_table)
    return out


def _documents_of_lines(joins: list[JoinFinding], keys: dict[str, TableKeys]) -> dict[str, list[str]]:
    """The document each table's rows are the lines of (an order line's order): its currency is theirs. A customer's
    or a product's currency is not the currency of what was sold to them."""
    out: dict[str, list[str]] = {}
    for j in joins:
        if j.trust in ("verified", "declared") and not j.to_calendar and j.from_table in keys \
                and _lines_of(keys[j.from_table], j):
            out.setdefault(j.from_table, []).append(j.to_table)
    return out


# What a unit column can say it measures (WeightUnitMeasureCode, SIZE_UOM): the unit of that alone.
_MEASURED = {"weight", "wgt", "size", "volume", "vol", "length", "width", "height", "depth", "dimension", "distance"}


def _unit_column(inventory: Inventory, profiles: dict[str, TableProfile], key: str,
                 reachable: dict[str, list[str]], *, kind: str,
                 measured: set[str] = frozenset()) -> tuple[tuple[str, str], list[str]] | None:
    currencies: list[tuple[tuple[str, str], list[str], bool]] = []
    for table_key in [key] + sorted(reachable.get(key, [])):
        table = inventory.tables[table_key]
        for column in table.columns:
            p = profiles[table_key].columns[column.name]
            if column.data_type != "text" or not p.top or p.distinct > 20 or (p.max_len or 99) > 6:
                continue
            values = {(t.value or "").strip().lower() for t in p.top}
            words = _words(column.name)
            if kind == "unit" and words & _MEASURED and not words & _MEASURED & measured:
                continue   # WeightUnitMeasureCode is the unit of a weight, not of a count of units in stock
            if kind == "unit" and (words & _UNIT_WORDS or values and values <= _UNIT_VALUES):
                return (table_key, column.name), sorted(t.value or "" for t in p.top)
            if kind == "currency" and values and values <= _CURRENCIES:
                # The company's own currency (BaseCurrencyCd, always USD) says nothing of a document's.
                said = len(values) > 1 and not words & (_IN_BASE | _ONE_CURRENCY)
                currencies.append(((table_key, column.name), sorted(t.value or "" for t in p.top), said))
    if currencies:
        column, values, _ = next((c for c in currencies if c[2]), currencies[0])
        return column, values
    return None
