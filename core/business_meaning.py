"""
core/business_meaning.py

What a warehouse's codes mean, read from the warehouse itself.

The built-in dictionary (core/schema_enrichment.py) reads a code the same way
in every column of every warehouse, so it holds only codes with one meaning.
The others, and the codes no vocabulary knows, are read here, for one tenant,
from what its warehouse carries:

- where a code stands: QR in a calendar is a quarter; DLY before FCT in a
  table's name is the table's grain, daily, and DLY before DT names a date's
  event, delivery; HND after ON is "on hand"; CTR after PFT is a center;
- what stands beside it: CTY and PRV in a table that holds an address are a
  city and a province;
- what it holds: a column whose every value is a country code is a country,
  whatever its name says -- a profit center's CO holds CA, and CO reads
  "company" everywhere else;
- who it is to the business: a seller the business pays, or whose items it
  stocks, is what its readers call a supplier.

Each reading is a PROPOSAL carrying its evidence. Nothing here takes effect:
an admin confirms a reading before it joins the tenant's vocabulary. A code no
rule can read is listed as well, with the columns it appears in, so what is
left for the admin to write is in one place rather than in a bad answer.

The inputs are names, types and -- when the caller sampled them -- a few
distinct values of the columns that can settle a reading. Nothing is sent to a
model.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.date_roles import _has_calendar_attributes
from core.identifier_intelligence import _expansion_lexicon, tokenize_identifier

log = logging.getLogger("querybot.business_meaning")

# Where a reading applies once confirmed.
CODE = "code"      # every column the code appears in
COLUMN = "column"  # one column name, wherever it appears
TABLE = "table"    # one table's name

# The codes the built-in dictionary leaves out because they mean different
# things in different columns, and what they can mean. A vocabulary pack may
# still read one of them one way; context is checked wherever they appear.
TWO_WAY_CODES: dict[str, tuple[str, ...]] = {
    "DLY": ("daily", "delivery"),
    "QR": ("quarter",),
    "CO": ("company", "country"),
    "CTY": ("city", "county"),
    "PRV": ("province", "previous"),
    "STT": ("state", "status", "start"),
    "INV": ("inventory", "invoice"),
    "LST": ("last", "list"),
    "EXT": ("external", "extended"),
    "CTR": ("center", "counter"),
    "HND": ("hand", "handling"),
    "RM": ("room", "raw material"),
    "CHN": ("chain", "channel"),
    "FT": ("feet", "full time"),
}

# Words a column name spells out. They are not codes, so an unread one is not
# a gap: DAY_OF_WK is "day of week", not three codes and a question.
_WORDS = frozenset({
    "OF", "AS", "TO", "BY", "IN", "AT", "FOR", "AND", "OR", "PER", "THE",
    "NO", "ID", "KEY", "DAY", "AGE", "BIN", "SHOW", "ALL", "NEW", "OLD", "END",
    "TOP", "LOW", "HIGH", "SIZE", "RANK", "LEVEL", "SCORE", "NOTE", "YES",
    "LAST", "NEXT", "FIRST", "FROM", "START", "OPEN", "CLOSE", "HOUR",
    "WEEK", "YEAR", "MONTH", "NAME", "CODE", "DATE", "TIME", "TYPE", "AREA",
    "ZONE", "ROLE", "RULE", "SITE", "TEAM", "UNIT", "USER", "HOLD", "LINE",
    "MODE", "PLAN", "SALE", "SALES", "STOCK", "COST", "PRICE", "BRAND", "SKU",
})

# Load and audit columns say nothing a reader asks about.
_INFRA_PREFIXES = ("AZ_", "ETL_", "DW_", "SYS_", "META_", "STG_", "CDC_")

_TEXT_TYPES = ("CHAR", "TEXT", "STRING", "CLOB")

# ── Rules: where a code stands ───────────────────────────────────────────────

_CALENDAR_READINGS = {"QR": "quarter", "QTR": "quarter", "DOW": "day of week", "DOY": "day of year"}
_CALENDAR_UNITS = frozenset({"YR", "YEAR", "MTH", "MONTH", "WK", "WEEK", "QR", "QTR", "QUARTER", "HDY", "HOLIDAY"})

_TABLE_KINDS = frozenset({"FCT", "FACT", "SNP", "SNAPSHOT", "AGG", "SMY", "HIST"})
_GRAIN_READINGS = {
    "DLY": "daily", "WKLY": "weekly", "WKY": "weekly", "MTHLY": "monthly",
    "QTRLY": "quarterly", "YRLY": "yearly",
}

_DATE_TOKENS = frozenset({"DT", "DATE", "TS", "DTM", "DTTM"})
_EVENT_READINGS = {"DLY": "delivery"}

# A code read by the word before it: {code: {previous token: reading}}.
_AFTER: dict[str, dict[str, str]] = {
    "HND": {"ON": "hand"},
    "CTR": {t: "center" for t in (
        "PFT", "PROFIT", "CST", "COST", "WRK", "WORK", "DST", "DIST", "DC",
        "CALL", "SVC", "SRV", "RSP", "DATA", "TRN")},
    "INV": {t: "inventory" for t in ("PHY", "STK", "CYC", "CYCLE")},
    "RM": {"SHOW": "room"},
    "FT": {"SQ": "feet"},
    "CHN": {
        "SLY": "chain", "SUPPLY": "chain",
        "SLS": "channel", "SAL": "channel", "SALES": "channel", "MKT": "channel",
    },
    "PRV": {"STT": "province"},
}

# A code read by the word after it: {code: {next token: reading}}.
_BEFORE: dict[str, dict[str, str]] = {
    "LST": {
        **{t: "last" for t in (
            "UPD", "RCT", "SLD", "PCH", "ISS", "MOV", "CNT", "ORD", "DLV", "SHP",
            "TXN", "ACT", "CHG", "MOD", "VST", "PMT", "PAY", "INV", "IVC")},
        **{t: "list" for t in ("PRC", "PCE", "PRICE")},
    },
    "PRV": {t: "previous" for t in (
        "YR", "MTH", "WK", "QR", "QTR", "PRD", "DAY", "DT", "YEAR", "MONTH", "PERIOD")},
    "SQ": {"FT": "square", "M": "square", "MT": "square", "MTR": "square"},
    "EXT": {
        **{t: "external" for t in ("ID", "REF", "SYS", "KEY", "CD", "NUM", "NO", "SRC")},
        **{t: "extended" for t in ("PRC", "PCE", "AMT", "CST", "VAL", "COST", "PRICE")},
    },
    "STT": {"PRV": "state"},
}

# A column that repeats another in a second language: ITM_FR_NM beside ITM_NM.
_LANGUAGE_CODES = {"FR": "french", "FRA": "french", "FRE": "french", "EN": "english", "ENG": "english"}

# Two codes that read as one word. The second is absorbed into the first, so
# the reading belongs to the column: read code by code, RE_CLS would be
# "reclass class".
_PAIRS = {("CO", "OP"): "co-op", ("RE", "CLS"): "reclass"}
_ABSORBED = "\x00absorbed"

# ── Rules: what stands beside it ─────────────────────────────────────────────

_ADDRESS_LINE = frozenset({"ADR", "ADDR", "ADDRESS"})
_ADDRESS_POSTAL = frozenset({"PSL", "ZIP", "POSTAL", "POSTCODE", "PSTL", "PST"})
_ADDRESS_PLACE = frozenset({"CTY", "CITY", "PRV", "PROVINCE", "STT", "STATE", "CNTRY", "CTRY", "COUNTRY"})
_ADDRESS_READINGS = {
    "CTY": ("city", 75), "PRV": ("province", 75), "STT": ("state", 75),
    "CNTRY": ("country", 80), "CTRY": ("country", 80), "CO": ("country", 60),
}

# ── Rules: what it holds ─────────────────────────────────────────────────────

_COUNTRIES_2 = frozenset("""
AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM
BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX
CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG
GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR
IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV
LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE
NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO
RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF
TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF
WS YE YT ZA ZM ZW
""".split())
_COUNTRIES_3 = frozenset({
    "CAN", "USA", "MEX", "FRA", "DEU", "GBR", "ITA", "ESP", "PRT", "NLD", "BEL",
    "CHE", "AUT", "IRL", "CHN", "JPN", "KOR", "IND", "BRA", "AUS", "NZL", "ZAF",
})
_COUNTRY_NAMES = frozenset({
    "CANADA", "UNITED STATES", "USA", "MEXICO", "FRANCE", "GERMANY", "UNITED KINGDOM",
    "ITALY", "SPAIN", "CHINA", "JAPAN", "INDIA", "BRAZIL", "AUSTRALIA",
})
_PROVINCES = frozenset({"AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"})
_STATES = frozenset("""
AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE
NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR
""".split())
_POSTAL = re.compile(r"^(?:[A-Z]\d[A-Z] ?\d[A-Z]\d|\d{5}(?:-\d{4})?)$")
_PLACE_NAME = re.compile(r"^[A-Z][A-Z .'\-]{2,}$")

# The placeholder rows a dimension keeps for keys it could not match.
_PLACEHOLDERS = re.compile(r"^(?:NO[_ ]?VALUE|NO[_ ]?MATCH|NULL VALUE.*|VALUE PROVIDED.*|N/?A|NONE|UNKNOWN|-)$")

# ── Rules: who it is to the business ─────────────────────────────────────────

# A party read literally, and what the business's readers call it. A seller
# the business pays, or whose items it stocks, sells TO the business.
_PARTY_WORDS = {"seller": "supplier"}
_PAYEE_TOKENS = frozenset({"PYE", "PAYEE", "PYEE"})
_STOCK_TOKENS = frozenset({"HND", "OH", "STK", "PCH", "RCT", "RCPT"})
_KEY_TOKENS = frozenset({"KEY", "ID", "SK", "CD", "CODE"})


@dataclass
class Proposal:
    """One reading for an admin to confirm, with what it rests on."""

    scope: str
    subject: str
    reading: str
    rule: str
    confidence: int
    evidence: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    where: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope, "subject": self.subject, "reading": self.reading,
            "rule": self.rule, "confidence": self.confidence,
            "evidence": list(self.evidence), "synonyms": list(self.synonyms),
            "where": list(self.where),
        }


@dataclass(frozen=True)
class _Reading:
    word: str
    rule: str
    confidence: int
    evidence: str


@dataclass(frozen=True)
class _Place:
    table: str                # bare table name, upper case
    column: str               # the column as discovered; "" for the table's own name
    tokens: tuple[str, ...]
    index: int

    @property
    def token(self) -> str:
        return self.tokens[self.index]

    @property
    def label(self) -> str:
        return f"{self.table}.{self.column}" if self.column else self.table


def _bare(name: str) -> str:
    return str(name or "").strip().strip("[]\"`").split(".")[-1].strip("[]\"`").upper()


def _normal_value(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().upper()


# The meanings sampled values can settle, per code, and how each is told.
_VALUE_CANDIDATES = {
    "CO": {"company", "country"},
    "CTY": {"city", "county"},
    "PRV": {"province", "previous", "state or province"},
    "STT": {"state", "status", "start", "state or province"},
    "CNTRY": {"country"},
    "CTRY": {"country"},
}
_VALUE_WORDING = {
    "country": "a country code", "province": "a province code", "state": "a state code",
    "state or province": "a state or province code", "city": "a place name",
    "county": "a county name",
}


def value_kinds(values: Iterable[object]) -> tuple[set[str], list[str]]:
    """What a column's sampled values are: the kinds every real value fits
    (country, province, state, postal code, place name), and the distinct real
    values, placeholder rows and blanks left out."""
    real = sorted({v for v in (_normal_value(x) for x in values) if v and not _PLACEHOLDERS.match(v)})
    if not real:
        return set(), []
    kinds: set[str] = set()
    if all(v in _COUNTRIES_2 or v in _COUNTRIES_3 or v in _COUNTRY_NAMES for v in real):
        kinds.add("country")
    if all(v in _PROVINCES for v in real):
        kinds.add("province")
    if all(v in _STATES for v in real):
        kinds.add("state")
    if all(v in _PROVINCES or v in _STATES for v in real) and not kinds & {"province", "state"}:
        kinds.add("state or province")
    if all(_POSTAL.match(v) for v in real):
        kinds.add("postal code")
    if (
        len(real) >= 3 and all(_PLACE_NAME.match(v) for v in real)
        and not any(v in _COUNTRIES_2 or v in _PROVINCES or v in _STATES for v in real)
    ):
        kinds.add("county" if any(v.endswith(" COUNTY") for v in real) else "city")
    return kinds, real


def _examples(values: list[str], limit: int = 4) -> str:
    shown = ", ".join(values[:limit])
    return shown + (", ..." if len(values) > limit else "")


class _Warehouse:
    """The tenant's tables as the rules see them."""

    def __init__(
        self, table_columns: dict[str, Any], vocab: Any,
        values: dict[str, list[object]] | None, ignore: Iterable[str],
    ) -> None:
        from core.vocab_packs import get_active_vocab

        self.vocab = vocab if vocab is not None else get_active_vocab()
        self.lexicon = _expansion_lexicon(self.vocab)
        # "TABLE.COLUMN", however qualified the table was: the rules look up
        # the bare table name.
        self.values = {
            ".".join(str(k).upper().split(".")[-2:]): list(v or []) for k, v in (values or {}).items()
        }
        self.ignore = {str(t).upper() for t in ignore or ()}
        self.tables: dict[str, list[tuple[str, str]]] = {}
        for table, columns in (table_columns or {}).items():
            if isinstance(columns, dict):
                pairs = [(str(c), str(t or "")) for c, t in columns.items()]
            else:
                pairs = [(str(c[0]), str(c[1] or "")) if isinstance(c, (list, tuple)) else (str(c), "")
                         for c in columns or []]
            self.tables.setdefault(_bare(table), pairs)
        self.tokens: dict[tuple[str, str], tuple[str, ...]] = {}
        for table, columns in self.tables.items():
            self.tokens[(table, "")] = tuple(tokenize_identifier(table, vocab=self.vocab))
            for column, _ in columns:
                self.tokens[(table, column)] = tuple(tokenize_identifier(column, vocab=self.vocab))
        self.calendars = {
            table: [c for c, _ in columns if set(self.tokens[(table, c)]) & _CALENDAR_UNITS]
            for table, columns in self.tables.items()
            if _has_calendar_attributes({c.upper() for c, _ in columns})
        }
        self.addresses = {
            table: [c for c, _ in columns if set(self.tokens[(table, c)]) & (_ADDRESS_LINE | _ADDRESS_POSTAL | _ADDRESS_PLACE)]
            for table, columns in self.tables.items()
            if self._holds_address(table, columns)
        }

    def _holds_address(self, table: str, columns: list[tuple[str, str]]) -> bool:
        tokens = {t for c, _ in columns for t in self.tokens[(table, c)]}
        signs = [bool(tokens & _ADDRESS_LINE), bool(tokens & _ADDRESS_POSTAL),
                 len(tokens & _ADDRESS_PLACE) >= 2]
        return sum(signs) >= 2

    def places(self) -> Iterable[_Place]:
        for (table, column), tokens in self.tokens.items():
            if column.upper().startswith(_INFRA_PREFIXES):
                continue
            for index in range(len(tokens)):
                yield _Place(table, column, tokens, index)

    def is_code(self, token: str) -> bool:
        """A token that needs a reading: not a number, a letter, a word, or a
        name the caller said to leave alone (the tenant's own)."""
        return (
            len(token) >= 2 and not token.isdigit() and token not in _WORDS
            and token not in self.ignore
        )

    def column_type(self, table: str, column: str) -> str:
        return next((t for c, t in self.tables.get(table, []) if c == column), "")


def _from_values(place: _Place, wh: _Warehouse) -> _Reading | None:
    """A name whose values settle which of its meanings it has."""
    if not place.column:
        return None
    sampled = wh.values.get(f"{place.table}.{place.column}".upper())
    if not sampled:
        return None
    kinds, real = value_kinds(sampled)
    if not kinds:
        return None
    chosen = sorted(kinds & _VALUE_CANDIDATES.get(place.token, set()))
    if len(chosen) != 1:
        return None
    word = chosen[0]
    what = _VALUE_WORDING[word]
    if len(real) == 1:
        return _Reading(word, "values", 90, f"its only value is {what} ({real[0]})")
    return _Reading(word, "values", 90, f"every one of its {len(real)} sampled values is {what} ({_examples(real)})")


def _from_calendar(place: _Place, wh: _Warehouse) -> _Reading | None:
    word = _CALENDAR_READINGS.get(place.token)
    if not word or not place.column or place.table not in wh.calendars:
        return None
    return _Reading(word, "calendar", 90,
                    f"{place.table} is a calendar: {_examples(wh.calendars[place.table])}")


def _from_grain(place: _Place, wh: _Warehouse) -> _Reading | None:
    word = _GRAIN_READINGS.get(place.token)
    nxt = place.tokens[place.index + 1] if place.index + 1 < len(place.tokens) else ""
    if not word or nxt not in _TABLE_KINDS:
        return None
    siblings = sorted(
        t for t in wh.tables
        if t != place.table and len(t.split("_")) == len(place.table.split("_"))
        and t.split("_")[-1] in _TABLE_KINDS and t.rsplit("_", 2)[0] == place.table.rsplit("_", 2)[0]
    )
    evidence = f"{place.table}: the word before {nxt} says what one row covers"
    if siblings:
        evidence += f", as in {', '.join(siblings)}"
    return _Reading(word, "grain", 85, evidence)


def _from_event_date(place: _Place, wh: _Warehouse) -> _Reading | None:
    word = _EVENT_READINGS.get(place.token)
    nxt = place.tokens[place.index + 1] if place.index + 1 < len(place.tokens) else ""
    if not word or not place.column or nxt not in _DATE_TOKENS:
        return None
    return _Reading(word, "event_date", 85, f"{place.column}: a date is named for the event it records")


def _from_neighbours(place: _Place, wh: _Warehouse) -> _Reading | None:
    tokens, i = place.tokens, place.index
    prev = tokens[i - 1] if i > 0 else ""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    if (prev, place.token) in _PAIRS:
        return _Reading(_ABSORBED, "pair", 80, "")
    pair = _PAIRS.get((place.token, nxt))
    if pair:
        return _Reading(pair, "pair", 80, f"{place.label}: {place.token}_{nxt} is one word, \"{pair}\"")
    word = _AFTER.get(place.token, {}).get(prev)
    if word:
        return _Reading(word, "phrase", 80, f"{place.label}: after {prev}, {place.token} reads \"{word}\"")
    word = _BEFORE.get(place.token, {}).get(nxt)
    if word:
        return _Reading(word, "phrase", 80, f"{place.label}: before {nxt}, {place.token} reads \"{word}\"")
    return None


def _from_twin(place: _Place, wh: _Warehouse) -> _Reading | None:
    """FR in ITM_FR_NM, beside ITM_NM: the same attribute in French."""
    word = _LANGUAGE_CODES.get(place.token)
    if not word or not place.column:
        return None
    twin = "_".join(t for i, t in enumerate(place.tokens) if i != place.index)
    if twin not in {c.upper() for c, _ in wh.tables.get(place.table, [])}:
        return None
    return _Reading(word, "twin", 90, f"{place.column} repeats {twin} in {word}")


def _from_address(place: _Place, wh: _Warehouse) -> _Reading | None:
    reading = _ADDRESS_READINGS.get(place.token)
    if not reading or not place.column or place.table not in wh.addresses:
        return None
    word, confidence = reading
    # Values outrank neighbours both ways: a CO beside an address that holds
    # "100" and "200" is a company, whatever stands next to it.
    sampled = wh.values.get(f"{place.table}.{place.column}".upper())
    if sampled:
        kinds, real = value_kinds(sampled)
        if real and word not in kinds:
            return None
    return _Reading(word, "address", confidence,
                    f"{place.table} holds an address: {_examples(wh.addresses[place.table])}")


# Most specific first: what a column holds outranks what its name suggests.
_RULES = (
    _from_values, _from_calendar, _from_grain, _from_event_date, _from_twin,
    _from_neighbours, _from_address,
)


def _read(place: _Place, wh: _Warehouse) -> _Reading | None:
    for rule in _RULES:
        reading = rule(place, wh)
        if reading is not None:
            return reading
    return None


def _column_reading(table: str, column: str, wh: _Warehouse) -> str:
    """The whole name read with every rule that applies to its codes."""
    tokens = wh.tokens[(table, column)]
    words: list[str] = []
    for index, token in enumerate(tokens):
        reading = _read(_Place(table, column, tokens, index), wh)
        if reading is not None:
            if reading.word != _ABSORBED:
                words.append(reading.word)
            continue
        words.append(wh.lexicon.get(token, token.lower()))
    return " ".join(w for w in words if w)


def _party_proposals(wh: _Warehouse) -> list[Proposal]:
    """A seller the business pays, or whose items it stocks, is its supplier."""
    proposals = []
    for code, literal in sorted((c, r) for c, r in wh.lexicon.items() if r in _PARTY_WORDS):
        tables = [t for t in wh.tables if code in wh.tokens[(t, "")]]
        if not tables:
            continue
        evidence = []
        for table in tables:
            payees = [c for c, _ in wh.tables[table] if set(wh.tokens[(table, c)]) & _PAYEE_TOKENS]
            if payees:
                evidence.append(f"{table} carries a payee ({', '.join(payees)}): the business pays this party")
        keys = {c.upper() for t in tables for c, _ in wh.tables[t] if c.upper().startswith(f"{t}_") or c.upper() == f"{t}_KEY"}
        for table, columns in wh.tables.items():
            if table in tables:
                continue
            names = {c.upper() for c, _ in columns}
            stock = [
                c for c, _ in columns
                if set(wh.tokens[(table, c)]) & _STOCK_TOKENS and not set(wh.tokens[(table, c)]) & _KEY_TOKENS
            ]
            if names & keys and stock:
                evidence.append(f"{table} holds stock ({_examples(stock, 2)}) by this party")
        if not evidence:
            continue
        where = sorted(f"{t}.{c}" for t in wh.tables for c, _ in wh.tables[t] if code in wh.tokens[(t, c)])
        proposals.append(Proposal(
            CODE, code, _PARTY_WORDS[literal], "party", 70,
            evidence=evidence[:3], synonyms=["vendor", literal], where=where,
        ))
    return proposals


# Spellings of one word. A pack that reads CTR as "centre" agrees with a rule
# that reads it "center"; neither is a correction of the other.
_SAME_WORD = {"centre": "center", "colour": "color", "catalogue": "catalog"}


def _same(left: str, right: str) -> bool:
    def norm(text: str) -> str:
        return " ".join(_SAME_WORD.get(w, w) for w in str(text or "").lower().split())
    return norm(left) == norm(right)


def propose_meanings(
    table_columns: dict[str, Any],
    vocab=None,
    values: dict[str, list[object]] | None = None,
    ignore: Iterable[str] = (),
) -> list[Proposal]:
    """Readings to propose for one tenant's tables.

    `table_columns` maps each table to its columns ({column: type} or
    [(column, type)]); `values` maps "TABLE.COLUMN" (the table qualified or
    not) to a column's sampled values;
    `ignore` names tokens that need no reading (the tenant's own name).

    A code nobody reads that reads the same wherever it appears is proposed
    once, for every column; one that reads differently in different places is
    proposed column by column, as is a code read together with its neighbour
    (CO_OP, "co-op"); one the vocabulary already reads is proposed only where
    its context says otherwise. A code no rule reads is listed with an empty
    reading, for the admin to write.
    """
    wh = _Warehouse(table_columns, vocab, values, ignore)
    found: dict[str, list[tuple[_Place, _Reading | None]]] = defaultdict(list)
    for place in wh.places():
        token = place.token
        if not wh.is_code(token):
            continue
        current = wh.lexicon.get(token, "")
        reading = _read(place, wh)
        if current and token not in TWO_WAY_CODES and reading is None:
            continue
        found[token].append((place, reading))

    by_subject: dict[tuple[str, str], Proposal] = {}

    def add(proposal: Proposal) -> None:
        key = (proposal.scope, proposal.subject)
        held = by_subject.get(key)
        if held is None:
            by_subject[key] = proposal
            return
        held.where = sorted(set(held.where) | set(proposal.where))
        held.evidence = list(dict.fromkeys([*held.evidence, *proposal.evidence]))[:3]
        held.confidence = min(held.confidence, proposal.confidence)

    def by_column(place: _Place, reading: _Reading, note: str = "") -> None:
        scope, subject = (COLUMN, place.column.upper()) if place.column else (TABLE, place.table)
        add(Proposal(
            scope, subject, _column_reading(place.table, place.column, wh),
            reading.rule, reading.confidence,
            evidence=[e for e in (reading.evidence, note) if e], where=[place.label],
        ))

    for token in sorted(found):
        places = found[token]
        current = wh.lexicon.get(token, "")
        read = [(p, r) for p, r in places if r is not None and r.word != _ABSORBED]
        unread = [p for p, r in places if r is None]
        if current:
            for place, reading in read:
                if not _same(reading.word, current):
                    by_column(place, reading, f"{token} reads \"{current}\" elsewhere")
            continue
        per_code = [(p, r) for p, r in read if r.rule != "pair"]
        words = {r.word for _, r in per_code}
        if len(words) == 1 and len(per_code) == len(read) and not unread:
            evidence = list(dict.fromkeys(r.evidence for _, r in read if r.evidence))
            add(Proposal(
                CODE, token, next(iter(words)), read[0][1].rule, min(r.confidence for _, r in read),
                evidence=evidence[:3], where=sorted({p.label for p, _ in read}),
            ))
            continue
        for place, reading in read:
            by_column(place, reading)
        if unread:
            add(Proposal(
                CODE, token, "", "unread", 0,
                evidence=[f"no rule reads {token} in these names"],
                where=sorted({p.label for p in unread}),
            ))

    for proposal in _party_proposals(wh):
        add(proposal)
    return list(by_subject.values())


# ── At discovery ─────────────────────────────────────────────────────────────

# Values read per column: enough to say what a column holds, few enough that
# the evidence stays a sample.
_VALUES_PER_COLUMN = 50


def _discovered_tables(schema_dir: str) -> dict[str, list[tuple[str, str]]]:
    """{table as discovered: [(column, type)]} from the discovered schema."""
    import json
    from pathlib import Path

    from core.schema import _normalize_schema

    path = Path(schema_dir) / "_schema.json" if schema_dir else None
    if path is None or not path.exists():
        return {}
    try:
        master = _normalize_schema(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        log.warning("Business meanings: could not read %s: %s", path, exc)
        return {}
    return {
        str(fqn): [
            (str(col["name"]), str(col.get("type") or ""))
            for col in info.get("columns") or []
            if isinstance(col, dict) and col.get("name")
        ]
        for fqn, info in master.items()
        if not str(fqn).startswith("__") and isinstance(info, dict)
    }


def propose_for_account(account_id: str, *, base_dir: str = "clients") -> dict[str, int]:
    """Discovery's pass: read the tenant's discovered tables and keep what
    their codes appear to mean for the admin to confirm.

    The values come from the tenant's value index, which discovery builds
    first and which holds only what its privacy gates let it keep (masked,
    sensitive and, for a regulated tenant, uncleared columns are never in
    it); nothing is read from the warehouse here. The tenant's own name is
    not a code. Returns counts: proposed, of them unread, kept, dropped.
    """
    import store
    from core.value_index import sample_values_by_column
    from core.vocab_packs import vocab_for_account

    state = store.get_client_state(account_id) or {}
    tables = _discovered_tables(str(state.get("schema_dir") or ""))
    if not tables:
        log.info("Business meanings for %s: no discovered schema", account_id)
        return {"proposed": 0, "unread": 0, "kept": 0, "dropped": 0}
    values = {
        f"{row['table_fqn']}.{row['column']}": row["values"]
        for row in sample_values_by_column(
            account_id, per_column=_VALUES_PER_COLUMN, max_columns=2000, base_dir=base_dir,
        )
    }
    client = store.get_client(account_id) or {}
    own_name = re.findall(r"[A-Za-z0-9]+", str(client.get("client_name") or ""))
    proposals = propose_meanings(
        {fqn: columns for fqn, columns in tables.items()},
        vocab=vocab_for_account(account_id), values=values, ignore=own_name,
    )
    saved = store.save_business_meanings(account_id, [p.as_dict() for p in proposals])
    return {
        "proposed": len(proposals),
        "unread": sum(1 for p in proposals if not p.reading),
        **saved,
    }

