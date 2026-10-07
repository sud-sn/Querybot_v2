"""Naming styles for the synthetic warehouses.

Every domain is written once with descriptive names (``sales_order_lines``,
``order_date_key``) and rendered in each style, so the same data and the same
questions test whether QueryBot understands a warehouse whatever it is called:

* ``descriptive``  as written: ``order_lines.order_date_key``
* ``warehouse``    uppercase codes: ``SLS_ORD_LN_FCT.ORD_DT_KEY``
* ``pascal``       a Kimball star: ``FactOrderLines.OrderDateKey``
* ``generic``      no meaning at all: ``T04.C07``, and no declared keys

The warehouse style abbreviates word by word from one dictionary, so a word is
always abbreviated the same way across tables, as real code-named warehouses do.
"""

from __future__ import annotations

import hashlib
import re

STYLES = ("descriptive", "warehouse", "pascal", "generic")

# One abbreviation per word. Words not listed are shortened by _abbreviate().
_ABBREVIATIONS = {
    "account": "ACCT", "accounts": "ACCT", "actual": "ACT", "address": "ADDR", "amount": "AMT", "at": "TS",
    "balance": "BAL", "billing": "BIL", "budget": "BDGT", "by": "BY", "calendar": "CAL", "cancelled": "CNL",
    "categories": "CAT", "category": "CAT", "channel": "CHNL", "city": "CTY", "closed": "CLS", "code": "CD",
    "cost": "CST", "count": "CNT", "country": "CTRY", "credit": "CR", "currency": "CUR", "customer": "CUST",
    "customers": "CUST", "date": "DT", "day": "DAY", "debit": "DR", "delivered": "DLV", "department": "DEPT",
    "departments": "DEPT", "description": "DESC", "discount": "DSC", "due": "DUE", "employee": "EMP",
    "employees": "EMP", "end": "END", "entries": "ENT", "entry": "ENT", "event": "EVT", "events": "EVT",
    "expected": "EXP", "first": "FST", "fiscal": "FSC", "flag": "FLG", "gross": "GRS", "group": "GRP",
    "groups": "GRP", "hand": "HND", "headcount": "HC", "hire": "HIRE", "home": "HM", "id": "KEY",
    "invoice": "INV", "invoices": "INV", "is": "IS", "item": "ITM", "items": "ITM", "journal": "JRNL",
    "key": "KEY", "last": "LST", "ledger": "LDG", "level": "LVL", "line": "LN", "lines": "LN", "list": "LST",
    "loaded": "LD", "location": "LOC", "locations": "LOC", "manager": "MGR", "margin": "MRG", "measure": "MSR",
    "modified": "MOD", "month": "MTH", "monthly": "MTH", "movement": "MVT", "movements": "MVT", "name": "NM",
    "net": "NET", "number": "NO", "of": "", "on": "ON", "opened": "OPN", "order": "ORD", "orders": "ORD",
    "paid": "PD", "parent": "PRNT", "payment": "PMT", "payments": "PMT", "pct": "PCT", "percent": "PCT",
    "period": "PER", "plan": "PLN", "plans": "PLN", "planned": "PLN", "price": "PRC", "priority": "PRI",
    "product": "PRD", "products": "PRD", "purchase": "PUR", "quantity": "QTY", "quarter": "QTR",
    "reason": "RSN", "received": "RCV", "receipt": "RCPT", "receipts": "RCPT", "refund": "RFD",
    "region": "RGN", "regions": "RGN", "rejected": "RJT", "requested": "REQ", "resolved": "RSLV",
    "return": "RTN", "returns": "RTN", "revenue": "REV", "salary": "SAL", "sales": "SLS", "segment": "SEG",
    "ship": "SHP", "shipped": "SHP", "sku": "SKU", "snapshot": "SNP", "snapshots": "SNP", "start": "STRT",
    "status": "STS", "stock": "STK", "store": "STR", "stores": "STR", "subscription": "SUB",
    "subscriptions": "SUB", "supplier": "SUPP", "suppliers": "SUPP", "target": "TGT", "targets": "TGT",
    "tax": "TAX", "termination": "TERM", "ticket": "TKT", "tickets": "TKT", "type": "TYP", "unit": "UNT",
    "units": "UNT", "updated": "UPD", "user": "USR", "value": "VAL", "warehouse": "WHS",
    "warehouses": "WHS", "week": "WK", "weekend": "WKND", "year": "YR",
}

_KIND_SUFFIX = {"fact": "FCT", "snapshot": "SNP", "dimension": "DIM", "bridge": "BRG", "calendar": "DIM"}
_PASCAL_PREFIX = {"fact": "Fact", "snapshot": "Fact", "dimension": "Dim", "bridge": "Bridge", "calendar": "Dim"}


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w]


def _abbreviate(word: str, extra: dict[str, str] | None = None) -> str:
    if extra and word in extra:
        return extra[word]
    if word in _ABBREVIATIONS:
        return _ABBREVIATIONS[word]
    if word.isdigit() or len(word) <= 3:
        return word.upper()
    # First letter, then consonants, four letters at most: "transfer" -> "TRNS".
    tail = [c for c in word[1:] if c not in "aeiou"]
    return (word[0] + "".join(tail))[:4].upper()


def warehouse_table(name: str, kind: str, extra: dict[str, str] | None = None) -> str:
    parts = [a for a in (_abbreviate(w, extra) for w in _words(name)) if a]
    if kind == "calendar":
        return "CAL_DIM"
    return "_".join(parts + [_KIND_SUFFIX.get(kind, "TBL")])


def warehouse_column(name: str, extra: dict[str, str] | None = None) -> str:
    return "_".join(a for a in (_abbreviate(w, extra) for w in _words(name)) if a)


def pascal_table(name: str, kind: str) -> str:
    return _PASCAL_PREFIX.get(kind, "") + "".join(w.capitalize() for w in _words(name))


def pascal_column(name: str) -> str:
    return "".join(w.capitalize() if w not in ("id",) else "Key" for w in _words(name))


def generic_order(names: list[str], salt: str) -> list[str]:
    """A fixed, meaningless order: generic table numbers must not follow the design."""
    return sorted(names, key=lambda n: hashlib.sha256(f"{salt}:{n}".encode()).hexdigest())


def rename_map(tables: dict[str, tuple[str, list[str]]], style: str, salt: str,
               abbreviations: dict[str, str] | None = None) -> dict[str, tuple[str, dict[str, str]]]:
    """``{logical table: (kind, [logical columns])}`` -> ``{logical table: (physical, {column: physical})}``.

    ``abbreviations`` adds or overrides warehouse-style abbreviations for one domain.
    """
    if style not in STYLES:
        raise ValueError(f"unknown naming style {style!r}")
    out: dict[str, tuple[str, dict[str, str]]] = {}
    if style == "generic":
        for i, table in enumerate(generic_order(list(tables), salt), start=1):
            _, columns = tables[table]
            out[table] = (f"T{i:02d}", {c: f"C{j:02d}" for j, c in enumerate(columns, start=1)})
        return out
    for table, (kind, columns) in tables.items():
        if style == "descriptive":
            out[table] = (table, {c: c for c in columns})
        elif style == "warehouse":
            out[table] = (warehouse_table(table, kind, abbreviations),
                          {c: warehouse_column(c, abbreviations) for c in columns})
        else:
            out[table] = (pascal_table(table, kind), {c: pascal_column(c) for c in columns})
    for table, (physical, columns) in out.items():
        if len(set(columns.values())) != len(columns):
            raise ValueError(f"{style}: two columns of {table} share a name: {columns}")
    if len({physical for physical, _ in out.values()}) != len(out):
        raise ValueError(f"{style}: two tables share a name")
    return out
