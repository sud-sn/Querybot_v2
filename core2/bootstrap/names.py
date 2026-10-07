"""What a name can tell, and no more: tokens, abbreviations, readable words.

Names are evidence, never proof, in core2. These helpers turn a name into tokens
(``CUST_ORD_DT_KEY``, ``CustomerOrderDateKey`` and ``customer_order_date_key``
all become customer/order/date/key), compare tokens allowing for abbreviation,
and make a readable fallback label when no AI label exists. A name that carries
no meaning (``C07``, ``T04``, ``COL_12``) is recognised as such and gives no
evidence at all.
"""

from __future__ import annotations

import re

# Suffix tokens that only say "this is a key".
KEY_SUFFIXES = {"id", "key", "sk", "fk", "pk", "no", "nbr", "num", "code", "cd", "ref", "dms", "dim"}
# Tokens that only say what kind of table this is.
TABLE_AFFIXES = {"dim", "dms", "dimension", "fact", "fct", "fac", "f", "d", "tbl", "t", "vw", "v", "lkp",
                 "lookup", "ref", "mst", "master", "dw", "stg", "bridge", "brg", "snp", "snapshot", "agg"}
_OPAQUE_WORDS = {"c", "col", "column", "field", "fld", "attr", "attribute", "t", "tab", "table", "var", "x", "f", "v"}

# Numbers that are levels (taken at a point in time) and numbers that are flows
# (amounts for a period). On a periodic snapshot a level is taken at period end
# and a flow is summed; names are the evidence for which is which.
LEVEL_WORDS = {"balance", "bal", "hand", "oh", "hnd", "stock", "stk", "inventory", "level", "headcount", "hc", "fte",
               "seats", "mrr", "arr", "outstanding", "backlog", "available", "avl", "allocated", "alc", "reserved",
               "rsv", "position", "open"}
FLOW_WORDS = {"purchase", "purchased", "pch", "pur", "bought",
              "cost", "cst", "salary", "sal", "sales", "sls", "sold", "sld", "sale", "revenue", "rev", "received", "rcv",
              "receipt", "rct", "issued", "iss", "issue", "paid", "budget", "bdgt", "target", "tgt", "plan", "forecast",
              "quota", "hours", "spent", "spend", "movement", "mvt", "shipped", "ship", "shp", "transfer", "tfr",
              "returned", "rtn", "ret", "return", "adjusted", "adj", "consumed", "produced", "scrap", "delivered", "dlv",
              "invoiced"}

# A fallback reading of common abbreviations (weak evidence; AI labels and admins win).
EXPANSIONS = {
    "acct": "account", "acg": "accounting", "act": "actual", "asg": "assignment", "cfm": "confirmed",
    "ann": "annual", "bck": "back", "dmd": "demand", "drc": "direct", "isp": "inspection", "lis": "list",
    "ngv": "negative", "pik": "pick", "psv": "positive", "rjc": "rejected", "rt": "rate", "efc": "effective",
    "fnn": "finance", "ret": "return", "phy": "physical", "acc": "account", "ctr": "centre",
    "chg": "change", "lmt": "limit", "pln": "planned", "pry": "primary", "crd": "credit", "dbt": "debit",
    "addr": "address", "adj": "adjustment", "alc": "allocated", "amt": "amount",
    "avg": "average", "avl": "available", "bal": "balance", "bdgt": "budget", "bil": "billing", "brg": "bridge",
    "cal": "calendar", "cat": "category", "cd": "code", "chnl": "channel", "cls": "closed", "cnl": "cancelled",
    "cnt": "count", "cny": "currency", "cr": "credit", "crn": "current", "cst": "cost", "ctry": "country",
    "cty": "city", "cur": "currency", "cus": "customer", "cust": "customer", "dept": "department", "desc": "description",
    "dlv": "delivered", "dly": "daily", "dr": "debit", "dsc": "discount", "dt": "date", "dvn": "division",
    "emp": "employee", "ent": "entry", "evt": "event", "exp": "expected", "fct": "fact", "flg": "flag",
    "fsc": "fiscal", "fst": "first", "grp": "group", "grs": "gross", "hc": "headcount", "hm": "home",
    "hnd": "hand", "inv": "invoice", "itm": "item", "ivc": "invoice", "jrnl": "journal", "lct": "location",
    "ld": "loaded", "ldg": "ledger", "lin": "line", "ln": "line", "loc": "location", "lst": "last", "lvl": "level",
    "mgn": "margin", "mgr": "manager", "mod": "modified", "mrg": "margin", "msr": "measure", "mth": "month",
    "mvt": "movement", "nm": "name", "no": "number", "nbr": "number", "num": "number", "oh": "on hand",
    "opn": "opened", "ord": "order", "pch": "purchase", "pct": "percent", "pd": "paid", "per": "period",
    "pfm": "performance", "pft": "profit", "pmt": "payment", "prc": "price", "prd": "product",
    "pri": "priority", "prnt": "parent", "prv": "province", "pst": "postal", "pur": "purchase", "qtr": "quarter",
    "qty": "quantity", "rcpt": "receipt", "rct": "receipt", "rcv": "received", "req": "requested",
    "rev": "revenue", "rfd": "refund", "rgn": "region", "rjt": "rejected", "rqs": "requested", "rslv": "resolved",
    "rsn": "reason", "rsv": "reserved", "rtn": "return", "sld": "sold", "tfr": "transfer", "sal": "salary", "seg": "segment", "shp": "ship", "sls": "sales",
    "snp": "snapshot", "sts": "status", "stk": "stock", "str": "store", "strt": "start", "sub": "subscription",
    "sup": "supplier", "supp": "supplier", "tgt": "target", "tkt": "ticket", "ts": "timestamp", "typ": "type",
    "uom": "unit of measure", "upd": "updated", "unt": "unit", "usr": "user", "val": "value", "vch": "voucher",
    "whs": "warehouse", "wh": "warehouse", "wk": "week", "wknd": "weekend", "yr": "year",
}


def tokens(name: str) -> list[str]:
    """``CustOrderDT_KEY2`` -> ``["cust", "order", "dt", "key", "2"]``."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    spaced = re.sub(r"([A-Za-z])(\d)", r"\1 \2", spaced)
    spaced = re.sub(r"(\d)([A-Za-z])", r"\1 \2", spaced)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def opaque(name: str) -> bool:
    """A name that says nothing: ``C07``, ``T04``, ``COL_12``, ``FIELD3``."""
    parts = tokens(name)
    return not parts or all(p.isdigit() or p in _OPAQUE_WORDS for p in parts)


_IRREGULAR = {"statuses": "status", "buses": "bus", "bonuses": "bonus", "campuses": "campus", "people": "person",
              "children": "child", "analyses": "analysis", "data": "data", "series": "series"}


def singular(word: str) -> str:
    """"categories" -> "category", "addresses" -> "address", "warehouses" -> "warehouse", "status" stays."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes", "zzes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def plural(phrase: str) -> str:
    """'order line' -> 'order lines', 'category' -> 'categories', 'address' -> 'addresses'."""
    head, _, word = phrase.rpartition(" ")
    lower = word.lower()
    if not word or lower.endswith(("rows", "data", "staff", "people")):
        out = word
    elif lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        out = word[:-1] + "ies"
    elif lower.endswith(("s", "x", "z", "ch", "sh")):
        out = word + "es"
    else:
        out = word + "s"
    return f"{head} {out}" if head else out


def _skeleton(word: str) -> str:
    return word[0] + re.sub(r"[aeiou]", "", word[1:]) if word else word


def same_word(a: str, b: str) -> bool:
    """Equal, or one an abbreviation of the other (``cust``/``customer``, ``prd``/``product``)."""
    a, b = singular(a), singular(b)
    if a == b:
        return True
    if a.isdigit() or b.isdigit():
        return False
    if EXPANSIONS.get(a) == b or EXPANSIONS.get(b) == a:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 3:
        return False
    return long_.startswith(short) or _skeleton(long_).startswith(_skeleton(short)) and len(short) >= 3


def core_column(name: str) -> list[str]:
    """Column tokens without trailing key words: ``order_date_key`` -> order, date."""
    parts = tokens(name)
    while len(parts) > 1 and parts[-1] in KEY_SUFFIXES:
        parts.pop()
    return parts


def core_table(name: str) -> list[str]:
    """Table tokens without kind affixes, singular: ``DIM_CUSTOMERS`` -> customer."""
    parts = [p for p in tokens(name) if p not in TABLE_AFFIXES and not p.isdigit()]
    return [singular(p) for p in parts] or [singular(p) for p in tokens(name)]


def ends_with(longer: list[str], tail: list[str]) -> bool:
    if not tail or len(tail) > len(longer):
        return False
    return all(same_word(x, y) for x, y in zip(longer[-len(tail):], tail))


_DATE_TOKENS = {"dt", "date", "ts", "timestamp", "day", "time"}
_KEY_ONLY = {"dms", "dim", "key", "sk", "fk"}
_CURRENCY_TOKENS = {"cd", "code", "key", "id", "rate", "rt", "exch", "exchange", "sym", "symbol", "nm", "name",
                    "conv", "conversion", "iso"}


def _expand(token: str, neighbours: list[str], data_type: str | None) -> str:
    """One token in words. A few abbreviations mean different things in different places."""
    if token == "dsc":
        # A text column's DSC is its description; a number's is a discount.
        return "description" if data_type == "text" else EXPANSIONS[token]
    if token == "cls":
        # CLS_DT is when something closed; ABC_CLS is a classification.
        return "closed" if set(neighbours) & _DATE_TOKENS else "class"
    if token == "prd" and (set(neighbours) - {"prd"} <= _KEY_ONLY or set(neighbours) & {"bal", "balance"}):
        # PRD_DMS_KEY and ITM_BAL_PRD_FCT are periods; PRD_GRP_DMS_KEY, PRD_NM are products.
        return "period"
    if token == "dly":
        # CFM_DLY_DT is a delivery date; ITM_BAL_DLY_FCT is a daily balance.
        return "delivery" if set(neighbours) & _DATE_TOKENS else "daily"
    if token == "cur":
        # CUR_CD is a currency; CUR_ON_HND_QTY is the current quantity.
        return "currency" if set(neighbours) & _CURRENCY_TOKENS else "current"
    return EXPANSIONS.get(token, token)


# Written in capitals in a name read out: "GL account", not "Gl account".
ACRONYMS = {"gl", "sku", "abc", "po", "vat", "gst", "hst", "pst", "kpi", "ar", "ap"}


def _word(token: str, parts: list[str], data_type: str | None) -> str:
    return token.upper() if token in ACRONYMS else _expand(token, parts, data_type)


def read_tokens(parts: list[str]) -> str:
    """Tokens in words, each read with its neighbours (as :func:`readable` reads a name)."""
    return " ".join(_word(t, parts, None) for t in parts)


def readable(name: str, data_type: str | None = None) -> str:
    """A fallback label from a name: ``CUST_ORD_DT_KEY`` -> "Customer order date key"."""
    if opaque(name):
        return name
    parts = tokens(name)
    words = [_word(t, parts, data_type) for t in parts]
    text = " ".join(words)
    return text[:1].upper() + text[1:]
