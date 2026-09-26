"""
One form for each inflection of the verbs a business measure is named by.

"How many units did we sell" and a metric called "Units sold" shared no word
but "units" until "sell" and "sold" were the same word, so the question matched
no metric and was asked which dataset it meant. Only verbs are folded, with
their own nouns where the noun is the same event ("orders", "deliveries"), and
"buy" with "purchase", which name one event: a general stemmer would make
"sales" (revenue) the same word as "sold" (units).
"""

from __future__ import annotations

import re

_FORMS: dict[str, str] = {}
for _base, _forms in {
    "sell": ("sells", "selling", "sold"),
    "purchase": ("purchases", "purchasing", "purchased", "buy", "buys", "buying", "bought"),
    "receive": ("receives", "receiving", "received"),
    "ship": ("ships", "shipping", "shipped", "shipment", "shipments"),
    "deliver": ("delivers", "delivering", "delivered", "delivery", "deliveries"),
    "return": ("returns", "returning", "returned"),
    "order": ("orders", "ordering", "ordered"),
    "allocate": ("allocates", "allocating", "allocated", "allocation", "allocations"),
    "consume": ("consumes", "consuming", "consumed", "consumption"),
    "produce": ("produces", "producing", "produced", "production"),
    "issue": ("issues", "issuing", "issued"),
    "transfer": ("transfers", "transferring", "transferred"),
    "invoice": ("invoices", "invoicing", "invoiced"),
    "spend": ("spends", "spending", "spent"),
}.items():
    _FORMS[_base] = _base
    for _form in _forms:
        _FORMS[_form] = _base


def base_form(word: str) -> str:
    """The verb a word is a form of, or the word itself."""
    return _FORMS.get(word, word)


# How the answer is cut and over what, never which measure: "deliveries and
# physical inventory counts by month in 2022" scored "month-end inventory
# value" on the word "month", and was answered with it.
_GRAIN_OR_WINDOW = re.compile(
    r"\b(?:by|per|each|every|a)\s+(?:day|week|month|quarter|year)s?\b"
    r"|\b(?:the\s+)?(?:last|past|previous|prior|next|this|current)\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|quarters?|years?)\b",
    re.I,
)


def without_grain_or_window(question: str) -> str:
    """The question less the words that cut it by a period or bound it to one."""
    return _GRAIN_OR_WINDOW.sub(" ", question or "")
