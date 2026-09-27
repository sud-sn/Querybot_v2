"""
One form for each inflection of the verbs a business measure is named by.

"How many units did we sell" and a metric called "Units sold" shared no word
but "units" until "sell" and "sold" were the same word, so the question matched
no metric and was asked which dataset it meant. Only verbs are folded, with
their own nouns where the noun is the same event ("orders", "deliveries"),
"buy" with "purchase", which name one event, and "stock" with "inventory",
which name one thing: a general stemmer would make "sales" (revenue) the same
word as "sold" (units).
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
    # The goods a business holds, called either. French "stock" is read as
    # "inventory", and a metric named "reserved stock" shared only "reserved"
    # with "reserved inventory", as every inventory metric shared "inventory".
    "inventory": ("inventories", "stock", "stocks"),
}.items():
    _FORMS[_base] = _base
    for _form in _forms:
        _FORMS[_form] = _base


def base_form(word: str) -> str:
    """The verb a word is a form of, or the word itself."""
    return _FORMS.get(word, word)


def forms_of(word: str) -> tuple[str, ...]:
    """Every form of the word a word is a form of, itself among them:
    "stock" is "inventory", "inventories", "stock" and "stocks"."""
    base = base_form(word)
    return tuple(sorted({word, base, *(form for form, of in _FORMS.items() if of == base)}))


# How the answer is cut and over what, never which measure: "deliveries and
# physical inventory counts by month in 2022" scored "month-end inventory
# value" on the word "month", and was answered with it. The French is read as
# the reader typed it, which the scorers read beside the canonical English:
# "par mois", "tous les mois", "les 3 derniers mois", "le mois dernier",
# "l'année en cours", "ce mois-ci", "le mois le plus récent". "Fin de mois"
# is not one of them: it names the month-end measures.
_FR_PERIOD = r"(?:jours?|semaines?|mois|trimestres?|semestres?|ans?|ann[ée]es?)"
_FR_WHICH = r"(?:derni[eè]re?s?|pr[ée]c[ée]dente?s?|pass[ée]e?s?|prochaine?s?)"
_GRAIN_OR_WINDOW = re.compile(
    r"\b(?:by|per|each|every|a)\s+(?:day|week|month|quarter|year)s?\b"
    r"|\b(?:the\s+)?(?:last|past|previous|prior|next|this|current)\s+(?:\d+\s+)?"
    r"(?:days?|weeks?|months?|quarters?|years?)\b"
    r"|\b(?:the\s+)?(?:most\s+recent|latest|newest)\s+(?:day|week|month|quarter|year)\b"
    rf"|\b(?:par|chaque)\s+{_FR_PERIOD}\b"
    rf"|\b(?:tous|toutes)\s+les\s+{_FR_PERIOD}\b"
    rf"|\b(?:\d+\s+)?{_FR_WHICH}\s+(?:\d+\s+)?{_FR_PERIOD}\b"
    rf"|\b{_FR_PERIOD}\s+(?:{_FR_WHICH}|en\s+cours|(?:le|la)\s+plus\s+r[ée]cente?)\b"
    rf"|\b(?:ce|cet|cette)\s+{_FR_PERIOD}(?:-ci)?\b",
    re.I,
)


# A count of goods is said in units or in quantity: "quantity sold" is "units
# sold", "units ordered" the "quantity ordered". Only the plural is a count:
# a unit price is no quantity.
_QUANTITY_WORDS = re.compile(r"\b(?:units|quantities|qty|qtys)\b", re.I)


def with_one_quantity_word(text: str) -> str:
    """The text with every word for a count of goods said as "quantity"."""
    return _QUANTITY_WORDS.sub("quantity", text or "")


def without_grain_or_window(question: str) -> str:
    """The question less the words that cut it by a period or bound it to one."""
    return _GRAIN_OR_WINDOW.sub(" ", question or "")


# French function words carry no meaning of a measure: "valeur de notre stock"
# scored "valeur du stock en fin de mois" above "valeur du stock" on the word
# "de", and was answered with month-end value.
FRENCH_FUNCTION_WORDS = frozenset({
    "de", "du", "des", "la", "le", "les", "un", "une", "en", "et", "au", "aux",
    "est", "sont", "quel", "quelle", "quels", "quelles", "notre", "nos", "votre",
    "vos", "leur", "leurs", "par", "pour", "sur", "dans", "avec", "ce", "cet",
    "cette", "ces", "qui", "que", "combien", "nous", "vous", "avons", "avez",
})


# French words for a value, an amount, a quantity, a number or a total say no
# more about WHICH measure than their English twins in
# core.source_resolution.GENERIC_MEASURE_WORDS: "quantité reçue par entrepôt"
# shared only "quantité" with five stock metrics and put all five in scope,
# where "received quantity by warehouse" matched none. Spelled as the scorers'
# tokenisers leave them, which keep [a-z0-9] only: "quantité" is "quantit".
FRENCH_GENERIC_MEASURE_WORDS = frozenset({
    "valeur", "valeurs", "montant", "montants", "quantit", "quantite", "quantites",
    "nombre", "nombres", "totale", "totales", "totaux",
})
