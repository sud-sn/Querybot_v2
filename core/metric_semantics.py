from __future__ import annotations

import re
from typing import Any


_SEMANTIC_FAMILIES: dict[str, dict[str, Any]] = {
    "attendance": {
        "synonyms": [
            "attendance", "absence", "absent", "absenteeism", "late",
            "lateness", "leave", "present",
        ],
        "business_meaning": "Measures workforce presence, attendance state, or absence-related activity.",
        "why_it_matters": "This can affect staffing capacity, schedule reliability, and workforce continuity.",
        "safe_next_steps": ["compare by department", "compare by manager", "compare by month"],
    },
    "attrition": {
        "synonyms": ["attrition", "resignation", "turnover", "exit", "termination"],
        "business_meaning": "Represents employee exits or workforce loss over a period.",
        "why_it_matters": "This can affect hiring demand, retention planning, and workforce stability.",
        "safe_next_steps": ["compare by department", "compare by month", "compare by tenure band"],
    },
    "nationality_count": {
        "synonyms": ["nationality", "citizenship", "employee nationality"],
        "business_meaning": "Represents the distribution of employees across nationality groups.",
        "why_it_matters": "This may matter for workforce diversity, localization targets, and hiring mix analysis.",
        "safe_next_steps": ["compare top groups", "analyze concentration", "compare by department"],
    },
    "headcount": {
        "synonyms": ["employee", "employees", "headcount", "staff", "workforce"],
        "business_meaning": "Represents the size or distribution of the workforce.",
        "why_it_matters": "This is useful for workforce planning, organization design, and staffing analysis.",
        "safe_next_steps": ["compare by department", "compare by location", "compare over time"],
    },
    "revenue": {
        "synonyms": ["revenue", "sales", "income", "billing", "charges"],
        "business_meaning": "Represents earned commercial value across customers, products, or periods.",
        "why_it_matters": "This helps track business performance, demand mix, and concentration risk.",
        "safe_next_steps": ["compare top segments", "compare by month", "analyze concentration"],
    },
    "count": {
        # "How many", never "total" or "volume": total sales and sales volume
        # are amounts, not a number of records.
        "synonyms": ["count", "number of", "how many", "combien", "nombre de"],
        "business_meaning": "Represents the volume or frequency of records in the selected scope.",
        "why_it_matters": "This helps quantify scale before investigating patterns or segment differences.",
        "safe_next_steps": ["break down by category", "compare top groups", "compare over time"],
    },
}

# What a measure is when no business family names it, read from the measure
# itself (core.analysis_contract.measure_additivity). Every unnamed measure
# used to be "count": inventory value by warehouse reached the analysis
# prompt as "the volume or frequency of records".
_MEASURE_FAMILIES: dict[str, dict[str, Any]] = {
    "level": {
        "business_meaning": (
            "Represents a level held at a point in time, such as stock on hand or a balance: "
            "it adds up across items or locations, not across dates."
        ),
        "why_it_matters": (
            "A level shows what is held now; it is read at its latest snapshot and compared "
            "between snapshots, never summed over them."
        ),
        "safe_next_steps": ["compare by location", "compare by category", "compare with the previous snapshot"],
    },
    "rate": {
        "business_meaning": "Represents a price, rate or average per unit: it is compared, not added up.",
        "why_it_matters": "A change in a rate moves every total built on it.",
        "safe_next_steps": ["compare by category", "compare over time", "look at the highest and lowest"],
    },
    "amount": {
        "business_meaning": "Represents the amount of the measure shown, over the rows in scope.",
        "why_it_matters": "Where the amount is concentrated and how it moves show where to look next.",
        "safe_next_steps": ["compare top groups", "compare over time", "break down by category"],
    },
    # A measure whose kind nothing in its name says -- SOLDE, PRIX_MOYEN, EFFECTIF,
    # INV_VAL, ORDERS -- is described as no kind: an average price is no
    # "amount", and a number of customers no amount of anything.
    "measure": {
        "business_meaning": "Represents the measure shown, over the rows in scope.",
        "why_it_matters": "How it is spread across the groups shown, and how it moves, show where to look next.",
        "safe_next_steps": ["compare top groups", "compare over time", "break down by category"],
    },
}


def _tokenize(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in re.split(r"[^a-z0-9_]+", (part or "").lower()):
            token = token.strip("_ ")
            if token:
                tokens.add(token)
    return tokens


# Modifiers that turn a raw column total into a CALCULATED metric — "open
# order quantity" means ordered MINUS received, not SUM(ordered). Deliberately
# narrow (inventory/fulfillment/finance subtraction words only): broad terms
# like growth/change/margin/rate are handled by dedicated engines or are
# legitimately single columns, and flagging them would be noise.
_DERIVED_MODIFIERS = (
    "open", "net", "outstanding", "remaining", "unfulfilled", "unfilled",
    "backlog", "backordered", "uninvoiced", "unreceived", "unshipped",
    "undelivered", "unpaid", "shortfall", "surplus", "deficit", "buildup",
    "leakage",
)
_METRIC_NOUNS = (
    "quantity", "qty", "amount", "amt", "value", "balance", "revenue",
    "sales", "cost", "units", "volume", "orders", "order", "inventory",
    "stock", "total",
)
_DERIVED_PHRASE_RE = re.compile(
    r"\b(" + "|".join(_DERIVED_MODIFIERS) + r")\s+(?:\w+\s+){0,2}"
    r"(" + "|".join(_METRIC_NOUNS) + r")\b",
    re.IGNORECASE,
)


def detect_derived_metric_gap(
    question: str,
    *,
    has_metric_formula: bool = False,
    has_term_expression: bool = False,
) -> str:
    """Return the derived-metric phrase when the question asks for a
    calculated business metric that has NO approved formula behind it.

    "what is the open order quantity by purchase order date" reads as
    ordered − received, but with no registry metric and no business term
    carrying a canonical_expression, the LLM can only SUM a single raw
    column — a business-wrong answer delivered with full confidence.
    This detector lets the confidence layer say so instead.

    Returns "" when a formula source matched (the pipeline injects the
    approved calculation, so there is no gap) or no derived phrase exists.
    """
    if has_metric_formula or has_term_expression:
        return ""
    match = _DERIVED_PHRASE_RE.search(question or "")
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(0)).strip().lower()


def detect_metric_semantics(
    question: str,
    *,
    context: dict | None = None,
    business_context: str = "",
) -> dict[str, Any]:
    """
    Return a safe semantic description for the metric/question.

    This is intentionally descriptive rather than client-specific. It can be
    shared with the LLM without leaking raw rows or internal rules.

    A business family (revenue, headcount, ...) is chosen by whole words of
    the question and the result's columns: "late" is attendance, "latest" is
    not, and an ORDER_NUMBER column counts nothing. Where none is named, the
    measure's own kind decides (_measure_family), never "count" by default.
    """
    ctx = context or {}
    words = _words(
        question,
        business_context[:500],
        ctx.get("label_col", ""),
        ctx.get("value_col", ""),
        " ".join(ctx.get("numeric_cols") or []),
        " ".join(ctx.get("text_cols") or []),
    )

    best_key = ""
    best_score = 0
    for key, family in _SEMANTIC_FAMILIES.items():
        score = 0
        read: set[tuple[str, ...]] = set()
        for synonym in family.get("synonyms", []):
            phrase = _phrase(synonym)
            # "employee" and "employees" are one word: said once, scored once.
            one_word = tuple(_singular(word) for word in phrase)
            if phrase and one_word not in read and _holds_phrase(words, phrase):
                read.add(one_word)
                score += max(2, len(phrase))
        if score > best_score:
            best_key = key
            best_score = score

    selected = _SEMANTIC_FAMILIES[best_key] if best_key else None
    if selected is None:
        best_key = _measure_family(ctx)
        selected = _SEMANTIC_FAMILIES["count"] if best_key == "count" else _MEASURE_FAMILIES[best_key]
    return {
        "key": best_key,
        "business_meaning": selected["business_meaning"],
        "why_it_matters": selected["why_it_matters"],
        "safe_next_steps": list(selected.get("safe_next_steps", [])),
    }


def _words(*parts: str) -> list[str]:
    """The words of the question and of the result's column names, in order:
    NET_AMOUNT and NetAmount are "net", "amount"."""
    words: list[str] = []
    for part in parts:
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(part or ""))
        words += [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]
    return words


def _phrase(synonym: str) -> list[str]:
    return [word for word in re.split(r"[^a-z0-9]+", synonym.lower()) if word]


def _singular(word: str) -> str:
    """"revenues" is "revenue", "absences" "absence": a word is read without
    the s that makes it plural, as the family words are."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _holds_phrase(words: list[str], phrase: list[str]) -> bool:
    size = len(phrase)
    singular_words = [_singular(word) for word in words]
    singular_phrase = [_singular(word) for word in phrase]
    return any(singular_words[index:index + size] == singular_phrase
               for index in range(len(singular_words) - size + 1))


# The words that say a column counts records or events, in the abbreviations a
# warehouse writes them in: ORDER_COUNT, ORDER_CNT, NB_COMMANDES, NBR_ORDERS.
_COUNT_WORDS = frozenset({"count", "cnt", "nb", "nbr"})


def _measure_family(ctx: dict) -> str:
    """What the result's measure is, by its own kind: a balance is a level, a
    price or an average a rate, a count of events a count, a sum of money or
    of a quantity an amount; a measure whose name says none of them is no kind
    at all."""
    from core.analysis_contract import measure_additivity

    column = str(ctx.get("value_col") or next(iter(ctx.get("numeric_cols") or []), "") or "")
    if not column:
        return "measure"
    aggregation, basis = measure_additivity(column)
    if basis == "event_count" or _COUNT_WORDS & set(_words(column)):
        return "count"
    if aggregation == "semi_additive":
        return "level"
    if aggregation == "non_additive":
        return "rate"
    return "amount" if aggregation == "additive" else "measure"
