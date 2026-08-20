"""Deterministic analytical-intent planning for governed data questions.

The planner is deliberately metadata-only.  It never sees result rows and it
does not generate SQL.  Its job is to turn conversational language into a
small, auditable set of analytical slots that the semantic and SQL layers can
resolve against governed workspace metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable


_TIME_RE = re.compile(
    r"\b(today|today'?s|yesterday|this\s+(?:week|month|quarter|year)|"
    r"last\s+(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
    r"(?:observed\s+(?:business\s+)?)?"
    r"(?:days?|weeks?|months?|quarters?|years?)|year\s+to\s+date|ytd|mtd|qtd|"
    r"q[1-4](?:\s+(?:19|20)\d{2})?|"
    r"(?:first|second|third|fourth)\s+quarter(?:\s+(?:19|20)\d{2})?|"
    r"quarter\s+[1-4](?:\s+(?:19|20)\d{2})?|"
    r"latest(?:\s+available)?(?:\s+(?:day|date|period))?)\b",
    re.I,
)
_NAMED_QUARTER_RE = re.compile(
    r"\b(?:q\s*([1-4])|quarter\s+([1-4])|"
    r"(first|second|third|fourth)\s+quarter)"
    r"(?:\s+(?:of\s+)?((?:19|20)\d{2}))?\b",
    re.I,
)
_CALENDAR_BASIS_RE = re.compile(
    r"\b(?:calendar\s+(?:year|quarter|q[1-4])|"
    r"use\s+calendar\s+(?:quarters?|periods?)|calendar\s+basis)\b",
    re.I,
)
_FISCAL_BASIS_RE = re.compile(
    r"\b(?:fiscal|financial)\s+(?:year|quarter|q[1-4]|calendar|basis)|"
    r"\b(?:fy\s*\d{2,4}|fq[1-4])\b|"
    r"\buse\s+(?:the\s+)?(?:fiscal|financial)\s+(?:quarters?|calendar|periods?)\b",
    re.I,
)
_FISCAL_START_RE = re.compile(
    r"\b(?:fiscal|financial)\s+year\s+(?:starts?|begins?)\s+(?:in|on)?\s*"
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.I,
)
_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_DAILY_SNAPSHOT_RE = re.compile(
    r"\b(?:today'?s?\s+(?:data|numbers?|snapshot|summary|performance)|"
    r"(?:data|numbers?|snapshot|summary|performance)\s+(?:for\s+)?today|"
    r"daily\s+(?:snapshot|brief|summary)|what(?:'s|\s+is|\s+was)\s+"
    r"(?:my|our|the)?\s*today'?s?\s+(?:data|numbers?|status))\b",
    re.I,
)
_DATA_OVERVIEW_RE = re.compile(
    r"^\s*(?:show|give|get|tell)\s+(?:me\s+)?(?:my|our|the)?\s*"
    r"(?:data|numbers?|overview|summary|snapshot|insights?)\s*[?.!]*$",
    re.I,
)
_COMPARISON_RE = re.compile(
    r"\b(?:compare|versus|vs\.?|against|difference|change\s+from|"
    r"year[- ]over[- ]year|month[- ]over[- ]month|week[- ]over[- ]week)\b",
    re.I,
)
_TREND_RE = re.compile(r"\b(?:trend|over\s+time|time\s+series|movement|trajectory)\b", re.I)
_RANKING_RE = re.compile(r"\b(?:top|bottom|highest|lowest|best|worst|rank)\b", re.I)
_DISTRIBUTION_RE = re.compile(
    r"\b(?:distribution|share|mix|composition|breakdown|split)\b", re.I
)
_CAUSAL_RE = re.compile(
    r"\b(?:why|what\s+(?:caused|drove|contributed)|root\s+cause|drivers?)\b",
    re.I,
)
_ENTITY_LOOKUP_RE = re.compile(
    r"\b(?:who|which\s+(?:customers?|people|persons?|employees?|suppliers?)|"
    r"find|identify|list)\b",
    re.I,
)
_BUSINESS_CONCEPT_RE = re.compile(
    r"\b(churn(?:ed|ing)?|retention|reactivat(?:ed|ion)|new\s+customers?|"
    r"lost\s+customers?|inactive\s+(?:customers?|items?)|stockout|"
    r"slow[- ]moving|overdue|attrition)\b",
    re.I,
)
_BUSINESS_EVENT_RE = re.compile(
    r"\b(orders?|invoices?|shipments?|returns?|transactions?|payments?|"
    r"purchases?|receipts?|deliveries?|claims?|prescriptions?|tickets?)\b",
    re.I,
)
_EVENT_COUNT_CUE_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|count(?:\s+of)?|total(?:\s+number\s+of)?)\b|"
    r"\b(?:most|fewest|highest|lowest)\s+(?:number\s+of\s+)?"
    r"(?:(?:customer|sales|purchase)\s+)?"
    r"(?:orders?|invoices?|shipments?|returns?|transactions?|payments?|purchases?|"
    r"receipts?|deliveries?|claims?|prescriptions?|tickets?)\b|"
    r"\b(?:fewer|more|reduced|decreased|declined|increased|grew|grown|growth\s+in)\s+"
    r"(?:orders?|invoices?|shipments?|returns?|transactions?|payments?|purchases?|"
    r"receipts?|deliveries?|claims?|prescriptions?|tickets?)\b|"
    r"\b(?:by|based\s+on)\s+(?:orders?|invoices?|shipments?|returns?|transactions?|"
    r"payments?|purchases?|receipts?|deliveries?|claims?|prescriptions?|tickets?)\b",
    re.I,
)
_EVENT_VALUE_CUE_RE = re.compile(
    r"\b(?:value|amount|revenue|sales|quantity|units?|cost|margin|discount|price|"
    r"duration|days?)\b",
    re.I,
)
# What a question asks to count is the SUBJECT of the count phrase, not any
# countable noun that happens to appear later in the sentence. "How many
# customers placed orders in June" asks about customers; reading the first
# event noun anywhere in the text answered a question about order volume.
_COUNT_SUBJECT_CUE_RE = re.compile(
    r"\b(?:how\s+many|number\s+of|counts?\s+of|total\s+number\s+of)\s+", re.I
)
_COUNT_SUBJECT_SKIP = {
    "active", "different", "distinct", "individual", "separate", "total", "unique",
}
_COUNT_SUBJECT_STOP = {
    "a", "an", "and", "are", "at", "based", "by", "did", "do", "does", "each",
    "for", "from", "had", "has", "have", "i", "in", "is", "its", "of", "on", "or",
    "our", "per", "that", "the", "their", "there", "these", "they", "this",
    "those", "to", "was", "we", "were", "which", "who", "whose", "with",
    "without", "you",
}
# Counting these is a calculation or a question about the product, not a count
# of a business population, so they must never resolve to a governed identifier.
_NON_ENTITY_COUNT_SUBJECT = {
    "column", "dataset", "date", "day", "field", "hour", "minute", "month",
    "percent", "percentage", "quarter", "query", "question", "record", "report",
    "result", "row", "second", "table", "time", "week", "year",
}
_PLURAL_EXCEPTIONS = {
    "children": "child", "men": "man", "people": "person", "women": "woman",
}
# A count subject ends where a predicate about it begins. The -ed/-ing test
# covers regular English; the rest is a closed class of irregular past tenses,
# not business vocabulary, so it does not need to grow per workspace. A word
# wrongly treated as a verb only shortens the subject, which at worst leaves
# the reading unchanged.
_IRREGULAR_VERBS = {
    "bought", "came", "did", "gave", "got", "held", "kept", "lost", "made",
    "paid", "ran", "saw", "sent", "sold", "spent", "took", "went", "won",
}
# A population question asks how large a business population is and nothing
# else. The shape is matched against the WHOLE question deliberately: any
# surviving predicate ("how many drugs are in stock", "how many customers
# ordered last month") narrows the population to activity, which the master
# table alone cannot answer, so those must not match and must keep their
# existing treatment.
_POPULATION_COUNT_RE = re.compile(
    r"^\W*(?:so\s+|and\s+|also\s+)?"
    r"(?:what(?:'s|\s+is)\s+the\s+(?:total\s+)?number\s+of|"
    r"how\s+many|(?:total\s+)?number\s+of|counts?\s+of)\s+"
    r"(?:(?:active|different|distinct|individual|separate|total|unique)\s+)*"
    r"(?P<entity>[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2}?)"
    r"(?:\s+(?:do|does|did)\s+(?:we|you|i|they)\s+(?:currently\s+)?(?:have|hold))?"
    r"(?:\s+(?:are|is)\s+there)?"
    r"(?:\s+(?:exist|exists))?"
    r"(?:\s+in\s+(?:total|the\s+(?:system|database|data|business|company)|"
    r"our\s+(?:system|database|data)))?"
    r"(?:\s+(?:overall|altogether|currently))?"
    r"(?:\s+(?:by|per|for\s+each|grouped\s+by|split\s+by|across)\s+"
    r"[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,3})?"
    r"\W*$",
    re.I,
)
_ENTITY_GRAIN_RE = re.compile(
    r"\b(?:by|per|for\s+each)\s+(?:each\s+)?"
    r"(customers?|people|persons?|employees?|suppliers?|items?|products?|"
    r"warehouses?|orders?|invoices?)\b",
    re.I,
)
_VAGUE_RECENT_EVENT_CHANGE_RE = re.compile(
    r"\b(?:recently|recent)\b.*\b(?:orders?|invoices?|shipments?|returns?|transactions?|"
    r"payments?|purchases?|receipts?|deliveries?|claims?|prescriptions?|tickets?)\b|"
    r"\b(?:orders?|invoices?|shipments?|returns?|transactions?|payments?|purchases?|"
    r"receipts?|deliveries?|claims?|prescriptions?|tickets?)\b.*\b(?:recently|recent)\b",
    re.I,
)
_OUTPUT_RE = re.compile(
    r"\b(pie\s+chart|bar\s+chart|line\s+chart|area\s+chart|chart|graph|"
    r"table|kpi|dashboard)\b",
    re.I,
)
_DIMENSION_RE = re.compile(
    r"\b(?:by|per|across|grouped\s+by|split\s+by|for\s+each)\s+"
    r"([a-z][a-z0-9 _-]{1,45}?)(?=\s+(?:for|in|during|where|with|from|"
    r"on|today|yesterday|this|last|latest|versus|vs\.?|as\s+a)\b|[?.!,]|$)",
    re.I,
)
_TOP_N_RE = re.compile(r"\b(?:top|bottom)\s+(\d{1,3})\b", re.I)
_CLARIFICATION_RE = re.compile(
    r"Clarification for the same request:\s*(.+?)\.\s*\n",
    re.I | re.S,
)


@dataclass(frozen=True)
class ClarificationRequest:
    slot: str
    question: str
    options: tuple[dict[str, str], ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "question": self.question,
            "options": [dict(option) for option in self.options],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnalyticalPlan:
    intent: str
    metrics: tuple[str, ...] = ()
    business_concepts: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    time_range: str = ""
    quarter_periods: tuple[str, ...] = ()
    date_role: str = "unresolved"
    calendar_basis: str = "unresolved"
    fiscal_year_start_month: int | None = None
    calendar_basis_source: str = ""
    comparison: str = ""
    entity_grain: str = ""
    measure_semantics: str = ""
    counted_entity: str = ""
    population_entity: str = ""
    output: str = "auto"
    top_n: int | None = None
    assumptions: tuple[str, ...] = ()
    unresolved_slots: tuple[str, ...] = ()
    confidence: float = 0.0
    clarification: ClarificationRequest | None = None

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "metrics": list(self.metrics),
            "business_concepts": list(self.business_concepts),
            "dimensions": list(self.dimensions),
            "filters": list(self.filters),
            "time_range": self.time_range,
            "quarter_periods": list(self.quarter_periods),
            "date_role": self.date_role,
            "calendar_basis": self.calendar_basis,
            "fiscal_year_start_month": self.fiscal_year_start_month,
            "calendar_basis_source": self.calendar_basis_source,
            "comparison": self.comparison,
            "entity_grain": self.entity_grain,
            "measure_semantics": self.measure_semantics,
            "counted_entity": self.counted_entity,
            "population_entity": self.population_entity,
            "output": self.output,
            "top_n": self.top_n,
            "assumptions": list(self.assumptions),
            "unresolved_slots": list(self.unresolved_slots),
            "confidence": round(self.confidence, 3),
            "clarification": self.clarification.to_dict() if self.clarification else None,
        }

    def prompt_context(self) -> str:
        """Return a compact control block for the governed SQL prompt."""
        payload = self.to_dict()
        payload.pop("clarification", None)
        return (
            "ANALYTICAL INTENT PLAN (deterministic, metadata-only):\n"
            + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            + "\nUse this plan to preserve the requested grain, time context, "
            "comparison, and output. Resolve names only from governed semantic "
            "metadata. Never invent a metric, business definition, field, or join."
        )


def _catalog_phrases(item: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("name", "term", "category", "synonyms", "aliases"):
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(str(value) for value in raw)
        elif raw:
            text = str(raw)
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                values.extend(str(value) for value in parsed)
            else:
                values.extend(re.split(r"[,;|]", text))
    return [value.strip() for value in values if value and value.strip()]


def _matched_catalog_names(question: str, catalog: Iterable[dict[str, Any]]) -> list[str]:
    lowered = question.casefold()
    matches: list[str] = []
    for item in catalog:
        canonical = str(item.get("name") or item.get("term") or "").strip()
        if not canonical:
            continue
        phrases = _catalog_phrases(item) or [canonical]
        if any(
            re.search(r"\b" + re.escape(phrase.casefold()) + r"\b", lowered)
            for phrase in phrases if len(phrase.strip()) >= 2
        ):
            if canonical not in matches:
                matches.append(canonical)
    return matches


def _subject_options(metrics: Iterable[dict[str, Any]]) -> tuple[dict[str, str], ...]:
    metrics = list(metrics)
    categories: list[str] = []
    for metric in metrics:
        category = str(metric.get("category") or "").strip()
        if category and category.casefold() not in {c.casefold() for c in categories}:
            categories.append(category)
    labels = categories[:4]
    if len(labels) < 2:
        labels = []
        for metric in metrics:
            name = str(metric.get("name") or "").strip()
            if name and name.casefold() not in {label.casefold() for label in labels}:
                labels.append(name)
            if len(labels) >= 4:
                break
    return tuple(
        {"id": f"subject-{index}", "label": label, "value": label}
        for index, label in enumerate(labels, start=1)
    )


def _normalised_terms(text: str) -> set[str]:
    """Small lexical set for relevance filtering; never resolves schema names."""
    stop = {
        "and", "are", "best", "bottom", "by", "each", "for", "from", "highest",
        "in", "last", "lowest", "me", "month", "months", "of", "per", "rank",
        "show", "the", "this", "top", "total", "what", "which", "year", "years",
    }
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(text or "").casefold()):
        if len(token) < 2 or token in stop or token.isdigit():
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        terms.add(token)
    return terms


def _relevant_metric_options(
    question: str,
    metrics: Iterable[dict[str, Any]],
) -> tuple[dict[str, str], ...]:
    """Offer only metrics with evidence in the user's words.

    The old fallback returned the first four account metrics. That made a
    request for order count offer unrelated measures such as COGS or allocated
    cost. Empty options are intentional: free text is safer than a fabricated
    shortlist when the catalog has no compatible measure.
    """
    question_terms = _normalised_terms(question)
    ranked: list[tuple[int, str]] = []
    for metric in metrics or ():
        label = str(metric.get("name") or "").strip()
        if not label:
            continue
        phrases = " ".join(_catalog_phrases(metric))
        overlap = question_terms & _normalised_terms(phrases)
        # One generic shared word (for example "customer" in both an order
        # question and "Customer Discount Amount") is not enough evidence to
        # offer a governed measure. Exact metric-name matches are handled by
        # the primary matcher; this fallback is deliberately conservative.
        if len(overlap) >= 2:
            ranked.append((len(overlap), label))
    ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    labels = list(dict.fromkeys(label for _, label in ranked))[:4]
    return tuple(
        {"id": f"metric-{index}", "label": label, "value": label}
        for index, label in enumerate(labels, start=1)
    )


def _unambiguous_relevant_metric(
    question: str,
    metrics: Iterable[dict[str, Any]],
) -> str:
    """Return a metric only when the user's own words separate it clearly.

    Snapshot questions commonly omit the full governed metric name while still
    naming both its subject and cadence (for example, ``daily inventory
    value``).  Requiring an exact catalog phrase made the agent ask users to
    choose among Daily, Monthly, and ERP inventory measures even though the
    requested grain already disambiguated them.  This metadata-only scorer is
    deliberately conservative: a candidate must overlap at least two
    meaningful terms and beat the runner-up, so generic requests such as
    ``today's data`` still ask for a subject.
    """
    question_terms = _normalised_terms(question)
    ranked: list[tuple[int, str]] = []
    for metric in metrics or ():
        label = str(metric.get("name") or "").strip()
        if not label:
            continue
        phrases = " ".join(_catalog_phrases(metric))
        overlap = question_terms & _normalised_terms(phrases)
        if overlap:
            ranked.append((len(overlap), label))
    ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    if not ranked or ranked[0][0] < 2:
        return ""
    runner_up = ranked[1][0] if len(ranked) > 1 else 0
    return ranked[0][1] if ranked[0][0] > runner_up else ""


def _singular(word: str) -> str:
    lowered = str(word or "").casefold()
    if lowered in _PLURAL_EXCEPTIONS:
        return _PLURAL_EXCEPTIONS[lowered]
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("ches", "shes", "sses", "xes", "zes")):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def _is_plural(word: str) -> bool:
    lowered = str(word or "").casefold()
    return lowered in _PLURAL_EXCEPTIONS or (
        lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3
    )


def _looks_like_verb(word: str) -> bool:
    lowered = str(word or "").casefold()
    if lowered in _IRREGULAR_VERBS:
        return True
    return len(lowered) > 4 and lowered.endswith(("ed", "ing"))


def _count_subject(question: str) -> str:
    """Return the noun phrase the count phrase is asking about, or ``""``."""
    text = str(question or "")
    match = _COUNT_SUBJECT_CUE_RE.search(text)
    if not match:
        return ""
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z][\w-]*", text[match.end():]):
        word = raw.casefold()
        if not words and word in _COUNT_SUBJECT_SKIP:
            continue
        if word in _COUNT_SUBJECT_STOP or _looks_like_verb(word):
            break
        words.append(word)
        if len(words) >= 3:
            break
    if not words:
        return ""
    subject = " ".join(words[:-1] + [_singular(words[-1])])
    return "" if subject in _NON_ENTITY_COUNT_SUBJECT else subject


def detect_business_event_count(question: str) -> str:
    """Return the requested countable event, or ``""`` when it asks for value.

    This is schema-independent language interpretation. The physical business
    identifier is resolved later from governed semantic metadata; this helper
    never guesses a column or table.
    """
    text = str(question or "")
    match = _BUSINESS_EVENT_RE.search(text)
    if not match or _EVENT_VALUE_CUE_RE.search(text):
        return ""
    if not _EVENT_COUNT_CUE_RE.search(text):
        return ""
    raw = match.group(1).casefold()
    irregular = {"deliveries": "delivery", "purchases": "purchase"}
    event = irregular.get(raw, raw[:-1] if raw.endswith("s") else raw)
    # An explicit count phrase names its own subject. When that subject is a
    # different business thing, the event noun is a predicate about it, not
    # what the user asked to count.
    subject = _count_subject(text)
    if subject and event not in {subject, subject.split()[-1]}:
        return ""
    return event


def detect_population_count(question: str) -> str:
    """Return the entity whose whole population is being asked for, or ``""``.

    This is the "how many customers do we have" family: a request for the size
    of a business population, with no period, measure or activity narrowing it.
    Such a count belongs to the entity's own master table — answering it from
    whichever fact happened to win source arbitration silently drops every
    member with no activity on that fact.
    """
    text = str(question or "").strip()
    if _EVENT_VALUE_CUE_RE.search(text):
        return ""
    match = _POPULATION_COUNT_RE.match(text)
    if not match:
        return ""
    words = [
        word.casefold()
        for word in re.findall(r"[A-Za-z][\w-]*", match.group("entity"))
    ]
    # "How many" takes a plural. Requiring one keeps a trailing predicate out
    # of the entity: in "how many suppliers delivered late" the only way the
    # shape matches at all is by swallowing "delivered late" as part of the
    # name, and "late" is not a population.
    if not words or not _is_plural(words[-1]):
        return ""
    entity = " ".join(words[:-1] + [_singular(words[-1])])
    return "" if entity in _NON_ENTITY_COUNT_SUBJECT else entity


def _latest_clarification(question: str) -> str:
    matches = _CLARIFICATION_RE.findall(question or "")
    return matches[-1].strip() if matches else ""


def _normalise_calendar_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Return the small, safe calendar subset accepted by the planner."""
    raw = profile if isinstance(profile, dict) else {}
    basis = str(
        raw.get("basis")
        or raw.get("calendar_basis")
        or raw.get("calendar_mode")
        or ""
    ).strip().casefold()
    if basis in {"financial", "fiscal_year", "financial_year"}:
        basis = "fiscal"
    if basis not in {"calendar", "fiscal"}:
        basis = ""
    try:
        start_month = int(raw.get("fiscal_year_start_month") or 0)
    except (TypeError, ValueError):
        start_month = 0
    if not 1 <= start_month <= 12:
        start_month = 0
    return {
        "basis": basis,
        "fiscal_year_start_month": start_month or None,
        "source": str(raw.get("source") or "").strip(),
    }


def _named_quarter(text: str) -> str:
    match = _NAMED_QUARTER_RE.search(text or "")
    if not match:
        return ""
    word_numbers = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
    number = match.group(1) or match.group(2) or word_numbers.get(
        str(match.group(3) or "").casefold(), ""
    )
    year = str(match.group(4) or "")
    return f"Q{number}{' ' + year if year else ''}"


def _named_quarters(text: str) -> tuple[str, ...]:
    """Return all requested business quarters in analytical order.

    "Q1 to Q3" means an inclusive trend (Q1, Q2, Q3), while "Q1 and
    Q3" means two explicitly selected periods. Calendar/fiscal interpretation
    is resolved separately; this function only preserves the requested shape.
    """
    found: list[str] = []
    word_numbers = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
    for match in _NAMED_QUARTER_RE.finditer(text or ""):
        number = match.group(1) or match.group(2) or word_numbers.get(
            str(match.group(3) or "").casefold(), ""
        )
        year = str(match.group(4) or "")
        label = f"Q{number}{' ' + year if year else ''}"
        if number and label not in found:
            found.append(label)
    if len(found) == 2 and re.search(
        r"\b(?:from\s+)?q\s*[1-4]\s+(?:to|through|thru|until|-)+\s+q\s*[1-4]\b",
        text or "",
        re.I,
    ):
        start = int(re.search(r"[1-4]", found[0]).group(0))
        end = int(re.search(r"[1-4]", found[1]).group(0))
        if start <= end:
            year = found[0].split(" ", 1)[1] if " " in found[0] else ""
            return tuple(f"Q{quarter}{' ' + year if year else ''}" for quarter in range(start, end + 1))
    return tuple(found)


def _fiscal_start_month(text: str) -> int | None:
    match = _FISCAL_START_RE.search(text or "")
    if not match:
        return None
    return _MONTH_NUMBERS.get(match.group(1).casefold())


def _fiscal_month_options() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "id": f"fiscal-start-{month}",
            "label": name.title(),
            "value": f"The fiscal year starts in {name.title()}",
        }
        for name, month in _MONTH_NUMBERS.items()
    )


def plan_analytical_intent(
    question: str,
    *,
    metrics: Iterable[dict[str, Any]] = (),
    terms: Iterable[dict[str, Any]] = (),
    calendar_profile: dict[str, Any] | None = None,
) -> AnalyticalPlan:
    """Build a structured plan from the question and governed catalog names."""
    text = str(question or "").strip()
    metrics = tuple(metrics or ())
    terms = tuple(terms or ())
    clarified = _latest_clarification(text)
    profile = _normalise_calendar_profile(calendar_profile)
    metric_matches = _matched_catalog_names(text, metrics)
    concept_match = _BUSINESS_CONCEPT_RE.search(text)
    business_concepts = [concept_match.group(1).lower()] if concept_match else []
    counted_entity = detect_business_event_count(text)
    # Advisory, not governing: a population entity only becomes the
    # counted entity once the semantic layer resolves it to a master
    # table, so an unresolvable noun costs nothing.
    population_entity = "" if counted_entity else detect_population_count(text)
    if counted_entity:
        business_concepts.append(f"{counted_entity} count")
    known_concepts = _matched_catalog_names(text, terms) if concept_match else []
    for concept in known_concepts:
        if concept not in business_concepts:
            business_concepts.append(concept)

    if _DAILY_SNAPSHOT_RE.search(text):
        intent = "daily_snapshot"
    elif _DATA_OVERVIEW_RE.search(text):
        intent = "data_overview"
    elif _CAUSAL_RE.search(text):
        intent = "causal_analysis"
    elif _COMPARISON_RE.search(text):
        intent = "comparison"
    elif _RANKING_RE.search(text):
        intent = "ranking"
    elif _DISTRIBUTION_RE.search(text):
        intent = "distribution"
    elif _TREND_RE.search(text):
        intent = "trend"
    elif _ENTITY_LOOKUP_RE.search(text) and business_concepts:
        intent = "entity_lookup"
    else:
        intent = "metric_query"

    # A subject/category phrase can match several governed metrics at once
    # (for example, every Inventory metric has category ``Inventory``).  When
    # cadence and measure wording uniquely separate one candidate, narrow the
    # plan before deciding whether a clarification is required.
    if intent in {"daily_snapshot", "data_overview"} and len(metric_matches) > 1:
        inferred_metric = _unambiguous_relevant_metric(text, metrics)
        if inferred_metric:
            metric_matches = [inferred_metric]

    dimensions = tuple(
        dict.fromkeys(match.group(1).strip(" -") for match in _DIMENSION_RE.finditer(text))
    )
    metric_words = {value.casefold() for value in metric_matches}
    dimensions = tuple(
        value for value in dimensions if value.casefold() not in metric_words
    )
    time_match = _TIME_RE.search(text)
    quarter_periods = _named_quarters(text)
    named_quarter = quarter_periods[0] if quarter_periods else _named_quarter(text)
    time_range = named_quarter or (time_match.group(1).lower() if time_match else "")
    output_match = _OUTPUT_RE.search(text)
    output = output_match.group(1).lower() if output_match else "auto"
    top_match = _TOP_N_RE.search(text)
    top_n = int(top_match.group(1)) if top_match else None
    grain_match = _ENTITY_GRAIN_RE.search(text)
    entity_match = re.search(
        r"\b(customers?|people|persons?|employees?|suppliers?|items?|products?|"
        r"warehouses?|orders?|invoices?)\b",
        text,
        re.I,
    )
    entity_grain = (
        grain_match.group(1).lower()
        if grain_match
        else (entity_match.group(1).lower() if entity_match else (dimensions[0] if dimensions else ""))
    )
    entity_grain = {
        "people": "person",
        "persons": "person",
        "deliveries": "delivery",
    }.get(entity_grain, entity_grain[:-1] if entity_grain.endswith("s") else entity_grain)
    if intent == "ranking" and entity_grain and (not dimensions or grain_match):
        dimensions = (entity_grain,)
    comparison = "period_or_segment_comparison" if _COMPARISON_RE.search(text) else ""
    assumptions: list[str] = []
    if time_range in {"today", "today's", "todays", "today’s"}:
        assumptions.append("Use the latest complete governed business date; disclose if it differs from wall-clock today.")
    if clarified:
        assumptions.append(f"User clarification: {clarified[:180]}")

    explicit_fiscal = bool(_FISCAL_BASIS_RE.search(text))
    explicit_calendar = bool(_CALENDAR_BASIS_RE.search(text))
    if explicit_fiscal and not explicit_calendar:
        calendar_basis = "fiscal"
        calendar_basis_source = "question"
    elif explicit_calendar and not explicit_fiscal:
        calendar_basis = "calendar"
        calendar_basis_source = "question"
    else:
        calendar_basis = str(profile.get("basis") or "unresolved")
        calendar_basis_source = str(profile.get("source") or ("profile" if calendar_basis != "unresolved" else ""))
    fiscal_start_month = _fiscal_start_month(text)
    if fiscal_start_month is None and calendar_basis == "fiscal":
        fiscal_start_month = profile.get("fiscal_year_start_month")
    if calendar_basis == "calendar":
        assumptions.append("Interpret named quarters using calendar quarters.")
    elif calendar_basis == "fiscal" and fiscal_start_month:
        month_name = next(
            name.title() for name, month in _MONTH_NUMBERS.items()
            if month == fiscal_start_month
        )
        assumptions.append(f"Interpret named quarters using a fiscal year starting in {month_name}.")

    unresolved: list[str] = []
    clarification: ClarificationRequest | None = None
    if intent in {"daily_snapshot", "data_overview"} and not metric_matches:
        if clarified:
            metric_matches = [clarified]
        else:
            inferred_metric = _unambiguous_relevant_metric(text, metrics)
            if inferred_metric:
                metric_matches = [inferred_metric]
                assumptions.append(
                    f"Resolved the governed measure from explicit subject and cadence wording: {inferred_metric}."
                )
            else:
                unresolved.append("subject")
                options = _subject_options(metrics)
                clarification = ClarificationRequest(
                    slot="subject",
                    question=(
                        "Which business area or metric should I use for this snapshot?"
                        if intent == "daily_snapshot"
                        else "Which business area or metric would you like me to analyse?"
                    ),
                    options=options,
                    reason="The request does not identify a governed measure or subject area.",
                )
    elif intent == "ranking" and not metric_matches and not business_concepts:
        unresolved.append("metric")
        clarification = ClarificationRequest(
            slot="metric",
            question="What measure should I use to rank them?",
            options=_relevant_metric_options(text, metrics),
            reason="A ranking requires a governed measure.",
        )

    if business_concepts:
        is_defined = bool(known_concepts or metric_matches or counted_entity)
        if not is_defined and not clarified and clarification is None:
            unresolved.append("business_definition")
            clarification = ClarificationRequest(
                slot="business_definition",
                question=(
                    f"How should I define '{business_concepts[0]}' for this analysis? "
                    "Please include the business rule or threshold to use."
                ),
                reason="This analytical concept has no matching governed definition.",
            )

    if (
        counted_entity
        and _VAGUE_RECENT_EVENT_CHANGE_RE.search(text)
        and not clarified
        and clarification is None
    ):
        unresolved.append("recent_window")
        clarification = ClarificationRequest(
            slot="recent_window",
            question=(
                "What comparison window should I use for 'recently'?"
            ),
            options=(
                {
                    "id": "recent-7-days",
                    "label": "Last 7 vs previous 7 days",
                    "value": "Compare the last 7 observed business days with the previous 7 observed business days.",
                },
                {
                    "id": "recent-30-days",
                    "label": "Last 30 vs previous 30 days",
                    "value": "Compare the last 30 observed business days with the previous 30 observed business days.",
                },
                {
                    "id": "recent-90-days",
                    "label": "Last 90 vs previous 90 days",
                    "value": "Compare the last 90 observed business days with the previous 90 observed business days.",
                },
            ),
            reason="A decrease requires both a recent period and a comparable baseline.",
        )

    # A bare Q1/Q2/Q3/Q4 is not universally a calendar quarter. Do not make
    # that business decision in SQL generation. Configured or thread-level
    # metadata may resolve it; otherwise ask before retrieval and execution.
    if named_quarter and calendar_basis == "unresolved" and clarification is None:
        unresolved.append("calendar_basis")
        clarification = ClarificationRequest(
            slot="calendar_basis",
            question=(
                f"Should I interpret {named_quarter} using calendar quarters "
                "or your fiscal quarters?"
            ),
            options=(
                {
                    "id": "calendar-basis-calendar",
                    "label": "Calendar quarters",
                    "value": "Use calendar quarters for this request",
                },
                {
                    "id": "calendar-basis-fiscal",
                    "label": "Fiscal quarters",
                    "value": "Use fiscal quarters for this request",
                },
            ),
            reason="The workspace has no approved calendar basis for a bare quarter reference.",
        )
    elif (
        calendar_basis == "fiscal"
        and (named_quarter or explicit_fiscal)
        and not fiscal_start_month
        and clarification is None
    ):
        unresolved.append("fiscal_year_start_month")
        clarification = ClarificationRequest(
            slot="fiscal_year_start_month",
            question="Which month does your fiscal year start?",
            options=_fiscal_month_options(),
            reason="Fiscal Q1 cannot be calculated safely without the fiscal year start month.",
        )

    signals = sum(
        bool(value)
        for value in (metric_matches, business_concepts, dimensions, time_range, output_match)
    )
    confidence = min(0.98, 0.52 + (signals * 0.09))
    if clarification:
        confidence = min(confidence, 0.69)

    return AnalyticalPlan(
        intent=intent,
        metrics=tuple(metric_matches),
        business_concepts=tuple(business_concepts),
        dimensions=dimensions,
        time_range=time_range,
        quarter_periods=quarter_periods,
        date_role="governed_default" if time_range else "unresolved",
        calendar_basis=calendar_basis,
        fiscal_year_start_month=fiscal_start_month,
        calendar_basis_source=calendar_basis_source,
        comparison=comparison,
        entity_grain=entity_grain,
        measure_semantics="count_distinct_business_identifier" if counted_entity else "",
        counted_entity=counted_entity,
        population_entity=population_entity,
        output=output,
        top_n=top_n,
        assumptions=tuple(assumptions),
        unresolved_slots=tuple(unresolved),
        confidence=confidence,
        clarification=clarification,
    )
