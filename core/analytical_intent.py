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
    r"last\s+(?:week|month|quarter|year)|year\s+to\s+date|ytd|mtd|qtd|"
    r"latest(?:\s+available)?(?:\s+(?:day|date|period))?)\b",
    re.I,
)
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
_OUTPUT_RE = re.compile(
    r"\b(pie\s+chart|bar\s+chart|line\s+chart|area\s+chart|chart|graph|"
    r"table|kpi|dashboard)\b",
    re.I,
)
_DIMENSION_RE = re.compile(
    r"\b(?:by|per|across|grouped\s+by|split\s+by|for\s+each)\s+"
    r"([a-z][a-z0-9 _-]{1,45}?)(?=\s+(?:for|in|during|where|with|from|"
    r"today|yesterday|this|last|versus|vs\.?|as\s+a)\b|[?.!,]|$)",
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
    date_role: str = "unresolved"
    comparison: str = ""
    entity_grain: str = ""
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
            "date_role": self.date_role,
            "comparison": self.comparison,
            "entity_grain": self.entity_grain,
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


def _latest_clarification(question: str) -> str:
    matches = _CLARIFICATION_RE.findall(question or "")
    return matches[-1].strip() if matches else ""


def plan_analytical_intent(
    question: str,
    *,
    metrics: Iterable[dict[str, Any]] = (),
    terms: Iterable[dict[str, Any]] = (),
) -> AnalyticalPlan:
    """Build a structured plan from the question and governed catalog names."""
    text = str(question or "").strip()
    metrics = tuple(metrics or ())
    terms = tuple(terms or ())
    clarified = _latest_clarification(text)
    metric_matches = _matched_catalog_names(text, metrics)
    concept_match = _BUSINESS_CONCEPT_RE.search(text)
    business_concepts = [concept_match.group(1).lower()] if concept_match else []
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

    dimensions = tuple(
        dict.fromkeys(match.group(1).strip(" -") for match in _DIMENSION_RE.finditer(text))
    )
    metric_words = {value.casefold() for value in metric_matches}
    dimensions = tuple(
        value for value in dimensions if value.casefold() not in metric_words
    )
    time_match = _TIME_RE.search(text)
    time_range = time_match.group(1).lower() if time_match else ""
    output_match = _OUTPUT_RE.search(text)
    output = output_match.group(1).lower() if output_match else "auto"
    top_match = _TOP_N_RE.search(text)
    top_n = int(top_match.group(1)) if top_match else None
    entity_match = re.search(
        r"\b(customers?|people|persons?|employees?|suppliers?|items?|products?|"
        r"warehouses?|orders?|invoices?)\b",
        text,
        re.I,
    )
    entity_grain = entity_match.group(1).lower() if entity_match else (dimensions[0] if dimensions else "")
    if intent == "ranking" and entity_grain and not dimensions:
        dimensions = (entity_grain,)
    comparison = "period_or_segment_comparison" if _COMPARISON_RE.search(text) else ""
    assumptions: list[str] = []
    if time_range in {"today", "today's", "todays", "today’s"}:
        assumptions.append("Use the latest complete governed business date; disclose if it differs from wall-clock today.")
    if clarified:
        assumptions.append(f"User clarification: {clarified[:180]}")

    unresolved: list[str] = []
    clarification: ClarificationRequest | None = None
    if intent in {"daily_snapshot", "data_overview"} and not metric_matches:
        if clarified:
            metric_matches = [clarified]
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
            options=_subject_options(metrics),
            reason="A ranking requires a governed measure.",
        )

    if business_concepts:
        is_defined = bool(known_concepts or metric_matches)
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
        date_role="governed_default" if time_range else "unresolved",
        comparison=comparison,
        entity_grain=entity_grain,
        output=output,
        top_n=top_n,
        assumptions=tuple(assumptions),
        unresolved_slots=tuple(unresolved),
        confidence=confidence,
        clarification=clarification,
    )
