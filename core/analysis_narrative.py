"""Turn computed evidence into sentences — the phrasing half of Phase C.

``core.analysis_evidence`` computes what is true about a result set.
This module says it. The split matters because it is what lets the same
finding be spoken in English or French, with or without a customer's name in
it, by a template or by a model — without anyone re-deriving the arithmetic.

Two phrasers implement one interface:

``TemplatePhraser``
    Catalogue sentences with named placeholders, formatted for the reader's
    language. Deterministic, offline, bilingual today, and reproducible: the
    same evidence yields byte-identical prose every time. This is the default
    for every tenant, not a regulated fallback.

``LLMPhraser``
    Receives the **evidence object only** — never rows, never cells — and
    returns prose. Then the rule nobody else applies: every figure in the
    model's output must appear in the evidence, or its output is discarded and
    the template's is used. A model cannot introduce a number here, because a
    number it introduced is by definition one we did not compute.

The check is the point. "The LLM only phrases what we computed" is a claim
that has to be enforced somewhere, and a paragraph is not evidence of its own
provenance. ``verify_against_evidence`` is where the claim becomes a test.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.analysis_evidence import (
    BELOW_AVERAGE_CLUSTER,
    COMOVEMENT,
    CONCENTRATION_LEADER,
    CONCENTRATION_PARETO,
    CORRELATION,
    LONG_TAIL,
    OUTLIERS,
    RANGE_SPAN,
    SKEW_RIGHT,
    SPREAD_HIGH,
    SPREAD_LOW,
    TREND_DOWN,
    TREND_FLAT,
    TREND_REVERSAL,
    TREND_UP,
    AnalysisEvidence,
    Finding,
)
from core.i18n import format_count, format_decimal, format_percent, t

log = logging.getLogger("querybot.analysis_narrative")

# A "{name}" that survived interpolation.
_UNFILLED_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


# ══════════════════════════════════════════════════════════════════════════════
# Selection
# ══════════════════════════════════════════════════════════════════════════════

# Below this, a finding is arithmetically true and not worth a sentence. The
# floor exists because rank_findings will happily return a 0.02 observation
# when a result has nothing else in it, and "these values range from 4 to 9"
# reads as padding rather than analysis.
MIN_MATERIALITY = 0.15

# More than this and the reader stops reading. Four is the number of distinct
# things a summary can say before it becomes a second table.
DEFAULT_LIMIT = 4


def select_findings(
    evidence: AnalysisEvidence,
    *,
    limit: int = DEFAULT_LIMIT,
    floor: float = MIN_MATERIALITY,
) -> list[Finding]:
    """The findings worth saying out loud, strongest first.

    ``evidence.findings`` is already ranked and de-duplicated; selection is
    the separate decision of how many of them clear the bar. Kept apart from
    ranking so the bar can move without touching the arithmetic.
    """
    return [f for f in evidence.findings if f.materiality >= floor][:limit]


# ══════════════════════════════════════════════════════════════════════════════
# Number formatting for a sentence
# ══════════════════════════════════════════════════════════════════════════════

# Which placeholders in each sentence are percentages, and which are counts of
# rows rather than measured amounts. Everything else is a measured value and
# gets the decimal treatment. Getting this wrong is visible immediately —
# "3.00 of 8.00 rows" — which is why it is a table rather than a heuristic on
# the placeholder name.
_PERCENT_FIELDS = {"pct", "share"}
_COUNT_FIELDS = {"count", "n_total", "n_top", "n_bottom", "n", "turns"}
_RAW_FIELDS = {"r", "r_squared", "cv", "ratio", "multiple"}


def format_number_for(name: str, value: float, lang: str) -> str:
    """One placeholder value, written the way ``lang`` writes it."""
    if name in _PERCENT_FIELDS:
        return format_percent(value, 1 if value % 1 else 0, lang=lang)
    if name in _COUNT_FIELDS:
        return format_count(value, lang=lang)
    if name in _RAW_FIELDS:
        return format_decimal(value, 2, lang=lang)
    return format_decimal(value, lang=lang)


# ══════════════════════════════════════════════════════════════════════════════
# Message ids
# ══════════════════════════════════════════════════════════════════════════════

# Kinds whose sentence names a value out of the customer's data. Each has a
# ".unlabelled" entry saying the same thing without the name.
#
# A static set, not ``finding.needs_label``: by the time a label-free
# narrative is built, ``build_evidence(include_labels=False)`` has already
# emptied ``labels``, so the finding no longer knows it would have carried
# one. Reading the runtime flag chose the labelled sentence and left a bare
# "{leader}" on the page — visible only because ``t()`` deliberately leaves an
# unsupplied placeholder in place rather than costing the reader their answer.
LABEL_BEARING_KINDS = frozenset({CONCENTRATION_LEADER})


def message_id_for(finding: Finding, *, labels_available: bool) -> str:
    """The catalogue id whose sentence says this finding.

    Correlation splits on the sign of r rather than carrying "positive" or
    "negative" as a placeholder — an adjective inside a placeholder cannot be
    translated, and that is exactly how ``core.stat_signals`` ended up with
    English baked into a structured signal.
    """
    if finding.kind == CORRELATION:
        r = finding.numbers.get("r", 0.0)
        return "narrative.correlation.negative" if r < 0 else "narrative.correlation.positive"
    base = f"narrative.{finding.kind}"
    if finding.kind in LABEL_BEARING_KINDS and not labels_available:
        return f"{base}.unlabelled"
    return base


# Which column of a finding fills which placeholder. A finding carries its
# columns positionally (the detector knows what it measured); the sentence
# needs them by name.
_COLUMN_SLOTS: dict[str, tuple[str, ...]] = {
    TREND_UP: ("column",),
    TREND_DOWN: ("column",),
    TREND_FLAT: ("column",),
    TREND_REVERSAL: ("column",),
    CONCENTRATION_LEADER: ("group_column", "column"),
    CONCENTRATION_PARETO: ("group_column", "column"),
    LONG_TAIL: ("group_column", "column"),
    SPREAD_HIGH: ("column",),
    SPREAD_LOW: ("column",),
    SKEW_RIGHT: ("column",),
    BELOW_AVERAGE_CLUSTER: ("column",),
    RANGE_SPAN: ("column",),
    OUTLIERS: ("column",),
    CORRELATION: ("x_column", "y_column"),
    COMOVEMENT: ("x_column", "y_column"),
}


def humanise_column(name: str) -> str:
    """A column name a reader can read.

    ``NET_REVENUE_AMT`` in the middle of a sentence reads as a system
    identifier and undoes the work of writing prose at all. This is
    presentation only — the finding keeps the real name, so the trace and the
    proof pack still say which column was measured.
    """
    cleaned = re.sub(r"[_]+", " ", str(name or "")).strip()
    if not cleaned:
        return ""
    if cleaned.isupper() or cleaned.islower():
        return cleaned.title()
    return cleaned


def placeholders_for(finding: Finding, lang: str, *, labels_available: bool) -> dict[str, str]:
    """Every ``{name}`` the finding's sentence needs, already formatted."""
    values: dict[str, str] = {
        name: format_number_for(name, value, lang)
        for name, value in finding.numbers.items()
    }
    for slot, column in zip(_COLUMN_SLOTS.get(finding.kind, ("column",)), finding.columns):
        values[slot] = humanise_column(column)
    if labels_available:
        values.update(finding.labels)
    return values


# ══════════════════════════════════════════════════════════════════════════════
# Phrasers
# ══════════════════════════════════════════════════════════════════════════════

class Phraser(Protocol):
    name: str

    def phrase(self, findings: list[Finding], *, lang: str,
               labels_available: bool) -> list[str]: ...


class TemplatePhraser:
    """Catalogue sentences. No model, no network, no variance."""

    name = "template"

    def phrase(self, findings: list[Finding], *, lang: str = "en",
               labels_available: bool = True) -> list[str]:
        out: list[str] = []
        for finding in findings:
            msg_id = message_id_for(finding, labels_available=labels_available)
            values = placeholders_for(finding, lang, labels_available=labels_available)
            sentence = t(msg_id, lang=lang, **values)
            if _UNFILLED_PLACEHOLDER.search(sentence):
                # t() leaves an unsupplied placeholder in place rather than
                # raising, which is right for an answer path and means a
                # missing value reaches the reader as "{leader}". Drop the
                # sentence instead of printing scaffolding, and log it —
                # this is a catalogue/placeholder mismatch, always a bug.
                log.warning("narrative sentence %s left a placeholder unfilled: %r",
                            msg_id, sentence)
                continue
            out.append(sentence)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# The narrative
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Narrative:
    """Sentences plus the record of how they came to be written."""

    sentences: tuple[str, ...] = ()
    title: str = ""
    phrasing: str = "template"
    evidence_id: str = ""
    finding_kinds: tuple[str, ...] = ()
    labels_included: bool = True
    substituted: bool = False
    substitution_reason: str = ""
    rows_sent_to_llm: int = 0
    values_sent_to_llm: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.sentences


def evidence_id(evidence: AnalysisEvidence) -> str:
    """A stable id for one set of computed findings.

    The proof pack cites it, so a reviewer can take any sentence in any answer
    and pull up the computation behind it. Derived from the findings alone —
    same findings, same id, regardless of who phrased them or in what
    language.
    """
    payload = [
        {
            "kind": f.kind,
            "columns": list(f.columns),
            "numbers": {k: round(v, 6) for k, v in sorted(f.numbers.items())},
            "materiality": f.materiality,
        }
        for f in evidence.findings
    ]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def build_narrative(
    evidence: AnalysisEvidence,
    *,
    lang: str = "en",
    limit: int = DEFAULT_LIMIT,
    phraser: Phraser | None = None,
) -> Narrative:
    """Say what the evidence found, in ``lang``.

    Never raises and never returns None: a result with nothing notable in it
    gets the "nothing stands out" sentence, which is itself a useful answer
    and considerably better than the static apology regulated tenants used to
    receive in place of any analysis at all.
    """
    phraser = phraser or TemplatePhraser()
    labels_available = evidence.labels_included
    findings = select_findings(evidence, limit=limit)

    if not findings:
        return Narrative(
            sentences=(t("narrative.nothing_notable", lang=lang),),
            title=t("narrative.title", lang=lang),
            phrasing=phraser.name,
            evidence_id=evidence_id(evidence),
            labels_included=labels_available,
        )

    try:
        sentences = phraser.phrase(
            findings, lang=lang, labels_available=labels_available)
    except Exception as exc:
        # A phrasing failure must not cost the answer. Warning, not debug:
        # this is the only signal that the narrative silently degraded.
        log.warning("phraser %s failed: %s", getattr(phraser, "name", "?"), exc)
        sentences = TemplatePhraser().phrase(
            findings, lang=lang, labels_available=labels_available)
        return Narrative(
            sentences=tuple(sentences),
            title=t("narrative.title", lang=lang),
            phrasing="template",
            evidence_id=evidence_id(evidence),
            finding_kinds=tuple(f.kind for f in findings),
            labels_included=labels_available,
            substituted=True,
            substitution_reason="phraser_failed",
        )

    return Narrative(
        sentences=tuple(sentences),
        title=t("narrative.title", lang=lang),
        phrasing=getattr(phraser, "name", "template"),
        evidence_id=evidence_id(evidence),
        finding_kinds=tuple(f.kind for f in findings),
        labels_included=labels_available,
    )


# ══════════════════════════════════════════════════════════════════════════════
# The checked-numbers rule
# ══════════════════════════════════════════════════════════════════════════════

# Thousands groups written with a space: a space between a digit and exactly
# three digits, in any of the space characters a number is grouped with.
# Removed before parsing so "4 517" and "4\u202f517" both reach 4517.
_GROUPING_SPACE = re.compile(r"(?<=\d)[ \u00a0\u202f\u2009](?=\d{3}(?!\d))")

# A figure, once grouping spaces are gone: digits with comma/dot separators.
_NUMBER_IN_PROSE = re.compile(r"-?\d+(?:[.,]\d+)*")

# A figure below this is not worth challenging: "one row", "the top 3",
# ordinals and small counts appear in fluent prose without being claims about
# the data, and rejecting a paragraph over the word "two" would make the rule
# unusable.
SMALL_NUMBER_CEILING = 12.0

# Rounding tolerance. A model writing "66%" where the evidence holds 66.2 is
# rounding, not inventing.
_ROUNDING_TOLERANCE = 0.51


def _parse_figure(token: str) -> float | None:
    """One figure from prose, read the way both languages write numbers.

    The hard case is a single separator: "4,517" is four and a half thousand
    to an English reader and four point five one seven to a French one, and
    the paragraph does not say which. It is read as a thousands group —
    a separator followed by exactly three digits and nothing else is grouped
    far more often than not, and the consequence of being wrong runs the safe
    way: the larger reading is the one that constitutes a claim about the
    data, so reading it that way means the claim gets checked. Reading it as
    4.517 waves it through as a small number without checking anything, which
    is how an invented figure reaches the page.
    """
    cleaned = token.strip()
    if not cleaned:
        return None
    commas = cleaned.count(",")
    dots = cleaned.count(".")
    if commas and dots:
        # Both present: whichever comes last is the decimal point.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif commas or dots:
        separator = "," if commas else "."
        tail = cleaned.rsplit(separator, 1)[1]
        grouped = (commas + dots) > 1 or len(tail) == 3
        if grouped:
            cleaned = cleaned.replace(separator, "")
        else:
            cleaned = cleaned.replace(separator, ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    """Every figure a paragraph asserts, in either language's notation."""
    normalised = _GROUPING_SPACE.sub("", text or "")
    found: list[float] = []
    for token in _NUMBER_IN_PROSE.findall(normalised):
        value = _parse_figure(token)
        if value is not None:
            found.append(value)
    return found


def verify_against_evidence(
    text: str, evidence: AnalysisEvidence,
) -> tuple[bool, str]:
    """Is every figure in ``text`` one the evidence actually computed?

    Returns ``(ok, reason)``. This is the enforcement point for the claim that
    a model only phrases what was computed: a figure that is not in the
    evidence was not computed, whatever the paragraph around it says.

    Deliberately permissive in two directions, because a rule that rejects
    correct prose gets switched off. Small numbers pass (a model writing "the
    top 3" is not making a claim about the data), and a figure within a
    rounding step of a computed one passes.
    """
    known = evidence.all_numbers()
    for value in numbers_in(text):
        if abs(value) <= SMALL_NUMBER_CEILING:
            continue
        if any(abs(value - k) <= _ROUNDING_TOLERANCE for k in known):
            continue
        if _is_derivable(value, evidence):
            continue
        return False, f"uncomputed_figure:{value:g}"
    return True, ""


def _is_derivable(value: float, evidence: AnalysisEvidence) -> bool:
    """Is ``value`` one computed figure expressed as a percentage of another?

    "Revenue ended at 188% of where it started" is a legitimate restatement of
    a finding that holds both endpoints, and rejecting it would make the rule
    fight fluent prose rather than invention.

    Bounded to pairs from a SINGLE finding, and to percentages. Across the
    whole evidence there are N² ratios, and with even a handful of findings
    that set is dense enough that almost any figure lands within tolerance of
    one — which is not a check, it is the appearance of one. That version
    accepted an invented "53" because two numbers from unrelated findings
    happened to divide that way.
    """
    if not (0.0 < value <= 1000.0):
        return False
    for finding in evidence.findings:
        values = list(finding.numbers.values())
        for numerator in values:
            for denominator in values:
                if denominator == 0.0 or denominator == numerator:
                    continue
                if abs(numerator / denominator * 100.0 - value) <= _ROUNDING_TOLERANCE:
                    return True
    return False


def enforce_checked_numbers(
    text: str,
    evidence: AnalysisEvidence,
    *,
    lang: str = "en",
    limit: int = DEFAULT_LIMIT,
) -> Narrative:
    """Accept a model's prose only if every figure in it was computed.

    On rejection the template narrative is returned and the substitution is
    recorded — the reader still gets an analysis, and the record says the
    model's version was discarded and why.
    """
    fallback = build_narrative(evidence, lang=lang, limit=limit)
    ok, reason = verify_against_evidence(text, evidence)
    if not ok:
        log.warning("LLM narrative rejected (%s); using computed sentences", reason)
        return Narrative(
            sentences=fallback.sentences,
            title=fallback.title,
            phrasing="template",
            evidence_id=fallback.evidence_id,
            finding_kinds=fallback.finding_kinds,
            labels_included=fallback.labels_included,
            substituted=True,
            substitution_reason=reason,
        )
    return Narrative(
        sentences=tuple(s for s in (text or "").split("\n") if s.strip()),
        title=fallback.title,
        phrasing="llm",
        evidence_id=fallback.evidence_id,
        finding_kinds=fallback.finding_kinds,
        labels_included=fallback.labels_included,
    )


# ══════════════════════════════════════════════════════════════════════════════
# The prompt a model may see
# ══════════════════════════════════════════════════════════════════════════════

def evidence_for_prompt(evidence: AnalysisEvidence, *, lang: str = "en") -> dict[str, Any]:
    """The only thing an LLM phraser is ever given.

    Findings, their numbers and their column names — no rows, no cells, and no
    data-derived label unless the evidence was built with labels included. A
    caller that wants to widen this must widen ``AnalysisEvidence`` first,
    where the governance decision already lives, rather than reaching past it
    for the rows.
    """
    return {
        "row_count": evidence.row_count,
        "language": lang,
        "labels_included": evidence.labels_included,
        "findings": [
            {
                "kind": f.kind,
                "columns": list(f.columns),
                "numbers": dict(f.numbers),
                **({"labels": dict(f.labels)} if f.labels else {}),
            }
            for f in select_findings(evidence)
        ],
    }
