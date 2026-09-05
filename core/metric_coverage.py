"""Does this metric answer the questions people will actually ask about it?

A metric that compiles is not a metric that works. The one someone defines in
chat on Tuesday is asked about on Wednesday as "top 10 customers by it", on
Thursday as "how has it changed this quarter", and on Friday by a name nobody
wrote down — and each of those is a different requirement on the same
definition. Nothing checked any of them, so the first time anyone found out
was when a user got "I could not generate a query for that".

This module derives the question shapes a metric has to survive, resolves
each against what the metric actually declares, and returns a report whose
gaps name the asset to add rather than the failure that occurred. That turns
"works for real questions" from an aspiration into a number that can go up.

The check is deliberately offline: no LLM, no warehouse, no execution. What
it verifies is whether the metric's own declarations — its name and synonyms,
its approved dimensions, its grain, its bound business dates — are sufficient
for each shape. Those are exactly the things an admin can fix, and they are
where the failures actually are.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.i18n import grain_label, t

log = logging.getLogger("querybot.metric_coverage")

# How many approved dimensions to build variations from. A metric with
# twenty dimensions does not need sixty questions to prove the point; the
# first few are the ones people ask about, and a report nobody reads is
# worth as much as no report.
MAX_DIMENSIONS = 4

# The grains a question can ask for. Ordered coarse-to-fine so a metric that
# only supports months is not judged on days first.
GRAINS = ("year", "quarter", "month", "week", "day")

# Which grains a metric declaring one grain can also answer. A metric
# reported daily can be rolled up; one reported monthly cannot be split down.
_ROLLUP: dict[str, tuple[str, ...]] = {
    "day": ("day", "week", "month", "quarter", "year"),
    "week": ("week", "month", "quarter", "year"),
    "month": ("month", "quarter", "year"),
    "quarter": ("quarter", "year"),
    "year": ("year",),
}


@dataclass(frozen=True)
class Variation:
    """One question shape a metric has to survive."""

    kind: str
    message_id: str
    question: str
    dimension: str = ""
    grain: str = ""


@dataclass(frozen=True)
class Gap:
    """A variation the metric cannot answer, and what to add so it can."""

    variation: Variation
    reason_id: str
    reason: str
    missing: str = ""


@dataclass(frozen=True)
class CoverageReport:
    variations: tuple[Variation, ...] = ()
    gaps: tuple[Gap, ...] = ()
    summary: str = ""

    @property
    def total(self) -> int:
        return len(self.variations)

    @property
    def resolvable(self) -> int:
        return self.total - len(self.gaps)

    @property
    def complete(self) -> bool:
        return self.total > 0 and not self.gaps

    def gaps_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for gap in self.gaps:
            counts[gap.reason_id] = counts.get(gap.reason_id, 0) + 1
        return counts


# ══════════════════════════════════════════════════════════════════════════════
# What the metric declares
# ══════════════════════════════════════════════════════════════════════════════

_SPLIT = re.compile(r"[,;|\n]+")


def declared_dimensions(metric: dict) -> list[str]:
    """The dimensions this metric is approved to be broken down by.

    ``allowed_dimensions`` is unstructured admin free text, so it is split on
    the separators an admin actually types rather than assumed to be a JSON
    list — the same reading ``core.metric_scope`` gives it.
    """
    raw = str((metric or {}).get("allowed_dimensions") or "")
    return [part.strip() for part in _SPLIT.split(raw) if part.strip()]


def declared_grain(metric: dict) -> str:
    return str((metric or {}).get("grain") or "").strip().lower()


def answerable_grains(metric: dict) -> tuple[str, ...]:
    """Which time grains this metric can be reported at.

    A metric stored daily rolls up to weeks, months, quarters and years; one
    stored monthly cannot be split into days. Getting this backwards is how a
    coverage report ends up demanding work that is not possible.
    """
    grain = declared_grain(metric)
    if not grain:
        return ()
    return _ROLLUP.get(grain, (grain,))


def declared_examples(metric: dict) -> list[str]:
    """The example questions stored against this metric.

    Split on newlines only: an example question can contain a comma, and
    splitting on one turns "revenue, net of returns, by region" into three
    questions that find nothing.
    """
    raw = str((metric or {}).get("example_questions") or "")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def searchable_names(metric: dict) -> list[str]:
    """Every phrase a question could use to find this metric."""
    names = [str((metric or {}).get("name") or "").strip()]
    names.extend(
        part.strip()
        for part in _SPLIT.split(str((metric or {}).get("synonyms") or ""))
    )
    return [n for n in names if n]


# ══════════════════════════════════════════════════════════════════════════════
# The question shapes
# ══════════════════════════════════════════════════════════════════════════════

def variations_for(metric: dict, *, lang: str = "en") -> list[Variation]:
    """Every question shape this metric should be able to answer.

    Built from the metric's own declarations so the set reflects what it
    claims to support: a metric with three approved dimensions gets three
    dimension questions, not a fixed list that happens to name someone else's
    columns.
    """
    name = str((metric or {}).get("name") or "").strip()
    if not name:
        return []

    dimensions = declared_dimensions(metric)[:MAX_DIMENSIONS]
    if not dimensions:
        # A metric with no approved dimensions must still be MEASURED against
        # the shapes that need one. Skipping them shrinks the denominator, so
        # the least-curated metric scores best -- "answers 2 of 7" for a bare
        # metric versus "answers 25 of 25" for a complete one is a report that
        # rewards doing nothing. The placeholder makes the absence visible.
        dimensions = [t("coverage.any_dimension", lang=lang)]
    grains = answerable_grains(metric) or ("month",)
    primary_grain = grains[0]
    out: list[Variation] = []

    # `for_dimension` / `at_grain` are what the Variation records; **values
    # are the sentence's placeholders. Named apart because "grain" is both a
    # field on the Variation and a placeholder in the question, and one name
    # for the two collides on every call that needs them together.
    def _add(kind: str, message_id: str, *, for_dimension: str = "",
             at_grain: str = "", **values) -> None:
        out.append(Variation(
            kind=kind,
            message_id=message_id,
            question=t(message_id, lang=lang, metric=name, **values),
            dimension=for_dimension,
            grain=at_grain,
        ))

    _add("base", "coverage.q.base")
    _add("base", "coverage.q.total")

    for grain in grains:
        label = grain_label(grain, 1, lang=lang)
        _add("grain", "coverage.q.grain", at_grain=grain, grain=label)
        _add("grain", "coverage.q.grain_last", at_grain=grain, grain=label,
             grain_plural=grain_label(grain, 6, lang=lang))

    for dimension in dimensions:
        _add("dimension", "coverage.q.dimension",
             for_dimension=dimension, dimension=dimension)
        _add("ranking", "coverage.q.ranking",
             for_dimension=dimension, dimension=dimension)
        _add("ranking", "coverage.q.ranking_bottom",
             for_dimension=dimension, dimension=dimension)
        _add("share", "coverage.q.share",
             for_dimension=dimension, dimension=dimension)
        _add("dimension", "coverage.q.dimension_grain",
             for_dimension=dimension, at_grain=primary_grain,
             dimension=dimension,
             grain=grain_label(primary_grain, 1, lang=lang))

    _add("comparison", "coverage.q.comparison_prior", at_grain=primary_grain,
         grain=grain_label(primary_grain, 1, lang=lang))
    _add("comparison", "coverage.q.comparison_year", at_grain=primary_grain)
    _add("trend", "coverage.q.trend", at_grain=primary_grain)

    # The metric's own example questions. Every shape above interpolates the
    # metric's name, so the name check can never fail on one -- these are the
    # phrasings that genuinely can, because a human wrote them from how the
    # business talks rather than from what the metric happens to be called.
    # A metric whose own examples do not find it is the commonest real
    # failure this report exists to catch.
    for example in declared_examples(metric):
        out.append(Variation(
            kind="example", message_id="", question=example))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Resolving a variation
# ══════════════════════════════════════════════════════════════════════════════

def check_variation(
    variation: Variation,
    metric: dict,
    *,
    date_roles: list[dict] | None = None,
    lang: str = "en",
) -> Gap | None:
    """Can this metric answer this shape? ``None`` when it can.

    Order matters: the most fundamental missing thing is reported, so an
    admin fixing one gap does not immediately meet a second on the same
    question. A metric with no date role and no dimensions should be told
    about the date role first, because that blocks more shapes.
    """
    kind = variation.kind

    if kind in {"grain", "comparison", "trend"}:
        if not date_roles:
            return Gap(variation, "coverage.gap.no_date_role",
                       t("coverage.gap.no_date_role", lang=lang), "date_role")
        if not declared_grain(metric):
            return Gap(variation, "coverage.gap.no_grain",
                       t("coverage.gap.no_grain", lang=lang), "grain")

    if kind in {"dimension", "ranking", "share"}:
        approved = {d.casefold() for d in declared_dimensions(metric)}
        if not approved:
            return Gap(variation, "coverage.gap.no_dimensions",
                       t("coverage.gap.no_dimensions", lang=lang), "dimensions")
        if variation.dimension and variation.dimension.casefold() not in approved:
            return Gap(
                variation, "coverage.gap.dimension_not_allowed",
                t("coverage.gap.dimension_not_allowed", lang=lang,
                  dimension=variation.dimension),
                variation.dimension,
            )
        # A dimension question that also names a period needs both.
        if variation.grain and not date_roles:
            return Gap(variation, "coverage.gap.no_date_role",
                       t("coverage.gap.no_date_role", lang=lang), "date_role")

    if kind == "example" and not question_finds_metric(variation.question, metric):
        # Only for phrasings we did not write. Every generated shape
        # interpolates the metric's name, so applying this to them would
        # check that a template contains what the template put there.
        return Gap(variation, "coverage.gap.name_not_found",
                   t("coverage.gap.name_not_found", lang=lang), "synonym")
    return None


def question_finds_metric(question: str, metric: dict) -> bool:
    """Would a question phrased this way locate the metric by name?

    Whole-phrase, case-insensitive, on word boundaries -- the same shape
    ``core.metric_scope`` matches with. A metric whose own generated question
    cannot find it is not a hypothetical: it happens whenever the display
    name and the phrasing people use have drifted apart.
    """
    text = str(question or "").casefold()
    for name in searchable_names(metric):
        phrase = re.escape(name.casefold())
        if re.search(rf"(?<![a-z0-9]){phrase}(?![a-z0-9])", text):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# The report
# ══════════════════════════════════════════════════════════════════════════════

def coverage_report(
    metric: dict,
    *,
    date_roles: list[dict] | None = None,
    lang: str = "en",
) -> CoverageReport:
    """Which question shapes this metric answers, and what the rest need.

    Never raises: a coverage report is advice, and advice that can break a
    metric save is worse than no advice.
    """
    try:
        variations = variations_for(metric, lang=lang)
    except Exception as exc:
        log.warning("Coverage variations unavailable for %r: %s",
                    (metric or {}).get("name"), exc)
        return CoverageReport()

    gaps: list[Gap] = []
    for variation in variations:
        try:
            gap = check_variation(variation, metric, date_roles=date_roles, lang=lang)
        except Exception as exc:
            log.warning("Coverage check failed for %r: %s", variation.question, exc)
            continue
        if gap is not None:
            gaps.append(gap)

    total = len(variations)
    resolvable = total - len(gaps)
    if total and not gaps:
        summary = t("coverage.summary.complete", lang=lang, total=total)
    elif total:
        stem = "coverage.summary.one" if resolvable == 1 else "coverage.summary.other"
        summary = t(stem, lang=lang, resolvable=resolvable, total=total)
    else:
        summary = ""
    return CoverageReport(
        variations=tuple(variations), gaps=tuple(gaps), summary=summary)


def report_for_metric_id(
    account_id: str, metric_id: int, *, lang: str = "en",
) -> CoverageReport:
    """Coverage for one saved metric, reading its date bindings from the store."""
    import store

    try:
        metric = next(
            (m for m in store.list_metrics(account_id, active_only=False)
             if int(m.get("id") or 0) == int(metric_id)),
            None,
        )
    except Exception as exc:
        log.warning("Metric %s unavailable for coverage: %s", metric_id, exc)
        return CoverageReport()
    if not metric:
        return CoverageReport()

    try:
        date_roles = store.list_metric_date_contexts(
            account_id, metric_ids=[int(metric_id)])
    except Exception as exc:
        log.warning("Date contexts unavailable for metric %s: %s", metric_id, exc)
        date_roles = []
    return coverage_report(metric, date_roles=date_roles, lang=lang)
