"""Computed evidence for an analytical narrative — no LLM, no raw rows out.

Every analytical summary this product produces should be traceable to a
number it computed, not to a model's impression of a table. This module is
the computation half of that: it runs the deterministic analysers the
codebase already owns over a governed result set and returns typed
``Finding`` objects carrying the numbers, the columns and a materiality
score. ``core.analysis_narrative`` turns those into sentences.

Two properties make this worth having as its own layer.

**Findings carry numbers, not English.** ``core.stat_signals`` already
computes most of the statistics here, but bakes an English ``label`` into
every signal, so its output cannot be spoken in French and cannot be
re-phrased without re-deriving the arithmetic. A ``Finding`` carries
``numbers`` — named floats destined for ``str.format`` placeholders — and
leaves the words to the phrasing layer.

**Data-derived labels are separated from aggregates and governed.** A
category leader's name is a value out of the customer's data; the share it
holds is an aggregate. They travel in different fields (``labels`` vs
``numbers``) so a regulated tenant can be given the second without the
first. ``build_evidence(..., include_labels=False)`` drops them, and the
phrasing layer has a label-free sentence for every finding kind that can
carry one. That is what lets a regulated tenant have a real analytical
summary rather than a static apology.

Nothing here calls an LLM, opens a connection, or logs a value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Any

log = logging.getLogger("querybot.analysis_evidence")


# ══════════════════════════════════════════════════════════════════════════════
# Finding kinds
# ══════════════════════════════════════════════════════════════════════════════
# A kind is a stable identifier: it is the suffix of the i18n message id the
# phrasing layer looks up, so renaming one is a breaking change to the
# catalogue. Families group kinds that say the same *sort* of thing, and
# de-duplication happens per (family, primary column) — otherwise a single
# skewed column produces "high variance", "right-skewed" and "outliers
# present" as three separate bullets about one fact.

TREND_UP = "trend_up"
TREND_DOWN = "trend_down"
TREND_FLAT = "trend_flat"
TREND_REVERSAL = "trend_reversal"

CONCENTRATION_LEADER = "concentration_leader"
CONCENTRATION_PARETO = "concentration_pareto"
LONG_TAIL = "long_tail"

SPREAD_HIGH = "spread_high"
SPREAD_LOW = "spread_low"
SKEW_RIGHT = "skew_right"
BELOW_AVERAGE_CLUSTER = "below_average_cluster"

OUTLIERS = "outliers"

CORRELATION = "correlation"
COMOVEMENT = "comovement"

RANGE_SPAN = "range_span"

_FAMILIES: dict[str, str] = {
    TREND_UP: "trend",
    TREND_DOWN: "trend",
    TREND_FLAT: "trend",
    TREND_REVERSAL: "trend",
    CONCENTRATION_LEADER: "concentration",
    CONCENTRATION_PARETO: "concentration",
    LONG_TAIL: "concentration",
    SPREAD_HIGH: "spread",
    SPREAD_LOW: "spread",
    SKEW_RIGHT: "spread",
    BELOW_AVERAGE_CLUSTER: "spread",
    OUTLIERS: "outliers",
    CORRELATION: "relationship",
    COMOVEMENT: "relationship",
    RANGE_SPAN: "spread",
}

# Per-kind weight applied to the normalised effect size. Two findings with the
# same raw strength are not equally worth a sentence: a reversal in a trend is
# the most useful thing you can tell someone about a time series, while "this
# column is uniform" is a footnote. Weights are the only place a judgement
# about *interestingness* lives; effect size stays arithmetic.
_WEIGHTS: dict[str, float] = {
    TREND_REVERSAL: 1.00,
    TREND_UP: 0.95,
    TREND_DOWN: 0.95,
    CONCENTRATION_LEADER: 0.90,
    CONCENTRATION_PARETO: 0.85,
    OUTLIERS: 0.80,
    CORRELATION: 0.75,
    LONG_TAIL: 0.70,
    SKEW_RIGHT: 0.60,
    COMOVEMENT: 0.55,
    SPREAD_HIGH: 0.50,
    BELOW_AVERAGE_CLUSTER: 0.45,
    RANGE_SPAN: 0.35,
    TREND_FLAT: 0.30,
    SPREAD_LOW: 0.25,
}


# At most this many findings from any one family survive ranking.
MAX_PER_FAMILY = 2


def family_of(kind: str) -> str:
    """The de-duplication family a finding kind belongs to."""
    return _FAMILIES.get(kind, kind)


# ══════════════════════════════════════════════════════════════════════════════
# The typed result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Finding:
    """One computed statement about a result set.

    ``numbers`` are aggregates and are always safe to send anywhere — they are
    the placeholder values the phrasing layer formats. ``labels`` are strings
    taken from the customer's data (a category leader's name, a period label)
    and are governed: ``build_evidence(include_labels=False)`` returns findings
    with ``labels`` empty, and every kind that can carry one has a label-free
    sentence in the catalogue.
    """

    kind: str
    columns: tuple[str, ...] = ()
    numbers: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    materiality: float = 0.0

    @property
    def family(self) -> str:
        return family_of(self.kind)

    @property
    def primary_column(self) -> str:
        return self.columns[0] if self.columns else ""

    @property
    def needs_label(self) -> bool:
        """True when this finding was built with a data-derived label."""
        return bool(self.labels)


@dataclass(frozen=True)
class AnalysisEvidence:
    """Everything computed about one result set, ranked and de-duplicated."""

    findings: tuple[Finding, ...] = ()
    row_count: int = 0
    numeric_columns: tuple[str, ...] = ()
    label_columns: tuple[str, ...] = ()
    temporal_column: str = ""
    labels_included: bool = True

    def of_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def of_family(self, family: str) -> list[Finding]:
        return [f for f in self.findings if f.family == family]

    def top(self, n: int) -> list[Finding]:
        return list(self.findings[:n])

    def all_numbers(self) -> list[float]:
        """Every number this evidence asserts.

        ``core.analysis_narrative`` checks a model's prose against this list:
        a figure in the narrative that is not here was not computed, and the
        prose is rejected in favour of the template.
        """
        out: list[float] = []
        for finding in self.findings:
            out.extend(finding.numbers.values())
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Numeric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(value: Any) -> float | None:
    """Parse a cell to a float, or None. NaN is discarded, not propagated."""
    try:
        parsed = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return None if parsed != parsed else parsed


def _column_values(rows: list[dict], col: str) -> list[float]:
    out = []
    for row in rows:
        parsed = _to_float(row.get(col))
        if parsed is not None:
            out.append(parsed)
    return out


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _score(kind: str, effect: float) -> float:
    """Materiality = normalised effect size × the kind's weight.

    Effect is clamped into 0..1 by the caller's own normalisation, so this
    stays comparable across kinds that measure entirely different things —
    a share, a correlation coefficient and a percentage change.
    """
    return round(_clamp(effect) * _WEIGHTS.get(kind, 0.5), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Column classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_columns(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, label) by how many cells parse as numbers.

    Deliberately the same majority rule ``core.stat_signals`` uses, so the two
    layers cannot disagree about which column is the measure.
    """
    if not rows:
        return [], []
    numeric, labels = [], []
    for col in rows[0].keys():
        values = _column_values(rows, col)
        if len(values) >= max(1, len(rows) // 2):
            numeric.append(col)
        else:
            labels.append(col)
    return numeric, labels


def find_temporal_column(rows: list[dict], label_columns: list[str]) -> str:
    """The label column that holds period labels, or "".

    Delegates to ``core.stat_signals._is_temporal_col`` rather than carrying a
    second copy of its regexes — that function already encodes the fix for
    "Nova"/"Maritime"/"Mayfield" being read as month abbreviations, and a
    second implementation would drift away from it.
    """
    if not rows:
        return ""
    try:
        from core.stat_signals import _is_temporal_col
    except Exception:  # pragma: no cover - import guard
        log.warning("stat_signals unavailable; no temporal column detected")
        return ""
    for col in label_columns:
        try:
            if _is_temporal_col(col, rows):
                return col
        except Exception as exc:
            log.warning("temporal detection failed for %s: %s", col, exc)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Detectors — each returns findings for one aspect of the result
# ══════════════════════════════════════════════════════════════════════════════
# Every detector is pure, takes already-released rows, and returns [] rather
# than raising. They are separately testable on purpose: the ranking is only
# trustworthy if each input to it is.

# A first-to-last change below this reads as flat rather than directional.
# Same band as core.stat_signals._FLAT_TREND_PCT and the narrative layer's
# "stable" wording — three classifiers on one series must not disagree.
FLAT_TREND_PCT = 5.0

# A leader below this share is not worth calling dominance.
_LEADER_MIN_SHARE = 0.35
# Top-20% share above this is a Pareto concentration.
_PARETO_MIN_SHARE = 0.60
# ...but only when it exceeds the leader's own share by at least this much,
# otherwise the two findings are one fact stated twice.
_PARETO_ADDS_OVER_LEADER = 0.15
# |r| below this is not worth reporting as a relationship.
_CORRELATION_MIN_ABS_R = 0.5
# Coefficient of variation bands.
_CV_HIGH = 0.6
_CV_LOW = 0.08


def trend_findings(rows: list[dict], value_col: str, period_col: str) -> list[Finding]:
    """Direction, size and any reversal in a series ordered by period."""
    values = [_to_float(row.get(value_col)) for row in rows]
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return []

    first, last = values[0], values[-1]
    found: list[Finding] = []

    if first != 0:
        pct = (last - first) / abs(first) * 100.0
        magnitude = abs(pct)
        if magnitude < FLAT_TREND_PCT:
            kind = TREND_FLAT
        elif last > first:
            kind = TREND_UP
        else:
            kind = TREND_DOWN
        found.append(Finding(
            kind=kind,
            columns=(value_col, period_col),
            numbers={"pct": round(magnitude, 1), "first": first, "last": last},
            materiality=_score(kind, magnitude / 100.0),
        ))

    # A reversal is worth more than a direction: it is the thing a reader
    # cannot see from the headline number. Detected on the sign of consecutive
    # deltas, ignoring the flat band so noise around zero does not register as
    # a turn.
    deltas = [b - a for a, b in zip(values, values[1:])]
    scale = max(abs(v) for v in values) or 1.0
    signs = [
        (1 if d > 0 else -1)
        for d in deltas
        if abs(d) / scale * 100.0 >= FLAT_TREND_PCT
    ]
    turns = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    if turns:
        peak = max(values)
        trough = min(values)
        span = peak - trough
        # Scored by how much of the range the series gave back, not by how
        # many times it turned. One decisive turn is the whole story of a
        # series; three shallow ones are noise. Counting turns ranked
        # 100 -> 400 -> 300 -> 150 below "up 50%", which is the reading that
        # misleads: the series ended well below its peak.
        retracement = max(peak - last, first - trough) / span if span else 0.0
        found.append(Finding(
            kind=TREND_REVERSAL,
            columns=(value_col, period_col),
            numbers={"turns": float(turns), "peak": peak, "trough": trough},
            materiality=_score(TREND_REVERSAL, retracement),
        ))
    return found


def concentration_findings(
    rows: list[dict], value_col: str, label_col: str,
) -> list[Finding]:
    """How much of the total sits in how few rows."""
    pairs = [
        (str(row.get(label_col, "") or "")[:60], _to_float(row.get(value_col)))
        for row in rows
    ]
    pairs = [(name, v) for name, v in pairs if v is not None and v > 0]
    if len(pairs) < 2:
        return []
    total = sum(v for _, v in pairs)
    if total <= 0:
        return []

    found: list[Finding] = []
    ordered = sorted(pairs, key=lambda p: p[1], reverse=True)

    leader_name, leader_value = ordered[0]
    leader_share = leader_value / total
    if leader_share >= _LEADER_MIN_SHARE:
        found.append(Finding(
            kind=CONCENTRATION_LEADER,
            columns=(label_col, value_col),
            numbers={"share": round(leader_share * 100, 1), "value": leader_value},
            labels={"leader": leader_name},
            materiality=_score(CONCENTRATION_LEADER, leader_share),
        ))

    # Pareto is only a second fact when it is not merely restating the leader.
    # With one dominant row the top-20% share is arithmetically implied by the
    # leader's share, and reporting both says one thing twice — so it is
    # emitted when no row dominates, or when the top group adds materially
    # more than the leader alone.
    n_top = max(1, len(ordered) // 5)
    top_share = sum(v for _, v in ordered[:n_top]) / total
    pareto_adds_information = (
        leader_share < _LEADER_MIN_SHARE
        or top_share - leader_share >= _PARETO_ADDS_OVER_LEADER
    )
    if top_share >= _PARETO_MIN_SHARE and len(ordered) >= 5 and pareto_adds_information:
        found.append(Finding(
            kind=CONCENTRATION_PARETO,
            columns=(label_col, value_col),
            numbers={
                "share": round(top_share * 100, 1),
                "n_top": float(n_top),
                "n_total": float(len(ordered)),
            },
            materiality=_score(CONCENTRATION_PARETO, top_share),
        ))

    # The mirror image, and the more actionable half: how many rows together
    # account for almost nothing.
    n_bottom = max(1, len(ordered) // 2)
    tail_share = sum(v for _, v in ordered[-n_bottom:]) / total
    if len(ordered) >= 6 and tail_share <= 0.10:
        found.append(Finding(
            kind=LONG_TAIL,
            columns=(label_col, value_col),
            numbers={
                "share": round(tail_share * 100, 1),
                "n_bottom": float(n_bottom),
                "n_total": float(len(ordered)),
            },
            materiality=_score(LONG_TAIL, 1.0 - tail_share * 10.0),
        ))
    return found


def spread_findings(rows: list[dict], value_col: str) -> list[Finding]:
    """Dispersion, shape and the below-average cluster in one column."""
    values = _column_values(rows, value_col)
    if len(values) < 2:
        return []

    avg = mean(values)
    mid = median(values)
    sd = stdev(values)
    found: list[Finding] = []

    if avg != 0:
        cv = sd / abs(avg)
        if cv > _CV_HIGH:
            found.append(Finding(
                kind=SPREAD_HIGH,
                columns=(value_col,),
                numbers={"cv": round(cv, 2), "mean": avg, "sd": sd},
                materiality=_score(SPREAD_HIGH, min(1.0, cv / 2.0)),
            ))
        elif cv < _CV_LOW:
            found.append(Finding(
                kind=SPREAD_LOW,
                columns=(value_col,),
                numbers={"cv": round(cv, 3), "mean": avg},
                materiality=_score(SPREAD_LOW, 1.0 - cv / _CV_LOW),
            ))

    if avg > 0 and mid > 0 and avg > 1.5 * mid:
        ratio = avg / mid
        found.append(Finding(
            kind=SKEW_RIGHT,
            columns=(value_col,),
            numbers={"ratio": round(ratio, 2), "mean": avg, "median": mid},
            materiality=_score(SKEW_RIGHT, min(1.0, (ratio - 1.0) / 3.0)),
        ))

    below = [v for v in values if v < avg * 0.5]
    if below and len(below) >= max(1, len(values) // 5):
        share = len(below) / len(values)
        found.append(Finding(
            kind=BELOW_AVERAGE_CLUSTER,
            columns=(value_col,),
            numbers={
                "count": float(len(below)),
                "n_total": float(len(values)),
                "share": round(share * 100, 1),
            },
            materiality=_score(BELOW_AVERAGE_CLUSTER, share),
        ))

    low, high = min(values), max(values)
    if low != high and low != 0:
        span = abs(high - low) / abs(low)
        if span >= 1.0:
            found.append(Finding(
                kind=RANGE_SPAN,
                columns=(value_col,),
                numbers={"low": low, "high": high, "multiple": round(high / low, 1)}
                if low > 0 else {"low": low, "high": high},
                materiality=_score(RANGE_SPAN, min(1.0, span / 10.0)),
            ))
    return found


def outlier_findings(rows: list[dict], value_col: str) -> list[Finding]:
    """Outliers, delegated to the detector the product already ships."""
    if len(rows) < 4:
        return []
    try:
        from core.anomaly_detection import detect_anomalies
    except Exception:  # pragma: no cover - import guard
        log.warning("anomaly_detection unavailable; no outlier findings")
        return []
    try:
        result = detect_anomalies(rows, value_col=value_col, method="iqr")
    except Exception as exc:
        log.warning("outlier detection failed on %s: %s", value_col, exc)
        return []
    if not result or not result.flagged_rows:
        return []
    return [Finding(
        kind=OUTLIERS,
        columns=(value_col,),
        numbers={
            "count": float(result.flagged_rows),
            "n_total": float(result.total_rows),
            "share": round(result.flagged_fraction * 100, 1),
        },
        # A handful of outliers in a large set is the interesting case; a
        # result where half the rows are "anomalous" means the threshold, not
        # the data, is the story — so materiality peaks around 10% and falls
        # away on either side.
        materiality=_score(OUTLIERS, 1.0 - abs(result.flagged_fraction - 0.10) * 5.0),
    )]


def relationship_findings(
    rows: list[dict], x_col: str, y_col: str,
) -> list[Finding]:
    """A correlation between two measures, and whether they move together."""
    if len(rows) < 3:
        return []
    try:
        from core.correlation_analysis import compute_correlation
    except Exception:  # pragma: no cover - import guard
        log.warning("correlation_analysis unavailable; no relationship findings")
        return []
    try:
        result = compute_correlation(rows, x_col=x_col, y_col=y_col)
    except Exception as exc:
        log.warning("correlation failed on %s/%s: %s", x_col, y_col, exc)
        return []
    if result is None or result.pearson_r is None:
        return []

    found: list[Finding] = []
    r = result.pearson_r
    if abs(r) >= _CORRELATION_MIN_ABS_R:
        found.append(Finding(
            kind=CORRELATION,
            columns=(x_col, y_col),
            numbers={
                "r": round(r, 2),
                "r_squared": round(result.r_squared or 0.0, 2),
                "n": float(result.n),
            },
            materiality=_score(CORRELATION, abs(r)),
        ))
        return found

    # Below the correlation floor, a weaker but still useful statement: how
    # often the two columns sit on the same side of their own averages. This
    # survives non-linear relationships that Pearson misses.
    xs = [_to_float(row.get(x_col)) for row in rows]
    ys = [_to_float(row.get(y_col)) for row in rows]
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return found
    mx = mean(a for a, _ in pairs)
    my = mean(b for _, b in pairs)
    together = sum(1 for a, b in pairs if (a > mx) == (b > my))
    agreement = together / len(pairs)
    if agreement > 0.60:
        found.append(Finding(
            kind=COMOVEMENT,
            columns=(x_col, y_col),
            numbers={"share": round(agreement * 100, 1), "n": float(len(pairs))},
            materiality=_score(COMOVEMENT, agreement),
        ))
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Ranking and de-duplication
# ══════════════════════════════════════════════════════════════════════════════

def rank_findings(findings: list[Finding]) -> list[Finding]:
    """Order by materiality, keeping the strongest finding per family+column.

    De-duplication is per (family, primary column) rather than per kind: one
    skewed column otherwise yields high-variance, right-skew and a wide range
    as three separate statements about a single fact. Two different columns
    may each contribute one spread finding — that is genuinely two facts.

    A family is then capped at ``MAX_PER_FAMILY``, because "genuinely two
    facts" stops being true at three: a result with four measures would
    otherwise spend the whole summary observing that each of them has a wide
    range, and crowd out the trend and the concentration entirely.
    """
    ordered = sorted(findings, key=lambda f: (-f.materiality, f.kind, f.primary_column))
    seen: set[tuple[str, str]] = set()
    per_family: dict[str, int] = {}
    kept: list[Finding] = []
    for finding in ordered:
        key = (finding.family, finding.primary_column)
        if key in seen:
            continue
        if per_family.get(finding.family, 0) >= MAX_PER_FAMILY:
            continue
        seen.add(key)
        per_family[finding.family] = per_family.get(finding.family, 0) + 1
        kept.append(finding)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# The entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_evidence(
    rows: list[dict],
    *,
    include_labels: bool = True,
    max_numeric_columns: int = 3,
) -> AnalysisEvidence:
    """Compute every finding this result set supports, ranked.

    ``include_labels=False`` strips the data-derived strings (a category
    leader's name) while keeping the aggregates, so a regulated tenant gets
    the same findings with label-free wording rather than nothing at all.

    ``rows`` must already have passed the governed path; this function does
    not fetch, mask or release anything. It never raises: a detector that
    fails is logged at warning and contributes no findings, because a broken
    statistic must not cost the user their answer.
    """
    if not rows:
        return AnalysisEvidence(row_count=0, labels_included=include_labels)

    numeric_cols, label_cols = classify_columns(rows)
    temporal_col = find_temporal_column(rows, label_cols)

    # Bounded on purpose. A 40-column result would otherwise produce hundreds
    # of findings for the ranker to discard, and the columns that matter are
    # the leading measures — the same order the renderer displays.
    measures = numeric_cols[:max_numeric_columns]

    findings: list[Finding] = []
    for value_col in measures:
        if temporal_col:
            findings.extend(_guard(trend_findings, rows, value_col, temporal_col))
        for label_col in label_cols[:2]:
            if label_col == temporal_col:
                continue
            findings.extend(_guard(concentration_findings, rows, value_col, label_col))
        findings.extend(_guard(spread_findings, rows, value_col))
        outliers = _guard(outlier_findings, rows, value_col)
        # A single outlier in a result that already has a dominant leader is
        # that leader's row. Reporting both spends two of the reader's five
        # sentences pointing at one number from two directions.
        leader_here = any(
            f.kind == CONCENTRATION_LEADER and value_col in f.columns
            for f in findings
        )
        if not (leader_here and all(f.numbers.get("count") == 1.0 for f in outliers)):
            findings.extend(outliers)

    if len(measures) >= 2:
        findings.extend(_guard(relationship_findings, rows, measures[0], measures[1]))

    if not include_labels:
        findings = [
            Finding(
                kind=f.kind,
                columns=f.columns,
                numbers=f.numbers,
                labels={},
                materiality=f.materiality,
            )
            for f in findings
        ]

    return AnalysisEvidence(
        findings=tuple(rank_findings(findings)),
        row_count=len(rows),
        numeric_columns=tuple(numeric_cols),
        label_columns=tuple(label_cols),
        temporal_column=temporal_col,
        labels_included=include_labels,
    )


def _guard(detector, *args) -> list[Finding]:
    """Run a detector; a failure costs its findings, never the answer.

    Logged at warning rather than debug: this is a fail-open handler in a
    user-facing path, so the log is the only way "computing nothing" is
    distinguishable from "found nothing".
    """
    try:
        return detector(*args) or []
    except Exception as exc:
        log.warning("%s failed: %s", getattr(detector, "__name__", detector), exc)
        return []
