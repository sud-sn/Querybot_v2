"""
core/stat_signals.py

Pure-Python statistical pattern detector.

Computes named signals from a result row-set (list[dict]) — no LLM, no DB
call, no raw-value exposure.  Signals feed two downstream consumers:

  1. template_suggestions()  — instant zero-LLM follow-up question templates
  2. format_signals_for_llm() — compact signal summary sent to the LLM when
     templates can't fill 3 suggestions (Method 3 constrained-LLM fallback)

Design rules
  - Never returns raw row values; only aggregated/derived stats
  - All computation is O(n) or O(n log n) — safe on large result sets
  - Falls back gracefully when statistics library is unavailable
"""

from __future__ import annotations

import re
from statistics import mean, median, stdev
from typing import Any

# ── helpers ───────────────────────────────────────────────────────────────────

# Below this first-to-last change a series is reported as flat rather than
# given a direction. Matches the intent of the "stable" band the narrative
# layer already applies, so the two cannot contradict each other.
_FLAT_TREND_PCT = 5.0


def _to_float(v: Any) -> float | None:
    try:
        f = float(str(v).replace(",", ""))
        return None if (f != f) else f        # discard NaN
    except (TypeError, ValueError):
        return None


def _classify_columns(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, text_cols) based on the first row's values."""
    numeric, text = [], []
    for col in rows[0].keys():
        vals = [_to_float(r.get(col)) for r in rows if r.get(col) is not None]
        vals = [v for v in vals if v is not None]
        if len(vals) >= max(1, len(rows) // 2):
            numeric.append(col)
        else:
            text.append(col)
    return numeric, text


# A column NAME that names a period. This is the strong signal and is checked
# first.
_TEMPORAL_NAME_RE = re.compile(
    r"(?:^|[^a-z])(date|day|week|month|quarter|year|period|fiscal|yyyy|mm)(?:[^a-z]|$)",
    re.IGNORECASE,
)

# A VALUE that is itself a period label. Anchored, because an unanchored
# pattern matched month abbreviations as bare substrings anywhere in the text:
# "Nova" contains "nov", "Maritime" contains "mar", "Mayfield" contains "may"
# and "Septic" contains "sep". So a result grouped by profit centre read as a
# time series, and the answer offered "the trend is downward — what period
# drove the biggest change?" about a result with no period in it at all.
#
# Anchoring alone did not fix it. The month branch ended `[a-z]*`, which is
# free to eat the rest of the word before the anchor is reached, so every one
# of those names still matched — and so did "Marseille", "Augusta",
# "Octagon", "Decathlon" and any other word merely BEGINNING with a month
# abbreviation. The months and weekdays are therefore spelled out: an
# abbreviation, or the whole word, and nothing else. Same for the weekday
# branch, where "Satellite", "Frigo", "Monaco" and "Sunbelt" had the same
# problem.
_MONTHS = (
    r"january|february|march|april|june|july|august|september|october"
    r"|november|december"
    # French, because the product ships French tenants whose warehouses hold
    # French period labels.
    r"|janvier|f\u00e9vrier|fevrier|mars|avril|juin|juillet|ao\u00fbt|aout"
    r"|septembre|octobre|novembre|d\u00e9cembre|decembre"
    # Abbreviations. "may"/"mai" carry no separate long form.
    r"|jan|feb|mar|apr|may|mai|jun|jul|aug|sept|sep|oct|nov|dec"
)
_WEEKDAYS = (
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"
    r"|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
)
_TEMPORAL_VALUE_RE = re.compile(
    r"^(?:"
    r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?"          # 2026-06, 2026/06/30
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"            # 30-06-2026
    r"|(?:19|20)\d{2}"                            # 2026
    r"|q[1-4][\s-]?(?:19|20)?\d{2}"               # Q2 2026
    r"|(?:19|20)\d{2}[\s-]?q[1-4]"                # 2026-Q2
    r"|(?:" + _MONTHS + r")\.?"
    r"(?:[\s-]+(?:19|20)?\d{2})?"                 # Jun, June 2026
    r"|(?:" + _WEEKDAYS + r")\.?"                  # weekday labels
    r")$",
    re.IGNORECASE,
)


def _is_temporal_col(col_name: str, rows: list[dict]) -> bool:
    """Does this column hold period labels?

    Name first, then the values themselves — and a value only counts when the
    WHOLE value is a period label, not when a month abbreviation happens to
    appear inside a business name.
    """
    if _TEMPORAL_NAME_RE.search(str(col_name or "")):
        return True
    sample = [str(r.get(col_name, "") or "").strip() for r in rows[:6]]
    sample = [value for value in sample if value]
    if not sample:
        return False
    matched = sum(1 for value in sample if _TEMPORAL_VALUE_RE.match(value))
    # A majority, so one stray "May" among six branch names cannot carry it.
    return matched >= max(2, (len(sample) + 1) // 2)


def _cv(values: list[float]) -> float | None:
    """Coefficient of variation (std / mean).  None when mean == 0."""
    if len(values) < 2:
        return None
    m = mean(values)
    if m == 0:
        return None
    return stdev(values) / abs(m)


def _pareto_ratio(values: list[float]) -> float | None:
    """Fraction of total held by the top 20 % of rows."""
    if not values:
        return None
    total = sum(values)
    if total <= 0:
        return None
    n_top = max(1, len(values) // 5)
    top_sum = sum(sorted(values, reverse=True)[:n_top])
    return top_sum / total


# ══════════════════════════════════════════════════════════════════════════════
# Core signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(rows: list[dict]) -> list[dict]:
    """
    Analyse result rows and return a list of named statistical signal dicts.

    Each signal dict contains at minimum:
        {type: str, col: str | None, value: float | str | None, label: str}

    'label' is a short human-readable description used in LLM prompts.
    No raw row values are included.
    """
    if not rows or len(rows) < 2:
        return []

    numeric_cols, text_cols = _classify_columns(rows)
    signals: list[dict] = []

    # ── Per-numeric-column signals ────────────────────────────────────────────
    for col in numeric_cols:
        vals = [_to_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue

        mn   = mean(vals)
        md   = median(vals)
        mx   = max(vals)
        mi   = min(vals)
        sd   = stdev(vals) if len(vals) >= 2 else 0.0
        cv   = _cv(vals)
        par  = _pareto_ratio(vals)

        # High variance / outlier-prone
        if cv is not None and cv > 0.6:
            outlier_count = sum(1 for v in vals if v > mn + 2.5 * sd)
            signals.append({
                "type": "high_variance",
                "col": col,
                "value": round(cv, 2),
                "label": f"high variance in {col} (CV={cv:.2f})",
            })
            if outlier_count > 0:
                signals.append({
                    "type": "outlier_present",
                    "col": col,
                    "value": outlier_count,
                    "label": f"{outlier_count} outlier(s) in {col} (>mean+2.5σ)",
                })

        # Low variance — suspiciously uniform
        if cv is not None and cv < 0.08:
            signals.append({
                "type": "low_variance",
                "col": col,
                "value": round(cv, 3),
                "label": f"{col} values are very uniform (CV={cv:.3f})",
            })

        # Pareto concentration
        if par is not None and par > 0.60:
            n_top = max(1, len(vals) // 5)
            signals.append({
                "type": "pareto",
                "col": col,
                "value": round(par * 100, 1),
                "n_top": n_top,
                "label": f"top {n_top} rows hold {par*100:.0f}% of {col} (Pareto)",
            })

        # Right-skewed distribution
        if mn > 0 and md > 0 and mn > 1.5 * md:
            signals.append({
                "type": "skewed_right",
                "col": col,
                "value": round(mn / md, 2),
                "label": f"{col} is right-skewed (mean={mn:.1f} vs median={md:.1f})",
            })

        # Below-average gap — bottom tier significantly below mean
        below_avg = [v for v in vals if v < mn * 0.5]
        if len(below_avg) >= max(1, len(vals) // 5):
            signals.append({
                "type": "below_avg_gap",
                "col": col,
                "value": len(below_avg),
                "label": f"{len(below_avg)} rows in {col} are significantly below average",
            })

    # ── Multi-column signals ──────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        col_a, col_b = numeric_cols[0], numeric_cols[1]
        # Cross-column direction: check if top-5 rows rank similarly on both cols
        vals_a = [_to_float(r.get(col_a)) or 0.0 for r in rows]
        vals_b = [_to_float(r.get(col_b)) or 0.0 for r in rows]
        mn_a, mn_b = mean(vals_a), mean(vals_b)
        both_above = sum(
            1 for a, b in zip(vals_a, vals_b)
            if a > mn_a and b > mn_b
        )
        both_below = sum(
            1 for a, b in zip(vals_a, vals_b)
            if a < mn_a and b < mn_b
        )
        agreement = (both_above + both_below) / max(len(rows), 1)
        if agreement > 0.60:
            signals.append({
                "type": "cross_col_positive",
                "col": f"{col_a}+{col_b}",
                "value": round(agreement * 100, 1),
                "label": f"{agreement*100:.0f}% of rows move together on {col_a} and {col_b}",
            })
        signals.append({
            "type": "two_metrics",
            "col": f"{col_a}+{col_b}",
            "value": len(numeric_cols),
            "label": f"two numeric columns available: {col_a} and {col_b}",
        })

    # ── Categorical signals ───────────────────────────────────────────────────
    if text_cols and numeric_cols:
        # The dimension the reader asked about, not simply the first text
        # column. On a result grouped by two things the calendar repeats once
        # per warehouse, and describing the result along that axis reports the
        # imbalance BETWEEN MONTHS -- four periods at 25% each, no leader --
        # for a question about warehouses. The narrative layer and the evidence
        # engine were taught this in c7d0689 and 4452268; this detector runs on
        # the same rows and must not disagree with them.
        #
        # Imported inside the function: core.response_builder reaches this
        # module through core.analysis_evidence, so a module-scope import here
        # would close the cycle.
        from core.response_builder import _narrative_label_column

        label_col  = _narrative_label_column(rows, text_cols)
        value_col  = numeric_cols[0]

        # ── One row per label before any share is computed ────────────────────
        # A result grouped by two things carries a row per label PER PERIOD.
        # Dividing one of those rows by a total summed over all of them
        # understates every share by the period count -- on live data a genuine
        # 37.5% leader came out at 9.4%, under the threshold below, so the
        # signal did not merely understate: it disappeared. The identical
        # defect in core.analysis_evidence.concentration_findings was fixed in
        # 4452268; this is the same rule, from the same place.
        #
        # collapse_rows_by_label refuses when the measure may not be summed --
        # a margin percentage, a stock balance -- and then there is no share to
        # report rather than a share of a total nobody can compute.
        from core.analysis_contract import collapse_rows_by_label

        collapsed = collapse_rows_by_label(rows, label_col, value_col)
        if collapsed:
            total = sum(value for _, value in collapsed)
            if total > 0 and len(collapsed) >= 2:
                leader_val, leader_num = max(collapsed, key=lambda pair: pair[1])
                leader_pct = leader_num / total
                if leader_pct > 0.35:
                    # Leader dominates — group imbalance
                    signals.append({
                        "type": "group_imbalance",
                        "col": label_col,
                        "value": round(leader_pct * 100, 1),
                        "leader": str(leader_val)[:40],
                        "label": f"{label_col} is dominated by one group ({leader_pct*100:.0f}% share)",
                    })

        # Temporal column
        if _is_temporal_col(label_col, rows):
            # ── The series, in period order, or not at all ────────────────────
            # first→last was read straight off the rows in arrival order, and
            # with no check that a period appears once. So "revenue by
            # warehouse for the last three months" -- a row per warehouse per
            # month -- compared the FIRST warehouse in the first month with the
            # LAST warehouse in the last month and called the difference a
            # trend. On a quarter where every warehouse was exactly flat it
            # reported a 25% decline, and that signal is not decoration: it
            # produces the follow-up chip inviting the reader to investigate
            # the decline, and it grounds the model's other suggestions.
            #
            # The shared rule, from the module that already refuses this:
            # ordered by period, and None when a period repeats or the labels
            # have no known ordering.
            from core.analysis_evidence import _values_in_period_order

            ordered = _values_in_period_order(rows, value_col, label_col)
            first_v = ordered[0] if ordered and len(ordered) >= 2 else None
            last_v = ordered[-1] if ordered and len(ordered) >= 2 else None
            if first_v is not None and last_v is not None and first_v != 0:
                pct = abs((last_v - first_v) / first_v * 100)
                # A flat band, because there was none: any first != last was
                # called "upward" or "downward", so a 0.7% drift produced a
                # follow-up reading "the trend in Revenue is downward" beside a
                # narrative saying "holding steady — no urgent action". Two
                # classifiers, one series, opposite verdicts. The narrative
                # layer (core/response_builder) already has a stable band;
                # this is the same idea at the same place in the reasoning.
                if pct < _FLAT_TREND_PCT:
                    signals.append({
                        "type": "flat_trend",
                        "col": label_col,
                        "value": round(pct, 1),
                        "direction": "flat",
                        "label": f"{value_col} is broadly flat ({pct:.1f}% change first→last)",
                    })
                else:
                    direction = "upward" if last_v > first_v else "downward"
                    signals.append({
                        "type": "temporal",
                        "col": label_col,
                        "value": round(pct, 1),
                        "direction": direction,
                        "label": f"{direction} trend in {value_col} ({pct:.0f}% change first→last)",
                    })

    # ── Result-size signals ───────────────────────────────────────────────────
    n = len(rows)
    if n <= 5:
        signals.append({"type": "small_result", "col": None, "value": n,
                         "label": f"small result set ({n} rows)"})
    elif n > 50:
        cat = text_cols[0] if text_cols else None
        signals.append({"type": "large_result", "col": cat, "value": n,
                         "label": f"large result ({n} rows) — segmentation may help"})

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# Template-based suggestions (zero LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _suggestion_pairs(
    signals: list[dict],
    col_names: list[str],
    col_types: dict[str, str] | None = None,
) -> list[dict]:
    """Up to 3 follow-ups, each as {"question", "label"}.

    `question` is English and is what the planner re-reads when the chip is
    clicked -- the same label/wire split the drill chips use, and the reason
    the French copy below can read naturally instead of having to survive
    question_normalizer.canonicalise word by word.

    `label` is what the reader sees, in the reader's language, with every
    column named the way the business names it.

    Both halves were wrong before. The twelve templates here were English
    f-strings with no access to the catalogue, and this is Tier 1: insight.py
    short-circuits before the LLM as soon as three of them fire, which a plain
    ranking of six rows does. So the commonest result in the product produced
    three English chips carrying raw warehouse codes, under a fully translated
    heading. Live output was:

        What makes the top 1 ORDER_CNT account for 80% of REVENUE_AMT?

    -- English, raw codes, and a COUNT column named as the entity being
    ranked, because the name heuristic below has no "_cnt" suffix. That last
    part is why col_types is threaded through: compute_data_brief already
    knows ORDER_CNT is numeric because it looked at the values, and a real
    type beats any spelling rule.
    """
    from core.i18n import t
    from core.schema_enrichment import display_label

    def _numeric(col: str) -> bool:
        declared = str((col_types or {}).get(col) or "").strip().lower()
        if declared:
            return declared in {"numeric", "number", "int", "integer",
                                "float", "decimal", "real", "money"}
        return _looks_numeric_col(col)

    text_cols = [c for c in col_names if not _numeric(c)]
    num_cols  = [c for c in col_names if _numeric(c)]

    # Index signals by type for quick lookup
    by_type: dict[str, dict] = {}
    for s in signals:
        if s["type"] not in by_type:
            by_type[s["type"]] = s

    pairs: list[dict] = []

    def _add(msg_id: str, **kw) -> None:
        """One chip, rendered twice: English for the wire, the reader's for the eye."""
        if len(pairs) >= 3:
            return
        question = t(msg_id, lang="en", **kw)
        if not question or any(p["question"] == question for p in pairs):
            return
        pairs.append({"question": question, "label": t(msg_id, **kw)})

    def _entity(default_id: str = "ui.followup.entity.rows") -> dict:
        """The thing being counted, as a label pair for the {entity} slot."""
        if text_cols:
            name = display_label(text_cols[0])
            return {"en": name, "loc": name}
        return {"en": t(default_id, lang="en"), "loc": t(default_id)}

    def _col(name: str) -> str:
        return display_label(name)

    # Priority order: most analytically interesting first

    # 1 — Outliers (very specific, high user interest)
    if "outlier_present" in by_type:
        s = by_type["outlier_present"]
        _add("ui.followup.outliers", column=_col(s["col"]))

    # 2 — Pareto / concentration
    if "pareto" in by_type:
        s = by_type["pareto"]
        entity = _entity()
        _add("ui.followup.pareto", n=s["n_top"], entity=entity["loc"],
             pct=f"{s['value']:.0f}", column=_col(s["col"]))

    # 3 — Right-skewed (high outliers driving the mean)
    if "skewed_right" in by_type:
        s = by_type["skewed_right"]
        _add("ui.followup.skew_right", entity=_entity()["loc"],
             column=_col(s["col"]))

    # 4 — Group imbalance
    if "group_imbalance" in by_type:
        s = by_type["group_imbalance"]
        num_col = num_cols[0] if num_cols else s["col"]
        _add("ui.followup.group_imbalance", leader=s["leader"],
             pct=f"{s['value']:.0f}", column=_col(num_col))

    # 5 — High variance (without outliers already shown)
    if "high_variance" in by_type and "outlier_present" not in by_type:
        s = by_type["high_variance"]
        _add("ui.followup.high_variance", column=_col(s["col"]))

    # 6 — Below-average gap
    if "below_avg_gap" in by_type:
        s = by_type["below_avg_gap"]
        _add("ui.followup.below_avg", entity=_entity()["loc"],
             column=_col(s["col"]))

    # 7 — Cross-column positive association
    if "cross_col_positive" in by_type:
        s = by_type["cross_col_positive"]
        parts = s["col"].split("+")
        if len(parts) == 2:
            _add("ui.followup.cross_col", first=_col(parts[0]),
                 second=_col(parts[1]))

    # 8 — Two metrics scatter (if cross_col not already added)
    if "two_metrics" in by_type and "cross_col_positive" not in by_type:
        s = by_type["two_metrics"]
        parts = s["col"].split("+")
        if len(parts) == 2:
            _add("ui.followup.two_metrics", first=_col(parts[0]),
                 second=_col(parts[1]))

    # 9 — Temporal trend
    if "temporal" in by_type:
        s = by_type["temporal"]
        num_col = _col(num_cols[0]) if num_cols else t("ui.followup.metric")
        direction_id = f"ui.followup.direction.{s.get('direction') or 'changing'}"
        _add("ui.followup.trend", column=num_col, direction=t(direction_id))
    elif "flat_trend" in by_type:
        num_col = _col(num_cols[0]) if num_cols else t("ui.followup.metric")
        _add("ui.followup.flat_trend", column=num_col)

    # 10 — Low variance (curiosity)
    if "low_variance" in by_type:
        s = by_type["low_variance"]
        _add("ui.followup.low_variance", column=_col(s["col"]))

    # 11 — Large result — suggest segmentation by a column NOT already in the result
    if "large_result" in by_type:
        cat = by_type["large_result"]["col"]
        # Only suggest breakdown if there are OTHER text columns not already
        # the grouping dim
        other_text = [c for c in text_cols if c != cat]
        if other_text:
            _add("ui.followup.segment", column=_col(other_text[0]))

    # 12 — Small result — suggest drilling deeper
    if "small_result" in by_type and num_cols:
        _add("ui.followup.more_detail",
             entity=_entity("ui.followup.entity.these")["loc"])

    return pairs[:3]


def template_suggestions(
    signals: list[dict],
    col_names: list[str],
    col_types: dict[str, str] | None = None,
) -> list[str]:
    """The English questions only -- the form the planner and the LLM's
    "already suggested, do not repeat" list both need."""
    return [p["question"] for p in _suggestion_pairs(signals, col_names, col_types)]


def _looks_numeric_col(col: str) -> bool:
    """Rough heuristic — col names ending with typical metric suffixes."""
    c = col.lower()
    return any(c.endswith(s) for s in (
        "_usd", "_amt", "_amount", "_count", "_qty", "_total", "_sum",
        "_avg", "_rate", "_pct", "_percent", "_score", "_num",
    )) or c in ("count", "total", "amount", "revenue", "charge", "quantity")


# ══════════════════════════════════════════════════════════════════════════════
# LLM fallback helper
# ══════════════════════════════════════════════════════════════════════════════

def format_signals_for_llm(signals: list[dict], col_names: list[str]) -> str:
    """
    Build a compact, privacy-safe string for the LLM suggestion fallback.

    Contains only signal labels and column names — zero raw row values.
    """
    if not signals:
        return ""
    labels = [s["label"] for s in signals[:8]]
    cols   = ", ".join(col_names[:8])
    return (
        f"Columns: {cols}\n"
        f"Statistical patterns detected:\n"
        + "\n".join(f"  - {l}" for l in labels)
    )
