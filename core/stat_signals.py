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


_TEMPORAL_RE = re.compile(
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"q[1-4]|week|month|year|\b\d{4}[-/]\d{1,2}|\b20\d\d\b)",
    re.IGNORECASE,
)


def _is_temporal_col(col_name: str, rows: list[dict]) -> bool:
    sample = " ".join(str(r.get(col_name, "")) for r in rows[:6])
    return bool(_TEMPORAL_RE.search(sample))


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
        label_col  = text_cols[0]
        value_col  = numeric_cols[0]
        vals = [_to_float(r.get(value_col)) or 0.0 for r in rows]
        total = sum(vals)
        if total > 0 and len(rows) >= 2:
            leader_pct = max(vals) / total
            if leader_pct > 0.35:
                # Leader dominates — group imbalance
                leader_val = rows[vals.index(max(vals))].get(label_col, "")
                signals.append({
                    "type": "group_imbalance",
                    "col": label_col,
                    "value": round(leader_pct * 100, 1),
                    "leader": str(leader_val)[:40],
                    "label": f"{label_col} is dominated by one group ({leader_pct*100:.0f}% share)",
                })

        # Temporal column
        if _is_temporal_col(label_col, rows):
            # Detect trend direction from first→last value
            first_v = _to_float(rows[0].get(value_col))
            last_v  = _to_float(rows[-1].get(value_col))
            if first_v is not None and last_v is not None and first_v != 0:
                direction = "upward" if last_v > first_v else "downward"
                pct = abs((last_v - first_v) / first_v * 100)
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

def template_suggestions(
    signals: list[dict],
    col_names: list[str],
) -> list[str]:
    """
    Map detected signals to natural-language follow-up questions.

    Returns up to 3 questions.  Order is chosen for maximum analytical value:
    the most striking pattern comes first.
    """
    text_cols  = [c for c in col_names if not _looks_numeric_col(c)]
    num_cols   = [c for c in col_names if _looks_numeric_col(c)]

    # Index signals by type for quick lookup
    by_type: dict[str, dict] = {}
    for s in signals:
        if s["type"] not in by_type:
            by_type[s["type"]] = s

    suggestions: list[str] = []

    def _add(q: str) -> None:
        if q and len(suggestions) < 3 and q not in suggestions:
            suggestions.append(q)

    # Priority order: most analytically interesting first

    # 1 — Outliers (very specific, high user interest)
    if "outlier_present" in by_type:
        s = by_type["outlier_present"]
        _add(f"Show only the outliers in {s['col']} — values significantly above normal")

    # 2 — Pareto / concentration
    if "pareto" in by_type:
        s = by_type["pareto"]
        entity = text_cols[0] if text_cols else "rows"
        _add(f"What makes the top {s['n_top']} {entity} account for {s['value']:.0f}% of {s['col']}?")

    # 3 — Right-skewed (high outliers driving the mean)
    if "skewed_right" in by_type:
        s = by_type["skewed_right"]
        entity = text_cols[0] if text_cols else "rows"
        _add(f"Which {entity} are driving the high {s['col']} values?")

    # 4 — Group imbalance
    if "group_imbalance" in by_type:
        s = by_type["group_imbalance"]
        num_col = num_cols[0] if num_cols else s["col"]
        _add(f"Why does '{s['leader']}' have a {s['value']:.0f}% share in {num_col}?")

    # 5 — High variance (without outliers already shown)
    if "high_variance" in by_type and "outlier_present" not in by_type:
        s = by_type["high_variance"]
        entity = text_cols[0] if text_cols else "rows"
        _add(f"Who is significantly above and below average in {s['col']}?")

    # 6 — Below-average gap
    if "below_avg_gap" in by_type:
        s = by_type["below_avg_gap"]
        entity = text_cols[0] if text_cols else "rows"
        _add(f"Which {entity} are significantly below average in {s['col']}?")

    # 7 — Cross-column positive association
    if "cross_col_positive" in by_type:
        s = by_type["cross_col_positive"]
        parts = s["col"].split("+")
        if len(parts) == 2:
            _add(f"Show {parts[0]} vs {parts[1]} — do they move together?")

    # 8 — Two metrics scatter (if cross_col not already added)
    if "two_metrics" in by_type and "cross_col_positive" not in by_type:
        s = by_type["two_metrics"]
        parts = s["col"].split("+")
        if len(parts) == 2:
            _add(f"Show {parts[0]} vs {parts[1]} as a scatter chart")

    # 9 — Temporal trend
    if "temporal" in by_type:
        s = by_type["temporal"]
        num_col = num_cols[0] if num_cols else "the metric"
        direction = s.get("direction", "changing")
        _add(f"The trend in {num_col} is {direction} — what period drove the biggest change?")

    # 10 — Low variance (curiosity)
    if "low_variance" in by_type:
        s = by_type["low_variance"]
        _add(f"Why are {s['col']} values so uniform across all rows?")

    # 11 — Large result — suggest segmentation
    if "large_result" in by_type:
        cat = by_type["large_result"]["col"]
        if cat:
            _add(f"Break this down by {cat} to find the main patterns")

    # 12 — Small result — suggest drilling deeper
    if "small_result" in by_type and num_cols:
        entity = text_cols[0] if text_cols else "these"
        _add(f"Show more detail about each {entity}")

    return suggestions[:3]


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
