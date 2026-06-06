"""
core/result_transforms.py

Pure Python row transformations for the Contribution and Outlier chips
(Sprint D).

Design principles
─────────────────
• Both transforms operate entirely on the cached result rows already in
  memory — zero LLM calls, zero DB round-trips, instant response.
• Neither function mutates the input; both return new lists of dicts.
• Every edge case (empty rows, zero total, zero stdev, too few rows) is
  handled explicitly so the caller always gets a predictable return type.

Public API
──────────
  add_contribution_pct(rows, metric_col, …) → list[dict]
      Append a "Share %" column to each row.

  filter_outliers(rows, metric_col, …) → (list[dict], dict)
      Return rows above mean + threshold × stdev, plus a stats summary.

  describe_contribution_sql(metric_col, total) → str
      Human-readable pseudo-SQL comment for the trust block.

  describe_outlier_sql(metric_col, stats) → str
      Human-readable pseudo-SQL comment for the trust block.
"""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(val: Any) -> float | None:
    try:
        f = float(str(val).replace(",", ""))
        return None if f != f else f   # discard NaN
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# add_contribution_pct
# ══════════════════════════════════════════════════════════════════════════════

def add_contribution_pct(
    rows: list[dict],
    metric_col: str,
    *,
    pct_col_name: str = "",
    sort_desc: bool = True,
) -> tuple[list[dict], dict]:
    """
    Append a percentage-share column to every row.

    Each row's share = (row_value / total_value) × 100, rounded to 1 dp.
    Rows where the metric is None/non-numeric receive a share of 0.0.

    Parameters
    ──────────
    rows        : original result rows (not mutated)
    metric_col  : name of the column whose share should be computed
    pct_col_name: name for the new share column (defaults to
                  "{metric_col} Share %")
    sort_desc   : sort result by metric_col descending (default True)

    Returns
    ───────
    (transformed_rows, stats)

    stats keys: total, metric_col, pct_col, row_count, ok (bool)
    Returns (rows, {"ok": False, "reason": "..."}) on every failure path.
    """
    if not rows:
        return rows, {"ok": False, "reason": "no_rows"}
    if not metric_col:
        return rows, {"ok": False, "reason": "no_metric_col"}

    values = [_to_float(r.get(metric_col)) for r in rows]
    numeric_values = [v for v in values if v is not None]
    if not numeric_values:
        return rows, {"ok": False, "reason": "metric_not_numeric"}

    total = sum(numeric_values)
    if total == 0:
        return rows, {"ok": False, "reason": "zero_total"}

    col_name = pct_col_name or f"{metric_col} Share %"
    result: list[dict] = []
    for row, val in zip(rows, values):
        new_row = dict(row)
        if val is None:
            new_row[col_name] = 0.0
        else:
            new_row[col_name] = round(val / total * 100, 1)
        result.append(new_row)

    if sort_desc:
        result.sort(
            key=lambda r: _to_float(r.get(metric_col)) or 0.0,
            reverse=True,
        )

    return result, {
        "ok": True,
        "total": round(total, 2),
        "metric_col": metric_col,
        "pct_col": col_name,
        "row_count": len(result),
    }


# ══════════════════════════════════════════════════════════════════════════════
# filter_outliers
# ══════════════════════════════════════════════════════════════════════════════

def filter_outliers(
    rows: list[dict],
    metric_col: str,
    *,
    threshold: float = 1.5,
) -> tuple[list[dict], dict]:
    """
    Return rows where metric_col > mean + threshold × stdev.

    Requires at least 3 rows so that stdev is meaningful and the filter
    is not vacuously tight or vacuously loose.

    Parameters
    ──────────
    rows       : original result rows (not mutated)
    metric_col : column to apply the outlier filter to
    threshold  : stdev multiplier (default 1.5, i.e. ~top 7% in a normal
                 distribution — conservative enough to be honest)

    Returns
    ───────
    (filtered_rows, stats)

    stats keys: total_rows, outlier_rows, mean, std_dev, threshold_value,
                threshold, metric_col, ok (bool)
    Returns ([], {"ok": False, "reason": "..."}) on every failure path.
    """
    if not rows:
        return [], {"ok": False, "reason": "no_rows"}
    if not metric_col:
        return [], {"ok": False, "reason": "no_metric_col"}
    if len(rows) < 3:
        return [], {"ok": False, "reason": "too_few_rows",
                    "detail": "Outlier detection needs at least 3 rows."}

    values = [_to_float(r.get(metric_col)) for r in rows]
    numeric_pairs = [(r, v) for r, v in zip(rows, values) if v is not None]
    if len(numeric_pairs) < 3:
        return [], {"ok": False, "reason": "too_few_numeric_values"}

    numeric_vals = [v for _, v in numeric_pairs]
    avg = mean(numeric_vals)

    # stdev requires ≥2 values (already guaranteed above)
    std = stdev(numeric_vals)

    if std == 0:
        # All values are identical — nothing is an outlier
        return [], {
            "ok": False, "reason": "zero_variance",
            "detail": "All values are equal — no outliers exist.",
            "mean": round(avg, 2), "std_dev": 0.0,
        }

    cutoff = avg + threshold * std
    filtered = [r for r, v in numeric_pairs if v > cutoff]

    if not filtered:
        return [], {
            "ok": False, "reason": "no_outliers",
            "detail": (
                f"No rows exceed {metric_col} > "
                f"{avg:.0f} + {threshold}×{std:.0f} = {cutoff:.0f}. "
                "The values are relatively evenly distributed."
            ),
            "mean": round(avg, 2),
            "std_dev": round(std, 2),
            "threshold_value": round(cutoff, 2),
        }

    # Sort outliers by metric descending
    filtered.sort(key=lambda r: _to_float(r.get(metric_col)) or 0.0, reverse=True)

    return filtered, {
        "ok": True,
        "total_rows": len(rows),
        "outlier_rows": len(filtered),
        "mean": round(avg, 2),
        "std_dev": round(std, 2),
        "threshold_value": round(cutoff, 2),
        "threshold": threshold,
        "metric_col": metric_col,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Human-readable SQL descriptions (for trust block transparency)
# ══════════════════════════════════════════════════════════════════════════════

def describe_contribution_sql(metric_col: str, total: float) -> str:
    """Return a pseudo-SQL comment describing the contribution transform."""
    return (
        f"-- % share added from cached result rows\n"
        f"-- Formula: {metric_col} / {total:,.0f} (total) × 100"
    )


def describe_outlier_sql(metric_col: str, stats: dict) -> str:
    """Return a pseudo-SQL comment describing the outlier filter."""
    avg = stats.get("mean", 0)
    std = stats.get("std_dev", 0)
    cut = stats.get("threshold_value", 0)
    thr = stats.get("threshold", 1.5)
    total = stats.get("total_rows", 0)
    kept  = stats.get("outlier_rows", 0)
    return (
        f"-- Outlier filter applied to cached result rows\n"
        f"-- WHERE {metric_col} > {avg:.0f} + {thr}×{std:.0f} = {cut:.0f}"
        f"  ({kept} of {total} rows)"
    )
