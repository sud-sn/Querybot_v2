"""
core/anomaly_detection.py
──────────────────────────
Statistical anomaly detection as a first-class query route.

Answers questions like:
  "Which weeks had unusually high sick leave?"
  "Flag any months where attrition spiked abnormally"
  "Show me outlier employees by salary"
  "Are there any anomalies in the data?"
  "Find unusual transactions"

Design
──────
• Two methods, both pure Python (no external ML dependencies):
    Z-SCORE  — flags rows where |z| > threshold (default 2.5)
               Best for normally distributed data (sales, revenue)
    IQR      — flags rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
               Best for skewed/non-normal data (wait times, salaries)

• Auto-selects method based on distribution skewness of the data.
• Produces a flagged result with anomaly_score and anomaly_flag columns.
• Builds an LLM-safe brief (no raw values sent to LLM) for narrative generation.

Entry points
────────────
  detect_anomaly_intent(question) → bool
  detect_anomalies(rows, value_col, method, threshold) → AnomalyResult
  build_anomaly_brief(result) → dict   (safe for LLM)
  build_anomaly_sql_hint() → str       (prompt injection)
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from statistics import mean, stdev, median
from typing import Any, Literal


# ══════════════════════════════════════════════════════════════════════════════
# Detection patterns
# ══════════════════════════════════════════════════════════════════════════════

_ANOMALY_PATTERNS = [
    re.compile(r"\b(anomal(?:y|ies|ous)|outlier(?:s)?|unusual(?:ly)?|abnormal(?:ly)?|irregular(?:ly)?)\b", re.I),
    re.compile(r"\bspike[sd]?\b", re.I),
    re.compile(r"\b(flag|mark|highlight|identify)\s+(?:the\s+)?(?:unusual(?:ly)?|abnormal(?:ly)?|anomalous|outlier(?:s)?)\b", re.I),
    re.compile(r"\b(?:statistically\s+)?significant\s+(?:change|deviation|departure)\b", re.I),
    re.compile(r"\b(?:find|detect|show|which|where)\s+.{0,30}(?:anomal|unusual(?:ly)?|abnormal(?:ly)?|outlier(?:s)?)\b", re.I),
    re.compile(r"\b(?:exceed|outside)\s+(?:normal|typical|expected|average)\b", re.I),
    re.compile(r"\bnot\s+normal\b", re.I),
    re.compile(r"\bstandard\s+deviation\b", re.I),
    re.compile(r"\boutside\s+(?:the\s+)?(?:normal\s+)?(?:range|bounds|threshold)\b", re.I),
    re.compile(r"\bunusually\s+(?:high|low|large|small|elevated|depressed)\b", re.I),
]


def detect_anomaly_intent(question: str) -> bool:
    """Return True if the question asks for anomaly / outlier detection."""
    return any(p.search(question) for p in _ANOMALY_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly result model
# ══════════════════════════════════════════════════════════════════════════════

AnomalyMethod = Literal["zscore", "iqr", "auto"]


@dataclass
class AnomalyResult:
    method: str                        # "zscore" or "iqr"
    value_col: str
    threshold: float
    total_rows: int
    flagged_rows: int
    flagged_fraction: float            # fraction of rows flagged (0.0–1.0)
    mean_val: float | None
    std_val: float | None
    median_val: float | None
    iqr_lower: float | None
    iqr_upper: float | None
    rows: list[dict] = field(default_factory=list)   # all rows, with anomaly cols added
    flagged: list[dict] = field(default_factory=list)  # anomalous rows only


# ══════════════════════════════════════════════════════════════════════════════
# Core detection logic
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _compute_iqr_bounds(values: list[float]) -> tuple[float, float, float, float]:
    """Return (Q1, Q3, lower_fence, upper_fence)."""
    sv = sorted(values)
    n  = len(sv)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1  = sv[q1_idx]
    q3  = sv[q3_idx]
    iqr = q3 - q1
    return q1, q3, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _skewness(values: list[float]) -> float:
    """Approximate Fisher-Pearson skewness; returns 0.0 on insufficient data."""
    n = len(values)
    if n < 3:
        return 0.0
    m  = mean(values)
    try:
        s = stdev(values)
    except Exception:
        return 0.0
    if s == 0:
        return 0.0
    return sum(((v - m) / s) ** 3 for v in values) * n / ((n - 1) * (n - 2))


def detect_anomalies(
    rows: list[dict],
    value_col: str,
    method: AnomalyMethod = "auto",
    zscore_threshold: float = 2.5,
    include_all_rows: bool = True,
) -> AnomalyResult:
    """
    Flag anomalous rows in the result set.

    Parameters
    ──────────
    rows              — the data rows (list of dicts)
    value_col         — numeric column to analyse
    method            — "zscore", "iqr", or "auto" (auto selects by skewness)
    zscore_threshold  — |z| cutoff for Z-score method (default 2.5)
    include_all_rows  — if True, all rows are returned with anomaly cols;
                        if False, only flagged rows are returned

    Adds to each row:
      anomaly_score  — Z-score (z-method) or distance from nearest fence (iqr)
      anomaly_flag   — True / False
    """
    # Extract numeric values with original index
    indexed = [(i, _to_float(row.get(value_col))) for i, row in enumerate(rows)]
    valid   = [(i, v) for i, v in indexed if v is not None]

    if len(valid) < 4:
        # Not enough data for meaningful anomaly detection
        flagged_rows = [{**rows[i], "anomaly_score": None, "anomaly_flag": False}
                        for i, _ in indexed]
        return AnomalyResult(
            method=method, value_col=value_col, threshold=zscore_threshold,
            total_rows=len(rows), flagged_rows=0, flagged_fraction=0.0,
            mean_val=None, std_val=None, median_val=None,
            iqr_lower=None, iqr_upper=None,
            rows=flagged_rows, flagged=[],
        )

    values = [v for _, v in valid]
    m      = mean(values)
    med    = median(values)
    try:
        s = stdev(values)
    except Exception:
        s = 0.0

    # Auto-select method by skewness
    if method == "auto":
        skew = _skewness(values)
        method = "iqr" if abs(skew) > 1.0 else "zscore"

    iqr_lower = iqr_upper = q1 = q3 = None

    if method == "iqr":
        q1, q3, iqr_lower, iqr_upper = _compute_iqr_bounds(values)

    enriched_rows: list[dict] = []
    flagged: list[dict] = []

    for i, row in enumerate(rows):
        v = _to_float(row.get(value_col))
        if v is None:
            score = None
            flag  = False
        elif method == "zscore":
            score = round((v - m) / s, 3) if s > 0 else 0.0
            flag  = abs(score) >= zscore_threshold
        else:  # IQR
            if v < (iqr_lower or float("-inf")):
                score = round(iqr_lower - v, 4)
                flag  = True
            elif v > (iqr_upper or float("inf")):
                score = round(v - iqr_upper, 4)
                flag  = True
            else:
                score = 0.0
                flag  = False

        enriched = {**row, "anomaly_score": score, "anomaly_flag": flag}
        enriched_rows.append(enriched)
        if flag:
            flagged.append(enriched)

    flagged_count = len(flagged)

    return AnomalyResult(
        method=method,
        value_col=value_col,
        threshold=zscore_threshold,
        total_rows=len(rows),
        flagged_rows=flagged_count,
        flagged_fraction=round(flagged_count / len(rows), 3) if rows else 0.0,
        mean_val=round(m, 4),
        std_val=round(s, 4),
        median_val=round(med, 4),
        iqr_lower=round(iqr_lower, 4) if iqr_lower is not None else None,
        iqr_upper=round(iqr_upper, 4) if iqr_upper is not None else None,
        rows=enriched_rows if include_all_rows else flagged,
        flagged=flagged,
    )


def infer_value_col(rows: list[dict]) -> str:
    """
    Infer the primary numeric column from the first row.
    Returns the first column whose values are all (or mostly) numeric.
    """
    if not rows:
        return ""
    for col in rows[0].keys():
        numeric_count = sum(1 for r in rows if _to_float(r.get(col)) is not None)
        if numeric_count >= len(rows) * 0.75:
            return col
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# LLM brief builder (no raw data)
# ══════════════════════════════════════════════════════════════════════════════

def build_anomaly_brief(result: AnomalyResult) -> dict:
    """
    Build a statistical summary safe for LLM consumption.
    Raw row values are NEVER included — only aggregate statistics.
    """
    return {
        "method": result.method,
        "column_analysed": result.value_col,
        "total_rows": result.total_rows,
        "flagged_rows": result.flagged_rows,
        "flagged_fraction_pct": round(result.flagged_fraction * 100, 1),
        "mean": result.mean_val,
        "std_dev": result.std_val,
        "median": result.median_val,
        "iqr_lower_fence": result.iqr_lower,
        "iqr_upper_fence": result.iqr_upper,
        "zscore_threshold": result.threshold,
        "data_is_clean": result.flagged_rows == 0,
        "anomaly_rate": (
            "none" if result.flagged_rows == 0
            else "low" if result.flagged_fraction < 0.05
            else "moderate" if result.flagged_fraction < 0.15
            else "high"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SQL hint builder
# ══════════════════════════════════════════════════════════════════════════════

def build_anomaly_sql_hint(db_type: str = "azure_sql") -> str:
    """
    Return a SQL construction hint for anomaly detection queries.
    The LLM can embed this pattern when asked to detect anomalies in SQL.
    """
    is_tsql = db_type in ("azure_sql", "sql_server", "mssql")

    if is_tsql:
        stdev_fn = "STDEV"
    else:
        stdev_fn = "STDDEV"

    return (
        "ANOMALY DETECTION HINT:\n"
        "The user wants to flag statistically unusual rows. "
        "Use a Z-score window function:\n\n"
        "  WITH stats AS (\n"
        "    SELECT\n"
        "      *,\n"
        f"      AVG(CAST(metric_col AS FLOAT)) OVER () AS _mean,\n"
        f"      {stdev_fn}(CAST(metric_col AS FLOAT)) OVER () AS _std\n"
        "    FROM base_query\n"
        "  )\n"
        "  SELECT\n"
        "    *,\n"
        "    CASE WHEN _std > 0 THEN (metric_col - _mean) / _std ELSE 0 END AS anomaly_zscore,\n"
        "    CASE WHEN ABS((metric_col - _mean) / NULLIF(_std, 0)) >= 2.5\n"
        "         THEN 1 ELSE 0 END AS is_anomaly\n"
        "  FROM stats\n\n"
        "Replace 'metric_col' with the numeric column being analysed.\n"
        "Replace 'base_query' with the FROM clause and filters.\n"
        "A threshold of 2.5 standard deviations is standard; adjust to 2.0 for stricter flagging."
    )
