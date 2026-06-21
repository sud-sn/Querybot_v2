"""
core/period_comparison.py

Prior-period comparison engine for the "vs prior period" chip (Sprint B).

Design principles
─────────────────
• Pure functions (detect_period_grain, compute_prior_period, etc.) are
  deterministic and free from LLM calls — tested independently.
• SQL rewrite is delegated to the LLM with a deliberately narrow prompt:
  the model is told to change *only* the date filter and return nothing else.
• The comparison narrative reuses the existing LLM insight infrastructure
  (`generate_insight`) by passing both briefs as drilldown context.
• Every step degrades gracefully — if the SQL rewrite fails, the date cannot
  be detected, or the prior-period query returns 0 rows, a clear user-facing
  message is returned rather than an error.

Entry point
───────────
  await generate_period_comparison(
      rows=..., question=..., original_sql=..., data_brief=...,
      db_cfg=..., known_tables=...,
      provider=..., model=..., api_key=...,
  )
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger("querybot.period_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Period detection (pure, no side effects)
# ══════════════════════════════════════════════════════════════════════════════

# Supported grains returned by detect_period_grain
GRAIN_MONTHLY   = "monthly"
GRAIN_QUARTERLY = "quarterly"
GRAIN_YEARLY    = "yearly"
GRAIN_UNKNOWN   = "unknown"

_YYYY_MM     = re.compile(r"^\d{4}[-/]\d{1,2}$")
_YYYY_ONLY   = re.compile(r"^\d{4}$")
_QN_YEAR     = re.compile(r"^Q[1-4][\s\-/]?\d{4}$", re.IGNORECASE)
_YEAR_QN     = re.compile(r"^\d{4}[\s\-/]?Q[1-4]$", re.IGNORECASE)
_MONTH_NAMES = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(uary|ruary|ch|il|e|y|ust|tember|ober|ember)?",
    re.IGNORECASE,
)
_MONTH_YEAR  = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[\s\-/]?\d{4}$",
    re.IGNORECASE,
)


def detect_period_grain(labels: list[str]) -> str:
    """
    Infer the time-series grain from the period labels returned in a result.

    Scans the first few labels; returns the first match.  Falls back to
    GRAIN_UNKNOWN when no pattern is recognised so callers can degrade
    gracefully rather than crash.

    Supported patterns
    ──────────────────
    monthly    : "2024-01", "2024/1", "Jan 2024", "Jan"
    quarterly  : "Q1 2024", "2024-Q1", "Q4-2023"
    yearly     : "2024", "2023"
    """
    for label in (labels or [])[:6]:
        s = str(label).strip()
        if _YYYY_MM.match(s) or _MONTH_YEAR.match(s) or _MONTH_NAMES.match(s):
            return GRAIN_MONTHLY
        if _QN_YEAR.match(s) or _YEAR_QN.match(s):
            return GRAIN_QUARTERLY
        if _YYYY_ONLY.match(s):
            return GRAIN_YEARLY
    return GRAIN_UNKNOWN


# ── Month-name helpers ────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

_MONTH_ABBR = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _parse_yyyymm(label: str) -> tuple[int, int] | None:
    """Return (year, month) from 'YYYY-MM' / 'YYYY/M' label, or None."""
    m = re.match(r"^(\d{4})[-/](\d{1,2})$", label.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _parse_quarter(label: str) -> tuple[int, int] | None:
    """Return (year, quarter) from 'Q1 2024' / '2024 Q1' / '2024-Q1', or None."""
    m = re.match(r"Q([1-4])[\s\-/]?(\d{4})", label.strip(), re.IGNORECASE)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = re.match(r"(\d{4})[\s\-/]?Q([1-4])", label.strip(), re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _fmt_yyyymm(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def _fmt_quarter(year: int, quarter: int) -> str:
    return f"Q{quarter} {year}"


# ══════════════════════════════════════════════════════════════════════════════
# Prior period computation (pure, no side effects)
# ══════════════════════════════════════════════════════════════════════════════

def compute_prior_period(
    first_label: str,
    last_label: str,
    grain: str,
) -> tuple[str, str]:
    """
    Compute the prior period boundaries (prior_first, prior_last) that have
    the same *duration* as the current window but are shifted back by one
    window length.

    Examples
    ────────
    monthly   "2025-01" → "2025-03"  ⟹  "2024-10" → "2024-12"  (3-month window)
    quarterly "Q1 2025" → "Q2 2025"  ⟹  "Q3 2024" → "Q4 2024"  (2-quarter window)
    yearly    "2024"    → "2025"      ⟹  "2022"    → "2023"      (2-year window)

    Returns ("", "") when the grain is unknown or parsing fails — callers
    must check for empty strings before proceeding.
    """
    if grain == GRAIN_MONTHLY:
        p1 = _parse_yyyymm(first_label)
        p2 = _parse_yyyymm(last_label)
        if p1 and p2:
            # Number of months in the window
            span = (p2[0] - p1[0]) * 12 + (p2[1] - p1[1]) + 1  # inclusive
            # Shift both endpoints back by `span` months
            pf = _shift_months(p1[0], p1[1], -span)
            pl = _shift_months(p2[0], p2[1], -span)
            return _fmt_yyyymm(*pf), _fmt_yyyymm(*pl)
        # Fall back: try month-name parsing
        pf_year, pf_month = _parse_month_name_label(first_label)
        pl_year, pl_month = _parse_month_name_label(last_label)
        if pf_month and pl_month:
            span = (pl_year - pf_year) * 12 + (pl_month - pf_month) + 1
            pf2 = _shift_months(pf_year, pf_month, -span)
            pl2 = _shift_months(pl_year, pl_month, -span)
            return _fmt_yyyymm(*pf2), _fmt_yyyymm(*pl2)

    elif grain == GRAIN_QUARTERLY:
        q1 = _parse_quarter(first_label)
        q2 = _parse_quarter(last_label)
        if q1 and q2:
            span = (q2[0] - q1[0]) * 4 + (q2[1] - q1[1]) + 1  # inclusive
            pf = _shift_quarters(q1[0], q1[1], -span)
            pl = _shift_quarters(q2[0], q2[1], -span)
            return _fmt_quarter(*pf), _fmt_quarter(*pl)

    elif grain == GRAIN_YEARLY:
        m1 = re.search(r"\d{4}", first_label)
        m2 = re.search(r"\d{4}", last_label)
        if m1 and m2:
            y1, y2 = int(m1.group()), int(m2.group())
            span = y2 - y1 + 1
            return str(y1 - span), str(y2 - span)

    return "", ""


def _shift_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = (year * 12 + month - 1) + delta
    return total // 12, (total % 12) + 1


def _shift_quarters(year: int, quarter: int, delta: int) -> tuple[int, int]:
    total = (year * 4 + quarter - 1) + delta
    return total // 4, (total % 4) + 1


def _parse_month_name_label(label: str) -> tuple[int, int]:
    """Return (year, month) from 'Jan 2025' / 'January' labels.  Year defaults to 0."""
    s = label.strip().lower()
    m = re.match(r"^([a-z]+)\s*(\d{4})?", s)
    if m:
        month = _MONTH_MAP.get(m.group(1)[:3], 0)
        year = int(m.group(2)) if m.group(2) else 0
        return year, month
    return 0, 0


# ══════════════════════════════════════════════════════════════════════════════
# LLM prompt builders (pure, no side effects)
# ══════════════════════════════════════════════════════════════════════════════

def build_prior_period_rewrite_prompt(
    original_sql: str,
    current_first: str,
    current_last: str,
    prior_first: str,
    prior_last: str,
    date_col_hint: str = "",
) -> tuple[str, str]:
    """
    Build the (system, user) prompt pair for rewriting the original SQL to
    target the prior period.

    The system prompt is deliberately minimal: the LLM's only job is to
    change the date filter.  Everything else must remain identical.
    """
    hint_block = (
        f"\nDate/time column hint from schema: {date_col_hint}"
        if date_col_hint else ""
    )
    system = (
        "You are a SQL date-range modifier. "
        "Your ONLY task is to rewrite the date filter in the SQL below to target a different time period. "
        "RULES:\n"
        "1. Change ONLY the date or time period filter(s) — do NOT alter SELECT columns, GROUP BY, JOINs, metrics, or aggregations.\n"
        "2. Return ONLY the modified SQL — no explanation, no markdown fences.\n"
        "3. If you cannot safely rewrite the date filter without changing query logic, return exactly: CANNOT_REWRITE\n"
        "4. Preserve all aliases, table qualifiers, and bracket notation exactly as in the original."
    )
    user = (
        f"Original SQL:\n{original_sql}\n\n"
        f"Current period in the result: {current_first} to {current_last}\n"
        f"Target period (prior): {prior_first} to {prior_last}{hint_block}\n\n"
        f"Rewrite the SQL so it retrieves data for the target period."
    )
    return system, user


def build_period_comparison_narrative_prompt(
    current_brief: dict,
    prior_brief: dict,
    question: str,
    current_label: str,
    prior_label: str,
    business_context: str = "",
) -> tuple[str, str]:
    """
    Build the (system, user) prompt for comparing two period data briefs.

    Both briefs are statistical summaries — no raw rows are included.
    """
    from core.insight import _format_brief_for_prompt  # local import to avoid circular

    biz_block = ""
    if business_context:
        biz_block = (
            "\n\nBUSINESS CONTEXT:\n"
            + business_context[:1800]
            + "\n\nUse this context to interpret column names and metrics.\n"
        )

    system = (
        "You are a senior business analyst comparing two time periods for a non-technical user.\n\n"
        "RULES:\n"
        "1. Use ONLY the numbers in the two data briefs below — never invent values.\n"
        "2. Translate column names to plain English.\n"
        "3. State whether the metric improved, declined, or was stable compared to the prior period.\n"
        "4. Keep language direct — no filler phrases.\n"
        "5. Highlight the most significant change (absolute AND percentage if available).\n"
        "6. Never claim causation unless the data directly supports it.\n"
        + biz_block
        + "\nRESPONSE FORMAT (required):\n"
        "HEADLINE: [one sentence — what changed and by how much]\n"
        "BODY: [2–3 sentences covering direction, magnitude, and what dimension drove the difference]\n"
        "DETAIL:\n- [metric value in current period]\n- [metric value in prior period]\n- [absolute and % change]\n"
        "NEXT: [one specific question to investigate next]\n"
    )
    user = (
        f"Question: {question}\n\n"
        f"CURRENT PERIOD ({current_label}):\n{_format_brief_for_prompt(current_brief)}\n\n"
        f"PRIOR PERIOD ({prior_label}):\n{_format_brief_for_prompt(prior_brief)}\n\n"
        "Compare these two periods."
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# SQL extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_date_col_hint(semantic_plan: dict | None) -> str:
    """
    Extract a date column hint from the semantic plan's date_dimension fields.
    Returns an empty string when the plan has no date role information.
    """
    if not semantic_plan or not semantic_plan.get("enabled"):
        return ""
    for field in (semantic_plan.get("fields") or []):
        if field.get("role") == "date_dimension":
            hint_parts = []
            if field.get("source_key_column"):
                hint_parts.append(f"fact FK: {field['source_key_column']}")
            if field.get("column"):
                hint_parts.append(f"dim key: {field['column']}")
            if field.get("table"):
                hint_parts.append(f"dim table: {field['table']}")
            return ", ".join(hint_parts) if hint_parts else ""
    return ""


def _clean_sql_response(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from an LLM SQL response."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0].strip()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

async def generate_period_comparison(
    *,
    rows: list[dict],
    question: str,
    original_sql: str,
    data_brief: dict,
    db_cfg: dict,
    known_tables: set[str] | None = None,
    provider: str,
    model: str,
    api_key: str,
    business_context: str = "",
    semantic_plan: dict | None = None,
    query_executor=None,
    **extra_kwargs,
) -> dict:
    """
    Fetch the prior period for the current result and generate a structured
    side-by-side comparison narrative.

    Pipeline
    ────────
    1. Read period info from data_brief.time_series
    2. Detect grain (monthly / quarterly / yearly)
    3. Compute prior-period boundaries
    4. Ask LLM to rewrite original SQL for the prior period
    5. Validate + execute the rewritten SQL against the live DB
    6. Compute data_brief for the prior-period result
    7. Ask LLM to compare both briefs and produce a structured narrative
    8. Return an assistant_analysis response (same shape as other actions)

    Returns a complete ``assistant_analysis`` dict on success, or a
    descriptive fallback dict when any step cannot proceed.
    """
    from core.insight import compute_data_brief
    from core.llm import llm_complete
    from core.schema import run_query
    from core.validator import validate_sql

    def _fallback(reason: str, suggestion: str = "") -> dict:
        return {
            "type": "assistant_analysis",
            "action": "compare_prior",
            "title": "Prior period comparison",
            "headline": "Could not fetch prior period data.",
            "body": reason,
            "bullets": [suggestion] if suggestion else [],
            "next_step": (
                suggestion or
                "Try asking directly: "
                "\"Show [metric] for [period A] vs [period B]\""
            ),
            "data_brief": data_brief,
            "source_question": question,
            "mode": data_brief.get("mode", "table"),
        }

    # ── Step 1: Extract period info from existing brief ──────────────────────
    ts = data_brief.get("time_series") or {}
    first_label = ts.get("first_period", "")
    last_label  = ts.get("last_period", "")

    if not first_label or not last_label:
        return _fallback(
            "Prior period comparison requires a time-series result with period labels.",
            "Try a query that groups by month, quarter, or year first.",
        )

    # ── Step 2: Detect grain ─────────────────────────────────────────────────
    grain = detect_period_grain([first_label, last_label])
    if grain == GRAIN_UNKNOWN:
        return _fallback(
            f"Could not recognise the period format in labels "
            f"'{first_label}' → '{last_label}'.",
            "This works best with YYYY-MM, Q1/Q2, or yearly labels.",
        )

    # ── Step 3: Compute prior-period boundaries ──────────────────────────────
    prior_first, prior_last = compute_prior_period(first_label, last_label, grain)
    if not prior_first or not prior_last:
        return _fallback(
            f"Could not compute the prior period for '{first_label}' → '{last_label}'."
        )

    log.info(
        "compare_prior: current=%s→%s  prior=%s→%s  grain=%s",
        first_label, last_label, prior_first, prior_last, grain,
    )

    # ── Step 4: LLM rewrites the SQL for the prior period ───────────────────
    date_col_hint = _extract_date_col_hint(semantic_plan)
    sys_prompt, usr_prompt = build_prior_period_rewrite_prompt(
        original_sql, first_label, last_label,
        prior_first, prior_last, date_col_hint,
    )

    try:
        raw_sql, _, _ = await llm_complete(
            sys_prompt, usr_prompt,
            provider, model, api_key,
            max_tokens=600,
            temperature=0.0,
            **extra_kwargs,
        )
    except Exception as exc:
        log.warning("compare_prior: LLM rewrite call failed: %s", exc)
        return _fallback(
            "The SQL rewriter encountered an error.",
            f"Try asking: \"Show the same metric for {prior_first} to {prior_last}\".",
        )

    prior_sql = _clean_sql_response(raw_sql)
    if not prior_sql or "CANNOT_REWRITE" in prior_sql.upper():
        return _fallback(
            "The original query uses a date filter that could not be automatically "
            "shifted to the prior period.",
            f"Try asking: \"Show [metric] for {prior_first} to {prior_last}\".",
        )

    # ── Step 5: Validate the rewritten SQL ──────────────────────────────────
    try:
        ok, reason, _code = validate_sql(
            prior_sql,
            known_tables or set(),
            db_cfg.get("db_type", "azure_sql"),
        )
        if not ok:
            log.warning("compare_prior: rewritten SQL failed validation: %s", reason)
            return _fallback(
                "The rewritten SQL for the prior period did not pass validation.",
                f"Try asking: \"Show [metric] for {prior_first} to {prior_last}\".",
            )
    except Exception as exc:
        log.warning("compare_prior: validation error: %s", exc)
        return _fallback("SQL validation error while preparing the prior period query.")

    # ── Step 6: Execute against the live DB ─────────────────────────────────
    try:
        if query_executor:
            governed = query_executor(db_cfg, prior_sql)
            prior_rows = governed.rows
            prior_sql = governed.sql
        else:
            prior_rows = run_query(
                db_cfg.get("credentials") or db_cfg,
                db_cfg.get("db_type", "azure_sql"),
                prior_sql,
            )
    except Exception as exc:
        log.warning("compare_prior: DB execution failed: %s", exc)
        return _fallback(
            f"The prior period query executed but encountered a database error: "
            f"{str(exc)[:120]}",
        )

    if not prior_rows:
        return _fallback(
            f"No data found for the prior period ({prior_first} to {prior_last}). "
            "This period may not have records in the database.",
            f"Try asking: \"Show [metric] for {prior_first} to {prior_last}\" to verify.",
        )

    # ── Step 7: Compute prior-period brief ───────────────────────────────────
    prior_brief = compute_data_brief(prior_rows, question)
    current_label_str = f"{first_label} – {last_label}" if first_label != last_label else first_label
    prior_label_str   = f"{prior_first} – {prior_last}" if prior_first != prior_last else prior_first

    # ── Step 8: Generate comparison narrative ───────────────────────────────
    sys_narr, usr_narr = build_period_comparison_narrative_prompt(
        data_brief, prior_brief,
        question,
        current_label_str,
        prior_label_str,
        business_context,
    )

    try:
        raw_narr, _, _ = await llm_complete(
            sys_narr, usr_narr,
            provider, model, api_key,
            max_tokens=500,
            temperature=0.3,
            **extra_kwargs,
        )
    except Exception as exc:
        log.warning("compare_prior: narrative LLM call failed: %s", exc)
        raw_narr = ""

    parsed = _parse_narrative(raw_narr, data_brief, prior_brief,
                               current_label_str, prior_label_str)

    return {
        "type": "assistant_analysis",
        "action": "compare_prior",
        "title": f"vs {prior_label_str}",
        "headline":  parsed["headline"],
        "body":      parsed["body"],
        "bullets":   parsed["bullets"],
        "next_step": parsed["next_step"],
        "secondary": f"Prior period: {prior_label_str}  ·  Current: {current_label_str}",
        "data_brief": data_brief,
        "prior_brief": prior_brief,
        "source_question": question,
        "mode": data_brief.get("mode", "time_series"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Narrative parsing + deterministic fallback
# ══════════════════════════════════════════════════════════════════════════════

def _parse_narrative(
    raw: str,
    current_brief: dict,
    prior_brief: dict,
    current_label: str,
    prior_label: str,
) -> dict:
    """
    Parse the structured LLM narrative.  Falls back to a deterministic
    stat-driven narrative when the LLM response is missing or unparseable.
    """
    result: dict[str, Any] = {
        "headline": "", "body": "", "bullets": [], "next_step": "",
    }

    if raw:
        for line in raw.splitlines():
            s = line.strip()
            if s.upper().startswith("HEADLINE:"):
                result["headline"] = s[9:].strip()
            elif s.upper().startswith("BODY:"):
                result["body"] = s[5:].strip()
            elif s.upper().startswith("NEXT:"):
                result["next_step"] = s[5:].strip()
            elif s.startswith(("- ", "• ")):
                result["bullets"].append(s.lstrip("-• ").strip())

    # Deterministic fallback when LLM produced nothing useful
    if not result["headline"]:
        result["headline"] = _deterministic_headline(
            current_brief, prior_brief, current_label, prior_label
        )
    if not result["body"]:
        result["body"] = (
            f"Comparing {current_label} with the prior period ({prior_label})."
        )
    if not result["bullets"]:
        result["bullets"] = _deterministic_bullets(current_brief, prior_brief)

    return result


def _deterministic_headline(
    current: dict,
    prior: dict,
    current_label: str,
    prior_label: str,
) -> str:
    """Generate a one-line headline from the two briefs without an LLM."""
    c_ts = current.get("time_series") or {}
    p_ts = prior.get("time_series") or {}
    c_val = c_ts.get("last_value") or _first_numeric_total(current)
    p_val = p_ts.get("last_value") or _first_numeric_total(prior)
    if c_val is not None and p_val is not None and p_val != 0:
        pct = round((c_val - p_val) / abs(p_val) * 100, 1)
        direction = "up" if pct > 0 else "down"
        sign = "+" if pct > 0 else ""
        return (
            f"Result is {direction} {abs(pct):.1f}% ({sign}{pct:.1f}%) "
            f"compared to {prior_label}."
        )
    return f"Comparing {current_label} vs {prior_label}."


def _deterministic_bullets(current: dict, prior: dict) -> list[str]:
    bullets = []
    for brief, label in ((current, "Current"), (prior, "Prior")):
        ts = brief.get("time_series") or {}
        total = _first_numeric_total(brief)
        if total is not None:
            bullets.append(f"{label} period total: {total:,.2f}")
        elif ts.get("last_value") is not None:
            bullets.append(f"{label} period last value: {ts['last_value']:,.2f}")
    return bullets[:3]


def _first_numeric_total(brief: dict) -> float | None:
    """Return the first numeric column's total from a data brief, or None."""
    for _col, stats in (brief.get("numeric_summaries") or {}).items():
        if isinstance(stats, dict) and stats.get("total") is not None:
            return float(stats["total"])
    return None
