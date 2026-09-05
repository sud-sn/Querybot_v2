"""
core/insight.py

Dynamic LLM-powered analysis engine.

Design principle: the LLM never receives the ROW SET. It receives a "data
brief" — aggregated summaries, trend metrics, distribution shapes and category
breakdowns.

Stated precisely, because the older wording ("NEVER sees raw data") was not
true and invited people to widen the brief on the strength of it: the brief
DOES carry individual cell values where the aggregate IS a cell — a
single-value result passes its number, and a category breakdown passes its
labels. What it never carries is the rows themselves, and for a regulated
tenant it is not built at all: `result_llm_features_allowed` gates every one
of these entry points and returns a static response instead. The LLM interprets these patterns and
generates a business-language narrative.

Two entry points:

1. generate_insight()
   Called when a user clicks an action button (explain, analyze, compare,
   predict) or when the system detects a "why" follow-up question.
   Takes the result rows + context, computes a data brief, sends it to
   the LLM, returns a structured insight response.

2. generate_drilldown_insight()
   Called when the user asks "why did X change" — runs additional
   drill-down queries against the DB, computes briefs from each,
   then sends the combined brief to the LLM for causal interpretation.
"""

import logging
import math
import re
from statistics import mean, median, stdev
from typing import Any, Optional

from core.i18n import t as _t

log = logging.getLogger("querybot.insight")


# ══════════════════════════════════════════════════════════════════════════════
# Analytical question detection
# ══════════════════════════════════════════════════════════════════════════════

_WHY_PATTERNS = [
    r"\bwhy\b",
    r"\breason\b",
    r"\bcause[sd]?\b",
    r"\bexplain\b",
    r"\bwhat.*(drove|driving|caused|behind)\b",
    r"\b(drop|decline|decrease|fall|fell|dip)\b.*\b(why|reason|cause)\b",
    r"\bwhy\b.*(drop|decline|decrease|increase|rise|spike|jump|change)\b",
    r"\bwhat\s+happened\b",
    r"\bwhat\s+changed\b",
    r"\bbreak\s*down\b.*\bwhy\b",
    r"\banalyze\b",
    r"\banalysis\b",
    r"\binsight\b",
]


def is_insight_question(question: str) -> bool:
    """Return True if the question is asking for causal/analytical insight
    rather than a simple data retrieval."""
    q = question.lower()
    return any(re.search(p, q) for p in _WHY_PATTERNS)


# Strict subset of _WHY_PATTERNS: only explicit causal wording. Used by the
# query pipeline to decide whether a FRESH query deserves an appended why-
# analysis (an extra LLM round trip on every match, so "analyze"/"explain"/
# "insight" — which often just mean "show me" — are deliberately excluded;
# those still get the full treatment on the cached-result follow-up path).
_CAUSAL_STRICT_PATTERNS = [
    r"\bwhy\b",
    r"\breason(s)?\s+(for|behind|why)\b",
    r"\bwhat.*(drove|driving|caused|behind)\b",
    r"\broot\s+cause\b",
    r"\bwhat\s+happened\s+(to|with)\b",
    r"\bwhat\s+changed\b",
]


def is_causal_question(question: str) -> bool:
    """Return True only for explicitly causal questions ("why did X drop",
    "what drove Y") — the ones a plain data table cannot answer by itself."""
    q = question.lower()
    return any(re.search(p, q) for p in _CAUSAL_STRICT_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# Unified analytical intent detection
# Delegates to the specialised modules; keeps query_pipeline imports clean.
# ══════════════════════════════════════════════════════════════════════════════

def detect_analytical_intents(question: str) -> dict:
    """
    Run all analytical intent detectors against the question and return a
    summary dict.  The pipeline uses this to decide which analytics route
    to activate.

    Keys returned
    ─────────────
    window          WindowIntent | None   — rolling avg, running total, rank, delta
    relative_date   RelativeDateIntent | None — last N days / this week vs last week
    contribution    bool                  — % share / mix analysis
    anomaly         bool                  — outlier / spike detection
    multi_period    MultiPeriodIntent | None — 2+ named periods to compare
    budget_vs_actual bool                 — variance to budget/target/plan
    cohort          bool                  — cohort retention analysis
    correlation     bool                  — correlation / scatter between metrics
    pivot           bool                  — pivot / cross-tab table
    funnel          bool                  — funnel / conversion stage analysis
    forecast        bool                  — trend forecast / projection
    fiscal          bool                  — fiscal year / FY period reference
    histogram       bool                  — distribution / frequency histogram
    boxplot         bool                  — quartile / box-plot analysis
    whatif          bool                  — what-if / scenario analysis
    """
    from core.window_analytics import detect_window_intent
    from core.relative_date_range import detect_relative_date_question
    from core.contribution_analysis import detect_contribution_intent
    from core.anomaly_detection import detect_anomaly_intent
    from core.multi_period import detect_multi_period_intent
    from core.budget_vs_actual import detect_bva_intent
    from core.cohort_analysis import detect_cohort_intent
    from core.correlation_analysis import detect_correlation_intent
    from core.pivot_table import detect_pivot_intent
    from core.funnel_analysis import detect_funnel_intent
    from core.forecast import detect_forecast_intent
    from core.fiscal_calendar import detect_fiscal_intent
    from core.distribution_analysis import detect_histogram_intent, detect_boxplot_intent
    from core.whatif import detect_whatif_intent

    contribution = detect_contribution_intent(question)
    # Composition and histogram are mutually exclusive.  Without this guard,
    # "distribution of revenue by state" was converted into row-level bins,
    # discarding the requested state grain before chart selection.
    histogram = detect_histogram_intent(question) and not contribution

    return {
        "window":           detect_window_intent(question),
        "relative_date":    detect_relative_date_question(question),
        "contribution":     contribution,
        "anomaly":          detect_anomaly_intent(question),
        "multi_period":     detect_multi_period_intent(question),
        "budget_vs_actual": detect_bva_intent(question),
        "cohort":           detect_cohort_intent(question),
        "correlation":      detect_correlation_intent(question),
        "pivot":            detect_pivot_intent(question),
        "funnel":           detect_funnel_intent(question),
        "forecast":         detect_forecast_intent(question),
        "fiscal":           detect_fiscal_intent(question),
        "histogram":        histogram,
        "boxplot":          detect_boxplot_intent(question),
        "whatif":           detect_whatif_intent(question),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Data brief computation — NO raw data leaves this module
# ══════════════════════════════════════════════════════════════════════════════

def _to_float(val: Any) -> Optional[float]:
    try:
        return float(str(val).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _safe_pct(a: float, b: float) -> Optional[float]:
    if a == 0:
        return None
    return round(((b - a) / abs(a)) * 100, 2)


def _classify_columns(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Split columns into numeric and text."""
    if not rows:
        return [], []
    numeric, text = [], []
    for h in rows[0].keys():
        is_num = True
        seen = False
        for r in rows:
            v = r.get(h)
            if v is None or v == "":
                continue
            seen = True
            if _to_float(v) is None:
                is_num = False
                break
        if is_num and seen:
            numeric.append(h)
        else:
            text.append(h)
    return numeric, text


def _looks_temporal(labels: list[str]) -> bool:
    sample = " ".join(v.lower() for v in labels[:8] if v)
    # Word-boundary tokens — short abbreviations that could false-positive
    # on business words (e.g. "marketing" contains "mar") so we use \b
    wb_tokens = [
        "jan", "feb", "mar", "apr", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec", "q1", "q2", "q3", "q4",
    ]
    # Substring tokens — full words or longer tokens safe from false positives
    substr_tokens = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "week", "month", "quarter", "year", "date", "day",
    ]
    if bool(re.search(r"\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b", sample)):
        return True
    if any(tok in sample for tok in substr_tokens):
        return True
    if any(re.search(r"\b" + tok + r"\b", sample) for tok in wb_tokens):
        return True
    return False


def _is_sensitive_field(name: str) -> bool:
    """
    Return True if a column name suggests it contains personally identifiable
    or sensitive data that should be redacted before being sent to an LLM.

    Two normalization forms are checked:
      • underscore form  — non-alphanumeric chars replaced with _
        catches: first_name, EMPLOYEE_ID, phone_number
      • bare form        — underscores also removed
        catches: CamelCase without separators: FirstName, PatientID, PostalCode
    """
    _norm     = re.sub(r"[^a-z0-9]+", "_", (name or "").lower())
    from core.compliance.classifier import is_non_person_catalog_name
    if is_non_person_catalog_name(_norm):
        return False
    _bare     = _norm.replace("_", "")        # "postalcode", "medicalrecordnumber"
    keywords  = [
        # Personal names
        "name",
        # Contact details
        "email", "phone", "mobile", "telephone", "fax", "address",
        # Government / national identifiers
        "ssn", "national_id", "nationalid",
        "national_insurance", "nationalinsurance",
        "nhs",
        "mrn",
        "passport", "tax_id", "taxpayer",
        "driving_licence", "driving_license", "license_no",
        # Internal identity references
        "employee_id", "employeeid",
        "employee_number", "employeenumber",
        "staff_id", "staffid",
        "user_id", "userid",
        "person_id", "personid",
        "id_number", "idnumber",
        "member_id", "memberid",
        "patient_id", "patientid",
        "account_no", "accountno",
        "account_number", "accountnumber",
        # Dates of birth / age
        "dob", "birth", "birthdate",
        # Location PII
        "postcode", "postal_code", "postalcode",
        "zip_code", "zipcode",
        # Medical record numbers
        "medical_record", "medicalrecord",
        "record_number", "recordnumber",
        "encounter_id", "encounterid",
    ]
    return any(kw in _norm or kw in _bare for kw in keywords)


def _display_label(label: str, label_col: str) -> str:
    if _is_sensitive_field(label_col):
        return "redacted segment"
    return label


def compute_data_brief(
    rows: list[dict],
    question: str = "",
    *,
    result_scope: dict | None = None,
    context: dict | None = None,
) -> dict:
    """
    Compute a statistical data brief from result rows.

    ⚠️  THIS BRIEF CONTAINS REAL DATA VALUES. The previous docstring claimed it
    "NEVER contains raw row values"; that was false and dangerously so, because
    every new consumer read it as a safety guarantee. It carries, among others:
      • brief["value"]                      — the literal single-cell result
      • category_breakdown.top_5/bottom_3   — actual dimension labels + amounts
      • time_series from_period/to_period   — actual period labels
      • numeric min/max/mean/total          — derived from real rows

    The only filter is _is_sensitive_field(), which matches on COLUMN NAME
    keywords, so a value in a column named PRODUCT_NAME or DRUG survives intact.

    Consequently every consumer that hands this to an LLM MUST first check
    core.compliance.policy_engine.result_llm_features_allowed(account_id).
    Today's consumers — the why-insight narration, the drilldown planner,
    period comparison and follow-up suggestions — all do. If you add a fifth,
    gate it too; the brief itself is not a safe boundary.
    """
    from core.metric_semantics import detect_metric_semantics
    from core.response_builder import infer_result_scope, summarize_result_context

    ctx = context or summarize_result_context(rows, question)
    scope = result_scope or ctx.get("result_scope") or infer_result_scope(rows, question, mode=ctx.get("mode", "table"))

    if not rows:
        return {
            "row_count": 0,
            "mode": "empty",
            "summary": "No data returned for this query.",
            "result_scope": scope,
            "metric_semantics": detect_metric_semantics(question, context=ctx),
        }

    numeric_cols, text_cols = _classify_columns(rows)
    brief: dict[str, Any] = {
        "row_count": len(rows),
        "column_count": len(rows[0]),
        "columns": {
            col: "numeric" for col in numeric_cols
        },
        "mode": "table",
        "result_scope": scope,
        "metric_semantics": detect_metric_semantics(question, context=ctx),
    }
    brief["columns"].update({col: "text" for col in text_cols})

    # ── Single-value result ──────────────────────────────────────────────────
    if len(rows) == 1 and len(rows[0]) == 1:
        col = next(iter(rows[0].keys()))
        brief["mode"] = "single_value"
        brief["value_column"] = col
        brief["value"] = _to_float(rows[0][col]) or str(rows[0][col])
        return brief

    # ── Numeric summaries ────────────────────────────────────────────────────
    numeric_summaries = {}
    for col in numeric_cols:
        values = [_to_float(r.get(col)) for r in rows]
        values = [v for v in values if v is not None]
        if not values:
            continue
        s = {
            "count": len(values),
            "total": round(sum(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "mean": round(mean(values), 2),
            "median": round(median(values), 2),
        }
        if len(values) >= 3:
            s["std_dev"] = round(stdev(values), 2)
        # Concentration — what % does top-3 represent?
        sorted_vals = sorted(values, reverse=True)
        if len(sorted_vals) >= 3 and s["total"] > 0:
            top3_pct = round(sum(sorted_vals[:3]) / s["total"] * 100, 1)
            s["top_3_concentration_pct"] = top3_pct
        numeric_summaries[col] = s

    brief["numeric_summaries"] = numeric_summaries

    # ── Category breakdowns ──────────────────────────────────────────────────
    if text_cols and numeric_cols:
        label_col = text_cols[0]
        value_col = numeric_cols[0]
        # Same formatting the table and KPI apply, so a trend sentence does
        # not print 2026-01-01 beside a headline that says 2026-01.
        from core.response_builder import narrative_period_labels
        labels = narrative_period_labels([r.get(label_col, "") for r in rows])
        values = [_to_float(r.get(value_col)) or 0.0 for r in rows]
        paired = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)

        cat_breakdown = {
            "label_column": label_col,
            "value_column": value_col,
            "category_count": len(set(labels)),
            "top_5": [{"label": _display_label(l, label_col), "value": round(v, 2)} for l, v in paired[:5]],
            "bottom_3": [{"label": _display_label(l, label_col), "value": round(v, 2)} for l, v in paired[-3:]],
            "labels_redacted": _is_sensitive_field(label_col),
        }

        total = sum(values)
        if total > 0 and len(paired) >= 2:
            leader_pct = round(paired[0][1] / total * 100, 1)
            cat_breakdown["leader_share_pct"] = leader_pct
            cat_breakdown["leader_vs_runner_up_gap"] = round(
                paired[0][1] - paired[1][1], 2
            )

        brief["category_breakdown"] = cat_breakdown

        # ── Time series analysis ─────────────────────────────────────────────
        if _looks_temporal(labels):
            brief["mode"] = "time_series"
            ts = _compute_time_series_brief(labels, values)
            brief["time_series"] = ts
        else:
            brief["mode"] = "ranking"
    elif numeric_cols:
        brief["mode"] = "numeric_table"
    else:
        brief["mode"] = "text_table"

    return brief


def _compute_time_series_brief(
    labels: list[str], values: list[float]
) -> dict:
    """Compute trend metrics for a time-ordered series."""
    n = len(values)
    first, last = values[0], values[-1]
    pct_change = _safe_pct(first, last)

    # Period-over-period changes
    pop_changes = []
    for i in range(1, n):
        delta = values[i] - values[i - 1]
        pct = _safe_pct(values[i - 1], values[i])
        pop_changes.append({
            "from_period": labels[i - 1],
            "to_period": labels[i],
            "absolute_change": round(delta, 2),
            "pct_change": pct,
        })

    # Find biggest swings — a single-period result (e.g. "which month has the
    # highest X" resolving to one row) has no period-over-period change to
    # compare; leave both None rather than crash on an empty pop_changes.
    biggest_drop = min(pop_changes, key=lambda x: x["absolute_change"]) if pop_changes else None
    biggest_gain = max(pop_changes, key=lambda x: x["absolute_change"]) if pop_changes else None

    # Peak and trough
    peak_i = max(range(n), key=lambda i: values[i])
    trough_i = min(range(n), key=lambda i: values[i])

    # Volatility (average absolute change)
    abs_changes = [abs(values[i] - values[i - 1]) for i in range(1, n)]
    volatility = round(mean(abs_changes), 2) if abs_changes else 0

    # Trend direction
    if pct_change is None:
        direction = "indeterminate"
    elif pct_change > 5:
        direction = "increasing"
    elif pct_change < -5:
        direction = "decreasing"
    else:
        direction = "stable"

    # Half-period comparison
    half_comparison = None
    if n >= 4:
        mid = n // 2
        first_half_avg = round(mean(values[:mid]), 2)
        second_half_avg = round(mean(values[mid:]), 2)
        half_comparison = {
            "first_half_period": f"{labels[0]} to {labels[mid - 1]}",
            "first_half_avg": first_half_avg,
            "second_half_period": f"{labels[mid]} to {labels[-1]}",
            "second_half_avg": second_half_avg,
            "half_pct_change": _safe_pct(first_half_avg, second_half_avg),
        }

    # Consecutive decline/increase streaks
    decline_streak = 0
    current_streak = 0
    for i in range(1, n):
        if values[i] < values[i - 1]:
            current_streak += 1
            decline_streak = max(decline_streak, current_streak)
        else:
            current_streak = 0

    increase_streak = 0
    current_streak = 0
    for i in range(1, n):
        if values[i] > values[i - 1]:
            current_streak += 1
            increase_streak = max(increase_streak, current_streak)
        else:
            current_streak = 0

    return {
        "direction": direction,
        "first_period": labels[0],
        "first_value": round(first, 2),
        "last_period": labels[-1],
        "last_value": round(last, 2),
        "overall_pct_change": pct_change,
        "peak": {"period": labels[peak_i], "value": round(values[peak_i], 2)},
        "trough": {"period": labels[trough_i], "value": round(values[trough_i], 2)},
        "biggest_period_drop": biggest_drop,
        "biggest_period_gain": biggest_gain,
        "volatility": volatility,
        "half_comparison": half_comparison,
        "longest_decline_streak": decline_streak,
        "longest_increase_streak": increase_streak,
        "period_count": n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Drill-down SQL generation for "why" questions
# ══════════════════════════════════════════════════════════════════════════════

def build_drilldown_prompt(
    original_question: str,
    follow_up: str,
    original_sql: str,
    data_brief: dict,
    db_type: str,
    context: str,
) -> tuple[str, str]:
    """
    Build a prompt that asks the LLM to generate drill-down SQL queries
    to investigate a "why" question. The LLM sees the data brief and the
    original SQL, and must produce 1-3 breakdown queries.
    """
    from core.llm import _SQL_SYNTAX, _DB_LABELS

    label = _DB_LABELS.get(db_type, db_type)
    syntax = _SQL_SYNTAX.get(db_type, "- Use standard ANSI SQL\n")

    system = (
        f"You are a {label} SQL expert. "
        "The user has received query results and is now asking a follow-up "
        "question to understand WHY the data looks the way it does.\n\n"
        "Your job: generate 1 to 3 drill-down SQL queries that would help "
        "explain the pattern described in the data brief below.\n\n"
        "RULES:\n"
        "- Use ONLY tables and columns from the Knowledge Base below.\n"
        "- Each query should investigate a DIFFERENT dimension or breakdown.\n"
        "- Focus on GROUP BY queries that reveal which categories, periods, "
        "or segments drove the overall pattern.\n"
        f"{syntax}"
        "- Return ONLY the SQL queries, each on its own line, separated by "
        "a line containing just '---'.\n"
        "- No markdown fences, no explanation.\n"
        "- If you cannot generate useful drill-downs, reply: NO_DRILLDOWN\n\n"
        f"Knowledge Base:\n{context[:3000]}"
    )

    user = (
        f"Original question: {original_question}\n"
        f"Original SQL: {original_sql}\n\n"
        f"Data brief (statistical summary — no raw data):\n"
        f"{_format_brief_for_prompt(data_brief)}\n\n"
        f"Follow-up question: {follow_up}\n\n"
        "Generate drill-down queries to investigate this."
    )
    return system, user


def _build_safe_llm_payload(
    action: str,
    question: str,
    data_brief: dict,
    *,
    follow_up: str = "",
) -> dict:
    payload: dict[str, Any] = {
        "question": question,
        "action": action,
        "follow_up": follow_up,
        "mode": data_brief.get("mode", "table"),
        "result_scope": data_brief.get("result_scope", {}),
        "metric_semantics": data_brief.get("metric_semantics", {}),
    }

    if data_brief.get("mode") == "ranking":
        category = data_brief.get("category_breakdown") or {}
        top_5 = category.get("top_5") or []
        leader = top_5[0] if top_5 else {}
        runner_up = top_5[1] if len(top_5) > 1 else {}
        payload["distribution_stats"] = {
            "category_count": category.get("category_count"),
            "top_3_share_pct": (data_brief.get("numeric_summaries", {}).get(category.get("value_column", ""), {}) or {}).get("top_3_concentration_pct"),
            "leader_share_pct": category.get("leader_share_pct"),
            "leader_vs_runner_up_gap": category.get("leader_vs_runner_up_gap"),
            "labels_redacted": category.get("labels_redacted", False),
        }
        payload["comparison_stats"] = {
            "leader": leader.get("label"),
            "leader_value": leader.get("value"),
            "runner_up": runner_up.get("label"),
            "runner_up_value": runner_up.get("value"),
            "gap": category.get("leader_vs_runner_up_gap"),
            "leader_share_pct": category.get("leader_share_pct"),
        }
    elif data_brief.get("mode") == "time_series":
        payload["time_series_stats"] = data_brief.get("time_series", {})
    elif data_brief.get("mode") == "numeric_table":
        numeric_summaries = data_brief.get("numeric_summaries", {})
        first_key = next(iter(numeric_summaries.keys()), "")
        payload["distribution_stats"] = numeric_summaries.get(first_key, {})
    elif data_brief.get("mode") == "single_value":
        # The one shape where the omission was total. A single-cell result has
        # no ranking, no series and no distribution, so every branch above
        # missed it and the payload carried NO NUMBER AT ALL -- the model was
        # asked to explain a figure it had never been shown, and answered from
        # the question wording alone.
        payload["single_value_stats"] = {
            "column": data_brief.get("value_column"),
            "value": data_brief.get("value"),
        }
    else:
        # `table` and `text_table` fall here. A table of numbers still has
        # summaries worth reasoning about; a table of text genuinely has none,
        # and then this stays empty rather than inventing something.
        numeric_summaries = data_brief.get("numeric_summaries", {})
        if numeric_summaries:
            first_key = next(iter(numeric_summaries.keys()), "")
            payload["distribution_stats"] = numeric_summaries.get(first_key, {})

    # Shape-independent: how many rows the reader is looking at, and how wide.
    # Without it an answer cannot say "across 3 categories" or "over 57 rows",
    # which is the difference between a description and a summary.
    payload["result_shape"] = {
        "row_count": data_brief.get("row_count"),
        "column_count": data_brief.get("column_count"),
        "columns": data_brief.get("columns", {}),
    }

    return payload


def build_action_contract(
    action: str,
    question: str,
    data_brief: dict,
    *,
    follow_up: str = "",
    grounding: dict | None = None,
) -> dict:
    payload = _build_safe_llm_payload(action, question, data_brief, follow_up=follow_up)
    semantics = payload.get("metric_semantics", {})
    scope = payload.get("result_scope", {})
    contract: dict[str, Any] = {
        "action": action,
        "question": question,
        "mode": payload.get("mode"),
        "result_scope": scope,
        "metric_semantics": semantics,
        # How much the reader is actually looking at. Every action wants it —
        # "across 3 categories", "over 57 rows" — so it is set once here
        # rather than in each branch below.
        "result_shape": payload.get("result_shape", {}),
        # How the number was governed: which business date it resolved to,
        # what scope was applied. The product computes all of this and then
        # told the model none of it, so an answer could state a total but
        # never what the total was OF.
        "grounding": dict(grounding or {}),
    }

    if action == "explain":
        contract["task"] = "Explain what this specific returned result means, including its scope limits."
        contract["headline_number"] = None
        if payload.get("mode") == "ranking":
            comparison = payload.get("comparison_stats", {})
            contract["headline_number"] = comparison.get("leader_value")
            contract["top_item"] = comparison.get("leader")
        elif payload.get("mode") == "time_series":
            ts = payload.get("time_series_stats", {})
            contract["headline_number"] = ts.get("last_value")
            contract["top_item"] = ts.get("last_period")
        elif payload.get("mode") == "single_value":
            single = payload.get("single_value_stats", {})
            contract["headline_number"] = single.get("value")
            contract["top_item"] = single.get("column")
    elif action == "analyze":
        contract["task"] = "Describe pattern shape, spread, concentration, or volatility in the returned result."
        contract["distribution_stats"] = payload.get("distribution_stats", {})
        contract["time_series_stats"] = payload.get("time_series_stats", {})
    elif action == "compare":
        contract["task"] = "Compare leader versus runner-up, or start versus end, using only returned comparison stats."
        contract["comparison_stats"] = payload.get("comparison_stats", {})
        contract["time_series_stats"] = payload.get("time_series_stats", {})
    elif action == "why":
        contract["task"] = "Provide business framing for why this result matters without inventing causes."
        contract["comparison_stats"] = payload.get("comparison_stats", {})
        contract["distribution_stats"] = payload.get("distribution_stats", {})
        contract["safe_next_steps"] = semantics.get("safe_next_steps", [])
    elif action == "predict":
        contract["task"] = "Project the next returned period from the observed series only."
        contract["time_series_stats"] = payload.get("time_series_stats", {})
    elif action == "decide":
        contract["task"] = (
            "Frame an advisory decision brief from the returned result: the key "
            "finding, why it matters, ONE recommended next step to investigate, and "
            "what to verify before acting. Never claim causation or prescribe a "
            "business decision."
        )
        contract["comparison_stats"]   = payload.get("comparison_stats", {})
        contract["distribution_stats"] = payload.get("distribution_stats", {})
        contract["time_series_stats"]  = payload.get("time_series_stats", {})
        contract["safe_next_steps"]    = semantics.get("safe_next_steps", [])

    if follow_up:
        contract["follow_up"] = follow_up
    return contract


def build_answer_grounding(
    *,
    semantic_plan: dict | None = None,
    result_scope: dict | None = None,
    row_count: int = 0,
    truncated: bool = False,
    tables: set[str] | list[str] | None = None,
) -> dict:
    """Collect how a figure was governed, for the answer to restate.

    Every fact here is already computed somewhere in the pipeline and was
    simply never carried to the model. Nothing is derived or guessed: a key is
    present only when the pipeline actually established it, because
    `_format_grounding_for_prompt` renders whatever it is given and the model
    is told to treat it as true.

    Deliberately total-failure-tolerant. This decorates an answer; it must
    never be the reason a question fails, so a malformed plan yields fewer
    facts rather than an exception.
    """
    grounding: dict[str, Any] = {}
    plan = semantic_plan or {}

    try:
        disclosures = [d for d in (plan.get("date_disclosures") or []) if isinstance(d, dict)]
        # The label is the sentence the date layer already wrote for a human
        # ("year to date, to the newest date present in CUS_ORD_IVC_FCT").
        labels = [str(d.get("label") or "").strip() for d in disclosures]
        labels = [label for label in labels if label]
        if labels:
            grounding["date_context"] = labels[:3]
        anchored = next(
            (d for d in disclosures if d.get("resolution_source") or d.get("inference_source")),
            None,
        )
        if anchored:
            source = anchored.get("resolution_source") or anchored.get("inference_source")
            table = anchored.get("table") or ""
            grounding["business_date"] = (
                f"{source} on {table}" if table else str(source)
            )
    except Exception:  # noqa: BLE001 — decoration, never a failure path
        pass

    try:
        scope = result_scope or {}
        # badge_key, not badge: the badge is now translated for the reader,
        # and a French token here would silently change what the model is
        # asked. The answer's language is set by the prompt's language rule.
        badge = str(scope.get("badge_key") or scope.get("badge")
                    or scope.get("kind") or "").strip()
        if badge:
            grounding["scope"] = badge
    except Exception:  # noqa: BLE001
        pass

    # Completeness is stated only when we know it is INCOMPLETE. Saying "full
    # read" on every other answer trains the reader to stop reading the line,
    # which is exactly when the truncation notice needs to land.
    if truncated:
        grounding["completeness"] = (
            f"TRUNCATED — only the first {row_count} rows were read, so totals "
            "and averages below cover that subset, not the whole result"
        )

    try:
        names = sorted({str(t).strip() for t in (tables or []) if str(t).strip()})
        if names and len(names) <= 6:
            grounding["tables"] = names
    except Exception:  # noqa: BLE001
        pass

    return grounding


def _format_grounding_for_prompt(grounding: dict) -> str:
    """Render how the figure was produced, for the answer to restate.

    A governed product computes all of this — which business date the query
    resolved to, what scope was applied, whether the read was complete — and
    then told the model none of it. So an answer could state a total and never
    say what the total was OF, which is the difference between a number and a
    defensible number.

    Only non-empty facts are emitted. A half-filled provenance block invites
    the model to fill the gaps, which is the opposite of the point.
    """
    if not grounding:
        return ""
    labels = (
        ("business_date", "Business date the query resolved to"),
        ("date_context", "Date context"),
        ("period_label", "Period covered"),
        ("scope", "Scope applied"),
        ("completeness", "Completeness"),
        ("metric_source", "Metric definition"),
        ("tables", "Tables read"),
    )
    lines = []
    for key, label in labels:
        value = grounding.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value if item)
            if not value:
                continue
        lines.append(f"- {label}: {value}")
    if not lines:
        return ""
    return (
        "\nHOW THIS RESULT WAS PRODUCED — state the relevant parts in your "
        "answer, and never invent one that is absent here:\n" + "\n".join(lines)
    )


# The narration model writes the sentences the reader actually reads, so the
# reader's language is a prompt rule, not a post-processing step -- translating
# a generated English paragraph afterwards would need a second model call and
# would still be a translation of an analysis rather than an analysis.
#
# Three things are pinned deliberately:
#   * The structural labels stay English. parse_insight_response matches
#     "HEADLINE:", "BODY:", "DETAIL:", "SECTION:" and "NEXT:" literally, so a
#     translated label is a response that parses as unlabelled prose.
#   * Column names and category values are the customer's schema and data.
#     Translating "Marge brute" into "Gross margin" makes the answer stop
#     matching the table under it.
#   * Numbers keep the formatting they arrive with. They are formatted upstream
#     against the column's format spec, and a model re-punctuating them is a
#     model changing values.
_LANGUAGE_RULES: dict[str, str] = {
    "fr": (
        "LANGUE :\n"
        "Rédigez toute votre réponse en français, dans un français professionnel "
        "et naturel -- pas une traduction mot à mot de l'anglais.\n"
        "Exceptions, à laisser exactement telles quelles :\n"
        "- Les étiquettes de structure HEADLINE:, SECTION:, BODY:, DETAIL: et "
        "NEXT: restent en anglais, en majuscules, suivies de deux-points.\n"
        "- Les noms de colonnes et les valeurs de catégories proviennent de la "
        "base du client : citez-les tels quels, ne les traduisez pas.\n"
        "- Les nombres, dates et devises sont déjà mis en forme : reprenez-les "
        "caractère pour caractère.\n\n"
    ),
}


def _language_rule(lang: str | None = None) -> str:
    """The output-language rule for the active reader, or "" for English.

    English is the empty string on purpose: every existing prompt is written in
    English, so adding "answer in English" to it would be a change to a prompt
    that has been tuned, for no behavioural gain.
    """
    from core.i18n import get_active_language, normalise_language
    tag = normalise_language(lang if lang is not None else get_active_language())
    return _LANGUAGE_RULES.get(tag, "")


def build_insight_prompt_from_contract(
    action_contract: dict,
    *,
    follow_up: str = "",
    drilldown_briefs: list[dict] | None = None,
    business_context: str = "",
) -> tuple[str, str]:
    """
    Build per-action LLM prompts with precise output structure.

    Each action has a specific instruction set so the LLM produces
    structured, business-grounded output — not generic statistics prose.
    """
    biz_block = ""
    if business_context:
        biz_block = (
            "\n\nBUSINESS CONTEXT (from the connected Knowledge Base):\n"
            + business_context[:2500]
            + "\n\nUse this context to interpret column names and business terminology. "
            "Only reference domain-specific causes if the business context supports them.\n"
        )

    action  = action_contract.get("action", "explain")
    mode    = action_contract.get("mode", "table")
    scope   = action_contract.get("result_scope", {})
    scope_note = scope.get("note", "Based on returned rows.")
    scope_badge = scope.get("badge_key") or scope.get("badge", "")

    # ── Shared base rules ────────────────────────────────────────────────────
    base_rules = (
        "You are a senior business analyst interpreting query results for a non-technical user.\n\n"
        + _language_rule() +
        "RULES:\n"
        "1. Use ONLY the numbers and labels from the data brief. Never invent values.\n"
        "2. Turn column names into the plain language of your answer "
        "(STATUSCOUNT → 'count of employees per status').\n"
        "3. Never claim certainty about causes unless the data directly supports it.\n"
        "4. Keep language direct — no filler phrases like 'it is worth noting'.\n"
        "5. Scope note to include verbatim at the end of BODY: "
        f'"{scope_note}"\n'
        + biz_block
    )

    # ── Per-action system instructions ───────────────────────────────────────
    if action == "explain":
        system = base_rules + (
            "\nTASK — EXPLAIN RESULT:\n"
            "Produce a result explanation with this exact structure:\n\n"
            "HEADLINE: One sentence: what metric is shown, grouped or filtered by what dimension.\n"
            "BODY: 2-3 sentences covering:\n"
            "  - What the numbers represent in plain English\n"
            "  - What dimensions (columns) are being used\n"
            "  - A brief scope note (use the scope note provided above verbatim)\n"
            "DETAIL:\n"
            "  - Bullet 1: The highest value and what it represents\n"
            "  - Bullet 2: The lowest value and what it represents\n"
            "  - Bullet 3: How many categories or periods are shown\n"
            "NEXT: One sentence — what the user should look at next to dig deeper.\n\n"
            "Example style for HEADLINE: "
            "'You are seeing daily attendance behaviour split by status — "
            "STATUSCOUNT shows how many employees fell into each attendance category on each date.'\n"
        )

    elif action == "analyze":
        system = base_rules + (
            "\nTASK — ANALYZE TREND / PATTERN:\n"
            "Produce a pattern analysis with this exact structure:\n\n"
            "HEADLINE: One sentence on the dominant pattern or most significant finding.\n"
            "BODY: 3-4 sentences covering:\n"
            "  - Which category or period is consistently highest\n"
            "  - Whether any category is increasing, decreasing, or volatile\n"
            "  - Whether there are notable spikes, drops, or unusual dates/values\n"
            "  - Whether the overall pattern looks stable or volatile\n"
            "DETAIL:\n"
            "  - Bullet 1: Highest point (when and how much)\n"
            "  - Bullet 2: Lowest point (when and how much)\n"
            "  - Bullet 3: Stable vs volatile assessment with evidence\n"
            "NEXT: A specific follow-up — e.g. filter by a specific category or drill down by date.\n\n"
            "Example style: 'In time remains the dominant attendance status across the visible dates. "
            "Late appears more variable than In time, suggesting inconsistency on certain days "
            "rather than a steady trend.'\n"
            "Do NOT say 'the data shows' — say directly what the pattern is.\n"
        )

    elif action == "compare":
        system = base_rules + (
            "\nTASK — COMPARE PERIODS OR CATEGORIES:\n"
            "Produce a comparison with this exact structure:\n\n"
            "HEADLINE: One sentence naming what is being compared and the key difference.\n"
            "BODY: 3-4 sentences covering:\n"
            "  - Which two periods or categories are being compared\n"
            "  - The absolute difference between them\n"
            "  - The percentage difference (if meaningful)\n"
            "  - Which dimension or status changed the most\n"
            "DETAIL:\n"
            "  - Bullet 1: Comparison of the most recent vs earliest period (or leader vs runner-up)\n"
            "  - Bullet 2: The largest single-period change (biggest gain or biggest drop)\n"
            "  - Bullet 3: Whether the second half of the series improved or worsened vs the first\n"
            "NEXT: What comparison to run next (e.g. same period by department, or week-on-week).\n\n"
            "Example style: 'Compared with 2016-01-03, 2016-01-04 shows a higher In time count "
            "and a lower Late count — attendance quality improved on the later date.'\n"
        )

    elif action == "why":
        system = base_rules + (
            "\nTASK — WHY THIS PATTERN (business interpretation):\n"
            "Produce a grounded business interpretation with this exact structure:\n\n"
            "HEADLINE: One sentence on what might be driving this pattern — phrased as a hypothesis, not a fact.\n"
            "BODY: 3-4 sentences covering:\n"
            "  - Likely drivers based on the shape of the data (NOT invented business knowledge)\n"
            "  - Your confidence level: low / medium / high, with a one-line reason\n"
            "  - What additional breakdown is needed to confirm the cause\n"
            "DETAIL:\n"
            "  - Bullet 1: Most likely driver (labelled as hypothesis)\n"
            "  - Bullet 2: Second possible driver\n"
            "  - Bullet 3: Specific drill-down that would confirm or reject the hypothesis\n"
            "NEXT: The next investigation step — e.g. filter by department, shift, or team.\n\n"
            "Example style: 'This pattern may be driven by operational differences across days, "
            "such as staffing schedules or late-arrival pressure on specific dates. "
            "To confirm the cause, compare attendance by department, shift, or employee group.'\n"
            "NEVER claim a cause is definitive unless the data directly shows it.\n"
            "NEVER invent domain-specific causes not supported by the business context.\n"
        )

    elif action == "predict":
        system = base_rules + (
            "\nTASK — PREDICT NEXT PERIOD:\n"
            "Produce a directional forecast with this exact structure:\n\n"
            "HEADLINE: One sentence on what the next period is likely to look like.\n"
            "BODY: 2-3 sentences covering:\n"
            "  - The predicted direction or value for the next period\n"
            "  - Confidence level (low / medium / moderate) with reason\n"
            "  - An explicit assumption note — this is a projection, not a guarantee\n"
            "DETAIL:\n"
            "  - Bullet 1: Last observed period and its value\n"
            "  - Bullet 2: Average step change used in the projection\n"
            "  - Bullet 3: What would cause the forecast to be wrong (key risk)\n"
            "NEXT: What to watch or track to validate whether the prediction holds.\n\n"
            "Example style: 'If the current pattern continues, In time is likely to remain the "
            "leading status next period, with Late staying within a similar range. "
            "Confidence is moderate because day-to-day variation is still visible in the series.'\n"
            "Always frame as directional estimate, never a hard promise.\n"
        )

    elif action == "decide":
        system = base_rules + (
            "\nTASK — RECOMMEND NEXT STEP (advisory decision brief):\n"
            "You are a decision-support analyst. You surface findings and suggested "
            "checks. You NEVER tell the user what business decision to make, and you "
            "NEVER claim a cause is proven.\n"
            "Produce an advisory decision brief with this exact structure:\n\n"
            "HEADLINE: The single decision-relevant finding, in one sentence.\n"
            "BODY: 2-3 sentences — what the result implies for the business, framed as "
            "observation not instruction (use hedged language: 'may indicate', "
            "'worth checking'). End with the scope note verbatim.\n"
            "DETAIL:\n"
            "  - Finding: the strongest signal, taken only from the stats provided\n"
            "  - Implication: what it could mean for a decision (hedged, not asserted)\n"
            "  - Risk/caveat: the main reason NOT to over-read this result\n"
            "NEXT: ONE concrete, safe step to verify or drill into BEFORE acting.\n\n"
            "Example style: 'Revenue is concentrated in a handful of accounts, which "
            "may expose the book to churn risk. Before reallocating effort, confirm "
            "whether the concentration reflects a few large contracts or a reporting "
            "artifact.'\n"
            "Frame everything as a recommendation to investigate — never as a directive.\n"
        )

    else:
        system = base_rules + "\nTASK: Provide a concise, grounded interpretation of this result.\n"

    # ── User message — structured data brief ─────────────────────────────────
    system += (
        "\nHOW TO WRITE IT\n"
        "- Open by naming what the figures cover — the period, the filter, the "
        "date the data runs to — using only what the provenance block states. "
        "A reader should never have to ask what the total was OF.\n"
        "- Use the actual figures. \"Revenue rose 12% to $13.1M\" is an answer; "
        "\"revenue increased significantly\" is not.\n"
        "- Say what it means, not only what it is. The number is already on "
        "screen in the table above your answer.\n"
        "- No preamble. Start with the finding.\n"
        "\nRESPONSE FORMAT (required):\n"
        "HEADLINE: ...\n"
        "BODY: ...\n"
        "DETAIL:\n- ...\n- ...\n- ...\n"
        "NEXT: ...\n"
        "\nBODY may run to several lines; they are kept.\n"
        "\nIf the question asked TWO OR MORE distinct things, answer each under "
        "its own heading instead of running them together in one BODY:\n"
        "HEADLINE: ...\n"
        "SECTION: <the first thing asked, as a heading>\n"
        "BODY: ...\n"
        "DETAIL:\n- ...\n"
        "SECTION: <the second thing asked>\n"
        "BODY: ...\n"
        "NEXT: ...\n"
        "Use SECTION only for parts the reader actually asked about — never to "
        "subdivide a single question, and never more than four.\n"
    )

    user_parts = [
        f"Action: {action}",
        f"Question asked: {action_contract.get('question', '')}",
        f"Mode: {mode}",
        f"Scope: {scope_badge}",
        f"\nData brief:\n{_format_brief_for_prompt(action_contract)}",
    ]
    grounding_block = _format_grounding_for_prompt(action_contract.get("grounding") or {})
    if grounding_block:
        user_parts.append(grounding_block)
    if follow_up:
        user_parts.insert(1, f"Follow-up context: {follow_up}")
    if drilldown_briefs:
        user_parts.append("\nDrill-down breakdowns:")
        for i, db in enumerate(drilldown_briefs, 1):
            user_parts.append(f"\nBreakdown {i}:\n{_format_brief_for_prompt(db)}")
    return system, "\n".join(user_parts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM insight generation
# ══════════════════════════════════════════════════════════════════════════════

def build_insight_prompt(
    action: str,
    question: str,
    data_brief: dict,
    follow_up: str = "",
    drilldown_briefs: list[dict] | None = None,
    business_context: str = "",
) -> tuple[str, str]:
    """
    Build the system + user prompt for LLM insight generation.
    
    The LLM receives the data brief, never the row set. See the module
    docstring for what that does and does not include.
    business_context provides the KB/RAG documents so the LLM
    understands what the business does, what columns mean, and
    what metrics represent — without this, it would hallucinate.
    """
    # Extract a concise business context block from the RAG KB docs
    # The full KB can be very long — we take the most relevant sections:
    # business vocab, column meanings, and metric definitions.
    biz_block = ""
    if business_context:
        # Truncate to keep the prompt focused — the LLM needs business
        # understanding, not the full schema for SQL generation.
        biz_block = (
            "\n\nBUSINESS CONTEXT (from the connected Knowledge Base):\n"
            + business_context[:2500]
            + "\n\nUse this context to interpret column names, understand what "
            "the business does, and ground your analysis in real domain meaning. "
            "If a column name maps to a specific business concept in this context, "
            "use the business term — not the raw column name.\n"
        )

    system = (
        "You are a senior business analyst interpreting query results for "
        "a non-technical business user.\n\n"
        "CRITICAL RULES:\n"
        "1. You are given a STATISTICAL SUMMARY (data brief) of the query "
        "results — not the raw data. Your job is to interpret the patterns "
        "and provide clear, actionable business insight.\n"
        "2. Always reference specific numbers from the brief — do not "
        "invent or hallucinate data points.\n"
        "3. Use plain business language. Translate column names to readable "
        "labels using the business context provided (e.g., if the context "
        "says TOTAL_CHARGES is prescription revenue, say 'prescription "
        "revenue' — not 'TOTAL_CHARGES').\n"
        "4. Keep your response concise: 2-4 sentences for the main "
        "finding, then 2-3 bullet points for supporting detail.\n"
        "5. If drill-down data is provided, use it to explain causal "
        "factors — which segments, categories, or time periods drove "
        "the observed pattern.\n"
        "6. End with one specific, actionable suggestion for what the "
        "user should investigate next.\n"
        "7. If the data is insufficient to draw conclusions, say so "
        "honestly rather than speculating. NEVER invent business reasons "
        "that are not supported by the data brief or business context.\n"
        "8. GROUNDING RULE: Only reference business concepts, seasonal "
        "patterns, or domain-specific factors if they appear in the "
        "business context below. If no business context is provided, "
        "limit your analysis to what the numbers show — do not guess "
        "at industry-specific causes.\n\n"
        "RESPONSE FORMAT (use this exact structure):\n"
        "HEADLINE: [One sentence main finding]\n"
        "BODY: [2-4 sentence explanation with specific numbers]\n"
        "DETAIL: [2-3 bullet points with supporting evidence]\n"
        "NEXT: [One actionable next step]\n"
        + biz_block
    )

    brief_text = _format_brief_for_prompt(data_brief)

    action_context = {
        "explain": "Explain what this result means in business terms.",
        "analyze": "Provide a detailed statistical analysis of the patterns.",
        "compare": "Compare the key segments or time periods in this data.",
        "predict": "Based on the observed trend, project what might happen next.",
        "why": "Explain WHY the data shows this pattern — what drove it.",
    }.get(action, "Provide your analysis of this data.")

    user_parts = [
        f"Question asked: {question}",
        f"\nData brief:\n{brief_text}",
        f"\nTask: {action_context}",
    ]

    if follow_up:
        user_parts.insert(1, f"Follow-up: {follow_up}")

    if drilldown_briefs:
        user_parts.append("\nDrill-down breakdowns (additional context):")
        for i, db in enumerate(drilldown_briefs, 1):
            user_parts.append(f"\nBreakdown {i}:\n{_format_brief_for_prompt(db)}")

    return system, "\n".join(user_parts)


def _format_brief_for_prompt(brief: dict) -> str:
    """Format a data brief dict into readable text for the LLM prompt."""
    lines = []

    if brief.get("task"):
        lines.append(f"Mode: {brief.get('mode', 'unknown')}")
        scope = brief.get("result_scope") or {}
        if scope:
            lines.append(f"Scope: {scope.get('badge', scope.get('kind', 'returned result'))}")
            if scope.get("note"):
                lines.append(f"Scope note: {scope['note']}")
        semantics = brief.get("metric_semantics") or {}
        if semantics:
            lines.append(f"Business meaning: {semantics.get('business_meaning', '')}")
            lines.append(f"Why it matters: {semantics.get('why_it_matters', '')}")
        if brief.get("top_item") is not None:
            lines.append(f"Top item: {brief.get('top_item')} = {brief.get('headline_number')}")
        if brief.get("distribution_stats"):
            lines.append(f"Distribution stats: {brief.get('distribution_stats')}")
        if brief.get("comparison_stats"):
            lines.append(f"Comparison stats: {brief.get('comparison_stats')}")
        if brief.get("time_series_stats"):
            lines.append(f"Time-series stats: {brief.get('time_series_stats')}")
        if brief.get("safe_next_steps"):
            lines.append(f"Safe next steps: {', '.join(brief['safe_next_steps'])}")
        return "\n".join(lines)

    lines.append(f"Result: {brief.get('row_count', 0)} rows, mode: {brief.get('mode', 'unknown')}")

    if brief.get("mode") == "single_value":
        lines.append(f"Single value — {brief.get('value_column')}: {brief.get('value')}")
        return "\n".join(lines)

    if brief.get("mode") == "empty":
        lines.append("No data returned.")
        return "\n".join(lines)

    # Numeric summaries
    for col, stats in brief.get("numeric_summaries", {}).items():
        lines.append(
            f"  {col}: total={stats['total']}, min={stats['min']}, "
            f"max={stats['max']}, mean={stats['mean']}, median={stats['median']}"
        )
        if "std_dev" in stats:
            lines.append(f"    std_dev={stats['std_dev']}")
        if "top_3_concentration_pct" in stats:
            lines.append(f"    top 3 items account for {stats['top_3_concentration_pct']}% of total")

    # Category breakdown
    cat = brief.get("category_breakdown")
    if cat:
        lines.append(f"\nBreakdown by {cat['label_column']}:")
        lines.append(f"  {cat['category_count']} distinct categories")
        if cat.get("top_5"):
            top_str = ", ".join(
                f"{item['label']}={item['value']}" for item in cat["top_5"]
            )
            lines.append(f"  Top 5: {top_str}")
        if cat.get("leader_share_pct"):
            lines.append(f"  Leader holds {cat['leader_share_pct']}% of total")
        if cat.get("leader_vs_runner_up_gap"):
            lines.append(f"  Leader vs runner-up gap: {cat['leader_vs_runner_up_gap']}")

    # Time series
    ts = brief.get("time_series")
    if ts:
        lines.append(f"\nTime trend ({ts['period_count']} periods):")
        lines.append(f"  Direction: {ts['direction']}")
        lines.append(f"  Start: {ts['first_period']} = {ts['first_value']}")
        lines.append(f"  End: {ts['last_period']} = {ts['last_value']}")
        if ts.get("overall_pct_change") is not None:
            lines.append(f"  Overall change: {ts['overall_pct_change']}%")
        lines.append(f"  Peak: {ts['peak']['period']} = {ts['peak']['value']}")
        lines.append(f"  Trough: {ts['trough']['period']} = {ts['trough']['value']}")
        bd = ts.get("biggest_period_drop", {})
        if bd:
            lines.append(
                f"  Biggest drop: {bd.get('from_period')} → {bd.get('to_period')} "
                f"({bd.get('absolute_change')}, {bd.get('pct_change')}%)"
            )
        bg = ts.get("biggest_period_gain", {})
        if bg:
            lines.append(
                f"  Biggest gain: {bg.get('from_period')} → {bg.get('to_period')} "
                f"({bg.get('absolute_change')}, {bg.get('pct_change')}%)"
            )
        lines.append(f"  Volatility (avg abs change): {ts['volatility']}")
        if ts.get("half_comparison"):
            hc = ts["half_comparison"]
            lines.append(
                f"  First half avg: {hc['first_half_avg']} vs "
                f"second half avg: {hc['second_half_avg']} "
                f"({hc.get('half_pct_change')}%)"
            )
        if ts.get("longest_decline_streak", 0) >= 2:
            lines.append(f"  Longest consecutive decline: {ts['longest_decline_streak']} periods")
        if ts.get("longest_increase_streak", 0) >= 2:
            lines.append(f"  Longest consecutive increase: {ts['longest_increase_streak']} periods")

    return "\n".join(lines)


def parse_insight_response(raw: str) -> dict:
    """Parse the structured LLM response into a dict.

    Two things this used to do badly, both silently.

    Every line that did not open with a known label was DISCARDED — the if/elif
    chain had no else — so a BODY written across several lines kept only its
    first, and there was no way to tell from the output that anything had been
    lost. Continuation lines now attach to whatever block is open.

    And there was no way to answer a question that has two parts. A reader
    asking "are they priced at a premium, and how much revenue depends on
    them?" wants both answered under their own headings; the contract had one
    flat body, so the model was told to pick. `SECTION: <title>` opens a titled
    block, and `sections` carries them. The four original keys keep their exact
    old meaning, so every existing renderer is unaffected by a response that
    does not use sections.
    """
    result: dict = {
        "headline": "", "body": "", "bullets": [], "next_step": "", "sections": [],
    }
    # The block that continuation lines and bullets belong to: the answer
    # itself, or the section currently open.
    current: dict = result
    # Did the model use the format at all? This used to be inferred from "no
    # headline and no body", which stops working once unlabelled prose fills
    # the body — the salvage path would never fire again, and a reply with no
    # labels would arrive with no headline.
    saw_label = False

    def _append_body(container: dict, text: str) -> None:
        container["body"] = f"{container['body']} {text}".strip() if container["body"] else text

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("HEADLINE:"):
            saw_label = True
            result["headline"] = stripped[len("HEADLINE:"):].strip()
            current = result
        elif upper.startswith("SECTION:"):
            saw_label = True
            title = stripped[len("SECTION:"):].strip()
            current = {"title": title, "body": "", "bullets": []}
            result["sections"].append(current)
        elif upper.startswith("BODY:"):
            saw_label = True
            _append_body(current, stripped[len("BODY:"):].strip())
        elif upper.startswith("DETAIL:"):
            saw_label = True
            detail_text = stripped[len("DETAIL:"):].strip()
            # Parse bullet points — could be on same line or following lines
            if detail_text.startswith("- ") or detail_text.startswith("• "):
                current["bullets"].append(detail_text.lstrip("-•").strip())
            elif detail_text:
                current["bullets"].append(detail_text)
        elif stripped.startswith("- ") or stripped.startswith("• "):
            saw_label = True
            current["bullets"].append(stripped.lstrip("-•").strip())
        elif upper.startswith("NEXT:"):
            saw_label = True
            # One per answer, never per section: it is what the reader does
            # next, and a list of them is a menu rather than a suggestion.
            result["next_step"] = stripped[len("NEXT:"):].strip()
        else:
            # Unlabelled prose. Previously dropped on the floor, which is how a
            # three-sentence explanation arrived as one sentence.
            _append_body(current, stripped)

    # Salvage — the model ignored the format entirely, so the whole reply is
    # the answer and its first line is the closest thing to a headline it has.
    if not saw_label:
        result["body"] = raw.strip()
        lines = raw.strip().splitlines()
        if lines:
            result["headline"] = lines[0][:120]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main entry points
# ══════════════════════════════════════════════════════════════════════════════

async def generate_insight(
    rows: list[dict],
    question: str,
    action: str = "explain",
    follow_up: str = "",
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: str = "",
    drilldown_briefs: list[dict] | None = None,
    business_context: str = "",
    original_sql: str = "",
    grounding: dict | None = None,
    **extra_kwargs,
) -> dict:
    """
    Generate a dynamic LLM-powered insight from query results.
    
    The LLM receives the data brief, never the row set. See the module
    docstring for what that does and does not include.
    business_context provides KB/RAG docs so the LLM understands
    the domain, column meanings, and business terminology.
    
    Returns:
        {
            "type": "assistant_analysis",
            "action": action,
            "title": str,
            "headline": str,
            "body": str,
            "bullets": list[str],
            "next_step": str,
            "data_brief": dict,  # the statistical summary (for transparency)
            "source_question": str,
        }
    """
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_component

    from core.response_builder import summarize_result_context

    context = summarize_result_context(rows, question, sql=original_sql)
    brief = compute_data_brief(
        rows,
        question,
        result_scope=context.get("result_scope"),
        context=context,
    )
    action_contract = build_action_contract(
        action,
        question,
        brief,
        follow_up=follow_up,
        grounding=grounding,
    )
    system, user_msg = build_insight_prompt_from_contract(
        action_contract,
        follow_up=follow_up,
        drilldown_briefs=drilldown_briefs,
        business_context=business_context,
    )

    try:
        with llm_audit_component("analysis_narrative"):
            raw_response, tok_in, tok_out = await llm_complete(
                system, user_msg, provider, model, api_key,
                max_tokens=600, temperature=0.3,
                # Prose a person reads: a narrative that stops mid-sentence is
                # still worth more than no narrative, and nothing downstream
                # parses it for a governed decision.
                allow_truncated=True,
                **extra_kwargs,
            )
        parsed = parse_insight_response(raw_response)
    except Exception as e:
        log.error("Insight generation failed: %s", e)
        parsed = {
            "headline": _t("analysis.failed_headline"),
            "body": _t("analysis.failed_body", detail=str(e)[:100]),
            "bullets": [],
            "next_step": _t("analysis.failed_next"),
        }

    # The action key routes the card; only the title is copy.
    _KNOWN_TITLES = ("explain", "analyze", "compare", "predict", "why", "decide")
    title = _t(f"analysis.title.{action}" if action in _KNOWN_TITLES
               else "analysis.title.default")

    return {
        "type": "assistant_analysis",
        "action": action,
        "title": title,
        "headline": parsed.get("headline", ""),
        "body": parsed.get("body", ""),
        "bullets": parsed.get("bullets", []),
        # Titled parts, when the question had more than one. Empty for every
        # single-part answer, so a renderer that ignores it loses nothing.
        "sections": parsed.get("sections", []),
        "next_step": parsed.get("next_step", ""),
        "secondary": parsed.get("next_step", ""),  # compat with existing UI
        "data_brief": brief,
        "action_contract": action_contract,
        "result_scope": brief.get("result_scope", {}),
        "source_question": question,
        "mode": brief.get("mode", "table"),
    }


async def generate_drilldown_insight(
    rows: list[dict],
    question: str,
    follow_up: str,
    original_sql: str,
    db_cfg: dict,
    context: str,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    api_key: str = "",
    known_tables: set[str] | None = None,
    business_context: str = "",
    query_executor=None,
    **extra_kwargs,
) -> dict:
    """
    Full drill-down pipeline for "why" questions:
    
    1. Compute data brief from original results
    2. Ask LLM to generate drill-down SQL queries
    3. Execute each drill-down query
    4. Compute data briefs from drill-down results
    5. Send all briefs to LLM for causal interpretation
    
    The LLM never sees the row set at any step — see the module docstring.
    """
    from core.llm import llm_complete
    from core.llm_audit import llm_audit_component
    from core.schema import run_query
    from core.validator import validate_sql

    from core.response_builder import summarize_result_context

    context_summary = summarize_result_context(rows, question, sql=original_sql)
    brief = compute_data_brief(
        rows,
        question,
        result_scope=context_summary.get("result_scope"),
        context=context_summary,
    )
    drilldown_briefs = []

    # Step 1: Ask LLM for drill-down queries
    dd_system, dd_user = build_drilldown_prompt(
        question, follow_up, original_sql, brief,
        db_cfg["db_type"], context,
    )

    try:
        with llm_audit_component("drilldown_planner"):
            dd_response, _, _ = await llm_complete(
                dd_system, dd_user, provider, model, api_key,
                max_tokens=800, temperature=0.2, **extra_kwargs,
            )

        if "NO_DRILLDOWN" not in dd_response.upper():
            # Parse drill-down SQL queries
            dd_sqls = [
                s.strip() for s in dd_response.split("---")
                if s.strip() and len(s.strip()) > 10
            ]

            # Clean markdown fences
            cleaned = []
            for sql in dd_sqls:
                if sql.startswith("```"):
                    sql = "\n".join(sql.split("\n")[1:]).rsplit("```", 1)[0].strip()
                cleaned.append(sql)
            dd_sqls = cleaned[:3]  # Max 3 drill-downs

            # Step 2: Validate and execute each drill-down
            _known = known_tables or set()

            for sql in dd_sqls:
                try:
                    ok, reason, code = validate_sql(
                        sql, _known, db_cfg["db_type"]
                    )
                    if ok:
                        if query_executor:
                            governed = query_executor(db_cfg, sql)
                            dd_rows = governed.rows
                            sql = governed.sql
                        else:
                            dd_rows = run_query(
                                db_cfg["credentials"], db_cfg["db_type"], sql
                            )
                        if dd_rows:
                            dd_brief = compute_data_brief(dd_rows, follow_up)
                            dd_brief["drilldown_sql_intent"] = sql[:100]
                            drilldown_briefs.append(dd_brief)
                except Exception as e:
                    log.debug("Drill-down query failed: %s", str(e)[:80])

    except Exception as e:
        log.warning("Drill-down generation failed: %s", e)

    # Step 3: Generate insight with all briefs
    # Use the RAG context as business context for the final insight —
    # it contains KB docs with column meanings, business terms, and domain info.
    _biz_ctx = business_context or context
    return await generate_insight(
        rows=rows,
        question=question,
        action="why",
        follow_up=follow_up,
        provider=provider,
        model=model,
        api_key=api_key,
        drilldown_briefs=drilldown_briefs if drilldown_briefs else None,
        business_context=_biz_ctx,
        original_sql=original_sql,
        **extra_kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Result-aware follow-up question suggestions
# ══════════════════════════════════════════════════════════════════════════════

async def generate_followup_suggestions(
    brief: dict,
    question: str,
    result_scope: dict,
    db_cfg: dict,
    account_id: str,
    audit_enabled: bool = False,
    audit_request_id: str = "",
    signals: list[dict] | None = None,
) -> list[str]:
    """
    Generate 3 result-aware follow-up questions from brief metadata.
    Reads category_breakdown and numeric_summaries from brief — raw rows never
    reach this function (PII boundary).
    Returns [] on any failure — follow-ups are a UX enhancement, not critical.
    Regulated tenants get [] unconditionally (see result_llm_features_allowed)
    with a proof-of-refusal audit row instead of ever building a prompt.
    """
    from core.compliance.policy_engine import result_llm_features_allowed
    if not result_llm_features_allowed(account_id):
        from core.llm_audit import llm_audit_scope, record_llm_blocked
        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="followup_suggestions",
        ):
            record_llm_blocked(
                "followup_suggestions",
                "follow-up suggestions blocked — regulated tenant, LLM never received result rows.",
            )
        return []

    if not brief:
        return []

    columns   = brief.get("columns") or {}
    row_count = brief.get("row_count", 0)
    if row_count == 0 or len(columns) < 2:
        return []

    col_names = list(columns.keys()) if isinstance(columns, dict) else [str(c) for c in columns]
    col_types = columns if isinstance(columns, dict) else {}

    # Build top-values context from category_breakdown — no raw row values sent to LLM
    cat       = brief.get("category_breakdown") or {}
    label_col = cat.get("label_column", "")
    top_vals_ctx = ""
    if label_col and not _is_sensitive_field(label_col):
        top_5 = cat.get("top_5") or []
        safe_labels = [
            item["label"] for item in top_5
            if item.get("label") and item["label"] != "redacted segment"
        ]
        if safe_labels:
            top_vals_ctx = f"Top {label_col} values: {', '.join(str(l) for l in safe_labels[:5])}\n"

    # ── Tier 1: Statistical signals → instant template suggestions ───────────
    #
    # `signals` used to be initialised empty here and never filled, so this
    # whole tier was dead: every chip came from the LLM below, ungrounded,
    # while the comments and the prompt described statistical grounding that
    # never happened. The cause was structural — compute_signals() needs the
    # result ROWS, and rows deliberately stop at the PII boundary before this
    # function. Signals carry no raw values by construction, so the caller
    # computes them where the rows are and passes them in.
    signals = list(signals or [])
    suggestions: list[str] = []
    if signals:
        try:
            from core.stat_signals import template_suggestions
            suggestions = [q for q in template_suggestions(signals, col_names) if q]
        except Exception as exc:
            log.warning(
                "Statistical follow-up templates unavailable (%s) — falling back "
                "to model-written suggestions with signal context", exc,
            )

    if len(suggestions) >= 3:
        return [s for s in suggestions if s][:3]

    # ── Tier 2: LLM gap-fill with signal context only (no raw rows) ──────────
    needed = 3 - len(suggestions)

    # Build signal context: only labels + column names, never row values
    if signals:
        from core.stat_signals import format_signals_for_llm
        signal_ctx = format_signals_for_llm(signals, col_names)
    else:
        # No statistical signals — fall back to brief stats + category labels
        num_lines = []
        for cn, stats in (brief.get("numeric_summaries") or {}).items():
            if stats.get("min") is not None:
                num_lines.append(f"  {cn}: min={stats['min']}, max={stats['max']}, mean={stats.get('mean','?')}")
        _unknown = "?"
        signal_ctx = (
            f"Columns: {', '.join(f'{c} ({col_types.get(c, _unknown)})' for c in col_names[:8])}\n"
            + top_vals_ctx
            + ("\n".join(num_lines) if num_lines else "")
        )

    existing_str = (
        f"\nAlready suggested (do NOT repeat): {suggestions}\n" if suggestions else ""
    )

    # Columns already present in the result — LLM must not suggest grouping by these
    existing_cols_str = ", ".join(col_names[:8])

    user_msg = (
        f"The user asked: {question!r}\n"
        f"Result: {row_count} rows\n"
        f"Columns already in this result: {existing_cols_str}\n"
        f"{signal_ctx}"
        f"{existing_str}\n"
        f"Generate exactly {needed} short follow-up question(s) (max 12 words each) "
        f"that a business analyst would naturally ask, based ONLY on the detected "
        f"patterns listed above.\n"
        f"Rules:\n"
        f"  - Each question must map to a pattern listed above — do NOT invent new patterns\n"
        f"  - Questions must be answerable by SQL (aggregations, filters, rankings, groupings)\n"
        f"  - Do NOT suggest 'break down by X' or 'group by X' when X is already a column "
        f"in the result — the result is ALREADY broken down by those columns\n"
        f"  - For two numeric columns: phrase as 'Show X vs Y' not 'Are they correlated'\n"
        f"  - Never suggest statistical functions (correlation coeff, regression, p-value)\n"
        f"Return ONLY a JSON array of {needed} string(s). No markdown. No explanation.\n"
        f'Example (3 needed): ["Who had the highest total charge?", '
        f'"Show bottom 5 by prescription count", "Which items are above average?"]'
    )

    system_msg = (
        "You generate statistically-grounded follow-up questions for a SQL analytics chatbot. "
        "You ONLY phrase questions from the detected statistical patterns provided — you never "
        "invent new patterns. Questions must be SQL-answerable. "
        "Return a JSON array only. No markdown fences. No explanation."
    )

    try:
        import json as _json
        import store as _store
        from core.llm import llm_complete, resolve_provider
        from core.llm_audit import llm_audit_scope

        _client = _store.get_client(account_id) or {}
        provider, model, api_key, az_kwargs = resolve_provider(_client, purpose="query")
        with llm_audit_scope(
            account_id=account_id,
            question=question,
            enabled=audit_enabled,
            request_id=audit_request_id,
            question_id=audit_request_id,
            component="followup_suggestions",
        ):
            raw, _, _ = await llm_complete(
                system_msg, user_msg, provider, model, api_key,
                max_tokens=160, temperature=0.5, **az_kwargs,
            )

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        llm_suggestions = _json.loads(clean)
        if not isinstance(llm_suggestions, list):
            llm_suggestions = []

        for s in llm_suggestions:
            s = str(s).strip()[:80]
            if s and s not in suggestions:
                suggestions.append(s)
            if len(suggestions) >= 3:
                break

        # Normalise: strings only, stripped, max 80 chars, max 3
        result = [s for s in suggestions if s][:3]
        log.debug("Follow-up suggestions (template+LLM): %s", result)
        return result

    except Exception as exc:
        import logging as _log
        _log.getLogger("querybot.insight").debug(
            "Follow-up suggestion generation failed (non-critical): %s", exc
        )
        return []
