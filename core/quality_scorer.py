"""
core/quality_scorer.py

Quality scoring and classification for the self-learning loop (Sprint).

Score dimensions (max 100 — no feedback):
  SQL validation          25   pass/fail
  Execution success       20   pass/fail
  Metric compliance       15   partial (0.0–1.0 float → prorated pts)
  Schema/ACL compliance   15   partial
  Entity-graph compliance 10   partial
  No repair               10   mutually exclusive with successful_repair
  Successful repair        5   mutually exclusive with no_repair
  Usable non-null result   5   pass/fail (zero rows = no penalty, no bonus)

Feedback adjustments (applied AFTER the technical score):
  Net positive (more ups than downs)  +10
  Net negative (more downs than ups)  -30
  Tied / no votes                      0

Classification thresholds:
  final_score ≥ 85                → "positive"
  60 ≤ final_score < 85           → "review"
  final_score < 60 or net-negative → "negative"

Design notes
────────────
• score_trace() is a pure function — no DB, no LLM, no I/O.
• Partial compliance scoring (0.0–1.0 float) prevents a single missed
  metric from equalling a completely wrong query.
• net_feedback_delta() converts raw vote counts to ±10/−30 so one very
  active user cannot game the score with multiple votes.
• All constants are module-level for easy tweaking without code changes.
"""

from __future__ import annotations

from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Score-weight constants
# ══════════════════════════════════════════════════════════════════════════════

W_SQL_VALIDATION          = 25
W_EXECUTION_SUCCESS       = 20
W_METRIC_COMPLIANCE       = 15
W_SCHEMA_ACL_COMPLIANCE   = 15
W_ENTITY_GRAPH_COMPLIANCE = 10
W_NO_REPAIR               = 10   # mutually exclusive with W_SUCCESSFUL_REPAIR
W_SUCCESSFUL_REPAIR       =  5
W_USABLE_RESULT           =  5

FEEDBACK_POSITIVE = +10
FEEDBACK_NEGATIVE = -30

THRESHOLD_POSITIVE  = 85
THRESHOLD_REVIEW_LO = 60


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def score_trace(
    *,
    validation_passed: bool,
    execution_success: bool,
    had_repair: bool,
    repair_succeeded: bool,
    metric_compliance: float = 1.0,
    schema_acl_compliance: float = 1.0,
    entity_graph_compliance: float = 1.0,
    row_count: int = 0,
    feedback_delta: int = 0,
) -> tuple[int, dict[str, Any]]:
    """
    Compute the quality score and evidence breakdown for one trace.

    Parameters
    ──────────
    validation_passed       : SQL passed the validator
    execution_success       : DB execution completed without error
    had_repair              : SQL went through a repair cycle
    repair_succeeded        : The repair produced valid SQL (relevant only when had_repair=True)
    metric_compliance       : fraction of required metrics correctly referenced (0.0–1.0)
    schema_acl_compliance   : fraction of used tables that are ACL-allowed (0.0–1.0)
    entity_graph_compliance : fraction of joins that comply with entity graph (0.0–1.0)
    row_count               : number of rows returned (0 = no bonus, no penalty)
    feedback_delta          : net feedback adjustment from net_feedback_delta()

    Returns
    ───────
    (final_score, evidence_dict)

    evidence_dict is stored verbatim in learning_candidate.evidence (JSON) so
    admins can see exactly why a candidate scored what it scored.
    """
    evidence: dict[str, Any] = {}
    raw = 0

    # ── SQL validation ────────────────────────────────────────────────────────
    sql_pts = W_SQL_VALIDATION if validation_passed else 0
    raw += sql_pts
    evidence["sql_validation"] = sql_pts

    # ── Execution ────────────────────────────────────────────────────────────
    exec_pts = W_EXECUTION_SUCCESS if execution_success else 0
    raw += exec_pts
    evidence["execution_success"] = exec_pts

    # ── Metric compliance (partial) ──────────────────────────────────────────
    mc = max(0.0, min(1.0, float(metric_compliance)))
    metric_pts = round(W_METRIC_COMPLIANCE * mc)
    raw += metric_pts
    evidence["metric_compliance"] = metric_pts
    if mc < 1.0:
        evidence["metric_compliance_pct"] = round(mc * 100)

    # ── Schema/ACL compliance (partial) ──────────────────────────────────────
    sc = max(0.0, min(1.0, float(schema_acl_compliance)))
    schema_pts = round(W_SCHEMA_ACL_COMPLIANCE * sc)
    raw += schema_pts
    evidence["schema_acl_compliance"] = schema_pts

    # ── Entity-graph compliance (partial) ────────────────────────────────────
    eg = max(0.0, min(1.0, float(entity_graph_compliance)))
    eg_pts = round(W_ENTITY_GRAPH_COMPLIANCE * eg)
    raw += eg_pts
    evidence["entity_graph_compliance"] = eg_pts

    # ── Repair dimension (mutually exclusive) ────────────────────────────────
    if not had_repair:
        raw += W_NO_REPAIR
        evidence["repair"] = f"no_repair (+{W_NO_REPAIR})"
    elif repair_succeeded:
        raw += W_SUCCESSFUL_REPAIR
        evidence["repair"] = f"repair_succeeded (+{W_SUCCESSFUL_REPAIR})"
    else:
        evidence["repair"] = "repair_failed (+0)"

    # ── Usable result ────────────────────────────────────────────────────────
    if execution_success and row_count > 0:
        raw += W_USABLE_RESULT
        evidence["usable_result"] = W_USABLE_RESULT
    elif row_count == 0 and execution_success:
        evidence["usable_result"] = 0
        evidence["zero_rows_note"] = "zero rows — not penalised, not rewarded"
    else:
        evidence["usable_result"] = 0

    # ── Feedback adjustment ───────────────────────────────────────────────────
    evidence["technical_score"] = raw
    if feedback_delta != 0:
        evidence["feedback_delta"] = feedback_delta

    final = max(0, raw + feedback_delta)
    evidence["final_score"] = final
    return final, evidence


def classify_score(
    score: int,
    has_net_negative_feedback: bool = False,
) -> str:
    """
    Map a score (and optional negative-feedback flag) to a candidate type.

    Returns one of: "positive" | "review" | "negative"
    """
    if has_net_negative_feedback or score < THRESHOLD_REVIEW_LO:
        return "negative"
    if score >= THRESHOLD_POSITIVE:
        return "positive"
    return "review"


def net_feedback_delta(positive_votes: int, negative_votes: int) -> int:
    """
    Convert raw vote counts to a single feedback delta.

    Rules:
      net > 0  (more ups than downs)  → FEEDBACK_POSITIVE  (+10)
      net < 0  (more downs than ups)  → FEEDBACK_NEGATIVE  (-30)
      net == 0 (tied or no votes)     → 0

    This prevents a single active user from gaming the score with
    multiple votes while still capturing the direction of feedback.
    """
    net = positive_votes - negative_votes
    if net > 0:
        return FEEDBACK_POSITIVE
    if net < 0:
        return FEEDBACK_NEGATIVE
    return 0
