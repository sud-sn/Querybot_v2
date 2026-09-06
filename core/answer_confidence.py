"""How much friction an answer met, said in the reader's own language.

Every string here used to be an English literal and this module imported no
i18n at all — so the one panel a reader opens to decide whether to trust a
number was the one panel that never spoke their language. A French user got
"Confiance" as a heading over "SQL passed schema validation."

The ids resolve through the ambient language ContextVar, which both callers
already run inside: core.result_renderer._send_results and
core.pipeline_helpers._build_zero_row_message are reached from
_handle_query_impl, which activates the reader's language around the whole
turn. `lang` is still accepted for a caller that has one and no activation.
"""

from __future__ import annotations

from typing import Any

from core.i18n import format_count, plural, t


def _level(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _label(level: str, lang: str | None = None) -> str:
    return t(f"confidence.level.{level if level in _LEVELS else 'medium'}", lang=lang)


_LEVELS = ("high", "medium", "low")


def build_answer_confidence(
    *,
    validation_code: str = "ok",
    row_count: int | None = None,
    retry_count: int = 0,
    has_semantic_plan: bool = False,
    has_graph_context: bool = False,
    tables_used: list[str] | None = None,
    empty_tables: list[str] | None = None,
    null_metric_issue: bool = False,
    derived_metric_gap: str = "",
    weak_retrieval: bool = False,
    retrieval_unscored: bool = False,
    zero_match_result: bool = False,
    graph_scope: str = "",
    graph_resolution_failed: bool = False,
    semantic_planning_failed: bool = False,
    fanout_risk: bool = False,
    result_verification: dict[str, Any] | None = None,
    candidate_selection: dict[str, Any] | None = None,
    corroboration: dict[str, Any] | None = None,
    lang: str | None = None,
) -> dict[str, Any]:
    """
    Convert technical query signals into a compact business-facing confidence score.

    The score is intentionally simple and deterministic. It is not a truth
    guarantee; it tells the user how much friction the answer encountered.

    zero_match_result: True for a single-row diagnostic aggregate whose own
    match-count column is zero (see response_builder.detect_zero_match_result)
    -- a real physical row exists, but it represents no matching data, not a
    successful single-value answer. Scored as if row_count were 0 regardless
    of the physical count passed in, and mutually exclusive with
    null_metric_issue (a real, non-zero match count with a missing metric).
    """
    validation = (validation_code or "ok").lower()
    rows = 0 if row_count is None else max(int(row_count), 0)
    if zero_match_result:
        rows = 0
        null_metric_issue = False
    retries = max(int(retry_count or 0), 0)
    used_tables = [str(t) for t in (tables_used or []) if str(t).strip()]
    empty = [str(t) for t in (empty_tables or []) if str(t).strip()]

    score = 70
    reasons: list[str] = []
    warnings: list[str] = []

    if validation in {"ok", "pass", "trusted_metric"}:
        score += 15
        reasons.append(t("confidence.reason.validation_passed", lang=lang))
    else:
        # validation is always a non-empty string here (normalised above)
        score -= 25
        warnings.append(t("confidence.warn.validation_attention", lang=lang))

    if retries:
        score -= min(20, 10 * retries)
        warnings.append(t("confidence.warn.repair_retry", lang=lang))
    else:
        reasons.append(t("confidence.reason.no_retry", lang=lang))

    if row_count is not None:
        if rows > 0:
            score += 10
            reasons.append(plural(
                "confidence.reason.rows_returned", rows, lang=lang,
                # French takes the singular at zero as well as one, and writes
                # thousands with a narrow no-break space -- neither of which
                # an f-string with a bolted-on "s" can produce.
                rows=format_count(rows, lang=lang),
            ))
        else:
            score -= 20
            warnings.append(t("confidence.warn.no_rows", lang=lang))

    if empty:
        score -= 35
        listed = ", ".join(empty[:3])
        suffix = "..." if len(empty) > 3 else ""
        warnings.append(t("confidence.warn.empty_table", lang=lang,
                          tables=f"{listed}{suffix}"))
    elif used_tables:
        reasons.append(t("confidence.reason.known_tables", lang=lang))

    if null_metric_issue:
        score -= 25
        warnings.append(t("confidence.warn.null_metric", lang=lang))

    if derived_metric_gap:
        score -= 25
        warnings.append(t("confidence.warn.derived_metric_gap", lang=lang,
                          metric=derived_metric_gap))

    if weak_retrieval:
        score -= 20
        warnings.append(t("confidence.warn.weak_retrieval", lang=lang))
    elif retrieval_unscored:
        # The re-ranker produced no scores, so the relevance floor never ran and
        # weak_retrieval could not be raised. Retrieval was unfiltered — which
        # is not the same as relevant, and used to be indistinguishable from it.
        score -= 10
        warnings.append(t("confidence.warn.retrieval_unscored", lang=lang))

    if has_semantic_plan:
        score += 5
        reasons.append(t("confidence.reason.semantic_plan", lang=lang))
    elif semantic_planning_failed:
        # Field planning RAISED rather than finding nothing. Four guarantees
        # went with it in one step: term-to-column bindings, the required join
        # path, the superseded-column list, and the temporal policy that scopes
        # the window. The answer was written from raw prose and looks identical
        # to a planned one.
        score -= 30
        warnings.append(t("confidence.warn.semantic_planning_failed", lang=lang))

    if has_graph_context:
        if str(graph_scope or "").lower() == "suggested_fallback":
            score -= 35
            warnings.append(t("confidence.warn.suggested_relationships", lang=lang))
        else:
            score += 5
            reasons.append(t("confidence.reason.graph_used", lang=lang))
    elif graph_resolution_failed and len(set(used_tables)) > 1:
        # Entity-graph resolution raised instead of returning "no graph", so
        # nothing checked the joins in this SQL against the approved ones. A
        # single-table answer has no joins to check and is unaffected; a
        # multi-table one is the model's own join plan, executed. Without this
        # the answer scored identically to a query that needed no governance.
        score -= 35
        warnings.append(t("confidence.warn.graph_resolution_failed", lang=lang))

    if fanout_risk:
        score -= 35
        warnings.append(t("confidence.warn.fanout_risk", lang=lang))

    # When the plan left a decision open, the pipeline asks the question a
    # second way and lets the verifier choose (core.candidate_selection). Two
    # candidates the verifier likes equally, returning different numbers, mean
    # the question was ambiguous in a way the plan did not capture -- the
    # answer shown is one of them, and saying so is the difference between an
    # explanation and a coin toss presented as fact.
    selection = candidate_selection or {}
    selection_reason = str(selection.get("reason") or "")
    if selection_reason == "verified_candidates_disagree":
        score = min(score - 25, 49)
        warnings.append(t("confidence.warn.candidates_disagree", lang=lang))
    elif selection_reason == "no_candidate_verified":
        score -= 15
        warnings.append(t("confidence.warn.no_candidate_verified", lang=lang))
    elif selection_reason.startswith("agreement_of_"):
        score += 5
        reasons.append(t("confidence.reason.candidates_agree", lang=lang))

    # A second subject area was asked the same question and its answer
    # compared (core.corroboration_run). Scored separately from the candidate
    # check above because it is a different claim: two candidates disagreeing
    # means the QUESTION was ambiguous, two areas disagreeing means the
    # BUSINESS has two answers to it -- and only one of those is something the
    # reader can resolve by rephrasing.
    #
    # `checked` is the gate, and it is the whole point of the field: a second
    # opinion that failed to run reports checked=False, and neither the credit
    # nor the warning applies. Reading `agrees` alone would score every
    # failed run as a disagreement.
    second_opinion = corroboration or {}
    if second_opinion.get("checked"):
        # `line` is the reader-facing sentence, already translated and already
        # carrying both figures and the gap -- see
        # core.corroboration_run.describe_second_opinion. Scoring does not do
        # i18n, so it takes the sentence when the caller rendered one and
        # states the finding plainly when nobody did.
        line = str(second_opinion.get("line") or "").strip()
        if second_opinion.get("agrees"):
            score += 5
            reasons.append(
                line or t("confidence.reason.second_area_agrees", lang=lang))
        else:
            # 49, not 59. The reader-facing surface shows a warning inline
            # only when the verdict is LOW, and puts everything else behind a
            # collapsed disclosure -- so a cap at 59 landed one point above
            # the threshold and hid the disagreement behind a click nobody has
            # a reason to make. The candidate-disagreement branch above caps
            # at 49 for the same reason. Two areas disagreeing is at least as
            # serious as two candidates disagreeing: it means the business has
            # two answers to the question, which the reader cannot resolve by
            # rephrasing it.
            score = min(score - 20, 49)
            warnings.append(
                line or t("confidence.warn.second_area_disagrees", lang=lang))

    verification = result_verification or {}
    verification_status = str(verification.get("status") or "").lower()
    if verification_status == "unavailable":
        # The shape check could not run. Absent verification used to cost
        # nothing, so a failed check scored identically to a passed one — the
        # defence against a schema-valid but business-wrong answer reporting
        # "no issues found" precisely because it never looked.
        score -= 15
        warnings.append(t("confidence.warn.verification_unavailable", lang=lang))
    elif verification_status == "pass":
        score += 5
        reasons.append(t("confidence.reason.shape_matched", lang=lang))
    elif verification_status in {"warning", "fail"}:
        score -= 10 if verification_status == "warning" else 30
        details = list(verification.get("errors") or []) + list(
            verification.get("warnings") or []
        )
        # `details` comes from core.result_verifier and is English today;
        # the fallback beside it is not, so the reader gets their own
        # language whenever the verifier had nothing specific to say.
        warnings.append(
            str(details[0]) if details
            else t("confidence.warn.shape_mismatch", lang=lang)
        )
        if verification_status == "fail":
            # A safe, executable query can still answer the wrong shape.  Do
            # not present that result as medium/high confidence merely because
            # schema validation passed.
            score = min(score, 49)

    # A repaired query may be usable, but compilation and execution alone do
    # not prove business correctness. Never present a repaired result as high
    # confidence until it is covered by deterministic result assertions.
    if retries:
        score = min(score, 75)

    score = max(0, min(100, score))
    level = _level(score)
    return {
        "score": score,
        "level": level,
        "label": _label(level, lang),
        "reasons": reasons[:5],
        "warnings": warnings[:5],
    }
