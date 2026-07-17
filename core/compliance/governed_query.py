from __future__ import annotations

from dataclasses import dataclass

import store
from core.compliance.models import PolicyContext, PolicyDecision
from core.compliance.policy_engine import evaluate
from core.compliance.result_guard import protect_rows
from core.compliance.sql_guard import SqlPolicyAnalysis, analyze_sql, inject_row_policies
from core.schema import run_query
from core.validator import validate_sql


class PolicyDeniedError(PermissionError):
    def __init__(self, decision: PolicyDecision):
        super().__init__(decision.explanation or decision.reason_code)
        self.decision = decision


@dataclass
class GovernedQueryResult:
    rows: list[dict]
    sql: str
    decision: PolicyDecision
    analysis: SqlPolicyAnalysis
    row_obligations: list[dict]


def execute_governed_query(
    credentials: dict,
    db_type: str,
    sql: str,
    *,
    context: PolicyContext,
    known_tables: set[str],
    table_columns: dict[str, dict[str, str]] | None = None,
    allowed_tables: set[str] | None = None,
    semantic_context: dict | None = None,
    max_rows: int = 200,
) -> GovernedQueryResult:
    ok, reason, code = validate_sql(
        sql,
        known_tables,
        db_type,
        allowed_tables,
        table_columns,
        semantic_context,
    )
    if not ok:
        raise ValueError(f"{code}: {reason}")

    analysis = analyze_sql(sql, db_type)
    decision = evaluate(context, analysis.resources)
    if analysis.has_star:
        classified = bool(decision.masking or decision.aggregate_only)
        store_classified = store.get_classification_map(context.account_id)
        if classified or store_classified:
            used_tables = set(analysis.tables)
            if any(
                key.rsplit(".", 1)[0] in used_tables
                or any(key.rsplit(".", 1)[0].endswith("." + table) for table in used_tables)
                for key in store_classified
            ):
                decision.allowed = False
                decision.reason_code = "classified_select_star"
                decision.explanation = "SELECT * is blocked on classified tables."
    if not decision.effective_allowed:
        raise PolicyDeniedError(decision)

    aggregate_sources = {
        source
        for output, sources in analysis.lineage.items()
        if output in analysis.aggregate_outputs
        for source in sources
    }
    required_aggregate = {resource.key for resource in decision.aggregate_only}
    if required_aggregate - aggregate_sources:
        decision.allowed = False
        decision.reason_code = "aggregate_only_violation"
        decision.explanation = "One or more fields may only be returned as aggregates."
        if not decision.shadow:
            raise PolicyDeniedError(decision)

    rewritten_sql, row_obligations = inject_row_policies(sql, db_type, context)
    raw_rows = run_query(credentials, db_type, rewritten_sql, max_rows=max_rows)

    release_context = PolicyContext(**{**context.__dict__, "action": "result_release"})
    release_decision = evaluate(release_context, analysis.resources)
    if not release_decision.effective_allowed:
        raise PolicyDeniedError(release_decision)
    combined_masking = {**decision.masking, **release_decision.masking}

    # ── Per-user attestation override ────────────────────────────────────────
    # An internal user with a recorded (unrevoked) confidentiality attestation
    # sees real values instead of masked ones. Display-side only: the LLM
    # boundary is untouched (llm_context gating and result_llm_features_allowed
    # don't consult this), and query-shape policies (aggregate_only, row
    # obligations) still apply — this releases only the masking obligations.
    # Every unmasked release is written to the hash-chained decision log so
    # an auditor can answer "who saw unmasked values, and when".
    if combined_masking and store.user_attestation_valid(
        context.account_id, context.user_id
    ):
        try:
            store.log_policy_decision(
                account_id=context.account_id,
                user_id=context.user_id,
                action="result_release",
                purpose_id=context.purpose_id,
                channel=context.channel,
                allowed=True,
                reason_code="attested_unmasked_release",
                resources=sorted(combined_masking.keys()),
                obligations={"masking_waived": combined_masking},
                policy_version=context.policy_version or 0,
            )
            combined_masking = {}
        except Exception:
            # An unmasked release without its audit row is worse than a
            # masked result — if the log can't be written, keep the masking.
            pass

    release_decision.masking = combined_masking
    release_decision.row_obligations = row_obligations
    rows = protect_rows(
        raw_rows,
        release_decision,
        analysis.lineage,
        account_id=context.account_id,
        mask_exempt_outputs=analysis.mask_exempt_outputs,
    )
    return GovernedQueryResult(
        rows=rows,
        sql=rewritten_sql,
        decision=release_decision,
        analysis=analysis,
        row_obligations=row_obligations,
    )
