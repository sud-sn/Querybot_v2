from __future__ import annotations

from dataclasses import dataclass

import store
from core.compliance.models import PolicyContext, PolicyDecision
from core.compliance.policy_engine import evaluate
from core.compliance.result_guard import protect_rows
from core.compliance.sql_guard import (
    SqlPolicyAnalysis,
    aggregate_only_violations,
    analyze_sql,
    inject_row_policies,
)
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
    # True when the underlying fetch stopped at its row cap, so ``rows`` is a
    # truncated prefix rather than the whole result. Consumers that aggregate
    # across rows (quartiles, histograms, correlation, cohort matrices) must
    # refuse rather than present a statistic computed over the prefix.
    truncated: bool = False


def governed_reader_for_user(
    account_id: str,
    user: dict | None,
    *,
    channel: str,
    purpose_id: str = "",
    db_cfg: dict | None = None,
):
    """A ``run(sql) -> GovernedQueryResult`` closure scoped to `user`'s row
    policy and table grants, or ``None`` when there is no user to scope the
    read to.

    For diagnostic reads that run ALONGSIDE a main governed answer -- a
    coverage-gap probe (core/date_coverage.py), a zero-row table count
    (core/pipeline_helpers.py) -- and that used to bypass governance
    entirely via a raw core.schema.run_query call. That is the exact defect
    class already fixed for core/alert_engine.py's date-anchor probe (see
    that module's ``_owner_execution``, which this mirrors): a user whose
    row policy restricts them to a subset of a shared fact table got a
    coverage verdict or a zero-row diagnosis computed over rows they are not
    authorized to see.

    Returns ``None`` rather than an ungoverned fallback closure when
    `account_id` or `user` is falsy -- the CALLER decides whether an
    ungoverned read is still acceptable for its own diagnostic (and must say
    so explicitly, loudly), this function will not make that call silently.

    Not for the main answer's own execution, which already builds its own
    context once per request and should keep doing so -- core/query_pipeline
    .py's ``_execute_with_policy`` closure is that context, already built
    with the request's real known_tables/table_columns; a diagnostic running
    in the SAME request should reuse it rather than calling this a second
    time to rebuild an equivalent one.
    """
    if not account_id or not user:
        return None

    import store

    from core.compliance.policy_engine import resolve_context
    from core.schema import load_known_tables, load_schema_columns

    state = store.get_client_state(account_id) or {}
    context = resolve_context(
        account_id, user, action="query_execution",
        channel=channel, purpose_id=purpose_id,
    )
    known_tables = load_known_tables(state.get("schema_dir", ""))
    table_columns = load_schema_columns(state.get("schema_dir", ""))
    allowed_tables = store.get_allowed_tables(user)
    cfg = db_cfg or {}
    credentials = cfg.get("credentials") or cfg
    db_type = cfg.get("db_type", "azure_sql")

    def run(sql: str) -> GovernedQueryResult:
        return execute_governed_query(
            credentials, db_type, sql,
            context=context, known_tables=known_tables,
            table_columns=table_columns, allowed_tables=allowed_tables,
        )

    return run


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

    required_aggregate = {resource.key for resource in decision.aggregate_only}
    if aggregate_only_violations(analysis, required_aggregate):
        decision.allowed = False
        decision.reason_code = "aggregate_only_violation"
        decision.explanation = "One or more fields may only be returned as aggregates."
        if not decision.shadow:
            raise PolicyDeniedError(decision)

    rewritten_sql, row_obligations = inject_row_policies(sql, db_type, context)
    raw_rows = run_query(credentials, db_type, rewritten_sql, max_rows=max_rows)
    # Read the flag off the fetch result immediately: masking and row-policy
    # transforms below return plain lists and would drop it.
    rows_truncated = bool(getattr(raw_rows, "truncated", False))

    release_context = PolicyContext(**{**context.__dict__, "action": "result_release"})
    release_decision = evaluate(release_context, analysis.resources)
    if decision.reason_code == "aggregate_only_violation":
        # Only reachable in shadow mode -- enforce mode already raised above.
        # release_decision comes from a SEPARATE evaluate() call, scoped to
        # the result_release action, which knows nothing about a violation
        # found against the query_execution decision. Left unpropagated, the
        # decision this function actually RETURNS reports "policy_allow" for
        # a query shadow mode was specifically supposed to flag -- undetected
        # in shadow mode is worse than undetected nowhere, because shadow mode
        # exists so an operator can see what enforce mode would have blocked.
        release_decision.allowed = False
        release_decision.reason_code = "aggregate_only_violation"
        release_decision.explanation = decision.explanation
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
        truncated=rows_truncated,
    )
