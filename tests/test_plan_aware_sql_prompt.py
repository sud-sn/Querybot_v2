from core.llm import build_sql_system_prompt
from core.pipeline_helpers import allow_progressive_sql_repair


def _semantic_plan(**request_overrides):
    request = {
        "status": "compiled",
        "question": "What is total revenue?",
        "intent": "metric_query",
        "source_fact": "sales.fact_invoice",
        "source_facts": ["sales.fact_invoice"],
        "subrequests": [],
        "measures": [{"table": "sales.fact_invoice", "column": "revenue"}],
        "dimensions": [],
        "temporal_operations": [],
        "joins": [],
        "metrics": [{"name": "Revenue", "formula": "SUM(revenue)"}],
        "derived_measure": {},
        "filters": [],
        "time_range": "",
        "quarter_periods": [],
        "comparison": "",
        "top_n": None,
    }
    request.update(request_overrides)
    return {
        "enabled": True,
        "fields": [{"term": "revenue", "table": "sales.fact_invoice", "column": "revenue"}],
        "analytical_request_plan": request,
    }


def test_direct_call_without_compiled_plan_keeps_legacy_rule_catalogue():
    prompt = build_sql_system_prompt("azure_sql", "Table sales.fact_invoice(revenue)")

    assert "CORRELATION / SCATTER RULE" in prompt
    assert "FACT-TO-FACT JOIN RULE" in prompt
    assert "MOVING AVERAGE RULE" in prompt


def test_scalar_metric_prompt_drops_irrelevant_advanced_rules():
    prompt = build_sql_system_prompt(
        "azure_sql",
        "Table sales.fact_invoice(revenue, invoice_date_key)",
        semantic_plan=_semantic_plan(),
    )

    assert "APPROVED METRIC FORMULA RULE" in prompt
    assert "CORRELATION / SCATTER RULE" not in prompt
    assert "FACT-TO-FACT JOIN RULE" not in prompt
    assert "MOVING AVERAGE RULE" not in prompt
    assert "MONTH-OVER-MONTH / QUARTER-OVER-QUARTER RULE" not in prompt
    assert "Knowledge Base — available tables" in prompt


def test_temporal_comparison_keeps_only_relevant_date_and_comparison_rules():
    prompt = build_sql_system_prompt(
        "azure_sql",
        "Table sales.fact_invoice(revenue, invoice_date_key); Table shared.date_dim(date_key, full_date)",
        semantic_plan=_semantic_plan(
            question="Compare revenue for the current month and previous month",
            intent="comparison",
            temporal_operations=[{"kind": "relative_window", "role": "invoice_date"}],
            joins=[{"left_table": "sales.fact_invoice", "right_table": "shared.date_dim"}],
            time_range="current month and previous month",
            comparison="period_or_segment_comparison",
        ),
    )

    assert "MONTH-OVER-MONTH / QUARTER-OVER-QUARTER RULE" in prompt
    assert "DATE-DIMENSION SURROGATE KEY RULE" in prompt
    assert "NULL-SAFE JOIN RULE" in prompt
    assert "MOVING AVERAGE RULE" not in prompt
    assert "ANTI-JOIN RULE" not in prompt


def test_compound_plan_keeps_fact_isolation_rule():
    prompt = build_sql_system_prompt(
        "azure_sql",
        "Table sales.fact_invoice; Table inventory.fact_snapshot; Table shared.warehouse",
        semantic_plan=_semantic_plan(
            question="Compare revenue with inventory by warehouse",
            intent="comparison",
            source_facts=["sales.fact_invoice", "inventory.fact_snapshot"],
            subrequests=[
                {"source_fact": "sales.fact_invoice", "aggregate_before_join": True},
                {"source_fact": "inventory.fact_snapshot", "aggregate_before_join": True},
            ],
            dimensions=[{"table": "shared.warehouse", "column": "warehouse_name"}],
            comparison="period_or_segment_comparison",
        ),
    )

    assert "FACT-TO-FACT JOIN RULE" in prompt
    assert "STAR-SCHEMA JOIN ORDER RULE" in prompt
    assert "CORRELATION / SCATTER RULE" not in prompt


def test_ranking_and_threshold_plan_keeps_having_and_ranking_rules():
    prompt = build_sql_system_prompt(
        "azure_sql",
        "Table sales.fact_invoice; Table shared.customer",
        semantic_plan=_semantic_plan(
            question="Show the top 5 customers with revenue above 350",
            intent="ranking",
            dimensions=[{"table": "shared.customer", "column": "customer_name"}],
            top_n=5,
        ),
    )

    assert "HAVING RULE" in prompt
    assert "RANKING RULE" in prompt
    assert "STAR-SCHEMA JOIN ORDER RULE" in prompt
    assert "MOVING AVERAGE RULE" not in prompt


def test_progressive_repair_allows_one_changed_reason_code():
    assert allow_progressive_sql_repair(
        {"graph_plan_mismatch"},
        "field_plan_mismatch",
        1,
    )


def test_progressive_repair_stops_same_reason_code_loop():
    assert not allow_progressive_sql_repair(
        {"field_plan_mismatch"},
        "field_plan_mismatch",
        1,
    )


def test_progressive_repair_honours_hard_attempt_cap():
    assert not allow_progressive_sql_repair(
        {"graph_plan_mismatch"},
        "field_plan_mismatch",
        2,
    )


def test_query_pipeline_wires_bounded_progressive_repair():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "allow_progressive_sql_repair(" in source
    assert 'max_attempts=2' in source
    assert 'component="sql_repair_progressive"' in source
    assert "repair_attempt" in source


def test_compiled_plan_gates_even_when_no_question_is_supplied():
    """The two gates are independent.

    rule_applies() returns True for every gated rule when there is no question
    to match against -- that is its keep-everything default, not a verdict. The
    second gate used to read that default as an endorsement and skip its own
    filtering entirely, so any caller that omitted the question received the
    full rule catalogue despite having a compiled plan.
    """
    prompt = build_sql_system_prompt(
        "azure_sql",
        "Table sales.fact_invoice(revenue, invoice_date_key)",
        semantic_plan=_semantic_plan(),
    )
    for rule in (
        "CORRELATION / SCATTER RULE",
        "MOVING AVERAGE RULE",
        "FACT-TO-FACT JOIN RULE",
        "MONTH-OVER-MONTH / QUARTER-OVER-QUARTER RULE",
    ):
        assert rule not in prompt, (
            f"{rule} survived a compiled scalar-metric plan; the second gate is inert"
        )


def test_a_question_can_still_protect_a_rule_the_plan_would_cut():
    """The reason the first gate is allowed to vote at all: a question that
    plainly asks for a correlation must keep the correlation rule even when the
    compiled plan is a plain scalar metric that never mentions one."""
    plan = _semantic_plan()
    without = build_sql_system_prompt(
        "azure_sql", "Table sales.fact_invoice(revenue, discount)",
        semantic_plan=plan, question="what is total revenue?",
    )
    with_q = build_sql_system_prompt(
        "azure_sql", "Table sales.fact_invoice(revenue, discount)",
        semantic_plan=plan, question="is revenue correlated with discount?",
    )
    assert "CORRELATION / SCATTER RULE" not in without
    assert "CORRELATION / SCATTER RULE" in with_q, (
        "the question gate must be able to protect a rule the plan would drop"
    )


def test_every_sql_prompt_call_site_passes_the_question():
    """Structural guard. The progressive-repair call site omitted question=,
    which silently disabled rule gating on exactly the path that most needed a
    focused prompt -- the retry after a failure. A missing keyword argument
    leaves no trace at runtime, so pin it here."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "build_sql_system_prompt"
    ]
    assert sites, "no build_sql_system_prompt call sites found"
    missing = [n.lineno for n in sites if "question" not in {k.arg for k in n.keywords}]
    assert not missing, (
        f"build_sql_system_prompt called without question= at line(s) {missing}; "
        "rule gating is disabled there"
    )
