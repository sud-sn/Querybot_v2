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
