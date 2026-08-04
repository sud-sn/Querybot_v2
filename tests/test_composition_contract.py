from __future__ import annotations

from pathlib import Path

import pytest

from core.analysis_contract import build_analysis_contract, format_analysis_contract
from core.chart import build_chart_payload, detect_chart_type
from core.chart_spec import infer_chart_spec
from core.contribution_analysis import (
    detect_composition_intent,
    detect_contribution_intent,
    infer_numeric_col,
)
from core.distribution_analysis import detect_histogram_intent
from core.insight import detect_analytical_intents
from core.query_semantics import analyze_query_intent
from core.validator import validate_sql_detailed


COMPOSITION_QUESTION = "Give me the distribution of revenue by pharmacy states"


def test_distribution_by_category_is_composition_not_histogram():
    assert detect_composition_intent(COMPOSITION_QUESTION)
    assert detect_contribution_intent(COMPOSITION_QUESTION)
    # The low-level lexical detector still sees "distribution of"; the unified
    # resolver is responsible for making the two routes mutually exclusive.
    assert detect_histogram_intent(COMPOSITION_QUESTION)
    intents = detect_analytical_intents(COMPOSITION_QUESTION)
    assert intents["contribution"] is True
    assert intents["histogram"] is False
    assert analyze_query_intent(COMPOSITION_QUESTION)["wants_share"] is True


@pytest.mark.parametrize(
    "question",
    [
        "Show the distribution of net revenue by pharmacy state as a pie chart",
        "Show the distribution of gross revenue by customer region",
        "Show the distribution of booked revenue across business unit",
        "Show the distribution of average unit cost grouped by supplier",
        "Show the distribution of custom contract value per account tier",
    ],
)
def test_modified_and_client_specific_measures_keep_category_grain(question):
    intents = detect_analytical_intents(question)
    assert intents["contribution"] is True
    assert intents["histogram"] is False


def test_statistical_distribution_remains_histogram():
    question = "Show the distribution of individual transaction amounts"
    intents = detect_analytical_intents(question)
    assert intents["contribution"] is False
    assert intents["histogram"] is True
    assert analyze_query_intent(question)["wants_share"] is False


def test_explicit_statistical_words_override_composition_phrase():
    question = "Show the distribution of revenue by state in histogram bins"
    assert not detect_composition_intent(question)
    intents = detect_analytical_intents(question)
    assert intents["histogram"] is True
    assert intents["contribution"] is False


def test_unregistered_additive_measure_builds_governed_contract():
    plan = {
        "enabled": True,
        "fields": [
            {
                "term": "net revenue",
                "table": "ERP.F_SALES",
                "column": "NET_REVENUE_AMT",
                "role": "measure",
                "source": "semantic_field",
                "confidence": 91,
            },
            {
                "term": "customer region",
                "table": "ERP.D_CUSTOMER",
                "column": "REGION_NAME",
                "role": "display_dimension",
                "source": "semantic_model",
                "enforcement": "required",
            },
        ],
    }
    contract = build_analysis_contract(
        "distribution of net revenue by customer region",
        analytical_intents={"contribution": True},
        semantic_plan=plan,
        metric_formulas=[],
    )
    assert contract["enabled"] is True
    assert contract["mode"] == "composition"
    assert contract["measure_source"] == "semantic_field"
    assert contract["measures"][0]["classification"] == "additive"
    assert contract["dimensions"][0]["column"] == "REGION_NAME"
    rendered = format_analysis_contract(contract)
    assert "one row per requested business category" in rendered
    assert "ERP.F_SALES.NET_REVENUE_AMT" in rendered
    assert "not a statistical histogram" in rendered


def test_approved_metric_formula_takes_precedence_in_contract():
    contract = build_analysis_contract(
        "revenue distribution by region",
        analytical_intents={"contribution": True},
        semantic_plan={
            "enabled": True,
            "fields": [{
                "term": "revenue",
                "table": "CRM.F_INVOICE",
                "column": "REVENUE_AMT",
                "role": "measure",
            }],
        },
        metric_formulas=[{
            "name": "Governed revenue",
            "sql_template": "SUM(CASE WHEN IS_POSTED = 1 THEN REVENUE_AMT ELSE 0 END)",
        }],
    )
    assert contract["measure_source"] == "approved_metric"
    assert len(contract["measures"]) == 1
    assert contract["measures"][0]["authoritative"] is True
    assert "CASE WHEN IS_POSTED" in format_analysis_contract(contract)


def test_non_additive_unregistered_measure_is_not_declared_summable():
    contract = build_analysis_contract(
        "distribution of margin rate by region",
        analytical_intents={"contribution": True},
        semantic_plan={
            "enabled": True,
            "fields": [{
                "term": "margin rate",
                "table": "FIN.F_PROFIT",
                "column": "MARGIN_PCT",
                "role": "measure",
            }],
        },
        metric_formulas=[],
    )
    assert contract["measures"][0]["classification"] == "non_additive"
    assert "never SUM" in format_analysis_contract(contract)


def _composition_context():
    return {
        "production_sql": True,
        "question": COMPOSITION_QUESTION,
        "analysis_contract": {
            "enabled": True,
            "mode": "composition",
            "requires_grouping": True,
            "requires_aggregate": True,
        },
    }


TABLE_COLUMNS = {
    "ERP.F_SALES": {
        "STATE_CODE": "varchar",
        "NET_REVENUE_AMT": "decimal",
    }
}


def test_composition_validator_rejects_row_level_histogram_shape():
    result = validate_sql_detailed(
        "SELECT STATE_CODE, NET_REVENUE_AMT FROM ERP.F_SALES",
        set(TABLE_COLUMNS),
        "azure_sql",
        table_columns=TABLE_COLUMNS,
        semantic_context=_composition_context(),
    )
    assert result.ok is False
    assert result.code == "composition_shape"


def test_composition_validator_rejects_overall_total_without_category_grain():
    result = validate_sql_detailed(
        "SELECT SUM(NET_REVENUE_AMT) AS NET_REVENUE FROM ERP.F_SALES",
        set(TABLE_COLUMNS),
        "azure_sql",
        table_columns=TABLE_COLUMNS,
        semantic_context=_composition_context(),
    )
    assert result.ok is False
    assert result.code == "composition_shape"


def test_composition_validator_accepts_grouped_aggregate():
    result = validate_sql_detailed(
        "SELECT STATE_CODE, SUM(NET_REVENUE_AMT) AS NET_REVENUE "
        "FROM ERP.F_SALES GROUP BY STATE_CODE",
        set(TABLE_COLUMNS),
        "azure_sql",
        table_columns=TABLE_COLUMNS,
        semantic_context=_composition_context(),
    )
    assert result.ok is True, result.reason


def test_composition_validator_rejects_direct_non_additive_aggregation():
    columns = {
        "FIN.F_MARGIN": {
            "REGION": "varchar",
            "MARGIN_PCT": "decimal",
        }
    }
    context = _composition_context()
    context["analysis_contract"] = {
        **context["analysis_contract"],
        "measures": [{
            "name": "margin rate",
            "table": "FIN.F_MARGIN",
            "column": "MARGIN_PCT",
            "source": "semantic_field",
            "classification": "non_additive",
        }],
    }
    result = validate_sql_detailed(
        "SELECT REGION, AVG(MARGIN_PCT) AS MARGIN_PCT "
        "FROM FIN.F_MARGIN GROUP BY REGION",
        set(columns),
        "azure_sql",
        table_columns=columns,
        semantic_context=context,
    )
    assert result.ok is False
    assert result.code == "composition_shape"
    assert "non-additive" in result.reason


def _share_rows():
    return [
        {"STATE": "RI", "NET_REVENUE": 10933.0, "contribution_pct": 30.0},
        {"STATE": "MA", "NET_REVENUE": 10401.0, "contribution_pct": 28.0},
        {"STATE": "CT", "NET_REVENUE": 9100.0, "contribution_pct": 25.0},
        {"STATE": "NY", "NET_REVENUE": 6200.0, "contribution_pct": 17.0},
    ]


def test_small_composition_with_postprocessed_share_defaults_to_pie():
    rows = _share_rows()
    spec = infer_chart_spec(rows, COMPOSITION_QUESTION)
    assert spec["recommended_type"] == "pie"
    assert spec["y"][0]["column"] == "NET_REVENUE"
    assert detect_chart_type(rows, COMPOSITION_QUESTION) == "pie"
    payload = build_chart_payload(
        rows, None, question=COMPOSITION_QUESTION, title=COMPOSITION_QUESTION
    )
    assert payload["chart_type"] == "pie"
    assert payload["y_keys"] == ["NET_REVENUE"]


def test_negative_composition_values_fall_back_to_bar():
    rows = [
        {"REGION": "North", "NET_PROFIT": 100.0},
        {"REGION": "South", "NET_PROFIT": -25.0},
    ]
    spec = infer_chart_spec(rows, "distribution of profit by region")
    assert spec["recommended_type"] == "bar"
    assert any("negative values" in warning for warning in spec["warnings"])


def test_large_composition_uses_readable_bar_fallback():
    rows = [
        {"STATE": f"S{i}", "NET_REVENUE": float(i + 1)}
        for i in range(12)
    ]
    spec = infer_chart_spec(rows, "distribution of revenue by state")
    assert spec["recommended_type"] == "bar"
    assert "pie" not in spec["allowed_types"]


def test_contribution_postprocessing_ignores_numeric_identifier_column():
    rows = [
        {"PHARMACY_ID": 101, "NET_REVENUE": 500.0},
        {"PHARMACY_ID": 102, "NET_REVENUE": 750.0},
    ]
    assert infer_numeric_col(rows) == "NET_REVENUE"


def test_query_pipeline_carries_contract_to_generation_and_validation():
    source = Path("core/query_pipeline.py").read_text(encoding="utf-8")
    assert "format_analysis_contract(_analysis_contract)" in source
    assert '"analysis_contract": _analysis_contract' in source
    assert 'last_code == "composition_shape"' in source
