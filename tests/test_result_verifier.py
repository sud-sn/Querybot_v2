from decimal import Decimal

from core.analytical_intent import AnalyticalPlan
from core.answer_confidence import build_answer_confidence
from core.result_cache import ResultCache
from core.result_commands import execute_result_command, parse_result_command
from core.result_verifier import verify_result_shape


def test_metric_registry_binding_wins_and_valid_numeric_result_passes():
    report = verify_result_shape(
        [{"month": "2026-01", "revenue": Decimal("125.50")},
         {"month": "2026-02", "revenue": Decimal("140.00")}],
        analytical_plan=AnalyticalPlan(
            intent="trend",
            metrics=("Revenue",),
            dimensions=("month",),
            time_range="this year",
            output="line chart",
        ),
        resolution_plan={
            "metrics": [{"name": "Revenue", "source": "metric_registry"}],
        },
    )

    assert report["status"] == "pass"
    assert report["metric_binding_source"] == "metric_registry"
    assert report["numeric_columns"] == ["revenue"]


def test_trend_without_period_fails_shape_verification():
    report = verify_result_shape(
        [{"customer": "A", "revenue": 10}, {"customer": "B", "revenue": 12}],
        analytical_plan={"intent": "trend", "metrics": ["Revenue"]},
        resolution_plan={"metrics": [{"name": "Revenue"}]},
    )

    assert report["status"] == "fail"
    assert any("date or period" in error for error in report["errors"])


def test_trend_accepts_plausible_numeric_erp_period_key():
    report = verify_result_shape(
        [
            {"PRD_DMS_KEY": 202601, "Revenue": 10},
            {"PRD_DMS_KEY": 202602, "Revenue": 12},
        ],
        analytical_plan={"intent": "trend", "metrics": ["Revenue"]},
        resolution_plan={"metrics": [{"name": "Revenue"}]},
    )

    assert report["status"] == "pass"
    assert report["time_columns"] == ["PRD_DMS_KEY"]


def test_trend_rejects_invalid_numeric_period_identifier():
    report = verify_result_shape(
        [
            {"PRD_DMS_KEY": 202699, "Revenue": 10},
            {"PRD_DMS_KEY": 202698, "Revenue": 12},
        ],
        analytical_plan={"intent": "trend", "metrics": ["Revenue"]},
        resolution_plan={"metrics": [{"name": "Revenue"}]},
    )

    assert report["status"] == "fail"
    assert report["time_columns"] == []


def test_top_n_over_return_is_visible_warning():
    report = verify_result_shape(
        [{"customer": chr(65 + index), "revenue": 100 - index} for index in range(5)],
        analytical_plan={
            "intent": "ranking",
            "dimensions": ["customer"],
            "top_n": 3,
        },
    )

    assert report["status"] == "warning"
    assert "top 3" in report["warnings"][0]


def test_distribution_percentages_must_reconcile():
    report = verify_result_shape(
        [
            {"state": "A", "revenue": 70, "contribution_pct": 70},
            {"state": "B", "revenue": 20, "contribution_pct": 20},
        ],
        analytical_plan={"intent": "distribution", "dimensions": ["state"]},
    )

    assert report["status"] == "warning"
    assert any("sum to" in warning for warning in report["warnings"])


def test_kpi_rejects_diagnostic_count_in_place_of_approved_metric():
    report = verify_result_shape(
        [{"row_count": 25}],
        analytical_plan={"intent": "metric_query", "output": "kpi"},
        resolution_plan={"metrics": [{"name": "Revenue"}]},
    )

    assert report["status"] == "fail"
    assert any("approved metric" in error.lower() for error in report["errors"])


def test_period_change_rows_are_reconciled_without_exposing_them_to_llm():
    report = verify_result_shape(
        [{
            "ENTITY": "Customer A",
            "CURRENT_PERIOD_COUNT": 8,
            "PRIOR_PERIOD_COUNT": 12,
            "ABSOLUTE_CHANGE": -4,
            "PERCENTAGE_CHANGE": Decimal("-33.33"),
        }],
        analytical_plan={"intent": "entity_lookup", "dimensions": ["entity"]},
        request_plan={
            "analytical_recipe": {
                "kind": "period_over_period_entity_change",
                "direction": "decrease",
            },
        },
    )

    assert report["status"] == "pass"
    assert any("reconcile" in check.lower() for check in report["checks"])


def test_period_change_rejects_wrong_direction_or_arithmetic():
    report = verify_result_shape(
        [{
            "ENTITY": "Customer A",
            "CURRENT_PERIOD_COUNT": 14,
            "PRIOR_PERIOD_COUNT": 12,
            "ABSOLUTE_CHANGE": 99,
        }],
        request_plan={
            "analytical_recipe": {
                "kind": "period_over_period_entity_change",
                "direction": "decrease",
            },
        },
    )

    assert report["status"] == "fail"
    assert any("do not reconcile" in error for error in report["errors"])
    assert any("decrease" in error for error in report["errors"])


def test_failed_shape_caps_answer_confidence_below_medium():
    confidence = build_answer_confidence(
        validation_code="ok",
        row_count=2,
        has_semantic_plan=True,
        result_verification={
            "status": "fail",
            "errors": ["A trend requires a visible date or period column."],
        },
    )

    assert confidence["score"] <= 49
    assert confidence["level"] == "low"
    assert any("trend requires" in warning.lower() for warning in confidence["warnings"])


def test_generic_chart_request_uses_best_fit_cached_result_without_llm():
    cache = ResultCache()
    session_id = "tenant:user:thread"
    result_id = cache.store(
        session_id,
        [{"month": "2026-01", "revenue": 10}, {"month": "2026-02", "revenue": 12}],
        "revenue trend",
        "SELECT month, revenue FROM sales",
    )

    command = parse_result_command("provide this result as a chart")
    assert command is not None
    assert command.action == "presentation"
    assert command.presentation_type == "auto"

    outcome = execute_result_command(
        session_id,
        command,
        cache=cache,
        source_result_id=result_id,
    )
    assert outcome.ok
    assert outcome.snapshot["metadata"]["chart_type_override"] == "auto"


def test_explicit_cached_chart_type_is_preserved_for_portal_renderer():
    cache = ResultCache()
    session_id = "tenant:user:thread"
    result_id = cache.store(
        session_id,
        [{"state": "A", "revenue": 10}, {"state": "B", "revenue": 12}],
        "revenue by state",
        "SELECT state, revenue FROM sales",
    )
    command = parse_result_command("change this result into a pie chart")
    outcome = execute_result_command(
        session_id,
        command,
        cache=cache,
        source_result_id=result_id,
    )

    assert outcome.ok
    assert outcome.snapshot["metadata"]["chart_type_override"] == "pie"
