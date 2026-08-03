import json

from core.dashboard_planner import (
    build_dashboard_plan_input,
    compile_dashboard_plan_response,
    looks_like_multi_widget_request,
)


def test_dashboard_plan_compiles_bounded_business_questions():
    raw = json.dumps({
        "operation": "create_dashboard",
        "name": "Executive sales",
        "widgets": [
            {"question": "monthly revenue this year", "title": "Revenue trend", "chart_type": "line", "tab": "Overview"},
            {"question": "top 10 products by revenue this year", "title": "Top products", "chart_type": "bar", "tab": "Products"},
        ],
        "tabs": ["Overview", "Products"],
        "refresh_schedule": "daily",
        "visibility": "team",
        "confidence": 0.94,
    })
    plan, error = compile_dashboard_plan_response(raw)
    assert error == ""
    assert plan and len(plan.widgets) == 2
    assert plan.widgets[0].chart_type == "line"
    assert plan.refresh_schedule == "daily"
    assert plan.visibility == "team"


def test_dashboard_plan_rejects_sql_and_unbounded_work():
    sql_plan = json.dumps({
        "operation": "create_dashboard",
        "widgets": [{"question": "SELECT * FROM sales"}],
    })
    plan, error = compile_dashboard_plan_response(sql_plan)
    assert plan is None and "not SQL" in error

    too_many = json.dumps({
        "operation": "create_dashboard",
        "widgets": [{"question": f"metric {index}"} for index in range(7)],
    })
    plan, error = compile_dashboard_plan_response(too_many)
    assert plan is None and "six" in error


def test_dashboard_planner_prompt_is_metadata_only():
    built = build_dashboard_plan_input(
        "revenue dashboard",
        [{"name": "Net revenue", "description": "Approved finance metric", "formula": "secret"}],
    )
    assert "Net revenue" in built.system_prompt
    assert "secret" not in built.system_prompt
    assert "never write sql" in built.system_prompt.lower()


def test_multi_widget_detection_requires_explicit_structure():
    assert looks_like_multi_widget_request("revenue trend; top products; order KPI")
    assert looks_like_multi_widget_request("revenue plus order count")
    assert not looks_like_multi_widget_request("revenue and profit by month")
