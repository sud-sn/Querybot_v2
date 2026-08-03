import asyncio
import json

import pytest

from core.analysis_code_planner import (
    build_python_plan_input,
    compile_python_plan_response,
    extract_user_python,
    parse_python_analysis_plan,
)


ROWS = [
    {"month": "2026-01", "revenue": 100, "secret_value": "never-send-this"},
    {"month": "2026-02", "revenue": 120, "secret_value": "also-private"},
]


def test_planner_prompt_contains_metadata_but_no_rows_values_or_sql():
    built = build_python_plan_input("Calculate a rolling average", ROWS)
    payload = json.loads(built.user_prompt)
    assert payload["row_count"] == 2
    assert {item["name"] for item in payload["columns"]} == {"month", "revenue", "secret_value"}
    assert "never-send-this" not in built.user_prompt
    assert "also-private" not in built.user_prompt
    assert "SELECT " not in built.user_prompt
    assert "Do not use attributes" in built.system_prompt


def test_user_python_extraction_requires_explicit_source_marker():
    assert extract_user_python("""Run this:\n```python\nresult = [{"x": 1}]\n```""") == 'result = [{"x": 1}]'
    assert extract_user_python("run this python: result = [{\"x\": 2}]") == 'result = [{"x": 2}]'
    assert extract_user_python("use python to calculate a rolling average") == ""


def test_compiler_accepts_safe_plan_and_rejects_unsafe_source():
    good, error = compile_python_plan_response(json.dumps({
        "operation": "python_analysis",
        "title": "Revenue average",
        "code": 'result = [{"average": mean(values(rows, "revenue"))}]',
        "confidence": 0.92,
        "explanation": "Calculate the mean over returned revenue values.",
    }))
    assert error == ""
    assert good and good.confidence == 0.92 and len(good.code_hash) == 64

    bad, error = compile_python_plan_response(json.dumps({
        "operation": "python_analysis",
        "title": "Unsafe",
        "code": "import os\nresult = []",
        "confidence": 1,
        "explanation": "",
    }))
    assert bad is None
    assert "rejected" in error


def test_async_planner_uses_metadata_only_contract():
    captured = {}

    async def complete(**kwargs):
        captured.update(kwargs)
        return json.dumps({
            "operation": "python_analysis",
            "title": "Top revenue",
            "code": 'result = top(rows, 5, "revenue", True)',
            "confidence": 0.9,
            "explanation": "Rank the released rows.",
        }), 20, 30

    plan, error = asyncio.run(
        parse_python_analysis_plan("show top five using Python", ROWS, complete)
    )
    assert error == ""
    assert plan and plan.title == "Top revenue"
    assert "never-send-this" not in captured["user"]
    assert captured["temperature"] == 0.0
