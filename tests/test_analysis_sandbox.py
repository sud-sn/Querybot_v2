import pytest

from core.analysis_sandbox import (
    UnsafeAnalysisCode,
    _analyze,
    plan_analysis_operations,
    run_governed_python_analysis,
    run_isolated_analysis,
    validate_python_analysis,
)


ROWS = [
    {"month": f"2026-{index:02d}", "revenue": value, "orders": value * 2}
    for index, value in enumerate([10, 11, 9, 10, 10, 11, 9, 10, 10, 100], 1)
]


def test_planner_uses_only_allowlisted_operations():
    assert plan_analysis_operations(
        "Analyze this result deeply for correlations, outliers, and trends", ROWS
    ) == ["profile", "correlation", "outliers"]
    assert plan_analysis_operations("Find trends in this data", ROWS) == ["trend"]


def test_profile_correlation_outlier_and_trend_math():
    profile = _analyze(ROWS, "profile")
    correlation = _analyze(ROWS, "correlation")
    outliers = _analyze(ROWS, "outliers")
    trend = _analyze(ROWS, "trend")

    assert profile["metadata"]["input_rows"] == 10
    assert {row["column"] for row in profile["rows"]} == {"revenue", "orders"}
    assert correlation["rows"][0]["correlation"] == 1.0
    assert outliers["rows"][0]["row"] == 10
    assert trend["rows"][-1]["change"] == 90.0


def test_isolated_worker_returns_json_shaped_result():
    result = run_isolated_analysis(ROWS, "correlation", timeout_seconds=5)
    assert result.operation == "correlation"
    assert result.metadata["input_rows"] == 10
    assert result.rows[0]["strength"] == "strong"


def test_arbitrary_code_or_operation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        run_isolated_analysis(ROWS, "__import__('os').system('whoami')")


GROUP_CODE = """
totals = {}
for row in rows:
    region = str(get(row, "region", "Unknown"))
    amount = number(get(row, "revenue", 0))
    if amount is not None:
        if region not in totals:
            totals[region] = 0
        totals[region] += amount
result = [{"region": region, "revenue": round(amount, 2)} for region, amount in items(totals)]
"""


def test_governed_python_supports_custom_grouping_and_derived_rows():
    rows = [
        {"region": "North", "revenue": 10},
        {"region": "South", "revenue": 7.5},
        {"region": "North", "revenue": 2.5},
    ]
    validation = validate_python_analysis(GROUP_CODE)
    result = run_governed_python_analysis(rows, GROUP_CODE, timeout_seconds=5)

    assert len(validation.code_hash) == 64
    assert {"get", "items", "number", "round", "str"} <= set(validation.helper_calls)
    assert result.rows == [
        {"region": "North", "revenue": 12.5},
        {"region": "South", "revenue": 7.5},
    ]
    assert result.metadata["database_queried"] is False
    assert result.metadata["rows_sent_to_llm"] == 0


def test_governed_python_supports_rolling_calculations():
    code = """
window = []
result = []
for row in rows:
    amount = number(get(row, "revenue"))
    if amount is not None:
        window = window + [amount]
        if len(window) > 3:
            window = window[1:]
        result = result + [{"month": get(row, "month"), "rolling_avg": round(mean(window), 2)}]
"""
    result = run_governed_python_analysis(ROWS[:4], code, timeout_seconds=5)
    assert [row["rolling_avg"] for row in result.rows] == [10.0, 10.5, 10.0, 10.0]


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = []",
        "result = open('secret.txt').read()",
        "result = rows[0].get('revenue')",
        "result = ().__class__.__mro__",
        "while True:\n    pass\nresult = []",
        "def calculate():\n    return []\nresult = calculate()",
        "result = eval('1 + 1')",
        "rows = []\nresult = rows",
        "result = [x for row in rows for x in range(10000)]",
        "result = rows * 1001",
    ],
)
def test_governed_python_rejects_escape_and_resource_surfaces(code):
    with pytest.raises(UnsafeAnalysisCode):
        validate_python_analysis(code)


def test_governed_python_requires_explicit_result_and_bounded_output():
    with pytest.raises(UnsafeAnalysisCode, match="result"):
        validate_python_analysis("total = sum([1, 2, 3])")
    with pytest.raises(ValueError, match="2000"):
        run_governed_python_analysis(
            ROWS,
            'result = [{"value": i} for i in range(2001)]',
            timeout_seconds=5,
        )
