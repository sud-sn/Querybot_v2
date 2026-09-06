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


def test_single_value_kpi_and_currency_strings_remain_numeric_for_profile():
    plain = _analyze([{"Total Orders": 360}], "profile")
    formatted = _analyze([{"Total Orders": "₹360"}], "profile")
    accounting = _analyze([{"Revenue": "(INR 1,250.50)"}], "profile")

    assert plain["rows"][0]["mean"] == 360.0
    assert formatted["rows"][0]["mean"] == 360.0
    assert accounting["rows"][0]["mean"] == -1250.5


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


# ── The exec namespace ────────────────────────────────────────────────────────
# The worker used to run the analysis as exec(code, {"__builtins__": {}}, env),
# which put the 32 helpers in LOCALS. A comprehension compiles to its own code
# object, and a code object resolves a free name against GLOBALS -- so every
# helper was invisible inside one and the analysis died with "NameError: name
# 'round' is not defined" AFTER the validator had approved it. GROUP_CODE above
# ends in a list comprehension, which is why the test for it had been failing
# and was being carried as an "environment" failure.
#
# Only the comprehension forms the validator permits are covered: GeneratorExp
# and Lambda are refused outright (they raise UnsafeAnalysisCode), so a test
# for those would prove nothing about this fix.

_SCOPE_SHAPES = [
    ("list comprehension",
     'result = [{"m": get(row, "month"), "v": round(number(get(row, "revenue")), 2)}\n'
     '          for row in rows]'),
    ("nested list comprehension",
     'peak = max([abs(number(get(row, "revenue"))) for row in rows])\n'
     'result = [{"peak": round(peak, 2)}]'),
    ("set comprehension",
     'months = {str(get(row, "month")) for row in rows}\n'
     'result = [{"months": len(months)}]'),
]


@pytest.mark.parametrize("shape,code", _SCOPE_SHAPES, ids=[s for s, _ in _SCOPE_SHAPES])
def test_helpers_are_visible_inside_every_scope_that_makes_its_own_frame(shape, code):
    validate_python_analysis(code)          # the validator lets it through ...
    result = run_governed_python_analysis(ROWS, code, timeout_seconds=5)
    assert result.rows, shape               # ... so this has to actually run


@pytest.mark.parametrize("code", [
    'result = [{"t": sum(number(get(row, "revenue")) for row in rows)}]',
    'f = lambda r: number(get(r, "revenue"))\nresult = [{"v": f(row)} for row in rows]',
])
def test_the_scopes_this_fix_does_not_reach_are_refused_not_broken(code):
    # Recorded so the coverage above is not mistaken for the whole story:
    # these two never reach the namespace at all.
    with pytest.raises(UnsafeAnalysisCode):
        validate_python_analysis(code)


class _Pipe:
    """The worker's end of the process boundary."""

    def __init__(self):
        self.payload = None

    def send(self, value):
        self.payload = value

    def close(self):
        pass


def _run_worker_past_the_validator(code):
    """Execute the real worker with the validator opened.

    The validator is the gate, so the only way to reach the namespace behind
    it is to open the gate. That is a boundary mock -- the worker still
    compiles and executes the code itself, which is the thing under test.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import core.analysis_sandbox as sandbox

    stub = SimpleNamespace(code_hash="0" * 64, ast_nodes=0, helper_calls=())
    pipe = _Pipe()
    # _apply_worker_limits is the OTHER boundary here, and it is not optional
    # to mock: production calls this entry point in a forked child, so its
    # setrlimit(RLIMIT_CPU, 4) / RLIMIT_AS 512MB / RLIMIT_NOFILE 16 land on
    # that child. Called in-process they land on pytest, which is then killed
    # a few thousand tests later — a failure that looks nothing like its cause.
    with patch.object(sandbox, "validate_python_analysis", return_value=stub), \
            patch.object(sandbox, "_apply_worker_limits", lambda: None):
        sandbox._python_worker_entry(pipe, ROWS, code)
    return pipe.payload


def test_the_namespace_is_still_sealed_against_builtins():
    """Defence in depth, asserted where it lives.

    Every other escape test in this module goes through
    validate_python_analysis, so all of them pass against a worker that
    executes with the real builtins module in scope -- and moving the helpers
    into globals is exactly what makes exec inject builtins when the key is
    absent. This asserts the pin, not the gate.
    """
    payload = _run_worker_past_the_validator(
        "result = [{'leaked': __import__('os').getcwd()}]")
    assert payload["ok"] is False
    assert "__import__" in payload["error"]


def test_the_seal_test_fails_for_the_right_reason():
    # The same harness on code that uses only the helper table must succeed,
    # or the assertion above would pass on a worker that is simply broken.
    payload = _run_worker_past_the_validator(
        'result = [{"n": round(number(get(row, "revenue")), 1)} for row in rows]')
    assert payload["ok"] is True
    assert len(payload["rows"]) == len(ROWS)
