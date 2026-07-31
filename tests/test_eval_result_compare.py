from decimal import Decimal

from evals.result_compare import compare_result_rows


def test_result_comparison_is_order_independent_by_default():
    case = {
        "expected_rows": [
            {"month": "2026-01", "revenue": 10},
            {"month": "2026-02", "revenue": 20},
        ]
    }
    actual = [
        {"MONTH": "2026-02", "REVENUE": Decimal("20")},
        {"MONTH": "2026-01", "REVENUE": Decimal("10")},
    ]

    result = compare_result_rows(case, actual)

    assert result is not None
    assert result.matched is True
    assert result.status == "matched"


def test_result_comparison_applies_numeric_tolerance():
    case = {
        "expected_result": {
            "rows": [{"margin": 0.333}],
            "numeric_tolerance": 0.001,
        }
    }

    result = compare_result_rows(case, [{"margin": Decimal("0.3339")}])

    assert result is not None
    assert result.matched is True


def test_result_comparison_reports_missing_expected_row():
    case = {"expected_rows": [{"department": "Finance", "headcount": 4}]}

    result = compare_result_rows(case, [{"department": "Sales", "headcount": 4}])

    assert result is not None
    assert result.matched is False
    assert result.status == "value_mismatch"


def test_result_comparison_supports_row_count_only():
    result = compare_result_rows(
        {"min_row_count": 1, "max_row_count": 2},
        [{"value": 1}, {"value": 2}, {"value": 3}],
    )

    assert result is not None
    assert result.matched is False
    assert result.status == "row_count_mismatch"


def test_result_comparison_is_optional_for_legacy_cases():
    assert compare_result_rows({"expected_tables": ["SALES"]}, [{"value": 1}]) is None
