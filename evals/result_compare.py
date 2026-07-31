"""Result-equivalence checks for golden SQL evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class ResultComparison:
    matched: bool
    status: str
    detail: str = ""


def _row_map(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().casefold(): value for key, value in row.items()}


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None


def _values_equal(actual: Any, expected: Any, tolerance: Decimal) -> bool:
    actual_num = _decimal(actual)
    expected_num = _decimal(expected)
    if actual_num is not None and expected_num is not None:
        return abs(actual_num - expected_num) <= tolerance
    if isinstance(actual, (date, datetime)):
        actual = actual.isoformat()
    if isinstance(expected, (date, datetime)):
        expected = expected.isoformat()
    if actual is None or expected is None:
        return actual is expected
    return str(actual).strip() == str(expected).strip()


def _rows_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    columns: list[str],
    tolerance: Decimal,
) -> bool:
    actual_ci = _row_map(actual)
    expected_ci = _row_map(expected)
    selected = [str(c).strip().casefold() for c in columns] if columns else list(expected_ci)
    return all(
        key in actual_ci
        and key in expected_ci
        and _values_equal(actual_ci[key], expected_ci[key], tolerance)
        for key in selected
    )


def compare_result_rows(case: dict, actual_rows: list[dict[str, Any]]) -> ResultComparison | None:
    """Compare rows when a case declares result expectations.

    Supported case fields::

        expected_rows: [{...}]
        expected_result:
          rows: [{...}]
          order_matters: false
          numeric_tolerance: 0.01
          columns: [month, revenue]
        expected_row_count: 12
        min_row_count: 1
        max_row_count: 12
    """
    spec = case.get("expected_result")
    if isinstance(spec, list):
        spec = {"rows": spec}
    elif not isinstance(spec, dict):
        spec = {}

    expected_rows = case.get("expected_rows", spec.get("rows"))
    has_expectation = expected_rows is not None or any(
        key in case for key in ("expected_row_count", "min_row_count", "max_row_count")
    )
    if not has_expectation:
        return None

    actual_rows = actual_rows if isinstance(actual_rows, list) else []
    expected_count = case.get("expected_row_count")
    min_count = case.get("min_row_count")
    max_count = case.get("max_row_count")
    if expected_count is not None and len(actual_rows) != int(expected_count):
        return ResultComparison(False, "row_count_mismatch", f"expected {int(expected_count)} rows, got {len(actual_rows)}")
    if min_count is not None and len(actual_rows) < int(min_count):
        return ResultComparison(False, "row_count_mismatch", f"expected at least {int(min_count)} rows, got {len(actual_rows)}")
    if max_count is not None and len(actual_rows) > int(max_count):
        return ResultComparison(False, "row_count_mismatch", f"expected at most {int(max_count)} rows, got {len(actual_rows)}")

    if expected_rows is None:
        return ResultComparison(True, "matched")
    if not isinstance(expected_rows, list) or not all(isinstance(row, dict) for row in expected_rows):
        return ResultComparison(False, "invalid_expectation", "expected rows must be a list of objects")
    if len(actual_rows) != len(expected_rows):
        return ResultComparison(False, "row_count_mismatch", f"expected {len(expected_rows)} rows, got {len(actual_rows)}")

    try:
        tolerance = Decimal(str(spec.get("numeric_tolerance", case.get("numeric_tolerance", 0))))
    except InvalidOperation:
        return ResultComparison(False, "invalid_expectation", "numeric_tolerance must be numeric")
    columns = spec.get("columns") or case.get("result_columns") or []
    order_matters = bool(spec.get("order_matters", case.get("result_order_matters", False)))

    if order_matters:
        for index, (actual, expected) in enumerate(zip(actual_rows, expected_rows), start=1):
            if not _rows_equal(actual, expected, columns, tolerance):
                return ResultComparison(False, "value_mismatch", f"row {index} did not match expected values")
        return ResultComparison(True, "matched")

    unmatched = list(actual_rows)
    for index, expected in enumerate(expected_rows, start=1):
        match_index = next(
            (i for i, actual in enumerate(unmatched) if _rows_equal(actual, expected, columns, tolerance)),
            None,
        )
        if match_index is None:
            return ResultComparison(False, "value_mismatch", f"expected row {index} was not present in the result")
        unmatched.pop(match_index)
    return ResultComparison(True, "matched")
