from pathlib import Path

import pytest

from evals.business_user import run_business_suite, summarize


ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "sample_hr": 20,
    "sample_manufacturing": 10,
    "sample_compounding_pharmacy": 10,
    "sample_finance": 12,
    "sample_banking": 13,
}


@pytest.mark.parametrize(("suite_name", "expected_count"), SUITES.items())
def test_business_questions_pass_end_to_end(suite_name, expected_count):
    results = run_business_suite(ROOT / "evals" / suite_name)
    failures = {
        result.id: result.failures
        for result in results
        if not result.passed
    }
    assert len(results) == expected_count
    assert not failures
    assert summarize(results)["pass_rate"] == 100.0


def test_combined_suite_covers_requested_business_metrics():
    results = [
        result
        for suite_name in SUITES
        for result in run_business_suite(ROOT / "evals" / suite_name)
    ]
    categories = {result.category for result in results}
    assert len(results) == 65
    assert {
        "attendance",
        "attrition",
        "budget",
        "compensation",
        "data_quality",
        "deposits",
        "expense",
        "filters",
        "governance",
        "lending",
        "operations",
        "production",
        "profitability",
        "quality",
        "receivables",
        "retention",
        "revenue",
        "risk",
        "trend",
        "workforce",
    } <= categories
