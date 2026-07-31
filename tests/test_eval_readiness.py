from pathlib import Path
from unittest.mock import patch

from evals.readiness import evaluate_baseline_readiness


def test_readiness_fails_closed_without_client_or_cases(tmp_path: Path):
    with patch("evals.readiness.store.init_db"), patch(
        "evals.readiness.store.get_client", return_value=None
    ), patch(
        "evals.readiness.resolve_provider", side_effect=RuntimeError("no query model configured")
    ), patch("evals.readiness._qdrant_reachable", return_value=(False, "offline")), patch(
        "evals.readiness.importlib.util.find_spec", return_value=None
    ):
        report = evaluate_baseline_readiness(
            "missing", "FINANCE", tmp_path / "missing.yaml"
        )

    assert report["ready"] is False
    failed = {item["name"] for item in report["checks"] if not item["passed"]}
    assert {"client", "database", "query_model", "golden_suite", "vector_store"} <= failed


def test_readiness_counts_result_and_safety_assertions(tmp_path: Path):
    cases = tmp_path / "golden.yaml"
    cases.write_text(
        """cases:
  - id: one
    question: total?
    expected_rows: [{total: 1}]
  - id: two
    question: delete?
    expected_validation: ddl
""",
        encoding="utf-8",
    )
    client = {"state": "READY", "db_config_id": 1, "state_data": "{}"}
    with patch("evals.readiness.store.init_db"), patch(
        "evals.readiness.store.get_client", return_value=client
    ), patch("evals.readiness.store.get_db_config", return_value={"db_type": "azure_sql"}), patch(
        "evals.readiness.resolve_provider", return_value=("openai", "model", "key", {})
    ), patch("evals.readiness._qdrant_reachable", return_value=(True, "online")), patch(
        "evals.readiness.importlib.util.find_spec", return_value=object()
    ):
        report = evaluate_baseline_readiness(
            "acct", "FINANCE", cases, minimum_cases=2
        )

    assertion_check = next(item for item in report["checks"] if item["name"] == "result_assertions")
    assert assertion_check["passed"] is True
    assert assertion_check["detail"] == "2/2 cases have result or safety assertions"
