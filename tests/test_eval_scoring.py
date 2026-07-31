import asyncio
import json
from unittest.mock import AsyncMock, patch

from evals.run import _score_sql, run_eval_suite


def _score(case, sql, execute_result=None):
    with patch("evals.run.validate_sql", return_value=(True, "", "")):
        return _score_sql(
            case,
            sql,
            known_tables={"PHARMA_LAB.F_RX_FILL"},
            table_columns={},
            db_type="azure_sql",
            execute_result=execute_result,
        )


def test_bracketed_azure_table_matches_canonical_expected_table():
    result = _score(
        {
            "id": "azure_table",
            "question": "fills",
            "expected_tables": ["PHARMA_LAB.F_RX_FILL"],
            "min_score": 0.85,
        },
        "SELECT COUNT(*) FROM [PHARMA_LAB].[F_RX_FILL]",
        ("passed", 1, ""),
    )

    assert result.passed is True
    assert result.score == 100.0
    assert result.failures == []


def test_offline_score_uses_only_available_checks():
    result = _score(
        {
            "id": "offline",
            "question": "fills",
            "expected_tables": ["PHARMA_LAB.F_RX_FILL"],
            "min_score": 0.85,
        },
        "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_FILL",
    )

    assert result.execution_status == "skipped"
    assert result.passed is True
    assert result.score == 100.0


def test_requested_execution_failure_reduces_score_and_fails_gate():
    result = _score(
        {
            "id": "execution_failure",
            "question": "fills",
            "expected_tables": ["PHARMA_LAB.F_RX_FILL"],
            "min_score": 0.85,
        },
        "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_FILL",
        ("failed", 0, "database unavailable"),
    )

    assert result.passed is False
    assert result.score == 75.0
    assert result.execution_status == "failed"
    assert "execution failed: database unavailable" in result.failures


def test_generate_mode_replaces_historical_sql(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [{
        "id": "fresh",
        "question": "count fills",
        "generated_sql": "SELECT 1 AS historical_sql",
        "expected_tables": ["PHARMA_LAB.F_RX_FILL"],
    }]}), encoding="utf-8")
    fresh_sql = "SELECT COUNT(*) FROM [PHARMA_LAB].[F_RX_FILL]"

    with (
        patch("evals.run.store.init_db"),
        patch("evals.run.store.get_client", return_value={"state_data": "{}"}),
        patch("evals.run._client_db_config", return_value={"db_type": "azure_sql"}),
        patch("evals.run.load_known_tables", return_value={"PHARMA_LAB.F_RX_FILL"}),
        patch("evals.run.load_schema_columns", return_value={}),
        patch("evals.run._generate_sql", AsyncMock(return_value=fresh_sql)) as generate_sql,
        patch("evals.run.validate_sql", return_value=(True, "", "")),
        patch("evals.run._write_reports"),
        patch("evals.run.store.previous_eval_run", return_value=None),
        patch("evals.run.store.save_eval_run", return_value=7),
        patch("evals.run.store.save_eval_case_result"),
        patch("core.semantic_contract.contract_fingerprint", return_value="contract"),
    ):
        results, run_id = asyncio.run(
            run_eval_suite(
                account_id="Demo_2",
                schema="PHARMA_LAB",
                cases_path=cases_path,
                generate=True,
                execute=False,
                out_dir=tmp_path / "report",
            )
        )

    generate_sql.assert_awaited_once()
    assert run_id == 7
    assert results[0].generated_sql == fresh_sql
    assert "historical_sql" not in results[0].generated_sql
