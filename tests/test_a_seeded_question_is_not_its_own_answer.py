"""
A question taken from query history is not its own answer.

"Build golden suite from query history" took the most-asked questions and the
SQL that answered each, and wrote the tables that SQL reads as the tables the
case expected. An eval run replayed that SQL and found those tables in it, so
every case passed whatever its answer was, and the evaluations page reported
"SQL accuracy gate: 100%" -- as did Model Health, and the regression check
after every semantic change.

Now a seeded case says it came from usage and expects nothing. A run replays
its SQL, which must still validate, and leaves it out of the pass rate until
an admin gives it an expected result and the run executes it; the readiness
check counts it only then. A case written by hand is scored as before.

The store, the suite file, the eval run and the evaluations page are real.
The schema the SQL is validated against and the warehouse it runs on are the
only stand-ins.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
import yaml
from starlette.requests import Request

ACCOUNT = "acct-seeded-evals"
SUITE = Path("evals") / "clients" / ACCOUNT / "dbo" / "golden_questions.yaml"
ASKED = (
    ("How many orders did we ship last month?", "SELECT COUNT(*) AS ORDERS FROM dbo.ORDERS", 3),
    ("Orders by warehouse", "SELECT WAREHOUSE, COUNT(*) AS ORDERS FROM dbo.ORDERS GROUP BY WAREHOUSE", 2),
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A store of the test's own, with questions the workspace answered."""
    import store

    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    monkeypatch.chdir(tmp_path)
    store.init_db()
    store.upsert_client(ACCOUNT, "portal")
    db_id = store.save_db_config("azure_sql", "Warehouse", {
        "server": "dw.example.net", "database": "DW", "user": "reader", "password": "a-password"})
    store.update_client_meta(ACCOUNT, db_config_id=db_id)
    with store.get_db() as conn:
        for question, sql, times in ASKED:
            for _ in range(times):
                conn.execute(
                    """INSERT INTO answer_trace (account_id, question_text_sanitized, route, selected_schema,
                           generated_sql, query_row_count, status) VALUES (?, ?, 'normal_sql', 'dbo', ?, 3, 'success')""",
                    (ACCOUNT, question, sql))
    monkeypatch.setattr("evals.run.load_known_tables", lambda schema_dir: {"DBO.ORDERS"})
    monkeypatch.setattr("evals.run.load_schema_columns", lambda schema_dir: {})
    return store


def _seed() -> list[dict]:
    from evals.seed import seed_golden_suite

    seed_golden_suite(ACCOUNT)
    return yaml.safe_load(SUITE.read_text(encoding="utf-8"))["cases"]


def _write(cases: list[dict]) -> None:
    SUITE.parent.mkdir(parents=True, exist_ok=True)
    SUITE.write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")


def _run(store, *, execute: bool = False, warehouse_rows: int = 3) -> dict:
    """An eval run of the suite, as the evaluations page starts one."""
    from evals.run import run_eval_suite

    rows = [{"ORDERS": index} for index in range(warehouse_rows)]
    with patch("evals.run.run_query", return_value=rows):
        _, run_id = asyncio.run(run_eval_suite(
            account_id=ACCOUNT, schema="dbo", cases_path=SUITE, execute=execute, out_dir=Path("report")))
    return store.get_eval_run(run_id)


def _evaluations_page() -> str:
    from admin import routes

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request({"type": "http", "method": "GET", "path": f"/admin/clients/{ACCOUNT}/evals", "root_path": "",
                       "scheme": "http", "query_string": urlencode({}).encode(), "server": ("testserver", 80),
                       "client": ("127.0.0.1", 1), "headers": []}, receive)
    with patch.object(routes, "_is_auth", return_value=True):
        return asyncio.run(routes.client_evals(request, ACCOUNT)).body.decode()


class TestASeededCase:

    def test_expects_nothing_of_the_sql_it_replays(self, workspace):
        cases = _seed()
        assert [case["question"] for case in cases] == [question for question, _, _ in ASKED]
        for case in cases:
            assert case["origin"] == "usage"
            assert "expected_tables" not in case

    def test_is_replayed_and_not_scored(self, workspace):
        _seed()
        run = _run(workspace)
        assert (run["total_cases"], run["passed_cases"], run["status"]) == (0, 0, "awaiting_answers")
        assert [case["validation_status"] for case in run["cases"]] == ["passed", "passed"]
        assert all(case["failures"][-1].startswith("not scored: from query history") for case in run["cases"])
        page = _evaluations_page()
        assert "SQL accuracy gate" not in page and "SQL accuracy baseline required" in page

    def test_nor_one_seeded_before_this_release(self, workspace):
        """Its expected tables are the ones its own SQL reads."""
        question, sql, _ = ASKED[0]
        _write([{"id": "auto_0123456789", "question": question, "generated_sql": sql,
                 "expected_tables": ["DBO.ORDERS"], "min_score": 0.85}])
        run = _run(workspace)
        assert (run["total_cases"], run["status"]) == (0, "awaiting_answers")

    def test_a_suite_of_them_is_no_drop_from_the_run_before(self, workspace):
        """The last run of the same file passed them all."""
        _seed()
        workspace.save_eval_run(account_id=ACCOUNT, schema_name="dbo", case_file=str(SUITE), total_cases=2,
                                passed_cases=2, status="passed")
        assert not _run(workspace)["regressed"]
        assert workspace.latest_regressed_run(ACCOUNT) is None


class TestAnExpectedAnswer:

    def _answered(self) -> None:
        cases = _seed()
        cases[0]["expected_row_count"] = 3
        _write(cases)

    def test_scores_the_case_in_a_run_that_executes_it(self, workspace):
        self._answered()
        run = _run(workspace, execute=True)
        assert (run["total_cases"], run["passed_cases"], run["status"]) == (1, 1, "passed")

    def test_that_does_not_hold_fails_it(self, workspace):
        self._answered()
        run = _run(workspace, execute=True, warehouse_rows=2)
        assert (run["total_cases"], run["passed_cases"], run["status"]) == (1, 0, "failed")

    def test_is_not_checked_by_a_run_that_does_not_execute(self, workspace):
        self._answered()
        assert _run(workspace)["total_cases"] == 0

    def test_is_what_the_readiness_check_counts(self, workspace):
        from evals.readiness import evaluate_baseline_readiness

        def suite_check() -> dict:
            report = evaluate_baseline_readiness(ACCOUNT, "dbo", SUITE, minimum_cases=1)
            return next(check for check in report["checks"] if check["name"] == "golden_suite")

        _seed()
        check = suite_check()
        assert (check["passed"], check["detail"]) == (
            False, "0 cases with an expected answer (minimum 1); 2 still need one")
        self._answered()
        assert suite_check()["passed"] is True


def test_a_case_written_by_hand_is_scored_as_before(workspace):
    question, sql, _ = ASKED[0]
    _write([{"id": "orders-last-month", "question": question, "generated_sql": sql,
             "expected_tables": ["DBO.ORDERS"]}])
    run = _run(workspace)
    assert (run["total_cases"], run["passed_cases"], run["status"]) == (1, 1, "passed")
