"""Offline business-user evaluation against deterministic sample data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlglot import transpile

from core.validator import validate_sql_detailed


@dataclass
class BusinessCaseResult:
    suite: str
    id: str
    category: str
    question: str
    passed: bool
    validation_status: str
    execution_status: str
    answer_status: str
    row_count: int
    failures: list[str]


def _load_suite(suite_dir: Path) -> tuple[dict, list[dict]]:
    schema = json.loads((suite_dir / "schema.json").read_text(encoding="utf-8"))
    payload = yaml.safe_load(
        (suite_dir / "business_questions.yaml").read_text(encoding="utf-8")
    ) or {}
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("business_questions.yaml must contain a cases list")
    return schema, cases


def _table_metadata(schema: dict) -> tuple[set[str], dict[str, dict[str, str]]]:
    known_tables: set[str] = set()
    table_columns: dict[str, dict[str, str]] = {}
    for fqn, table in schema.items():
        upper_fqn = str(fqn).upper()
        parts = upper_fqn.split(".")
        variants = {upper_fqn, parts[-1]}
        if len(parts) >= 2:
            variants.add(".".join(parts[-2:]))
        columns = {
            str(column["name"]).upper(): str(column.get("type") or "")
            for column in table.get("columns") or []
        }
        for variant in variants:
            known_tables.add(variant)
            table_columns.setdefault(variant, {}).update(columns)
    return known_tables, table_columns


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def _normalize_rows(rows: list[dict], *, ordered: bool) -> list[dict]:
    normalized = [
        {str(key).upper(): _normalize_value(value) for key, value in row.items()}
        for row in rows
    ]
    if not ordered:
        normalized.sort(key=lambda row: json.dumps(row, sort_keys=True))
    return normalized


def _execute(connection: sqlite3.Connection, sql: str) -> list[dict]:
    sqlite_sql = transpile(sql, read="tsql", write="sqlite")[0]
    cursor = connection.execute(sqlite_sql)
    return [dict(row) for row in cursor.fetchall()]


def run_business_suite(suite_dir: Path) -> list[BusinessCaseResult]:
    """Validate and execute every case in a local business-question suite."""
    suite_dir = Path(suite_dir)
    schema, cases = _load_suite(suite_dir)
    known_tables, table_columns = _table_metadata(schema)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "YEAR",
        1,
        lambda value: int(str(value)[:4]) if value else None,
    )
    try:
        connection.executescript(
            (suite_dir / "setup.sql").read_text(encoding="utf-8")
        )
        results: list[BusinessCaseResult] = []
        for case in cases:
            sql = str(case.get("generated_sql") or "").strip()
            expected_validation = str(
                case.get("expected_validation") or "passed"
            ).lower()
            allowed_tables = case.get("allowed_tables")
            allowed = (
                {str(table).upper() for table in allowed_tables}
                if allowed_tables is not None
                else None
            )
            verdict = validate_sql_detailed(
                sql,
                known_tables,
                "azure_sql",
                allowed,
                table_columns,
            )
            actual_validation = "passed" if verdict.ok else verdict.code
            failures: list[str] = []
            execution_status = "skipped"
            answer_status = "skipped"
            row_count = 0

            if actual_validation != expected_validation:
                failures.append(
                    "validation mismatch: "
                    f"expected {expected_validation}, got {actual_validation}"
                )

            if expected_validation == "passed" and verdict.ok:
                try:
                    actual_rows = _execute(connection, sql)
                    execution_status = "passed"
                    row_count = len(actual_rows)
                    ordered = bool(case.get("order_matters", False))
                    actual = _normalize_rows(actual_rows, ordered=ordered)
                    expected = _normalize_rows(
                        case.get("expected_rows") or [],
                        ordered=ordered,
                    )
                    if actual == expected:
                        answer_status = "passed"
                    else:
                        answer_status = "failed"
                        failures.append(
                            "answer mismatch: "
                            f"expected {expected!r}, got {actual!r}"
                        )
                except Exception as exc:
                    execution_status = "failed"
                    answer_status = "failed"
                    failures.append(f"execution failed: {exc}")

            results.append(
                BusinessCaseResult(
                    suite=suite_dir.name,
                    id=str(case.get("id") or ""),
                    category=str(case.get("category") or "general"),
                    question=str(case.get("question") or ""),
                    passed=not failures,
                    validation_status=actual_validation,
                    execution_status=execution_status,
                    answer_status=answer_status,
                    row_count=row_count,
                    failures=failures,
                )
            )
        return results
    finally:
        connection.close()


def summarize(results: list[BusinessCaseResult]) -> dict:
    total = len(results)
    passed = sum(result.passed for result in results)
    categories: dict[str, dict[str, int]] = {}
    suites: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = categories.setdefault(result.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(result.passed)
        suite_bucket = suites.setdefault(
            result.suite,
            {"passed": 0, "total": 0},
        )
        suite_bucket["total"] += 1
        suite_bucket["passed"] += int(result.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round((passed / total * 100) if total else 0.0, 2),
        "suites": suites,
        "categories": categories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite directory. Repeat to run multiple suites.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every evals/sample_* business suite.",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON result path")
    args = parser.parse_args()

    if args.all:
        suite_dirs = sorted(
            path
            for path in Path("evals").glob("sample_*")
            if (path / "business_questions.yaml").exists()
        )
    else:
        suite_dirs = [
            Path(path)
            for path in (args.suite or ["evals/sample_hr"])
        ]

    results: list[BusinessCaseResult] = []
    for suite_dir in suite_dirs:
        suite_results = run_business_suite(suite_dir)
        results.extend(suite_results)
        print(f"\n[{suite_dir.name}]")
        for result in suite_results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status:4} {result.id:24} {result.question}")
            for failure in result.failures:
                print(f"     {failure}")

    summary = summarize(results)
    print(
        f"\n{summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']}%)"
    )

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
