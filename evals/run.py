"""Run golden-question SQL accuracy regressions.

Example:
    python -m evals.run --client demo_client --schema HR --cases evals/clients/demo_client/HR/golden_questions.yaml

Cases may include either ``generated_sql`` for offline scoring or use
``--generate`` to ask the configured LLM to produce SQL from the client's KB.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import store
from core.examples import format_examples_for_prompt, retrieve_similar_examples
from core.knowledge import load_retriever
from core.llm import build_sql_system_prompt, llm_complete, resolve_provider
from core.schema import load_known_tables, run_query
from core.validator import validate_sql


@dataclass
class EvalCaseResult:
    id: str
    question: str
    score: float
    passed: bool
    generated_sql: str = ""
    validation_status: str = ""
    validation_error: str = ""
    execution_status: str = "skipped"
    row_count: int = 0
    failures: list[str] | None = None


def _load_cases(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML eval files require PyYAML. Install pyyaml or use JSON.") from exc
        data = yaml.safe_load(text) or []
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("cases", [])
    if not isinstance(data, list):
        raise ValueError("Eval case file must contain a list or {cases: [...]}")
    return data


def _contains_all(sql_upper: str, expected: list[str], label: str, failures: list[str]) -> int:
    matched = 0
    for item in expected or []:
        if str(item).upper() in sql_upper:
            matched += 1
        else:
            failures.append(f"missing {label}: {item}")
    return matched


def _contains_none(sql_upper: str, forbidden: list[str], label: str, failures: list[str]) -> int:
    clean = 0
    for item in forbidden or []:
        if str(item).upper() in sql_upper:
            failures.append(f"forbidden {label}: {item}")
        else:
            clean += 1
    return clean


def _score_sql(case: dict, sql: str, known_tables: set[str], db_type: str, execute_result: tuple[str, int, str] | None) -> EvalCaseResult:
    failures: list[str] = []
    sql = (sql or "").strip()
    sql_upper = sql.upper()
    if not sql:
        return EvalCaseResult(
            id=str(case.get("id", "")),
            question=str(case.get("question", "")),
            score=0.0,
            passed=False,
            generated_sql=sql,
            validation_status="missing_sql",
            failures=["no SQL generated"],
        )

    allowed_tables = set(case.get("allowed_tables") or known_tables)
    ok, reason, code = validate_sql(sql, known_tables, db_type, allowed_tables)
    validation_points = 25 if ok else 0
    if not ok:
        failures.append(f"validator failed: {reason}")

    expected_tables = case.get("expected_tables") or []
    expected_columns = case.get("expected_columns") or []
    expected_contains = case.get("expected_sql_contains") or []
    forbidden_contains = case.get("forbidden_sql_contains") or []
    forbidden_columns = case.get("forbidden_columns") or []

    total_expectations = (
        len(expected_tables) + len(expected_columns) + len(expected_contains)
        + len(forbidden_contains) + len(forbidden_columns)
    )
    expectation_score = 40
    if total_expectations:
        matched = 0
        matched += _contains_all(sql_upper, expected_tables, "table", failures)
        matched += _contains_all(sql_upper, expected_columns, "column", failures)
        matched += _contains_all(sql_upper, expected_contains, "SQL pattern", failures)
        matched += _contains_none(sql_upper, forbidden_contains, "SQL pattern", failures)
        matched += _contains_none(sql_upper, forbidden_columns, "column", failures)
        expectation_score = 40 * (matched / total_expectations)

    execution_score = 0
    execution_status = "skipped"
    row_count = 0
    if execute_result:
        execution_status, row_count, exec_err = execute_result
        execution_score = 25 if execution_status == "passed" else 0
        if exec_err:
            failures.append(f"execution failed: {exec_err}")

    privacy_score = 10
    for sensitive in case.get("forbidden_sensitive_terms") or []:
        if str(sensitive).upper() in sql_upper:
            privacy_score = 0
            failures.append(f"sensitive term surfaced: {sensitive}")

    score = round(validation_points + expectation_score + execution_score + privacy_score, 2)
    min_score = float(case.get("min_score", 0.85)) * 100
    passed = score >= min_score and not any(f.startswith("validator failed") for f in failures)
    return EvalCaseResult(
        id=str(case.get("id", "")),
        question=str(case.get("question", "")),
        score=score,
        passed=passed,
        generated_sql=sql,
        validation_status="passed" if ok else code,
        validation_error="" if ok else reason,
        execution_status=execution_status,
        row_count=row_count,
        failures=failures,
    )


async def _generate_sql(account_id: str, question: str, db_type: str, allowed_tables: set[str] | None) -> str:
    client = store.get_client(account_id) or {}
    provider, model, api_key, az_kwargs = resolve_provider(client, purpose="query")
    retriever = load_retriever(account_id)
    docs = retriever.retrieve(question, n=8, allowed_tables=allowed_tables)
    context = "\n\n---\n\n".join(docs)
    examples = retrieve_similar_examples(question, account_id, n=3, allowed_tables=allowed_tables)
    if examples:
        context = format_examples_for_prompt(examples) + "\n\n---\n\n" + context
    system = build_sql_system_prompt(db_type, context)
    sql, _, _ = await llm_complete(system, question, provider, model, api_key, max_tokens=512, **az_kwargs)
    if sql and sql.startswith("```"):
        sql = "\n".join(sql.split("\n")[1:]).rsplit("```", 1)[0].strip()
    return (sql or "").strip()


def _write_reports(results: list[EvalCaseResult], out_dir: Path, client: str, schema: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "client": client,
        "schema": schema,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "avg_score": round(sum(r.score for r in results) / max(len(results), 1), 2),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        failures = "<br>".join(html.escape(f) for f in (r.failures or []))
        rows.append(
            "<tr>"
            f"<td>{html.escape(status)}</td><td>{html.escape(r.id)}</td>"
            f"<td>{html.escape(str(r.score))}</td><td>{html.escape(r.question)}</td>"
            f"<td><pre>{html.escape(r.generated_sql)}</pre></td><td>{failures}</td>"
            "</tr>"
        )
    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>QueryBot Eval Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;vertical-align:top}}pre{{white-space:pre-wrap}}</style></head>
<body><h1>Eval Report: {html.escape(client)} / {html.escape(schema)}</h1>
<p>{payload['passed']} / {payload['total']} passed. Average score: {payload['avg_score']}</p>
<table><thead><tr><th>Status</th><th>ID</th><th>Score</th><th>Question</th><th>SQL</th><th>Failures</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (out_dir / "report.html").write_text(report, encoding="utf-8")


def _client_db_config(account_id: str) -> dict:
    client = store.get_client(account_id) or {}
    db_config_id = client.get("db_config_id")
    return store.get_db_config(db_config_id) if db_config_id else {}


async def _amain() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True, help="Client/account id")
    parser.add_argument("--schema", default="", help="Schema name for report grouping")
    parser.add_argument("--cases", required=True, help="YAML/JSON golden question file")
    parser.add_argument("--generate", action="store_true", help="Generate SQL using configured LLM when case.generated_sql is absent")
    parser.add_argument("--execute", action="store_true", help="Execute generated SQL against the configured DB")
    parser.add_argument("--out", default="", help="Output report directory")
    args = parser.parse_args()

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("evals") / "reports" / args.client / (args.schema or "default") / stamp
    results, run_id = await run_eval_suite(
        account_id=args.client,
        schema=args.schema or "default",
        cases_path=Path(args.cases),
        generate=args.generate,
        execute=args.execute,
        out_dir=out_dir,
    )

    passed = sum(1 for r in results if r.passed)
    print(f"{passed}/{len(results)} passed")
    print(f"Eval run id: {run_id}")
    print(f"Report: {out_dir / 'report.html'}")
    return 0 if passed == len(results) else 1


async def run_eval_suite(
    *,
    account_id: str,
    schema: str,
    cases_path: Path,
    generate: bool = False,
    execute: bool = False,
    out_dir: Path | None = None,
) -> tuple[list[EvalCaseResult], int]:
    """Run and persist a golden-question evaluation suite."""
    store.init_db()
    client = store.get_client(account_id) or {}
    state = json.loads(client.get("state_data") or "{}") if client else {}
    db_cfg = _client_db_config(account_id)
    db_type = db_cfg.get("db_type", "azure_sql")
    known_tables = load_known_tables(state.get("schema_dir", ""))
    cases = _load_cases(cases_path)
    results: list[EvalCaseResult] = []

    for case in cases:
        sql = (case.get("generated_sql") or "").strip()
        allowed = set(case.get("allowed_tables") or known_tables)
        if not sql and generate:
            sql = await _generate_sql(account_id, case["question"], db_type, allowed)

        execute_result = None
        if execute and sql:
            try:
                rows = run_query(db_cfg["credentials"], db_type, sql)
                execute_result = ("passed", len(rows), "")
            except Exception as exc:
                execute_result = ("failed", 0, str(exc)[:500])

        results.append(_score_sql(case, sql, known_tables, db_type, execute_result))

    if out_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("evals") / "reports" / account_id / (schema or "default") / stamp
    _write_reports(results, out_dir, account_id, schema or "default")

    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.score for r in results) / max(len(results), 1), 2)
    run_id = store.save_eval_run(
        account_id=account_id,
        schema_name=schema or "default",
        case_file=str(cases_path),
        total_cases=len(results),
        passed_cases=passed,
        avg_score=avg_score,
        status="passed" if passed == len(results) else "failed",
        report_path=str(out_dir / "report.html"),
    )
    for r in results:
        store.save_eval_case_result(
            run_id,
            case_id=r.id,
            question=r.question,
            score=r.score,
            passed=r.passed,
            generated_sql=r.generated_sql,
            validation_status=r.validation_status,
            validation_error=r.validation_error,
            execution_status=r.execution_status,
            row_count=r.row_count,
            failures=r.failures or [],
        )
    return results, run_id


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
