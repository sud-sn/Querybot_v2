"""Preflight checks for a production-path SQL accuracy baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import store
from core.kb_quality import load_kb_quality_report
from core.llm import resolve_provider
from evals.run import _load_cases


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str
    required: bool = True


def _check(name: str, passed: bool, detail: str, *, required: bool = True) -> ReadinessCheck:
    return ReadinessCheck(name=name, passed=bool(passed), detail=detail, required=required)


def _qdrant_reachable() -> tuple[bool, str]:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 6333)
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True, f"reachable at {host}:{port}"
    except OSError:
        return False, f"not reachable at {host}:{port}"


def evaluate_baseline_readiness(
    account_id: str,
    schema: str,
    cases_path: Path | None = None,
    *,
    minimum_cases: int = 100,
) -> dict:
    """Return a credential-safe readiness report for a live accuracy run."""
    store.init_db()
    client = store.get_client(account_id) or {}
    checks: list[ReadinessCheck] = []
    checks.append(_check("client", bool(client), "configured" if client else "client not found"))

    state: dict = {}
    if client:
        try:
            state = json.loads(client.get("state_data") or "{}")
        except (TypeError, ValueError):
            state = {}
    checks.append(_check(
        "workspace_state",
        client.get("state") == "READY",
        f"state={client.get('state') or 'missing'}",
    ))

    db_id = client.get("db_config_id") if client else None
    db_cfg = store.get_db_config(db_id) if db_id else None
    checks.append(_check(
        "database",
        bool(db_cfg),
        f"configured ({db_cfg.get('db_type')})" if db_cfg else "no database assigned",
    ))

    try:
        provider, model, api_key, _ = resolve_provider(client, purpose="query")
        checks.append(_check(
            "query_model",
            bool(api_key and model),
            f"configured ({provider}/{model})" if api_key and model else "model or key missing",
        ))
    except Exception as exc:
        checks.append(_check("query_model", False, str(exc)[:220]))

    schema_dir = Path(state.get("schema_dir") or "") if state.get("schema_dir") else None
    schema_files = list(schema_dir.glob("*.md")) if schema_dir and schema_dir.exists() else []
    checks.append(_check(
        "discovered_schema",
        bool(schema_files),
        f"{len(schema_files)} schema documents" if schema_files else "schema artifacts missing",
    ))

    kb_dir = Path(state.get("kb_dir") or "") if state.get("kb_dir") else None
    kb_files = list(kb_dir.glob("*_kb.md")) if kb_dir and kb_dir.exists() else []
    checks.append(_check(
        "knowledge_base",
        bool(kb_files),
        f"{len(kb_files)} table KB documents" if kb_files else "KB artifacts missing",
    ))
    quality = load_kb_quality_report(str(kb_dir)) if kb_dir else {}
    checks.append(_check(
        "kb_quality",
        quality.get("status") in {"ready", "needs_review"},
        f"status={quality.get('status') or 'missing'}, score={quality.get('score', 'n/a')}",
    ))
    contract_version = str(state.get("kb_built_contract_version") or "")
    checks.append(_check(
        "semantic_contract",
        bool(contract_version),
        f"version={contract_version}" if contract_version else "compiled contract missing",
    ))

    embedding_present = importlib.util.find_spec("sentence_transformers") is not None
    checks.append(_check(
        "embedding_runtime",
        embedding_present,
        "sentence-transformers installed" if embedding_present else "sentence-transformers is not installed",
    ))
    qdrant_ok, qdrant_detail = _qdrant_reachable()
    checks.append(_check("vector_store", qdrant_ok, qdrant_detail))

    case_path = cases_path or (
        Path("evals") / "clients" / account_id / schema / "golden_questions.yaml"
    )
    cases: list[dict] = []
    case_error = ""
    if case_path.exists():
        try:
            cases = _load_cases(case_path)
        except Exception as exc:
            case_error = str(exc)[:220]
    checks.append(_check(
        "golden_suite",
        len(cases) >= minimum_cases,
        (
            f"{len(cases)} cases (minimum {minimum_cases})"
            if cases else case_error or f"case file missing: {case_path}"
        ),
    ))
    asserted = sum(
        1 for case in cases
        if case.get("expected_rows") is not None
        or case.get("expected_result") is not None
        or case.get("expected_row_count") is not None
        or case.get("expected_validation") is not None
        or case.get("forbidden_sensitive_terms")
    )
    checks.append(_check(
        "result_assertions",
        bool(cases) and asserted == len(cases),
        f"{asserted}/{len(cases)} cases have result or safety assertions",
    ))

    blockers = [check for check in checks if check.required and not check.passed]
    return {
        "account_id": account_id,
        "schema": schema,
        "cases_path": str(case_path),
        "ready": not blockers,
        "blocker_count": len(blockers),
        "checks": [asdict(check) for check in checks],
    }


def main() -> None:
    # Windows consoles may default to cp1252 while configuration errors contain
    # Unicode arrows or punctuation. Readiness output must never crash while
    # reporting a blocker.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()
    report = evaluate_baseline_readiness(
        args.client,
        args.schema,
        Path(args.cases) if args.cases else None,
        minimum_cases=max(1, args.minimum_cases),
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            status = "PASS" if check["passed"] else "BLOCK"
            print(f"{status:5} {check['name']:20} {check['detail']}")
        print(f"\nReady: {'yes' if report['ready'] else 'no'} ({report['blocker_count']} blocker(s))")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
