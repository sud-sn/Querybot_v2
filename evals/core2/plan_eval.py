"""Level 3 of core2's evaluation: from the question, through a real AI, to the numbers.

Each golden question of a domain goes through the whole question path with the
AI a workspace would use: member names found, the plan asked for and checked,
resolved, compiled, run on the DuckDB build, and the rows compared with the
reference SQL. A case that must ask (``expect.asks``) passes when the answer is a
question back; one that must decline (``expect.declines``) when it is declined.

The synthetic warehouses hold no customer data, so any provider may see them.

    export QUERYBOT_EVAL_API_KEY=...
    python -m evals.core2.plan_eval retail --provider azure_openai --model <deployment> \\
        --endpoint https://<resource>.openai.azure.com --api-version 2024-10-21
    python -m evals.core2.plan_eval inventory --provider anthropic --model <model name>

Every AI answer is written to ``--record`` (default
``evals/core2/recorded/<domain>.<model>.json``); ``--replay <file>`` runs the
same cases from a recording without calling any provider, so a run can be
re-checked after a change to anything downstream of the AI.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import golden, learn, same_rows
from evals.core2.framework import materialize

RECORDED = Path(__file__).parent / "recorded"


def provider_complete(provider: str, model: str, api_key: str, *, endpoint: str = "",
                      api_version: str = "2024-10-21") -> Callable[[str, str], str]:
    """The planner's AI, called the way QueryBot calls it (temperature 0, cached prefix)."""
    from core.llm import llm_complete
    from core.prompt_cache import CachedPrompt

    def complete(stable: str, tail: str) -> str:
        text, _, _ = asyncio.run(llm_complete(
            CachedPrompt(stable=stable, volatile="Answer with one JSON object."), tail, provider,  # type: ignore[arg-type]
            model, api_key, max_tokens=1500, azure_endpoint=endpoint, azure_api_version=api_version,
            temperature=0.0))
        return text

    return complete


class Recorder:
    """Keeps every AI answer per case; replays them in the same order."""

    def __init__(self, complete: Callable[[str, str], str] | None, replay: dict[str, list[str]] | None = None):
        self.complete, self.replay = complete, replay
        self.answers: dict[str, list[str]] = {}
        self.case = ""

    def __call__(self, stable: str, tail: str) -> str:
        if self.replay is not None:
            queue = self.replay.get(self.case) or []
            text = queue.pop(0) if queue else "{}"
        else:
            assert self.complete is not None
            text = self.complete(stable, tail)
        self.answers.setdefault(self.case, []).append(text)
        return text


def _status(case: dict[str, Any], payload: dict[str, Any], warehouse: DuckDBWarehouse, ran_before: int,
            reference: DuckDBWarehouse) -> tuple[str, str]:
    expect = case.get("expect") or {}
    if expect.get("asks"):
        return ("ok", "asked") if payload.get("clarify") else ("wrong", "should have asked")
    if expect.get("declines"):
        return ("ok", "declined") if payload.get("unsupported") else ("wrong", "should have declined")
    if payload.get("clarify"):
        return "asked", str(payload["answer"]["headline"])[:160]
    if len(warehouse.log) == ran_before:
        return "no_query", str(payload["answer"]["headline"])[:160]
    got = warehouse.query(warehouse.log[-1])
    expected = reference.query(case["reference_sql"])
    diff = same_rows(expected.columns, expected.rows, got.columns, got.rows,
                     order_matters=bool(expect.get("order_matters")))
    return ("wrong", diff) if diff else ("ok", "")


def evaluate(domain_name: str, style: str, complete: Recorder, *, today: dt.date | None = None) -> list[dict]:
    spec = golden(domain_name)
    domain = domains.build(domain_name)
    built, model = learn(domain, style)
    warehouse, reference = DuckDBWarehouse(built.con), DuckDBWarehouse(materialize(domain, "descriptive").con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    services = Services(model=model, warehouse=warehouse, complete=complete, index=build_index(model, fetch),
                        today=today or dt.date.fromisoformat(str(spec["today"])))
    results = []
    for case in spec.get("questions", []):
        if "turns" in case:
            continue          # multi-turn cases: a later round
        complete.case = case["id"]
        ran_before = len(warehouse.log)
        start = time.perf_counter()
        try:
            payload = answer_question(case["question"], services, Session())
            status, detail = _status(case, payload, warehouse, ran_before, reference)
            plan = payload.get("plan")
        except Exception as exc:     # noqa: BLE001 - reported per case
            status, detail, plan = "error", f"{type(exc).__name__}: {exc}", None
        results.append({"id": case["id"], "question": case["question"], "status": status, "detail": detail,
                        "plan": plan, "seconds": round(time.perf_counter() - start, 2)})
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("domain")
    parser.add_argument("--style", default="warehouse")
    parser.add_argument("--provider", default="azure_openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default=os.environ.get("QUERYBOT_EVAL_ENDPOINT", ""))
    parser.add_argument("--api-version", default="2024-10-21")
    parser.add_argument("--record", default="")
    parser.add_argument("--replay", default="")
    args = parser.parse_args(argv)
    if args.replay:
        recorder = Recorder(None, json.loads(Path(args.replay).read_text())["answers"])
    else:
        key = os.environ.get("QUERYBOT_EVAL_API_KEY", "")
        if not key or not args.model:
            parser.error("set QUERYBOT_EVAL_API_KEY and --model (or --replay a recording)")
        recorder = Recorder(provider_complete(args.provider, args.model, key, endpoint=args.endpoint,
                                              api_version=args.api_version))
    results = evaluate(args.domain, args.style, recorder)
    ok = sum(r["status"] == "ok" for r in results)
    print(f"{args.domain} ({args.style}): {ok}/{len(results)} right")
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['id']} {r['status']}: {r['question']}\n      {r['detail']}\n      plan: {json.dumps(r['plan'])}")
    if not args.replay:
        out = Path(args.record) if args.record else RECORDED / f"{args.domain}.{args.model or 'model'}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"domain": args.domain, "style": args.style, "provider": args.provider,
                                   "model": args.model, "answers": recorder.answers, "results": results}, indent=1))
        print(f"recorded to {out}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
