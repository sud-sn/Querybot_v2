"""Follow-up or new question: does the new core read each turn of a conversation as a person meant it?

Each conversation in ``conversations/<domain>.yaml`` is a thread of questions about one synthetic
warehouse. Every turn after the first carries a label, written by hand:

* ``refine``: it continues the answer on screen ("only North", "by week instead", "the first one",
  "why did it drop?"). Right when the plan is a refinement and keeps what the turn says it keeps
  (``keeps``: measures, group_by, filters, window) from the answer before it.
* ``new``: a question of its own, whatever words it shares with the one before. Right when the plan
  is new and carries none of what the turn says it drops (``drops``) from the answer before it.
* ``unclear``: a short, complete question after a narrowed answer ("How many orders?" after net
  sales for one region in one month), which a person could mean either way. Right when the new
  core asks which was meant; answering either way is a guess.

A turn that is asked about when it is ``refine`` or ``new`` counts as asked, not right. The agreed
targets: at least 90% of turns right, at most 10% asked.

The turns go through the whole question path (members found, plan asked for and checked, answer
computed on the DuckDB build) with the AI a workspace uses, and every AI answer is recorded, so a
run can be replayed without a provider after a change downstream of the AI (see plan_eval):

    export QUERYBOT_EVAL_API_KEY=...
    python -m evals.core2.followup_eval --provider azure_openai --model <deployment> --endpoint https://...
    python -m evals.core2.followup_eval retail networking --replay evals/core2/recorded/followups.<model>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from core2.plan.ir import Plan
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn
from evals.core2.plan_eval import RECORDED, Recorder, provider_complete

CONVERSATIONS = Path(__file__).parent / "conversations"
LABELS = ("refine", "new", "unclear")
KINDS = {
    "refine": {"filter", "exclude", "regroup", "drill", "grain", "date_role", "period", "compare", "measure",
               "add_measure", "share", "topn", "sort", "chart", "reference", "why", "forecast"},
    "new": {"switch", "restate", "describe"},
    "unclear": {"bare"},
}
PARTS = ("measures", "group_by", "filters", "window")
TARGETS = {"right": 0.90, "asked": 0.10}


def conversations(domain: str, folder: Path | None = None) -> dict[str, Any]:
    path = (folder or CONVERSATIONS) / f"{domain}.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def available(folder: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (folder or CONVERSATIONS).glob("*.yaml"))


def _part(plan: Plan, part: str) -> list[str]:
    if part == "measures":
        return sorted(plan.measures + [d.name for d in plan.derived])
    if part == "group_by":
        return sorted(plan.group_by)
    if part == "filters":
        return sorted(json.dumps(f.model_dump(mode="json"), sort_keys=True) for f in plan.filters)
    return [json.dumps(plan.time.window.model_dump(mode="json", exclude_defaults=True), sort_keys=True)]


def _empty(plan: Plan, part: str) -> bool:
    return _part(plan, part) in ([], ["{}"])


def judge(turn: dict[str, Any], plan: Plan | None, *, asked: bool, previous: Plan | None) -> tuple[str, str]:
    """``right``, ``asked``, ``guessed`` (an unclear turn answered) or ``wrong``, and why."""
    label = turn["label"]
    if label == "unclear":
        if asked:
            return "right", "asked which was meant"
        reading = plan.follow_up if plan is not None else "nothing"
        return "guessed", f"answered as {reading} without asking"
    if asked:
        return "asked", "asked, though the turn is clear"
    if plan is None:
        return "wrong", "no plan"
    if label == "refine":
        if plan.kind != "query":
            return "wrong", f"answered as {plan.kind}, not as a change to the answer on screen"
        if plan.follow_up != "refine":
            return "wrong", "read as a new question"
        if previous is None:
            return "wrong", "no answer before it to refine"
        lost = [p for p in turn.get("keeps", []) if not set(_part(previous, p)) <= set(_part(plan, p))]
        return ("wrong", f"dropped the answer's {', '.join(lost)}") if lost else ("right", "")
    if plan.kind == "query" and plan.follow_up != "new":
        return "wrong", "read as a change to the answer on screen"
    if previous is not None and plan.kind == "query":
        kept = [p for p in turn.get("drops", []) if not _empty(previous, p)
                and set(_part(previous, p)) & set(_part(plan, p))]
        if kept:
            return "wrong", f"carried the last answer's {', '.join(kept)} into a new question"
    return "right", ""


@contextmanager
def _planned(into: list[Plan]) -> Iterator[None]:
    """Each plan the planner makes, kept: the reading of a turn is the plan, whatever happens after it
    (a describe answer, a refusal for a member the data lacks), and only answered plans reach the payload."""
    from core2 import service

    original = service.plan_question

    def kept(*args: Any, **kwargs: Any):
        outcome = original(*args, **kwargs)
        into.append(outcome.plan)
        return outcome

    service.plan_question = kept
    try:
        yield
    finally:
        service.plan_question = original


def evaluate(domain_name: str, complete: Recorder, *, style: str = "descriptive",
             folder: Path | None = None) -> list[dict[str, Any]]:
    spec = conversations(domain_name, folder)
    domain = domains.build(domain_name)
    built, model = learn(domain, style)
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    today = dt.date.fromisoformat(str(spec.get("today") or domain.today))
    services = Services(model=model, warehouse=warehouse, complete=complete,
                        index=build_index(model, fetch), today=today)
    results = []
    planned: list[Plan] = []
    for thread in spec.get("threads", []):
        session = Session()
        previous: Plan | None = None
        for n, turn in enumerate(thread["turns"]):
            complete.case = f"{domain_name}:{thread['id']}:{n}"
            start = time.perf_counter()
            planned.clear()
            try:
                with _planned(planned):
                    payload = answer_question(turn["q"], services, session)
                asked = bool(payload.get("clarify"))
                plan = planned[-1] if planned else (
                    Plan.model_validate(payload["plan"]) if payload.get("plan") else None)
                error = ""
            except Exception as exc:     # noqa: BLE001 - reported per turn
                payload, asked, plan, error = {}, False, None, f"{type(exc).__name__}: {exc}"
            if asked:
                session.pending = None   # the reader moves on to the next question as written
            if n and "label" in turn:
                status, detail = ("error", error) if error else judge(turn, plan, asked=asked, previous=previous)
                results.append({"domain": domain_name, "thread": thread["id"], "turn": n, "question": turn["q"],
                                "label": turn["label"], "kind": turn.get("kind", ""), "status": status,
                                "detail": detail,
                                "plan": plan.model_dump(mode="json", exclude_defaults=True) if plan else None,
                                "seconds": round(time.perf_counter() - start, 2)})
            if plan is not None and plan.kind == "query" and not asked:
                previous = plan
    return results


def signals(domain_name: str, *, style: str = "descriptive", folder: Path | None = None) -> list[dict[str, Any]]:
    """The readings decided in code (core2/plan/followup.py), with no AI: which turns they decide, and how right.

    The answer before each turn is stood in for by its question: the measures it names, and whether it set a
    period, a grouping or a member (what a narrowed answer carries).
    """
    from core2.plan.followup import measures_named, read_turn, scoped

    spec = conversations(domain_name, folder)
    built, model = learn(domains.build(domain_name), style)
    warehouse = DuckDBWarehouse(built.con)

    def fetch(slug: str) -> list[object]:
        column = model.columns[model.attributes[slug].column]
        table = model.tables[column.table]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM "{table.name}"').rows]

    index = build_index(model, fetch)
    out = []
    for thread in spec.get("threads", []):
        previous: str | None = None
        for n, turn in enumerate(thread["turns"]):
            q = turn["q"]
            if n and "label" in turn:
                stand_in = None if previous is None else Plan(intent="value", measures=sorted(
                    measures_named(model, previous)))
                reading = read_turn(q, previous=stand_in, model=model, members=[m.text for m in index.match(q)],
                                    previous_narrowed=previous is not None and scoped(previous,
                                                                                     len(index.match(previous))))
                want = {"refine": "refine", "new": "new", "unclear": "unsure"}[turn["label"]]
                status = "open" if reading.kind == "open" else ("right" if reading.kind == want else "wrong")
                out.append({"domain": domain_name, "thread": thread["id"], "turn": n, "question": q,
                            "label": turn["label"], "kind": turn.get("kind", ""), "reading": reading.kind,
                            "why": reading.why, "status": status})
            if turn.get("kind") != "describe":
                previous = q
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def shares(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(r["status"] for r in rows)
        n = len(rows) or 1
        return {"turns": len(rows), **{s: round(counts[s] / n, 3) for s in ("right", "asked", "guessed", "wrong", "error")}}

    by_label: dict[str, list] = defaultdict(list)
    by_kind: dict[str, list] = defaultdict(list)
    by_domain: dict[str, list] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)
        by_kind[f"{r['label']}:{r['kind']}"].append(r)
        by_domain[r["domain"]].append(r)
    return {"overall": shares(results), "by_label": {k: shares(v) for k, v in sorted(by_label.items())},
            "by_kind": {k: shares(v) for k, v in sorted(by_kind.items())},
            "by_domain": {k: shares(v) for k, v in sorted(by_domain.items())}}


def report(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    def line(name: str, s: dict[str, Any]) -> str:
        return (f"  {name:24} {s['turns']:4} turns  right {s['right']:5.0%}  asked {s['asked']:5.0%}  "
                f"guessed {s['guessed']:5.0%}  wrong {s['wrong']:5.0%}  error {s['error']:5.0%}")

    o = summary["overall"]
    lines = [f"follow-up or new: {o['turns']} turns, right {o['right']:.0%}, asked {o['asked']:.0%} "
             f"(targets: right >= {TARGETS['right']:.0%}, asked <= {TARGETS['asked']:.0%})"]
    for title in ("by_label", "by_domain", "by_kind"):
        lines.append(title.replace("_", " "))
        lines += [line(name, s) for name, s in summary[title].items()]
    misses = [r for r in results if r["status"] != "right"]
    if misses:
        lines.append("not right")
        lines += [f"  [{r['domain']} {r['thread']}.{r['turn']} {r['label']}:{r['kind']}] {r['status']}: "
                  f"{r['question']} -- {r['detail']}" for r in misses]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("domains", nargs="*", help="domains to run (default: every conversation file)")
    parser.add_argument("--style", default="descriptive")
    parser.add_argument("--provider", default="azure_openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--endpoint", default=os.environ.get("QUERYBOT_EVAL_ENDPOINT", ""))
    parser.add_argument("--api-version", default="2024-10-21")
    parser.add_argument("--record", default="")
    parser.add_argument("--replay", default="")
    parser.add_argument("--signals", action="store_true", help="only the readings decided in code; no AI")
    parser.add_argument("--set", default="", help="a set of conversations under conversations/ (heldout: never "
                        "used to tune the rules)")
    args = parser.parse_args(argv)
    folder = CONVERSATIONS / args.set if args.set else CONVERSATIONS
    if args.signals:
        rows = [r for name in (args.domains or available(folder)) for r in signals(name, style=args.style,
                                                                                   folder=folder)]
        decided = [r for r in rows if r["status"] != "open"]
        right = sum(r["status"] == "right" for r in decided)
        asked = sum(r["reading"] == "unsure" for r in rows)
        print(f"decided in code: {len(decided)}/{len(rows)} ({len(decided) / len(rows):.0%}); right {right}/"
              f"{len(decided)} ({right / max(1, len(decided)):.0%}); asked {asked} ({asked / len(rows):.0%}); "
              f"left to the AI {len(rows) - len(decided)}")
        for r in rows:
            if r["status"] == "wrong":
                print(f"  wrong [{r['domain']} {r['thread']}.{r['turn']} {r['label']}:{r['kind']}] read as "
                      f"{r['reading']}: {r['question']} -- {r['why']}")
        for r in rows:
            if r["status"] == "open":
                print(f"  open  [{r['domain']} {r['thread']}.{r['turn']} {r['label']}:{r['kind']}] {r['question']}")
        return 0
    if args.replay:
        recorder = Recorder(None, json.loads(Path(args.replay).read_text())["answers"])
    else:
        key = os.environ.get("QUERYBOT_EVAL_API_KEY", "")
        if not key or not args.model:
            parser.error("set QUERYBOT_EVAL_API_KEY and --model (or --replay a recording)")
        recorder = Recorder(provider_complete(args.provider, args.model, key, endpoint=args.endpoint,
                                              api_version=args.api_version))
    results = []
    for name in args.domains or available(folder):
        results += evaluate(name, recorder, style=args.style, folder=folder)
    summary = summarize(results)
    print(report(results, summary))
    if not args.replay:
        out = Path(args.record) if args.record else RECORDED / f"followups.{args.model or 'model'}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"style": args.style, "provider": args.provider, "model": args.model,
                                   "answers": recorder.answers, "summary": summary, "results": results}, indent=1))
        print(f"recorded to {out}")
    o = summary["overall"]
    return 0 if o["right"] >= TARGETS["right"] and o["asked"] <= TARGETS["asked"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
