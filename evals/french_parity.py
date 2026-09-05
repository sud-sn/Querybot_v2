"""
evals/french_parity.py

How much of a French question the product still understands.

Every intent detector in the pipeline is a hand-written English regex that runs
before any model sees the question, so a French question used to arrive with
almost nothing detected -- and nothing errored. This measures what
core/question_normalizer.py recovers, per case and in total, against the
English phrasing of the same question.

The number to watch is PARITY: the share of intents the English question
detects that the canonicalised French one also detects. Anything below 1.0 is a
question a French customer asks and gets a different answer to.

    python -m evals.french_parity
    python -m evals.french_parity --min 1.0 --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

CORPUS = Path(__file__).resolve().parent / "french_questions.yaml"


@dataclass
class CaseResult:
    id: str
    intent: str
    english: str
    french: str
    canonical: str
    english_signals: list[str]
    raw_signals: list[str]
    canonical_signals: list[str]
    missing: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing


def _signals(text: str) -> set[str]:
    """Everything the pipeline's front door detects in one question.

    The causal route and the row limit are in here alongside the fifteen
    analytical intents, because losing either changes the answer just as much
    -- a lost row limit hands the reader every customer, narrated as the top 10.
    """
    from core.insight import detect_analytical_intents, is_causal_question
    from core.query_semantics import analyze_query_intent, detect_top_n_intent

    found = {name for name, value in detect_analytical_intents(text).items() if value}
    found |= {f"flag:{name}" for name, value in analyze_query_intent(text).items() if value}
    if is_causal_question(text):
        found.add("causal")
    top_n = detect_top_n_intent(text)
    if top_n:
        found.add(f"top_n:{top_n.limit}:{top_n.direction}")
    return found


def run(corpus: Path = CORPUS) -> list[CaseResult]:
    from core.question_normalizer import canonical_question

    payload = yaml.safe_load(corpus.read_text(encoding="utf-8")) or {}
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    results = []
    for case in cases or []:
        english, french = case["english"], case["french"]
        canonical = canonical_question(french, "fr")
        english_signals = _signals(english)
        canonical_signals = _signals(canonical)
        results.append(CaseResult(
            id=case["id"],
            intent=case.get("intent", ""),
            english=english,
            french=french,
            canonical=canonical,
            english_signals=sorted(english_signals),
            raw_signals=sorted(_signals(french)),
            canonical_signals=sorted(canonical_signals),
            missing=sorted(english_signals - canonical_signals),
        ))
    return results


def summarise(results: list[CaseResult]) -> dict:
    expected = sum(len(r.english_signals) for r in results)
    raw = sum(len(set(r.raw_signals) & set(r.english_signals)) for r in results)
    recovered = expected - sum(len(r.missing) for r in results)
    return {
        "cases": len(results),
        "cases_passed": sum(1 for r in results if r.passed),
        "signals_expected": expected,
        "signals_raw_french": raw,
        "signals_recovered": recovered,
        "parity": round(recovered / expected, 4) if expected else 1.0,
        "parity_without_normaliser": round(raw / expected, 4) if expected else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=float, default=1.0,
                        help="fail below this parity (default 1.0)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = run()
    summary = summarise(results)

    if args.json:
        print(json.dumps({"summary": summary,
                          "cases": [asdict(r) for r in results]}, indent=2))
    else:
        for result in results:
            mark = "ok  " if result.passed else "MISS"
            print(f"{mark} {result.id:24} {result.french}")
            if not result.passed:
                print(f"     missing: {', '.join(result.missing)}")
                print(f"     canonical: {result.canonical}")
        print()
        print(f"cases            {summary['cases_passed']}/{summary['cases']}")
        print(f"signals          {summary['signals_recovered']}/{summary['signals_expected']}")
        print(f"parity           {summary['parity']:.1%}")
        print(f"without this     {summary['parity_without_normaliser']:.1%}")

    return 0 if summary["parity"] >= args.min else 1


if __name__ == "__main__":
    raise SystemExit(main())
