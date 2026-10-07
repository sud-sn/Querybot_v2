"""core2 must learn every synthetic warehouse right, in every naming style.

This is the level-1 gate of the core2 evaluation (docs/core-v2/DESIGN.md §11):
the bootstrap runs against each domain's DuckDB build and is graded against the
domain's ground truth. The thresholds are evals.core2.learn_eval.GATES (names
that carry meaning) and GENERIC_GATES (names that carry none, where only the
structure the values reveal can be asked for).
"""

from __future__ import annotations

import pytest

from evals.core2 import domains
from evals.core2.learn_eval import GATES, GENERIC_GATES, score
from evals.core2.naming import STYLES

CASES = [(name, style) for name in domains.available() for style in STYLES]
_built: dict[str, object] = {}


@pytest.mark.parametrize("name, style", CASES)
def test_the_learned_model_meets_the_gate(name, style):
    domain = _built.setdefault(name, domains.build(name))
    result, _ = score(domain, style)  # type: ignore[arg-type]
    gates = GENERIC_GATES if style == "generic" else GATES
    below = {part: f"{result.rate(part):.0%} < {minimum:.0%}" for part, minimum in gates.items()
             if part in result.parts and result.rate(part) < minimum}
    assert not below, "\n".join([str(below)] + result.misses)
