"""core2 must give the reference answer to every golden question, on every warehouse.

This is the level-2 gate of the core2 evaluation (docs/core-v2/DESIGN.md §11):
each golden plan is put in the learned model's own words, resolved, compiled and
run on DuckDB, and its rows must equal the reference SQL's rows. The same query
is compiled for Snowflake, Azure SQL and Oracle and must read back in each, with
every name resolving the way that warehouse folds names.

Names that carry meaning must answer everything; generic names may only miss
what could not be learned from values alone (a level-1 miss, reported as such).
"""

from __future__ import annotations

import pytest

from core2.compile.compiler import CompileError, check_references
from evals.core2 import domains
from evals.core2.compile_eval import evaluate, golden
from evals.core2.naming import STYLES

CASES = [(name, style) for name in domains.available() if golden(name) for style in STYLES]


@pytest.mark.parametrize("name, style", CASES)
def test_every_golden_question_gets_the_reference_answer(name, style):
    results = evaluate(name, style)
    assert results, f"{name} has golden questions but none ran"
    allowed = {"ok", "not_learned"} if style == "generic" else {"ok"}
    wrong = [f"{r.id}: {r.status}: {r.detail}\n{r.sql}" for r in results if r.status not in allowed]
    assert not wrong, "\n\n".join(wrong)


@pytest.mark.parametrize("dialect", ["snowflake", "oracle"])
def test_a_name_spelled_two_ways_is_caught_where_case_matters(dialect):
    # Defined unquoted (folded to Q), referenced quoted (q): a different name there.
    with pytest.raises(CompileError, match="names no table"):
        check_references('SELECT "q".x FROM (SELECT 1 AS x FROM t) AS q', dialect)
    check_references("SELECT q.x FROM (SELECT 1 AS x FROM t) AS q", dialect)
    check_references('SELECT "q".x FROM (SELECT 1 AS x FROM t) AS "q"', dialect)
