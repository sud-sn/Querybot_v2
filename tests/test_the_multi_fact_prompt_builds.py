"""
tests/test_the_multi_fact_prompt_builds.py

The governed two-fact question could not build its own SQL prompt.

When the join planner decides a question spans two physical facts -- revenue
and purchase cost by month -- it returns status "requires_isolated_aggregation",
and build_sql_system_prompt adds a mandatory section telling the model to build
one aggregate CTE per fact and join only the aggregates. That section serialises
the per-fact contracts with json.dumps.

core/llm.py never imported json. `git log -S` finds no commit in which it ever
did: the section was added on 2026-08-23 (de2eadf) calling a module that was not
there. So every two-fact question that reached free-form SQL generation -- any
the governed compiler declined, through any of its guard exits -- raised
NameError while building the prompt, outside any handler, and failed the turn.

Three test files reference requires_isolated_aggregation. None of them ever
built this prompt, which is how a NameError on a production path shipped green
and stayed that way for a month. It was found by the lint rule F821, switched on
for CI in the same commit as this test.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "QUERYBOT_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="qb-multifact-"), "querybot.db"),
)

from core.llm import build_sql_system_prompt  # noqa: E402

SUBPLANS = [
    {"fact": "DW.F_SALES", "measures": ["NET_AMT"], "grain": ["MONTH"]},
    {"fact": "DW.F_PURCHASES", "measures": ["COST_AMT"], "grain": ["MONTH"]},
]

MULTI_FACT = {"join_plan": {
    "status": "requires_isolated_aggregation",
    "fact_entities": ["F_SALES", "F_PURCHASES"],
    "common_dimensions": ["month"],
    "isolated_fact_plans": SUBPLANS,
}}


def _prompt(graph_context):
    return build_sql_system_prompt(
        "azure_sql", "F_SALES: NET_AMT decimal\nF_PURCHASES: COST_AMT decimal",
        graph_context=graph_context,
    )


def test_a_two_fact_question_gets_its_prompt():
    """The crash itself: this raised NameError before json was imported."""
    prompt = _prompt(MULTI_FACT)
    assert "Governed multi-fact execution plan" in prompt


def test_the_isolation_rule_reaches_the_model():
    """The point of the section. A prompt that builds but loses the rule would
    let the model join two facts directly -- the fan-out this whole path exists
    to prevent."""
    prompt = _prompt(MULTI_FACT)
    assert "one aggregate CTE per physical fact" in prompt
    assert "NEVER join two physical fact tables directly" in prompt
    assert "F_SALES, F_PURCHASES" in prompt
    assert "month" in prompt


def test_each_fact_contract_is_carried_whole_and_in_a_stable_order():
    """sort_keys matters: the same plan must produce the same prompt, or the
    prompt cache never hits for a question it has already seen."""
    prompt = _prompt(MULTI_FACT)
    assert json.dumps(SUBPLANS, sort_keys=True) in prompt

    reordered = {"join_plan": dict(MULTI_FACT["join_plan"], isolated_fact_plans=[
        {"grain": ["MONTH"], "measures": ["NET_AMT"], "fact": "DW.F_SALES"},
        {"grain": ["MONTH"], "measures": ["COST_AMT"], "fact": "DW.F_PURCHASES"},
    ])}
    assert _prompt(reordered) == prompt


def test_a_single_fact_question_is_unchanged():
    """The section is added only on the multi-fact status."""
    assert "Governed multi-fact execution plan" not in _prompt({})
