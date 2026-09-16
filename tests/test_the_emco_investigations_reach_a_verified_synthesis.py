# -*- coding: utf-8 -*-
"""Three EMCO investigations, run end to end through the real loop.

core/investigation_planner.py's own suite proves the loop's mechanics in
the abstract; this proves it against the three shapes it was built for --
manufacturing questions that need more than one governed fact to answer
well: scrap by work centre, on-time-in-full by customer, and margin by
product line. Each scenario scripts a realistic multi-step plan and a
realistic model-written synthesis, and checks that synthesis the same way
a real one would be checked -- so a scenario earns its place here only if
its wording is close enough to what a model would actually write that the
check has something real to accept or refuse.

The tool is replaced at its boundary (core.investigation.run_query_tool);
tests/test_investigation_tool.py already proves that function against a
real store with dispatch replaced one level further down.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import core.investigation as investigation
from core.investigation import ToolResult
from core.investigation_planner import run_investigation


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def allowed(monkeypatch):
    import core.compliance.policy_engine as pe

    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: True)


def _scripted_tool(outcomes: dict[str, ToolResult]):
    async def _tool(*, account_id, portal_user, run_id, question):
        return outcomes.get(question, ToolResult(ok=False, kind="query", question=question,
                                                  error="unscripted question"))
    return _tool


def _scripted_planner(replies: list[str]):
    queue = list(replies)

    async def _complete(system="", user="", *a, **k):
        return queue.pop(0), 10, 10

    return _complete


class TestScrapByWorkCentre:

    OBJECTIVE = "why did scrap rise at the Lyon plant in Q2 2024"

    def test_the_investigation_reaches_a_checked_synthesis(self, allowed):
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Mode: time_series\nTotal scrap quantity: 4200 units in Q2 2024, up from 3100 in Q1 2024."),
            "scrap quantity by work centre at the Lyon plant for Q2 2024": ToolResult(
                ok=True, kind="query", question="scrap quantity by work centre at the Lyon plant for Q2 2024",
                brief="Mode: ranking\nLeader: work centre WC10, 1800 units, 43% of total."),
            "scrap quantity for work centre WC10 at the Lyon plant for Q1 2024": ToolResult(
                ok=True, kind="query", question="scrap quantity for work centre WC10 at the Lyon plant for Q1 2024",
                brief="Mode: single_value\nWork centre WC10 scrap quantity: 900 units in Q1 2024."),
        }
        planner = _scripted_planner([
            '{"action": "query", "question": "scrap quantity by work centre at the Lyon plant for Q2 2024", "reason": "find which work centre drove the rise"}',
            '{"action": "query", "question": "scrap quantity for work centre WC10 at the Lyon plant for Q1 2024", "reason": "confirm WC10 also grew, not just its share"}',
            '{"action": "finish", "synthesis": "Scrap at the Lyon plant rose from 3100 to 4200 units between Q1 and Q2 2024. Work centre WC10 led the increase, from 900 to 1800 units, 43% of the Q2 total."}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-scrap", max_steps=4, complete=planner,
            ))
        assert outcome.phrasing == "llm"
        assert len(outcome.steps) == 3
        assert "WC10" in outcome.synthesis and "1800" in outcome.synthesis

    def test_an_invented_root_cause_is_refused_even_here(self, allowed):
        """A plausible-sounding but uncomputed explanation -- a real failure
        mode, not a strawman -- must still fall back to the template."""
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Total scrap quantity: 4200 units in Q2 2024, up from 3100 in Q1 2024."),
        }
        planner = _scripted_planner([
            '{"action": "finish", "synthesis": "Scrap rose because a new supplier delivered 600 defective coils in May."}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-scrap-2", max_steps=3, complete=planner,
            ))
        assert outcome.phrasing == "template"
        assert "4200" in outcome.synthesis


class TestOnTimeInFullByCustomer:

    OBJECTIVE = "which customers had the worst on-time-in-full rate last quarter"

    def test_the_investigation_reaches_a_checked_synthesis(self, allowed):
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Mode: ranking\nLowest: EMCO Corp EU, OTIF 68.5%, 210 orders."),
            "on-time-in-full rate for EMCO Corp EU by month last quarter": ToolResult(
                ok=True, kind="query", question="on-time-in-full rate for EMCO Corp EU by month last quarter",
                brief="Mode: time_series\nOTIF for EMCO Corp EU fell from 81.2% to 68.5% over the quarter."),
        }
        planner = _scripted_planner([
            '{"action": "query", "question": "on-time-in-full rate for EMCO Corp EU by month last quarter", "reason": "see whether the low rate is a trend or one bad month"}',
            '{"action": "finish", "synthesis": "EMCO Corp EU had the worst OTIF rate last quarter at 68.5% across 210 orders, and it fell steadily from 81.2% over the quarter rather than one bad month."}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-otif", max_steps=4, complete=planner,
            ))
        assert outcome.phrasing == "llm"
        assert "68.5%" in outcome.synthesis and "EMCO Corp EU" in outcome.synthesis

    def test_a_quoted_customer_name_the_steps_never_returned_is_refused(self, allowed):
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Lowest: EMCO Corp EU, OTIF 68.5%, 210 orders."),
        }
        import json as _json

        quoted_synthesis = _json.dumps({
            "action": "finish",
            "synthesis": '"Acme Distribution" had the worst rate at 68.5%.',
        })
        planner = _scripted_planner([quoted_synthesis])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-otif-2", max_steps=3, complete=planner,
            ))
        assert outcome.phrasing == "template"
        assert "EMCO Corp EU" in outcome.synthesis

    def test_a_bare_unquoted_customer_name_is_a_known_gap_not_a_silent_one(self, allowed):
        """verify_synthesis catches a figure that was not computed and a
        quoted or warehouse-shaped token that was not returned; it does not
        parse ordinary prose for a swapped proper noun with neither marker.
        core.situation_phraser's identical check has the same boundary. This
        test exists so that boundary is asserted, not merely unexercised --
        a future tightening of either check should have to change this test
        on purpose, not discover the gap by accident."""
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Lowest: EMCO Corp EU, OTIF 68.5%, 210 orders."),
        }
        planner = _scripted_planner([
            '{"action": "finish", "synthesis": "Acme Distribution had the worst rate at 68.5%."}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-otif-3", max_steps=3, complete=planner,
            ))
        assert outcome.phrasing == "llm"
        assert "Acme Distribution" in outcome.synthesis


class TestMarginByProductLine:

    OBJECTIVE = "why is gross margin down for the hydraulics product line this year"

    def test_a_failed_step_still_reaches_a_useful_synthesis(self, allowed):
        """The first governed question this objective produces may find
        nothing -- "this year" resolved against a fiscal calendar the
        workspace does not have loaded, say -- and the investigation must
        still make something of it rather than dying silently."""
        outcomes = {}  # every question is unscripted -> ok=False, generic error
        planner = _scripted_planner([
            '{"action": "finish", "synthesis": "Gross margin for the hydraulics product line could not be computed for this year in this workspace."}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-margin", max_steps=3, complete=planner,
            ))
        assert outcome.steps[0].result.ok is False
        # A synthesis that only restates the failure, adding no invented
        # figure, still passes -- the check is about invented CONTENT, not
        # about whether the underlying question succeeded.
        assert outcome.phrasing == "llm"

    def test_budget_exhaustion_still_produces_a_usable_trail(self, allowed):
        outcomes = {
            self.OBJECTIVE: ToolResult(
                ok=True, kind="query", question=self.OBJECTIVE,
                brief="Gross margin percent for hydraulics: 22.4% this year, down from 28.1% last year."),
            "cost of goods sold for the hydraulics product line this year versus last year": ToolResult(
                ok=True, kind="query",
                question="cost of goods sold for the hydraulics product line this year versus last year",
                brief="COGS for hydraulics: 4.1M this year, up from 3.2M last year."),
        }
        planner = _scripted_planner([
            '{"action": "query", "question": "cost of goods sold for the hydraulics product line this year versus last year", "reason": "margin fell -- check cost first"}',
        ])
        with patch.object(investigation, "run_query_tool", _scripted_tool(outcomes)):
            outcome = _run(run_investigation(
                objective=self.OBJECTIVE, account_id="acct-emco", portal_user={"id": 7},
                run_id="emco-margin-2", max_steps=2, complete=planner,
            ))
        assert len(outcome.steps) == 2
        assert outcome.phrasing == "template"
        assert "22.4%" in outcome.synthesis and "4.1M" in outcome.synthesis


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
