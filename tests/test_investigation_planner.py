# -*- coding: utf-8 -*-
"""Ask several questions instead of one, and prove the summary before it ships.

core.investigation_planner runs a bounded loop over the one tool an
investigation has (core.investigation.run_query_tool): the objective itself
is always the first question, and a model chooses every question after that
from nothing but the labels and figures the prior steps actually found.

The synthesis at the end is checked exactly as a reworded failure is
(core.situation_phraser): every figure it states must be one a step found,
every quoted term or column-like identifier must come from a step or the
objective. A synthesis that fails the check, a planner that returns garbage
or nothing, or a budget that runs out before the planner chooses to finish,
all fall back to the same template -- built from the steps alone, with no
model call.

The tool is replaced at its boundary (core.investigation.run_query_tool) for
every test in this module; core/test_investigation_tool.py already proves
that function against a real store with dispatch replaced one level further
down.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import core.investigation as investigation
from core.investigation import ToolResult
from core.investigation_planner import (
    InvestigationStep,
    build_planner_prompt,
    parse_planner_decision,
    run_investigation,
    template_synthesis,
    verify_synthesis,
)

OBJECTIVE = "why did scrap rise in Q2 2024"


def _step(index, question, *, ok=True, brief="", error=""):
    return InvestigationStep(
        index=index, question=question,
        result=ToolResult(ok=ok, kind="query", question=question, brief=brief, error=error,
                          result_id=f"r{index}" if ok else "", row_count=2 if ok else 0),
    )


def _run(coro):
    return asyncio.run(coro)


class TestThePlannerPrompt:

    def test_it_carries_the_objective_and_every_step(self):
        steps = [
            _step(1, OBJECTIVE, brief="Total scrap: 4200 units, up from 3100 in Q1."),
            _step(2, "scrap by work centre for Q2 2024", brief="Leader: WC10, 1800 units."),
        ]
        system, user = build_planner_prompt(OBJECTIVE, steps, steps_left=2)
        assert OBJECTIVE in user
        assert "Total scrap: 4200 units" in user
        assert "WC10, 1800 units" in user
        assert "action" in system and '"query"' in system and '"finish"' in system

    def test_a_failed_step_is_shown_as_such(self):
        steps = [_step(1, OBJECTIVE, ok=False, error="No scrap data for Q1 2023 in this workspace.")]
        _system, user = build_planner_prompt(OBJECTIVE, steps, steps_left=3)
        assert "Could not be answered: No scrap data for Q1 2023" in user

    def test_the_remaining_budget_is_told_plainly(self):
        system, _user = build_planner_prompt(OBJECTIVE, [], steps_left=0)
        assert "0 more question" in system
        assert "finish immediately" in system

    def test_a_french_reader_gets_the_french_rule(self):
        from core.i18n import prompt_language_rule

        system, _user = build_planner_prompt(OBJECTIVE, [], steps_left=1, lang="fr")
        assert prompt_language_rule("fr", shape="prose") in system


class TestParsingTheDecision:

    def test_a_query_decision(self):
        decision = parse_planner_decision(
            '{"action": "query", "question": "scrap by shift for Q2 2024", "reason": "narrow it down"}')
        assert decision.action == "query"
        assert decision.question == "scrap by shift for Q2 2024"

    def test_a_finish_decision(self):
        decision = parse_planner_decision('{"action": "finish", "synthesis": "Scrap rose because of WC10."}')
        assert decision.action == "finish"
        assert decision.synthesis == "Scrap rose because of WC10."

    def test_markdown_fences_are_stripped(self):
        decision = parse_planner_decision('```json\n{"action": "finish", "synthesis": "Done."}\n```')
        assert decision.action == "finish"

    @pytest.mark.parametrize("raw", [
        "not json at all",
        '{"action": "query"}',                                   # no question
        '{"action": "finish"}',                                  # no synthesis
        '{"action": "delete_everything", "question": "x"}',      # unsupported action
        '{"action": "query", "question": "x", "extra": "y"}',    # key outside the contract
        '[]',
        '',
    ])
    def test_anything_off_contract_is_an_empty_decision(self, raw):
        decision = parse_planner_decision(raw)
        assert decision.action == ""


class TestVerifyingTheSynthesis:

    def _steps(self):
        return [_step(1, OBJECTIVE, brief="Total scrap: 4200 units, up from 3100 in Q1 2024."),
                _step(2, "scrap by work centre", brief="Leader: work centre WC10, 1800 units.")]

    def test_a_faithful_restatement_passes(self):
        ok, why = verify_synthesis(
            "Scrap rose from 3100 to 4200 units in Q2 2024, led by work centre WC10 at 1800 units.",
            OBJECTIVE, self._steps())
        assert (ok, why) == (True, "")

    def test_an_invented_figure_is_refused(self):
        ok, why = verify_synthesis("Scrap rose to about 9000 units.", OBJECTIVE, self._steps())
        assert not ok and why.startswith("uncomputed_figure")

    def test_small_numbers_are_not_claims(self):
        ok, _ = verify_synthesis("The top 3 work centres drive most of it.", OBJECTIVE, self._steps())
        assert ok

    def test_an_unknown_quoted_term_is_refused(self):
        ok, why = verify_synthesis('The field "REBUT_PCT" explains it.', OBJECTIVE, self._steps())
        assert not ok and why.startswith("unknown_term")

    def test_an_unknown_identifier_is_refused(self):
        ok, why = verify_synthesis("NET_SLS_AMT explains the rise.", OBJECTIVE, self._steps())
        assert not ok and why.startswith("unknown_identifier")

    def test_a_failed_steps_own_reason_counts_as_evidence(self):
        steps = [_step(1, OBJECTIVE, ok=False, error="No scrap data for the Lyon plant in Q1 2024.")]
        ok, _ = verify_synthesis("No scrap data exists for the Lyon plant in Q1 2024.", OBJECTIVE, steps)
        assert ok

    def test_empty_and_oversized_are_both_refused(self):
        assert verify_synthesis("", OBJECTIVE, self._steps())[0] is False
        assert verify_synthesis("x" * 2000, OBJECTIVE, self._steps())[0] is False


class TestTheTemplateFallback:

    def test_it_lists_every_step_in_order(self):
        steps = [_step(1, OBJECTIVE, brief="Total scrap: 4200 units."),
                _step(2, "scrap by work centre", ok=False, error="Nothing matched.")]
        text = template_synthesis(OBJECTIVE, steps)
        assert OBJECTIVE in text
        assert text.index("Total scrap: 4200") < text.index("Nothing matched.")

    def test_it_speaks_the_readers_language(self):
        text = template_synthesis(OBJECTIVE, [_step(1, OBJECTIVE, brief="x")], lang="fr")
        assert "Investigation :" in text


@pytest.fixture
def planner(monkeypatch):
    """A scripted model: each call returns the next queued reply.

    Also allows this account's results to reach a model -- the same gate
    core.result_conversation and the analyst are held to, tested for itself
    in TestTheLoop's two regulated/no-profile cases below.
    """
    import core.compliance.policy_engine as pe

    monkeypatch.setattr(pe, "result_llm_features_allowed", lambda account_id: account_id == "acct-allowed")
    calls = {"prompts": []}
    replies: list[str] = []

    async def _complete(system="", user="", *a, **k):
        calls["prompts"].append((system, user))
        return replies.pop(0), 10, 10

    calls["replies"] = replies
    calls["complete"] = _complete
    return calls


def _fake_tool(outcomes: dict[str, ToolResult]):
    """A stand-in for run_query_tool keyed by the question asked."""
    async def _tool(*, account_id, portal_user, run_id, question):
        return outcomes.get(question, ToolResult(ok=False, kind="query", question=question,
                                                  error="unscripted question"))
    return _tool


class TestTheLoop:

    def test_step_one_always_asks_the_objective_verbatim(self, planner):
        seen = []

        async def _tool(*, account_id, portal_user, run_id, question):
            seen.append(question)
            return ToolResult(ok=True, kind="query", question=question, brief="Total: 4200 units.")

        planner["replies"].append('{"action": "finish", "synthesis": "Total is 4200 units."}')
        with patch.object(investigation, "run_query_tool", _tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r1", max_steps=3, complete=planner["complete"],
            ))
        assert seen == [OBJECTIVE]
        assert outcome.phrasing == "llm"
        assert outcome.synthesis == "Total is 4200 units."
        assert len(outcome.steps) == 1

    def test_a_query_decision_asks_a_second_question(self, planner):
        tool = _fake_tool({
            OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE,
                                  brief="Total scrap: 4200 units, up from 3100 in Q1 2024."),
            "scrap by work centre for Q2 2024": ToolResult(
                ok=True, kind="query", question="scrap by work centre for Q2 2024",
                brief="Leader: work centre WC10, 1800 units."),
        })
        planner["replies"].extend([
            '{"action": "query", "question": "scrap by work centre for Q2 2024", "reason": "narrow it down"}',
            '{"action": "finish", "synthesis": "Scrap rose from 3100 to 4200 units, led by WC10 at 1800 units."}',
        ])
        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r2", max_steps=4, complete=planner["complete"],
            ))
        assert [s.question for s in outcome.steps] == [OBJECTIVE, "scrap by work centre for Q2 2024"]
        assert outcome.phrasing == "llm"
        assert len(planner["prompts"]) == 2

    def test_a_rejected_synthesis_falls_back_to_the_template(self, planner):
        tool = _fake_tool({OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units.")})
        planner["replies"].append('{"action": "finish", "synthesis": "Scrap reached about 9000 units."}')
        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r3", max_steps=3, complete=planner["complete"],
            ))
        assert outcome.phrasing == "template"
        assert "4200" in outcome.synthesis

    def test_budget_exhaustion_without_a_finish_decision_uses_the_template(self, planner):
        tool = _fake_tool({
            OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units."),
            "step two": ToolResult(ok=True, kind="query", question="step two", brief="Leader: WC10."),
        })
        planner["replies"].append('{"action": "query", "question": "step two", "reason": "more detail"}')
        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r4", max_steps=2, complete=planner["complete"],
            ))
        assert len(outcome.steps) == 2
        assert outcome.phrasing == "template"
        # The planner was asked only once: after step 1, with one question
        # left, the loop must not ask it again after step 2 uses up the budget.
        assert len(planner["prompts"]) == 1

    def test_an_unreachable_planner_still_finishes_with_the_template(self, planner):
        tool = _fake_tool({OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units.")})

        async def _boom(**kwargs):
            raise RuntimeError("provider down")

        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r5", max_steps=3, complete=_boom,
            ))
        assert outcome.phrasing == "template"
        assert len(outcome.steps) == 1

    def test_garbage_from_the_planner_ends_the_loop_without_crashing(self, planner):
        tool = _fake_tool({OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units.")})
        planner["replies"].append("I would ask about work centres next.")
        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r6", max_steps=3, complete=planner["complete"],
            ))
        assert outcome.phrasing == "template"
        assert len(outcome.steps) == 1

    def test_a_failed_first_step_still_reaches_the_planner(self, planner):
        tool = _fake_tool({})  # every question is "unscripted" -> ok=False
        planner["replies"].append('{"action": "finish", "synthesis": "No data was found for this objective."}')
        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r7", max_steps=3, complete=planner["complete"],
            ))
        assert outcome.steps[0].result.ok is False
        assert outcome.phrasing == "llm"

    def test_a_regulated_tenant_never_reaches_the_tool_or_the_planner(self, monkeypatch, planner):
        import core.compliance.policy_engine as pe

        monkeypatch.setattr(pe, "result_llm_features_allowed", lambda _a: False)
        fake_tool = _fake_tool({})
        with patch.object(investigation, "run_query_tool", fake_tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-regulated", portal_user={"id": 7},
                run_id="r8", max_steps=3, complete=planner["complete"],
            ))
        assert outcome.steps == []
        assert outcome.refused
        assert planner["prompts"] == []

    def test_on_step_is_called_as_each_step_finishes(self, planner):
        tool = _fake_tool({
            OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units."),
            "step two": ToolResult(ok=True, kind="query", question="step two", brief="Leader: WC10."),
        })
        planner["replies"].extend([
            '{"action": "query", "question": "step two", "reason": "more detail"}',
            '{"action": "finish", "synthesis": "WC10 leads."}',
        ])
        seen = []
        with patch.object(investigation, "run_query_tool", tool):
            _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r10", max_steps=3, complete=planner["complete"], on_step=seen.append,
            ))
        assert [s.question for s in seen] == [OBJECTIVE, "step two"]
        # Called as each step finishes, not after the whole loop -- the first
        # callback must have already happened before the planner is asked
        # for the second question at all.
        assert seen[0].result.brief == "Total: 4200 units."

    def test_a_raising_on_step_does_not_break_the_loop(self, planner):
        tool = _fake_tool({OBJECTIVE: ToolResult(ok=True, kind="query", question=OBJECTIVE, brief="Total: 4200 units.")})
        planner["replies"].append('{"action": "finish", "synthesis": "Total is 4200 units."}')

        def _boom(_step):
            raise RuntimeError("UI is gone")

        with patch.object(investigation, "run_query_tool", tool):
            outcome = _run(run_investigation(
                objective=OBJECTIVE, account_id="acct-allowed", portal_user={"id": 7},
                run_id="r11", max_steps=3, complete=planner["complete"], on_step=_boom,
            ))
        assert outcome.phrasing == "llm"
        assert outcome.synthesis == "Total is 4200 units."

    def test_the_real_policy_fails_closed_without_a_profile(self, planner):
        outcome = _run(run_investigation(
            objective=OBJECTIVE, account_id="acct-investigation-no-profile", portal_user={"id": 7},
            run_id="r9", max_steps=3, complete=planner["complete"],
        ))
        assert outcome.steps == []
        assert outcome.refused


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
