"""
tests/test_a_measure_question_the_reader_can_answer.py

The option-less "which measure?", and the reader with no measure to name.

core.analytical_intent asks for a measure and offers nothing on purpose: when
no governed measure is relevant to the reader's words, a shortlist of unrelated
ones is worse than free text, and test_analytical_intent.py's
test_unknown_ranking_never_backfills_unrelated_metric_options pins that. Free
text is the intended reply mode.

What the pipeline did with that was send the bare sentence — no menu, and no
word telling the reader that a sentence of their own was expected. Worse,
against a workspace whose metric registry is empty in this reader's scope, no
sentence of theirs could have resolved it either: the loop asked its rounds and
then refused, having known from the first one that there was nothing to name.

Executed, not read. The gate sits 1,600 lines into a 6,700-line coroutine that
needs a warehouse and a websocket to reach, so it is lifted out of the shipped
function as a syntax tree and run here against a recording adapter with real
collaborators — the real planner, the real round counter, the real catalogue.
Asserting that the branch LOOKS right is exactly what lets a branch ship
looking right and behaving wrong.
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never the application database. Nothing here reads a table, but importing the
# store is enough to open one, so name a throwaway before any import does.
_tmpdir = tempfile.mkdtemp(prefix="querybot_measure_clarification_")
os.environ.setdefault("QUERYBOT_DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))

from core.analytical_intent import plan_analytical_intent  # noqa: E402
from core.clarification import (  # noqa: E402
    can_request_clarification,
    clarification_progress,
    measure_clarification_is_answerable,
)
from core.i18n import t  # noqa: E402


class _RecordingAdapter:
    """Records what the reader would have been sent, and in which shape."""

    def __init__(self):
        self.messages: list[str] = []
        self.prompts: list[tuple[str, list]] = []

    async def send_message(self, event, text, **kwargs):
        self.messages.append(text)

    async def send_clarification_prompt(self, event, question, options,
                                        pending_id=None):
        self.prompts.append((question, list(options)))


class _Event:
    user_id = "u1"
    raw: dict = {}


def _shipped_gate() -> str:
    """The analytical-plan clarification gate, out of the shipped function."""
    source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.If)
                and ast.unparse(node.test).startswith(
                    "_analytical_plan.needs_clarification")):
            return ast.unparse(node)
    raise AssertionError(
        "the analytical-plan clarification gate is no longer recognisable "
        "inside _handle_query_impl"
    )


def _run_gate(question, metrics, lang="en"):
    """Run the gate over one real plan and report what the reader got."""
    adapter = _RecordingAdapter()
    saved: list[dict] = []
    traces: list[dict] = []
    namespace = {
        "_analytical_plan": plan_analytical_intent(question, metrics=metrics),
        "_calendar_slot_requires_answer": False,
        "_planner_has_cached_result": False,
        "_planner_metrics": metrics,
        "can_request_clarification": can_request_clarification,
        "clarification_progress": clarification_progress,
        "measure_clarification_is_answerable": measure_clarification_is_answerable,
        "_save_pending_clarification": lambda q, c, meta: saved.append(meta),
        "adapter": adapter,
        "event": _Event(),
        "_t": t,
        "_trace_finish": lambda trace_id, **kw: traces.append(kw),
        "trace_id": "trace-1",
        "time": time,
        "start_ms": int(time.time() * 1000),
        "portal_user": {"lang": lang},
        "question": question,
    }
    # The gate returns, so it has to run inside a function; it awaits, so that
    # function has to be a coroutine.
    exec(compile(
        "async def _gate():\n" + textwrap.indent(_shipped_gate(), "    "),
        "<analytical-plan-gate>", "exec"), namespace)
    asyncio.run(namespace["_gate"]())
    return adapter, saved, traces


def test_the_gate_is_reached_at_all():
    """Without this, every assertion below would pass on a gate that was
    never entered."""
    adapter, _saved, _traces = _run_gate(
        "which warehouses have the highest total", metrics=[])
    assert adapter.messages, "the lifted gate produced nothing at all"


def test_a_measure_question_is_not_asked_when_there_is_no_measure_to_name():
    adapter, saved, traces = _run_gate(
        "which warehouses have the highest total", metrics=[], lang="fr")

    # The refusal this pipeline already gives an unresolved measure slot, in
    # the reader's own language, naming what the semantic layer is missing and
    # who can supply it.
    assert adapter.messages == [
        t("terminal.analytical_plan_unresolved", lang="fr",
          missing=t("slot.measure", lang="fr"))
    ]
    assert adapter.prompts == []
    # Nothing was asked, so nothing is pending and no round was spent.
    assert saved == []
    assert [trace["status"] for trace in traces] == ["error"]


def test_a_measure_question_is_still_asked_when_the_reader_has_measures():
    # Measures exist, none of them relevant to these words, so the planner
    # offers none -- but "by net revenue" is still an answer that can work,
    # and asking is right.
    metrics = [{"name": "Allocated Costs"}, {"name": "COGS"}]
    adapter, saved, traces = _run_gate(
        "which warehouses have the highest total", metrics=metrics)

    assert adapter.prompts == []          # no options: there is nothing to press
    assert len(adapter.messages) == 1
    sent = adapter.messages[0]
    assert t("clar.metric.ranking", lang="en") in sent
    # Naming COGS and Allocated Costs here would be the backfill
    # test_analytical_intent.py forbids, committed one layer up instead.
    assert "Allocated Costs" not in sent and "COGS" not in sent
    # ...but the reader is now told that their own words are the answer.
    assert t("clar.reply_in_plain_language", lang="en") in sent
    assert [meta["slot"] for meta in saved] == ["metric"]
    assert [trace["answer_type"] for trace in traces] == ["clarification"]


def test_an_option_less_business_definition_is_left_alone():
    """The narrowness of the rule, written as the case it must not swallow."""
    adapter, saved, traces = _run_gate(
        "which customers are at risk of churn by region", metrics=[])

    # No options, no measures in scope — and still a fair question: it asks for
    # the reader's own business rule, not for a name from the catalogue.
    assert len(adapter.messages) == 1
    assert "churn" in adapter.messages[0]
    assert [meta["slot"] for meta in saved] == ["business_definition"]
    assert [trace["answer_type"] for trace in traces] == ["clarification"]


def test_the_rule_itself_judges_only_the_measure_slots():
    assert measure_clarification_is_answerable("metric", (), []) is False
    assert measure_clarification_is_answerable("subject", (), []) is False
    assert measure_clarification_is_answerable(
        "metric", (), [{"name": "Revenue"}]) is True
    assert measure_clarification_is_answerable(
        "metric", ({"label": "Revenue"},), []) is True
    # A registry row with no usable name is not a measure anybody can name.
    assert measure_clarification_is_answerable(
        "metric", (), [{"name": "  "}]) is False
    for slot in ("business_definition", "recent_window", "calendar_basis",
                 "fiscal_year_start_month", "date_role", ""):
        assert measure_clarification_is_answerable(slot, (), []) is True


def test_the_slot_names_in_the_refusal_are_translated():
    """The refusal interpolates the slot name into a translated sentence. The
    names were English literals, so a French reader got a French sentence with
    an English noun phrase inside it."""
    for slot in ("source_fact", "measure", "count_target", "date_role",
                 "comparison_window"):
        key = f"slot.{slot}"
        english = t(key, lang="en")
        french = t(key, lang="fr")
        assert english != key, f"{key} is missing from the catalogue"
        assert french != key
        assert french != english, f"{key} is not translated"
