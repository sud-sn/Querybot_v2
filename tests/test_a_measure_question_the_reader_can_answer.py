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
import logging
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
    """A PlatformEvent as the clarification loop reads one.

    ``rounds_spent`` drives clarification_progress, which is what decides
    whether the reader can still be asked. The refusal below only fires once
    the loop is spent, so a test that never spends it cannot see it.
    """

    user_id = "u1"

    def __init__(self, rounds_spent: int = 0, max_rounds: int = 3):
        self.raw = {
            "_clarification_round": rounds_spent,
            "_clarification_max_rounds": max_rounds,
        }


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


def _run_gate(question, metrics, lang="en", metrics_read=True, rounds_spent=0):
    """Run the gate over one real plan and report what the reader got.

    ``metrics_read`` is the pipeline's own _planner_metrics_read: whether the
    registry was actually read this turn, as opposed to read and found empty.
    """
    adapter = _RecordingAdapter()
    saved: list[dict] = []
    traces: list[dict] = []
    namespace = {
        "_analytical_plan": plan_analytical_intent(question, metrics=metrics),
        "_calendar_slot_requires_answer": False,
        "_planner_has_cached_result": False,
        "_planner_metrics": metrics,
        "_planner_metrics_read": metrics_read,
        "can_request_clarification": can_request_clarification,
        "clarification_progress": clarification_progress,
        "measure_clarification_is_answerable": measure_clarification_is_answerable,
        "_save_pending_clarification": lambda q, c, meta: saved.append(meta),
        "adapter": adapter,
        "event": _Event(rounds_spent=rounds_spent),
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


def test_the_reader_is_still_asked_before_anyone_gives_up():
    """The rounds are not wasted even with an empty registry.

    A reply can resolve a ranking without naming a registered metric at all,
    by naming something countable: "total shipments by warehouse" against an
    empty catalogue resolves on counted_entity. That recovery is the reason the
    refusal waits for the loop to be spent instead of pre-empting the ask --
    asserted by executing the planner, so the reason is checked and not merely
    asserted.
    """
    from core.clarification import combine_with_clarification

    combined, _ = combine_with_clarification(
        "which warehouses have the highest total", "total shipments by warehouse")
    recovered = plan_analytical_intent(combined, metrics=())
    assert recovered.needs_clarification is False, (
        "an empty registry no longer admits a counted-entity recovery, so the "
        "refusal below could move earlier again"
    )

    adapter, saved, traces = _run_gate(
        "which warehouses have the highest total", metrics=[])
    assert [meta["slot"] for meta in saved] == ["metric"]
    assert t("clar.metric.ranking", lang="en") in adapter.messages[0]
    assert [trace["answer_type"] for trace in traces] == ["clarification"]


def test_a_spent_loop_says_what_is_missing_rather_than_asking_again():
    """With the rounds gone and no governed measure in scope, the honest reply
    names what the semantic layer lacks and who can supply it.

    terminal.needs_governed_context -- the message this path used to end with
    -- says "restate the request and specify the measure", which is the one
    thing the reader cannot usefully do here.
    """
    adapter, saved, traces = _run_gate(
        "which warehouses have the highest total", metrics=[], lang="fr",
        rounds_spent=3)

    assert adapter.messages == [
        t("terminal.analytical_plan_unresolved", lang="fr",
          missing=t("slot.measure", lang="fr"))
    ]
    assert adapter.prompts == []
    assert saved == []
    assert [trace["answer_type"] for trace in traces] == ["semantic_plan_incomplete"]


def test_a_spent_loop_on_an_answerable_slot_keeps_the_old_message():
    """The narrowness of the swap: a slot the reader COULD have filled still
    ends with the message that asks them to restate it."""
    adapter, _saved, traces = _run_gate(
        "which customers are at risk of churn by region", metrics=[],
        rounds_spent=3)

    assert adapter.messages == [
        t("terminal.needs_governed_context", lang="en", slot="business definition")
    ]
    assert [trace["answer_type"] for trace in traces] == ["clarification_limit"]


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


def test_a_subject_question_is_still_asked_with_an_empty_registry():
    """The regression this rule caused when it judged the `subject` slot too.

    The ranking slot re-asks until a reply matches the registry, so with
    nothing in it no reply resolves. The SUBJECT slot does the opposite: it
    takes the reader's words outright. Asserted by executing the planner, so
    the reason this slot is exempt is checked rather than asserted.
    """
    from core.clarification import combine_with_clarification

    # The planner's own behaviour, which is why the exemption exists.
    combined, _ = combine_with_clarification("show me my data", "pallets shipped")
    resolved = plan_analytical_intent(combined, metrics=())
    assert resolved.needs_clarification is False, (
        "a subject slot no longer resolves from free text, so the exemption "
        "below may no longer be warranted"
    )

    # ...so the gate must let the question through.
    adapter, saved, traces = _run_gate("show me my data", metrics=[])
    assert len(adapter.messages) == 1
    assert [meta["slot"] for meta in saved] == ["subject"]
    assert [trace["answer_type"] for trace in traces] == ["clarification"]


def test_an_unread_registry_keeps_asking_rather_than_refusing():
    """[] has two meanings and only one of them justifies refusing.

    store.list_metrics can raise; the pipeline catches it, fails open and
    re-plans with no metrics, leaving _planner_metrics empty. Refusing then
    would turn a deliberate fail-open into a turn-ending refusal whose trace
    asserts something untrue about the workspace.
    """
    adapter, saved, traces = _run_gate(
        "which warehouses have the highest total", metrics=[], metrics_read=False)

    assert [meta["slot"] for meta in saved] == ["metric"]
    assert [trace["answer_type"] for trace in traces] == ["clarification"]
    assert t("clar.metric.ranking", lang="en") in adapter.messages[0]


def test_the_rule_itself_judges_only_the_measure_slots():
    assert measure_clarification_is_answerable("metric", (), []) is False
    assert measure_clarification_is_answerable(
        "metric", (), [{"name": "Revenue"}]) is True
    assert measure_clarification_is_answerable(
        "metric", ({"label": "Revenue"},), []) is True
    # A registry row with no usable name is not a measure anybody can name.
    assert measure_clarification_is_answerable(
        "metric", (), [{"name": "  "}]) is False
    # Every other slot asks for the reader's own words, "subject" included --
    # it was in the measure set for one commit and that was the regression
    # test_a_subject_question_is_still_asked_with_an_empty_registry covers.
    for slot in ("subject", "business_definition", "recent_window",
                 "calendar_basis", "fiscal_year_start_month", "date_role", ""):
        assert measure_clarification_is_answerable(slot, (), []) is True


def _shipped_incomplete_plan_refusal() -> str:
    """The `status == "incomplete"` refusal, out of the shipped function."""
    source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.If)
                and ast.unparse(node.test) ==
                "_analytical_request_plan.get('status') == 'incomplete'"):
            return ast.unparse(node)
    raise AssertionError(
        "the incomplete-plan refusal is no longer recognisable inside "
        "_handle_query_impl"
    )


def _run_incomplete_refusal(missing, lang):
    """Execute that block and report what the reader was sent."""
    adapter = _RecordingAdapter()
    namespace = {
        "_analytical_request_plan": {"status": "incomplete", "missing_slots": missing},
        "_semantic_plan_question": "which warehouses have the highest total",
        "account_id": "acct",
        "adapter": adapter,
        "event": _Event(),
        "_t": t,
        "log": logging.getLogger("test.slotcopy"),
        "_trace_finish": lambda trace_id, **kw: None,
        "trace_id": "trace-1",
        "time": time,
        "start_ms": int(time.time() * 1000),
        "portal_user": {"lang": lang},
    }
    exec(compile(
        "async def _refusal():\n"
        + textwrap.indent(_shipped_incomplete_plan_refusal(), "    "),
        "<incomplete-plan-refusal>", "exec"), namespace)
    asyncio.run(namespace["_refusal"]())
    return adapter.messages


def test_the_slot_names_in_the_refusal_are_translated():
    """The refusal interpolates the slot names into a translated sentence, and
    they were English literals in a dict -- so a French reader got a French
    sentence with an English noun phrase inside it.

    The call site is EXECUTED, not just the catalogue read: checking only that
    the keys exist and differ would pass just as well with the English dict
    still in place.
    """
    sent = _run_incomplete_refusal(["measure", "date_role"], "fr")
    assert sent == [
        t("terminal.analytical_plan_unresolved", lang="fr",
          missing=", ".join([t("slot.measure", lang="fr"),
                             t("slot.date_role", lang="fr")]))
    ]
    # And nothing English survived in it.
    for english in ("the governed measure to calculate", "the business date to use"):
        assert english not in sent[0]


def test_a_slot_with_no_catalogue_label_keeps_its_own_name():
    """The fallback the dict used to provide, still provided: an unlabelled
    slot must not render as the raw id "slot.whatever"."""
    sent = _run_incomplete_refusal(["some_new_slot"], "en")
    assert "some new slot" in sent[0]
    assert "slot.some_new_slot" not in sent[0]


def test_every_slot_label_exists_in_both_languages():
    for slot in ("source_fact", "measure", "count_target", "date_role",
                 "comparison_window"):
        key = f"slot.{slot}"
        english = t(key, lang="en")
        french = t(key, lang="fr")
        assert english != key, f"{key} is missing from the catalogue"
        assert french != key
        assert french != english, f"{key} is not translated"
