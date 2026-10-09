"""The follow-up-or-new conversations are labelled so they can be scored, and the scoring is right.

Whether the new core reads a turn as a change to the answer on screen or as a new question is
measured on hand-labelled conversations about every synthetic warehouse (evals/core2/followup_eval.py).
These checks hold the set to what it must be to measure that at all: enough turns, every label one
the scorer knows, both readings well represented (a planner that always says "follow-up" must not
pass), and a few turns that are honestly unclear. The scoring is then checked on plans made here,
and end to end through the question path with a scripted AI in place of a provider.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import pytest
import yaml

from core2.plan.ir import Clarify, Filter, Plan, TimeSpec, Window
from evals.core2 import domains, followup_eval
from evals.core2.plan_eval import Recorder

FILES = followup_eval.available()


def _turns():
    for name in FILES:
        for thread in followup_eval.conversations(name)["threads"]:
            for n, turn in enumerate(thread["turns"]):
                yield name, thread["id"], n, turn


def test_every_business_type_has_conversations():
    assert set(FILES) == set(domains.DOMAINS) | set(domains.BENCHMARK)
    for name in FILES:
        spec = followup_eval.conversations(name)
        assert spec["domain"] == name
        dt.date.fromisoformat(str(spec["today"]))
        ids = [t["id"] for t in spec["threads"]]
        assert len(ids) == len(set(ids)), name


def test_every_turn_is_labelled_as_the_scorer_reads_it():
    for name, thread, n, turn in _turns():
        where = f"{name} {thread}.{n}"
        assert turn["q"].strip(), where
        if n == 0:
            assert "label" not in turn, f"{where}: a conversation's first question has nothing to follow"
            continue
        assert turn["label"] in followup_eval.LABELS, where
        assert turn["kind"] in followup_eval.KINDS[turn["label"]], where
        assert set(turn) <= {"q", "label", "kind", "keeps", "drops"}, where
        assert set(turn.get("keeps", [])) <= set(followup_eval.PARTS), where
        assert set(turn.get("drops", [])) <= set(followup_eval.PARTS), where
        assert "keeps" not in turn or turn["label"] == "refine", where
        assert "drops" not in turn or turn["label"] == "new", where


def test_the_set_is_big_and_balanced_enough_to_measure_both_readings():
    labels = Counter(turn["label"] for _, _, n, turn in _turns() if n)
    total = sum(labels.values())
    assert total >= 300
    # neither "always a follow-up" nor "always new" gets near the 90% target
    assert labels["refine"] / total < 0.75 and labels["new"] / total >= 0.2
    # honestly unclear turns exist, and asking about all of them stays within the 10% that may be asked
    assert 0.03 <= labels["unclear"] / total <= 0.10
    kinds = Counter(f"{turn['label']}:{turn['kind']}" for _, _, n, turn in _turns() if n)
    for label, names in followup_eval.KINDS.items():
        for kind in names:
            assert kinds[f"{label}:{kind}"] >= 1, f"no turn of kind {label}:{kind}"


# ── the scoring, on plans made here ─────────────────────────────────────────

LAST_MONTH = TimeSpec(window=Window(kind="previous", unit="month"))
NORTH = Filter(field="region.region_name", op="in", values=["North"])


def _plan(**kw) -> Plan:
    base = {"intent": "breakdown", "measures": ["net_sales"], "group_by": ["region.region_name"],
            "time": LAST_MONTH}
    return Plan(**{**base, **kw})


BEFORE = _plan(filters=[NORTH])


@pytest.mark.parametrize("turn,plan,asked,status,detail", [
    ({"label": "refine", "keeps": ["measures", "filters", "window"]},
     _plan(filters=[NORTH], group_by=["store.store_name"], follow_up="refine"), False, "right", ""),
    ({"label": "refine", "keeps": ["filters"]},
     _plan(group_by=["store.store_name"], follow_up="refine"), False, "wrong", "dropped the answer's filters"),
    ({"label": "refine", "keeps": ["window"]},
     _plan(filters=[NORTH], time=TimeSpec(), follow_up="refine"), False, "wrong", "dropped the answer's window"),
    ({"label": "refine"}, _plan(filters=[NORTH]), False, "wrong", "read as a new question"),
    ({"label": "refine"}, Plan(kind="describe_data"), False, "wrong", "answered as describe_data"),
    ({"label": "refine"}, None, True, "asked", "asked"),
    ({"label": "new", "drops": ["filters"]}, _plan(measures=["refunds"]), False, "right", ""),
    ({"label": "new", "drops": ["filters"]}, _plan(measures=["refunds"], filters=[NORTH]), False, "wrong",
     "carried the last answer's filters"),
    ({"label": "new"}, _plan(measures=["refunds"], follow_up="refine"), False, "wrong", "read as a change"),
    ({"label": "new"}, Plan(kind="describe_data"), False, "right", ""),
    ({"label": "unclear"}, Plan(kind="clarify", clarify=Clarify(about="other", question="Which?")), True,
     "right", "asked"),
    ({"label": "unclear"}, _plan(measures=["orders"], filters=[NORTH], follow_up="refine"), False, "guessed",
     "answered as refine"),
])
def test_a_turn_is_scored_by_its_label(turn, plan, asked, status, detail):
    got, why = followup_eval.judge(turn, plan, asked=asked, previous=BEFORE)
    assert got == status and detail in why, (got, why)


def test_a_part_the_answer_never_had_cannot_be_carried():
    """A new question that names last month itself has not carried "last month" when the answer before had no window."""
    turn = {"label": "new", "drops": ["window"]}
    assert followup_eval.judge(turn, _plan(), asked=False, previous=_plan(time=TimeSpec()))[0] == "right"


def test_the_summary_counts_each_status():
    results = [{"domain": "retail", "label": "refine", "kind": "filter", "status": s}
               for s in ("right", "right", "right", "asked")] + \
              [{"domain": "retail", "label": "unclear", "kind": "bare", "status": "guessed"}]
    summary = followup_eval.summarize(results)
    assert summary["overall"] == {"turns": 5, "right": 0.6, "asked": 0.2, "guessed": 0.2, "wrong": 0.0, "error": 0.0}
    assert summary["by_label"]["refine"]["right"] == 0.75


# ── end to end, with a scripted AI ──────────────────────────────────────────


@pytest.fixture(scope="module")
def retail_words():
    from evals.core2.compile_eval import learn

    _, model = learn(domains.build("retail"), "descriptive")
    assert "net_amount" in {m.slug for m in model.measures.values()}
    region = next(slug for slug in model.attributes if slug.startswith("region."))
    return "net_amount", region


def _conversation(tmp_path, monkeypatch):
    (tmp_path / "retail.yaml").write_text(yaml.safe_dump({"domain": "retail", "today": "2026-06-15", "threads": [
        {"id": "t1", "turns": [
            {"q": "Net sales by region last month"},
            {"q": "only North", "label": "refine", "kind": "filter", "keeps": ["measures", "window"]},
            {"q": "How many orders?", "label": "unclear", "kind": "bare"},
            {"q": "What data do you have about customers?", "label": "new", "kind": "describe"},
            {"q": "only Zzyzx", "label": "refine", "kind": "filter", "keeps": ["measures", "window"]},
        ]}]}))
    monkeypatch.setattr(followup_eval, "CONVERSATIONS", tmp_path)


def _replies(measure, region, *, refine_window):
    first = {"kind": "query", "intent": "breakdown", "measures": [measure], "group_by": [region],
             "time": {"window": {"kind": "previous", "unit": "month"}}}
    refine = {**first, "filters": [{"field": region, "op": "in", "values": ["North"]}], "follow_up": "refine"}
    if not refine_window:
        refine["time"] = {}
    ask = {"kind": "clarify", "clarify": {"about": "other", "question": "All orders, or North last month?",
                                          "options": ["All orders", "North, last month"]}}
    describe = {"kind": "describe_data", "about": ["customers"]}
    # a member the data does not have: refused after planning, yet read as the follow-up it is
    unknown = {**first, "filters": [{"field": region, "op": "in", "values": ["Zzyzx"]}], "follow_up": "refine"}
    return {"retail:t1:0": [json.dumps(first)], "retail:t1:1": [json.dumps(refine)], "retail:t1:2": [json.dumps(ask)],
            "retail:t1:3": [json.dumps(describe)], "retail:t1:4": [json.dumps(unknown)]}


def test_a_conversation_runs_through_the_question_path_and_is_scored(tmp_path, monkeypatch, retail_words):
    _conversation(tmp_path, monkeypatch)
    results = followup_eval.evaluate("retail", Recorder(None, _replies(*retail_words, refine_window=True)))
    assert [(r["turn"], r["status"]) for r in results] == [(1, "right"), (2, "right"), (3, "right"),
                                                            (4, "right")], results


def test_a_follow_up_that_loses_the_period_is_scored_wrong(tmp_path, monkeypatch, retail_words):
    _conversation(tmp_path, monkeypatch)
    results = followup_eval.evaluate("retail", Recorder(None, _replies(*retail_words, refine_window=False)))
    assert results[0]["status"] == "wrong" and "window" in results[0]["detail"], results[0]
