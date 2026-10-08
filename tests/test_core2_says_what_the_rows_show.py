"""Each new-core answer says what its rows show beyond the first sentence.

The new core answered with one sentence and nothing more: "Net amount by store:
Old Town Store leads with $124,495.93" left the reader to work out whether the
stores were close, whether one carried the rest, or how a month compared with
the ones before. Today's pipeline adds a summary, but a general one.

Now up to three findings follow the sentence, each worked out from the answer's
own rows and specific to its shape, under "What stands out":

* a breakdown: how much the top 3 make of the total; a ranking from the lowest,
  how little the lowest 3 make; an average or days, its range and median (never
  a share of a total it does not have);
* a series: the last complete period against the first; a period the data only
  partly covers is left out;
* a comparison by member: how many rose and fell, and the one that moved more
  than all of them together;
* one number: against the period before, from one more query, "unchanged" when
  nothing moved, and nothing when the data starts after that period began.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core2.answer.builder import fmt
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.resolve.resolver import Context, resolve
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
TEMPLATE = Path(__file__).resolve().parents[1] / "portal" / "templates" / "portal_chat.html"


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def _ask(retail, plan: dict) -> dict:
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con),
                        complete=Recorded(json.dumps({"kind": "query", **plan})), index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _by_store(answer: dict, column: str) -> list[float]:
    return sorted((float(r[column]) for r in answer["data"]["rows"]), reverse=True)


def test_a_breakdown_says_how_much_the_top_three_make(retail):
    answer = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                           "time": {"window": H1}})
    values = _by_store(answer, "net_amount")
    share = sum(values[:3]) / sum(values)
    assert f"The top 3 of the {len(values)} stores make {share:.0%} of the total." in answer["key_insights"]


def test_a_ranking_from_the_lowest_speaks_of_the_lowest(retail):
    answer = _ask(retail, {"intent": "rank", "measures": ["net_amount"], "group_by": ["store"],
                           "time": {"window": H1}, "sort": [{"by": "net_amount", "desc": False}]})
    values = _by_store(answer, "net_amount")
    share = sum(values[-3:]) / sum(values)
    assert f"The 3 lowest of the {len(values)} stores make {share:.0%} of the total." in answer["key_insights"]
    assert not any("top 3" in k or "times the next" in k for k in answer["key_insights"]), answer["key_insights"]


def test_an_average_gives_its_range_never_a_share_of_a_total(retail):
    answer = _ask(retail, {"intent": "breakdown", "group_by": ["store"], "time": {"window": H1},
                           "durations": [{"name": "Days to ship", "start": "order_date", "end": "ship_date"}]})
    rows = answer["data"]["rows"]
    low = min(rows, key=lambda r: r["days_to_ship"])
    high = max(rows, key=lambda r: r["days_to_ship"])
    said = " ".join(answer["key_insights"])
    assert f"From {fmt(low['days_to_ship'], 'days')} ({low['store_name']}) to " \
           f"{fmt(high['days_to_ship'], 'days')} ({high['store_name']}); half the stores are above " in said, said
    assert "of the total" not in said and "apart" not in said


def test_a_series_says_how_its_last_complete_period_compares_with_its_first(retail):
    answer = _ask(retail, {"intent": "trend", "measures": ["net_amount"], "time": {
        "grain": "month", "window": {"kind": "between", "start": "2025-01-01", "end": "2026-06-30"}}})
    rows = {r["period"][:7]: float(r["net_amount"]) for r in answer["data"]["rows"]}
    change = rows["2026-05"] / rows["2025-01"] - 1
    assert f"May 2026 is {abs(change):.0%} {'above' if change > 0 else 'below'} Jan 2025." in answer["key_insights"]
    assert not any("Jun 2026" in k for k in answer["key_insights"]), "June 2026 is only partly in the data"


def test_a_comparison_by_member_says_how_many_rose_and_who_outweighed_the_rest(retail):
    answer = _ask(retail, {"intent": "compare", "measures": ["net_amount"], "group_by": ["store"],
                           "time": {"window": APRIL, "compare": {"kind": "previous_period"}}})
    changes = {r["store_name"]: float(r["net_amount_change"]) for r in answer["data"]["rows"]}
    up, down = sum(1 for c in changes.values() if c > 0), sum(1 for c in changes.values() if c < 0)
    assert f"{up} of the {len(changes)} stores rose and {down} fell." in answer["key_insights"]
    net = sum(changes.values())
    most = min(changes, key=lambda s: changes[s]) if net < 0 else max(changes, key=lambda s: changes[s])
    assert abs(changes[most]) > abs(net), "the synthetic April has one store outweighing the rest"
    assert (f"{most} {'fell' if net < 0 else 'rose'} {fmt(abs(changes[most]), 'currency')}, more than the net "
            f"{'fall' if net < 0 else 'rise'} of {fmt(abs(net), 'currency')}: the other stores "
            f"{'rose' if net < 0 else 'fell'} {fmt(abs(net - changes[most]), 'currency')} together."
            ) in answer["key_insights"]


def test_one_number_is_set_against_the_period_before(retail):
    april = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {"window": APRIL}})
    march = _ask(retail, {"intent": "value", "measures": ["net_amount"], "time": {
        "window": {"kind": "between", "start": "2026-03-01", "end": "2026-03-31"}}})
    now, before = float(april["data"]["rows"][0]["net_amount"]), float(march["data"]["rows"][0]["net_amount"])
    moved = f"{'Up' if now >= before else 'Down'} {fmt(abs(now - before), 'currency')} " \
            f"({(now - before) / before * 100:+.1f}%) on March 2026, the period before."
    assert april["key_insights"] == [moved]


def test_nothing_moved_reads_unchanged(retail):
    stores = _ask(retail, {"intent": "value", "measures": ["number_of_stores"],
                           "time": {"date": "order_date", "window": APRIL}})
    assert stores["key_insights"] == ["Unchanged on March 2026, the period before."]
    compared = _ask(retail, {"intent": "compare", "measures": ["number_of_stores"], "time": {
        "date": "order_date", "window": APRIL, "compare": {"kind": "previous_period"}}})
    assert compared["answer"]["headline"].endswith(": 12, unchanged on March 2026."), compared["answer"]
    assert compared["answer"]["comparison"] == "unchanged on March 2026"


def test_no_period_before_when_the_data_starts_after_it_began(retail):
    con, model = retail
    logical = resolve(Plan.model_validate({"kind": "query", "intent": "value", "measures": ["number_of_order_lines"],
                                           "time": {"window": APRIL}}), model, Context(today=TODAY))
    first = logical.parts[0].date.role.first
    month = {"kind": "between", "start": str(first.replace(day=1)), "end": str(first.replace(day=28))}
    answer = _ask(retail, {"intent": "value", "measures": ["number_of_order_lines"], "time": {"window": month}})
    assert answer["data"]["rows"][0]["number_of_order_lines"] > 0
    assert answer["key_insights"] == [], "a count of 0 before the data starts is not a period with none in it"


def test_no_period_before_for_a_series_or_a_breakdown(retail):
    series = _ask(retail, {"intent": "value", "measures": ["number_of_order_lines"],
                           "time": {"grain": "month", "window": APRIL}})
    breakdown = _ask(retail, {"intent": "breakdown", "measures": ["number_of_order_lines"], "group_by": ["store"],
                              "time": {"window": APRIL}})
    assert breakdown["key_insights"], "a breakdown has findings of its own"
    assert not any("the period before" in k for k in series["key_insights"] + breakdown["key_insights"])


def test_findings_that_fail_cost_only_the_findings(retail, monkeypatch, caplog):
    from core2.answer import insights

    def broken(*args, **kwargs):
        raise ValueError("a value the findings cannot read")

    monkeypatch.setattr(insights, "summarize", broken)
    answer = _ask(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"],
                           "time": {"window": H1}})
    assert answer["answer"]["headline"].startswith("Net amount from Jan 2026 to Jun 2026: ")
    assert answer["key_insights"] == []
    assert "could not work out findings" in caplog.text


# ── the portal ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(shutil.which("node") is None, reason="node runs the portal's own script")
def test_the_portal_shows_each_finding_under_what_stands_out():
    pytest.importorskip("dukpy")
    from tests.test_the_answer_card import card

    findings = ["The top 3 of the 12 stores make 33% of the total.", "A <b> & B are close."]
    shown = card({"engine": "core2", "answer": {"headline": "Net amount by store."}, "key_insights": findings,
                  "coverage_caveats": ["Rows with no store are left out."], "trust": {"engine": "core2"}})
    assert shown.index("Rows with no store") < shown.index("What stands out") < shown.index("The top 3")
    assert shown.count('class="answer-note note"') == 2
    assert "A &lt;b&gt; &amp; B are close." in shown and "<b>" not in shown, "a finding is text, never markup"
    bare = card({"engine": "core2", "answer": {"headline": "Net amount by store."}, "key_insights": [],
                 "trust": {"engine": "core2"}})
    assert "What stands out" not in bare, "no findings, no empty heading"
