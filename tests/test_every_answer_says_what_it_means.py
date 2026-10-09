"""Every answer says what it means: findings for every shape, a short written summary, a "why" that holds up.

From the reader's test of the new core:

* Some answers came with no summary at all: a line per member ("by region by month") and a
  listing had no findings. They do now: who led and who rose and fell most; how a list falls
  into a small grouping and the range of a figure in it.
* Every answer with rows gets a short summary, written by the workspace's AI from the answer
  itself and sent after it. Every figure in it must be one the answer holds; one that is not
  stops the summary. Where the workspace keeps member values from the AI (a tenant under
  compliance, or value indexing off), an answer naming members gets none; one of numbers alone
  still does, with the question's personal data scrubbed.
* "Why did month-end stock change in Aug 2026?" compared August with July, which had no
  snapshot ("rose from 0"), added 14 units together, and offered 0/1 flags as what moved it. It
  now compares with the last month that has data and says so, counts in the main unit and
  says which, and leaves yes/no flags out of the groupings it checks.

Invented retail data; the AI is a stand-in that records what it was given.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core2.answer import summary
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
H1 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
APRIL = {"kind": "between", "start": "2026-04-01", "end": "2026-04-30"}
MAY = {"kind": "between", "start": "2026-05-01", "end": "2026-05-31"}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


@pytest.fixture(scope="module")
def no_april():
    built, model = learn(domains.build("retail"), "descriptive")
    built.con.execute("DELETE FROM main.order_lines WHERE order_date_key BETWEEN 20260401 AND 20260430")
    return built.con, model


def _answer(warehouse, plan: dict) -> dict:
    con, model = warehouse
    answer = json.dumps({"kind": "query", **plan})
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=lambda s, t: answer,
                        index=MemberIndex(), today=TODAY)
    return answer_question("q", services, Session())


def _notes(payload: dict) -> list[str]:
    return list((payload.get("trust") or {}).get("date_context") or [])


# ── findings for every shape ────────────────────────────────────────────────


def test_a_line_per_member_says_who_rose_and_who_fell_most(retail):
    payload = _answer(retail, {"intent": "trend", "measures": ["net_amount"], "group_by": ["region.name"],
                               "time": {"grain": "month", "window": H1}})
    found = " ".join(payload["key_insights"])
    assert "rose most" in found and "fell most" in found, found
    assert "Jun 2026" not in found, "a partly covered month is not a finding's end point"


def test_a_list_says_how_it_falls_into_a_small_grouping(retail):
    payload = _answer(retail, {"intent": "list", "group_by": ["store.name", "region.name"]})
    assert payload["key_insights"] == ["By region: North (3), South (3), Central (2) and 2 more."]


def test_a_list_of_names_alone_has_nothing_more_to_say(retail):
    assert _answer(retail, {"intent": "list", "group_by": ["store"]})["key_insights"] == []


# ── why ─────────────────────────────────────────────────────────────────────


def test_a_why_about_a_quantity_counts_its_main_unit_and_says_so(retail):
    payload = _answer(retail, {"intent": "drivers", "measures": ["quantity"], "time": {"window": APRIL}})
    headline = payload["answer"]["headline"]
    assert headline.startswith("Quantity fell 422 EA"), headline
    assert "unit of measure EA" not in headline, "the unit is said with the amounts, not again as a condition"
    assert any(n.startswith("Counted in EA, the unit of most rows") for n in _notes(payload))
    assert all(r["member"] != "EA" for r in payload["data"]["rows"]), "units are not what moved it"


def test_a_why_after_a_month_with_no_data_compares_with_the_last_month_that_has(no_april):
    payload = _answer(no_april, {"intent": "drivers", "measures": ["net_amount"], "time": {"window": MAY}})
    assert "from $194,876.48 in March 2026 to $180,136.22 in May 2026" in payload["answer"]["headline"]
    assert "Apr 2026 has no data: compared with Mar 2026, the last month before it that has." in _notes(payload)


def test_a_why_after_a_month_with_data_compares_with_it_as_before(retail):
    payload = _answer(retail, {"intent": "drivers", "measures": ["net_amount"], "time": {"window": MAY}})
    assert "in April 2026 to" in payload["answer"]["headline"]
    assert not any("has no data" in n for n in _notes(payload))


def _last_with_data(warehouse, window: dict):
    from core2.answer.drivers import _last_with_data as last, base_plan
    from core2.plan.ir import Plan
    from core2.resolve.resolver import Context, resolve

    con, model = warehouse
    ctx = Context(today=TODAY)
    base, _ = base_plan(Plan.model_validate({"intent": "drivers", "measures": ["net_amount"],
                                             "time": {"window": window}}), model, ctx)
    return last(base, resolve(base, model, ctx), model, ctx, DuckDBWarehouse(con))


def test_only_a_month_with_no_row_at_all_is_passed_over(retail, no_april):
    assert _last_with_data(retail, MAY) is None, "April has rows: it is the month compared with"
    window, name, unit = _last_with_data(no_april, MAY)
    assert (window.start, window.end, name, unit) == (dt.date(2026, 3, 1), dt.date(2026, 3, 31), "Mar 2026", "month")


def test_a_yes_no_flag_is_not_a_grouping_a_why_checks():
    from core2.answer.drivers import _a_flag

    def column(*values):
        return SimpleNamespace(profile=SimpleNamespace(top=[SimpleNamespace(value=v, count=1) for v in values]))

    assert _a_flag(column("0", "1")) and _a_flag(column("Y", "N", None)) and _a_flag(column("true", "false"))
    assert not _a_flag(column("Stocked", "Not stocked")) and not _a_flag(column("1", "2", "3"))
    assert not _a_flag(SimpleNamespace(profile=None))


# ── the written summary ─────────────────────────────────────────────────────


BY_STORE = {"type": "assistant_response", "engine": "core2", "question": "q",
            "answer": {"headline": "Old Town Store leads with $1,234,567.00 (12% of the total) across 12 stores."},
            "key_insights": ["The top 3 of the 12 stores make 33% of the total."], "coverage_caveats": [],
            "data": {"headers": ["store_name", "net_amount"], "header_labels": {"store_name": "Store",
                                                                                "net_amount": "Net amount"},
                     "rows": [{"store_name": "Old Town Store", "net_amount": 1234567.0},
                              {"store_name": "Riverside", "net_amount": 1113593.63}],
                     "total_rows": 12, "column_formats": {"store_name": "text", "net_amount": "currency"}}}
MONTHLY = {"engine": "core2", "question": "q", "answer": {"headline": "Net amount by month: $167,980.77 in Jan."},
           "data": {"headers": ["period", "net_amount"], "rows": [{"period": "2026-01-01", "net_amount": 167980.77}],
                    "column_formats": {"period": "text", "net_amount": "currency"},
                    "display_formats": {"period": {"type": "date", "style": "month_year_long"}}}}


def test_an_answer_naming_members_gets_a_summary_only_where_their_values_may_reach_the_ai():
    assert summary.eligible(BY_STORE, values_allowed=True)
    assert not summary.eligible(BY_STORE, values_allowed=False)
    assert summary.eligible(MONTHLY, values_allowed=False), "numbers and periods alone may"
    assert not summary.eligible({**MONTHLY, "data": {"rows": []}}, values_allowed=True)
    assert not summary.eligible({**MONTHLY, "engine": "legacy"}, values_allowed=True)


def test_a_summary_holds_only_figures_the_answer_holds():
    given = summary.material(BY_STORE, "net sales by store")
    ok = "Old Town Store leads at about $1.2M; the top 3 make 33% of the total, so watch the leaders."
    assert summary.checked(ok, BY_STORE, given) == ok
    assert summary.checked("Riverside made $1,113,594 and 2 stores lead.", BY_STORE, given)
    invented = "Old Town Store grew 27% on last year to $1.2M."
    assert summary.checked(invented, BY_STORE, given) is None
    assert summary.checked("Sales were $5.9M across the stores.", BY_STORE, given) is None


def test_a_summary_is_plain_sentences_of_a_reasonable_length():
    given = summary.material(BY_STORE, "q")
    assert summary.checked("", BY_STORE, given) is None
    assert summary.checked("- a list item", BY_STORE, given) is None
    assert summary.checked(" ".join(["word"] * 120), BY_STORE, given) is None


def test_the_ai_is_given_the_answer_as_shown_and_no_more_than_25_rows():
    many = {**BY_STORE, "data": {**BY_STORE["data"],
                                 "rows": [{"store_name": f"Store {i}", "net_amount": float(i)} for i in range(40)],
                                 "total_rows": 40}}
    given = summary.material(many, "net sales by store")
    assert "Question: net sales by store" in given and BY_STORE["answer"]["headline"] in given
    assert "The top 3 of the 12 stores make 33% of the total." in given
    assert "Store 24" in given and "Store 25" not in given and "15 more not shown" in given


def test_an_ai_that_fails_leaves_the_answer_without_a_summary():
    def broken(system, text):
        raise RuntimeError("the AI is down")

    assert summary.write(BY_STORE, "q", broken) is None


def _summarised(account: str, payload: dict, *, scrub) -> tuple[str | None, list[str]]:
    import core2.service as service

    asked: list[str] = []

    def planner(account_id, client, *, question="", question_id=""):
        def complete(system, text):
            asked.append(text)
            return "The leaders make most of it; watch them."
        return complete

    with patch.object(service, "question_scrubber", return_value=scrub), \
            patch("core2.bootstrap.ai.workspace_planner", planner), \
            patch("core.value_index.value_index_enabled", return_value=True):
        said = service.portal_summary(account, "net sales for Jane Doe by store", payload, question_id="c2-x")
    return said, asked


def test_under_compliance_an_answer_naming_members_never_reaches_the_ai():
    said, asked = _summarised("acct-none", BY_STORE, scrub=lambda text: text.replace("Jane Doe", "[name]"))
    assert said is None and asked == []


def test_under_compliance_an_answer_of_numbers_alone_is_summarised_with_the_question_scrubbed():
    said, asked = _summarised("acct-none", MONTHLY, scrub=lambda text: text.replace("Jane Doe", "[name]"))
    assert said == "The leaders make most of it; watch them."
    assert "[name]" in asked[0] and "Jane Doe" not in asked[0]


def test_without_compliance_every_answer_with_rows_is_summarised():
    said, asked = _summarised("acct-none", BY_STORE, scrub=None)
    assert said and "Old Town Store" in asked[0]


# ── it follows the answer ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tenant(tmp_path_factory):
    from tests import answer_harness as harness

    with harness.tenant_in(tmp_path_factory.mktemp("summaries")) as built:
        yield built


def test_the_summary_follows_the_answer_and_stays_with_it(tenant):
    import gateway.core2_bridge as bridge
    import store
    from tests import answer_harness as harness
    from tests import portal_harness as portal

    async def setting(account_id):
        return "core2"

    def answer(account_id, question, *args, **kwargs):
        payload = json.loads(json.dumps(BY_STORE))
        payload.update(question=question, trust={"engine": "core2", "sql": "SELECT 1 AS v", "row_count": 2,
                                                 "question_id": kwargs["question_id"]})
        payload["export_rows"] = payload["data"]["rows"]
        return payload

    told = "Old Town Store leads; the top 3 make 33% of the total."
    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", answer), \
            patch("core2.service.portal_summary", return_value=told) as summarise:
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(),
                                       idle_seconds=2.0) as conversation:
            frames = conversation.ask("net sales by store")["frames"]
    kinds = [f.get("type") for f in frames]
    assert "answer_summary" in kinds and kinds.index("answer_summary") > kinds.index("assistant_response")
    shown = next(f for f in frames if f.get("type") == "assistant_response")
    said = next(f for f in frames if f.get("type") == "answer_summary")
    assert said == {"type": "answer_summary", "question_id": shown["trust"]["question_id"], "summary": told}
    assert summarise.call_args.kwargs["question_id"] == shown["trust"]["question_id"]
    kept = json.loads(store.get_answer_trace(int(shown["trace_id"]))["answer_frame"])
    assert kept["summary"] == told                     # a reopened thread shows it too


def test_no_summary_is_sent_when_there_is_none_to_show(tenant):
    import gateway.core2_bridge as bridge
    from tests import answer_harness as harness
    from tests import portal_harness as portal

    async def setting(account_id):
        return "core2"

    def answer(account_id, question, *args, **kwargs):
        payload = json.loads(json.dumps(BY_STORE))
        payload["trust"] = {"engine": "core2", "sql": "SELECT 1 AS v", "row_count": 2, "question_id": kwargs["question_id"]}
        return payload

    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", answer), \
            patch("core2.service.portal_summary", return_value=None):
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(),
                                       idle_seconds=2.0) as conversation:
            frames = conversation.ask("net sales by store please")["frames"]
    assert "answer_summary" not in [f.get("type") for f in frames]


# ── the page ────────────────────────────────────────────────────────────────


def test_the_card_shows_its_summary_under_the_chart_and_before_the_findings():
    from tests.test_the_answer_card import Elements, card

    msg = {**BY_STORE, "summary": "The leaders make most of it.",
           "chart": {"title": "Net amount", "chart_type": "bar", "x_key": "store_name", "y_keys": ["net_amount"],
                     "rows": BY_STORE["data"]["rows"], "renderable_types": ["bar"]}}
    shown = Elements(card(msg))
    assert shown.first("answer-summary")["text"].endswith("The leaders make most of it.")
    assert shown.order("answer-visual", "answer-summary", "answer-insights") == sorted(
        shown.order("answer-visual", "answer-summary", "answer-insights"))
    assert Elements(card(BY_STORE)).first("answer-summary") is None
