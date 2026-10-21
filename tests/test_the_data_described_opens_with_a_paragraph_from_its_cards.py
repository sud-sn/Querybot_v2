"""An answer about the data itself opens with a paragraph in the workspace's AI's words.

"What data do you have?" is answered from the model, as an analyst would go through it: a lead, a card per table
(what one row is, its figures, what it links to, what to know before counting it), things to know, and questions to
try (core2/answer/describe.py). The workspace's AI then opens it with two to four sentences written from the cards
alone (core2/answer/overview.py), after the answer is shown:

* a paragraph with a figure the cards do not hold is not shown;
* where member values are kept from the AI (a tenant under compliance, or value indexing off) and for a workspace
  whose answers may not reach the AI, there is none, and the AI is never called;
* the AI egress log says the data's spans and values were sent;
* it is written in the reader's language, and sits under the lead, above the cards, live and when reopened.

Synthetic data only.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from unittest.mock import patch

import pytest

from core2.answer import overview
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built, model


class Recorded:
    def __init__(self, *answers: str):
        self.answers = list(answers)

    def __call__(self, stable: str, tail: str) -> str:
        return self.answers.pop(0)


def _described(retail) -> dict:
    built, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=Recorded('{"kind": "describe_data"}'),
                        index=MemberIndex(), today=TODAY)
    return answer_question("what data do you have?", services, Session(), question_id="c2-about")


def test_the_answer_about_the_data_is_marked_as_such_and_carries_its_question(retail):
    said = _described(retail)
    assert said["kind"] == "describe_data" and said["trust"]["question_id"] == "c2-about"
    assert said["answer"]["headline"].startswith("This data has 8 tables")


def test_the_ai_is_given_the_cards_as_shown_and_a_paragraph_holding_their_figures_is_shown(retail):
    said = _described(retail)
    given = overview.material(said, "what data do you have?")
    assert given.startswith("THE DATA AS DESCRIBED")
    assert "Order line (Activity)" in given and "One row per order line (32,451 rows)." in given
    assert "- Worth knowing: 73 order lines name a product the product list does not hold." in given
    sent: list[tuple[str, str]] = []

    def complete(system, text):
        sent.append((system, text))
        return ("Most questions start from the order lines, which name the customer, product and store of each "
                "sale; the returns and sales targets sit beside them. Check the 73 lines with no product on the "
                "list before you count by product.")

    shown = overview.write(said, "what data do you have?", complete)
    assert shown and shown.startswith("Most questions start from the order lines")
    assert sent[0][1] == given and "plain business English" in sent[0][0]


def test_a_paragraph_with_a_figure_the_cards_do_not_hold_is_not_shown(retail):
    said = _described(retail)
    invented = ("The order lines hold about 50,000 sales across 12 stores, so start there.")
    assert overview.write(said, "what data do you have?", lambda s, t: invented) is None
    assert overview.write(said, "q", lambda s, t: "## Overview\n- order lines") is None      # no headings or lists

    def broken(system, text):
        raise RuntimeError("the AI is down")

    assert overview.write(said, "q", broken) is None


def test_it_is_written_in_the_readers_language(retail):
    said = _described(retail)
    systems: list[str] = []

    def complete(system, text):
        systems.append(system)
        return "Les lignes de commande sont le point de départ : elles nomment le client, le produit et le magasin."

    assert overview.write(said, "quelles données avez-vous ?", complete, lang="fr")
    assert "plain business French" in systems[0]


def _standard_workspace() -> str:
    import store

    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode="standard", industry="standard")
    return account_id


def _opened(account: str, payload: dict, *, scrub, indexed: bool = True) -> tuple[str | None, int]:
    import core2.service as service

    asked: list[str] = []

    def writer(account_id, client, *, question="", question_id=""):
        def complete(system, text):
            asked.append(text)
            return "Start from the order lines; they name the customer, product and store of each sale."
        return complete

    with patch.object(service, "question_scrubber", return_value=scrub), \
            patch("core2.bootstrap.ai.summary_writer", writer), \
            patch("core.value_index.value_index_enabled", return_value=indexed):
        said = service.portal_overview(account, "what data do you have?", payload, question_id="c2-about")
    return said, len(asked)


def test_the_paragraph_follows_what_may_reach_the_workspaces_ai(retail):
    import store

    store.init_db()
    payload = _described(retail)
    standard = _standard_workspace()
    said, calls = _opened(standard, payload, scrub=None)
    assert said and calls == 1
    # Member values kept from the AI: under compliance, or value indexing off. The AI is never called.
    assert _opened(standard, payload, scrub=lambda q: q) == (None, 0)
    assert _opened(standard, payload, scrub=None, indexed=False) == (None, 0)
    # A workspace never set up for compliance counts as regulated: its answers never reach the AI.
    unset = f"acct{os.urandom(4).hex()}"
    store.upsert_client(unset, "Other Ltd")
    assert _opened(unset, payload, scrub=None) == (None, 0)
    # Only an answer about the data gets one.
    assert _opened(standard, {**payload, "kind": "smalltalk"}, scrub=None) == (None, 0)


def test_the_egress_log_says_the_datas_spans_and_values_were_sent(retail):
    from core.llm_audit import build_egress_manifest

    said = _described(retail)
    manifest = build_egress_manifest(overview.SYSTEM, overview.material(said, "what data do you have?"))
    assert manifest["values_sent"] and "data_description" in manifest["value_sources"], manifest


@pytest.fixture(scope="module")
def tenant(tmp_path_factory):
    from tests import answer_harness as harness

    with harness.tenant_in(tmp_path_factory.mktemp("overview")) as built:
        yield built


def test_the_paragraph_follows_the_answer_in_the_chat_and_stays_with_it(tenant, retail):
    import gateway.core2_bridge as bridge
    import store
    from tests import answer_harness as harness
    from tests import portal_harness as portal

    described = _described(retail)

    async def setting(account_id):
        return "core2"

    def answer(account_id, question, *args, **kwargs):
        payload = json.loads(json.dumps(described))
        payload["trust"] = {**payload["trust"], "question_id": kwargs["question_id"]}
        return payload

    told = "Start from the order lines; they name the customer, product and store of each sale."
    with patch.object(bridge, "engine", setting), patch("core2.service.portal_answer", answer), \
            patch("core2.service.portal_overview", return_value=told) as opening, \
            patch("core2.service.portal_summary") as summary:
        with portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(),
                                       idle_seconds=2.0) as conversation:
            frames = conversation.ask("what data do you have?")["frames"]
    kinds = [f.get("type") for f in frames]
    assert "answer_summary" in kinds and kinds.index("answer_summary") > kinds.index("assistant_response")
    shown = next(f for f in frames if f.get("type") == "assistant_response")
    said = next(f for f in frames if f.get("type") == "answer_summary")
    assert said == {"type": "answer_summary", "question_id": shown["trust"]["question_id"], "summary": told}
    assert opening.call_args.kwargs["question_id"] == shown["trust"]["question_id"]
    assert not summary.called
    kept = json.loads(store.get_answer_trace(int(shown["trace_id"]))["answer_frame"])
    assert kept["summary"] == told                     # a reopened thread shows it too


def test_the_card_shows_the_paragraph_under_the_lead_and_above_the_tables(retail):
    from tests.test_the_answer_card import Elements, card

    msg = {**_described(retail), "summary": "Start from the order lines."}
    shown = Elements(card(msg))
    assert shown.first("answer-summary")["text"].endswith("Start from the order lines.")
    order = shown.order("answer-headline", "answer-summary", "analysis-section")
    assert order == sorted(order), order
