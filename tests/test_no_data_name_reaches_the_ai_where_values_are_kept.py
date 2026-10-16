"""Where a workspace keeps its data's names from the AI, the chat reply and the written summary never carry one.

The AI words the chat's reply and writes an answer's summary from what the answer already says. Where member values
may not reach it (a workspace under compliance, or one with value indexing off), an answer whose words name a member
was meant to get neither -- but only some of those words were looked at:

* a single figure for one store has no name in its rows, so its summary was written from a sentence that names
  the store ("Net amount for store Riverside Store in May 2026 is ...");
* a condition kept from the question before is said by name ("I kept the condition for customer Lakeside
  Wholesale ..."), and its reply was sent with that name;
* a question read again quotes its first answer's sentence, which named a customer the question never did.

Now an answer says which names its words state (the members its conditions name), and each of these is kept from
the AI where values are. Where values may reach the AI, all of them still go as before.

Invented retail data; the AI is replaced and records what it is given.
"""

from __future__ import annotations

import os

import pytest

from tests.test_the_product_looks_before_it_answers import MAY, Chat, Told, _net, retail  # noqa: F401 - fixture

JANUARY = {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-01-31"}}


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    path = str(tmp_path / "querybot.db")
    monkeypatch.setenv("QUERYBOT_DB_PATH", path)
    monkeypatch.setenv("DB_PATH", path)
    import store
    from admin import routes

    for module in {id(m): m for m in (store, routes.store)}.values():
        for namespace in (module.crypto.__dict__, module.config_store.decrypt.__globals__):
            monkeypatch.setitem(namespace, "KEY_FILE", tmp_path / ".key")
    store.init_db()
    yield store


def _workspace(store, *, values: bool) -> str:
    """A standard workspace (not under compliance), with value indexing on or off."""
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.save_compliance_profile(account_id, mode="standard", industry="standard")
    if not values:
        state = store.get_client_state(account_id) or {}
        store.update_client_state(account_id, "READY", {**state, "value_index_enabled": False})
    return account_id


@pytest.fixture
def ai(monkeypatch):
    from core2.bootstrap import ai as bootstrap_ai

    given: list[str] = []

    def writer(account_id, client, *, question="", question_id=""):
        def complete(system: str, user: str) -> str:
            given.append(user)
            return "Nothing new to add."
        return complete

    monkeypatch.setattr(bootstrap_ai, "summary_writer", writer)
    return given


# ── what an answer's words name ───────────────────────────────────────────────


def _store(name: str) -> dict:
    return {"measures": ["net_amount"], "time": MAY, "filters": [{"field": "store.name", "op": "eq", "values": [name]}]}


def test_an_answer_says_which_names_its_words_state(retail):  # noqa: F811
    one = Chat(retail).ask('Net sales of "Riverside Store" in May 2026', _store("Riverside Store"))
    assert "Riverside Store" in one["answer"]["headline"]
    assert one["named"] == ["Riverside Store"]
    total = Chat(retail).ask("Net sales in May 2026", {"measures": ["net_amount"], "time": MAY})
    assert total["named"] == []


# ── the written summary ───────────────────────────────────────────────────────


def _one_store(retail) -> tuple[str, dict]:  # noqa: F811
    question = 'Net sales of "Riverside Store" in May 2026'
    got = Chat(retail).ask(question, _store("Riverside Store"))
    assert got["data"]["rows"][0]["net_amount"] and not got.get("reply"), "a figure, and nothing seen"
    return question, got


def test_a_summary_never_carries_a_name_its_conditions_state_where_values_are_kept(fresh_store, ai, retail):  # noqa: F811
    from core2.service import portal_summary

    question, got = _one_store(retail)
    assert portal_summary(_workspace(fresh_store, values=False), question, got) is None
    assert ai == [], "the store's name was sent to the AI"


def test_a_summary_is_written_as_before_where_values_may_reach_the_ai(fresh_store, ai, retail):  # noqa: F811
    from core2.service import portal_summary

    question, got = _one_store(retail)
    portal_summary(_workspace(fresh_store, values=True), question, got)
    assert len(ai) == 1 and "Riverside Store" in ai[0]


# ── the chat's reply ──────────────────────────────────────────────────────────


def _carried(retail) -> tuple[str, dict]:  # noqa: F811
    chat = Chat(retail)
    chat.ask('Net sales for "Lakeside Wholesale" in January 2026', {**_net(customer="Lakeside Wholesale"),
                                                                    "time": JANUARY})
    question = "and in May 2026?"
    got = chat.ask(question, {**_net(customer="Lakeside Wholesale"), "time": MAY, "follow_up": "refine"})
    assert any(s["kind"] == "carried" for s in got["reply"]["seen"])
    return question, got


def test_a_condition_kept_from_the_question_before_is_a_name_the_reply_states(retail):  # noqa: F811
    from core2.agent import talk

    _, got = _carried(retail)
    carried = next(s for s in got["reply"]["seen"] if s["kind"] == "carried")
    assert carried["members"] == ["Lakeside Wholesale"]
    assert talk.names_members(got["reply"])


def test_a_reply_never_carries_a_kept_condition_by_name_where_values_are_kept(fresh_store, ai, retail):  # noqa: F811
    from core2.service import portal_reply

    question, got = _carried(retail)
    assert portal_reply(_workspace(fresh_store, values=False), question, got) is None
    assert ai == [], "the customer kept from the question before was sent to the AI"


def test_a_question_read_again_never_sends_its_first_answer_by_name_where_values_are_kept(fresh_store, ai, retail):  # noqa: F811
    from core2.service import portal_reply

    chat = Told(retail)
    chat.plans += [{**_net(customer="Lakeside Wholesale"), "time": MAY}, {"measures": ["net_amount"], "time": MAY}]
    question = "Net sales in May 2026"
    got = chat.ask(question)
    again = got["reply"]["seen"][0]
    assert again["kind"] == "looked_again" and "Lakeside Wholesale" in again["en"]
    assert again["members"] == ["Lakeside Wholesale"]
    assert portal_reply(_workspace(fresh_store, values=False), question, got) is None
    assert ai == [], "the first answer's customer was sent to the AI"


def test_a_reply_naming_no_member_is_worded_by_the_ai_as_before(fresh_store, ai, retail):  # noqa: F811
    from core2.service import portal_reply

    question = "Net sales from October 2022 to March 2023"
    got = Chat(retail).ask(question, {"measures": ["net_amount"], "time": {"window": {
        "kind": "between", "start": "2022-10-01", "end": "2023-03-31"}}})
    assert [s["kind"] for s in got["reply"]["seen"]] == ["starts_late"] and got["named"] == []
    portal_reply(_workspace(fresh_store, values=False), question, got)
    assert len(ai) == 1
