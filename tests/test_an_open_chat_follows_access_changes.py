"""An open chat follows the person's access as it is now, not as it was when the chat opened.

The chat socket read the person once, when it opened. Deactivating them, resetting their password, or moving
them to another group or role reached nothing already open: the chat kept answering with the old access until
the browser tab was closed. Each message now reads the person again. A person who may no longer sign in is told
to sign in again and the chat closes, with the question unanswered; a new group or role applies from the next
question on.

Through the real portal socket; invented data only.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from tests import answer_harness as harness
from tests import portal_harness as portal
from tests.test_core2_answers_in_the_portal import CANNED, _answers, _NewCore
from tests.test_core2_answers_in_the_portal import tenant  # noqa: F401 - the fixture

QUESTION = "stock on hand by warehouse"
ENDED = {"en": "Your access to this workspace has changed. Sign in again to continue.",
         "fr": "Votre accès à cet espace de travail a changé. Reconnectez-vous pour continuer."}


class _Reader(_NewCore):
    """The new core, answering every question, and the person each question was asked as."""

    def __init__(self):
        super().__init__({})
        self.as_whom: list[dict] = []

    def __call__(self, account_id, question, portal_user=None, *args, **kwargs):
        self.asked.append(question)
        self.as_whom.append(dict(portal_user or {}))
        return {**CANNED, "answer": dict(CANNED["answer"])}


class _Pipeline:
    """Today's pipeline as the socket calls it: the person each question was asked as."""

    def __init__(self):
        self.as_whom: list[dict] = []

    async def __call__(self, account_id, event, adapter, bg, portal_user=None):
        self.as_whom.append(dict(portal_user or {}))


def _chat(tenant, engine: str, new_core, *, lang: str = "en"):  # noqa: F811
    import gateway.core2_bridge as bridge

    async def setting(account_id):
        return engine

    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(bridge, "engine", setting))
    stack.enter_context(patch("core2.service.portal_answer", new_core))
    conversation = stack.enter_context(
        portal.SocketConversation(tenant, harness.ACCOUNT, harness.saved_connection(), lang=lang, idle_seconds=2.0))
    return stack, conversation


@pytest.fixture(autouse=True)
def _readers_restored():
    """Every test leaves the harness readers as it found them: active, admins, their own password."""
    yield
    import store

    for lang in ("en", "fr"):
        with store.get_db() as conn:
            conn.execute("UPDATE portal_user SET is_active=1, role='admin', is_temp_pw=0, temp_pw_expires_at=NULL "
                         "WHERE account_id=? AND email=?", (harness.ACCOUNT, f"socket-reader-{lang}@harness.example"))


def _closed(turn, lang: str = "en") -> bool:
    return any(f.get("type") == "message" and f.get("content") == ENDED[lang] for f in turn["frames"])


@pytest.mark.parametrize("change", ["deactivated", "password changed", "password reset by an admin"])
def test_a_person_who_may_no_longer_sign_in_is_told_and_the_chat_closes(tenant, change):  # noqa: F811
    import store

    new_core = _Reader()
    stack, conversation = _chat(tenant, "core2", new_core)
    with stack:
        first = conversation.ask(QUESTION)
        assert [a["answer"]["headline"] for a in _answers(first)] == ["New core says 42."]
        if change == "deactivated":
            store.update_user(conversation.user_id, is_active=0)
        elif change == "password changed":
            store.change_password(conversation.user_id, "a-new-password-they-chose", is_temp=False)
        else:
            store.reset_user_password(conversation.user_id)
        second = conversation.ask(QUESTION)
    assert _closed(second), second["frames"]
    assert not _answers(second), "the question was answered after access ended"
    assert new_core.asked == [QUESTION], "the new core was asked after access ended"


@pytest.mark.parametrize("engine", ["legacy", "compare"])
def test_todays_pipeline_follows_the_same_rule(tenant, engine):  # noqa: F811
    import store

    pipeline, new_core = _Pipeline(), _Reader()
    stack, conversation = _chat(tenant, engine, new_core)
    with stack, patch("gateway.webhooks.dispatch", pipeline):
        conversation.ask(QUESTION)
        store.update_user(conversation.user_id, is_active=0)
        second = conversation.ask(QUESTION)
    assert _closed(second) and not _answers(second)
    assert len(pipeline.as_whom) == 1, "today's pipeline answered after access ended"
    assert len(new_core.as_whom) == (1 if engine == "compare" else 0)


def test_a_new_group_and_role_apply_from_the_next_question(tenant):  # noqa: F811
    import store

    new_core = _Reader()
    stack, conversation = _chat(tenant, "core2", new_core)
    group = store.create_group(harness.ACCOUNT, f"Regional {conversation.user_id} {id(new_core)}")
    with stack:
        conversation.ask(QUESTION)
        store.update_user(conversation.user_id, group_id=group, role="analyst")
        second = conversation.ask(QUESTION)
        store.update_user(conversation.user_id, role="admin")
        third = conversation.ask(QUESTION)
    assert not _closed(second) and not _closed(third)
    assert [len(_answers(t)) for t in (second, third)] == [1, 1]
    before, after, back = new_core.as_whom
    assert (before["role"], after["role"], back["role"]) == ("admin", "analyst", "admin")
    assert before.get("group_id") != group and after["group_id"] == group == back["group_id"]


def test_todays_pipeline_reads_the_new_group_too(tenant):  # noqa: F811
    import store

    pipeline = _Pipeline()
    stack, conversation = _chat(tenant, "legacy", _Reader())
    group = store.create_group(harness.ACCOUNT, f"Northern {conversation.user_id} {id(pipeline)}")
    with stack, patch("gateway.webhooks.dispatch", pipeline):
        conversation.ask(QUESTION)
        store.update_user(conversation.user_id, group_id=group)
        conversation.ask(QUESTION)
    before, after = pipeline.as_whom
    assert before.get("group_id") != group and after["group_id"] == group


def test_a_person_whose_access_holds_keeps_the_conversation(tenant):  # noqa: F811
    new_core = _Reader()
    stack, conversation = _chat(tenant, "core2", new_core)
    with stack:
        turns = [conversation.ask(QUESTION) for _ in range(4)]
    assert [len(_answers(t)) for t in turns] == [1, 1, 1, 1]
    assert not any(_closed(t) for t in turns)
    assert {w["id"] for w in new_core.as_whom} == {conversation.user_id}


def test_the_reason_is_said_in_the_readers_language(tenant):  # noqa: F811
    import store

    new_core = _Reader()
    stack, conversation = _chat(tenant, "core2", new_core, lang="fr")
    with stack:
        conversation.ask(QUESTION)
        store.update_user(conversation.user_id, is_active=0)
        second = conversation.ask(QUESTION)
    assert _closed(second, "fr"), second["frames"]
    assert new_core.asked == [QUESTION]
