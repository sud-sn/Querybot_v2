# -*- coding: utf-8 -*-
"""A French reader starts an investigation in French.

The investigation trigger (gateway/webhooks.py) knew only English: "investigate",
"look into", "dig into", "deep dive on", "root-cause analysis of". A French
reader who typed "enquête sur la valeur du stock par entrepôt" got an ordinary
answer to a question with "enquête" in it.

French has two words the trigger must not take for a request: "enquête" is also
a survey ("l'enquête sur la satisfaction"), and "creuse" an adjective
("période creuse"). So a French verb counts only where a request starts, and a
phrase no ordinary question contains ("analyse approfondie de", "cause profonde
de") counts anywhere.

Driven through the real socket, as the wiring tests for the English trigger
are: the planning loop is replaced at its own boundary
(core.investigation_planner.run_investigation), which is exactly the thing a
match reaches and a miss does not.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

# The store is conftest's scratch database for the run (tests/conftest.py);
# nothing here repoints or re-imports it, so modules imported before this one
# keep the same store object as every test after it.
import store  # noqa: E402
import core.investigation_planner as investigation_planner  # noqa: E402
import gateway.webhooks as wh  # noqa: E402
from core.investigation_planner import InvestigationOutcome  # noqa: E402


def _reader(lang: str = "fr"):
    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    store.update_client_meta(account_id, chat_ui_enabled=1, enable_llm_audit=1)
    user_id, _ = store.create_user(account_id, "Ada", f"{os.urandom(4).hex()}@x.com", password="a-password-they-chose")
    store.set_user_language(user_id, lang)
    return account_id, user_id


def _drive(account_id, user_id, text, *, drain=12):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import portal.routes as pr

    app = FastAPI()
    app.include_router(wh.router)
    client = TestClient(app)
    client.cookies.set(pr._COOKIE, pr._sign_session_value(user_id))
    with client.websocket_connect(f"/ws/chat/{account_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "message", "text": text})
        for _ in range(drain):
            frame = ws.receive_json()
            if isinstance(frame, dict) and frame.get("type") == "message" and frame.get("content"):
                break
            if isinstance(frame, dict) and frame.get("type") == "typing" and frame.get("active") is False:
                break


@pytest.fixture
def loop(monkeypatch):
    """The planning loop, replaced at its boundary; records what reached it."""
    seen: list[dict] = []

    async def _fake_loop(*, objective, lang=None, **kwargs):
        seen.append({"objective": objective, "lang": lang})
        return InvestigationOutcome(objective=objective, synthesis="Terminé.", phrasing="template")

    monkeypatch.setattr(investigation_planner, "run_investigation", _fake_loop)
    monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
    return seen


class TestFrenchAsksStartAnInvestigation:

    @pytest.mark.parametrize("text", [
        "Enquête sur la valeur du stock par entrepôt",
        "Peux-tu enquêter sur la baisse des ventes en 2022 ?",
        "Pouvez-vous investiguer les réceptions par division ?",
        "Penche-toi sur les achats par entrepôt",
        "Creuse les ventes par groupe d'articles",
        "Approfondis la valeur du stock par fournisseur",
        "Merci de mener une enquête sur les commandes en souffrance",
        "Lance une enquête sur l’écart de stock à Oshawa",
        "Fais une analyse approfondie des ventes par région",
        "Quelle est la cause profonde de la baisse des achats ?",
        # No request verb up front: the phrase itself asks for one.
        "J'aimerais une analyse approfondie de la valeur du stock par entrepôt",
        "Une analyse des causes de la baisse des ventes, s'il te plaît",
        "Bonjour. Enquête sur les transferts par mois",
        "S’il vous plaît, enquête sur les retours par mois",
    ])
    def test_the_readers_own_words_reach_the_loop(self, loop, text):
        account_id, user_id = _reader("fr")
        _drive(account_id, user_id, text)
        assert loop == [{"objective": text, "lang": "fr"}]


class TestOrdinaryFrenchIsNeverDiverted:

    @pytest.mark.parametrize("text", [
        # "creuse" as an adjective: an off-peak period.
        "Ventes en période creuse par mois",
        # "enquête" as a survey.
        "Combien de réponses à l'enquête sur la satisfaction client ?",
        # The verb with nothing to investigate.
        "Enquête sur",
        # A why-question stays one governed answer, as in English.
        "Pourquoi les ventes ont-elles baissé en mai ?",
        "Quel est le stock disponible par groupe d'articles ?",
    ])
    def test_it_does_not_reach_the_loop(self, monkeypatch, text):
        spy = AsyncMock()
        monkeypatch.setattr(investigation_planner, "run_investigation", spy)
        monkeypatch.setattr(wh, "resolve_provider", lambda *a, **k: ("test", "test", "k", {}))
        account_id, user_id = _reader("fr")
        _drive(account_id, user_id, text)
        spy.assert_not_called()


def test_an_english_ask_still_starts_one(loop):
    account_id, user_id = _reader("en")
    _drive(account_id, user_id, "investigate stock value by warehouse")
    assert loop == [{"objective": "investigate stock value by warehouse", "lang": "en"}]
