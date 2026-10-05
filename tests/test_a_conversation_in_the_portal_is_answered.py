"""
A conversation in the portal is answered: the question, the drill-down, the
"why", in English and in French.

Asked of an inventory warehouse through the portal -- the dispatcher, the
conversational analyst, the result kept between turns -- most of a sales demo's
questions came back without an answer:

* "stock on hand by warehouse" was put to the conversational analyst, a model,
  to decide whether it was a question for the data at all; the model offered
  to run it ("... let me know!") and the reader got a Proceed button for the
  question they had just asked.
* asked after "what is our total stock on hand?", it reached the governed
  compilers under the label the product joins a follow-up to its parent with
  ("Follow-up request:"), and "request" was read as a word narrowing what was
  asked -- so no governed answer.
* "break it down by item group" offered "down" as a member to filter by, and
  then no compiler would group by two things.
* "top 2 items by inventory value", asked after "inventory value by
  warehouse", names the result's measure, so it was handed to the result on
  screen -- which holds warehouses, not items -- and refused.
* "why is NORTH DEPOT the highest?" was refused as a question naming nothing
  to measure; "why is that?" after a drill-down lost the question the
  drill-down was of, and asked which dataset was meant.
* "seulement les 2 premiers" and "pourquoi ... ?" were read as English, so the
  first was not a result command and both were put to the analyst.

Each is asked here, of the harness's inventory tenant, through the portal.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness
from tests import portal_harness as portal


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("portal-conversation")) as built:
        yield built


def _conversation(warehouse, lang: str = "en") -> portal.Conversation:
    return portal.Conversation(warehouse, harness.ACCOUNT, harness.saved_connection(), lang=lang)


def _headline(turn) -> str:
    return str(((turn.answer or {}).get("answer") or {}).get("headline") or "")


def _columns(turn) -> list[str]:
    return list(turn.rows[0]) if turn.rows else []


def _analysis_titles(turn) -> list[str]:
    return [str(f.get("title") or "") for f in turn["frames"] if f.get("type") == "assistant_analysis"]


def _governed(turn) -> None:
    """Answered by a governed compiler, without being offered or put to the model."""
    assert turn.answer is not None, turn.texts
    assert not turn.offered, turn.texts
    assert not turn["model_wrote_sql"], turn["sql"]


class TestTheFirstQuestion:

    def test_a_breakdown_is_answered_not_offered(self, warehouse):
        turn = _conversation(warehouse).ask("stock on hand by warehouse")

        _governed(turn)
        assert not turn["asked_the_analyst"]
        assert _headline(turn) == "In FT, NORTH DEPOT leads at 900 FT."

    def test_the_lowest_stock_is_the_stock_on_hand(self, warehouse):
        turn = _conversation(warehouse).ask("which warehouses have the lowest stock?")

        _governed(turn)
        assert "STOCK_ON_HAND" in _columns(turn)
        assert _headline(turn) == "In FT, SOUTH DEPOT is lowest at 350 FT."

    def test_the_lowest_stock_in_french(self, warehouse):
        turn = _conversation(warehouse, "fr").ask("quels entrepôts ont les stocks les plus bas ?")

        _governed(turn)
        assert _headline(turn) == "En FT, SOUTH DEPOT est le plus bas avec 350 FT."

    def test_the_value_of_the_stock_in_french_is_the_value_alone(self, warehouse):
        turn = _conversation(warehouse, "fr").ask("valeur du stock par entrepôt")

        _governed(turn)
        assert _columns(turn) == ["WAREHOUSE", "INVENTORY_VALUE"]

    def test_a_monthly_quantity_is_drawn(self, warehouse):
        turn = _conversation(warehouse).ask("units sold by month")

        _governed(turn)
        assert (turn.answer.get("chart") or {}).get("chart_type") == "line"


class TestTheFollowUps:

    def test_a_breakdown_after_a_total_is_governed(self, warehouse):
        conversation = _conversation(warehouse)
        conversation.ask("what is our total stock on hand?")
        turn = conversation.ask("stock on hand by warehouse")

        _governed(turn)
        assert "WAREHOUSE" in _columns(turn)

    def test_a_drill_down_and_why(self, warehouse):
        conversation = _conversation(warehouse)
        conversation.ask("stock on hand by warehouse")
        drilled = conversation.ask("break it down by item group")

        _governed(drilled)
        assert {"ITEM_GROUP", "WAREHOUSE"} <= set(_columns(drilled))

        why = conversation.ask("why is that?")
        _governed(why)
        assert {"ITEM_GROUP", "WAREHOUSE"} <= set(_columns(why))
        assert _analysis_titles(why) == ["Why this pattern?"]

    def test_a_question_of_its_own_beside_a_result_is_asked(self, warehouse):
        conversation = _conversation(warehouse)
        conversation.ask("inventory value by warehouse")
        turn = conversation.ask("top 2 items by inventory value")

        _governed(turn)
        assert _columns(turn) == ["ITEM", "INVENTORY_VALUE"]
        assert [row["ITEM"] for row in turn.rows][0] == "COPPER PIPE"
        assert len(turn.rows) == 2

    def test_why_a_member_leads_is_answered_with_the_result(self, warehouse):
        conversation = _conversation(warehouse)
        conversation.ask("inventory value by warehouse")
        turn = conversation.ask("why is NORTH DEPOT the highest?")

        _governed(turn)
        assert _columns(turn) == ["WAREHOUSE", "INVENTORY_VALUE"]
        assert _analysis_titles(turn) == ["Why this pattern?"]

    def test_what_about_another_measure(self, warehouse):
        conversation = _conversation(warehouse)
        conversation.ask("stock on hand by warehouse")
        turn = conversation.ask("what about reserved quantity?")

        _governed(turn)
        assert "RESERVED_QUANTITY" in _columns(turn)


class TestInFrench:

    def test_only_the_first_two_is_a_result_command(self, warehouse):
        conversation = _conversation(warehouse, "fr")
        conversation.ask("stock disponible par entrepôt")
        turn = conversation.ask("seulement les 2 premiers")

        _governed(turn)
        assert not turn["asked_the_analyst"]
        assert len(turn.rows) == 2

    def test_why_is_asked_about_the_result(self, warehouse):
        conversation = _conversation(warehouse, "fr")
        conversation.ask("valeur du stock par entrepôt")
        turn = conversation.ask("pourquoi NORTH DEPOT est-il le plus élevé ?")

        _governed(turn)
        assert not turn["asked_the_analyst"]
        assert _analysis_titles(turn) == ["Pourquoi ce schéma ?"]
