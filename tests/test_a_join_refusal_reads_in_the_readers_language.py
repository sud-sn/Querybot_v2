"""
A join refusal reads in the reader's language, and names what it cannot reach in words.

When the confirmed join graph cannot reach part of a question, no query is run
and the reader is told so. That reply was English whatever the reader's
language -- a French reader on the sample tenant asking for reserved stock by
warehouse got "I couldn't build a trusted join plan for this question. The
confirmed entity graph cannot reach every business entity required by this
question" -- and it named what it could not reach by the warehouse's code
("PTY_DMS"). It now says what happened, what could not be reached in words
("Party") and what to do, in the reader's language; the graph's own account
stays in the trace.

A synthetic tenant (tests/answer_harness.py) whose monthly fact keeps units
sold and whose buyer is kept only on the daily snapshot: "units sold by buyer"
cannot be joined, and is refused.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("join-refusal")) as built:
        yield built


def _expected(lang: str, names: str = "") -> list[str]:
    reason = (i18n.t("fail.graph.reason", lang=lang, names=names) if names
              else i18n.t("fail.graph.reason_unnamed", lang=lang))
    return [i18n.t("fail.graph.headline", lang=lang), reason, i18n.t("fail.graph.next_step", lang=lang)]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Units sold by buyer", "en"),
        ("Unités vendues par acheteur", "fr"),
    ])
    def test_the_refusal(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["executed"] == [] and answer["model_wrote_sql"] is False
        ((kind, text),) = answer["replies"]
        assert (kind, text.split("\n\n")) == ("message", _expected(lang, "Party"))


_MODEL = {"tables": [
    {"qualified_name": "MART.WHS_DMS", "type": "dimension", "entity": "Whs"},
    {"qualified_name": "MART.ITM_BAL_PRD_FCT", "type": "fact", "entity": "Itm Bal Prd",
     "grain": "one row per item warehouse month"},
]}


class TestTheReply:

    @staticmethod
    def _reply(graph_ctx: dict, lang: str = "en") -> list[str]:
        from core.query_pipeline import graph_block_response
        from core.vocab_packs import builtin_vocab

        return graph_block_response(graph_ctx, lang=lang, model=_MODEL, vocab=builtin_vocab()).split("\n\n")

    def test_codes_in_words_and_role_names_as_they_are(self):
        reply = self._reply({"missing_entities": ["WHS_DMS", "Item Balance Effective Date", "WHS_DMS"],
                             "reason": "The graph cannot reach WHS_DMS."})
        assert reply == _expected("en", "Warehouse, Item Balance Effective Date")

    def test_a_role_named_in_french(self):
        assert self._reply({"missing_entities": ["Date de création"]}, "fr") == _expected("fr", "Date de création")

    def test_a_table_as_the_model_names_it(self):
        # As the source choice card names it: the model's grain gives its cadence.
        assert self._reply({"missing_entities": ["ITM_BAL_PRD_FCT"]}) == _expected("en", "Monthly Item Balance Period")

    def test_a_table_the_model_does_not_know(self):
        assert self._reply({"missing_entities": ["PTY_DMS"]}, "fr") == _expected("fr", "Party")

    def test_nothing_named(self):
        reply = self._reply({"reason": "No confirmed relationship connects the requested business entities."}, "fr")
        assert reply == _expected("fr")
