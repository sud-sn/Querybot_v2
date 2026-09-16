# -*- coding: utf-8 -*-
"""A French reader could not command the result card.

The card understands a closed set of commands -- sort, exclude, keep the top
N, show as a chart, undo -- through parse_result_command, which reads English.
A French question is canonicalised first, so the card hears the reader's
French as English; but the lexicon carried no verb the card understands.
"trie par montant" reached the parser as "trie by amount", "exclus Paris" as
"exclus paris", "montre-moi en camembert" as "show me en camembert", and
every one of them fell through to "I could not answer that". English
readers had a card they could command; French readers had a card they
could only look at.

Every test canonicalises real French and hands the result to the real
parser, or drives the real socket end to end.
"""

from __future__ import annotations

import pytest

from core.question_normalizer import canonical_question
from core.result_commands import parse_result_command


def _command(french: str):
    return parse_result_command(canonical_question(french, "fr"))


class TestTheCardHearsFrenchCommands:

    @pytest.mark.parametrize("french, target, direction", [
        ("trie par montant décroissant", "amount", "desc"),
        ("trier par ventes croissant", "sales", "asc"),
        ("triez par marge", "margin", "desc"),
        ("ordonne par quantité décroissante", "quantity", "desc"),
        ("trie par ventes du plus grand au plus petit", "sales", "desc"),
    ])
    def test_sorting(self, french, target, direction):
        command = _command(french)
        assert command is not None, canonical_question(french, "fr")
        assert (command.action, command.target_text, command.direction) == ("sort", target, direction)

    @pytest.mark.parametrize("french, target", [
        ("exclus Paris", "paris"),
        ("exclure Paris", "paris"),
        ("retire Lyon", "lyon"),
        ("enlève la ligne 2", "row 2"),
        ("supprime Marseille", "marseille"),
        ("retirez Paris de ce résultat", "paris"),
    ])
    def test_excluding(self, french, target):
        command = _command(french)
        assert command is not None, canonical_question(french, "fr")
        assert (command.action, command.target_text) == ("exclude", target)

    @pytest.mark.parametrize("french, limit", [
        ("garde les 3 premiers", 3),
        ("garde seulement les 3 premiers", 3),
        ("gardez uniquement les 5 premiers", 5),
        ("conserve les 2 premiers", 2),
        ("seulement les 3 premiers", 3),
        ("montre-moi les 10 premiers", 10),
    ])
    def test_keeping_the_top(self, french, limit):
        command = _command(french)
        assert command is not None, canonical_question(french, "fr")
        assert (command.action, command.limit) == ("keep_top", limit)

    @pytest.mark.parametrize("french, kind", [
        ("montre-moi un graphique", "auto"),
        ("affiche un graphique", "auto"),
        ("montre en camembert", "pie"),
        ("affiche en barres", "bar"),
        ("montre-moi en courbe", "line"),
        ("affiche en tableau", "table"),
        ("montre-moi en nuage de points", "scatter"),
    ])
    def test_presenting(self, french, kind):
        command = _command(french)
        assert command is not None, canonical_question(french, "fr")
        assert (command.action, command.presentation_type) == ("presentation", kind)

    def test_undo(self):
        for french in ("annuler la dernière modification", "reviens en arrière", "défaire"):
            command = _command(french)
            assert command is not None and command.action == "undo", canonical_question(french, "fr")


class TestTheVerbsDoNotEatTheNouns:
    """A verb added for the card must not rewrite the same letters inside a
    business word: "classe" is a product class, "retire" a retired status,
    "trie" lives inside "industrie"."""

    @pytest.mark.parametrize("french, expected", [
        ("ventes par classe de produit", "sales by classe of product"),
        ("ventes par industrie", "sales by industrie"),
        ("commandes annulées par client", "cancelled orders by customer"),
        ("ventes de croissants par magasin", "sales of croissants by store"),
        ("commandes en ligne par client", "orders online by customer"),
    ])
    def test_a_business_word_keeps_its_meaning(self, french, expected):
        assert canonical_question(french, "fr") == expected


class TestTheCardObeysOverTheSocket:
    """One French command, end to end: the frame a browser sends for a French
    reader, and the rows that come back."""

    def test_a_french_reader_excludes_a_row(self):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_the_card_keeps_the_conversation import _Card

        with _Card(lang="fr") as card:
            reply = card.ask("exclus East")
        assert reply.get("type") == "result_chat_response", reply
        assert reply["row_count"] == 3
        assert "East" not in [r.get("REGION") for r in reply["rows"]]

    def test_a_french_reader_keeps_the_top_two(self):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_the_card_keeps_the_conversation import _Card

        with _Card(lang="fr") as card:
            reply = card.ask("garde seulement les 2 premiers")
        assert reply.get("type") == "result_chat_response", reply
        assert [r.get("REGION") for r in reply["rows"]] == ["North", "South"]
