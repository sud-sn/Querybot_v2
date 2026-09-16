# -*- coding: utf-8 -*-
"""A French reader who asked for an analysis got a table.

The analyst gate reads the CANONICAL question -- the reader's French mapped to
the English the detectors know -- and the lexicon carried "pourquoi" but not
one analysis verb. Measured on the real functions before the fix:

    "analyser les ventes par région"          -> canonical "analyser the sales by region"  -> nothing
    "explique les ventes par entrepôt"        -> "explique the sales by warehouse"          -> nothing
    "résume les ventes par entrepôt"          -> "resume the sales by warehouse"            -> nothing
    "interprète la répartition du CA"         -> "interprete the breakdown of revenue"      -> nothing
    "qu'est-ce qui ressort des ventes"        -> "what ressort of sales"                     -> nothing

Only "analyse" (spelt the same in both languages) and "pourquoi" reached the
gate. In English, "summarize", "a summary of" and "insights on" missed too.

With the tenant default of "always" every result gets the analyst anyway, so
these words matter where the gate still decides: tenants on "on_request", and
the follow-up router that reads a question typed against the last result.
Every test executes canonical_question and the real gate.
"""

from __future__ import annotations

import pytest

from core.insight import analysis_action_for, is_insight_question
from core.question_normalizer import canonical_question


def gate(question: str, lang: str = "fr") -> str:
    return analysis_action_for(canonical_question(question, lang), mode="on_request")


class TestFrenchRequestsForAnalysisReachTheAnalyst:

    @pytest.mark.parametrize("question", [
        "analyser les ventes par région",
        "analysez les ventes nettes par entrepôt",
        "analyse les ventes nettes par entrepôt",
        "explique les ventes par entrepôt",
        "expliquer la répartition des ventes",
        "expliquez-moi les retours par entrepôt",
        "résume les ventes par entrepôt",
        "résumer les ventes de l'année",
        "donne-moi un résumé des retours par entrepôt",
        "interprète la répartition du chiffre d'affaires",
        "interpréter les marges par produit",
        "qu'est-ce qui ressort des ventes par région",
        "que ressort-il des ventes par région",
    ])
    def test_an_explicit_request_earns_one_call(self, question):
        assert gate(question) == "analyze", canonical_question(question, "fr")

    @pytest.mark.parametrize("question", [
        "pourquoi les ventes ont-elles baissé en mars",
        "explique pourquoi les ventes ont baissé",
    ])
    def test_a_causal_request_still_drills(self, question):
        assert gate(question) == "why"

    @pytest.mark.parametrize("question", [
        "ventes nettes par entrepôt",
        "les 5 meilleurs clients par ventes",
        "ventes du mois dernier",
    ])
    def test_a_plain_french_question_is_still_plain(self, question):
        assert gate(question) == ""

    def test_accents_are_optional_here_too(self):
        assert gate("resume les ventes par entrepot") == "analyze"
        assert gate("interprete la repartition") == "analyze"


class TestTheEnglishWordsThatWereMissing:

    @pytest.mark.parametrize("question", [
        "summarize sales by warehouse",
        "summarise the returns by month",
        "give me a summary of returns by warehouse",
        "insights on sales by customer",
        "any insight into margin by product?",
    ])
    def test_a_request_for_a_summary_or_insights_is_a_request_for_analysis(self, question):
        assert analysis_action_for(question, mode="on_request") == "analyze"
        assert is_insight_question(question) is True

    @pytest.mark.parametrize("question", [
        "sales summary report by warehouse",
    ])
    def test_a_noun_in_a_table_name_still_earns_it(self, question):
        """The word is the reader's; whether it names a report or asks for
        one, an extra explanation is the cheap mistake and a table with no
        analyst is the expensive one."""
        assert analysis_action_for(question, mode="on_request") == "analyze"

    @pytest.mark.parametrize("question", [
        "net sales by warehouse",
        "top 10 customers by margin",
        "list open orders",
    ])
    def test_plain_english_is_still_plain(self, question):
        assert analysis_action_for(question, mode="on_request") == ""


class TestTheFollowUpRouterReadsFrenchToo:
    """A question typed against the LAST result goes through a second gate in
    the chat socket, which read the raw text. It needs a socket to execute,
    so its call is checked as a syntax tree: is_insight_question(
    canonical_question(text, ...))."""

    def test_the_cached_result_gate_canonicalises_first(self):
        import ast
        import inspect
        import gateway.webhooks as wh

        tree = ast.parse(inspect.getsource(wh.ws_chat))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) == "is_insight_question"
        ]
        assert calls, "the socket no longer consults the insight gate at all"
        for call in calls:
            inner = call.args[0]
            assert isinstance(inner, ast.Call), ast.dump(call)
            assert getattr(inner.func, "id", getattr(inner.func, "attr", "")) == "canonical_question"
            assert getattr(inner.args[0], "id", None) == "text"
