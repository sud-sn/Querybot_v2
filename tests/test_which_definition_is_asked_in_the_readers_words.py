"""
"Which definition?" is asked in the reader's words.

A bare measure word -- "sales", "revenue", "ventes" -- that more than one
schema defines is asked about rather than guessed. The question was one
English sentence, whoever asked and whatever they asked about: "I found more
than one revenue definition. Which one should I use?" -- to a French reader,
and to a reader who had said "sales". The clarification it saved named
"revenue" too.

It now names the reader's own word, in the reader's language: "I found more
than one definition of “Sales”", "J’ai trouvé plusieurs définitions de
« Ventes »", and the clarification keeps that word.

tests/star_harness.py answers the question; the metric scope is made to find
two definitions, as a warehouse with two schemas that each define sales does.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("which-definition")) as built:
        yield built


@pytest.fixture
def two_definitions(monkeypatch):
    import core.query_pipeline as query_pipeline
    from core.metric_scope import MetricScopeResult

    monkeypatch.setattr(query_pipeline, "resolve_metric_scope", lambda *args, **kwargs: MetricScopeResult(
        metrics=[], ambiguous=True, options=["Net Sales", "Gross Sales"]))


class TestTheProductAsks:

    @pytest.mark.parametrize("question,lang,asked", [
        ("Sales", "en", "I found more than one definition of “Sales”. Which one should I use?"),
        ("Ventes ?", "fr", "J’ai trouvé plusieurs définitions de « Ventes ». Laquelle dois-je utiliser ?"),
    ])
    def test_in_the_readers_words(self, warehouse, two_definitions, question, lang, asked):
        answer = star.ask(warehouse, question, lang)
        clarifications = [body for kind, body in answer["replies"] if kind == "clarify"]
        assert [body["question"] for body in clarifications] == [asked]
        assert [option["label"] for option in clarifications[0]["options"]] == ["Net Sales", "Gross Sales"]
        assert answer["executed"] == []

    def test_the_clarification_keeps_the_readers_word(self, warehouse, two_definitions):
        from core.clarification import get_pending

        star.ask(warehouse, "Sales", "en")
        pending = get_pending(star.ACCOUNT, "harness-en", session_id=f"{star.ACCOUNT}:portal:harness")
        assert pending["clarification_meta"]["term"] == "Sales"


class TestTheQuestion:

    @pytest.mark.parametrize("question,lang,term", [
        ("revenue?", "en", "revenue"),
        ("  Chiffre d'affaires ?  ", "fr", "Chiffre d'affaires"),
        ("sales amount!", "en", "sales amount"),
    ])
    def test_the_term_is_the_readers(self, question, lang, term):
        from core.metric_scope import ambiguous_metric_question

        text, said = ambiguous_metric_question(question, lang)
        assert said == term and term in text
        assert ("Laquelle" in text) is (lang == "fr")
