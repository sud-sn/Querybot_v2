"""
A French question names its measure by the words an English one does.

"Quelle est la valeur de notre stock ?" was answered with month-end inventory
value, where "What is our inventory value?" was answered from the daily
snapshot. The French named neither metric's phrase verbatim, so both were
scored on the words they shared with it, and "de" was one of the words
"valeur du stock en fin de mois" shared: 30 to 20. Beside it, the scorers read
French words English ones had long stopped counting:

  * articles and question words ("de", "la", "notre", "quelle");
  * a generic word alone -- "quantité reçue par entrepôt" put five stock
    metrics in scope on "quantité", where "received quantity by warehouse"
    matched none;
  * a grain or a window -- "par mois", "le mois dernier", "ce mois-ci" -- which
    reached the month-end metrics on "mois".

None of them is evidence now. With "de" gone the two value metrics tied, and
the tie went to the first name; a phrase whose every word is in the question,
in another order, now counts a word more than one it half shares, whose other
words ("fin de mois") name a narrower measure nobody asked for.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from core.metric_scope import _phrase_score, resolve_metric_scope
from core.question_normalizer import canonical_question
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("french-measures")) as built:
        yield built


@pytest.fixture(scope="module")
def metrics(warehouse):
    """The tenant's metrics as discovery proposed them."""
    import store

    return {metric["name"]: metric for metric in store.list_metrics(harness.ACCOUNT)}


def _scoped(metrics: dict, question: str, lang: str) -> list[str]:
    canonical = canonical_question(question, lang) if lang == "fr" else question
    return [metric["name"] for metric in resolve_metric_scope(
        list(metrics.values()), canonical, None, reader_question=question).metrics]


def _rows(answer: dict, measure: str) -> list[dict]:
    (answered,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return answered["rows"]


class TestTheProductAnswersWithTheMeasureAsked:

    @pytest.mark.parametrize("question,lang", [
        ("Quelle est la valeur de notre stock ?", "fr"),
        ("What is the value of our stock?", "en"),
    ])
    def test_the_value_of_our_stock(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert not any("MONTH_END" in run["sql"] for run in answer["executed"])
        (row,) = _rows(answer, "INVENTORY_VALUE")
        assert row["INVENTORY_VALUE"] == pytest.approx(
            sum(on_hand * cost for *_keys, on_hand, _allocated, cost in harness.STOCK))

    def test_month_end_is_still_asked_for_by_name(self, warehouse):
        answer = harness.ask(warehouse, "Valeur du stock en fin de mois", "fr")
        assert any("AS MONTH_END_INVENTORY_VALUE" in run["sql"] for run in answer["executed"])


class TestWhatNamesNoMeasure:

    def test_a_french_article(self, metrics):
        month_end = metrics["Month-end inventory value"]
        assert _phrase_score(month_end, "la valeur de notre stock") == _phrase_score(month_end, "valeur stock")

    @pytest.mark.parametrize("question", ["Quantité reçue par entrepôt", "Quelle est la valeur des ventes ?"])
    def test_a_french_generic_word_alone(self, metrics, question):
        assert _scoped(metrics, question, "fr") == []

    @pytest.mark.parametrize("question", ["Received quantity by warehouse", "What is the value of sales?"])
    def test_as_its_english_twin(self, metrics, question):
        assert _scoped(metrics, question, "en") == []

    @pytest.mark.parametrize("window", [
        "par mois", "tous les mois", "le mois dernier", "des 3 derniers mois", "ce mois-ci",
        "du mois le plus récent", "de l'année en cours",
    ])
    def test_a_french_grain_or_window(self, metrics, window):
        month_end = metrics["Month-end stock on hand"]
        assert _phrase_score(month_end, f"Stock {window}") == _phrase_score(month_end, "Stock")

    def test_the_most_recent_month(self, metrics):
        month_end = metrics["Month-end stock on hand"]
        assert _phrase_score(month_end, "stock in the most recent month") == _phrase_score(month_end, "stock")

    def test_the_end_of_a_month_still_names_one(self, metrics):
        month_end = metrics["Month-end stock on hand"]
        assert _phrase_score(month_end, "Stock en fin de mois") > _phrase_score(month_end, "Stock")

    def test_the_prompts_candidates_read_french_the_same_way(self, metrics):
        from store.config_store import _score_metric_for_question

        assert _score_metric_for_question(metrics["Inventory value"], "Quelle est la valeur des ventes ?") == 0


class TestEveryWordOfThePhrase:

    @pytest.mark.parametrize("question,lang", [
        ("Quelle est la valeur de notre stock ?", "fr"),
        ("what is the value of our inventory", "en"),
    ])
    def test_outranks_half_of_a_longer_one(self, metrics, question, lang):
        assert _scoped(metrics, question, lang) == ["Inventory value"]

    def test_an_exact_phrase_still_wins(self, metrics):
        assert _scoped(metrics, "month-end inventory value", "en")[0] == "Month-end inventory value"
