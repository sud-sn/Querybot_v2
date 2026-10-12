"""A French reader's buttons under an answer are in French, and a click on one works as the English one does.

The answer's note and headline were French on a French screen, while the buttons under it -- "Why did net amount
change in Dec 2025?", "Forecast net amount for the next 3 months", "Median instead", "Including Cancelled" -- the
date badge ("by order date") and the chart's click menu were English, and a click sent the English words. Each now
comes with its French twin: the page shows it and a click sends it. A chip that changes one setting finds its plan
by either wording, with no AI; "about the answer above" or "a new question" is read in French too. The data's own
names (a figure, a store) stay as they are.

The invented retail shop and home-services firm of the test domains.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.plan.chips import french
from core2.plan.ir import Plan
from core2.plan.values import MemberIndex
from core2.service import ABOVE, AFRESH, Services, Session, _answer_question, answer_question, which_way
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)
Y2025 = {"window": {"kind": "between", "start": "2025-01-01", "end": "2025-12-31"}}


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return built.con, model


def _run(retail, plan: dict) -> dict:
    con, model = retail
    services = Services(model=model, warehouse=DuckDBWarehouse(con), complete=None, index=MemberIndex(), today=TODAY)
    return _answer_question("q", services, Session(), planned=Plan.model_validate({"kind": "query", **plan}))


def test_the_next_questions_of_a_trend_are_french_for_a_french_reader(retail):
    payload = _run(retail, {"intent": "trend", "measures": ["net_amount"], "time": {**Y2025, "grain": "month"}})
    chips = payload["follow_up_suggestions"]
    assert chips and all(c["label_fr"] and c["question_fr"] for c in chips)
    french_words = [c["question_fr"] for c in chips]
    assert "Pourquoi net amount a-t-il varié en déc. 2025 ?" in french_words
    assert "Prévision de net amount pour les 3 prochains mois" in french_words
    assert not any("Why" in w or "Forecast" in w for w in french_words)
    chart = payload["chart"]                                     # the chart's own title too
    assert chart["title"].endswith(" by month")
    assert chart["title_fr"] == chart["title"][:-len(" by month")] + " par mois"


def test_the_next_questions_of_a_breakdown_keep_its_period_and_conditions_in_french(retail):
    payload = _run(retail, {"intent": "breakdown", "measures": ["net_amount"], "group_by": ["store"], **{"time": Y2025},
                            "filters": [{"field": "region.name", "op": "eq", "values": ["North"]}]})
    french_words = [c["question_fr"] for c in payload["follow_up_suggestions"]]
    english = [c["question"] for c in payload["follow_up_suggestions"]]
    assert any(w.startswith("Net Amount par mois pour") or w.startswith("Net amount par mois pour")
               for w in french_words), french_words
    assert all(', pour region "North"' in w for w in french_words), french_words      # the condition, in French
    assert all(', for region "North"' in w for w in english), english
    assert any(w.startswith("Part de net amount par") and "en 2025" in w for w in french_words), french_words
    assert payload["chart"]["title_fr"].endswith(' par store, pour region North'), payload["chart"]["title_fr"]


def test_the_badges_are_french_for_a_french_reader(retail):
    payload = _run(retail, {"intent": "value", "measures": ["net_amount"],
                            "time": {"window": {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}}})
    assert {"kind": "period", "text": "Q1 2026", "text_fr": "T1 2026"} in payload["answer"]["badges"]
    assert {"kind": "date", "text": "by order date", "text_fr": "selon order date"} in payload["answer"]["badges"]


@pytest.mark.parametrize("words, said", [
    ("Median instead", "Médiane à la place"),
    ("90th percentile instead", "90e centile à la place"),
    ("6-month moving average", "Moyenne mobile sur 6 mois"),
    ("Total of the last 4 weeks", "Total des 4 dernières semaines"),
    ("Top 5 in each store", "Top 5 dans chaque store"),
    ("No orders in the last 180 days", "Sans orders depuis 180 jours"),
    ("Including Cancelled, Duplicate Entry and Voided", "En incluant Cancelled, Duplicate Entry et Voided"),
    ("A name of no known form", "A name of no known form"),
])
def test_a_chip_says_its_setting_in_french(words, said):
    assert french(words) == said


def test_a_french_click_on_a_setting_chip_runs_its_plan_with_no_ai():
    built, model = learn(domains.build("home_services"), "descriptive")
    plan = {"kind": "query", "intent": "value", "measures": ["number_of_bookings"], "time": Y2025}
    asked: list[str] = []

    def ai(stable: str, tail: str) -> str:
        asked.append(tail)
        return json.dumps(plan)

    services = Services(model=model, warehouse=DuckDBWarehouse(built.con), complete=ai, index=MemberIndex(),
                        today=TODAY)
    session = Session()
    first = answer_question("Combien de réservations en 2025 ?", services, session)
    chip = next(c for c in first["follow_up_suggestions"] if c.get("kind") == "setting")
    assert chip["label_fr"].startswith("En incluant") and chip["question_fr"] == chip["label_fr"]
    calls = len(asked)
    second = answer_question(chip["question_fr"], services, session)
    every = built.con.execute("SELECT COUNT(*) FROM bookings b JOIN calendar c ON b.booked_date_key = c.date_key "
                              "WHERE c.full_date BETWEEN '2025-01-01' AND '2025-12-31'").fetchone()[0]
    assert second["data"]["rows"] == [{"number_of_bookings": every}] and len(asked) == calls


def test_about_the_answer_above_or_a_new_question_is_read_in_french():
    assert which_way("À propos de la réponse ci-dessus", [ABOVE, AFRESH]) == "refine"
    assert which_way("Une nouvelle question", [ABOVE, AFRESH]) == "new"
    assert which_way("A new question", [ABOVE, AFRESH]) == "new"


def _page(msg: dict, lang: str) -> str:
    from tests.chat_js import run
    from tests.test_the_answer_card import FUNCTIONS, PREAMBLE

    script = (f"JSON.stringify(answerCardHtml({json.dumps(msg)}, {{chartId: 'c1', rcId: 'rc-1', pinToken: ''}}, "
              "{feedbackEnabled: false}));")
    return run(script, lang=lang, functions=FUNCTIONS, consts=["_NUMERIC_FORMATS", "_QB_CURRENCY_SYMBOL",
                                                               "ANSWER_CHART_TYPES"],
               preamble=PREAMBLE + f"\nwindow.QB_LANG = {json.dumps(lang)};")


def test_the_card_shows_and_sends_the_french_words_on_a_french_screen():
    msg = {"type": "assistant_response", "engine": "core2", "question": "q",
           "answer": {"headline": "Net amount: 10 $.", "badges": [
               {"kind": "period", "text": "Q1 2026", "text_fr": "T1 2026"},
               {"kind": "date", "text": "by order date", "text_fr": "selon order date"}]},
           "follow_up_suggestions": [
               {"label": "Median instead", "question": "Median instead", "label_fr": "Médiane à la place",
                "question_fr": "Médiane à la place", "kind": "setting"},
               {"label": "Forecast net amount for the next 3 months",
                "question": "Forecast net amount for the next 3 months",
                "label_fr": "Prévision de net amount pour les 3 prochains mois",
                "question_fr": "Prévision de net amount pour les 3 prochains mois"}]}
    french_card, english_card = _page(msg, "fr"), _page(msg, "en")
    assert "T1 2026" in french_card and "selon order date" in french_card and "Q1 2026" not in french_card
    assert 'data-question="Médiane à la place"' in french_card
    assert 'data-question="Prévision de net amount pour les 3 prochains mois"' in french_card
    assert "Forecast net amount" not in french_card
    assert 'data-question="Median instead"' in english_card and "Q1 2026" in english_card
