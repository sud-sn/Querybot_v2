"""
A total kept per unit of measure is not a ranking.

A total of a quantity is read per unit (core/units_of_measure.py), so "what is
our total stock on hand?" comes back as a row of feet and a row of eaches. The
card read those rows as a ranking -- "FT leads at 1,250", "1,030 above the next
result" -- and for units sold went on to "FT holds 91.1% of the total" and "FT
alone holds 91% of the total — a single point of dependency": 380 feet ahead of
37 eaches, and a share of a total that adds the two. On the sample tenant the
total stock on hand was headed "FT leads at 28,640." in both languages.

Each row is a total in its own unit, and none is ahead of another. The card
lists them by the unit's name and says they are not added together, and the
brief the summary, callouts, signal and analysis are written from takes no
leader, gap, share or total across them -- nor do the follow-ups offered, the
computed summary a regulated tenant reads, or the cards shown when the model
path fails. A reader who asks which unit holds the most is answered as asked;
money, which adds up across units, is still ranked; and so is a count, a rate
or an average kept beside a unit column, as none of them is a quantity of goods.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests import answer_harness as harness

_PER_UNIT = [
    {"UNT_OF_MSR": "FT", "STOCK_ON_HAND": 1250.0},
    {"UNT_OF_MSR": "EA", "STOCK_ON_HAND": 220.0},
]
_NOT_ADDED = "Quantities in different units are not added together."


def _card(rows: list[dict], question: str, lang: str = "en", column_formats: dict | None = None) -> dict:
    from core.response_builder import build_answer, infer_result_scope

    token = i18n.activate_language(lang)
    try:
        return build_answer(rows, question, infer_result_scope(rows, question, mode="ranking"),
                            column_formats=column_formats)
    finally:
        i18n.deactivate_language(token)


def _answered_card(answer: dict) -> dict:
    (card,) = [payload for kind, payload in answer["replies"]
               if isinstance(payload, dict) and payload.get("answer")]
    return card


class TestTheProductAnswers:

    @pytest.fixture(scope="class")
    def warehouse(self, tmp_path_factory):
        with harness.tenant_in(tmp_path_factory.mktemp("per-unit-totals")) as built:
            yield built

    def test_total_stock_on_hand(self, warehouse):
        answer = harness.ask(warehouse, "What is our total stock on hand?")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"] for row in answer["rows"]} == {"EA", "FT"}
        card = _answered_card(answer)["answer"]
        # By the unit's name, whatever order the rows came in: the card ranks
        # nothing.
        assert card["headline"] == "Stock On Hand by unit of measure: 1,250 FT and 220 EA."
        assert card["short_value"] == ""
        # In the note, which the portal shows whatever the card holds; the
        # comparison it prints only beside a single value.
        assert _NOT_ADDED in card["scope_note"]

    def test_in_french(self, warehouse):
        # The harness asks below the turn's own wrapper, which is what sets
        # the reader's language.
        token = i18n.activate_language("fr")
        try:
            answer = harness.ask(warehouse, "Quel est notre stock total en main ?", "fr")
        finally:
            i18n.deactivate_language(token)
        card = _answered_card(answer)["answer"]
        assert card["headline"] == "Stock On Hand par unité de mesure : 1 250 FT et 220 EA."
        assert "arrive en tête" not in card["headline"]
        assert "Les quantités de différentes unités ne sont pas additionnées." in card["scope_note"]

    def test_units_sold_share_no_total_of_both(self, warehouse):
        card = _answered_card(harness.ask(warehouse, "How many units did we sell in 2025?"))
        assert card["answer"]["headline"].startswith("Units Sold by unit of measure: ")
        assert "leads" not in (card.get("insight_summary") or "")
        assert "%" not in (card.get("insight_summary") or "")
        assert card.get("anomaly_callouts") in (None, [])
        assert not card.get("decision_signal")
        # Nor offers the share as a next step.
        assert "contribution" not in [chip["id"] for chip in card.get("next_actions") or []]


class TestTheCard:

    def test_every_unit_at_zero(self):
        rows = [{"UNT_OF_MSR": unit, "AVAILABLE_QUANTITY": 0.0} for unit in ("EA", "FT")]
        card = _card(rows, "What is our available quantity?")
        assert card["headline"] == "Available Quantity is 0 in every unit of measure."

    def test_every_unit_at_zero_in_french(self):
        rows = [{"UNT_OF_MSR": unit, "AVAILABLE_QUANTITY": 0.0} for unit in ("EA", "FT")]
        card = _card(rows, "Quelle est notre quantité disponible ?", "fr")
        assert card["headline"] == "Available Quantity est à 0 dans chaque unité de mesure."

    def test_three_units_are_listed_and_the_rest_counted(self):
        rows = [{"UNT_OF_MSR": unit, "STOCK_ON_HAND": value}
                for unit, value in (("ME", 1184.6), ("EA", 11305.0), ("FT", 28640.0), ("BX", 3.0), ("PK", 7.0))]
        card = _card(rows, "What is our total stock on hand?")
        assert card["headline"] == (
            "Stock On Hand by unit of measure: 28,640 FT, 11,305 EA, 1,184.60 ME and 2 more.")

    @pytest.mark.parametrize("question,lang,headline", [
        ("What is our month-end stock on hand?", "en", "Month End Stock On Hand by unit of measure: 605 FT and 220 EA."),
        ("Quel est notre stock en fin de mois ?", "fr",
         "Month End Stock On Hand par unité de mesure : 605 FT et 220 EA."),
    ])
    def test_month_end_stock_on_hand(self, question, lang, headline):
        """The starter metric's month-end stock is stock: a month's end is a
        period, not a length of time. Read as one, "FT leads at 605"."""
        rows = [{"UNT_OF_MSR": "FT", "MONTH_END_STOCK_ON_HAND": 605.0},
                {"UNT_OF_MSR": "EA", "MONTH_END_STOCK_ON_HAND": 220.0}]
        assert _card(rows, question, lang)["headline"] == headline

    def test_equal_totals_go_by_name(self):
        rows = [{"UNT_OF_MSR": unit, "STOCK_ON_HAND": 10.0} for unit in ("FT", "EA")]
        assert _card(rows, "What is our total stock on hand?")["headline"] == (
            "Stock On Hand by unit of measure: 10 EA and 10 FT.")

    def test_in_french_one_more_is_singular(self):
        rows = [{"UNT_OF_MSR": unit, "STOCK_ON_HAND": value}
                for unit, value in (("EA", 30.0), ("FT", 20.0), ("ME", 10.0), ("BX", 5.0))]
        card = _card(rows, "Quel est notre stock total en main ?", "fr")
        assert card["headline"] == "Stock On Hand par unité de mesure : 30 EA, 20 FT, 10 ME et 1 autre."

    @pytest.mark.parametrize("question", [
        "Which unit of measure has the most stock on hand?", "Quelle unité de mesure a le plus de stock ?",
        "What is the unit of measure with the most stock on hand?", "Which of our units of measure holds the most stock?",
        "UOM with the largest quantity on hand", "Stock on hand by unit of measure from highest to lowest",
        "Quelle est l'unité de mesure avec le plus de stock ?",
        "Stock en main par unité de mesure du plus grand au plus petit"])
    def test_a_ranking_asked_for_is_a_ranking(self, question):
        assert _card(_PER_UNIT, question)["headline"] == "FT leads at 1,250."

    @pytest.mark.parametrize("question", [
        # A name and a phrase that hold a ranking's word, and rank nothing.
        "How many units did we sell to Best Supply in 2025?",
        "Units sold at the Bottom Line depot",
        "Quel est notre stock en main le plus à jour ?",
        # About something else than the units' place among them.
        "What are the main units of measure we stock?",
        "Which unit of measure has the most recent stock on hand?",
    ])
    def test_a_word_that_ranks_nothing(self, question):
        headline = _card(_PER_UNIT, question)["headline"]
        assert headline.startswith("Stock On Hand ") and "220 EA" in headline and "1" in headline
        assert "leads" not in headline and "lowest" not in headline and "arrive en tête" not in headline

    def test_money_adds_up_across_units(self):
        rows = [{"UNT_OF_MSR": "FT", "INVENTORY_VALUE": 900.0}, {"UNT_OF_MSR": "EA", "INVENTORY_VALUE": 300.0}]
        card = _card(rows, "Inventory value by unit of measure", column_formats={"INVENTORY_VALUE": "currency"})
        assert card["headline"].startswith("FT leads at")

    @pytest.mark.parametrize("measure", [
        # Money the query did not format as money, a count, a rate, an average.
        "STOCK_VALUE", "ITEM_COUNT", "FILL_RATE", "AVERAGE_DAYS_ON_HAND", "NET_SALES", "AVERAGE_ITEM_COST"])
    def test_what_is_no_quantity_is_ranked(self, measure):
        rows = [{"UNT_OF_MSR": "FT", measure: 3280.0}, {"UNT_OF_MSR": "EA", measure: 740.0}]
        card = _card(rows, f"{measure.replace('_', ' ').lower()} by unit of measure")
        assert card["headline"].startswith("FT leads at")
        assert _NOT_ADDED not in (card.get("scope_note") or "")

    def test_one_units_rows_merged_by_unit(self):
        rows = [{"UNT_OF_MSR": unit, "MONTH": month, "UNITS_SOLD": value}
                for unit, month, value in (("EA", "Jan", 5.0), ("EA", "Feb", 7.0), ("FT", "Jan", 30.0))]
        card = _card(rows, "Units sold by unit of measure")
        assert card["headline"] == "Units Sold by unit of measure: 30 FT and 12 EA."

    def test_a_unit_with_no_figure_has_no_total(self):
        rows = [{"UNT_OF_MSR": "FT", "STOCK_ON_HAND": None}, {"UNT_OF_MSR": "EA", "STOCK_ON_HAND": 5.0}]
        assert _card(rows, "What is our total stock on hand?")["headline"] == (
            "Stock On Hand by unit of measure: 5 EA and no total in FT.")

    def test_a_unit_with_no_figure_is_not_zero(self):
        rows = [{"UNT_OF_MSR": "FT", "STOCK_ON_HAND": None}, {"UNT_OF_MSR": "EA", "STOCK_ON_HAND": 0.0}]
        assert _card(rows, "What is our total stock on hand?")["headline"] == (
            "Stock On Hand by unit of measure: 0 EA and no total in FT.")

    def test_no_unit_is_one_unit_however_blank(self):
        rows = [{"UNT_OF_MSR": None, "UNITS_SOLD": 5.0}, {"UNT_OF_MSR": "", "UNITS_SOLD": 3.0},
                {"UNT_OF_MSR": "EA", "UNITS_SOLD": 2.0}]
        assert _card(rows, "Units sold by unit of measure")["headline"] == (
            "Units Sold by unit of measure: 2 EA and 8 with no unit.")


class TestTheBrief:

    @staticmethod
    def _brief(rows: list[dict], question: str) -> dict:
        from core.insight import compute_data_brief

        return compute_data_brief(rows, question)

    def test_no_leader_share_or_total_across_units(self):
        brief = self._brief(_PER_UNIT, "What is our total stock on hand?")
        assert "category_breakdown" not in brief
        assert brief["per_unit"] == {
            "unit_column": "UNT_OF_MSR", "value_column": "STOCK_ON_HAND",
            "totals": [{"unit": "FT", "value": 1250.0}, {"unit": "EA", "value": 220.0}], "missing": []}
        summary = brief["numeric_summaries"]["STOCK_ON_HAND"]
        assert not {"total", "mean", "median", "std_dev", "min", "max"} & set(summary)

    def test_a_ranking_asked_for_keeps_its_breakdown(self):
        brief = self._brief(_PER_UNIT, "Which unit of measure has the most stock on hand?")
        assert brief["category_breakdown"]["top_5"][0]["label"] == "FT"
        assert "per_unit" not in brief


class TestTheAnalysis:
    """The why, explain and analyse the reader asks of the card, and the
    follow-ups it offers: written from each unit's total, and from nothing
    across them."""

    @staticmethod
    def _brief() -> dict:
        from core.insight import compute_data_brief

        return compute_data_brief(_PER_UNIT, "What is our total stock on hand?")

    def test_the_prompt_states_each_units_total(self):
        from core.insight import _format_brief_for_prompt

        # This read a total the brief no longer holds, and every "why" asked of
        # a result in several units ended in a KeyError.
        prompt = _format_brief_for_prompt(self._brief())
        assert "FT=1250.0, EA=220.0" in prompt
        assert "never add them, rank them, compare them or share them out" in prompt

    @pytest.mark.parametrize("action", ["why", "explain", "analyze", "compare"])
    def test_every_action_is_given_the_totals(self, action):
        from core.insight import _format_brief_for_prompt, build_action_contract

        contract = build_action_contract(action, "What is our total stock on hand?", self._brief())
        assert contract["per_unit_stats"]["totals"] == [{"unit": "FT", "value": 1250.0}, {"unit": "EA", "value": 220.0}]
        assert not (contract.get("comparison_stats") or {}).get("leader")
        assert "FT=1250.0" in _format_brief_for_prompt(contract)

    def test_the_drill_down_prompt(self):
        from core.insight import build_drilldown_prompt

        _, user = build_drilldown_prompt(
            "What is our total stock on hand?", "Why does FT have the most?", "SELECT 1", self._brief(), "azure_sql", "")
        assert "FT=1250.0" in user

    def test_the_follow_ups_offer_no_share(self):
        from core.stat_signals import compute_signals

        kinds = {signal["type"] for signal in compute_signals(_PER_UNIT)}
        assert not kinds & {"group_imbalance", "pareto", "high_variance", "outlier_present", "below_avg_gap",
                            "skewed_right", "low_variance"}

    def test_the_computed_summary(self):
        from core.response_builder import _regulated_analysis_fallback

        body = _regulated_analysis_fallback("analyze", _PER_UNIT)["body"]
        assert "%" not in body and "deviation" not in body

    @pytest.mark.parametrize("rows", [
        _PER_UNIT,
        # Items ranked by units sold, each in its own unit.
        [{"ITEM": "COPPER PIPE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 900.0},
         {"ITEM": "BRASS ELBOW", "UNT_OF_MSR": "EA", "UNITS_SOLD": 160.0},
         {"ITEM": "BALL VALVE", "UNT_OF_MSR": "EA", "UNITS_SOLD": 40.0},
         {"ITEM": "PEX TUBE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 10.0}],
    ])
    def test_a_regulated_tenant_is_given_each_units_total(self, rows):
        """With every finding across units withheld, the computed summary said
        the values were "close to evenly spread" -- as false as the share it
        replaced -- and never gave a total."""
        from core.response_builder import _regulated_analysis_fallback

        analysis = _regulated_analysis_fallback("why", rows)
        assert "each total is in its own unit of measure" in analysis["body"]
        assert analysis["body"].split(":")[0] in {"1,250 FT and 220 EA", "910 FT and 200 EA"}
        assert "evenly" not in analysis["body"] and analysis["rows_sent_to_llm"] == 0

    def test_and_what_the_money_beside_them_shows(self):
        """Money adds up across units: its findings stand beside each unit's
        total, and the summary that follows the answer still carries them."""
        from core.response_builder import _regulated_analysis_fallback

        rows = [{"ITEM": "COPPER PIPE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 900.0, "NET_SALES": 5400.0},
                {"ITEM": "BRASS ELBOW", "UNT_OF_MSR": "EA", "UNITS_SOLD": 160.0, "NET_SALES": 800.0},
                {"ITEM": "BALL VALVE", "UNT_OF_MSR": "EA", "UNITS_SOLD": 40.0, "NET_SALES": 600.0},
                {"ITEM": "PEX TUBE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 10.0, "NET_SALES": 400.0}]
        analysis = _regulated_analysis_fallback("why", rows)
        assert analysis["body"].startswith("910 FT and 200 EA: each total is in its own unit of measure")
        assert "COPPER PIPE alone accounts for 75% of the total Net Sales" in analysis["body"]
        assert analysis["finding_kinds"]

    @pytest.mark.parametrize("first,values,more", [
        # A date, and the trend of a quantity kept in feet one day and in
        # eaches the next, which is none: "Units Shipped fell 95.6%".
        ("SHIPPED_DATE", (20250101, 20250102, 20250103), ""),
        # A count of orders adds up across units, and is found as it is.
        ("ORDERS_SHIPPED", (3, 12, 1), " FT alone accounts for 75% of the total Orders Shipped (12). Orders Shipped "
                                       "ranges from 1 to 12 across the result."),
        # A date shipped on: "40,500,204 EA".
        ("SHIPPED_ON_DATE", (20250101, 20250102, 20250103), ""),
    ])
    def test_a_date_or_a_count_before_it_is_no_quantity(self, first, values, more):
        """A ship date keyed yyyymmdd, or a count of orders, in the first
        numeric column: "40,500,413 EA and 20,250,131 FT" was no total of anything."""
        from core.response_builder import _regulated_analysis_fallback

        rows = [{"UNT_OF_MSR": unit, first: value, "UNITS_SHIPPED": units}
                for unit, value, units in zip(("EA", "FT", "EA"), values, (30.0, 900.0, 10.0))]
        body = _regulated_analysis_fallback("why", rows)["body"]
        assert body == ("900 FT and 40 EA: each total is in its own unit of measure, so none is ranked against, "
                        "compared with or added to another." + more), body

    def test_nor_is_the_customer_sold_to(self):
        """Customer numbers were totalled per unit: "2,004 EA and 2,002 FT"."""
        from core.response_builder import _regulated_analysis_fallback

        rows = [{"UNT_OF_MSR": unit, "SOLD_TO_ID": customer, "UNITS_SHIPPED": units}
                for unit, customer, units in zip(("EA", "FT", "EA"), (1001, 1002, 1003), (30.0, 900.0, 10.0))]
        body = _regulated_analysis_fallback("why", rows)["body"]
        assert body.startswith("900 FT and 40 EA: each total is in its own unit of measure"), body

    def test_nor_is_a_trend_read_across_them(self):
        """Receipts one month in feet and the next in eaches "peaked at 900
        and bottomed at 40", and the summary was sent after the answer."""
        from core.response_builder import _regulated_analysis_fallback

        rows = [{"MONTH": month, "ITEM": item, "UNT_OF_MSR": unit, "UNITS_RECEIVED": units}
                for month, item, unit, units in (("2025-01", "COPPER PIPE", "FT", 900.0),
                                                 ("2025-02", "BRASS ELBOW", "EA", 160.0),
                                                 ("2025-03", "BALL VALVE", "EA", 40.0),
                                                 ("2025-04", "PEX PIPE", "FT", 700.0))]
        fallback = _regulated_analysis_fallback("why", rows)
        assert (fallback["body"], fallback["finding_kinds"]) == (
            "1,600 FT and 200 EA: each total is in its own unit of measure, so none is ranked against, compared "
            "with or added to another.", [])

    def test_nor_by_the_card_the_chips_or_the_signals(self):
        """The same receipts "trended down 22.2% from 2025-01 to 2025-04",
        with a "Biggest gain: +1,650.0%" callout, a chip asking what drove the
        drop, and a "downward trend" signal."""
        from core.insight import compute_data_brief
        from core.response_builder import (
            _build_anomaly_callouts, _build_insight_summary, compute_chip_eligibility, summarize_result_context)
        from core.stat_signals import compute_signals

        rows = [{"MONTH": month, "UNT_OF_MSR": unit, "UNITS_RECEIVED": units}
                for month, unit, units in (("2025-01", "FT", 900.0), ("2025-02", "EA", 160.0),
                                           ("2025-03", "EA", 40.0), ("2025-04", "FT", 700.0))]
        question = "Units received by month"
        context = summarize_result_context(rows, question, "SELECT MONTH, UNT_OF_MSR, SUM(Q) FROM F GROUP BY 1, 2")
        brief = compute_data_brief(rows, question, result_scope=context.get("result_scope"), context=context)
        assert context.get("pct_change") is None and not context.get("comparison_stats")
        assert not brief.get("time_series")
        assert _build_insight_summary(rows, context, brief) == ""
        assert _build_anomaly_callouts(brief) == []
        assert {chip["id"] for chip in compute_chip_eligibility(context, brief)} == {"download_csv"}
        assert not [signal for signal in compute_signals(rows) if signal["type"] in {"temporal", "flat_trend"}]

    @pytest.mark.parametrize("lang,words", [("en", "more than one unit of measure"), ("fr", "plusieurs unités de mesure")])
    def test_the_narrative_says_why_nothing_is_compared(self, lang, words):
        from core.analysis_evidence import build_evidence
        from core.analysis_narrative import build_narrative

        (sentence,) = build_narrative(build_evidence(_PER_UNIT), lang=lang).sentences
        assert words in sentence and "evenly" not in sentence and "uniforme" not in sentence

    def test_the_fallback_gives_no_spread_of_zero(self):
        """The spread is left out across units, and the fallback card printed
        "Spread from highest to lowest returned value: 0" in its place."""
        from core.response_builder import build_analysis_response, summarize_result_context

        rows = [{"ITEM": "COPPER PIPE", "UNT_OF_MSR": "FT", "UNITS_SOLD": 900.0},
                {"ITEM": "BRASS ELBOW", "UNT_OF_MSR": "EA", "UNITS_SOLD": 40.0}]
        context = summarize_result_context(rows, "Which items sold the most units?")
        bullets = build_analysis_response("analyze", context).get("bullets") or []
        assert not any("Spread" in bullet for bullet in bullets)

    @pytest.mark.parametrize("action", ["explain", "analyze", "compare", "why"])
    def test_the_card_when_the_model_path_fails(self, action):
        from core.response_builder import build_analysis_response, summarize_result_context

        context = summarize_result_context(_PER_UNIT, "What is our total stock on hand?")
        body = build_analysis_response(action, context)["body"]
        assert body.startswith("1,250 FT and 220 EA: each total is in its own unit of measure")
        assert "ahead" not in body and "%" not in body


class TestTheRule:

    @pytest.mark.parametrize("question", [
        "What is our total stock on hand?", "Quel est notre stock total en main ?",
        "How many units did we sell in 2022?", "Stock on hand by unit of measure",
        "How many units did we sell to Best Supply in 2025?", "Quel est notre stock en main le plus à jour ?",
        # A superlative about something else, beside the units: nothing ranks them.
        "What is our most recent stock on hand by unit of measure?",
        "Most up-to-date stock on hand by unit of measure",
        "Stock on hand by unit of measure for items with at least one sale",
        "Units sold by unit of measure, at most 30 days ago",
        "What sort of stock do we hold, by unit of measure?",
        "Stock on hand at our largest warehouse, by unit of measure",
        "Quel est notre stock en main le plus à jour par unité de mesure ?",
        # One after a preposition, and "units" alone, are about something else.
        "Which units of measure have at least 100 in stock?",
        "Which units of measure have at most 10 on hand?",
        "What units of measure do we hold at our largest warehouse?",
        "Stock on hand by unit of measure for the items with the most units sold",
        "Stock on hand by unit of measure for our top-ranked suppliers",
        "Quelle unité de mesure pour l'entrepôt le plus grand ?",
        "Quelles unités de mesure ont au moins 100 en stock ?",
        "Stock on hand by unit of measure for units with the highest turnover",
        "Stock on hand by unit of measure for the items top-ranked by sales",
        # The unit an item is ordered in, not an order to sort by.
        "Stock on hand by order UOM", "Quantity ordered by order UOM"])
    def test_totals_per_unit(self, question):
        from core.units_of_measure import rows_per_unit

        assert rows_per_unit(_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", question) is True

    @pytest.mark.parametrize("rows,label,measure,question,fmt", [
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which unit of measure has the most stock?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Quelle unité de mesure a le plus de stock ?", ""),
        # Other ways to ask for the units in order.
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which unit of measure do we sell the most?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Largest unit of measure by stock on hand", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock on hand by unit of measure, sorted descending", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Classement des unités de mesure par stock", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Top 2 units of measure by stock on hand", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which units do we hold the most stock in?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Order the units of measure by stock on hand", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which unit of measure leads in stock on hand?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Units of measure ranked by stock on hand", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Ranking of units of measure by stock on hand", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Sort units of measure by stock", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which unit of measure accounts for the most stock?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Which unit of measure do we currently hold the most stock in?",
         ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Quelle unité de mesure a le stock en main le plus élevé ?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock on hand by unit of measure from the highest to the lowest",
         ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock on hand by unit of measure, highest-to-lowest", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock on hand by unit of measure from high to low", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock en main par unité de mesure du plus haut au plus bas", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Stock en main par unité de mesure du plus bas au plus élevé", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND",
         "Stock en main par unité de mesure de la plus grande à la plus petite", ""),
        (_PER_UNIT[:1], "UNT_OF_MSR", "STOCK_ON_HAND", "What is our total stock on hand?", ""),
        ([{"ITEM": "COPPER PIPE", "STOCK_ON_HAND": 9.0}, {"ITEM": "BRASS ELBOW", "STOCK_ON_HAND": 4.0}],
         "ITEM", "STOCK_ON_HAND", "What is our total stock on hand?", ""),
        (_PER_UNIT, "UNT_OF_MSR", "STOCK_ON_HAND", "Inventory value by unit of measure", "currency"),
        ([{"UNT_OF_MSR": "FT", "ITEM_COUNT": 3.0}, {"UNT_OF_MSR": "EA", "ITEM_COUNT": 2.0}],
         "UNT_OF_MSR", "ITEM_COUNT", "How many items by unit of measure?", ""),
    ])
    def test_not_totals_per_unit(self, rows, label, measure, question, fmt):
        from core.units_of_measure import rows_per_unit

        assert rows_per_unit(rows, label, measure, question, measure_format=fmt) is False

    @pytest.mark.parametrize("column,quantity", [
        ("STOCK_ON_HAND", True), ("UNITS_SOLD", True), ("TOTAL_QTY", True), ("StockOnHandQty", True),
        ("Quantité disponible", True), ("SALES_QTY", True),
        ("ON_HAND", True), ("TOTAL_SOLD", True), ("Unités vendues", True),
        ("STOCK_VALUE", False), ("INVENTORY_VALUE", False), ("ITEM_COUNT", False), ("FILL_RATE", False),
        ("AVERAGE_DAYS_ON_HAND", False), ("NET_SALES", False), ("UNIT_PRICE", False), ("STOCK_ON_HAND_USD", False),
        # Time, a ratio of two quantities, and the hand of a product's name.
        ("WEEKS_ON_HAND", False), ("STOCK_COVER_WEEKS", False), ("UNITS_PER_CASE", False),
        ("DAYS_OF_SUPPLY", False), ("MONTHS_OF_SUPPLY", False), ("QTY_PER_ORDER", False),
        ("HAND_TOOLS_SALES", False),
        # A period is no length of time: the stock and units of one.
        ("MONTH_END_STOCK_ON_HAND", True), ("UNITS_SOLD_THIS_MONTH", True), ("WEEK_END_QTY", True),
        ("Stock fin de mois", True), ("PER_END_QTY", True), ("OPEN_ORDER_QTY_PER_WAREHOUSE", True),
        ("COVER_STOCK_QTY", True),
        # A date, key or code named after a movement, and a count of documents.
        ("SHIPPED_DATE", False), ("RECEIVED_DT_KEY", False), ("SOLD_TO_CUSTOMER_NO", False),
        ("ORDERED_BY_USER_ID", False), ("ORDERS_SHIPPED", False), ("LINES_RECEIVED", False),
        # Only where they end the name, and not "to date" or "on time".
        ("UNITS_SOLD_TO_DATE", True), ("QTY_SHIPPED_ON_TIME", True), ("ON_TIME_QTY", True),
        ("QTY_NO_CHARGE", True), ("NUM_UNITS", True),
        # Stock cover and months of stock are lengths of time, and a quantity
        # per transaction, shipment or case a ratio.
        ("STOCK_COVER", False), ("STOCK_COVER_WKS", False), ("Couverture de stock (mois)", False),
        ("Mois de stock", False), ("UNITS_PER_TRANSACTION", False), ("QTY_PER_SHIPMENT", False),
        ("PER_CASE_QTY", False),
        # "To" and "on" before a date or an identifier are no "to date": a
        # ship date and a customer number were summed per unit of measure.
        ("SHIPPED_ON_DATE", False), ("RECEIVED_ON_DT", False), ("ShippedOnDate", False), ("SOLD_TO_ID", False),
        ("SOLD_TO_NO", False), ("SoldToId", False),
        # A window a quantity is counted over is no length of time, stock
        # to cover it is stock, and a month named is a period.
        ("UNITS_SOLD_LAST_4_WKS", True), ("QTY_SOLD_13_WKS", True), ("UNITS_SOLD_3_MTHS", True),
        ("Unités vendues 4 dernières semaines", True), ("COVER_STOCK", True), ("Stock du mois de mars", True),
    ])
    def test_a_quantity_of_goods(self, column, quantity):
        from core.units_of_measure import is_quantity_measure

        assert is_quantity_measure(column) is quantity
