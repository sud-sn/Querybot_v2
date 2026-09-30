"""
A French reply names what it counts in French, and writes its numbers in French.

The card's sentences name the column its rows are counted by, and took that
name from the English label a column's code expands to: "North arrive en
tête avec 900 (92.8 % du total), sur 3 regions", "sur 3 item groups", "Sur 2
categories, 1 a augmenté". The share beside it was the raw number, with a
point for its decimal. The noun is now French where the product knows the
column's word (core.i18n.count_noun), singular or plural as the count is, and
otherwise the sentence's own French word; the share is written with a comma.
English reads as it did.
"""

from __future__ import annotations

import pytest

from core import i18n
from core.response_builder import build_assistant_response

RANKING = [{"region": "North", "revenue": 900}, {"region": "South", "revenue": 50},
           {"region": "East", "revenue": 20}]
GROUPS = [{"item_group": group, "revenue": value} for group, value in (("PIPE", 900), ("VALVES", 50), ("FITTINGS", 20))]
GRADES = [{"widget_grade": grade, "revenue": value} for grade, value in (("A", 900), ("B", 50), ("C", 20))]
PERIOD = [{"category": "Pumps", "revenue_2025": 100, "revenue_2026": 150},
          {"category": "Valves", "revenue_2025": 200, "revenue_2026": 180}]


@pytest.fixture
def french():
    token = i18n.activate_language("fr")
    try:
        yield
    finally:
        i18n.deactivate_language(token)


def _card(rows: list[dict], question: str, period: bool = False) -> dict:
    return build_assistant_response(
        question=question, rows=rows, sql="SELECT * FROM t", duration_ms=1,
        display_context={"period_comparison": {"labels": ["2025", "2026"]}} if period else None)


class TestTheNoun:

    @pytest.mark.parametrize("label,count,noun", [
        ("Item Group", 5, "groupes d'articles"), ("Warehouse Name", 3, "entrepôts"),
        ("Warehouse Name", 1, "entrepôt"), ("Unit of Measure", 2, "unités de mesure"),
        ("Region", 3, "régions"), ("Category", 2, "catégories"),
        # A label written as a plural, with a hyphen or a typographic
        # apostrophe, or without its accents, is still the noun it names.
        ("Regions", 3, "régions"), ("Clients", 3, "clients"), ("Sub-Category", 3, "sous-catégories"),
        ("Unité d\u2019affaires", 2, "unités d'affaires"), ("Entrepot", 3, "entrepôts"),
        # Below two, French counts in the singular.
        ("Warehouse", 0, "entrepôt"), ("Warehouse", 1.5, "entrepôt")])
    def test_in_french(self, label, count, noun):
        from core.i18n import count_noun

        assert count_noun(label, count, "answer.entries", "fr") == noun

    def test_a_word_it_does_not_know_is_the_sentences_own(self):
        from core.i18n import count_noun

        assert count_noun("Widget Grade", 3, "answer.entries", "fr") == "entrées"
        # And agrees with its count: "Sur 1 groupes" was the answer's.
        assert [count_noun("", count, "answer.groups", "fr") for count in (1, 3)] == ["groupe", "groupes"]
        assert count_noun("Widget Grade", 1, "answer.entries", "fr") == "entrée"
        assert [count_noun("", count, "answer.groups", "en") for count in (1, 3)] == ["groups", "groups"]

    @pytest.mark.parametrize("label,noun", [
        ("Warehouse Name", "warehouse names"), ("Region", "regions"), ("Category", "categories"),
        ("Item Group", "item groups")])
    def test_english_is_unchanged(self, label, noun):
        from core.i18n import count_noun

        assert count_noun(label, 3, "answer.entries", "en") == noun


class TestTheCard:

    @pytest.mark.parametrize("rows,question,summary", [
        (RANKING, "revenue by region", "North arrive en tête avec 900 (92,8 % du total), sur 3 régions."),
        (GROUPS, "revenue by item group",
         "PIPE arrive en tête avec 900 (92,8 % du total), sur 3 groupes d'articles."),
        (GRADES, "revenue by widget grade", "A arrive en tête avec 900 (92,8 % du total), sur 3 entrées."),
    ])
    def test_a_ranking(self, french, rows, question, summary):
        assert _card(rows, question)["insight_summary"] == summary

    def test_a_second_measures_share(self, french):
        rows = [{"WAREHOUSE": warehouse, "UNITS_SOLD": units, "NET_SALES": sales}
                for warehouse, units, sales in (("NORTH", 900.0, 1234.5), ("SOUTH", 50.0, 20.0), ("EAST", 20.0, 10.0))]
        assert "NORTH arrive en tête pour Net Sales avec 1\u202f234,50 (97,6 % du total)." in _card(
            rows, "units sold and net sales by warehouse")["insight_summary"]

    def test_a_concentration(self, french):
        rows = [{"WAREHOUSE": f"W{n}", "NET_SALES": sales}
                for n, sales in enumerate((5000.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0), 1)]
        assert "Les 3 premières entrées représentent 94,5 % du total — forte concentration" in [
            callout["message"] for callout in _card(rows, "net sales by warehouse")["anomaly_callouts"]]

    def test_its_callout(self, french):
        callouts = [callout["message"] for callout in _card(RANKING, "revenue by region")["anomaly_callouts"]]
        assert callouts == ["North détient 92,8 % du total"]

    def test_the_period_note(self, french):
        assert _card(PERIOD, "revenue 2025 vs 2026", period=True)["insight_summary"].startswith(
            "Sur 2 catégories, 1 a augmenté et 1 a diminué entre 2025 et 2026.")

    @pytest.mark.parametrize("rows,question,summary", [
        (RANKING, "revenue by region", "North leads at 900 (92.8% of total) across 3 regions."),
        (GROUPS, "revenue by item group", "PIPE leads at 900 (92.8% of total) across 3 item groups."),
        (GRADES, "revenue by widget grade", "A leads at 900 (92.8% of total) across 3 widget grades."),
    ])
    def test_english_is_unchanged(self, rows, question, summary):
        card = _card(rows, question)
        assert card["insight_summary"] == summary
        assert [callout["message"] for callout in card["anomaly_callouts"]] == [
            f"{rows[0][next(iter(rows[0]))]} holds 92.8% of the total"]

    def test_a_share_past_a_thousand_percent_is_not_grouped(self):
        """A total shrunk by a negative row makes a share of 14333.9%, which
        English wrote ungrouped and still does."""
        card = _card([{"REGION": "NORTH", "NET_SALES": 14333.9}, {"REGION": "SOUTH", "NET_SALES": -14233.9}],
                     "net sales by region")
        assert "(14333.9% of total)" in card["insight_summary"]
        assert [callout["message"] for callout in card["anomaly_callouts"]] == ["NORTH holds 14333.9% of the total"]
