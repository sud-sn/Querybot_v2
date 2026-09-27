"""
A calendar's day names are read in the reader's language.

A label kept twice -- EnglishMonthName and FrenchMonthName -- is shown to a
French reader in French. A twin was recognised only where the name ends in a
label's word (NAME, DESC, LABEL...). A calendar names its weekday
EnglishDayNameOfWeek and FrenchDayNameOfWeek: the label's word is followed by
what it names the unit of, the two were not seen as twins, and "Ventes par
jour de la semaine" came back "Sunday arrive en tete".

A name that ends in a label's word followed by "of" and a unit is a label too:
DAY_NAME_OF_WEEK is the week's day's name.

tests/star_harness.py keeps its calendar's day names in English and French.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("day-names")) as built:
        yield built


class TestTheProductAnswers:

    def test_in_french(self, warehouse):
        answer = star.ask(warehouse, "Ventes par jour de la semaine", "fr")
        assert [row["PERIOD"] for row in answer["rows"]] == [
            "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi"]

    def test_in_english(self, warehouse):
        answer = star.ask(warehouse, "Sales by day of the week")
        assert [row["PERIOD"] for row in answer["rows"]] == [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


class TestTheRule:

    @pytest.mark.parametrize("columns,twins", [
        (["EnglishDayNameOfWeek", "FrenchDayNameOfWeek", "DayNumberOfWeek"],
         [("EnglishDayNameOfWeek", "FrenchDayNameOfWeek")]),
        (["DAY_NAME_OF_WEEK", "DAY_NAME_OF_WEEK_FR"], [("DAY_NAME_OF_WEEK", "DAY_NAME_OF_WEEK_FR")]),
        (["EnglishMonthName", "FrenchMonthName"], [("EnglishMonthName", "FrenchMonthName")]),
        (["EnglishDayNumberOfWeek", "FrenchDayNumberOfWeek"], []),
        (["START_OF_WEEK", "START_OF_WEEK_FR"], []),
        (["DESC_TYPE_CODE", "DESC_TYPE_CODE_FR"], []),
    ])
    def test_the_twins_of_a_calendar(self, columns, twins):
        from core.label_language import language_twins

        assert [(twin["base"], twin["twin"]) for twin in language_twins(columns)] == twins
