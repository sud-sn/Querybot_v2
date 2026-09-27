"""
A grain that comes round again groups by the calendar's own day, week or month.

"Items created by day of the week" asks for seven numbers; the product gave one
a day, since "by day" read as a daily series, and "by weekday" as a weekly one.
"By week of the year" gave the weeks of every year one after another, and "by
month of the year" twelve months a year -- each right only when the data held
a single year. On the sample tenant the weekday question was answered with
a count for every creation date.

Now "by day of the week" (or weekday), "by week of the year" (or week number)
and "by month of the year" -- and their French, "par jour de la semaine",
"par semaine de l'année" -- are read as the cycle they name. The governed
compiler groups by the calendar's own number for it and shows its name, in
the reader's language where the calendar keeps two, in the calendar's order.
A month is read from the date itself where the calendar keeps no month, as a
period key is; a weekday's or a week's number only from the calendar, since a
server's settings number them differently. A level is not compiled by
weekday, and the model is told to group by the cycle where it writes the
query.

A synthetic tenant (tests/answer_harness.py) whose calendar keeps the day of the
week, its English and French names and the week of the year.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import answer_harness as harness

_DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
         "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("repeating-grains")) as built:
        yield built


def _created() -> list[tuple[int, dt.date]]:
    return [(row[1], dt.datetime.strptime(str(row[3]), "%Y%m%d").date()) for row in harness.STOCK]


def _cycle(key, label, distinct_items: bool) -> list[tuple]:
    """(label, count) in the calendar's order."""
    groups: dict = {}
    for item, day in _created():
        groups.setdefault(key(day), (label(day), []))[1].append(item)
    return [(name, len(set(items)) if distinct_items else len(items)) for _, (name, items) in sorted(groups.items())]


def _series(answer: dict, measure: str) -> list[tuple]:
    assert answer["model_wrote_sql"] is False
    assert {key for row in answer["rows"] for key in row} == {"PERIOD", measure}
    return [(row["PERIOD"], row[measure]) for row in answer["rows"]]


class TestTheProductAnswers:

    def test_items_by_day_of_the_week(self, warehouse):
        answer = harness.ask(warehouse, "Items created by day of the week")
        assert _series(answer, "ITEM_COUNT") == _cycle(
            lambda day: day.isoweekday(), lambda day: _DAYS["en"][day.weekday()], distinct_items=True)

    def test_records_by_weekday(self, warehouse):
        answer = harness.ask(warehouse, "How many item-warehouse records were created by weekday?")
        assert _series(answer, "ITEM_WAREHOUSE_RECORD_COUNT") == _cycle(
            lambda day: day.isoweekday(), lambda day: _DAYS["en"][day.weekday()], distinct_items=False)

    def test_in_french(self, warehouse):
        answer = harness.ask(warehouse, "Combien de fiches ont été créées par jour de la semaine ?", "fr")
        assert _series(answer, "RECORD_COUNT") == _cycle(
            lambda day: day.isoweekday(), lambda day: _DAYS["fr"][day.weekday()], distinct_items=False)

    def test_by_week_of_the_year(self, warehouse):
        answer = harness.ask(warehouse, "Items created by week of the year in 2025")
        assert _series(answer, "ITEM_COUNT") == _cycle(
            lambda day: day.isocalendar()[1], lambda day: day.isocalendar()[1], distinct_items=True)

    def test_by_month_of_the_year(self, warehouse):
        answer = harness.ask(warehouse, "Items created by month of the year")
        assert _series(answer, "ITEM_COUNT") == _cycle(
            lambda day: day.month, lambda day: _MONTHS[day.month - 1], distinct_items=True)

    def test_a_month_of_the_year_from_a_period_key(self, warehouse):
        # The monthly snapshot's period is a yyyymm key with no calendar.
        answer = harness.ask(warehouse, "Units sold by month of the year")
        assert answer["model_wrote_sql"] is False
        sold: dict = {}
        for _whs, item, period, units, *_rest in harness.MOVES:
            key = (period % 100, harness.ITEMS[item][3])
            sold[key] = sold.get(key, 0) + units
        assert {(row["PERIOD"], row["UNT_OF_MSR"]): row["UNITS_SOLD"] for row in answer["rows"]} == sold

    def test_a_month_series_is_unchanged(self, warehouse):
        answer = harness.ask(warehouse, "Items created by month in 2025")
        assert [row["PERIOD"] for row in answer["rows"]] == sorted({day.replace(day=1) for _i, day in _created()})

    def test_a_level_is_not_compiled_by_weekday(self, warehouse):
        answer = harness.ask(warehouse, "Stock on hand by day of the week")
        assert answer["model_wrote_sql"] is True


class TestTheReading:

    @pytest.mark.parametrize("question,cycle", [
        ("Items created by day of the week", "day_of_week"),
        ("Units sold by weekday", "day_of_week"),
        ("Sales per days of the week", "day_of_week"),
        ("Items created by week of the year in 2025", "week_of_year"),
        ("Receipts by week number", "week_of_year"),
        ("Units sold by month of the year", "month_of_year"),
        ("Units sold on weekdays", ""),
        ("Units sold by week", ""),
        ("Items created by day", ""),
    ])
    def test_the_cycle(self, question, cycle):
        from core.contextual_dates import requested_cycle

        assert requested_cycle(question) == cycle

    @pytest.mark.parametrize("question,cycle", [
        ("Articles créés par jour de la semaine", "day_of_week"),
        ("Articles créés par semaine de l'année en 2025", "week_of_year"),
        ("Unités vendues par mois de l'année", "month_of_year"),
    ])
    def test_in_french(self, question, cycle):
        from core.contextual_dates import requested_cycle
        from core.question_normalizer import canonical_question

        assert requested_cycle(canonical_question(question, "fr")) == cycle

    def test_a_weekday_is_a_days_grain(self):
        from core.contextual_dates import requested_temporal_grain

        assert requested_temporal_grain("Units sold by weekday") == "day"


class TestTheCalendar:

    def test_the_weekday_attributes(self):
        from core.contextual_dates import infer_calendar_attributes

        columns = {"MART.DT_DMS": {"DMS_DT": "date", "DAY_OF_WK": "int", "DAY_NM": "nvarchar", "WK_OF_YR": "int"}}
        attributes = infer_calendar_attributes("MART.DT_DMS", columns)
        assert (attributes["day_of_week"], attributes["day_name"], attributes["week"]) == ("DAY_OF_WK", "DAY_NM", "WK_OF_YR")


_SALES_DATE = {"fact_table": "MART.SALES_FCT", "fact_column": "SLS_DT_DMS_KEY",
               "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY", "date_value_column": "DMS_DT",
               "date_key_type": "surrogate_fk", "business_role": "Sales date", "name": "Sales date",
               "temporal_grain": "day",
               "calendar_attributes": {"date": "DMS_DT", "year": "YR", "month_number": "MTH",
                                       "day_of_week": "DAY_OF_WK", "day_name": "DAY_NM"}}


class TestTheModelIsToldTheSame:

    def test_group_by_the_day_of_the_week(self):
        from core.contextual_dates import build_contextual_date_plan
        from core.query_pipeline import _governed_date_anchor_repair_lines
        from core.semantic_planner import format_semantic_field_plan

        plan = build_contextual_date_plan(_SALES_DATE, "net sales by day of the week", snapshot=False)
        prompt = format_semantic_field_plan(plan)
        assert "REQUIRED GROUPING: group by the day of week, business_date.[DAY_OF_WK] and show" in prompt
        assert "REQUIRED DAY BUCKET" not in prompt
        assert "Group by the day of week of" in prompt
        assert "group by the day of week of" in _governed_date_anchor_repair_lines(plan)

    def test_a_month_is_still_a_bucket(self):
        from core.contextual_dates import build_contextual_date_plan
        from core.semantic_planner import format_semantic_field_plan

        prompt = format_semantic_field_plan(build_contextual_date_plan(_SALES_DATE, "net sales by month", snapshot=False))
        assert "REQUIRED MONTH BUCKET" in prompt and "REQUIRED GROUPING" not in prompt


class TestTheCompiler:

    @staticmethod
    def _refs(cycle: str, attributes: dict) -> tuple[str, str]:
        from core.pipeline_helpers import _cycle_refs

        policy = {"cycle": cycle, "calendar_attributes": attributes, "dimension_table": "MART.DT_DMS"}
        return _cycle_refs(policy, "sales_date", {}, "azure_sql", "sales_date.[DMS_DT]")

    def test_a_weekday_only_from_the_calendar(self):
        assert self._refs("day_of_week", {"date": "DMS_DT"}) == ("", "")

    def test_a_week_only_from_the_calendar(self):
        assert self._refs("week_of_year", {"date": "DMS_DT"}) == ("", "")

    def test_a_month_from_the_date(self):
        assert self._refs("month_of_year", {"date": "DMS_DT"}) == ("MONTH(sales_date.[DMS_DT])", "")
