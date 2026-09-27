"""
A calendar's time attributes are never measures.

"Items created by week of year in 2025" was refused on the sample tenant, where
"Items created by week of the year in 2025" was answered, and French "par
semaine de l'année" -- read as "by week of year" -- was refused with it. The
field planner found the calendar's week-number column by its own name, "week
of year", and bound it as a measure the answer had to compute; the compiler,
asked to count the items and to add up the weeks, declined. With one word
more, the planner matched the column less well, left it optional, and the
cycle was answered as it should be.

A synthetic tenant (tests/answer_harness.py) whose calendar keeps the week of
the year; its planner is made to bind that week as the sample's did.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import pytest

from tests import answer_harness as harness

_WEEK = {"term": "week of year", "table": "WH.MART.DT_DMS", "column": "WK_OF_YR", "role": "measure"}


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("calendar-attributes")) as built:
        yield built


def _planner_binding_the_week(original):
    """The field planner, binding the calendar's week number as a measure
    wherever the question names it by the column's own name."""
    def plan(question, *args, **kwargs):
        found = original(question, *args, **kwargs)
        if "week of year" not in str(question).lower():
            return found
        found = dict(found or {})
        found["enabled"] = True
        found["fields"] = [*(found.get("fields") or []), dict(_WEEK)]
        return found
    return plan


def _items_by_week_of_2025() -> list[tuple]:
    weeks: dict = {}
    for row in harness.STOCK:
        day = dt.datetime.strptime(str(row[3]), "%Y%m%d").date()
        if day.year == 2025:
            weeks.setdefault(day.isocalendar()[1], set()).add(row[1])
    return [(week, len(items)) for week, items in sorted(weeks.items())]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Items created by week of year in 2025", "en"),
        ("Items created by week of the year in 2025", "en"),
    ])
    def test_items_by_week_of_year(self, warehouse, question, lang):
        from core import query_pipeline

        with patch.object(query_pipeline, "build_semantic_field_plan",
                          _planner_binding_the_week(query_pipeline.build_semantic_field_plan)):
            answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert [(row["PERIOD"], row["ITEM_COUNT"]) for row in answer["rows"]] == _items_by_week_of_2025()


def _calendar_binding() -> dict:
    """A date role on the calendar, enriched as the pipeline enriches it."""
    from core.contextual_dates import enrich_date_binding_calendar_attributes

    columns = {"WH.MART.DT_DMS": {"DT_DMS_KEY": "int", "DMS_DT": "date", "YR": "int", "QTR_NO": "int", "MTH": "int",
                                  "MTH_NM": "nvarchar", "WK_OF_YR": "int", "DAY_OF_WK": "int", "DAY_NM": "nvarchar",
                                  "HDY_FLG": "int"}}
    return enrich_date_binding_calendar_attributes(
        {"dimension_table": "MART.DT_DMS", "date_value_column": "DMS_DT"}, columns)


def _demoted(*fields, bindings=None) -> list:
    from core.analytical_request_plan import demote_calendar_measures

    plan = {"fields": [dict(field) for field in fields]}
    demote_calendar_measures(plan, [_calendar_binding()] if bindings is None else bindings)
    return [field.get("enforcement") for field in plan["fields"]]


def _measure(column: str, table: str = "WH.MART.DT_DMS") -> dict:
    return {"term": column.lower(), "table": table, "column": column, "role": "measure"}


class TestTheRule:

    @pytest.mark.parametrize("column", ["WK_OF_YR", "YR", "QTR_NO", "MTH", "MTH_NM", "DAY_OF_WK", "DAY_NM"])
    def test_a_time_attribute_is_not_a_measure(self, column):
        assert _demoted(_measure(column)) == ["optional"]

    def test_what_else_the_calendar_keeps_stays(self):
        # A holiday flag is counted and added up; the calendar's own date is
        # a latest or an earliest.
        assert _demoted(_measure("HDY_FLG"), _measure("DMS_DT")) == [None, None]

    def test_only_on_the_calendar(self):
        assert _demoted(_measure("YR", "WH.MART.SALES_FCT")) == [None]

    def test_only_as_a_measure(self):
        week = dict(_measure("WK_OF_YR"), role="dimension")
        assert _demoted(week) == [None]

    def test_no_calendar_no_demotion(self):
        assert _demoted(_measure("WK_OF_YR"), bindings=[]) == [None]


class TestTheOrder:

    def test_a_count_of_records_is_read_once_the_week_is_not_a_measure(self):
        # The plan the pipeline holds for "items created by week of year",
        # with the week bound as the sample's planner binds it.
        from core.analytical_request_plan import demote_what_the_question_does_not_compute

        plan = {
            "source_scope": {"selected_fact": "MART.ITM_BAL_DLY_FCT"},
            "temporal_policies": [{"resolution_source": "explicit_date_role", "fact_table": "MART.ITM_BAL_DLY_FCT",
                                   "business_role": "Creation Date", "fact_column": "ITM_WHS_CRN_DT_DMS_KEY"}],
            "fields": [
                {"term": "item", "table": "WH.MART.ITM_DMS", "column": "ITM_NM", "role": "display_dimension",
                 "aliases": ["item", "item dimension key"], "source_key_column": "ITM_DMS_KEY",
                 "source_key_table": "WH.MART.ITM_BAL_DLY_FCT", "display_required": True},
                dict(_WEEK),
                {"term": "Creation Date", "table": "MART.DT_DMS", "column": "DMS_DT", "role": "date_dimension",
                 "enforcement": "required"},
            ],
        }
        demote_what_the_question_does_not_compute(plan, {"record_count": "item"}, [], [_calendar_binding()])
        assert [field.get("enforcement") for field in plan["fields"]] == ["optional", "optional", "required"]
