"""
A grain with no window is a series over every date the data holds.

"Units sold by month", "units sold each year" and "number of receipts by
quarter" named a grain and no window, and the date plan gave them no policy at
all: the compilers had no date to bucket, and every such question went to the
model -- the plainest trend a reader asks for. On the sample tenant "items
created by year" and "by day of the week" were among them.

A question that asks for a grain and names no window now carries a series
policy over every date the governed date holds: grouped by that grain, with no
filter and no anchor. The prompts say so when the model writes the query. A
level asked for by a grain is still read at the last snapshot of each period.

A synthetic tenant (tests/answer_harness.py) whose monthly snapshot keeps its
units sold and receipts; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("series")) as built:
        yield built


def _series(measure: int, bucket, by_unit: bool) -> dict:
    """The harness's monthly rows summed into buckets; the year row is not a month."""
    totals: dict = {}
    for row in harness.MOVES:
        item, period = row[1], row[2]
        start = bucket(dt.date(period // 100, period % 100, 1))
        key = (start, harness.ITEMS[item][3]) if by_unit else start
        totals[key] = totals.get(key, 0) + row[measure]
    return totals


def _month(day: dt.date) -> dt.date:
    return day.replace(day=1)


def _year(day: dt.date) -> dt.date:
    return day.replace(month=1, day=1)


def _quarter(day: dt.date) -> dt.date:
    return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)


_SOLD, _RECEIPTS = 3, 5


def _answered(answer: dict, measure: str, by_unit: bool) -> dict:
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return {
        ((row["PERIOD"], row["UNT_OF_MSR"]) if by_unit else row["PERIOD"]): row[measure]
        for row in run["rows"]
    }


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Units sold by month", "en"),
        ("Unités vendues par mois", "fr"),
    ])
    def test_each_month(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "UNITS_SOLD", by_unit=True) == pytest.approx(_series(_SOLD, _month, True))

    def test_each_year(self, warehouse):
        answer = harness.ask(warehouse, "Units sold each year")
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "UNITS_SOLD", by_unit=True) == pytest.approx(_series(_SOLD, _year, True))

    def test_each_quarter(self, warehouse):
        answer = harness.ask(warehouse, "Number of receipts by quarter")
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "NUMBER_OF_RECEIPTS", by_unit=False) == _series(_RECEIPTS, _quarter, False)


_DAILY_DATE = {"fact_table": "MART.SALES_FCT", "fact_column": "SLS_DT_DMS_KEY",
               "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY", "date_value_column": "DMS_DT",
               "date_key_type": "surrogate_fk", "business_role": "Sales date", "name": "Sales date",
               "temporal_grain": "day"}


def _policies(question: str, **kwargs) -> list[dict]:
    from core.contextual_dates import build_contextual_date_plan

    return build_contextual_date_plan(_DAILY_DATE, question, **kwargs).get("temporal_policies") or []


class TestTheDatePlan:

    def test_a_grain_alone_is_a_series(self):
        (policy,) = _policies("net sales by month", snapshot=False)
        assert (policy["kind"], policy["requested_grain"]) == ("all_dates", "month")

    def test_no_grain_is_no_series(self):
        assert _policies("net sales by warehouse", snapshot=False) == []

    def test_a_window_keeps_its_own_policy(self):
        assert [policy["kind"] for policy in _policies("net sales by month in 2025", snapshot=False)] == [
            "named_period"]

    def test_a_date_the_reader_typed_is_never_every_date(self):
        # Not read into a window here, it is the model's to write -- never
        # widened to every date the data holds.
        assert _policies("net sales by day on March 2, 2026", snapshot=False) == []

    def test_a_level_is_still_read_at_its_snapshots(self):
        assert [policy["kind"] for policy in _policies("stock by month", snapshot=True)] == ["latest_snapshot"]


class TestTheModelIsToldTheSame:

    def test_no_anchor_and_no_filter(self):
        from core.query_pipeline import _governed_date_anchor_repair_lines
        from core.semantic_planner import format_semantic_field_plan

        from core.contextual_dates import build_contextual_date_plan

        plan = build_contextual_date_plan(_DAILY_DATE, "net sales by month", snapshot=False)
        prompt = format_semantic_field_plan(plan)
        assert "no date filter, no anchor" in prompt
        assert "REQUIRED ANCHOR" not in prompt
        repair = _governed_date_anchor_repair_lines(plan)
        assert "with no date filter" in repair
        assert "REQUIRED ANCHOR" not in repair
