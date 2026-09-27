"""
A period the question names is answered by the product's own SQL.

"Units sold in 2025", "Q1 2022", "March 2025", "the first half of 2025",
"stock at the end of 2025" and their French twins name a calendar period
outright. Only relative windows were read ("the last 6 months"), so these
carried no window at all: the governed date was bound, and the period was
left for the model to write, or dropped. Several gates also missed them --
"H1 2025" and "premier semestre 2025" were not date questions at all, and
"March 2025", being a row of the period table's descriptions as well, was
taken for a member to filter on.

Now the period is read into literal bounds, a start and an exclusive end that
need no anchor and no clock, and compiled on the measure's own date: a flow is
summed inside the bounds, a level is read at the last snapshot inside them.
"The most recent month" is the data's newest month. A window total of a
quantity is kept per unit. A compiled query's empty result is the answer,
told as the period having no records, not regenerated into another question.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from core import contextual_dates
from core.contextual_dates import build_contextual_date_plan, detect_temporal_window, question_names_a_calendar_period
from core.question_normalizer import canonical_question
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("periods")) as built:
        yield built


def read_named_period(question: str) -> dict:
    # Read at call time, so each test fails on a tree without the reader
    # rather than the module failing to import.
    return contextual_dates.read_named_period(question)


def _bounds(question: str) -> tuple[str, str] | None:
    period = read_named_period(question)
    return (period["start"], period["end"]) if period else None


class TestThePeriodIsRead:

    @pytest.mark.parametrize("question,bounds", [
        ("units sold in 2022", ("2022-01-01", "2023-01-01")),
        ("units sold throughout 2025", ("2025-01-01", "2026-01-01")),
        ("stock at the end of 2025", ("2025-01-01", "2026-01-01")),
        ("Q1 2022", ("2022-01-01", "2022-04-01")),
        ("sales in the third quarter of 2024", ("2024-07-01", "2024-10-01")),
        ("units sold in the first half of 2025", ("2025-01-01", "2025-07-01")),
        ("H2 2024", ("2024-07-01", "2025-01-01")),
        ("the second semester of 2024", ("2024-07-01", "2025-01-01")),
        ("units sold in March 2025", ("2025-03-01", "2025-04-01")),
        ("units sold in 2025-03", ("2025-03-01", "2025-04-01")),
        ("sales in Dec 2024", ("2024-12-01", "2025-01-01")),
        ("units sold from January to March 2025", ("2025-01-01", "2025-04-01")),
        ("sales between 2023 and 2024", ("2023-01-01", "2025-01-01")),
    ])
    def test_in_english(self, question, bounds):
        assert _bounds(question) == bounds

    @pytest.mark.parametrize("question,bounds", [
        ("Unités vendues en 2022", ("2022-01-01", "2023-01-01")),
        ("Unités vendues au premier semestre 2025", ("2025-01-01", "2025-07-01")),
        ("Unités vendues en mars 2025", ("2025-03-01", "2025-04-01")),
        ("Ventes de février 2025", ("2025-02-01", "2025-03-01")),
        ("Ventes d'août 2024", ("2024-08-01", "2024-09-01")),
        ("Unités vendues de janvier à mars 2025", ("2025-01-01", "2025-04-01")),
        ("Stock disponible à la fin de 2025", ("2025-01-01", "2026-01-01")),
    ])
    def test_in_french_as_typed_and_as_canonicalised(self, question, bounds):
        assert _bounds(question) == bounds
        assert _bounds(canonical_question(question, "fr")) == bounds

    def test_a_french_quarter_as_canonicalised(self):
        assert _bounds(canonical_question("Unités vendues au T1 2025", "fr")) == ("2025-01-01", "2025-04-01")

    @pytest.mark.parametrize("question", [
        "orders over 2000",
        "the top 2000 items",
        "units sold in the last 3 months",
        "sales 2022 vs 2021",
        "Q1 2025 compared to Q1 2024",
        "fiscal 2025 revenue",
        "units sold from March to January 2025",
    ])
    def test_what_is_not_one_stated_calendar_period(self, question):
        assert read_named_period(question) == {}


class TestItIsADateQuestion:

    @pytest.mark.parametrize("question", [
        "units sold in H1 2025",
        "units sold throughout 2025",
        canonical_question("Unités vendues au premier semestre 2025", "fr"),
    ])
    def test_the_gate_opens(self, question):
        assert question_names_a_calendar_period(question)

    def test_an_amount_is_not_a_period(self):
        assert not question_names_a_calendar_period("orders over 2000")

    @pytest.mark.parametrize("question,kind", [
        ("units purchased in the most recent month", "this_month"),
        ("sales in the latest quarter", "this_quarter"),
        (canonical_question("Unités achetées le mois le plus récent", "fr"), "this_month"),
    ])
    def test_the_most_recent_period_is_the_datas_newest(self, question, kind):
        assert detect_temporal_window(question)["kind"] == kind


_PERIOD_DATE = {"fact_table": "MART.ITM_BAL_PRD_FCT", "fact_column": "PRD_DMS_KEY",
                "date_key_type": "yyyymm_integer", "business_role": "Period", "name": "Period",
                "temporal_grain": "month"}
_BALANCE_DATE = {"fact_table": "MART.ITM_BAL_DLY_FCT", "fact_column": "ITM_BAL_EFC_DT_DMS_KEY",
                 "dimension_table": "MART.DT_DMS", "dimension_key": "DT_DMS_KEY", "date_value_column": "DMS_DT",
                 "date_key_type": "surrogate_fk", "business_role": "Balance date", "name": "Balance date",
                 "temporal_grain": "day"}


class TestTheDatePlanStatesIt:

    def test_a_named_period_is_stated_not_anchored(self):
        (policy,) = build_contextual_date_plan(_PERIOD_DATE, "units sold in March 2025")["temporal_policies"]
        assert (policy["kind"], policy["anchor_policy"]) == ("named_period", "stated")
        assert (policy["start"], policy["end"]) == ("2025-03-01", "2025-04-01")

    def test_a_level_in_it_is_read_at_its_last_snapshot(self):
        (policy,) = build_contextual_date_plan(
            _BALANCE_DATE, "stock on hand at the end of 2025", snapshot=True)["temporal_policies"]
        assert policy["kind"] == "named_period" and policy["reads_a_level"] is True

    def test_a_relative_window_is_still_relative(self):
        (policy,) = build_contextual_date_plan(_PERIOD_DATE, "units sold in the last 3 months")["temporal_policies"]
        assert policy["kind"] == "last_n"


def _sold(first: int, last: int, by_month: bool = False) -> dict:
    """Units sold per unit (and month) over yyyymm first..last, month rows only."""
    totals: dict = {}
    for _whs, item, period, sold, _bought, _receipts, _cost in harness.MOVES:
        if first <= period <= last:
            key = (period, harness.ITEMS[item][3]) if by_month else harness.ITEMS[item][3]
            totals[key] = totals.get(key, 0) + sold
    return totals


def _rows(answer: dict, measure: str) -> list[dict]:
    """The rows of the query that answered: a coverage check may run after it."""
    (answered,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return answered["rows"]


def _by_unit(answer: dict, measure: str) -> dict:
    return {row["UNT_OF_MSR"]: row[measure] for row in _rows(answer, measure)}


def _by_month(answer: dict) -> dict:
    return {(int(str(row["PERIOD"])[:7].replace("-", "")), row["UNT_OF_MSR"]): row["UNITS_SOLD"]
            for row in _rows(answer, "UNITS_SOLD")}


class TestTheProductAnswersIt:

    @pytest.mark.parametrize("question,lang,first,last", [
        ("Units sold in 2025", "en", 202501, 202512),
        ("Unités vendues en 2025", "fr", 202501, 202512),
        ("Units sold in March 2025", "en", 202503, 202503),
        ("Unités vendues en mars 2025", "fr", 202503, 202503),
        ("Units sold in the first half of 2025", "en", 202501, 202506),
        ("Unités vendues au premier semestre 2025", "fr", 202501, 202506),
    ])
    def test_a_flow_is_summed_inside_the_period(self, warehouse, question, lang, first, last):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        # The whole-year row sits beside its months and is never added in.
        assert _by_unit(answer, "UNITS_SOLD") == pytest.approx(_sold(first, last))

    @pytest.mark.parametrize("question,lang", [
        ("Units sold by month in 2025", "en"),
        ("Unités vendues par mois en 2025", "fr"),
    ])
    def test_by_month_inside_the_period(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _by_month(answer) == pytest.approx(_sold(202501, 202512, by_month=True))

    def test_a_level_is_read_at_the_last_snapshot_inside_the_period(self, warehouse):
        answer = harness.ask(warehouse, "Stock on hand at the end of March 2026")
        assert answer["model_wrote_sql"] is False
        latest: dict = {}
        for _whs, item, _buyer, _created, on_hand, _allocated, _cost in harness.STOCK:
            latest[harness.ITEMS[item][3]] = latest.get(harness.ITEMS[item][3], 0) + on_hand
        # The earlier snapshot in March holds a thousand more of everything.
        assert _by_unit(answer, "STOCK_ON_HAND") == pytest.approx(latest)

    def test_a_period_with_no_records_says_so(self, warehouse):
        # The daily snapshot starts in 2026: 2025 has none, and that is the
        # answer -- not a regeneration told to try another anchor.
        answer = harness.ask(warehouse, "Stock on hand at the end of 2025")
        assert answer["model_wrote_sql"] is False and _rows(answer, "STOCK_ON_HAND") == []
        said = str(answer["replies"])
        assert "2025-01-01 to 2025-12-31" in said and "join path" not in said


class TestTheLatestAndTheLast:

    def test_the_most_recent_month_is_the_datas_newest(self, warehouse):
        newest = max(period for _w, _i, period, *_rest in harness.MOVES)
        answer = harness.ask(warehouse, "Units sold in the most recent month")
        assert answer["model_wrote_sql"] is False
        assert _by_unit(answer, "UNITS_SOLD") == pytest.approx(_sold(newest, newest))

    def test_a_window_total_is_kept_per_unit(self, warehouse):
        # The last 3 months to the newest (June 2025): April to June.
        answer = harness.ask(warehouse, "Units sold in the last 3 months")
        assert answer["model_wrote_sql"] is False
        assert _by_unit(answer, "UNITS_SOLD") == pytest.approx(_sold(202504, 202506))

    def test_by_month_over_a_window(self, warehouse):
        answer = harness.ask(warehouse, "Units sold by month over the last 6 months")
        assert answer["model_wrote_sql"] is False
        assert _by_month(answer) == pytest.approx(_sold(202501, 202506, by_month=True))


class TestTheModelIsToldTheBounds:

    def test_a_member_in_a_named_period(self, warehouse):
        # A member is the planner's to filter on: the model writes this one,
        # told the period's bounds rather than to anchor on the data.
        answer = harness.ask(warehouse, "Units sold for BRASS ELBOW in March 2025")
        assert answer["model_wrote_sql"] is True
        prompt = answer["prompts"][0]
        assert "'2025-03-01'" in prompt and "'2025-04-01'" in prompt
        assert "A named period is stated" in prompt
        assert "NAMED PERIOD FILTER DETECTED" not in prompt


class TestAQuarterNeedsItsCalendar:
    """A quarter is January to March only on a calendar basis; a fiscal year's
    first quarter is other months, and an unsettled basis is asked."""

    @staticmethod
    def _policies(question: str, basis: str | None = None) -> list[tuple]:
        kwargs = {"calendar_basis": basis} if basis else {}
        return [(policy["kind"], policy.get("start"), policy.get("end")) for policy in
                build_contextual_date_plan(_PERIOD_DATE, question, **kwargs).get("temporal_policies") or []]

    def test_on_a_calendar_basis(self):
        assert self._policies("units sold in Q1 2025", "calendar") == [("named_period", "2025-01-01", "2025-04-01")]

    @pytest.mark.parametrize("basis", [None, "unresolved", "fiscal"])
    def test_not_otherwise(self, basis):
        assert self._policies("units sold in Q1 2025", basis) == []

    def test_a_half_is_the_calendars_unless_the_year_is_fiscal(self):
        assert self._policies("units sold in the first half of 2025") == [
            ("named_period", "2025-01-01", "2025-07-01")]
        assert self._policies("units sold in the first half of 2025", "fiscal") == []

    def test_a_year_and_a_month_are_the_same_on_any_basis(self):
        assert self._policies("units sold in 2025", "fiscal") == [("named_period", "2025-01-01", "2026-01-01")]
        assert self._policies("units sold in March 2025", "fiscal") == [("named_period", "2025-03-01", "2025-04-01")]

    def test_the_hint_states_bounds_only_on_a_known_calendar(self):
        from core.query_semantics import build_generic_query_hints

        assert "'2025-01-01'" in build_generic_query_hints("units sold in Q1 2025", calendar_basis="calendar")
        assert "'2025-01-01'" not in build_generic_query_hints("units sold in Q1 2025", calendar_basis="fiscal")

    @pytest.mark.parametrize("question", ["Unités vendues au premier trimestre 2025", "Unités vendues au T1 2025"])
    def test_a_french_quarter_is_asked_about_like_an_english_one(self, question):
        from core.analytical_intent import plan_analytical_intent

        plan = plan_analytical_intent(question)
        assert plan.clarification is not None and plan.clarification.slot == "calendar_basis"
        assert plan.quarter_periods == ("Q1 2025",)

    @pytest.mark.parametrize("question,lang", [
        ("Units sold in Q1 2025", "en"), ("Unités vendues au premier trimestre 2025", "fr")])
    def test_the_reader_is_asked_which_quarters(self, warehouse, question, lang):
        # Where the warehouse keeps a fiscal year to read a quarter by; one
        # that keeps none has only the calendar's
        # (tests/test_a_quarter_is_the_calendars_without_a_fiscal_year.py).
        from unittest.mock import patch

        import core.schema as schema

        original = schema.load_schema_columns

        def with_a_fiscal_year(schema_dir):
            tables = {name: dict(columns) for name, columns in original(schema_dir).items()}
            for name in tables:
                if name.upper().endswith("DT_DMS"):
                    tables[name]["FSC_YR"] = "int"
            return tables

        # Nor has an earlier question in the harness's one thread chosen one.
        from core import query_pipeline

        query_pipeline.conversation_state_store.clear(harness.ACCOUNT, f"{harness.ACCOUNT}:portal:harness")
        with patch.object(schema, "load_schema_columns", with_a_fiscal_year):
            answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False and answer["executed"] == []
        assert [kind for kind, *_ in answer["replies"]] == ["clarify"]

    def test_named_as_a_calendar_quarter_it_is_answered(self, warehouse):
        answer = harness.ask(warehouse, "Units sold in calendar Q1 2025")
        assert answer["model_wrote_sql"] is False
        assert _by_unit(answer, "UNITS_SOLD") == pytest.approx(_sold(202501, 202503))


class TestARepairIsToldTheBoundsToo:

    def test_no_anchor_to_copy(self):
        from core.query_pipeline import _governed_date_anchor_repair_lines

        plan = build_contextual_date_plan(_PERIOD_DATE, "units sold in March 2025")
        lines = _governed_date_anchor_repair_lines(plan)
        assert ">= '2025-03-01' and < '2025-04-01'" in lines
        assert "REQUIRED ANCHOR" not in lines and "MAX(" not in lines


class TestAnEmptyPeriodIsExplained:

    @staticmethod
    def _rca(lang: str, business_role: str = "Balance date") -> dict:
        from core import i18n
        from core.answer_rca import build_business_rca

        plan = {"enabled": True, "temporal_policies": [{
            "kind": "named_period", "start": "2025-01-01", "end": "2026-01-01", "label": "2025",
            "business_role": business_role}]}
        token = i18n.activate_language(lang)
        try:
            return build_business_rca(row_count=0, tables_used=["A", "B"], graph_context={"enabled": True},
                                      semantic_plan=plan)
        finally:
            i18n.deactivate_language(token)

    def test_the_reason_names_the_period_and_its_date(self):
        reason = self._rca("en")["most_likely_reason"]
        assert "2025-01-01 to 2025-12-31" in reason and "Balance date" in reason

    def test_in_french(self):
        reason = self._rca("fr")["most_likely_reason"]
        assert "du 2025-01-01 au 2025-12-31" in reason and "Balance date" in reason

    def test_without_a_date_name(self):
        reason = self._rca("en", business_role="")["most_likely_reason"]
        assert "2025-01-01 to 2025-12-31" in reason and "“" not in reason
