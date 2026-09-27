"""
A quarter is the calendar's where the warehouse keeps no fiscal year.

"Units sold in the first quarter of 2022" was answered on the sample tenant
with a question: "Should I interpret Q1 2022 using calendar quarters or your
fiscal quarters?" -- in English and French, every time, while "the first half
of 2022" was answered. Nothing in its warehouse is named for a fiscal year:
the calendar keeps its years, quarters, months and weeks, and no fiscal
period anywhere. The choice offered was one the warehouse could not honour.

A bare quarter is now asked about only where a fiscal year is kept -- a table
or a column named for one -- or cannot be told; an admin's setting still
decides it, and a question that names its basis still does.

A synthetic tenant (tests/answer_harness.py) whose warehouse keeps no fiscal
year, and one that is made to keep one.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("quarters")) as built:
        yield built


def _sold_in_the_first_quarter_of_2025() -> dict:
    sold: dict = {}
    for _whs, item, period, units, *_rest in harness.MOVES:
        if 202501 <= period <= 202503:
            unit = harness.ITEMS[item][3]
            sold[unit] = sold.get(unit, 0) + units
    return sold


def _answered(answer: dict) -> dict:
    assert answer["model_wrote_sql"] is False
    return {row["UNT_OF_MSR"]: row["UNITS_SOLD"] for row in answer["rows"]}


def _forget_the_threads_calendar() -> None:
    """A basis an earlier question in the harness's one thread chose ("calendar
    Q1 2025") is that thread's until it is forgotten."""
    from core import query_pipeline

    query_pipeline.conversation_state_store.clear(harness.ACCOUNT, f"{harness.ACCOUNT}:portal:harness")


def _asked(answer: dict) -> list:
    return [body for kind, body in answer["replies"] if kind == "clarify"]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Units sold in the first quarter of 2025", "en"),
        ("Units sold in Q1 2025", "en"),
        ("Quantité vendue au premier trimestre 2025", "fr"),
    ])
    def test_the_calendars_quarter(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert _asked(answer) == []
        assert _answered(answer) == pytest.approx(_sold_in_the_first_quarter_of_2025())

    def test_a_warehouse_that_keeps_a_fiscal_year_is_still_asked(self, warehouse):
        import core.schema as schema

        original = schema.load_schema_columns

        def with_a_fiscal_year(schema_dir):
            tables = {name: dict(columns) for name, columns in original(schema_dir).items()}
            for name in tables:
                if name.upper().endswith("DT_DMS"):
                    tables[name]["FSC_YR"] = "int"
            return tables

        _forget_the_threads_calendar()
        with patch.object(schema, "load_schema_columns", with_a_fiscal_year):
            answer = harness.ask(warehouse, "Units sold in the first quarter of 2025")
        assert len(_asked(answer)) == 1 and answer["executed"] == []


def _keeps(tmp_path, tables: dict) -> bool:
    from core.query_pipeline import _warehouse_keeps_a_fiscal_year

    schema = {f"WH.MART.{name}": {"columns": [{"name": column, "type": "int"} for column in columns],
                                  "pk_columns": [], "schema": "MART", "database": "WH"}
              for name, columns in tables.items()}
    (tmp_path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return _warehouse_keeps_a_fiscal_year(str(tmp_path))


class TestTheEvidence:

    def test_a_calendar_with_no_fiscal_year(self, tmp_path):
        assert _keeps(tmp_path, {"DT_DMS": ["DT_DMS_KEY", "YR", "QR", "WK_OF_YR"],
                                 "STOCK_FCT": ["FIN_GDS_QTY", "FSCALE_FCTR"]}) is False

    @pytest.mark.parametrize("tables", [
        {"FISCAL_PERIOD_DMS": ["PRD_KEY"]},
        {"DT_DMS": ["DT_DMS_KEY", "FSC_YR"]},
        {"DT_DMS": ["DT_DMS_KEY", "FY_START_DT"]},
        {"DT_DMS": ["DT_DMS_KEY", "PRD_FY"]},
        {"DT_DMS": ["DT_DMS_KEY", "FISCAL_QUARTER"]},
    ])
    def test_a_fiscal_year_by_any_of_its_names(self, tmp_path, tables):
        assert _keeps(tmp_path, tables) is True

    def test_a_warehouse_it_cannot_read_is_asked_about(self, tmp_path):
        from core.query_pipeline import _warehouse_keeps_a_fiscal_year

        assert _warehouse_keeps_a_fiscal_year(str(tmp_path / "missing")) is True
        assert _warehouse_keeps_a_fiscal_year("") is True


class TestTheWords:

    @pytest.mark.parametrize("question,named", [
        ("Units sold in Q1 2022", True),
        ("Units sold in the first quarter of 2022", True),
        ("Unités vendues au premier trimestre 2022", True),
        ("Unités vendues en T1 2022", True),
        ("Units sold in the first half of 2022", False),
        ("Units sold by quarter", False),
    ])
    def test_a_quarter_named_by_its_number(self, question, named):
        from core.analytical_intent import names_a_quarter

        assert names_a_quarter(question) is named
