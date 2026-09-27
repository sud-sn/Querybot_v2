"""
An answer never adds quantities kept in different units.

A total of a quantity is kept per unit of measure (core/units_of_measure.py):
"reserved quantity by warehouse" comes back with a row for each warehouse's
eaches, feet, rolls and bags. The query was right; the card was not. It merged
each warehouse's rows into one number to rank them -- "NORTH DEPOT leads at
120", 100 feet and 20 eaches -- and told the reader how far ahead it was of the
next. The brief the insights and chart notes are written from did the same, and
shared out a total of mixed units.

Rows in more than one unit are no longer merged, as a balance across months
is not: the card says what came back, and the brief leaves the breakdown out.
Rows in one unit still merge, rows grouped by the unit itself merge per unit,
and a result with one row per label -- a ranking of items, each in its own
unit -- is what the query returned.
"""

from __future__ import annotations

import pytest

from core import i18n
from tests import answer_harness as harness

# Allocated quantity by warehouse, as the governed compiler returns it: a row
# per warehouse and unit.
_BY_UNIT = [
    {"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "FT", "ALLOCATED_QUANTITY": 0.0},
    {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "FT", "ALLOCATED_QUANTITY": 100.0},
    {"WAREHOUSE": "SOUTH DEPOT", "UNT_OF_MSR": "EA", "ALLOCATED_QUANTITY": 60.0},
    {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "EA", "ALLOCATED_QUANTITY": 20.0},
]


def _card(rows: list[dict], question: str, lang: str = "en") -> dict:
    from core.response_builder import build_answer, infer_result_scope

    token = i18n.activate_language(lang)
    try:
        return build_answer(rows, question, infer_result_scope(rows, question))
    finally:
        i18n.deactivate_language(token)


class TestTheProductAnswers:

    @pytest.fixture(scope="class")
    def warehouse(self, tmp_path_factory):
        with harness.tenant_in(tmp_path_factory.mktemp("units-in-the-answer")) as built:
            yield built

    def test_allocated_quantity_by_warehouse(self, warehouse):
        answer = harness.ask(warehouse, "Allocated quantity by warehouse")
        assert answer["model_wrote_sql"] is False
        assert {row["UNT_OF_MSR"] for row in answer["rows"]} == {"EA", "FT"}
        (card,) = [payload["answer"] for kind, payload in answer["replies"]
                   if isinstance(payload, dict) and payload.get("answer")]
        assert card["headline"] == "Returned 4 rows for Allocated quantity by warehouse."
        assert "above the next result" not in card["comparison"]


class TestTheCard:

    def test_no_leader_across_units(self):
        card = _card(_BY_UNIT, "Allocated quantity by warehouse")
        assert card["headline"] == "Returned 4 rows for Allocated quantity by warehouse."
        assert card["short_value"] == ""

    def test_in_french(self):
        card = _card(_BY_UNIT, "Quantité allouée par entrepôt", "fr")
        assert card["headline"].startswith("4 lignes")
        assert "120" not in card["headline"]

    def test_one_unit_still_leads(self):
        rows = [row for row in _BY_UNIT if row["UNT_OF_MSR"] == "FT"] + [
            {"WAREHOUSE": "NORTH DEPOT", "UNT_OF_MSR": "FT", "ALLOCATED_QUANTITY": 5.0},
        ]
        card = _card(rows, "Allocated quantity by warehouse")
        assert card["headline"] == "NORTH DEPOT leads at 105."


class TestTheBrief:

    def test_no_breakdown_across_units(self):
        from core.insight import compute_data_brief

        breakdown = compute_data_brief(_BY_UNIT, "Allocated quantity by warehouse")["category_breakdown"]
        assert (breakdown["top_5"], breakdown.get("total"), breakdown.get("leader_share_pct")) == ([], None, None)


class TestTheRule:

    def test_a_label_in_two_units_is_not_merged(self):
        from core.analysis_contract import collapse_rows_by_label

        assert collapse_rows_by_label(_BY_UNIT, "WAREHOUSE", "ALLOCATED_QUANTITY") is None

    def test_one_unit_across_months_is(self):
        from core.analysis_contract import collapse_rows_by_label

        rows = [{"WAREHOUSE": w, "MONTH": m, "UNT_OF_MSR": "EA", "UNITS_SOLD": v}
                for w, m, v in (("NORTH", "Jan", 5), ("NORTH", "Feb", 7), ("SOUTH", "Jan", 3))]
        assert collapse_rows_by_label(rows, "WAREHOUSE", "UNITS_SOLD") == [("NORTH", 12.0), ("SOUTH", 3.0)]

    def test_labels_in_different_units_are_not_merged(self):
        # Each warehouse in one unit, but a total of feet is not ahead of a
        # total of eaches.
        from core.analysis_contract import collapse_rows_by_label

        rows = [{"WAREHOUSE": w, "MONTH": m, "UNT_OF_MSR": u, "UNITS_SOLD": v}
                for w, m, u, v in (("NORTH", "Jan", "EA", 5), ("NORTH", "Feb", "EA", 7),
                                   ("SOUTH", "Jan", "FT", 30), ("SOUTH", "Feb", "FT", 20))]
        assert collapse_rows_by_label(rows, "WAREHOUSE", "UNITS_SOLD") is None

    def test_totals_by_unit_merge_per_unit(self):
        from core.analysis_contract import collapse_rows_by_label

        rows = [{"UNT_OF_MSR": u, "MONTH": m, "UNITS_SOLD": v}
                for u, m, v in (("EA", "Jan", 5), ("EA", "Feb", 7), ("FT", "Jan", 3))]
        assert collapse_rows_by_label(rows, "UNT_OF_MSR", "UNITS_SOLD") == [("EA", 12.0), ("FT", 3.0)]

    def test_a_row_per_label_is_what_came_back(self):
        # Top items by quantity, each in its own unit: nothing is added.
        from core.analysis_contract import collapse_rows_by_label

        rows = [{"ITEM": "COPPER PIPE", "UNT_OF_MSR": "FT", "QTY": 900},
                {"ITEM": "BRASS ELBOW", "UNT_OF_MSR": "EA", "QTY": 160}]
        assert collapse_rows_by_label(rows, "ITEM", "QTY") == [("COPPER PIPE", 900.0), ("BRASS ELBOW", 160.0)]
