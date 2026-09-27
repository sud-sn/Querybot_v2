"""
A dimension answers to the last word of its name.

"Receipts by division in 2022" was never answered on the sample tenant: the
division dimension is the profit center division, and the planner matched a
dimension only as its whole concept -- "division" alone is one word of three.
The reader was told the query could not be trusted; "by profit center division"
would have worked.

A dimension whose name no other dimension's name ends on now answers to that
last word where the question asks by it: "by division", "for each group",
"which division" -- not "group the stock by warehouse". A word two dimensions
end on names neither -- "group" beside an item group and a product group is a
question, not a guess -- and a generic word ("type", "status", "name") never
stands alone. "Groupe(s)" is read as "group(s)" in French.

A synthetic tenant (tests/answer_harness.py) whose item group is its only group.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("head-nouns")) as built:
        yield built


def _by_group(answer: dict) -> dict:
    assert answer["model_wrote_sql"] is False
    return {(row["ITEM_GROUP"], row["UNT_OF_MSR"]): row["STOCK_ON_HAND"] for row in answer["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Stock on hand by group", "en"),
        ("Stock en main par groupe", "fr"),
    ])
    def test_by_group_is_by_item_group(self, warehouse, question, lang):
        by_item_group = _by_group(harness.ask(warehouse, "Stock on hand by item group"))
        assert by_item_group
        assert _by_group(harness.ask(warehouse, question, lang)) == by_item_group

    def test_group_as_a_verb_asks_for_no_group(self, warehouse):
        answer = harness.ask(warehouse, "Group the stock on hand by warehouse")
        assert answer["model_wrote_sql"] is False
        assert {key for row in answer["rows"] for key in row} == {"WAREHOUSE", "UNT_OF_MSR", "STOCK_ON_HAND"}


def _dimension(display_table: str, name: str) -> dict:
    return {"display_table": display_table, "display_column": "DSC", "source_key": display_table.split(".")[-1] + "_KEY",
            "name": name}


class TestTheRule:

    @staticmethod
    def _short(*dimensions: dict) -> dict:
        from core.semantic_model import _dimension_short_names

        return _dimension_short_names([{"qualified_name": "MART.STOCK", "dimensions": list(dimensions)}])

    def test_a_head_noun_no_other_dimension_ends_on(self):
        assert self._short(_dimension("MART.PC_DVN", "Profit Center Division"),
                           _dimension("MART.WHS", "Warehouse")) == {"MART.PC_DVN": "division"}

    def test_a_head_noun_two_dimensions_share(self):
        assert self._short(_dimension("MART.ITM_GRP", "Item Group"),
                           _dimension("MART.PDC_GRP", "Product Group")) == {}

    def test_one_dimension_reached_from_two_facts(self):
        from core.semantic_model import _dimension_short_names

        group = _dimension("MART.ITM_GRP", "Item Group")
        tables = [{"qualified_name": "MART.DAILY", "dimensions": [group]},
                  {"qualified_name": "MART.MONTHLY", "dimensions": [dict(group)]}]
        assert _dimension_short_names(tables) == {"MART.ITM_GRP": "group"}

    @pytest.mark.parametrize("name", ["Customer Type", "Customer Status", "Customer Name", "Customer No"])
    def test_not_a_generic_word(self, name):
        # Named as the customer's own: the customer key's role, refined.
        from core.semantic_model import _dimension_label

        dimension = {**_dimension("MART.X", name), "source_key": "CUS_DMS_KEY"}
        assert _dimension_label("CUS_DMS_KEY", dimension, "DSC")[0] == name
        assert self._short(dimension) == {}

    @pytest.mark.parametrize("question,asked", [
        ("Receipts by division in 2022", True),
        ("Stock for each division", True),
        ("Which divisions sold the most?", True),
        ("Group the stock by warehouse", False),
        ("Division totals", False),
    ])
    def test_asked_by(self, question, asked):
        from core.semantic_model import _asked_by

        word = "group" if "Group" in question else "division"
        assert _asked_by(question, word) is asked

    @pytest.mark.parametrize("question,ending", [
        ("Stock en main par groupe", "by group"),
        ("Stock en main de tous les groupes", "groups"),
    ])
    def test_groupe_in_french(self, question, ending):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr").endswith(ending)
