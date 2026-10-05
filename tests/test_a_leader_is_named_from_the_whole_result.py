"""
A leader is named from the whole result, never from the rows the cap kept.

"Stock on hand by item group" grouped 304 item-group-and-unit rows with no
ORDER BY, so the 200-row cap kept whichever rows the database returned first,
and the card named the largest of those its leader: one run "In EA, the
second-largest group leads", the next the largest. A leader named from rows
the cap cut is now withheld unless the statement sorted the whole result by
what it measures, that way round; the card says how many rows came back,
beside the caveat that says the result is larger.

Synthetic rows and the harness tenant; no customer data.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core import i18n
from tests import answer_harness as harness

_CAP = 200


def _headline(rows: list[dict], question: str, sql: str) -> str:
    from core.response_builder import build_answer, infer_result_scope

    token = i18n.activate_language("en")
    try:
        return build_answer(rows, question, infer_result_scope(rows, question, sql, mode="ranking"))["headline"]
    finally:
        i18n.deactivate_language(token)


def _groups(count: int) -> list[dict]:
    """The head of a larger result: GROUP 1 .. GROUP n, largest first."""
    return [{"ITEM_GROUP": f"GROUP {n}", "INVENTORY_VALUE": float(1000 - n)} for n in range(1, count + 1)]


def _groups_by_unit(count: int) -> list[dict]:
    """Each group in eaches and in feet: a row per group and unit."""
    return [row for n in range(1, count // 2 + 1) for row in (
        {"ITEM_GROUP": f"GROUP {n}", "UNT_OF_MSR": "EA", "STOCK_ON_HAND": float(5000 - n)},
        {"ITEM_GROUP": f"GROUP {n}", "UNT_OF_MSR": "FT", "STOCK_ON_HAND": float(3000 - n)})]


_UNSORTED = "SELECT g AS ITEM_GROUP, SUM(v) AS INVENTORY_VALUE FROM t GROUP BY g"
_RETURNED = "Returned 200 rows for inventory value by item group."


class TestACutResult:

    def test_unsorted_names_no_leader(self):
        assert _headline(_groups(_CAP), "inventory value by item group", _UNSORTED) == _RETURNED

    @pytest.mark.parametrize("order", [
        "ORDER BY INVENTORY_VALUE DESC", "ORDER BY SUM(v) DESC", "ORDER BY 2 DESC",
        "ORDER BY [INVENTORY_VALUE] DESC -- (largest first)",
    ])
    def test_sorted_by_what_it_measures_largest_first_names_its_leader(self, order):
        headline = _headline(_groups(_CAP), "inventory value by item group", f"{_UNSORTED}\n{order}")
        assert headline.startswith("GROUP 1 leads at")

    @pytest.mark.parametrize("sql", [
        _UNSORTED + "\nORDER BY INVENTORY_VALUE ASC",
        _UNSORTED + "\nORDER BY ITEM_GROUP DESC",
        # The first column is the label, not the figure.
        _UNSORTED + "\nORDER BY 1 DESC",
    ])
    def test_sorted_otherwise_names_no_leader(self, sql):
        assert _headline(_groups(_CAP), "inventory value by item group", sql) == _RETURNED

    def test_sorted_by_another_figure_names_no_leader(self):
        rows = [{"ITEM_GROUP": f"GROUP {n}", "QUANTITY": float(n), "INVENTORY_VALUE": float(1000 - n)}
                for n in range(1, _CAP + 1)]
        sql = "SELECT g AS ITEM_GROUP, SUM(q) AS QUANTITY, SUM(v) AS INVENTORY_VALUE FROM t GROUP BY g"
        assert " leads at " not in _headline(rows, "quantity by item group", sql + "\nORDER BY INVENTORY_VALUE DESC")

    def test_an_apostrophe_in_a_name_hides_no_sort(self):
        sql = ("SELECT g AS ITEM_GROUP, SUM(v) AS INVENTORY_VALUE FROM [Owner's groups] t GROUP BY g"
               "\nORDER BY INVENTORY_VALUE DESC")
        assert _headline(_groups(_CAP), "inventory value by item group", sql).startswith("GROUP 1 leads at")

    def test_the_lowest_is_named_only_from_a_result_sorted_lowest_first(self):
        rows = list(reversed(_groups(_CAP)))
        question = "which item groups have the lowest inventory value"
        assert _headline(rows, question, _UNSORTED + "\nORDER BY INVENTORY_VALUE DESC").startswith(
            "Returned 200 rows")
        assert _headline(rows, question, _UNSORTED + "\nORDER BY INVENTORY_VALUE ASC").startswith(f"GROUP {_CAP} ")

    def test_a_leader_in_one_unit_is_held_to_the_same(self):
        rows = _groups_by_unit(_CAP)
        question = "stock on hand by item group"
        unsorted = "SELECT g AS ITEM_GROUP, u AS UNT_OF_MSR, SUM(q) AS STOCK_ON_HAND FROM t GROUP BY g, u"
        assert " leads at " not in _headline(rows, question, unsorted)
        assert _headline(rows, question, unsorted + "\nORDER BY STOCK_ON_HAND DESC") == (
            "In EA, GROUP 1 leads at 4,999 EA.")

    def test_a_result_the_cap_did_not_reach_is_read_as_before(self):
        assert _headline(_groups(12), "inventory value by item group", _UNSORTED).startswith("GROUP 1 leads at")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("leader-from-the-whole")) as built:
        yield built


class TestThroughThePipeline:

    def test_an_unsorted_breakdown_the_cap_cut_names_no_leader(self, warehouse):
        # Two rows stand for the cap: the harness's warehouse has no more.
        import core.response_builder as response_builder

        with patch.object(response_builder, "_PREVIEW_ROW_CAP", 2):
            answer = harness.ask(warehouse, "stock on hand by warehouse")

        (card,) = [payload for _kind, payload in answer["replies"] if isinstance(payload, dict) and payload.get("answer")]
        assert "ORDER BY" not in answer["sql"].upper().split("GROUP BY")[-1]
        assert card["answer"]["headline"].startswith("Returned ")

    def test_a_ranking_sorted_its_way_names_its_end(self, warehouse):
        answer = harness.ask(warehouse, "which warehouses have the lowest stock?")

        (card,) = [payload for _kind, payload in answer["replies"] if isinstance(payload, dict) and payload.get("answer")]
        assert answer["sql"].rstrip().endswith("ASC"), answer["sql"]
        assert " is lowest at " in card["answer"]["headline"]
