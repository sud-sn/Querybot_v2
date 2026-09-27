"""
A tied join is told in words, and asks only for a word that can change it.

When two join paths reach a dimension equally well, the answer takes the
better-ranked one and says so. On the sample tenant the notice named the
dimension by its table and both paths by their tables cut at the underscores:

    Using the **Itm Bal Dly Fct via Itm Grp Dms** relationship to reach
    ITM_GRP_DMS. Ask again naming *Itm Dms via Itm Grp Dms* to use the other one.

and it asked the reader to name a path no word can take: a question is weighed
by the words of a path's own edges -- their labels and roles -- and neither of
these paths had any. On a tenant whose snapshot names a buyer and a seller,
"inventory value by party" was told it reached "PTY_DMS".

The dimension and the tables are now named as the tenant's vocabulary spells
them; the reader is asked to name the word that takes the other path, only
where one does -- not a word both paths carry, nor one the question already
says -- and is otherwise told which table the value was read from.

A synthetic tenant (tests/answer_harness.py) whose snapshot reaches the party
table by a buyer key and a seller key, and small graphs of the sample's shape.
"""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from tests import answer_harness as harness

_ON_HAND, _COST = 4, 6
FACT = "ITM_BAL_DLY_FCT"


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("tied-joins")) as built:
        yield built


def _value_by(party_of_row) -> dict:
    """The newest snapshot's inventory value by the name of each row's party."""
    totals: dict = {}
    for n, row in enumerate(harness.STOCK):
        name = harness.PARTIES[party_of_row(n, row)][1]
        totals[name] = totals.get(name, 0) + row[_ON_HAND] * row[_COST]
    return totals


def _notices(answer: dict) -> list[str]:
    return [str(body) for kind, body in answer["replies"] if kind == "message" and "ℹ️" in str(body)]


def _values(answer: dict) -> dict:
    assert answer["model_wrote_sql"] is False
    name = next(column for column in answer["rows"][0] if column != "INVENTORY_VALUE")
    return {row[name]: row["INVENTORY_VALUE"] for row in answer["rows"]}


class TestTheProductAnswers:

    def test_the_party_is_named_as_the_reader_knows_it(self, warehouse):
        from core.i18n import t

        answer = harness.ask(warehouse, "Inventory value by party")
        assert _values(answer) == pytest.approx(_value_by(lambda n, row: row[2]))
        assert _notices(answer) == [t("disclosure.relationship.ranked", lang="en",
                                      chosen="Buyer", target="Party", alternative="Seller")]

    def test_the_word_it_names_takes_the_other_path(self, warehouse):
        notice = _notices(harness.ask(warehouse, "Inventory value by party"))[0]
        word = re.search(r"\*([^*]+)\*(?= to use)", notice).group(1)
        answer = harness.ask(warehouse, f"Inventory value by {word.lower()}")
        assert _values(answer) == pytest.approx(_value_by(lambda n, row: harness.SELLER[n]))
        assert _notices(answer) == []


def _entity(name: str, kind: str, display: str = "") -> dict:
    # A display name copied from the table, as discovery writes one.
    return {"entity_name": name, "table_name": name, "schema_name": "MART", "entity_type": kind,
            "display_name": display or name.replace("_", " "), "status": "confirmed"}


def _edge(number: int, source: str, target: str, column: str, label: str = "", confidence: float = 100) -> dict:
    return {"id": number, "from_entity": source, "to_entity": target, "from_column": column, "to_column": column,
            "relationship_key": f"DBFK:{number}", "relationship_type": "many_to_one", "join_type": "INNER",
            "label": label, "generated_by": "db_fk", "source_enforced": 1, "status": "confirmed",
            "validation_status": "valid", "confidence_score": confidence}


def _graph(*edges) -> dict:
    return {"entities": [_entity(FACT, "fact", "Stock Snapshot"), _entity("ITM_DMS", "dimension"),
                         _entity("ITM_GRP_DMS", "dimension"), _entity("PTY_DMS", "dimension")],
            "relationships": list(edges), "properties": []}


# The sample's shape: the snapshot keeps the item's group beside the item, and
# the item keeps its group too. Neither road has a name of its own.
UNNAMED = _graph(_edge(1, FACT, "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", confidence=87.5),
                 _edge(2, FACT, "ITM_DMS", "ITM_DMS_KEY"),
                 _edge(3, "ITM_DMS", "ITM_GRP_DMS", "ITM_GRP_DMS_KEY"))
# Two keys into one party table, each with its role: as a label, or as a
# business role the relationship keeps.
ROLES = _graph(_edge(1, FACT, "PTY_DMS", "BYR_PTY_DMS_KEY", "Buyer"),
               _edge(2, FACT, "PTY_DMS", "SLR_PTY_DMS_KEY", "Seller"))
ROLES_KEPT = _graph(dict(_edge(1, FACT, "PTY_DMS", "BYR_PTY_DMS_KEY"), business_role="Buyer"),
                    dict(_edge(2, FACT, "PTY_DMS", "SLR_PTY_DMS_KEY"), business_role="Seller"))
# The other road is named by its first step alone.
NAMED_FIRST = _graph(_edge(1, FACT, "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", confidence=87.5),
                     _edge(2, FACT, "ITM_DMS", "ITM_DMS_KEY", "Catalogue Item"),
                     _edge(3, "ITM_DMS", "ITM_GRP_DMS", "ITM_GRP_DMS_KEY"))
# The other road's only name is one the road taken carries too.
SHARED = _graph(_edge(1, FACT, "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", "Catalogue", confidence=87.5),
                _edge(2, FACT, "ITM_DMS", "ITM_DMS_KEY"),
                _edge(3, "ITM_DMS", "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", "Catalogue"))
# The other road's own name is one the question already says.
SAID = _graph(_edge(1, FACT, "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", "Item Group"),
              _edge(2, FACT, "ITM_DMS", "ITM_DMS_KEY", "Item"),
              _edge(3, "ITM_DMS", "ITM_GRP_DMS", "ITM_GRP_DMS_KEY", "Item Group"))


def _resolved(graph: dict, question: str, target: str) -> dict:
    from core.graph_resolver import resolve_for_question

    result = resolve_for_question(question, "tenant", "azure_sql", graph=graph,
                                  required_entities={FACT, target}, metric_formula_tables={FACT})
    assert result["planning_status"] == "selected"
    return result


def _told(graph: dict, question: str, target: str, lang: str = "en") -> str:
    from core.query_pipeline import ranked_relationship_disclosure

    result = _resolved(graph, question, target)
    return ranked_relationship_disclosure(result.get("ranked_relationship") or {}, lang=lang)


class TestTheNotice:

    @pytest.mark.parametrize("lang,told", [
        ("en", "ℹ️ Item Group is read from **Stock Snapshot**, not from *Item*."),
        ("fr", "ℹ️ Item Group provient de **Stock Snapshot**, et non de *Item*."),
    ])
    def test_an_unnamed_road_is_told_where_it_was_read_from(self, lang, told):
        assert _told(UNNAMED, "stock on hand by item group", "ITM_GRP_DMS", lang) == told

    @pytest.mark.parametrize("graph", [ROLES, ROLES_KEPT], ids=["label", "business role"])
    @pytest.mark.parametrize("lang", ["en", "fr"])
    def test_a_named_road_is_asked_for_by_its_name(self, graph, lang):
        from core.i18n import t

        assert _told(graph, "stock on hand by party", "PTY_DMS", lang) == t(
            "disclosure.relationship.ranked", lang=lang, chosen="Buyer", target="Party", alternative="Seller")

    def test_the_word_asked_for_is_the_one_that_takes_the_road(self):
        assert _told(NAMED_FIRST, "stock on hand by item group", "ITM_GRP_DMS") == (
            "ℹ️ Using the **Stock Snapshot via Item Group** relationship to reach Item Group. "
            "Ask again naming *Catalogue Item* to use the other one.")
        taken = _resolved(NAMED_FIRST, "stock on hand by catalogue item group", "ITM_GRP_DMS")
        assert "[MART].[ITM_DMS]" in taken["join_skeleton"]
        assert not taken.get("ranked_relationship")

    def test_a_name_both_roads_carry_is_not_asked_for(self):
        assert _told(SHARED, "stock on hand by item group", "ITM_GRP_DMS") == (
            "ℹ️ Item Group is read from **Stock Snapshot**, not from *Item*.")

    def test_a_name_the_question_already_says_is_not_asked_for(self):
        assert _told(SAID, "stock on hand by item group", "ITM_GRP_DMS") == (
            "ℹ️ Item Group is read from **Stock Snapshot**, not from *Item*.")

    def test_a_broken_vocabulary_costs_the_words_not_the_answer(self):
        import core.source_resolution as source_resolution

        def broken(*args, **kwargs):
            raise RuntimeError("vocabulary unavailable")

        with patch.object(source_resolution, "_business_source_label", broken):
            told = _told(UNNAMED, "stock on hand by item group", "ITM_GRP_DMS")
        assert told == "ℹ️ ITM GRP DMS is read from **Stock Snapshot**, not from *ITM DMS*."

    def test_roads_read_from_one_table_are_told_by_their_names(self):
        from core.query_pipeline import ranked_relationship_disclosure

        ranked = {"target": "ITM_GRP_DMS", "target_label": "Item Group", "redirect": "",
                  "chosen": "Stock Snapshot via Item via Item Group",
                  "alternatives": ["Stock Snapshot via Warehouse via Item via Item Group"],
                  "options": [{"from_label": "Item"}, {"from_label": "Item"}]}
        assert ranked_relationship_disclosure(ranked, lang="en") == (
            "ℹ️ Item Group is read from **Stock Snapshot via Item via Item Group**, "
            "not from *Stock Snapshot via Warehouse via Item via Item Group*.")


class TestTheRoadsNames:

    def test_an_unnamed_walk_names_each_table_once(self):
        from core.graph_resolver import _path_label

        walk = [UNNAMED["relationships"][1], UNNAMED["relationships"][2]]
        assert _path_label(walk, UNNAMED["entities"]) == "Stock Snapshot via Item via Item Group"

    def test_a_walk_is_named_in_the_order_it_is_walked(self):
        from core.graph_resolver import _path_label

        # From the group to the snapshot, and back out along the item's key.
        walk = [dict(UNNAMED["relationships"][0], _direction="backward"),
                dict(UNNAMED["relationships"][1], _direction="forward")]
        assert _path_label(walk, UNNAMED["entities"]) == "Item Group via Stock Snapshot via Item"

    def test_a_named_step_ends_a_walk(self):
        from core.graph_resolver import _path_label

        walk = [UNNAMED["relationships"][1], dict(UNNAMED["relationships"][2], label="Item Group")]
        assert _path_label(walk, UNNAMED["entities"]) == "Stock Snapshot via Item via Item Group"
        walk = [dict(UNNAMED["relationships"][1], label="Item"), UNNAMED["relationships"][2]]
        assert _path_label(walk, UNNAMED["entities"]) == "Item via Item Group"
