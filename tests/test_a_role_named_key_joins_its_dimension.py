"""
A key named for a role joins the dimension that plays it; nothing else does.

A balance fact carries BYR_PTY_DMS_KEY -- the party acting as the item's buyer
-- and four ABC classifications, ABC_CLS_MNL/_VOL/_FRQ/_CTB_DMS_KEY, all keys of
ABC_CLS_DMS. Discovery looked for a table named exactly like each key's stem
(BYR_PTY, ABC_CLS_MNL), found none, and joined none of them: "stock by buyer"
and "stock by ABC class" had no path at all.

A key on a fact now joins the dimension its name contains when the words
around that name are a role: a known qualifier (buyer, ship-to, primary), or
one of a family of variants of the same dimension with no plain key to it.
Each role is its own edge into the dimension, labelled with the role, as a
database's foreign keys to one table already are -- so "ABC class volume" takes
the volume classification, "buyer" the buyer, and "ABC class" alone takes the
top-ranked one and names the others. A word that names a dimension of its own
is never stripped: SLR_STS_DMS_KEY is the supplier's STATUS and
ITM_BIN_LOC_DMS_KEY the bin location, dimensions that were not selected, so
neither is joined to the supplier or the item.

Also: a fact's key that may be NULL is a LEFT join (an INNER join drops those
rows from every total grouped by it), and a table's own key is found when its
name starts with DIM_ (str.lstrip removed characters, not the prefix).

Drives build_entity_graph_from_schema from a schema file, then the resolver's
entity detection and path finding on the graph it built. Synthetic tables in a
mart's naming convention; no customer data.
"""

from __future__ import annotations

import json

import pytest

from core.graph_resolver import detect_entities, find_join_path_with_diagnostics
from core.schema import build_entity_graph_from_schema


def _table(own_key: str, *columns: tuple[str, str], nullable: tuple[str, ...] = (),
           pk: bool = True) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": name in nullable, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key] if pk else [],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


def _dim(key: str, code: str) -> dict:
    return _table(key, (key, "int"), (code, "varchar"), (code.replace("_CD", "_DSC"), "varchar"))


FACT = "ITM_BAL_DLY_FCT"
ABC_ROLES = ["ABC_CLS_MNL_DMS_KEY", "ABC_CLS_VOL_DMS_KEY",
             "ABC_CLS_FRQ_DMS_KEY", "ABC_CLS_CTB_DMS_KEY"]
SCHEMA = {
    f"WH.MART.{FACT}": _table(
        f"{FACT}_KEY",
        (f"{FACT}_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("SLR_DMS_KEY", "int"), ("BYR_PTY_DMS_KEY", "int"), ("SHP_TO_PTY_DMS_KEY", "int"),
        *((column, "int") for column in ABC_ROLES),
        ("SLR_STS_DMS_KEY", "int"), ("PTY_STS_DMS_KEY", "int"),
        ("ITM_BIN_LOC_DMS_KEY", "int"), ("CUS_TIER_DMS_KEY", "int"),
        ("SLR_TIER_DMS_KEY", "int"), ("SLR_SIZE_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("PRI_PRD_DMS_KEY", "int"),
        ("ITM_STK_STS_DMS_KEY", "int"), ("ON_HND_QTY", "decimal"),
        nullable=("ITM_STK_STS_DMS_KEY",),
    ),
    "WH.MART.ITM_DMS": _dim("ITM_DMS_KEY", "ITM_CD"),
    "WH.MART.WHS_DMS": _dim("WHS_DMS_KEY", "WHS_CD"),
    "WH.MART.SLR_DMS": _table(
        "SLR_DMS_KEY", ("SLR_DMS_KEY", "int"), ("SLR_CD", "varchar"),
        ("BYR_PTY_DMS_KEY", "int"), ("RSP_PTY_DMS_KEY", "int"),
    ),
    "WH.MART.PTY_DMS": _dim("PTY_DMS_KEY", "PTY_CD"),
    "WH.MART.CUS_DMS": _dim("CUS_DMS_KEY", "CUS_CD"),
    "WH.MART.ABC_CLS_DMS": _dim("ABC_CLS_DMS_KEY", "ABC_CLS_CD"),
    "WH.MART.ITM_STK_STS_DMS": _dim("ITM_STK_STS_DMS_KEY", "ITM_STK_STS_CD"),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int"),
    ),
    "WH.MART.PRD_DMS": _table(
        "PRD_DMS_KEY", ("PRD_DMS_KEY", "int"), ("PRD_CD", "varchar"),
        ("PRD_YR", "int"), ("PRD_MTH", "int"),
    ),
}


def _graph(tmp_path, schema: dict) -> dict:
    (tmp_path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return build_entity_graph_from_schema(str(tmp_path))


@pytest.fixture
def graph(tmp_path) -> dict:
    return _graph(tmp_path, SCHEMA)


def _edges(graph: dict, from_entity: str) -> dict[str, dict]:
    return {
        rel["from_column"]: rel for rel in graph["relationships"]
        if rel["from_entity"] == from_entity
    }


def _table_of(graph: dict, entity_name: str) -> str:
    return next(e["table_name"] for e in graph["entities"] if e["entity_name"] == entity_name)


def _route(graph: dict, question: str) -> tuple[list[tuple[str, str]], dict]:
    """The joins the resolver takes for a question, as (column, entity)."""
    detected = detect_entities(question, graph, required_tables=[f"MART.{FACT}"])
    path, diagnostics = find_join_path_with_diagnostics(detected, graph, question=question)
    return [(step["from_column"], step["to_entity"]) for step in path], diagnostics


class TestARoleJoinsItsDimension:

    def test_the_buyer_is_a_party(self, graph):
        edge = _edges(graph, FACT)["BYR_PTY_DMS_KEY"]
        assert (edge["to_entity"], edge["to_column"], edge["label"]) == (
            "PTY_DMS", "PTY_DMS_KEY", "Buyer")

    def test_one_known_role_on_its_own_is_enough(self, tmp_path):
        """No family to go on: the buyer is the only party key on the table."""
        schema = {
            "WH.MART.ORD_LIN_FCT": _table(
                "ORD_LIN_FCT_KEY", ("ORD_LIN_FCT_KEY", "bigint"),
                ("BYR_PTY_DMS_KEY", "int"), ("ORD_QTY", "decimal"),
            ),
            "WH.MART.PTY_DMS": _dim("PTY_DMS_KEY", "PTY_CD"),
        }
        edge = _edges(_graph(tmp_path, schema), "ORD_LIN_FCT")["BYR_PTY_DMS_KEY"]
        assert (edge["to_entity"], edge["to_column"], edge["label"]) == (
            "PTY_DMS", "PTY_DMS_KEY", "Buyer")

    def test_a_two_word_role_names_what_it_qualifies(self, graph):
        edge = _edges(graph, FACT)["SHP_TO_PTY_DMS_KEY"]
        assert (edge["to_entity"], edge["label"]) == ("PTY_DMS", "Ship To Party")

    def test_four_classifications_are_four_joins(self, graph):
        edges = _edges(graph, FACT)
        assert all(column in edges for column in ABC_ROLES)
        assert {edges[column]["to_entity"] for column in ABC_ROLES} == {"ABC_CLS_DMS"}
        assert {edges[column]["to_column"] for column in ABC_ROLES} == {"ABC_CLS_DMS_KEY"}
        assert len({edges[column]["label"] for column in ABC_ROLES}) == 4
        assert edges["ABC_CLS_VOL_DMS_KEY"]["label"] == "Abc Class Volume"

    def test_the_plain_keys_still_join_directly(self, graph):
        edges = _edges(graph, FACT)
        assert edges["SLR_DMS_KEY"]["to_entity"] == "SLR_DMS"
        assert edges["ITM_DMS_KEY"]["to_entity"] == "ITM_DMS"


class TestAQuestionTakesTheRoleItNames:

    def test_abc_class_volume(self, graph):
        route, _ = _route(graph, "stock on hand by abc class volume")
        assert route == [("ABC_CLS_VOL_DMS_KEY", "ABC_CLS_DMS")]

    def test_manual_abc_class_in_any_order(self, graph):
        route, _ = _route(graph, "stock on hand by manual abc class")
        assert route == [("ABC_CLS_MNL_DMS_KEY", "ABC_CLS_DMS")]

    def test_the_buyer_alone(self, graph):
        """One join. The supplier's own buyer key is not a road to the party."""
        route, _ = _route(graph, "stock on hand by buyer")
        assert route == [("BYR_PTY_DMS_KEY", "PTY_DMS")]

    def test_the_dimension_alone_takes_one_role_and_names_the_others(self, graph):
        route, diagnostics = _route(graph, "stock on hand by abc class")
        assert len(route) == 1 and route[0][0] in ABC_ROLES
        assert [a["target"] for a in diagnostics["ambiguous_targets"]] == ["ABC_CLS_DMS"]
        assert diagnostics["unreachable"] == []


class TestNothingElseDoes:

    @pytest.mark.parametrize("column", [
        "SLR_STS_DMS_KEY",       # the supplier's status: a dimension of its own
        "PTY_STS_DMS_KEY",       # the party's status, though no plain party key is here
        "ITM_BIN_LOC_DMS_KEY",   # the item's bin location: a dimension of its own
        "CUS_TIER_DMS_KEY",      # a word that is not a known role, on its own
        "PRI_PRD_DMS_KEY",       # a period key: the calendar's roles are date roles
    ])
    def test_a_key_for_another_dimension_is_left_unjoined(self, graph, column):
        assert column not in _edges(graph, FACT)

    def test_a_family_beside_its_plain_key_is_not_a_role(self, graph):
        """SLR_TIER and SLR_SIZE next to SLR_DMS_KEY are the supplier's
        attributes kept as dimensions of their own, not the supplier twice."""
        edges = _edges(graph, FACT)
        assert "SLR_TIER_DMS_KEY" not in edges and "SLR_SIZE_DMS_KEY" not in edges

    def test_a_dimensions_own_role_keys_are_left_to_the_admin(self, graph):
        edges = _edges(graph, "SLR_DMS")
        assert "BYR_PTY_DMS_KEY" not in edges and "RSP_PTY_DMS_KEY" not in edges

    def test_a_date_key_stays_with_its_calendar(self, graph):
        edge = _edges(graph, FACT)["ITM_BAL_EFC_DT_DMS_KEY"]
        assert edge["generated_by"] == "date_role"
        assert _table_of(graph, edge["to_entity"]) == "DT_DMS"
        assert sum(r["from_column"] == "ITM_BAL_EFC_DT_DMS_KEY"
                   for r in graph["relationships"]) == 1


class TestAKeyThatMayBeNullKeepsItsRows:

    def test_a_nullable_key_is_a_left_join(self, graph):
        edge = _edges(graph, FACT)["ITM_STK_STS_DMS_KEY"]
        assert (edge["join_type"], edge["optionality"]) == ("LEFT", "optional")

    def test_a_key_that_cannot_be_null_stays_inner(self, graph):
        edges = _edges(graph, FACT)
        assert (edges["ITM_DMS_KEY"]["join_type"], edges["ITM_DMS_KEY"]["optionality"]) == (
            "INNER", "required")
        assert edges["BYR_PTY_DMS_KEY"]["join_type"] == "INNER"


class TestATablesOwnKeyIsFound:

    def test_dim_prefixed_table_without_a_declared_key(self, tmp_path):
        """DIM_MANAGER had its key read as DEPARTMENT_ID: lstrip("DIM_") left
        "ANAGER", so MANAGER_ID was never tried and the first _ID column won.
        Every join to the manager then compared the wrong column."""
        schema = {
            "WH.MART.FACT_TIMESHEET": _table(
                "FACT_TIMESHEET_ID", ("FACT_TIMESHEET_ID", "bigint"),
                ("MANAGER_ID", "int"), ("HOURS_QTY", "decimal"),
            ),
            "WH.MART.DIM_MANAGER": _table(
                "", ("DEPARTMENT_ID", "int"), ("MANAGER_ID", "int"), ("MANAGER_NAME", "varchar"),
                pk=False,
            ),
        }
        graph = _graph(tmp_path, schema)
        manager = next(e for e in graph["entities"] if e["table_name"] == "DIM_MANAGER")
        assert manager["pk_column"] == "MANAGER_ID"
        edge = _edges(graph, "FACT_TIMESHEET")["MANAGER_ID"]
        assert (edge["to_entity"], edge["to_column"]) == ("DIM_MANAGER", "MANAGER_ID")
