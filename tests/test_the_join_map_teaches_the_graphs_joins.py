"""
The knowledge base's join map teaches the joins the governed planner accepts.

The SQL model reads _join_map.md. Its relationship section came from a pass
that joined any column name two tables shared, so on an inventory mart with a
daily and a monthly balance fact it taught, among 72 joins:

- the two facts joined to each other, on every dimension key and on every
  measure both carry (ON_HND_QTY, ITM_CST) -- the fan-out the governed path
  forbids;
- a fact joined to the profit-centre dimension on the region key, which the
  profit centre only references -- one fact row per profit centre in the
  region;
- no role at all: the buyer's key and the four ABC classifications were not
  there, because no other table shares those column names.

The section is now the entity graph's own key edges (the same function
discovery uses): each table to the dimension that owns the key,
with the edge's join type, and a role under an alias of its own. The database's
constraints, the tenant's alias joins and the date roles keep their passes.

Drives _build_join_map on a schema master and parses the SQL it teaches.
Synthetic tables in a mart's naming convention; no customer data.
"""

from __future__ import annotations

import json
import re

import pytest

from core.schema import _build_join_map, build_entity_graph_from_schema


def _table(own_key: str, *columns: tuple[str, str], nullable: tuple[str, ...] = ()) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": name in nullable, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


def _dim(key: str, code: str, *extra: tuple[str, str]) -> dict:
    return _table(key, (key, "int"), (code, "varchar"), *extra)


DAILY, MONTHLY = "WH.MART.ITM_BAL_DLY_FCT", "WH.MART.ITM_BAL_PRD_FCT"
MASTER = {
    DAILY: _table(
        "ITM_BAL_DLY_FCT_KEY",
        ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("RGN_DMS_KEY", "int"), ("BYR_PTY_DMS_KEY", "int"),
        ("ABC_CLS_MNL_DMS_KEY", "int"), ("ABC_CLS_VOL_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ITM_STK_STS_DMS_KEY", "int"),
        ("ON_HND_QTY", "decimal"), ("ITM_CST", "decimal"), ("DEL_REC_IND", "varchar"),
        nullable=("ITM_STK_STS_DMS_KEY",),
    ),
    MONTHLY: _table(
        "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("RGN_DMS_KEY", "int"), ("PRD_DMS_KEY", "int"),
        ("ON_HND_QTY", "decimal"), ("ITM_CST", "decimal"), ("DEL_REC_IND", "varchar"),
    ),
    "WH.MART.PFT_CTR_DMS": _dim("PFT_CTR_DMS_KEY", "PFT_CTR_CD", ("RGN_DMS_KEY", "int")),
    "WH.MART.RGN_DMS": _dim("RGN_DMS_KEY", "RGN_CD"),
    "WH.MART.ITM_DMS": _dim("ITM_DMS_KEY", "ITM_CD"),
    "WH.MART.WHS_DMS": _dim("WHS_DMS_KEY", "WHS_CD"),
    "WH.MART.PTY_DMS": _dim("PTY_DMS_KEY", "PTY_CD"),
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

# Every JOIN the document teaches, with or without a stated join type.
_JOIN_RE = re.compile(
    r"^(?:(?P<type>INNER|LEFT) )?JOIN \[(?P<right>[^\]]+)\](?: AS \[(?P<alias>[^\]]+)\])? "
    r"ON \[(?P<left>[^\]]+)\]\.(?P<lcol>\w+) = \[(?P<rhs>[^\]]+)\]\.(?P<rcol>\w+)$",
    re.M,
)


@pytest.fixture(scope="module")
def join_map() -> str:
    return _build_join_map(MASTER)


def _section(doc: str, heading: str) -> str:
    start = doc.index(heading)
    rest = doc[start + len(heading):]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _joins(text: str) -> list[dict]:
    return [match.groupdict() for match in _JOIN_RE.finditer(text)]


def _pairs(text: str) -> set[tuple[str, str, str, str]]:
    return {(j["left"], j["lcol"], j["right"], j["rcol"]) for j in _joins(text)}


class TestNoJoinTheGovernedPathWouldRefuse:

    def test_the_two_facts_are_never_joined(self, join_map):
        facts = {DAILY, MONTHLY}
        assert not [p for p in _pairs(join_map) if {p[0], p[2]} <= facts]

    @pytest.mark.parametrize("column", ["ON_HND_QTY", "ITM_CST", "DEL_REC_IND"])
    def test_no_join_on_a_shared_measure_or_flag(self, join_map, column):
        assert not [p for p in _pairs(join_map) if column in (p[1], p[3])]

    def test_a_fact_is_not_joined_to_a_dimension_that_only_references_the_key(self, join_map):
        """PFT_CTR_DMS carries RGN_DMS_KEY; the region owns it."""
        assert not [p for p in _pairs(join_map)
                    if p[0] in {DAILY, MONTHLY} and p[2] == "WH.MART.PFT_CTR_DMS"]
        assert (DAILY, "RGN_DMS_KEY", "WH.MART.RGN_DMS", "RGN_DMS_KEY") in _pairs(join_map)
        assert ("WH.MART.PFT_CTR_DMS", "RGN_DMS_KEY", "WH.MART.RGN_DMS",
                "RGN_DMS_KEY") in _pairs(join_map)


class TestItIsTheGraph:

    def test_the_dimension_joins_are_the_graphs_key_edges(self, join_map, tmp_path):
        """The graph discovery builds from the same schema file."""
        (tmp_path / "_schema.json").write_text(json.dumps(MASTER), encoding="utf-8")
        graph = build_entity_graph_from_schema(str(tmp_path))
        table = {e["entity_name"]: f"WH.MART.{e['table_name']}" for e in graph["entities"]}
        expected = {
            (table[r["from_entity"]], r["from_column"], table[r["to_entity"]], r["to_column"])
            for r in graph["relationships"] if r["generated_by"] == "heuristic"
        }
        taught = {
            (j["left"], j["lcol"], j["right"], j["rcol"])
            for j in _joins(_section(join_map, "## Dimension Joins"))
        }
        assert taught == expected

    def test_a_key_that_may_be_empty_is_taught_as_a_left_join(self, join_map):
        joins = {j["lcol"]: j["type"] for j in _joins(_section(join_map, "## Dimension Joins"))
                 if j["left"] == DAILY}
        assert joins["ITM_STK_STS_DMS_KEY"] == "LEFT"
        assert joins["ITM_DMS_KEY"] == "INNER"


class TestARoleIsTaughtUnderItsOwnAlias:

    def test_each_role_of_one_table_has_an_alias(self, join_map):
        roles = [j for j in _joins(join_map) if j["right"] == "WH.MART.ABC_CLS_DMS"
                 and j["left"] == DAILY]
        assert {j["lcol"]: j["alias"] for j in roles} == {
            "ABC_CLS_MNL_DMS_KEY": "abc_class_manual",
            "ABC_CLS_VOL_DMS_KEY": "abc_class_volume",
        }
        assert all(j["rhs"] == j["alias"] for j in roles)

    def test_the_buyer(self, join_map):
        buyer = next(j for j in _joins(join_map) if j["lcol"] == "BYR_PTY_DMS_KEY")
        assert (buyer["right"], buyer["alias"], buyer["rcol"]) == (
            "WH.MART.PTY_DMS", "buyer", "PTY_DMS_KEY")


class TestTheOtherPassesStay:

    def test_the_date_roles(self, join_map):
        dates = _section(join_map, "## Role-Playing Date Dimension Joins")
        assert "ITM_BAL_EFC_DT_DMS_KEY" in dates

    def test_the_databases_own_constraints(self):
        master = dict(MASTER)
        master["__db_fk_constraints__"] = [{
            "source": "azure_sql", "constraint_name": "FK_BAL_ITEM",
            "parent_schema": "MART", "parent_table": "ITM_BAL_DLY_FCT",
            "parent_col": "ITM_DMS_KEY", "ref_schema": "MART", "ref_table": "ITM_DMS",
            "ref_col": "ITM_DMS_KEY", "ordinal": 1, "enforced": True,
        }]
        assert "## DB-Enforced FK Constraints" in _build_join_map(master)
