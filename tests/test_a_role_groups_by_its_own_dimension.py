"""
A role groups by its own dimension, through its own key.

"Inventory value by buyer" was left to the model, and so was its French twin.
The buyer is a role: BYR_PTY_DMS_KEY holds the party who buys, and the join
graph knew it -- an edge from the fact to the party dimension labelled
"Buyer". The field planner did not. It read the key by its own words, "buyer
party", which "by buyer" does not say, and would have looked for the key's
label on a table named for BYR_PTY, which no warehouse has: the party
dimension keys its rows by PTY_DMS_KEY.

A key the graph names a role for now answers to the role's name, and shows
its role's dimension -- the party's name -- joined through the pair the edge
declares, BYR_PTY_DMS_KEY = PTY_DMS_KEY. "Acheteur" is read as "buyer".

A synthetic tenant (tests/answer_harness.py) whose daily snapshot records a
buyer for each row; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness

FACT = "WH.MART.ITM_BAL_DLY_FCT"
PARTY = "WH.MART.PTY_DMS"
COLUMNS = {
    FACT: {"WHS_DMS_KEY": "int", "BYR_PTY_DMS_KEY": "int", "ON_HND_QTY": "decimal", "ITM_CST": "decimal"},
    PARTY: {"PTY_DMS_KEY": "int", "PTY_CD": "nvarchar", "PTY_NM": "nvarchar"},
    "WH.MART.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_DSC": "nvarchar"},
}
BUYER = {"table": "MART.ITM_BAL_DLY_FCT", "column": "BYR_PTY_DMS_KEY",
         "to_table": "MART.PTY_DMS", "to_column": "PTY_DMS_KEY", "label": "Buyer"}


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("roles")) as built:
        yield built


def _by_buyer(amount) -> dict:
    """An amount of the newest snapshot per buyer's name, from the harness rows."""
    totals: dict = {}
    for row in harness.STOCK:
        name = harness.PARTIES[row[2]][1]
        totals[name] = totals.get(name, 0) + amount(row)
    return totals


def _answered(answer: dict, measure: str) -> dict:
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return {row["BUYER"]: row[measure] for row in run["rows"]}


def _plan(question: str, columns: dict = COLUMNS, role_keys=(BUYER,)) -> dict:
    from core.semantic_planner import build_semantic_field_plan

    roles = {"role_keys": list(role_keys)} if role_keys else {}
    return build_semantic_field_plan(question, columns, None, **roles)


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Inventory value by buyer", "en"),
        ("Valeur du stock par acheteur", "fr"),
    ])
    def test_the_value_of_each_buyers_stock(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        on_hand, cost = 4, 6
        assert _answered(answer, "INVENTORY_VALUE") == pytest.approx(
            _by_buyer(lambda row: row[on_hand] * row[cost]))

    def test_the_top_buyer(self, warehouse):
        answer = harness.ask(warehouse, "Top 1 buyer by inventory value")
        assert answer["model_wrote_sql"] is False
        values = _by_buyer(lambda row: row[4] * row[6])
        leader = max(values, key=values.get)
        assert _answered(answer, "INVENTORY_VALUE") == pytest.approx({leader: values[leader]})


class TestThePlan:

    def test_the_role_is_shown_by_its_dimensions_name(self):
        (field,) = _plan("Inventory value by buyer")["fields"]
        assert (field["term"], field["table"], field["column"]) == ("buyer", PARTY, "PTY_NM")
        assert (field["source_key_column"], field["display_key_column"]) == ("BYR_PTY_DMS_KEY", "PTY_DMS_KEY")

    def test_it_is_joined_through_the_roles_own_pair(self):
        # A measure the planner binds on the fact anchors the join.
        plan = _plan("Sum of ON_HND_QTY by buyer")
        assert [(join["from"], join["to"], join["conditions"]) for join in plan["joins"]] == [
            (FACT, PARTY, [("BYR_PTY_DMS_KEY", "PTY_DMS_KEY")])]

    def test_the_label_is_read_on_the_roles_own_table(self):
        # Another table keyed by the party and named like a party dimension
        # would win the label on its name.
        columns = dict(COLUMNS, **{"WH.MART.DIM_PTY": {"PTY_DMS_KEY": "int", "PTY_NM": "nvarchar"}})
        (field,) = _plan("Inventory value by buyer", columns)["fields"]
        assert field["table"] == PARTY

    def test_a_key_the_graph_names_no_role_for_is_read_by_its_words(self):
        assert _plan("Inventory value by buyer", role_keys=())["fields"] == []

    def test_acheteur_is_buyer(self):
        from core.question_normalizer import canonical_question

        assert "buyer" in canonical_question("Valeur du stock par acheteur", "fr").split()


class TestTheGraphsRoles:

    def test_a_labelled_key_from_a_fact_to_a_dimension(self):
        from core.graph_resolver import role_keys

        def entity(name, kind, table=None):
            return {"entity_name": name, "table_name": table or name, "schema_name": "MART", "entity_type": kind}

        def edge(source, column, target, to_column, label):
            return {"from_entity": source, "from_column": column, "to_entity": target,
                    "to_column": to_column, "label": label}

        graph = {
            "entities": [entity("ITM_BAL_DLY_FCT", "fact"), entity("PTY_DMS", "dimension"),
                         entity("SLR_DMS", "dimension"), entity("WHS_DMS", "dimension"),
                         entity("Creation Date", "dimension", "DT_DMS")],
            "relationships": [
                edge("ITM_BAL_DLY_FCT", "BYR_PTY_DMS_KEY", "PTY_DMS", "PTY_DMS_KEY", "Buyer"),
                edge("ITM_BAL_DLY_FCT", "WHS_DMS_KEY", "WHS_DMS", "WHS_DMS_KEY", ""),
                edge("ITM_BAL_DLY_FCT", "ITM_WHS_CRN_DT_DMS_KEY", "Creation Date", "DT_DMS_KEY", "Creation Date"),
                edge("SLR_DMS", "BYR_PTY_DMS_KEY", "PTY_DMS", "PTY_DMS_KEY", "Buyer"),
            ],
        }
        assert role_keys(graph) == [BUYER]


class TestTheRepair:
    """A model that grouped by the role's key is repaired to its label, joined
    through the dimension's own key."""

    _COLS = {"MART.ITM_BAL_DLY_FCT": COLUMNS[FACT], "MART.PTY_DMS": COLUMNS[PARTY]}

    def test_the_join_uses_the_dimensions_key(self):
        from core.pipeline_helpers import attempt_field_plan_repair
        from core.validator import validate_sql

        plan = {"enabled": True, "fields": [{
            "term": "buyer", "table": "MART.PTY_DMS", "column": "PTY_NM",
            "role": "display_dimension", "display_required": True,
            "source_key_column": "BYR_PTY_DMS_KEY", "display_key_column": "PTY_DMS_KEY",
            "source_key_table": "MART.ITM_BAL_DLY_FCT",
        }], "joins": []}
        sql = ("SELECT f.BYR_PTY_DMS_KEY, SUM(f.ON_HND_QTY) AS ON_HAND "
               "FROM MART.ITM_BAL_DLY_FCT f GROUP BY f.BYR_PTY_DMS_KEY")
        fixed = attempt_field_plan_repair(
            sql, "azure_sql", set(self._COLS), None, self._COLS, {"semantic_plan": plan})
        assert fixed
        ok, reason, _ = validate_sql(fixed, set(self._COLS), "azure_sql", table_columns=self._COLS,
                                     semantic_context={"semantic_plan": plan})
        assert ok, reason
        import sqlglot

        (join,) = sqlglot.parse_one(fixed, dialect="tsql").find_all(sqlglot.exp.Join)
        assert sorted(column.name for column in join.args["on"].find_all(sqlglot.exp.Column)) == [
            "BYR_PTY_DMS_KEY", "PTY_DMS_KEY"]
