# -*- coding: utf-8 -*-
""""Net sales and returns by warehouse" planned a path through Customer.

Two facts and one grain. What the plan needed was two edges, one per fact,
each landing on Warehouse:

    Invoice.WHS_DMS_KEY -> Warehouse.WHS_DMS_KEY
    Return.WHS_DMS_KEY  -> Warehouse.WHS_DMS_KEY

What the path finder produced, measured on the real resolver against an
M3-shaped graph:

    Invoice.CUS_DMS_KEY -> Customer.CUS_DMS_KEY
    Return.CUS_DMS_KEY  -> Customer.CUS_DMS_KEY      (backward)
    Invoice.WHS_DMS_KEY -> Warehouse.WHS_DMS_KEY

    common_dimensions = []
    isolated_fact_plans[*].group_by = []

The join tree grew from the anchor fact and treated every other detected
entity -- the second fact included -- as a target to REACH. A fact can only be
reached backward through a dimension, so Return was reached through whichever
shared dimension ranked cheapest, and its own edge to the grain the question
named was never in the plan. compile_join_plan then found no dimension both
facts touched; analytical_request_plan copies common_dimensions into every
isolated sub-plan's GROUP BY; and a per-warehouse comparison compiled to two
grand totals -- or, when the model wrote the correct SQL and joined Return to
Warehouse anyway, to a refusal for using an edge the plan had left out. As a
bonus, "via Customer" and "via Warehouse" tied and were disclosed as a ranked
choice for a question that had already said which.

Facts are roots, not destinations. Each is searched on its own against the
requested dimensions and the union is the plan. A second defect in the same
family fell out of writing these tests: compile_join_plan counted only edges
landing DIRECTLY on a requested dimension, so a grain reached through a
snowflake hop (Invoice -> Item -> Product Category) read as shared by nothing.
It now walks the planned edges transitively, with other facts as walls.

Every test drives ``resolve_for_question`` -- the entry point the pipeline
calls before SQL generation -- and asserts on the plan it hands back. The last
one feeds that plan to the validator with the SQL the plan asks for, so the
write side and the read side are checked against each other with nothing
passed between them by the test.
"""

from __future__ import annotations

import unittest

import pytest
import sqlglot

from core.graph_resolver import resolve_for_question
from core.join_planner import compile_join_plan
from core.validator import _join_plan_contract_errors

SCHEMA = "EMDW_DMART"


def _entity(name, table, kind, **extra):
    return {
        "entity_name": name, "display_name": name.replace("_", " "),
        "table_name": table, "schema_name": SCHEMA, "entity_type": kind,
        "status": "confirmed", **extra,
    }


def _edge(id_, source, target, left, right):
    return {
        "id": id_, "from_entity": source, "to_entity": target,
        "from_column": left, "to_column": right,
        "relationship_type": "many_to_one", "join_type": "INNER",
        "status": "confirmed", "validation_status": "validated",
    }


# EMCO's mart in shape: three facts, a snowflaked product dimension, and one
# date dimension played as three roles. Names are M3's, because the naming
# conventions are half of what the deterministic layers read.
GRAPH = {
    "entities": [
        _entity("Invoice", "CUS_ORD_IVC_FCT", "fact"),
        _entity("Return", "CUS_RTN_FCT", "fact"),
        _entity("Inventory_Balance", "INV_BAL_FCT", "fact"),
        _entity("Customer", "CUS_DMS", "dimension"),
        _entity("Warehouse", "WHS_DMS", "dimension"),
        _entity("Item", "ITM_DMS", "dimension"),
        _entity("Product_Category", "PRD_CAT_DMS", "dimension"),
        _entity("Invoice_Date", "DT_DMS", "dimension",
                role_alias="invoice_date", business_role="Invoice Date"),
        _entity("Return_Date", "DT_DMS", "dimension",
                role_alias="return_date", business_role="Return Date"),
    ],
    "relationships": [
        _edge(1, "Invoice", "Customer", "CUS_DMS_KEY", "CUS_DMS_KEY"),
        _edge(2, "Invoice", "Warehouse", "WHS_DMS_KEY", "WHS_DMS_KEY"),
        _edge(3, "Invoice", "Item", "ITM_DMS_KEY", "ITM_DMS_KEY"),
        _edge(4, "Item", "Product_Category", "PRD_CAT_DMS_KEY", "PRD_CAT_DMS_KEY"),
        _edge(5, "Invoice", "Invoice_Date", "CUS_IVC_DT_DMS_KEY", "DT_DMS_KEY"),
        _edge(7, "Return", "Customer", "CUS_DMS_KEY", "CUS_DMS_KEY"),
        _edge(8, "Return", "Warehouse", "WHS_DMS_KEY", "WHS_DMS_KEY"),
        _edge(9, "Return", "Return_Date", "RTN_DT_DMS_KEY", "DT_DMS_KEY"),
        _edge(10, "Inventory_Balance", "Warehouse", "WHS_DMS_KEY", "WHS_DMS_KEY"),
        _edge(11, "Inventory_Balance", "Item", "ITM_DMS_KEY", "ITM_DMS_KEY"),
    ],
    "properties": [],
}


def resolve(question, *entities, anchor="CUS_ORD_IVC_FCT"):
    """The real entry point, with the store kept out of it."""
    return resolve_for_question(
        question, "tenant", "azure_sql", graph=GRAPH,
        required_entities=set(entities),
        metric_formula_tables={anchor},
        use_suggested=False,
    )


def _edge_ids(result):
    return sorted(result.get("edge_ids") or [])


def _touches(result, entity):
    return any(
        entity in (edge.get("from_entity"), edge.get("to_entity"))
        for edge in result.get("resolved_edges") or []
    )


class TestEachFactCarriesItsOwnEdgeToTheGrain:

    def test_sales_and_returns_by_warehouse_plans_both_warehouse_edges(self):
        """The defect. Two facts, one grain, two edges -- and nothing else."""
        result = resolve("net sales and returns by warehouse",
                         "Invoice", "Return", "Warehouse")
        assert result["planning_status"] == "requires_isolated_aggregation"
        assert _edge_ids(result) == [2, 8]

    def test_the_grain_is_recognised_as_common_to_both_facts(self):
        result = resolve("net sales and returns by warehouse",
                         "Invoice", "Return", "Warehouse")
        plan = result["join_plan"]
        assert plan["common_dimensions"] == ["Warehouse"]
        assert [sub["group_by"] for sub in plan["isolated_fact_plans"]] == (
            [["Warehouse"], ["Warehouse"]]
        )

    def test_no_edge_to_a_dimension_nobody_asked_for(self):
        """Customer was the cheapest way to *reach* Return. It is not part of
        the question, and joining it in both CTEs is a silent change to the
        row set."""
        result = resolve("net sales and returns by warehouse",
                         "Invoice", "Return", "Warehouse")
        assert not _touches(result, "Customer")
        assert not _touches(result, "Item")

    def test_the_second_fact_is_never_a_destination(self):
        """Every planned edge points forward from a fact to a dimension. A
        backward edge is the signature of reaching a fact through a shared
        dimension, which is the fan-out shape the planner exists to refuse."""
        result = resolve("net sales and returns by warehouse",
                         "Invoice", "Return", "Warehouse")
        for edge in result["resolved_edges"]:
            assert edge["direction"] == "forward", edge

    def test_and_there_is_no_ranked_choice_for_a_question_that_already_chose(self):
        result = resolve("net sales and returns by warehouse",
                         "Invoice", "Return", "Warehouse")
        assert not result.get("ranked_relationship")

    def test_three_facts_conform_to_one_grain(self):
        result = resolve("net sales, returns and inventory balance by warehouse",
                         "Invoice", "Return", "Inventory_Balance", "Warehouse")
        assert result["planning_status"] == "requires_isolated_aggregation"
        assert _edge_ids(result) == [2, 8, 10]
        assert result["join_plan"]["common_dimensions"] == ["Warehouse"]


class TestAGrainReachedThroughASnowflakeHop:

    def test_sales_and_inventory_by_product_category_share_the_category(self):
        """Neither fact touches Product Category directly; both reach it
        through Item. That IS their common grain."""
        result = resolve("net sales and inventory balance by product category",
                         "Invoice", "Inventory_Balance", "Product_Category")
        assert result["planning_status"] == "requires_isolated_aggregation"
        assert result["join_plan"]["common_dimensions"] == ["Product_Category"]
        assert all(sub["group_by"] == ["Product_Category"]
                   for sub in result["join_plan"]["isolated_fact_plans"])

    def test_the_shared_hop_is_planned_once(self):
        """Item -> Product Category is reached from both facts. It is one
        physical join; two copies would mean two rows in the plan for one
        relationship and a duplicate in the validator's expected edges."""
        result = resolve("net sales and inventory balance by product category",
                         "Invoice", "Inventory_Balance", "Product_Category")
        ids = result["edge_ids"]
        assert sorted(ids) == [3, 4, 11]
        assert len(ids) == len(set(ids))


class TestWhatCannotBeConformed:

    def test_two_facts_with_no_grain_are_two_scalars(self):
        """"Net sales and returns" -- no dimension at all. Nothing to join;
        each fact aggregates alone and the totals sit side by side. Before,
        this was blocked as "cannot reach Return"."""
        result = resolve("net sales and returns", "Invoice", "Return")
        assert result["planning_status"] == "requires_isolated_aggregation"
        assert result["edge_ids"] == []
        assert result["join_plan"]["fact_entities"] == ["Invoice", "Return"]
        assert all(sub["group_by"] == []
                   for sub in result["join_plan"]["isolated_fact_plans"])

    def test_a_fact_that_cannot_reach_the_grain_blocks_and_says_which(self):
        """Return has no path to Product Category. Conforming both facts to a
        grain one of them cannot express is not a comparison, and the reason
        names the fact rather than listing the dimension as missing."""
        result = resolve("net sales and returns by product category",
                         "Invoice", "Return", "Product_Category")
        assert result["planning_status"] == "blocked"
        assert "Return" in result["reason"]
        assert "Product_Category" in result["reason"]
        assert "Product_Category" in result["missing_entities"]

    def test_the_fact_that_can_reach_it_is_not_blamed(self):
        result = resolve("net sales and returns by product category",
                         "Invoice", "Return", "Product_Category")
        assert "Invoice has no governed path" not in result["reason"]


class TestWhatMustNotChange:

    def test_a_single_fact_path_is_as_before(self):
        result = resolve("net sales by warehouse", "Invoice", "Warehouse")
        assert result["planning_status"] == "selected"
        assert _edge_ids(result) == [2]
        assert result["join_skeleton"].startswith(f"FROM [{SCHEMA}].[CUS_ORD_IVC_FCT]")

    def test_a_single_fact_snowflake_is_as_before(self):
        result = resolve("net sales by product category", "Invoice", "Product_Category")
        assert result["planning_status"] == "selected"
        assert _edge_ids(result) == [3, 4]

    def test_a_single_fact_that_cannot_reach_its_grain_still_blocks(self):
        result = resolve("returns by product category", "Return", "Product_Category",
                         anchor="CUS_RTN_FCT")
        assert result["planning_status"] == "blocked"
        assert "Product_Category" in result["missing_entities"]

    def test_the_anti_join_existence_path_is_untouched(self):
        """The one legitimate fact-to-fact traversal: a LEFT JOIN to find what
        is missing. It must keep going through the original tree, where the
        second fact IS the destination."""
        graph = {
            "entities": [
                _entity("Order", "F_RX_ORDER", "fact"),
                _entity("Fill", "F_RX_FILL", "fact"),
            ],
            "relationships": [{
                "id": 1, "from_entity": "Order", "to_entity": "Fill",
                "from_column": "ORDER_ID", "to_column": "ORDER_ID",
                "relationship_type": "one_to_many", "join_type": "LEFT",
                "status": "confirmed", "validation_status": "valid",
            }],
            "properties": [],
        }
        result = resolve_for_question(
            "orders without fills", "tenant", "azure_sql", graph=graph,
            intent={"wants_missing_records": True}, use_suggested=False,
        )
        assert result["anchor"] == "Order"
        assert result["join_skeleton"].startswith(f"FROM [{SCHEMA}].[F_RX_ORDER]")
        assert "LEFT  JOIN" in result["join_skeleton"]


class TestThePlannerOnItsOwnTerms:
    """compile_join_plan is a layer with a contract of its own and is fed
    hand-built paths by its own tests. Its transitive walk must not cross
    another fact: a dimension is not a fact's grain because a DIFFERENT fact
    in the same plan happens to reach it."""

    def test_a_dimension_behind_another_fact_is_not_shared(self):
        graph = {
            "entities": [
                _entity("A", "A_FCT", "fact"), _entity("B", "B_FCT", "fact"),
                _entity("X", "X_DMS", "dimension"), _entity("Y", "Y_DMS", "dimension"),
            ],
            "relationships": [
                _edge(1, "A", "X", "X_KEY", "X_KEY"),
                _edge(2, "B", "X", "X_KEY", "X_KEY"),
                _edge(3, "B", "Y", "Y_KEY", "Y_KEY"),
            ],
            "properties": [],
        }
        # A -> X, B -> X, B -> Y. Walked freely, A reaches Y through B and
        # A's sub-plan would GROUP BY a column A does not have.
        plan = compile_join_plan(graph, ["A", "B", "X", "Y"], list(graph["relationships"]),
                                 metric_formula_tables={"A_FCT"})
        assert plan["status"] == "requires_isolated_aggregation"
        assert plan["common_dimensions"] == ["X"]
        assert [sub["group_by"] for sub in plan["isolated_fact_plans"]] == [["X"], ["X"]]


class TestTheValidatorAcceptsWhatThePlannerNowAsksFor(unittest.TestCase):
    """Write side to read side. The plan the resolver produces is handed to
    the validator together with the SQL that plan describes -- two CTEs, each
    grouped to the shared grain, joined on it -- and must pass."""

    def test_two_ctes_conformed_to_warehouse_pass_the_contract(self):
        context = resolve("net sales and returns by warehouse",
                          "Invoice", "Return", "Warehouse")
        sql = """
            WITH sales AS (
                SELECT inv.WHS_DMS_KEY, SUM(inv.NET_SLS_AMT) AS NET_SALES
                FROM EMDW_DMART.CUS_ORD_IVC_FCT inv
                GROUP BY inv.WHS_DMS_KEY
            ), returns AS (
                SELECT rtn.WHS_DMS_KEY, SUM(rtn.RTN_AMT) AS RETURNS
                FROM EMDW_DMART.CUS_RTN_FCT rtn
                GROUP BY rtn.WHS_DMS_KEY
            )
            SELECT w.WHS_DSC, s.NET_SALES, r.RETURNS
            FROM EMDW_DMART.WHS_DMS w
            LEFT JOIN sales s ON s.WHS_DMS_KEY = w.WHS_DMS_KEY
            LEFT JOIN returns r ON r.WHS_DMS_KEY = w.WHS_DMS_KEY
        """
        tree = sqlglot.parse_one(sql, read="tsql")
        self.assertEqual(_join_plan_contract_errors(tree, context), [])

    def test_and_the_raw_fact_to_fact_shape_is_still_refused(self):
        context = resolve("net sales and returns by warehouse",
                          "Invoice", "Return", "Warehouse")
        sql = """
            SELECT w.WHS_DSC, SUM(inv.NET_SLS_AMT), SUM(rtn.RTN_AMT)
            FROM EMDW_DMART.CUS_ORD_IVC_FCT inv
            JOIN EMDW_DMART.WHS_DMS w ON inv.WHS_DMS_KEY = w.WHS_DMS_KEY
            JOIN EMDW_DMART.CUS_RTN_FCT rtn ON rtn.WHS_DMS_KEY = w.WHS_DMS_KEY
            GROUP BY w.WHS_DSC
        """
        tree = sqlglot.parse_one(sql, read="tsql")
        errors = _join_plan_contract_errors(tree, context)
        self.assertTrue(errors)
        self.assertEqual(errors[0]["code"], "raw_fact_to_fact_join")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
