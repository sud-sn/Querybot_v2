"""
A table of stock movements is a ledger of events, not a stock position.

Stock is held as levels and changed by movements, and a warehouse can model
either as a fact: ITM_BAL_DLY_FCT holds the position of each item on each day,
STK_MVT_FCT one row per receipt, issue or transfer. The table role classifier
called any table whose name mentioned stock -- INVENTORY, STOCK, or since the
last change the STK abbreviation -- a periodic snapshot. On a snapshot, a
quantity whose name says nothing about time is a level, so a movement table's
ITM_QTY became semi-additive: "units moved this year" was answered with one
day's movements, the knowledge base told the SQL model never to sum them
across dates, and the question-time check pinned the question to one snapshot.

The name now decides by what the table records: an explicit snapshot word
(SNAPSHOT, BALANCE, SNAP, BAL) is a snapshot; a stock table is a snapshot
unless its name says it holds movements (TRANSACTION, MOVEMENT, MVT, TRN,
ADJ, RCT, TFR, ...), and "in transit" stock is still a level.

Drives the classifier, the semantic model built from a schema file, the
question-time snapshot check and the knowledge-base hints. Synthetic tables in
a mart's naming convention; no customer data.
"""

from __future__ import annotations

import json

import pytest

from core.contextual_dates import measures_are_semi_additive
from core.naming_convention import get_naming_hints
from core.semantic_model import build_semantic_model
from core.table_role_classifier import classify_table, is_periodic_snapshot_fact

MOVEMENT_TABLES = [
    "MART.STK_MVT_FCT",
    "MART.STK_TRN_FCT",
    "MART.STK_ADJ_FCT",
    "MART.STK_RCT_FCT",
    "MART.STK_TFR_FCT",
    "MART.INVENTORY_TRANSACTION_FCT",
    "MART.INVENTORY_ADJUSTMENT_FCT",
    "MART.STOCK_MOVEMENT_FCT",
    "MART.STOCK_LEDGER_FCT",
    "[MART].[STK_MVT_FCT]",
]

LEVEL_TABLES = [
    "MART.ITM_BAL_DLY_FCT",
    "MART.ITM_BAL_PRD_FCT",
    "MART.STK_LVL_FCT",
    "MART.STOCK_FCT",
    "MART.STOCK_ON_HAND_FCT",
    "MART.INVENTORY_SNAPSHOT_FCT",
    "MART.STK_IN_TRN_FCT",
    "MART.ITM_IN_TRN_BAL_FCT",
    "MART.INVENTORY_MOVEMENT_SNAPSHOT_FCT",
    "MART.GL_BAL_FCT",
]


class TestTheNameSaysWhichItHolds:

    @pytest.mark.parametrize("table", MOVEMENT_TABLES)
    def test_a_movement_table_is_a_transaction_fact(self, table):
        bare = table.split(".")[-1].strip("[]")
        role = classify_table(bare)
        assert (role.role, role.fact_type) == ("fact", "transaction")
        assert is_periodic_snapshot_fact(table) is False

    @pytest.mark.parametrize("table", LEVEL_TABLES)
    def test_a_level_table_is_still_a_snapshot(self, table):
        assert is_periodic_snapshot_fact(table) is True

    def test_a_table_about_something_else_is_neither(self):
        assert is_periodic_snapshot_fact("MART.GLOBAL_SALES_FCT") is False
        assert is_periodic_snapshot_fact("MART.CUS_ORD_IVC_FCT") is False


def _table(own_key: str, *columns: tuple[str, str]) -> dict:
    return {
        "columns": [
            {"name": name, "type": dtype, "nullable": True, "comment": ""}
            for name, dtype in columns
        ],
        "pk_columns": [own_key],
        "row_count": 1000,
        "comment": "",
        "schema": "MART",
        "database": "WH",
    }


SCHEMA = {
    "WH.MART.STK_MVT_FCT": _table(
        "STK_MVT_FCT_KEY",
        ("STK_MVT_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("MVT_DT_DMS_KEY", "int"), ("ITM_QTY", "decimal"), ("EXT_CST_AMT", "decimal"),
    ),
    "WH.MART.ITM_BAL_DLY_FCT": _table(
        "ITM_BAL_DLY_FCT_KEY",
        ("ITM_BAL_DLY_FCT_KEY", "bigint"), ("ITM_DMS_KEY", "int"), ("WHS_DMS_KEY", "int"),
        ("ITM_BAL_EFC_DT_DMS_KEY", "int"), ("ITM_QTY", "decimal"), ("RSV_QTY", "decimal"),
    ),
    "WH.MART.ITM_DMS": _table(
        "ITM_DMS_KEY", ("ITM_DMS_KEY", "int"), ("ITM_CD", "varchar"), ("ITM_DSC", "varchar"),
    ),
    "WH.MART.DT_DMS": _table(
        "DT_DMS_KEY", ("DT_DMS_KEY", "int"), ("DMS_DT", "date"), ("YR", "int"), ("MTH", "int"),
    ),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("movement_model")
    (directory / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return build_semantic_model(str(directory))


def _tables(model) -> dict:
    return {table["table"]: table for table in model["tables"]}


def _aggregation(table: dict, column: str) -> str:
    return next(
        str(field.get("aggregation") or "")
        for field in table["fields"] if field.get("column") == column
    )


class TestTheSemanticModel:

    def test_the_movement_fact_is_a_transaction_fact(self, model):
        movements = _tables(model)["STK_MVT_FCT"]
        assert (movements["type"], movements["fact_type"]) == ("fact", "transaction")

    def test_its_quantities_add_up_across_dates(self, model):
        movements = _tables(model)["STK_MVT_FCT"]
        assert _aggregation(movements, "ITM_QTY") == "additive"
        assert _aggregation(movements, "EXT_CST_AMT") == "additive"

    def test_the_balance_fact_beside_it_keeps_its_levels(self, model):
        balances = _tables(model)["ITM_BAL_DLY_FCT"]
        assert balances["fact_type"] == "periodic_snapshot"
        assert _aggregation(balances, "ITM_QTY") == "semi_additive"
        assert _aggregation(balances, "RSV_QTY") == "semi_additive"


def _measure(column: str, table: str) -> dict:
    return {"role": "measure", "column": column, "table": table}


class TestTheQuestionTimeCheck:
    """The check that pins a question to one snapshot reads the table's name."""

    def test_a_movement_quantity_is_not_pinned_to_one_snapshot(self):
        assert measures_are_semi_additive([_measure("ITM_QTY", "MART.STK_MVT_FCT")]) is False
        assert measures_are_semi_additive([_measure("ITM_QTY", "MART.STK_TRN_FCT")]) is False

    def test_the_same_name_on_a_balance_fact_is(self):
        assert measures_are_semi_additive([_measure("ITM_QTY", "MART.ITM_BAL_DLY_FCT")]) is True


class TestTheKnowledgeBaseHints:
    """The SQL model is told the grain in the table's hint block."""

    def test_a_movement_table_gets_no_snapshot_grain_line(self):
        hints = get_naming_hints(["ITM_QTY", "EXT_CST_AMT"], table_name="STK_MVT_FCT")
        assert "SNAPSHOT GRAIN" not in hints

    def test_a_balance_table_does(self):
        hints = get_naming_hints(["ITM_QTY", "RSV_QTY"], table_name="ITM_BAL_DLY_FCT")
        assert "SNAPSHOT GRAIN [ITM_BAL_DLY_FCT]" in hints
