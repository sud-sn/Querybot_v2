"""
An inventory measure is classified by what it is, not by the suffix it ends in.

A periodic balance fact carries three kinds of measure side by side, and they
aggregate differently:

- levels -- stock on hand, allocated, on order, aging buckets. The quantity at
  the snapshot. Summable across items and warehouses, never across months.
- movements and counts of events -- purchased, sold, shipped, transferred,
  reclassified, number of receipts. What happened DURING the period. Summable
  across months: March's purchases plus April's are the two months' purchases.
- unit values -- item cost, average item cost, prices. Never summable at all;
  stock is valued as quantity times cost.

The governed path used to know one thing: a measure on a snapshot fact is
semi-additive. So every movement on a balance fact was reported for one period
when a year was asked for, every unit cost was summed across items, and the
counts and the oldest aging bucket were not measures at all because they carry
no _QTY or _AMT suffix. The knowledge base the SQL model reads said the
opposite of the governed path for the stock levels: "_QTY -- safe to SUM
across all dimensions".

These drive the real builders -- the semantic model from a schema file, the
snapshot decision the pipeline calls, the knowledge-base hints, the narrative's
merge rule -- and assert on what they return. Synthetic tables in a mart's
naming convention; no customer data.
"""

from __future__ import annotations

import json
import re

import pytest

from core.analysis_contract import (
    collapse_rows_by_label,
    measure_additivity,
    measure_class_for_column,
    measure_class_for_metric,
)
from core.contextual_dates import question_has_snapshot_intent
from core.naming_convention import get_naming_hints
from core.schema_enrichment import _role_for_column, enrich_columns
from core.semantic_model import build_semantic_model
from core.table_role_classifier import classify_table, is_periodic_snapshot_fact


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


BALANCE_FACT = "WH.MART.ITM_BAL_PRD_FCT"
INVOICE_FACT = "WH.MART.CUS_ORD_IVC_FCT"

LEVELS = ("CUR_ON_HND_QTY", "ALC_ON_HND_QTY", "ALC_QTY", "ORD_QTY", "RSV_QTY",
          "AGE_0_9_QTY", "AGE_24_PLU")
# An ERP measure code says nothing to the name rules; the table's grain decides.
ERP_LEVELS = ("ALQT",)
MOVEMENTS = ("PCH_QTY", "SLD_QTY", "DRC_SHP_QTY", "PSV_TFR_QTY", "NGV_RE_CLS_QTY")
COUNTS = ("NUM_OF_RCT", "NUM_OF_RET", "NUM_OF_PHY_INV")
UNIT_VALUES = ("ITM_CST", "AVG_ITM_CST")

SCHEMA = {
    BALANCE_FACT: _table(
        "ITM_BAL_PRD_FCT_KEY",
        ("ITM_BAL_PRD_FCT_KEY", "bigint"),
        ("ITM_DMS_KEY", "int"),
        ("WHS_DMS_KEY", "int"),
        ("PRD_DMS_KEY", "int"),
        *((name, "decimal(18,4)") for name in LEVELS if name != "AGE_24_PLU"),
        ("AGE_24_PLU", "int"),
        *((name, "decimal(18,4)") for name in ERP_LEVELS),
        *((name, "decimal(18,4)") for name in MOVEMENTS),
        *((name, "int") for name in COUNTS),
        *((name, "decimal(18,4)") for name in UNIT_VALUES),
        ("AVG_ON_HND_QTY", "decimal(18,4)"),
    ),
    INVOICE_FACT: _table(
        "CUS_ORD_IVC_FCT_KEY",
        ("CUS_ORD_IVC_FCT_KEY", "bigint"),
        ("CUS_DMS_KEY", "int"),
        ("ITM_DMS_KEY", "int"),
        ("CUS_IVC_DT_DMS_KEY", "int"),
        ("SOP_CUS_IVC_LIN_AMT", "decimal(18,2)"),
        ("IVC_QTY", "decimal(18,4)"),
        ("ORD_QTY", "decimal(18,4)"),
        ("ITM_CST", "decimal(18,4)"),
    ),
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("inventory_model")
    (directory / "_schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")
    return build_semantic_model(str(directory))


def _fields(model: dict, table: str) -> dict[str, dict]:
    for entry in model["tables"]:
        if entry["table"] == table:
            return {field["column"]: field for field in entry["fields"]}
    raise AssertionError(f"{table} missing from the model")


class TestTheNameSaysWhatTheMeasureIs:

    @pytest.mark.parametrize("column", [
        "ON_HND_QTY", "CUR_ON_HND_QTY", "ALC_ON_HND_QTY", "ONHAND_QTY",
        "AGE_0_9_QTY", "AGE_24_PLU", "AGED_30_60_AMT", "YTD_SLS_AMT",
    ])
    def test_a_stock_level_is_semi_additive(self, column):
        assert measure_additivity(column) == ("semi_additive", "balance")

    @pytest.mark.parametrize("column", [
        "ITM_CST", "AVG_ITM_CST", "UNT_CST", "STD_CST", "SLS_PRC", "UnitPrice",
    ])
    def test_a_price_or_cost_per_unit_is_never_summed(self, column):
        assert measure_additivity(column) == ("non_additive", "unit_value")

    @pytest.mark.parametrize("column", ["PCH_QTY", "SLD_QTY", "DRC_SHP_QTY",
                                        "PSV_TFR_QTY", "NGV_RE_CLS_QTY", "NET_SLS_AMT"])
    def test_a_movement_is_a_flow(self, column):
        assert measure_additivity(column) == ("additive", "flow")

    @pytest.mark.parametrize("column", ["NUM_OF_RCT", "NUM_OF_PHY_INV", "NBR_OF_DLV"])
    def test_a_number_of_events_is_a_count(self, column):
        assert measure_additivity(column) == ("additive", "event_count")

    @pytest.mark.parametrize("column", ["EXT_CST", "LIN_CST_AMT", "PCE_QTY", "ORD_QTY"])
    def test_an_extended_amount_or_a_quantity_is_not_a_unit_value(self, column):
        assert measure_additivity(column)[0] == "additive"

    @pytest.mark.parametrize("column", ["PENDING_QTY", "SPENDING_AMT"])
    def test_ending_inside_another_word_is_not_a_closing_balance(self, column):
        assert measure_additivity(column) == ("additive", "generic")

    @pytest.mark.parametrize("column", ["HND_CHG_AMT", "HANDLING_AMT"])
    def test_hand_without_on_is_not_stock_on_hand(self, column):
        assert measure_additivity(column)[0] != "semi_additive"

    @pytest.mark.parametrize("column", ["AGE", "EMP_AGE", "AGE_GRP_CD"])
    def test_an_age_without_bucket_bounds_is_not_an_aging_bucket(self, column):
        assert measure_additivity(column)[1] != "balance"

    def test_an_admin_declaration_outranks_the_name(self):
        assert measure_additivity("ITM_CST", {"aggregation_semantics": "additive"}) == (
            "additive", "declared")

    def test_the_average_of_a_level_is_an_average(self):
        assert measure_additivity("AVG_ON_HND_QTY") == ("non_additive", "average_or_rate")


class TestAFormulaTakesItsQuantitysAdditivity:
    """A unit cost times a quantity is valued stock or valued movement."""

    @pytest.mark.parametrize("formula, expected", [
        ("SUM(f.SLD_QTY * f.ITM_CST)", "additive"),
        ("SUM(ON_HND_QTY * AVG_ITM_CST)", "semi_additive"),
        ("SUM(ITM_CST)", "non_additive"),
        ("MAX(SLS_PRC)", "non_additive"),
        ("SUM(PCH_QTY)", "additive"),
    ])
    def test_the_metric_class(self, formula, expected):
        assert measure_class_for_metric({"name": "m", "sql_template": formula}) == expected


class TestTheModelKnowsEachKindOfMeasure:

    @pytest.mark.parametrize("column", LEVELS + ERP_LEVELS)
    def test_a_level_on_the_snapshot_is_semi_additive(self, model, column):
        assert _fields(model, "ITM_BAL_PRD_FCT")[column]["aggregation"] == "semi_additive"

    @pytest.mark.parametrize("column", MOVEMENTS + COUNTS)
    def test_a_movement_or_a_count_on_the_snapshot_still_adds_up(self, model, column):
        assert _fields(model, "ITM_BAL_PRD_FCT")[column]["aggregation"] == "additive"

    @pytest.mark.parametrize("column", UNIT_VALUES + ("AVG_ON_HND_QTY",))
    def test_a_unit_value_or_an_average_is_never_summed(self, model, column):
        assert _fields(model, "ITM_BAL_PRD_FCT")[column]["aggregation"] == "non_additive"

    @pytest.mark.parametrize("column", COUNTS + ("AGE_24_PLU",))
    def test_counts_and_the_last_aging_bucket_are_measures(self, model, column):
        assert _fields(model, "ITM_BAL_PRD_FCT")[column]["role"] == "measure"
        balance = next(t for t in model["tables"] if t["table"] == "ITM_BAL_PRD_FCT")
        assert column in {measure["column"] for measure in balance["measures"]}

    def test_the_transaction_fact_keeps_its_flows_additive(self, model):
        fields = _fields(model, "CUS_ORD_IVC_FCT")
        for column in ("SOP_CUS_IVC_LIN_AMT", "IVC_QTY", "ORD_QTY"):
            assert fields[column]["aggregation"] == "additive", column

    def test_a_unit_cost_on_a_transaction_fact_is_still_a_unit_value(self, model):
        assert _fields(model, "CUS_ORD_IVC_FCT")["ITM_CST"]["aggregation"] == "non_additive"


class TestTheColumnRoleSeesCountsAndBuckets:

    def test_untyped_counts_and_buckets_are_measures(self):
        roles = {item.column: item.role for item in enrich_columns(
            ["NUM_OF_RCT", "NBR_OF_DLV", "AGE_24_PLU", "AGE_GRP_CD"])}
        assert roles["NUM_OF_RCT"] == "measure"
        assert roles["NBR_OF_DLV"] == "measure"
        assert roles["AGE_24_PLU"] == "measure"
        assert roles["AGE_GRP_CD"] != "measure"

    @pytest.mark.parametrize("data_type", ["int", "bigint", "decimal(18,2)", "NUMBER(10)"])
    def test_a_numeric_count_is_a_measure(self, data_type):
        assert _role_for_column("NUM_OF_RCT", data_type)[0] == "measure"

    def test_a_text_column_named_like_a_count_is_not(self):
        assert _role_for_column("NUM_OF_RCT", "varchar(20)")[0] != "measure"


class TestTheQuestionAnchorsOnTheGrain:
    """What the pipeline calls to decide whether to read one snapshot."""

    TABLE = "MART.ITM_BAL_PRD_FCT"

    def _anchors(self, column, table=TABLE, **extra):
        field = {"role": "measure", "column": column, **extra}
        if table:
            field["table"] = table
        return question_has_snapshot_intent(
            "how much do we have by warehouse", measure_fields=[field])

    def test_the_question_carries_no_wording_signal(self):
        assert not question_has_snapshot_intent("how much do we have by warehouse")

    @pytest.mark.parametrize("column", ["ALC_QTY", "ORD_QTY", "RSV_QTY", "ITM_CST"])
    def test_a_level_named_like_anything_anchors_on_a_snapshot_fact(self, column):
        assert self._anchors(column)

    @pytest.mark.parametrize("column", MOVEMENTS + COUNTS)
    def test_a_movement_or_a_count_never_anchors(self, column):
        assert not self._anchors(column)

    def test_the_same_column_without_its_table_does_not_anchor(self):
        assert not self._anchors("ALC_QTY", table="")

    def test_the_same_column_on_a_transaction_fact_does_not_anchor(self):
        assert not self._anchors("ORD_QTY", table="MART.CUS_ORD_IVC_FCT")

    def test_an_admin_declaration_still_wins(self):
        assert not self._anchors("ALC_QTY", aggregation_semantics="additive")

    def test_a_bracketed_table_name_is_read(self):
        assert self._anchors("ALC_QTY", table="[MART].[ITM_BAL_PRD_FCT]")


class TestASnapshotIsNamedInWholeWords:

    @pytest.mark.parametrize("table", ["ITM_BAL_PRD_FCT", "ITM_BAL_DLY_FCT",
                                       "STK_SNP_FCT", "INVENTORY_POSITION_FCT"])
    def test_a_snapshot_fact(self, table):
        assert classify_table(table).fact_type == "periodic_snapshot"
        assert is_periodic_snapshot_fact(f"MART.{table}")

    @pytest.mark.parametrize("table", ["GLOBAL_SALES_FCT", "VERBAL_ORD_FCT", "CUS_ORD_IVC_FCT"])
    def test_bal_inside_another_word_is_not_a_snapshot(self, table):
        assert classify_table(table).fact_type == "transaction"
        assert not is_periodic_snapshot_fact(table)

    def test_a_dimension_is_never_a_snapshot_fact(self):
        assert not is_periodic_snapshot_fact("MART.ITM_BAL_DMS")


class TestTheKnowledgeBaseSaysWhatTheModelSays:
    """The text the SQL model reads, parsed back into verdicts."""

    @staticmethod
    def _verdicts(hints: str) -> dict[str, str]:
        verdicts = {}
        for line in hints.splitlines():
            match = re.match(r"- (\w+) \[.*?aggregation=(\w+)", line)
            if match:
                verdicts[match.group(1)] = match.group(2)
        return verdicts

    def _hints(self, table: str) -> str:
        columns = [column["name"] for column in SCHEMA[table]["columns"]]
        return get_naming_hints(columns, table.split(".")[-1])

    def test_every_measure_reads_as_the_model_classifies_it(self, model):
        for table in (BALANCE_FACT, INVOICE_FACT):
            verdicts = self._verdicts(self._hints(table))
            fields = _fields(model, table.split(".")[-1])
            for column, field in fields.items():
                # An ERP code has no naming-convention line; its hint is the
                # ERP dictionary's, and the snapshot rule above covers it.
                if field["role"] != "measure" or column in ERP_LEVELS:
                    continue
                assert verdicts.get(column) == field["aggregation"], (table, column)

    def test_the_snapshot_says_what_may_not_be_summed_across_periods(self):
        hints = self._hints(BALANCE_FACT)
        assert "SNAPSHOT GRAIN [ITM_BAL_PRD_FCT]" in hints
        line = next(line for line in hints.splitlines() if line.startswith("- ALC_QTY "))
        assert "NEVER SUM across dates or periods" in line

    def test_a_movement_on_the_snapshot_is_spelled_out_as_summable(self):
        hints = self._hints(BALANCE_FACT)
        line = next(line for line in hints.splitlines() if line.startswith("- PCH_QTY "))
        assert "Safe to SUM across periods" in line

    def test_the_transaction_fact_has_no_snapshot_rule(self):
        hints = self._hints(INVOICE_FACT)
        assert "SNAPSHOT GRAIN" not in hints
        assert self._verdicts(hints)["IVC_QTY"] == "additive"


class TestTheNarrativeMergesOnlyWhatAddsUp:
    """Rows sharing a label across two months are merged only when summable."""

    ROWS = [
        {"warehouse": "North", "month": "2026-03", "value": 40},
        {"warehouse": "North", "month": "2026-04", "value": 60},
        {"warehouse": "South", "month": "2026-03", "value": 10},
    ]

    def test_a_movement_is_summed_across_months(self):
        assert collapse_rows_by_label(
            self.ROWS, "warehouse", "value", measure_name="PCH_QTY",
        ) == [("North", 100.0), ("South", 10.0)]

    @pytest.mark.parametrize("column", ["CUR_ON_HND_QTY", "ITM_CST", "AGE_24_PLU"])
    def test_a_level_or_a_unit_value_is_not(self, column):
        assert collapse_rows_by_label(
            self.ROWS, "warehouse", "value", measure_name=column) is None

    def test_a_count_of_events_is_summable(self):
        assert measure_class_for_column("NUM_OF_RCT") == "additive"
