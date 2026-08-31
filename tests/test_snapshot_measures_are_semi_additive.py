"""
tests/test_snapshot_measures_are_semi_additive.py

Catalogue check B4 — "show me the inventory value by warehouse" must use
ITM_BAL_PRD_FCT without summing a balance across periods.

Semi-additivity was recognised only from a column-name SUFFIX (_BAL, _INV).
EMCO's inventory value column is BAL_VAL_AMT, where BAL is a PREFIX, so
endswith("_BAL") is False, it fell through to the _AMT rule, and the KB
published it as "additive — safe to SUM across all dimensions".

Summing a month-end balance across 18 months of snapshots overstates inventory
eighteen-fold. No validator fires: SUM of a decimal column is valid SQL and the
number looks plausible.

Additivity is a property of the TABLE's grain, not of how a column happens to
be spelled. The table role classifier already identifies ITM_BAL_PRD_FCT as a
periodic_snapshot, so trust that rather than the naming convention.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_semiadd_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.analysis_contract import _measure_class, measure_class_for_metric  # noqa: E402
from core.contextual_dates import question_has_snapshot_intent  # noqa: E402
from core.naming_convention import match_column_suffix  # noqa: E402
from core.schema_enrichment import enrich_columns  # noqa: E402
from core.semantic_model import _field_entry  # noqa: E402
from core.table_role_classifier import classify_schema_tables  # noqa: E402

SNAPSHOT = "EMDW_DMART.ITM_BAL_PRD_FCT"
TRANSACTION = "EMDW_DMART.CUS_ORD_IVC_FCT"

SCHEMA = {
    SNAPSHOT: {
        "columns": [
            {"name": "ITM_BAL_PRD_FCT_KEY", "type": "bigint"},
            {"name": "ITM_DMS_KEY", "type": "int"},
            {"name": "WHS_DMS_KEY", "type": "int"},
            {"name": "PRD_DMS_KEY", "type": "int"},
            {"name": "BAL_DT_DMS_KEY", "type": "int"},
            {"name": "OH_QTY", "type": "decimal"},
            {"name": "AVL_QTY", "type": "decimal"},
            {"name": "BAL_VAL_AMT", "type": "decimal"},
        ],
        "pk_columns": ["ITM_BAL_PRD_FCT_KEY"],
    },
    TRANSACTION: {
        "columns": [
            {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
            {"name": "CUS_DMS_KEY", "type": "int"},
            {"name": "CUS_IVC_DT_DMS_KEY", "type": "int"},
            {"name": "SOP_CUS_IVC_LIN_AMT", "type": "decimal"},
            {"name": "IVC_QTY", "type": "decimal"},
        ],
        "pk_columns": ["CUS_ORD_IVC_FCT_KEY"],
    },
}


def _aggregations(fqn):
    roles = classify_schema_tables(SCHEMA)
    classification = roles.get(fqn)
    snapshot = bool(
        classification
        and classification.role == "fact"
        and classification.fact_type in {"periodic_snapshot", "accumulating_snapshot"}
    )
    meta = SCHEMA[fqn]
    columns = [c["name"] for c in meta["columns"]]
    return {
        entry["column"]: entry["aggregation"]
        for entry in (
            _field_entry(item, meta, snapshot_fact=snapshot)
            for item in enrich_columns(columns)
        )
    }


class TestTheDefectItself(unittest.TestCase):

    def test_the_suffix_rule_genuinely_misses_a_prefixed_balance(self):
        # The premise: nothing in the naming convention catches BAL_VAL_AMT.
        rule = match_column_suffix("BAL_VAL_AMT")
        self.assertIsNotNone(rule)
        self.assertNotEqual(
            rule.aggregation, "semi_additive",
            "premise no longer holds — the suffix rule now catches this",
        )

    def test_the_classifier_does_recognise_the_snapshot(self):
        roles = classify_schema_tables(SCHEMA)
        self.assertEqual(roles[SNAPSHOT].fact_type, "periodic_snapshot")


class TestSnapshotMeasuresAreNotSummableAcrossTime(unittest.TestCase):

    def test_the_inventory_value_is_semi_additive(self):
        self.assertEqual(_aggregations(SNAPSHOT).get("BAL_VAL_AMT"), "semi_additive")

    def test_the_quantity_balances_are_semi_additive_too(self):
        aggregations = _aggregations(SNAPSHOT)
        for column in ("OH_QTY", "AVL_QTY"):
            with self.subTest(column=column):
                self.assertEqual(aggregations.get(column), "semi_additive")

    def test_keys_are_not_relabelled_as_measures(self):
        aggregations = _aggregations(SNAPSHOT)
        for column in ("ITM_DMS_KEY", "WHS_DMS_KEY", "BAL_DT_DMS_KEY"):
            with self.subTest(column=column):
                self.assertNotEqual(aggregations.get(column), "semi_additive")

    def test_a_transaction_fact_keeps_additive_measures(self):
        # Revenue on an invoice line IS a flow and must stay summable.
        aggregations = _aggregations(TRANSACTION)
        self.assertNotEqual(
            aggregations.get("SOP_CUS_IVC_LIN_AMT"), "semi_additive",
        )


class TestTheRegistryMetricPathSeesItToo(unittest.TestCase):
    """The schema layer above learns semi-additivity from the TABLE's grain.

    `measure_class_for_metric` cannot: a registry metric carries a formula and
    a name, not a table role, so it classifies from the column name alone --
    and it called `_measure_class(token, {})` with an empty field dict, so the
    enrichment verdict proved above never reached it. On abbreviated ERP names
    that fell straight through to the _AMT / _QTY rule and returned "additive".

    That is not cosmetic. `metrics_are_semi_additive` is the GOVERNED half of
    `question_has_snapshot_intent`, and its docstring promises it holds "for
    any domain whose snapshot measure happens not to be called inventory or
    balance (headcount, assets under management, open subscriptions)". Those
    are exactly the domains with no wording fallback, so a wrong verdict here
    left them with no snapshot detection at all.
    """

    def test_the_abbreviated_balance_column_is_semi_additive(self):
        self.assertEqual(
            measure_class_for_metric(
                {"name": "Inventory Value", "sql_template": "SUM(BAL_VAL_AMT)"}
            ),
            "semi_additive",
        )

    def test_a_flow_measure_is_still_additive(self):
        for formula in ("SUM(SOP_CUS_IVC_LIN_AMT)", "SUM(PCH_ORD_LIN_CAD_AMT)"):
            with self.subTest(formula=formula):
                self.assertEqual(
                    measure_class_for_metric({"name": "m", "sql_template": formula}),
                    "additive",
                )

    def test_an_admin_declaration_still_wins(self):
        """Naming is the fallback, never an override of governed metadata."""
        self.assertEqual(
            measure_class_for_metric({
                "name": "Inventory Value",
                "sql_template": "SUM(BAL_VAL_AMT)",
                "aggregation_semantics": "additive",
            }),
            "additive",
        )

    def test_tokens_match_whole_words_not_substrings(self):
        """"BAL" as a substring also fires on GLOBAL_AMT, and a false
        semi-additive verdict suppresses a legitimate SUM."""
        for column in ("GLOBAL_AMT", "HANDLING_AMT", "VERBAL_SCORE_AMT"):
            with self.subTest(column=column):
                self.assertNotEqual(_measure_class(column, {}), "semi_additive")

    def test_the_abbreviations_that_are_too_ambiguous_stay_out(self):
        """INV is invoice far more often than inventory; CLS is class; OPN is
        an open order. Claiming these would break additive flow measures."""
        for column in ("INV_LIN_AMT", "CLS_CD_AMT", "OPN_ORD_AMT"):
            with self.subTest(column=column):
                self.assertEqual(_measure_class(column, {}), "additive")

    def test_a_camel_case_mart_is_covered(self):
        """The spelled-out set needs the underscore form: StockOnHandQty
        tokenises to {STOCK, ON, HAND, QTY} and matches "ON_HAND" nowhere."""
        self.assertEqual(_measure_class("StockOnHandQty", {}), "semi_additive")


class TestTheGovernedSnapshotPromiseHolds(unittest.TestCase):
    """End of the chain: the thing the pipeline actually calls.

    These execute `question_has_snapshot_intent` rather than asserting on the
    classifier, because the classifier being right is only useful if the
    governed branch above the wording fallback consumes it.
    """

    HEADCOUNT = {"name": "Month-end headcount", "sql_template": "SUM(EOM_BAL_HC)"}

    def test_a_domain_with_no_wording_cue_is_still_detected(self):
        question = "what is my month-end headcount by division"
        # The premise: wording alone cannot save this one.
        self.assertFalse(question_has_snapshot_intent(question))
        # The governed metric must.
        self.assertTrue(
            question_has_snapshot_intent(question, matched_metrics=[self.HEADCOUNT])
        )

    def test_a_flow_metric_does_not_acquire_snapshot_intent(self):
        revenue = {"name": "Revenue", "sql_template": "SUM(SOP_CUS_IVC_LIN_AMT)"}
        self.assertFalse(
            question_has_snapshot_intent(
                "what is my revenue by customer", matched_metrics=[revenue]
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
