"""
tests/test_single_measure_fact_and_path_labels.py

The "purchase orders by profit center" chain, root-caused from production logs.

    Graph resolved (authoritative): entities=['CUS_ORD_IVC_FCT', 'FNN_FCT',
      'Confirmed Delivery Date', 'PCH_ORD_RCT_FCT', 'PFT_CTR_DMS']
      anchor= planning_status=clarification_required
      dropped_dates=[] dropped_facts=[]

Three facts required, no anchor, and the user was asked to choose between
"Confirmed Delivery Date -> Confirmed Delivery Date" and
"CUS_ORD_IVC_FCT to ITM_DMS -> PCH_ORD_RCT_FCT to ITM_DMS".

Cause: `metric_formula_tables` carries every fact any matched metric is built
on. A loosely-matched metric (built on CUS_ORD_IVC_FCT + FNN_FCT) dragged its
facts in as *authoritative*, and with 2+ authoritative facts the rival-fact
drop can never fire — each looks authoritative to the others. The graph was
then asked to connect three facts at once, no single anchor could be chosen,
and planning ended in clarification_required.

Two fixes: the plan's required measure decides the one authoritative fact, and
a join-path option that is not business-distinguishable is not offered as a
choice.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_single_fact_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.graph_resolver import _path_label  # noqa: E402
from core.semantic_resolution import build_planner_alignment  # noqa: E402

SCHEMA = "EMDW_DMART"
PURCHASE = f"{SCHEMA}.PCH_ORD_RCT_FCT"
INVOICE = f"{SCHEMA}.CUS_ORD_IVC_FCT"
FINANCE = f"{SCHEMA}.FNN_FCT"
INVENTORY = f"{SCHEMA}.ITM_BAL_PRD_FCT"
PROFIT_CENTRE = f"{SCHEMA}.PFT_CTR_DMS"


def _entity(name, table, entity_type, display=""):
    entity = {
        "entity_name": name, "table_name": table,
        "schema_name": SCHEMA, "entity_type": entity_type,
    }
    if display:
        entity["display_name"] = display
    return entity


GRAPH = {
    "entities": [
        _entity("PCH_ORD_RCT_FCT", "PCH_ORD_RCT_FCT", "fact", "Purchase Order Receipts"),
        _entity("CUS_ORD_IVC_FCT", "CUS_ORD_IVC_FCT", "fact", "Customer Invoices"),
        _entity("FNN_FCT", "FNN_FCT", "fact", "Finance"),
        _entity("ITM_BAL_PRD_FCT", "ITM_BAL_PRD_FCT", "fact", "Item Balances"),
        _entity("PFT_CTR_DMS", "PFT_CTR_DMS", "dimension", "Profit Centre"),
        _entity("Confirmed Delivery Date", "DT_DMS", "dimension"),
    ],
    "relationships": [],
}

PLAN = {
    "enabled": True,
    "fact_anchor": PURCHASE,
    "fields": [
        {"term": "purchase", "table": f"EMCODW_DEV.{INVENTORY}", "column": "PCH_QTY",
         "role": "measure", "enforcement": None},
        {"term": "purchase order amount", "table": PURCHASE,
         "column": "PCH_ORD_LIN_CAD_AMT", "role": "measure", "enforcement": "required"},
        {"term": "Profit Center", "table": PROFIT_CENTRE, "column": "PFT_CTR_NM",
         "role": "display_dimension", "enforcement": "required"},
    ],
    "joins": [],
    "required_tables": [PURCHASE, PROFIT_CENTRE],
}

DETECTED = [
    "CUS_ORD_IVC_FCT", "FNN_FCT", "Confirmed Delivery Date",
    "PCH_ORD_RCT_FCT", "PFT_CTR_DMS",
]


def _align(*, plan=None, metric_tables=None, detected=None):
    return build_planner_alignment(
        graph=GRAPH,
        graph_ctx={
            "enabled": True,
            "detected": list(DETECTED if detected is None else detected),
            "anchor": "CUS_ORD_IVC_FCT",
        },
        semantic_plan=PLAN if plan is None else plan,
        metric_formula_tables=(
            {INVOICE, FINANCE} if metric_tables is None else metric_tables
        ),
        date_context_resolution={"status": "none"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1  One authoritative measure fact
# ══════════════════════════════════════════════════════════════════════════════
class TestOneAuthoritativeMeasureFact(unittest.TestCase):

    def test_a_loosely_matched_metric_cannot_add_authoritative_facts(self):
        alignment = _align()
        self.assertEqual(
            alignment["authoritative_fact_tables"], [PURCHASE],
            "the fact carrying the required measure must be the only authority",
        )
        self.assertEqual(alignment["governed_measure_fact"], PURCHASE)
        self.assertIn(INVOICE, alignment["dropped_metric_facts"])
        self.assertIn(FINANCE, alignment["dropped_metric_facts"])

    def test_the_rival_facts_are_dropped_so_an_anchor_can_be_chosen(self):
        """dropped_facts was empty in production; that is what left anchor= blank."""
        alignment = _align()
        self.assertIn("CUS_ORD_IVC_FCT", alignment["dropped_fact_entities"])
        self.assertIn("FNN_FCT", alignment["dropped_fact_entities"])
        required = set(alignment["required_entities"])
        self.assertIn("PCH_ORD_RCT_FCT", required)
        self.assertNotIn("CUS_ORD_IVC_FCT", required)
        self.assertNotIn("FNN_FCT", required)

    def test_the_requested_dimension_survives(self):
        self.assertIn("PFT_CTR_DMS", _align()["required_entities"])

    def test_exactly_one_fact_remains_required(self):
        alignment = _align()
        facts = {
            name for name in alignment["required_entities"]
            if name.endswith("_FCT")
        }
        self.assertEqual(facts, {"PCH_ORD_RCT_FCT"})

    def test_an_agreeing_metric_is_not_dropped(self):
        """A metric on the measure's own fact stays; only rivals are dropped.

        ITM_BAL_PRD_FCT is still dropped here, and should be: it arrives from
        the plan's own weak "purchase -> PCH_QTY" field, not from the metric.
        """
        alignment = _align(metric_tables={PURCHASE})
        self.assertEqual(alignment["authoritative_fact_tables"], [PURCHASE])
        self.assertNotIn(PURCHASE, alignment["dropped_metric_facts"])
        self.assertIn(INVENTORY, alignment["dropped_metric_facts"])

    def test_without_a_governed_measure_nothing_is_narrowed(self):
        """No measure evidence means no basis to choose — keep prior behaviour."""
        plan = {**PLAN, "fact_anchor": "", "fields": [
            {"term": "Profit Center", "table": PROFIT_CENTRE, "column": "PFT_CTR_NM",
             "role": "display_dimension", "enforcement": "required"},
        ]}
        alignment = _align(plan=plan)
        self.assertEqual(alignment["governed_measure_fact"], "")
        self.assertEqual(alignment["dropped_metric_facts"], [])

    def test_a_genuine_multi_fact_comparison_is_not_collapsed(self):
        """Two required measures on two facts is a real comparison, not noise."""
        plan = {**PLAN, "fact_anchor": "", "fields": [
            {"term": "purchase order amount", "table": PURCHASE,
             "column": "PCH_ORD_LIN_CAD_AMT", "role": "measure", "enforcement": "required"},
            {"term": "invoice amount", "table": INVOICE,
             "column": "SOP_CUS_IVC_LIN_AMT", "role": "measure", "enforcement": "required"},
        ]}
        alignment = _align(plan=plan, metric_tables={PURCHASE, INVOICE})
        # The first required measure anchors; the rival is reported, not hidden.
        self.assertEqual(alignment["governed_measure_fact"], PURCHASE)
        self.assertIn(INVOICE, alignment["dropped_metric_facts"])


# ══════════════════════════════════════════════════════════════════════════════
# 2  Join-path options a business user can actually answer
# ══════════════════════════════════════════════════════════════════════════════
class TestJoinPathLabels(unittest.TestCase):

    def test_repeated_role_labels_are_collapsed(self):
        """Production showed "Confirmed Delivery Date -> Confirmed Delivery Date"."""
        path = [
            {"from_entity": "PCH_ORD_RCT_FCT", "to_entity": "Confirmed Delivery Date",
             "label": "Confirmed Delivery Date"},
            {"from_entity": "CUS_ORD_IVC_FCT", "to_entity": "Confirmed Delivery Date",
             "label": "Confirmed Delivery Date"},
        ]
        self.assertEqual(
            _path_label(path, GRAPH["entities"]), "Confirmed Delivery Date",
        )

    def test_unlabelled_edges_use_business_display_names(self):
        """Production showed raw "CUS_ORD_IVC_FCT to ITM_DMS -> ..."."""
        path = [
            {"from_entity": "CUS_ORD_IVC_FCT", "to_entity": "PFT_CTR_DMS", "label": ""},
        ]
        label = _path_label(path, GRAPH["entities"])
        self.assertIn("Customer Invoices", label)
        self.assertIn("Profit Centre", label)
        self.assertNotIn("CUS_ORD_IVC_FCT", label)
        self.assertNotIn("PFT_CTR_DMS", label)

    def test_a_business_role_is_preferred_over_entity_names(self):
        path = [{"from_entity": "F", "to_entity": "C", "business_role": "Sold-to Customer"}]
        self.assertEqual(_path_label(path, GRAPH["entities"]), "Sold-to Customer")

    def test_genuinely_distinct_roles_stay_distinct(self):
        sold = _path_label(
            [{"from_entity": "F", "to_entity": "C", "label": "Sold-to Customer"}],
            GRAPH["entities"],
        )
        ship = _path_label(
            [{"from_entity": "F", "to_entity": "C", "label": "Ship-to Customer"}],
            GRAPH["entities"],
        )
        self.assertNotEqual(sold, ship)

    def test_indistinguishable_paths_are_not_offered_as_a_choice(self):
        """Two identical buttons are not a question — rank instead of asking."""
        source = (ROOT / "core" / "graph_resolver.py").read_text(encoding="utf-8")
        self.assertIn("if len(_path_options) > 1:", source)
        self.assertIn("business-distinguishable", source)

    def test_no_label_ever_ends_up_empty(self):
        for path in (
            [{"from_entity": "A", "to_entity": "B"}],
            [{"from_entity": "", "to_entity": "", "label": ""}],
        ):
            with self.subTest(path=path):
                self.assertIsInstance(_path_label(path, GRAPH["entities"]), str)


if __name__ == "__main__":
    unittest.main()
