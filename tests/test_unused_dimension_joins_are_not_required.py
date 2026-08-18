"""
tests/test_unused_dimension_joins_are_not_required.py

Catalogue check B1 — "total amount of confirmed purchase orders by profit
center". After the profit-centre binding was fixed the label was right, the
metric was right and CFM_FLG = 1 was applied, but the NUMBER was ~7.5x too
large:

    FROM PCH_ORD_RCT_FCT pch
    LEFT JOIN  PFT_CTR_DMS pft      ON pch.PFT_CTR_DMS_KEY = pft.PFT_CTR_DMS_KEY
    INNER JOIN PFT_CTR_CUS_DAT pft2 ON pft2.PFT_CTR_DMS_KEY = pft.PFT_CTR_DMS_KEY
    INNER JOIN CUS_DMS cus          ON pft2.CUS_DMS_KEY = cus.CUS_DMS_KEY
    WHERE pch.CFM_FLG = 1
    GROUP BY pft.PFT_CTR_NM

PFT_CTR_CUS_DAT is a many-to-many bridge, so every purchase-order row was
counted once per customer in that profit centre. Valid SQL, real declared
foreign keys, correct grouping label, no validator error — just a wrong total.

The question never mentions customers. CUS_DMS was in the plan's
required_tables because build_runtime_semantic_plan emitted a REQUIRED join for
every dimension that merely SCORED, whether or not it contributed a field, and
required_semantic_tables marks both endpoints of a required join as required.

A join earns "required" only by carrying a field the answer must show.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_joinreq_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_model import build_runtime_semantic_plan  # noqa: E402
from core.semantic_plan_utils import required_semantic_tables  # noqa: E402

PCH = "EMDW_DMART.PCH_ORD_RCT_FCT"
PFT = "EMDW_DMART.PFT_CTR_DMS"
CUS = "EMDW_DMART.CUS_DMS"

MODEL = {
    "tables": [{
        "qualified_name": PCH,
        "schema": "EMDW_DMART",
        "fields": [],
        "dimensions": [
            {"name": "Profit Center", "source_key": "PFT_CTR_DMS_KEY",
             "display_table": PFT, "display_column": "PFT_CTR_NM",
             "display_key": "PFT_CTR_DMS_KEY", "confidence": 90},
            {"name": "Customer", "source_key": "CUS_DMS_KEY",
             "display_table": CUS, "display_column": "CUS_NM",
             "display_key": "CUS_DMS_KEY", "confidence": 90},
        ],
    }],
}

QUESTION = "what is the total amount of confirmed purchase orders by profit center"


def _plan(question=QUESTION):
    return build_runtime_semantic_plan(
        "", question=question, selected_schema="EMDW_DMART", model=MODEL,
    )


class TestOnlyDimensionsTheAnswerShowsAreRequired(unittest.TestCase):

    def test_the_customer_table_is_not_required_by_a_profit_centre_question(self):
        self.assertNotIn(CUS, required_semantic_tables(_plan()))

    def test_the_profit_centre_table_is_still_required(self):
        self.assertIn(PFT, required_semantic_tables(_plan()))

    def test_the_join_that_carries_the_answer_stays_required(self):
        joins = {(j["from"], j["to"]): j.get("enforcement") for j in _plan()["joins"]}
        self.assertEqual(joins.get((PCH, PFT)), "required")

    def test_a_join_to_a_table_the_answer_does_not_show_never_forces_it(self):
        # Absent is fine; present-but-optional is fine. What must not happen is
        # a required edge, because required_semantic_tables promotes BOTH of its
        # endpoints into the plan.
        joins = {(j["from"], j["to"]): j.get("enforcement") for j in _plan()["joins"]}
        self.assertNotEqual(joins.get((PCH, CUS)), "required")

    def test_a_scored_but_unshown_dimension_is_demoted_not_required(self):
        # "purchase order" scores the Customer dimension through CUS_DMS_KEY
        # without the answer ever showing a customer.
        plan = _plan("total confirmed purchase order amount by profit center name")
        for join in plan["joins"]:
            if join["to"] == CUS:
                self.assertEqual(join.get("enforcement"), "optional")

    def test_a_question_that_does_ask_for_customers_still_requires_them(self):
        required = required_semantic_tables(_plan("purchase order amount by customer"))
        self.assertIn(CUS, required)

    def test_a_question_naming_both_requires_both(self):
        required = required_semantic_tables(
            _plan("purchase order amount by profit center and customer"))
        self.assertIn(CUS, required)
        self.assertIn(PFT, required)


if __name__ == "__main__":
    unittest.main(verbosity=2)
