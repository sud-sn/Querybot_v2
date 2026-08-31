"""
tests/test_inventory_vocabulary_reaches_the_balance_fact.py

Catalogue case B4 -- "show me the inventory value by warehouse" -- could not be
answered, and the reason turned out not to be a bug in any rule. The tenant's
inventory fact had no business vocabulary at all.

`packs/infor_m3.json` carried BAL -> balance, QTY -> quantity, AMT -> amount,
ITM -> item and WHS -> warehouse, but not VAL, OH or AVL. So the balance
columns expanded only halfway: BAL_VAL_AMT read as "balance val amount", with
the measure word left as meaningless noise, and the fact itself had no
table_dict entry so its entity label was "Itm Bal Prd". Nothing connected the
word "inventory" to ITM_BAL_PRD_FCT, and `build_semantic_field_plan` returned a
single field -- the warehouse dimension -- with `fact_anchor` empty and no
measure at all.

Note that adding VAL -> value alone would have changed nothing: "value" is in
`_RUNTIME_MATCH_STOPWORDS` (core/semantic_model.py), deliberately, because
every measure in every registry is a value of something. The alias has to carry
the SUBJECT word, which is why `direct_aliases` gained "inventory value" rather
than the abbreviation alone.

These tests run the real planner inside the real pack, because the whole claim
is about what the pack makes reachable. A test that read the JSON and asserted
a key exists would pass while the planner still saw nothing.
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

_tmpdir = tempfile.mkdtemp(prefix="querybot_invvocab_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.semantic_planner import build_semantic_field_plan  # noqa: E402
from core.vocab_packs import (  # noqa: E402
    _clone_builtin,
    _merge_pack,
    activate_vocab,
    deactivate_vocab,
    load_pack,
)

SNAPSHOT = "EMDW_DMART.ITM_BAL_PRD_FCT"
INVOICE = "EMDW_DMART.CUS_ORD_IVC_FCT"
PURCHASE = "EMDW_DMART.PCH_ORD_RCT_FCT"

# The mart as deploy/emco_dmart/01_create_model.sql declares it.
TABLES = {
    SNAPSHOT: {
        "ITM_DMS_KEY": "int", "WHS_DMS_KEY": "int", "PRD_DMS_KEY": "int",
        "BAL_DT_DMS_KEY": "int", "OH_QTY": "decimal", "ALC_QTY": "decimal",
        "AVL_QTY": "decimal", "PCH_QTY": "decimal", "UNT_CST_AMT": "decimal",
        "BAL_VAL_AMT": "decimal",
    },
    "EMDW_DMART.WHS_DMS": {"WHS_DMS_KEY": "int", "WHS_NM": "varchar"},
    INVOICE: {"SOP_CUS_IVC_LIN_AMT": "decimal", "CUS_DMS_KEY": "int"},
    PURCHASE: {"PCH_ORD_LIN_CAD_AMT": "decimal", "SUP_DMS_KEY": "int"},
}
FACTS = {SNAPSHOT, INVOICE, PURCHASE}


def _m3_vocab():
    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
    return vocab


def _plan(question: str) -> dict:
    vocab = _m3_vocab()
    token = activate_vocab(vocab)
    try:
        return build_semantic_field_plan(
            question, TABLES, allowed_tables=set(TABLES),
            selected_schema="EMDW_DMART", vocab=vocab, fact_tables=FACTS,
        )
    finally:
        deactivate_vocab(token)


def _measures(plan: dict) -> list[tuple[str, str]]:
    return [
        (str(f.get("table") or ""), str(f.get("column") or ""))
        for f in plan.get("fields") or []
        if str(f.get("role") or "") == "measure"
    ]


class TestTheInventoryQuestionResolves(unittest.TestCase):

    def test_b4_anchors_on_the_balance_fact(self):
        self.assertEqual(_plan("show me the inventory value by warehouse")
                         .get("fact_anchor"), SNAPSHOT)

    def test_b4_binds_the_balance_value_measure(self):
        self.assertIn((SNAPSHOT, "BAL_VAL_AMT"),
                      _measures(_plan("show me the inventory value by warehouse")))

    def test_the_warehouse_dimension_survives(self):
        """The one field that already worked must not be displaced by the new
        measure."""
        plan = _plan("show me the inventory value by warehouse")
        dimensions = [
            (str(f.get("table") or ""), str(f.get("column") or ""))
            for f in plan.get("fields") or []
            if "dimension" in str(f.get("role") or "")
        ]
        self.assertIn(("EMDW_DMART.WHS_DMS", "WHS_NM"), dimensions)

    def test_the_quantity_balances_are_reachable_too(self):
        for question, column in (
            ("show me the stock on hand by warehouse", "OH_QTY"),
            ("what is the available quantity by warehouse", "AVL_QTY"),
            ("show the allocated quantity by warehouse", "ALC_QTY"),
        ):
            with self.subTest(question=question):
                self.assertIn((SNAPSHOT, column), _measures(_plan(question)))


class TestTheNewVocabularyStaysInItsLane(unittest.TestCase):
    """The inventory fact carries PCH_QTY, and "purchase" bound to it once
    before -- see the comment at core/semantic_model.py:2085. Making the fact
    reachable for inventory questions must not widen that."""

    def test_a_revenue_question_does_not_reach_the_balance_fact(self):
        measures = _measures(_plan("what is my revenue by each customer"))
        self.assertNotIn(SNAPSHOT, [table for table, _ in measures])

    def test_the_purchase_order_question_is_completely_unaffected(self):
        """B1 is the case the OPPOSITE bug was fixed for, so the bar for a
        vocabulary addition is that it changes B1 by nothing at all.

        This asserts the planner's whole measure verdict, not merely that the
        answer is acceptable -- a weaker check would pass while the new aliases
        quietly shifted the ranking underneath it.

        Note what this deliberately does NOT claim: PCH_QTY IS emitted here as
        a candidate, with `enforcement` unset, and that predates this pack
        entirely (verified by running this question against the pack with and
        without the inventory entries -- the output is identical). Demoting it
        is `_scope_plan_to_single_fact`'s job one layer up, which
        tests/test_measure_fact_and_timeout_handling.py already guards. This
        file has no business re-litigating that at the wrong layer.
        """
        plan = _plan(
            "what is the total amount of confirmed purchase orders by profit center"
        )
        self.assertEqual(_measures(plan), [(SNAPSHOT, "PCH_QTY")])
        self.assertEqual(plan.get("fact_anchor"), SNAPSHOT)


class TestTheAbbreviationsExpandFully(unittest.TestCase):

    def test_no_meaningless_token_survives_expansion(self):
        """"balance val amount" was the actual expansion. A token that is still
        an abbreviation after expansion is one the pack does not know."""
        from core.semantic_planner import _column_words

        vocab = _m3_vocab()
        token = activate_vocab(vocab)
        try:
            words = [w.lower() for w in _column_words("BAL_VAL_AMT", vocab=vocab)]
        finally:
            deactivate_vocab(token)
        self.assertIn("value", words)
        self.assertNotIn("val", words)


if __name__ == "__main__":
    unittest.main(verbosity=2)
