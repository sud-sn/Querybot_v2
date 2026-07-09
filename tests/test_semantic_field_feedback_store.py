"""
tests/test_semantic_field_feedback_store.py

Storage round-trip for the "Business terms" box: save_semantic_field_feedback
must persist suggested_synonyms alongside meaning/use_case, and it must
survive the migration path (v33: ALTER TABLE ADD COLUMN suggested_synonyms)
for databases created before this feature existed.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_sff.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

import store


class SemanticFieldFeedbackSynonymsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store.upsert_client("acct_sff", "portal")

    def test_suggested_synonyms_round_trips(self):
        feedback_id = store.save_semantic_field_feedback(
            account_id="acct_sff",
            portal_user_id=None,
            table_fqn="EMDW_DMART.PCH_ORD_RCT_FCT",
            schema_name="EMDW_DMART",
            table_name="PCH_ORD_RCT_FCT",
            column_name="PCH_ORD_AUM_QTY",
            current_meaning="Purchase order quantity",
            current_use_case="",
            suggested_meaning="Purchase order quantity",
            suggested_use_case="Used when a question refers to purchase quantity",
            suggested_synonyms="purchase quantity, number of items purchased",
            user_comment="Needed for phrasing matching",
        )
        rows = store.list_semantic_field_feedback("acct_sff")
        row = next(r for r in rows if r["id"] == feedback_id)
        self.assertEqual(row["suggested_synonyms"], "purchase quantity, number of items purchased")

    def test_default_empty_when_not_supplied(self):
        feedback_id = store.save_semantic_field_feedback(
            account_id="acct_sff",
            portal_user_id=None,
            table_fqn="EMDW_DMART.PCH_ORD_RCT_FCT",
            schema_name="EMDW_DMART",
            table_name="PCH_ORD_RCT_FCT",
            column_name="PCH_ORD_LIN_AMT",
            suggested_meaning="Purchase order line amount",
        )
        rows = store.list_semantic_field_feedback("acct_sff")
        row = next(r for r in rows if r["id"] == feedback_id)
        self.assertEqual(row["suggested_synonyms"], "")

    def test_migration_adds_column_to_pre_existing_table(self):
        # A DB created before v33 has semantic_field_feedback without the
        # column; init_db's migration list must ALTER TABLE it in, not crash.
        from store.db import get_table_columns
        with store.get_db() as conn:
            cols = get_table_columns(conn, "semantic_field_feedback")
        self.assertIn("suggested_synonyms", cols)


if __name__ == "__main__":
    unittest.main()
