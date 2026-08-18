"""
tests/test_date_role_join_typing.py

Role-playing date entities exist to carry a JOIN to a date dimension. Both the
entity-graph builder and the KB join-map builder selected the columns for that
join by NAME alone, never consulting the column's storage type.

On the EMCO-shaped Infor M3 mart that let the infrastructure audit column
AZ_LST_UPD_TS (datetime2) match the modified-date pattern, so the graph offered

    CUS_ORD_IVC_FCT.AZ_LST_UPD_TS = DT_DMS.DT_DMS_KEY      -- datetime2 = int

at 88% confidence, and minted a second "Last Modified Date" dimension beside the
one LST_MOD_DT_DMS_KEY already owned. Bulk-accepting everything above 85% — the
documented client workflow — therefore installed four type-invalid edges and a
duplicate date entity into the governed graph.

The Date Roles resolver already refused the same column ("no native date type,
declared date-dimension FK, or unambiguous date-key mapping"). Only the graph
and join-map paths disagreed, so the fix is to make them ask the same question.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_dateroles_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from unittest.mock import patch  # noqa: E402

import core.graph_autopopulate as autopop  # noqa: E402
from core.date_roles import detect_date_role, joins_date_dimension  # noqa: E402
from core.schema import build_entity_graph_from_schema  # noqa: E402
from core.vocab_packs import (  # noqa: E402
    _clone_builtin,
    _merge_pack,
    activate_vocab,
    deactivate_vocab,
    get_active_vocab,
    load_pack,
)


def _m3_vocab():
    vocab = _clone_builtin()
    _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
    return vocab

FACT = "CHATBOT_DB.EMDW_DMART.CUS_ORD_IVC_FCT"
DATE_DIM = "CHATBOT_DB.EMDW_DMART.DT_DMS"

SCHEMA = {
    FACT: {
        "columns": [
            {"name": "CUS_ORD_IVC_FCT_KEY", "type": "bigint"},
            {"name": "CUS_IVC_DT_DMS_KEY", "type": "int"},
            {"name": "CUS_ORD_DT_DMS_KEY", "type": "int"},
            {"name": "CNL_ORD_DT_DMS_KEY", "type": "int"},
            {"name": "LST_MOD_DT_DMS_KEY", "type": "int"},
            # Infrastructure audit column: a real datetime2, not a key.
            {"name": "AZ_LST_UPD_TS", "type": "datetime2"},
            {"name": "SOP_CUS_IVC_LIN_AMT", "type": "decimal"},
        ],
    },
    DATE_DIM: {
        "columns": [
            {"name": "DT_DMS_KEY", "type": "int"},
            {"name": "DMS_DT", "type": "date"},
            {"name": "CLD_YR_NO", "type": "int"},
            {"name": "MTH_NM", "type": "nvarchar"},
        ],
    },
}


def _graph(vocab=None):
    """Build the graph the way production does — inside the account's pack."""
    token = activate_vocab(vocab) if vocab is not None else None
    try:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "_schema.json").write_text(
                json.dumps(SCHEMA), encoding="utf-8")
            return build_entity_graph_from_schema(d)
    finally:
        if token is not None:
            deactivate_vocab(token)


class TestANativeDateNeverJoinsADateDimension(unittest.TestCase):

    def test_the_predicate_helper_separates_keys_from_values(self):
        self.assertTrue(joins_date_dimension("CUS_IVC_DT_DMS_KEY", "int"))
        for native in ("date", "datetime", "datetime2", "smalldatetime",
                       "datetimeoffset", "timestamp with time zone"):
            with self.subTest(type=native):
                self.assertFalse(joins_date_dimension("AZ_LST_UPD_TS", native))

    def test_sql_server_rowversion_is_not_mistaken_for_a_date(self):
        # A bare `timestamp` in T-SQL is a binary rowversion. It is not a date
        # value, so it must not be exempted from the join rule on that basis.
        self.assertTrue(joins_date_dimension("ROWVER", "timestamp"))

    def test_the_graph_emits_no_datetime_to_int_edge(self):
        offending = [
            r for r in _graph()["relationships"]
            if r.get("from_column", "").upper() == "AZ_LST_UPD_TS"
        ]
        self.assertEqual(
            offending, [],
            "a datetime2 column was joined to an integer dimension key",
        )

    def test_the_surrogate_date_keys_still_build_their_joins(self):
        by_col = {
            r["from_column"].upper(): r
            for r in _graph()["relationships"]
            if r.get("generated_by") == "date_role"
        }
        self.assertIn("CUS_IVC_DT_DMS_KEY", by_col)
        self.assertEqual(by_col["CUS_IVC_DT_DMS_KEY"]["to_column"], "DT_DMS_KEY")

    def test_no_duplicate_entity_is_minted_for_one_business_role(self):
        names = [e["entity_name"] for e in _graph(_m3_vocab())["entities"]]
        self.assertEqual(
            sorted(n for n in names if "Mod" in n),
            ["Last Modified Date"],
            "the audit column minted a second entity for the same role",
        )
        self.assertEqual(len(names), len(set(names)), "duplicate entity names")


class TestTheGraphIsBuiltWithTheAccountVocabulary(unittest.TestCase):
    """KB generation and the query runtime both activate the client's ERP
    pack before reading identifiers; graph construction did not, so it named
    the same column differently from the two layers that consume it."""

    def test_the_pack_is_active_while_the_graph_is_built(self):
        vocab = _m3_vocab()
        seen = {}
        with patch.object(autopop, "log"),                 patch("core.vocab_packs.vocab_for_account", return_value=vocab):
            with autopop._account_vocab("Emco_test"):
                seen["active"] = get_active_vocab()
        self.assertIs(seen["active"], vocab)

    def test_the_previous_vocabulary_is_restored_afterwards(self):
        before = get_active_vocab()
        with patch("core.vocab_packs.vocab_for_account", return_value=_m3_vocab()):
            with autopop._account_vocab("Emco_test"):
                pass
        self.assertIs(get_active_vocab(), before)

    def test_a_pack_failure_is_logged_loudly_not_swallowed(self):
        with patch("core.vocab_packs.vocab_for_account",
                   side_effect=RuntimeError("pack missing")),                 patch.object(autopop, "log") as log:
            with autopop._account_vocab("Emco_test"):
                pass
        self.assertTrue(log.warning.called, "silent fallback to builtin vocabulary")

    def test_the_m3_pack_changes_what_the_graph_names_the_role(self):
        builtin = [e["entity_name"] for e in _graph()["entities"]]
        packed = [e["entity_name"] for e in _graph(_m3_vocab())["entities"]]
        self.assertIn("Lst Mod Date", builtin)
        self.assertIn("Last Modified Date", packed)


class TestTheInforM3MartVocabulary(unittest.TestCase):
    """The pack modelled M3's own field codes but not the warehouse names
    derived from them, so roles arrived unreadable or merged."""

    def setUp(self):
        vocab = _clone_builtin()
        _merge_pack(vocab, load_pack("infor_m3"), "infor_m3")
        self.vocab = vocab
        self._token = activate_vocab(vocab)

    def tearDown(self):
        deactivate_vocab(self._token)

    def _role(self, column):
        return detect_date_role(column, vocab=self.vocab)

    def test_two_spellings_of_one_concept_resolve_to_one_role(self):
        self.assertEqual(
            self._role("LST_MOD_DT_DMS_KEY").key,
            self._role("AZ_LST_UPD_TS").key,
        )

    def test_distinct_dates_on_one_fact_keep_distinct_roles(self):
        # Both live on CUS_ORD_IVC_FCT. Collapsing them onto order_date made
        # the role -> column mapping ambiguous within a single table.
        self.assertNotEqual(
            self._role("CUS_ORD_DT_DMS_KEY").key,
            self._role("CNL_ORD_DT_DMS_KEY").key,
        )
        self.assertEqual(
            self._role("CNL_ORD_DT_DMS_KEY").key, "cancelled_order_date",
        )

    def test_one_role_across_two_facts_is_still_conformed(self):
        # The opposite case: the same business role on different facts SHOULD
        # share a key. Splitting these would break cross-fact date governance.
        self.assertEqual(
            self._role("CUS_ORD_DT_DMS_KEY").key,
            self._role("PCH_ORD_DT_DMS_KEY").key,
        )

    def test_roles_carry_labels_a_user_could_actually_type(self):
        for column, label in (
            ("ACG_DT_DMS_KEY", "Accounting Date"),
            ("CFM_DLY_DT_DMS_KEY", "Confirmed Delivery Date"),
            ("RQS_DLY_DT_DMS_KEY", "Requested Delivery Date"),
            ("PLN_DLY_DT_DMS_KEY", "Planned Delivery Date"),
            ("ASG_DT_DMS_KEY", "Assignment Date"),
        ):
            with self.subTest(column=column):
                self.assertEqual(self._role(column).label, label)

    def test_the_governed_invoice_default_is_untouched(self):
        self.assertEqual(self._role("CUS_IVC_DT_DMS_KEY").key, "invoice_date")
        self.assertEqual(self._role("PCH_ORD_RCT_DT_DMS_KEY").key, "receipt_date")


if __name__ == "__main__":
    unittest.main(verbosity=2)
