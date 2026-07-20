"""Sprint 2 detector #5: compliance/ACL visibility on metrics.

Not in the original ten-item plan matrix — added because compliance
classification was previously invisible to the compiler entirely: a
masking/classification change never moved contract_version, so a published
metric's actual masking behavior could silently drift out of sync with the
classification an admin sees on the Compliance page.

classifications is wired in as a proper compiled source
(core/semantic_contract.py::_compile_contract_internal), keyed
"TABLE_FQN.COLUMN" (the exact shape core.semantic_ids.field_id() produces
minus its "field:" prefix) - this test file also covers that wiring, not
just the pure detector function.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.semantic_conflicts import detect_compliance_gaps  # noqa: E402
from core.semantic_contract import compile_contract  # noqa: E402

_CLASSIFICATIONS = {
    "ERP.FACT_SALES.CUST_NAME": {
        "sensitivity": "RESTRICTED", "tags": ["PII"], "reviewed": 0, "mask_strategy": "redact",
    },
    "ERP.FACT_SALES.CUST_EMAIL": {
        "sensitivity": "RESTRICTED", "tags": ["PII"], "reviewed": 1,
        "mask_strategy": "safe_alias_identifier",
    },
    "ERP.FACT_SALES.NET_AMT": {
        "sensitivity": "INTERNAL", "tags": [], "reviewed": 1, "mask_strategy": "redact",
    },
}


def _compile(*, metrics=None, classifications=None, account_id="acct-compliance-unit"):
    with patch("core.semantic_model.load_semantic_model", return_value={}), \
         patch("store.list_metrics", return_value=metrics or []), \
         patch("store.list_metric_date_contexts", return_value=[]), \
         patch("store.get_full_graph", return_value={}), \
         patch("store.list_terms", return_value=[]), \
         patch("store.get_classification_map", return_value=classifications or {}), \
         patch("core.field_overrides.load_field_overrides", return_value={}):
        return compile_contract(account_id, "C:/tmp/compliance-unit-kb")


class ClassificationSourceWiringTests(unittest.TestCase):
    def test_classifications_land_on_the_compiled_contract(self):
        contract = _compile(classifications=_CLASSIFICATIONS)
        self.assertEqual(contract["classifications"], _CLASSIFICATIONS)

    def test_missing_classification_source_degrades_not_crashes(self):
        with patch("core.semantic_model.load_semantic_model", return_value={}), \
             patch("store.list_metrics", return_value=[]), \
             patch("store.list_metric_date_contexts", return_value=[]), \
             patch("store.get_full_graph", return_value={}), \
             patch("store.list_terms", return_value=[]), \
             patch("store.get_classification_map", side_effect=RuntimeError("boom")), \
             patch("core.field_overrides.load_field_overrides", return_value={}):
            contract = compile_contract("acct-compliance-broken", "C:/tmp/compliance-broken-kb")
        self.assertEqual(contract["classifications"], {})


class UnreviewedRestrictedColumnTests(unittest.TestCase):
    def test_unreviewed_restricted_column_is_a_warning(self):
        contract = _compile(
            metrics=[{"id": 1, "name": "Customer List", "base_table": "ERP.FACT_SALES",
                      "required_columns": "CUST_NAME, NET_AMT"}],
            classifications=_CLASSIFICATIONS,
        )
        conflicts = detect_compliance_gaps(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "unreviewed_restricted_column")
        self.assertEqual(conflicts[0]["severity"], "WARNING")
        self.assertTrue(conflicts[0]["suggestions"])

    def test_reviewed_restricted_column_is_info_only(self):
        contract = _compile(
            metrics=[{"id": 2, "name": "Email List", "base_table": "ERP.FACT_SALES",
                      "required_columns": "CUST_EMAIL"}],
            classifications=_CLASSIFICATIONS,
        )
        conflicts = detect_compliance_gaps(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "restricted_column_in_metric")
        self.assertEqual(conflicts[0]["severity"], "INFO")
        self.assertEqual(conflicts[0]["evidence"]["mask_strategy"], "safe_alias_identifier")

    def test_non_sensitive_column_not_flagged(self):
        contract = _compile(
            metrics=[{"id": 3, "name": "Revenue", "base_table": "ERP.FACT_SALES",
                      "required_columns": "NET_AMT"}],
            classifications=_CLASSIFICATIONS,
        )
        self.assertEqual(detect_compliance_gaps(contract), [])

    def test_bare_table_name_still_resolves(self):
        # base_table is free text (core/metric_scope.py) - classification
        # keys are always fully qualified, so the detector must bridge the two.
        contract = _compile(
            metrics=[{"id": 5, "name": "X", "base_table": "FACT_SALES",
                      "required_columns": "CUST_NAME"}],
            classifications=_CLASSIFICATIONS,
        )
        conflicts = detect_compliance_gaps(contract)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["code"], "unreviewed_restricted_column")

    def test_no_classification_data_at_all_no_conflicts_no_crash(self):
        contract = _compile(
            metrics=[{"id": 4, "name": "X", "base_table": "ERP.FACT_SALES",
                      "required_columns": "CUST_NAME"}],
        )
        self.assertEqual(detect_compliance_gaps(contract), [])

    def test_metric_without_required_columns_skipped(self):
        contract = _compile(
            metrics=[{"id": 6, "name": "X", "base_table": "ERP.FACT_SALES"}],
            classifications=_CLASSIFICATIONS,
        )
        self.assertEqual(detect_compliance_gaps(contract), [])

    def test_conflict_key_is_specific_to_metric_and_column(self):
        contract = _compile(
            metrics=[
                {"id": 1, "name": "A", "base_table": "ERP.FACT_SALES", "required_columns": "CUST_NAME"},
                {"id": 2, "name": "B", "base_table": "ERP.FACT_SALES", "required_columns": "CUST_NAME"},
            ],
            classifications=_CLASSIFICATIONS,
        )
        conflicts = detect_compliance_gaps(contract)
        self.assertEqual(len(conflicts), 2)
        keys = {c["conflict_key"] for c in conflicts}
        self.assertEqual(len(keys), 2)  # distinct per metric, not deduped away


if __name__ == "__main__":
    unittest.main()
