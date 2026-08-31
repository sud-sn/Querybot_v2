"""
tests/test_value_index_write_gate.py

What a regulated tenant may STORE, as distinct from what it may SEE.

core/value_resolver.py has always filtered what value grounding shows a
regulated tenant. Nothing filtered what got written: core/value_index.py
imported nothing from core.compliance, so an admin-reviewed PHI classification
did not stop the harvest, and up to per_column_cap real distinct values per
column were copied into clients/{account_id}/value_index.sqlite.

Every test here reads the SQLite file directly with the stdlib driver rather
than going through a loader. That is the point: the defect was that data sat on
disk regardless of what the read path filtered, so a test that asks the read
path what it can see would have passed against the defect.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tmpdir = tempfile.mkdtemp(prefix="querybot_vigate_")
os.environ.setdefault("DB_PATH", str(Path(_tmpdir) / "test_querybot.db"))
os.environ.setdefault("QUERYBOT_KEY_FILE", str(Path(_tmpdir) / "test_key"))

from core.value_index import build_value_index, select_filterable_columns  # noqa: E402

DIM = "CHATBOT_DB.CLIN.DIM_PATIENT"
SCHEMA = {
    DIM: {
        "database": "CHATBOT_DB", "schema": "CLIN", "table": "DIM_PATIENT",
        "columns": [
            {"name": "PAT_DMS_KEY", "type": "int"},
            # Abbreviated, so the spelled-out PII pattern does not catch it.
            {"name": "PAT_NM", "type": "nvarchar"},
            {"name": "DIAGNOSIS_CD", "type": "nvarchar"},
            {"name": "CLINIC_NM", "type": "nvarchar"},
        ],
    },
}
PATIENT_ROWS = [{"PAT_NM": "Marguerite Oyelaran"}, {"PAT_NM": "Tomas Iwuchukwu"}]
CLINIC_ROWS = [{"CLINIC_NM": "Riverside"}, {"CLINIC_NM": "Northgate"}]
DIAGNOSIS_ROWS = [{"DIAGNOSIS_CD": "E11.9"}, {"DIAGNOSIS_CD": "I10"}]

_BY_COLUMN = {"PAT_NM": PATIENT_ROWS, "CLINIC_NM": CLINIC_ROWS, "DIAGNOSIS_CD": DIAGNOSIS_ROWS}


def _fake_query(_creds, _db_type, sql, max_rows=0):
    for column, rows in _BY_COLUMN.items():
        if column in sql:
            return rows
    return []


def _values_on_disk(base_dir: str, account_id: str) -> set[str]:
    """Read the index file itself. Not a loader, not the resolver."""
    path = Path(base_dir) / account_id / "value_index.sqlite"
    if not path.exists():
        return set()
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT value FROM column_value")}
    finally:
        conn.close()


class _Harness(unittest.TestCase):
    account = "acct_regulated"

    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="vi_base_")

    def _build(self, *, regulated=True, pack="healthcare_pharmacy_v1",
               classifications=None, industry="healthcare_pharmacy"):
        profile = {"policy_pack_key": pack, "industry": industry}
        with patch("core.compliance.policy_engine.is_regulated", return_value=regulated), \
             patch("store.get_compliance_profile", return_value=profile), \
             patch("store.get_classification_map", return_value=classifications or {}), \
             patch("core.schema.load_schema_json", return_value=SCHEMA):
            return build_value_index(
                self.account, {"x": 1}, "azure_sql", "unused-schema-dir",
                run_query_fn=_fake_query, base_dir=self.base,
            )

    def _on_disk(self):
        return _values_on_disk(self.base, self.account)


class TestTheGateRefusesWhatItShould(_Harness):

    def test_an_unclassified_column_is_not_written_to_disk(self):
        """The fail-closed choice, stated. At query time an unclassified column
        merely fails to clear; here it would be persisted, so the default has
        to be stricter."""
        stats = self._build(classifications={})
        self.assertEqual(self._on_disk(), set())
        self.assertEqual(stats["columns_indexed"], 0)

    def test_an_abbreviated_phi_column_is_not_written_to_disk(self):
        """PAT_NM holds exactly what PATIENT_NAME holds. The spelled-out PII
        pattern catches one and not the other, which is why the gate cannot be
        left to naming alone."""
        self._build(classifications={
            f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": ["PHI"]},
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        on_disk = self._on_disk()
        self.assertNotIn("Marguerite Oyelaran", on_disk)
        self.assertNotIn("Tomas Iwuchukwu", on_disk)

    def test_a_reviewed_but_sensitive_column_is_refused(self):
        self._build(classifications={
            f"{DIM}.DIAGNOSIS_CD".upper(): {"reviewed": True, "tags": ["PHI"]},
        })
        self.assertNotIn("E11.9", self._on_disk())

    def test_an_unreviewed_clearance_is_not_a_clearance(self):
        """Tags that say the column is harmless do not count until a human has
        reviewed them — an unreviewed row is a machine guess."""
        self._build(classifications={
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": False, "tags": []},
        })
        self.assertEqual(self._on_disk(), set())

    def test_a_regulated_tenant_with_no_pack_indexes_nothing(self):
        """No pack means no definition of "sensitive", so nothing can be
        established as safe. Same reading the query path takes."""
        self._build(pack="", classifications={
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        self.assertEqual(self._on_disk(), set())

    def test_unreadable_compliance_state_refuses_rather_than_proceeds(self):
        profile_boom = patch("store.get_compliance_profile", side_effect=RuntimeError("db down"))
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             profile_boom, \
             patch("core.schema.load_schema_json", return_value=SCHEMA):
            build_value_index(
                self.account, {"x": 1}, "azure_sql", "unused",
                run_query_fn=_fake_query, base_dir=self.base,
            )
        self.assertEqual(self._on_disk(), set())


class TestTheGateStillAllowsGroundedValues(_Harness):
    """A gate that refuses everything is not a gate, it is an outage."""

    def test_a_reviewed_non_sensitive_column_is_indexed(self):
        stats = self._build(classifications={
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        on_disk = self._on_disk()
        self.assertIn("Riverside", on_disk)
        self.assertIn("Northgate", on_disk)
        self.assertEqual(stats["columns_indexed"], 1)

    def test_the_cleared_column_is_indexed_while_its_neighbour_is_refused(self):
        """Both columns sit on the same table. The gate is per column."""
        self._build(classifications={
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
            f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": ["PHI"]},
        })
        on_disk = self._on_disk()
        self.assertIn("Riverside", on_disk)
        self.assertNotIn("Marguerite Oyelaran", on_disk)

    def test_an_unregulated_tenant_is_completely_unaffected(self):
        """This change must not narrow anything for the tenants who are not
        regulated — they have no classifications at all."""
        self._build(regulated=False, classifications={})
        on_disk = self._on_disk()
        self.assertIn("Riverside", on_disk)
        self.assertIn("Marguerite Oyelaran", on_disk)

    def test_the_build_reports_what_it_refused(self):
        """A build that indexes nothing is a governed outcome, not a failure,
        and the stats must let an operator tell those apart."""
        stats = self._build(classifications={
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        self.assertTrue(stats["regulated"])
        self.assertEqual(stats["industry"], "healthcare_pharmacy")
        self.assertGreater(stats["columns_refused_unclassified"], 0)


class TestTheRegulatedClassifierRuns(unittest.TestCase):
    """detect_sensitive_columns(columns, industry="") runs its banking and
    healthcare pass only when industry is set. The call site omitted it, so
    that pass never ran during a build."""

    def test_industry_reaches_the_detector(self):
        seen: dict = {}

        def _spy(columns, industry=""):
            seen["industry"] = industry
            return {}

        with patch("core.masking.detect_sensitive_columns", _spy):
            select_filterable_columns(SCHEMA, industry="healthcare_pharmacy")
        self.assertEqual(seen.get("industry"), "healthcare_pharmacy")

    def test_the_default_is_still_no_industry(self):
        seen: dict = {}

        def _spy(columns, industry=""):
            seen["industry"] = industry
            return {}

        with patch("core.masking.detect_sensitive_columns", _spy):
            select_filterable_columns(SCHEMA)
        self.assertEqual(seen.get("industry"), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReclassificationRetractsWhatWasHarvested(_Harness):
    """Building was the only moment the gate ran.

    A column reclassified as PHI after the build kept its values on disk
    indefinitely: the compliance profile save rewrote classifications and never
    touched this file, making it the one artifact where correcting a
    classification had no effect.
    """

    CLEARED = {f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
               f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": []}}

    def _purge_with(self, classifications):
        from core.value_index import purge_uncleared_columns
        profile = {"policy_pack_key": "healthcare_pharmacy_v1",
                   "industry": "healthcare_pharmacy"}
        with patch("core.compliance.policy_engine.is_regulated", return_value=True), \
             patch("store.get_compliance_profile", return_value=profile), \
             patch("store.get_classification_map", return_value=classifications):
            return purge_uncleared_columns(self.account, base_dir=self.base)

    def test_values_harvested_under_the_old_classification_are_removed(self):
        self._build(classifications=self.CLEARED)
        self.assertIn("Marguerite Oyelaran", self._on_disk())

        result = self._purge_with({
            f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": ["PHI"]},
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })

        on_disk = self._on_disk()
        self.assertNotIn("Marguerite Oyelaran", on_disk)
        self.assertEqual(result["purged_columns"], 1)
        self.assertGreater(result["purged_values"], 0)

    def test_the_cleared_neighbour_survives_the_purge(self):
        """A purge that empties the index is indistinguishable from a broken
        one, so the surviving half is the half that proves it worked."""
        self._build(classifications=self.CLEARED)
        self._purge_with({
            f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": ["PHI"]},
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        self.assertIn("Riverside", self._on_disk())

    def test_revoking_review_alone_retracts_the_column(self):
        """Tags unchanged, review withdrawn. An unreviewed row is a machine
        guess, and it must not keep values on disk."""
        self._build(classifications=self.CLEARED)
        self._purge_with({
            f"{DIM}.PAT_NM".upper(): {"reviewed": False, "tags": []},
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        self.assertNotIn("Marguerite Oyelaran", self._on_disk())

    def test_the_deleted_values_are_not_left_in_the_file(self):
        """A DELETE leaves rows readable in SQLite's free pages. A retraction
        that leaves the plaintext recoverable is not a retraction."""
        self._build(classifications=self.CLEARED)
        self._purge_with({
            f"{DIM}.PAT_NM".upper(): {"reviewed": True, "tags": ["PHI"]},
            f"{DIM}.CLINIC_NM".upper(): {"reviewed": True, "tags": []},
        })
        raw = (Path(self.base) / self.account / "value_index.sqlite").read_bytes()
        self.assertNotIn(b"Marguerite Oyelaran", raw)

    def test_purging_an_unregulated_tenant_changes_nothing(self):
        from core.value_index import purge_uncleared_columns
        self._build(regulated=False, classifications={})
        before = self._on_disk()
        with patch("core.compliance.policy_engine.is_regulated", return_value=False):
            result = purge_uncleared_columns(self.account, base_dir=self.base)
        self.assertEqual(result["reason"], "tenant_not_regulated")
        self.assertEqual(self._on_disk(), before)

    def test_purging_with_no_index_is_not_an_error(self):
        from core.value_index import purge_uncleared_columns
        with patch("core.compliance.policy_engine.is_regulated", return_value=True):
            result = purge_uncleared_columns("never_built", base_dir=self.base)
        self.assertEqual(result["reason"], "no_index")


class TestAReviewedClassificationOutranksTheNamingGuess(unittest.TestCase):
    """The naming heuristic is a regex over a column name. A classification is
    a human decision about that exact column. Where both have an opinion the
    human has to win, or the heuristic silently vetoes the admin.

    Concretely: a compounding pharmacy's drug catalog lives in
    DIM_FORMULA.GENERIC_NAME, which the PII patterns read as "drug name" and
    exclude. That is the one column a question like "revenue by Naltrexone" is
    about, so the veto did not merely lose a column -- it kept value grounding
    from working on the exact case it exists for.
    """

    PHARMA = {
        "DB.PHARMA.DIM_FORMULA": {
            "database": "DB", "schema": "PHARMA", "table": "DIM_FORMULA",
            "columns": [
                {"name": "FORMULA_KEY", "type": "int"},
                {"name": "GENERIC_NAME", "type": "nvarchar"},
            ],
        },
    }
    KEY = "DB.PHARMA.DIM_FORMULA.GENERIC_NAME"

    def _columns(self, cleared):
        return [c["column"] for c in select_filterable_columns(
            self.PHARMA, account_id="a", industry="healthcare_pharmacy", cleared=cleared,
        )]

    def test_the_naming_heuristic_alone_still_excludes_it(self):
        """The premise. If this ever stops holding, the override below is
        testing nothing."""
        self.assertEqual([c["column"] for c in select_filterable_columns(self.PHARMA)], [])

    def test_an_admin_clearance_re_admits_the_column(self):
        self.assertEqual(
            self._columns(lambda t, c: f"{t}.{c}".upper() == self.KEY),
            ["GENERIC_NAME"],
        )

    def test_an_admin_who_tags_it_sensitive_still_excludes_it(self):
        """The override runs in one direction only: it re-admits what the gate
        cleared, and can never admit what the gate refused."""
        self.assertEqual(self._columns(lambda t, c: False), [])

    def test_an_unregulated_tenant_is_unchanged(self):
        """No classifications exist to consult, so the heuristic remains the
        only check and behaviour is exactly what it was."""
        self.assertEqual(self._columns(None), [])
