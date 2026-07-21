"""Governed SQL-plan parity across Portal and external channels."""

from __future__ import annotations

import sqlite3
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core.validator import (
    SqlValidationResult,
    repair_unambiguous_unknown_columns,
    validate_sql_detailed,
)
from store.trace_store import find_reusable_validated_sql_plan


@contextmanager
def _db_context(conn: sqlite3.Connection):
    yield conn


class CrossChannelSqlPlanReuseTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            CREATE TABLE query_log (
                id INTEGER PRIMARY KEY,
                account_id TEXT,
                question TEXT,
                sql_generated TEXT,
                success INTEGER,
                question_id TEXT,
                created_at TEXT
            );
            CREATE TABLE answer_trace (
                id INTEGER PRIMARY KEY,
                account_id TEXT,
                question_id TEXT,
                selected_schema TEXT,
                allowed_tables_snapshot TEXT,
                db_type TEXT,
                contract_version TEXT,
                sql_validation_status TEXT,
                status TEXT
            );
            """
        )
        self.conn.execute(
            """INSERT INTO query_log
               (id, account_id, question, sql_generated, success, question_id, created_at)
               VALUES (1, 'client-a', ?, ?, 1, 'portal-q1', '2026-07-21')""",
            (
                "Which drugs have inventory expiring within 210 days?",
                "SELECT fin.SNAPSHOT_DATE_ID, fin.ON_HAND_QUANTITY "
                "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin",
            ),
        )
        # Simulates an older repaired Portal success whose trace retained the
        # validation state of the failed first draft.
        self.conn.execute(
            """INSERT INTO answer_trace
               (id, account_id, question_id, selected_schema,
                allowed_tables_snapshot, db_type, contract_version,
                sql_validation_status, status)
               VALUES (10, 'client-a', 'portal-q1', '', ?,
                       'azure_sql', 'contract-v4', 'fail', 'success')""",
            ('["PHARMA_LAB.D_DATE", "PHARMA_LAB.F_INVENTORY_SNAPSHOT"]',),
        )
        self.db_patch = patch(
            "store.trace_store.get_db",
            side_effect=lambda: _db_context(self.conn),
        )
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.conn.close()

    def _find(self, **overrides):
        args = {
            "account_id": "client-a",
            "question": "Which drugs have inventory expiring within 210 days?",
            "selected_schema": "PHARMA_LAB",
            "allowed_tables": [
                "PHARMA_LAB.F_INVENTORY_SNAPSHOT",
                "PHARMA_LAB.D_DATE",
            ],
            "db_type": "azure_sql",
            "contract_version": "contract-v4",
        }
        args.update(overrides)
        return find_reusable_validated_sql_plan(**args)

    def test_reuses_successful_portal_plan_for_same_governance_scope(self):
        # Legacy Portal traces may not record selected_schema; the sole schema
        # in the exact ACL table snapshot is inferred without broadening scope.
        plan = self._find(question="which DRUGS have inventory expiring within 210 days?!")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["query_log_id"], 1)
        self.assertIn("ON_HAND_QUANTITY", plan["sql_generated"])

    def test_does_not_reuse_across_tenants(self):
        self.assertIsNone(self._find(account_id="client-b"))

    def test_does_not_reuse_when_schema_acl_dialect_or_contract_changes(self):
        self.assertIsNone(self._find(selected_schema="OTHER"))
        self.assertIsNone(self._find(allowed_tables=["PHARMA_LAB.F_INVENTORY_SNAPSHOT"]))
        self.assertIsNone(self._find(db_type="snowflake"))
        self.assertIsNone(self._find(contract_version="contract-v5"))


class UnambiguousColumnRepairTests(unittest.TestCase):
    def test_validator_maps_common_erp_identifier_variants(self):
        table = "PHARMA_LAB.F_INVENTORY_SNAPSHOT"
        sql = (
            "SELECT fin.SNAPSHOT_DT_DMS_KEY, fin.ON_HAND_QTY, "
            "fin.EXPIRATION_DT_DMS_KEY "
            "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin"
        )
        result = validate_sql_detailed(
            sql,
            {table},
            "azure_sql",
            {table},
            {table: {
                "SNAPSHOT_DATE_ID": "int",
                "ON_HAND_QUANTITY": "decimal",
                "EXPIRY_DATE_ID": "int",
            }},
            None,
        )
        suggestions = {error["column"]: error.get("suggestions") for error in result.errors}
        self.assertEqual(suggestions["SNAPSHOT_DT_DMS_KEY"], ["SNAPSHOT_DATE_ID"])
        self.assertEqual(suggestions["ON_HAND_QTY"], ["ON_HAND_QUANTITY"])
        self.assertEqual(suggestions["EXPIRATION_DT_DMS_KEY"], ["EXPIRY_DATE_ID"])

    def test_repairs_all_repeated_legacy_columns_from_validator_candidates(self):
        sql = (
            "SELECT fin.SNAPSHOT_DT_DMS_KEY, SUM(fin.ON_HAND_QTY) AS QTY "
            "FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin "
            "WHERE fin.EXPIRATION_DT_DMS_KEY > fin.SNAPSHOT_DT_DMS_KEY "
            "GROUP BY fin.SNAPSHOT_DT_DMS_KEY"
        )
        validation = SqlValidationResult(
            ok=False,
            reason="Unknown columns",
            code="unknown_column",
            errors=[
                {
                    "code": "unknown_column",
                    "alias": "fin",
                    "column": "SNAPSHOT_DT_DMS_KEY",
                    "suggestions": ["SNAPSHOT_DATE_ID"],
                },
                {
                    "code": "unknown_column",
                    "alias": "fin",
                    "column": "ON_HAND_QTY",
                    "suggestions": ["ON_HAND_QUANTITY"],
                },
                {
                    "code": "unknown_column",
                    "alias": "fin",
                    "column": "EXPIRATION_DT_DMS_KEY",
                    "suggestions": ["EXPIRY_DATE_ID"],
                },
            ],
        )
        repaired = repair_unambiguous_unknown_columns(sql, validation, "azure_sql")
        self.assertTrue(repaired)
        self.assertNotIn("SNAPSHOT_DT_DMS_KEY", repaired)
        self.assertNotIn("ON_HAND_QTY", repaired)
        self.assertNotIn("EXPIRATION_DT_DMS_KEY", repaired)
        self.assertGreaterEqual(repaired.count("SNAPSHOT_DATE_ID"), 3)
        self.assertIn("ON_HAND_QUANTITY", repaired)
        self.assertIn("EXPIRY_DATE_ID", repaired)

    def test_refuses_ambiguous_repair(self):
        validation = SqlValidationResult(
            ok=False,
            reason="Unknown column",
            code="unknown_column",
            errors=[{
                "code": "unknown_column",
                "alias": "fin",
                "column": "QTY",
                "suggestions": ["ON_HAND_QUANTITY", "RECEIVED_QUANTITY"],
            }],
        )
        repaired = repair_unambiguous_unknown_columns(
            "SELECT fin.QTY FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin",
            validation,
            "azure_sql",
        )
        self.assertEqual(repaired, "")


if __name__ == "__main__":
    unittest.main()
