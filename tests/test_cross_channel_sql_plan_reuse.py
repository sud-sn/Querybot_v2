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
import store.trace_store as trace_store


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
                status TEXT,
                query_row_count INTEGER DEFAULT 0,
                request_source TEXT DEFAULT ''
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
                sql_validation_status, status, query_row_count, request_source)
               VALUES (10, 'client-a', 'portal-q1', '', ?,
                       'azure_sql', 'contract-v4', 'fail', 'success', 12, 'portal')""",
            ('["PHARMA_LAB.D_DATE", "PHARMA_LAB.F_INVENTORY_SNAPSHOT"]',),
        )
        # Patch the exact module object imported by this test. Several legacy
        # suites delete and re-import ``store.*`` during collection; a string
        # patch would then target the replacement module while this test's
        # imported function still reads globals from the original module.
        self.db_patch = patch.object(
            trace_store,
            "get_db",
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
        return trace_store.find_reusable_validated_sql_plan(**args)

    def test_reuses_successful_portal_plan_for_same_governance_scope(self):
        # Legacy Portal traces may not record selected_schema; the sole schema
        # in the exact ACL table snapshot is inferred without broadening scope.
        plan = self._find(question="which DRUGS have inventory expiring within 210 days?!")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["query_log_id"], 1)
        self.assertIn("ON_HAND_QUANTITY", plan["sql_generated"])
        self.assertEqual(plan["query_row_count"], 12)

    def test_skips_newer_zero_row_plan_and_reuses_positive_plan(self):
        self.conn.execute(
            """INSERT INTO query_log
               (id, account_id, question, sql_generated, success, question_id, created_at)
               VALUES (2, 'client-a', ?, ?, 1, 'teams-q2', '2026-07-21 11:38:00')""",
            (
                "Which drugs have inventory expiring within 210 days?",
                "SELECT fin.SNAPSHOT_DATE_ID FROM PHARMA_LAB.F_INVENTORY_SNAPSHOT fin "
                "WHERE fin.SNAPSHOT_DATE_ID < 0",
            ),
        )
        self.conn.execute(
            """INSERT INTO answer_trace
               (id, account_id, question_id, selected_schema,
                allowed_tables_snapshot, db_type, contract_version,
                sql_validation_status, status, query_row_count, request_source)
               VALUES (20, 'client-a', 'teams-q2', 'PHARMA_LAB', ?,
                       'azure_sql', 'contract-v4', 'pass', 'success', 0, 'teams')""",
            ('["PHARMA_LAB.F_INVENTORY_SNAPSHOT"]',),
        )

        plan = self._find()
        self.assertIsNotNone(plan)
        self.assertEqual(plan["query_log_id"], 1)
        self.assertEqual(plan["request_source"], "portal")

    def test_reuses_plan_when_current_acl_authorizes_all_referenced_tables(self):
        # Unrelated ACL differences between Portal and Teams must not split the
        # plan pool when the SQL only references a table both users may access.
        plan = self._find(allowed_tables=["PHARMA_LAB.F_INVENTORY_SNAPSHOT"])
        self.assertIsNotNone(plan)
        self.assertEqual(plan["query_log_id"], 1)

    def test_catalog_qualified_sql_matches_schema_table_acl(self):
        self.conn.execute(
            """UPDATE query_log
                  SET sql_generated=?
                WHERE id=1""",
            (
                "SELECT fin.SNAPSHOT_DATE_ID FROM "
                "CHATBOT_DB.PHARMA_LAB.F_INVENTORY_SNAPSHOT fin",
            ),
        )
        self.assertIsNotNone(self._find(
            allowed_tables=["PHARMA_LAB.F_INVENTORY_SNAPSHOT"],
        ))

    def test_does_not_reuse_across_tenants(self):
        self.assertIsNone(self._find(account_id="client-b"))

    def test_does_not_reuse_when_schema_acl_dialect_or_contract_changes(self):
        self.assertIsNone(self._find(selected_schema="OTHER"))
        self.assertIsNone(self._find(allowed_tables=["PHARMA_LAB.D_DATE"]))
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


class ReusedPlanStalenessGuardTests(unittest.TestCase):
    """A cached plan from before a resolver/validator fix (e.g. the fan-out
    fix that tightened detect_entities()) must not keep being reused forever
    just because contract_version hasn't changed — contract_version tracks
    semantic DATA changes, not CODE changes."""

    def setUp(self):
        self.graph_ctx = {
            "enabled": True,
            "detected": ["F_RX_FILL", "D_PHARMACY"],
            "entities": [
                {"entity_name": "F_RX_FILL", "table_name": "F_RX_FILL"},
                {"entity_name": "D_PHARMACY", "table_name": "D_PHARMACY"},
                {"entity_name": "F_INVENTORY_SNAPSHOT", "table_name": "F_INVENTORY_SNAPSHOT"},
                {"entity_name": "F_PURCHASE_RECEIPT", "table_name": "F_PURCHASE_RECEIPT"},
                {"entity_name": "F_RX_ORDER", "table_name": "F_RX_ORDER"},
            ],
        }

    def _stale(self, sql, graph_ctx=None):
        from core.pipeline_helpers import reused_plan_is_stale_for_graph
        return reused_plan_is_stale_for_graph(
            sql, graph_ctx if graph_ctx is not None else self.graph_ctx, "azure_sql",
        )

    def test_cached_fanout_plan_is_stale_when_fresh_detection_narrower(self):
        # Exactly the live-production case: a plan cached before the fan-out
        # fix joins 3 extra fact tables the current resolver no longer detects.
        sql = (
            "SELECT dph.PHARMACY_NAME, SUM(frx.NET_REVENUE_AMT) AS TOTAL_NET_REVENUE "
            "FROM PHARMA_LAB.F_RX_FILL frx "
            "INNER JOIN PHARMA_LAB.D_PHARMACY dph ON frx.PHARMACY_ID = dph.PHARMACY_ID "
            "INNER JOIN PHARMA_LAB.F_INVENTORY_SNAPSHOT fin ON fin.PHARMACY_ID = dph.PHARMACY_ID "
            "INNER JOIN PHARMA_LAB.F_PURCHASE_RECEIPT fpu ON fpu.PHARMACY_ID = dph.PHARMACY_ID "
            "INNER JOIN PHARMA_LAB.F_RX_ORDER frx2 ON frx2.PHARMACY_ID = dph.PHARMACY_ID"
        )
        self.assertTrue(self._stale(sql))

    def test_matching_plan_is_not_stale(self):
        sql = (
            "SELECT dph.PHARMACY_NAME, SUM(frx.NET_REVENUE_AMT) AS TOTAL_NET_REVENUE "
            "FROM PHARMA_LAB.F_RX_FILL frx "
            "INNER JOIN PHARMA_LAB.D_PHARMACY dph ON frx.PHARMACY_ID = dph.PHARMACY_ID"
        )
        self.assertFalse(self._stale(sql))

    def test_cte_names_are_never_treated_as_extra_tables(self):
        # A genuinely fresh, fan-out-guard-produced CTE plan must not be
        # rejected just because its CTE aliases aren't in the entity list.
        sql = (
            "WITH revenue_by_pharmacy AS ("
            "SELECT dph.PHARMACY_NAME, SUM(frx.NET_REVENUE_AMT) AS TOTAL_REVENUE "
            "FROM PHARMA_LAB.F_RX_FILL frx "
            "INNER JOIN PHARMA_LAB.D_PHARMACY dph ON frx.PHARMACY_ID = dph.PHARMACY_ID "
            "GROUP BY dph.PHARMACY_NAME) "
            "SELECT * FROM revenue_by_pharmacy"
        )
        self.assertFalse(self._stale(sql))

    def test_no_check_when_graph_disabled(self):
        self.assertFalse(self._stale("SELECT 1 FROM ANYTHING", {"enabled": False}))

    def test_review_only_graph_always_invalidates_historical_plan(self):
        self.assertTrue(self._stale(
            "SELECT COUNT(*) FROM PHARMA_LAB.F_RX_ORDER o "
            "JOIN PHARMA_LAB.BR_RX_DIAGNOSIS b ON b.RX_ORDER_ID = o.RX_ORDER_ID",
            {"enabled": False, "review_only": True, "entities": self.graph_ctx["entities"]},
        ))

    def test_no_check_when_nothing_detected(self):
        self.assertFalse(self._stale("SELECT 1 FROM ANYTHING", {"enabled": True, "detected": []}))

    def test_malformed_sql_does_not_raise(self):
        self.assertFalse(self._stale("not even sql {{{"))


class ReusedPlanSemanticContractGuardTests(unittest.TestCase):
    TABLE = "QBOT_LIVE_TEST.F_SALES_INVOICE"
    DATE_TABLE = "QBOT_LIVE_TEST.D_DATE"
    COLUMNS = {
        TABLE: {
            "INVOICE_DATE_SK": "int",
            "NET_REVENUE_AMOUNT": "decimal",
        },
        DATE_TABLE: {
            "DATE_SK": "int",
            "FULL_DATE": "date",
        },
    }
    PLAN = {
        "enabled": True,
        "fields": [],
        "joins": [],
        "required_tables": [],
        "temporal_policies": [{
            "kind": "last_n",
            "amount": 5,
            "unit": "day",
            "anchor_policy": "latest_available",
            "fact_table": TABLE,
            "fact_column": "INVOICE_DATE_SK",
            "date_table": DATE_TABLE,
            "date_column": "FULL_DATE",
            "dimension_table": DATE_TABLE,
            "dimension_key": "DATE_SK",
            "date_key_type": "surrogate_fk",
            "business_role": "invoice_date",
        }],
    }

    def _code(self, sql):
        from core.pipeline_helpers import reused_plan_semantic_staleness_code
        return reused_plan_semantic_staleness_code(
            sql,
            set(self.COLUMNS),
            "azure_sql",
            set(self.COLUMNS),
            self.COLUMNS,
            {"semantic_plan": self.PLAN},
            {
                "kind": "last_n",
                "amount": 5,
                "unit": "day",
                "anchor_policy": "latest_available",
            },
        )

    def test_unbounded_daily_plan_is_stale_for_current_five_day_contract(self):
        sql = (
            "SELECT d.FULL_DATE, SUM(f.NET_REVENUE_AMOUNT) AS DAILY_REVENUE "
            "FROM QBOT_LIVE_TEST.F_SALES_INVOICE f "
            "JOIN QBOT_LIVE_TEST.D_DATE d ON f.INVOICE_DATE_SK = d.DATE_SK "
            "GROUP BY d.FULL_DATE"
        )
        self.assertEqual(self._code(sql), "temporal_anchor_missing")

    def test_current_five_day_plan_remains_reusable(self):
        sql = (
            "WITH anchor AS ("
            "SELECT MAX(d.FULL_DATE) AS max_business_date "
            "FROM QBOT_LIVE_TEST.F_SALES_INVOICE f "
            "JOIN QBOT_LIVE_TEST.D_DATE d ON f.INVOICE_DATE_SK = d.DATE_SK) "
            "SELECT d.FULL_DATE, SUM(f.NET_REVENUE_AMOUNT) AS DAILY_REVENUE "
            "FROM QBOT_LIVE_TEST.F_SALES_INVOICE f "
            "JOIN QBOT_LIVE_TEST.D_DATE d ON f.INVOICE_DATE_SK = d.DATE_SK "
            "CROSS JOIN anchor a "
            "WHERE d.FULL_DATE > DATEADD(day, -5, a.max_business_date) "
            "AND d.FULL_DATE <= a.max_business_date "
            "GROUP BY d.FULL_DATE"
        )
        self.assertEqual(self._code(sql), "")

    def test_temporal_reuse_fails_closed_when_plan_merge_lost_policy(self):
        from core.pipeline_helpers import reused_plan_semantic_staleness_code

        code = reused_plan_semantic_staleness_code(
            "SELECT d.FULL_DATE, SUM(f.NET_REVENUE_AMOUNT) "
            "FROM QBOT_LIVE_TEST.F_SALES_INVOICE f "
            "JOIN QBOT_LIVE_TEST.D_DATE d ON f.INVOICE_DATE_SK = d.DATE_SK "
            "GROUP BY d.FULL_DATE",
            set(self.COLUMNS),
            "azure_sql",
            set(self.COLUMNS),
            self.COLUMNS,
            {"semantic_plan": {"enabled": True, "temporal_policies": []}},
            {"kind": "last_n", "amount": 5, "unit": "day"},
        )

        self.assertEqual(code, "current_temporal_policy_missing")

    def test_reuse_gate_is_wired_before_plan_selection(self):
        from pathlib import Path

        source = Path("core/query_pipeline.py").read_text(encoding="utf-8")
        guard = source.index("_reuse_staleness_code = reused_plan_semantic_staleness_code")
        selection = source.index("elif _reused_plan:", guard)
        self.assertLess(guard, selection)

    def test_pipeline_passes_effective_window_to_cache_guard(self):
        from pathlib import Path

        source = Path("core/query_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"temporal_window": _effective_temporal_window', source)
        self.assertIn(
            "_generation_semantic_context,\n            _effective_temporal_window,",
            source,
        )


class ReusedPlanPinnedAnchorLiteralTests(unittest.TestCase):
    """A cached plan may pin the business date as a literal instead of deriving
    it with an in-query MAX(). That is legitimate — core/date_anchor.py resolves
    the anchor once from the fact's own rows, and the validator accepts the
    resolved literal in place of the MAX(). It also means the plan is only
    correct for as long as that literal IS the anchor.

    This is the reported field scenario: a dev warehouse had its range moved
    from 2025-04-17 back to 2024-08-11, and the same answer kept coming back.
    The reuse layer's protection against that is entirely implicit — it falls
    out of the anchor literal no longer matching — so nothing failed if the
    protection was removed. These tests make it explicit.
    """

    TABLE = "QBOT_LIVE_TEST.F_SALES_INVOICE"
    COLUMNS = {TABLE: {"INVOICE_DATE": "date", "NET_REVENUE_AMOUNT": "decimal"}}
    POLICY = {
        "kind": "today",
        "anchor_policy": "latest_available",
        "fact_table": TABLE,
        "fact_column": "INVOICE_DATE",
        "date_column": "INVOICE_DATE",
    }

    def _code(self, sql, anchor_value):
        from core.pipeline_helpers import reused_plan_semantic_staleness_code

        return reused_plan_semantic_staleness_code(
            sql,
            set(self.COLUMNS),
            "azure_sql",
            set(self.COLUMNS),
            self.COLUMNS,
            {
                "semantic_plan": {"enabled": True, "temporal_policies": [self.POLICY]},
                "resolved_date_anchor": {
                    "value": anchor_value,
                    "fact_table": self.TABLE,
                    "fact_column": "INVOICE_DATE",
                },
            },
            {"kind": "today", "amount": 0, "unit": "day",
             "anchor_policy": "latest_available"},
        )

    def _sql(self, day):
        return (
            "SELECT SUM(NET_REVENUE_AMOUNT) AS TOTAL "
            f"FROM {self.TABLE} WHERE INVOICE_DATE = '{day}'"
        )

    def test_a_plan_pinned_to_the_current_anchor_is_reusable(self):
        self.assertEqual(self._code(self._sql("2025-04-17"), "2025-04-17"), "")

    def test_a_plan_pinned_to_a_superseded_anchor_is_refused(self):
        """The warehouse moved and the cached plan did not. Reusing it returns
        the previous period's number as though it were today's."""
        self.assertEqual(
            self._code(self._sql("2025-04-17"), "2024-08-11"),
            "temporal_anchor_missing",
        )

    def test_the_anchor_moving_backwards_is_caught_too(self):
        """A reload can move the range earlier, not only later. Nothing here
        may assume data only ever advances."""
        self.assertEqual(
            self._code(self._sql("2024-08-11"), "2025-04-17"),
            "temporal_anchor_missing",
        )

    def test_an_anchor_probed_for_a_different_fact_does_not_license_the_literal(self):
        """anchor_for_policy() exists so an anchor probed from one fact cannot
        validate a plan over another. Without it, any date string matching some
        other fact's anchor would pass."""
        from core.pipeline_helpers import reused_plan_semantic_staleness_code

        code = reused_plan_semantic_staleness_code(
            self._sql("2025-04-17"),
            set(self.COLUMNS),
            "azure_sql",
            set(self.COLUMNS),
            self.COLUMNS,
            {
                "semantic_plan": {"enabled": True, "temporal_policies": [self.POLICY]},
                "resolved_date_anchor": {
                    "value": "2025-04-17",
                    "fact_table": "QBOT_LIVE_TEST.F_OTHER_FACT",
                    "fact_column": "POSTED_DATE",
                },
            },
            {"kind": "today", "amount": 0, "unit": "day",
             "anchor_policy": "latest_available"},
        )
        self.assertEqual(code, "temporal_anchor_missing")


if __name__ == "__main__":
    unittest.main()
