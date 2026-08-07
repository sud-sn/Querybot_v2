import re
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "azure_sql_live_regression"
DDL = FIXTURE_DIR / "01_create_schema.sql"
SEED = FIXTURE_DIR / "02_seed_data.sql"
VALIDATE = FIXTURE_DIR / "03_validate_expected_results.sql"
README = FIXTURE_DIR / "README.md"


class TestAzureSqlLiveRegressionFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ddl = DDL.read_text(encoding="utf-8")
        cls.seed = SEED.read_text(encoding="utf-8")
        cls.validate = VALIDATE.read_text(encoding="utf-8")
        cls.readme = README.read_text(encoding="utf-8")
        cls.all_sql = "\n".join((cls.ddl, cls.seed, cls.validate))

    def test_all_package_files_exist_and_are_substantial(self):
        for path in (DDL, SEED, VALIDATE, README):
            self.assertTrue(path.is_file(), path)
        self.assertGreater(len(self.ddl), 15_000)
        self.assertGreater(len(self.seed), 18_000)
        self.assertGreater(len(self.validate), 9_000)

    def test_scripts_have_no_go_batch_separator(self):
        self.assertIsNone(re.search(r"(?im)^\s*GO\s*(?:--.*)?$", self.all_sql))

    def test_mutating_scripts_are_transactional_and_rerunnable(self):
        for sql in (self.ddl, self.seed):
            self.assertIn("SET XACT_ABORT ON", sql)
            self.assertIn("BEGIN TRANSACTION", sql)
            self.assertIn("ROLLBACK TRANSACTION", sql)
        self.assertIn("DROP TABLE IF EXISTS", self.ddl)
        self.assertIn("DELETE FROM [QBOT_LIVE_TEST]", self.seed)

    def test_required_tables_are_created(self):
        required = {
            "D_DATE",
            "D_REGION",
            "D_WAREHOUSE",
            "D_CUSTOMER_SCD2",
            "D_PRODUCT",
            "D_CATEGORY",
            "B_PRODUCT_CATEGORY",
            "F_SALES_INVOICE",
            "F_INVENTORY_DAILY",
            "F_INVENTORY_MONTHLY",
            "ERP_ITM_BAL_PRD_FCT",
            "M3_MITBAL",
            "F_RETURNS",
            "F_ORDERS",
            "F_SHIPMENTS",
            "D_DUPLICATE_CODE",
            "F_BAD_CODE",
            "TEST_EXPECTED_CASES",
        }
        for table in required:
            self.assertIn(f"[QBOT_LIVE_TEST].[{table}]", self.ddl)

    def test_date_storage_regressions_are_covered(self):
        for token in (
            "INVOICE_DATE_SK",
            "ORDER_DATE_SK",
            "DATE_SK",
            "FULL_DATE",
            "POSTING_YYYYMM",
            "SNAPSHOT_YYYYMMDD",
            "SNAPSHOT_DATE",
            "SNAPSHOT_AT_UTC",
            "PERIOD_YYYYMM",
            "PRD_DMS_KEY",
            "MLPERY",
            "MLLMDT",
            "UPDATED_AT_UTC",
        ):
            self.assertIn(token, self.all_sql)

    def test_calendar_future_date_trap_and_fact_anchor_are_present(self):
        self.assertIn("2026-12-31", self.seed)
        self.assertIn("MAX([FULL_DATE]) AS [max_fact_date]", self.validate)
        self.assertNotIn("MAX([FULL_DATE]) FROM [QBOT_LIVE_TEST].[D_DATE]", self.validate)

    def test_expected_catalog_contains_nineteen_cases(self):
        case_ids = {
            int(value)
            for value in re.findall(
                r"(?m)^\s*\((\d+),\s*'(?:star_join|snowflake_join|surrogate_date_role|relative_last_days|date_role_disambiguation|yyyy_mm|yyyy_mm_dd|native_date|timestamp|daily_vs_monthly_grain|scd2|bridge_allocation|multi_fact_isolation|role_playing_return_date|governed_anti_join|m3_cryptic_yyyy_mm|unsafe_relationship|formatting_follow_up|erp_prd_dms_yyyymm)'",
                self.seed,
            )
        }
        self.assertEqual(case_ids, set(range(1, 20)))

    def test_multi_fact_reference_aggregates_each_fact_separately(self):
        self.assertIn("WITH [march_revenue] AS", self.validate)
        self.assertIn("[latest_inventory] AS", self.validate)
        self.assertNotRegex(
            self.validate,
            r"(?is)F_SALES_INVOICE\]\s+\w+\s+(?:INNER\s+|LEFT\s+)?JOIN\s+\[QBOT_LIVE_TEST\]\.\[F_INVENTORY_DAILY",
        )

    def test_unsafe_relationship_profile_checks_target_duplicates(self):
        self.assertIn("HAVING COUNT(*) > 1", self.validate)
        self.assertIn("[fanout_ratio]", self.validate)


if __name__ == "__main__":
    unittest.main()
