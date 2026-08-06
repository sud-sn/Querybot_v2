import re
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "azure_sql_join_planner_v2.sql"


class TestAzureSqlJoinPlannerFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = FIXTURE.read_text(encoding="utf-8")

    def test_fixture_exists_and_is_substantial(self):
        self.assertTrue(FIXTURE.is_file())
        self.assertGreater(len(self.sql), 15000)

    def test_fixture_has_no_batch_separator(self):
        self.assertIsNone(re.search(r"(?im)^\s*GO\s*(?:--.*)?$", self.sql))

    def test_fixture_is_rerunnable_and_transactional(self):
        self.assertIn("DROP TABLE IF EXISTS", self.sql)
        self.assertIn("SET XACT_ABORT ON", self.sql)
        self.assertIn("BEGIN TRANSACTION", self.sql)
        self.assertIn("ROLLBACK TRANSACTION", self.sql)

    def test_fixture_covers_production_join_shapes(self):
        required = {
            "DIM_DATE",
            "DIM_REGION",
            "DIM_WAREHOUSE",
            "DIM_CUSTOMER_SCD2",
            "DIM_PRODUCT",
            "DIM_CATEGORY",
            "BRIDGE_PRODUCT_CATEGORY",
            "FACT_SALES",
            "FACT_INVENTORY_DAILY",
            "FACT_INVENTORY_MONTHLY",
            "FACT_RETURNS",
            "FACT_BAD_RELATIONSHIP",
            "DIM_DUPLICATE_CODE",
            "TEST_EXPECTED_CASES",
        }
        for table in required:
            self.assertIn(f"[QBOT_JOIN_TEST].[{table}]", self.sql)

    def test_fixture_includes_role_playing_dates_and_scd2(self):
        self.assertIn("INVOICE_DATE_KEY", self.sql)
        self.assertIn("ORDER_DATE_KEY", self.sql)
        self.assertIn("RETURN_DATE_KEY", self.sql)
        self.assertIn("ORIGINAL_INVOICE_DATE_KEY", self.sql)
        self.assertIn("EFFECTIVE_FROM_DATE", self.sql)
        self.assertIn("CURRENT_FLAG", self.sql)

    def test_bridge_has_allocation_and_composite_key(self):
        self.assertIn("ALLOCATION_PCT", self.sql)
        self.assertRegex(
            self.sql,
            r"PRIMARY KEY \(\[PRODUCT_KEY\], \[CATEGORY_KEY\]\)",
        )

    def test_acceptance_catalog_has_ten_cases(self):
        case_ids = {
            int(value) for value in re.findall(
                r"(?m)^\s*\((\d+),\s*'(?:single_fact_star|snowflake|role_playing_date|scd2|bridge_allocation|daily_snapshot|monthly_snapshot|multi_fact_isolation|return_date_role|relationship_profile)'",
                self.sql,
            )
        }
        self.assertEqual(case_ids, set(range(1, 11)))

    def test_multi_fact_acceptance_query_aggregates_facts_separately(self):
        self.assertIn("WITH [sales_agg] AS", self.sql)
        self.assertIn("[inventory_agg] AS", self.sql)
        self.assertNotRegex(
            self.sql,
            r"(?is)FACT_SALES\]\s+\w+\s+(?:INNER\s+|LEFT\s+)?JOIN\s+\[QBOT_JOIN_TEST\]\.\[FACT_INVENTORY_DAILY",
        )


if __name__ == "__main__":
    unittest.main()
