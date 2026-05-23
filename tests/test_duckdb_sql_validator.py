import unittest

from core.duckdb_sql_validator import validate_duckdb_result_sql
from core.result_cache import ResultCache


class DuckDBSQLValidatorTests(unittest.TestCase):
    def assert_ok(self, sql: str):
        verdict = validate_duckdb_result_sql(sql)
        self.assertTrue(verdict.ok, verdict)

    def assert_rejected(self, sql: str, code: str | None = None):
        verdict = validate_duckdb_result_sql(sql)
        self.assertFalse(verdict.ok, verdict)
        if code:
            self.assertEqual(verdict.code, code)

    def test_simple_select_from_result_passes(self):
        self.assert_ok('SELECT * FROM result ORDER BY "Revenue" DESC')

    def test_cte_over_result_passes(self):
        self.assert_ok(
            "WITH ranked AS (SELECT *, ROW_NUMBER() OVER () AS rn FROM result) "
            "SELECT * FROM ranked WHERE rn <= 5"
        )

    def test_other_table_rejected(self):
        self.assert_rejected("SELECT * FROM customers", "invalid_table")

    def test_multi_statement_rejected(self):
        self.assert_rejected("SELECT * FROM result; DROP TABLE result", "multi_statement")

    def test_copy_rejected(self):
        self.assert_rejected("COPY result TO 'x.csv'", "not_select")

    def test_file_reader_rejected(self):
        self.assert_rejected("SELECT * FROM read_csv('secrets.csv')", "forbidden")

    def test_pragma_rejected(self):
        self.assert_rejected("PRAGMA database_list", "not_select")

    def test_result_cache_fails_closed_for_invalid_sql(self):
        cache = ResultCache()
        cache.store("s1", [{"name": "A", "amount": 10}, {"name": "B", "amount": 20}])
        self.assertEqual(cache.query("s1", "SELECT * FROM read_csv('secrets.csv')"), [])


if __name__ == "__main__":
    unittest.main()
