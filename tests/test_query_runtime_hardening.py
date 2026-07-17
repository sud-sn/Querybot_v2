import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.schema as schema


class QueryRuntimeTimeoutTests(unittest.TestCase):
    def test_timeout_configuration_is_bounded(self):
        self.assertEqual(schema._query_timeout_seconds({}), 120)
        self.assertEqual(schema._query_timeout_seconds({"query_timeout_seconds": 0}), 1)
        self.assertEqual(schema._query_timeout_seconds({"query_timeout_seconds": 9999}), 600)
        self.assertEqual(schema._query_timeout_seconds({"query_timeout_seconds": "bad"}), 120)

    def test_azure_sql_sets_driver_timeout_before_execute(self):
        cursor = MagicMock()
        cursor.description = [("A",)]
        cursor.fetchmany.return_value = [(1,)]
        conn = MagicMock()
        conn.cursor.return_value = cursor

        with patch("core.schema._az_connect", return_value=conn):
            rows = schema._run_azure_sql(
                {"query_timeout_seconds": 45}, "SELECT A FROM S.T", max_rows=10
            )

        self.assertEqual(conn.timeout, 45)
        cursor.execute.assert_called_once_with("SELECT A FROM S.T")
        cursor.fetchmany.assert_called_once_with(10)
        self.assertEqual(rows, [{"A": 1}])
        conn.close.assert_called_once()

    def test_oracle_sets_call_timeout_in_milliseconds(self):
        cursor = MagicMock()
        cursor.description = [("A",)]
        cursor.fetchmany.return_value = [(1,)]
        conn = MagicMock()
        conn.cursor.return_value = cursor

        with patch("core.schema._ora_connect", return_value=conn):
            schema._run_oracle({"query_timeout_seconds": 30}, "SELECT A FROM S.T")

        self.assertEqual(conn.call_timeout, 30_000)
        cursor.execute.assert_called_once_with("SELECT A FROM S.T")
        conn.close.assert_called_once()

    def test_snowflake_sets_statement_timeout_before_query(self):
        cursor = MagicMock()
        cursor.fetchmany.return_value = [{"A": 1}]
        conn = MagicMock()
        conn.cursor.return_value = cursor

        connector = types.ModuleType("snowflake.connector")
        connector.DictCursor = object()
        package = types.ModuleType("snowflake")
        package.connector = connector

        with patch.dict(sys.modules, {
            "snowflake": package,
            "snowflake.connector": connector,
        }), patch("core.schema._sf_connect", return_value=conn):
            rows = schema._run_snowflake(
                {"query_timeout_seconds": 25}, "SELECT A FROM S.T", max_rows=5
            )

        calls = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(calls[0], "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 25")
        self.assertEqual(calls[1], "SELECT A FROM S.T")
        cursor.fetchmany.assert_called_once_with(5)
        self.assertEqual(rows, [{"A": 1}])
        conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
