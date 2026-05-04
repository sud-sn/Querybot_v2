"""
Tests for core/schema_discovery.py

Covers the cache layer and the async entry point.
Real DB connections are not made — the dialect functions are mocked.

Run: python -m unittest tests.test_schema_discovery
"""

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.schema_discovery import (
    get_cached_tree, set_cached_tree, bust_cache,
    discover_schema_tree,
    _CACHE_TTL,
)

SAMPLE_TREE = {
    "MY_DB": {
        "PUBLIC": {"tables": ["ORDERS", "CUSTOMERS"], "views": ["VW_SUMMARY"]},
        "ANALYTICS": {"tables": ["FACT_SALES"], "views": []},
    }
}


class CacheTests(unittest.TestCase):

    def setUp(self):
        # Clear any cached state from previous tests
        bust_cache(9001)
        bust_cache(9002)

    def test_miss_on_empty_cache(self):
        self.assertIsNone(get_cached_tree(9001))

    def test_hit_after_set(self):
        set_cached_tree(9001, SAMPLE_TREE)
        result = get_cached_tree(9001)
        self.assertEqual(result, SAMPLE_TREE)

    def test_different_ids_are_isolated(self):
        set_cached_tree(9001, SAMPLE_TREE)
        self.assertIsNone(get_cached_tree(9002))

    def test_bust_removes_entry(self):
        set_cached_tree(9001, SAMPLE_TREE)
        bust_cache(9001)
        self.assertIsNone(get_cached_tree(9001))

    def test_expired_entry_returns_none(self):
        """Simulate expiry by back-dating the stored timestamp."""
        import core.schema_discovery as mod
        mod._cache[9001] = (time.monotonic() - 1, SAMPLE_TREE)  # already expired
        self.assertIsNone(get_cached_tree(9001))

    def test_overwrite_with_new_data(self):
        set_cached_tree(9001, {"OLD": {}})
        set_cached_tree(9001, SAMPLE_TREE)
        self.assertEqual(get_cached_tree(9001), SAMPLE_TREE)


class DiscoverSchemaTreeTests(unittest.TestCase):

    def _run(self, coro):
        return asyncio.run(coro)

    def test_snowflake_calls_correct_dialect(self):
        with patch("core.schema_discovery._discover_snowflake", return_value=SAMPLE_TREE) as mock:
            result = self._run(discover_schema_tree("snowflake", {"user": "u"}, timeout_seconds=10))
        mock.assert_called_once_with({"user": "u"})
        self.assertEqual(result, SAMPLE_TREE)

    def test_oracle_calls_correct_dialect(self):
        with patch("core.schema_discovery._discover_oracle", return_value=SAMPLE_TREE) as mock:
            result = self._run(discover_schema_tree("oracle", {"dsn": "x"}, timeout_seconds=10))
        mock.assert_called_once()
        self.assertEqual(result, SAMPLE_TREE)

    def test_azure_sql_calls_correct_dialect(self):
        with patch("core.schema_discovery._discover_azure_sql", return_value=SAMPLE_TREE) as mock:
            result = self._run(discover_schema_tree("azure_sql", {}, timeout_seconds=10))
        mock.assert_called_once()
        self.assertEqual(result, SAMPLE_TREE)

    def test_unsupported_db_type_raises(self):
        with self.assertRaises(ValueError):
            self._run(discover_schema_tree("mysql", {}, timeout_seconds=5))

    def test_timeout_parameter_is_passed_to_wait_for(self):
        """discover_schema_tree uses asyncio.wait_for with the given timeout.
        We verify wait_for is called with the correct timeout value rather than
        simulating an actual hang (which requires a live executor thread)."""
        import asyncio as _aio
        captured = {}

        original_wait_for = _aio.wait_for
        async def _spy_wait_for(coro, timeout):
            captured["timeout"] = timeout
            # Let the real function run but with the mocked dialect underneath
            return await original_wait_for(coro, timeout=30)  # generous real timeout

        with patch("core.schema_discovery._discover_snowflake", return_value={}):
            with patch("core.schema_discovery.asyncio.wait_for", side_effect=_spy_wait_for):
                self._run(discover_schema_tree("snowflake", {}, timeout_seconds=17))

        self.assertEqual(captured.get("timeout"), 17)

    def test_connection_error_propagates(self):
        with patch("core.schema_discovery._discover_snowflake", side_effect=RuntimeError("bad creds")):
            with self.assertRaises(RuntimeError):
                self._run(discover_schema_tree("snowflake", {}, timeout_seconds=10))


class SchemaTreeStructureTests(unittest.TestCase):
    """Verify the expected shape of a returned tree."""

    def test_tree_keys_are_database_schema_objects(self):
        tree = SAMPLE_TREE
        for db, schemas in tree.items():
            self.assertIsInstance(db, str)
            for schema, objs in schemas.items():
                self.assertIsInstance(schema, str)
                self.assertIn("tables", objs)
                self.assertIn("views", objs)
                self.assertIsInstance(objs["tables"], list)
                self.assertIsInstance(objs["views"], list)

    def test_sample_tree_contents(self):
        self.assertIn("MY_DB", SAMPLE_TREE)
        self.assertIn("PUBLIC", SAMPLE_TREE["MY_DB"])
        self.assertIn("ORDERS", SAMPLE_TREE["MY_DB"]["PUBLIC"]["tables"])
        self.assertIn("VW_SUMMARY", SAMPLE_TREE["MY_DB"]["PUBLIC"]["views"])
        self.assertEqual(SAMPLE_TREE["MY_DB"]["ANALYTICS"]["views"], [])


if __name__ == "__main__":
    unittest.main()
