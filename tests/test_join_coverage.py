import unittest
from unittest.mock import patch

from core.join_coverage import check_join_coverage, _ORPHAN_RATE_WARNING_THRESHOLD


class CheckJoinCoverageTests(unittest.TestCase):
    def _edge(self, rel_id=1, from_entity="Orders", to_entity="Customer"):
        return {"id": rel_id, "from_entity": from_entity, "to_entity": to_entity}

    def test_empty_edges_returns_empty_list(self):
        self.assertEqual(check_join_coverage("acct1", []), [])

    def test_above_threshold_produces_caveat_naming_entities_and_percent(self):
        with patch("store.get_relationship", return_value={"orphan_rate": 15.0}):
            messages = check_join_coverage("acct1", [self._edge()])
        self.assertEqual(len(messages), 1)
        self.assertIn("Orders", messages[0])
        self.assertIn("Customer", messages[0])
        self.assertIn("15%", messages[0])

    def test_at_or_below_threshold_produces_no_caveat(self):
        with patch("store.get_relationship", return_value={"orphan_rate": _ORPHAN_RATE_WARNING_THRESHOLD}):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])
        with patch("store.get_relationship", return_value={"orphan_rate": 2.0}):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])

    def test_unvalidated_relationship_sentinel_produces_no_caveat(self):
        with patch("store.get_relationship", return_value={"orphan_rate": -1.0}):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])

    def test_missing_relationship_row_skips_silently(self):
        with patch("store.get_relationship", return_value=None):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])

    def test_store_lookup_exception_skips_silently_not_raise(self):
        with patch("store.get_relationship", side_effect=RuntimeError("db down")):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])

    def test_malformed_edge_dict_skipped_not_raise(self):
        with patch("store.get_relationship", return_value={"orphan_rate": 15.0}):
            self.assertEqual(check_join_coverage("acct1", [{}, {"id": "not-an-int"}]), [])

    def test_multiple_edges_each_produce_their_own_caveat(self):
        def _get(account_id, rel_id):
            return {1: {"orphan_rate": 10.0}, 2: {"orphan_rate": 20.0}}.get(rel_id)

        with patch("store.get_relationship", side_effect=_get):
            messages = check_join_coverage("acct1", [
                self._edge(rel_id=1, from_entity="Orders", to_entity="Customer"),
                self._edge(rel_id=2, from_entity="Orders", to_entity="Product"),
            ])
        self.assertEqual(len(messages), 2)

    def test_missing_orphan_rate_field_skips_silently(self):
        with patch("store.get_relationship", return_value={}):
            self.assertEqual(check_join_coverage("acct1", [self._edge()]), [])

    def test_falls_back_to_relationship_row_entity_names_when_edge_lacks_them(self):
        with patch("store.get_relationship", return_value={
            "orphan_rate": 15.0, "from_entity": "RelFrom", "to_entity": "RelTo",
        }):
            messages = check_join_coverage("acct1", [{"id": 1}])
        self.assertIn("RelFrom", messages[0])
        self.assertIn("RelTo", messages[0])


if __name__ == "__main__":
    unittest.main()
