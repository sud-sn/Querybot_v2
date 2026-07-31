import time
import unittest
from unittest.mock import patch

from core.plan_preview import build_plan_preview, PendingPlanPreviewStore, PlanPreview


class BuildPlanPreviewTests(unittest.TestCase):
    def _resolution(self, **overrides):
        base = {
            "enabled": True, "detected": ["Orders", "Customer"],
            "anchor": "Orders", "graph_scope": "confirmed",
        }
        base.update(overrides)
        return base

    def test_single_table_summary(self):
        with patch("store.get_full_graph", return_value={}), \
             patch("core.graph_resolver.resolve_for_question",
                   return_value=self._resolution(detected=["Orders"])):
            preview = build_plan_preview("net revenue last 7 days", "acct1", "azure_sql")
        self.assertIn("Orders", preview.summary)
        self.assertEqual(preview.tables, ("Orders",))
        self.assertNotIn("unreviewed", preview.summary)

    def test_joined_tables_summary(self):
        with patch("store.get_full_graph", return_value={}), \
             patch("core.graph_resolver.resolve_for_question",
                   return_value=self._resolution()):
            preview = build_plan_preview("revenue by customer", "acct1", "azure_sql")
        self.assertIn("Orders", preview.summary)
        self.assertIn("Customer", preview.summary)
        self.assertEqual(preview.tables, ("Orders", "Customer"))

    def test_suggested_fallback_adds_caveat(self):
        with patch("store.get_full_graph", return_value={}), \
             patch("core.graph_resolver.resolve_for_question",
                   return_value=self._resolution(graph_scope="suggested_fallback")):
            preview = build_plan_preview("revenue by customer", "acct1", "azure_sql")
        self.assertIn("unreviewed", preview.summary)

    def test_no_resolution_falls_back_to_generic_message(self):
        with patch("store.get_full_graph", return_value={}), \
             patch("core.graph_resolver.resolve_for_question",
                   return_value={"enabled": False, "detected": [], "anchor": "", "graph_scope": ""}):
            preview = build_plan_preview("something vague", "acct1", "azure_sql")
        self.assertEqual(preview.tables, ())
        self.assertIn("normal way", preview.summary)


class PendingPlanPreviewStoreTests(unittest.TestCase):
    def test_set_then_get_returns_the_preview(self):
        store = PendingPlanPreviewStore()
        preview = PlanPreview(question="q", summary="s", tables=("T",), graph_scope="confirmed")
        store.set("acct", "sess", preview)
        self.assertEqual(store.get("acct", "sess"), preview)

    def test_get_returns_none_when_nothing_pending(self):
        store = PendingPlanPreviewStore()
        self.assertIsNone(store.get("acct", "sess"))

    def test_clear_removes_the_pending_entry(self):
        store = PendingPlanPreviewStore()
        preview = PlanPreview(question="q", summary="s", tables=(), graph_scope="")
        store.set("acct", "sess", preview)
        store.clear("acct", "sess")
        self.assertIsNone(store.get("acct", "sess"))

    def test_isolated_per_account_and_session(self):
        store = PendingPlanPreviewStore()
        p1 = PlanPreview(question="q1", summary="s1", tables=(), graph_scope="")
        p2 = PlanPreview(question="q2", summary="s2", tables=(), graph_scope="")
        store.set("acct1", "sess", p1)
        store.set("acct2", "sess", p2)
        self.assertEqual(store.get("acct1", "sess"), p1)
        self.assertEqual(store.get("acct2", "sess"), p2)

    def test_entry_expires_after_ttl(self):
        clock = {"now": 1000.0}
        store = PendingPlanPreviewStore(ttl_seconds=30)
        with patch("core.plan_preview.time.time", side_effect=lambda: clock["now"]):
            store.set("acct", "sess", PlanPreview(question="q", summary="s", tables=(), graph_scope=""))
            clock["now"] += 31
            self.assertIsNone(store.get("acct", "sess"))


if __name__ == "__main__":
    unittest.main()
