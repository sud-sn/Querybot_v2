from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TestDashboardLayoutContract(unittest.TestCase):
    def _read(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8")

    def test_pinned_chart_schema_has_grid_layout_columns(self):
        text = self._read("store/db.py")
        for column in ("grid_x", "grid_y", "grid_w", "grid_h"):
            self.assertIn(column, text)
            self.assertIn(f'("pinned_chart", "{column}"', text)

    def test_store_exposes_layout_update_method(self):
        text = self._read("store/user_store.py")
        self.assertIn("def update_pinned_chart_layouts", text)
        self.assertIn("WHERE id=? AND user_id=?", text)
        self.assertIn("grid_x=?, grid_y=?, grid_w=?, grid_h=?", text)

        init_text = self._read("store/__init__.py")
        self.assertIn("update_pinned_chart_layouts", init_text)

    def test_portal_exposes_dashboard_layout_api(self):
        text = self._read("portal/routes.py")
        self.assertIn('@router.post("/api/update-chart-layout")', text)
        self.assertIn("store.update_pinned_chart_layouts", text)

    def test_dashboard_template_uses_gridstack_with_fallback(self):
        text = self._read("portal/templates/portal_dashboard.html")
        self.assertIn("gridstack-all.js", text)
        self.assertIn("GridStack.init", text)
        self.assertIn("saveDashboardLayoutSoon", text)
        self.assertIn("/portal/api/update-chart-layout", text)
        self.assertIn("initFallbackDashboardDrag", text)
        for attr in ("gs-x", "gs-y", "gs-w", "gs-h"):
            self.assertIn(attr, text)


if __name__ == "__main__":
    unittest.main()
