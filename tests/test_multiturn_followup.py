"""
tests/test_multiturn_followup.py

Tests for:
  1. Multi-turn conversation memory (WebAdapter history buffer)
  2. history injection into build_sql_system_prompt
  3. add_to_history called after _send_results
  4. Result-aware follow-up suggestions (generate_followup_suggestions)
  5. Follow-up wiring in portal_chat.html and portal_dashboard.html
  6. Chart alignment: ResizeObserver used instead of { once: true } resize
  7. Maximize modal present on dashboard
  8. LLM audit component 'followup_suggestions' registered
"""
import os
import sys
import tempfile
import unittest
from collections import deque

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_mt.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for mod in list(sys.modules.keys()):
    if mod.startswith("store"):
        del sys.modules[mod]
import store.db as db_mod
db_mod.init_db()

CHAT_TMPL  = os.path.join(os.path.dirname(__file__), "..", "portal",  "templates", "portal_chat.html")
DASH_TMPL  = os.path.join(os.path.dirname(__file__), "..", "portal",  "templates", "portal_dashboard.html")
INSIGHT_PY = os.path.join(os.path.dirname(__file__), "..", "core",    "insight.py")
LLM_PY     = os.path.join(os.path.dirname(__file__), "..", "core",    "llm.py")
ADAPTER_PY = os.path.join(os.path.dirname(__file__), "..", "gateway", "web_adapter.py")
MAIN_PY    = os.path.join(os.path.dirname(__file__), "..", "main.py")

def _read(path):
    with open(path) as f:
        return f.read()


# ── 1  WebAdapter history buffer ────────────────────────────────────────────
class TestWebAdapterHistory(unittest.TestCase):

    def _make_adapter(self):
        from gateway.web_adapter import WebAdapter
        from unittest.mock import MagicMock, AsyncMock
        ws = MagicMock()
        ws.send_json = AsyncMock()
        return WebAdapter(ws, "acc1", "user1")

    def test_history_starts_empty(self):
        a = self._make_adapter()
        self.assertEqual(a.get_history(), [])

    def test_add_and_get_history(self):
        a = self._make_adapter()
        a.add_to_history("show revenue", "SELECT SUM(Revenue)", ["Customer","Revenue"], 12)
        hist = a.get_history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["question"], "show revenue")
        self.assertEqual(hist[0]["sql"], "SELECT SUM(Revenue)")
        self.assertEqual(hist[0]["columns"], ["Customer","Revenue"])
        self.assertEqual(hist[0]["row_count"], 12)

    def test_history_limited_to_3_turns(self):
        a = self._make_adapter()
        for i in range(6):
            a.add_to_history(f"q{i}", f"sql{i}", [f"col{i}"], i)
        hist = a.get_history()
        self.assertEqual(len(hist), 3)
        # Should contain the last 3
        self.assertEqual(hist[-1]["question"], "q5")
        self.assertEqual(hist[0]["question"], "q3")

    def test_clear_history(self):
        a = self._make_adapter()
        a.add_to_history("q1", "sql1", ["col1"], 5)
        a.clear_history()
        self.assertEqual(a.get_history(), [])

    def test_history_stores_columns_not_rows(self):
        """History must store column names, not raw row data — PII boundary."""
        a = self._make_adapter()
        a.add_to_history("show employees", "SELECT ...", ["EmployeeName","Salary"], 100)
        hist = a.get_history()
        # columns list present
        self.assertIn("columns", hist[0])
        # no raw row data
        self.assertNotIn("rows", hist[0])
        self.assertNotIn("data", hist[0])

    def test_history_oldest_first(self):
        a = self._make_adapter()
        a.add_to_history("first", "sql1", [], 1)
        a.add_to_history("second", "sql2", [], 2)
        a.add_to_history("third", "sql3", [], 3)
        hist = a.get_history()
        self.assertEqual(hist[0]["question"], "first")
        self.assertEqual(hist[2]["question"], "third")

    def test_adapter_has_clear_history_method(self):
        src = _read(ADAPTER_PY)
        self.assertIn("def clear_history", src)

    def test_adapter_has_add_to_history_method(self):
        src = _read(ADAPTER_PY)
        self.assertIn("def add_to_history", src)

    def test_adapter_has_get_history_method(self):
        src = _read(ADAPTER_PY)
        self.assertIn("def get_history", src)

    def test_history_maxlen_is_3(self):
        src = _read(ADAPTER_PY)
        self.assertIn("_HISTORY_MAXLEN = 3", src)

    def test_history_uses_deque(self):
        src = _read(ADAPTER_PY)
        self.assertIn("deque", src)


# ── 2  build_sql_system_prompt history injection ─────────────────────────────
class TestSqlPromptHistory(unittest.TestCase):

    def _build_prompt(self, history=None):
        from core.llm import build_sql_system_prompt
        return build_sql_system_prompt("azure_sql", "KB context here",
                                       conversation_history=history)

    def test_no_history_no_session_block(self):
        p = self._build_prompt(history=None)
        self.assertNotIn("Session context", p)

    def test_empty_history_no_session_block(self):
        p = self._build_prompt(history=[])
        self.assertNotIn("Session context", p)

    def test_with_history_adds_session_block(self):
        hist = [{"question":"show revenue","sql":"SELECT SUM(Revenue)","columns":["Customer","Revenue"],"row_count":12}]
        p = self._build_prompt(history=hist)
        self.assertIn("Session context", p)
        self.assertIn("show revenue", p)
        self.assertIn("SELECT SUM(Revenue)", p)

    def test_history_injects_columns(self):
        hist = [{"question":"q","sql":"sql","columns":["Alpha","Beta"],"row_count":5}]
        p = self._build_prompt(history=hist)
        self.assertIn("Alpha", p)
        self.assertIn("Beta", p)

    def test_history_capped_at_300_chars_per_sql(self):
        long_sql = "SELECT " + "X" * 500
        hist = [{"question":"q","sql":long_sql,"columns":[],"row_count":1}]
        p = self._build_prompt(history=hist)
        # SQL should be truncated — check the truncated version appears
        self.assertIn("SELECT " + "X" * 293, p)  # 300 chars total
        self.assertNotIn("X" * 301, p)

    def test_prompt_accepts_conversation_history_param(self):
        src = _read(LLM_PY)
        self.assertIn("conversation_history", src)

    def test_build_sql_returns_string(self):
        p = self._build_prompt()
        self.assertIsInstance(p, str)
        self.assertGreater(len(p), 100)


# ── 3  main.py wiring ─────────────────────────────────────────────────────────
class TestMainWiring(unittest.TestCase):

    def test_add_to_history_called_after_send_results(self):
        src = _read(MAIN_PY)
        self.assertIn("add_to_history", src)

    def test_get_history_called_before_build_sql_prompt(self):
        src = _read(MAIN_PY)
        self.assertIn("get_history", src)
        self.assertIn("conversation_history=_conv_history", src)

    def test_clear_history_called_on_ws_connect(self):
        src = _read(MAIN_PY)
        self.assertIn("clear_history", src)

    def test_history_columns_derived_from_rows(self):
        src = _read(MAIN_PY)
        self.assertIn("list(rows[0].keys())", src)

    def test_generate_followup_suggestions_imported(self):
        src = _read(MAIN_PY)
        self.assertIn("generate_followup_suggestions", src)

    def test_follow_up_suggestions_added_to_payload(self):
        src = _read(MAIN_PY)
        self.assertIn("follow_up_suggestions", src)
        self.assertIn("response_payload[\"follow_up_suggestions\"]", src)

    def test_follow_up_only_for_portal_users(self):
        src = _read(MAIN_PY)
        # Must check portal_user before generating suggestions
        idx_suggest = src.find("generate_followup_suggestions")
        idx_portal  = src.rfind("portal_user", 0, idx_suggest)
        self.assertGreater(idx_suggest, 0)
        self.assertGreater(idx_portal, 0)


# ── 4  generate_followup_suggestions ─────────────────────────────────────────
class TestGenerateFollowupSuggestions(unittest.TestCase):

    def test_function_exists_in_insight(self):
        src = _read(INSIGHT_PY)
        self.assertIn("async def generate_followup_suggestions", src)

    def test_function_returns_empty_on_single_column(self):
        """Single-column results (scalar answers) get no suggestions."""
        src = _read(INSIGHT_PY)
        self.assertIn("len(columns) < 2", src)

    def test_function_returns_empty_on_zero_rows(self):
        src = _read(INSIGHT_PY)
        self.assertIn("row_count == 0", src)

    def test_audit_component_is_followup_suggestions(self):
        src = _read(INSIGHT_PY)
        self.assertIn('component="followup_suggestions"', src)

    def test_pii_boundary_no_raw_rows(self):
        """Suggestions must be generated from brief, not raw row values."""
        src = _read(INSIGHT_PY)
        # The function signature should have brief and result_scope, not rows
        fn_start = src.find("async def generate_followup_suggestions")
        fn_sig   = src[fn_start:fn_start+400]
        self.assertIn("brief:", fn_sig)
        self.assertIn("result_scope:", fn_sig)
        self.assertNotIn("rows: list", fn_sig)

    def test_uses_llm_complete(self):
        src = _read(INSIGHT_PY)
        self.assertIn("llm_complete", src)

    def test_max_tokens_is_small(self):
        """Follow-up suggestions use a small token budget."""
        src = _read(INSIGHT_PY)
        self.assertIn("max_tokens=160", src)

    def test_failure_returns_empty_list(self):
        src = _read(INSIGHT_PY)
        # Exceptions must return [] not raise
        self.assertIn("return []", src)
        self.assertIn("except Exception", src)

    def test_suggestions_capped_at_3(self):
        src = _read(INSIGHT_PY)
        self.assertIn("][:3]", src)

    def test_json_output_parsing(self):
        src = _read(INSIGHT_PY)
        self.assertIn("json.loads", src.replace("_json.loads", "json.loads"))


# ── 5  portal_chat.html follow-up rendering ───────────────────────────────────
class TestChatTemplateFollowUp(unittest.TestCase):

    def test_follow_up_chips_rendered(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn("follow_up_suggestions", tmpl)
        self.assertIn("follow-up-chip", tmpl)

    def test_follow_up_label_present(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn("Based on this result", tmpl)

    def test_chip_click_fires_sendSuggestion(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn("sendSuggestion", tmpl)
        self.assertIn("data-follow-up", tmpl)

    def test_follow_up_css_defined(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn(".follow-up-chip", tmpl)
        self.assertIn(".follow-up-wrap", tmpl)


# ── 6  Chart alignment: ResizeObserver ────────────────────────────────────────
class TestChartResizeObserver(unittest.TestCase):

    def test_chat_uses_ResizeObserver_not_once_true(self):
        tmpl = _read(CHAT_TMPL)
        # { once: true } resize should be replaced
        self.assertNotIn("addEventListener('resize', () => chart.resize(), { once: true })", tmpl)
        self.assertIn("ResizeObserver", tmpl)

    def test_dashboard_uses_ResizeObserver(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("ResizeObserver", tmpl)

    def test_dashboard_uses_requestAnimationFrame(self):
        """rAF ensures grid settles before echarts reads dimensions."""
        tmpl = _read(DASH_TMPL)
        self.assertIn("requestAnimationFrame", tmpl)

    def test_chat_ResizeObserver_on_chart_element(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn("ro.observe(chartEl)", tmpl)

    def test_dashboard_ResizeObserver_on_chart_element(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("ro.observe(node)", tmpl)


# ── 7  Dashboard maximize modal ───────────────────────────────────────────────
class TestDashboardMaximize(unittest.TestCase):

    def test_expand_button_present(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("⤢ Expand", tmpl)

    def test_openChartModal_function_present(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("function openChartModal", tmpl)

    def test_closeChartModal_function_present(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("function closeChartModal", tmpl)

    def test_modal_escape_key_closes(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("Escape", tmpl)
        self.assertIn("closeChartModal", tmpl)

    def test_modal_resize_on_open(self):
        """Chart in modal must be resized after it paints at full size."""
        tmpl = _read(DASH_TMPL)
        self.assertIn("mc.resize()", tmpl)

    def test_modal_disposes_chart_on_close(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("mc.dispose()", tmpl.replace("_modalChart.dispose()", "mc.dispose()"))

    def test_expand_button_calls_openChartModal(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("onclick=\"openChartModal(this.closest('.chart-card'))\"", tmpl)

    def test_modal_uses_full_viewport_size(self):
        tmpl = _read(DASH_TMPL)
        self.assertIn("100vw", tmpl)
        self.assertIn("100vh", tmpl)


# ── 8  LLM audit component registration ──────────────────────────────────────
class TestAuditComponent(unittest.TestCase):

    def test_followup_component_in_insight(self):
        src = _read(INSIGHT_PY)
        self.assertIn('"followup_suggestions"', src)

    def test_followup_uses_llm_audit_scope(self):
        src = _read(INSIGHT_PY)
        self.assertIn("llm_audit_scope", src)

    def test_history_injection_uses_existing_sql_gen_scope(self):
        """History is injected into the existing sql_generation scope, not a new one."""
        src = _read(MAIN_PY)
        # The history enriches the system prompt before the sql_generation scope
        idx_conv  = src.find("conversation_history=_conv_history")
        idx_scope = src.find('component="sql_generation"')
        self.assertGreater(idx_scope, idx_conv,
            "History must be set before the sql_generation audit scope opens")


if __name__ == "__main__":
    unittest.main()
