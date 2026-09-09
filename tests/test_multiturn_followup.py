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

CHAT_TMPL        = os.path.join(os.path.dirname(__file__), "..", "portal",  "templates", "portal_chat.html")
DASH_TMPL        = os.path.join(os.path.dirname(__file__), "..", "portal",  "templates", "portal_dashboard.html")
LLM_PY           = os.path.join(os.path.dirname(__file__), "..", "core",    "llm.py")
ADAPTER_PY       = os.path.join(os.path.dirname(__file__), "..", "gateway", "web_adapter.py")
MAIN_PY          = os.path.join(os.path.dirname(__file__), "..", "main.py")
QUERY_PIPELINE_PY = os.path.join(os.path.dirname(__file__), "..", "core",   "query_pipeline.py")
RESULT_RENDERER_PY = os.path.join(os.path.dirname(__file__), "..", "core",  "result_renderer.py")

def _read(path):
    with open(path, encoding="utf-8") as f:
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
        self.assertNotIn("## Session context", p)

    def test_empty_history_no_session_block(self):
        p = self._build_prompt(history=[])
        self.assertNotIn("## Session context", p)

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


# ── 3  pipeline wiring ────────────────────────────────────────────────────────
class TestMainWiring(unittest.TestCase):

    def test_add_to_history_called_after_send_results(self):
        src = _read(QUERY_PIPELINE_PY)
        self.assertIn("add_to_history", src)

    def test_get_history_called_before_build_sql_prompt(self):
        src = _read(QUERY_PIPELINE_PY)
        self.assertIn("get_history", src)
        self.assertIn("conversation_history=_conv_history", src)

    def test_clear_history_called_on_ws_connect(self):
        src = _read(ADAPTER_PY)
        self.assertIn("clear_history", src)

    def test_history_columns_derived_from_rows(self):
        src = _read(QUERY_PIPELINE_PY)
        self.assertIn("list(rows[0].keys())", src)

    def test_generate_followup_suggestions_imported(self):
        src = _read(RESULT_RENDERER_PY)
        self.assertIn("generate_followup_suggestions", src)

    def test_follow_up_suggestions_added_to_payload(self):
        src = _read(RESULT_RENDERER_PY)
        self.assertIn("follow_up_suggestions", src)
        self.assertIn("response_payload[\"follow_up_suggestions\"]", src)

    def test_follow_up_only_for_portal_users(self):
        src = _read(RESULT_RENDERER_PY)
        # Must check portal_user before generating suggestions
        idx_suggest = src.rfind("generate_followup_suggestions")
        idx_portal  = src.rfind("portal_user", 0, idx_suggest)
        self.assertGreater(idx_suggest, 0)
        self.assertGreater(idx_portal, 0)


# ── 4  generate_followup_suggestions ─────────────────────────────────────────
class TestGenerateFollowupSuggestions(unittest.TestCase):

    # ── Executed, not grepped ────────────────────────────────────────────
    #
    # Nine tests here read core/insight.py as text and asserted that fragments
    # appeared in it somewhere: "len(columns) < 2", "row_count == 0",
    # "return []", "][:3]", "max_tokens=160", "llm_complete", "json.loads",
    # 'component="followup_suggestions"'. The file is 1,800 lines, so every one
    # of them passed on a substring that could have been in an unrelated
    # function, a comment, or a docstring — and would have kept passing with
    # the guard it names deleted.
    #
    # Two were worse than useless: assertIn("return []") and
    # assertIn("except Exception") are true of most modules in this repo.
    #
    # Below, each guard is exercised through the real coroutine. Only the two
    # boundaries are mocked: the LLM call and the compliance posture.

    BRIEF = {
        "row_count": 6,
        "columns": ["WHS_NM", "REVENUE_AMT"],
        "category_breakdown": {
            "label_column": "WHS_NM", "value_column": "REVENUE_AMT",
            "top_5": [{"label": "Halifax", "value": 100.0}],
            "category_count": 3,
        },
    }

    _DEFAULT = object()

    def suggestions(self, brief=_DEFAULT, reply='["a", "b", "c", "d", "e"]',
                    regulated=False, raises=None):
        """Run the real coroutine, mocking only what sits outside it.

        Three boundaries: the compliance posture, provider resolution (it
        reads the encrypted credential store, and raises on a tenant with no
        API key — which the function's own except-Exception would then swallow
        into [], making every assertion below pass for the wrong reason), and
        the model call itself.

        The model mock records the audit scope in force while the call is in
        flight, because that is where `component` actually lives — it is a
        llm_audit_scope argument, not an llm_complete one.
        """
        import asyncio
        from unittest.mock import AsyncMock, patch

        import core.compliance.policy_engine as policy_engine
        import core.llm as llm
        from core.insight import generate_followup_suggestions
        from core.llm_audit import _AUDIT_SCOPE

        self.audit_scope = {}

        def _respond(*_args, **_kwargs):
            self.audit_scope = dict(_AUDIT_SCOPE.get() or {})
            if raises:
                raise raises
            return (reply, 1, 1)

        call = AsyncMock(side_effect=_respond)
        self.last_call = call
        with patch.object(llm, "llm_complete", new=call), \
             patch.object(llm, "resolve_provider",
                          return_value=("openai", "gpt-4o-mini", "k", {})), \
             patch.object(policy_engine, "is_regulated", return_value=regulated):
            return asyncio.run(generate_followup_suggestions(
                brief=self.BRIEF if brief is self._DEFAULT else brief,
                question="revenue by warehouse", result_scope={}, db_cfg={},
                account_id="acct"))

    def prompt(self):
        """The user message the function actually built."""
        return self.last_call.await_args.args[1]

    def questions(self, **kw):
        """The English half of each chip -- what the planner re-reads.

        The chips became {"question", "label"} pairs so a French reader can be
        shown French copy while the text that gets re-planned stays English.
        These tests are about the question half; the label half has its own
        tests in tests/test_followup_chip_language.py.
        """
        return [p["question"] for p in self.suggestions(**kw)]

    def test_it_suggests_something_for_an_ordinary_result(self):
        # The control. Without it, every test below passes on a function that
        # returns [] unconditionally.
        self.assertEqual(self.questions(), ["a", "b", "c"])

    def test_suggestions_are_capped_at_three(self):
        self.assertEqual(
            self.questions(reply='["a","b","c","d","e","f","g"]'),
            ["a", "b", "c"])

    def test_a_single_column_result_gets_none(self):
        # A scalar answer has nothing to break down.
        self.assertEqual(
            self.suggestions({**self.BRIEF, "columns": ["REVENUE_AMT"]}), [])

    def test_a_zero_row_result_gets_none(self):
        self.assertEqual(self.suggestions({**self.BRIEF, "row_count": 0}), [])

    def test_an_empty_brief_gets_none(self):
        self.assertEqual(self.suggestions({}), [])

    def test_a_missing_brief_gets_none_rather_than_raising(self):
        # A caller with no brief at all passes None, and `brief.get` on it
        # would raise from OUTSIDE the try -- so this one guard is the
        # difference between no chips and no answer.
        self.assertEqual(self.suggestions(None), [])

    def test_a_repeated_suggestion_is_not_shown_twice(self):
        self.assertEqual(
            self.questions(reply='["same", "same", "other", "third"]'),
            ["same", "other", "third"])

    def test_an_overlong_suggestion_is_trimmed_not_dropped(self):
        # These render as chips in a row; one that wraps to three lines breaks
        # the layout, and dropping it silently leaves the row short.
        got = self.questions(reply='["%s", "b", "c"]' % ("x" * 200))
        self.assertEqual([len(s) for s in got], [80, 1, 1])

    def test_the_model_is_not_even_called_for_those(self):
        # The guards must short-circuit, not filter afterwards: reaching the
        # LLM at all is the egress this function is careful about.
        self.suggestions({**self.BRIEF, "row_count": 0})
        self.last_call.assert_not_awaited()

    def test_a_regulated_tenant_gets_none(self):
        self.assertEqual(self.suggestions(regulated=True), [])

    def test_and_the_model_is_not_called_for_a_regulated_tenant(self):
        self.suggestions(regulated=True)
        self.last_call.assert_not_awaited()

    def test_a_failing_model_costs_the_suggestions_not_the_answer(self):
        self.assertEqual(self.suggestions(raises=RuntimeError("boom")), [])

    def test_a_reply_that_is_not_json_costs_the_suggestions_not_the_answer(self):
        self.assertEqual(self.suggestions(reply="not json at all"), [])

    def test_the_token_budget_stays_small(self):
        # Asserted on the call the function actually makes.
        self.suggestions()
        self.assertEqual(self.last_call.await_args.kwargs.get("max_tokens"), 160)

    def test_the_audit_names_this_component(self):
        # Read off the scope that was actually in force during the call, not
        # off the source text -- the egress row is filed under this name.
        self.suggestions()
        self.assertEqual(self.audit_scope.get("component"),
                         "followup_suggestions")

    def test_no_raw_row_value_can_reach_the_prompt(self):
        """The PII boundary, asserted on the signature rather than on a slice
        of source text 400 characters wide."""
        import inspect

        from core.insight import generate_followup_suggestions

        parameters = inspect.signature(generate_followup_suggestions).parameters
        self.assertIn("brief", parameters)
        self.assertIn("result_scope", parameters)
        self.assertNotIn("rows", parameters,
                         "this function must never receive raw result rows")

    # ── The PII boundary, on the prompt that was actually built ──────────
    #
    # Category labels DO reach the model -- that is the grounding the whole
    # function exists for. What must not reach it is a label drawn from a
    # sensitive column, or the redaction sentinel standing in for one. The
    # three tests below fix all three halves of that rule, so weakening any
    # one of them fails here rather than shipping.

    def test_a_safe_category_label_does_reach_the_model(self):
        # The control. Without it the two tests below pass on a prompt that
        # carries no labels at all.
        self.suggestions()
        self.assertIn("Halifax", self.prompt())

    def test_a_label_from_a_sensitive_column_does_not(self):
        self.suggestions({
            **self.BRIEF,
            "columns": ["PATIENT_NAME", "REVENUE_AMT"],
            "category_breakdown": {"label_column": "PATIENT_NAME",
                                   "top_5": [{"label": "Jane Roe", "value": 1.0}]},
        })
        self.assertNotIn("Jane Roe", self.prompt())

    def test_the_redaction_sentinel_never_reaches_the_model(self):
        # Upstream masking replaces a value with this exact string; forwarding
        # it would tell the model a segment exists and was hidden.
        self.suggestions({
            **self.BRIEF,
            "category_breakdown": {
                "label_column": "WHS_NM",
                "top_5": [{"label": "redacted segment", "value": 1.0},
                          {"label": "Halifax", "value": 2.0}],
            },
        })
        prompt = self.prompt()
        self.assertNotIn("redacted segment", prompt)
        self.assertIn("Halifax", prompt)


# ── 5  portal_chat.html follow-up rendering ───────────────────────────────────
class TestChatTemplateFollowUp(unittest.TestCase):

    def test_follow_up_chips_rendered(self):
        tmpl = _read(CHAT_TMPL)
        self.assertIn("follow_up_suggestions", tmpl)
        self.assertIn("follow-up-chip", tmpl)

    def test_follow_up_label_present(self):
        """The label above the follow-up chips.

        Was assertIn("Based on this result", <the template source>), which
        broke the moment the copy moved into the catalogue -- and would have
        passed just as happily if the label had been deleted from the DOM and
        left in a comment. Asserted on the resolved copy now, in both
        languages, so it fails when the label goes missing rather than when it
        gets translated.
        """
        from core import i18n

        for lang in i18n.SUPPORTED_LANGUAGES:
            with self.subTest(lang=lang):
                label = i18n.lookup("ui.chat.suggestions.label", lang)
                self.assertTrue(label.strip())
                self.assertNotEqual(label, "ui.chat.suggestions.label")
        self.assertIn("ui.chat.suggestions.label", _read(CHAT_TMPL))

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
        # The label moved into the message catalogue, so the source now holds
        # it whatever renders. Assert the rendered button instead -- which also
        # covers the fact that it is only drawn on a card that can expand.
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from dashboard_render import CHART, render, visible

        markup = visible(render(charts=[CHART]))
        self.assertIn("⤢ Expand", markup)
        self.assertIn("openChartModal(this.closest('.chart-card'))", markup)

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

    # The two greps that were here -- '"followup_suggestions"' in the module
    # source, and "llm_audit_scope" in the module source -- are replaced by
    # TestGenerateFollowupSuggestions.test_the_audit_names_this_component,
    # which reads the scope actually in force while the model call is in
    # flight. Either grep passed on a match anywhere in 1,800 lines, including
    # in the docstring that names the component.

    def test_history_injection_uses_existing_sql_gen_scope(self):
        """History is injected into the existing sql_generation scope, not a new one."""
        src = _read(QUERY_PIPELINE_PY)
        # The history enriches the system prompt before the sql_generation scope
        idx_conv  = src.find("conversation_history=_conv_history")
        idx_scope = src.find('component="sql_generation"')
        self.assertGreater(idx_scope, idx_conv,
            "History must be set before the sql_generation audit scope opens")


if __name__ == "__main__":
    unittest.main()
