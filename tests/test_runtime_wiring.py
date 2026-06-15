"""
tests/test_runtime_wiring.py

Tests that catch the class of bugs that reached production:

  Bug 1: NameError — _build_chat_suggestions not defined in portal/routes.py
  Bug 2: 422 on semantic feedback form — status field not sent by form.submit()

These are runtime-wiring tests. They don't test business logic — they verify
that every function called by a route actually exists and is importable, and
that every HTML form that POSTs to a route sends all required fields.

Why these tests are needed:
  - ast.parse() confirms syntax but not that names resolve at runtime
  - Jinja ChainableUndefined silently swallows undefined template names
  - Unit tests mock everything so NameErrors in the real module are invisible
  - Browser form.submit() quirks can't be caught by backend tests alone
    but we CAN verify the HTML has the hidden field the backend requires

Run: python -m unittest tests.test_runtime_wiring
"""

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def _collect_defined_names(path: str, *, include_assigns: bool = False) -> set[str]:
    """
    Parse a Python source file and return the set of names that are defined or
    imported at any scope level.  Used to verify that every name called at
    runtime actually resolves — catching NameError bugs before they hit prod.

    include_assigns: also collect bare Name targets from assignment statements
    (e.g. ``MY_CONST = ...``).  Useful for modules that expose constants as
    their public API.
    """
    src = _read(path)
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif include_assigns and isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


# ══════════════════════════════════════════════════════════════════════════════
# Bug 1 class: functions called in route handlers must actually be defined
# ══════════════════════════════════════════════════════════════════════════════

class PortalRoutesFunctionPresenceTests(unittest.TestCase):
    """
    Every helper function called inside portal/routes.py must be defined
    (or imported) in the same file.  A missing def causes NameError at
    runtime when the route is first hit — not at startup.

    This test replicates the exact failure:
        NameError: name '_build_chat_suggestions' is not defined
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read("portal/routes.py")
        cls.defined = _collect_defined_names("portal/routes.py", include_assigns=True)

    def _assert_defined(self, name: str):
        self.assertIn(
            name, self.defined,
            f"'{name}' is called in portal/routes.py but is not defined or imported. "
            f"This causes NameError at runtime when the route is first hit."
        )

    def test_build_chat_suggestions_defined(self):
        """The function called on every /portal/chat request must exist."""
        self._assert_defined("_build_chat_suggestions")

    def test_get_available_schemas_defined(self):
        """Schema selector helper must exist."""
        self._assert_defined("_get_available_schemas")

    def test_get_portal_user_defined(self):
        self._assert_defined("_get_portal_user")

    def test_login_redirect_defined(self):
        self._assert_defined("_login_redirect")

    def test_resp_defined(self):
        self._assert_defined("_resp")

    def test_guess_dynamic_suggestions_defined(self):
        self._assert_defined("_guess_dynamic_suggestions")

    def test_guess_safe_metric_suggestions_defined(self):
        self._assert_defined("_guess_safe_metric_suggestions")

    def test_build_chat_suggestions_is_callable(self):
        """
        Verify _build_chat_suggestions is defined as a function (not just a name)
        in portal/routes.py. Uses AST to avoid requiring fastapi at test time.
        """
        src = _read("portal/routes.py")
        tree = ast.parse(src)
        fn_defs = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn(
            "_build_chat_suggestions", fn_defs,
            "_build_chat_suggestions is not defined as a function in portal/routes.py — "
            "this is the exact Bug 1 that caused 500 on /portal/chat"
        )

    def test_get_available_schemas_is_callable(self):
        """Verify _get_available_schemas is defined as a function."""
        src = _read("portal/routes.py")
        tree = ast.parse(src)
        fn_defs = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn(
            "_get_available_schemas", fn_defs,
            "_get_available_schemas is not defined as a function in portal/routes.py"
        )

    def test_build_chat_suggestions_body_not_inside_another_function(self):
        """
        The specific failure mode: _build_chat_suggestions was accidentally
        placed as dead code inside _get_available_schemas after its return.
        Verify it's a top-level function def, not nested inside another.
        """
        src = _read("portal/routes.py")
        tree = ast.parse(src)
        # Find top-level function definitions only
        top_level_fns = {
            node.name for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn(
            "_build_chat_suggestions", top_level_fns,
            "_build_chat_suggestions is not a top-level function — "
            "it may be nested inside another function as dead code "
            "(the exact Bug 1 failure mode)"
        )


class AdminRoutesFunctionPresenceTests(unittest.TestCase):
    """Same check for admin/routes.py."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read("admin/routes.py")
        cls.defined = _collect_defined_names("admin/routes.py")

    def _assert_defined(self, name: str):
        self.assertIn(name, self.defined,
                      f"'{name}' missing in admin/routes.py")

    def test_is_auth_defined(self):
        self._assert_defined("_is_auth")

    def test_resp_defined(self):
        self._assert_defined("_resp")

    def test_semantic_feedback_review_defined(self):
        self._assert_defined("semantic_feedback_review")

    def test_db_connection_error_message_defined(self):
        self._assert_defined("_db_connection_error_message")


# ══════════════════════════════════════════════════════════════════════════════
# Bug 2 class: HTML forms must contain all fields the backend route requires
# ══════════════════════════════════════════════════════════════════════════════

class SemanticFeedbackFormTests(unittest.TestCase):
    """
    Verify the semantic feedback review form sends all fields the backend
    requires.

    The exact Bug 2 failure:
      POST /admin/clients/{id}/semantic-feedback/{id}/review
      422 Unprocessable Entity: Field required: status

    Root cause: the approve button used type="submit" name="status" but
    onclick called this.form.submit() — programmatic form submission in
    browsers does NOT include the clicked button's value.

    The fix: a hidden <input name="status"> field that JS sets before submit.
    This test verifies the hidden field exists in the template.
    """

    @classmethod
    def setUpClass(cls):
        cls.html = _read("admin/templates/client_kb.html")

    def test_hidden_status_field_exists(self):
        """
        The form must contain a hidden input named 'status'.
        Without it, programmatic form.submit() drops the status value
        and the backend gets a 422.
        """
        # Look for <input type="hidden" name="status" ...>
        has_hidden = bool(re.search(
            r'<input[^>]*type=["\']hidden["\'][^>]*name=["\']status["\']',
            self.html, re.I
        ) or re.search(
            r'<input[^>]*name=["\']status["\'][^>]*type=["\']hidden["\']',
            self.html, re.I
        ))
        self.assertTrue(
            has_hidden,
            "client_kb.html is missing <input type='hidden' name='status'>. "
            "Without this, programmatic form.submit() does not send the status "
            "field and the backend returns 422."
        )

    def test_submit_review_js_function_exists(self):
        """
        The submitReview() JS function must exist in the template.
        It sets the hidden status field before form submission.
        """
        self.assertIn(
            "function submitReview",
            self.html,
            "submitReview JS function missing from client_kb.html"
        )

    def test_approve_button_calls_submit_review(self):
        """Approve button must call submitReview(), not form.submit() directly."""
        self.assertIn(
            "submitReview",
            self.html,
            "Approve button must call submitReview() to set status before submit"
        )
        # Must NOT use this.form.submit() on an approval button
        # (that drops the button value)
        self.assertNotIn(
            "this.form.submit()",
            self.html,
            "form.submit() used directly — this drops the button value. "
            "Use submitReview() instead."
        )

    def test_review_form_has_action_with_review_path(self):
        """The review form action must point to the /review endpoint."""
        self.assertIn(
            "/review",
            self.html,
            "Review form action must include /review path"
        )

    def test_admin_note_textarea_present(self):
        """Admin note textarea must be present in the review form."""
        self.assertIn(
            'name="admin_note"',
            self.html,
            "admin_note textarea missing from review form"
        )


# ══════════════════════════════════════════════════════════════════════════════
# General: all route files parse cleanly and key route paths are registered
# ══════════════════════════════════════════════════════════════════════════════

class RouteRegistrationTests(unittest.TestCase):
    """
    Verify that key routes are registered and reachable.
    This catches the case where a decorator is missing or a route is
    defined inside a wrong scope.
    """

    def test_portal_chat_route_registered(self):
        """GET /portal/chat must be registered."""
        src = _read("portal/routes.py")
        self.assertIn('"/chat"', src,
                      "/portal/chat route not found in portal/routes.py")

    def test_semantic_feedback_review_route_registered(self):
        """POST /clients/{id}/semantic-feedback/{id}/review must be registered."""
        src = _read("admin/routes.py")
        self.assertIn("semantic-feedback", src,
                      "semantic-feedback route not in admin/routes.py")
        self.assertIn("/review", src,
                      "/review endpoint not in admin/routes.py")

    def test_portal_routes_no_syntax_errors(self):
        ast.parse(_read("portal/routes.py"))

    def test_admin_routes_no_syntax_errors(self):
        ast.parse(_read("admin/routes.py"))

    def test_main_no_syntax_errors(self):
        ast.parse(_read("main.py"))

    def test_portal_chat_template_no_undefined_jinja_calls(self):
        """
        Verify portal_chat.html doesn't call undefined Jinja macros or
        access undefined variables that would crash rendering.

        We render with a minimal context and ChainableUndefined to catch
        obvious issues — but also verify key functions used in JS are present.
        """
        html = _read("portal/templates/portal_chat.html")
        # These JS functions must be defined in the template
        for fn in ["sendMessage", "sendSuggestion", "selectSchema",
                   "_refreshSuggestionsForSchema", "_activeSchema"]:
            self.assertIn(fn, html,
                          f"JS symbol '{fn}' missing from portal_chat.html")

    def test_portal_chat_schema_hint_in_ws_send(self):
        """
        schema_hint must be included in the WebSocket payload sent to the server.
        """
        html = _read("portal/templates/portal_chat.html")
        self.assertIn("schema_hint", html,
                      "schema_hint not sent in WebSocket payload")

    def test_portal_chat_token_kpi_present(self):
        """The chat portal must render the user token KPI."""
        html = _read("portal/templates/portal_chat.html")
        self.assertIn("token-kpi-pill", html)
        self.assertIn("Tokens this month", html)

    def test_schema_selector_is_not_nested_inside_suggestions_only(self):
        """Schema selector should be available even when suggestions are empty."""
        html = _read("portal/templates/portal_chat.html")
        schema_pos = html.index("schema-control-row")
        suggestions_pos = html.index("suggestionsWrap")
        self.assertLess(schema_pos, suggestions_pos)

    def test_schema_scope_reaches_backend_pipeline(self):
        """Selected schema should constrain backend retrieval/validation scope."""
        src = _read("core/query_pipeline.py")
        self.assertIn("query_scope_tables", src)
        self.assertIn("rag_filter = query_scope_tables", src)
        self.assertIn("ACTIVE SCHEMA", src)

    def test_admin_semantic_feedback_realtime_wiring(self):
        """Semantic Layer edit requests should notify admin without refresh."""
        admin_src = _read("admin/routes.py")
        portal_src = _read("portal/routes.py")
        base_html = _read("admin/templates/base.html")
        detail_html = _read("admin/templates/client_detail.html")

        self.assertIn('"/ws/notifications"', admin_src)
        self.assertIn('"/api/semantic-feedback/pending-count"', admin_src)
        self.assertIn("notify_semantic_feedback_changed", admin_src)
        self.assertIn("notify_semantic_feedback_changed", portal_src)
        self.assertIn("semanticPendingBadge", base_html)
        self.assertIn("new WebSocket", base_html)
        self.assertIn("semanticPendingInline", detail_html)

    def test_admin_kb_build_live_progress_wiring(self):
        """Admin KB generation should show live progress instead of relying on refresh."""
        admin_src = _read("admin/routes.py")
        knowledge_src = _read("core/knowledge.py")
        setup_html = _read("admin/templates/client_setup.html")
        admin_notifications = _read("core/admin_notifications.py")

        self.assertIn('"/clients/{account_id}/setup/status"', admin_src)
        self.assertIn("notify_kb_build_changed", admin_src)
        self.assertIn("progress_callback=_on_kb_progress", admin_src)
        self.assertIn("kb_progress", admin_src)
        self.assertIn("notify_kb_build_changed", admin_notifications)
        self.assertIn("progress_callback", knowledge_src)
        self.assertIn("Completed {table_name}", knowledge_src)
        self.assertIn("kbLiveStatus", setup_html)
        self.assertIn("qb-admin-notification", setup_html)
        self.assertIn("Building… ${current}/${total} completed", setup_html)
        self.assertIn("kbProgressPercent", setup_html)
        self.assertIn("Storing Knowledge Base in vector index", knowledge_src)
        self.assertIn("Knowledge Base stored in vector index", knowledge_src)
        self.assertIn("Success - Knowledge Base stored and ready", admin_src)

    def test_portal_dashboard_remaining_token_kpi_wiring(self):
        """Portal dashboard must show remaining monthly token allowance."""
        portal_src = _read("portal/routes.py")
        dashboard_html = _read("portal/templates/portal_dashboard.html")
        store_src = _read("store/config_store.py")
        client_detail_html = _read("admin/templates/client_detail.html")

        self.assertIn("get_monthly_token_status", portal_src)
        self.assertIn("Remaining tokens", dashboard_html)
        self.assertIn("remaining_label", dashboard_html)
        self.assertIn("token_limit_monthly", store_src)
        self.assertIn("Monthly token limit", client_detail_html)

    def test_portal_query_limit_kpi_and_notifications_wiring(self):
        """Portal must surface the same monthly query limit used by admin."""
        portal_src = _read("portal/routes.py")
        dashboard_html = _read("portal/templates/portal_dashboard.html")
        chat_html = _read("portal/templates/portal_chat.html")
        portal_base = _read("portal/templates/portal_base.html")
        store_src = _read("store/config_store.py")

        self.assertIn("def _query_limit_status", portal_src)
        self.assertIn('"/api/query-limit-status"', portal_src)
        self.assertIn("get_monthly_query_count", portal_src)
        self.assertIn("Query limit", dashboard_html)
        self.assertIn("query_status.remaining_label", dashboard_html)
        self.assertIn("query-kpi-pill", chat_html)
        self.assertIn("refreshQueryLimitStatus", chat_html)
        self.assertIn("Queries left", chat_html)
        self.assertIn("qb-query-limit-status", portal_base)
        self.assertIn("pollQueryLimit", portal_base)
        self.assertIn("query_limit_monthly", store_src)

    def test_admin_semantic_pending_count_professional_ui(self):
        """Admin semantic review count should be rendered as a review queue chip."""
        base_html = _read("admin/templates/base.html")
        detail_html = _read("admin/templates/client_detail.html")

        self.assertIn("semantic-review-chip", base_html)
        self.assertIn("semantic-review-count", detail_html)
        self.assertIn("pending Semantic Layer review", detail_html)
        self.assertIn("99+", base_html)

    def test_admin_system_provider_toggle_ui(self):
        """System page should show only the selected LLM provider details."""
        system_html = _read("admin/templates/system.html")

        self.assertIn("provider-switch", system_html)
        self.assertIn('data-provider-button="anthropic"', system_html)
        self.assertIn('data-provider-panel="azure_openai"', system_html)
        self.assertIn("syncProvider", system_html)
        self.assertIn("filterModelSelect", system_html)
        self.assertIn('data-provider="{{ provider }}"', system_html)

    def test_portal_semantic_review_live_update_wiring(self):
        """Portal Semantic Layer should update in place after admin review."""
        portal_src = _read("portal/routes.py")
        portal_base = _read("portal/templates/portal_base.html")
        kb_html = _read("portal/templates/portal_kb.html")
        admin_src = _read("admin/routes.py")
        patch_src = _read("core/semantic_kb_patch.py")
        layer_src = _read("core/semantic_layer.py")

        self.assertIn('"/ws/notifications"', portal_src)
        self.assertIn("notify_portal_semantic_feedback_changed", admin_src)
        self.assertIn("semantic_feedback_reviewed", portal_base)
        self.assertIn("qb-semantic-feedback-reviewed", kb_html)
        self.assertIn("field-meaning", kb_html)
        self.assertIn("_approved_table_row_cells", patch_src)
        self.assertIn("_parse_confidence_value", layer_src)

    def test_admin_semantic_review_opens_patched_kb_file(self):
        """After approval, admin should be taken to the actual patched markdown file."""
        admin_src = _read("admin/routes.py")
        admin_kb_html = _read("admin/templates/client_kb.html")
        patch_src = _read("core/semantic_kb_patch.py")

        self.assertIn("locate_kb_file_for_feedback", patch_src)
        self.assertIn('item["patch_file"]', admin_src)
        self.assertIn("patched_file = locate_kb_file_for_feedback", admin_src)
        self.assertIn("feedback_status={status}", admin_src)
        self.assertIn("file={quote(patched_file)}#kb-editor", admin_src)
        self.assertIn("The patched KB file is opened below", admin_kb_html)
        self.assertIn("Open patched KB file", admin_kb_html)
        self.assertIn('id="kb-editor"', admin_kb_html)

    def test_portal_semantic_template_field_scope(self):
        """portal_kb.html must not reference field before the field loop."""
        kb_html = _read("portal/templates/portal_kb.html")
        loop_pos = kb_html.index("{% for field in table.fields %}")
        self.assertNotIn(
            "{{ field", kb_html[:loop_pos],
            "portal_kb.html references field before the field loop, causing Jinja UndefinedError"
        )
        self.assertRegex(
            kb_html,
            r'<tr class="semantic-field-row"[^>]*data-table-fqn="{{ table\.fqn }}"[^>]*data-column-name="{{ field\.column }}"',
        )
        self.assertIn('data-review-target="{{ row_id }}"', kb_html)
        self.assertNotIn("toggleReview('{{ row_id }}')", kb_html)

    def test_portal_semantic_suggest_edit_toggle_syncs_inline_display(self):
        """Suggest edit must still open after search code has touched row display."""
        kb_html = _read("portal/templates/portal_kb.html")
        self.assertIn('onclick="toggleReview(this.dataset.reviewTarget, this)"', kb_html)
        self.assertIn('aria-expanded="false"', kb_html)
        self.assertIn("function _syncReviewRowDisplay(row)", kb_html)
        self.assertIn("row.style.display = row.classList.contains('open') ? 'table-row' : 'none';", kb_html)
        self.assertIn("_syncReviewRowDisplay(row);", kb_html)
        self.assertIn("_syncReviewRowDisplay(rr);", kb_html)
        self.assertNotIn("rr.style.display = match ? rr.classList.contains('open') ? '' : 'none' : 'none'", kb_html)

    def test_user_portal_uses_semantic_layer_label(self):
        """User portal should not expose old Knowledge Base wording."""
        dashboard_html = _read("portal/templates/portal_dashboard.html")
        kb_html = _read("portal/templates/portal_kb.html")

        self.assertIn("Browse Semantic Layer", dashboard_html)
        self.assertIn("View Semantic Layer", dashboard_html)
        self.assertIn("live Semantic Layer directly", kb_html)
        self.assertNotIn("Browse Knowledge Base", dashboard_html)
        self.assertNotIn("View KB docs", dashboard_html)

    def test_kb_file_links_are_url_encoded(self):
        """Admin KB editor links must survive spaces and special chars in filenames."""
        admin_kb_html = _read("admin/templates/client_kb.html")
        portal_src = _read("portal/routes.py")

        self.assertIn("item.patch_file|urlencode", admin_kb_html)
        self.assertIn("f|urlencode", admin_kb_html)
        self.assertIn("from urllib.parse import quote", portal_src)
        self.assertIn("quote(table['schema'] or '')", portal_src)

    def test_metric_formula_builder_ui_wiring(self):
        """Metric Registry must expose the interactive formula-builder fields."""
        html = _read("admin/templates/client_metrics.html")
        for token in [
            "Formula Builder",
            "fn-helper-btn",   # replaces formula-snippet pill toolbar
            "ƒ Functions",
            "formula-editor",
            "formula-status",
            'name="formula_type"',
            'name="result_format"',
            'name="required_columns"',
            'name="allowed_dimensions"',
            'name="example_questions"',
            'name="grain"',
            "NULLIF",
            "Ratio",
        ]:
            self.assertIn(token, html)

    def test_metric_formula_backend_wiring(self):
        """Metric formula metadata must persist and reach SQL generation."""
        routes_src = _read("admin/routes.py")
        store_src = _read("store/config_store.py")
        db_src = _read("store/db.py")
        main_src = _read("core/query_pipeline.py")
        llm_src = _read("core/llm.py")
        init_src = _read("store/__init__.py")

        for field in [
            "formula_type",
            "result_format",
            "required_columns",
            "allowed_dimensions",
            "example_questions",
            "grain",
        ]:
            self.assertIn(field, routes_src)
            self.assertIn(field, store_src)
            self.assertIn(field, db_src)

        self.assertIn("list_metric_formula_context", store_src)
        self.assertIn("list_metric_formula_context", init_src)
        self.assertIn("APPROVED METRIC FORMULAS", main_src)
        self.assertIn("metric_formula_context", main_src)
        self.assertIn("APPROVED METRIC FORMULA RULE", llm_src)
        self.assertIn('(metric.get("formula_type") or "query").lower() != "query"', store_src)


if __name__ == "__main__":
    unittest.main()
