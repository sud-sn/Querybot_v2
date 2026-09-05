from pathlib import Path
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
CHAT = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "css" / "chat_workspace.css").read_text(encoding="utf-8")
WEB_ADAPTER = (ROOT / "gateway" / "web_adapter.py").read_text(encoding="utf-8")
WEBHOOKS = (ROOT / "gateway" / "webhooks.py").read_text(encoding="utf-8")


def test_open_ended_clarification_has_a_free_text_reply_path():
    assert "if (!options.length)" in CHAT
    assert "clarification-freeform" in CHAT
    assert "payload.text = text" in CHAT
    assert "type:'clarification_response'" in CHAT
    assert "Shift + Enter for a new line" in _catalogue()["ui.chat.clar.help"]


def test_ranked_date_choices_keep_a_custom_business_date_input_below_them():
    assert "options.some((opt) => Boolean(opt.allow_free_text))" in CHAT
    assert "Search by business date name" in _catalogue()["ui.chat.clar.search_dates"]
    assert "For example: invoice date" in _catalogue()["ui.chat.clar.date_example"]
    assert "business_suggestions" in CHAT
    assert "clarification-business-suggestion" in CHAT
    assert ".clarification-business-suggestion" in CSS


def test_date_suggestions_do_not_render_physical_schema_fields():
    clarification_block = CHAT.split("function renderClarificationPrompt", 1)[1].split(
        "function _submitClarification", 1
    )[0]
    assert "fact_column" not in clarification_block
    assert "fact_table" not in clarification_block

    adapter_block = WEB_ADAPTER.split(
        "async def send_clarification_prompt", 1
    )[1].split("async def upload_file", 1)[0]
    assert "_public_clarification_options(options)" in adapter_block
    assert '"business_suggestions"' in WEB_ADAPTER
    assert '"fact_column"' not in adapter_block
    assert '"fact_table"' not in adapter_block

    from gateway.web_adapter import _public_clarification_options

    public = _public_clarification_options([{
        "id": "date_role_1",
        "label": "Invoice Date",
        "value": "Invoice Date",
        "allow_free_text": True,
        "business_suggestions": ["Invoice Date", "Order Date"],
        "fact_table": "SALES.F_INVOICE",
        "fact_column": "INVOICE_DATE_KEY",
        "dimension_table": "SALES.D_DATE",
    }])
    assert public == [{
        "id": "date_role_1",
        "label": "Invoice Date",
        "value": "Invoice Date",
        "allow_free_text": True,
        "business_suggestions": ["Invoice Date", "Order Date"],
    }]


def test_portal_date_reply_uses_full_server_side_choices_and_preserves_binding():
    response_block = WEBHOOKS.split(
        'if msg_type == "clarification_response":', 1
    )[1].split("await handle_query(", 1)[0]
    assert 'cmeta.get("all_options") or opts' in response_block
    assert "resolve_date_option_text" in response_block
    assert '"_clarification_selected_option"' in response_block


def test_portal_count_reply_preserves_hidden_governed_target():
    response_block = WEBHOOKS.split(
        'if msg_type == "clarification_response":', 1
    )[1].split("await handle_query(", 1)[0]
    assert '{"metric_date_context", "count_target"}' in response_block
    assert 'cmeta.get("source")' in response_block
    assert '"_clarification_selected_option"' in response_block


def test_result_action_targets_the_clicked_governed_snapshot():
    assert "function sendAnalysisAction(action, context, label, resultId)" in CHAT
    assert "result_id: resultId || ''" in CHAT
    assert "trust.result_id || msg.data?.result_id || ''" in CHAT
    action_block = WEBHOOKS.split("if action:", 1)[1].split(
        "if not text:", 1
    )[0]
    assert "requested_result_id" in action_block
    assert "result_cache.get_snapshot(" in action_block
    assert "adopt_cached_snapshot(adapter, action_snapshot)" in action_block
    assert '_t("reply.result.expired_analysis")' in action_block
    from core import i18n
    assert "no longer available for analysis" in i18n.t(
        "reply.result.expired_analysis", lang="en")
    assert "const actionResultId = trust.result_id || msg.data?.result_id || '';" in CHAT
    assert "const nextActions = actionResultId &&" in CHAT


def test_durable_thread_result_restore_is_user_thread_scoped_and_exact():
    from core.result_cache import result_cache
    from gateway.web_adapter import WebAdapter
    from gateway.webhooks import _restore_durable_thread_result

    adapter = WebAdapter(
        AsyncMock(), "tenant-a", "web_17", thread_id="thread-a",
        portal_user_id=17,
    )
    trace = {
        "question_id": "result-123",
        "question_text_sanitized": (
            "Revenue by month. Clarification for the same request: Invoice Date. "
            "Use this clarification to interpret the original request; do not treat "
            "it as a separate question."
        ),
        "generated_sql": "select invoice_month, sum(revenue) from governed_sales",
        "result_rows": '[{"INVOICE_MONTH":"2026-01","REVENUE":125}]',
        "status": "success",
        "contract_version": "contract-7",
    }
    try:
        trace["portal_user_id"] = 17
        trace["session_id"] = "tenant-a:web_17:thread:thread-a"
        with patch(
            "gateway.webhooks.store.get_answer_trace_by_question_id", return_value=trace
        ) as get_trace, patch(
            "gateway.webhooks.get_client_db",
            return_value={"id": 41, "db_type": "azure_sql"},
        ):
            snapshot = _restore_durable_thread_result(
                "tenant-a", 17, adapter, result_id="result-123"
            )

        assert snapshot["result_id"] == "result-123"
        assert snapshot["rows"] == [{"INVOICE_MONTH": "2026-01", "REVENUE": 125}]
        assert snapshot["question"] == "Revenue by month."
        assert adapter.last_result_id == "result-123"
        get_trace.assert_called_once_with(
            "tenant-a", "result-123",
        )
    finally:
        result_cache.clear(adapter.session_id)


def test_durable_thread_result_restore_rejects_errors_and_empty_rows():
    from core.result_cache import result_cache
    from gateway.web_adapter import WebAdapter
    from gateway.webhooks import _restore_durable_thread_result

    adapter = WebAdapter(
        AsyncMock(), "tenant-b", "web_22", thread_id="thread-b",
        portal_user_id=22,
    )
    traces = [
        {
            "question_id": "failed-result",
            "question_text_sanitized": "Failed",
            "generated_sql": "select 1",
            "result_rows": '[{"VALUE":1}]',
            "status": "error",
        },
        {
            "question_id": "empty-result",
            "question_text_sanitized": "Empty",
            "generated_sql": "select 1 where 1=0",
            "result_rows": "[]",
            "status": "success",
        },
    ]
    try:
        with patch(
            "gateway.webhooks.store.list_answer_traces", return_value=traces
        ):
            snapshot = _restore_durable_thread_result("tenant-b", 22, adapter)
        assert snapshot == {}
        assert not result_cache.has_result(adapter.session_id)
    finally:
        result_cache.clear(adapter.session_id)


def test_durable_exact_result_restore_rejects_another_user_or_thread():
    from core.result_cache import result_cache
    from gateway.web_adapter import WebAdapter
    from gateway.webhooks import _restore_durable_thread_result

    adapter = WebAdapter(
        AsyncMock(), "tenant-c", "web_31", thread_id="owned-thread",
        portal_user_id=31,
    )
    foreign_trace = {
        "question_id": "foreign-result",
        "portal_user_id": 99,
        "session_id": "tenant-c:web_99:thread:other-thread",
        "question_text_sanitized": "Revenue",
        "generated_sql": "select revenue from sales",
        "result_rows": '[{"REVENUE":500}]',
        "status": "success",
    }
    try:
        with patch(
            "gateway.webhooks.store.get_answer_trace_by_question_id",
            return_value=foreign_trace,
        ):
            snapshot = _restore_durable_thread_result(
                "tenant-c", 31, adapter, result_id="foreign-result"
            )
        assert snapshot == {}
        assert not result_cache.has_result(adapter.session_id)
    finally:
        result_cache.clear(adapter.session_id)


def _catalogue():
    """The message catalogue the chat page ships to the browser.

    Several strings below moved out of the template and into core/i18n.py. A
    source-text assertion would now pass against a page that resolves the id to
    nothing, so these read what the browser actually receives.
    """
    from tests.chat_render import render, catalogue
    return catalogue(render(lang="en"))


def test_result_actions_acknowledge_complete_download_and_timeout_visibly():
    assert "assistant_action_ack" in WEBHOOKS
    assert "action_id: actionId" in CHAT
    assert "function finishAnalysisAction" in CHAT
    # A catalogue id in the source now, so this asserts the sentence still
    # reaches the browser.
    assert "This action did not finish in time" in \
        _catalogue()["ui.chat.system.action_timed_out"]
    assert "msg.type === 'assistant_export'" in CHAT
    assert "new Blob([msg.content || '']" in CHAT
    assert "_bound_action_payload(_dd_result)" in WEBHOOKS
    assert "_bound_action_payload(_cp_result)" in WEBHOOKS
    assert "_bound_action_payload(insight)" in WEBHOOKS
    assert 'resolved.setdefault("action_id", action_id)' in WEBHOOKS
    assert _catalogue()["ui.chat.err.action_accepted"] == \
        "The governed action was accepted and is being completed."


def test_only_live_server_clarification_is_actionable_after_restore():
    assert "stale:true" in CHAT
    assert "if (opts.stale)" in CHAT
    assert "reconnect_pending = get_pending(" in WEBHOOKS
    assert "pending_id=str(reconnect_meta.get" in WEBHOOKS


def test_outbound_messages_expose_delivery_and_manual_recovery_states():
    assert "function _setUserMessageState" in CHAT
    assert "function _retryUserMessage" in CHAT
    assert "function _editUserMessage" in CHAT
    # Now a catalogue id, so this asserts the string still REACHES the browser
    # rather than that it is spelled out in the template -- which would pass
    # for a page that resolves the id to nothing.
    from tests.chat_render import render as _render_chat, catalogue
    assert catalogue(_render_chat(lang="en"))["ui.chat.interrupted_retry"] == \
        "Interrupted · Retry when connected"
    on_open = CHAT.split("ws.onopen =", 1)[1].split("ws.onmessage =", 1)[0]
    assert "_retryUserMessage(" not in on_open
    assert '.msg-user[data-delivery-state="interrupted"] .message-retry' in CSS


def test_offline_composer_preserves_draft_instead_of_silently_dropping_send():
    assert "DRAFT_STORAGE_KEY" in CHAT
    assert "function _saveComposerDraft" in CHAT
    assert "function _restoreComposerDraft" in CHAT
    assert "You are offline. Your message is saved" in \
        _catalogue()["ui.chat.toast.offline_saved"]
    assert "_clearComposerDraft();" in CHAT


def test_governed_recovery_statuses_have_human_readable_progress_copy():
    """Executed rather than read: STATUS_FALLBACK is built from the catalogue
    now, so the question is whether these three stages RESOLVE to a label and a
    detail, not whether their keys appear in the source."""
    from tests.chat_js import run as run_chat_js

    out = run_chat_js(
        "JSON.stringify({fallback: STATUS_FALLBACK});",
        consts=["STATUS_FALLBACK"],
    )["fallback"]
    for stage in ("recovering_sql", "reusing_sql", "retrying_query"):
        label, detail = out[stage]
        assert label and not label.startswith("stage."), stage
        assert detail and not detail.startswith("stage."), stage
