from pathlib import Path


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
    assert "Shift + Enter for a new line" in CHAT


def test_ranked_date_choices_keep_a_custom_business_date_input_below_them():
    assert "options.some((opt) => Boolean(opt.allow_free_text))" in CHAT
    assert "Search by business date name" in CHAT
    assert "For example: invoice date" in CHAT
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
    assert "That result is no longer available for analysis" in action_block


def test_result_actions_acknowledge_complete_download_and_timeout_visibly():
    assert "assistant_action_ack" in WEBHOOKS
    assert "action_id: actionId" in CHAT
    assert "function finishAnalysisAction" in CHAT
    assert "This action did not finish in time" in CHAT
    assert "msg.type === 'assistant_export'" in CHAT
    assert "new Blob([msg.content || '']" in CHAT


def test_only_live_server_clarification_is_actionable_after_restore():
    assert "stale:true" in CHAT
    assert "if (opts.stale)" in CHAT
    assert "reconnect_pending = get_pending(" in WEBHOOKS
    assert "pending_id=str(reconnect_meta.get" in WEBHOOKS


def test_outbound_messages_expose_delivery_and_manual_recovery_states():
    assert "function _setUserMessageState" in CHAT
    assert "function _retryUserMessage" in CHAT
    assert "function _editUserMessage" in CHAT
    assert "Interrupted · Retry when connected" in CHAT
    on_open = CHAT.split("ws.onopen =", 1)[1].split("ws.onmessage =", 1)[0]
    assert "_retryUserMessage(" not in on_open
    assert '.msg-user[data-delivery-state="interrupted"] .message-retry' in CSS


def test_offline_composer_preserves_draft_instead_of_silently_dropping_send():
    assert "DRAFT_STORAGE_KEY" in CHAT
    assert "function _saveComposerDraft" in CHAT
    assert "function _restoreComposerDraft" in CHAT
    assert "You are offline. Your message is saved" in CHAT
    assert "_clearComposerDraft();" in CHAT


def test_governed_recovery_statuses_have_human_readable_progress_copy():
    assert "recovering_sql:" in CHAT
    assert "reusing_sql:" in CHAT
    assert "retrying_query:" in CHAT
