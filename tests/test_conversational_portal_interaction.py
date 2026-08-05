from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAT = (ROOT / "portal" / "templates" / "portal_chat.html").read_text(encoding="utf-8")
CSS = (ROOT / "static" / "css" / "chat_workspace.css").read_text(encoding="utf-8")


def test_open_ended_clarification_has_a_free_text_reply_path():
    assert "if (!options.length)" in CHAT
    assert "clarification-freeform" in CHAT
    assert "payload.text = text" in CHAT
    assert "type:'clarification_response'" in CHAT
    assert "Shift + Enter for a new line" in CHAT


def test_ranked_date_choices_keep_a_custom_business_date_input_below_them():
    assert "options.some((opt) => Boolean(opt.allow_free_text))" in CHAT
    assert "None of these? Enter the business date to use" in CHAT
    assert "For example: accounting date" in CHAT


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
