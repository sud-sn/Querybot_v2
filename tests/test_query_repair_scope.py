from core.query_pipeline import (
    _entity_field_unavailable_reason,
    _unknown_column_is_cross_schema,
)


def test_pipeline_promotes_entity_conflict_to_terminal_validation_code():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
        encoding="utf-8"
    )
    guard = source.index('code = "entity_field_unavailable"')
    retry = source.index("retryable = (not ok")

    assert guard < retry
    assert "_entity_field_reason = _entity_field_unavailable_reason(" in source
    assert "_column_validation.errors" in source


def test_unknown_column_same_schema_does_not_crash_or_cross_scope():
    reason = (
        "Column STATE_CODE is unknown. Exact column exists on: "
        "CHATBOT_DB.PHARMA_LAB.D_PATIENT, PHARMA_LAB.D_PHARMACY"
    )

    assert _unknown_column_is_cross_schema(reason, "PHARMA_LAB") is False


def test_unknown_column_only_in_another_schema_is_cross_scope():
    reason = (
        "Column STATE_CODE is unknown. EXACT COLUMN EXISTS ON: "
        "CHATBOT_DB.CRM.D_PRESCRIBER"
    )

    assert _unknown_column_is_cross_schema(reason, "PHARMA_LAB") is True


def test_unknown_column_without_location_evidence_is_not_cross_scope():
    assert _unknown_column_is_cross_schema(
        "Column STATE_CODE is unknown.", "PHARMA_LAB"
    ) is False


def test_entity_field_conflict_explains_and_deduplicates_alternatives():
    reason = _entity_field_unavailable_reason([{
        "code": "unknown_column",
        "table": "PHARMA_LAB.D_PRESCRIBER",
        "column": "STATE_CODE",
        "suggestions": [],
        "candidate_tables": [
            "CHATBOT_DB.PHARMA_LAB.D_PATIENT",
            "PHARMA_LAB.D_PATIENT",
            "PHARMA_LAB.D_PHARMACY",
        ],
    }])

    assert "STATE_CODE is not available for prescriber (D_PRESCRIBER)" in reason
    assert reason.count("patient (D_PATIENT)") == 1
    assert "pharmacy (D_PHARMACY)" in reason
    assert "change the meaning" in reason


def test_entity_field_conflict_is_generic_for_other_clients():
    reason = _entity_field_unavailable_reason([{
        "code": "unknown_column",
        "table": "ERP.D_CUSTOMER_ACCOUNT",
        "column": "REGION_NAME",
        "suggestions": [],
        "candidate_tables": ["ERP.D_SHIP_TO", "CRM.D_CONTACT"],
    }])

    assert "customer account (D_CUSTOMER_ACCOUNT)" in reason
    assert "ship to (D_SHIP_TO)" in reason
    assert "contact (D_CONTACT)" in reason


def test_entity_field_conflict_does_not_block_safe_same_table_spelling_repair():
    reason = _entity_field_unavailable_reason([{
        "code": "unknown_column",
        "table": "ERP.D_CUSTOMER",
        "column": "CUSTMER_NAME",
        "suggestions": ["CUSTOMER_NAME"],
        "candidate_tables": [],
    }])

    assert reason == ""


def test_entity_field_conflict_does_not_block_unqualified_join_repair():
    reason = _entity_field_unavailable_reason([{
        "code": "unknown_column",
        "table": "",
        "column": "REGION_NAME",
        "suggestions": [],
        "candidate_tables": ["ERP.D_REGION"],
    }])

    assert reason == ""
