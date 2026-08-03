from core.query_pipeline import _unknown_column_is_cross_schema


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
