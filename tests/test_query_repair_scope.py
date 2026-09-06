from core.query_pipeline import (
    _entity_field_unavailable_reason,
    _unknown_column_is_cross_schema,
)


def test_an_entity_conflict_becomes_terminal_rather_than_a_repair():
    """Executed, not grepped.

    Moving an exact column from one table to another changes the business
    entity being asked about -- prescriber state for pharmacy state -- so it
    must stop the run with an explanation rather than continue into the LLM
    repair budget. This used to be a source scan asserting the guard appeared
    before "retryable = (", because the code was inline in a 4,000-line
    function and could not be called. It is in core.sql_attempt now, so the
    behaviour itself is the assertion.
    """
    from unittest.mock import patch

    from core.sql_attempt import ValidationScope, validate_with_repairs

    scope = ValidationScope(
        known_tables={"PHARMA_LAB.D_PRESCRIBER"}, db_type="azure_sql",
        allowed_tables={"PHARMA_LAB.D_PRESCRIBER"},
        table_columns={"PHARMA_LAB.D_PRESCRIBER": {"PRESCRIBER_ID": "int"}},
    )

    class _Detail:
        ok = False
        errors = [{
            "code": "unknown_column",
            "table": "PHARMA_LAB.D_PRESCRIBER",
            "column": "STATE_CODE",
            "suggestions": [],
            "candidate_tables": ["PHARMA_LAB.D_PATIENT", "PHARMA_LAB.D_PHARMACY"],
        }]

    with (
        patch("core.validator.validate_sql",
              return_value=(False, "Column STATE_CODE is unknown.", "unknown_column")),
        patch("core.validator.validate_sql_detailed", return_value=_Detail()),
        # No unambiguous same-table fix exists, which is what makes this a
        # conflict rather than a spelling repair.
        patch("core.validator.repair_unambiguous_unknown_columns", return_value=""),
    ):
        attempt = validate_with_repairs(
            "SELECT STATE_CODE FROM PHARMA_LAB.D_PRESCRIBER", scope)

    assert attempt.ok is False
    assert attempt.code == "entity_field_unavailable"
    assert "STATE_CODE is not available for prescriber (D_PRESCRIBER)" in attempt.reason
    assert "change the meaning" in attempt.reason
    assert attempt.repairs == ()


def test_an_unambiguous_column_is_repaired_rather_than_made_terminal():
    """The other side of the same decision.

    The negative above proves nothing unless a genuine spelling repair still
    goes through -- otherwise "always terminal" would pass it.
    """
    from unittest.mock import patch

    from core.sql_attempt import ValidationScope, validate_with_repairs

    scope = ValidationScope(
        known_tables={"S.T"}, db_type="azure_sql", allowed_tables={"S.T"},
        table_columns={"S.T": {"STATE_CODE": "varchar"}},
    )
    fixed = "SELECT STATE_CODE FROM S.T"

    class _Bad:
        ok = False
        errors = [{"code": "unknown_column", "table": "S.T",
                   "column": "STATE_COD", "suggestions": ["STATE_CODE"],
                   "candidate_tables": []}]

    class _Good:
        ok = True
        errors = []

    with (
        patch("core.validator.validate_sql",
              return_value=(False, "Column STATE_COD is unknown.", "unknown_column")),
        patch("core.validator.validate_sql_detailed", side_effect=[_Bad(), _Good()]),
        patch("core.validator.repair_unambiguous_unknown_columns", return_value=fixed),
    ):
        attempt = validate_with_repairs("SELECT STATE_COD FROM S.T", scope)

    assert attempt.ok is True
    assert attempt.sql == fixed
    assert "unknown_column_repair" in attempt.repairs


def test_validation_runs_before_the_llm_retry_block():
    """The one genuinely positional half of the old scan.

    Whether the deterministic repairs run BEFORE the pipeline decides a query
    is retryable is an ordering fact inside a function that needs a live
    websocket, a warehouse and a compliance context to execute. The check is
    now one line instead of three, because the whole sequence is one call.
    """
    import inspect

    import core.query_pipeline as qp

    source = inspect.getsource(qp._handle_query_impl)
    assert source.index("_attempt = await run_attempt(") < source.index("retryable = (")


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
