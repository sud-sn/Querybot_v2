from core.count_target_resolver import (
    count_target_clarification_options,
    resolve_count_target,
)


def _model(fields, *, grain="one row per customer order line"):
    return {
        "tables": [{
            "qualified_name": "TENANT_X.F_ORDER_LINE",
            "schema": "TENANT_X",
            "type": "fact",
            "entity": "customer order",
            "grain": grain,
            "fields": fields,
        }]
    }


def _field(column, expanded, **extra):
    return {
        "column": column,
        "expanded_name": expanded,
        "role": extra.pop("role", "identifier"),
        "aggregation": extra.pop("aggregation", "identifier"),
        "confidence": extra.pop("confidence", 90),
        "status": extra.pop("status", "generated"),
        **extra,
    }


def test_header_business_number_beats_line_and_surrogate_identifiers():
    model = _model([
        _field(
            "ORDER_NUMBER", "Sales Order Number", status="approved",
            approved_meaning="Business number shared by every line of one sales order.",
            business_candidates=["order", "sales order number"],
        ),
        _field(
            "ORDER_LINE_NUMBER", "Sales Order Line Number",
            approved_meaning="Sequence number for one item line within an order.",
            business_candidates=["order line"],
        ),
        _field(
            "ORDER_SK", "Order Key", role="dimension_key",
            aggregation="identifier", business_candidates=["order key"],
        ),
        _field(
            "CUSTOMER_SK", "Customer Key", role="dimension_key",
            aggregation="identifier", business_candidates=["customer key"],
        ),
        _field(
            "ORDER_AMOUNT", "Order Amount", role="measure",
            aggregation="additive", business_candidates=["order value"],
        ),
    ])

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
    )

    assert result["status"] == "selected"
    assert result["selected"]["column"] == "ORDER_NUMBER"
    assert not result["selected"]["line_level"]


def test_close_business_identifiers_require_business_facing_clarification():
    model = _model([
        _field(
            "EXTERNAL_ORDER_NUMBER", "Customer Order Number", status="approved",
            approved_meaning="Order reference supplied by the customer.",
            business_candidates=["order number"],
        ),
        _field(
            "INTERNAL_ORDER_NUMBER", "Internal Sales Order Number", status="approved",
            approved_meaning="Order reference assigned by the ERP.",
            business_candidates=["order number"],
        ),
    ], grain="one row per sales order")

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
    )
    options = count_target_clarification_options(result)

    assert result["status"] == "ambiguous"
    assert len(options) == 2
    assert all("TENANT_X" not in option["label"] for option in options)
    assert all("_" not in option["label"] for option in options)
    assert {option["target_column"] for option in options} == {
        "EXTERNAL_ORDER_NUMBER", "INTERNAL_ORDER_NUMBER",
    }


def test_confirmed_business_choice_resolves_exact_hidden_target():
    model = _model([
        _field(
            "EXTERNAL_ORDER_NUMBER", "Customer Order Number", status="approved",
            approved_meaning="Order reference supplied by the customer.",
            business_candidates=["order number"],
        ),
        _field(
            "INTERNAL_ORDER_NUMBER", "Internal Sales Order Number", status="approved",
            approved_meaning="Order reference assigned by the ERP.",
            business_candidates=["order number"],
        ),
    ], grain="one row per sales order")
    confirmed = {
        "target_table": "TENANT_X.F_ORDER_LINE",
        "target_column": "INTERNAL_ORDER_NUMBER",
    }

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
        confirmed_option=confirmed,
    )

    assert result["status"] == "selected"
    assert result["reason"] == "user confirmed business meaning"
    assert result["selected"]["column"] == "INTERNAL_ORDER_NUMBER"


def test_missing_identifier_fails_closed_instead_of_counting_arbitrary_field():
    model = _model([
        _field(
            "ORDER_AMOUNT", "Order Amount", role="measure",
            aggregation="additive", business_candidates=["order value"],
        ),
        _field(
            "CUSTOMER_SK", "Customer Key", role="dimension_key",
            aggregation="identifier", business_candidates=["customer key"],
        ),
    ])

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
    )

    assert result["status"] == "missing"
    assert result["selected"] == {}


def test_opaque_erp_column_uses_expanded_business_metadata():
    model = _model([
        _field(
            "ORNO", "Order Number", status="approved",
            approved_meaning="Business identifier for one customer order.",
            business_candidates=["sales order", "customer order number"],
        ),
    ])

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
    )

    assert result["status"] == "selected"
    assert result["selected"]["column"] == "ORNO"


def test_client_approved_alias_metadata_resolves_an_opaque_identifier():
    model = _model([{
        "column": "DOCREF",
        "role": "attribute",
        "aggregation": "identifier",
        "business_name": "Customer Order Reference",
        "business_meaning": "Stable reference shared by all lines of one order.",
        "approved_synonyms": ["order number", "sales document"],
        "confidence": 95,
        "status": "approved",
    }])

    result = resolve_count_target(
        "order", model,
        source_scope={"selected_fact": "TENANT_X.F_ORDER_LINE"},
    )

    assert result["status"] == "selected"
    assert result["selected"]["column"] == "DOCREF"
    assert result["selected"]["business_name"] == "Customer Order Reference"
