from core.graph_resolver import resolve_for_question


def _graph(status="confirmed"):
    return {
        "entities": [
            {"entity_name": "Order", "table_name": "F_RX_ORDER", "schema_name": "dbo",
             "entity_type": "fact", "display_name": "Prescription Orders", "status": status},
            {"entity_name": "Fill", "table_name": "F_RX_FILL", "schema_name": "dbo",
             "entity_type": "fact", "display_name": "Prescription Fills", "status": status},
        ],
        "relationships": [{
            "id": 1, "from_entity": "Order", "to_entity": "Fill",
            "from_column": "ORDER_ID", "to_column": "ORDER_ID",
            "relationship_type": "one_to_many", "join_type": "LEFT",
            "status": status, "validation_status": "valid", "generated_by": "manual",
        }],
        "properties": [],
    }


def test_suggested_graph_is_review_only_for_normal_sql_generation():
    result = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=_graph("suggested"), intent={"wants_missing_records": True},
    )
    assert result["enabled"] is False
    assert result["review_only"] is True
    assert result["graph_scope"] == "review_only"
    assert len(result["entities"]) == 2


def test_admin_diagnostic_can_explicitly_preview_suggested_graph():
    result = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=_graph("suggested"), intent={"wants_missing_records": True},
        use_suggested=True,
    )
    assert result["enabled"] is True
    assert result["graph_scope"] == "suggested_fallback"


def test_anti_join_anchors_records_named_before_without():
    result = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=_graph(), intent={"wants_missing_records": True},
    )
    assert result["anchor"] == "Order"
    assert result["join_skeleton"].startswith("FROM [dbo].[F_RX_ORDER]")
    assert "LEFT  JOIN [dbo].[F_RX_FILL]" in result["join_skeleton"]
