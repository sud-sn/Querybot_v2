from core.graph_resolver import _confirmed_subgraph, resolve_for_question


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


def test_confirmed_subgraph_excludes_pending_and_rejected_rows_at_every_level():
    graph = {
        "entities": [
            {"entity_name": "Confirmed", "status": "confirmed"},
            {"entity_name": "Legacy"},
            {"entity_name": "Pending", "status": "suggested"},
            {"entity_name": "Rejected", "status": "rejected"},
        ],
        "relationships": [
            {"id": 1, "from_entity": "Confirmed", "to_entity": "Legacy",
             "status": "confirmed"},
            {"id": 2, "from_entity": "Confirmed", "to_entity": "Legacy",
             "status": "suggested"},
            {"id": 3, "from_entity": "Confirmed", "to_entity": "Legacy",
             "status": "rejected"},
            {"id": 4, "from_entity": "Confirmed", "to_entity": "Pending",
             "status": "confirmed"},
        ],
        "properties": [
            {"entity_name": "Confirmed", "column_name": "APPROVED",
             "status": "confirmed"},
            {"entity_name": "Legacy", "column_name": "LEGACY"},
            {"entity_name": "Confirmed", "column_name": "PENDING",
             "status": "suggested"},
            {"entity_name": "Confirmed", "column_name": "REJECTED",
             "status": "rejected"},
            {"entity_name": "Pending", "column_name": "APPROVED_ON_PENDING_ENTITY",
             "status": "confirmed"},
        ],
    }

    confirmed = _confirmed_subgraph(graph)

    assert {e["entity_name"] for e in confirmed["entities"]} == {
        "Confirmed", "Legacy",
    }
    assert [r["id"] for r in confirmed["relationships"]] == [1]
    assert {(p["entity_name"], p["column_name"]) for p in confirmed["properties"]} == {
        ("Confirmed", "APPROVED"), ("Legacy", "LEGACY"),
    }


def test_rejected_relationship_is_never_used_even_in_admin_preview():
    graph = _graph("confirmed")
    graph["relationships"][0]["status"] = "rejected"

    normal = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=graph, intent={"wants_missing_records": True},
    )
    preview = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=graph, intent={"wants_missing_records": True},
        use_suggested=True,
    )

    assert normal["enabled"] is False
    assert preview["enabled"] is False


def test_normal_resolution_returns_only_confirmed_properties():
    graph = _graph("confirmed")
    graph["properties"] = [
        {"entity_name": "Order", "column_name": "ORDER_ID", "status": "confirmed"},
        {"entity_name": "Order", "column_name": "PENDING_ALIAS", "status": "suggested"},
        {"entity_name": "Fill", "column_name": "REJECTED_ALIAS", "status": "rejected"},
    ]

    result = resolve_for_question(
        "prescription orders without fills", "tenant", "azure_sql",
        graph=graph, intent={"wants_missing_records": True},
    )

    assert result["enabled"] is True
    assert [p["column_name"] for p in result["properties"]] == ["ORDER_ID"]
