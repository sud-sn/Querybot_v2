from contextlib import contextmanager

from core import graph_autopopulate
import core.schema as schema_module
from core.relationship_identity import physical_relationship_signature


def _authoritative_graph():
    return {
        "entities": [
            {
                "entity_name": "F_SALES", "table_name": "F_SALES",
                "schema_name": "dbo", "pk_column": "", "entity_type": "fact",
                "status": "confirmed", "generated_by": "db_schema",
                "confidence_score": 100, "reason": "trusted FK endpoint",
            },
            {
                "entity_name": "D_CUSTOMER", "table_name": "D_CUSTOMER",
                "schema_name": "dbo", "pk_column": "CUSTOMER_SK",
                "entity_type": "dimension", "status": "confirmed",
                "generated_by": "db_schema", "confidence_score": 100,
                "reason": "trusted FK endpoint",
            },
        ],
        "relationships": [{
            "from_entity": "F_SALES", "to_entity": "D_CUSTOMER",
            "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
            "relationship_key": "DBFK:FK_SALES_CUSTOMER",
            "relationship_type": "many_to_one", "join_type": "INNER",
            "join_conditions": [], "status": "confirmed",
            "generated_by": "db_fk", "source_enforced": True,
            "confidence_score": 100,
        }],
    }


class _FakeConnection:
    def __init__(self, state):
        self.state = state

    def execute(self, sql, params=()):
        if "UPDATE entity_graph" in sql:
            self.state["promotions"].append(params[-1])
        return self

    def executemany(self, sql, rows):
        if "UPDATE entity_relationships SET is_active=0" in sql:
            self.state["retired"].extend(row[1] for row in rows)
        return self


def _install_fake_store(monkeypatch, relationships):
    state = {"promotions": [], "retired": [], "upserts": []}
    entities = [
        {"entity_name": "F_SALES", "status": "suggested", "generated_by": "heuristic"},
        {"entity_name": "D_CUSTOMER", "status": "suggested", "generated_by": "heuristic"},
    ]

    monkeypatch.setattr(
        graph_autopopulate.store, "list_entities",
        lambda account_id, active_only=False: list(entities),
    )
    monkeypatch.setattr(
        graph_autopopulate.store, "list_relationships",
        lambda account_id, active_only=True: [
            row for row in relationships if row["id"] not in state["retired"]
        ],
    )

    @contextmanager
    def fake_db():
        yield _FakeConnection(state)

    monkeypatch.setattr(graph_autopopulate.store, "get_db", fake_db)
    monkeypatch.setattr(
        graph_autopopulate.store, "save_entity",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("entity should be promoted")),
    )
    monkeypatch.setattr(
        graph_autopopulate.store, "upsert_relationship_by_identity",
        lambda **kwargs: state["upserts"].append(kwargs) or 99,
    )
    monkeypatch.setattr(schema_module, "build_entity_graph_from_schema", lambda _: _authoritative_graph())
    return state


def test_trusted_fk_promotes_endpoints_and_retires_exact_suggestion(monkeypatch):
    stale = {
        "id": 7, "from_entity": "F_SALES", "to_entity": "D_CUSTOMER",
        "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
        "relationship_key": "HEURISTIC:F_SALES:D_CUSTOMER:CUSTOMER_SK",
        "join_conditions": "[]", "status": "suggested", "generated_by": "heuristic",
    }
    state = _install_fake_store(monkeypatch, [stale])

    graph_autopopulate.auto_populate_from_schema("tenant", "unused")

    assert set(state["promotions"]) == {"F_SALES", "D_CUSTOMER"}
    assert state["retired"] == [7]
    assert state["upserts"][0]["relationship_key"] == "DBFK:FK_SALES_CUSTOMER"
    assert state["upserts"][0]["status"] == "confirmed"


def test_trusted_fk_does_not_duplicate_manual_exact_relationship(monkeypatch):
    manual = {
        "id": 8, "from_entity": "F_SALES", "to_entity": "D_CUSTOMER",
        "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
        "relationship_key": "MANUAL:CUSTOMER_ROLE", "join_conditions": [],
        "status": "confirmed", "generated_by": "manual",
    }
    state = _install_fake_store(monkeypatch, [manual])

    graph_autopopulate.auto_populate_from_schema("tenant", "unused")

    assert state["retired"] == []
    assert state["upserts"] == []


def test_physical_identity_reconciles_reversed_edges_but_preserves_roles():
    forward = {
        "from_entity": "F_SALES", "to_entity": "D_CUSTOMER",
        "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
    }
    reversed_edge = {
        "from_entity": "D_CUSTOMER", "to_entity": "F_SALES",
        "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
    }
    billing_role = {
        "from_entity": "F_SALES", "to_entity": "D_BILL_TO_CUSTOMER",
        "from_column": "CUSTOMER_SK", "to_column": "CUSTOMER_SK",
    }

    assert physical_relationship_signature(forward) == physical_relationship_signature(reversed_edge)
    assert physical_relationship_signature(forward) != physical_relationship_signature(billing_role)
