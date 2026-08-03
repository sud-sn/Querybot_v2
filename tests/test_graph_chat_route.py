"""
admin/routes.py::graph_api_chat -- the conversational front door onto the
entity graph. Real SQLite (store.init_db()); admin._is_auth is patched to
bypass the ops-panel cookie check (this codebase's established pattern --
see tests/test_admin_reports.py); the LLM call (core.llm.llm_complete) is
patched so no network call happens and the response is fully controlled.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import store
from admin import routes


def _arun(coro):
    return asyncio.run(coro)


def _fake_request(message: str) -> MagicMock:
    req = MagicMock()
    req.json = AsyncMock(return_value={"message": message})
    return req


class GraphChatRouteTests(unittest.TestCase):
    def setUp(self):
        store.init_db()
        # resolve_provider is mocked directly (rather than seeding a real
        # system API key via store.set_system) to avoid a pre-existing,
        # already-documented full-suite-only flakiness in store.crypto
        # (module-reimport duplicate Fernet-key identity across test files
        # -- unrelated to this route, see project memory
        # project_compliance_llm_boundary.md's "Suite gotcha" note).
        # graph_api_chat does `from core.llm import ... resolve_provider`
        # locally at call time, so the patch target is core.llm itself, not
        # a module-level admin.routes attribute.
        self._resolve_provider_patch = patch(
            "core.llm.resolve_provider",
            return_value=("anthropic", "claude-test", "test-key", {}),
        )
        self._resolve_provider_patch.start()
        self.account_id = f"acct-graphchat-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")

        self._tmp = tempfile.TemporaryDirectory()
        schema_dir = Path(self._tmp.name) / "schema"
        schema_dir.mkdir()
        schema = {
            "DBO.F_ORDERS": {
                "columns": [
                    {"name": "ORDER_ID", "type": "int"},
                    {"name": "CUSTOMER_ID", "type": "int"},
                    {"name": "ORDER_DATE", "type": "date"},
                ],
                "pk_columns": ["ORDER_ID"],
            },
            "DBO.DIM_CUSTOMER": {
                "columns": [
                    {"name": "CUSTOMER_ID", "type": "int"},
                    {"name": "CUSTOMER_NAME", "type": "varchar"},
                ],
                "pk_columns": ["CUSTOMER_ID"],
            },
        }
        (schema_dir / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        self.schema_dir = schema_dir
        store.update_client_state(self.account_id, "active", {"schema_dir": str(schema_dir)})

    def tearDown(self):
        self._resolve_provider_patch.stop()
        self._tmp.cleanup()
        with store.get_db() as conn:
            for table in ("entity_properties", "entity_relationships", "entity_graph"):
                conn.execute(f"DELETE FROM {table} WHERE account_id=?", (self.account_id,))
            conn.execute("DELETE FROM client WHERE account_id=?", (self.account_id,))

    def _mock_llm(self, response: dict):
        return patch(
            "core.llm.llm_complete",
            new=AsyncMock(return_value=(json.dumps(response), 10, 10)),
        )

    def test_rejects_unauthenticated(self):
        with patch.object(routes, "_is_auth", return_value=False):
            with self.assertRaises(Exception):
                _arun(routes.graph_api_chat(_fake_request("map orders"), self.account_id))

    def test_empty_message_clarifies_without_calling_llm(self):
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.graph_api_chat(_fake_request(""), self.account_id))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "clarify")

    def test_register_entity_lands_as_suggested_not_confirmed(self):
        with patch.object(routes, "_is_auth", return_value=True), self._mock_llm({
            "action": "register_entity",
            "table_name": "DBO.DIM_CUSTOMER",
            "confidence": 0.9,
        }):
            resp = _arun(routes.graph_api_chat(
                _fake_request("this is the mapping for the customer table"), self.account_id,
            ))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "ok")
        entity = store.get_entity(self.account_id, "DIM_CUSTOMER")
        self.assertIsNotNone(entity)
        self.assertEqual(entity["status"], "suggested")
        self.assertEqual(entity["generated_by"], "chat")

    def test_create_join_lands_as_suggested_with_validation_status(self):
        with patch.object(routes, "_is_auth", return_value=True), self._mock_llm({
            "action": "create_join",
            "from_table": "DBO.F_ORDERS",
            "to_table": "DBO.DIM_CUSTOMER",
            "from_column": "CUSTOMER_ID",
            "to_column": "CUSTOMER_ID",
            "join_type": "left",
            "confidence": 0.85,
        }):
            resp = _arun(routes.graph_api_chat(
                _fake_request("join orders to customer on customer id"), self.account_id,
            ))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "ok")
        self.assertIn("validation_status", body)

        rels = store.list_relationships(self.account_id, active_only=False)
        self.assertEqual(len(rels), 1)
        rel = rels[0]
        self.assertEqual(rel["status"], "suggested")
        self.assertEqual(rel["generated_by"], "chat")
        self.assertEqual(rel["from_column"], "CUSTOMER_ID")
        self.assertEqual(rel["to_column"], "CUSTOMER_ID")

        # Both sides should have been auto-registered as suggested entities
        # since neither existed yet.
        from_ent = store.get_entity(self.account_id, "F_ORDERS")
        to_ent = store.get_entity(self.account_id, "DIM_CUSTOMER")
        self.assertIsNotNone(from_ent)
        self.assertIsNotNone(to_ent)
        self.assertEqual(from_ent["status"], "suggested")
        self.assertEqual(to_ent["status"], "suggested")

    def test_create_join_reuses_existing_entity_instead_of_duplicating(self):
        store.save_entity(
            self.account_id, "Customer", "DIM_CUSTOMER", schema_name="DBO",
            status="confirmed", generated_by="manual",
        )
        with patch.object(routes, "_is_auth", return_value=True), self._mock_llm({
            "action": "create_join",
            "from_table": "DBO.F_ORDERS",
            "to_table": "DBO.DIM_CUSTOMER",
            "from_column": "CUSTOMER_ID",
            "to_column": "CUSTOMER_ID",
        }):
            _arun(routes.graph_api_chat(
                _fake_request("join orders to customer"), self.account_id,
            ))
        rels = store.list_relationships(self.account_id, active_only=False)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0]["to_entity"], "Customer")
        # The pre-existing confirmed entity must not be duplicated or downgraded.
        entities = [e for e in store.list_entities(self.account_id, active_only=False)
                    if e["table_name"] == "DIM_CUSTOMER"]
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["status"], "confirmed")

    def test_hallucinated_table_returns_clarify_not_a_write(self):
        with patch.object(routes, "_is_auth", return_value=True), self._mock_llm({
            "action": "register_entity",
            "table_name": "DBO.NOT_A_REAL_TABLE",
        }):
            resp = _arun(routes.graph_api_chat(
                _fake_request("map the fake table"), self.account_id,
            ))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "clarify")
        self.assertIsNone(store.get_entity(self.account_id, "NOT_A_REAL_TABLE"))

    def test_no_schema_discovered_clarifies(self):
        store.update_client_state(self.account_id, "active", {"schema_dir": ""})
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.graph_api_chat(_fake_request("map orders"), self.account_id))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "clarify")
        self.assertIn("Discover", body["message"])

    def test_batch_explicit_mapping_is_schema_validated_and_staged_without_llm(self):
        message = "\n".join([
            "F_ORDERS is a fact",
            "DIM_CUSTOMER is a dimension",
            "F_ORDERS.CUSTOMER_ID -> DIM_CUSTOMER.CUSTOMER_ID many-to-one",
        ])
        with patch.object(routes, "_is_auth", return_value=True), patch(
            "core.llm.llm_complete", new=AsyncMock(side_effect=AssertionError("LLM must not run")),
        ):
            resp = _arun(routes.graph_api_chat(_fake_request(message), self.account_id))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["staged"], {"entities": 2, "relationships": 1})
        self.assertEqual(store.get_entity(self.account_id, "F_ORDERS")["entity_type"], "fact")
        self.assertEqual(store.get_entity(self.account_id, "DIM_CUSTOMER")["entity_type"], "dimension")

    def test_validator_failure_defers_validation_instead_of_returning_http_500(self):
        with patch.object(routes, "_is_auth", return_value=True), patch(
            "core.relationship_validator.validate_relationship",
            side_effect=RuntimeError("database temporarily unavailable"),
        ):
            resp = _arun(routes.graph_api_chat(
                _fake_request("F_ORDERS.CUSTOMER_ID -> DIM_CUSTOMER.CUSTOMER_ID"),
                self.account_id,
            ))
        body = json.loads(bytes(resp.body))
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["validation_status"], "untested")
        self.assertEqual(len(store.list_relationships(self.account_id, active_only=False)), 1)

    def test_chat_never_downgrades_a_confirmed_entity(self):
        store.save_entity(
            self.account_id, "Orders", "F_ORDERS", schema_name="DBO",
            entity_type="fact", status="confirmed", generated_by="manual",
        )
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.graph_api_chat(
                _fake_request("Change F_ORDERS from fact to bridge"), self.account_id,
            ))
        body = json.loads(bytes(resp.body))
        entity = store.get_entity(self.account_id, "Orders")
        self.assertEqual(entity["status"], "confirmed")
        self.assertEqual(entity["entity_type"], "fact")
        self.assertIn("left unchanged", body["message"])


class GraphChatTemplateWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "admin" / "templates" / "client_graph.html").read_text(encoding="utf-8")

    def test_chat_button_and_panel_present(self):
        self.assertIn('id="chat-btn"', self.source)
        self.assertIn('id="chat-panel"', self.source)
        self.assertIn("toggleChatPanel", self.source)

    def test_send_posts_to_graph_chat_endpoint_via_api_base(self):
        self.assertIn("sendGraphChatMessage", self.source)
        self.assertIn("${API_BASE}/chat", self.source)

    def test_success_reply_does_not_auto_reload_only_offers_a_manual_refresh(self):
        # A chat-created suggestion never goes live automatically -- the user
        # must explicitly choose to jump to the review queue, matching how
        # graph_api_chat itself never calls _after_semantic_approval.
        start = self.source.index("async function sendGraphChatMessage")
        end = self.source.index("\n}", self.source.index("finally", start))
        block = self.source[start:end]
        self.assertIn("Refresh to review", block)
        self.assertNotIn("location.reload()\n", block.split("Refresh to review")[0])

    def test_new_thread_is_a_real_navigation_fallback(self):
        base = (ROOT / "portal" / "templates" / "portal_base.html").read_text(encoding="utf-8")
        self.assertIn('href="/portal/chat?new=1"', base)
        self.assertIn("return startNewChat(event)", base)


if __name__ == "__main__":
    unittest.main()
