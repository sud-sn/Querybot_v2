# -*- coding: utf-8 -*-
"""The customer saw a table and three computed sentences and asked where the
analyst was.

After every answer the card carries a deterministic summary: a headline, the
leader's share or the trend, a decision signal. The model-written analyst
explanation -- a few sentences on what the result shows, in the reader's
language, sent as a second message -- was gated on the wording of the
question: only "why ..." and, later, an explicit "analyse ..." earned it. For
"net sales by warehouse" the gate returned nothing, by design, to save one
model round trip per question.

That is now the tenant's choice rather than the code's. A client row carries
analysis_mode, "always" by default: every result gets the analyst, causal
wording still gets the drill-down treatment, and "on_request" keeps the old
gate exactly. Regulated tenants are unaffected either way -- the pipeline's
own compliance gate refuses before any row reaches a model, and that refusal
has its own tests.

Every test here executes the real thing: the setting reader, the decision
function, the store, and the admin handler that edits the row. The one place
that decides in production is the query pipeline, which needs a warehouse,
so its call is read as a syntax tree rather than executed.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import unittest
import uuid
from unittest.mock import patch

import pytest

from core.insight import ANALYSIS_MODES, analysis_action_for, analysis_mode_for


class TestTheTenantSetting:

    def test_a_row_that_never_chose_gets_the_analyst(self):
        assert analysis_mode_for({}) == "always"
        assert analysis_mode_for(None) == "always"
        assert analysis_mode_for({"analysis_mode": None}) == "always"
        assert analysis_mode_for({"analysis_mode": ""}) == "always"

    def test_on_request_is_honoured(self):
        assert analysis_mode_for({"analysis_mode": "on_request"}) == "on_request"

    def test_case_and_whitespace_do_not_change_the_choice(self):
        assert analysis_mode_for({"analysis_mode": " On_Request "}) == "on_request"
        assert analysis_mode_for({"analysis_mode": "ALWAYS"}) == "always"

    def test_a_value_that_is_neither_is_the_default_not_a_downgrade(self):
        """A typo in a hand-edited row must not quietly switch the analyst
        off; the promised behaviour is the safe reading."""
        assert analysis_mode_for({"analysis_mode": "sometimes"}) == "always"

    def test_the_two_modes_are_the_only_two(self):
        assert set(ANALYSIS_MODES) == {"always", "on_request"}


class TestWhatEveryAnswerEarnsByDefault:

    @pytest.mark.parametrize("question", [
        "net sales by warehouse",
        "revenue by region",
        "top 10 customers by margin",
        "sales last month",
        "how many open orders are there",
        "ventes nettes par entrepôt",
    ])
    def test_a_plain_question_gets_the_analyst(self, question):
        """The half of the old gate that was wrong about the product."""
        assert analysis_action_for(question, mode="always") == "analyze"

    @pytest.mark.parametrize("question", [
        "why did revenue drop last quarter?",
        "what drove the increase in North?",
        "what changed between March and April?",
    ])
    def test_causal_wording_still_gets_the_causal_treatment(self, question):
        assert analysis_action_for(question, mode="always") == "why"

    def test_a_clarified_question_is_still_explained(self):
        """The dispatcher re-submits the reader's RESOLVED question after a
        clarification, so the result is theirs to have explained."""
        assert analysis_action_for("net sales by warehouse", mode="always",
                                   is_clarification=True) == "analyze"
        assert analysis_action_for("why did net sales drop in March",
                                   mode="always", is_clarification=True) == "why"

    def test_nothing_at_all_is_not_a_question(self):
        assert analysis_action_for("", mode="always") == ""

    def test_the_mode_is_not_optional(self):
        """A caller that forgot the tenant would silently get one of two
        behaviours. It gets a TypeError instead."""
        with pytest.raises(TypeError):
            analysis_action_for("net sales by warehouse")  # type: ignore[call-arg]


class TestOnRequestIsTheOldGateExactly:

    @pytest.mark.parametrize("question,expected", [
        ("net sales by warehouse", ""),
        ("revenue by region", ""),
        ("analyse revenue by region", "analyze"),
        ("what stands out in sales by region", "analyze"),
        ("why did revenue drop last quarter?", "why"),
        ("", ""),
    ])
    def test_the_table_from_before(self, question, expected):
        assert analysis_action_for(question, mode="on_request") == expected

    def test_a_clarification_reply_earns_nothing_on_request(self):
        assert analysis_action_for("why did revenue drop?", mode="on_request",
                                   is_clarification=True) == ""


def _arun(coro):
    return asyncio.run(coro)


class TestTheSettingIsStoredAndEdited(unittest.TestCase):
    """The real store and the real admin handler, on a throwaway client."""

    def setUp(self):
        import store
        store.init_db()
        self.store = store
        self.account = f"acct-analyst-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account, "portal")

    def tearDown(self):
        with self.store.get_db() as conn:
            conn.execute("DELETE FROM client WHERE account_id = ?", (self.account,))

    def _update(self, **kwargs):
        import admin.routes as routes
        from starlette.datastructures import FormData

        async def _form():
            return FormData([])

        req = type("R", (), {"form": staticmethod(_form)})()
        defaults = dict(
            client_name="", db_config_id="", llm_provider="", llm_model="",
            query_limit_monthly="", token_limit_monthly="", enable_llm_audit="",
            portal_only="", teams_platform_config_id="",
        )
        defaults.update(kwargs)
        with patch.object(routes, "_is_auth", return_value=True):
            return _arun(routes.client_update(req, self.account, **defaults))

    def test_a_new_client_is_born_with_the_analyst_on(self):
        client = self.store.get_client(self.account)
        self.assertEqual(client["analysis_mode"], "always")
        self.assertEqual(analysis_mode_for(client), "always")

    def test_the_store_records_on_request(self):
        self.store.update_client_meta(self.account, analysis_mode="on_request")
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "on_request")

    def test_the_store_refuses_a_third_value(self):
        with self.assertRaises(ValueError):
            self.store.update_client_meta(self.account, analysis_mode="sometimes")
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "always")

    def test_the_admin_form_switches_it_off_and_on(self):
        resp = self._update(analysis_mode="on_request")
        self.assertEqual(resp.status_code, 303)
        self.assertIn("saved=1", resp.headers["location"])
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "on_request")
        self._update(analysis_mode="always")
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "always")

    def test_a_form_that_does_not_carry_the_field_keeps_the_default(self):
        self._update()
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "always")

    def test_a_tampered_form_value_is_stored_as_the_default(self):
        """The route normalises through the same reader the pipeline uses,
        so the row and the behaviour agree."""
        self.store.update_client_meta(self.account, analysis_mode="on_request")
        resp = self._update(analysis_mode="sometimes")
        self.assertEqual(resp.status_code, 303)
        self.assertEqual(self.store.get_client(self.account)["analysis_mode"], "always")

    def test_the_admin_page_offers_both_choices(self):
        """The template is rendered by the real page handler for this client,
        with the row set to on_request, and the option marked selected is the
        stored one."""
        import admin.routes as routes
        self.store.update_client_meta(self.account, analysis_mode="on_request")
        from starlette.requests import Request
        scope = {"type": "http", "method": "GET", "path": f"/admin/clients/{self.account}",
                 "headers": [], "query_string": b"", "session": {}}
        req = Request(scope)
        with patch.object(routes, "_is_auth", return_value=True):
            resp = _arun(routes.client_detail(req, self.account))
        html = resp.body.decode("utf-8")
        self.assertIn('name="analysis_mode"', html)
        self.assertRegex(html, r'value="on_request"[^>]*selected')
        self.assertNotRegex(html, r'value="always"[^>]*selected')


class TestThePipelineAsksTheTenant(unittest.TestCase):
    """The one production caller. It needs a warehouse to execute, so its
    call is checked as a syntax tree: analysis_action_for(..., mode=
    analysis_mode_for(client), ...) inside _handle_query_impl."""

    def test_the_decision_is_made_with_the_clients_mode(self):
        import core.query_pipeline as qp
        tree = ast.parse(inspect.getsource(qp._handle_query_impl))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) == "analysis_action_for"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one analyst decision in the pipeline")
        mode = {kw.arg: kw.value for kw in calls[0].keywords}.get("mode")
        self.assertIsInstance(mode, ast.Call)
        self.assertEqual(getattr(mode.func, "id", getattr(mode.func, "attr", "")), "analysis_mode_for")
        self.assertEqual([getattr(a, "id", None) for a in mode.args], ["client"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
