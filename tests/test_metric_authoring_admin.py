"""
tests/test_metric_authoring_admin.py

Phase 5: describing a metric in the admin metrics tab.

These EXECUTE the route. The audit that scoped this phase was explicit about
why: its sibling on the portal side is covered by assertions that slice
gateway/webhooks.py as a string and compare character offsets, and those were
green throughout the release in which the accept route could not survive
contact with half its inputs. A test that reads source proves the source says
something, not that the code does it.

The governance rule this phase must not break: composing a metric NEVER writes
the registry. It writes a proposal, and the human accept route is the only door
in. That an administrator is also the approver does not change it -- one entry
path is the property being protected, and a second path that could drift from
the first is the thing being refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_authoring_admin.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402
from admin import routes  # noqa: E402
from core.metric_authoring import parse_explicit_metric_definition  # noqa: E402
from core.metric_dryrun import DryRunOutcome  # noqa: E402

store.init_db()

SCHEMA = {
    "DW.SALES_FACT": {"AMOUNT", "DISCOUNT", "CUS_NO", "INVOICE_DT"},
    "DW.CUSTOMER_DIM": {"CUS_NO", "ACT_FLG", "CUS_NM"},
    # A synthetic entry the graph chat filters with startswith("__"); it is not
    # a table and must never be offered as one.
    "__meta": {"IGNORED"},
}


@pytest.fixture
def account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _request(message="", history=None):
    req = MagicMock()
    req.json = AsyncMock(return_value={"message": message, "history": history or []})
    return req


def _call(account_id, message="", *, history=None, dryrun=None, llm=None, auth=True):
    """Drive the real route with the live database and the LLM stubbed."""
    outcome = dryrun or DryRunOutcome(
        status="ok", detail="bound", probe_kind="single_table",
        tables_probed=("DW.SALES_FACT",), value=42,
    )
    stack = [
        patch.object(routes, "_is_auth", return_value=auth),
        patch.object(routes, "_load_metric_schema_columns", return_value=dict(SCHEMA)),
        patch("core.metric_dryrun.dry_run_metric_formula",
              AsyncMock(return_value=outcome)),
    ]
    if llm is not None:
        stack += [
            patch("core.llm.resolve_provider",
                  return_value=("openai", "gpt-4o", "k", {})),
            patch("core.llm.llm_complete",
                  AsyncMock(return_value=(llm, 10, 10))),
        ]
    for ctx in stack:
        ctx.start()
    try:
        resp = asyncio.run(routes.metric_authoring_chat(
            _request(message, history), account_id))
    finally:
        for ctx in reversed(stack):
            ctx.stop()
    return resp.status_code, json.loads(bytes(resp.body))


class TestTheRouteRefusesBeforeItThinks:
    def test_an_unauthenticated_caller_gets_401(self, account):
        code, body = _call(account, "revenue per customer", auth=False)
        assert code == 401 and body["status"] == "error"

    def test_an_empty_message_asks_rather_than_calling_the_model(self, account):
        """No LLM is patched here, so if the route reached one this would raise."""
        code, body = _call(account, "")
        assert code == 200 and body["status"] == "clarify"

    def test_an_absurdly_long_message_is_refused(self, account):
        code, body = _call(account, "x" * 5000)
        assert body["status"] == "error" and "too long" in body["detail"].lower()

    def test_no_discovered_schema_says_so(self, account):
        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_load_metric_schema_columns", return_value={}):
            resp = asyncio.run(routes.metric_authoring_chat(
                _request("revenue per customer"), account))
        body = json.loads(bytes(resp.body))
        assert body["status"] == "error" and "Discover schema" in body["detail"]


class TestThePasteFormNeedsNoModel:
    """An administrator who already knows the answer should not wait for a
    model to rediscover it -- and should not spend a token on it either."""

    def test_an_explicit_definition_is_parsed_without_an_llm(self, account):
        # No LLM patched: reaching one would raise rather than pass silently.
        code, body = _call(account, "Net Revenue = SUM(AMOUNT) - SUM(DISCOUNT) FROM DW.SALES_FACT")
        assert code == 200 and body["status"] == "ok", body
        assert body["source"] == "pasted"
        assert body["draft"]["sql_template"] == "SUM(AMOUNT) - SUM(DISCOUNT)"
        assert body["draft"]["base_table"] == "DW.SALES_FACT"
        assert body["draft"]["confidence"] == 1.0

    def test_an_unknown_column_is_named_not_guessed_at(self, account):
        code, body = _call(account, "Bad = SUM(NOPE) FROM DW.SALES_FACT")
        assert body["status"] == "clarify" and "NOPE" in body["reply"]

    def test_a_table_outside_the_manifest_is_refused(self, account):
        code, body = _call(account, "Bad = SUM(AMOUNT) FROM DW.SOMEONE_ELSE")
        assert body["status"] == "clarify" and "SOMEONE_ELSE" in body["reply"]

    def test_sql_pasted_after_the_formula_cannot_ride_along(self, account):
        code, body = _call(account, "Evil = SUM(AMOUNT); DROP TABLE X FROM DW.SALES_FACT")
        assert body["status"] == "clarify"
        assert "will not accept" in body["reply"]

    def test_the_synthetic_schema_entry_is_never_offered(self):
        """__meta is not a table. The route filters it before anything sees it."""
        draft, err = parse_explicit_metric_definition(
            "X = SUM(IGNORED) FROM __meta",
            {k: sorted(v) for k, v in SCHEMA.items() if not k.startswith("__")},
        )
        assert draft is None and "__meta" in err


class TestComposingThroughTheModel:
    PLAN = json.dumps({
        "operation": "define_metric",
        "name": "Revenue Per Active Customer",
        "mode": "ratio",
        "result_format": "currency",
        "numerator": {"aggregation": "SUM", "measure_ref": "COL_REF_1"},
        "denominator": {"aggregation": "COUNT_DISTINCT", "measure_ref": "COL_REF_5"},
        "base_table_ref": "TABLE_REF_1",
        "confidence": 0.9,
    })

    def test_a_described_calculation_becomes_a_reviewable_proposal(self, account):
        code, body = _call(account, "revenue per active customer", llm=self.PLAN)
        assert code == 200 and body["status"] == "ok", body
        assert body["source"] == "composed"
        assert body["proposal_id"]
        assert "NULLIF" in body["draft"]["sql_template"], "a ratio must be null-guarded"

    def test_the_model_never_authors_sql_text(self, account):
        """The plan carries structured slots and opaque refs. A model that
        returns a formula string gets nothing through."""
        sneaky = json.dumps({
            "operation": "define_metric", "name": "X", "mode": "aggregate",
            "sql_template": "SELECT * FROM secrets",
            "aggregation": "SUM", "measure_ref": "COL_REF_1",
            "base_table_ref": "TABLE_REF_1", "confidence": 0.9,
        })
        code, body = _call(account, "compose something", llm=sneaky)
        assert body["status"] == "clarify"
        assert "unsupported fields" in body["reply"]

    def test_a_table_the_model_names_directly_is_not_honoured(self, account):
        named = json.dumps({
            "operation": "define_metric", "name": "X", "mode": "aggregate",
            "aggregation": "SUM", "measure_ref": "AMOUNT",
            "base_table_ref": "DW.SALES_FACT", "confidence": 0.9,
        })
        code, body = _call(account, "compose something", llm=named)
        assert body["status"] == "clarify", body

    def test_unparseable_model_output_is_reported_not_swallowed(self, account):
        code, body = _call(account, "compose something", llm="I think maybe SUM?")
        assert body["status"] == "clarify" and body["reply"]


class TestTheGatesBlock:
    def test_a_failed_dry_run_stops_it(self, account):
        code, body = _call(
            account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT",
            dryrun=DryRunOutcome(status="error", detail="Invalid column name"),
        )
        assert body["status"] == "clarify"
        assert "could not be proven" in body["reply"]
        assert store.list_metric_proposals(account) == []

    def test_a_SKIPPED_dry_run_also_stops_it(self, account):
        """"Not an error" is not "verified". skipped means nothing was probed --
        no database configured, no discovered tables -- and accepting that as
        proof is the defect this gate was fixed for on the portal side."""
        code, body = _call(
            account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT",
            dryrun=DryRunOutcome(status="skipped", detail="No database configured"),
        )
        assert body["status"] == "clarify"
        assert store.list_metric_proposals(account) == []

    def test_a_probed_value_is_never_recorded_in_the_proposal(self, account):
        """DryRunOutcome.value is real data. A proposal row is read back by the
        queue renderer, so a measured value has no business living in it."""
        _call(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        proposal = store.list_metric_proposals(account)[0]
        assert "42" not in json.dumps(proposal.get("dryrun") or {})
        assert "value" not in (proposal.get("dryrun") or {})


class TestItProposesAndNeverPublishes:
    """The one rule Phase 5 exists inside."""

    def test_composing_writes_a_proposal_and_no_metric(self, account):
        assert store.list_metrics(account, active_only=False) == []
        code, body = _call(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        assert body["status"] == "ok"
        assert store.list_metrics(account, active_only=False) == [], (
            "the chat route wrote the registry directly"
        )
        pending = store.list_metric_proposals(account, status="pending")
        assert len(pending) == 1
        assert pending[0]["generated_by"] == "admin_chat"

    def test_the_route_never_recompiles_the_semantic_contract(self, account):
        """_after_semantic_approval changes answers to questions nobody has
        asked yet. Only the human accept route may call it."""
        with patch.object(routes, "_after_semantic_approval") as recompile:
            _call(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        recompile.assert_not_called()

    def test_accepting_the_proposal_is_what_creates_the_metric(self, account):
        """End to end: compose, then accept, and only then does it exist."""
        _, body = _call(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        proposal_id = body["proposal_id"]

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval") as recompile, \
                patch.object(routes, "_notify_metric_proposal_reviewed", create=True):
            resp = asyncio.run(routes.metric_proposal_accept(
                MagicMock(), account, proposal_id))
        assert resp.status_code == 200, bytes(resp.body)
        recompile.assert_called_once()

        names = [m["name"] for m in store.list_metrics(account, active_only=False)]
        assert names == ["Net Revenue"]


class TestTheAuditScopeCoversTheEgress:
    def test_the_model_call_is_wrapped_in_the_admin_component(self, account):
        """A metric composed by a model is an LLM egress with the client's
        schema in the prompt; it has to be attributable."""
        seen = {}

        import core.llm_audit as audit
        real_scope = audit.llm_audit_scope

        def _spy(**kwargs):
            seen.update(kwargs)
            return real_scope(**kwargs)

        with patch("core.llm_audit.llm_audit_scope", _spy):
            _call(account, "revenue per active customer",
                  llm=TestComposingThroughTheModel.PLAN)
        assert seen.get("component") == "metric_authoring_admin", seen

    def test_the_paste_path_makes_no_llm_call_to_audit(self, account):
        """Nothing leaves the process, so there is nothing to record."""
        calls = []

        import core.llm_audit as audit
        real_scope = audit.llm_audit_scope

        def _spy(**kwargs):
            calls.append(kwargs.get("component"))
            return real_scope(**kwargs)

        with patch("core.llm_audit.llm_audit_scope", _spy):
            _call(account, "Net Revenue = SUM(AMOUNT) FROM DW.SALES_FACT")
        assert "metric_authoring_admin" not in calls


class TestThePanelIsOnThePage:
    def test_the_chat_partial_is_included_and_posts_to_the_route(self):
        from metrics_template import metrics_template

        page = metrics_template()
        assert 'id="authorChatForm"' in page, "the composer is not on the page"
        assert "metrics/api/chat" in page, "the panel does not call the route"
        assert "data-accept=" in page, "no way to accept without leaving the panel"

    def test_the_panel_escapes_what_it_renders(self):
        """The reply and the compiled formula are echoed back into innerHTML."""
        from metrics_template import metrics_template

        page = metrics_template()
        assert "function esc(" in page
        assert "esc(d.sql_template)" in page, "the formula must be escaped"
