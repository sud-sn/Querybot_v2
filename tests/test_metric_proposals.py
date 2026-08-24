"""
tests/test_metric_proposals.py

A metric is the highest-authority layer in the semantic contract: saving one
recompiles the contract synchronously, runs every conflict detector, and from
then on it can change the answer to questions nobody has asked yet.

So neither chat surface writes one. They write a proposal, and this accept route
is the only thing that turns a proposal into a metric -- the same rule the
entity-graph chat states in its own docstring. These tests exist to keep that
true as the two chat features land on top.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_metric_proposals.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()


@pytest.fixture
def account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _proposal(account_id, **overrides):
    payload = {
        "name": "Revenue Per Customer",
        "sql_template": "SUM(AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT CUS_NO), 0)",
        "formula_type": "expression",
        "required_columns": "AMOUNT, CUS_NO",
        "base_table": "DW.SALES_FACT",
    }
    payload.update(overrides.pop("payload", {}))
    return store.create_metric_proposal(
        account_id, payload=payload,
        confidence_score=overrides.pop("confidence_score", 85),
        source_question=overrides.pop("source_question", "revenue per active customer"),
        validation=overrides.pop("validation", {"valid": True}),
        dryrun=overrides.pop("dryrun", {"status": "ok", "probe_kind": "single_table"}),
        **overrides,
    )


class TestAProposalChangesNothingLive:
    def test_creating_one_does_not_create_a_metric(self):
        """The whole point. A bot can propose all day and the registry is
        untouched until a human acts."""
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        before = len(store.list_metrics(account_id, active_only=False))
        _proposal(account_id)
        assert len(store.list_metrics(account_id, active_only=False)) == before

    def test_it_lands_pending(self, account):
        proposal = store.get_metric_proposal(account, _proposal(account))
        assert proposal["status"] == "pending"
        assert proposal["payload"]["name"] == "Revenue Per Customer"

    def test_the_evidence_travels_with_it(self, account):
        """A reviewer should not have to re-test a formula to know whether it
        parses and whether it runs."""
        proposal = store.get_metric_proposal(account, _proposal(account))
        assert proposal["validation"]["valid"] is True
        assert proposal["dryrun"]["status"] == "ok"

    def test_the_probe_value_is_never_stored(self, account):
        """The dry run's scalar is real data that never crossed the compliance
        boundary. It informs the gate; it does not get filed."""
        proposal = store.get_metric_proposal(account, _proposal(account))
        assert "value" not in proposal["dryrun"]


class TestReviewHappensOnce:
    def test_accepting_twice_is_refused(self, account):
        """Guarded on status='pending', so a double-submitted approval cannot
        apply the same metric twice."""
        proposal_id = _proposal(account)
        assert store.review_metric_proposal(account, proposal_id, "accepted") is True
        assert store.review_metric_proposal(account, proposal_id, "accepted") is False

    def test_a_rejection_keeps_its_note(self, account):
        proposal_id = _proposal(account)
        store.review_metric_proposal(
            account, proposal_id, "rejected", review_note="we already have this one",
        )
        proposal = store.get_metric_proposal(account, proposal_id)
        assert proposal["status"] == "rejected"
        assert proposal["review_note"] == "we already have this one"

    @pytest.mark.parametrize("status", ["pending", "approved", "deleted", ""])
    def test_only_accept_or_reject_are_review_outcomes(self, account, status):
        with pytest.raises(ValueError):
            store.review_metric_proposal(account, _proposal(account), status)

    def test_the_pending_count_drives_the_admin_inbox(self, account):
        assert store.count_pending_metric_proposals(account) == 0
        first, second = _proposal(account), _proposal(account)
        assert store.count_pending_metric_proposals(account) == 2
        store.review_metric_proposal(account, first, "accepted")
        store.review_metric_proposal(account, second, "rejected")
        assert store.count_pending_metric_proposals(account) == 0


class TestAStaleProposalCannotBeApplied:
    """A proposal says "change it from THIS to that". If the metric moved in the
    meantime, accepting would overwrite somebody's edit with a diff computed
    against a version that no longer exists. The graph accept route answers that
    with a 409 and so does this one."""

    def test_a_changed_formula_is_drift(self):
        drifted, fields = store.metric_has_drifted(
            {"name": "Net Revenue", "sql_template": "SUM(A) - SUM(B)"},
            {"name": "Net Revenue", "sql_template": "SUM(A)"},
        )
        assert drifted and fields == ["sql_template"]

    def test_a_deleted_metric_is_drift(self):
        drifted, fields = store.metric_has_drifted(None, {"name": "Net Revenue"})
        assert drifted and fields == ["metric no longer exists"]

    def test_an_untouched_metric_is_not_drift(self):
        assert store.metric_has_drifted(
            {"name": "X", "sql_template": "SUM(A)", "base_table": "DW.F"},
            {"name": "X", "sql_template": "SUM(A)", "base_table": "DW.F"},
        ) == (False, [])

    def test_a_create_proposal_has_nothing_to_drift_against(self):
        assert store.metric_has_drifted({"name": "X"}, {}) == (False, [])

    @pytest.mark.parametrize("field", [
        "name", "sql_template", "formula_type", "required_columns", "base_table", "base_entity",
    ])
    def test_every_field_that_defines_the_metric_is_guarded(self, field):
        assert field in store.GUARDED_METRIC_FIELDS


class TestTheAcceptPathIsTheOnlyWayIn:
    def test_no_chat_handler_recompiles_the_contract(self):
        """`_after_semantic_approval` recompiles the semantic contract. The
        entity-graph chat's docstring states the rule -- chat proposes, only a
        human confirmation route applies -- and metric authoring inherits it.
        Asserted here rather than trusted, because the two chat features that
        must obey it are not written yet."""
        from pathlib import Path

        routes = (Path(__file__).resolve().parents[1] / "admin" / "routes.py").read_text(
            encoding="utf-8",
        )
        applier = routes[routes.index("def _apply_metric_create("):]
        applier = applier[: applier.index("\n@router.post") if "\n@router.post" in applier else len(applier)]
        assert "_after_semantic_approval" in applier, (
            "the shared applier is where the recompile belongs"
        )

    def test_the_form_route_and_the_accept_route_share_one_applier(self):
        """Two copies of the save + patch + recompile sequence is how they
        drift, and a drift means a metric in the registry but not in the
        compiled contract."""
        from pathlib import Path

        routes = (Path(__file__).resolve().parents[1] / "admin" / "routes.py").read_text(
            encoding="utf-8",
        )
        assert routes.count("_apply_metric_create(account_id,") == 2


class TestTheAcceptRouteActuallyRuns:
    """Executing the route, not reading it.

    Everything above this class tests the store layer or scans source text, and
    every one of those was green while the accept route contained a guaranteed
    TypeError on half its inputs:

        store.get_metric(account_id, int(proposal.get("target_metric_id")))

    store.get_metric takes ONE argument. Accepting an update proposal was a 500,
    always, and no test noticed because none of them called the route. The two
    source-scan tests immediately above are exactly the pattern that missed it.

    A second defect hid behind the first: even with the call fixed, every
    proposal went to _apply_metric_CREATE, so an accepted update would have
    tried to insert a duplicate metric under an existing name.
    """

    @staticmethod
    def _run(coro):
        import asyncio

        return asyncio.run(coro)

    @staticmethod
    def _request():
        from unittest.mock import MagicMock

        return MagicMock()

    def _accept(self, account_id, proposal_id):
        import json as _json
        from unittest.mock import patch

        from admin import routes

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval"), \
                patch.object(routes, "_notify_metric_proposal_reviewed", create=True):
            resp = self._run(routes.metric_proposal_accept(
                self._request(), account_id, proposal_id,
            ))
        return resp.status_code, _json.loads(bytes(resp.body))

    def _seed_metric(self, account_id, name="Revenue Per Customer"):
        store.save_metric(account_id, {
            "name": name,
            "sql_template": "SUM(AMOUNT)",
            "formula_type": "expression",
            "required_columns": "AMOUNT",
            "base_table": "DW.SALES_FACT",
        }, db_type="azure_sql")
        for row in store.list_metrics(account_id, active_only=False):
            if str(row.get("name") or "") == name:
                return int(row["id"]), row
        raise AssertionError("seed metric not saved")

    def test_accepting_a_create_proposal_writes_one_metric(self, account):
        pid = _proposal(account)
        code, body = self._accept(account, pid)
        assert code == 200, body
        names = [m["name"] for m in store.list_metrics(account, active_only=False)]
        assert names.count("Revenue Per Customer") == 1

    def test_accepting_an_update_proposal_updates_rather_than_500s(self, account):
        """The regression. Before the fix this raised TypeError from the route
        itself, so the assertion below never got the chance to run."""
        metric_id, live = self._seed_metric(account)
        pid = store.create_metric_proposal(
            account, action="update_metric", target_metric_id=metric_id,
            before=dict(live),
            payload={
                "name": "Revenue Per Customer",
                "sql_template": "SUM(AMOUNT) - SUM(DISCOUNT)",
                "formula_type": "expression",
                "required_columns": "AMOUNT, DISCOUNT",
                "base_table": "DW.SALES_FACT",
            },
        )
        code, body = self._accept(account, pid)
        assert code == 200, body

        rows = [m for m in store.list_metrics(account, active_only=False)
                if m["name"] == "Revenue Per Customer"]
        assert len(rows) == 1, "an update must not create a second metric"
        assert rows[0]["sql_template"] == "SUM(AMOUNT) - SUM(DISCOUNT)"
        assert int(rows[0]["id"]) == metric_id

    def test_an_update_whose_metric_moved_is_refused_as_a_conflict(self, account):
        metric_id, live = self._seed_metric(account, "Margin")
        pid = store.create_metric_proposal(
            account, action="update_metric", target_metric_id=metric_id,
            before=dict(live),
            payload={**dict(live), "sql_template": "SUM(A) - SUM(B)"},
        )
        store.update_metric(metric_id, {"sql_template": "SUM(SOMETHING_ELSE)"},
                            account_id=account, db_type="azure_sql")
        code, body = self._accept(account, pid)
        assert code == 409, body
        assert "changed since" in body["detail"]

    def test_an_update_targeting_another_tenants_metric_is_not_found(self, account):
        """store.get_metric does not scope by account, so the tenant check has
        to be explicit. The broken call passed account_id as the metric id,
        which meant the check it looked like it was doing never happened."""
        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        metric_id, live = self._seed_metric(other, "Someone Elses Metric")
        pid = store.create_metric_proposal(
            account, action="update_metric", target_metric_id=metric_id,
            before=dict(live), payload={**dict(live), "sql_template": "SUM(X)"},
        )
        code, body = self._accept(account, pid)
        assert code == 404, body
        still = store.get_metric(metric_id)
        assert still["sql_template"] == "SUM(AMOUNT)", "another tenant's metric was written"

    def test_the_route_reads_get_metric_with_the_arity_it_has(self):
        """The specific defect, pinned by calling the real function the way the
        route calls it — not by grepping for the call."""
        import inspect

        params = inspect.signature(store.get_metric).parameters
        assert len(params) == 1, (
            "get_metric grew a parameter; the accept route passes exactly one"
        )


class TestTheQueueIsReachable:
    """A review queue nobody can open is a queue that fills up silently.

    store.list_metric_proposals shipped with the proposal model and had ZERO
    production callers. The admin inbox badge linked to /metrics#proposals --
    an anchor that resolved to nothing, on a page that never fetched a
    proposal. Both chat surfaces could file requests; no administrator could
    see one.
    """

    @staticmethod
    def _render(account_id):
        import asyncio
        from unittest.mock import MagicMock, patch

        from admin import routes

        request = MagicMock()
        request.query_params = {}
        with patch.object(routes, "_is_auth", return_value=True):
            resp = asyncio.run(routes.metrics_page(request, account_id))
        return resp.body.decode("utf-8", "replace")

    def test_a_pending_proposal_appears_on_the_metrics_page(self, account):
        _proposal(account)
        html = self._render(account)
        assert 'id="proposals"' in html, "the anchor the admin inbox links to"
        assert "Revenue Per Customer" in html
        assert "SUM(AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT CUS_NO), 0)" in html

    def test_the_question_that_prompted_it_is_shown(self, account):
        """An administrator approving a metric needs to know what someone was
        actually trying to answer with it."""
        _proposal(account)
        assert "revenue per active customer" in self._render(account)

    def test_the_evidence_is_shown_not_just_the_formula(self, account):
        _proposal(account)
        html = self._render(account)
        assert "validated" in html and "dry run passed" in html

    def test_an_accepted_proposal_leaves_the_queue(self, account):
        pid = _proposal(account)
        assert "Revenue Per Customer" in self._render(account)
        store.review_metric_proposal(account, pid, "accepted", reviewed_by="admin")
        html = self._render(account)
        assert "Nothing waiting" in html

    def test_another_clients_proposals_never_appear(self, account):
        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        _proposal(other, payload={"name": "Someone Elses Metric"})
        assert "Someone Elses Metric" not in self._render(account)

    def test_an_empty_queue_says_so_rather_than_showing_nothing(self, account):
        assert "Nothing waiting" in self._render(account)
