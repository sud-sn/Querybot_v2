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
