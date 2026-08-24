"""
tests/test_adhoc_metric_drafts.py

store/adhoc_metric_store.py had ZERO test coverage. Not thin coverage -- none:
a search for its function names across tests/ matched only source-text scans in
another module that never call them.

It holds a thread-scoped metric definition that answers a live user's questions
and can be promoted into the shared registry, so its access control, its expiry
and its supersession are all load-bearing. This exercises them.

The ACL case is the one that mattered. The plan required source_tables to be
re-checked against the user's grants on EVERY read. The listing read did it; the
read used by the promotion frame did not, so a user whose table grant was
revoked mid-thread could still turn that draft into a proposal naming a table
they could no longer query.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_adhoc_drafts.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

SESSION = "thread-1"
USER_ID = 7
TABLES = ["DW.SALES_FACT", "DW.CUSTOMER_DIM"]


@pytest.fixture
def account():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    return account_id


def _draft(account_id, *, name="Revenue Per Customer", session=SESSION, user=USER_ID,
           tables=None):
    return store.save_session_metric_draft(
        account_id, session, user,
        {
            "name": name,
            "sql_template": "SUM(AMOUNT) * 1.0 / NULLIF(COUNT(DISTINCT CUS_NO), 0)",
            "formula_type": "expression",
            "required_columns": "AMOUNT, CUS_NO",
            "base_table": "DW.SALES_FACT",
        },
        source_tables=list(TABLES if tables is None else tables),
        validation={"valid": True},
        dryrun={"status": "ok"},
        confidence=0.9,
        source_question="revenue per active customer",
    )


class TestARevokedGrantKillsTheDraft:
    """The rule the plan stated: re-checked on every read, not on the one the
    author happened to remember."""

    def test_the_listing_read_drops_a_draft_whose_table_was_revoked(self, account):
        _draft(account)
        assert store.active_session_metrics(account, SESSION, TABLES)
        assert store.active_session_metrics(account, SESSION, ["DW.SOMETHING_ELSE"]) == []

    def test_the_promotion_read_also_drops_it(self, account):
        """The gap. This read is what the metric_promotion_request frame uses,
        and it checked only the account -- so the one path that turns a draft
        into a durable artifact was the one path with no ACL re-check."""
        draft_id = _draft(account)
        assert store.get_session_metric_draft(account, draft_id, allowed_tables=TABLES)
        assert store.get_session_metric_draft(
            account, draft_id, allowed_tables=["DW.SOMETHING_ELSE"],
        ) is None

    def test_an_unqualified_grant_still_matches_a_qualified_draft(self, account):
        """An ACL may store a bare table name while a draft records a
        fully-qualified one. Both reads compare on the bare name too, and they
        have to agree about it -- which is why there is now one rule and not
        two copies."""
        draft_id = _draft(account)
        bare = ["SALES_FACT", "CUSTOMER_DIM"]
        assert store.active_session_metrics(account, SESSION, bare)
        assert store.get_session_metric_draft(account, draft_id, allowed_tables=bare)

    def test_a_draft_with_no_recorded_tables_is_not_blocked(self, account):
        draft_id = _draft(account, tables=[])
        assert store.get_session_metric_draft(account, draft_id, allowed_tables=["ANY.THING"])

    def test_omitting_the_acl_reads_it_unfiltered(self, account):
        """The argument is optional so non-user-facing callers keep working.
        That is deliberate, and worth pinning so nobody 'tidies' it into a
        default-deny that silently breaks promotion."""
        draft_id = _draft(account)
        assert store.get_session_metric_draft(account, draft_id) is not None


class TestTheDraftIsScopedToItsOwner:
    def test_another_account_cannot_read_it(self, account):
        draft_id = _draft(account)
        other = f"acct{os.urandom(4).hex()}"
        store.upsert_client(other, "Other Ltd")
        assert store.get_session_metric_draft(other, draft_id) is None

    def test_a_different_thread_does_not_see_it(self, account):
        _draft(account)
        assert store.active_session_metrics(account, "some-other-thread", TABLES) == []

    def test_the_owner_is_recorded_so_the_frame_can_check_it(self, account):
        draft_id = _draft(account)
        assert int(store.get_session_metric_draft(account, draft_id)["portal_user_id"]) == USER_ID


class TestOnlyTheNewestDraftIsLive:
    def test_a_second_draft_supersedes_the_first(self, account):
        first = _draft(account, name="First")
        second = _draft(account, name="Second")
        live = store.active_session_metrics(account, SESSION, TABLES)
        assert [m["name"] for m in live] == ["Second"]
        assert store.get_session_metric_draft(account, first)["status"] != "active"
        assert store.get_session_metric_draft(account, second)["status"] == "active"

    def test_promoting_marks_it_so_it_cannot_be_promoted_twice(self, account):
        draft_id = _draft(account)
        assert store.mark_draft_promoted(account, draft_id, 123) is True
        assert store.get_session_metric_draft(account, draft_id)["status"] != "active"
        assert store.active_session_metrics(account, SESSION, TABLES) == []

    def test_discarding_takes_it_out_of_the_thread(self, account):
        draft_id = _draft(account)
        assert store.discard_session_metric_draft(account, draft_id, USER_ID) is True
        assert store.active_session_metrics(account, SESSION, TABLES) == []

    def test_another_user_cannot_discard_someone_elses_draft(self, account):
        draft_id = _draft(account)
        assert store.discard_session_metric_draft(account, draft_id, USER_ID + 1) is False
        assert store.active_session_metrics(account, SESSION, TABLES)


class TestItExpires:
    def test_the_ttl_default_is_four_hours(self, monkeypatch):
        import store.adhoc_metric_store as mod

        monkeypatch.delenv("ADHOC_METRIC_TTL_SECONDS", raising=False)
        assert mod._ttl_seconds() == 4 * 60 * 60

    @pytest.mark.parametrize("value,expected", [
        ("600", 600), ("60", 60), ("1", 60), ("0", 60), ("nonsense", 4 * 60 * 60),
    ])
    def test_the_env_override_has_a_floor_and_survives_garbage(
        self, monkeypatch, value, expected,
    ):
        """A one-second TTL would make the feature look broken rather than
        configurable, so there is a floor; a typo must not crash the read."""
        import store.adhoc_metric_store as mod

        monkeypatch.setenv("ADHOC_METRIC_TTL_SECONDS", value)
        assert mod._ttl_seconds() == expected

    def test_an_expired_draft_stops_being_offered(self, account, monkeypatch):
        monkeypatch.setenv("ADHOC_METRIC_TTL_SECONDS", "60")
        _draft(account)
        assert store.active_session_metrics(account, SESSION, TABLES)

        # Age it past its own expiry rather than sleeping.
        from datetime import datetime, timedelta

        from store.db import get_db

        stale = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE session_metric_draft SET expires_at=? WHERE account_id=?",
                (stale, account),
            )
        assert store.active_session_metrics(account, SESSION, TABLES) == []

    def test_purging_removes_expired_rows(self, account):
        _draft(account)
        from datetime import datetime, timedelta

        from store.db import get_db

        stale = (datetime.utcnow() - timedelta(days=2)).isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE session_metric_draft SET expires_at=? WHERE account_id=?",
                (stale, account),
            )
        assert store.purge_expired_drafts() >= 1


class TestWhatThePipelineReceives:
    """active_session_metrics feeds straight into the metric candidate list, so
    its rows have to be shaped like registry metrics or the planner will not
    recognise them."""

    def test_a_draft_looks_like_a_metric_and_says_it_is_not_one(self, account):
        _draft(account)
        metric = store.active_session_metrics(account, SESSION, TABLES)[0]
        assert metric["name"] == "Revenue Per Customer"
        assert metric["sql_template"]
        assert metric["_adhoc"] is True
        assert metric["_pinned_thread_metric"] is True

    def test_it_carries_its_own_draft_id_for_the_promotion_frame(self, account):
        draft_id = _draft(account)
        metric = store.active_session_metrics(account, SESSION, TABLES)[0]
        assert int(metric["_draft_id"]) == draft_id
