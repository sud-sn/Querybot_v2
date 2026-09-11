"""
tests/test_an_alert_reads_as_its_owner.py

An alert's window was anchored on data its owner cannot see.

An alert belongs to a person, and check_alert_now already ran its main query as
that person: execute_governed_query under resolve_context(account, user,
channel="alert") with their allowed tables, plus a second evaluate() for the
"alert" action. So "whose access does a scheduled job use" was never an open
question on this product -- it was answered, in that function, for the query.

The business-date anchor probe did not use the answer. _refresh_relative_window
went through core.schema.run_query, which injects no row policy, so on a
workspace that restricts rows the probe read the WHOLE fact and the alert's own
query then read one slice of it. Executed against the real injector, for an
owner restricted to REGION_CD = 'EAST':

    main query : SELECT ... FROM EMDW.CUS_ORD_IVC_FCT AS anchor_fact
                 WHERE anchor_fact.REGION_CD = 'EAST'
    anchor probe: SELECT ... FROM EMDW.CUS_ORD_IVC_FCT AS anchor_fact

The window literal came from the newest date in EVERY region. When EAST loads
later than the rest -- which is the ordinary reason to have the restriction at
all -- that window ends past the end of the data the owner can see, and the
alert compares against a period that is empty for them. It fires, or fails to,
on a comparison nobody could reproduce from their own screen.

Two constructions of the same thing, built separately and drifted. There is one
now, and both callers use it, so they cannot disagree again.

The scope travels with the value and that is not incidental: core/date_anchor.py
keys its cache and its durable row on the row-policy fingerprint, so returning
an owner-filtered anchor under the empty scope would hand one reader's date to
the whole workspace -- the defect the scope exists to prevent, reintroduced
through the scheduler.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACT = "EMDW.CUS_ORD_IVC_FCT"
ANCHOR_POLICY = {
    "anchor_policy": "latest_available",
    "fact_table": FACT, "fact_column": "IVC_DT", "date_column": "IVC_DT",
}
DB_CFG = {"db_type": "azure_sql", "credentials": {}}


class AlertOwnerCase(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-alert-owner-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "a.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-alert-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "Test Ltd")
        self.user_id, _ = store.create_user(
            self.account_id, "East Rep", f"east-{uuid.uuid4().hex[:6]}@x.test",
            role="analyst")
        # A real owner has a table grant. get_allowed_tables returns an EMPTY
        # set for an analyst with no group and no overrides, and an empty set
        # means no access at all -- so without this every read below is refused
        # for want of a grant, which is a different test (and is one, at the
        # bottom of this file).
        store.set_user_extra_tables(self.user_id, self.account_id, [FACT])

        import core.date_anchor as date_anchor
        date_anchor.clear_cache(self.account_id, persistent=True)
        self._date_anchor = date_anchor

    def tearDown(self):
        self._date_anchor.clear_cache(self.account_id, persistent=True)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _restrict_the_owner(self, region="EAST"):
        import store
        store.replace_row_policies(self.account_id, 0, [{
            "name": "region", "subject_type": "user",
            "subject_id": str(self.user_id), "table_fqn": FACT,
            "condition": {"field": "REGION_CD", "operator": "=", "value": region},
        }])

    def _alert(self, *, owned=True, sql=None, anchor_value="2026-09-10"):
        alert = {
            "id": "alert-1", "db_type": "azure_sql", "status": "active",
            "question": "revenue last 7 days",
            "anchor_policy": dict(ANCHOR_POLICY),
            "anchor_value": anchor_value,
            "sql": sql or (f"SELECT SUM(NET_AMT) AS Revenue FROM {FACT} "
                           f"WHERE IVC_DT <= '{anchor_value}'"),
            "metric_col": "Revenue", "baseline_value": 100.0,
            "condition": "change_pct", "threshold": 10.0,
        }
        if owned:
            alert["account_id"] = self.account_id
            alert["user_id"] = self.user_id
        return alert

    def _capture_warehouse_sql(self, rows=None):
        """Patch the ONE call that reaches the warehouse from inside the
        governed executor, after row policies have been injected. Asserting on
        what arrives there is the only way to tell a governed read from an
        ungoverned one."""
        seen: list[str] = []

        def _run(credentials, db_type, sql, max_rows=200):
            seen.append(sql)
            return list(rows if rows is not None
                        else [{"max_business_date": "2026-09-08"}])

        return seen, _run


class TestTheProbeRunsAsTheOwner(AlertOwnerCase):

    def _refresh(self, alert, *, probe_rows=None):
        import core.alert_engine as ae

        seen, governed_run = self._capture_warehouse_sql(probe_rows)
        ungoverned: list[str] = []

        def _plain(credentials, db_type, sql, **kw):
            ungoverned.append(sql)
            return list(probe_rows or [{"max_business_date": "2026-09-08"}])

        with (
            patch("core.compliance.governed_query.run_query", governed_run),
            patch("core.schema.run_query", _plain),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            result = ae._refresh_relative_window(alert, DB_CFG)
        return result, seen, ungoverned

    def test_the_owners_row_policy_reaches_the_probe(self):
        self._restrict_the_owner()
        _, governed, ungoverned = self._refresh(self._alert())
        self.assertTrue(governed, "the probe never reached the governed executor")
        self.assertEqual(ungoverned, [],
                         "the probe still bypassed row policies")
        self.assertIn("REGION_CD", governed[0])
        self.assertIn("EAST", governed[0])

    def test_an_unrestricted_owner_reads_the_whole_fact(self):
        """No policy applies, so nothing is injected -- and the probe still
        goes through the governed executor, which is what makes that a fact
        rather than an assumption."""
        _, governed, ungoverned = self._refresh(self._alert())
        self.assertTrue(governed)
        self.assertEqual(ungoverned, [])
        self.assertNotIn("REGION_CD", governed[0])

    def test_the_window_is_moved_to_the_owners_newest_date(self):
        self._restrict_the_owner()
        (sql, problem), _, _ = self._refresh(
            self._alert(), probe_rows=[{"max_business_date": "2026-09-08"}])
        self.assertEqual(problem, "")
        self.assertIn("2026-09-08", sql)
        self.assertNotIn("2026-09-10", sql)

    def test_an_alert_with_no_owner_still_uses_the_plain_reader(self):
        """Alerts that predate owner tracking. check_alert_now runs their main
        query ungoverned too, so the probe matching it is what keeps the window
        and the query consistent."""
        _, governed, ungoverned = self._refresh(self._alert(owned=False))
        self.assertEqual(governed, [])
        self.assertTrue(ungoverned)

    def test_a_deleted_owner_is_refused_rather_than_read_ungoverned(self):
        alert = self._alert()
        alert["user_id"] = 999999
        (sql, problem), governed, ungoverned = self._refresh(alert)
        self.assertEqual(problem, "alert_user_missing")
        self.assertEqual(governed, [])
        self.assertEqual(ungoverned, [],
                         "an alert whose owner is gone read the warehouse anyway")


class TestTheAnchorIsFiledUnderTheOwnersScope(AlertOwnerCase):
    """core/date_anchor.py keys its cache and its durable row on the row-policy
    fingerprint. A filtered value under the empty scope is one reader's date
    served to the workspace."""

    def _scope_of(self, alert):
        import core.alert_engine as ae

        with (
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            return ae._owner_execution(alert, DB_CFG).scope

    def test_a_restricted_owner_gets_their_own_scope(self):
        self._restrict_the_owner()
        scope = self._scope_of(self._alert())
        self.assertTrue(scope, "a restricted owner shares the unrestricted bucket")
        self.assertNotEqual(scope, "")

    def test_it_is_the_same_fingerprint_the_injector_implies(self):
        """If these two disagree the cache is keyed on something other than
        what the probe actually ran under."""
        import store
        from core.compliance.policy_engine import resolve_context
        from core.compliance.sql_guard import row_policy_scope

        self._restrict_the_owner()
        context = resolve_context(
            self.account_id, store.get_user(self.user_id),
            action="query_execution", channel="alert", purpose_id="")
        self.assertEqual(self._scope_of(self._alert()),
                         row_policy_scope(context, [FACT]))

    def test_an_unrestricted_owner_shares_the_workspace_anchor(self):
        """Otherwise every workspace with any row policy loses anchor caching
        for everybody."""
        self.assertEqual(self._scope_of(self._alert()), "")

    def test_two_differently_restricted_owners_do_not_share(self):
        import store

        self._restrict_the_owner("EAST")
        east = self._scope_of(self._alert())
        second, _ = store.create_user(
            self.account_id, "West Rep", f"west-{uuid.uuid4().hex[:6]}@x.test",
            role="analyst")
        store.replace_row_policies(self.account_id, 0, [
            {"name": "e", "subject_type": "user", "subject_id": str(self.user_id),
             "table_fqn": FACT,
             "condition": {"field": "REGION_CD", "operator": "=", "value": "EAST"}},
            {"name": "w", "subject_type": "user", "subject_id": str(second),
             "table_fqn": FACT,
             "condition": {"field": "REGION_CD", "operator": "=", "value": "WEST"}},
        ])
        west_alert = self._alert()
        west_alert["user_id"] = second
        self.assertNotEqual(east, self._scope_of(west_alert))

    def test_no_owner_means_the_unfiltered_bucket(self):
        self.assertEqual(self._scope_of(self._alert(owned=False)), "")

    def _refresh_then_read_the_store(self, alert):
        """Run the real refresh and look at what the DURABLE store now holds.

        Everything above checks what _owner_execution COMPUTES. This checks
        what _refresh_relative_window PASSES, which is a different thing and
        the one that matters -- the scope reaches the cache and the
        business_date_anchor row through resolve_business_anchor's last
        argument, and a test that only exercises the helper cannot see it go
        missing. It did not: hardcoding "" at that call site passed every other
        assertion in this file.
        """
        import store

        import core.alert_engine as ae

        with (
            patch("core.compliance.governed_query.run_query",
                  lambda *a, **k: [{"max_business_date": "2026-09-08"}]),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            ae._refresh_relative_window(alert, DB_CFG)
        return store

    def test_the_scope_reaches_the_durable_row(self):
        self._restrict_the_owner()
        alert = self._alert()
        scope = self._scope_of(alert)
        store = self._refresh_then_read_the_store(alert)

        mine = store.load_business_date_anchor(
            self.account_id, FACT, "IVC_DT", scope)
        self.assertEqual(mine.get("value"), "2026-09-08")

    def test_and_not_the_workspace_wide_row(self):
        """The defect this guards. An owner-filtered date stored under the
        empty scope is served to every unrestricted reader in the workspace --
        the cross-reader leak, arriving through the scheduler."""
        self._restrict_the_owner()
        store = self._refresh_then_read_the_store(self._alert())
        self.assertEqual(
            store.load_business_date_anchor(self.account_id, FACT, "IVC_DT", ""),
            {},
            "a restricted alert owner's business date was filed as the "
            "workspace's, where every other reader picks it up")

    def test_an_unrestricted_owner_does_fill_the_shared_row(self):
        """The other direction: nothing applies, so the value IS the
        workspace's, and forking the cache for it would cost a probe per alert
        for no reason."""
        store = self._refresh_then_read_the_store(self._alert())
        self.assertEqual(
            store.load_business_date_anchor(
                self.account_id, FACT, "IVC_DT", "").get("value"),
            "2026-09-08")


class TestTheQueryAndTheProbeCannotDriftAgain(AlertOwnerCase):
    """They were two separate constructions of the same thing. One now serves
    both, and this asserts they really are the same one."""

    def test_check_alert_now_runs_its_query_through_the_same_reader(self):
        import core.alert_engine as ae

        self._restrict_the_owner()
        alert = self._alert(anchor_value="2026-09-08")  # already current
        seen: list[str] = []

        def _run(credentials, db_type, sql, max_rows=200):
            seen.append(sql)
            if "max_business_date" in sql:
                return [{"max_business_date": "2026-09-08"}]
            return [{"Revenue": 120.0}]

        with (
            patch.object(ae, "get_alert", return_value=alert),
            patch.object(ae, "_load", return_value=[alert]),
            patch.object(ae, "_save"),
            patch("core.compliance.governed_query.run_query", _run),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            result = ae.check_alert_now(alert["id"], DB_CFG)

        self.assertTrue(result.get("ok"), result)
        self.assertTrue(seen, "nothing reached the warehouse")
        for sql in seen:
            self.assertIn("REGION_CD", sql,
                          f"this read was not filtered by the owner's policy: {sql}")

    def test_one_construction_serves_both(self):
        """Read from the tree, because the defect was two call sites building
        the same thing and only one of them being governed. _owner_execution is
        the one; nothing else may assemble a second."""
        import ast
        import inspect

        import core.alert_engine as ae

        tree = ast.parse(inspect.getsource(ae))
        builders = [
            node.name for node in tree.body           # module level only:
            if isinstance(node, ast.FunctionDef)      # the runner is nested
            and any(isinstance(inner, ast.Call)       # inside the one builder
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "execute_governed_query"
                    for inner in ast.walk(node))
        ]
        self.assertEqual(builders, ["_owner_execution"], builders)

    def test_an_alert_with_no_owner_is_unchanged(self):
        """The legacy path still runs ungoverned, for both the probe and the
        query. Quietly governing it would break every alert created before
        owners were recorded."""
        import core.alert_engine as ae

        alert = self._alert(owned=False, anchor_value="2026-09-08")
        plain: list[str] = []
        governed: list[str] = []

        def _plain(credentials, db_type, sql, **kw):
            plain.append(sql)
            if "max_business_date" in sql:
                return [{"max_business_date": "2026-09-08"}]
            return [{"Revenue": 120.0}]

        with (
            patch.object(ae, "get_alert", return_value=alert),
            patch.object(ae, "_load", return_value=[alert]),
            patch.object(ae, "_save"),
            patch("core.schema.run_query", _plain),
            patch("core.compliance.governed_query.run_query",
                  lambda *a, **k: governed.append(a[2]) or []),
        ):
            result = ae.check_alert_now(alert["id"], DB_CFG)

        self.assertTrue(result.get("ok"), result)
        self.assertTrue(plain)
        self.assertEqual(governed, [])


if __name__ == "__main__":
    unittest.main()


class TestAnOwnerWhoMayNotReadTheFact(AlertOwnerCase):
    """Making the probe governed means it can now be REFUSED. That is the
    point, and the refusal has to be legible.

    Nothing that worked stops working: check_alert_now runs the alert's main
    query under the same rules a moment later, so an owner without a grant was
    already getting query_failed. What changes is that the reason names the
    cause instead of pointing at the warehouse.
    """

    def setUp(self):
        super().setUp()
        import store
        store.set_user_extra_tables(self.user_id, self.account_id, [])

    def test_the_reason_names_the_access_problem(self):
        import core.alert_engine as ae

        def _never(*args, **kwargs):
            raise AssertionError("the warehouse was read despite the refusal")

        with (
            patch("core.compliance.governed_query.run_query", _never),
            patch("core.schema.run_query", _never),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            sql, problem = ae._refresh_relative_window(self._alert(), DB_CFG)
        self.assertEqual(problem, "alert_owner_lacks_access")

    def test_it_is_not_reported_as_a_missing_anchor(self):
        """anchor_unavailable sent an admin to look at the date dimension for
        a problem that is a table grant."""
        import core.alert_engine as ae

        with (
            patch("core.compliance.governed_query.run_query",
                  lambda *a, **k: []),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            _, problem = ae._refresh_relative_window(self._alert(), DB_CFG)
        self.assertNotEqual(problem, "anchor_unavailable")

    def test_the_same_owner_was_already_refused_on_the_main_query(self):
        """So this commit stops no alert that was working. Asserted rather than
        assumed, because "it was already broken" is exactly the claim a reader
        should not have to take on trust."""
        import core.alert_engine as ae

        alert = self._alert(anchor_value="2026-09-08")  # nothing to re-anchor
        with (
            patch.object(ae, "get_alert", return_value=alert),
            patch.object(ae, "_load", return_value=[alert]),
            patch.object(ae, "_save"),
            patch("core.compliance.governed_query.run_query",
                  lambda *a, **k: [{"Revenue": 120.0}]),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            result = ae.check_alert_now(alert["id"], DB_CFG)
        self.assertFalse(result.get("ok"))

    def test_a_warehouse_failure_does_not_read_as_a_permissions_problem(self):
        """The distinction that matters. A timeout is caught inside
        resolve_business_anchor and surfaces as anchor_unavailable -- which is
        pre-existing and fine. What must not happen is a timeout being reported
        as a table grant, or a table grant as a timeout: they send an admin to
        different screens."""
        import core.alert_engine as ae
        import store

        store.set_user_extra_tables(self.user_id, self.account_id, [FACT])
        with (
            patch("core.compliance.governed_query.run_query",
                  side_effect=TimeoutError("warehouse timed out")),
            patch("core.schema.load_known_tables", return_value={FACT}),
            patch("core.schema.load_schema_columns", return_value={}),
        ):
            _, problem = ae._refresh_relative_window(self._alert(), DB_CFG)
        self.assertNotEqual(problem, "alert_owner_lacks_access")
        self.assertEqual(problem, "anchor_unavailable")


class TestTheRefusalDetector(unittest.TestCase):
    """_is_access_refusal decides which of the two messages an admin gets."""

    def test_a_policy_denial_is_a_refusal(self):
        import core.alert_engine as ae
        from core.compliance.governed_query import PolicyDeniedError
        from core.compliance.models import PolicyDecision

        denial = PolicyDeniedError(PolicyDecision(
            allowed=False, reason_code="policy_denied"))
        self.assertTrue(ae._is_access_refusal(denial))

    def test_a_validation_refusal_is_too(self):
        import core.alert_engine as ae
        for message in ("access_denied: no", "unknown_table: nope",
                        "You don't have access to the following table(s)"):
            self.assertTrue(ae._is_access_refusal(ValueError(message)), message)

    def test_an_ordinary_failure_is_not(self):
        import core.alert_engine as ae
        for exc in (TimeoutError("timed out"), ConnectionError("reset"),
                    ValueError("syntax error near FROM"),
                    RuntimeError("deadlock detected")):
            self.assertFalse(ae._is_access_refusal(exc), exc)
