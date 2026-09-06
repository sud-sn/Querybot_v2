"""
tests/test_client_sources.py

A2 (first stage) — the connections a workspace can answer from.

A workspace has held exactly one warehouse connection since the product
started. A tenant whose sales data is in Snowflake and whose finance ledger is
in Azure SQL has to be two workspaces: two knowledge bases, two graphs, and no
way to check one area's number against the other's.

Everything here is additive, and most of this file is about proving that.
`client.db_config_id` stays authoritative for a workspace that has declared no
sources, the startup backfill gives every existing tenant one default source
pointing at the connection they already had, and one resolver decides which
connection a question runs against — so a scoping mistake is one function to
read rather than a hundred call sites to audit.

A real database throughout. The backfill is a migration and the resolver is a
seam between two tables; neither can be tested honestly against a mock.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import uuid


class RealDatabase(unittest.TestCase):

    def setUp(self):
        import store

        self._dir = tempfile.mkdtemp(prefix="qb-sources-")
        self._saved = {k: os.environ.get(k) for k in ("DB_PATH", "QUERYBOT_DB_PATH")}
        path = os.path.join(self._dir, "d.db")
        os.environ["DB_PATH"] = path
        os.environ["QUERYBOT_DB_PATH"] = path
        store.init_db()
        self.account_id = f"acct-src-{uuid.uuid4().hex[:8]}"
        store.upsert_client(self.account_id, "portal")
        self.snowflake = self._db_config("snowflake-prod")
        self.azure = self._db_config("azure-finance")

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self._dir, ignore_errors=True)

    def _db_config(self, name: str) -> int:
        import store

        with store.get_db() as conn:
            return int(conn.execute(
                "INSERT INTO db_config (name, db_type, credentials_encrypted) "
                "VALUES (?,?,?)", (name, "azure_sql", "unused-here"),
            ).lastrowid)

    def _resolved_config_id(self, **kwargs):
        """Which db_config get_client_db went to fetch.

        A spy rather than a real fetch, because get_db_config DECRYPTS, and by
        the time a full-suite run reaches this module the process has been
        through a dozen modules that repoint QUERYBOT_KEY_FILE at their own
        mkdtemp and re-import store — so a Fernet round-trip fails inside a
        single test with the key file and every binding verifiably identical
        (tests/test_egress_posture.py documents the same condition and works
        around it with a subprocess).

        Decryption is not what A2 changed. Which id the resolver chose is.
        """
        from unittest.mock import patch

        import core.pipeline_context as pipeline_context

        # Patched on the store object pipeline_context ACTUALLY BOUND, not on
        # whatever `import store` resolves to here. Several modules in this
        # suite repoint QUERYBOT_KEY_FILE and re-import store, so the two are
        # not always the same object -- patching the wrong one leaves the real
        # get_db_config running and the failure looks like a Fernet bug.
        with patch.object(pipeline_context.store, "get_db_config",
                          side_effect=lambda cfg: {"id": "SPY", "resolved": cfg}) as spy:
            result = pipeline_context.get_client_db(self.account_id, **kwargs)
        if result is None:
            self.assertFalse(spy.called, "a config was fetched for a refused resolve")
            return None
        return result["resolved"]

    def _set_legacy_connection(self, db_config_id: int):
        import store

        with store.get_db() as conn:
            conn.execute("UPDATE client SET db_config_id=? WHERE account_id=?",
                         (db_config_id, self.account_id))

    def _defaults(self):
        import store

        return [s for s in store.list_client_sources(self.account_id)
                if s["is_default"]]


class TestNothingChangesForAWorkspaceThatHasNotOptedIn(RealDatabase):

    def test_a_workspace_with_no_sources_uses_the_column_it_always_did(self):
        import store

        self._set_legacy_connection(self.snowflake)
        self.assertEqual(store.list_client_sources(self.account_id), [])
        self.assertEqual(store.resolve_db_config_id(self.account_id), self.snowflake)

    def test_a_workspace_with_no_connection_at_all_resolves_to_nothing(self):
        import store

        self.assertIsNone(store.resolve_db_config_id(self.account_id))

    def test_get_client_db_still_works_without_arguments(self):
        self._set_legacy_connection(self.snowflake)
        self.assertEqual(self._resolved_config_id(), self.snowflake)

    def test_an_unknown_account_resolves_to_nothing(self):
        from core.pipeline_context import get_client_db

        self.assertIsNone(get_client_db("nobody"))


class TestTheBackfillGivesExistingTenantsTheirConnection(RealDatabase):

    def test_a_client_with_a_connection_gets_one_default_source(self):
        import store

        self._set_legacy_connection(self.snowflake)
        store.init_db()

        sources = store.list_client_sources(self.account_id)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["db_config_id"], self.snowflake)
        self.assertTrue(sources[0]["is_default"])

    def test_it_is_idempotent(self):
        import store

        self._set_legacy_connection(self.snowflake)
        for _ in range(3):
            store.init_db()
        self.assertEqual(len(store.list_client_sources(self.account_id)), 1)

    def test_a_client_with_no_connection_gets_no_phantom_source(self):
        import store

        store.init_db()
        self.assertEqual(store.list_client_sources(self.account_id), [])

    def test_it_does_not_touch_a_workspace_that_already_has_sources(self):
        import store

        self._set_legacy_connection(self.snowflake)
        store.save_client_source(self.account_id, "Finance", self.azure)
        store.init_db()

        sources = store.list_client_sources(self.account_id)
        # A deliberately-configured workspace is not "fixed" by a restart.
        self.assertEqual([s["name"] for s in sources], ["Finance"])

    def test_a_deleted_source_is_not_resurrected_by_a_restart(self):
        import store

        self._set_legacy_connection(self.snowflake)
        store.init_db()
        only = store.list_client_sources(self.account_id)[0]
        store.save_client_source(self.account_id, "Finance", self.azure)
        store.delete_client_source(self.account_id, only["id"])
        store.init_db()
        self.assertEqual([s["name"] for s in store.list_client_sources(self.account_id)],
                         ["Finance"])


class TestExactlyOneDefault(RealDatabase):

    def test_the_first_source_is_the_default_whatever_was_asked_for(self):
        import store

        # A workspace with sources and no default has no answer to "which
        # connection does a question with no domain run against?", and the
        # resolver would have to invent one.
        store.save_client_source(self.account_id, "Sales", self.snowflake,
                                 is_default=False)
        self.assertEqual(len(self._defaults()), 1)

    def test_adding_a_second_source_does_not_change_the_default(self):
        import store

        first = store.save_client_source(self.account_id, "Sales", self.snowflake)
        store.save_client_source(self.account_id, "Finance", self.azure)
        self.assertEqual([s["id"] for s in self._defaults()], [first])

    def test_promoting_one_demotes_the_other(self):
        import store

        store.save_client_source(self.account_id, "Sales", self.snowflake)
        second = store.save_client_source(self.account_id, "Finance", self.azure)
        self.assertTrue(store.set_default_source(self.account_id, second))
        self.assertEqual([s["id"] for s in self._defaults()], [second])

    def test_a_source_that_is_not_ours_cannot_be_promoted(self):
        import store

        store.upsert_client("other-tenant", "portal")
        theirs = store.save_client_source("other-tenant", "Theirs", self.azure)
        self.assertFalse(store.set_default_source(self.account_id, theirs))

    def test_deleting_the_default_promotes_another(self):
        import store

        first = store.save_client_source(self.account_id, "Sales", self.snowflake)
        store.save_client_source(self.account_id, "Finance", self.azure)
        store.delete_client_source(self.account_id, first)
        # Left with sources and no default, a workspace cannot answer a
        # question that names no domain: that is an outage, not tidiness.
        self.assertEqual(len(self._defaults()), 1)
        self.assertEqual(self._defaults()[0]["name"], "Finance")

    def test_deleting_the_only_source_leaves_the_legacy_fallback(self):
        import store

        self._set_legacy_connection(self.snowflake)
        only = store.save_client_source(self.account_id, "Finance", self.azure)
        store.delete_client_source(self.account_id, only)
        self.assertEqual(store.resolve_db_config_id(self.account_id), self.snowflake)

    def test_deleting_something_that_is_not_there_says_so(self):
        import store

        self.assertFalse(store.delete_client_source(self.account_id, 999))


class TestTheResolverIsTheOnlyPlaceThatDecides(RealDatabase):

    def setUp(self):
        super().setUp()
        import store

        self._set_legacy_connection(self.snowflake)
        self.sales = store.save_client_source(
            self.account_id, "Sales", self.snowflake, domain_id=11)
        self.finance = store.save_client_source(
            self.account_id, "Finance", self.azure, domain_id=22)

    def test_a_named_source_wins(self):
        import store

        self.assertEqual(
            store.resolve_db_config_id(self.account_id, source_id=self.finance),
            self.azure)

    def test_a_domain_picks_its_source(self):
        import store

        self.assertEqual(
            store.resolve_db_config_id(self.account_id, domain_id=22), self.azure)

    def test_a_domain_with_no_source_falls_back_to_the_default(self):
        import store

        self.assertEqual(
            store.resolve_db_config_id(self.account_id, domain_id=99), self.snowflake)

    def test_a_named_source_beats_the_domain(self):
        import store

        self.assertEqual(
            store.resolve_db_config_id(self.account_id, source_id=self.sales,
                                       domain_id=22),
            self.snowflake)

    def test_an_unknown_source_refuses_rather_than_falling_back(self):
        import store

        # Falling through would run the question against a connection the
        # caller did not ask for and report the answer as though it had.
        self.assertIsNone(
            store.resolve_db_config_id(self.account_id, source_id=999))

    def test_another_tenants_source_is_not_reachable_by_id(self):
        import store

        store.upsert_client("other-tenant", "portal")
        theirs = store.save_client_source("other-tenant", "Theirs", self.azure)
        self.assertIsNone(
            store.resolve_db_config_id(self.account_id, source_id=theirs))
        self.assertIsNone(store.get_client_source(self.account_id, theirs))

    def test_an_inactive_source_is_not_used(self):
        import store

        store.save_client_source(self.account_id, "Finance", self.azure,
                                 source_id=self.finance, is_active=False)
        self.assertIsNone(
            store.resolve_db_config_id(self.account_id, source_id=self.finance))
        self.assertEqual(
            store.resolve_db_config_id(self.account_id, domain_id=22), self.snowflake)

    def test_get_client_db_carries_the_choice_through(self):
        self.assertEqual(self._resolved_config_id(source_id=self.finance), self.azure)
        self.assertEqual(self._resolved_config_id(domain_id=22), self.azure)
        self.assertEqual(self._resolved_config_id(), self.snowflake)

    def test_get_client_db_refuses_an_unknown_source(self):
        from core.pipeline_context import get_client_db

        self.assertIsNone(get_client_db(self.account_id, source_id=999))


class TestSourcesAreNamedAndTenantScoped(RealDatabase):

    def test_a_source_needs_a_name(self):
        import store

        for blank in ("", "   ", None):
            with self.subTest(name=blank):
                with self.assertRaises(ValueError):
                    store.save_client_source(self.account_id, blank, self.snowflake)

    def test_two_sources_cannot_share_a_name(self):
        import sqlite3

        import store

        store.save_client_source(self.account_id, "Sales", self.snowflake)
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_client_source(self.account_id, "Sales", self.azure)

    def test_two_tenants_can_both_have_a_sales_source(self):
        import store

        store.upsert_client("other-tenant", "portal")
        store.save_client_source(self.account_id, "Sales", self.snowflake)
        store.save_client_source("other-tenant", "Sales", self.azure)
        self.assertEqual(len(store.list_client_sources(self.account_id)), 1)
        self.assertEqual(len(store.list_client_sources("other-tenant")), 1)

    def test_listing_is_scoped_to_one_tenant(self):
        import store

        store.upsert_client("other-tenant", "portal")
        store.save_client_source("other-tenant", "Theirs", self.azure)
        self.assertEqual(store.list_client_sources(self.account_id), [])

    def test_the_default_is_listed_first(self):
        import store

        # The default is named LAST alphabetically on purpose: with "Alpha" as
        # the default, plain name ordering produces the same list and the
        # assertion passes whether or not the default is being hoisted at all.
        store.save_client_source(self.account_id, "Alpha", self.snowflake)
        zulu = store.save_client_source(self.account_id, "Zulu", self.azure)
        store.set_default_source(self.account_id, zulu)
        self.assertEqual(
            [s["name"] for s in store.list_client_sources(self.account_id)],
            ["Zulu", "Alpha"])

    def test_updating_a_source_keeps_its_id(self):
        import store

        source_id = store.save_client_source(self.account_id, "Sales", self.snowflake)
        again = store.save_client_source(self.account_id, "Sales Renamed",
                                         self.azure, source_id=source_id)
        self.assertEqual(again, source_id)
        saved = store.get_client_source(self.account_id, source_id)
        self.assertEqual(saved["name"], "Sales Renamed")
        self.assertEqual(saved["db_config_id"], self.azure)


class TestADomainCanBeCheckedOnItsOwnConnection(RealDatabase):
    """
    The capability the whole of A2 exists for: checking one subject area's
    number against another's when the second area's tables are in a different
    warehouse.

    Corroboration (A3) already runs the same question under the secondary
    domain's scope. Without a connection to go with the scope it runs that
    query against the primary's warehouse, where the second area's tables do
    not exist — so the check fails as access_denied and corroboration reads
    "not checked" on exactly the workspaces it was built for.
    """

    def _domain(self, name: str, tables: list[str]) -> int:
        import store

        store.save_domain(self.account_id, name, tables=tables)
        return int(store.get_domain(self.account_id, name)["id"])

    def test_a_domain_with_its_own_source_resolves_to_that_connection(self):
        import store

        self._set_legacy_connection(self.snowflake)
        finance = self._domain("Finance", ["FIN.GL"])
        store.save_client_source(self.account_id, "Ledger", self.azure,
                                 domain_id=finance)
        self.assertEqual(
            store.db_config_for_domain_name(self.account_id, "Finance"), self.azure)

    def test_a_domain_with_no_source_of_its_own_says_so(self):
        import store

        self._set_legacy_connection(self.snowflake)
        self._domain("Sales", ["SALES.ORDERS"])
        # A default source EXISTS, and this must still answer None. Falling
        # back to it here would report "Sales lives on that connection" when
        # nobody said so, and the caller would then hand the second opinion an
        # override it did not need — which is how a redundant override becomes
        # a wrong one the day the default changes.
        store.save_client_source(self.account_id, "Warehouse", self.snowflake)
        self.assertIsNotNone(store.default_source(self.account_id))
        # None means "use whatever the caller was already using", not
        # "refuse": a workspace with one connection and several domains is
        # the normal case, and every domain in it shares that connection.
        self.assertIsNone(
            store.db_config_for_domain_name(self.account_id, "Sales"))

    def test_a_domain_that_does_not_exist_says_so(self):
        import store

        for name in ("Nonexistent", "", None):
            with self.subTest(name=name):
                self.assertIsNone(
                    store.db_config_for_domain_name(self.account_id, name))

    # Two properties of db_config_for_domain_name are NOT tested here, and
    # deliberately: the tenant scoping of its get_domain lookup, and its
    # blank-name guard. Both are redundant with what runs after them —
    # source_for_domain is itself account-scoped, so a domain id found across
    # the tenant boundary still resolves to no source, and a blank name finds
    # no domain either way. Removing either changes nothing observable, so a
    # test for them could only be written by contriving a state the product
    # cannot reach. They stay in the code as cheap defence; claiming coverage
    # for them would be claiming coverage that cannot fail.

    def test_another_tenants_domain_is_not_reachable_by_name(self):
        import store

        store.upsert_client("other-tenant", "portal")
        store.save_domain("other-tenant", "Finance", tables=["FIN.GL"])
        theirs = int(store.get_domain("other-tenant", "Finance")["id"])
        store.save_client_source("other-tenant", "Ledger", self.azure,
                                 domain_id=theirs)
        # Same domain NAME, different tenant. Resolving by name across the
        # tenant boundary would run one customer's corroboration against
        # another customer's warehouse.
        self.assertIsNone(
            store.db_config_for_domain_name(self.account_id, "Finance"))

    def test_the_executor_runs_against_the_connection_it_was_handed(self):
        # The seam A2 turns on. _execute_with_policy is a closure over the
        # primary db_cfg; the override is what lets the second opinion reach
        # a different warehouse, and getting it wrong is silent — the query
        # runs, against the wrong database, and returns a confident number.
        import inspect

        import core.query_pipeline as pipeline

        source = inspect.getsource(pipeline._handle_query_impl)
        self.assertIn("target = db_cfg_override or db_cfg", source)
        self.assertIn('target["credentials"],', source)
        self.assertIn('target["db_type"],', source)
        # And nothing below it still reaches past the override.
        executor = source[source.index("def _execute_with_policy("):]
        executor = executor[:executor.index("return execute_governed_query(") + 600]
        self.assertNotIn('db_cfg["credentials"]', executor)

    def test_the_second_opinion_is_told_which_connection_to_use(self):
        import inspect

        import core.query_pipeline as pipeline

        source = inspect.getsource(pipeline._handle_query_impl)
        self.assertIn("db_cfg_override=_second_db_cfg", source)
        self.assertIn("store.db_config_for_domain_name(", source)

    def test_the_trace_records_which_connection_confirmed_it(self):
        import inspect

        import core.query_pipeline as pipeline

        # "Confirmed against Finance" means something different when Finance
        # is a different database, and a trace that does not say so cannot
        # tell the two apart afterwards.
        source = inspect.getsource(pipeline._handle_query_impl)
        self.assertIn('"connection": (_second_db_cfg or {}).get("name") or "primary"',
                      source)


class TestNoCrossSourceSqlIsPossibleYet(RealDatabase):
    """
    The rule the whole design rests on: two connections mean two governed
    executions and a local combine over released rows, never one query.

    A warehouse cannot join to a warehouse it cannot see, and a plan that
    emitted SQL naming tables from two connections would be a plan no
    validator on either side could check. Nothing in this stage can produce
    one — the resolver returns a single db_config_id — and this test exists to
    fail loudly if that ever stops being true.
    """

    def test_the_resolver_returns_one_connection_or_none(self):
        import store

        store.save_client_source(self.account_id, "Sales", self.snowflake)
        store.save_client_source(self.account_id, "Finance", self.azure)
        for kwargs in ({}, {"domain_id": 1}, {"source_id": None}):
            with self.subTest(**kwargs):
                resolved = store.resolve_db_config_id(self.account_id, **kwargs)
                self.assertIsInstance(resolved, int)

    def test_get_client_db_returns_one_config_or_none(self):
        import store

        store.save_client_source(self.account_id, "Sales", self.snowflake)
        store.save_client_source(self.account_id, "Finance", self.azure)
        resolved = self._resolved_config_id()
        # One id, not a list of them. Two connections mean two governed
        # executions and a local combine, never one query.
        self.assertIsInstance(resolved, int)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
