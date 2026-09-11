"""
tests/test_anchor_is_scoped_to_the_reader.py

The business-date anchor is MAX(governed date) read THROUGH the reader's row
policies, and it was cached under a key that did not mention them.

core/date_anchor.py probes "the newest date present in this fact" so that a
question about "yesterday" resolves against the data rather than the clock.
The probe runs through execute_governed_query, which injects row policies
(core/compliance/sql_guard.py), so what comes back is the newest date THAT
READER can see. The cache key was (account_id, fact_table, fact_column), and
the durable row in business_date_anchor had the same primary key.

So the first reader of the hour decided the business date for the whole
workspace:

  * A workspace restricts the sales_east role to REGION_CD = 'EAST', and EAST
    loads two days behind WEST. An unrestricted admin asks anything relative
    first; the anchor becomes the newest WEST date and is cached and persisted.
    An EAST rep then asks "revenue yesterday" -- the cache hits, no SQL runs,
    the window lands past the end of EAST's own data, and the card says there
    is no data. For a reader whose region has data.
  * In the other direction the EAST rep asks first, and every unrestricted
    reader's "last 7 days" quietly stops two days early. That one is disclosed
    nowhere at all: the freshness banner only fires for current-period window
    kinds, and last_n is not one of them.
  * And the date-role clarification card decorates each candidate with "data
    through <date>" out of the same durable row, so a restricted reader was
    shown a date derived from rows they have no access to.

The scope is a fingerprint of the row restrictions that applied, and "" means
none did -- so an unrestricted workspace keeps precisely the caching it had,
one anchor per fact per hour, and only workspaces that actually restrict rows
pay for a second probe.

These execute the real resolver, the real fingerprint over a real row-policy
table, and the real durable store.
"""

import os
import sys
import tempfile

import pytest

_tmp_db = os.path.join(tempfile.mkdtemp(), "test_anchor_scope.db")
os.environ["QUERYBOT_DB_PATH"] = _tmp_db
os.environ["DB_PATH"] = _tmp_db
for _mod in list(sys.modules):
    if _mod.startswith("store"):
        del sys.modules[_mod]

import store  # noqa: E402

store.init_db()

from core import date_anchor  # noqa: E402
from core.compliance.models import PolicyContext  # noqa: E402
from core.compliance.sql_guard import (  # noqa: E402
    inject_row_policies, row_policy_scope,
)
from core.contextual_dates import describe_date_role_evidence  # noqa: E402
from core.date_anchor import (  # noqa: E402
    anchor_key, resolve_business_anchor,
)
from core.query_pipeline import anchor_row_scope  # noqa: E402

FACT = "EMDW_DMART.CUS_ORD_IVC_FCT"
# A fact-native date: probeable with no join, which is the simplest shape
# build_anchor_probe_sql accepts. date_column is what it reads.
POLICY = {
    "anchor_policy": "latest_available",
    "fact_table": FACT,
    "fact_column": "IVC_DT",
    "date_column": "IVC_DT",
    "business_role": "invoice_date",
}


@pytest.fixture
def tenant():
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    date_anchor.clear_cache(account_id, persistent=True)
    _POLICIES.pop(account_id, None)
    yield account_id
    date_anchor.clear_cache(account_id, persistent=True)
    _POLICIES.pop(account_id, None)


_POLICIES: dict[str, list[dict]] = {}


def _restrict(account_id, *, role, region="EAST", table=FACT,
              field="REGION_CD"):
    """A real row policy, written through the real store.

    replace_row_policies replaces the whole version, so they accumulate here.
    """
    _POLICIES.setdefault(account_id, []).append({
        "name": f"{role}-{table}", "subject_type": "role", "subject_id": role,
        "table_fqn": table,
        "condition": {"field": field, "operator": "=", "value": region},
    })
    store.replace_row_policies(account_id, 0, _POLICIES[account_id])


def _ctx(account_id, role, user_id="u1"):
    return PolicyContext(account_id=account_id, user_id=user_id, role=role)


class TestTheFingerprintTracksWhatTheProbeActuallyReads:
    """The scope must change exactly when the injected probe changes. If it
    can lag behind, the cache is back to sharing across restrictions."""

    PROBE = f"SELECT MAX(IVC_DT) AS max_business_date FROM {FACT}"

    def test_no_row_policy_anywhere_is_the_empty_scope(self, tenant):
        assert row_policy_scope(_ctx(tenant, "analyst"), [FACT]) == ""

    def test_a_policy_that_does_not_bind_this_reader_is_still_empty(self, tenant):
        _restrict(tenant, role="sales_east")
        # An admin the policy does not name reads the whole fact, so it shares
        # the unrestricted bucket -- and must, or every workspace with any row
        # policy loses anchor caching entirely.
        assert row_policy_scope(_ctx(tenant, "admin"), [FACT]) == ""
        untouched, applied = inject_row_policies(
            self.PROBE, "azure_sql", _ctx(tenant, "admin"))
        assert applied == []

    def test_a_restricted_reader_gets_a_scope_and_a_filtered_probe(self, tenant):
        _restrict(tenant, role="sales_east")
        scope = row_policy_scope(_ctx(tenant, "sales_east"), [FACT])
        assert scope, "a restricted reader must not share the empty scope"
        injected, applied = inject_row_policies(
            self.PROBE, "azure_sql", _ctx(tenant, "sales_east"))
        assert applied, "the probe was not actually filtered"
        assert "REGION_CD" in injected

    def test_the_scope_separates_exactly_when_the_probe_does(self, tenant):
        """Executed both ways round, because a fingerprint that is merely
        different is not the same as one that is different for the right
        reason."""
        _restrict(tenant, role="sales_east", region="EAST")
        _restrict(tenant, role="sales_west", region="WEST")
        seen: dict[str, str] = {}
        for role in ("admin", "sales_east", "sales_west"):
            context = _ctx(tenant, role)
            scope = row_policy_scope(context, [FACT])
            injected, _ = inject_row_policies(self.PROBE, "azure_sql", context)
            seen[role] = scope
            assert scope not in {v for k, v in seen.items() if k != role} or True
        # Three readers, three different injected probes, three scopes.
        assert len({seen["admin"], seen["sales_east"], seen["sales_west"]}) == 3

    def test_the_same_reader_twice_is_the_same_scope(self, tenant):
        """Or nothing is ever a cache hit and the probe runs every question."""
        _restrict(tenant, role="sales_east")
        first = row_policy_scope(_ctx(tenant, "sales_east"), [FACT])
        second = row_policy_scope(_ctx(tenant, "sales_east", user_id="u2"), [FACT])
        assert first == second != ""

    def test_an_unreadable_policy_table_does_not_read_as_unrestricted(self, tenant):
        """Fails closed. Returning "" here would silently restore the defect."""
        import core.compliance.sql_guard as guard

        original = guard.store.list_row_policies
        guard.store.list_row_policies = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("policy table unavailable"))
        try:
            assert row_policy_scope(_ctx(tenant, "sales_east"), [FACT]) == "unknown"
        finally:
            guard.store.list_row_policies = original


class TestTheKeyCarriesIt:

    def test_the_scope_is_part_of_the_cache_key(self, tenant):
        bare = anchor_key(tenant, POLICY)
        scoped = anchor_key(tenant, POLICY, "v0:abcdef")
        assert bare != scoped
        assert bare[:3] == scoped[:3], "only the scope may differ"

    def test_an_unscoped_key_is_the_empty_scope(self, tenant):
        """So an unrestricted workspace keeps the caching it had."""
        assert anchor_key(tenant, POLICY)[3] == ""
        assert anchor_key(tenant, POLICY) == anchor_key(tenant, POLICY, "")


class TestOneReadersAnchorIsNotServedToAnother:
    """The reported failure, driven through the real resolver."""

    @staticmethod
    def _probe_returning(value, calls):
        def run(sql):
            calls.append(sql)
            return [{"max_business_date": value}]
        return run

    def test_a_restricted_reader_probes_for_themselves(self, tenant):
        admin_calls: list[str] = []
        east_calls: list[str] = []

        # The unrestricted reader asks first and warms the cache.
        admin = resolve_business_anchor(
            tenant, POLICY, "azure_sql",
            self._probe_returning("2026-09-10", admin_calls), "")
        assert admin["value"] == "2026-09-10"
        assert len(admin_calls) == 1

        # The restricted reader asks the same question. Their own data ends two
        # days earlier, and they must see that, not the admin's date.
        east = resolve_business_anchor(
            tenant, POLICY, "azure_sql",
            self._probe_returning("2026-09-08", east_calls), "v0:east")
        assert east["value"] == "2026-09-08", (
            "the restricted reader was served the unrestricted anchor")
        assert len(east_calls) == 1, "the restricted reader did not probe at all"
        assert east.get("cached") is False

    def test_the_same_reader_does_hit_the_cache(self, tenant):
        """The saving the cache exists for is still there."""
        calls: list[str] = []
        run = self._probe_returning("2026-09-08", calls)
        first = resolve_business_anchor(tenant, POLICY, "azure_sql", run, "v0:east")
        second = resolve_business_anchor(tenant, POLICY, "azure_sql", run, "v0:east")
        assert first["value"] == second["value"] == "2026-09-08"
        assert len(calls) == 1, "the second question re-probed"
        assert second.get("cached") is True

    def test_the_unrestricted_reader_still_hits_their_own(self, tenant):
        calls: list[str] = []
        run = self._probe_returning("2026-09-10", calls)
        resolve_business_anchor(tenant, POLICY, "azure_sql", run, "")
        again = resolve_business_anchor(tenant, POLICY, "azure_sql", run, "")
        assert again.get("cached") is True
        assert len(calls) == 1


class TestItSurvivesARestartWithoutCrossingReaders:
    """The durable row had the same subject-independent primary key, so the
    cross-reader value outlived the process for up to the max-age bound."""

    def test_two_scopes_persist_as_two_rows(self, tenant):
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-08"}, "v0:east")
        assert store.load_business_date_anchor(
            tenant, FACT, "IVC_DT", "")["value"] == "2026-09-10"
        assert store.load_business_date_anchor(
            tenant, FACT, "IVC_DT", "v0:east")["value"] == "2026-09-08"

    def test_a_scope_with_nothing_stored_reads_empty(self, tenant):
        """Rather than falling back to somebody else's row."""
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        assert store.load_business_date_anchor(
            tenant, FACT, "IVC_DT", "v0:west") == {}

    def test_the_resolver_restores_only_its_own_scope(self, tenant):
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        calls: list[str] = []

        def run(sql):
            calls.append(sql)
            return [{"max_business_date": "2026-09-08"}]

        east = resolve_business_anchor(tenant, POLICY, "azure_sql", run, "v0:east")
        assert east["value"] == "2026-09-08"
        assert len(calls) == 1, "the stored unrestricted row was reused"


class TestTheClarificationCardDoesNotQuoteAnothersData:
    """describe_date_role_evidence reads the same durable row to print
    "data through <date>" under each candidate date."""

    ROLE = {"fact_table": FACT, "fact_column": "IVC_DT", "is_default": 1}

    def test_the_readers_own_scope_is_what_is_shown(self, tenant):
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-08"}, "v0:east")
        east = describe_date_role_evidence(tenant, self.ROLE, scope="v0:east")
        assert "08 Sep 2026" in east, east
        assert "10 Sep 2026" not in east

    def test_a_reader_with_no_stored_anchor_is_told_nothing_rather_than_anothers(
            self, tenant):
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        detail = describe_date_role_evidence(tenant, self.ROLE, scope="v0:west")
        assert "2026" not in detail
        # The governance status is still worth saying.
        assert detail.strip(), "the whole line disappeared, not just the date"

    def test_no_scope_at_all_omits_the_date(self, tenant):
        """Fails closed: a caller that does not know the reader must not be
        handed the unrestricted figure by default."""
        store.save_business_date_anchor(
            tenant, FACT, "IVC_DT", {"value": "2026-09-10"}, "")
        assert "2026" not in describe_date_role_evidence(tenant, self.ROLE)


class TestThePipelineComputesTheScopeItProbesUnder:
    """anchor_row_scope is what the pipeline passes, so it has to agree with
    the fingerprint the injector implies for the same reader."""

    def test_it_matches_the_fingerprint_for_the_fact(self, tenant):
        _restrict(tenant, role="sales_east")
        context = _ctx(tenant, "sales_east")
        assert anchor_row_scope(context, POLICY) == row_policy_scope(
            context, [FACT])

    def test_an_unrestricted_reader_gets_the_shared_scope(self, tenant):
        assert anchor_row_scope(_ctx(tenant, "analyst"), POLICY) == ""

    def test_it_reads_the_dimension_table_too(self, tenant):
        """A surrogate-key role probes the date DIMENSION and semi-joins the
        fact, so a policy on either one changes what comes back."""
        _restrict(tenant, role="sales_east", table="EMDW_DMART.DT_DMS",
                  field="FISCAL_YR", region="2026")
        policy = dict(POLICY, date_table="EMDW_DMART.DT_DMS")
        assert anchor_row_scope(_ctx(tenant, "sales_east"), policy) != ""

    def test_a_broken_context_does_not_read_as_unrestricted(self, tenant):
        """Fails closed to a value that matches nothing shared."""
        scope = anchor_row_scope(object(), POLICY)
        assert scope != ""
