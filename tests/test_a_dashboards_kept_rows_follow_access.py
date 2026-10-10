"""What a dashboard keeps for its owner follows the owner's access.

A scheduled refresh keeps a source's rows, encrypted, for the dashboard's owner, released under the access the
owner had then: their role, group and tables, and the workspace's row rules, masking, purposes and attestations.
They were served until they expired whatever changed meanwhile -- a narrower group, a new row rule, a
deactivation -- and a refresh failing after such a change (the warehouse refusing an owner it no longer serves)
served them with no end. Every change to that access now clears the rows it governs, in the same write: one
person's, a group's members', or the whole workspace's. Access that ends by itself, with nothing written when it
does -- an attestation's term, an emergency grant -- bounds how long rows are kept.

A scratch store holds the people, dashboards, sources and kept rows; the warehouse is the only stand-in.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import core.dashboard_refresh as refresh
import store

ROWS = [{"REGION": "NORTH", "UNITS": 840.0}]
SQL = "SELECT REGION, SUM(UNITS) AS UNITS FROM SALES GROUP BY REGION"


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _stamp(at: datetime) -> str:
    return at.strftime("%Y-%m-%d %H:%M:%S")


def _workspace(monkeypatch=None):
    """A workspace with two groups and three owners -- two in the first group, one in the second -- each with a
    daily dashboard of one source whose rows are kept."""
    account_id = f"kept-rows-{uuid.uuid4().hex[:10]}"
    store.upsert_client(account_id, "portal")
    east, west = (store.create_group(account_id, name) for name in ("East", "West"))
    db_id = store.save_db_config("azure_sql", f"Warehouse {account_id}", {
        "server": "dw.example.net", "database": "DW", "user": "reader", "password": "a-password"})
    owners = {}
    for name, group in (("first", east), ("second", east), ("third", west)):
        user_id, _ = store.create_user(account_id, name.title(), f"{name}@{account_id}.example.com",
                                       password="a-long-password-1", role="analyst", group_id=group)
        board = store.create_dashboard(account_id, user_id, "t", f"Units, {name}")
        store.update_dashboard_controls(board["id"], user_id, account_id, refresh_schedule="daily")
        source = store.create_data_source(board["id"], user_id, account_id, name="Units", question="Units by region",
                                          sql_query=SQL, db_config_id=db_id)
        owners[name] = SimpleNamespace(id=user_id, board=board, source=source)
    return SimpleNamespace(account_id=account_id, east=east, west=west, owners=owners, db_id=db_id)


def _keep(source: dict, ttl: int = 86400) -> None:
    store.save_source_cache(source, ROWS, policy_version=0, contract_version="", ttl_seconds=ttl)


def _kept(workspace) -> set[str]:
    """Whose rows are kept, by owner name."""
    with store.get_db() as conn:
        users = {int(r["user_id"]) for r in conn.execute(
            "SELECT user_id FROM dashboard_source_cache WHERE account_id=? AND rows_encrypted<>''",
            (workspace.account_id,))}
    return {name for name, owner in workspace.owners.items() if owner.id in users}


@pytest.fixture
def two_workspaces():
    """This workspace, every owner's rows kept; and another workspace, its rows kept too."""
    here, there = _workspace(), _workspace()
    for workspace in (here, there):
        for owner in workspace.owners.values():
            _keep(owner.source)
    return here, there


EVERYONE = {"first", "second", "third"}


# Changes to one person, a group, or the whole workspace; and whose rows each ends.
CHANGES = {
    "a new group": (lambda w: store.update_user(w.owners["first"].id, group_id=w.west), {"first"}),
    "a new role": (lambda w: store.update_user(w.owners["first"].id, role="admin"), {"first"}),
    "deactivated": (lambda w: store.update_user(w.owners["first"].id, is_active=0), {"first"}),
    "deleted": (lambda w: store.delete_user(w.owners["first"].id), {"first"}),
    "their own tables": (lambda w: store.set_user_extra_tables(w.owners["first"].id, w.account_id, ["SALES"]),
                         {"first"}),
    "their group's tables": (lambda w: store.set_group_tables(w.east, w.account_id, ["SALES"]), {"first", "second"}),
    "their group deleted": (lambda w: store.delete_group(w.east), {"first", "second"}),
    "a row rule": (lambda w: store.replace_row_policies(w.account_id, 0, [
        {"name": "North only", "subject_type": "group", "subject_id": str(w.east), "table_fqn": "SALES",
         "condition": {"field": "REGION", "op": "=", "value": "NORTH"}}]), EVERYONE),
    "a policy rule": (lambda w: store.replace_policy_rules(w.account_id, 0, [
        {"name": "No personal data", "resource_pattern": "PII", "effect": "deny"}]), EVERYONE),
    "a column's classification": (lambda w: store.save_classification(
        w.account_id, "SALES", "CUSTOMER_EMAIL", sensitivity="pii", identifiability="direct", tags=["EMAIL"]),
        EVERYONE),
    "the purposes": (lambda w: store.replace_purposes(w.account_id, [
        {"purpose_key": "reporting", "name": "Reporting", "permissions": {"PII": ["query_execution"]}}]), EVERYONE),
    "an attestation signed": (lambda w: store.save_user_attestation(w.account_id, str(w.owners["first"].id)),
                              EVERYONE),
    "an emergency grant": (lambda w: store.create_break_glass_grant(
        w.account_id, str(w.owners["first"].id), "INC-1", "outage", ["SALES"], ["query_execution"],
        _stamp(_now() + timedelta(hours=2))), EVERYONE),
    "the enforcement mode": (lambda w: store.save_compliance_profile(w.account_id, enforcement_mode="enforce"),
                             EVERYONE),
    "a policy version made active": (lambda w: store.activate_policy_version(
        w.account_id, store.create_policy_version(w.account_id, {"rules": []}, created_by="admin")), EVERYONE),
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_to_access_clears_the_rows_it_governs_and_no_others(two_workspaces, change):
    here, there = two_workspaces
    make, ended = CHANGES[change]
    make(here)
    assert _kept(here) == EVERYONE - ended
    assert _kept(there) == EVERYONE, "another workspace's rows were cleared"


def test_an_attestation_revoked_clears_the_rows(two_workspaces):
    here, there = two_workspaces
    attestation = store.save_user_attestation(here.account_id, str(here.owners["second"].id))
    for owner in here.owners.values():
        _keep(owner.source)
    assert store.revoke_user_attestation(here.account_id, attestation, revoked_by="admin")
    assert _kept(here) == set() and _kept(there) == EVERYONE


@pytest.mark.parametrize("change", [
    "a new name",
    "a language",
    "a password",
    "the egress posture",
    "the same enforcement mode",
])
def test_what_does_not_change_access_keeps_the_rows(two_workspaces, change):
    here, _ = two_workspaces
    first = here.owners["first"].id
    {
        "a new name": lambda: store.update_user(first, name="First Renamed"),
        "a language": lambda: store.set_user_language(first, "fr"),
        "a password": lambda: store.change_password(first, "another-long-password", is_temp=False),
        "the egress posture": lambda: store.save_compliance_profile(here.account_id, egress_posture="strict"),
        "the same enforcement mode": lambda: store.save_compliance_profile(
            here.account_id, enforcement_mode=store.get_compliance_profile(here.account_id)["enforcement_mode"]),
    }[change]()
    assert _kept(here) == EVERYONE


def test_a_platform_user_approved_into_a_group_clears_their_rows(two_workspaces):
    here, _ = two_workspaces
    first = here.owners["first"]
    with store.get_db() as conn:
        conn.execute("UPDATE portal_user SET email=?, zoom_user_id=? WHERE id=?",
                     ("teams-first@platform.internal", "teams-first", first.id))
    _, pending = store.upsert_pending_user(here.account_id, "teams", "teams-first", "First", "{}")
    store.approve_pending_user(int(pending["id"]), here.account_id, here.west)
    assert store.get_user(first.id)["group_id"] == here.west
    assert _kept(here) == EVERYONE - {"first"}


# ── The owner's next view, after a change ────────────────────────────────────

@pytest.fixture
def warehouse(monkeypatch):
    """The warehouse: the rows it answers with and every query it ran."""
    state = SimpleNamespace(rows=list(ROWS), ran=[], ttl=86400)

    def governed(credentials, db_type, sql, *args, **kwargs):
        state.ran.append({"sql": sql, "allowed_tables": kwargs.get("allowed_tables")})
        return SimpleNamespace(rows=list(state.rows), sql=sql, decision=SimpleNamespace(cache_ttl_seconds=state.ttl))

    monkeypatch.setattr("core.compliance.governed_query.execute_governed_query", governed)
    return state


def _owners_chart(owner) -> dict:
    from portal.routes import _refresh_chart
    from tests.dashboard_render import CHART

    return _refresh_chart({**CHART, "data_source_id": owner.source["id"], "user_id": owner.id, "chart_type": "bar",
                           "sql_query": SQL}, store.get_db_config(owner.source["db_config_id"]), store.get_user(owner.id))


def test_after_a_change_the_owners_next_view_runs_under_the_access_they_have_now(warehouse):
    here = _workspace()
    owner = here.owners["first"]
    store.set_group_tables(here.east, here.account_id, ["SALES"])
    assert refresh.refresh_dashboard_source({**owner.source, "refresh_schedule": "daily"}) is True
    assert len(warehouse.ran) == 1
    assert _owners_chart(owner)["from_cache"] is True and len(warehouse.ran) == 1

    store.set_group_tables(here.east, here.account_id, ["RETURNS"])
    warehouse.rows = [{"REGION": "NORTH", "UNITS": 12.0}]
    chart = _owners_chart(owner)
    assert chart["from_cache"] is False and len(warehouse.ran) == 2
    assert warehouse.ran[-1]["allowed_tables"] == {"RETURNS"}, "the view ran under the old tables"
    # Kept again, under the access as it is now.
    again = _owners_chart(owner)
    assert again["from_cache"] is True and len(warehouse.ran) == 2


def test_after_a_change_the_scheduler_refreshes_the_source_at_once(warehouse):
    here = _workspace()
    owner = here.owners["first"]
    assert refresh.refresh_dashboard_source({**owner.source, "refresh_schedule": "daily"}) is True
    due = {s["id"] for s in store.list_due_dashboard_sources()}
    assert owner.source["id"] not in due
    store.update_user(owner.id, group_id=here.west)
    assert owner.source["id"] in {s["id"] for s in store.list_due_dashboard_sources()}


def test_a_shared_viewer_never_reads_the_owners_rows(warehouse):
    here = _workspace()
    owner, viewer = here.owners["first"], here.owners["third"]
    _keep(owner.source)
    result = refresh.execute_dashboard_source(owner.source, store.get_user(viewer.id))
    assert result.from_cache is False and len(warehouse.ran) == 1


# ── Access that ends by itself ───────────────────────────────────────────────

def _expires(owner) -> datetime:
    with store.get_db() as conn:
        row = conn.execute("SELECT expires_at FROM dashboard_source_cache WHERE source_id=?",
                           (owner.source["id"],)).fetchone()
    return datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S") if row else None


def _refresh(owner) -> None:
    assert refresh.refresh_dashboard_source({**owner.source, "refresh_schedule": "daily"}) is True


@pytest.mark.parametrize("access", ["attestation", "emergency grant"])
def test_rows_are_kept_no_longer_than_access_that_ends_by_itself(warehouse, access):
    here = _workspace()
    owner = here.owners["first"]
    ends = _now() + timedelta(minutes=30)
    if access == "attestation":
        store.save_user_attestation(here.account_id, str(owner.id), expires_at=_stamp(ends))
    else:
        store.create_break_glass_grant(here.account_id, str(owner.id), "INC-2", "outage", ["SALES"],
                                       ["query_execution"], _stamp(ends))
    _refresh(owner)
    assert abs((_expires(owner) - ends).total_seconds()) <= 2, "kept past the end of the access it was released under"


def test_an_attestation_ending_on_a_bare_date_ends_at_the_start_of_that_day(warehouse):
    here = _workspace()
    owner = here.owners["first"]
    day = (_now() + timedelta(days=2)).date()
    store.save_user_attestation(here.account_id, str(owner.id), expires_at=day.isoformat())
    warehouse.ttl = 7 * 86400
    _refresh(owner)
    start = datetime.combine(day, datetime.min.time())
    assert timedelta(0) <= start - _expires(owner) <= timedelta(seconds=2)


@pytest.mark.parametrize("access", ["attestation", "emergency grant"])
def test_rows_are_not_kept_when_that_access_ends_within_the_minute(warehouse, access):
    here = _workspace()
    owner = here.owners["first"]
    ends = _stamp(_now() + timedelta(seconds=30))
    if access == "attestation":
        store.save_user_attestation(here.account_id, str(owner.id), expires_at=ends)
    else:
        store.create_break_glass_grant(here.account_id, str(owner.id), "INC-3", "outage", ["SALES"],
                                       ["query_execution"], ends)
    _refresh(owner)
    assert _expires(owner) is None
    assert _owners_chart(owner)["row_count"] == 1, "the owner's view still answers, live"


@pytest.mark.parametrize("held", ["nothing", "an attestation with no end", "an attestation already over",
                                  "another person's grant", "a revoked attestation"])
def test_rows_are_kept_their_own_life_when_nothing_held_ends(warehouse, held):
    here = _workspace()
    owner, other = here.owners["first"], here.owners["second"]
    if held == "an attestation with no end":
        store.save_user_attestation(here.account_id, str(owner.id))
    elif held == "an attestation already over":
        store.save_user_attestation(here.account_id, str(owner.id), expires_at=_stamp(_now() - timedelta(days=1)))
    elif held == "another person's grant":
        store.create_break_glass_grant(here.account_id, str(other.id), "INC-4", "outage", ["SALES"],
                                       ["query_execution"], _stamp(_now() + timedelta(minutes=30)))
    elif held == "a revoked attestation":
        attestation = store.save_user_attestation(here.account_id, str(owner.id),
                                                  expires_at=_stamp(_now() + timedelta(minutes=30)))
        store.revoke_user_attestation(here.account_id, attestation)
    _refresh(owner)
    assert abs((_expires(owner) - (_now() + timedelta(days=1))).total_seconds()) <= 5
