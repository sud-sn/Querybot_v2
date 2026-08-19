"""
tests/test_admin_inbox.py

The admin dashboard opened on six totals — clients, ready, queries, cost,
platforms, databases — none of which is a thing anyone can act on. The work
waiting for an operator lived one client and two clicks away, so the honest
answer to "what needs me today" was that nobody knew.

These exercise admin/inbox.py and the dashboard route itself, because the first
build of this feature was correct per-client and useless in aggregate: it emitted
63 rows across 48 clients, 57 of them the same missing-database item. Row count
under fleet growth is the property that matters, and only a test that feeds it a
fleet can see it.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from admin.inbox import build_inbox, inbox_summary

ROOT = Path(__file__).resolve().parents[1]

_DB = {"db-1"}


# Patch targets are "admin.inbox.store.*", not "store.*". Somewhere earlier in
# the full suite a test replaces sys.modules["store"], so patching "store.X"
# binds to a module object that admin/inbox.py is not the one holding -- the
# patch applies to nothing, build_inbox reads the real database, and every
# assertion here fails only when run alongside the rest of the suite. Resolving
# through admin.inbox reaches whichever object that module actually uses.


def _clients(n: int, *, state: str = "READY", db: str | None = "db-1", prefix: str = "c"):
    return [
        {"account_id": f"{prefix}{i}", "client_name": f"Client {i}",
         "state": state, "db_config_id": db}
        for i in range(n)
    ]


class _Signals:
    """Context manager: no signals except the ones named."""

    def __init__(self, access=None, flagged=None, conflicts=None, errors=None, semantic=None):
        self.access = access or {}
        self.flagged = flagged or {}
        self.conflicts = conflicts or {}
        self.errors = errors or {}
        self.semantic = semantic or {}
        self._patches: list = []

    def _by_severity(self, severity: str = ""):
        return dict(self.errors) if severity == "ERROR" else dict(self.conflicts)

    def __enter__(self):
        self._patches = [
            patch("admin.inbox.store.pending_access_by_account", return_value=self.access),
            patch("admin.inbox.store.flagged_answers_by_account", return_value=self.flagged),
            patch("admin.inbox.store.open_conflicts_by_account", side_effect=self._by_severity),
            patch("admin.inbox.store.semantic_feedback_pending_summary", return_value={
                "clients": [{"account_id": k, "pending": v} for k, v in self.semantic.items()]
            }),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


# ── The property the first build got wrong ───────────────────────────────────

@pytest.mark.parametrize("fleet", [1, 5, 48, 500])
def test_row_count_does_not_grow_with_the_fleet(fleet):
    """One row per client per signal is a wall, not an inbox. The rows are kinds
    of problem, so the list length is bounded by how many kinds exist."""
    clients = _clients(fleet, db=None)
    with _Signals(
        access={c["account_id"]: 2 for c in clients},
        errors={c["account_id"]: 1 for c in clients},
        conflicts={c["account_id"]: 1 for c in clients},
        flagged={c["account_id"]: 3 for c in clients},
    ):
        items = build_inbox(clients, _DB)

    assert len(items) <= 8, (
        f"{fleet} clients each with 4 problems produced {len(items)} rows; the "
        f"inbox must group by signal, not by client"
    )
    # And the counts must still be complete, not truncated to what is displayed.
    access = next(i for i in items if i["kind"] == "access")
    assert access["total"] == 2 * fleet
    assert access["client_count"] == fleet


def test_a_grouped_row_names_a_few_clients_and_counts_the_rest():
    clients = _clients(30, db=None)
    with _Signals(access={c["account_id"]: 1 for c in clients}):
        row = next(i for i in build_inbox(clients, _DB) if i["kind"] == "access")

    assert len(row["shown"]) == 4, "a grouped row should name only a handful"
    assert row["more"] == 26, f"expected the remaining 26 counted, got {row['more']}"
    assert row["client_count"] == 30


# ── Trust ────────────────────────────────────────────────────────────────────

def test_counts_for_dead_accounts_are_ignored():
    """The conflict and access tables outlive the clients that produced them.
    This deployment's tables hold rows for accounts that no longer exist, and
    presenting those as work would make the inbox untrustworthy on first read."""
    clients = _clients(2)
    with _Signals(
        access={"c0": 1, "ghost-account": 9},
        errors={"long-deleted": 40},
        conflicts={"long-deleted": 40},
    ):
        items = build_inbox(clients, _DB)

    accounts = {m["account_id"] for i in items for m in i["members"]}
    assert accounts == {"c0"}, f"work attributed to dead accounts: {accounts - {'c0'}}"
    assert not any(i["kind"].startswith("conflict") for i in items), (
        "a deleted account's conflicts became a row"
    )


def test_every_link_points_at_the_page_that_fixes_the_item():
    clients = _clients(3)
    with _Signals(
        access={"c0": 1},
        flagged={"c1": 2},
        semantic={"c2": 1},
        errors={"c0": 1},
        conflicts={"c0": 1},
    ):
        items = build_inbox(clients, _DB)

    assert items, "no items built"
    for item in items:
        for member in item["members"]:
            assert member["href"].startswith(f"/admin/clients/{member['account_id']}/"), (
                f"{item['kind']}: {member['href']} is not scoped to its client"
            )
        if item["single"]:
            assert item["href"] == item["members"][0]["href"]
        else:
            # No row-level link when several clients are involved: there is no
            # honest single destination, and the per-client links are the actions.
            assert item["href"] == "", (
                f"{item['kind']} has {item['client_count']} clients but still a "
                f"row-level link to {item['href']}"
            )


# ── Ranking ──────────────────────────────────────────────────────────────────

def test_blocked_work_outranks_decaying_work_which_outranks_the_rest():
    clients = _clients(4)
    with _Signals(
        access={"c0": 1},               # action
        flagged={"c1": 99},             # warn, much larger count
        errors={"c2": 1},               # action
        conflicts={"c2": 1, "c3": 50},  # c3 is warning-level only -> info
    ):
        items = build_inbox(clients, _DB)

    order = [i["severity"] for i in items]
    assert order == sorted(order, key=lambda s: {"action": 0, "warn": 1, "info": 2}[s]), (
        f"severity order broken: {order} — a big warn count must not outrank a "
        f"small blocker"
    )
    assert order[0] == "action"


# ── The database signals ─────────────────────────────────────────────────────

def test_no_advice_to_assign_a_database_when_none_exists():
    """Every client is unassigned as a consequence of there being no databases at
    all, and "assign a database" is then advice the operator cannot act on. The
    dashboard's own "No database configured" alert is the real item."""
    clients = _clients(5, db=None)
    with _Signals():
        assert not [i for i in build_inbox(clients, set()) if i["kind"] == "setup-db"]
        assert [i for i in build_inbox(clients, _DB) if i["kind"] == "setup-db"], (
            "with a database configured, unassigned clients are real work"
        )


def test_a_client_pointing_at_a_deleted_database_is_its_own_signal():
    """This reads as "assigned" on the client row while every query fails, so it
    cannot share a row with the never-assigned case."""
    clients = _clients(1, db="db-that-was-deleted")
    with _Signals():
        items = build_inbox(clients, _DB)

    kinds = {i["kind"] for i in items}
    assert "setup-db-missing" in kinds, kinds
    assert "setup-db" not in kinds


@pytest.mark.parametrize("state,kind", [
    ("NEW", "setup-discover"),
    ("SCHEMA_READY", "setup-kb"),
])
def test_a_client_that_cannot_answer_yet_is_surfaced(state, kind):
    with _Signals():
        items = build_inbox(_clients(1, state=state), _DB)
    assert {i["kind"] for i in items} == {kind}


@pytest.mark.parametrize("state", ["KB_BUILDING", "SCHEMA_BUILDING", "READY"])
def test_a_build_in_flight_is_not_reported_as_waiting_on_anyone(state):
    with _Signals():
        assert build_inbox(_clients(1, state=state), _DB) == []


# ── The empty case ───────────────────────────────────────────────────────────

def test_a_clean_deployment_produces_no_items():
    with _Signals():
        items = build_inbox(_clients(3), _DB)
    assert items == []
    assert inbox_summary(items) == {"action": 0, "warn": 0, "info": 0, "clients": 0}


def test_no_clients_at_all_is_not_an_error():
    with _Signals():
        assert build_inbox([], _DB) == []


# ── The rendered page ────────────────────────────────────────────────────────

def _render_dashboard():
    from admin import routes

    class FakeReq:
        def __init__(self):
            self.url = type("U", (), {"path": "/admin/"})()
            self.cookies = {}
            self.headers = {}
            self.query_params = {}
            self.session = {}

    with patch.object(routes, "_is_auth", return_value=True), \
         patch.object(routes, "_first_run", return_value=False):
        resp = asyncio.run(routes.dashboard(FakeReq()))
    return resp.body.decode("utf-8", "replace")


def test_the_dashboard_renders_the_inbox_above_the_totals():
    """Totals are not work. If the inbox renders below them it is decoration."""
    html = _render_dashboard()
    assert 'class="inbox"' in html, "the dashboard route no longer passes an inbox"
    if 'class="metrics"' in html:
        assert html.index('class="inbox"') < html.index('class="metrics"'), (
            "the inbox renders below the totals"
        )


def test_severity_is_not_carried_by_colour_alone():
    """A colour-blind operator, or a greyscale print, must still see the ranking."""
    html = _render_dashboard()
    if 'class="inbox-item' not in html:
        pytest.skip("this deployment's inbox is empty; nothing to check")

    for row in re.findall(r'<li class="inbox-item inbox-item--(\w+)">(.*?)</li>', html, re.S):
        severity, body = row
        word = re.search(r'class="inbox-sev">\s*([^<]+?)\s*<', body)
        assert word, f"{severity} row has no severity word, only a colour"
        assert word.group(1) in {"Blocked", "Degrading", "FYI"}, word.group(1)


def test_the_inbox_rail_colour_is_never_the_only_difference():
    """Paired with the test above: the rail must exist, but the CSS must not be
    the sole carrier."""
    css = (ROOT / "static" / "css" / "admin.css").read_text(encoding="utf-8")
    for severity in ("action", "warn", "info"):
        assert re.search(rf"\.inbox-item--{severity}\s*\{{", css), (
            f"no rail styling for {severity}"
        )
        assert re.search(rf"\.inbox-item--{severity}\s+\.inbox-sev", css), (
            f"{severity} has a rail but no styled severity word"
        )
