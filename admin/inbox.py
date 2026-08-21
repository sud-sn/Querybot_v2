"""What needs the operator today, across every client.

The dashboard used to open on six totals — clients, ready, queries, cost,
platforms, databases — none of which is a thing anyone can act on. The work
waiting for an operator lived one client and two clicks away: access requests
under Access, flagged answers under Quality, conflicts under Model Health, a
half-finished setup visible only by reading a state badge. So the console
answered "how big is this deployment" when the question on arrival is "what is
waiting for me".

This turns the counts in store/admin_inbox.py into a ranked list, each entry
carrying the link to the one page where the work can be done.

GROUPED BY SIGNAL, NOT BY CLIENT. One row per client per signal does not
survive contact with a real fleet: the first build of this produced 63 rows
across 48 clients, 57 of them the same missing-database item, which is a wall
rather than an inbox. A signal affecting many clients collapses into a single
row that names the first few and counts the rest, so the list stays about as
long as there are kinds of problem — seven at most — however many clients exist.

Ranking is by who is blocked, not by count:

  action  someone or something is stopped until a human acts — a person waiting
          for access, a model the compiler will not trust, a client that cannot
          answer anything yet
  warn    quality is decaying and will not recover on its own — wrong answers
          nobody has reviewed, field mappings awaiting a decision
  info    worth knowing, safe to leave — a warning-level conflict backlog

Counts are only ever joined against live clients. The conflict and access tables
outlive the clients that produced them — this deployment's tables hold rows for
a dozen accounts that no longer exist — and showing those as work would make the
inbox untrustworthy on first read.
"""

from __future__ import annotations

import store

# Rank order for sorting. Lower sorts first.
_SEVERITY_RANK = {"action": 0, "warn": 1, "info": 2}

# How many client names a grouped row shows before collapsing to "+N more".
_NAMES_SHOWN = 4

# States where the client cannot answer a question yet, and what to do about it.
# *_BUILDING is deliberately absent: a build in flight is not waiting on anyone.
_INCOMPLETE_SETUP = {
    "NEW": ("setup-discover", "Schema not discovered yet", "Run discovery"),
    "SCHEMA_READY": ("setup-kb", "Knowledge base not built", "Build the KB"),
}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def build_inbox(clients: list[dict], db_ids: set[str] | None = None) -> list[dict]:
    """Rank the work waiting across ``clients``, grouped by signal.

    ``db_ids`` is the set of configured database ids, used to spot a client
    pointing at a database that no longer exists — which reads as "assigned" on
    the client row but leaves every page needing a connection inert.
    """
    live = {c["account_id"]: c for c in clients if c.get("account_id")}
    if not live:
        return []

    access = store.pending_access_by_account()
    flagged = store.flagged_answers_by_account()
    metric_proposals = store.pending_metric_proposals_by_account()
    errors = store.open_conflicts_by_account("ERROR")
    all_conflicts = store.open_conflicts_by_account()
    semantic = {
        row["account_id"]: int(row.get("pending") or 0)
        for row in (store.semantic_feedback_pending_summary() or {}).get("clients", [])
        if row.get("account_id")
    }

    # kind -> group. Insertion order is irrelevant; the sort at the end decides.
    groups: dict[str, dict] = {}

    def hit(kind, severity, label, detail, path, cta, account_id, count=0):
        group = groups.setdefault(kind, {
            "kind": kind, "severity": severity, "label": label,
            "detail": detail, "cta": cta, "path": path,
            "members": [], "total": 0,
        })
        client = live[account_id]
        group["members"].append({
            "account_id": account_id,
            "client_name": client.get("client_name") or account_id,
            "count": count,
            # Every client is its own link to the page that fixes it. A grouped
            # row pointing at the client list would land the operator in a list
            # of everything and leave them to find the affected ones again.
            "href": f"/admin/clients/{account_id}{path}",
        })
        group["total"] += count or 1

    for account_id, client in live.items():
        # ── Someone is waiting ───────────────────────────────────────────────
        n = access.get(account_id, 0)
        if n:
            hit("access", "action", "waiting for access",
                "people who asked to use the bot and have not heard back",
                "/pending-users", "Review requests", account_id, n)

        # ── The client cannot answer anything ────────────────────────────────
        assigned = client.get("db_config_id")
        if not assigned:
            # Only when there is something to assign. With no databases
            # configured at all, every client is unassigned as a consequence, and
            # "assign a database" is advice the operator cannot act on -- the
            # dashboard's own "No database configured" alert is the real item.
            if db_ids:
                hit("setup-db", "action", "no database assigned",
                    "nothing can be queried until a database is connected",
                    "/settings", "Assign a database", account_id)
        elif db_ids is not None and assigned not in db_ids:
            hit("setup-db-missing", "action", "pointing at a database that no longer exists",
                "the client looks configured but every query will fail",
                "/settings", "Reassign a database", account_id)
        elif client.get("state") in _INCOMPLETE_SETUP:
            kind, label, cta = _INCOMPLETE_SETUP[client["state"]]
            hit(kind, "action", label,
                "the client cannot answer questions yet",
                "/setup", cta, account_id)

        # ── The model is not trustworthy ─────────────────────────────────────
        n = errors.get(account_id, 0)
        if n:
            hit("conflict-error", "action", "unresolved model errors",
                "the compiler will not trust the model until these are resolved",
                "/model-health/conflicts", "Resolve conflicts", account_id, n)

        # ── Quality is decaying ──────────────────────────────────────────────
        n = flagged.get(account_id, 0)
        if n:
            hit("flagged", "warn", "flagged answers awaiting review",
                "users marked these wrong and nothing improves until someone looks",
                "/learning-queue", "Review answers", account_id, n)

        n = semantic.get(account_id, 0)
        if n:
            hit("semantic", "warn", "field mappings awaiting a decision",
                "the semantic layer cannot settle these names on its own",
                "/kb#semantic-feedback", "Review mappings", account_id, n)

        n = metric_proposals.get(account_id, 0)
        if n:
            # "warn", not "action": whoever asked already has their answer --
            # the logic ran in their own thread. What is pending is only
            # whether it becomes available to everyone else.
            hit("metric-proposal", "warn", "metric definitions awaiting approval",
                "users composed these in chat and are waiting for them to become shared",
                "/metrics#proposals", "Review metric requests", account_id, n)

        # ── Worth knowing ────────────────────────────────────────────────────
        n = all_conflicts.get(account_id, 0) - errors.get(account_id, 0)
        if n > 0:
            hit("conflict-warn", "info", "model warnings",
                "no action required, but the backlog is worth a look",
                "/model-health/conflicts", "View conflicts", account_id, n)

    items = []
    for group in groups.values():
        members = sorted(
            group["members"],
            key=lambda m: (-m["count"], m["client_name"].lower()),
        )
        single = len(members) == 1
        # Signals that carry a real count (5 flagged answers) read differently
        # from ones that are only ever "this client is in a bad state", where the
        # total is just how many clients are in it.
        counted = any(m["count"] for m in members)

        items.append({
            "kind": group["kind"],
            "severity": group["severity"],
            "label": group["label"],
            "detail": group["detail"],
            "cta": group["cta"],
            "total": group["total"],
            "counted": counted,
            "client_count": len(members),
            "members": members,
            "shown": members[:_NAMES_SHOWN],
            "more": max(0, len(members) - _NAMES_SHOWN),
            # A one-client signal gets a row-level link straight to its fix; with
            # several, the per-client links are the actions and a row-level one
            # would have nowhere honest to point.
            "href": members[0]["href"] if single else "",
            "single": single,
        })

    items.sort(key=lambda i: (
        _SEVERITY_RANK.get(i["severity"], 9),
        -i["total"],
        i["label"],
    ))
    return items


def inbox_summary(items: list[dict]) -> dict[str, int]:
    """Headline counts for the dashboard's inbox heading."""
    summary = {"action": 0, "warn": 0, "info": 0, "clients": 0}
    for item in items:
        if item["severity"] in summary:
            summary[item["severity"]] += 1
    summary["clients"] = len({
        member["account_id"] for item in items for member in item["members"]
    })
    return summary
