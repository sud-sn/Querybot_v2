"""
tests/test_stale_answers_and_suggestions.py

Two faults reported from the field, and the reason neither was caught.

FAULT 1 — "ask today, ask again tomorrow, same answer". A dev database had its
date range moved from 2025-04-17 back to 2024-08-11 and the product kept
answering on the old range. The anchor itself was right: "today" is resolved
against the newest date the DATA holds, never the wall clock. What was wrong was
how long that answer was trusted — a persisted anchor was served for up to
86400s, and measured on the live path a 23.9h-old anchor came back with zero
probes. A full day of confident answers against a range the warehouse no longer
had, visually identical to correct ones.

Nothing invalidated it when the data changed, either. There WAS a route to clear
it by hand (admin/routes.py date-roles/refresh-anchor) and nothing in the product
ever called it — the test that was supposed to cover that asserted on routes.py
source text, so a live endpoint with no caller read as covered.

FAULT 2 — suggested questions that fail when clicked. The gate meant to prevent
this only ever detected one narrow defect ("jointless twins"). On any graph
without one it returned True for every question AND marked itself verified, so
it looked like it was working exactly when it was doing nothing. Metric-registry
suggestions were not gated at all.

These tests drive the real functions. The reachability gate in particular is
tested through _graph_reachability_check rather than by reading the file, because
reading the file is how the previous version's emptiness went unnoticed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import core.date_anchor as anchor
import core.suggestions as suggestions

FACT = "DB.dbo.SALES_FCT"
# A fact-native business date: no dimension, so the probe is a plain MAX over
# the fact's own governed date column. date_column is what makes the policy
# physically complete enough to probe at all.
POLICY = {
    "fact_table": FACT,
    "fact_column": "INV_DT",
    "date_column": "INV_DT",
    "anchor_policy": "latest_available",
    "kind": "today",
}


# ── Fault 1: the anchor must not outlive the data it describes ───────────────

def _stored(age_hours: float, value: str = "2025-04-17") -> dict:
    stamp = (datetime.utcnow() - timedelta(hours=age_hours)).isoformat(sep=" ")
    return {
        "value": value, "fact_table": FACT, "fact_column": "INV_DT",
        "source": "probed_from_fact_rows", "resolved_at": stamp,
    }


@pytest.fixture(autouse=True)
def _clean():
    anchor.clear_cache()
    yield
    anchor.clear_cache()


def test_the_warehouse_is_re_read_once_the_stored_anchor_ages_out():
    """The user's scenario, driven through the real entry point rather than the
    private helper the old tests reached for."""
    probes: list[str] = []

    def probe(_sql):
        probes.append(_sql)
        return [{"anchor": "2024-08-11"}]

    # Just inside the bound: served from the store, no probe, OLD value.
    with patch("store.load_business_date_anchor", return_value=_stored(0.2)):
        fresh = anchor.resolve_business_anchor("acct", POLICY, "azure_sql", probe)
    assert fresh.get("value") == "2025-04-17"
    assert fresh.get("cached") is True
    assert not probes, "a stored anchor inside the bound should not re-probe"

    anchor.clear_cache()

    # Past the bound: the warehouse is asked again and the NEW range wins.
    with patch("store.load_business_date_anchor", return_value=_stored(5)), \
         patch("store.save_business_date_anchor", return_value=None):
        aged = anchor.resolve_business_anchor("acct", POLICY, "azure_sql", probe)
    assert probes, "an aged anchor must trigger a fresh probe"
    assert aged.get("value") == "2024-08-11", (
        "the moved date range was not picked up, which is the reported fault"
    )


def test_the_staleness_bound_is_hours_not_a_day():
    """86400 was the default. It assumes data only ever moves forward, on a
    schedule we guessed."""
    assert anchor._DEFAULT_MAX_AGE_SECONDS <= 3600, (
        f"the persisted anchor may be served for "
        f"{anchor._DEFAULT_MAX_AGE_SECONDS / 3600:.0f}h after the data changed"
    )


def test_discovery_invalidates_the_anchor():
    """Discovery is the product's own signal that the source changed. Without
    this the stored anchor survives a reload until it ages out, and the only
    thing that cleared it was a route nothing called."""
    import inspect
    import admin.routes as routes

    source = inspect.getsource(routes.admin_discover_schema)
    assert "clear_cache" in source and "persistent=True" in source, (
        "schema discovery does not invalidate the business-date anchor, so a "
        "reload is not noticed until the age bound expires"
    )


def test_clearing_the_anchor_persistently_forgets_the_stored_row():
    """clear_cache(persistent=False) would drop memory and read the same value
    straight back out of the store, which is a refresh that does nothing."""
    calls: list[str] = []
    with patch("store.clear_business_date_anchor",
               side_effect=lambda acct, *a, **k: calls.append(acct) or 1):
        anchor.clear_cache("acct", persistent=True)
    assert calls == ["acct"], "the durable anchor row was not cleared"


# ── Fault 2: a suggestion the product cannot answer must not be offered ──────

TWO_ISLANDS = {
    "entities": [
        {"entity_name": "Sales",    "table_name": "SALES_FCT",  "schema_name": "dbo"},
        {"entity_name": "Customer", "table_name": "CUST_DIM",   "schema_name": "dbo"},
        {"entity_name": "Ticket",   "table_name": "TICKET_FCT", "schema_name": "dbo"},
        {"entity_name": "Agent",    "table_name": "AGENT_DIM",  "schema_name": "dbo"},
    ],
    "relationships": [
        {"from_entity": "Sales",  "to_entity": "Customer"},
        {"from_entity": "Ticket", "to_entity": "Agent"},
    ],
}

DETECTED = {
    "revenue by customer": ["Sales", "Customer"],
    "tickets by agent":    ["Ticket", "Agent"],
    "revenue by agent":    ["Sales", "Agent"],
    "tickets by customer": ["Ticket", "Customer"],
    "total revenue":       ["Sales"],
}


def _gate():
    return patch("core.graph_resolver.detect_entities",
                 side_effect=lambda q, g: DETECTED.get(q, [])), \
           patch("store.get_full_graph", return_value=TWO_ISLANDS)


@pytest.mark.parametrize("question,allowed", [
    ("revenue by customer", True),   # one island, joinable
    ("tickets by agent",    True),
    ("revenue by agent",    False),  # spans two islands: no governed path
    ("tickets by customer", False),
    ("total revenue",       True),   # single entity needs no join
])
def test_a_question_spanning_unjoinable_entities_is_withheld(question, allowed):
    """The old gate returned True for every one of these, because this graph has
    no jointless twin and that was the only thing it looked for."""
    detect, graph = _gate()
    with detect, graph:
        check = suggestions._graph_reachability_check("acct")
        assert check(question) is allowed, (
            f"{question!r} should be {'offered' if allowed else 'withheld'}"
        )


def test_the_gate_reports_itself_unverified_when_it_falls_open():
    """Failing open is right when filtering a trusted source and wrong when
    promoting an untrusted one, so the caller has to be able to tell."""
    with patch("store.get_full_graph", return_value={"entities": []}):
        check = suggestions._graph_reachability_check("acct")
    assert check("anything") is True, "an unreadable graph must not withhold"
    assert getattr(check, "verified", None) is False, (
        "a gate that fell open reported itself as having checked something"
    )


def test_metric_suggestions_go_through_the_same_gate():
    """Tier 2 was the one source offered unchecked. A metric's SQL template
    having been valid says nothing about whether the NL pipeline can re-plan the
    sentence the suggestion is phrased as."""
    import inspect

    source = inspect.getsource(suggestions.get_suggestions)
    tier2 = source[source.index("Tier 2"):source.index("Tier 3")]
    assert "_reachable(" in tier2, (
        "metric-registry suggestions bypass the reachability gate"
    )
