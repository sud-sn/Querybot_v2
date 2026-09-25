"""
Two columns of one table shown under one name go to the admin.

The item dimension of a real inventory warehouse, read offline, keeps
PDC_GRP_DMS_KEY and PRU_GRP_DMS_KEY, and both were shown as "Product Group
Dimension Key": PDC and PRU both read "product". No label told them apart,
nor did the knowledge base's description of the table -- and nothing in the
warehouse says what PRU is, while PDC_GRP_DMS_KEY joins PDC_GRP_DMS.

Discovery now finds the columns of one table a reader is shown under one name
and puts them in the admin's queue with the evidence: which columns, the codes
that read alike, and which of them joins a table its name spells -- that one
is what the name says, and the others need a meaning. With nothing to tell
them apart, each does. The admin writes a reading, which becomes the column's
label, or leaves it as it reads; a decision that tells a pair apart takes the
rest of the pair out of the queue.

Synthetic names in a real inventory warehouse's naming convention; no
customer data. `store` is imported where it is used (see
tests/test_unknown_members_are_found_at_discovery.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.business_meaning import COLUMN, propose_meanings
from core.vocab_packs import _clone_builtin, builtin_vocab

TABLES = {
    "MART.ITM_BAL_DLY_FCT": {c: "int" for c in (
        "ITM_BAL_DLY_FCT_KEY", "ITM_DMS_KEY", "VND_DMS_KEY", "PFT_CTR_DMS_KEY", "ON_HND_QTY")},
    "MART.ITM_DMS": {c: "nvarchar" for c in (
        "ITM_DMS_KEY", "ITM_NM", "PDC_GRP_DMS_KEY", "PRU_GRP_DMS_KEY", "ETL_UPD_TS", "ETL_UPDATE_TS")},
    "MART.PDC_GRP_DMS": {c: "nvarchar" for c in ("PDC_GRP_DMS_KEY", "PDC_GRP_CD", "PDC_GRP_DSC")},
    # Retired items keep the item's name too: across tables, one name is expected.
    "MART.ITM_ARC_DMS": {c: "nvarchar" for c in ("ITM_ARC_DMS_KEY", "ITM_NM")},
    # Two spellings of one name, and no key to say which is which.
    "MART.VND_DMS": {c: "nvarchar" for c in ("VND_DMS_KEY", "VND_NM", "VENDOR_NM")},
    "MART.PFT_CTR_DMS": {c: "nvarchar" for c in (
        "PFT_CTR_DMS_KEY", "PFT_CTR_NM", "PC_CO", "PC_COMPANY")},
}
# PC_CO holds a country code: its values say it is not the company.
VALUES = {"PFT_CTR_DMS.PC_CO": ["CA"]}


def _collisions(tables=TABLES, vocab=None, values=None) -> dict[str, dict]:
    return {
        p.subject: p.as_dict()
        for p in propose_meanings(tables, vocab=vocab or builtin_vocab(), values=values)
        if p.rule == "collision"
    }


class TestWhatIsFlagged:

    def test_the_column_that_joins_nothing_is_the_one_to_name(self):
        flagged = _collisions()["PRU_GRP_DMS_KEY"]
        assert (flagged["scope"], flagged["reading"], flagged["confidence"]) == (COLUMN, "", 0)
        assert flagged["evidence"] == [
            'PRU_GRP_DMS_KEY and PDC_GRP_DMS_KEY on ITM_DMS are both shown as "product group dimension key"',
            'PDC and PRU both read "product"',
            "PDC_GRP_DMS_KEY joins PDC_GRP_DMS, so it is the product group dimension key",
        ]
        assert flagged["where"] == ["ITM_DMS.PRU_GRP_DMS_KEY", "ITM_DMS.PDC_GRP_DMS_KEY"]
        assert "PDC_GRP_DMS_KEY" not in _collisions()

    def test_with_nothing_to_tell_them_apart_each_is_flagged(self):
        flagged = _collisions()
        assert flagged["VND_NM"]["evidence"] == [
            'VND_NM and VENDOR_NM on VND_DMS are both shown as "vendor name"',
            'VENDOR and VND both read "vendor"',
        ]
        assert flagged["VENDOR_NM"]["where"] == ["VND_DMS.VENDOR_NM", "VND_DMS.VND_NM"]

    def test_when_each_joins_a_table_each_is_flagged(self):
        tables = {**TABLES, "MART.PRU_GRP_DMS": {"PRU_GRP_DMS_KEY": "int", "PRU_GRP_DSC": "nvarchar"}}
        flagged = _collisions(tables)
        assert {"PDC_GRP_DMS_KEY", "PRU_GRP_DMS_KEY"} <= set(flagged)

    def test_what_is_not_a_collision(self):
        flagged = _collisions()
        # One name in two tables, and the platform's own columns.
        assert not {"ITM_NM", "ETL_UPD_TS", "ETL_UPDATE_TS"} & set(flagged)
        # Nothing sampled: PC_CO reads "company" like PC_COMPANY.
        assert set(flagged) == {"PRU_GRP_DMS_KEY", "VND_NM", "VENDOR_NM", "PC_CO", "PC_COMPANY"}

    def test_a_whole_name_reading_tells_them_apart(self):
        vocab = _clone_builtin()
        vocab.column_dict["PRUGRPDMSKEY"] = ("pricing group dimension key", [])
        assert "PRU_GRP_DMS_KEY" not in _collisions(vocab=vocab)

    def test_a_column_already_read_otherwise_keeps_that_reading(self):
        proposals = {p.subject: p for p in propose_meanings(TABLES, vocab=builtin_vocab(), values=VALUES)}
        country = proposals["PC_CO"]
        assert (country.rule, country.reading) == ("values", "profit center country")
        assert country.confidence > 0
        assert 'PC_CO and PC_COMPANY on PFT_CTR_DMS are both shown as "profit center company"' in (
            country.evidence)
        assert proposals["PC_COMPANY"].rule == "collision"


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    return request


def _discover(account_id) -> dict[str, int]:
    """Discovery's pass, with the value index as the boundary: PC_CO's sampled
    value is what the index would hold."""
    from core.business_meaning import propose_for_account

    sampled = [{"table_fqn": "WH.MART.PFT_CTR_DMS", "column": "PC_CO", "values": ["CA"]}]
    with patch("core.value_index.sample_values_by_column", return_value=sampled):
        return propose_for_account(account_id)


@pytest.fixture
def account(tmp_path):
    """A tenant discovered from a schema directory, its vocabulary read from a
    scratch client directory (never the repository's)."""
    import store
    from core.vocab_packs import _account_cache

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "Test Ltd")
    schema = {
        f"WH.{fqn}": {
            "columns": [{"name": c, "type": t} for c, t in columns.items()],
            "schema": "MART", "database": "WH",
        }
        for fqn, columns in TABLES.items()
    }
    (tmp_path / "_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    store.update_client_state(account_id, "READY", {"schema_dir": str(tmp_path)})
    with tempfile.TemporaryDirectory() as tmp, \
            patch("core.vocab_packs._CLIENTS_DIR", Path(tmp)):
        _account_cache.pop(account_id, None)
        counts = _discover(account_id)
        rules = [m["rule"] for m in store.list_business_meanings(account_id)]
        assert (counts["alike"], counts["unread"]) == (4, rules.count("unread"))
        yield account_id
        _account_cache.pop(account_id, None)


def _meanings(account_id, status="suggested") -> dict[str, dict]:
    import store

    return {m["subject"]: m for m in store.list_business_meanings(account_id, statuses={status})}


def _decide(account_id, subject, action, reading=""):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval"):
        return asyncio.run(routes.meanings_decide(
            _request(), account_id, scope=COLUMN, subject=subject, action=action,
            reading=reading, synonyms=""))


def _label(account_id, column):
    from core.schema_enrichment import display_label
    from core.vocab_packs import activate_vocab, deactivate_vocab, vocab_for_account

    token = activate_vocab(vocab_for_account(account_id))
    try:
        return display_label(column)
    finally:
        deactivate_vocab(token)


def _page(account_id):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_resp", side_effect=lambda request, name, ctx: ctx):
        return asyncio.run(routes.meanings_page(_request(), account_id))


class TestTheAdminDecides:

    def test_discovery_keeps_them_for_the_admin(self, account):
        page = _page(account)
        assert sorted(m["subject"] for m in page["alike"]) == [
            "PC_COMPANY", "PRU_GRP_DMS_KEY", "VENDOR_NM", "VND_NM"]
        assert not {m["subject"] for m in page["to_write"]} & {"PRU_GRP_DMS_KEY", "VND_NM"}

    def test_the_page_shows_the_evidence_and_asks_for_a_reading(self, account):
        from admin import routes

        class _Url:
            path = f"/admin/clients/{account}/meanings"

        class _FakeRequest:
            url = _Url()
            query_params: dict = {}
            session = {"admin_id": "admin_user_1"}
            scope = {"type": "http"}
            cookies: dict = {}
            headers: dict = {}

        context = _page(account)
        context["client"] = {"account_id": account, "client_name": "Test Ltd", "state": "READY"}
        html = routes.templates.get_template("client_meanings.html").render(request=_FakeRequest(), **context)
        section = html[html.index("Columns shown under one name (4)"):]
        assert "PDC_GRP_DMS_KEY joins PDC_GRP_DMS, so it is the product group dimension key" in section
        assert 'name="subject" value="PRU_GRP_DMS_KEY"' in section
        assert "Leave as it reads" in section

    def test_the_admins_reading_becomes_the_label(self, account):
        assert _label(account, "PRU_GRP_DMS_KEY") == _label(account, "PDC_GRP_DMS_KEY")
        _decide(account, "PRU_GRP_DMS_KEY", "confirm", reading="pricing group dimension key")
        assert _label(account, "PRU_GRP_DMS_KEY") == "Pricing Group Dimension Key"
        assert _label(account, "PDC_GRP_DMS_KEY") == "Product Group Dimension Key"
        assert "PRU_GRP_DMS_KEY" in _meanings(account, "confirmed")

    def test_naming_one_of_a_pair_takes_the_other_out_of_the_queue(self, account):
        _decide(account, "VENDOR_NM", "confirm", reading="vendor legal name")
        assert "VND_NM" not in _meanings(account)
        assert "VENDOR_NM" in _meanings(account, "confirmed")
        # The rest of the queue is as discovery left it.
        assert {"PRU_GRP_DMS_KEY", "PC_COMPANY", "PC_CO"} <= set(_meanings(account))

    def test_a_confirmed_reading_that_still_collides_keeps_the_pair_queued(self, account):
        _decide(account, "VENDOR_NM", "confirm", reading="vendor name")
        assert "VND_NM" in _meanings(account)

    def test_leaving_it_as_it_reads_is_kept(self, account):
        _decide(account, "PRU_GRP_DMS_KEY", "reject")
        _discover(account)
        assert "PRU_GRP_DMS_KEY" in _meanings(account, "rejected")
        assert "PRU_GRP_DMS_KEY" not in {m["subject"] for m in _page(account)["alike"]}


class TestTheRefresh:

    def test_it_leaves_the_other_rules_proposals_alone(self, account):
        import store

        before = {k: (m["rule"], m["reading"]) for k, m in _meanings(account).items()}
        store.save_business_meanings(account, [], rule="collision")
        after = {k: (m["rule"], m["reading"]) for k, m in _meanings(account).items()}
        assert after == {k: v for k, v in before.items() if v[0] != "collision"}
        assert after["PC_CO"] == ("values", "profit center country")

    def test_a_subject_another_rule_proposed_keeps_that_proposal(self, account):
        from core.business_meaning import refresh_collisions

        refresh_collisions(account)
        assert (_meanings(account)["PC_CO"]["rule"], _meanings(account)["PC_CO"]["reading"]) != (
            "collision", "")

    def test_a_tenant_never_discovered_has_nothing_to_refresh(self):
        import store
        from core.business_meaning import refresh_collisions

        store.init_db()
        account_id = f"acct{os.urandom(4).hex()}"
        store.upsert_client(account_id, "Test Ltd")
        assert refresh_collisions(account_id) == {"kept": 0, "dropped": 0}
