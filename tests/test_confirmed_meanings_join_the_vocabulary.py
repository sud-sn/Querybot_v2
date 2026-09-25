"""
tests/test_confirmed_meanings_join_the_vocabulary.py

Discovery proposes what a warehouse's codes mean; the admin confirms them on
the Business Meanings page, and a confirmed meaning joins the tenant's
vocabulary: a code's reading is how every column with it is labelled, written
up and matched; a column's or a table's reading is its name; synonyms are
further words a question can use.

"Done" for this was stated as questions, not code: "supplier", "vendor" and
"fournisseur" reach the seller dimension, "country" reaches the profit
center's country column, "on hand" reaches the stock on hand -- once, and only
once, the admin has confirmed what the evidence proposed.

Each test goes from the admin route that records the decision to the read
path that uses it, with nothing handed between them by the test. Synthetic
names in a real inventory warehouse's naming convention; no customer data.
`store` is imported where it is used (see
tests/test_unknown_members_are_found_at_discovery.py).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.business_meaning import CODE, COLUMN, TABLE, confirmed_vocabulary, propose_meanings
from core.vocab_packs import _clone_builtin, builtin_vocab

TABLES = {
    "MART.ITM_BAL_DLY_FCT": {c: "int" for c in (
        "ITM_BAL_DLY_FCT_KEY", "ITM_DMS_KEY", "SLR_DMS_KEY", "WHS_DMS_KEY", "PFT_CTR_DMS_KEY",
        "ON_HND_QTY", "ALC_ON_HND_QTY")},
    "MART.SLR_DMS": {c: "nvarchar" for c in ("SLR_DMS_KEY", "SLR_CD", "SLR_NM", "PYE_CD")},
    "MART.PFT_CTR_DMS": {c: "nvarchar" for c in (
        "PFT_CTR_DMS_KEY", "PFT_CTR_NM", "PC_ADR_LIN_1", "PC_CTY", "PC_PRV", "PC_PSL_CD", "PC_CO")},
    "MART.WHS_DMS": {c: "nvarchar" for c in ("WHS_DMS_KEY", "WHS_DSC")},
    "MART.ITM_DMS": {c: "nvarchar" for c in ("ITM_DMS_KEY", "ITM_NM", "QXR_IND")},
}
VALUES = {
    "PFT_CTR_DMS.PC_CO": ["CA"],
    "PFT_CTR_DMS.PC_PRV": ["ON", "QC", "AB"],
    "PFT_CTR_DMS.PC_CTY": ["LONDON", "CALGARY", "LAVAL"],
}


def _request(query=None):
    request = MagicMock()
    request.query_params = query or {}
    request.session = {"admin_id": "admin_user_1"}
    return request


@pytest.fixture
def account():
    """A tenant whose discovery proposed meanings, its vocabulary read from a
    scratch client directory (never the repository's)."""
    import store
    from core.vocab_packs import _account_cache

    store.init_db()
    account_id = f"acct{os.urandom(4).hex()}"
    store.upsert_client(account_id, "portal")
    proposals = propose_meanings(TABLES, vocab=builtin_vocab(), values=VALUES)
    store.save_business_meanings(account_id, [p.as_dict() for p in proposals])
    with tempfile.TemporaryDirectory() as tmp, \
            patch("core.vocab_packs._CLIENTS_DIR", Path(tmp)):
        _account_cache.pop(account_id, None)
        yield account_id
        _account_cache.pop(account_id, None)


def _decide(account_id, scope, subject, action, **form):
    from admin import routes

    with patch.object(routes, "_is_auth", return_value=True), \
            patch.object(routes, "_after_semantic_approval") as recompiled:
        response = asyncio.run(routes.meanings_decide(
            _request(), account_id, scope=scope, subject=subject, action=action,
            reading=form.get("reading", ""), synonyms=form.get("synonyms", "")))
    return response, recompiled


def _confirm_as_proposed(account_id, scope, subject):
    import store

    proposed = next(m for m in store.list_business_meanings(account_id)
                    if (m["scope"], m["subject"]) == (scope, subject))
    return _decide(account_id, scope, subject, "confirm",
                   reading=proposed["reading"], synonyms=", ".join(proposed["synonyms"]))


def _tenant(account_id):
    from core.vocab_packs import vocab_for_account

    return vocab_for_account(account_id)


def _label(account_id, column):
    from core.schema_enrichment import display_label
    from core.vocab_packs import activate_vocab, deactivate_vocab

    token = activate_vocab(_tenant(account_id))
    try:
        return display_label(column)
    finally:
        deactivate_vocab(token)


def _fields(account_id, question):
    from core.semantic_planner import build_semantic_field_plan

    plan = build_semantic_field_plan(question, TABLES, vocab=_tenant(account_id))
    return {(f["table"], f["column"]) for f in plan.get("fields", [])}


class TestTheConfirmedLayer:

    def _meaning(self, scope, subject, reading, synonyms=(), found_in=(), status="confirmed"):
        return {"scope": scope, "subject": subject, "status": status,
                "decided_reading": reading, "decided_synonyms": list(synonyms),
                "found_in": list(found_in)}

    def test_a_code_is_read_everywhere_and_its_synonyms_reach_its_columns(self):
        layer = confirmed_vocabulary([self._meaning(
            CODE, "SLR", "supplier", ["vendor"],
            ["SLR_DMS.SLR_NM", "SLR_DMS.SLR_CD", "ITM_BAL_DLY_FCT.SLR_DMS_KEY"])], builtin_vocab())
        assert layer["abbreviations"] == {"SLR": "supplier"}
        assert layer["direct_aliases"]["SLR_NM"] == ["supplier", "vendor", "vendor name"]
        # A code column does not answer to the bare noun: its name column does.
        assert layer["direct_aliases"]["SLR_CD"] == ["vendor code"]
        assert layer["direct_aliases"]["SLR_DMS_KEY"] == ["vendor dimension key"]

    def test_a_place_column_answers_to_the_place(self):
        layer = confirmed_vocabulary([self._meaning(CODE, "CTY", "city", (), ["PFT_CTR_DMS.PC_CTY"])])
        assert layer["direct_aliases"]["PC_CTY"] == ["city"]

    def test_a_column_and_a_table(self):
        vocab = _clone_builtin()
        vocab.table_dict["ITM_BAL_DLY_FCT"] = {"label": "old", "type": "fact"}
        layer = confirmed_vocabulary([
            self._meaning(COLUMN, "PC_CO", "profit center country", ["country"]),
            self._meaning(TABLE, "ITM_BAL_DLY_FCT", "item balance daily fact"),
        ], vocab)
        assert layer["column_dict"] == {"PCCO": {"label": "profit center country", "synonyms": ["country"]}}
        assert layer["table_dict"] == {"ITM_BAL_DLY_FCT": {"label": "item balance daily fact", "type": "fact"}}

    def test_only_a_confirmed_reading_counts(self):
        layer = confirmed_vocabulary([
            self._meaning(CODE, "QR", "quarter", status="suggested"),
            self._meaning(CODE, "HND", "hand", status="rejected"),
            self._meaning(CODE, "QXR", "", status="confirmed"),
        ])
        assert layer["abbreviations"] == {} and layer["direct_aliases"] == {}

    def test_a_columns_existing_words_are_kept(self):
        vocab = _clone_builtin()
        vocab.direct_aliases["SLR_NM"] = {"merchant"}
        layer = confirmed_vocabulary([self._meaning(CODE, "SLR", "supplier", (), ["SLR_DMS.SLR_NM"])], vocab)
        assert layer["direct_aliases"]["SLR_NM"] == ["merchant", "supplier"]


class TestTheTenantsVocabularyReadsThem:

    def test_a_proposal_alone_changes_nothing(self, account):
        assert _label(account, "SLR_NM") == "Seller Name"
        assert _label(account, "PC_CO") == "Profit Center Company"

    def test_a_confirmed_code_is_how_its_columns_read(self, account):
        response, recompiled = _confirm_as_proposed(account, CODE, "SLR")
        assert response.status_code == 303 and "saved=confirmed" in response.headers["location"]
        assert recompiled.called
        assert _label(account, "SLR_NM") == "Supplier Name"
        assert _label(account, "SLR_DMS_KEY") == "Supplier Dimension Key"

    def test_a_confirmed_column_reads_as_confirmed(self, account):
        _confirm_as_proposed(account, COLUMN, "PC_CO")
        assert _label(account, "PC_CO") == "Profit Center Country"
        # CO still reads "company" everywhere else.
        assert _label(account, "SLR_CO_CD") == "Seller Company Code"

    def test_undoing_it_takes_it_back_out(self, account):
        _confirm_as_proposed(account, CODE, "SLR")
        assert _label(account, "SLR_NM") == "Supplier Name"
        _decide(account, CODE, "SLR", "reset")
        assert _label(account, "SLR_NM") == "Seller Name"

    def test_a_rejected_reading_is_not_used(self, account):
        _decide(account, CODE, "SLR", "reject")
        assert _label(account, "SLR_NM") == "Seller Name"

    def test_the_admins_edit_is_what_is_used(self, account):
        _decide(account, CODE, "SLR", "confirm", reading="vendor", synonyms="")
        assert _label(account, "SLR_NM") == "Vendor Name"


class TestQuestionsReachTheRightColumns:

    def test_supplier_vendor_and_fournisseur_after_confirming(self, account):
        from core.question_normalizer import canonical_question

        slr = ("MART.SLR_DMS", "SLR_NM")
        assert slr not in _fields(account, "stock on hand by supplier")
        _confirm_as_proposed(account, CODE, "SLR")
        assert slr in _fields(account, "stock on hand by supplier")
        assert slr in _fields(account, "top 10 vendors by stock on hand")
        assert slr in _fields(account, canonical_question("quantité en stock par fournisseur", "fr"))

    def test_country_after_confirming(self, account):
        country = ("MART.PFT_CTR_DMS", "PC_CO")
        assert country not in _fields(account, "stock on hand by country")
        _confirm_as_proposed(account, COLUMN, "PC_CO")
        assert country in _fields(account, "stock on hand by country")

    def test_city_and_province_after_confirming(self, account):
        _confirm_as_proposed(account, CODE, "CTY")
        _confirm_as_proposed(account, CODE, "PRV")
        assert ("MART.PFT_CTR_DMS", "PC_CTY") in _fields(account, "stock on hand by city")
        assert ("MART.PFT_CTR_DMS", "PC_PRV") in _fields(account, "stock on hand by province")

    def test_on_hand_after_confirming(self, account):
        _confirm_as_proposed(account, CODE, "HND")
        assert _label(account, "ON_HND_QTY") == "On Hand Quantity"
        assert ("MART.ITM_BAL_DLY_FCT", "ON_HND_QTY") in _fields(account, "on hand quantity by warehouse")


class TestTheAdminPage:

    def _context(self, account_id):
        from admin import routes

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_resp", side_effect=lambda r, t, c: c):
            return asyncio.run(routes.meanings_page(_request(), account_id))

    def test_the_page_lists_what_to_review_and_what_to_write(self, account):
        context = self._context(account)
        review = [(m["scope"], m["subject"]) for m in context["to_review"]]
        assert (CODE, "SLR") in review and (COLUMN, "PC_CO") in review
        confidences = [m["confidence"] for m in context["to_review"]]
        assert confidences == sorted(confidences, reverse=True)
        assert [m["subject"] for m in context["to_write"]] == ["QXR"]
        assert context["strong_count"] == sum(1 for c in confidences if c >= context["strong"])

    def test_the_page_renders_the_evidence_and_the_forms(self, account):
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

        context = self._context(account)
        context["client"] = {"account_id": account, "client_name": "Acme", "state": "READY"}
        html = routes.templates.get_template("client_meanings.html").render(request=_FakeRequest(), **context)
        assert "SLR_DMS carries a payee (PYE_CD)" in html
        assert 'name="subject" value="PC_CO"' in html
        assert f'action="/admin/clients/{account}/meanings/confirm-strong"' in html
        assert "Codes nothing reads (1)" in html
        assert 'href="/admin/clients/' + account + '/meanings"' in html

    def test_confirm_all_strong_leaves_the_weaker_ones(self, account):
        import store
        from admin import routes

        with patch.object(routes, "_is_auth", return_value=True), \
                patch.object(routes, "_after_semantic_approval"):
            response = asyncio.run(routes.meanings_confirm_strong(_request(), account))
        assert response.status_code == 303
        status = {(m["scope"], m["subject"]): m["status"] for m in store.list_business_meanings(account)}
        assert status[(CODE, "HND")] == "confirmed"
        assert status[(CODE, "SLR")] == "suggested"
        assert status[(CODE, "QXR")] == "suggested"
        assert _label(account, "ON_HND_QTY") == "On Hand Quantity"

    def test_a_code_nothing_reads_needs_the_admins_reading(self, account):
        import store

        response, _ = _decide(account, CODE, "QXR", "confirm", reading="")
        assert "error=" in response.headers["location"]
        _decide(account, CODE, "QXR", "confirm", reading="quick reorder")
        assert _label(account, "QXR_IND") == "Quick Reorder Indicator"
        assert next(m for m in store.list_business_meanings(account)
                    if m["subject"] == "QXR")["status"] == "confirmed"

    def test_an_unknown_action_or_subject_is_refused(self, account):
        response, recompiled = _decide(account, CODE, "SLR", "approve")
        assert "error=" in response.headers["location"] and not recompiled.called
        response, recompiled = _decide(account, CODE, "NOPE", "reject")
        assert "error=" in response.headers["location"] and not recompiled.called
