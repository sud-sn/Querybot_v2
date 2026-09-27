"""
A dimension is named as the tenant's vocabulary spells it.

"Inventory value by region", "by stock status" and "by business area" were
never answered on the sample tenant, in English or French. A dimension's name
is its table's, cut at the underscores -- "Itm Stk Sts", "Acme Rgn" -- and
nothing read it as the vocabulary does: the region's head noun was "rgn", and
"by region" named nothing. Worse, a key whose prefix a pack names two ways
("Item / Product") was labelled with the prefix alone, so the item's stock
status was called "Item Product", as the item is. And a name that starts with
what it belongs to -- the ITEM's stock status, the ITEM's business area -- was
not asked for by the rest of it: "by stock status" shares two words of three
with it, and "status" and "area" are too generic, or too common, to name one.

The French never reached it either: "statut" was left untranslated, "secteur
d'activité" was read as "segment activite", and French "stock" is read as
"inventory", so "par statut de stock" asks by the inventory status.

A synthetic tenant (tests/answer_harness.py) whose snapshot keeps each row's
stock status in a dimension of its own, named as the sample's is.
"""

from __future__ import annotations

import json

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("stock-statuses")) as built:
        yield built


def _on_hand_by_status() -> dict:
    """The newest snapshot's stock on hand by (stock status, unit)."""
    totals: dict = {}
    for n, (_whs, item, *_rest) in enumerate(harness.STOCK):
        on_hand = harness.STOCK[n][4]
        key = (harness.STOCK_STATUSES[harness.STOCK_STATUS[n]][1], harness.ITEMS[item][3])
        totals[key] = totals.get(key, 0) + on_hand
    return totals


def _answered(answer: dict) -> dict:
    assert answer["model_wrote_sql"] is False
    return {(row["ITEM_STOCK_STATUS"], row["UNT_OF_MSR"]): row["STOCK_ON_HAND"] for row in answer["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question", [
        "Stock on hand by stock status",
        "Stock on hand for each stock status",
        # French "stock" is read as inventory, and an English reader says either.
        "Stock on hand by inventory status",
    ])
    def test_by_stock_status(self, warehouse, question):
        assert _answered(harness.ask(warehouse, question)) == pytest.approx(_on_hand_by_status())


class TestInFrenchWithThePack:
    """The distribution pack reads "statut de stock" as the stock status, as it
    does on the sample tenant, where the portal activates the tenant's
    vocabulary before the question is read."""

    @pytest.fixture(autouse=True)
    def packs(self, warehouse):
        import store
        from core.vocab_packs import activate_vocab, deactivate_vocab, forget_account_vocab, vocab_for_account

        before = store.get_client(harness.ACCOUNT).get("erp_packs") or "[]"
        store.update_client_meta(harness.ACCOUNT, erp_packs=json.dumps(["wholesale_distribution"]))
        forget_account_vocab(harness.ACCOUNT)
        token = activate_vocab(vocab_for_account(harness.ACCOUNT))
        try:
            yield
        finally:
            deactivate_vocab(token)
            store.update_client_meta(harness.ACCOUNT, erp_packs=before)
            forget_account_vocab(harness.ACCOUNT)

    @pytest.mark.parametrize("question", [
        "Stock disponible par statut de stock",
        "Stock disponible par statut du stock",
    ])
    def test_by_stock_status(self, warehouse, question):
        assert _answered(harness.ask(warehouse, question, "fr")) == pytest.approx(_on_hand_by_status())


class TestTheName:

    @pytest.mark.parametrize("source_key,declared,label", [
        # A prefix a pack names two ways is refined by either reading.
        ("ITM_STK_STS_DMS_KEY", "Itm Stk Sts", "Item Stock Status"),
        ("WHS_RGN_DMS_KEY", "Whs Rgn", "Warehouse Region"),
        ("ACME_RGN_DMS_KEY", "Acme Rgn", "Acme Region"),
        # A key no entity prefix names reads through the vocabulary as well.
        ("BYR_PTY_DMS_KEY", "Pty", "Buyer Party"),
        # A name a prefix spells, and the admin's own, are left as they were.
        ("RGN_DMS_KEY", "Rgn", "Region"),
        ("CUS_SEG_DMS_KEY", "Customer Segment", "Customer Segment"),
    ])
    def test_spelled(self, source_key, declared, label):
        from core.semantic_model import _dimension_label

        assert _dimension_label(source_key, {"name": declared}, "X_DSC")[0] == label

    def test_the_item_is_still_the_item(self):
        from core.semantic_model import _dimension_label

        assert _dimension_label("ITM_DMS_KEY", {"name": "Itm"}, "ITM_NM") == ("Item Product", "item product")

    def test_on_the_model(self, warehouse):
        import store
        from core.semantic_model import _dimension_label, load_semantic_model

        model = load_semantic_model(store.get_client_state(harness.ACCOUNT)["kb_dir"])
        found = {
            dimension["source_key"]: _dimension_label(dimension["source_key"], dimension, dimension["display_column"])[0]
            for table in model.get("tables") or []
            for dimension in table.get("dimensions") or []
            if dimension.get("source_key") == "ITM_STK_STS_DMS_KEY" and dimension.get("display_column")
        }
        assert found == {"ITM_STK_STS_DMS_KEY": "Item Stock Status"}

    def test_the_head_noun_is_spelled(self):
        from core.semantic_model import _dimension_short_names

        tables = [{"qualified_name": "MART.STOCK", "dimensions": [
            {"name": "Acme Rgn", "source_key": "ACME_RGN_DMS_KEY", "display_table": "MART.ACME_RGN_DMS",
             "display_column": "ACME_RGN_DSC"},
        ]}]
        assert _dimension_short_names(tables) == {"MART.ACME_RGN_DMS": "region"}


def _dimension(name: str, table: str) -> dict:
    return {"name": name, "source_key": f"{table}_KEY", "display_table": f"MART.{table}", "display_column": "DSC"}


class TestTheRule:

    @staticmethod
    def _tails(*dimensions) -> dict:
        from core.semantic_model import _dimension_tail_names

        return _dimension_tail_names([{"qualified_name": "MART.STOCK", "dimensions": list(dimensions)}])

    def test_the_name_less_what_it_belongs_to(self):
        assert self._tails(_dimension("Item Business Area", "ITM_BUS_ARA_DMS")) == {
            "MART.ITM_BUS_ARA_DMS": "business area"}

    def test_a_tail_two_dimensions_end_on_names_neither(self):
        assert self._tails(_dimension("Item Stock Status", "ITM_STK_STS_DMS"),
                           _dimension("Order Stock Status", "ORD_STK_STS_DMS")) == {}

    def test_a_two_word_name_has_no_tail(self):
        assert self._tails(_dimension("Item Group", "ITM_GRP_DMS")) == {}

    def test_a_generic_tail_names_nothing(self):
        assert self._tails(_dimension("Customer Type Name", "CUS_TYP_DMS")) == {}

    @pytest.mark.parametrize("question,asked", [
        ("value by stock status", "stock status"),
        ("value by the inventory status", "inventory status"),
        ("value for each stock status", "stock status"),
        ("value by inventory statuses", "inventory status"),
        ("stock status report", ""),
    ])
    def test_asked_by_in_any_form(self, question, asked):
        from core.semantic_model import _asked_by_in_any_form

        assert _asked_by_in_any_form(question, "stock status") == asked

    def test_the_forms_of_a_word(self):
        from core.word_forms import forms_of

        assert {"stock", "inventory"} <= set(forms_of("stock")) and forms_of("status") == ("status",)


class TestTheWords:

    @pytest.mark.parametrize("question,canonical", [
        ("Valeur du stock par statut de stock", "inventory value by inventory status"),
        ("Valeur du stock par secteur d'activité", "inventory value by business area"),
        ("Valeur du stock par domaine d'activité", "inventory value by business area"),
    ])
    def test_with_the_distribution_pack(self, question, canonical):
        from core.question_normalizer import canonical_question
        from core.vocab_packs import _clone_builtin, _merge_pack, activate_vocab, deactivate_vocab, load_pack

        vocab = _clone_builtin()
        _merge_pack(vocab, load_pack("wholesale_distribution"), "wholesale_distribution")
        token = activate_vocab(vocab)
        try:
            assert canonical_question(question, "fr") == canonical
        finally:
            deactivate_vocab(token)

    def test_statut_is_status(self):
        from core.question_normalizer import canonical_question

        assert canonical_question("Nombre de clients par statut", "fr") == "count of customers by status"
