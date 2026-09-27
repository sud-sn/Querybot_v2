"""
A dimension answers to what the tenant's vocabulary calls it.

"Inventory value by branch" was never answered on the sample tenant. The
distribution pack its admin selected calls the profit center's name column
"branch name" and its code "branch code", but a dimension was matched only by
its own name -- the profit center -- so "by branch" named nothing and the query
was left to the model and refused.

A term the vocabulary (a pack, or the admin's own column terms) gives a
dimension's label or code column, less the word that makes it a label, now
names the dimension where the question asks by it: "branch name" makes the
profit center a branch. A name two dimensions are given names neither, and
"customer type" labels no customer.

A synthetic tenant (tests/answer_harness.py) whose admin calls the warehouse's
description column "site name".
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("vocabulary-names")) as built:
        yield built


@pytest.fixture(scope="module")
def site_name(warehouse):
    """The admin's term for the warehouse's description column."""
    from core.vocab_packs import forget_account_vocab
    from store.table_description_store import get_table_description, save_table_description

    before = get_table_description(harness.ACCOUNT, "MART.WHS_DMS") or {}
    save_table_description(harness.ACCOUNT, "MART.WHS_DMS", column_synonyms={"WHS_DSC": ["site name"]})
    forget_account_vocab(harness.ACCOUNT)
    try:
        yield
    finally:
        save_table_description(harness.ACCOUNT, "MART.WHS_DMS", description=before.get("description") or "",
                               synonyms=before.get("synonyms") or "",
                               column_synonyms=before.get("column_synonym_map") or "")
        forget_account_vocab(harness.ACCOUNT)


def _by_warehouse(answer: dict) -> dict:
    assert answer["model_wrote_sql"] is False
    return {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row["STOCK_ON_HAND"] for row in answer["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Stock on hand by site", "en"),
        ("Stock en main par site", "fr"),
    ])
    def test_by_site_is_by_warehouse(self, warehouse, site_name, question, lang):
        expected = _by_warehouse(harness.ask(warehouse, "Stock on hand by warehouse"))
        assert expected
        assert _by_warehouse(harness.ask(warehouse, question, lang)) == expected

    def test_a_site_not_asked_by(self, warehouse, site_name):
        answer = harness.ask(warehouse, "Stock on hand at the site")
        assert answer["model_wrote_sql"] is False
        assert {key for row in answer["rows"] for key in row} == {"UNT_OF_MSR", "STOCK_ON_HAND"}


def _dimension(display_table: str, display_column: str, code_column: str = "") -> dict:
    return {"display_table": display_table, "display_column": display_column, "code_column": code_column,
            "source_key": display_table.split(".")[-1] + "_KEY", "name": display_table.split(".")[-1]}


class TestTheRule:

    @staticmethod
    def _names(aliases: dict, *dimensions: dict) -> dict:
        from core.semantic_model import _dimension_vocabulary_names
        from core.vocab_packs import MergedVocab

        vocab = MergedVocab(direct_aliases={column: set(terms) for column, terms in aliases.items()})
        return _dimension_vocabulary_names([{"qualified_name": "MART.STOCK", "dimensions": list(dimensions)}], vocab)

    def test_a_label_names_its_dimension(self):
        assert self._names({"PC_NM": ["profit center name", "branch name"]}, _dimension("MART.PC", "PC_NM")) == {
            "MART.PC": {"profit center", "branch"}}

    def test_so_does_a_code(self):
        assert self._names({"PC_CD": ["branch code"]}, _dimension("MART.PC", "PC_NM", "PC_CD")) == {"MART.PC": {"branch"}}

    def test_a_name_two_dimensions_are_given(self):
        assert self._names({"PC_NM": ["branch name"], "WHS_NM": ["branch name"]},
                           _dimension("MART.PC", "PC_NM"), _dimension("MART.WHS", "WHS_NM")) == {}

    @pytest.mark.parametrize("term", ["customer type", "type name", "branch", "name", "desc"])
    def test_not_a_label(self, term):
        assert self._names({"CUS_NM": [term]}, _dimension("MART.CUS", "CUS_NM")) == {}
