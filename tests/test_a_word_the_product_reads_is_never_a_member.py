"""
A word the product reads as vocabulary is never a member.

The value resolver offers the reader's own words to the value index, and a
member it matches sends the question to the planner, filtered on that member.
Its English exclusions -- column names, entity words, the glossary, the
measures' words -- had no French twins, and words of the question matched
members:

  * "valeur du stock par pays" matched "pays" to a warehouse coded like it,
    where "by country" never reached the index;
  * "inventory value by buyer" matched "buyer", the name of a role-named key,
    to a party whose name holds the word;
  * "les 10 articles ayant la plus grande valeur" asked which of three members
    "ayant" (having) meant;
  * "était" was read from its second letter, as "tait", and narrowed the
    question to a party whose code ends with it;
  * "avons-nous", "a-t-il" and "mois-ci" were each a candidate.

The words the tenant's columns are named with, and the French the
canonicaliser and the tenant's packs rewrite, are never a member on their own
-- though a member's name may hold one: "l'entrepôt Laval" names ENTREPOT
LAVAL. And "quel était" is "what was": "which etait notre inventory" asked for
a which-thing and was not compiled.

A synthetic tenant (tests/answer_harness.py) with a warehouse coded PAY and one
coded ENTREPOT and named ENTREPOT LAVAL, neither holding stock; the warehouse
and the model are the only stand-ins.
"""

from __future__ import annotations

import json

import pytest

from core import value_resolver
from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("vocabulary")) as built:
        yield built


def _columns() -> dict:
    """The tenant's columns as the pipeline loads them for the resolver."""
    import store
    from core.schema import load_schema_columns

    return load_schema_columns(store.get_client_state(harness.ACCOUNT)["schema_dir"])


def _vocabulary(account: str, columns: dict | None) -> dict:
    """The resolver's vocabulary argument, as the pipeline passes it. Read at
    call time: on a tree without it the words are not excluded, and each test
    fails on what the resolver does rather than on an import."""
    build = getattr(value_resolver, "build_vocabulary_words", None)
    return {"vocabulary": build(account, columns)} if build else {}


def _candidates(question: str) -> list[str]:
    columns = _columns()
    return value_resolver.extract_candidate_phrases(
        question, value_resolver.build_known_terms(harness.ACCOUNT, columns),
        **_vocabulary(harness.ACCOUNT, columns))


def _members(question: str) -> list[tuple[str, str]]:
    columns = _columns()
    resolved = value_resolver.resolve_literals(
        harness.ACCOUNT, question, known_terms=value_resolver.build_known_terms(harness.ACCOUNT, columns),
        **_vocabulary(harness.ACCOUNT, columns))
    return [(item["phrase"], item.get("value")) for bucket in ("verified", "in_lists", "narrowed", "clarify")
            for item in resolved.get(bucket) or []]


class TestTheProductAnswers:

    def test_by_warehouse_in_french(self, warehouse):
        # A warehouse is coded ENTREPOT: "par entrepôt" named it, and the
        # question went to the model filtered on that one warehouse.
        answer = harness.ask(warehouse, "Valeur du stock par entrepôt", "fr")
        assert answer["model_wrote_sql"] is False
        (run,) = [run for run in answer["executed"] if "AS INVENTORY_VALUE" in run["sql"]]
        expected: dict = {}
        for whs, _item, _buyer, _created, on_hand, _allocated, cost in harness.STOCK:
            expected[harness.WAREHOUSES[whs][1]] = expected.get(harness.WAREHOUSES[whs][1], 0) + on_hand * cost
        assert {row["WAREHOUSE"]: row["INVENTORY_VALUE"] for row in run["rows"]} == pytest.approx(expected)

    def test_a_question_asked_in_the_past(self, warehouse):
        answer = harness.ask(warehouse, "Quel était notre stock en main en mars 2026 ?", "fr")
        assert answer["model_wrote_sql"] is False
        (run,) = [run for run in answer["executed"] if "AS STOCK_ON_HAND" in run["sql"]]
        expected: dict = {}
        for _whs, item, _buyer, _created, on_hand, _allocated, _cost in harness.STOCK:
            expected[harness.ITEMS[item][3]] = expected.get(harness.ITEMS[item][3], 0) + on_hand
        assert {row["UNT_OF_MSR"]: row["STOCK_ON_HAND"] for row in run["rows"]} == pytest.approx(expected)


class TestWhatIsNeverAMember:

    def test_a_french_word_the_canonicaliser_reads(self, warehouse):
        assert _members("Valeur du stock par pays") == []

    def test_an_accented_word_of_vocabulary(self, warehouse):
        assert _candidates("Valeur du stock par entrepôt") == []

    def test_a_word_a_column_is_named_with(self, warehouse):
        assert _candidates("Inventory value by buyer") == []

    def test_the_tail_of_an_accented_word(self, warehouse):
        assert _candidates("Quel était notre stock en main à la fin de 2025 ?") == []

    @pytest.mark.parametrize("question", [
        "Combien d'unités avons-nous vendues en 2025 ?",
        "Combien de fournisseurs y a-t-il ?",
        "Quel est notre stock en main ce mois-ci ?",
    ])
    def test_grammar_joined_by_hyphens(self, warehouse, question):
        assert _candidates(question) == []

    def test_a_verb_a_degree_and_a_sentences_first_capital(self, warehouse):
        found = {phrase.lower() for phrase in _candidates("Les 2 articles ayant la plus grande valeur de stock")}
        assert not found & {"ayant", "grande", "les 2"} and not any("ayant" in phrase for phrase in found)

    def test_the_french_of_a_selected_pack(self, warehouse):
        import store

        account = f"{harness.ACCOUNT}-pack"
        store.upsert_client(account, "Pack Reader Ltd")
        store.update_client_meta(account, erp_packs=json.dumps(["wholesale_distribution"]))
        assert {"ville", "acheteur"} <= _vocabulary(account, None).get("vocabulary", set())


class TestWhatIsStillAMember:

    def test_a_member_named_with_a_word_of_vocabulary(self, warehouse):
        assert ("entrepôt Laval", "ENTREPOT LAVAL") in _members("Valeur du stock de l'entrepôt Laval")

    def test_a_code_joined_by_a_hyphen(self, warehouse):
        assert "BE-1" in _candidates("Stock on hand for item BE-1")

    def test_a_name(self, warehouse):
        assert ("BRASS ELBOW", "BRASS ELBOW") in _members("Units sold for BRASS ELBOW in 2025")
