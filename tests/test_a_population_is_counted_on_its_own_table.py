"""
A population is counted on its own table, without its placeholder members.

"How many warehouses do we have?" was never answered on the sample tenant, nor
"how many suppliers", nor "number of profit centres by province". The product
knew to count a population on the table that defines it -- the warehouse table,
not a fact -- and then:

- counted the placeholder rows a dimension keeps for a value that was empty or
  matched nothing among the warehouses, and the validator refused it;
- kept the warehouse's own name as a breakdown and the join to it from a fact
  the question never read, which the join graph then required;
- asked the reader of "how many items do we have" which of two facts to read,
  though a population is counted on neither;
- found no table defining "suppliers" or "profit centres": it read the
  columns by the names the model was written with ("seller code", "profit ctr
  code"), not as the tenant's vocabulary reads them today;
- and read nothing of the French: "combien d'entrepôts avons-nous" asked for
  no population, and "par province" set the province aside as a modifier of
  the count instead of its breakdown.

A synthetic tenant (tests/answer_harness.py) whose warehouse, item and item
group tables each keep a placeholder member.
"""

from __future__ import annotations

import json

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("populations")) as built:
        yield built


def _members(table: dict) -> int:
    return sum(1 for key in table if key != 0)


def _items_by_group() -> dict:
    counted: dict = {}
    for key, (_code, _name, group, _unit) in harness.ITEMS.items():
        if key:
            counted[harness.GROUPS[group][1]] = counted.get(harness.GROUPS[group][1], 0) + 1
    return counted


def _count(answer: dict, column: str) -> int:
    assert answer["model_wrote_sql"] is False
    ((row,),) = [answer["rows"]]
    return row[column]


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("How many warehouses do we have?", "en"),
        ("How many warehouses are there?", "en"),
        ("Combien d'entrepôts avons-nous ?", "fr"),
        ("Combien y a-t-il d'entrepôts ?", "fr"),
    ])
    def test_warehouses(self, warehouse, question, lang):
        assert _count(harness.ask(warehouse, question, lang), "WAREHOUSE_COUNT") == _members(harness.WAREHOUSES)

    def test_items_are_counted_on_the_item_table(self, warehouse):
        answer = harness.ask(warehouse, "How many items do we have?")
        assert _count(answer, "ITEM_COUNT") == _members(harness.ITEMS)

    def test_items_by_item_group(self, warehouse):
        answer = harness.ask(warehouse, "Number of items by item group")
        assert answer["model_wrote_sql"] is False
        assert {row["ITEM_GROUP"]: row["ITEM_COUNT"] for row in answer["rows"]} == _items_by_group()

    def test_by_the_admins_name_for_it(self, warehouse):
        # The admin calls the warehouse code a "site code": the sites are the
        # warehouses.
        from core.vocab_packs import forget_account_vocab
        from store.table_description_store import get_table_description, save_table_description

        before = get_table_description(harness.ACCOUNT, "MART.WHS_DMS") or {}
        save_table_description(harness.ACCOUNT, "MART.WHS_DMS", column_synonyms={"WHS_CD": ["site code"]})
        forget_account_vocab(harness.ACCOUNT)
        try:
            answer = harness.ask(warehouse, "How many sites do we have?")
        finally:
            save_table_description(harness.ACCOUNT, "MART.WHS_DMS", description=before.get("description") or "",
                                   synonyms=before.get("synonyms") or "",
                                   column_synonyms=before.get("column_synonym_map") or "")
            forget_account_vocab(harness.ACCOUNT)
        assert _count(answer, "SITE_COUNT") == _members(harness.WAREHOUSES)


class TestInFrenchWithThePack:
    """The distribution pack reads "articles" as items, as it does on the
    sample tenant, where the portal activates the tenant's vocabulary before
    the question is read."""

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

    def test_items_by_item_group(self, warehouse):
        answer = harness.ask(warehouse, "Nombre d'articles par groupe d'articles", "fr")
        assert answer["model_wrote_sql"] is False
        assert {row["ITEM_GROUP"]: row["ITEM_COUNT"] for row in answer["rows"]} == _items_by_group()


class TestTheWords:

    @pytest.mark.parametrize("question,canonical", [
        ("Combien d'entrepôts avons-nous ?", "how many warehouses do we have ?"),
        ("Combien y a-t-il d'entrepôts ?", "how many warehouses ?"),
        ("Combien de clients y a-t-il ?", "how many customers are there ?"),
    ])
    def test_in_french(self, question, canonical):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == canonical


_MASTER = {"qualified_name": "MART.PC_DMS", "type": "dimension", "fields": [
    {"column": "PC_DMS_KEY", "role": "dimension_key", "expanded_name": "pc dimension key"},
    {"column": "PC_CD", "role": "attribute", "expanded_name": "pc code"},
    {"column": "PC_PRV", "role": "attribute", "expanded_name": "pc province"},
]}


class TestTheRule:

    def test_the_vocabulary_names_the_population(self):
        from core.count_target_resolver import resolve_population_count_target
        from core.vocab_packs import MergedVocab

        model = {"tables": [_MASTER]}
        vocab = MergedVocab(direct_aliases={"PC_CD": {"profit centre", "branch code"}})
        found = resolve_population_count_target("profit centre", model, vocab=vocab)
        assert (found["status"], found["selected"]["column"]) == ("selected", "PC_CD")
        assert resolve_population_count_target("branch", model, vocab=vocab)["selected"]["column"] == "PC_CD"
        assert resolve_population_count_target("profit centre", model)["status"] == "missing"

    def test_the_plan_counts_the_population(self):
        from core.analytical_request_plan import demote_counted_population

        plan = {
            "count_target": {"selected": {"table": "MART.PC_DMS", "column": "PC_CD"}},
            "fields": [
                {"term": "profit centre", "table": "MART.PC_DMS", "column": "PC_NM", "role": "display_dimension"},
                {"term": "province", "table": "MART.PC_DMS", "column": "PC_PRV", "role": "attribute"},
            ],
            "joins": [{"from": "MART.STOCK_FCT", "to": "MART.PC_DMS"}, {"from": "MART.PC_DMS", "to": "MART.RGN_DMS"}],
        }
        demote_counted_population(plan, {"population_entity": "profit centre", "dimensions": ["province"]})
        assert [field.get("enforcement") for field in plan["fields"]] == ["optional", None]
        assert [join.get("enforcement") for join in plan["joins"]] == ["optional", None]

    def test_no_fact_the_first_pass_found(self):
        from core.semantic_resolution import build_planner_alignment

        graph = {"entities": [
            {"entity_name": "STOCK_FCT", "schema_name": "MART", "table_name": "STOCK_FCT", "entity_type": "fact"},
            {"entity_name": "PC_DMS", "schema_name": "MART", "table_name": "PC_DMS", "entity_type": "dimension"},
        ]}
        plan = {"count_target": {"selected": {"table": "MART.PC_DMS", "column": "PC_CD"}},
                "fields": [{"term": "province", "table": "MART.PC_DMS", "column": "PC_PRV", "role": "attribute"}]}
        aligned = build_planner_alignment(graph=graph, graph_ctx={"detected": ["STOCK_FCT", "PC_DMS"]},
                                          semantic_plan=plan)
        assert (aligned["required_entities"], aligned["dropped_fact_entities"]) == (["PC_DMS"], ["STOCK_FCT"])

