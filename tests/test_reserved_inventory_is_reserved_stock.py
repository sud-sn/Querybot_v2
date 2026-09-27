"""
Reserved inventory is reserved stock, whatever a pack calls a table.

"Stock réservé par entrepôt" was refused on the sample tenant: "I couldn't
build a trusted join plan for this question". The product reads French "stock"
as "inventory", and two things then went wrong with "reserved inventory by
warehouse". The reserved quantity's metric is named "reserved stock", which
shared one word with the question, as every inventory metric shared
"inventory": five metrics on two tables were in scope, and the join plan could
not bring both tables to one date. An English reader asking for reserved
inventory met the same. And the distribution ERP pack calls the monthly balance
table "inventory by warehouse": those words, inside the question, were read as
the reader naming that table, which keeps no reserved quantity, over the
approved metric whose table does.

"Stock" and "inventory" are one word to the metric matchers now, and a table
name that says how its table is cut ("by warehouse", "per location") names the
table by what comes before: "reserved inventory by warehouse" names inventory,
not a source.

A synthetic tenant (tests/answer_harness.py) whose daily snapshot keeps the
allocated and reserved quantities, asked with and without the distribution
packs; the warehouse and the model are the only stand-ins.
"""

from __future__ import annotations

import json

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("reserved-inventory")) as built:
        yield built


def _by_warehouse(amounts: list) -> dict:
    """An amount of the newest snapshot per warehouse name and unit, row for row with STOCK."""
    totals: dict = {}
    for (whs, item, *_rest), amount in zip(harness.STOCK, amounts):
        key = (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])
        totals[key] = totals.get(key, 0) + amount
    return totals


_ALLOCATED = [row[5] for row in harness.STOCK]


def _answered(answer: dict, measure: str) -> dict:
    assert answer["model_wrote_sql"] is False
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row[measure] for row in run["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,measure,amounts", [
        ("Reserved inventory by warehouse", "RESERVED_QUANTITY", harness.RESERVED),
        ("Allocated inventory by warehouse", "ALLOCATED_QUANTITY", _ALLOCATED),
    ])
    def test_inventory_is_stock(self, warehouse, question, measure, amounts):
        assert _answered(harness.ask(warehouse, question), measure) == pytest.approx(_by_warehouse(amounts))


class TestWithTheDistributionPacks:
    """The packs the sample tenant's admin selected, one of which names the
    monthly balance table "inventory by warehouse"."""

    @pytest.fixture(autouse=True)
    def packs(self, warehouse):
        import store
        from core.vocab_packs import forget_account_vocab

        before = store.get_client(harness.ACCOUNT).get("erp_packs") or "[]"
        store.update_client_meta(harness.ACCOUNT, erp_packs=json.dumps(["infor_m3", "wholesale_distribution"]))
        forget_account_vocab(harness.ACCOUNT)
        try:
            yield
        finally:
            store.update_client_meta(harness.ACCOUNT, erp_packs=before)
            forget_account_vocab(harness.ACCOUNT)

    @pytest.mark.parametrize("question,lang,measure,amounts", [
        ("Stock réservé par entrepôt", "fr", "RESERVED_QUANTITY", harness.RESERVED),
        ("Reserved inventory by warehouse", "en", "RESERVED_QUANTITY", harness.RESERVED),
        ("Stock alloué par entrepôt", "fr", "ALLOCATED_QUANTITY", _ALLOCATED),
    ])
    def test_the_measure_is_read_where_it_is_kept(self, warehouse, question, lang, measure, amounts):
        answer = harness.ask(warehouse, question, lang)
        assert _answered(answer, measure) == pytest.approx(_by_warehouse(amounts))


_DAILY, _MONTHLY = "MART.DAILY_BALANCE", "MART.MONTHLY_BALANCE"


def _source(question: str, synonyms: list[str], *, bound: set[str] | None = None) -> dict:
    from core.source_resolution import resolve_source_scope
    from core.vocab_packs import MergedVocab

    model = {"tables": [
        {"qualified_name": _DAILY, "schema": "MART", "type": "fact",
         "fields": [{"column": "RSV_QTY", "role": "measure", "expanded_name": "Reserved Quantity"}]},
        {"qualified_name": _MONTHLY, "schema": "MART", "type": "fact",
         "fields": [{"column": "ON_HND_QTY", "role": "measure", "expanded_name": "On Hand Quantity"}]},
    ]}
    vocab = MergedVocab(table_dict={"MONTHLY_BALANCE": {"label": "", "synonyms": synonyms, "type": "fact"}})
    return resolve_source_scope(question, model, vocab=vocab, authoritative_fact_tables=bound or set())


class TestTheSourceRule:

    @pytest.mark.parametrize("name,question", [
        ("inventory by warehouse", "reserved inventory by warehouse"),
        ("inventory per warehouse", "reserved inventory per warehouse"),
    ])
    def test_a_name_cut_by_a_dimension_is_its_subject(self, name, question):
        scope = _source(question, [name], bound={_DAILY})
        assert (scope["selected_fact"], scope["reason"]) == (_DAILY, "approved metric source binding")

    def test_the_name_before_the_cut_is_still_a_source(self):
        scope = _source("stock balance by period by warehouse", ["stock balance by period"], bound={_DAILY})
        assert (scope["selected_fact"], scope["reason"]) == (_MONTHLY, "explicit tenant source terminology")

    def test_a_two_word_name_is_still_a_source(self):
        scope = _source("reserved stock balance", ["stock balance"], bound={_DAILY})
        assert (scope["selected_fact"], scope["reason"]) == (_MONTHLY, "explicit tenant source terminology")


def _metric(name: str, synonyms: str) -> dict:
    return {"name": name, "synonyms": synonyms}


_RESERVED = _metric("Reserved quantity", "reserved quantity, reserved stock, quantity reserved")
_OTHERS = [
    _metric("Inventory value", "inventory value, stock value, value of stock"),
    _metric("Stock on hand", "stock on hand, on hand, inventory on hand"),
    _metric("Month-end inventory value", "month-end inventory value, month-end stock value"),
]


class TestTheWords:

    @pytest.mark.parametrize("word", ["stock", "stocks", "inventories"])
    def test_one_word(self, word):
        from core.word_forms import base_form

        assert base_form(word) == "inventory"

    def test_the_metric_scope(self):
        from core.metric_scope import resolve_metric_scope

        scope = resolve_metric_scope([_RESERVED, *_OTHERS], "reserved inventory by warehouse", {})
        assert [metric["name"] for metric in scope.metrics] == ["Reserved quantity"]
