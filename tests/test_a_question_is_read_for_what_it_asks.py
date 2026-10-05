"""
A question is read for what it asks: which measure, which condition, whether
it is one at all, and which question a follow-up continues.

* "which warehouses have the lowest stock?" named no metric: "stock" alone was
  a word of seven stock metrics' names, they tied, and the one it meant --
  the stock on hand -- sorted last and was cut. The question was asked which
  measure it meant, or refused over a join.
* "show me items with no stock" was answered with every item's stock: "no"
  reads as a word that narrows nothing, and no compiler writes "= 0".
* "how is the North Depot warehouse doing?" names no measure, and was told to ask
  an administrator to confirm a relationship.
* the portal put most questions to a model to decide whether they were
  questions for the data -- and "thanks, that is the best" must still not be.
* "why is that?" asked after a drill-down was planned as the drill-down's
  words alone, which named no measure.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests import answer_harness as harness


# ── Which measure ────────────────────────────────────────────────────────────

def _starter_metrics() -> list[dict]:
    """The starter metrics of a daily stock snapshot with a unit cost, an
    allocated, a reserved and an available part, as core.starter_metrics
    proposes them."""
    from core.starter_metrics import starter_metrics

    model = {"tables": [{
        "type": "fact", "fact_type": "periodic_snapshot", "qualified_name": "WH.MART.ITM_BAL_DLY_FCT",
        "fields": [{"column": column, "role": "measure"} for column in (
            "ON_HND_QTY", "ITM_CST", "ALC_ON_HND_QTY", "RSV_QTY", "AVL_QTY", "ORD_QTY")],
    }]}
    return [metric.as_metric() for metric in starter_metrics(model)]


def _named(question: str, reader_question: str = "") -> list[str]:
    from core.metric_scope import resolve_metric_scope

    return [metric["name"] for metric in resolve_metric_scope(
        _starter_metrics(), question, None, reader_question=reader_question).metrics]


class TestWhichMeasure:

    def test_stock_alone_is_the_stock_on_hand(self):
        synonyms = {metric["name"]: metric["synonyms"] for metric in _starter_metrics()}

        assert {"stock", "inventory"} <= {word.strip() for word in synonyms["Stock on hand"].split(",")}
        assert not any({"stock", "inventory"} & {word.strip() for word in text.split(",")}
                       for name, text in synonyms.items() if name != "Stock on hand")

    @pytest.mark.parametrize("question, metrics", [
        ("which warehouses have the lowest stock?", ["Stock on hand"]),
        ("stock by warehouse", ["Stock on hand"]),
        ("stock value by warehouse", ["Inventory value"]),
        ("inventory value by warehouse", ["Inventory value"]),
        ("allocated stock by warehouse", ["Allocated quantity"]),
        ("reserved stock by warehouse", ["Reserved quantity"]),
    ])
    def test_in_english(self, question, metrics):
        assert _named(question) == metrics

    def test_in_french_the_value_of_the_stock_is_the_value(self):
        assert _named("value of inventory by warehouse", "valeur du stock par entrepôt") == ["Inventory value"]

    def test_in_french_the_stock_is_the_stock_on_hand(self):
        assert _named("inventory en main by warehouse", "stock en main par entrepôt") == ["Stock on hand"]


# ── Which condition ──────────────────────────────────────────────────────────

class TestNoneOfIt:

    @pytest.mark.parametrize("question", [
        "show me items with no stock", "items that have no stock", "out of stock items", "items never sold",
        "products not sold last month", "warehouses with zero stock", "items without stock",
        "articles sans inventory", "articles en rupture of inventory", "articles with aucun inventory",
        "articles qui ont pas of inventory", "the articles with a inventory nul",
    ])
    def test_is_a_condition_no_compiler_writes(self, question):
        from core.pipeline_helpers import _VALUE_CONDITION

        assert _VALUE_CONDITION.search(question)

    @pytest.mark.parametrize("question", [
        "stock on hand by warehouse", "top 10 items by inventory value", "units sold by month",
        "what about allocated quantity?", "which warehouses have the lowest stock?",
        "number of receipts by item group", "inventory disponible by warehouse", "how many items do we have",
        "is there any stock in North Depot", "in stock by warehouse",
    ])
    def test_a_question_without_one_is_not_read_as_one(self, question):
        from core.pipeline_helpers import _VALUE_CONDITION

        assert not _VALUE_CONDITION.search(question)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("read-for-what-it-asks")) as built:
        yield built


class TestTheProductAnswers:

    def test_items_with_no_stock_are_not_every_items_stock(self, warehouse):
        answer = harness.ask(warehouse, "show me items with no stock")

        # Not compiled without its condition: the SQL writer is given it.
        assert answer["model_wrote_sql"]

    def test_a_question_naming_no_measure_is_asked_for_one(self, warehouse):
        import core.query_pipeline as qp

        resolve = qp._graph_resolve

        def unreachable(*args, **kwargs):
            # The join graph cannot reach part of the question, as on a
            # warehouse whose member matched a dimension nothing joins.
            found = resolve(*args, **kwargs)
            if "authoritative_fact_tables" in kwargs:
                found = {**found, "enabled": True, "planning_status": "blocked", "missing_entities": ["WHS_DMS"]}
            return found

        with patch.object(qp, "_graph_resolve", unreachable):
            no_measure = harness.ask(warehouse, "how is the NORTH DEPOT warehouse doing?")
            a_measure = harness.ask(warehouse, "stock on hand by warehouse")

        (reply,) = [str(payload) for _kind, payload in no_measure["replies"]]
        assert reply.startswith("I understand the analytical request")
        assert "the governed measure to calculate" in reply
        assert "join plan" not in reply
        (reply,) = [str(payload) for _kind, payload in a_measure["replies"]]
        assert reply.startswith("I couldn't build a trusted join plan for this question.")


# ── Whether it is a question for the data ────────────────────────────────────

class TestTheFrontDoor:

    @pytest.mark.parametrize("message", [
        "stock on hand by warehouse", "which warehouses have the lowest stock?", "top 10 items by inventory value",
        "best customers", "gross margin %", "units sold last 6 months", "how is the North Depot warehouse doing?",
        "number of receipts by item group", "what is our total stock on hand?",
    ])
    def test_a_question_for_the_data(self, message):
        from core.dispatcher import _looks_like_data_request

        assert _looks_like_data_request(message)

    @pytest.mark.parametrize("message", [
        "thanks, that is the best", "you are the best", "that was the most helpful", "by the way, thanks",
        "as per your answer that is fine", "hello there", "how are you doing?", "what can you do?",
        "tell me a joke", "perfect, thanks a lot",
    ])
    def test_said_to_the_assistant(self, message):
        from core.dispatcher import _looks_like_data_request

        assert not _looks_like_data_request(message)


# ── Which question a follow-up continues ─────────────────────────────────────

class TestTheLineage:

    def test_a_follow_up_of_a_follow_up_keeps_both(self):
        from core.clarification import combine_with_result_context, extract_original_question

        drilled = combine_with_result_context("stock on hand by warehouse", "break it down by item group")
        why = combine_with_result_context(drilled, "why is that?")

        assert why.count("Context from the active governed result:") == 1
        assert extract_original_question(why) == (
            "stock on hand by warehouse Follow-up request: break it down by item group\n"
            "Follow-up request: why is that?")

    def test_a_result_is_continued_as_it_was_planned(self):
        from core.governed_result_followup import contextualize_source_query_fallback, source_question
        from core.result_cache import ResultCache

        cache = ResultCache(max_sessions=2)
        planned = "stock on hand by warehouse\nFollow-up request: break it down by item group"
        cache.store("s", [{"ITEM_GROUP": "PIPE", "STOCK_ON_HAND": 1.0}], question="break it down by item group",
                    result_id="drilled")
        cache.note_planning_question("s", "drilled", planned)

        assert source_question("s", source_result_id="drilled", cache=cache) == planned
        assert contextualize_source_query_fallback(
            "why is that?", "s", source_result_id="drilled", cache=cache,
        ).endswith("Context from the active governed result: " + " ".join(planned.split()))

    def test_a_result_with_no_plan_of_its_own_is_continued_as_asked(self):
        from core.governed_result_followup import source_question
        from core.result_cache import ResultCache

        cache = ResultCache(max_sessions=2)
        cache.store("s", [{"WAREHOUSE": "A", "STOCK_ON_HAND": 1.0}], question="stock on hand by warehouse",
                    result_id="source")

        assert source_question("s", source_result_id="source", cache=cache) == "stock on hand by warehouse"
        assert source_question("s", source_result_id="missing", cache=cache) == ""
