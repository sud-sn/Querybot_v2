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
        # A bare "inventory" beside another measure's word is that measure's.
        ("value of our inventory by warehouse", ["Inventory value"]),
        ("what's the value of our inventory?", ["Inventory value"]),
        ("available inventory by warehouse", ["Available quantity"]),
        ("allocated inventory by warehouse", ["Allocated quantity"]),
        ("reserved inventory by warehouse", ["Reserved quantity"]),
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
        "net sales without tax by month", "revenue by region without returns", "total sales without VAT last year",
        "products not used", "revenue sans taxes by month", "aucun filtre, sales by month",
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
        assert reply.startswith("Which figure should I look at? This workspace can answer about ")
        assert "Stock on hand" in reply
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
        from core.dispatcher import _plainly_a_business_question

        assert _plainly_a_business_question(message)

    @pytest.mark.parametrize("message", [
        "thanks, that is the best", "you are the best", "that was the most helpful", "by the way, thanks",
        "as per your answer that is fine", "hello there", "how are you doing?", "what can you do?",
        "tell me a joke", "perfect, thanks a lot", "What does gross margin mean?", "How is net margin calculated?",
        "Can you explain what a KPI is?", "Can you help me reset my account password?", "is 30% margin good?",
        "what does on hand mean?", "Is our data refreshed daily?", "Thank you, see you in 2025", "100% agree",
        "merci by avance",
    ])
    def test_said_to_the_assistant(self, message):
        from core.dispatcher import _plainly_a_business_question

        assert not _plainly_a_business_question(message)


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


# ── Whether a turn beside a result is about it ───────────────────────────────

_REVENUE = [{"name": "Revenue", "synonyms": "sales, revenue"}, {"name": "Inventory value", "synonyms": "inventory value"},
            {"name": "Units sold", "synonyms": "units sold"}]


class TestBesideAResult:

    @pytest.mark.parametrize("question, columns, asked_anew", [
        ("top 10 items by inventory value", ["WAREHOUSE", "INVENTORY_VALUE"], True),
        ("units sold last 6 months", ["PERIOD", "UNITS_SOLD"], True),
        ("revenue by product", ["REGION", "REVENUE"], True),
        ("and the revenue last quarter?", ["REGION", "REVENUE"], False),
        ("same revenue but for last month", ["REGION", "REVENUE"], False),
        ("just the top 5 by sales", ["REGION", "REVENUE"], False),
        ("sort by sales", ["REGION", "REVENUE"], False),
        ("what about revenue by product?", ["REGION", "REVENUE"], False),
    ])
    def test_a_question_of_its_own(self, question, columns, asked_anew):
        from core.query_router import asks_a_new_question

        assert asks_a_new_question(question, columns, _REVENUE, lang="en") is asked_anew

    @pytest.mark.parametrize("question, lang, about_it", [
        ("why is that?", "en", True),
        ("explain this result", "en", True),
        ("why is North Depot the highest?", "en", True),
        ("pourquoi North Depot est-il le plus élevé ?", "fr", True),
        ("summarize returns by month for 2024", "en", False),
        ("explain what the most important KPI is for a distributor", "en", False),
        ("analyse the top 10 customers this year", "en", False),
        ("Why did sales drop last month across all regions?", "en", False),
        ("what should I focus on first, and why?", "en", False),
    ])
    def test_a_question_about_it(self, question, lang, about_it):
        from core.query_router import asks_about_the_result

        assert asks_about_the_result(question, _REVENUE, lang=lang) is about_it


# ── A figure the data does not keep, and a plain list ────────────────────────

class TestWhatTheDataKeeps:

    _INVENTORY = {"MART.ITM_BAL_DLY_FCT": {"ON_HND_QTY": "decimal", "ITM_CST": "decimal", "PFT_CTR_DMS_KEY": "int",
                                           "STL_PRC_IND": "nvarchar"},
                  "MART.ITM_DMS": {"ITM_NET_WT": "decimal", "ITM_GRS_WT": "decimal"}}
    _FACTS = {"MART.ITM_BAL_DLY_FCT"}

    @pytest.mark.parametrize("question, word", [
        ("what is my revenue trend", "revenue"),
        ("what is my revenue for each profit centre", "revenue"),
        ("what is our gross margin?", "margin"),
        ("sales by month", "sales"),
        ("stock by profit centre", ""),
        ("units sold by month", ""),
    ])
    def test_a_figure_an_inventory_does_not_keep(self, question, word):
        from core.metric_scope import measure_the_data_lacks

        metrics = [{"name": "Stock on hand", "synonyms": "stock"}, {"name": "Units sold", "synonyms": "units sold"}]
        assert measure_the_data_lacks(question, metrics, self._INVENTORY, self._FACTS) == word

    def test_a_workspace_that_keeps_the_money_is_not_told_it_does_not(self):
        from core.metric_scope import measure_the_data_lacks

        assert measure_the_data_lacks("revenue by month", [], {"MART.SLS_FCT": {"NET_SLS_AMT": "decimal"}}) == ""
        assert measure_the_data_lacks("revenue by month", [{"name": "Revenue"}], self._INVENTORY, self._FACTS) == ""

    @pytest.mark.parametrize("question, thing", [
        ("List me the item names", "item names"), ("what are the item groups?", "item groups"),
        ("show me the suppliers", "suppliers"), ("list warehouses", "warehouses"),
        ("list the availables items present", "items"), ("list the available items", "items"),
        ("what items do we have?", "items"), ("which items are available?", "items"),
        ("which warehouses are there?", "warehouses"),
        ("show stock by warehouse", ""), ("list items with no stock", ""), ("what are the top 5 items", ""),
        # Narrowed by a member or a condition: never listed whole.
        ("list items in the FITTINGS group", ""), ("list customers from Ontario", ""),
        ("list the items for warehouse 301", ""), ("list items whose name starts with A", ""),
    ])
    def test_a_plain_listing(self, question, thing):
        from core.listing import listing_target

        assert listing_target(question) == thing

    def test_a_listing_is_written_for_the_warehouse(self):
        from core.listing import listing_sql

        field = {"term": "item name", "table": "WH.MART.ITM_DMS", "column": "ITM_NM"}
        assert listing_sql(field, "azure_sql") == (
            "SELECT DISTINCT TOP 200 listed.[ITM_NM] AS [ITEM_NAME] FROM [MART].[ITM_DMS] AS listed "
            "WHERE listed.[ITM_NM] IS NOT NULL ORDER BY listed.[ITM_NM]")
        assert "LIMIT 200" in listing_sql(field, "snowflake")


class TestTheProductSaysWhatItKeeps:

    def test_revenue_of_an_inventory(self, warehouse):
        answer = harness.ask(warehouse, "what is my revenue trend")

        (reply,) = [str(payload) for _kind, payload in answer["replies"]]
        assert reply.startswith("This workspace's data has no revenue figures")
        assert "Stock on hand" in reply and not answer["model_wrote_sql"]

    def test_in_french(self, warehouse):
        answer = harness.ask(warehouse, "quel est notre chiffre d'affaires par mois ?", "fr")

        (reply,) = [str(payload) for _kind, payload in answer["replies"]]
        assert reply.startswith("Les données de cet espace ne contiennent pas de chiffre d'affaires")

    def test_the_item_names(self, warehouse):
        answer = harness.ask(warehouse, "list item names")

        assert not answer["model_wrote_sql"]
        assert {row["ITEM_NAME"] for row in answer["rows"]} >= {"BRASS ELBOW", "STEEL TEE", "COPPER PIPE", "PEX PIPE"}

    def test_the_available_items(self, warehouse):
        answer = harness.ask(warehouse, "list the available items")

        assert not answer["model_wrote_sql"]
        assert {"BRASS ELBOW", "COPPER PIPE"} <= set(answer["rows"][0].values()) | {
            value for row in answer["rows"] for value in row.values()}

    def test_the_warehouses_in_french(self, warehouse):
        answer = harness.ask(warehouse, "liste des entrepôts", "fr")

        assert not answer["model_wrote_sql"]
        assert {"NORTH DEPOT", "SOUTH DEPOT"} <= {row["WAREHOUSE"] for row in answer["rows"]}
