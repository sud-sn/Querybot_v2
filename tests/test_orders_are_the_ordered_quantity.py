"""
An orders question is answered with the quantity ordered, not the back-ordered.

A daily snapshot keeping both an ordered and a back-ordered quantity had a
metric for the back orders only. "Orders by warehouse" shared the one word
"orders" with the metric's "back orders"; the field plan read "orders" as the
ordered quantity itself, but the metric matched, the registry outranked the
field, and the answer was the back-ordered quantity. "Ordered quantity by
warehouse" and "quantity ordered" were worse off still: the field plan read
the noun "order" but not its verb, so they found no field at all and fell to
the back-ordered metric too. The French "commandes" followed the English.

Three rules now hold, each tested below, directly and through the product: a
metric for a stock state the question does not name -- back-ordered,
reserved, allocated -- yields to the measure the question does name; a
quantity is found by its noun's verb, before or after it; and a snapshot's
ordered quantity has a starter metric of its own.

The first rule is about states only. A metric that says more than the
question in any other way -- "Net Sales" to "sales", "Total Revenue USD" to
"revenue" -- is the governed reading of what was asked, and is never set
aside for a column of the bare word. And "ordered" after a quantity another
word has named is how it is sorted: "reserved quantity ordered by warehouse"
asks for the reserved quantity.

A synthetic tenant (tests/answer_harness.py); the warehouse and the model are
the only stand-ins.
"""

from __future__ import annotations

import pytest

from tests import answer_harness as harness


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with harness.tenant_in(tmp_path_factory.mktemp("orders")) as built:
        yield built


def _by_warehouse(amounts: list) -> dict:
    """An amount of the newest snapshot per warehouse name and unit, row for row with STOCK."""
    totals: dict = {}
    for (whs, item, *_rest), amount in zip(harness.STOCK, amounts):
        key = (harness.WAREHOUSES[whs][1], harness.ITEMS[item][3])
        totals[key] = totals.get(key, 0) + amount
    return totals


def _answered(answer: dict, measure: str) -> dict:
    (run,) = [run for run in answer["executed"] if f"AS {measure}" in run["sql"]]
    return {(row["WAREHOUSE"], row["UNT_OF_MSR"]): row[measure] for row in run["rows"]}


class TestTheProductAnswers:

    @pytest.mark.parametrize("question,lang", [
        ("Orders by warehouse", "en"),
        ("Ordered quantity by warehouse", "en"),
        ("Quantity ordered by warehouse", "en"),
        ("Commandes par entrepôt", "fr"),
        ("Quantité commandée par entrepôt", "fr"),
    ])
    def test_the_ordered_quantity(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert "RSV_BCK_ORD_QTY" not in " ".join(run["sql"] for run in answer["executed"])
        assert _answered(answer, "ORDERED_QUANTITY") == pytest.approx(_by_warehouse(harness.ORDERED))

    @pytest.mark.parametrize("question,lang", [
        ("Back orders by warehouse", "en"),
        ("Back ordered quantity by warehouse", "en"),
        ("Commandes en souffrance par entrepôt", "fr"),
    ])
    def test_back_orders_are_still_the_back_ordered(self, warehouse, question, lang):
        answer = harness.ask(warehouse, question, lang)
        assert answer["model_wrote_sql"] is False
        assert _answered(answer, "BACK_ORDERED_QUANTITY") == pytest.approx(_by_warehouse(harness.BACK_ORDERED))

    @pytest.mark.parametrize("question", [
        "Reserved quantity ordered by warehouse", "Allocated quantity ordered by warehouse"])
    def test_ordered_by_sorts_another_quantity(self, warehouse, question):
        answer = harness.ask(warehouse, question)
        assert answer["model_wrote_sql"] is False
        assert "ORDERED_QUANTITY" not in " ".join(run["sql"] for run in answer["executed"])


class TestAMetricForAStateYields:
    """core.metric_scope: a metric whose matching phrases name a stock state
    the question does not gives way to the measure the question names."""

    BACK_ORDERED = {
        "name": "Back-ordered quantity", "required_columns": "BCK_ORD_QTY",
        "synonyms": "back-ordered quantity, back orders, quantity on back order",
    }
    ORDERED_FIELD = {"fields": [{"term": "order", "table": "WH.SALES.STOCK_FACT", "column": "ORD_QTY",
                                 "role": "measure"}]}

    def _scope(self, question: str, semantic_plan: dict | None) -> list[str]:
        from core.metric_scope import resolve_metric_scope

        result = resolve_metric_scope([self.BACK_ORDERED], question, None, semantic_plan=semantic_plan)
        return [metric["name"] for metric in result.metrics]

    def test_yields_to_the_measure_the_question_names(self):
        assert self._scope("orders by warehouse", self.ORDERED_FIELD) == []

    def test_stays_when_the_question_names_no_measure_whole(self):
        assert self._scope("orders by warehouse", None) == ["Back-ordered quantity"]

    def test_is_kept_when_the_question_says_all_of_it(self):
        assert self._scope("back orders by warehouse", self.ORDERED_FIELD) == ["Back-ordered quantity"]

    def test_a_field_found_by_other_words_is_no_rival(self):
        # The metric is matched by "customers"; the "order" the field plan
        # read in "placed an order" names no measure of its own.
        from core.metric_scope import resolve_metric_scope

        buyers = {"name": "Number of buying customers", "required_columns": "CUSTOMER_ID",
                  "synonyms": "buying customers, active customers, customers who purchase"}
        result = resolve_metric_scope([buyers], "how many customers placed an order in 2025", None,
                                      semantic_plan=self.ORDERED_FIELD)
        assert [metric["name"] for metric in result.metrics] == ["Number of buying customers"]

    def test_a_field_the_metric_reads_is_no_rival(self):
        own_field = {"fields": [{"term": "order", "table": "WH.SALES.STOCK_FACT", "column": "BCK_ORD_QTY",
                                 "role": "measure"}]}
        assert self._scope("orders by warehouse", own_field) == ["Back-ordered quantity"]

    def test_a_french_phrase_that_says_more_yields_too(self):
        from core.metric_scope import resolve_metric_scope

        metric = dict(self.BACK_ORDERED, synonyms="commandes en souffrance")
        result = resolve_metric_scope([metric], "orders by warehouse", None, semantic_plan=self.ORDERED_FIELD,
                                      reader_question="Commandes par entrepôt")
        assert result.metrics == []

    def test_its_example_question_does_not_keep_it(self):
        from core.metric_scope import resolve_metric_scope

        metric = dict(self.BACK_ORDERED, example_questions="Show back orders by warehouse")
        result = resolve_metric_scope([metric], "orders by warehouse", None, semantic_plan=self.ORDERED_FIELD)
        assert result.metrics == []

    def test_a_reserved_metric_yields_to_the_stock_named(self):
        from core.metric_scope import resolve_metric_scope

        reserved = {"name": "Reserved quantity", "required_columns": "RSV_QTY",
                    "synonyms": "reserved quantity, reserved stock, réservé"}
        on_hand = {"fields": [{"term": "stock", "table": "WH.SALES.STOCK_FACT", "column": "ON_HND_QTY",
                               "role": "measure"}]}
        assert resolve_metric_scope([reserved], "stock by warehouse", None, semantic_plan=on_hand).metrics == []
        assert [m["name"] for m in resolve_metric_scope(
            [reserved], "reserved stock by warehouse", None, semantic_plan=on_hand).metrics] == ["Reserved quantity"]

    def test_a_committed_metric_yields_too(self):
        from core.metric_scope import resolve_metric_scope

        committed = {"name": "Committed stock", "required_columns": "CMT_QTY", "synonyms": "committed stock"}
        on_hand = {"fields": [{"term": "stock", "table": "WH.SALES.STOCK_FACT", "column": "ON_HND_QTY",
                               "role": "measure"}]}
        assert resolve_metric_scope([committed], "stock by warehouse", None, semantic_plan=on_hand).metrics == []

    def test_a_metric_whose_own_column_is_named_stays(self):
        # "reserve" is a state the question does not say, and the plan found
        # another balance as well -- but also the metric's own.
        from core.metric_scope import resolve_metric_scope

        reserve = {"name": "Reserve fund balance", "required_columns": "RSV_FND_BAL",
                   "synonyms": "reserve fund balance"}
        plan = {"fields": [{"term": "fund balance", "table": "WH.FIN.FUND_FACT", "column": "RSV_FND_BAL",
                            "role": "measure"},
                           {"term": "balance", "table": "WH.FIN.FUND_FACT", "column": "OPN_BAL", "role": "measure"}]}
        result = resolve_metric_scope([reserve], "what is the fund balance", None, semantic_plan=plan)
        assert [m["name"] for m in result.metrics] == ["Reserve fund balance"]

    @pytest.mark.parametrize("metric,question,column", [
        ({"name": "Net Sales", "sql_template": "SUM(UnitPrice * OrderQuantity)", "synonyms": "net sales"},
         "sales by product category in 2025", "SalesAmount"),
        ({"name": "Total Revenue USD", "required_columns": "REVENUE_USD", "synonyms": "total revenue usd"},
         "revenue by region", "REVENUE"),
        ({"name": "Gross margin", "required_columns": "GROSS_MARGIN_AMT", "synonyms": "gross margin"},
         "margin by product", "MARGIN"),
        ({"name": "Promotional discount", "required_columns": "PROMO_DISC_AMT", "synonyms": "promotional discount"},
         "discount by customer", "DISCOUNT"),
    ])
    def test_a_governed_metric_is_not_set_aside_for_a_bare_column(self, metric, question, column):
        from core.metric_scope import resolve_metric_scope

        bare = {"fields": [{"term": question.split()[0], "table": "WH.SALES.FACT", "column": column,
                            "role": "measure"}]}
        result = resolve_metric_scope([metric], question, None, semantic_plan=bare)
        assert [m["name"] for m in result.metrics] == [metric["name"]]


class TestAQuantityIsFoundByItsVerb:
    """core.semantic_planner: "order quantity" is the ordered quantity and the
    quantity ordered; a participle another word qualifies is not."""

    COLUMNS = {"WH.SALES.STOCK_FACT": {"ORD_QTY": "decimal", "WAREHOUSE_ID": "int"}}

    def _measures(self, question: str) -> list[str]:
        from core.semantic_planner import build_semantic_field_plan

        plan = build_semantic_field_plan(question, self.COLUMNS)
        return [field["column"] for field in plan.get("fields", []) if field.get("role") == "measure"]

    @pytest.mark.parametrize("question", [
        "order quantity by warehouse",
        "ordered quantity by warehouse",
        "quantity ordered by warehouse",
        "what is the ordered quantity",
    ])
    def test_the_verb_finds_the_quantity(self, question):
        assert self._measures(question) == ["ORD_QTY"]

    @pytest.mark.parametrize("question", [
        "Which warehouse has the highest ordered quantity?",
        "average ordered quantity by warehouse",
        "monthly ordered quantity",
        "2025 ordered quantity by warehouse",
        # And after it: only a word naming another quantity makes "ordered
        # by" a sort.
        "Which warehouse has the highest quantity ordered?",
        "What is the largest quantity ordered by warehouse?",
        "Monthly quantity ordered by warehouse",
        "average quantity ordered by warehouse",
        "2025 quantity ordered by warehouse",
        # Another quantity's word, with no "by" after: still the quantity ordered.
        "What was the net quantity ordered in 2025?", "net quantity ordered last month",
        "Inventory quantity ordered in March",
    ])
    def test_whatever_word_comes_before_it(self, question):
        assert self._measures(question) == ["ORD_QTY"]

    def test_in_french(self):
        from core.question_normalizer import canonical_question

        assert self._measures(canonical_question("Quel entrepôt a la plus grande quantité commandée ?", "fr")) == [
            "ORD_QTY"]

    @pytest.mark.parametrize("question", ["back ordered quantity by warehouse", "reserved ordered quantity"])
    def test_a_participle_a_state_qualifies_is_not_the_quantity(self, question):
        assert self._measures(question) == []

    @pytest.mark.parametrize("question", [
        "Reserved quantity ordered by warehouse", "Allocated quantity ordered by warehouse",
        "Available quantity ordered by warehouse", "sold quantity ordered by month",
        "Inventory quantity ordered by warehouse", "In transit quantity ordered by warehouse",
        "Onhand quantity ordered by warehouse", "Net quantity ordered by month"])
    def test_ordered_after_another_quantity_is_how_it_is_sorted(self, question):
        assert "ORD_QTY" not in self._measures(question)


class TestTheStarterMetric:

    @staticmethod
    def _names(*columns: str) -> dict:
        from core.starter_metrics import starter_metrics

        model = {"tables": [{
            "type": "fact", "fact_type": "periodic_snapshot", "qualified_name": "WH.SALES.STOCK_FACT",
            "fields": [{"column": column, "role": "measure"} for column in columns],
        }]}
        return {metric.name: metric for metric in starter_metrics(model)}

    def test_an_ordered_quantity_is_proposed(self):
        metric = self._names("ORD_QTY", "BCK_ORD_QTY")["Ordered quantity"]
        assert metric.required_columns == ("ORD_QTY",)
        assert "quantité commandée" in metric.synonyms

    @pytest.mark.parametrize("column", ["AVG_ORD_QTY", "BCK_ORD_QTY", "PCH_ORD_QTY", "RSV_ORD_QTY"])
    def test_other_order_figures_are_not_the_ordered_quantity(self, column):
        assert "Ordered quantity" not in self._names(column)
