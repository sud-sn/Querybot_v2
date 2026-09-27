"""
One French order is an order.

"Combien de clients ont passé une commande en 2025 ?" was refused, where
"How many customers placed an order in 2025?" was answered. "Commande" is
read as "orders" -- the word a French reader uses for the orders counted --
so "une commande" came out "a orders", the Number of Orders metric was the
only one its words matched, and the count of customers who placed one was
never among the answers.

"Une commande" is now read as "an order", as the English is written: the
customers who placed one are counted by the metric that counts them.
"Combien de commandes" is still how many orders, and a named kind of order
is one of that kind: "une commande en souffrance" is a back order, as
"commande en souffrance" is.

tests/star_harness.py keeps two years of orders from four customers and the
metrics Number of Buying Customers and Number of Orders.
"""

from __future__ import annotations

import pytest

from tests import star_harness as star


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    with star.tenant_in(tmp_path_factory.mktemp("une-commande")) as built:
        yield built


class TestTheProductAnswers:

    def test_the_customers_who_placed_an_order(self, warehouse):
        answer = star.ask(warehouse, "Combien de clients ont passé une commande en 2025 ?", "fr")
        assert answer["model_wrote_sql"] is False
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[5] for line in star.orders() if line[2].year == 2025})]]

    def test_how_many_orders_is_still_orders(self, warehouse):
        answer = star.ask(warehouse, "Combien de commandes en 2025 ?", "fr")
        assert [list(row.values()) for row in answer["rows"]] == [
            [len({line[0] for line in star.orders() if line[2].year == 2025})]]


class TestTheReading:

    @pytest.mark.parametrize("question,read", [
        ("Combien de clients ont passé une commande en 2025 ?", "how many customers placed an order en 2025 ?"),
        ("Combien de commandes en 2025 ?", "how many orders en 2025 ?"),
        ("Une commande en souffrance", "a back order"),
        ("Une commande annulée", "a cancelled order"),
    ])
    def test_in_english(self, question, read):
        from core.question_normalizer import canonical_question

        assert canonical_question(question, "fr") == read

    def test_every_kind_of_order_is_one_of_that_kind(self):
        from core.question_normalizer import _LEXICON, canonical_question

        kinds = {key: read for key, read in _LEXICON.items() if key.startswith("commande ")}
        assert kinds
        for key, read in kinds.items():
            article = "an" if read[0] in "aeiou" else "a"
            assert canonical_question(f"une {key}", "fr") == f"{article} {read}"
