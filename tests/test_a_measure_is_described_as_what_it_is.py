"""
The analysis is told what a measure is, not that every measure is a count.

The brief carries a "business meaning" into the prompt the analysis is
written from. It was chosen by words: revenue, headcount, attendance and a
"count" family that took every measure no other family named. So "inventory
value by warehouse" reached the model as "the volume or frequency of records
in the selected scope", and a question about the latest stock read as one
about attendance -- "late" was found inside "latest" -- as did any result
with an order number, "number" inside ORDER_NUMBER.

Now a family is named by whole words, plural or not, and a measure no family
names is described by its own kind (core.analysis_contract.measure_additivity):
a balance is a level held at a point in time, a price or an average a rate, a
count of events a count, a sum of money or of a quantity an amount -- and a
measure whose name says none of them (SOLDE, PRIX_MOYEN, EFFECTIF, INV_VAL,
ORDERS) is described as no kind: an average price is no "amount".

Real rows through the real brief and the real prompt builder: the assertion
is on the text the model receives.
"""

import pytest

from core.insight import (
    build_action_contract,
    build_insight_prompt_from_contract,
    compute_data_brief,
)


def _prompt(rows, question, action="why"):
    brief = compute_data_brief(rows, question)
    contract = build_action_contract(action, question, brief)
    _system, user = build_insight_prompt_from_contract(contract)
    return brief, user


STOCK = [
    {"WAREHOUSE": "North", "INVENTORY_VALUE": 5200.0},
    {"WAREHOUSE": "South", "INVENTORY_VALUE": 3100.0},
    {"WAREHOUSE": "East", "INVENTORY_VALUE": 900.0},
]
RECORDS = "volume or frequency of records"


class TestAMeasureNoFamilyNamesIsDescribedByItsKind:

    def test_inventory_value_is_a_level_not_a_count(self):
        brief, user = _prompt(STOCK, "inventory value by warehouse")
        assert brief["metric_semantics"]["key"] == "level"
        assert "a level held at a point in time" in user
        assert RECORDS not in user

    def test_the_latest_stock_is_not_attendance(self):
        rows = [{"ITEM_GROUP": "Pipe", "ON_HAND_QTY": 400.0}, {"ITEM_GROUP": "Valves", "ON_HAND_QTY": 250.0}]
        brief, user = _prompt(rows, "latest stock on hand by item group")
        assert brief["metric_semantics"]["key"] == "level"
        assert "workforce" not in user

    @pytest.mark.parametrize("question,rows,kind,words", [
        ("units sold by month", [{"MONTH": "2026-01", "UNITS_SOLD": 120.0}, {"MONTH": "2026-02", "UNITS_SOLD": 95.0}],
         "amount", "the amount of the measure shown"),
        ("average price by category", [{"CATEGORY": "Tents", "AVG_PRICE": 420.0}, {"CATEGORY": "Ropes", "AVG_PRICE": 95.0}],
         "rate", "a price, rate or average per unit"),
        ("orders by region", [{"REGION": "North", "ORDER_COUNT": 40}, {"REGION": "South", "ORDER_COUNT": 25}],
         "count", RECORDS),
    ])
    def test_each_kind_of_measure(self, question, rows, kind, words):
        brief, user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == kind
        assert words in user

    def test_an_order_number_counts_nothing(self):
        rows = [{"CUSTOMER": "Zed Co", "ORDER_NUMBER": 10432, "NET_AMOUNT": 800.0},
                {"CUSTOMER": "Acme", "ORDER_NUMBER": 10433, "NET_AMOUNT": 300.0}]
        brief, user = _prompt(rows, "amount by customer")
        assert brief["metric_semantics"]["key"] == "amount"
        assert RECORDS not in user


    def test_a_result_that_is_only_order_numbers_is_no_count(self):
        # "number" is a count only as "number of": the order numbers of a customer are no number of orders.
        rows = [{"CUSTOMER": "Zed Co", "ORDER_NUMBER": 10432}, {"CUSTOMER": "Acme", "ORDER_NUMBER": 10433}]
        brief, user = _prompt(rows, "order numbers by customer")
        assert brief["metric_semantics"]["key"] != "count"
        assert RECORDS not in user


class TestAFamilyIsNamedByWholeWords:

    @pytest.mark.parametrize("question,value,family", [
        ("revenue by region", "NET_AMOUNT", "revenue"),
        ("headcount by department", "HEADCOUNT", "headcount"),
        ("how many orders by region", "ORDERS", "count"),
        ("combien de commandes par région", "N", "count"),
        ("absence days by team", "ABSENCE_DAYS", "attendance"),
    ])
    def test_a_named_family_still_wins(self, question, value, family):
        rows = [{"GROUP": "A", value: 10.0}, {"GROUP": "B", value: 4.0}]
        brief, _user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == family

    def test_total_and_volume_are_amounts_not_counts(self):
        rows = [{"REGION": "North", "QTY_SHIPPED": 300.0}, {"REGION": "South", "QTY_SHIPPED": 120.0}]
        brief, user = _prompt(rows, "total shipped volume by region")
        assert brief["metric_semantics"]["key"] == "amount"
        assert RECORDS not in user


class TestAMeasureWhoseNameSaysNoKindIsNoKind:

    @pytest.mark.parametrize("column,question", [
        ("SOLDE", "solde par entrepôt"),
        ("PRIX_MOYEN", "prix moyen par catégorie"),
        ("TAUX_MARGE", "taux de marge par famille"),
        ("QTE_DISPONIBLE", "quantité disponible par entrepôt"),
        ("INV_VAL", "valeur du stock par entrepôt"),
        ("ORDERS", "orders by region"),
        ("CUSTOMERS", "customers by region"),
        ("EFFECTIF", "effectif par équipe"),
    ])
    def test_it_is_described_as_the_measure_shown_and_nothing_more(self, column, question):
        rows = [{"GROUP": "A", column: 120.0}, {"GROUP": "B", column: 45.0}]
        brief, user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == "measure"
        assert "Represents the measure shown, over the rows in scope" in user
        assert "the amount of the measure shown" not in user and RECORDS not in user


class TestACountIsKnownByItsAbbreviation:

    @pytest.mark.parametrize("column,question", [
        ("ORDER_COUNT", "orders by region"),
        ("ORDER_CNT", "orders by region"),
        ("NB_COMMANDES", "commandes par région"),
        ("NBR_ORDERS", "orders by region"),
        # The same names as a star schema or a dbt mart writes them.
        ("OrderCnt", "orders by region"),
        ("NbCommandes", "commandes par région"),
        ("nbrOrders", "orders by region"),
    ])
    def test_a_count_column_is_a_count(self, column, question):
        rows = [{"REGION": "North", column: 40}, {"REGION": "South", column: 25}]
        brief, user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == "count"
        assert RECORDS in user

    def test_a_quantity_is_an_amount_and_no_count(self):
        rows = [{"REGION": "North", "AVAILABLE_QTY": 40.0}, {"REGION": "South", "AVAILABLE_QTY": 25.0}]
        brief, _user = _prompt(rows, "available quantity by region")
        assert brief["metric_semantics"]["key"] == "amount"


class TestAFamilyWordIsReadPluralOrNot:

    @pytest.mark.parametrize("question,value,family", [
        ("Revenues by region", "X", "revenue"),
        ("absences by team", "X", "attendance"),
        ("resignations by team", "X", "attrition"),
        ("terminations by department", "X", "attrition"),
        ("sales by region", "X", "revenue"),
    ])
    def test_a_plural_names_its_family(self, question, value, family):
        rows = [{"GROUP": "A", value: 10.0}, {"GROUP": "B", value: 4.0}]
        brief, _user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == family

    @pytest.mark.parametrize("question,family", [
        # The word "employee" names headcount, and nothing else in these
        # questions does: two spellings of it are one word, said once.
        ("employee attrition by department", "attrition"),
        ("employee turnover by department", "attrition"),
        ("employees who resigned: attrition by department", "attrition"),
        ("employee absenteeism by department", "attendance"),
        ("late arrivals by employee", "attendance"),
        ("attendance rate by employee", "attendance"),
        ("leave days by employee", "attendance"),
        ("employees by department", "headcount"),
    ])
    def test_an_employee_is_one_word_however_it_is_spelled(self, question, family):
        rows = [{"GROUP": "A", "X": 10.0}, {"GROUP": "B", "X": 4.0}]
        brief, _user = _prompt(rows, question)
        assert brief["metric_semantics"]["key"] == family
