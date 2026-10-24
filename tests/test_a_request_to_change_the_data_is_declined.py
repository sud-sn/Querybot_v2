"""A request to change the data is declined, in the reader's language, with the rows it names one click away.

"Supprime les mouvements annulés de 2024" was answered with a count of them, and "close out all the open
bookings from 2024 for me" with a list of 5,000 booking numbers, as if done. Solviz reads data and never changes
it: such a request is told so before any planning, and offered as a question that shows the rows it names. A
question that only uses the same words ("change in sales", "close rate by rep", "cancelled orders by month") is
a question.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core2.model.schema import SemanticModel
from core2.plan.values import MemberIndex
from core2.service import Services, Session, answer_question, changes_data


@pytest.mark.parametrize("request_, shown", [
    ("Supprime les mouvements annulés de 2024, s'il te plaît", "Montre les mouvements annulés de 2024"),
    ("Supprime les fiches des employés partis avant 2024", "Montre les fiches des employés partis avant 2024"),
    ("Delete all the orders from the two test customers", "Show the orders from the two test customers"),
    ("Delete orders up to 2020", "Show the orders up to 2020"),
    ("close out all the open bookings from 2024 for me", "Show the open bookings from 2024"),
    ("Set the status of the open bookings to closed", "Show the status of the open bookings"),
    ("Mark all the 2023 invoices as paid", "Show the 2023 invoices"),
    # A price or a figure put up or down is a change too: "Increase the price of all air fryers by 10%" was answered
    # with last month's air fryer sales.
    ("Increase the price of all air fryers by 10%", "Show the price of all air fryers"),
    ("Raise the credit limit of all the Hospitality customers to $50,000", "Show the credit limit of all the "
                                                                           "Hospitality customers"),
    ("Reduce the discount on the clearance promotions by 5 points", "Show the discount on the clearance promotions"),
    ("Augmente le prix des friteuses de 10 %", "Montre les prix des friteuses"),
])
def test_a_request_to_change_the_data_names_the_rows_to_show(request_, shown):
    assert changes_data(request_) == shown


@pytest.mark.parametrize("question", ["Show the open bookings", "How many orders were cancelled?",
                                      "Cancelled orders by month", "Change in sales by region", "Close rate by rep",
                                      "Add up the sales by region", "Update me on sales this month",
                                      "Removed items by category", "Increase in sales by month",
                                      "Reduced price items by category", "Raised invoices this week",
                                      "Lower margin products in 2025",
                                      # A change to the answer on screen, never to the data (the bank warehouse's
                                      # follow-up was declined as a request to change records).
                                      "drop the count and the average, and show each balance as % of the "
                                      "region's total instead", "remove the cost column", "drop the margin",
                                      "add the count next to it",
                                      # Words that change data only as a command, never as a reading.
                                      "Refund rate by store", "Write-offs by month", "Waived fees by month",
                                      "Create a dashboard of sales by region",
                                      "write off amounts by region in 2025", "refund this year vs last year",
                                      "charge this year by month", "credit card 2025 spend by state",
                                      "refund orders 2025 by month", "add a refund rate", "add a discount breakdown",
                                      "apply a 10% discount to all 2025 orders and show the new revenue",
                                      "apply a 10% discount to last year's sales: what would revenue be?"])
def test_a_question_with_the_same_words_is_a_question(question):
    assert changes_data(question) is None


@pytest.mark.parametrize("request_", [
    # Requests that write a value or act on records stay declined, whatever words about the answer they hold.
    "Update the total of invoice 42 to $500", "Set the rate for all loans to 5%", "Cancel the open orders and list them",
    "Remove all the cancelled orders, then show the rest", "Change the cost of every item in category Bakery to 2.50",
    "Reset the counts for all stores", "Mark the invoices as paid instead"])
def test_a_request_that_writes_stays_declined(request_):
    assert changes_data(request_) is not None


@pytest.mark.parametrize("edit", [
    "add the units sold and how many orders each one was on", "add the sales rep who took the order",
    "drop the practice and add how many fills we dispensed", "remove the West region",
    "ajoute le magasin de chaque client", "add the region of each order"])
def test_with_an_answer_on_screen_adding_or_taking_off_a_field_changes_the_answer(edit):
    # "add the units sold ..." after a report was declined as a request to add records.
    assert changes_data(edit) is not None, "with nothing on screen it reads as a change to the data"
    assert changes_data(edit, on_screen=True) is None


@pytest.mark.parametrize("edit", [
    "add the units sold to the table", "add the margin next to net sales", "add the number of orders on each one",
    "add the category to each order line", "add the margin column to each order", "add the customer name to each order",
    "add the payer to the claims", "add the reject reason to each claim", "add the balance to each account",
    "add the city to every order", "remove the flag column", "remove the holds column", "ajoute la région à chaque commande",
    "remove the products in books", "drop the stores from the books category", "add the discount to each order line",
    "add the shipping fee to each order line", "add the credit to each customer", "add the rebate to each customer",
    "add the late fee to the customers", "remove the rows for customer 1042", "add a discount column",
    "add the cost to each order", "add the cost to each order instead", "add the unit cost to each order",
    "add the costs to the invoices", "add the refund amount to each invoice", "add the author to the books",
    "add the price to the books", "add a credit limit column", "add a payment method breakdown",
    "ajoute une colonne crédit", "ajoute le nombre de commandes 2024"])
def test_a_field_put_beside_the_answer_is_never_read_as_put_on_records(edit):
    assert changes_data(edit, on_screen=True) is None


@pytest.mark.parametrize("request_", [
    "add the new customer Acme", "add the refund to the ledger", "remove all the cancelled orders",
    "delete the West region", "add the 5 units to the stock of Pasta", "add the order as paid",
    "ajoute les nouvelles fiches client", "drop every store in the West",
    # Something put on records or taken off them, however little it names.
    "add the shipping fee to every order", "add the discount to order SO-1001", "add the late fee to Acme's account",
    "remove the cancelled orders from the books", "remove the hold on these accounts",
    "remove the refund on order 5512", "ajoute la remise à la commande 5512",
    # Anything put on a numbered record, costs and vouchers on every order, rows on or off the books, holds lifted.
    "add the coupon to order SO-1001", "add the item to order 5512", "add the shipping cost to every order",
    "add the voucher to every order", "add the tag to these accounts", "add the refund to the books",
    "add these invoices to the books", "remove the test orders from the books for 2024", "remove the hold for these accounts",
    "remove the freeze for these accounts", "remove the holds", "remove the credit from customer 1042",
    "ajoute les frais de port à chaque commande", "ajoute les produits à la commande 5512",
    "retire les frais sur la facture 77", "ajoute la note à la facture 77",
    # Money on records named with their owner ("customer accounts"), these put on the books.
    "add the late fee to all customer accounts", "add the credit to the customer account",
    "add the late fee to customers' accounts", "add the discount to all customer orders",
    "add the late fee to each customer invoice", "add the late fee to customer Acme's account",
    "add these to the books", "add a $50 credit to each customer"])
def test_with_an_answer_on_screen_a_new_record_or_every_record_is_still_data(request_):
    assert changes_data(request_, on_screen=True) is not None


@pytest.mark.parametrize("request_, shown", [
    ("add a $50 credit to customer 1042", "Show the customer 1042"),
    ("credit customer 1042 with $50", "Show the customer 1042"),
    ("apply a 10% discount to every order", "Show the orders"),
    ("add a late fee to every order", "Show the orders"),
    ("write off these invoices", "Show the invoices"),
    ("waive the late fee on invoice INV-77", "Show the invoice INV-77"),
    ("ajoute un crédit de 50 € au client 1042", "Montre le client 1042"),
    ("add a late fee to every order in 2025", "Show the orders in 2025"),
    ("charge customers 1042 and 1043 $5", "Show the customers 1042 and 1043"),
    # A hold put on or lifted, money applied: verbs that never edit the answer.
    ("lift the hold on account 7", "Show the account 7"), ("put a hold on customer 1042", "Show the customer 1042"),
    ("apply the discount to every order", "Show the orders"),
    ("applique la remise à toutes les commandes", "Montre les commandes")])
def test_money_put_on_a_record_is_declined_with_the_record_to_show(request_, shown):
    # "Add a $50 credit to customer 1042" was answered as a question, with or without an answer on screen.
    assert changes_data(request_) == shown
    assert changes_data(request_, on_screen=True) == shown


def test_it_is_declined_before_any_planning():
    asked = []
    services = Services(model=SemanticModel(client_id="t"), warehouse=None,
                        complete=lambda s, t: asked.append(t) or "{}", index=MemberIndex(), today=dt.date(2026, 6, 1))
    payload = answer_question("Supprime les fiches des employés partis avant 2024", services, Session())
    assert not asked
    assert payload["kind"] == "changes_data" and "jamais les modifier" in payload["answer"]["headline"]
    assert payload["follow_up_suggestions"][0]["question"].startswith("Montre les fiches")


def test_with_an_answer_on_screen_add_a_field_goes_to_the_planner():
    # "add the units sold and how many orders each one was on" after a top-5 list was declined before any planning.
    from core2.plan.ir import Plan
    from core2.plan.planner import Turn

    asked = []
    services = Services(model=SemanticModel(client_id="t"), warehouse=None,
                        complete=lambda s, t: asked.append(t) or "{}", index=MemberIndex(), today=dt.date(2026, 6, 1))
    session = Session(turns=[Turn("Top 5 products by net sales in May 2026",
                                  Plan.model_validate({"kind": "query", "intent": "rank", "measures": ["net_amount"]}))])
    payload = answer_question("add the units sold and how many orders each one was on", services, session)
    assert payload.get("kind") != "changes_data" and asked, "planned as a change to the answer on screen"
    payload = answer_question("add the units sold and how many orders each one was on", services, Session())
    assert payload["kind"] == "changes_data", "with nothing on screen it is still read as a change to the data"


@pytest.mark.parametrize("request_, shown", [
    # Typed as people type them: a record's short name and code with no "the" ("update cust C10001 credit limit to
    # 50k" was answered as a question about what Solviz can do), and a new value in thousands.
    ("update cust C10001 credit limit to 50k", "Show the cust C10001 credit limit"),
    ("set acct 1042 status to closed", "Show the acct 1042 status"),
    ("change item 123 price to 5", "Show the item 123 price"),
    ("cancel order SO-1001", "Show the order SO-1001"),
    ("mark inv 1234 as paid", "Show the inv 1234"),
    ("pls update vendor V-22 terms to net 30", "Show the vendor V-22 terms"),
    ("raise the credit limit for acct 1042 to 60k", "Show the credit limit for acct 1042"),
    # Found in review: the currency after the new value was shown as part of what to show.
    ("update item 123 price to 12.50 USD", "Show the item 123 price"),
    ("update order #42 status to shipped", "Show the order #42 status"),
])
def test_a_record_named_by_its_short_name_and_code_is_declined(request_, shown):
    assert changes_data(request_) == shown


@pytest.mark.parametrize("question", ["change in cust C10001 sales by month", "update on order SO-1001",
                                      "close rate for rep 12", "cancelled orders for cust C10001",
                                      "set of customers in region 3",
                                      # A year after the record's word is a period, never its number.
                                      "cut orders 2024 by region", "create invoice 2024 summary",
                                      "increase orders 2025 vs 2024", "add orders 2025 to the comparison",
                                      # Found in review: a quarter, a fiscal year, a short year or a number of days
                                      # read as the record's code.
                                      "cut orders q3 by region", "cut orders FY24 by region", "cut inv q1 by cust",
                                      "increase orders q2 vs q1", "cut orders 24 by region",
                                      "add orders q3 to the comparison", "create invoice q1 summary",
                                      "cut invoices 30 days past due by customer", "cut orders h2 by rep",
                                      # And in a second review: "cut" slices, and more ways to write a period or an
                                      # amount.
                                      "cut sku 1234 sales by week", "cut orders wk12 by store",
                                      "cut orders m12 by region", "cut orders 2024q3 by region",
                                      "cut orders FY24Q1 by region", "cut orders 202501 by region",
                                      "cut invoices 120+ days past due by region",
                                      "cut orders 100 units or more by region"])
def test_a_question_about_a_record_by_its_code_is_a_question(question):
    assert changes_data(question) is None
