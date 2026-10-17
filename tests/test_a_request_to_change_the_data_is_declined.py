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
                                      "add the count next to it"])
def test_a_question_with_the_same_words_is_a_question(question):
    assert changes_data(question) is None


@pytest.mark.parametrize("request_", [
    # Requests that write a value or act on records stay declined, whatever words about the answer they hold.
    "Update the total of invoice 42 to $500", "Set the rate for all loans to 5%", "Cancel the open orders and list them",
    "Remove all the cancelled orders, then show the rest", "Change the cost of every item in category Bakery to 2.50",
    "Reset the counts for all stores", "Mark the invoices as paid instead"])
def test_a_request_that_writes_stays_declined(request_):
    assert changes_data(request_) is not None


def test_it_is_declined_before_any_planning():
    asked = []
    services = Services(model=SemanticModel(client_id="t"), warehouse=None,
                        complete=lambda s, t: asked.append(t) or "{}", index=MemberIndex(), today=dt.date(2026, 6, 1))
    payload = answer_question("Supprime les fiches des employés partis avant 2024", services, Session())
    assert not asked
    assert payload["kind"] == "changes_data" and "jamais les modifier" in payload["answer"]["headline"]
    assert payload["follow_up_suggestions"][0]["question"].startswith("Montre les fiches")
