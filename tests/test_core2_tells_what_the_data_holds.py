"""The new core answers questions about the data itself, read off what it learned.

"What can you tell me about my data?" was answered with one paragraph of bold
headline: every subject's first six measure names run together, "…" where it
stopped, and the breakdowns after them -- beside today's pipeline's readable
summary. The answer is now built from the model, never written by the AI: a
one-sentence lead naming the subjects and the period the data covers, a section
per subject (its measures, what they break down by, its dates), a line saying
what can be asked, and example questions that each answer.

A question about one thing gets that thing: a measure's definition, how it adds
up, the date it is counted by and the period covered; a breakdown's size and
what it groups by; a date's range; a subject in full. A reader limited to some
tables is told only about those. Synthetic retail warehouse, recorded AI.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from core2.answer.describe import describe
from core2.plan.ir import Plan
from core2.plan.planner import RULES, stable_prompt
from core2.resolve.resolver import Context, resolve
from core2.compile.compiler import compile_query
from core2.service import Services, Session, answer_question
from core2.plan.values import MemberIndex
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.compile_eval import learn

TODAY = dt.date(2026, 6, 15)


@pytest.fixture(scope="module")
def retail():
    built, model = learn(domains.build("retail"), "descriptive")
    return model, built


def _bullets(said) -> list[str]:
    return [b for s in said.sections for b in s["bullets"]]


def _plan_for(model, question: str) -> Plan:
    """An example question as the plan it names: "<measure> by <breakdown|month> in <year>"."""
    found = re.fullmatch(r"(.+) by (.+?)(?: in (\d{4}))?", question)
    assert found, question
    measure, by, year = found.groups()
    slug = next(m.slug for m in model.measures.values() if m.business_name == measure)
    plan: dict = {"kind": "query", "intent": "breakdown", "measures": [slug]}
    if by == "month":
        plan.update(intent="trend", time={"grain": "month"})
    else:
        plan["group_by"] = [next(e.slug for e in model.entities.values() if e.business_name.lower() == by)]
    if year:
        plan.setdefault("time", {})["window"] = {"kind": "between", "start": f"{year}-01-01", "end": f"{year}-12-31"}
    return Plan.model_validate(plan)


def test_the_whole_data_is_a_lead_a_section_per_subject_and_questions_that_answer(retail):
    model, built = retail
    said = describe(model, [], today=TODAY)
    # The period is what happened: the sales targets run to December 2026, which is not data yet.
    assert said.headline == ("This data covers 3 subjects: Order line, Return and Sales target, "
                             "with data from January 2023 to June 2026.")
    assert said.note.startswith("Ask for a total, a breakdown, a trend")
    assert [s["title"] for s in said.sections] == ["Order line", "Return", "Sales target"]
    order_line = said.sections[0]["bullets"]
    assert order_line[0].startswith("Measures: net amount, gross amount")         # sales first, costs after
    assert order_line[1] == "Broken down by: customer, product, store, category and region"
    assert order_line[2] == "Dates: order date (the default), delivered date and ship date; January 2023 to June 2026"
    assert "…" not in json.dumps(said.sections)                                   # nothing cut mid-list
    # Every example question names a measure and a breakdown that reach each other, in a year with data.
    assert said.examples == ["Net amount by store in 2026", "Refund amount by category in 2026",
                             "Target amount by store in 2026", "Net amount by month in 2026"]
    warehouse = DuckDBWarehouse(built.con)
    for question in said.examples:
        logical = resolve(_plan_for(model, question), model, Context(today=TODAY))
        rows = warehouse.query(compile_query(logical, model, "duckdb").sql).rows
        assert rows, question


def test_a_measure_is_its_definition_how_it_adds_up_and_when(retail):
    model, _ = retail
    said = describe(model, ["net_amount"], today=TODAY)
    assert said.headline == "Net amount is the total net amount across order lines."
    assert _bullets(said) == [
        "It adds up across any breakdown and period.",
        "Amounts are in USD.",
        "It is counted by order date, from January 2023 to June 2026.",
        "It can be broken down by customer, product, store, category and region.",
    ]
    assert said.examples == ["Net amount by month in 2026", "Net amount by store in 2026"]
    orders = describe(model, ["number_of_orders"], today=TODAY)
    assert orders.headline == "Number of orders is the number of distinct order numbers across order lines."
    margin = describe(model, ["margin_percent"], today=TODAY)
    assert "It does not add up: it is worked out again for each breakdown and period." in _bullets(margin)


def test_a_breakdown_is_its_size_what_groups_it_and_what_it_breaks_down(retail):
    model, _ = retail
    said = describe(model, ["customer"], today=TODAY)
    assert said.headline == "There are 240 customers in the data, each named by its customer name."
    assert _bullets(said)[0] == "They can be grouped and filtered by city, customer code and segment."
    assert said.examples[0] == "List the customers"
    segment = describe(model, ["customer.segment"], today=TODAY)
    assert segment.headline == "Segment has 3 values: Online, Retail and Wholesale."
    # Where member values stay out of answers built for prompts, they stay out of this one too.
    assert describe(model, ["customer.segment"], today=TODAY, values=False).headline == "Segment has 3 values."


def test_a_date_is_its_range_beside_the_others(retail):
    model, _ = retail
    said = describe(model, ["order_date"], today=TODAY)
    assert said.headline == "Order date (order line) runs from January 2023 to June 2026."
    assert "Return (return date): 19 January 2023 to 14 June 2026." in _bullets(said)


def test_a_subject_named_as_the_catalog_names_it_is_described_in_full(retail):
    model, _ = retail
    said = describe(model, ["## Return"], today=TODAY)
    assert said.headline == "Here is what you can ask about Return."
    assert describe(model, ["returns"], today=TODAY).headline == said.headline        # as a reader says it
    assert describe(model, ["time:month_of_year"], today=TODAY).headline.startswith("This data covers")
    assert said.sections[0]["bullets"][0] == "Measures: refund amount and return quantity"


def test_what_the_data_does_not_have_is_said_and_the_rest_offered(retail):
    model, _ = retail
    said = describe(model, ["ebitda"], today=TODAY)
    assert said.headline == "This data has nothing called ebitda."
    assert said.note.startswith("This data covers 3 subjects") and len(said.sections) == 3


def test_a_reader_is_told_only_about_the_tables_they_may_use(retail):
    model, _ = retail
    returns = next(k for k, t in model.tables.items() if t.business_name == "Return")
    said = describe(model, [], today=TODAY, allowed=set(model.tables) - {returns})
    text = json.dumps([said.headline, said.note, said.sections, said.examples])
    assert said.headline.startswith("This data covers 2 subjects: Order line and Sales target")
    assert "Refund" not in text and "refund" not in text and "Return" not in text
    assert describe(model, ["refund_amount"], today=TODAY, allowed=set(model.tables) - {returns}).headline \
        == "This data has nothing called refund amount."


class _Recorded:
    def __init__(self, answer: dict):
        self.answer, self.sent = json.dumps(answer), []

    def __call__(self, stable, tail):
        self.sent.append(tail)
        return self.answer


def test_through_the_service_it_is_a_card_with_sections_chips_and_no_query(retail):
    model, built = retail

    class Watched(DuckDBWarehouse):
        ran: list[str] = []

        def query(self, sql, *, max_rows=None):
            Watched.ran.append(sql)
            return super().query(sql, max_rows=max_rows)

    services = Services(model=model, warehouse=Watched(built.con), complete=_Recorded({"kind": "describe_data"}),
                        index=MemberIndex(), today=TODAY, data_source="duckdb")
    frame = answer_question("what can you tell me about my data?", services, Session())
    assert frame["type"] == "assistant_response" and frame["engine"] == "core2" and frame["data"] is None
    assert frame["answer"]["headline"].startswith("This data covers 3 subjects")
    assert frame["answer"]["scope_note"].startswith("Ask for a total")
    assert [s["title"] for s in frame["sections"]] == ["Order line", "Return", "Sales target"]
    assert [c["question"] for c in frame["follow_up_suggestions"]][0] == "Net amount by store in 2026"
    assert not Watched.ran and not frame.get("unsupported")
    json.dumps(frame)
    asked = answer_question("how is net amount calculated?", Services(
        model=model, warehouse=Watched(built.con), complete=_Recorded({"kind": "describe_data", "about": ["net_amount"]}),
        index=MemberIndex(), today=TODAY), Session())
    assert asked["answer"]["headline"] == "Net amount is the total net amount across order lines."


def test_the_planner_is_told_which_questions_are_about_the_data_itself(retail):
    model, _ = retail
    assert '"what does X mean", "how is X calculated", "what dates does the data cover"' in RULES.replace("\n   ", " ")
    prompt = stable_prompt(model)
    assert re.search(r'Q: how is [a-z ]+ calculated\?\nA: \{"kind": "describe_data", "about": \["[a-z_]+"\]\}', prompt)
    assert '"about"' in prompt          # the plan's schema carries the field


def test_an_admins_count_does_not_lead_and_a_budget_set_ahead_asks_about_this_year(retail):
    model, _ = retail
    copy = model.model_copy(deep=True)
    lines = next(m for m in copy.measures.values() if m.slug == "number_of_order_lines")
    lines.kind, lines.business_name = "import", "Number of sales lines"         # an admin's metric, still a count
    gross = next(m for m in copy.measures.values() if m.slug == "gross_amount")
    gross.business_name = "Net amount"                                          # two columns, one name
    said = describe(copy, [], today=TODAY)
    listed = said.sections[0]["bullets"][0]
    assert listed.startswith("Measures: net amount, quantity") and said.examples[0].startswith("Net")
    assert listed.endswith("and number of sales lines")
    _, finance = learn(domains.build("finance"), "descriptive")
    budget = describe(finance, [], today=TODAY)
    assert "Budget" in budget.headline and budget.examples and not any("2027" in q for q in budget.examples)
    assert budget.headline.endswith("with data from July 2023 to June 2026.")   # the budget runs to June 2027


def test_an_example_never_needs_the_reader_to_name_a_role():
    """Customers reached as bill-to or ship-to: "order amount by customer" would come back as a question."""
    import numpy as np
    import pandas as pd

    from tests.test_core2_asks_which_role_and_follows_it import _learn

    rng = np.random.default_rng(3)
    n = 3000
    customers = pd.DataFrame({"customer_id": np.arange(1, 41), "customer_name": [f"Client {i:02d}" for i in range(1, 41)]})
    stores = pd.DataFrame({"store_id": np.arange(1, 7), "store_name": [f"Store {i}" for i in range(1, 7)]})
    bill = rng.integers(1, 41, n)
    orders = pd.DataFrame({"order_id": np.arange(1, n + 1), "bill_to_customer_id": bill,
                           "ship_to_customer_id": np.where(rng.random(n) < 0.6, bill, rng.integers(1, 41, n)),
                           "store_id": rng.integers(1, 7, n), "order_amount": np.round(rng.uniform(10, 900, n), 2)})
    _, model = _learn({"customers": customers, "stores": stores, "orders": orders})
    said = describe(model, [], today=TODAY)
    assert said.examples and all("by customer" not in q for q in said.examples), said.examples
    assert any(q.endswith("by store") for q in said.examples), said.examples
    _, alone = _learn({"customers": customers, "orders": orders.drop(columns=["store_id"])})
    assert not [q for q in describe(alone, [], today=TODAY).examples if " by " in q and "month" not in q]
