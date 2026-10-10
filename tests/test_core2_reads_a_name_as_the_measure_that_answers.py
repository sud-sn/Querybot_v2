"""A name that is both a member and a measure is read as the one that answers; a refusal says what it looked at.

On a real workspace "Cost of goods sold by item group in Q1 2026" was refused:
"Item Group Name is not linked to Finance". The words found a general-ledger
account of that name, the plan summed the ledger for that account, and the
ledger has no link to item groups -- while the cost on the invoice lines, named
"Cost Amount", does. Renamed "Cost of goods sold" by an admin, the question was
answered. The planner now reads such a name as the measure when the member's
table cannot be broken down as asked: the rule says so, and a plan that fails
that way goes back with the measures that do reach the breakdown, those sharing
the question's words first.

And the refusal itself showed nothing under "How this answer was produced" but
"No SQL query was executed": it now shows the measure considered and its table,
the breakdown, the filters and the dates, and why it stopped.

The warehouse is the synthetic retail one with a general ledger added beside it,
linked to nothing: "Cost of goods sold" is one of its accounts, and "Cost
amount" is on the order lines, which reach the product categories.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from core2.bootstrap.build import BuildOptions, build_model
from core2.bootstrap.inventory import from_duckdb
from core2.plan.planner import RULES, check
from core2.plan.values import build_index
from core2.service import Services, Session, answer_question
from core2.warehouse.runner import DuckDBWarehouse
from evals.core2 import domains
from evals.core2.framework import materialize

TODAY = dt.date(2026, 6, 15)
QUESTION = "Cost of goods sold by category in Q1 2026"
Q1 = {"kind": "between", "start": "2026-01-01", "end": "2026-03-31"}
# The member reading: the ledger's amount for the account the words found.
LEDGER = {"kind": "query", "intent": "breakdown", "measures": ["amount"], "group_by": ["category.name"],
          "filters": [{"field": "gl_line.account_name", "op": "eq", "values": ["Cost of goods sold"]}],
          "time": {"window": Q1}}
# The measure reading: the cost on the order lines.
COST = {"kind": "query", "intent": "breakdown", "measures": ["cost_amount"], "group_by": ["category.name"],
        "time": {"window": Q1}}


@pytest.fixture(scope="module")
def ledger():
    built = materialize(domains.build("retail"), "descriptive")
    con = built.con
    con.execute("""CREATE TABLE gl_lines AS SELECT i AS gl_line_id,
        CASE i % 3 WHEN 0 THEN 'Cost of goods sold' WHEN 1 THEN 'Rent' ELSE 'Wages' END AS account_name,
        CAST(100 + (i * 37) % 900 AS DECIMAL(12, 2)) AS amount,
        DATE '2025-01-01' + CAST(i % 500 AS INTEGER) AS posting_date FROM range(1, 1501) t(i)""")
    con.execute("ALTER TABLE gl_lines ADD PRIMARY KEY (gl_line_id)")
    warehouse = DuckDBWarehouse(con)
    model = build_model(warehouse, from_duckdb(warehouse, declared_fks=built.declared_fks),
                        options=BuildOptions(workers=1))
    return model, warehouse


class Recorded:
    def __init__(self, *answers: dict):
        self.answers = [json.dumps(a) for a in answers]
        self.sent: list[str] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append(tail)
        return self.answers.pop(0)


def _index(model, warehouse):
    def fetch(slug):
        column = model.columns[model.attributes[slug].column]
        return [r[0] for r in warehouse.query(f'SELECT DISTINCT "{column.name}" FROM '
                                              f'{model.tables[column.table].name}').rows]
    return build_index(model, fetch)


def _ask(model, warehouse, ai, **kw):
    services = Services(model=model, warehouse=warehouse, complete=ai, index=_index(model, warehouse), today=TODAY,
                        data_source="duckdb", **kw)
    return answer_question(QUESTION, services, Session())


def test_the_words_find_the_ledger_account(ledger):
    model, warehouse = ledger
    matches = _index(model, warehouse).match(QUESTION)
    assert [(m.attribute, m.value) for m in matches] == [("gl_line.account_name", "Cost of goods sold")]


def test_the_rule_tells_the_planner_to_take_the_reading_that_answers():
    assert "A name in VALUE MATCHES can also be what a measure measures" in " ".join(RULES.split())
    assert "otherwise use the measure" in RULES


def test_a_member_reading_that_cannot_be_broken_down_goes_back_with_the_measures_that_can(ledger):
    model, warehouse = ledger
    ai = Recorded(LEDGER, COST)
    payload = _ask(model, warehouse, ai)
    repair = ai.sent[1]
    assert "- Category name is not linked to GL line." in repair
    reaching = repair.split("Measures that can be broken down that way: ", 1)[1].split(". If the question", 1)[0]
    listed = reaching.split(", ")
    # Those sharing the question's words first ("cost", "category"), then by name.
    assert listed[:3] == ["cost_amount (Cost amount)", "number_of_categories (Number of categories)",
                          "discount_amount (Discount amount)"]
    assert len(listed) == 8 and not any(name.startswith(("amount ", "number_of_gl_lines")) for name in listed)
    assert "use that measure instead" in repair
    # The plan the repair returns is answered: the cost of the order lines, by category.
    assert payload["data"] and payload["trust"]["plan_repaired"] is True
    assert "cost_amount" in payload["trust"]["sql"].lower() and "gl_lines" not in payload["trust"]["sql"].lower()


def test_the_hint_is_for_the_planner_and_the_problem_for_the_reader(ledger):
    model, _ = ledger
    from core2.plan.ir import Plan

    problems, hint = check(Plan.model_validate(LEDGER), model, TODAY, QUESTION)
    assert problems == ["Category name is not linked to GL line."]
    assert hint.startswith("Measures that can be broken down that way: cost_amount (Cost amount), ")
    assert check(Plan.model_validate(COST), model, TODAY, QUESTION) == ([], "")
    _, hint = check(Plan.model_validate(LEDGER), model, TODAY, "refunds by category")
    assert hint.index("refund_amount (Refund amount)") < hint.index("cost_amount (Cost amount)")


def test_when_the_repair_fails_too_the_question_back_says_what_was_looked_at_and_why_it_stopped(ledger):
    model, warehouse = ledger
    payload = _ask(model, warehouse, Recorded(LEDGER, LEDGER))
    assert payload["clarify"] and payload["data"] is None
    assert payload["trust"]["considered"] == [
        {"key": "measure", "value": "Amount (GL line)"},
        {"key": "by", "value": "Category name (Category)"},
        {"key": "filter", "value": "Account name (GL line) is Cost of goods sold"},
        {"key": "dates", "value": "2026-01-01 to 2026-03-31"},
    ]
    assert payload["trust"]["stopped"] == "Category name is not linked to GL line."   # no planner hint for the reader


def test_a_repair_that_gives_up_still_says_what_was_looked_at(ledger):
    model, warehouse = ledger
    gave_up = {"kind": "unsupported", "notes": ["the ledger has no link to categories"]}
    payload = _ask(model, warehouse, Recorded(LEDGER, gave_up))
    assert payload["unsupported"] and payload["answer"]["headline"] == (
        "I cannot answer that from this data: the ledger has no link to categories")
    assert [c["value"] for c in payload["trust"]["considered"]][:2] == ["Amount (GL line)", "Category name (Category)"]
    assert payload["trust"]["stopped"] == "the ledger has no link to categories"
    assert payload["trust"]["data_source"] == "duckdb" and "sql" not in payload["trust"]


def test_a_refusal_after_the_plan_says_what_was_looked_at(ledger):
    model, warehouse = ledger
    by_year = {**COST, "time": {"window": Q1, "grain": "month", "compare": {"kind": "same_period_last_year"}}}
    categories = next(k for k, t in model.tables.items() if t.business_name == "Category")
    payload = _ask(model, warehouse, Recorded(by_year), allowed_tables=set(model.tables) - {categories})
    assert "do not have access" in payload["answer"]["headline"]
    assert payload["trust"]["considered"] == [
        {"key": "measure", "value": "Cost amount (Order line)"},
        {"key": "by", "value": "Category name (Category)"},
        {"key": "dates", "value": "2026-01-01 to 2026-03-31, month by month, compared with the same period last year"},
    ]
    assert payload["trust"]["stopped"] == "Category is not available to you"


def test_a_query_the_database_refuses_keeps_its_sql_and_says_why_it_stopped(ledger):
    model, warehouse = ledger

    class Refusing:
        dialect, db_type = warehouse.dialect, warehouse.db_type

        def query(self, sql, *, max_rows=None):
            if "cost_amount" in sql:      # the member index's reads still answer
                raise RuntimeError("[42000] The SELECT permission was denied (229)")
            return warehouse.query(sql, max_rows=max_rows)

    payload = _ask(model, Refusing(), Recorded(COST))
    assert payload["answer"]["headline"] == ("The question was understood but the query could not run "
                                             "(the database refused it).")
    assert payload["trust"]["stopped"] == "The query could not run: the database refused it"
    assert payload["trust"]["sql"] and payload["trust"]["considered"][0]["value"] == "Cost amount (Order line)"


def test_the_portal_has_a_label_for_everything_a_refusal_says_in_each_language():
    from core.i18n import catalogue_for

    for lang in ("en", "fr"):
        labels = catalogue_for(lang)
        for key in ("stopped", "considered.measure", "considered.by", "considered.filter", "considered.dates"):
            assert labels.get(f"ui.chat.trust.{key}"), (lang, key)


def test_for_a_regulated_workspace_the_hint_ranks_by_the_question_the_ai_saw(ledger):
    """Member values never reach the AI there: the ranking must not say which words a placeholder hides.

    (So there the AI cannot read the account's name as the cost measure either: it never sees the words.)
    """
    model, warehouse = ledger
    as_seen = {**LEDGER, "filters": [{**LEDGER["filters"][0], "values": ["\u27e8v1\u27e9"]}]}   # the placeholder
    ai = Recorded(as_seen, COST)
    _ask(model, warehouse, ai, values_allowed=False)
    assert "QUESTION: \u27e8v1\u27e9 by category in Q1 2026" in ai.sent[0]
    assert all("Cost of goods sold" not in sent and "cost of goods sold" not in sent.casefold() for sent in ai.sent)
    reaching = ai.sent[1].split("Measures that can be broken down that way: ", 1)[1].split(", ")
    # "category" is in what the AI saw; "cost" was inside the masked member name.
    assert reaching[0] == "number_of_categories (Number of categories)"
