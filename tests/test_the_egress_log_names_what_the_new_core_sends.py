"""The AI egress log says when a new-core prompt carried the data's values, and when it did not.

Each AI call's record (core/llm_audit.py, the Compliance page's AI egress log) says whether real data values left
in the prompt, read from the prompt itself: a section that carries values starts with a heading only it writes. The
headings it knew were the old engine's, so every call of the new core was recorded "0 data values" -- also the
planner's when it was handed a customer's name, a summary handed 25 rows of the answer, and Learn handed a column's
common values. Now each value-bearing section of the new core is named:

* a member the question names, and the previous plan's member values (planner tail), where values may reach the AI;
* the catalog's listed values and ranges (planner, stable part);
* the answer's rows (summary) and its facts (analyst note);
* a column's common values and ranges (Learn naming), and a field's common values (a metric written from words).

And where member values are withheld (a workspace under compliance), the same prompts are recorded as carrying none.

Prompts are built by the production builders on an invented warehouse; nothing is sent anywhere.
"""

from __future__ import annotations

import json

from core.llm_audit import build_egress_manifest
from core2.service import Session, answer_question
from tests.test_core2_answers_a_question_end_to_end import _domain, _index, _services


class Both:
    """The AI, answering from a script; it keeps both halves of what it was sent."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.sent: list[tuple[str, str]] = []

    def __call__(self, stable: str, tail: str) -> str:
        self.sent.append((stable, tail))
        return self.answers.pop(0)


def _sources(system: str, user: str) -> list[str]:
    manifest = build_egress_manifest(system, user)
    assert manifest["values_sent"] == bool(manifest["value_sources"])
    return manifest["value_sources"]


def _customer(model) -> str:
    return next(a.slug for a in model.attributes.values() if model.columns[a.column].name == "customer_name")


def _plan(slug: str, value: str, year: int, **extra) -> str:
    return json.dumps({"kind": "query", "intent": "value", "measures": ["net_amount"],
                       "filters": [{"field": slug, "op": "eq", "values": [value]}],
                       "time": {"window": {"kind": "between", "start": f"{year}-01-01", "end": f"{year}-12-31"}},
                       **extra})


def _two_turns(values_allowed: bool) -> list[tuple[str, str]]:
    _, built, model, _ = _domain("retail")
    from core2.warehouse.runner import DuckDBWarehouse

    warehouse = DuckDBWarehouse(built.con)
    slug = _customer(model)
    first = "Northline Distribution 58" if values_allowed else "⟨v1⟩"
    ai = Both(_plan(slug, first, 2025), _plan(slug, first, 2024, follow_up="refine"))
    services = _services(model, warehouse, ai, index=_index(model, warehouse), values_allowed=values_allowed)
    session = Session()
    answer_question('net sales for "Northline Distribution 58" in 2025', services, session)
    answer_question("and in 2024?", services, session)
    return ai.sent


# ── the planner ──────────────────────────────────────────────────────────────


def test_a_member_the_question_names_and_the_listed_values_are_recorded_as_sent():
    (stable, tail), _ = _two_turns(values_allowed=True)
    assert "Northline Distribution 58" in tail
    sources = _sources(stable, tail)
    assert "member_matches" in sources and "catalog_values" in sources


def test_a_follow_up_records_the_previous_plans_member_values():
    _, (stable, tail) = _two_turns(values_allowed=True)
    assert "Northline Distribution 58" in tail.rsplit("\nQUESTION:", 1)[0]
    assert "previous_plan_values" in _sources(stable, tail)


def test_with_values_withheld_the_same_questions_are_recorded_as_sending_none():
    for stable, tail in _two_turns(values_allowed=False):
        assert "Northline" not in stable + tail
        assert _sources(stable, tail) == [], (tail, _sources(stable, tail))


# ── the answer's summary and note ────────────────────────────────────────────


def test_a_summary_records_the_answers_rows_and_a_note_its_facts():
    from core2.answer import notes, summary

    payload = {"engine": "core2", "answer": {"headline": "Net amount by region"},
               "data": {"headers": ["region", "net_amount"], "rows": [{"region": "North", "net_amount": 120.0}],
                        "total_rows": 1}}
    assert "answer_rows" in _sources(summary.SYSTEM, summary.material(payload, "net amount by region"))
    reading = {"facts": [{"section": "what", "en": "North is the largest at 120.", "size": 1}]}
    assert "answer_facts" in _sources(notes.SYSTEM, notes._given(reading, "en"))


# ── Learn and the metric writer ──────────────────────────────────────────────


def test_learn_records_a_columns_common_values_and_withholds_them_when_not_allowed():
    from core2.bootstrap import labels
    from core2.model.schema import SemanticModel

    _, _, model, _ = _domain("retail")
    sent: list[tuple[str, str]] = []

    def complete(system: str, user: str) -> str:
        sent.append((system, user))
        return '{"tables": {}}'

    labels.label(model, complete)
    assert any("profile_values" in _sources(s, u) for s, u in sent)
    closed = SemanticModel.model_validate(model.model_dump())
    for column in closed.columns.values():
        column.values_allowed = False
    sent.clear()
    labels.label(closed, complete)
    assert sent and not any(_sources(s, u) for s, u in sent), [_sources(s, u) for s, u in sent]


def test_a_metric_written_from_words_records_the_fields_common_values():
    from core2.model.authoring import describe_prompt
    from core2.model.schema import SemanticModel

    _, _, model, _ = _domain("retail")
    assert "field_values" in _sources(describe_prompt(model), "The admin's description: sales to wholesale")
    closed = SemanticModel.model_validate(model.model_dump())
    for column in closed.columns.values():
        column.values_allowed = False
    assert _sources(describe_prompt(closed), "The admin's description: sales to wholesale") == []


# ── ranges and spans go where values go ─────────────────────────────────────


def test_with_values_withheld_dates_are_spanned_by_their_years_only():
    import re

    from core2.plan.catalog import catalog_text

    _, _, model, _ = _domain("retail")
    shown, withheld = catalog_text(model), catalog_text(model, values_allowed=False)
    dates = lambda text: text.split("DATES (", 1)[1].split("\n\n", 1)[0]
    assert re.search(r"\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", dates(shown))
    assert re.search(r"\| \d{4} to \d{4}", dates(withheld)) and not re.search(r"\d{4}-\d{2}", dates(withheld))
