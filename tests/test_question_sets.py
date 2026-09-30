"""
Every question of every question set (tests/question_sets), in English and in
French, asked of its warehouse through the product, and held to the answer its
reference query computes from the warehouse itself.

A wrong figure fails. So does a question the product answered without the
model that comes to need it, and one it answered that it no longer does. Where
a set records how the product falls short today -- a question it needs the
model for, or one it gets wrong -- that is expected, and the day it changes the
test says so, so the note goes.
"""

from __future__ import annotations

import pytest

from tests import answer_harness
from tests import question_sets as qs
from tests.question_sets import distribution, ledger, outfitters

SETS = (distribution, outfitters, ledger)
CASES = [(module, question, lang) for module in SETS for question in module.QUESTIONS for lang in ("en", "fr")]


@pytest.fixture(scope="session")
def roots(tmp_path_factory):
    return {module.NAME: tmp_path_factory.mktemp(f"question-set-{module.NAME}") for module in SETS}


def test_every_question_is_asked_in_both_languages_and_its_ids_are_its_own():
    ids = [question.id for module in SETS for question in module.QUESTIONS]
    assert len(ids) == len(set(ids))
    for module in SETS:
        for question in module.QUESTIONS:
            assert question.en.strip() and question.fr.strip(), question.id
            assert question.reference.strip() or question.kind == "no data", question.id
            assert set(question.today) <= {"en", "fr"}, question.id


@pytest.mark.parametrize("module,question,lang", CASES,
                         ids=[f"{module.NAME}-{question.id}-{lang}" for module, question, lang in CASES])
def test_the_answer(module, question, lang, roots):
    with module.tenant_in(roots[module.NAME]) as warehouse:
        expected = qs.reference_rows(warehouse, question.reference) if question.reference else []
        if question.kind == "answer":
            assert expected, f"{question.id}: the reference query finds nothing to answer with"
        answer = module.ask(warehouse, question.en if lang == "en" else question.fr, lang, question.choose)
    verdict, why = qs.verdict(question, answer, expected, answer_harness.MARKER)
    note = question.today.get(lang, "")
    if verdict == "PASS":
        assert not note, f"{question.id} {lang} is answered right now; the set still says: {note}"
        return
    if verdict == "NEEDS THE MODEL" and note.startswith("model"):
        pytest.skip(f"the release runner asks this of the real model ({note})")
    if note and not note.startswith("model"):
        pytest.xfail(f"{note} -- {verdict}: {why}")
    pytest.fail(f"{question.id} {lang}: {verdict}: {why}")
