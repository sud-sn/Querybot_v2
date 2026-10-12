"""A follow-up that splits or ranks the answer on screen, naming no figure of its own, is of that figure.

The profitability stress test, after "How does the first half of 2026 compare with the first half of 2025?":
"Break the first half of 2026 down by profit centre" was planned afresh as a list of profit centres, with no
figure; "Which division grew the most compared with the first half of 2025?" came back with the cost, or a question
back. "Profit centre" also read as the measure called Profit (the British spelling and the longer name were not
matched first), and "how many profit centres" as no count. A list ("Which devices are firewalls?") or a figure of
the question's own ("the average visit duration by team") is never such a split.

Invented data only.
"""

from __future__ import annotations

import pytest

from core2.bootstrap.names import spelling
from core2.plan import followup
from core2.plan.ir import Plan
from evals.core2 import domains
from evals.core2.compile_eval import learn

H1_26 = {"kind": "between", "start": "2026-01-01", "end": "2026-06-30"}
H1_25 = {"kind": "between", "start": "2025-01-01", "end": "2025-06-30"}
SPLIT = "it splits or ranks the answer on screen and names no figure of its own"


@pytest.fixture(scope="module")
def retail():
    return learn(domains.build("retail"), "descriptive")[1]


def _previous() -> Plan:
    return Plan.model_validate({"kind": "query", "intent": "compare", "measures": ["net_amount"],
                                "time": {"window": H1_26, "compare": {"kind": "window", "window": H1_25}}})


@pytest.mark.parametrize("question", [
    "Break the first half of 2026 down by region",
    "Which region grew the most compared with the first half of 2025?",
    "And by store category?",
])
def test_a_split_or_a_ranking_with_no_figure_of_its_own_is_of_the_figure_on_screen(retail, question):
    reading = followup.read_turn(question, previous=_previous(), model=retail, members=[])
    assert reading.kind == "refine", reading


@pytest.mark.parametrize("question", [
    "Which stores are in the north?",
    "What is the average number of days to ship by store?",
    "Net amount by region for 2025",
])
def test_a_list_or_a_figure_of_its_own_is_not_that_split(retail, question):
    reading = followup.read_turn(question, previous=_previous(), model=retail, members=[])
    assert reading.why != SPLIT, reading


@pytest.mark.parametrize("british, american", [("centre", "center"), ("colour", "color"), ("metres", "meters"),
                                               ("organisation", "organization"), ("cancelled", "canceled"),
                                               ("entre", "entre"), ("store", "store")])
def test_british_spellings_are_read_as_the_names_are_written(british, american):
    assert spelling(british) == american
