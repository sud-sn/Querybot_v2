"""A French quarter or half ("T1 2025", "S2") is a period, as Q1 and H2 are.

French readers write a quarter as a trimestre, T1 to T4, and a half as a semestre, S1 or S2. "Et par rapport
au T1 2025 ?" after a question about the first quarter of 2026 was read as keeping the period of the answer
above (no period named), so it compared the wrong quarters; and a code T1 or S2 could be taken for a member
of some field. The planner is told what the French words mean, a follow-up that writes one names its own
period, and T1 or S2 alone is never matched as a member unless the question names the field just before it.
"""

from __future__ import annotations

import pytest

from core2.plan.followup import keeps_period
from core2.plan.planner import RULES
from core2.plan.values import PERIOD_WORD


@pytest.mark.parametrize("word", ["T1", "t4", "S1", "s2", "Q3", "H1", "FY26"])
def test_a_quarter_or_half_written_either_way_is_a_period_word(word):
    assert PERIOD_WORD.match(word.casefold())


@pytest.mark.parametrize("question", ["Et par rapport au T1 2025, comment évoluent les ventes ?",
                                      "et pour le T3 2026 ?", "compare with Q1 2025"])
def test_a_follow_up_writing_a_quarter_names_its_own_period(question):
    assert not keeps_period(question)


def test_a_follow_up_naming_none_keeps_the_period_above():
    assert keeps_period("et par magasin ?")


def test_the_planner_is_told_what_the_french_words_mean():
    assert "T1 2025" in RULES and "trimestre" in RULES and "exercice" in RULES
