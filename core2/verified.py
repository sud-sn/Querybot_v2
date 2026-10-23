"""Verified answers: a question an admin checked, answered again from the plan they checked.

An admin marks an answer verified (Questions > Verified answers, or from the question log). When a reader asks that
question again, in the same words however they are written, the service answers it from the verified plan with no
AI reading it afresh, and the answer says who verified it (core2/service.py). As Databricks' trusted assets,
Snowflake's verified query repository and Looker's golden queries do.

The match is deliberately narrow: the same words once case, punctuation, spacing and a "please" are set aside (a
sign that changes the meaning, "> 5" or "-5%", is kept). Two questions that look alike but differ by a word ("2024"
for "2025", "store" for "region") are different questions, read afresh. The plan still runs under the reader's own
access, and one the data no longer answers is set aside.

Only an answer read from the question's own words alone can be verified (core2/service.py marks it): never a
follow-up or a chip (it carries the answer above), one under a reader's standing condition, one using a figure the
conversation defined, or one whose plan fixed a day the question names no year for ("since January").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Words, and the signs that change what a question asks: "> 5 orders" is not "< 5 orders", "-5%" not "5%".
_WORD = re.compile(r"[^\W_]+(?:[.,'’][^\W_]+)*|[<>=≤≥≠%$€£¥+]|-(?=\s*\d)", re.UNICODE)
_POLITE = {"please", "pls", "plz", "svp"}


@dataclass(frozen=True)
class Verified:
    id: int
    question: str
    plan: dict[str, Any]
    by: str
    at: str


def key(question: str) -> str:
    """The question as its words alone: "What were net sales in 2025?" and "please, what were NET sales in 2025"
    are one question. "" for a question with no words."""
    words = [w.casefold() for w in _WORD.findall(str(question or ""))]
    while words and words[0] in _POLITE:
        words.pop(0)
    while words and words[-1] in _POLITE:
        words.pop()
    return " ".join(words)[:500]


def badge(found: Verified) -> dict[str, str]:
    """The answer's first badge: who checked this answer (an admin with no name on their account, said so)."""
    who = (found.by or "").strip()
    return {"kind": "verified", "text": f"Verified by {who or 'an admin'}",
            "text_fr": f"Vérifié par {who or 'un administrateur'}"}
