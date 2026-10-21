"""The opening paragraph of an answer about the data itself, by the workspace's AI, from the cards below it.

describe.py builds that answer from the model: a lead, a card per table, things to know, questions to try. The
paragraph says in two to four sentences what the data lets a reader see of the business, as an analyst opens a
walk-through: what it records, how the tables fit together, and a good place to start. It is written from the cards
as the reader sees them and from nothing else, and every figure in it must be one the cards hold: a paragraph with
any other number is not shown, and the cards stand alone. It follows the answer, so the answer never waits for it.

The cards hold the data's spans and the values its tables leave out by default, so where member values are kept
from the AI (a tenant under compliance, or value indexing off) there is no paragraph.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from core2.answer.summary import checked

log = logging.getLogger("querybot.core2")

LANGUAGES = {"en": "plain business English", "fr": "plain business French"}
SYSTEM = """You write the opening paragraph of a description of a company's business data, for a business reader.

You are given the description as the reader sees it: its lead sentence, a card per table (what one row is, its
figures, the fields that describe it, its dates, what it links to, what to know before counting it) and notes.
Write two to four short sentences that say what this data lets the reader see of the business, as an analyst
would open a walk-through: what it records, how the tables fit together, and a good first question to ask. Say it
to the reader directly, in {language}.

Rules:
- Use only what you are given. Never name a table, figure or field that is not in it, and never guess what the
  business sells, who its customers are or where it is.
- Use only figures that appear in what you are given, written the same way. No sums, differences or counts of
  your own.
- Do not repeat the lead sentence; build on it.
- No headings, lists, markdown or quotation marks. Do not mention SQL, queries, databases, columns or schemas.
Reply with the sentences only."""


def eligible(payload: dict[str, Any], *, values_allowed: bool) -> bool:
    """Does this answer get an opening paragraph, given what may reach the workspace's AI?"""
    return (payload.get("kind") == "describe_data" and payload.get("engine") == "core2" and values_allowed
            and bool(payload.get("sections")))


def material(payload: dict[str, Any], question: str) -> str:
    """What the AI is given: the description as shown, and nothing else."""
    answer = payload.get("answer") or {}
    # The heading tells the AI egress log that the data's spans and values were sent (core/llm_audit.py).
    parts = ["THE DATA AS DESCRIBED", f"Question: {question}", f"Lead: {answer.get('headline') or ''}"]
    for section in payload.get("sections") or []:
        title = str(section.get("title") or "")
        label = str(section.get("label") or "")
        lines = [f"{title} ({label})" if label else title]
        if section.get("body"):
            lines.append(str(section["body"]))
        lines += [f"- {b}" for b in section.get("bullets") or [] if b]
        parts.append("\n".join(lines))
    if answer.get("scope_note"):
        parts.append(f"Note: {answer['scope_note']}")
    return "\n\n".join(parts)


def write(payload: dict[str, Any], question: str, complete: Callable[[str, str], str], *,
          lang: str = "en") -> str | None:
    """The opening paragraph of ``payload`` by the workspace's AI, checked; None when there is none to show."""
    given = material(payload, question)
    try:
        said = complete(SYSTEM.format(language=LANGUAGES.get(lang, LANGUAGES["en"])), given)
    except Exception as exc:  # noqa: BLE001 - the cards stand without their opening
        log.warning("The workspace's AI could not write the opening of a data description: %s", exc)
        return None
    return checked(said, payload, given)
