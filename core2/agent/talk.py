"""The reply an answer gives in the chat, in the reader's language, from what was seen when it was looked at.

core2/agent/look.py says what it saw in sentences of its own, with the figures and days each one states. The
workspace's AI then says it as a person would in a chat: what was found, what it means for the question, and the
next question worth a click -- from those sentences alone, never the rows. A checker reads the reply before the
reader does: every figure in it must be one the sentences hold (rounded as people round), it says no cause they do
not give, and it stays a few plain sentences. Otherwise the reply is the sentences as they are.

Whether the AI is asked at all is decided where the answer is (core2/service.py portal_reply): never where the
reply names people, nor where it names members of a workspace that keeps member names from the AI, nor where the
workspace's answers may not reach the AI.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any, Callable

from core2.answer.notes import CAUSE
from core2.answer.summary import _held, _sentences, figures

log = logging.getLogger("querybot.core2")

MAX_WORDS = 80
_PERCENT = re.compile(r"\d[\d.,]*\s?%")
# A reason guessed rather than found: "this could indicate partial data", "cela pourrait indiquer".
_GUESS = re.compile(r"\b(?:could|might|may|seems? to|appears? to)\s+(?:indicate|suggest|mean|point to|be due)|"
                    r"\bsuggests? that\b|\bpourrait\s+(?:indiquer|signifier|s'expliquer)|\bsemble\s+indiquer|"
                    r"\bpeut\s+indiquer", re.IGNORECASE)
SYSTEM = """You are the data assistant in a company's chat, talking with a manager about their own data.

An answer to their question was just worked out, then checked. You are given the question, the answer's own
sentence, what the checks found, and the next questions they can click under your reply. Write your reply in the
chat as a thoughtful analyst would say it to them: one to three short sentences, warm and direct, in {language}.
Say what was found and what it means for their question. When next questions are given, end with one short
question that invites the most useful of them (their buttons are shown under your reply); when none are given, ask
nothing.

Rules:
- Use only the figures and dates in what you are given, written the same way or rounded as people round them.
  Never work out a new figure: no sums, differences, averages or percentages of your own.
- Never say why something happened, and never guess at a reason (missing or partial data, stock, a closure), unless
  what you are given says so.
- Keep every name exactly as written.
- Plain sentences: no headings, lists, markdown or quotation marks. Do not mention SQL, queries, tables or columns.
Reply with your sentences only."""


def names_members(reply: dict[str, Any]) -> bool:
    return any(s.get("members") for s in reply.get("seen") or [])


def material(reply: dict[str, Any], question: str, headline: str, lang: str = "en") -> str:
    """What the AI is given: the question, the answer's sentence, what the checks found and the next questions."""
    key = "fr" if lang == "fr" else "en"
    seen = [f"- {s.get(key) or s.get('en')}" for s in reply.get("seen") or []]
    offers = [f"- {o.get('label_fr' if key == 'fr' else 'label') or o.get('label')}" for o in reply.get("offers") or []]
    parts = [f"Question: {question}", f"Answer: {headline}", "What the checks found:\n" + "\n".join(seen)]
    if offers:
        parts.append("Next questions they can click:\n" + "\n".join(offers))
    return "\n\n".join(parts)


def _known(reply: dict[str, Any], given: str) -> list[float]:
    known = [value for value, _ in figures(given.replace(" ", "").replace(" ", " "))]
    for s in reply.get("seen") or []:
        known += [float(v) for v in s.get("figures") or [] if isinstance(v, (int, float))]
        for day in s.get("days") or []:
            try:
                d = dt.date.fromisoformat(str(day)[:10])
            except ValueError:
                continue
            known += [float(d.day), float(d.year)]
    return known


def checked(said: str, reply: dict[str, Any], given: str) -> str | None:
    """The AI's reply when every figure in it is one the checks found, it gives no cause they do not, and it is a
    few plain sentences; None otherwise."""
    text = " ".join(_sentences(said).split()).strip().strip('"')
    if not text or len(text.split()) > MAX_WORDS or text.startswith(("#", "-", "*", "{", "[")):
        return None
    if re.search(r"\b(?:SQL|quer(?:y|ies)|columns?)\b", text):     # never how it is stored ("Table" may be a name)
        return None
    known = _known(reply, given)
    plain = text.replace("\u202f", "").replace("\u00a0", " ")
    stray = [v for v, step in figures(plain) if not _held(v, step, known)]
    # A share is a claim however small ("about 6% of the month"): it must be one the checks found.
    stray += [v for v, step in figures(" ".join(_PERCENT.findall(plain)))
              if not any(abs(v - k) <= max(step, 0.051) for k in known)]
    if stray:
        log.info("A reply was not shown: it states %s, which the checks did not find", stray[:3])
        return None
    if CAUSE.search(text) or _GUESS.search(text):
        log.info("A reply was not shown: it gives a cause the checks did not find")
        return None
    return text


def write(reply: dict[str, Any], question: str, headline: str, complete: Callable[[str, str], str], *,
          lang: str = "en") -> str | None:
    """The reply in the AI's words, checked; None when the facts' own sentences stand."""
    given = material(reply, question, headline, lang)
    try:
        said = complete(SYSTEM.format(language="French" if lang == "fr" else "English"), given)
    except Exception as exc:  # noqa: BLE001 - the facts' own sentences stand
        log.warning("The workspace's AI could not write a reply: %s", exc)
        return None
    return checked(said, reply, given)
