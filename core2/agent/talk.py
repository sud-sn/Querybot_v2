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
# "This suggests demand was limited", "the figures may not reflect all sales", "cela pourrait signifier": what a
# reader would take as a finding the checks never made.
_GUESS = re.compile(r"\b(?:could|might|may|seems? to|appears? to)\s+(?:not\b|indicate|suggest|mean|point to|be due|"
                    r"reflect)|\bsuggests?\b|\bimplies\b|\blikely\b|\bprobably\b|\bpourrai(?:en)?t\b|"
                    r"\bsemblen?t?\b|\bpeut\s+(?:indiquer|signifier)|\bsugg[èe]re\b|\bprobablement\b",
                    re.IGNORECASE)
_ASKED = re.compile(r"[^.!?]*\?\s*$")
# A day of the month said as a number ("the 1st to the 10th", "le 1er"): small, yet a claim -- one of the days seen.
_ORDINAL = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th|er|e)\b", re.IGNORECASE)
# "The data only covers April, September and December", "les informations pour les autres mois ne sont pas
# disponibles": a figure resting on a few days is what happened on those days, never data missing on the others.
# Only where the data itself starts inside the period may the reply say it does not cover all of it.
_COVERAGE = re.compile(r"\b(?:data|records?|information|données|informations?)\b[^.;]*?\b(?:missing|incomplete|"
                       r"partial|not (?:available|complete|cover)|only covers?|does(?:n't| not) cover|unavailable|"
                       r"manquant|incompl[eè]t|partiel|ne couvr|pas disponibles?|indisponibles?|ne sont pas)|"
                       r"\b(?:missing|incomplete|partial)\s+(?:data|records?|information)\b|"
                       r"\bdonnées\s+(?:manquantes|partielles|incomplètes)\b", re.IGNORECASE)
# French figures: "5 797 $" groups thousands with a space, "12,3 k $" writes the decimal with a comma.
_THOUSANDS = re.compile(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))")
_DECIMAL = re.compile(r"(?<=\d),(?=\d)")
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
  what you are given says so. A figure resting on a few days means it happened on those days only: the data is
  complete, so never say the data or information is missing, partial or does not cover the other days.
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


def _plain_figures(text: str, lang: str) -> str:
    text = text.replace("\u00a0", " ")
    if lang == "fr":
        text = _DECIMAL.sub(".", _THOUSANDS.sub("", text))
    return text.replace("\u202f", "")


def _known(reply: dict[str, Any], given: str, lang: str = "en") -> list[float]:
    known = [value for value, _ in figures(_plain_figures(given, lang))]
    for s in reply.get("seen") or []:
        known += [float(v) for v in s.get("figures") or [] if isinstance(v, (int, float))]
        for day in s.get("days") or []:
            try:
                d = dt.date.fromisoformat(str(day)[:10])
            except ValueError:
                continue
            known += [float(d.day), float(d.year)]
    return known + [abs(k) for k in known]     # a fall of $5,054 is the change of -$5,054, said as a fall


def checked(said: str, reply: dict[str, Any], given: str, lang: str = "en") -> str | None:
    """The AI's reply when every figure in it is one the checks found, it gives no cause they do not, and it is a
    few plain sentences; None otherwise. With no next question offered, a question it ends on is left out: there is
    no button to answer it with."""
    text = " ".join(_sentences(said).split()).strip().strip('"')
    if not reply.get("offers"):
        while _ASKED.search(text) and _ASKED.sub("", text).strip():
            text = _ASKED.sub("", text).strip()
        if text.endswith("?"):
            return None
    if not text or len(text.split()) > MAX_WORDS or text.startswith(("#", "-", "*", "{", "[")):
        return None
    if re.search(r"\b(?:SQL|quer(?:y|ies)|columns?)\b", text):     # never how it is stored ("Table" may be a name)
        return None
    known = _known(reply, given, lang)
    plain = _plain_figures(text, lang)
    stray = [v for v, step in figures(plain) if not _held(v, step, known)]
    # A share is a claim however small ("about 6% of the month"): it must be one the checks found.
    stray += [v for v, step in figures(" ".join(_PERCENT.findall(plain)))
              if not any(abs(v - k) <= max(step, 0.051) for k in known)]
    days = {int(d[8:10]) for x in reply.get("seen") or [] for d in x.get("days") or [] if len(str(d)) >= 10}
    stray += [float(n) for n in _ORDINAL.findall(plain) if int(n) not in days]
    if stray:
        log.info("A reply was not shown: it states %s, which the checks did not find", stray[:3])
        return None
    starts_late = any(x.get("kind") == "starts_late" for x in reply.get("seen") or [])
    if _COVERAGE.search(text) and not starts_late:
        log.info("A reply was not shown: it says data is missing where the checks found it is not")
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
    return checked(said, reply, given, lang)
