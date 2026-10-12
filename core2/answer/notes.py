"""The analyst's note under an answer: what stands out, why it moved and what to watch, from its facts.

The facts are worked out in code from the answer's rows (facts.py) and arrive with the answer, so the card can
show them at once, each as the sentence the code wrote. The workspace's AI then writes the note an analyst
would: the same facts, grouped and in plain words, in the reader's language. It is given the facts alone --
never the rows -- and a checker reads every sentence it writes before the reader does:

* a figure the facts do not hold (rounded as people round: "$1.2M" for 1,234,567) and the sentence is
  dropped;
* "because", "due to", "driven by" and the like only where a fact names who moved the figure (a driver);
  anything else is said as what happened, not why;
* nothing but sentences: no headings, lists or markdown.

Where the workspace keeps member names from the AI (a regulated tenant, value indexing off) and the facts name
members, or the AI is not there, the note is the facts' own sentences, grouped the same way. A person's name
never reaches the AI in any workspace: a reading of rows that name people has no AI note.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from core2.answer.facts import STANDS, WATCH, WHY
from core2.answer.summary import _held, figures

log = logging.getLogger("querybot.core2")

SECTIONS = (STANDS, WHY, WATCH)
LIMITS = {STANDS: 4, WHY: 2, WATCH: 2}
CAUSE = re.compile(r"\b(?:because|due to|driven by|owing to|as a result of|thanks to|caused by|explained by|"
                   r"parce que|en raison de|grâce à|du fait de|à cause de|porté par|portée par)\b", re.IGNORECASE)
SYSTEM = """You are a business analyst writing the note under a chart for a manager.

You are given facts already worked out from the data, each with its section: stands_out (what is typical, what
is not, what changed), why (who moved the figure) and watch (what to keep an eye on). Write the note: for each
section, one to three short sentences that say what those facts mean together, as an analyst would say it to
the manager. Lead with what matters most.

Rules:
- Use only the figures in the facts, written the same way. Never work out a new figure: no sums, differences,
  averages or percentages of your own.
- Say why something moved only when a fact in the why section names who moved it. Otherwise say what happened.
- Keep every name exactly as written in the facts.
- Plain sentences: no headings, lists, markdown or quotation marks. Do not mention SQL, queries or columns.
- Write in {language}.
Answer with one JSON object: {{"stands_out": [...], "why": [...], "watch": [...]}}, each a list of sentences
(empty when that section has no facts)."""


def template(reading: dict[str, Any], lang: str = "en") -> dict[str, Any]:
    """The note as the facts' own sentences, the largest first in each section."""
    key = "fr" if lang == "fr" else "en"
    facts = sorted(reading.get("facts") or [], key=lambda f: -float(f.get("size") or 0))
    note: dict[str, Any] = {s: [] for s in SECTIONS}
    for f in facts:
        section = f.get("section") if f.get("section") in SECTIONS else STANDS
        if len(note[section]) < LIMITS[section] and f.get(key):
            note[section].append(f[key])
    how = "how_fr" if key == "fr" else "how"
    note["how"] = [{"fact": f.get(key) or f.get("en"), "how": f.get(how) or f.get("how") or "",
                    "query": f.get("query") or ""} for f in facts if f.get("how")]
    note["lead"] = (reading.get("lead") or {}).get(key) or ""
    note["by"] = "facts"
    return note


def names_members(reading: dict[str, Any]) -> bool:
    return any(f.get("members") for f in reading.get("facts") or [])


def _given(reading: dict[str, Any], lang: str) -> str:
    key = "fr" if lang == "fr" else "en"
    lines = [f"- [{f.get('section')}] {f.get(key) or f.get('en')}"
             for f in sorted(reading.get("facts") or [], key=lambda f: -float(f.get("size") or 0))]
    return "Facts:\n" + "\n".join(lines)


def _known(reading: dict[str, Any]) -> list[float]:
    known: list[float] = []
    for f in reading.get("facts") or []:
        for text in (f.get("en"), f.get("fr")):
            known += [value for value, _ in figures(str(text or "").replace(" ", "").replace(" ", " "))]
        known += [float(v) for v in f.get("figures") or [] if isinstance(v, (int, float))]
    return known


def checked(said: Any, reading: dict[str, Any]) -> dict[str, list[str]] | None:
    """The AI's note with every sentence that states a figure the facts do not hold, or a cause no driver fact
    gives, left out; None when nothing is left."""
    if not isinstance(said, dict):
        return None
    known = _known(reading)
    drivers = any(f.get("section") == WHY for f in reading.get("facts") or [])
    out: dict[str, list[str]] = {}
    for section in SECTIONS:
        kept = []
        for sentence in said.get(section) or []:
            if not isinstance(sentence, str):
                continue
            text = " ".join(sentence.split()).strip().strip('"')
            if not text or text.startswith(("#", "-", "*")) or len(text.split()) > 60:
                continue
            plain = text.replace(" ", "").replace(" ", " ")
            stray = [v for v, step in figures(plain) if not _held(v, step, known)]
            if stray:
                log.info("A note sentence was dropped: it states %s, which the facts do not hold", stray[:3])
                continue
            if CAUSE.search(text) and not drivers:
                log.info("A note sentence was dropped: it gives a cause no fact names")
                continue
            kept.append(text)
        out[section] = kept[:LIMITS[section]]
    return out if any(out.values()) else None


def _parse(reply: str) -> Any:
    text = str(reply or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                return None
    return None


def write(reading: dict[str, Any], complete: Callable[[str, str], str] | None, *, lang: str = "en",
          values_allowed: bool = True, personal: bool = False) -> dict[str, Any]:
    """The note for a reading: the AI's, checked, where it may see the facts; else the facts' own sentences."""
    note = template(reading, lang)
    if complete is None or personal or (names_members(reading) and not values_allowed):
        return note
    system = SYSTEM.format(language="French" if lang == "fr" else "English")
    try:
        said = _parse(complete(system, _given(reading, lang)))
    except Exception as exc:  # noqa: BLE001 - the facts' own sentences stand
        log.warning("The workspace's AI could not write a note: %s", exc)
        return note
    kept = checked(said, reading)
    if kept is None:
        return note
    # A section the AI left empty keeps the facts' own sentences: nothing worked out is lost.
    for section in SECTIONS:
        if kept.get(section):
            note[section] = kept[section]
    note["by"] = "ai"
    return note
