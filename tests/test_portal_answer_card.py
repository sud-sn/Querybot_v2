"""
tests/test_portal_answer_card.py

The portal answer card is what a user actually reads, and it had the answer
backwards. Measured in a browser against a realistic payload, before this
change:

  .answer-chip     ("4.82M SEK · +12.4% vs Q1")   11px   <- the answer
  .result-kicker   ("Result")                     11px
  .answer-headline (a sentence restating it)      18px

The value the question asked for rendered at the same size as the word
"Result" and 61% the size of the sentence beside it. Above the result sat
five separate bands — insight summary, decision signal, coverage caveats and
two flavours of anomaly callout — each with its own box and its own 12-13px
text, in whatever order the code built them; together they occupied more
vertical height than the answer did. Confidence was a sixth band, permanently
expanded, duplicating what the provenance disclosure already held.

After: 34px value leading, one ranked notes region, confidence as a pill whose
reasoning lives inside the disclosure. Card height 814px -> 531px on the same
payload, with the answer three times larger.

SCOPE OF THESE TESTS. The card is built in JavaScript and the suite has no JS
runtime, so behaviour was verified in a real browser driving the real
appendAssistantResponse() — value promoted, notes ranked caveat-then-watch-
then-note, pill toggling the disclosure, warnings ordered before reasons, the
no-value fallback promoting the headline, no empty containers, and contrast
>= 4.75 in both themes. What is checked here is the part that can regress
silently in source: the type hierarchy the redesign exists to establish, and
the bands it collapsed staying collapsed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "portal" / "templates" / "portal_chat.html"


@pytest.fixture(scope="module")
def chat_source() -> str:
    return CHAT.read_text(encoding="utf-8")


def _font_size(source: str, selector: str) -> float:
    """Read font-size off a single-line CSS rule in the template's style block."""
    rule = re.search(re.escape(selector) + r"\{([^}]*)\}", source)
    assert rule, f"no CSS rule for {selector}"
    size = re.search(r"font-size:\s*([\d.]+)px", rule.group(1))
    assert size, f"{selector} declares no font-size"
    return float(size.group(1))


def test_the_answer_is_the_largest_thing_on_the_card(chat_source):
    """The whole point. A restatement of the question must not outrank the
    number the question asked for."""
    value = _font_size(chat_source, ".answer-value")
    headline = _font_size(chat_source, ".answer-headline")

    assert value > headline * 1.8, (
        f"the answer value is {value}px against a {headline}px headline; the "
        f"value must clearly lead, not merely edge ahead"
    )
    assert value >= 28, f"{value}px is not a headline number"


def test_the_headline_only_takes_the_lead_when_there_is_no_value(chat_source):
    """A list or a chart has no single number, and the card must not open on a
    whisper in that case."""
    lead = _font_size(chat_source, ".answer-headline.is-lead")
    headline = _font_size(chat_source, ".answer-headline")
    assert lead > headline, "the no-value fallback does not promote the headline"

    assert "answer-headline is-lead" in chat_source, (
        "nothing emits the lead variant, so the fallback renders at supporting size"
    )
    assert re.search(r"hasValue\s*=\s*!!answer\.short_value", chat_source), (
        "the lead/supporting choice is no longer driven by whether a value exists"
    )


@pytest.mark.parametrize("dead", [
    "result-kicker",     # labelled the obvious, at the size of the value
    "answer-chip",       # the 11px answer this replaced
    "insight-summary",   # four bands folded into .answer-notes
    "insight-callout",
    "decision-signal",
    "confidence-card",   # now a pill plus detail inside the disclosure
    "confidence-badge",
    "confidence-lines",
])
def test_the_collapsed_bands_do_not_come_back(chat_source, dead):
    assert f'class="{dead}' not in chat_source, f"{dead} is being emitted again"
    assert f'class="[^"]*{dead}' not in chat_source
    assert not re.search(rf"^\.{re.escape(dead)}\{{", chat_source, re.M), (
        f"dead CSS rule for {dead}"
    )


def test_what_to_know_is_one_ranked_region(chat_source):
    """Five bands with five different treatments gave the reader no order to
    read them in. One region, three tiers, ranked by consequence."""
    rank = re.search(r"NOTE_RANK\s*=\s*\{([^}]*)\}", chat_source)
    assert rank, "the note ranking is gone"
    order = dict(re.findall(r"(\w+):\s*(\d+)", rank.group(1)))
    assert order == {"caveat": "0", "watch": "1", "note": "2"}, (
        f"tier order changed: {order} — caveats must sort first, since they "
        f"change what the number means"
    )

    for source_field in ("coverage_caveats", "decision_signal",
                         "anomaly_callouts", "insight_summary"):
        assert source_field in chat_source, f"{source_field} no longer reaches the card"

    for tier in ("caveat", "watch", "note"):
        assert re.search(rf"^\.answer-note\.{tier}\{{", chat_source, re.M), (
            f"tier {tier} has no styling, so it is indistinguishable"
        )


def test_confidence_states_a_verdict_and_hides_its_reasoning(chat_source):
    """It used to be 74px of permanently-expanded bullets above the result."""
    assert "trust-pill" in chat_source, "the confidence verdict is gone"
    assert "confidenceDetailHtml" in chat_source, "confidence reasoning is gone"

    # The detail must be rendered inside the disclosure, not in the card body.
    details = re.search(r"<details class=\"trust-box\"[^>]*>.*?confidenceDetailHtml",
                        chat_source, re.S)
    assert details, "confidence reasoning is not inside the provenance disclosure"

    # Warnings before reasons: someone opening this wants to know what is wrong.
    warn_at = chat_source.index("confidence.warnings")
    reason_at = chat_source.index("confidence.reasons")
    assert warn_at < reason_at, (
        "reasons are listed before warnings; the reader is looking for the risk"
    )


def test_no_container_renders_with_nothing_in_it(chat_source):
    """An answer with no artifact and no question text used to draw an empty
    33px bubble, and a card with neither confidence nor sources drew an empty
    trust row. Both were invisible in code and obvious on screen."""
    assert re.search(r"else if \(msg\.question\)", chat_source), (
        "the result bubble no longer checks for text before rendering"
    )
    assert re.search(r"\(confidencePillHtml \|\| citationsHtml\)", chat_source), (
        "the trust row renders unconditionally again"
    )


def test_the_template_still_compiles(chat_source):
    """Cheap execution check: everything above reads source, so a Jinja syntax
    error introduced while editing would otherwise reach the browser first."""
    from jinja2 import Environment

    env = Environment()
    body = re.sub(r"\{%\s*extends.*?%\}", "", chat_source, flags=re.S)
    try:
        env.parse(body)
    except Exception as exc:  # pragma: no cover - the assertion is the report
        pytest.fail(f"portal_chat.html no longer parses as Jinja: {exc}")
