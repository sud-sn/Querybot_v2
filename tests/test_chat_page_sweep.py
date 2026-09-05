"""
tests/test_chat_page_sweep.py

Two guards over the whole chat page, and the diagnostic card's split between
wire format and copy.

The sweeps exist because portal_chat.html is 5,650 lines and the failure mode
is one string left behind: a French page with a single English toast in it is
worse than an English page, because the reader now cannot tell which parts
they are meant to understand.
"""

from __future__ import annotations

import re

import pytest

from core import i18n
from tests.chat_js import source
from tests.chat_render import catalogue, render

CHAT = source()

# The markers core/answer_formatter.py writes into the message, which the page
# scans for. Wire format, not copy -- see the note beside the diagnostic ids in
# core/i18n.py.
WIRE_MARKERS = (
    "Most likely reason:", "Suggested next step:", "Technical details:",
    "SQL tried:", "Confidence:", "Why:",
)


class TestEveryIdThePageUsesResolves:

    def _used(self):
        # The lookbehind matters: without it `getElementById('artifact')` and
        # every other name ending in "t" reads as a call to t().
        pattern = r"(?<![A-Za-z0-9_$.])%s\(\s*'([a-z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)'"
        return set(re.findall(pattern % "t", CHAT)) | \
            set(re.findall(pattern % "plural", CHAT))

    def test_the_sweep_finds_something_to_check(self):
        """Guards itself: a change to how the page calls t() would otherwise
        make every assertion below pass over an empty set."""
        assert len(self._used()) > 150

    def test_every_id_exists_in_both_languages(self):
        """A missing id renders as "ui.chat.toast.x" on screen, which is the
        least explicable thing a reader can be shown."""
        missing = []
        for msg_id in sorted(self._used()):
            stems = ([msg_id] if msg_id in i18n.MESSAGES
                     else [f"{msg_id}.one", f"{msg_id}.other"])
            for stem in stems:
                for lang in ("en", "fr"):
                    if i18n.t(stem, lang=lang) in (stem, ""):
                        missing.append((stem, lang))
        assert not missing, missing

    def test_every_id_ships_to_the_browser(self):
        """catalogue_for is what the page receives. An id in MESSAGES that the
        catalogue filters out would resolve in Python and not in the page."""
        shipped = catalogue(render(lang="fr"))
        for msg_id in sorted(self._used()):
            keys = ([msg_id] if msg_id in i18n.MESSAGES
                    else [f"{msg_id}.one", f"{msg_id}.other"])
            for key in keys:
                assert key in shipped, key


class TestNothingObviousWasLeftBehind:

    def _literals(self):
        """String literals in the page's script that read like user copy."""
        js = CHAT[CHAT.index("\n<script>\n// The message catalogue"):]
        js = re.sub(r"/\*.*?\*/", " ", js, flags=re.S)
        js = re.sub(r"(?m)^\s*//.*$", " ", js)
        out = set()
        for match in re.finditer(r"""(['"])((?:\\.|(?!\1)[^\\\n])*)\1""", js):
            value = match.group(2).strip()
            if len(value) < 8 or " " not in value:
                continue
            if value.startswith((".", "#", "ui.", "answer.", "chip.", "stage.")):
                continue
            if re.search(r"[{}<>]|var\(|px|:\s*\d|=|\bfunction\b|/", value):
                continue
            # A class list is lowercase words and hyphens and nothing else.
            if re.fullmatch(r"[a-z0-9-]+(?: [a-z0-9-]+)*", value):
                continue
            # A CSS selector fragment ("thead th", "input[type=x]").
            if re.fullmatch(r"[a-z]+(?: [a-z\[\]=\"'-]+)+", value):
                continue
            # An SVG path: digits, single letters and separators only.
            if re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZz0-9.,\s-]+", value):
                continue
            out.add(value)
        return out

    def test_no_user_copy_is_left_as_a_literal(self):
        """The whole point of this file. Every exception below is named and
        justified rather than filtered by a pattern that would also hide the
        next real one."""
        allowed = set(WIRE_MARKERS) | {
            # Substrings of a SERVER message, matched to classify an error.
            # Translating them would break the match, not the display.
            "not currently available",
            # console.warn / console.error prefixes: developer output, never
            # shown to a reader.
            "History load failed:", "Thread restore failed:",
            "Chart render failed",
            # A CSS selector and an inline handler, neither of them copy.
            "button, textarea",
            "openHistoryThread(this.dataset.threadId, this)",
        }
        leftover = sorted(self._literals() - allowed)
        assert not leftover, leftover


class TestTheDiagnosticCardSplitsWireFromCopy:

    def test_the_parse_markers_stay_english(self):
        """core/answer_formatter.py writes them and the same text degrades to
        plain Teams and Zoom messages. A card that silently stops parsing
        renders as raw text with no signal that anything went wrong."""
        for marker in WIRE_MARKERS:
            assert marker in CHAT, marker

    def test_the_markers_match_what_the_formatter_writes(self):
        """The two files have to agree exactly, and neither imports the other.
        Executed against the real formatter output."""
        from core.answer_formatter import format_failure_business_response

        text = format_failure_business_response(
            rca={"headline": "I could not answer this question.",
                 "technical_notes": ["a note"]},
            sql="SELECT 1", sql_preview_fn=lambda s: s,
        )
        for marker in ("Most likely reason:", "Suggested next step:",
                       "Technical details:", "SQL tried:"):
            assert marker in text, marker
            assert marker in CHAT, marker

    def test_the_displayed_labels_are_translated(self):
        """Same words, different job: these are drawn on the card."""
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.diag.reason"] == "Raison la plus probable"
        assert fr["ui.chat.diag.next_step"] == "Prochaine étape suggérée"
        assert fr["ui.chat.diag.sql_tried"] == "SQL tenté"

    def test_the_labels_are_not_the_markers(self):
        """If a label ever became the marker, translating it would silently
        stop the card from parsing."""
        for msg_id in ("ui.chat.diag.reason", "ui.chat.diag.next_step",
                       "ui.chat.diag.technical", "ui.chat.diag.sql_tried"):
            assert not i18n.t(msg_id, lang="en").endswith(":"), msg_id


class TestTheSchemaLockNoLongerKeepsItsOwnCopy:

    def test_the_notice_and_the_hint_use_the_markup_s_ids(self):
        """updateSchemaModeCopy mirrored four strings the markup already had,
        and they had drifted -- three dots against an ellipsis in the
        placeholder, a comma against a middot in the hint."""
        assert CHAT.count("t('ui.chat.multi_schema_body')") == 2
        assert CHAT.count("t('ui.chat.hint_all_schemas')") == 2
        assert CHAT.count("t('ui.chat.composer_placeholder')") == 2

    def test_the_locked_variants_are_french(self):
        fr = catalogue(render(lang="fr"))
        assert fr["ui.chat.schema_locked_title"] == "Schéma {schema} verrouillé."
        assert "{schema}" in fr["ui.chat.schema_locked_placeholder"]

    def test_the_schema_name_is_the_customer_s(self):
        assert i18n.placeholders("ui.chat.schema_locked_title") == {"schema"}
        assert i18n.t("ui.chat.schema_locked_title", lang="fr", schema="FIN") == \
            "Schéma FIN verrouillé."
