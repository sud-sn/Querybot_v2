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

    # A "/" opens a regex when the previous significant character cannot end
    # an expression -- otherwise it is division. The standard heuristic, and
    # sufficient for this page.
    _BEFORE_REGEX = set("(,=:[!&|?{};+-*%~^") | {"\n"}

    @classmethod
    def _starts_a_regex(cls, js: str, i: int) -> bool:
        if js[i + 1 : i + 2] in ("/", "*", ""):
            return False
        j = i - 1
        while j >= 0 and js[j] in " \t":
            j -= 1
        if j < 0:
            return True
        if js[j] in cls._BEFORE_REGEX:
            return True
        # `return /re/` and `typeof /re/`
        return bool(re.search(r"\b(return|typeof|case|in|of|new|delete)$",
                              js[max(0, j - 9):j + 1]))

    @staticmethod
    def _skip_regex(js: str, i: int) -> int:
        """Index just past the closing "/" (and any flags) of a regex."""
        i, n, in_class = i + 1, len(js), False
        while i < n:
            c = js[i]
            if c == "\\":
                i += 2
                continue
            if c == "\n":
                return i          # not a regex after all; give up safely
            if c == "[":
                in_class = True
            elif c == "]":
                in_class = False
            elif c == "/" and not in_class:
                i += 1
                while i < n and js[i].isalpha():
                    i += 1
                return i
            i += 1
        return i

    @staticmethod
    def _skip_interpolation(js: str, i: int) -> int:
        """Index just past the ``}`` closing a ${...}, i starting inside it."""
        depth, n = 1, len(js)
        while i < n and depth:
            c = js[i]
            if c == "\\":
                i += 2
            elif c == "{":
                depth, i = depth + 1, i + 1
            elif c == "}":
                depth, i = depth - 1, i + 1
            elif c == "/" and TestNothingObviousWasLeftBehind._starts_a_regex(js, i):
                # A regex inside an interpolation, which can hold a quote:
                #     `"${String(v).replace(/"/g, \'""\')}"`
                # is the CSV escaper on this page. Read as a string start, that
                # lone " pairs with one inside the next literal and the lexer
                # desyncs for 95,000 characters -- silently skipping a third of
                # the file, which is the same "reports its blind spot as clean"
                # failure this scanner exists to catch.
                i = TestNothingObviousWasLeftBehind._skip_regex(js, i)
            elif c in "'\"":
                j = i + 1
                while j < n and js[j] != c:
                    j += 2 if js[j] == "\\" else 1
                i = j + 1
            elif c == "`":
                # A template literal nested inside an interpolation, which can
                # hold interpolations of its own. Recursing here is the whole
                # trick: skipping to the "matching" backtick without it loses
                # sync, and every span after that reports CODE as copy.
                i, _ = TestNothingObviousWasLeftBehind._read_template(js, i)
            else:
                i += 1
        return i

    @staticmethod
    def _read_template(js: str, i: int):
        """(index past the closing backtick, static text) for js[i] == '`'."""
        i, out, n = i + 1, [], len(js)
        while i < n:
            c = js[i]
            if c == "\\":
                i += 2
            elif c == "`":
                return i + 1, "".join(out)
            elif c == "$" and js[i + 1 : i + 2] == "{":
                # Interpolations are values, not copy.
                i = TestNothingObviousWasLeftBehind._skip_interpolation(js, i + 2)
                out.append(" ")
            else:
                out.append(c)
                i += 1
        return i, "".join(out)

    @classmethod
    def _string_literals(cls, js: str):
        """Every string literal in `js`, template literals included.

        Hand-lexed. A regex cannot do this: `a${x ? `b` : ""}c` holds three
        literals and a regex pairs the wrong backticks, so everything after the
        first nested one comes back as JavaScript wearing quotes. Both shortcuts
        were tried -- a naive regex reported twenty pieces of code as English,
        and a non-nesting one silently found nothing at all.
        """
        i, n = 0, len(js)
        while i < n:
            c = js[i]
            if c == "/" and cls._starts_a_regex(js, i):
                # A REGEX LITERAL, not a division. This one matters more than
                # it sounds: formatBotText holds
                #     /```(?:[a-zA-Z0-9_-]+)?\n?([\s\S]*?)```/g
                # -- six backticks inside a regex. Read as template literals
                # they desync the lexer for the entire rest of the file, and
                # every span after that comes back as JavaScript wearing
                # quotes. That is what the first three attempts at this
                # scanner were actually reporting.
                i = cls._skip_regex(js, i)
            elif c in "'\"":
                j = i + 1
                while j < n:
                    if js[j] == "\\":
                        j += 2
                    elif js[j] == c or js[j] == "\n":
                        break
                    else:
                        j += 1
                if j < n and js[j] == c:
                    yield "quoted", js[i + 1 : j]
                    i = j + 1
                else:
                    # Unterminated: an apostrophe inside a regex literal, say.
                    # Stepping over one character is right; swallowing to the
                    # next quote would report the code between them as copy.
                    i += 1
            elif c == "`":
                i, text = cls._read_template(js, i)
                yield "template", text
            else:
                i += 1

    def _literals(self):
        """String literals in the page's script that read like user copy."""
        js = CHAT[CHAT.index("\n<script>\n// The message catalogue"):]
        js = re.sub(r"/\*.*?\*/", " ", js, flags=re.S)
        js = re.sub(r"(?m)^\s*//.*$", " ", js)
        out = set()
        # Template literals as well as quoted strings. Reading only quoted
        # strings, this file passed 11/11 while fourteen pieces of English copy
        # sat in backticks a few hundred lines apart -- the memory badge's
        # "N turns of context", "Downloaded N rows as CSV.", "Open KPI in
        # workspace", "Based on:", "Queries left ·", "Governed agent", the
        # dashboard link, two artifact tabs beside a translated third, and both
        # forecast captions. A scanner that cannot read the syntax the page
        # actually uses reports its own blind spot as a clean bill of health.
        for kind, raw in self._string_literals(js):
            # HTML around the copy is markup, not copy -- and discarding a
            # literal for CONTAINING markup, which the filter below used to do,
            # is what hid the two artifact tabs.
            value = re.sub(r"<[^<>]*>", " ", raw)
            value = re.sub(r"\s+", " ", value).strip()
            # A template literal is written for interpolation, so a short one
            # with no space is still copy -- ">Data</button>" is four
            # characters and was one of the two tabs sitting beside a
            # translated third.
            if kind == "quoted" and (len(value) < 8 or " " not in value):
                continue
            if kind == "template" and len(value) < 4:
                continue
            if value.startswith((".", "#", "ui.", "answer.", "chip.", "stage.")):
                continue
            if re.search(r"[{}<>]|var\(|px|:\s*\d|=|\bfunction\b|/", value):
                continue
            # The next two filters are about CSS, and CSS is never written as a
            # template literal -- a class list needs no interpolation. Applied
            # to one, "N turns of context" is all lowercase words and reads as
            # a class list, which is how the memory badge survived a scanner
            # written to find exactly it.
            if kind == "quoted":
                # A class list is lowercase words and hyphens and nothing else.
                if re.fullmatch(r"[a-z0-9-]+(?: [a-z0-9-]+)*", value):
                    continue
                # A CSS selector fragment ("thead th", "input[type=x]").
                if re.fullmatch(r"[a-z]+(?: [a-z\[\]=\"\'-]+)+", value):
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
            # A CSS selector and two inline handlers, none of them copy.
            "button, textarea",
            "openHistoryThread(this.dataset.threadId, this)",
            # ── Everything below is what the template-literal pass newly ──
            # ── reaches. ──
            # Each one is named rather than filtered by a pattern, because a
            # pattern broad enough to hide these would hide the next real
            # leak too -- which is the whole reason this file exists.
            #
            # A regex replacement template and two code-fence placeholders,
            # with their ${idx} stripped.
            "$1 $2 $3", "@@CODEBLOCK @@", "@@INLINECODE @@",
            # The product's own name beside the timestamp separator.
            "QueryBot •",
            # Internal keys, DOM id prefixes and class names, with their
            # ${...} stripped. None is copy; each would break a lookup or a
            # selector if it were translated.
            ": : :", "__abs_", "action- -", "-typing", "left", "rows",
            "· · ·", "rc-bubble", "result-kpi rc-kpi",
            "artifact-chart-", "assistant-chart-", "chat-chart-",
            "clarification-", "querybot.activeSchema.",
            "querybot.chatRailCollapsed:",
            # The product's name beside a separator, and two message shells
            # whose copy is now a catalogue id.
            "QueryBot • Ask", "QueryBot • ⛶",
            # localStorage key templates, with their ${account}/${thread}
            # interpolations stripped. Not copy: renaming one would orphan
            # every reader's cached thread.
            "qb_chat_cache: : :", "querybot.activeThread: :",
            "querybot.draft: : :", "querybot.history: :",
            "querybot.history: : :",
            # Inline CSS for the three-segment confidence meter.
            "background:linear-gradient(90deg, 0%, 33%, 33%, 66%, 66%)",
            # Two already-translated values joined by a separator; the static
            # part is the separator.
            "visual · ·",
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
