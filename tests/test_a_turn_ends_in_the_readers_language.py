"""
tests/test_a_turn_ends_in_the_readers_language.py

Twenty-four messages that END a turn were English literals.

Each one is the last thing a reader sees on a turn that produced no answer: the
monthly query limit, a three-minute timeout, no table access, a semantic conflict,
an analytical plan the layer could not compile. They were built inline in
core/query_pipeline.py and handed straight to send_message, so a French reader's
turn ended in a language they may not read, at the one moment they need to know
what to do next:

    ⏱ Query timed out after 3 minutes. Try adding a filter (e.g. date range or
      specific customer) to narrow the result.
    ❌ Monthly query limit reached (500/500).
    🔒 *No table access assigned.*

Step 1d fixed three reader-facing disclosures and QUARANTINED these twenty-four by
their opening text, so a twenty-fifth would fail the suite while the backlog was
worked. This is that backlog done, and that quarantine list is now empty.

Two payloads are deliberately NOT catalogue lookups and never will be:

    f"❌ {last_reason}"                      a policy explanation a person wrote
    f"{_clarification_prompt}\\n\\n{options}"  both halves already translated

Everything else in the turn-ending path now comes from the catalogue, with the
reader's own language passed explicitly rather than read from the ContextVar --
these sites sit on paths that return early, before the language activation that
covers answer construction.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.i18n import MESSAGES, t

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every message that can end a turn, and the placeholders it must keep.
TERMINAL_KEYS = {
    "terminal.no_database": (),
    "terminal.query_limit_reached": ("used", "limit"),
    "terminal.query_limit_warning": ("used", "limit"),
    "terminal.token_limit_reached": ("used", "limit"),
    "terminal.token_limit_warning": ("used", "limit"),
    "terminal.config_error": ("detail",),
    "terminal.no_tables_in_schema": ("schema",),
    "terminal.no_table_access": (),
    "terminal.no_tables_available": (),
    "terminal.needs_governed_context": ("slot",),
    "terminal.blocked_by_policy": ("reason",),
    "terminal.metric_blocked_by_policy": ("reason",),
    "terminal.cached_operation_unsafe": (),
    "terminal.trend_timeout": (),
    "terminal.query_timeout": (),
    "terminal.kb_not_ready": (),
    "terminal.count_entity_unmapped": ("entity",),
    "terminal.count_identifier_unconfirmed": (),
    "terminal.count_plan_uncompilable": (),
    "terminal.analytical_plan_unresolved": ("missing",),
    "terminal.semantic_conflict": ("conflicts",),
    "terminal.temporal_contract_uncompilable": (),
    "terminal.ai_error": ("detail",),
}


class TestEveryTurnEndingMessageIsTranslated:

    @pytest.mark.parametrize("key", sorted(TERMINAL_KEYS))
    def test_it_exists_in_both_languages(self, key):
        assert key in MESSAGES, key
        assert MESSAGES[key].get("en"), key
        assert MESSAGES[key].get("fr"), key

    @pytest.mark.parametrize("key", sorted(TERMINAL_KEYS))
    def test_the_french_was_written_not_copied(self, key):
        assert MESSAGES[key]["fr"] != MESSAGES[key]["en"], key

    @pytest.mark.parametrize("key", sorted(TERMINAL_KEYS))
    def test_every_placeholder_survives_translation(self, key):
        import re

        expected = set(TERMINAL_KEYS[key])
        for lang in ("en", "fr"):
            found = set(re.findall(r"\{(\w+)\}", MESSAGES[key][lang]))
            assert found == expected, (key, lang, found, expected)

    @pytest.mark.parametrize("key", sorted(TERMINAL_KEYS))
    def test_it_renders_without_leaving_a_placeholder_behind(self, key):
        values = {name: "X" for name in TERMINAL_KEYS[key]}
        for lang in ("en", "fr"):
            rendered = t(key, lang=lang, **values)
            assert "{" not in rendered, (key, lang, rendered)
            assert rendered.strip(), (key, lang)

    def test_a_limit_message_reads_as_a_sentence_in_french(self):
        assert t("terminal.query_limit_reached", lang="fr",
                 used=500, limit=500) == \
            "❌ Limite mensuelle de requêtes atteinte (500/500)."

    def test_a_timeout_still_tells_the_reader_what_to_do(self):
        french = t("terminal.query_timeout", lang="fr")
        assert "expiré" in french
        assert "filtre" in french
        assert "timed out" not in french


class TestTheCallSitesPassTheLanguage:
    """These sits inside a 6,600-line coroutine that needs a warehouse and a
    websocket, so the calls are read as a SYNTAX TREE. They also sit on paths
    that RETURN EARLY -- before the activation that covers answer construction --
    so reading the language from the ContextVar would not be enough, and whether
    a `lang=` argument is present is a structural fact rather than a substring.
    """

    @staticmethod
    def _terminal_calls() -> list[ast.Call]:
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        function = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == "_handle_query_impl")
        found = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "_t":
                continue
            if (node.args and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).startswith("terminal.")):
                found.append(node)
        return found

    def test_every_terminal_key_is_actually_used(self):
        used = {call.args[0].value for call in self._terminal_calls()}
        assert used == set(TERMINAL_KEYS), (
            sorted(set(TERMINAL_KEYS) - used), sorted(used - set(TERMINAL_KEYS)))

    def test_each_one_is_given_the_readers_language(self):
        for call in self._terminal_calls():
            keywords = {keyword.arg for keyword in call.keywords}
            assert "lang" in keywords, ast.dump(call.args[0])

    def test_the_language_comes_from_the_portal_user(self):
        """Not display_context, which on a cached-result hit is a snapshot -- a
        reader who switched to French would keep being answered in English."""
        for call in self._terminal_calls():
            lang = next(kw.value for kw in call.keywords if kw.arg == "lang")
            rendered = ast.dump(lang)
            assert "portal_user" in rendered, call.args[0].value
            assert "display_context" not in rendered, call.args[0].value

    def test_only_two_literal_payloads_are_left_in_the_turn_ending_path(self):
        """Both are pass-throughs of text that is already a reader's: a policy
        explanation a person wrote, and two already-translated variables."""
        source = (ROOT / "core" / "query_pipeline.py").read_text(encoding="utf-8")
        function = next(
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == "_handle_query_impl")
        literals = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "send_message":
                continue
            for arg in node.args[1:]:
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    literals.append(ast.get_source_segment(source, arg) or "")
        assert len(literals) == 2, literals
        assert any("last_reason" in text for text in literals), literals
        assert any("_clarification_prompt" in text for text in literals), literals
