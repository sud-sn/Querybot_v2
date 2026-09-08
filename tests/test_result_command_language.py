# -*- coding: utf-8 -*-
"""tests/test_result_command_language.py

F5 · Every message the result chat produces was an English literal.

core/result_commands.py is the whole conversational layer over a cached result
-- exclude these rows, sort it, summarise it, format that column, undo. Roughly
sixty user-visible strings, none of them in the catalogue, all delivered in the
same websocket JSON object as fields that ARE translated. A French reader
excluding a row got:

    Created a filtered result with 3 rows excluded.

beside a card whose every other line was French.

── Two invariants this file pins ────────────────────────────────────────────

THE LABEL IS TRANSLATED, THE QUESTION IS NOT. A clarification option carries a
`label` the reader reads and a `resolved_question` core/dispatcher.py re-plans.
Dispatcher reads resolved_question BEFORE value and label, so translating a
label cannot change what the pipeline is asked. `value` used to be a copy of
the label, which would have put a French string where a question belongs; it
carries the question now.

PLURALS GO THROUGH THE CATALOGUE'S RULE. "1 row excluded" / "2 rows excluded"
was `{'s' if n != 1 else ''}`, which is right for English and wrong for French,
where zero takes the singular. core.i18n.plural owns that rule and the two
forms are separate entries.

Every test executes the real command path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import i18n  # noqa: E402


def under(lang, fn, *args, **kw):
    token = i18n.activate_language(lang)
    try:
        return fn(*args, **kw)
    finally:
        i18n.deactivate_language(token)


ROWS = [{"WHS_NM": "Dallas", "BAL_VAL_AMT": 13_557_410.0},
        {"WHS_NM": "Chennai", "BAL_VAL_AMT": 9_100_000.0}]


class TestTheResolverSpeaksTheReadersLanguage(unittest.TestCase):

    def resolve(self, lang, rows, target):
        from core.result_commands import resolve_result_column

        return under(lang, resolve_result_column, rows, target)

    def test_a_missing_field_is_reported_in_french(self):
        english = self.resolve("en", ROWS, "nonsense")[1]
        french = self.resolve("fr", ROWS, "nonsense")[1]
        self.assertNotEqual(english, french)
        self.assertNotIn("field", french)

    def test_an_empty_result_is_reported_in_french(self):
        self.assertNotEqual(self.resolve("en", [], "x")[1],
                            self.resolve("fr", [], "x")[1])

    def test_the_english_is_unchanged(self):
        self.assertEqual(self.resolve("en", ROWS, "nonsense")[1],
                         "That field is not present in the current result.")

    def test_a_column_that_resolves_carries_no_message(self):
        column, error = self.resolve("fr", ROWS, "WHS_NM")
        self.assertEqual(column, "WHS_NM")
        self.assertEqual(error, "")


class TestTheCountedMessages(unittest.TestCase):
    """Where English's "s" rule and French's disagree."""

    def excluded(self, lang, count):
        from core.result_commands import _plural

        return under(lang, _plural, "reply.rc.excluded", count)

    def test_english_pluralises_from_two(self):
        self.assertIn("1 row excluded", self.excluded("en", 1))
        self.assertIn("2 rows excluded", self.excluded("en", 2))

    def test_english_treats_zero_as_plural(self):
        self.assertIn("0 rows excluded", self.excluded("en", 0))

    def test_french_treats_zero_as_singular(self):
        # The rule the inline {'s' if n != 1 else ''} got wrong: French takes
        # the singular for zero.
        self.assertIn("0 ligne exclue", self.excluded("fr", 0))
        self.assertIn("1 ligne exclue", self.excluded("fr", 1))
        self.assertIn("2 lignes exclues", self.excluded("fr", 2))


class TestEveryMessageIsInTheCatalogue(unittest.TestCase):
    """The sweep, rather than one assertion per message."""

    IDS = [k for k in i18n.MESSAGES if k.startswith("reply.rc.")]

    def test_the_namespace_is_populated(self):
        self.assertGreater(len(self.IDS), 40, "the reply.rc.* namespace is thin")

    def test_every_one_has_a_distinct_french_form(self):
        for msg_id in self.IDS:
            entry = i18n.MESSAGES[msg_id]
            with self.subTest(msg_id=msg_id):
                self.assertTrue(entry.get("fr"))
                self.assertNotEqual(entry["en"], entry["fr"])

    def test_the_module_holds_no_reader_facing_literal_any_more(self):
        """Executed as an AST walk, not a substring search.

        Reads every string constant that is not a docstring and looks like a
        sentence. A literal that survives here is a message a French reader
        would still get in English.
        """
        import ast
        import re

        source = (Path(__file__).resolve().parents[1] / "core"
                  / "result_commands.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                found = ast.get_docstring(node, clean=False)
                if found:
                    docstrings.add(found)

        # Wording the PLANNER parses, not copy: a clarification option's
        # resolved_question is fed back to the pipeline, and the operator words
        # are matched against a user's typing.
        wire = re.compile(
            r"^(?:format |keep the |exclude |sort |show |summari[sz]e |"
            r"January 2026|Fractions|Currency|Number|Percentage|"
            # _format_instruction's output: the same idea as _format_label but
            # in the PLANNER's language, fed back through resolved_question.
            r"values are (?:fractions|already)|percentage, values|"
            r"a number with)")
        # An f-string is judged whole. Its Constant pieces are fragments --
        # " as full month and year" is half of "format {column} as full month
        # and year", which is planner wording, and reading the pieces alone
        # cannot tell that.
        in_fstring = {id(part) for node in ast.walk(tree)
                      if isinstance(node, ast.JoinedStr)
                      for part in node.values}

        def rendered(node):
            return "".join(part.value if isinstance(part, ast.Constant) else "{}"
                           for part in node.values)

        leftover = []
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text = rendered(node).strip()
            elif (isinstance(node, ast.Constant)
                  and isinstance(node.value, str)
                  and id(node) not in in_fstring):
                if node.value in docstrings:
                    continue
                text = node.value.strip()
            else:
                continue
            if len(text) < 20 or not re.search(r"[a-z] [a-z]", text):
                continue
            if wire.match(text) or text.startswith(("SELECT", "http", "{")):
                continue
            leftover.append(text)
        self.assertEqual(sorted(set(leftover)), [], sorted(set(leftover)))


class TestTheFormatLabelTheReaderIsShown(unittest.TestCase):
    """"Formatted BAL_VAL_AMT as a number with 2 decimal places."

    _format_label and _format_instruction say the same thing in two languages:
    one to a person, one to the planner. Only the first is translated, and the
    split is the point -- _format_instruction's output goes into a
    clarification option's resolved_question and is fed back to the pipeline.
    """

    SPECS = (
        {"type": "date", "style": "month_year_long"},
        {"type": "currency", "currency_code": "USD"},
        {"type": "percentage", "scale": "fraction"},
        {"type": "number", "fraction_digits": 2},
    )

    def label(self, lang, spec):
        from core.result_commands import _format_label

        return under(lang, _format_label, spec)

    def test_every_shape_is_translated(self):
        for spec in self.SPECS:
            with self.subTest(spec=spec):
                self.assertNotEqual(self.label("en", spec),
                                    self.label("fr", spec))

    def test_the_english_is_unchanged(self):
        self.assertEqual(self.label("en", self.SPECS[0]), "full month and year")
        self.assertEqual(self.label("en", self.SPECS[2]), "percentage")
        self.assertEqual(self.label("en", self.SPECS[3]),
                         "a number with 2 decimal places")

    def test_the_currency_code_is_data(self):
        self.assertIn("USD", self.label("fr", self.SPECS[1]))

    def test_an_unknown_date_style_still_gets_a_label(self):
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                label = self.label(lang, {"type": "date", "style": "invented"})
                self.assertTrue(label.strip())
                self.assertNotIn("reply.rc.", label)

    def test_the_planner_s_wording_stays_english(self):
        from core.result_commands import _format_instruction

        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                self.assertEqual(
                    under(lang, _format_instruction, self.SPECS[2]),
                    "percentage, values are fractions 0 to 1")


class TestTheOptionsKeepTheirEnglishQuestion(unittest.TestCase):
    """Translating a label must not change what the pipeline is asked."""

    def format_options(self, lang):
        import core.result_commands as rc

        outcome = under(lang, rc._format_clarification, "src", 2,
                        i18n.lookup("reply.rc.which_format", lang),
                        [(i18n.lookup("reply.rc.fmt.month_year", lang),
                          "format this result as MMM-YY")])
        return outcome.clarification_options

    def test_the_label_is_translated(self):
        english = self.format_options("en")[0]["label"]
        french = self.format_options("fr")[0]["label"]
        self.assertNotEqual(english, french)

    def test_the_question_is_not(self):
        for lang in ("en", "fr"):
            with self.subTest(lang=lang):
                option = self.format_options(lang)[0]
                self.assertEqual(option["resolved_question"],
                                 "format this result as MMM-YY")

    def test_the_value_carries_the_question_not_the_label(self):
        # It used to be a copy of the label, which after translation would put
        # a French string where a re-plannable question belongs.
        option = self.format_options("fr")[0]
        self.assertEqual(option["value"], option["resolved_question"])

    def test_dispatcher_reads_the_question_first(self):
        # The precedence that makes translating a label safe, asserted rather
        # than assumed.
        option = self.format_options("fr")[0]
        chosen = str(option.get("resolved_question")
                     or option.get("value") or option.get("label") or "").strip()
        self.assertEqual(chosen, "format this result as MMM-YY")


if __name__ == "__main__":
    unittest.main()
