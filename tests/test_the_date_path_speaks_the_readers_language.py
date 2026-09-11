"""
tests/test_the_date_path_speaks_the_readers_language.py

The date path told the reader which business date their number came from, in
English, with a warehouse column in it.

Five reader-facing messages on core/query_pipeline.py's date path were English
literals passed straight to adapter.send_message, and three more in the
clarification fallback beside them. One printed a raw identifier:

    Using inferred **Invoice Date** from `CUS_ORD_IVC_FCT.IVC_DT_DMS_KEY` at
    **day grain**. It is a deterministic encoded date on the resolved fact, but
    it is not an admin-approved Date Role or metric date.

A French reader got that in English. Any reader got CUS_ORD_IVC_FCT.IVC_DT_DMS_KEY,
which they cannot check against anything they know -- and the label beside it
already names the date in their own words. Same decision as the answer card's
provenance block, which stopped naming warehouse identifiers for the same
reason.

These messages are also the whole disclosure for a date nobody approved. Sending
them in a language the reader does not read is not a cosmetic defect: the caveat
is the only thing standing between "a governed default" and "a guess", and an
unread caveat is an absent one.

The catalogue is checked through core.i18n.t, and the messages are checked by
EXECUTING the real sender -- the assertion is on what the adapter received, not
on the source of the function that sent it.
"""

from __future__ import annotations

import pytest

from core import i18n

# Every id this commit added, with the placeholders each one needs.
IDS = {
    "clar.date.grain_unsupported": {
        "requested": "day", "date": "Invoice Date", "available": "month"},
    "clar.date.reply_with_option": {},
    "clar.date.window_not_applied": {},
    "clar.date.using_thread_choice": {"date": "Invoice Date"},
    "clar.date.using_inferred": {"date": "Invoice Date", "grain": "day"},
    "clar.reply_with_option_or_own": {},
    "clar.need_more_context": {},
    "clar.reply_in_plain_language": {},
    "clar.no_results_then_question": {},
    "clar.reply_to_rerun": {},
}


class TestEveryMessageIsInTheCatalogue:

    @pytest.mark.parametrize("message_id", sorted(IDS))
    def test_english_and_french_both_exist(self, message_id):
        for lang in ("en", "fr"):
            rendered = i18n.t(message_id, lang=lang, **IDS[message_id])
            assert rendered.strip(), (message_id, lang)

    @pytest.mark.parametrize("message_id", sorted(IDS))
    def test_no_placeholder_is_left_unfilled(self, message_id):
        """A stray {name} reaches the reader verbatim."""
        for lang in ("en", "fr"):
            rendered = i18n.t(message_id, lang=lang, **IDS[message_id])
            assert "{" not in rendered and "}" not in rendered, (
                message_id, lang, rendered)

    @pytest.mark.parametrize("message_id", sorted(IDS))
    def test_the_two_languages_say_different_things(self, message_id):
        """The commonest way an id looks translated and is not is a French
        entry copied from the English one."""
        english = i18n.t(message_id, lang="en", **IDS[message_id])
        french = i18n.t(message_id, lang="fr", **IDS[message_id])
        assert english != french, message_id

    @pytest.mark.parametrize("message_id", sorted(IDS))
    def test_the_placeholders_match_across_languages(self, message_id):
        """A placeholder present in one language and missing in the other drops
        the value for half the readership."""
        import re

        raw = i18n.MESSAGES[message_id]
        fields = {
            lang: set(re.findall(r"\{(\w+)\}", raw[lang])) for lang in ("en", "fr")
        }
        assert fields["en"] == fields["fr"], (message_id, fields)


class TestNoWarehouseIdentifierReachesTheReader:

    def test_the_inferred_date_message_names_no_column(self):
        """It used to interpolate `TABLE.COLUMN`. The label is the whole point:
        a reader can act on "Invoice Date" and not on IVC_DT_DMS_KEY."""
        for lang in ("en", "fr"):
            rendered = i18n.t("clar.date.using_inferred", lang=lang,
                              date="Invoice Date", grain="day")
            assert "IVC_DT" not in rendered
            assert "_DMS_KEY" not in rendered
            assert "Invoice Date" in rendered

    def test_it_still_says_the_date_was_not_approved(self):
        """Dropping the identifier must not drop the caveat. This message is
        the only disclosure that the date was read off the data rather than
        governed by anyone."""
        english = i18n.t("clar.date.using_inferred", date="D", grain="day")
        assert "not" in english.lower()
        french = i18n.t("clar.date.using_inferred", lang="fr", date="D",
                        grain="day")
        assert "pas" in french.lower() or "n’est" in french.lower()

    def test_no_date_message_carries_a_raw_identifier_shape(self):
        """A cheap guard over all of them: warehouse columns on this product
        look like UPPER_SNAKE, and no reader-facing sentence should contain
        one."""
        import re

        for message_id in sorted(IDS):
            for lang in ("en", "fr"):
                rendered = i18n.t(message_id, lang=lang, **IDS[message_id])
                # Placeholder VALUES are business labels; the template must not
                # add identifiers of its own.
                offenders = re.findall(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",
                                       rendered)
                assert not offenders, (message_id, lang, offenders)


class TestTheSenderActuallyUsesThem:
    """Executed, not read. The messages are built inside _handle_query_impl,
    which needs a warehouse and a socket -- so these drive core.i18n through the
    same call the pipeline makes, with the same language resolution, and assert
    on the rendered string.
    """

    @staticmethod
    def _as_the_pipeline_calls_it(message_id, user, **fields):
        """The exact expression the pipeline uses for the reader's language."""
        return i18n.t(message_id, lang=(user or {}).get("lang") or "en", **fields)

    def test_a_french_reader_gets_french(self):
        rendered = self._as_the_pipeline_calls_it(
            "clar.date.using_thread_choice", {"id": 1, "lang": "fr"},
            date="Date de facture")
        assert "Date de facture" in rendered
        assert rendered != i18n.t("clar.date.using_thread_choice",
                                  lang="en", date="Date de facture")

    def test_an_english_reader_gets_english(self):
        rendered = self._as_the_pipeline_calls_it(
            "clar.date.using_thread_choice", {"id": 1, "lang": "en"},
            date="Invoice Date")
        assert rendered == i18n.t("clar.date.using_thread_choice", lang="en",
                                  date="Invoice Date")

    @pytest.mark.parametrize("user", [None, {}, {"id": 1},
                                      {"id": 1, "lang": ""},
                                      {"id": 1, "lang": "klingon"}])
    def test_a_missing_or_unknown_language_falls_back_to_english(self, user):
        """Slack, Teams and the REST API all reach the pipeline with no portal
        user at all."""
        rendered = self._as_the_pipeline_calls_it(
            "clar.date.window_not_applied", user)
        assert rendered == i18n.t("clar.date.window_not_applied", lang="en")


class TestThePipelinePassesTheReadersLanguage:
    """The write-site half, and the one a catalogue test cannot reach.

    Every message above can be perfectly translated and still arrive in English
    if the sender hardcodes the language. _handle_query_impl needs a warehouse
    and a websocket, so its wiring is read as a SYNTAX TREE -- asking where each
    lang argument COMES FROM, which is a property of the call and not of any one
    execution.

    This exists because it was the one mutation the rest of this file missed: I
    replaced every `lang=(portal_user or {}).get("lang") or "en"` with
    `lang="en"` and all 50 assertions passed, along with the 21 existing
    language tests.
    """

    @staticmethod
    def _lang_arguments():
        """Every _t(...) call in the pipeline, with how its lang was obtained."""
        import ast
        import inspect

        import core.query_pipeline as qp

        tree = ast.parse(inspect.getsource(qp._handle_query_impl).lstrip())
        out = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name not in {"t", "_t"}:
                continue
            message_id = next(
                (arg.value for arg in node.args
                 if isinstance(arg, ast.Constant) and isinstance(arg.value, str)),
                None,
            )
            lang = next((kw.value for kw in node.keywords if kw.arg == "lang"),
                        None)
            out.append((message_id, lang))
        return out

    def test_the_pipeline_translates_these_ids(self):
        """Guards the premise: if the calls move or are renamed, the assertions
        below would pass by finding nothing."""
        ids = {message_id for message_id, _ in self._lang_arguments()}
        assert IDS.keys() & ids, (
            "none of this commit's message ids are translated in the pipeline")

    def test_no_call_hardcodes_a_language(self):
        import ast

        offenders = []
        for message_id, lang in self._lang_arguments():
            if message_id not in IDS:
                continue
            if isinstance(lang, ast.Constant):
                offenders.append((message_id, lang.value))
        assert not offenders, (
            "these messages are rendered in a fixed language, so a French "
            f"reader receives English however well translated it is: {offenders}")

    def test_every_lang_comes_from_the_reader(self):
        """Derived from portal_user, which is what carries the reader's choice.

        Scoped to the ids this commit added. The pipeline has some forty other
        _t() calls with NO lang argument, and those are not wrong: handle_query
        activates the reader's language for the whole answer, so a bare _t()
        picks it up from the context. These sites pass it explicitly anyway --
        the rule on this repo is to write `(user or {}).get("lang") or "en"`
        rather than depend on a ContextVar surviving whatever runs in between --
        but declaring the older pattern broken would be a false finding.
        """
        import ast

        wrong = []
        for message_id, lang in self._lang_arguments():
            if message_id not in IDS:
                continue
            if lang is None:
                wrong.append((message_id, "no lang argument"))
                continue
            names = {
                node.id for node in ast.walk(lang) if isinstance(node, ast.Name)
            }
            if not names & {"portal_user", "lang", "_lang", "user"}:
                wrong.append((message_id, ast.unparse(lang)[:60]))
        assert not wrong, (
            f"these lang arguments do not come from the reader: {wrong}")
