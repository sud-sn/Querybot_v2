"""
tests/test_answer_language.py

The reader's language, activated for the whole answer.

Every deterministic producer downstream -- build_answer, the insight summary,
the decision signal, the coverage caveats -- reads the language through
core.i18n's ContextVar rather than taking a parameter. Threading one through
build_assistant_response's collaborators is a diff a new call site silently
forgets, and this codebase has already been bitten by a consumer reading a
ContextVar default and degrading with no error.

So the activation itself is the thing worth testing, and it is tested by
EXECUTING handle_query -- the real one -- with the pipeline body stubbed, and
recording what the language was while the body ran.
"""

from __future__ import annotations

import asyncio

import pytest

from core import i18n


def _run_handle_query(portal_user):
    """Call the real handle_query and report the language its body saw."""
    import core.query_pipeline as qp

    seen = {}

    async def _fake_impl(account_id, event, adapter, question, user, **kw):
        seen["lang"] = i18n.get_active_language()
        return "done"

    original = qp._handle_query_impl
    qp._handle_query_impl = _fake_impl
    try:
        result = asyncio.run(qp.handle_query(
            "acct", object(), object(), "revenue by region", portal_user,
        ))
    finally:
        qp._handle_query_impl = original
    return seen.get("lang"), result


class TestTheAnswerLanguageFollowsTheUser:

    def test_a_french_user_gets_a_french_answer_context(self):
        lang, result = _run_handle_query({"id": 1, "lang": "fr"})
        assert lang == "fr"
        assert result == "done"

    def test_an_english_user_is_unchanged(self):
        assert _run_handle_query({"id": 1, "lang": "en"})[0] == "en"

    def test_a_pre_migration_row_without_the_column_falls_back(self):
        """store/db.py logs a failed ALTER at debug and continues, so a
        deployment can boot with portal_user.lang missing. The dict simply
        lacks the key, with no exception to warn anyone."""
        assert _run_handle_query({"id": 1})[0] == "en"

    def test_no_portal_user_at_all_falls_back(self):
        """Slack, Teams and the REST API all reach handle_query with
        portal_user=None."""
        assert _run_handle_query(None)[0] == "en"

    def test_a_junk_value_does_not_reach_the_catalogue(self):
        assert _run_handle_query({"id": 1, "lang": "klingon"})[0] == "en"

    def test_the_activation_is_released_afterwards(self):
        """The token is reset in the finally, and this observes it.

        The obvious version of this test -- call the helper, then read the
        language -- cannot fail: asyncio.run() builds a fresh Context for the
        coroutine, so a set() inside it was never going to be visible outside
        whether or not anything reset it. It has to run in ONE context, which
        is also the real case: a clarification retry calls handle_query a
        second time inside the same request.
        """
        import core.query_pipeline as qp

        seen = []

        async def _record(account_id, event, adapter, question, user, **kw):
            seen.append(i18n.get_active_language())
            return "done"

        async def _two_answers_in_one_context():
            outer = i18n.activate_language("fr")
            try:
                seen.append(i18n.get_active_language())          # fr
                await qp.handle_query("acct", object(), object(), "q",
                                      {"lang": "en"})            # en inside
                seen.append(i18n.get_active_language())          # fr again
            finally:
                i18n.deactivate_language(outer)

        original = qp._handle_query_impl
        qp._handle_query_impl = _record
        try:
            asyncio.run(_two_answers_in_one_context())
        finally:
            qp._handle_query_impl = original
        assert seen == ["fr", "en", "fr"], seen

    def test_it_is_released_even_when_the_pipeline_raises(self):
        import core.query_pipeline as qp

        async def _boom(*a, **kw):
            raise RuntimeError("pipeline exploded")

        original = qp._handle_query_impl
        qp._handle_query_impl = _boom
        try:
            with pytest.raises(RuntimeError):
                asyncio.run(qp.handle_query("acct", object(), object(), "q",
                                            {"id": 1, "lang": "fr"}))
        finally:
            qp._handle_query_impl = original
        assert i18n.get_active_language() == "en"

    def test_two_concurrent_answers_do_not_borrow_each_others_language(self):
        """The reason this is a ContextVar and not a module global."""
        import core.query_pipeline as qp

        async def _slow(account_id, event, adapter, question, user, **kw):
            await asyncio.sleep(0)
            return i18n.get_active_language()

        original = qp._handle_query_impl
        qp._handle_query_impl = _slow
        try:
            async def both():
                return await asyncio.gather(*[
                    qp.handle_query("acct", object(), object(), "q", {"lang": lang})
                    for lang in ("fr", "en") * 8
                ])
            results = asyncio.run(both())
        finally:
            qp._handle_query_impl = original
        assert results[0::2] == ["fr"] * 8
        assert results[1::2] == ["en"] * 8
