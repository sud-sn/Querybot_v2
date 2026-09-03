"""
tests/test_prompt_cache_preload.py

Nothing in this codebase cached a prompt: `cache_control` appeared zero times,
and the prompt was the wrong shape for it anyway. `build_sql_system_prompt`
gates rules on the question, so its opening bytes changed from one question to
the next, and a prefix that changes is a prefix that caches nothing.

Two pieces here, and the second is worthless without the first:

- `core/kb_preload` sends every table the user is permitted to see, in a fixed
  order, instead of searching for a handful. That is what makes the block the
  same for every question.
- `core/prompt_cache` marks the breakpoint between that block and everything
  the question touches, and shapes it for whichever provider is configured.

The assertions that matter are the ones about *stability*: a split whose two
halves are wrong caches silently and forever, reporting no error at all.
"""

import unittest
from unittest.mock import patch

from core.kb_preload import _cache, invalidate_account_kb, preload_account_kb
from core.llm import build_sql_system_prompt
from core.prompt_cache import (
    CachedPrompt,
    anthropic_system_blocks,
    as_prompt_text,
    cache_usage,
    prompt_cache_enabled,
)

_DOCS = {
    "DW.DBO.SALES_FCT": "# DW.DBO.SALES_FCT\n## Columns\n- AMOUNT decimal\n",
    "DW.DBO.CUST_DMS": "# DW.DBO.CUST_DMS\n## Columns\n- CUST_NAME varchar\n",
    "DW.DBO.ITEM_DMS": "# DW.DBO.ITEM_DMS\n## Columns\n- ITEM_DESC varchar\n",
}


def _fake_fetch(account_id, fqn, sections=None):
    return _DOCS.get(fqn.upper())


def _preload(account_id, tables, **kwargs):
    with patch("core.vector_store.fetch_docs_for_fqn", _fake_fetch):
        return preload_account_kb(account_id, tables, **kwargs)


class ThePreloadedBlockIsStable(unittest.TestCase):
    """The whole point: identical bytes for every question on an account."""

    def setUp(self):
        _cache.clear()
        self.addCleanup(_cache.clear)

    def test_every_permitted_table_is_included(self):
        context, tables = _preload("acct", set(_DOCS))
        self.assertEqual(len(tables), 3)
        for fqn in _DOCS:
            self.assertIn(fqn, context)

    def test_the_order_does_not_depend_on_how_tables_were_supplied(self):
        """A prefix that reorders itself caches nothing, and says nothing."""
        forwards, _ = _preload("acct", list(_DOCS))
        _cache.clear()
        backwards, _ = _preload("acct", list(reversed(list(_DOCS))))
        self.assertEqual(forwards, backwards)

    def test_a_table_the_user_may_not_query_is_never_fetched(self):
        """Preloading changes how much of the permitted KB is sent, not which."""
        context, tables = _preload("acct", {"DW.DBO.SALES_FCT"})
        self.assertEqual(tables, ["DW.DBO.SALES_FCT"])
        self.assertNotIn("CUST_NAME", context)

    def test_an_account_with_no_documents_returns_nothing_to_send(self):
        """So the caller falls back to retrieval instead of prompting blind."""
        with patch("core.vector_store.fetch_docs_for_fqn", lambda *a, **k: None):
            context, tables = preload_account_kb("empty", {"DW.DBO.SALES_FCT"})
        self.assertEqual((context, tables), ("", []))

    def test_the_budget_stops_before_a_document_is_cut_in_half(self):
        big = {f"T{i}": "#" * 5000 for i in range(10)}
        with patch("core.vector_store.fetch_docs_for_fqn",
                   lambda a, fqn, sections=None: big.get(fqn.upper())):
            context, tables = preload_account_kb("acct", set(big), cap=12000)
        self.assertLess(len(tables), len(big))
        self.assertLessEqual(len(context), 12000)

    def test_a_rebuild_invalidates_the_cached_block(self):
        first, _ = _preload("acct", set(_DOCS))
        invalidate_account_kb("acct")
        with patch("core.vector_store.fetch_docs_for_fqn",
                   lambda a, fqn, sections=None: "# CHANGED\n"):
            second, _ = preload_account_kb("acct", set(_DOCS))
        self.assertNotEqual(first, second)
        self.assertIn("CHANGED", second)

    def test_writing_the_kb_index_invalidates_the_preload(self):
        """Not a separate call to remember: the write path itself expires it."""
        import core.vector_store as vector_store

        _preload("acct", set(_DOCS))
        _preload("acct", {"DW.DBO.SALES_FCT"})
        self.assertEqual(len([k for k in _cache if k[0] == "acct"]), 2)

        vector_store._invalidate_bm25_cache("acct")
        self.assertEqual([k for k in _cache if k[0] == "acct"], [])

    def test_two_users_with_different_grants_do_not_evict_each_other(self):
        _preload("acct", set(_DOCS))
        _preload("acct", {"DW.DBO.SALES_FCT"})
        self.assertEqual(len([k for k in _cache if k[0] == "acct"]), 2)

        with patch("core.vector_store.fetch_docs_for_fqn") as fetch:
            wide, _ = preload_account_kb("acct", set(_DOCS))
            narrow, _ = preload_account_kb("acct", {"DW.DBO.SALES_FCT"})
        fetch.assert_not_called()
        self.assertIn("CUST_NAME", wide)
        self.assertNotIn("CUST_NAME", narrow)


class TheBreakpointFallsBetweenStableAndVarying(unittest.TestCase):
    """A split with the wrong halves caches silently and forever."""

    KB = "### SALES_FCT\n- AMOUNT (decimal) invoice amount\n" * 400

    def _prompt(self, question, **kwargs):
        return build_sql_system_prompt(
            "azure_sql", "TERM: revenue -> SALES_FCT.AMOUNT",
            question=question, stable_context=self.KB, **kwargs,
        )

    def test_the_stable_half_is_identical_across_different_questions(self):
        a = self._prompt("total revenue by customer this year", return_parts=True)
        b = self._prompt("how does this month compare to last year", return_parts=True)
        self.assertEqual(a.stable, b.stable)

    def test_the_varying_half_actually_varies(self):
        """Otherwise the split is real but the gating it was built for is not."""
        a = self._prompt("total revenue by customer this year", return_parts=True)
        b = self._prompt("how does this month compare to last year", return_parts=True)
        self.assertNotEqual(a.volatile, b.volatile)

    def test_splitting_changes_nothing_about_what_is_sent(self):
        question = "top 10 products by quantity sold"
        split = self._prompt(question, return_parts=True)
        flat = self._prompt(question)
        self.assertEqual(split.text, flat)

    def test_without_a_preload_there_is_no_breakpoint(self):
        prompt = build_sql_system_prompt(
            "azure_sql", "some context", question="anything", return_parts=True,
        )
        self.assertEqual(prompt.stable, "")
        self.assertFalse(prompt.cacheable)

    def test_the_knowledge_base_leads_the_prompt(self):
        prompt = self._prompt("total revenue", return_parts=True)
        self.assertIn("AMOUNT (decimal) invoice amount", prompt.stable)
        self.assertIn("STRICT RULES", prompt.volatile)


class ShapingForTheProvider(unittest.TestCase):

    def test_anthropic_gets_two_blocks_with_the_breakpoint_marked(self):
        prompt = CachedPrompt(stable="x" * 20000, volatile="the question part")
        blocks = anthropic_system_blocks(prompt)
        self.assertIsInstance(blocks, list)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
        self.assertNotIn("cache_control", blocks[1])
        self.assertEqual(blocks[1]["text"], "the question part")

    def test_a_prefix_too_small_to_cache_is_sent_as_a_plain_string(self):
        """Marking it would change the request and cache nothing."""
        prompt = CachedPrompt(stable="short", volatile="rest")
        self.assertIsInstance(anthropic_system_blocks(prompt), str)

    def test_a_plain_string_is_passed_straight_through(self):
        self.assertEqual(anthropic_system_blocks("just a prompt"), "just a prompt")

    def test_other_providers_see_exactly_the_concatenation(self):
        prompt = CachedPrompt(stable="A", volatile="B")
        self.assertEqual(as_prompt_text(prompt), "A\n\nB")
        self.assertEqual(len(prompt), len("A\n\nB"))

    def test_cache_counters_are_read_when_the_sdk_reports_them(self):
        usage = type("U", (), {"cache_read_input_tokens": 900,
                               "cache_creation_input_tokens": 0})()
        self.assertEqual(cache_usage(usage), {"cache_read_input_tokens": 900})

    def test_an_sdk_that_reports_no_cache_counters_is_not_an_error(self):
        self.assertEqual(cache_usage(object()), {})


class TheSwitch(unittest.TestCase):
    """Off by default: preloading reorders the prompt that decides answers."""

    def test_default_is_off(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("QUERYBOT_PROMPT_CACHE", None)
            self.assertFalse(prompt_cache_enabled())

    def test_it_can_be_turned_on(self):
        with patch.dict("os.environ", {"QUERYBOT_PROMPT_CACHE": "on"}):
            self.assertTrue(prompt_cache_enabled())


if __name__ == "__main__":
    unittest.main()
