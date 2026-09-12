"""
tests/test_one_wide_table_does_not_kill_the_kb_build.py

A 210-column fact took the entire knowledge-base build down with it.

A table's KB document is eight fixed sections plus ONE ROW PER COLUMN in the
field table, so its length is linear in the column count. The output ceiling was
the constant 4096, and core/knowledge.py's own module docstring said it was
"so all 7 sections always fit" -- true for the narrow tables anyone had tried.

Measured on a synthetic 210-column EMDW_DMART fact: the prompt alone is 105,093
characters (~26k tokens), and the model was asked to document all 210 fields
across all eight sections in 4,096 output tokens. That is about 19 tokens per
field with the section headers included, so the provider stopped mid-document
every time.

Stopping mid-document is correctly refused. core/llm.py raises
LLMTruncatedError by design -- "a truncated completion is not a shorter answer,
it is a different one" -- and for a KB document that is right: one missing
"## Business Synonyms" header silently disables the whole deterministic synonym
layer for that table. But nothing caught it. The exception left the per-table
loop and ABORTED THE BUILD:

    build_kb            RAISED LLMTruncatedError
    KB documents        _business_kb.md only
    tables documented   0 of 3

CUS_RTN_FCT and WHS_DMS are narrow and would both have succeeded. They were
never attempted.

This matters beyond one wide table, and it is why this is a blocker rather than
a nuisance: ticking a vocabulary pack forces a FULL KB rebuild. An admin
enabling infor_m3 on an EMDW_DMART mart would have lost every KB document the
tenant had, and got back nothing.

Three changes, and all three are asserted below. The ceiling is derived from the
column count. An estimate that still falls short is retried once at the cap,
because the per-column allowance is a calibration and being wrong about one
table should cost a second call rather than the table. And a document that
cannot finish even at the cap skips THAT TABLE, loudly and durably, while the
rest of the build completes.

Raising a ceiling costs nothing that is not used: max_tokens is a limit, not a
request, so at or below 40 columns every budget here is byte-identical to the
constant it replaces.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from core.knowledge import (
    _KB_DOC_MAX_TOKENS,
    _kb_complete,
    kb_doc_token_budget,
    kb_query_token_budget,
    kb_vocab_token_budget,
)
from core.llm import LLMTruncatedError

WIDE = ("CUS_ORD_IVC_FCT", 210)
NARROW = ("CUS_RTN_FCT", 40)
TINY = ("WHS_DMS", 12)


def schema_md(table: str, column_count: int) -> str:
    """A schema document in the shape core/schema.py writes one."""
    rows = "\n".join(
        f"| `COL_{index:03d}_AMT` | decimal(18) | No |  |"
        for index in range(column_count)
    )
    return (
        f"# EMDW_DMART.{table}\n\n"
        f"**Type:** BASE TABLE  **Schema:** EMDW_DMART\n\n"
        f"**SQL table name:** `EMDW_DMART.{table}`\n\n"
        f"**Row count:** 9200000  **Scale:** Large\n\n"
        f"## Columns\n\n"
        f"| Column | Type | Nullable | Distinct Values |\n"
        f"|--------|------|:--------:|-----------------|\n{rows}\n"
    )


class TestTheBudgetFollowsTheWidth:

    def test_a_narrow_table_asks_for_exactly_what_it_used_to(self):
        """4096 and 3000 were the constants. Below the free width nothing about
        an existing tenant's build changes."""
        for column_count in (0, 1, 12, 40):
            assert kb_doc_token_budget(column_count) == 4096, column_count
            assert kb_query_token_budget(column_count) == 3000, column_count
        for table_count in (0, 1, 10):
            assert kb_vocab_token_budget(table_count) == 3000, table_count

    @pytest.mark.parametrize("column_count,expected", [
        (41, 4096 + 48),
        (56, 4096 + 48 * 16),
        (121, 4096 + 48 * 81),
        (210, 4096 + 48 * 170),
        (227, 4096 + 48 * 187),
    ])
    def test_a_wide_table_asks_for_more(self, column_count, expected):
        assert kb_doc_token_budget(column_count) == expected

    def test_the_measured_case_gets_three_times_the_room(self):
        """210 columns at 19 tokens each was the defect; this is the fix in one
        number."""
        assert kb_doc_token_budget(210) == 12_256
        assert kb_doc_token_budget(210) / 210 > 55

    @pytest.mark.parametrize("column_count", [288, 400, 5000])
    def test_it_stops_at_a_cap_every_provider_accepts(self, column_count):
        assert kb_doc_token_budget(column_count) == _KB_DOC_MAX_TOKENS

    def test_it_never_decreases_with_width(self):
        budgets = [kb_doc_token_budget(n) for n in range(0, 500, 7)]
        assert budgets == sorted(budgets)

    def test_the_vocabulary_budget_follows_the_table_count(self):
        assert kb_vocab_token_budget(11) == 3300
        assert kb_vocab_token_budget(20) == 6000
        assert kb_vocab_token_budget(500) == 12_000

    @pytest.mark.parametrize("bad", [None, -1, -500])
    def test_a_nonsense_width_falls_back_to_the_base(self, bad):
        assert kb_doc_token_budget(bad) == 4096


class TestOneCompletionRetriesThenGivesUp(unittest.IsolatedAsyncioTestCase):

    @staticmethod
    def _provider(needs: int):
        """A provider that truncates until max_tokens reaches `needs`."""
        calls: list[int] = []

        async def fake(system, user, provider, model, api_key,
                       max_tokens=1024, **kw):
            calls.append(max_tokens)
            if max_tokens < needs:
                raise LLMTruncatedError(
                    f"truncated at {max_tokens}", text="## Overview\npartial")
            return "## Overview\nfinished\n", 100, 200

        return fake, calls

    async def _complete(self, needs, budget):
        fake, calls = self._provider(needs)
        with patch("core.llm.llm_complete", side_effect=fake):
            text, reason = await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "anthropic",
                "claude-sonnet-5", "key", max_tokens=budget)
        self.reason = reason
        return text, calls

    async def test_a_document_that_fits_is_returned_on_the_first_call(self):
        text, calls = await self._complete(needs=1000, budget=12_256)
        self.assertEqual(text, "## Overview\nfinished\n")
        self.assertEqual(calls, [12_256])

    async def test_a_short_estimate_is_retried_once_at_the_cap(self):
        """The per-column allowance is a calibration. Being wrong about one
        table should cost a second call, not the table."""
        text, calls = await self._complete(needs=13_000, budget=12_256)
        self.assertEqual(text, "## Overview\nfinished\n")
        self.assertEqual(calls, [12_256, _KB_DOC_MAX_TOKENS])

    async def test_a_document_that_cannot_finish_returns_none(self):
        text, calls = await self._complete(needs=40_000, budget=12_256)
        self.assertIsNone(text)
        self.assertEqual(calls, [12_256, _KB_DOC_MAX_TOKENS])
        self.assertEqual(self.reason, "output truncated at the token ceiling")

    async def test_an_estimate_already_at_the_cap_is_not_repeated(self):
        """Retrying the identical call would spend a second call to learn
        nothing."""
        text, calls = await self._complete(needs=40_000, budget=_KB_DOC_MAX_TOKENS)
        self.assertIsNone(text)
        self.assertEqual(calls, [_KB_DOC_MAX_TOKENS])

    async def test_a_configuration_error_still_propagates(self):
        """A refused key or a dead endpoint is not a document that was too
        long, and swallowing it would report a build that never ran.

        The rule sharpened after this test was written: a rejection that is
        about THIS request -- an output ceiling above what the deployment
        allows, a content filter -- is contained per table like a truncation.
        See test_a_wide_fact_is_not_the_whole_build for that half. Everything
        that would reject the next table identically still lands here."""
        async def boom(*a, **kw):
            raise RuntimeError("401 invalid api key")

        with patch("core.llm.llm_complete", side_effect=boom):
            with self.assertRaises(RuntimeError):
                await _kb_complete("document for X", "s", "u", "anthropic",
                                   "m", "k", max_tokens=4096)


class _Build(unittest.TestCase):
    """build_kb against a fake provider. The LLM and the vector index are the
    boundaries; everything between them is the real build."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="qb-kb-build-")
        self.schema_dir = os.path.join(self.root, "schema")
        self.kb_dir = os.path.join(self.root, "kb")
        for path in (self.schema_dir, self.kb_dir):
            os.makedirs(path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_schema(self, tables=(WIDE, NARROW, TINY)):
        for table, column_count in tables:
            with open(os.path.join(self.schema_dir, f"{table}.md"), "w") as handle:
                handle.write(schema_md(table, column_count))

    @staticmethod
    def classify(user: str) -> str:
        """Which of the build's three calls this is.

        Read from the prompt's own markers rather than from which table name
        appears first: the business-vocabulary prompt names every table, so
        matching on the table name attributed that call to whichever table
        happened to sort first.
        """
        if "Generate the complete Knowledge Base document" in user:
            return "table_doc"
        if "## Overview" in user:
            return "query_examples"
        return "business_vocab"

    def build(self, needs):
        """`needs(stage, table) -> int` is what the provider demands of a call."""
        calls: list[tuple[str, str, int]] = []

        async def fake(system, user, provider, model, api_key,
                       max_tokens=1024, **kw):
            stage = self.classify(user)
            table = "" if stage == "business_vocab" else next(
                (name for name, _ in (WIDE, NARROW, TINY) if name in user), "")
            calls.append((stage, table, max_tokens))
            required = needs(stage, table)
            if required and max_tokens < required:
                raise LLMTruncatedError(
                    f"truncated at {max_tokens}", text="## Overview\npartial")
            return "## Overview\nfinished\n", 100, 200

        from core import knowledge

        with patch("core.llm.llm_complete", side_effect=fake), \
                patch.object(knowledge, "_embed_kb_files_qdrant",
                             return_value=None):
            count = asyncio.run(knowledge.build_kb(
                schema_dir=self.schema_dir, kb_dir=self.kb_dir,
                chroma_dir="acct", business_desc="a plumbing distributor",
                provider="anthropic", model="claude-sonnet-5", api_key="key",
                account_id="acct"))
        return count, calls

    def docs(self):
        return sorted(name for name in os.listdir(self.kb_dir)
                      if name.endswith("_kb.md") and not name.startswith("_"))

    def examples(self):
        return sorted(name for name in os.listdir(self.kb_dir)
                      if name.endswith("_queries.md"))

    def failures(self):
        path = os.path.join(self.kb_dir, "_build_failures.json")
        if not os.path.exists(path):
            return []
        with open(path) as handle:
            return json.load(handle)


def only_the_wide_doc(required):
    """The wide fact's Stage 1 call needs `required` tokens; nothing else does."""
    def needs(stage, table):
        return required if (stage == "table_doc" and table == WIDE[0]) else 0
    return needs


class TestTheWideFactNoLongerTakesTheBuildWithIt(_Build):

    def test_every_table_is_documented_when_the_budget_reaches(self):
        self.write_schema()
        count, _calls = self.build(only_the_wide_doc(12_000))
        self.assertEqual(count, 3)
        self.assertEqual(self.docs(), ["CUS_ORD_IVC_FCT_kb.md",
                                       "CUS_RTN_FCT_kb.md", "WHS_DMS_kb.md"])
        self.assertEqual(self.failures(), [])

    def test_the_wide_fact_is_asked_for_its_own_budget(self):
        self.write_schema()
        _count, calls = self.build(only_the_wide_doc(0))
        wide = [tokens for stage, table, tokens in calls
                if stage == "table_doc" and table == WIDE[0]]
        self.assertEqual(wide, [kb_doc_token_budget(210)])
        narrow = [tokens for stage, table, tokens in calls
                  if stage == "table_doc" and table == NARROW[0]]
        self.assertEqual(narrow, [4096], "a 40-column table must be unchanged")
        vocab = [tokens for stage, _t, tokens in calls
                 if stage == "business_vocab"]
        self.assertEqual(vocab, [kb_vocab_token_budget(3)])

    def test_a_short_estimate_is_rescued_by_the_retry(self):
        self.write_schema()
        count, calls = self.build(only_the_wide_doc(13_000))
        self.assertEqual(count, 3)
        self.assertIn("CUS_ORD_IVC_FCT_kb.md", self.docs())
        wide = [tokens for stage, table, tokens in calls
                if stage == "table_doc" and table == WIDE[0]]
        self.assertEqual(wide, [kb_doc_token_budget(210), _KB_DOC_MAX_TOKENS])

    def test_a_table_that_cannot_fit_fails_alone(self):
        self.write_schema()
        count, _calls = self.build(only_the_wide_doc(40_000))
        self.assertEqual(self.docs(), ["CUS_RTN_FCT_kb.md", "WHS_DMS_kb.md"])
        self.assertEqual(count, 2, "the two narrow tables still built")

    def test_and_says_which_table_and_why(self):
        self.write_schema()
        self.build(only_the_wide_doc(40_000))
        failures = self.failures()
        self.assertEqual(len(failures), 1, failures)
        self.assertEqual(failures[0]["table"], "CUS_ORD_IVC_FCT")
        self.assertEqual(failures[0]["stage"], "table_doc")
        self.assertEqual(failures[0]["columns"], 210)
        self.assertEqual(failures[0]["max_tokens"], kb_doc_token_budget(210))

    def test_a_skipped_table_leaves_no_half_written_document(self):
        """A document missing its ## Business Synonyms header silently disables
        the deterministic synonym layer for that table, so a partial one is
        worse than none."""
        self.write_schema()
        self.build(only_the_wide_doc(40_000))
        self.assertNotIn("CUS_ORD_IVC_FCT_kb.md", self.docs())
        self.assertNotIn("CUS_ORD_IVC_FCT_queries.md", self.examples())

    def test_a_clean_rebuild_clears_a_stale_failure_record(self):
        """Otherwise last week's failure reads as current."""
        self.write_schema()
        self.build(only_the_wide_doc(40_000))
        self.assertTrue(self.failures())
        self.build(only_the_wide_doc(0))
        self.assertEqual(self.failures(), [])


class TestTheOtherTwoCallsAreContainedToo(_Build):

    def test_a_lost_query_example_does_not_lose_the_document(self):
        """Stage 2 writes worked examples. The table is documented either way,
        so it is recorded but not skipped."""
        self.write_schema()

        def needs(stage, table):
            return 40_000 if (stage == "query_examples"
                              and table == WIDE[0]) else 0

        count, _calls = self.build(needs)
        self.assertEqual(count, 3)
        self.assertIn("CUS_ORD_IVC_FCT_kb.md", self.docs())
        self.assertNotIn("CUS_ORD_IVC_FCT_queries.md", self.examples())
        stages = [item["stage"] for item in self.failures()]
        self.assertEqual(stages, ["query_examples"])

    def test_a_lost_business_vocabulary_does_not_stop_table_one(self):
        """This call runs BEFORE the loop, so a truncation here used to mean not
        one table was ever attempted."""
        self.write_schema()

        def needs(stage, _table):
            return 40_000 if stage == "business_vocab" else 0

        count, _calls = self.build(needs)
        self.assertEqual(count, 3)
        self.assertEqual(self.docs(), ["CUS_ORD_IVC_FCT_kb.md",
                                       "CUS_RTN_FCT_kb.md", "WHS_DMS_kb.md"])
        self.assertNotIn("_business_kb.md", os.listdir(self.kb_dir))
