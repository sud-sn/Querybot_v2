"""
tests/test_example_harvest_quality.py

The second, independent cause of "suggested questions in a new thread fail when
you click them".

validated_examples feeds two things: few-shot grounding in the SQL prompt, and
the tier-1 suggestion chips. It was filled by copying every query_log row with
success=1, and success=1 is a much weaker statement than it reads:

  * A question that returned NO ROWS is logged successful, correctly — nothing
    went wrong. It is still not an example of how to answer anything, and
    offering it back as a chip reproduces the empty answer exactly.
  * A follow-up answered from the in-memory result snapshot is logged
    successful too. Its SQL is DuckDB dialect over a temporary table; run
    against the warehouse it is a hard error, which is the other half of what
    a user sees when a chip "just fails".

So the product learned from its own empty answers and re-offered them, and
banked SQL that could only ever run somewhere the warehouse is not.

Fixing the harvest alone would not have fixed the symptom: the bad rows are
already in the table on any deployed instance, and their embeddings are already
in Qdrant. The bar is therefore retroactive, and the vector index is cleared
before the survivors are re-embedded — upsert never deletes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolate the DB before store is imported anywhere in this module.
os.environ["QUERYBOT_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_harvest.db")
for _mod in [m for m in list(sys.modules) if m.startswith("store")]:
    del sys.modules[_mod]

import store.config_store as config_store  # noqa: E402
import store.db as db_mod  # noqa: E402
from store.db import get_db  # noqa: E402

db_mod.init_db()

ACCOUNT = "harvest-acct"

# (question, sql, row_count, llm_provider, llm_model, should_survive)
LOGGED = [
    ("total revenue by region", "SELECT R, SUM(A) FROM F GROUP BY R", 12,
     "openai", "gpt-x", True),
    ("invoices last month", "SELECT * FROM F WHERE D >= '2024-07-01'", 40,
     "result_chat_db_fallback", "azure_sql", True),
    ("revenue for a customer that does not exist", "SELECT SUM(A) FROM F WHERE C='X'", 0,
     "openai", "gpt-x", False),
    ("who is below average", "SELECT * FROM s WHERE v < (SELECT MEDIAN(v) FROM s)", 5,
     "governed_result_cache", "duckdb", False),
    ("show the ratio of charges to fills", "SELECT a/b FROM s", 3,
     "governed_result_cache", "duckdb", False),
]

KB_AUTHORED = "authored during the KB build"


def _seed():
    """query_log as the pipeline writes it, plus validated_examples as the OLD
    harvest left it: everything copied across, because success=1 was the only
    test."""
    with get_db() as conn:
        conn.execute("DELETE FROM query_log WHERE account_id=?", (ACCOUNT,))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validated_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT NOT NULL,
                question TEXT NOT NULL, sql_query TEXT NOT NULL,
                table_name TEXT DEFAULT '', source TEXT DEFAULT 'query_log',
                created_at TEXT DEFAULT (datetime('now')))
        """)
        conn.execute("DELETE FROM validated_examples WHERE account_id=?", (ACCOUNT,))
        conn.execute(
            "INSERT OR IGNORE INTO client (account_id, client_name, platform_type)"
            " VALUES (?,?,?)", (ACCOUNT, "Harvest Test", "portal"),
        )
        for question, sql, rows, provider, model, _keep in LOGGED:
            conn.execute(
                "INSERT INTO query_log (account_id, question, sql_generated,"
                " row_count, success, llm_provider, llm_model) VALUES (?,?,?,?,1,?,?)",
                (ACCOUNT, question, sql, rows, provider, model),
            )
            conn.execute(
                "INSERT INTO validated_examples (account_id, question, sql_query,"
                " source) VALUES (?,?,?,'query_log')", (ACCOUNT, question, sql),
            )
        conn.execute(
            "INSERT INTO validated_examples (account_id, question, sql_query,"
            " source) VALUES (?,?,?,'kb_stage2')",
            (ACCOUNT, KB_AUTHORED, "SELECT 1 FROM F"),
        )


def _questions():
    return sorted(
        e["question"] for e in config_store.get_validated_examples(ACCOUNT)
    )


def test_a_zero_row_answer_is_not_an_example():
    """This is the chip that "does nothing" when clicked: it produced an empty
    result once, and was banked as a question worth suggesting."""
    _seed()
    config_store.harvest_successful_queries(ACCOUNT)
    assert "revenue for a customer that does not exist" not in _questions()


def test_in_memory_duckdb_sql_is_not_an_example():
    """That SQL ran against an already-fetched snapshot. Against the warehouse
    it is a hard error, which is the other way a chip fails."""
    _seed()
    config_store.harvest_successful_queries(ACCOUNT)
    surviving = _questions()
    assert "who is below average" not in surviving
    assert "show the ratio of charges to fills" not in surviving


def test_real_warehouse_answers_are_kept():
    """The harvest still has to do its job — including the DB-fallback route,
    whose SQL is genuine warehouse SQL despite running from a result chat."""
    _seed()
    config_store.harvest_successful_queries(ACCOUNT)
    surviving = _questions()
    assert "total revenue by region" in surviving
    assert "invoices last month" in surviving


def test_examples_from_the_kb_build_are_left_alone():
    """Different provenance, different quality bar — the purge only reverses
    what this harvest itself created."""
    _seed()
    config_store.harvest_successful_queries(ACCOUNT)
    assert KB_AUTHORED in _questions()


def test_the_bar_is_retroactive():
    """Every deployed instance already has these rows. A fix that only gates
    NEW arrivals would leave the reported symptom exactly as it is."""
    _seed()
    before = _questions()
    assert "who is below average" in before, "precondition: the bad rows exist"

    removed = config_store.purge_unqualified_examples(ACCOUNT)
    assert removed == 3, f"expected 3 unqualified rows removed, got {removed}"
    assert _questions() == sorted([
        "total revenue by region", "invoices last month", KB_AUTHORED,
    ])


def test_purging_is_idempotent():
    _seed()
    config_store.purge_unqualified_examples(ACCOUNT)
    assert config_store.purge_unqualified_examples(ACCOUNT) == 0


def test_a_harvest_that_adds_nothing_still_re_embeds_after_a_purge():
    """upsert_examples never deletes, so a pruned question keeps its Qdrant
    point and keeps reaching the prompt. Re-embedding only when `added > 0`
    would skip exactly the run that removed something."""
    import inspect

    import core.examples as examples

    source = inspect.getsource(examples.harvest_and_embed)
    assert "purge_unqualified_examples" in source
    assert "if added or removed:" in source
    assert "delete_examples(account_id)" in source, (
        "the survivors are re-upserted without clearing the index first, so "
        "every pruned point stays where it was"
    )


def test_the_index_clear_targets_only_examples():
    import inspect

    import core.vector_store as vector_store

    source = inspect.getsource(vector_store.delete_examples)
    assert 'MatchValue(value="example")' in source
    assert "account_id" in source
