"""
tests/test_the_starter_questions_are_answerable.py

The greeting's starter questions, and the two ways they used to fail when
clicked.

core/conversational.py offers example questions in a greeting, after a vague
message, and in the "what can you do" reply. Without a signed-in reader it
reads them straight out of the query log on `success=1` — which is far weaker
than it reads, and is the exact test store/config_store.py's _harvest_qualifies
was written to replace. That function's own comment names the two cases:

  * a question that returned NO ROWS is logged successful, correctly, because
    nothing went wrong. Offered back as a starter question it reproduces the
    empty answer exactly, and is "the single most common thing a user sees
    behind 'the suggested question did nothing'";
  * a follow-up answered from the in-memory result snapshot is logged
    successful too, and its SQL is DuckDB dialect over a temporary table. Run
    against the warehouse it is a hard error.

The harvest was given that bar, and retroactively. This path never was, so the
two rows kept reaching readers from here.

The second half of the file is about who the questions belong to. With a
signed-in reader the function delegates to the workspace guide, which applies
that reader's table ACL — and then caught any failure and fell through to the
account-wide list below, at debug level. A governance decision made by an
exception handler, silently. Offering no starter questions is the safe failure;
offering someone else's tables is not.

Every test executes the real function against a database of its own.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolate the database before store is imported anywhere in this module.
os.environ["QUERYBOT_DB_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="qb-starter-questions-"), "test_starters.db")
for _mod in [m for m in list(sys.modules) if m.startswith("store")]:
    del sys.modules[_mod]

import store.db as db_mod  # noqa: E402
from store.db import get_db  # noqa: E402

db_mod.init_db()

from core.conversational import _example_questions  # noqa: E402

ACCOUNT = "starters-acct"

ANSWERED = "total revenue by region"
DB_FALLBACK = "invoices last month"
EMPTY = "revenue for a customer that does not exist"
IN_MEMORY_ROUTE = "who is below average"
IN_MEMORY_DIALECT = "show the ratio of charges to fills"

# (question, row_count, llm_provider, llm_model, should_be_offered)
#
# The route and the dialect are two separate exclusions, so they get a row
# each: a row carrying both would let either clause alone account for it, and
# a test that cannot tell them apart cannot notice one of them going missing.
# DB_FALLBACK is the case that must survive -- a result-chat question answered
# by going back to the warehouse, whose SQL is genuine warehouse SQL.
LOGGED = [
    (ANSWERED, 12, "openai", "gpt-x", True),
    (DB_FALLBACK, 40, "result_chat_db_fallback", "azure_sql", True),
    (EMPTY, 0, "openai", "gpt-x", False),
    (IN_MEMORY_ROUTE, 5, "governed_result_cache", "azure_sql", False),
    (IN_MEMORY_DIALECT, 3, "openai", "duckdb", False),
]


def _seed():
    """query_log exactly as the pipeline writes it — every row success=1,
    because every one of them is a query that did not raise."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO client (account_id, client_name, platform_type)"
            " VALUES (?,?,?)", (ACCOUNT, "Starters Test", "portal"),
        )
        conn.execute("DELETE FROM query_log WHERE account_id=?", (ACCOUNT,))
        for question, rows, provider, model, _offered in LOGGED:
            conn.execute(
                "INSERT INTO query_log (account_id, question, sql_generated,"
                " row_count, success, llm_provider, llm_model)"
                " VALUES (?,?,?,?,1,?,?)",
                (ACCOUNT, question, "SELECT 1", rows, provider, model),
            )


def _offered(limit: int = 10) -> list[str]:
    """What a system greeting with no signed-in reader would show."""
    _seed()
    # The metric registry is the earlier source in this function; empty it so
    # the query-log half is what the assertions are about.
    with patch("store.list_metrics", return_value=[]):
        return [q.rstrip("?").strip().lower() for q in
                _example_questions(ACCOUNT, limit=limit)]


def test_the_scan_finds_anything_at_all():
    """Without this, every assertion below would pass on an empty list."""
    assert ANSWERED in _offered()


def test_a_question_that_returned_no_rows_is_not_offered():
    """The chip that "does nothing" when clicked: it produced an empty result
    once, and was offered back as a question worth asking."""
    assert EMPTY not in _offered()


def test_an_in_memory_follow_up_is_not_offered():
    """Its SQL ran against an already-fetched snapshot in DuckDB dialect.
    Against the warehouse it is a hard error — the other way a chip fails.

    Both markers, separately: the route it was answered through and the
    dialect its SQL is written in are two independent exclusions, and a row
    carrying both would hide the loss of either."""
    offered = _offered()
    assert IN_MEMORY_ROUTE not in offered
    assert IN_MEMORY_DIALECT not in offered


def test_a_real_warehouse_answer_is_still_offered():
    """The bar has to keep doing the job as well as refusing: a greeting with
    no starter questions at all is its own defect — and the result-chat
    question that went back to the warehouse is a genuine example, despite
    having been asked from a card."""
    assert sorted(_offered()) == sorted([ANSWERED, DB_FALLBACK])


def test_a_signed_in_reader_gets_their_own_scope_or_nothing():
    """The guide applies this reader's table ACL. The legacy list below it is
    account-wide, so falling through on a failure would hand them questions
    about tables they cannot see."""
    _seed()
    with patch("core.workspace_guide.build_workspace_guide",
               side_effect=RuntimeError("semantic model unreadable")):
        assert _example_questions(ACCOUNT, portal_user={"id": 1, "lang": "en"}) == []


def test_the_fall_through_is_not_silent(caplog):
    """A fail-open handler on a user-facing path is correct design, and the
    log is then the only signal that the feature is dead. This one was at
    debug."""
    import logging

    _seed()
    with caplog.at_level(logging.WARNING, logger="querybot.conversational"):
        with patch("core.workspace_guide.build_workspace_guide",
                   side_effect=RuntimeError("semantic model unreadable")):
            _example_questions(ACCOUNT, portal_user={"id": 1, "lang": "en"})
    assert any(record.levelno >= logging.WARNING for record in caplog.records), (
        "a reader silently lost their governed starter questions"
    )


def test_a_signed_in_reader_gets_the_guides_questions_when_it_works():
    """The delegation still has to happen — a test that only proves the
    failure path would pass on a function that always returns []."""
    with patch("core.workspace_guide.build_workspace_guide",
               return_value={"examples": ["Show monthly revenue?", "b?", "c?", "d?"]}):
        assert _example_questions(
            ACCOUNT, limit=2, portal_user={"id": 1, "lang": "en"},
        ) == ["Show monthly revenue?", "b?"]
