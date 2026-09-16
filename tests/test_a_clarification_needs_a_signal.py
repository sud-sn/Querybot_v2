"""
tests/test_a_clarification_needs_a_signal.py

The bot may only ask "which measure did you mean?" when the reader said
something a measure could disambiguate.

A client asked how many pallets shipped from a warehouse. No SQL could be
written for it, and the pipeline answered "which measure do you want?" over net
revenue and gross margin — two measures the question had never named. The menu
was not invented by the model out of nothing: core/clarification.py's
_scored_terms_for_ambiguity_menu hands over the tenant's whole vocabulary when
the question scores against nothing, and the constrained-menu model dutifully
picks two. That fail-open is deliberate and fine — the caller is supposed to
have decided there IS an ambiguity before calling.

One caller had. The zero-row path in core/query_pipeline.py computed the
decision inline and only asked when the glossary could point at something the
question actually said. The CANNOT_GENERATE branch, added for the same failure
mode, had no such check — because the rule existed only as three local
variables inside another branch, so there was nothing for it to call.

So the rule is now one named predicate, and this file is its contract plus a
syntax-tree check that both branches consult it.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Never the application database. Nothing here reads a table — the store is
# patched at its one entry point — but importing `store` must not be the thing
# that opens data/querybot.db.
os.environ.setdefault(
    "QUERYBOT_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="qb-clar-signal-"), "querybot.db"),
)

QUESTION = "how many pallets did we ship from the north warehouse yesterday"
SCOPE = {"DBO.F_SHIPMENT"}


def _metric(term: str) -> dict:
    """A registered metric, shaped as match_terms_in_question returns one."""
    return {"id": 1, "term": term, "kind": "metric",
            "requires_clarification": 0, "clarification_options": ""}


def _needs_picking(term: str) -> dict:
    """A term an admin marked ambiguous AND gave options for."""
    return {"id": 2, "term": term, "kind": "dimension",
            "requires_clarification": 1,
            "clarification_options": '[{"label": "Billing active"}, '
                                     '{"label": "Signed in this month"}]'}


def test_a_measure_menu_is_only_offered_when_the_question_named_one():
    """The whole contract of the gate: silence unless the glossary can point
    at something the question actually said. Everything downstream of a True
    here — the constrained-menu model, the chips, the saved pending row — is
    built on the premise that the reader was ambiguous, and in the reported
    case they were not."""
    import store
    from core.clarification import has_ambiguity_signal

    def _decide(matches):
        # The store is the only boundary, patched on the `store` package the
        # helper imports. No glossary table is read and no database is opened.
        with patch.object(store, "match_terms_in_question",
                          return_value=matches) as matched:
            return has_ambiguity_signal("acct", QUESTION, SCOPE), matched

    # The reported defect. The question named nothing in the glossary, so
    # there is nothing to ask about — whatever the vocabulary contains.
    assert _decide([])[0] is False

    # One measure is a reading, not a choice between readings.
    assert _decide([_metric("net revenue")])[0] is False

    # A dimension the question named is not a measure ambiguity either.
    assert _decide([{"id": 4, "term": "warehouse", "kind": "dimension",
                     "requires_clarification": 0,
                     "clarification_options": ""}])[0] is False

    # Marked ambiguous but never given options: that can only produce an
    # option-less card, which is a different defect and not a signal here.
    assert _decide([{"id": 3, "term": "active customer", "kind": "dimension",
                     "requires_clarification": 1,
                     "clarification_options": ""}])[0] is False

    # And the two shapes that ARE worth a question still are, or the gate has
    # simply switched clarification off.
    assert _decide([_needs_picking("active customer")])[0] is True
    assert _decide([_metric("net revenue"), _metric("gross margin")])[0] is True

    # The reader's own table scope reaches the glossary, so a term their row
    # policy hides can never become the signal that interrogates them.
    _decision, matched = _decide([])
    assert matched.call_args.args[2] == SCOPE

    # A glossary we could not read is not a signal. This is a SQLite read
    # underneath; a locked database must produce the caller's own diagnostic,
    # not an invented question.
    with patch.object(store, "match_terms_in_question",
                      side_effect=RuntimeError("database is locked")):
        assert has_ambiguity_signal("acct", QUESTION, SCOPE) is False


def test_a_glossary_we_could_not_read_says_so_out_loud(caplog):
    """Failing open here switches clarification off at BOTH call sites, so the
    log is the only way to tell a workspace with nothing ambiguous from a
    glossary the product cannot reach. It was at debug, where nobody sees it —
    and before the gate existed this exception at least crashed the turn
    visibly."""
    import logging

    import store
    from core.clarification import has_ambiguity_signal

    with caplog.at_level(logging.WARNING, logger="querybot.clarification"):
        with patch.object(store, "match_terms_in_question",
                          side_effect=RuntimeError("database is locked")):
            assert has_ambiguity_signal("acct", QUESTION, SCOPE) is False

    loud = [
        record for record in caplog.records
        if record.levelno >= logging.WARNING
        and "ambiguity signal" in record.getMessage()
    ]
    assert loud, (
        "clarification was switched off for this turn and nothing above debug "
        "said so"
    )


def test_both_pipeline_branches_ask_the_gate_before_asking_the_reader():
    """The wiring, read as a syntax tree.

    _handle_query_impl needs a warehouse, a knowledge base, a live socket and
    a model to reach either branch, so this is the documented exception: read
    as an AST rather than as text, so it pins the invariant — one gate per
    clarification, each before its own ask — rather than a spelling.
    """
    import core.query_pipeline as qp

    fn = next(
        node for node in ast.walk(ast.parse(inspect.getsource(qp)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_handle_query_impl"
    )

    def _calls(node, name: str) -> list[int]:
        return sorted(
            n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
            and name in (getattr(n.func, "attr", ""), getattr(n.func, "id", ""))
        )

    asks = _calls(fn, "check_ambiguity_glossary_first")
    gates = _calls(fn, "has_ambiguity_signal")

    assert len(asks) == 2, (
        "a clarification entry point was added or removed; every one of them "
        f"needs its own relevance gate: {asks}"
    )
    assert len(gates) == len(asks), (
        f"{len(asks)} places ask the reader to disambiguate, {len(gates)} "
        "first check there is anything to disambiguate"
    )
    # Interleaved: each gate is consulted before the ask it guards, and no two
    # asks share one gate.
    assert gates[0] < asks[0] < gates[1] < asks[1], (gates, asks)

    # Counting and ordering are not enough: both gates would still be present,
    # in the same order, with their polarity inverted -- and inverted is the
    # reported defect restored. So each one's SENSE is pinned, from the If that
    # encloses it.
    #
    # The two sites are deliberately opposite. At the CANNOT_GENERATE branch
    # the gate guards the ASK, so it appears un-negated. At the zero-row branch
    # it guards the root-cause card that runs INSTEAD of asking, so it appears
    # under `not`.
    def _gate_calls(tree) -> list[ast.Call]:
        return [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "has_ambiguity_signal"
        ]

    negated: dict[int, bool] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not _gate_calls(node.test):
            continue
        # A `not` anywhere above the call inside this test negates it. Checking
        # whether the unparsed test STARTS with "not" would miss the real
        # shape at the first site, where the gate is the second operand of an
        # `and` — `can_request_clarification(event) and not has_ambiguity_...`.
        under_not = {
            call.lineno
            for unary in ast.walk(node.test)
            if isinstance(unary, ast.UnaryOp) and isinstance(unary.op, ast.Not)
            for call in _gate_calls(unary.operand)
        }
        for call in _gate_calls(node.test):
            negated[call.lineno] = call.lineno in under_not

    assert len(negated) == 2, negated
    ask_line, rca_line = sorted(negated)
    assert negated[ask_line] is False, (
        f"the gate guarding the clarification at line {ask_line} is negated — "
        "the bot now asks precisely when there is nothing to ask about, which "
        "is the reported defect inverted rather than fixed"
    )
    assert negated[rca_line] is True, (
        f"the gate guarding the root-cause card at line {rca_line} is not "
        "negated — the zero-row path sends the card when it should be asking, "
        "and asks when it should send the card"
    )
