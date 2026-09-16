"""
tests/test_a_pruned_suggestion_stays_pruned.py

The third cause of "the questions in the chat tab mostly did not produce
results", and the only one that survived the tier reordering and the harvest
bar.

The suggestion cache is written from every Q: line in the Stage-2 files, before
any of that SQL has been near a database. Validation then compiles each one and
prunes the cache down to the survivors. That much has worked for a while.

What did not work is the second lap. When the first validation finds failures,
the KB build asks the model to repair them, rewrites the Stage-2 files, and
validates again -- and the rebuild of the suggestion cache sat AFTER that second
validation. build_suggestion_cache reads the files, not the validation result,
so it restored every question the prune had just removed. The builds that got
this treatment were exactly the builds that had failures, which is to say the
workspaces whose panels most needed the prune.

The first test here executes the real functions to show the hazard is real: a
prune followed by a rebuild loses the prune entirely. The second reads the KB
build route as a syntax tree and asserts the rebuild precedes the validation
that prunes. That route needs a live warehouse, an LLM and a running admin
session before it reaches the branch in question, so its statement ORDER is the
one thing about it that can be checked here -- and the order is the whole
defect.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.suggestions import (  # noqa: E402
    _CACHE_FILENAME,
    build_suggestion_cache,
    prune_suggestion_cache,
)

COMPILES = "Revenue by region last quarter"
DOES_NOT = "Scrap rate by work centre"

STAGE_TWO = f"""# DW.DBO.F_SALES

Q: {COMPILES}
SQL: SELECT region, SUM(amount) FROM DW.DBO.F_SALES GROUP BY region

Q: {DOES_NOT}
SQL: SELECT work_centre, scrap_rate FROM DW.DBO.F_SALES
"""


def _kb_dir(tmp: str) -> str:
    (Path(tmp) / "F_SALES_queries.md").write_text(STAGE_TWO, encoding="utf-8")
    return tmp


def _cached(kb_dir: str) -> list[str]:
    raw = (Path(kb_dir) / _CACHE_FILENAME).read_text(encoding="utf-8")
    return [entry["question"] for entry in json.loads(raw)]


def test_rebuilding_the_cache_after_a_prune_undoes_the_prune():
    """The hazard itself, executed. build_suggestion_cache reads the Stage-2
    files, so it cannot know what validation rejected -- which is precisely why
    it must never be the last step."""
    with tempfile.TemporaryDirectory() as tmp:
        kb_dir = _kb_dir(tmp)

        build_suggestion_cache(kb_dir)
        assert sorted(_cached(kb_dir)) == sorted([COMPILES, DOES_NOT])

        assert prune_suggestion_cache(kb_dir, {COMPILES}) == 1
        assert _cached(kb_dir) == [COMPILES]

        # The step that used to run last.
        build_suggestion_cache(kb_dir)
        assert DOES_NOT in _cached(kb_dir), (
            "this assertion documents the hazard -- if it ever fails, "
            "build_suggestion_cache has learned to respect the prune and the "
            "ordering rule below can be relaxed"
        )


def _repair_branch_statements() -> list[ast.stmt]:
    """The body of the `files_rewritten > 0` branch inside _do_build."""
    tree = ast.parse((ROOT / "admin" / "routes.py").read_text(encoding="utf-8"))
    build = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_do_build"
    )
    for node in ast.walk(build):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.dump(node.test)
        if "files_rewritten" in test_src:
            return node.body
    raise AssertionError(
        "the post-repair branch is gone from admin.routes._do_build -- if the "
        "second validation lap was removed, this test should be too"
    )


def _first_line_calling(statements: list[ast.stmt], name: str) -> int:
    """Earliest line in the branch at which ``name`` is called.

    Taken by line number, not by walk order: ast.walk is breadth-first, so the
    first node it yields is not necessarily the first one in the source.
    """
    found = [
        node.lineno
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", "") or getattr(node.func, "attr", "")) == name
    ]
    if not found:
        raise AssertionError(f"{name} is never called in the post-repair branch")
    return min(found)


def test_the_cache_rebuild_runs_before_the_validation_that_prunes_it():
    """Order is the guarantee: the prune has to be the last word, so the
    rebuild that reads the repaired files has to come first."""
    body = _repair_branch_statements()

    # The rebuild is imported under an alias, so look for the alias the branch
    # actually calls rather than the imported name.
    rebuild_line = _first_line_calling(body, "_repair_bsc")
    validation_line = _first_line_calling(body, "_run_example_validation")

    assert rebuild_line < validation_line, (
        f"the suggestion cache is rebuilt at line {rebuild_line}, after the "
        f"validation at line {validation_line} that prunes it -- every question "
        "the second validation rejected is back in the panel"
    )


def test_the_parent_can_finish_the_prune_the_worker_may_not_have_run():
    """The rebuild is safe only while the prune after it actually happens.

    That prune lives in the validation child process, which is terminated
    outright on timeout and swallows its own exceptions -- so the parent's
    "the validation prunes afterwards" is an assumption, not a fact. This is
    the same rule, applied from the parent against the same stored evidence.
    """
    from unittest.mock import patch

    from core.suggestions import prune_suggestion_cache_to_validated

    with tempfile.TemporaryDirectory() as tmp:
        kb_dir = _kb_dir(tmp)
        build_suggestion_cache(kb_dir)
        assert sorted(_cached(kb_dir)) == sorted([COMPILES, DOES_NOT])

        import store
        with patch.object(store, "get_validated_examples", return_value=[
            {"question": COMPILES, "sql_query": "SELECT 1"},
        ]) as validated:
            assert prune_suggestion_cache_to_validated(kb_dir, "acct") == 1

        assert _cached(kb_dir) == [COMPILES]
        # Read from the store, not from the files the rebuild came from --
        # rebuilding from those is what put the rejected question back.
        assert validated.call_args.args[0] == "acct"


def test_the_late_prune_keeps_everything_when_the_store_vouches_for_it():
    """It must narrow to the validated set, not empty the panel: an account
    whose examples all validated loses nothing."""
    from unittest.mock import patch

    from core.suggestions import prune_suggestion_cache_to_validated

    with tempfile.TemporaryDirectory() as tmp:
        kb_dir = _kb_dir(tmp)
        build_suggestion_cache(kb_dir)

        import store
        with patch.object(store, "get_validated_examples", return_value=[
            {"question": COMPILES}, {"question": DOES_NOT},
        ]):
            assert prune_suggestion_cache_to_validated(kb_dir, "acct") == 0
        assert sorted(_cached(kb_dir)) == sorted([COMPILES, DOES_NOT])


def test_the_late_prune_runs_when_the_second_validation_did_not_finish():
    """Wiring, read as a syntax tree: the route needs a live warehouse and an
    LLM to reach this branch, and what matters is which statuses trigger it."""
    body = _repair_branch_statements()
    guarded = [
        node for statement in body for node in ast.walk(statement)
        if isinstance(node, ast.If)
        and "prune_suggestion_cache_to_validated" in ast.dump(node)
    ]
    assert guarded, (
        "nothing prunes the cache when the second validation does not finish, "
        "so a timed-out build leaves the rebuilt, unvalidated question set in "
        "the panel"
    )
    # The constants alone are not the guarantee: `status in {...}` and
    # `status not in {...}` carry the identical constants, and the second is
    # the defect. So the comparison SHAPE is pinned too, and the set is pinned
    # by equality rather than by a lower bound.
    test = guarded[0].test
    assert isinstance(test, ast.Compare), ast.dump(test)
    assert isinstance(test.ops[0], ast.In), (
        f"the late prune is guarded by {type(test.ops[0]).__name__}, not In — "
        "inverted, it prunes on the statuses that did not need it and skips "
        "the ones that did"
    )
    assert ast.unparse(test.left) == "validation_result.get('status')", \
        ast.unparse(test.left)
    statuses = {
        node.value for node in ast.walk(test.comparators[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # A user stop terminates the worker where it stands, exactly like a
    # timeout, so it belongs with the other two.
    assert statuses == {"timeout", "error", "stopped"}, statuses
