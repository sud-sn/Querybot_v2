"""No shipped module contains a gate that has already been decided.

This exists because of a real slip on this branch. A mutation probe --
`if False and len(_anchor_policies) == 1:` -- was left in
core/query_pipeline.py and COMMITTED. It disabled the whole
`latest_available` business-date anchor: every "latest day" question stopped
resolving its anchor from the data. The full suite passed, 9056 tests, because
no test executes that call site (`_handle_query_impl` needs a warehouse, a
socket and a model to run), and the bundled inert_code_scan.py does not model
short-circuit gates -- it looks for statements after a `return` and for locals
that stay empty, not for a condition that can never be true.

So this is the guard for the class, not a pin on that one line. A gate whose
condition is decided at parse time means one of its branches is unreachable:
either a disabled feature that still reads as live code, or a leftover probe.
Both look exactly like working code in review and in CI.

`_decided_gates` is read as a SYNTAX TREE. That is deliberate and it is the
narrow case where a source-level check is the right instrument: the question
is not what a function returns, it is whether a branch can be entered at all,
which is a property of the parse and not of any one execution. It is also the
only way to reach wiring inside `_handle_query_impl`. The classifier that
decides "already decided" is itself driven by executed tests below, over
snippets that must be caught and snippets that must not -- a detector that
quietly started answering "nothing is dead" would make this whole file pass
vacuously, which is the failure mode it is written against.

Deliberately dead code is silenced the same way the bundled scanner does it:
put `# inert-ok: <reason>` on the line of the gate.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Shipped code. tests/ is excluded: a test may legitimately build a dead gate
# as a fixture, and this file does exactly that below.
SHIPPED = ("core", "gateway", "store", "admin", "portal")

SILENCE = "inert-ok:"


def _is_falsy(node):
    """True when this expression cannot be true, whatever the inputs are."""
    if isinstance(node, ast.Constant):
        return not node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            # `False and anything()` -- short-circuits before the rest is read.
            return any(_is_falsy(value) for value in node.values)
        return all(_is_falsy(value) for value in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_truthy(node.operand)
    return False


def _is_truthy(node):
    """True when this expression cannot be false, whatever the inputs are."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.Or):
            return any(_is_truthy(value) for value in node.values)
        return all(_is_truthy(value) for value in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_falsy(node.operand)
    return False


def _decided_gates(source, name="<source>"):
    """Every branch in `source` that cannot be entered.

    Returns (line, what is dead, the condition as written).
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)) and _is_falsy(node.test):
            dead = "the body can never run"
        elif isinstance(node, ast.If) and node.orelse and _is_truthy(node.test):
            dead = "the else can never run"
        elif isinstance(node, ast.IfExp) and (_is_falsy(node.test)
                                              or _is_truthy(node.test)):
            dead = "one branch of the conditional can never be chosen"
        else:
            continue
        # `while True:` is an idiom, not a mistake -- it has no dead branch.
        if isinstance(node, ast.While) and _is_truthy(node.test):
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if SILENCE in line:
            continue
        found.append((node.lineno, dead, ast.unparse(node.test)[:80]))
    return found


def _shipped_files():
    for package in SHIPPED:
        root = REPO / package
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


class TestTheDetectorItself:
    """A guard whose detector answers "nothing is dead" passes vacuously."""

    def test_it_catches_the_probe_that_actually_shipped(self):
        found = _decided_gates(
            "def f(xs):\n"
            "    if False and len(xs) == 1:\n"
            "        return 'anchor resolved'\n"
            "    return 'no anchor'\n"
        )
        assert [(line, dead) for line, dead, _ in found] == [
            (2, "the body can never run")], found

    def test_it_catches_the_other_spellings(self):
        for snippet, why in [
            ("if False:\n    x = 1\n", "a bare disabled gate"),
            ("if 0:\n    x = 1\n", "zero"),
            ("if None:\n    x = 1\n", "None"),
            ("if '':\n    x = 1\n", "an empty string"),
            ("if []:\n    x = 1\n", "an empty list"),
            ("if ready and False:\n    x = 1\n", "false on the right"),
            ("if False or 0:\n    x = 1\n", "every branch of an or"),
            ("if not True:\n    x = 1\n", "a negated truth"),
            ("while False:\n    x = 1\n", "a loop that never turns"),
            ("if True:\n    x = 1\nelse:\n    x = 2\n", "a dead else"),
            ("x = 1 if False else 2\n", "a decided conditional expression"),
        ]:
            assert _decided_gates(snippet), f"missed {why}: {snippet!r}"

    def test_it_leaves_real_conditions_alone(self):
        for snippet, why in [
            ("if xs:\n    x = 1\n", "an ordinary truth test"),
            ("if len(xs) == 1:\n    x = 1\n", "a comparison"),
            ("if ready and len(xs) == 1:\n    x = 1\n", "a compound test"),
            ("if a or b:\n    x = 1\n", "an or of two unknowns"),
            ("if not xs:\n    x = 1\n", "a negated unknown"),
            ("while True:\n    break\n", "the while-True idiom"),
            ("if True:\n    x = 1\n", "a truthy gate with no else -- nothing dead"),
            ("x = 1 if xs else 2\n", "a real conditional expression"),
        ]:
            assert not _decided_gates(snippet), f"false alarm on {why}: {snippet!r}"

    def test_the_silencing_comment_works(self):
        assert not _decided_gates(
            "if False:  # inert-ok: kept for the 2027 migration\n    x = 1\n")

    def test_it_reads_the_real_repository(self):
        """If the file list is empty every assertion below is vacuous."""
        files = list(_shipped_files())
        assert len(files) > 100, f"only found {len(files)} shipped modules"
        assert any(p.name == "query_pipeline.py" for p in files), \
            "the pipeline is not being scanned"


class TestNoShippedModuleHasADisabledGate:

    def test_no_disabled_gates(self):
        offenders = []
        for path in _shipped_files():
            for line, dead, condition in _decided_gates(
                    path.read_text(encoding="utf-8"), path.name):
                offenders.append(
                    f"{path.relative_to(REPO)}:{line} -- {dead}: `{condition}`")
        assert not offenders, (
            "these branches are decided at parse time, so the code in them "
            "cannot run. A leftover mutation probe reads exactly like this, "
            "and it disabled the business-date anchor on this branch for a "
            "full commit. Restore the condition, delete the dead branch, or "
            "mark it `# inert-ok: <reason>` if it is deliberate:\n  "
            + "\n  ".join(offenders))


class TestTheAnchorSiteIsStillWired:
    """The specific site the slip disabled.

    `_handle_query_impl` cannot be executed in a test -- it needs a warehouse,
    a websocket and a model -- so its wiring is read from the tree. This asks
    only that the call is reachable, not how it is spelled or where it sits.
    """

    @staticmethod
    def _pipeline():
        return ast.parse(
            (REPO / "core" / "query_pipeline.py").read_text(encoding="utf-8"))

    def test_resolve_business_anchor_is_called_from_a_live_branch(self):
        tree = self._pipeline()
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_business_anchor"
        ]
        assert calls, ("core/query_pipeline.py no longer resolves a business "
                       "date anchor at all")

        # Walk down from the module, carrying whether we are inside a gate that
        # can never be entered. A call inside one is present but unreachable.
        reachable = []

        def walk(node, dead):
            for child in ast.iter_child_nodes(node):
                child_dead = dead
                if isinstance(child, ast.Call) and not dead \
                        and isinstance(child.func, ast.Name) \
                        and child.func.id == "resolve_business_anchor":
                    reachable.append(child.lineno)
                if isinstance(child, (ast.If, ast.While)):
                    body_dead = dead or _is_falsy(child.test)
                    for statement in child.body:
                        walk(statement, body_dead)
                    else_dead = dead or (isinstance(child, ast.If)
                                         and _is_truthy(child.test))
                    for statement in child.orelse:
                        walk(statement, else_dead)
                    walk(child.test, dead)
                    continue
                walk(child, child_dead)

        walk(self._pipeline(), False)
        assert reachable, (
            "every call to resolve_business_anchor in core/query_pipeline.py "
            "sits inside a branch that can never be entered, so 'latest "
            "available' date questions silently resolve no anchor")
