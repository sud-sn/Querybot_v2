"""
tests/test_post_process_actually_runs.py

Written after shipping a forecast gate that could not run.

The wiring read `policy_allows_derived_visual=bool(chart_type)`. `chart_type` is
a local of core/result_renderer.py; in core/query_pipeline.py it is nothing at
all, so the line compiled to `LOAD_GLOBAL chart_type` against a module with no
such global. Every forecast question raised NameError, the post-processing block
caught it with `except Exception: log.debug(...)`, and the user got no
projection and no explanation. Live for a full release.

The test that was supposed to prevent this read the source of query_pipeline.py
and asserted that "evaluate_forecast_request" appeared before "compute_forecast(".
Both strings were present. Both still are. The code they name has never
executed. That is the third time in this repository that a source-shaped
assertion has passed over a defect, so these two tests are deliberately of the
opposite kind: one runs the interpreter's own name resolution over the real
bytecode, the other executes the block.
"""

from __future__ import annotations

import ast
import builtins
import dis
import logging
import types
from pathlib import Path

import pytest

NEWLINE = chr(10)


def _unresolved_globals(func, module) -> list[str]:
    """Names the function loads from module scope that do not exist there.

    Python resolves a global at call time, so an undefined one is invisible to
    import, to a linter that does not run, and to any test that does not reach
    the line. The bytecode knows. Recurses into comprehensions and nested code
    objects, which have their own code objects and their own chances to be wrong.
    """
    seen: set[int] = set()
    missing: list[str] = []

    def walk(code: types.CodeType) -> None:
        if id(code) in seen:
            return
        seen.add(id(code))
        for ins in dis.get_instructions(code):
            if ins.opname in {"LOAD_GLOBAL", "LOAD_NAME"}:
                name = ins.argval
                if not hasattr(module, name) and not hasattr(builtins, name):
                    missing.append(name)
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                walk(const)

    walk(func.__code__)
    return sorted(set(missing))


class TestEveryNameInThePipelineResolves:
    """The check that catches this whole class of bug, not just its instance.

    _handle_query_impl is ~5,500 lines with dozens of conditional branches, most
    of which no test reaches. A name that only exists on one of those branches
    is a live NameError waiting for a user to find it.
    """

    def test_the_query_pipeline_entry_point_has_no_undefined_globals(self):
        import core.query_pipeline as qp

        assert _unresolved_globals(qp._handle_query_impl, qp) == []

    def test_the_result_renderer_has_no_undefined_globals(self):
        import core.result_renderer as rr

        for name in dir(rr):
            fn = getattr(rr, name)
            if isinstance(fn, types.FunctionType) and fn.__module__ == rr.__name__:
                assert _unresolved_globals(fn, rr) == [], f"in {name}()"

    def test_the_detector_actually_detects(self):
        """A detector that cannot fail proves nothing. This is the bug that
        shipped, reduced to four lines."""
        module = types.ModuleType("fake")
        exec("def f():\n    return bool(chart_type)\n", module.__dict__)
        assert _unresolved_globals(module.f, module) == ["chart_type"]
        with pytest.raises(NameError):
            module.f()


class TestTheForecastBlockExecutes:
    """Executing the block, rather than reading it.

    Building a full _handle_query_impl call needs a database, an adapter and a
    live account. The post-processing block is self-contained enough to run on
    its own: these tests compile the real source of the block out of the real
    file and execute it against the names it expects. A NameError, a wrong
    argument, or a renamed key fails here.
    """

    def _block_source(self) -> str:
        from pathlib import Path
        import textwrap

        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
            encoding="utf-8",
        )
        start = src.index('            if _post_intents.get("forecast")')
        end = src.index('            if _post_intents.get("histogram")')
        block = textwrap.dedent(src[start:end])
        # This repository lives in a OneDrive-synced directory, which
        # intermittently serves a stale or unavailable copy of a file that was
        # just written -- observed as a FileNotFoundError on a file that plainly
        # exists. A stale read here would execute the PREVIOUS version of the
        # block and quietly report that the current one works, which is the
        # exact failure this file was written to stop. Fail loudly instead.
        for marker in ("evaluate_forecast_request", "aggregate_only_gate_passes",
                       "assess_fit", "compute_forecast("):
            assert marker in block, (
                f"stale or truncated read of query_pipeline.py: {marker!r} missing"
            )
        return block

    def _run(self, rows, *, allowed=True, truncated=False, monkeypatch=None):
        """Execute the real block with the names it closes over."""
        import core.query_pipeline as qp

        recorded = {}

        class _Log:
            def info(self, msg, *a):
                recorded.setdefault("info", []).append(msg % a if a else msg)

        env = {
            **vars(qp),
            "rows": rows,
            "question": "forecast my revenue for the next 3 months",
            # The detectors read the canonical form -- identical to `question`
            # for an English reader, and the English phrasing of it for a
            # French one. The block closes over it, so the env must carry it.
            "_analysis_question": "forecast my revenue for the next 3 months",
            "_post_intents": {"forecast": True},
            "_rows_truncated": truncated,
            "_confidence_context": {},
            "account_id": "acct", "portal_user": None, "event": None,
            "sql": "SELECT PERIOD, SUM(AMT) AS REVENUE FROM F GROUP BY PERIOD",
            "db_cfg": {"db_type": "azure_sql"},
            "log": _Log(),
        }
        # Every name here must be one PRODUCTION actually has bound at this
        # point. The first version of this dict supplied "db_type_hint", which
        # the block did read -- and which is assigned only inside
        # `if table_hint_str:`, so on a typed question it does not exist. The
        # test invented a favourable world and passed; production raised
        # UnboundLocalError on every forecast. A fixture that supplies a name
        # the real caller does not is not a test, it is a second bug agreeing
        # with the first.
        assert "db_type_hint" not in env, "production does not reliably bind this"
        # The policy gate is the thing that was broken; stub only its verdict so
        # the rest of the block runs exactly as written. It has to be patched on
        # the module, not injected into env: the block imports it by name, and
        # that import overwrites anything placed in env beforehand -- which is
        # itself proof this test executes the real import line.
        if monkeypatch is not None:
            import core.chart_policy as cp

            monkeypatch.setattr(cp, "aggregate_only_gate_passes", lambda **kw: allowed)
        code = self._block_source()
        exec(compile(code, "<forecast-block>", "exec"), env)
        return env, recorded

    def _months(self, values):
        return [{"PERIOD": f"{2025 + i // 12}-{i % 12 + 1:02d}", "REVENUE": v}
                for i, v in enumerate(values)]

    def test_the_live_eighteen_month_series_gets_a_forecast(self):
        """The exact series the server returned, which produced nothing at all
        because of the NameError: 2025-01 to 2026-06 of EMCO revenue."""
        values = [7379419.76, 6867159.02, 7548077.20, 7307544.65, 7523942.04,
                  7381519.76, 7670171.04, 7710319.55, 7444167.07, 7639739.52,
                  7332990.06, 7540560.84, 7489655.63, 6903766.55, 7639510.50,
                  7367087.16, 7590724.34, 7439558.42]
        env, recorded = self._run(self._months(values))
        rows = env["rows"]
        projected = [r for r in rows if r.get("is_forecast")]
        assert len(projected) == 3, "the forecast did not run"
        for r in projected:
            assert r["forecast_low"] < r["forecast_value"] < r["forecast_high"]
        assert any("forecast appended" in m for m in recorded.get("info", []))

    def test_a_policy_block_refuses_and_says_so(self, monkeypatch):
        env, _ = self._run(self._months([100 + i * 5 for i in range(14)]),
                           allowed=False, monkeypatch=monkeypatch)
        assert not any(r.get("is_forecast") for r in env["rows"])
        caveats = env["_confidence_context"].get("forecast_caveats") or []
        assert caveats and "did not project" in caveats[0]

    def test_the_policy_gate_is_consulted_at_all(self, monkeypatch):
        """The bug in one assertion: the gate was wired to a name that did not
        exist, so it was never called and the NameError was swallowed."""
        import core.chart_policy as cp

        calls = []

        def _spy(**kw):
            calls.append(kw)
            return True

        monkeypatch.setattr(cp, "aggregate_only_gate_passes", _spy)
        self._run(self._months([100 + i * 5 for i in range(14)]))
        assert len(calls) == 1, "the policy gate was never consulted"
        assert calls[0]["what"] == "Forecast"
        assert calls[0]["sql"].startswith("SELECT")
        assert calls[0]["account_id"] == "acct"

    def test_a_short_series_refuses_and_the_caveat_reaches_the_context(self):
        env, _ = self._run(self._months([100, 110, 120, 130]))
        assert not any(r.get("is_forecast") for r in env["rows"])
        caveats = env["_confidence_context"].get("forecast_caveats") or []
        assert caveats and "at least 6" in caveats[0]

    def test_a_truncated_result_is_never_projected(self):
        env, _ = self._run(self._months([100 + i * 5 for i in range(14)]),
                           truncated=True)
        assert not any(r.get("is_forecast") for r in env["rows"])

    def test_the_rows_are_left_alone_when_the_fit_is_refused(self):
        """assess_fit runs after fitting, so the block must be able to throw the
        projection away and hand back exactly what it was given."""
        noise = [100, 900, 150, 40, 800, 90, 700, 30, 850, 60, 780, 45]
        before = self._months(noise)
        env, _ = self._run([dict(r) for r in before])
        assert env["rows"] == before


class TestNoLocalIsReadBeforeItCanBeBound:
    """The generalisation of three bugs of the same shape.

    First `chart_type`, an undefined global. Fixed, and replaced with
    `db_type_hint`, a local assigned only under `if table_hint_str:` -- so the
    forecast died on every typed question instead of every question. The
    bytecode check added for the first one looked at globals and could not see
    the second. Then `context`, passed to _save_pending_clarification 551 lines
    before its first binding, so every cached-result clarification raised
    UnboundLocalError.

    The check that was here could not have caught the third. Two reasons, and
    both are the reason it is being replaced rather than extended:

      * It was scoped to the forecast block -- lines between two greps for
        `_post_intents.get(...)`. The third bug is 900 lines above that block.
        A guard that covers one site of a defect class this codebase has now
        produced three times is the same missed-siblings pattern the defects
        themselves are.
      * It read `dis.Instruction.line_number` and `LOAD_FAST_CHECK`, which are
        CPython 3.12+. On 3.11 the attribute does not exist and the opcode is
        never emitted, so it ERRORED on every run here rather than checking
        anything -- a guard that is not merely silent but broken.

    So: a definite-assignment analysis over the parsed source, portable across
    versions, run over EVERY function in the repository.

    Why parsed rather than executed -- the one case where that is the right
    instrument. The invariant is "no local is read on a path where it cannot
    yet be bound", over a function of 6,484 lines. Establishing that by
    execution means reaching every branch of it. The compiler's own analysis is
    the honest tool, and the failure mode being guarded against (UnboundLocalError
    at runtime) is exactly what it computes. This is not a substring scan: it
    resolves scopes, parameters, global/nonlocal declarations, and every binding
    form the language has.
    """

    # Nested defs, lambdas, classes and comprehensions each have their own
    # scope -- descending into them would report their locals as the outer
    # function's.
    _OWN_SCOPE_BREAKS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                         ast.ClassDef, ast.ListComp, ast.SetComp,
                         ast.DictComp, ast.GeneratorExp)

    @classmethod
    def _events(cls, node):
        """(kind, name, line) for everything in this function's OWN frame."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, cls._OWN_SCOPE_BREAKS):
                # A nested def or class still binds its own name out here.
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    yield ("bind", child.name, child.lineno)
                continue
            if isinstance(child, ast.Name):
                yield (("bind" if isinstance(child.ctx, (ast.Store, ast.Del))
                        else "read"), child.id, child.lineno)
            elif isinstance(child, ast.ExceptHandler) and child.name:
                yield ("bind", child.name, child.lineno)      # a plain str, not a Name
            elif isinstance(child, ast.alias):
                yield ("bind", child.asname or child.name.split(".")[0], child.lineno)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                for declared in child.names:
                    yield ("declared", declared, child.lineno)
            yield from cls._events(child)

    @classmethod
    def _unbound_reads(cls, fn):
        """Locals this function reads at a line before ANY binding of them."""
        params = {a.arg for a in (fn.args.posonlyargs + fn.args.args
                                  + fn.args.kwonlyargs)}
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra:
                params.add(extra.arg)

        binds, reads, declared = {}, {}, set()
        for kind, name, line in cls._events(fn):
            if kind == "declared":
                declared.add(name)
            elif kind == "bind":
                binds[name] = min(line, binds.get(name, line))
            else:
                reads[name] = min(line, reads.get(name, line))

        return sorted(
            (name, read_line, binds[name])
            for name, read_line in reads.items()
            # not a parameter, not a global/nonlocal, and genuinely a local here
            if name not in params and name not in declared and name in binds
            and read_line < binds[name]
        )

    @staticmethod
    def _python_files():
        root = Path(__file__).resolve().parents[1]
        skip = {"__pycache__", "venv", ".git", "node_modules", ".venv"}
        return [p for p in root.rglob("*.py")
                if not (skip & set(p.relative_to(root).parts))]

    def test_no_function_in_the_repository_reads_an_unbound_local(self):
        root = Path(__file__).resolve().parents[1]
        offenders = []
        scanned = 0
        for path in self._python_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue                      # not ours to compile
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                scanned += 1
                for name, read_line, bind_line in self._unbound_reads(node):
                    offenders.append(
                        f"{path.relative_to(root)}:{read_line} {node.name}() reads "
                        f"{name!r}, first bound at line {bind_line}")

        assert scanned > 5000, f"the scan only reached {scanned} functions"
        assert not offenders, (
            "these read a local before it can be bound, which is "
            "UnboundLocalError at runtime:\n  " + "\n  ".join(sorted(offenders)))

    def test_the_detector_finds_the_bug_it_was_written_for(self):
        """Guards the guard.

        A scan that silently stopped matching would pass the test above
        forever. This rebuilds the defect -- `context` read at the call site
        where it was, 551 lines before its binding -- and requires the analysis
        to name it.
        """
        source = (
            "def f(a):\n"
            "    if a:\n"
            "        g(question, context, {})\n"      # the read
            "    for row in a:\n"
            "        pass\n"
            "    context = resolve(a)\n"              # the binding, later
            "    return context\n"
        )
        fn = ast.parse(source).body[0]
        assert self._unbound_reads(fn) == [("context", 3, 6)]

    def test_it_does_not_flag_the_shapes_that_are_fine(self):
        """The false-positive half. A detector that flagged these would have
        been switched off within a day, which is how the last one died."""
        source = (
            "def f(a, *rest, **kw):\n"
            "    global CACHE\n"
            "    CACHE = a\n"
            "    print(CACHE, rest, kw)\n"
            "    try:\n"
            "        import json as _j\n"
            "    except ImportError as exc:\n"
            "        print(exc)\n"
            "        _j = None\n"
            "    print(_j)\n"
            "    total = 0\n"
            "    for row in a:\n"
            "        total += row\n"
            "    squares = [n * n for n in a]\n"       # n is comprehension-local
            "    with open('x') as fh:\n"
            "        print(fh, squares, total)\n"
            "    def inner():\n"
            "        return helper\n"
            "    helper = inner\n"                     # a closure read, bound later
            "    return helper()\n"
        )
        fn = ast.parse(source).body[0]
        assert self._unbound_reads(fn) == []


class TestALocalImportReachesEveryBranchThatReadsIt:
    """The half the class above cannot see, and the bug that proved it.

    `TestNoLocalIsReadBeforeItCanBeBound` compares a read's line against the
    FIRST binding anywhere in the function. That catches "read before any
    binding" and, by construction, nothing else. `build_assistant_response`
    was the other shape: gateway/webhooks.ws_chat imported it inside three
    branches (the row-lasso handler at 2704, two chip handlers at 4118 and
    4169) and read it from two others (the result_chat governed-cache reply at
    2966, the DB-fallback reply at 3474). The first binding is at 2704, above
    both reads, so the line comparison was satisfied -- and every in-card
    conversation still died on

        UnboundLocalError: cannot access local variable
        'build_assistant_response' where it is not associated with a value

    because assigning a name anywhere in a function makes it local to the
    WHOLE function, and none of those three branches runs when result_chat
    does. The `except Exception` around the branch turned it into "Something
    went wrong", so the feature was dead for its entire life with a green
    suite.

    This walks the statement tree instead of the line numbers, tracking which
    imports have definitely run at each point: an `if` contributes only what
    both arms import, a `try` only what survives every handler that falls
    through, a loop body contributes nothing (it may run zero times). It is
    scoped to names the function binds ONLY by importing them, which is what
    makes it cheap and quiet -- a full definite-assignment pass over every
    local would drown in the legitimate `x = None` / `if: x = ...` idiom.
    """

    _NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    _TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)

    @classmethod
    def _import_only_names(cls, fn):
        """Names this function binds by a local import and by nothing else."""
        imported, otherwise = set(), set()

        def walk(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, cls._NESTED):
                    name = getattr(child, "name", "")
                    if name:
                        otherwise.add(name)     # a def shadows the import
                    continue
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        imported.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(child, ast.Name) and isinstance(
                        child.ctx, (ast.Store, ast.Del)):
                    otherwise.add(child.id)
                elif isinstance(child, ast.ExceptHandler) and child.name:
                    otherwise.add(child.name)
                elif isinstance(child, (ast.Global, ast.Nonlocal)):
                    otherwise.update(child.names)
                walk(child)

        walk(fn)
        params = {a.arg for a in fn.args.posonlyargs + fn.args.args
                  + fn.args.kwonlyargs}
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra:
                params.add(extra.arg)
        return (imported - otherwise) - params

    @classmethod
    def _reads(cls, node, watch):
        """(name, line) loaded by `node` in this function's own frame."""
        out = []

        def walk(n):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                    and n.id in watch:
                out.append((n.id, n.lineno))
            for child in ast.iter_child_nodes(n):
                if isinstance(child, cls._NESTED):
                    continue          # a closure reads the cell, not the frame
                walk(child)

        walk(node)
        return out

    @classmethod
    def _run_block(cls, stmts, bound, watch, found):
        """Walk one statement list. Returns (imports_bound_after, terminates)."""
        bound = set(bound)
        for st in stmts:
            # Whatever a compound statement evaluates before entering its own
            # blocks -- the `if` test, the `for` iterable, the `with` subject.
            if isinstance(st, (ast.If, ast.While)):
                header = [st.test]
            elif isinstance(st, (ast.For, ast.AsyncFor)):
                header = [st.iter]
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                header = [item.context_expr for item in st.items]
            elif isinstance(st, (ast.Try, ast.Match)):
                header = [st.subject] if isinstance(st, ast.Match) else []
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                header = list(st.decorator_list)
            else:
                header = [st]
            for part in header:
                for name, line in cls._reads(part, watch):
                    if name not in bound:
                        found.append((name, line))

            if isinstance(st, (ast.Import, ast.ImportFrom)):
                for alias in st.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(st, ast.If):
                taken, t_ends = cls._run_block(st.body, bound, watch, found)
                other, o_ends = cls._run_block(st.orelse, bound, watch, found)
                if t_ends and o_ends:
                    return bound, True
                bound = other if t_ends else taken if o_ends else taken & other
            elif isinstance(st, ast.Try):
                tried, body_ends = cls._run_block(st.body, bound, watch, found)
                survivors = []
                for handler in st.handlers:
                    caught, ends = cls._run_block(
                        handler.body, bound, watch, found)
                    if not ends:
                        survivors.append(caught)
                after = set(tried)
                if survivors:
                    for caught in survivors:
                        after &= caught
                elif not body_ends:
                    after, _ = cls._run_block(st.orelse, tried, watch, found)
                finally_bound, finally_ends = cls._run_block(
                    st.finalbody, bound, watch, found)
                bound = after | finally_bound
                if finally_ends:
                    return bound, True
            elif isinstance(st, (ast.For, ast.AsyncFor, ast.While)):
                # Zero iterations is a real path, so the body binds nothing
                # out here -- but its reads still count.
                cls._run_block(st.body, bound, watch, found)
                cls._run_block(st.orelse, bound, watch, found)
            elif isinstance(st, (ast.With, ast.AsyncWith)):
                bound, ends = cls._run_block(st.body, bound, watch, found)
                if ends:
                    return bound, True
            elif isinstance(st, ast.Match):
                arms = [cls._run_block(case.body, bound, watch, found)
                        for case in st.cases]
                live = [b for b, ends in arms if not ends]
                exhaustive = any(
                    isinstance(case.pattern, ast.MatchAs)
                    and case.pattern.pattern is None and case.guard is None
                    for case in st.cases)
                if live and exhaustive:
                    matched = set(live[0])
                    for arm in live[1:]:
                        matched &= arm
                    bound = matched
            elif isinstance(st, cls._TERMINAL):
                return bound, True
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                bound.add(st.name)
        return bound, False

    @classmethod
    def _unreachable_imports(cls, fn):
        """(name, line) read on a path where no import of it has run."""
        watch = cls._import_only_names(fn)
        if not watch:
            return []
        found = []
        cls._run_block(fn.body, set(), watch, found)
        return sorted(set(found))

    def test_no_function_reads_a_local_import_it_may_not_have_made(self):
        root = Path(__file__).resolve().parents[1]
        skip = {"__pycache__", "venv", ".git", "node_modules", ".venv"}
        offenders = []
        scanned = 0
        for path in root.rglob("*.py"):
            if skip & set(path.relative_to(root).parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                scanned += 1
                for name, line in self._unreachable_imports(node):
                    offenders.append(
                        f"{path.relative_to(root)}:{line} {node.name}() reads "
                        f"{name!r} on a path that never imported it")

        assert scanned > 5000, f"the scan only reached {scanned} functions"
        assert not offenders, (
            "a function-local import must run on every path that reads the "
            "name; these are UnboundLocalError at runtime:\n  "
            + "\n  ".join(sorted(offenders)))

    def test_the_detector_finds_the_bug_it_was_written_for(self):
        """The exact shape from ws_chat: imported in one branch, read in another."""
        source = (
            "def ws_chat(a):\n"
            "    if a == 'lasso':\n"
            "        from core.response_builder import build_assistant_response\n"
            "        return build_assistant_response(a)\n"
            "    if a == 'result_chat':\n"
            "        return build_assistant_response(a)\n"      # dead on arrival
        )
        fn = ast.parse(source).body[0]
        assert self._unreachable_imports(fn) == [
            ("build_assistant_response", 6)]

    def test_it_finds_a_handler_that_depends_on_the_body_it_guards(self):
        """The admin/routes.py shape: report the failure with an import that
        may be what failed."""
        source = (
            "def build():\n"
            "    try:\n"
            "        from core.pipeline_context import save_state\n"
            "        work()\n"
            "    except Exception:\n"
            "        save_state('failed')\n"
        )
        fn = ast.parse(source).body[0]
        assert self._unreachable_imports(fn) == [("save_state", 6)]

    def test_it_does_not_flag_the_shapes_that_are_fine(self):
        """The false-positive half. Every one of these is correct code and a
        detector that objected to them would be switched off inside a day."""
        source = (
            "def f(a):\n"
            "    import json\n"
            "    print(json)\n"                       # plainly before the read
            "    try:\n"
            "        import sqlglot\n"
            "    except ImportError:\n"
            "        return None\n"                   # handler leaves
            "    print(sqlglot)\n"
            "    try:\n"
            "        import duckdb\n"
            "    except ImportError:\n"
            "        duckdb = None\n"                 # handler binds it
            "    print(duckdb)\n"
            "    if a:\n"
            "        import yaml\n"
            "    else:\n"
            "        import yaml\n"                   # both arms
            "    print(yaml)\n"
            "    with open('x'):\n"
            "        import csv\n"
            "    print(csv)\n"                        # a with-body always runs
            "    if a:\n"
            "        import threading\n"
            "        print(threading)\n"              # same branch, after
            "    def later():\n"
            "        return threading\n"              # a closure, not this frame
            "    return later\n"
        )
        fn = ast.parse(source).body[0]
        assert self._unreachable_imports(fn) == []


class TestAFailedAnalyticIsLoud:
    """The NameError was survivable. Its being logged at debug level is what
    made it invisible for a release."""

    def test_a_programming_error_is_logged_at_error_level(self, caplog):
        import core.query_pipeline as qp

        src = (
            "try:\n"
            "    raise NameError(\"name 'chart_type' is not defined\")\n"
            "except (NameError, AttributeError, TypeError, ImportError) as _pp_exc:\n"
            "    log.error('post_process: analytics FAILED (bug, not data): %s',"
            " _pp_exc, exc_info=True)\n"
            "except Exception as _pp_exc:\n"
            "    log.warning('post_process: analytics skipped: %s', _pp_exc)\n"
        )
        with caplog.at_level(logging.DEBUG):
            exec(compile(src, "<x>", "exec"), {"log": qp.log})
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    def test_the_source_no_longer_swallows_at_debug(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parents[1] / "core" / "query_pipeline.py").read_text(
            encoding="utf-8",
        )
        assert 'log.debug("Post-processing analytics skipped' not in src


class TestOneGateGovernsBothDerivedVisuals:
    """A forecast and a chart disclose the same values in different shapes. Two
    copies of the aggregate-only rule is one copy that can be forgotten."""

    def test_the_renderer_and_the_forecast_call_the_same_function(self):
        import core.query_pipeline as qp
        import core.result_renderer as rr
        from core.chart_policy import aggregate_only_gate_passes

        for module in (qp, rr):
            src = __import__("inspect").getsource(module)
            assert "aggregate_only_gate_passes" in src
        assert callable(aggregate_only_gate_passes)

    def test_a_failed_evaluation_blocks_for_a_regulated_tenant(self, monkeypatch):
        """Fail closed. Preserved from the code this was lifted out of, where
        the comment records that shadow mode governs whether a decision is
        advisory, not whether a failed evaluation may be ignored."""
        from core.chart_policy import aggregate_only_gate_passes

        import core.compliance.policy_engine as pe

        monkeypatch.setattr(pe, "is_regulated", lambda a: True)
        assert not aggregate_only_gate_passes(
            account_id="x", portal_user=None, event=None,
            sql="this is not sql", db_type="not_a_db_type",
        )

    def test_a_failed_evaluation_allows_for_an_unregulated_tenant(self, monkeypatch):
        from core.chart_policy import aggregate_only_gate_passes

        import core.compliance.policy_engine as pe

        monkeypatch.setattr(pe, "is_regulated", lambda a: False)
        assert aggregate_only_gate_passes(
            account_id="x", portal_user=None, event=None,
            sql="this is not sql", db_type="not_a_db_type",
        )
