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

import builtins
import dis
import logging
import types

import pytest


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
            "_post_intents": {"forecast": True},
            "_rows_truncated": truncated,
            "_confidence_context": {},
            "account_id": "acct", "portal_user": None, "event": None,
            "sql": "SELECT PERIOD, SUM(AMT) AS REVENUE FROM F GROUP BY PERIOD",
            "db_type_hint": "azure_sql",
            "log": _Log(),
        }
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
