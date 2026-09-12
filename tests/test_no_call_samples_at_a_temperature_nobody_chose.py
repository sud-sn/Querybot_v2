# -*- coding: utf-8 -*-
"""The knowledge base's synonyms and SQL examples were sampled at 0.7.

``llm_complete``'s temperature default was 0.7 -- the OpenAI SDK's historical
value, not anything this product picked. Of the call sites that state a
temperature the highest is 0.5, most are 0.0, and every SQL path pins 0.0. So
0.7 was reachable only by FORGETTING the argument, and three places had:

* ``core/knowledge.py`` stage 1, the table document;
* ``core/knowledge.py`` stage 2, the question->SQL example pairs -- executable
  SQL, held in the retriever and read by the compiler;
* ``core/knowledge.py``'s business vocabulary -- the synonym list the
  DETERMINISTIC matching layer reads;
* and ``gateway/webhooks.py``'s result-chat SQL fallback plus its two retries,
  which generate SQL that the main pipeline generates at 0.0.

The vocabulary is the one that bites hardest, and it is not a cosmetic issue. A
knowledge base is a build artifact: rebuilding it from an unchanged schema
should produce the same thing. At 0.7 it did not, so a question that matched a
metric before a rebuild could stop matching after one, with nothing in the
schema having changed and nothing anywhere recording why. The same module had
already noticed this for its repair path and clamped that to 0.1; the build it
repairs was left at 0.7.

Two fixes, and the second is the one that keeps this from coming back:

1. ``_kb_complete`` clamps to ``_KB_TEMPERATURE``, inside the function rather
   than at its three call sites, so a stage added later cannot forget it. It is
   a ceiling: a caller may ask for less.
2. ``llm_complete``'s default is 0.0. A default only ever reached by accident
   should be the conservative one.

The test that matters is the last class: it walks every ``llm_complete`` call
site in the product with an AST and checks that each one either states a
temperature or is a forwarding closure whose caller does. That is the assertion
that fails when the NEXT call site forgets -- the specific-value tests below
would all still pass.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import pathlib
import sys
import unittest
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import core.knowledge as knowledge  # noqa: E402
import core.llm as llm  # noqa: E402
from core.knowledge import _KB_TEMPERATURE, _kb_complete, kb_doc_token_budget  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
AZURE_KWARGS = {
    "azure_endpoint": "https://emco.openai.azure.com",
    "azure_api_version": "2024-02-01",
}


class _Recorder:
    """Stands in for the provider and records the sampling it was asked for."""

    def __init__(self):
        self.temperatures: list[float] = []

    def azure(self):
        async def complete(system, user, model, api_key, max_tokens,
                           endpoint, api_version, temperature=None):
            self.temperatures.append(temperature)
            return "## Overview\nfinished\n", 100, 200
        return complete


class TestTheDefaultIsNoLongerSomethingNobodyChose(unittest.IsolatedAsyncioTestCase):

    async def test_a_call_that_states_no_temperature_gets_zero(self):
        """0.7 used to arrive here. Executed through the real llm_complete."""
        recorder = _Recorder()
        with patch.object(llm, "_azure_openai_complete", recorder.azure()):
            await llm.llm_complete(
                "sys", "user", "azure_openai", "emco-prod", "key",
                max_tokens=512, **AZURE_KWARGS)
        self.assertEqual(recorder.temperatures, [0.0])

    async def test_a_stated_temperature_is_still_what_arrives(self):
        """The default must not have become a clamp: insight.py deliberately
        asks for 0.3 and 0.5 and must keep getting them."""
        recorder = _Recorder()
        with patch.object(llm, "_azure_openai_complete", recorder.azure()):
            for wanted in (0.0, 0.2, 0.3, 0.5):
                await llm.llm_complete(
                    "sys", "user", "azure_openai", "emco-prod", "key",
                    max_tokens=512, temperature=wanted, **AZURE_KWARGS)
        self.assertEqual(recorder.temperatures, [0.0, 0.2, 0.3, 0.5])

    @pytest.mark.filterwarnings("ignore")
    async def test_every_provider_function_agrees_on_the_default(self):
        """A direct call to one of the four provider functions -- which is what
        a test or a new caller does -- must not reintroduce 0.7."""
        import inspect
        for name in ("_anthropic_complete", "_openai_complete",
                     "_local_complete", "_azure_openai_complete"):
            default = inspect.signature(
                getattr(llm, name)).parameters["temperature"].default
            self.assertEqual(default, 0.0, name)


class TestTheKnowledgeBaseIsReproducible(unittest.IsolatedAsyncioTestCase):
    """_kb_complete, executed, because the clamp lives inside it."""

    async def _temperature_for(self, **extra):
        recorder = _Recorder()
        with patch.object(llm, "_azure_openai_complete", recorder.azure()):
            await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "azure_openai",
                "emco-prod", "key", max_tokens=kb_doc_token_budget(121),
                **AZURE_KWARGS, **extra)
        return recorder.temperatures

    async def test_a_kb_completion_is_near_deterministic_by_default(self):
        self.assertEqual(await self._temperature_for(), [_KB_TEMPERATURE])
        self.assertLessEqual(_KB_TEMPERATURE, 0.1)

    async def test_a_caller_may_ask_for_less(self):
        self.assertEqual(await self._temperature_for(temperature=0.0), [0.0])

    async def test_but_not_for_more(self):
        """A ceiling, not a default. Nothing about a KB document wants variance,
        and a caller that passes one through from a config should not get it."""
        self.assertEqual(await self._temperature_for(temperature=0.9),
                         [_KB_TEMPERATURE])

    async def test_the_retry_at_the_cap_keeps_the_same_sampling(self):
        """The second call must not differ from the first in anything but its
        ceiling, or the retry is testing a different question."""
        recorder = _Recorder()

        async def truncates_once(system, user, model, api_key, max_tokens,
                                 endpoint, api_version, temperature=None):
            recorder.temperatures.append(temperature)
            if len(recorder.temperatures) == 1:
                raise llm.LLMTruncatedError("truncated", text="## partial")
            return "## Overview\nfinished\n", 100, 200

        with patch.object(llm, "_azure_openai_complete", truncates_once):
            text, reason = await _kb_complete(
                "document for CUS_ORD_IVC_FCT", "sys", "user", "azure_openai",
                "emco-prod", "key", max_tokens=kb_doc_token_budget(121),
                **AZURE_KWARGS)
        self.assertEqual(reason, "")
        self.assertEqual(recorder.temperatures,
                         [_KB_TEMPERATURE, _KB_TEMPERATURE])

    def test_the_repair_path_uses_the_same_constant(self):
        """repair_failed_query_patterns calls llm_complete directly, so it keeps
        its own clamp -- but from the same constant, because two copies of a
        sampling rule is how they come to differ."""
        source = (REPO / "core" / "knowledge.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "repair_failed_query_patterns"
        )
        # Read as a syntax tree, not as text: the assertion is that the clamp
        # names the constant rather than repeating its value, and the value is
        # what changes when someone retunes it.
        clamps = [
            ast.unparse(node.value) for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(t).endswith(("['temperature']", '["temperature"]'))
                    for t in node.targets)
        ]
        assert clamps, "the repair path no longer clamps its temperature"
        for clamp in clamps:
            assert "_KB_TEMPERATURE" in clamp, clamp


class TestTheResultChatFallbackGeneratesSqlLikeThePipeline(unittest.TestCase):
    """The webhook path is a WebSocket handler 3,500 lines in, so its call is
    read as a syntax tree -- the narrow exception for wiring that cannot be
    executed without a socket, a warehouse and a model. The temperature that
    reaches a provider IS executed, above and in the sweep below."""

    def test_each_fallback_sql_call_pins_zero(self):
        source = (REPO / "gateway" / "webhooks.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        pinned = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", ""))
            if name != "llm_complete":
                continue
            # The fallback calls are the ones handed the _fb_* provider tuple.
            if not any(ast.unparse(arg).startswith("_fb_") for arg in node.args):
                continue
            temperature = next(
                (ast.literal_eval(kw.value) for kw in node.keywords
                 if kw.arg == "temperature"), None)
            pinned.append((node.lineno, temperature))

        assert len(pinned) == 3, f"expected three fallback SQL calls, got {pinned}"
        for line, temperature in pinned:
            assert temperature == 0.0, (
                f"gateway/webhooks.py:{line} generates SQL at {temperature}; "
                "the main pipeline pins 0.0 for the same job"
            )


class TestNoCallSiteReliesOnTheDefault(unittest.TestCase):
    """The one that fails when the NEXT call site forgets.

    Every llm_complete call in the product either states a temperature or is a
    ``**kwargs`` forwarder whose caller does. Relying on the default is not
    wrong now that the default is 0.0 -- it is UNSTATED, and unstated is how
    three call sites came to sample SQL at 0.7 without anyone deciding to.
    """

    ROOTS = ("core", "gateway", "admin", "evals", "main.py", "store")

    @staticmethod
    def _carries_a_temperature(tree, call):
        """Does a ``**name`` forward in this call actually carry a temperature?

        Two forwards are legitimate and one is not, and the difference is the
        whole point of the sweep:

        * ``**kwargs`` where kwargs is the enclosing function's own ``**``
          parameter -- a closure relaying what its caller passed. Fine: the
          caller is a planner, and every planner states one.
        * ``**kw`` where the enclosing function assigns ``kw["temperature"]``
          -- the clamp in _kb_complete and in repair_failed_query_patterns.
          Fine, and deliberate.
        * ``**az_kwargs`` -- resolve_provider's Azure endpoint and api-version,
          which have never carried a temperature. NOT fine: the call samples at
          whatever the default happens to be.
        """
        forwarded = {kw.value.id for kw in call.keywords
                     if kw.arg is None and isinstance(kw.value, ast.Name)}
        if not forwarded:
            return False

        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not (function.lineno <= call.lineno <= (function.end_lineno or 0)):
                continue
            if function.args.kwarg and function.args.kwarg.arg in forwarded:
                return True
            for node in ast.walk(function):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id in forwarded
                            and ast.unparse(target).endswith(
                                ("['temperature']", '["temperature"]'))):
                        return True
        return False

    def _call_sites(self):
        files: list[pathlib.Path] = []
        for root in self.ROOTS:
            path = REPO / root
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(sorted(path.rglob("*.py")))

        for path in files:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = (node.func.attr if isinstance(node.func, ast.Attribute)
                        else getattr(node.func, "id", ""))
                # llm_complete only. A _kb_complete call is exempt BY DESIGN --
                # the clamp lives inside that function precisely so its callers
                # need not state it, and TestTheKnowledgeBaseIsReproducible
                # above proves the clamp by executing it.
                if name != "llm_complete":
                    continue
                keywords = {kw.arg for kw in node.keywords if kw.arg}
                yield (
                    path.relative_to(REPO).as_posix(),
                    node.lineno,
                    "temperature" in keywords,
                    self._carries_a_temperature(tree, node),
                )

    def test_every_call_states_its_sampling_or_forwards_it(self):
        unstated = [
            f"{path}:{line}" for path, line, states, forwards in self._call_sites()
            if not states and not forwards
        ]
        self.assertEqual(
            unstated, [],
            "these llm_complete calls leave the temperature unstated, so what "
            "they sample at is whatever the default happens to be:\n  "
            + "\n  ".join(unstated),
        )

    def test_the_sweep_actually_found_the_call_sites(self):
        """A walk that silently matches nothing is a test that cannot fail."""
        found = list(self._call_sites())
        self.assertGreater(len(found), 20, found)
        modules = {path for path, _line, _states, _forwards in found}
        for expected in ("core/query_pipeline.py", "core/knowledge.py",
                         "gateway/webhooks.py", "core/insight.py"):
            self.assertIn(expected, modules)


class TestTheClampIsAnnouncedNowhereBecauseItIsNotAFallback(unittest.TestCase):
    """A clamp that fires on every build must not log on every build."""

    def test_a_default_kb_completion_is_silent(self):
        recorder = _Recorder()
        with patch.object(llm, "_azure_openai_complete", recorder.azure()):
            with self.assertNoLogs("querybot.knowledge", level=logging.WARNING):
                asyncio.run(_kb_complete(
                    "document for CUS_ORD_IVC_FCT", "sys", "user",
                    "azure_openai", "emco-prod", "key",
                    max_tokens=kb_doc_token_budget(40), **AZURE_KWARGS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
