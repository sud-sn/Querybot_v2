"""
tests/test_background_tasks_are_kept_alive.py

A fire-and-forget task has to be held until it finishes.

The event loop keeps only a weak reference to a task, so a bare
asyncio.create_task(...) whose result is discarded can be garbage-collected
mid-flight and cancelled without a word. Two places did exactly that: the
vector-store warm-up at startup and the evaluation run after a semantic
contract is published. Both now go through core.background_tasks.spawn.

Found by lint rule RUF006, switched on for CI in the same commit.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import background_tasks  # noqa: E402


def test_a_spawned_task_is_held_while_it_runs_and_released_after():
    async def _scenario():
        started, release = asyncio.Event(), asyncio.Event()
        done: list[str] = []

        async def _work():
            started.set()
            await release.wait()
            done.append("finished")

        task = background_tasks.spawn(_work(), name="kept-alive")
        await started.wait()
        # Nothing but the module holds it now. Collect: it must survive.
        gc.collect()
        assert task in background_tasks._RUNNING
        assert not task.done()

        release.set()
        await task
        # done_callback runs on the next loop iteration.
        await asyncio.sleep(0)
        return task, done

    task, done = asyncio.run(_scenario())
    assert done == ["finished"]
    assert task not in background_tasks._RUNNING, "a finished task was never released"


def test_a_failing_task_is_released_too():
    """Released on every ending, not just success -- otherwise every failed
    warm-up would stay in the set for the life of the process."""
    async def _scenario():
        async def _boom():
            raise RuntimeError("warm-up failed")

        task = background_tasks.spawn(_boom())
        try:
            await task
        except RuntimeError:
            pass
        await asyncio.sleep(0)
        return task

    task = asyncio.run(_scenario())
    assert task not in background_tasks._RUNNING


def test_the_name_is_kept_for_debugging():
    async def _scenario():
        async def _noop():
            return None

        task = background_tasks.spawn(_noop(), name="vector-store-warmup")
        await task
        return task.get_name()

    assert asyncio.run(_scenario()) == "vector-store-warmup"
