"""
core/background_tasks.py

asyncio.create_task() for work nobody awaits.

The event loop keeps only a WEAK reference to a task. A task that is created
and then stored nowhere can be garbage-collected while it is still running,
which cancels it silently: no exception, no log line, the work simply stops.
The documented remedy is to hold a strong reference until the task finishes,
and this module is that, in one place.

Two call sites needed it -- the vector-store warm-up at startup and the
evaluation run that follows a semantic-contract publish -- and both were bare
create_task() calls whose result was discarded.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine

# Tasks that are running now. Each removes itself when it finishes, so this
# holds only work in flight and never grows without bound.
_RUNNING: set[asyncio.Task] = set()


def spawn(coro: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task:
    """Schedule ``coro`` on the running loop and keep it alive until it ends."""
    task = asyncio.create_task(coro, name=name)
    _RUNNING.add(task)
    task.add_done_callback(_RUNNING.discard)
    return task
