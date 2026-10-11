"""An AI client is reused only on the event loop that made it.

An SDK client's pooled connections belong to the event loop that opened them. The new core makes each AI
call with asyncio.run in a worker thread, and a dashboard builds several tiles at once: a client cached from
one of those loops and used from another failed as "Azure OpenAI error: Connection error." (three calls in
thirty-two, with four tiles at once, against a real Azure deployment). One loop keeps reusing its client; a
call on a loop of its own gets a client of its own; a stand-in a test puts in the cache is used as it is.
"""

from __future__ import annotations

import asyncio
import threading

from core import llm

KEY = ("azure", "key", "https://sample.openai.azure.com", "2024-10-21", 180.0, 1)


def _made():
    made: list[object] = []

    def make():
        made.append(object())
        return made[-1]

    return made, make


def _clean():
    llm._llm_client_cache.pop(KEY, None)
    llm._llm_client_loops.pop(KEY, None)


def test_one_loop_reuses_its_client():
    _clean()
    made, make = _made()

    async def twice():
        return llm._cached_client(KEY, make), llm._cached_client(KEY, make)

    try:
        first, second = asyncio.run(twice())
        assert first is second and len(made) == 1
    finally:
        _clean()


def test_a_call_on_a_loop_of_its_own_gets_a_client_of_its_own():
    _clean()
    made, make = _made()

    async def one():
        return llm._cached_client(KEY, make)

    try:
        first, second = asyncio.run(one()), asyncio.run(one())
        assert first is not second and len(made) == 2
        seen: list[object] = []
        threads = [threading.Thread(target=lambda: seen.append(asyncio.run(one()))) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(c) for c in seen}) == 4                     # tiles built at once: never one client
    finally:
        _clean()


def test_outside_any_loop_the_client_is_kept():
    _clean()
    made, make = _made()
    try:
        assert llm._cached_client(KEY, make) is llm._cached_client(KEY, make) and len(made) == 1
    finally:
        _clean()


def test_a_stand_in_put_in_the_cache_is_used_as_it_is():
    _clean()
    made, make = _made()

    async def one():
        return llm._cached_client(KEY, make)

    try:
        asyncio.run(one())                                          # a real client was made for this key
        stand_in = object()
        llm._llm_client_cache[KEY] = stand_in
        assert asyncio.run(one()) is stand_in and len(made) == 1
    finally:
        _clean()


def test_the_azure_client_is_made_for_the_loop_it_is_used_on():
    llm._llm_client_cache.clear()
    llm._llm_client_loops.clear()

    async def client():
        return llm._get_azure_client("key", "https://sample.openai.azure.com", "2024-10-21")

    try:
        assert asyncio.run(client()) is not asyncio.run(client())

        async def twice():
            return await client(), await client()

        first, second = asyncio.run(twice())
        assert first is second
    finally:
        llm._llm_client_cache.clear()
        llm._llm_client_loops.clear()
