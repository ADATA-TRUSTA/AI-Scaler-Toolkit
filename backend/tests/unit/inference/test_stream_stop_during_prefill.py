"""Regression: a stop must end a stream that is still waiting for its first event."""

import asyncio

from service.inference.async_server_client import _events_until_stopped


async def _slow_stream():
    await asyncio.sleep(30)  # prefill: no event for a long time
    yield "never"


async def _two_events():
    yield 1
    yield 2


async def _stop_ends_a_stream_stuck_in_prefill():
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, stop.set)
    seen = []

    async def consume():
        async for event in _events_until_stopped(_slow_stream(), stop):
            seen.append(event)

    await asyncio.wait_for(consume(), timeout=5)
    assert seen == []


async def _events_pass_through_when_not_stopped():
    stop = asyncio.Event()
    assert [e async for e in _events_until_stopped(_two_events(), stop)] == [1, 2]
    assert [e async for e in _events_until_stopped(_two_events(), None)] == [1, 2]


def test_stop_ends_a_stream_stuck_in_prefill():
    asyncio.run(_stop_ends_a_stream_stuck_in_prefill())


def test_events_pass_through_when_not_stopped():
    asyncio.run(_events_pass_through_when_not_stopped())
