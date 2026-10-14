"""
/system/resources must not hold the event loop while it checks the path it was given.

The endpoint already runs its three probes off the loop -- they shell out to powershell
/ lscpu / nvidia-smi and take seconds. The protected-path guard in front of them looked
too cheap to need the same treatment, but it resolves a path the *caller* supplies, and
resolving one on an unreachable network mount waits out an SMB/DNS timeout of ten
seconds or more. One request could therefore stall every other one, in-flight token
streams included.
"""

import asyncio
import time

import pytest
from fastapi import HTTPException

import service.app as app_module


def test_the_protected_path_guard_does_not_stall_the_service(monkeypatch):
    """The guard is made slow on purpose; other coroutines must still get to run."""
    guard_duration = 0.3

    def slow_guard(path):
        time.sleep(guard_duration)  # what Path.resolve() does on an unreachable mount
        return True  # refused, so the endpoint answers without reaching the probes

    monkeypatch.setattr(app_module, "is_protected_system_path", slow_guard)

    async def scenario() -> tuple[int, int]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(ticker())
        try:
            with pytest.raises(HTTPException) as refusal:
                await app_module.get_system_resources(
                    mode="spec", disk_path="//nosuchserver/share", calc_size=True
                )
        finally:
            beat.cancel()
        return ticks, refusal.value.status_code

    ticks, status_code = asyncio.run(scenario())

    # Blocking the loop for the whole guard would leave the ticker at zero.
    assert ticks > 5, f"the event loop was stalled by the guard (only {ticks} ticks)"
    # The guard still has to refuse the path; running it off the loop changes when
    # the answer arrives, not what it is.
    assert status_code == 400


def test_a_path_the_guard_allows_is_not_refused(monkeypatch):
    """The wrapped call must pass the verdict through, not just the timing."""
    monkeypatch.setattr(app_module, "is_protected_system_path", lambda path: False)
    monkeypatch.setattr(app_module.system_monitor, "get_cpu_resource", lambda mode: {})
    monkeypatch.setattr(
        app_module.system_monitor, "get_gpu_resource", lambda mode: {"available": False, "gpus": []}
    )
    monkeypatch.setattr(
        app_module.system_monitor,
        "get_disk_resource",
        lambda path, calc_size, mode: {"mounts": [{"path": path, "folder_size_gb": 1.0}]},
    )

    result = asyncio.run(
        app_module.get_system_resources(mode="spec", disk_path="/data", calc_size=True)
    )

    # The walk was reached and asked about the path the caller named, rather than
    # being refused on the way in.
    assert [mount.path for mount in result.disk.mounts] == ["/data"]
