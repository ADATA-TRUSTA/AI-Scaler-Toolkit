"""The one place that talks to the TRUSTA-AST backend.

Why this module exists
----------------------
The planner has to know three things about the co-resident LLM: whether it is
up, how much of the card it is holding, and -- only if we intend to hand its
DRAM back -- which process is hosting it. Every one of those used to be
answered by naming an inference binary and poking its port. This pack is built
to sit next to AST, and AST hosts whichever engine it was configured with, so
any name the pack carries is wrong on some host: it read the wrong number the
moment the backend was serving something other than what the pack assumed.

AST already owns all of it and exposes it over HTTP, so we ask it there:

    GET /inference/status      is a model loaded, on what engine, holding what
    GET /system/resources      CPU/GPU/disk, vendor-probed by the backend

Degradation is deliberate and one-directional: if the backend cannot be
reached, callers assume the LLM IS resident and reserve for it. Being wrong in
that direction costs a smaller video; being wrong the other way evicts the
agent that is driving the render.

Nothing here ever calls the inference endpoints. Measuring must not perturb
what it measures, and `/inference/force_cleanup_gpu` in particular restarts the
worker -- that is exactly the eviction this pack exists to prevent.
"""
from __future__ import annotations
import json
import ntpath
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

WINDOWS = sys.platform == "win32"

TIMEOUT = float(os.environ.get("VIDGEN_AST_TIMEOUT", "4"))


def base_url() -> str:
    return (os.environ.get("VIDGEN_AST_URL") or "http://127.0.0.1:8571").rstrip("/")


def _get(path: str, timeout: float | None = None):
    """GET one AST endpoint. Returns the decoded body, or None if unreachable.

    None means "could not ask", never "the answer is no" -- callers have to
    tell those apart to pick the conservative branch.
    """
    try:
        with urllib.request.urlopen(base_url() + path,
                                    timeout=timeout or TIMEOUT) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def status() -> dict | None:
    return _get("/inference/status")


def health() -> dict | None:
    """/health, which carries the config the load actually ran with.

    Kept separate from status() because it answers a different question: status
    describes the model, health describes how the backend was told to host it --
    including, on a backend that shells out, where it pointed the thing it
    spawned. Neither endpoint is authoritative about the process itself.
    """
    return _get("/health")


def resources(mode: str = "usage") -> dict | None:
    return _get("/system/resources?mode=" + mode)


def reachable() -> bool:
    return _get("/health", timeout=2) is not None


def llm_loaded() -> bool | None:
    """True / False / None (backend not reachable)."""
    st = status()
    if st is None:
        return None
    return bool(st.get("loaded") or st.get("is_loading"))


def llm_vram_gb() -> float | None:
    """VRAM the hosted model reports holding, or None if it does not say.

    `reserved_gb` is preferred over `allocated_gb`: the allocator's reserved
    pool is what the driver actually cannot give to anyone else, which is the
    figure we must budget around. Backends that report per-device instead are
    summed over their GPU entries.

    This is the last word, not the first. How much of the card is spoken for is
    measured by the vendor probes in osmem.py; what this adds is the part they
    cannot see -- how much a model that is idle or unloaded right now will take
    back when it returns. The caller uses it only to raise the reservation,
    never to lower it, so a backend that says nothing (None) costs nothing.
    """
    st = status()
    if not st:
        return None
    mem = st.get("memory_usage") or {}

    # Flat shape: the worker reports the allocator's own totals.
    for key in ("reserved_gb", "allocated_gb"):
        val = mem.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)

    # Per-device shape: {"CUDA0": {"gb": 1.86}, "CPU_Mapped": {"gb": 17.23}}.
    # Host-mapped weights must not be counted -- they are DRAM, and adding them
    # here would inflate the VRAM reservation past the size of the card and
    # refuse every render. Anything not named for the CPU is device memory.
    total = 0.0
    for name, entry in mem.items():
        if not isinstance(entry, dict) or "cpu" in str(name).lower():
            continue
        val = entry.get("gb")
        if val is None and isinstance(entry.get("mib"), (int, float)):
            val = entry["mib"] / 1024.0
        if isinstance(val, (int, float)) and val > 0:
            total += float(val)
    return total or None


# Data files AST also reports paths for. Weights are never processes, so they
# are skipped rather than probed; anything else is offered to the caller and
# the OS decides whether it is real.
_NOT_A_PROCESS = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx",
                  ".json", ".yaml", ".yml", ".txt", ".md", ".log", ".conf")


def _walk_strings(obj, depth: int = 0):
    """Every string in the payload, two containers deep. AST decides which key
    carries the path; this pack deliberately does not know the key names."""
    if isinstance(obj, str):
        yield obj
    elif depth < 3:
        values = obj.values() if isinstance(obj, dict) else (
            obj if isinstance(obj, (list, tuple)) else ())
        for v in values:
            yield from _walk_strings(v, depth + 1)


def inference_process_candidates() -> list[str]:
    """Process names AST's status implies, best-effort, in payload order.

    The pack carries no engine name, no binary name and no field name. It asks
    AST for its status, takes every value shaped like a filesystem path, and
    hands back the basenames -- the caller then asks the OS which of them is
    actually running, and that answer is the only one trusted.

    Why it is built this way: which key names a spawned binary is AST's
    business and changes with the engine it hosts, and an engine that runs
    inside AST's own worker names no binary at all. Any name or key this pack
    hardcoded would be right on the host it was written on and silently wrong
    on the next one -- find_pid() matches nothing and the DRAM handover quietly
    does nothing. Letting AST supply the candidates and the OS confirm them
    means neither of us has to guess.

    Empty is a normal answer, and it means the handover is off for this run.
    """
    st = status()
    if not st:
        return []
    out, seen = [], set()
    for value in _walk_strings(st):
        value = value.strip()
        # ntpath handles both separators and drive letters, so a Windows path
        # is recognised even when this runs on Linux (WSL reading a
        # Windows-side backend). Requiring an ABSOLUTE path is what keeps prose
        # out: a status field reading "GPU: 53/53 layers" or a model id like
        # "org/model-name" contains a slash but is not rooted, and offering
        # those as process names would put noise in front of the operator.
        if not ntpath.isabs(value):
            continue
        base = ntpath.basename(value)
        if not base or base.lower().endswith(_NOT_A_PROCESS):
            continue
        names = [base]
        if WINDOWS and "." not in base:
            names.append(base + ".exe")    # the process table spells it out
        for name in names:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def ast_port() -> int | None:
    """The TCP port VIDGEN_AST_URL points at.

    The one thing about the backend that is known without asking it anything,
    and the reason the process side needs no engine name: whatever is listening
    here IS the backend, and whatever it spawned is somewhere below that pid in
    the process tree. This module only says where to look -- the looking needs
    the process table, which lives in osmem.
    """
    try:
        parsed = urllib.parse.urlsplit(base_url())
        if parsed.port:
            return int(parsed.port)
        return {"http": 80, "https": 443}.get(parsed.scheme)
    except ValueError:
        return None


def _walk_numbers(obj, depth: int = 0):
    """Every integer in the payload, three containers deep."""
    if isinstance(obj, bool):
        return                      # bools are ints in Python; ports are not
    if isinstance(obj, int):
        yield obj
    elif isinstance(obj, str):
        if obj.isdigit():
            yield int(obj)
    elif depth < 4:
        values = obj.values() if isinstance(obj, dict) else (
            obj if isinstance(obj, (list, tuple)) else ())
        for v in values:
            yield from _walk_numbers(v, depth + 1)


def listening_port_candidates() -> list[int]:
    """Numbers AST reports that could be a port it told something to listen on.

    Deliberately loose, and deliberately weak. The pack does not know which key
    carries a port any more than it knows which carries a path, so this takes
    every plausible integer from both endpoints -- which means `n_ctx: 64000`
    and `n_batch: 512` come along too.

    That is why it is only ever used to CONFIRM a process the tree walk already
    found, never to pick one. A port match on its own would eventually point at
    some unrelated service that happens to listen on a number in the backend's
    config, and trimming that would be a plain misfire.
    """
    out, seen = [], set()
    for payload in (status(), health()):
        if not payload:
            continue
        for n in _walk_numbers(payload):
            # Registered/dynamic range only: an engine a backend spawns does
            # not get a privileged port, and the cut drops the small integers
            # (batch sizes, thread counts, timeouts) that dominate the payload.
            if 1024 <= n <= 65535 and n not in seen:
                seen.add(n)
                out.append(n)
    return out


def describe() -> dict:
    """For doctor.py -- what the backend says, or why we could not ask."""
    st = status()
    if st is None:
        return {"url": base_url(), "reachable": False}
    return {"url": base_url(), "reachable": True,
            "loaded": bool(st.get("loaded")),
            "engine": st.get("engine"),
            "model": st.get("model_name"),
            "vram_gb": llm_vram_gb(),
            # Candidates only: confirming one needs the process table, which
            # lives in osmem, and this module stays dependency-free.
            "process_candidates": inference_process_candidates()}


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
