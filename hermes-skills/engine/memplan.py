"""Memory-aware planner for local Wan 2.2 video generation.

HARD RULE: never disturb co-running programs -- above all the local LLM the
co-resident backend is hosting. Slow (SSD paging) is acceptable; evicting or
pressuring another process is NOT. So this planner is conservative by design:

  * VRAM: we may use at most (free_vram - VRAM_MARGIN). Activations (latents)
    cannot be offloaded, so they must fit there or we refuse.
  * RAM:  we keep RAM_MARGIN free at all times. Only *pageable* weights are
    allowed to exceed RAM -- i.e. GGUF files, which are mmap'd from SSD and
    page back to disk under pressure WITHOUT committing RAM. Non-pageable fp16
    safetensors (the 5B) are therefore NEVER chosen automatically, because they
    commit ~10 GB of RAM and that is exactly what evicted the LLM once.
  * If a job cannot be made to fit the safe budget even at the floor settings,
    we REFUSE (ok=False) and say why. Refusing is the correct outcome.

Nothing here names an inference runtime. Which engine is loaded and how much of
the card it holds come from astapi.py, i.e. from the backend that owns the
model. WHICH processes may have their working set trimmed is answered by the OS
instead, from the one fact the backend cannot get wrong: whoever is listening on
the URL we already talk to is the backend, and what it spawned is below that pid
in the process tree. Measuring the LLM by naming an inference binary and poking
its port -- which is what this module used to do -- reads the wrong number the
moment the backend is hosting a different engine, and it silently reads nothing
at all when the model is remote.

Pages locked in RAM need no flag: proc_dram() already subtracts VmLck on Linux,
so an mlocking runtime yields 0 reclaimable by measurement rather than by
configuration.

Reclaim-aware budgeting (why "free RAM" is the wrong number)
-----------------------------------------------------------
The OS's "available" figure (GlobalMemoryStatusEx().ullAvailPhys on Windows,
MemAvailable on Linux) does NOT count pages sitting in another process's
resident set -- even when those pages are clean and file-backed and the kernel
would drop them for free the moment we asked for the memory. When the runtime
maps its weights instead of reading them (the default for most GGUF loaders),
most of its resident footprint is exactly that: mmap'd weights, discardable at
no cost and re-read from disk on its next token.

Measured on this box: during a normal render the OS already reclaims them --
the LLM's working set fell from 18.6 GB to 2.9 GB mid-job and was back after.
So the memory was always spendable; only the planner could not see it, and it
refused fp16 jobs the machine could actually run.

We therefore split the resident LLM into two parts, using the process counters
themselves rather than a config flag:

    reclaimable = resident - private           (clamped at 0)

read per-OS by osmem.proc_dram():

    Windows   WorkingSetSize - PrivateUsage           (PSAPI)
    Linux     VmRSS          - (RssAnon + VmLck)      (/proc/PID/status)

With mmap that is large (file-backed weights). With --no-mmap the weights are
dirty private pages, the private term exceeds the resident set, and the formula
yields 0 all by itself -- which is the correct answer, because evicting those
would mean pagefile/swap writes. No cmdline sniffing, no mode to get wrong.

Reclaimable bytes are then spendable, and the thing we must still protect is
the OS itself (VIDGEN_OS_FLOOR, default 1.5 GB) rather than the LLM's residency.
Set VIDGEN_LLM_RECLAIM=off to restore the old conservative behaviour.

WHEN the LLM's DRAM is handed back (ordering rule -- do not change this)
------------------------------------------------------------------------
Trimming the LLM's working set is safe but NOT free: the runtime stays alive
and keeps its VRAM, yet its next token has to fault ~10 GB of weights back in
from SSD. The agent driving this pipeline IS that LLM. So if we trim before ComfyUI
is actually rendering, we pay the whole re-read cost for nothing the moment the
launch fails, the planner refuses at preflight, or the user asks anything --
and the agent appears to hang while it re-reads its own weights.

Therefore the trim happens in exactly one place, `release_llm_dram()`, and that
function REFUSES unless ComfyUI is already executing a prompt. It verifies this
itself over HTTP rather than trusting its caller, so no call site can get the
order wrong. `choose()` never trims -- planning only measures.
"""
from __future__ import annotations
import json, math, os, struct, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow standalone run
import paths          # resolves hermes-comfy.conf into os.environ -- import first
import astapi         # the only thing that talks to the TRUSTA-AST backend
import osmem          # per-OS process/memory primitives
import sysmon

GB = 1024.0  # MiB per GiB

# ---- safety margins (env-tunable) ----------------------------------------
"""Protection knobs are one-way by default.

An agent that gets refused will try to argue with the refusal: one run ended
with the model issuing `export VIDGEN_LLM_VRAM_BUDGET=2` to force a job past the
guard, and the render then took the GPU to 15.8/16.3 GB with the LLM resident.
So the environment may only make these limits *stricter*. Loosening any of them
additionally requires VIDGEN_UNSAFE_OVERRIDE=1, which a model has no reason to
set and which is not documented in the skill.
"""
UNSAFE_OVERRIDE = os.environ.get("VIDGEN_UNSAFE_OVERRIDE") == "1"


def _protected(env_name: str, default: float, direction: str = "min") -> float:
    """Read a limit from the environment, refusing to weaken it.

    direction "min": the value is a floor (bigger = safer), e.g. margins.

    A rejected value says so on stderr. It used to be dropped in silence, which
    reads exactly like the setting working: a box with no co-resident LLM sets
    VIDGEN_LLM_VRAM_BUDGET=0 as the config file suggested, still loses 5.7 GB of
    a 16 GB card, and has nothing to tell it why. stdout stays clean because the
    skills parse it as JSON.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    if UNSAFE_OVERRIDE:
        return val
    kept = max(val, default) if direction == "min" else min(val, default)
    if kept != val:
        print("[memplan] %s=%g ignored: this limit only moves toward safety "
              "(keeping %g). Set VIDGEN_UNSAFE_OVERRIDE=1 to mean it."
              % (env_name, val, kept), file=sys.stderr)
    return kept


VRAM_MARGIN = _protected("VIDGEN_VRAM_MARGIN", 1.5)   # GB kept free on GPU

# A resident 30-35B class model holds ~7 GB of VRAM. Measuring free VRAM is not
# enough on its own: if the LLM happens to be unloaded at plan time we would
# happily size a job into ITS memory and then evict it the moment it comes
# back. So we always reserve this much, and only count anything above it as
# ours. Set to 0 to fall back to pure measurement (e.g. LLM permanently off).
LLM_VRAM_BUDGET = _protected("VIDGEN_LLM_VRAM_BUDGET", 7.0)


def _llm_vram_reserved_gb(free_vram_gb: float, total_vram_gb: float) -> tuple[float, str]:
    """GB of VRAM to hold back for the LLM, whether or not it is up right now.

    If the LLM is already resident its footprint is baked into `free_vram_gb`,
    so reserving again would double-count. We therefore reserve only the part of
    the budget that is NOT already accounted for by current occupancy.
    """
    if LLM_VRAM_BUDGET <= 0:
        return 0.0, "disabled (VIDGEN_LLM_VRAM_BUDGET=0)"
    budget = LLM_VRAM_BUDGET
    # If the backend reports the hosted model holding MORE than the configured
    # budget, believe it. Only upwards: a backend that under-reports (or hosts a
    # runtime that never fills the field in) must not be able to shrink the
    # reservation below the operator's floor.
    measured = astapi.llm_vram_gb()
    if measured and measured > budget:
        budget = measured
    occupied = max(0.0, total_vram_gb - free_vram_gb)   # everything non-free
    # Assume up to `budget` of what is already occupied belongs to the LLM;
    # only the shortfall still needs holding back.
    shortfall = max(0.0, budget - occupied)
    basis = "" if budget == LLM_VRAM_BUDGET else " (backend-reported)"
    if shortfall <= 0.05:
        return 0.0, ("LLM budget %.1f GB%s already covered by %.1f GB resident"
                     % (budget, basis, occupied))
    return shortfall, ("holding %.1f GB of the %.1f GB LLM budget%s (only %.1f GB currently resident)"
                       % (shortfall, budget, basis, occupied))


def _ram_margin(total_gb: float, avail_gb: float) -> float:
    """GB of RAM to keep free for the co-running LLM/OS -- ADAPTIVE by default.

    Rationale: the auto/14B path is GGUF SSD-mmap and never hard-commits RAM, so
    its real safety net is the SSD anti-lock floor below -- a big fixed margin
    there just refuses safe jobs on a RAM-tight box. So when RAM is scarce we
    shrink the protective margin; when it's plentiful we keep a generous one.
    The dangerous fp16 5B path stays safe regardless because it is gated by the
    absolute weights+overhead check (which low free-RAM fails on its own).

      * explicit override wins:   VIDGEN_RAM_MARGIN=<gb>
      * else scale to current free RAM (measured just before each run)

    There used to be a third branch here, relaxing the margin when an operator
    flag claimed the LLM's pages were mlocked. It is gone: the claim was
    unverifiable, wrong by default on most runtimes, and being wrong about it
    relaxes exactly the margin that protects the agent.
    """
    if avail_gb >= 12:
        adaptive = 6.0      # plenty free: full LLM protection
    elif avail_gb >= 6:
        adaptive = 3.0      # moderate
    else:
        adaptive = 1.0      # tight: lean on the SSD anti-lock floor for safety
    # An explicit override may raise the margin but not lower it (see _protected).
    return _protected("VIDGEN_RAM_MARGIN", adaptive)
# ---- the resident LLM's reclaimable DRAM ---------------------------------
OS_FLOOR = _protected("VIDGEN_OS_FLOOR", 1.5)                     # GB kept free for the OS
# auto (default) = count the LLM's reclaimable pages when budgeting, and hand
#                  them back for real once a render is confirmed running.
# off            = never count them and never trim (fully conservative).
# release        = kept as an alias of auto. It used to mean "trim up front",
#                  which is the ordering bug this module now refuses to commit.
RECLAIM_MODE = os.environ.get("VIDGEN_LLM_RECLAIM", "auto").lower()
COMFY_URL = paths.COMFY_URL                                       # matches video_gen


def _tree_targets() -> list[dict]:
    """The processes the backend hosts the model in, found through the process tree.

    The chain has no engine name anywhere in it:

        the backend URL -> port -> whoever is LISTENING there = the backend
        the backend's pid -> its descendants                  = what it spawned

    Recursion matters. A backend that shells out does it through a worker, so
    the engine sits two levels down; the reference box measured
    backend -> multiprocessing worker -> engine binary. A children-only walk
    would find the worker and stop one short of the process actually holding
    the weights.

    The same walk covers the in-process case for free: when the model lives
    inside the backend's own worker there is no engine child, the walk returns
    the worker, and the worker IS the right thing to trim. That case used to
    return nothing at all.

    A port the backend also mentions is used to CONFIRM a descendant, never to
    find one -- see astapi.listening_port_candidates() for why that direction
    is the only safe one.
    """
    port = astapi.ast_port()
    if not port:
        return []
    listeners = osmem.listening_pids()
    root = listeners.get(port)
    if not root:
        return []                       # backend not on this host (or this netns)
    mentioned = set(astapi.listening_port_candidates())
    serving = {pid: p for p, pid in listeners.items() if p in mentioned}

    out = []
    for pid in osmem.descendants(root):
        m = _proc_dram(pid)
        if not m:
            continue                    # exited between the walk and the read
        ws, priv = m
        out.append({"pid": pid, "name": osmem.proc_name(pid),
                    "ws_gb": round(ws, 2), "private_gb": round(priv, 2),
                    "reclaimable_gb": round(max(0.0, ws - priv), 2),
                    "port": serving.get(pid),
                    "via": "backend-child+port" if pid in serving else "backend-child"})
    # Port-confirmed first, then whatever holds the most droppable memory. Only
    # the ordering depends on the port; nothing is included or excluded by it.
    out.sort(key=lambda d: (d["port"] is None, -d["reclaimable_gb"]))
    return out


def _named_target(name: str) -> list[dict]:
    """One target from a process name -- the override and legacy-path route."""
    pid = _find_pid(name)
    if not pid:
        return []
    m = _proc_dram(pid)
    ws, priv = m if m else (0.0, 0.0)
    return [{"pid": pid, "name": name, "ws_gb": round(ws, 2),
             "private_gb": round(priv, 2),
             "reclaimable_gb": round(max(0.0, ws - priv), 2),
             "port": None, "via": "name"}]


def llm_targets() -> list[dict]:
    """Every process that may be trimmed, asked fresh rather than fixed at import.

    Three routes, in descending order of how much the OS confirmed:

      1. VIDGEN_LLM_PROCESS   -- an operator naming it outright.
      2. the process tree     -- the backend's own descendants (_tree_targets).
      3. paths in the payload -- the original route, kept because it is the one
         that still works when the backend is on another host or another OS
         (a WSL split install): there is no process tree to walk from here, but
         a reported binary name can still be matched against this OS.

    Empty is a normal answer and means the handover is off for this run.
    """
    override = os.environ.get("VIDGEN_LLM_PROCESS", "").strip()
    if override:
        # Returned WITHOUT confirming it is running, on purpose: an operator who
        # names a process that is not up should be told so (see llm_dram), not
        # silently ignored.
        return _named_target(override) or [
            {"pid": None, "name": override, "ws_gb": 0.0, "private_gb": 0.0,
             "reclaimable_gb": 0.0, "port": None, "via": "name"}]
    found = _tree_targets()
    if found:
        return found
    for name in astapi.inference_process_candidates():
        got = _named_target(name)
        if got:
            return got
    return []


def llm_process() -> str | None:
    """The name of the primary target, for messages and for doctor.py.

    A name is no longer how a process is FOUND -- llm_targets() resolves pids --
    because the tree walk routinely lands on a python.exe, and matching that by
    name would pick whichever python the process table happened to list first.
    """
    override = os.environ.get("VIDGEN_LLM_PROCESS", "").strip()
    if override:
        return override
    t = llm_targets()
    return (t[0].get("name") or None) if t else None

# Process/memory primitives are per-OS; osmem holds both implementations and
# documents why the (working_set - private) formula is meaningful on each.
# Keeping the old private names means the rest of this module is unchanged.
_find_pid = osmem.find_pid
_proc_dram = osmem.proc_dram


def _release_dram(pid: int):
    """Hand the LLM's resident pages back now instead of waiting for pressure.

    DRAM only -- the process keeps running and its VRAM allocation is untouched.
    This is a working-set trim, NOT a shutdown. Returns (ok, reason); the reason
    matters because on Linux a refusal is routine and harmless (the kernel
    reclaims those clean pages under pressure by itself). Internal: go through
    release_llm_dram(), which enforces the ordering rule.
    """
    return osmem.release_dram(pid)


_noted: set[str] = set()


def _note(msg: str) -> None:
    """Say it once, on stderr: stdout carries the JSON the skill parses."""
    if msg not in _noted:
        _noted.add(msg)
        print("[memplan]", msg, file=sys.stderr)


def llm_dram() -> dict:
    """How much DRAM the co-resident LLM holds, and how much of it is free to
    drop (clean file-backed pages). See the module docstring for the formula."""
    if RECLAIM_MODE == "off":
        return {"mode": "off", "reclaimable_gb": 0.0}
    targets = llm_targets()
    if not targets:
        # Nothing resolved. Correct only when the backend is not on this host;
        # otherwise it means the walk found the backend but nothing under it,
        # and the handover stays off until that changes.
        _note("nothing under the backend at %s resolves to a running process, "
              "so the DRAM handover is off this run (the VRAM guard is "
              "unaffected). Set VIDGEN_LLM_PROCESS to name the process "
              "directly if the backend is not on this host." % astapi.base_url())
        return {"mode": RECLAIM_MODE, "found": False, "reclaimable_gb": 0.0,
                "why": "no process under the backend to trim"}
    live = [d for d in targets if d.get("pid")]
    if not live:
        # Naming a process that is not running is the one failure here that
        # looks like success: budgeting counts 0 reclaimable and plans
        # conservatively, which on a tight host refuses renders that would
        # have fit. Say so rather than let it pass as "nothing to reclaim".
        proc = targets[0].get("name")
        _note("no process named %r is running, so the DRAM handover will do "
              "nothing this run (the VRAM guard is unaffected). Unset "
              "VIDGEN_LLM_PROCESS to let the backend's process tree answer "
              "instead." % proc)
        return {"mode": RECLAIM_MODE, "found": False, "proc": proc,
                "reclaimable_gb": 0.0}
    # Summed, not primary-only: a backend that spawns its engine leaves DRAM in
    # both the worker and the engine, and the budget is about the total the
    # render can expect back. `pid`/`proc`/`ws_gb` stay scalar and describe the
    # primary target, so every existing reader keeps working.
    head = live[0]
    return {"mode": RECLAIM_MODE, "found": True,
            "pid": head["pid"], "proc": head.get("name"), "via": head.get("via"),
            "ws_gb": round(sum(d.get("ws_gb") or 0.0 for d in live), 1),
            "private_gb": round(sum(d.get("private_gb") or 0.0 for d in live), 1),
            "reclaimable_gb": round(sum(d["reclaimable_gb"] for d in live), 1),
            "pids": [d["pid"] for d in live],
            "targets": live}


def comfy_is_rendering() -> bool:
    """True only if ComfyUI is up AND has a prompt executing right now.

    queue_running -- not just "the server answers". A server that booted but is
    sitting idle is exactly the case where trimming the LLM would be pure loss.
    """
    try:
        with urllib.request.urlopen(COMFY_URL + "/queue", timeout=5) as r:
            q = json.loads(r.read().decode())
    except Exception:
        return False
    return bool(q.get("queue_running"))


def release_llm_dram(reason: str = "") -> dict:
    """Give the resident LLM's DRAM back to the render. THE ONLY TRIM CALL SITE.

    Refuses unless ComfyUI is already executing a prompt, and checks that
    itself rather than trusting the caller -- see the ordering rule in the
    module docstring. Never raises: a failed trim must not fail a render.

    Returns a dict that is safe to log and to show a user; `released` is the
    only field a caller should branch on.
    """
    out = {"released": False, "reason": reason, "mode": RECLAIM_MODE}
    if RECLAIM_MODE == "off":
        out["why"] = "VIDGEN_LLM_RECLAIM=off -- trimming disabled by operator"
        return out
    if not comfy_is_rendering():
        out["why"] = ("refused: ComfyUI is not executing a prompt yet. The LLM's "
                      "pages are handed back only once the render is confirmed "
                      "running, so a failed launch never costs the agent its weights.")
        return out
    before = llm_dram()
    pid = before.get("pid")
    if not pid:
        out["why"] = before.get("why") or (
            "%s is not running -- nothing to trim" % (before.get("proc")
                                                      or "the hosted runtime"))
        return out
    if before.get("reclaimable_gb", 0.0) <= 0.5:
        out["why"] = ("only %.1f GB is clean/file-backed (mmap); trimming would "
                      "cost pagefile writes for no real gain"
                      % before.get("reclaimable_gb", 0.0))
        out["llm_dram"] = before
        return out
    # Trim every process the walk found, not just the largest. On a backend that
    # spawns its engine the worker and the engine each hold part of the total,
    # and stopping after the first leaves the rest resident.
    fallback = [{"pid": pid, "name": before.get("proc"),
                 "reclaimable_gb": before.get("reclaimable_gb", 0.0)}]
    trims, any_ok = [], False
    for target in (before.get("targets") or fallback):
        if not target.get("pid") or (target.get("reclaimable_gb") or 0.0) < 0.1:
            continue          # nothing droppable there; the syscall would be noise
        try:
            ok, why = _release_dram(target["pid"])
            ok = bool(ok)
        except Exception as e:                  # never let this break a render
            ok, why = False, "working-set trim failed: %r" % (e,)
        any_ok = any_ok or ok
        trims.append({"pid": target["pid"], "name": target.get("name"),
                      "ok": ok, "why": why})
    out["trims"] = trims
    if not any_ok:
        out["why"] = "; ".join("%s: %s" % (r["name"] or r["pid"], r["why"])
                               for r in trims) or (
            "no target held enough droppable memory to be worth a trim")
        out["llm_dram"] = before
        return out
    time.sleep(2)                               # let the pages land on the free list
    after = llm_dram()
    out.update(released=True, pid=pid,
               ws_before_gb=before.get("ws_gb"), ws_after_gb=after.get("ws_gb"),
               freed_gb=round(max(0.0, (before.get("ws_gb") or 0.0)
                                  - (after.get("ws_gb") or 0.0)), 1),
               note=("DRAM working set only: the model is still loaded and its "
                     "VRAM is untouched. Its next inference re-reads weights from "
                     "SSD, so do not prompt it until the render finishes."))
    return out


RAM_OVERHEAD = 2.5   # GB torch/python context -- only for fp16 (cold, full commit)
# For SSD-backed GGUF with ComfyUI ALREADY running, torch context is already
# resident; a new render's *incremental* hard-RAM is just working buffers. Extra
# pressure pages our own weight mmap back to SSD (slow), not the LLM.
SSD_INCR_RAM = _protected("VIDGEN_SSD_INCR_RAM", 1.0)

# ---- model catalogue (what is on disk) -----------------------------------
MODELS = {
    "14b": {  # GGUF -> SSD-mmap, the safe default under memory pressure
        "label": "Wan 2.2 I2V-A14B (GGUF Q5_K_M, dual-expert)",
        "template": "wan_i2v_14b_gguf.json",
        "vae": "wan_2.1_vae.safetensors",
        "ssd_backed": True, "dual_expert": True, "shift": 5.0,
    },
    "5b": {   # fp16 -> commits ~10 GB RAM, opt-in ONLY (never auto)
        "label": "Wan 2.2 TI2V-5B (fp16, RAM-committing)",
        "template": "wan_ti2v_5b.json",
        "vae": "wan2.2_vae.safetensors",
        "weights_ram_gb": 10.0,
        "ssd_backed": False, "dual_expert": False, "shift": 8.0,
    },
    "h3": {   # MiniMax-H3: the only model here that also generates AUDIO
        "label": "MiniMax-H3 (pruned int8_convrot, native stereo audio)",
        "template": "minimax_h3_r2v.json",          # ref2va (reference images)
        "template_chain": "minimax_h3_fl2va.json",  # fl2va (first_frame chaining)
        # ref2va weights + both conditionings at once. Needs the
        # hermes_minimax_h3_refkf plugin in ComfyUI/custom_nodes.
        "template_refkf": "minimax_h3_ref2va_fl.json",
        "vae": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        # int8 convrot safetensors stream straight from disk under DynamicVRAM,
        # exactly like the GGUF path: pages, never hard-commits RAM.
        "ssd_backed": True, "dual_expert": False,
        "shift_video": 12.0, "shift_audio": 3.0,
        "fps": 24, "audio": True, "grid": 17,       # length must be 17k+5
        "dim_multiple": 32,
        "native_short": 768,                        # trained at 1344x768
    },
}
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

# ---- GPUs that share system RAM (integrated graphics) -----------------------
# An integrated GPU has no memory of its own: what it reports as VRAM is system
# RAM. The budgets below were built for a discrete card, where VRAM and RAM are
# two pools, so on an Intel iGPU they were counted twice -- 64 GB of "VRAM" plus
# 79 GB of RAM planned against a 127 GB machine whose LLM keeps its MoE experts in
# that same RAM. "SSD-mmap, never commits RAM" stops being true there too: the
# pages land in GPU memory, which is RAM.
#
# What each tier keeps resident on the device while sampling, worst case: the
# diffusion weights (both experts of the dual-expert 14B) and its text encoder.
# Read off the files on disk and ComfyUI's own "loaded completely" lines.
DEVICE_RESIDENT_GB = {"14b": 20.2 + 6.3, "5b": 9.5 + 6.3, "h3": 19.5 + 14.6}

def _backend_says_integrated() -> bool | None:
    """The TRUSTA-AST backend's is_integrated for the card being measured, if known.

    Asked rather than inferred from sizes, because sizes do not tell: Windows lends
    an iGPU about half of system RAM -- 73 of 127 GB here -- which is also what a
    24 GB card in a 32 GB machine looks like.
    """
    vendor = getattr(paths, "GPU_VENDOR", "") or ""
    if not vendor:
        return None
    try:
        with urllib.request.urlopen(paths.AST_URL + "/system/resources?mode=spec",
                                    timeout=6) as r:
            gpus = (json.loads(r.read().decode()).get("gpu") or {}).get("gpus") or []
    except Exception:
        return None
    same = [g for g in gpus if str(g.get("vendor", "")).lower() == vendor]
    if not same:
        return None
    index = getattr(paths, "GPU_INDEX", "") or ""
    g = same[int(index)] if index.isdigit() and int(index) < len(same) else same[0]
    flag = g.get("is_integrated")
    return bool(flag) if flag is not None else None


def shares_system_ram() -> tuple[bool, str]:
    """Whether the GPU ComfyUI runs on is drawing from system RAM, and why we think so.

    VIDGEN_GPU_SHARED=1/0 decides outright; otherwise the backend's own account of
    the card. Unknown is treated as discrete, which is what this module assumed
    before it knew integrated GPUs existed.
    """
    raw = os.environ.get("VIDGEN_GPU_SHARED", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True, "VIDGEN_GPU_SHARED=1"
    if raw in ("0", "false", "no"):
        return False, ""
    if _backend_says_integrated():
        return True, "the backend reports this %s GPU as integrated" % paths.GPU_VENDOR
    return False, ""


def speed_note() -> str | None:
    """Said whenever ComfyUI is not on an NVIDIA card, where no timing here holds.

    Every time figure in this module -- the H3 curve, the segment box -- was fitted on
    NVIDIA renders. Measured on an Intel iGPU instead: Wan 5B at 848x480 took ~155 s
    per step, so a two-second clip is well over an hour, and its text encoder alone
    took eight minutes to load. An agent told nothing waits in the foreground, times
    out, and resubmits.
    """
    vendor = getattr(paths, "GPU_VENDOR", "") or ""
    if not vendor or vendor == "nvidia":
        return None
    name = {"intel": "an Intel", "amd": "an AMD"}.get(vendor, "a %s" % vendor)
    return ("ComfyUI is running on %s GPU, and every time estimate here was measured "
            "on NVIDIA cards. Expect far slower: on an Intel iGPU, Wan 5B at 480p took "
            "~155 s per step (over an hour for 2 s of video). Tell the user it will be "
            "slow, submit once, and collect the result with `status` instead of "
            "waiting in the foreground." % name)


def _with_speed_note(plan: dict) -> dict:
    note = speed_note()
    if note and plan.get("ok"):
        plan["warning"] = (plan.get("warning") + " | " + note) if plan.get("warning") else note
    return plan

# ---- H3 sizing model (calibrated on this box, RTX 5080 Laptop 16 GB) -------
# Latent grid tokens: (W/16)*(H/16)*ceil(frames/4).
#
# MEMORY is ~linear in tokens (SDPA/flash attention keeps no NxN matrix), which
# is why 1344x768x362 did NOT OOM -- it ran, just extremely slowly. The 3.0e-5
# figure was VERIFIED at the edge on 2026-08-13: five single shots planned at
# 7.2-7.4 GB against a 7.4 GB budget all completed, none OOMed.
#
# TIME, calibrated 2026-08-13 on 19 measured single-shot renders spanning
# 22k-246k tokens (see output/minimax-h3-bench/results*.csv). The old
# 4.09e-7 x t^1.87 was fitted on two points and overestimated by 77-83% at the
# top of the range, which is what cut single segments to ~5 s at 768p.
#
# The real curve is not one power law: the exponent is ~1.33 below 130k tokens
# and ~1.70 above it (five resolutions agree to within 0.05). A single fit over
# the whole range gives exp 1.467. We deliberately run 1.4 instead, which is
# slightly optimistic at the very top (predicts ~18 min where the truth is
# ~45 min) but keeps the 1500 s box ABOVE the VRAM ceiling -- so activation,
# not a wall-clock guess, decides how long a single shot can be. That is safe
# because VRAM caps a segment at ~246k tokens either way, so the worst case a
# too-cheap estimate can buy is a ~45 minute render, not a runaway.
H3_ACT_GB_PER_TOKEN = float(os.environ.get("VIDGEN_H3_ACT_PER_TOKEN", "3.0e-5"))
H3_TIME_C = float(os.environ.get("VIDGEN_H3_TIME_C", "3.101e-5"))
H3_TIME_EXP = float(os.environ.get("VIDGEN_H3_TIME_EXP", "1.4"))
# Keep any single submission under this wall time. Long single shots are what
# preceded the nvlddmkm driver resets on this box; splitting also costs ~64%
# less total time because of the quadratic above.
H3_MAX_SEGMENT_S = float(os.environ.get("VIDGEN_H3_MAX_SEGMENT_S", "1500"))
H3_MIN_SEGMENT = 39          # 17*2+5: smallest segment worth chaining
# EasyCache (ComfyUI core node) reuses the DiT output on steps where it barely
# changes. Measured 2026-10-28 on a 73-frame 704x1344 ref2va chunk, 8 steps:
# 305 -> 242 s wall (-21%) at SSIM 0.955 against the uncached render. 0.3
# skipped the same steps as 0.2; LazyCache was as fast but drifted (SSIM
# 0.889, a T-shirt print changed). 0 disables it.
H3_EASYCACHE = float(os.environ.get("VIDGEN_H3_EASYCACHE", "0.2"))
# Wall-time factor applied to the REPORTED estimate only. Chunk sizing keeps
# the uncached time, so a clip on which the cache skips nothing still fits
# the H3_MAX_SEGMENT_S box.
H3_EASYCACHE_TIME = 0.8


def h3_easycache(req: dict) -> float:
    """reuse_threshold for this request: --easycache / --no-easycache, else default."""
    v = req.get("easycache")
    return H3_EASYCACHE if v is None else max(0.0, float(v))


def _easycache_label(ec: float) -> str:
    return " | EasyCache %.2f" % ec if ec > 0 else " | no cache"
# H3's trained canvas area (1344x768). Any aspect is allowed, but the area is
# what the model was trained at, so a fitted canvas is capped here.
H3_NATIVE_PIXELS = 1344 * 768


def _h3_step_down(frames: int) -> int:
    """Largest 17k+5 grid length strictly below `frames`.

    The previous descent did `snap(seg-17) - 17` then snapped again, which from
    an on-grid value skips TWO steps (34 frames) -- so the search could walk
    straight past the longest length that actually fits and report a shorter
    one. At 1344x768 that cost 243 frames -> 226, i.e. 0.7 s of clip that the
    VRAM budget could have held.
    """
    k = (frames - 6) // 17
    return max(H3_MIN_SEGMENT, 17 * k + 5) if k >= 0 else H3_MIN_SEGMENT


def _h3_fit_segment(width: int, height: int, want: int, vram_budget: float,
                    ref_tokens: int = 0):
    """Largest grid-legal segment length that fits BOTH the activation budget
    and the wall-clock box. None if even H3_MIN_SEGMENT does not fit VRAM.

    ref_tokens is charged to every segment: segment 0 carries the reference
    images and chained segments carry a keyframe, so neither is free.
    """
    seg = h3_snap_length(want)
    while seg > H3_MIN_SEGMENT:
        if (h3_activation_gb(width, height, seg, ref_tokens) <= vram_budget
                and h3_time_estimate_s(width, height, seg, ref_tokens) <= H3_MAX_SEGMENT_S):
            return seg
        seg = _h3_step_down(seg)
        if seg == H3_MIN_SEGMENT:
            break
    if h3_activation_gb(width, height, H3_MIN_SEGMENT, ref_tokens) <= vram_budget:
        return H3_MIN_SEGMENT
    return None


def h3_tokens(width: int, height: int, length: int) -> int:
    return int((width // 16) * (height // 16) * ((length + 3) // 4))


# ---- reference-image tokens ------------------------------------------------
# MiniMaxH3ReferenceToVideo re-injects the reference condition latents at EVERY
# sampling step (they are never denoised), so they sit in the attention sequence
# exactly like video tokens and cost activation memory on the same linear curve.
# Until 08-14 this planner counted only the video grid. That was harmless under
# the node's default ref_image_size="match" -- a ref is scaled DOWN to the
# generation's own pixel area, so it can never exceed ~4k tokens -- but it makes
# "max" unsafe: a 3213x5712 phone photo becomes 2048x3648 = 29,184 tokens, more
# than half the video grid of a 5 s 480p clip, and the budget would not see it.
H3_REF_SHORT_EDGE = 2048          # mirrors REF_IMAGE_SHORT_EDGE in the node
H3_REF_MULTIPLE = 32              # mirrors CANVAS_MULTIPLE


def _image_dims(path: str):
    """(width, height) from the file header, stdlib only. None if unreadable.

    Covers the formats _stage_image can hand to LoadImage. HEIC/AVIF are not
    parsed here (they need a real decoder); callers treat None conservatively.
    """
    try:
        with open(os.path.expanduser(path), "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"BM":
                return struct.unpack("<ii", head[18:26])
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return struct.unpack("<HH", head[6:10])
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                f.seek(12)
                chunk = f.read(8)
                if chunk[:4] == b"VP8X":
                    b = f.read(10)
                    return (1 + int.from_bytes(b[4:7], "little"),
                            1 + int.from_bytes(b[7:10], "little"))
                if chunk[:4] == b"VP8 ":
                    b = f.read(10)
                    return (int.from_bytes(b[6:8], "little") & 0x3FFF,
                            int.from_bytes(b[8:10], "little") & 0x3FFF)
                return None
            if head[:2] == b"\xff\xd8":                      # JPEG: walk to SOFn
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":                 # resync on padding
                        b = f.read(1)
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        return None
                    m = marker[0]
                    if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                        f.read(3)                             # length + precision
                        h, w = struct.unpack(">HH", f.read(4))
                        return (w, h)
                    size = struct.unpack(">H", f.read(2))[0]
                    f.seek(size - 2, 1)
    except Exception:
        return None
    return None


def h3_ref_grid(w: int, h: int, gen_w: int, gen_h: int, mode: str):
    """Reproduce the node's per-reference sizing (aspect-preserving, /32 snap)."""
    if mode == "max":
        s = min(1.0, H3_REF_SHORT_EDGE / float(min(w, h)))
    else:
        s = min(1.0, math.sqrt((gen_w * gen_h) / float(w * h)))
    return (max(H3_REF_MULTIPLE, round(w * s / H3_REF_MULTIPLE) * H3_REF_MULTIPLE),
            max(H3_REF_MULTIPLE, round(h * s / H3_REF_MULTIPLE) * H3_REF_MULTIPLE))


def h3_ref_baseline_tokens(gen_w: int, gen_h: int) -> int:
    """Reference tokens already priced into H3_ACT_GB_PER_TOKEN.

    Every run in the 08-12/08-13 benchmark that calibrated the 3.0e-5 constant
    and the ~246k-token VRAM ceiling was an r2v with ONE reference image at the
    default "match", i.e. a ref scaled to the canvas area. That cost is baked
    into the constant, so charging it again would double-count and shrink limits
    we have actually measured (it cost 768p its verified 243-frame single shot).
    Only the EXCESS over one canvas-sized reference is billed.
    """
    return (gen_w // 16) * (gen_h // 16)


def h3_route(req: dict) -> str:
    """Which of H3's graphs this request needs. The single source of truth.

      fl2va      i2v: --image is frame 0; no reference images exist on that graph
      ref2va     r2v: the --ref images are the cast and H3 invents the framing
      ref2va+kf  r2v with --image: both at once, via the custom node in
                 ComfyUI/custom_nodes/hermes_minimax_h3_refkf

    The planner prices whatever this returns and the builder loads it, so they
    cannot drift apart -- and `--image` means exactly one thing per route
    instead of being guessed at twice.
    """
    if req.get("mode") == "i2v":
        return "fl2va"
    if (req.get("refs") or []) and req.get("image"):
        return "ref2va+kf"
    return "ref2va"


def h3_keyframes(req: dict) -> int:
    """How many stretched-to-canvas keyframes the chosen graph will carry."""
    if h3_route(req) == "ref2va":
        return 0
    return (1 if req.get("image") else 0) + (1 if req.get("last_frame") else 0)


def h3_refs(req: dict) -> list:
    """Reference-image paths, once the route has settled what --image means."""
    route = h3_route(req)
    if route == "fl2va":
        return []
    if route == "ref2va+kf":
        return list(req.get("refs") or [])
    # plain ref2va: a lone --image is the legacy spelling of --ref
    return list(req.get("refs") or ([req["image"]] if req.get("image") else []))


def h3_ref_tokens(req: dict, gen_w: int, gen_h: int) -> int:
    """Raw latent tokens contributed by the conditioning images.

    Keyframes and references are priced by different rules and a request can
    carry both. A keyframe is STRETCHED onto the canvas (the node resizes it
    with upscale_method "disabled"), so it always costs exactly one canvas grid
    and `ref_image_size` never touches it; pricing one as a reference would
    invent tens of thousands of tokens the graph never allocates -- a 3213x5712
    phone photo at "max" bills ~29k -- and shrink or refuse a clip that fits.

    Unreadable headers (HEIC, and anything exotic) fall back to the worst case
    the mode allows, so an unknown file can only ever make us more conservative.
    """
    if req.get("swaps"):
        # one render per --swap, each carrying only that character's pictures:
        # the most expensive pass is what has to fit
        return max(h3_ref_tokens(dict(req, refs=s.get("refs") or [], swaps=None), gen_w, gen_h)
                   for s in req["swaps"])
    total = (gen_w // 16) * (gen_h // 16) * h3_keyframes(req)
    mode = "max" if str(req.get("ref_image_size", "match")).lower() == "max" else "match"
    refs = h3_refs(req)
    for path in refs:
        dims = _image_dims(str(path))
        if dims is None or dims[0] <= 0 or dims[1] <= 0:
            # worst case: "match" is capped by the generation area; "max" by a
            # 2048 square, which is the largest grid any single ref can produce.
            tw, th = ((H3_REF_SHORT_EDGE, H3_REF_SHORT_EDGE) if mode == "max"
                      else (gen_w, gen_h))
        else:
            tw, th = h3_ref_grid(dims[0], dims[1], gen_w, gen_h, mode)
        total += (tw // 16) * (th // 16)
    return total


def h3_ref_billable(req: dict, gen_w: int, gen_h: int) -> int:
    """Reference tokens to charge on top of the video grid (see baseline note)."""
    return max(0, h3_ref_tokens(req, gen_w, gen_h)
               - h3_ref_baseline_tokens(gen_w, gen_h))


# ---- reference VIDEO (--ref-video) -----------------------------------------
# MiniMaxH3ReferenceToVideo also takes reference videos (<Video k>). Their
# latents ride through every step exactly like reference images, and a video is
# a stack of them: 5 s at the node's own 768-short canvas is ~149k tokens, more
# than the clip being generated. So the engine pre-scales every reference video
# (h3_media.refvideo, default short edge 384), and this prices what that
# produces. Measured 2026-10-27 at 864x480x124, 8 steps: 110 s with pictures
# only, 207-229 s with a 384-short reference video -- the 1.4 power law below
# predicts 2.0x for those token counts.
#
# A recast is rendered SHOT BY SHOT. Every chunk gets its own slice of the
# reference video, so unlike fl2va chaining nothing is seeded from a drifted
# frame -- each chunk is anchored on the original exactly like a single shot.
# That is what makes 720p x 14 s possible at all (as one shot it needs ~400k
# tokens; the card holds ~246k), and because time is superlinear in tokens,
# several short chunks also cost less in total than one long one. Chunks start
# at the reference's own cuts, so the seams land where the edit already is.
H3_REF_VIDEO_SHORT = 384
H3_REF_VIDEO_MAX_S = 60.0         # whole recast; every chunk is far shorter
H3_REF_VIDEO_MIN_S = 1.0
H3_REF_CHUNK_MAX = 345            # 17k+5 frames, 14.4 s: top of H3's trained range


def h3_snap_down(frames: int) -> int:
    """Largest 17k+5 length <= frames -- what the node keeps of a reference video."""
    frames = int(frames)
    return 5 if frames < 22 else frames - ((frames - 5) % 17)


def h3_video_latent_t(frames: int) -> int:
    """Latent frames the H3 video VAE makes of `frames` pixel frames (node's own rule)."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def h3_ref_video_dims(w: int, h: int, short: int) -> tuple[int, int]:
    """Size h3_media.refvideo encodes at: short edge -> `short`, /32 grid, never up."""
    s = min(1.0, short / float(min(w, h)))
    return (max(H3_REF_MULTIPLE, int(round(w * s / float(H3_REF_MULTIPLE))) * H3_REF_MULTIPLE),
            max(H3_REF_MULTIPLE, int(round(h * s / float(H3_REF_MULTIPLE))) * H3_REF_MULTIPLE))


def h3_ref_video_seconds(req: dict) -> float | None:
    """Usable reference seconds after --ref-video-start."""
    meta = req.get("ref_video_meta") or {}
    dur = meta.get("duration")
    if dur is None:
        return None
    return max(0.0, min(H3_REF_VIDEO_MAX_S, float(dur) - float(req.get("ref_video_start") or 0)))


def h3_ref_video_grid(req: dict, gen_w: int, gen_h: int) -> int:
    """Tokens per LATENT frame of the prepared reference video (canvas shape if unprobed)."""
    short = int(req.get("ref_video_size") or H3_REF_VIDEO_SHORT)
    meta = req.get("ref_video_meta") or {}
    tw, th = h3_ref_video_dims(meta.get("w") or gen_w, meta.get("h") or gen_h, short)
    return (tw // 16) * (th // 16)


def h3_ref_video_chunks(req: dict, total: int, max_len: int) -> list:
    """Split `total` frames of reference into chunks of <= max_len, cutting at its shots.

    Shots shorter than H3_MIN_SEGMENT ride along with the previous chunk when
    that still fits (a cut INSIDE a chunk is fine: H3 copies the edit), and a
    shot longer than max_len is split into equal parts. Frames are relative to
    --ref-video-start.
    """
    start = float(req.get("ref_video_start") or 0)
    cuts = sorted({int(round((c - start) * 24))
                   for c in (req.get("ref_video_meta") or {}).get("cuts") or []})
    bounds = [0] + [f for f in cuts if 0 < f < total] + [total]
    merged = []
    for a, b in zip(bounds, bounds[1:]):
        if merged and ((b - a) < H3_MIN_SEGMENT
                       or (merged[-1][1] - merged[-1][0]) < H3_MIN_SEGMENT)                 and (b - merged[-1][0]) <= max_len:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    chunks = []
    for a, b in merged:
        k = -(-(b - a) // max_len)
        step = -(-(b - a) // k)
        chunks += [(a + i * step, min(b, a + (i + 1) * step)) for i in range(k)]
    return chunks


def h3_chunk_swaps(req: dict, a: int, b: int) -> list:
    """Which --swap passes a chunk [a, b) gets: indices, [] = no one to replace.

    A swap with a time range (absolute seconds in the reference video) applies
    to a chunk it covers at least half of; one without a range applies
    everywhere. Without --swap the chunk gets one pass with the caller's prompt
    (returned as [None]).
    """
    swaps = req.get("swaps") or []
    if not swaps:
        return [None]
    start = float(req.get("ref_video_start") or 0)
    lo, hi = start + a / 24.0, start + b / 24.0
    picked = []
    for i, sw in enumerate(swaps):
        s0 = sw.get("start")
        s1 = sw.get("end")
        s0 = lo if s0 is None else float(s0)
        s1 = hi if s1 is None else float(s1)
        if max(0.0, min(hi, s1) - max(lo, s0)) >= 0.5 * (hi - lo):
            picked.append(i)
    return picked


def h3_snap_length(frames: int) -> int:
    """H3 samples on a 17k+5 frame grid; snap UP so we never under-deliver."""
    frames = max(5, int(frames))
    rem = (frames - 5) % 17
    return frames if rem == 0 else frames + (17 - rem)


def h3_activation_gb(width: int, height: int, length: int, ref_tokens: int = 0) -> float:
    return round(H3_ACT_GB_PER_TOKEN * (h3_tokens(width, height, length) + ref_tokens), 2)


def h3_time_estimate_s(width: int, height: int, length: int, ref_tokens: int = 0) -> float:
    return round(H3_TIME_C * ((h3_tokens(width, height, length) + ref_tokens) ** H3_TIME_EXP), 0)


def activation_gb(short_side: int, length: int) -> float:
    """Conservative VRAM (GB) for latents+attention that CANNOT be offloaded."""
    base = {480: 3.2, 576: 4.6, 720: 8.5}.get(short_side, 3.2)
    return round(base * (length / 49.0), 1)


def _dims_for(short_side: int, portrait: bool) -> tuple[int, int]:
    long_side = int(round(short_side * 16 / 9 / 16) * 16)
    return (short_side, long_side) if portrait else (long_side, short_side)


def _refuse(msg, budget, why="would risk co-running programs"):
    return {"ok": False, "refuse_reason": msg, "budget": budget,
            "plan": "REFUSED (%s): %s" % (why, msg)}


def _h3_dims(short: int, portrait: bool, mult: int) -> tuple[int, int]:
    """16:9 dimensions snapped to `mult`. 768 maps to H3's native 1344x768."""
    native = {768: 1344, 720: 1280, 608: 1056, 576: 1024, 480: 864}
    long_side = native.get(short) or max(mult, int(round(short * 16 / 9 / mult) * mult))
    short = max(mult, int(round(short / mult) * mult))
    return (short, long_side) if portrait else (long_side, short)


# Phrases that pin the camera. A pinned FIRST FRAME is only a condition: H3
# edits inside one shot, so an action prompt with no camera pinned makes it cut
# to the close-up at frame 0 and never show the frame it was given. Measured
# 2026-09-09: identical seed/steps/canvas, "…kisses his cheek" opened 54-77
# /255 away from the pinned image; adding "the camera stays wide and static
# throughout, no cuts" brought it to 7.35 -- while the clip still moved.
H3_CAMERA_CUES = ("static camera", "camera stays", "camera is static", "no cuts",
                  "locked off", "locked-off", "fixed camera", "camera does not move",
                  "camera doesn't move", "single continuous shot", "one continuous shot",
                  "wide shot throughout", "no camera movement", "tripod")


def h3_camera_pinned(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(c in p for c in H3_CAMERA_CUES)


# Phrases that ask H3 to RE-FRAME. Measured 2026-09-10 on the WSL run: a prompt
# carrying both a camera pin and one of these loses the pinned frame anyway
# (frame 0 came back 47-58/255 away, versus 7.1 for the same shot with the pin
# alone). The close-up wins, which is the whole point of asking for one -- so
# the two instructions cannot both be honoured and the caller has to choose.
#
# This combination is not a silly mistake: the "face does not look like the
# reference" section of SKILL.md used to prescribe exactly this close-up
# language, so an agent fixing a face on round 2 would break the anchor that
# worked on round 1. Catching it here is what stops that.
H3_REFRAME_CUES = ("close-up", "close up", "closeup", "close shot", "medium close",
                   "push in", "pushes in", "pushing in", "zoom", "dolly",
                   "faces large", "tighter shot", "tight shot", "cut to",
                   "camera moves", "camera pans", "pan across")


def h3_reframe_requested(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(c in p for c in H3_REFRAME_CUES)


def _h3_dims_fit(w: int, h: int, short: int, mult: int) -> tuple[int, int]:
    """Canvas that keeps the source aspect: short side as asked, long side follows.

    Mirrors adapt_canvas() in comfy_extras/nodes_minimax_h3.py (per-axis snap to
    the canvas multiple, capped at H3's trained pixel area) but honours the
    caller's requested short side instead of H3's native 768.
    """
    if w <= 0 or h <= 0:
        return (short, short)
    if w >= h:
        cw, ch = short * (w / float(h)), float(short)
    else:
        cw, ch = float(short), short * (h / float(w))
    cap = float(H3_NATIVE_PIXELS)
    if cw * ch > cap:
        s = math.sqrt(cap / (cw * ch))
        cw, ch = cw * s, ch * s
    return (max(mult, int(round(cw / mult)) * mult),
            max(mult, int(round(ch / mult)) * mult))


def _h3_portrait_from_image(req: dict) -> bool:
    """Orientation of the i2v start frame; False when the header cannot be read."""
    dims = _image_dims(str(req.get("image") or ""))
    return bool(dims and dims[1] > dims[0])


def _plan_h3_refvideo(req: dict, m: dict, vram_budget: float, free_ram: float,
                      reclaim: float, budget: dict, width: int, height: int,
                      short: int, follow_video: bool) -> dict:
    """Plan a recast (--ref-video): shot-split chunks, each a single ref2va shot.

    Resolution is never traded away here -- that is the point of chunking. The
    longest chunk that fits VRAM and the per-submission time box at the asked
    canvas sets max_len, and the reference is cut to that (at its own shots
    first). Only if not even a minimum chunk fits is the request refused.
    """
    fps = m["fps"]
    avail = h3_ref_video_seconds(req)
    total = int(avail * fps) if avail is not None else round(float(req.get("seconds") or 5.0) * fps)
    if req.get("seconds") is not None:
        total = min(total, round(float(req["seconds"]) * fps))
    grid = h3_ref_video_grid(req, width, height)
    ref_tokens = h3_ref_tokens(req, width, height)
    img_bill = max(0, ref_tokens - h3_ref_baseline_tokens(width, height))

    def bill(n):
        return img_bill + grid * h3_video_latent_t(n)

    def fits(n):
        return (h3_activation_gb(width, height, n, bill(n)) <= vram_budget
                and h3_time_estimate_s(width, height, n, bill(n)) <= H3_MAX_SEGMENT_S)

    max_len = min(h3_snap_length(total), H3_REF_CHUNK_MAX)
    while not fits(max_len):
        if max_len <= H3_MIN_SEGMENT:
            return _refuse(
                "even a %d-frame chunk at %dx%d with a %d-short reference video needs "
                "%.1f GB VRAM for activations, but only %.1f GB is spendable after holding "
                "%.1f GB for the LLM. Lower --ref-video-size (e.g. 256) or --short."
                % (H3_MIN_SEGMENT, width, height,
                   int(req.get("ref_video_size") or H3_REF_VIDEO_SHORT),
                   h3_activation_gb(width, height, H3_MIN_SEGMENT, bill(H3_MIN_SEGMENT)),
                   vram_budget, budget.get("llm_vram_held_gb", 0)), budget)
        max_len = _h3_step_down(max_len)

    floor = _protected("VIDGEN_SSD_MIN_FREE_RAM", 0.6)
    if free_ram + reclaim < SSD_INCR_RAM + floor:
        return _refuse("free RAM %.1f GB (+%.1f GB reclaimable) is below ~%.1f GB working "
                       "buffers + %.1f GB anti-lock floor."
                       % (free_ram, reclaim, SSD_INCR_RAM, floor), budget)

    swaps = req.get("swaps") or []
    start = float(req.get("ref_video_start") or 0)
    ec = h3_easycache(req)
    chunks, total_est, act, renders, used = [], 0.0, 0.0, 0, set()
    for a, b in h3_ref_video_chunks(req, total, max_len):
        n = b - a
        rl = h3_snap_length(n)                  # rendered on the grid, trimmed back to n
        picks = h3_chunk_swaps(req, a, b)
        est = h3_time_estimate_s(width, height, rl, bill(rl)) * len(picks)
        if ec > 0:
            est = round(est * H3_EASYCACHE_TIME, 0)
        if picks:
            act = max(act, h3_activation_gb(width, height, rl, bill(rl)))
        total_est += est
        renders += len(picks)
        used.update(i for i in picks if i is not None)
        chunks.append({"start": round(start + a / fps, 3), "end": round(start + b / fps, 3),
                       "frame_a": a, "frame_b": b, "render_frames": rl,
                       "swaps": picks, "estimate_s": round(est, 0),
                       "who": ([swaps[i]["who"] for i in picks] if swaps and picks else
                               "unchanged (original frames)" if swaps else "prompt")})

    notes = []
    for i, sw in enumerate(swaps):
        if i not in used:
            notes.append("--swap %r covers no shot (its range never spans half a chunk) and "
                         "will not be applied" % sw["who"])
    if not swaps and len(req.get("refs") or []) > 1:
        notes.append("H3 replaces ONE person per render: several --ref pictures are treated "
                     "as views of that one character. If they are different people, use "
                     "--swap once per person instead")
    if any(len(c["swaps"]) > 1 for c in chunks):
        notes.append("some shots get more than one --swap pass; the second pass on the same "
                     "shot does not always take (measured 10-27). Give each --swap a time "
                     "range (WHO=IMG@START-END) when the people are in different shots")
    warning = "; ".join(notes) or None

    # H3 wants /32 dims, so a 720 request lands on a 704 canvas; the joined
    # clip is scaled to the size that was actually asked for.
    s = max(1.0, short / float(min(width, height)))
    out_w = int(round(width * s / 2)) * 2
    out_h = int(round(height * s / 2)) * 2

    budget["activation_gb"] = round(act, 2)
    budget["committed_ram_gb"] = SSD_INCR_RAM
    budget["h3_canvas_from"] = "reference video aspect" if follow_video else "requested"
    budget["h3_ref_video_size"] = int(req.get("ref_video_size") or H3_REF_VIDEO_SHORT)
    budget["h3_ref_video_tokens_max_chunk"] = grid * h3_video_latent_t(max_len)
    budget["h3_max_chunk_frames"] = max_len
    budget["h3_time_estimate_s"] = round(total_est, 0)
    route_label = ("ref2va + <Video 1>: %d chunk(s) cut at the reference's shots, %d render(s)"
                   % (len(chunks), renders))
    return {
        "ok": True, "tier": "h3", "model": m, "template": m["template"],
        "template_chain": m["template_chain"], "template_refkf": m["template_refkf"],
        "h3_route": "ref2va", "warning": warning,
        "mode": req.get("mode", "r2v"), "width": width, "height": height,
        "out_width": out_w, "out_height": out_h,
        "length": total, "fps": fps, "steps": int(req.get("steps", 20)),
        "easycache": ec,
        "cfg": None, "split": None, "fast": False,
        "text_encoder": H3_TEXT_ENCODER, "cache_none": True, "audio": True,
        "segments": len(chunks), "segment_length": max_len, "chunks": chunks,
        "segment_estimate_s": h3_time_estimate_s(width, height, max_len, bill(max_len)),
        "total_estimate_s": round(total_est, 0),
        "shift_video": m["shift_video"], "shift_audio": m["shift_audio"],
        "budget": budget,
        "offload": {"weights": "int8 convrot safetensors -> DynamicVRAM stream (never commits RAM)",
                    "activations": "VRAM (cannot offload)", "cache_none": True,
                    "protects": "holds %.1f GB VRAM for the LLM + %.1f GB margin"
                                % (budget.get("llm_vram_held_gb", 0), VRAM_MARGIN)},
        "shrunk": [], "ref_image_size": req.get("ref_image_size", "match"),
        "ref_tokens": ref_tokens,
        "plan": ("%s [%s] | %dx%d%s x%df (%.2f s) @%dfps + audio | %d steps%s | chunks <= %df | "
                 "~%.0f min | act %.1f GB, spendable %.1f GB | ref video %d short%s"
                 % (m["label"], route_label, width, height,
                    (" -> %dx%d" % (out_w, out_h)) if (out_w, out_h) != (width, height) else "",
                    total, total / float(fps), fps, int(req.get("steps", 20)),
                    _easycache_label(ec), max_len,
                    total_est / 60.0, act, vram_budget,
                    int(req.get("ref_video_size") or H3_REF_VIDEO_SHORT),
                    (" | WARNING: %s" % warning) if warning else "")),
    }


def _plan_h3(req: dict, m: dict, vram_budget: float, free_ram: float,
             reclaim: float, budget: dict) -> dict:
    """Plan a MiniMax-H3 render.

    Differences from the Wan path that matter:
      * fps is fixed at 24 and length lives on a 17k+5 grid
      * dimensions must be multiples of 32; native training size is 1344x768
      * memory is ~linear in latent tokens but TIME is ~quadratic, so a long
        single shot is not an OOM risk, it is a wall-clock and driver-stability
        risk. We therefore split into chained segments instead of refusing.
      * it decodes an audio latent as well, so the graph has two VAEs
    """
    route = h3_route(req)
    # "a start frame is pinned", which is what drives orientation and the
    # stretch warning -- true for plain i2v and for r2v carrying an --image.
    keyframed = route in ("fl2va", "ref2va+kf")
    portrait = req.get("portrait")
    if portrait is None:
        # Wan centre-crops its start image; H3's fl2va keyframe is STRETCHED to
        # the canvas. A portrait start frame on the default landscape canvas
        # therefore comes back squashed, not cropped -- so when the caller did
        # not state an orientation, take it from the image rather than assuming
        # landscape. (r2v references keep their aspect, so this is i2v-only.)
        portrait = _h3_portrait_from_image(req) if keyframed else False
    portrait = bool(portrait)
    ref_video = bool(req.get("ref_video"))
    seconds = float(req.get("seconds") or 5.0)
    short = int(req.get("short_side", 480))
    fps = m["fps"]

    # A pinned start frame is stretched onto the canvas, never cropped, so the
    # canvas should be the frame's own shape unless the caller asked for a
    # specific one. Everything below sizes through dims_for() so the shrink
    # loop and the single-shot search cannot disagree about the shape. A
    # reference video sets the shape the same way: its framing is what H3
    # copies, so a mismatched canvas crops or squeezes the motion it follows.
    src_dims = _image_dims(str(req.get("image") or "")) if keyframed else None
    if ref_video and not src_dims:
        meta = req.get("ref_video_meta") or {}
        if meta.get("w") and meta.get("h"):
            src_dims = (int(meta["w"]), int(meta["h"]))
    follow_image = bool(src_dims) and req.get("portrait") is None

    def dims_for(s: int) -> tuple[int, int]:
        if follow_image:
            return _h3_dims_fit(src_dims[0], src_dims[1], s, m["dim_multiple"])
        return _h3_dims(s, portrait, m["dim_multiple"])

    width, height = dims_for(short)
    if ref_video:
        return _plan_h3_refvideo(req, m, vram_budget, free_ram, reclaim, budget,
                                 width, height, short, follow_image)
    length = h3_snap_length(round(seconds * fps))

    # A chained render only ever has ONE segment in flight, so the activation
    # budget and the wall-clock box both apply per SEGMENT -- not to the whole
    # clip. Shrinking the requested duration because the *total* would not fit
    # in one shot is exactly the mistake to avoid: that is what chaining is for.
    # Reference latents ride through every step, so they are part of the
    # activation bill. Under "match" they rescale with the canvas, so this has
    # to be recomputed whenever the shrink loop changes the resolution.
    ref_tokens = h3_ref_tokens(req, width, height)
    ref_bill = h3_ref_billable(req, width, height)

    shrinks = []
    seg_len = _h3_fit_segment(width, height, length, vram_budget, ref_bill)
    while seg_len is None and short > 480:
        short = {768: 608, 720: 608, 608: 480, 576: 480}.get(short, 480)
        width, height = dims_for(short)
        ref_tokens = h3_ref_tokens(req, width, height)
        ref_bill = h3_ref_billable(req, width, height)
        shrinks.append((short, length))
        seg_len = _h3_fit_segment(width, height, length, vram_budget, ref_bill)

    if seg_len is None:
        floor_act = h3_activation_gb(width, height, H3_MIN_SEGMENT, ref_bill)
        hint = ""
        if ref_bill > h3_tokens(width, height, H3_MIN_SEGMENT) // 2:
            hint = (" Most of that is your reference images (%d latent tokens at "
                    "ref_image_size=%s) -- crop them to head-and-shoulders, or drop "
                    "--ref-image-size max."
                    % (ref_tokens, req.get("ref_image_size", "match")))
        return _refuse("even a floor H3 segment (%dx%d, %d frames) needs ~%.1f GB VRAM for "
                       "activations but only %.1f GB is spendable after holding %.1f GB for "
                       "the LLM. Unload the LLM or lower VIDGEN_LLM_VRAM_BUDGET.%s"
                       % (width, height, H3_MIN_SEGMENT, floor_act, vram_budget,
                          budget.get("llm_vram_held_gb", 0), hint), budget)
    act = h3_activation_gb(width, height, seg_len, ref_bill)

    # RAM: int8 convrot safetensors stream from disk exactly like GGUF.
    floor = _protected("VIDGEN_SSD_MIN_FREE_RAM", 0.6)
    if free_ram + reclaim < SSD_INCR_RAM + floor:
        return _refuse("free RAM %.1f GB (+%.1f GB reclaimable) is below ~%.1f GB working "
                       "buffers + %.1f GB anti-lock floor."
                       % (free_ram, reclaim, SSD_INCR_RAM, floor), budget)

    # ---- segment planning ---------------------------------------------------
    segments = max(1, -(-length // seg_len))           # ceil

    # Single shot is the product, not a nice-to-have. Only segment 0 is
    # conditioned on the reference images (ref2va); every later segment is
    # fl2va seeded from the previous one's last frame, so identity drifts a
    # little further each time and a bad hand-off ruins everything after it.
    # H3 also edits within one segment on its own -- a 25 s single shot comes
    # back as wide -> medium -> detail -> back view with the character intact --
    # so a long take is not a compromise. Rather than quietly chaining and
    # handing back something that drifts, refuse and say exactly what fits.
    if segments > 1 and os.environ.get("VIDGEN_H3_ALLOW_CHAIN", "0") != "1":
        fits = [s for s in (768, 720, 608, 576, 480)
                if (lambda wh: (_h3_fit_segment(wh[0], wh[1], length, vram_budget,
                                                h3_ref_billable(req, wh[0], wh[1])) or 0) >= length)
                (dims_for(s))]
        alt = ("or keep %.2f s and drop to --short %d, which still fits in one shot"
               % (length / fps, max(fits)) if fits else
               "no resolution here fits that length in one shot")
        # What chaining costs you differs by mode, so state the true one: r2v
        # loses the reference images, i2v drifts away from the start frame.
        loses = ("only the FIRST segment starts from your --image (every later one is "
                 "seeded from the previous segment's last frame), so the shot drifts "
                 "away from the picture you gave"
                 if keyframed else
                 "only the FIRST segment sees your --ref images (later ones are seeded "
                 "from the previous segment's last frame), so faces drift segment by "
                 "segment")
        instead = ("run several separate i2v calls, each starting from a frame you "
                   "chose, and join them"
                   if keyframed else
                   "run several separate r2v calls, EACH with --ref, and join them: "
                   "every call re-anchors on your reference instead of on a drifted "
                   "frame")
        return _refuse(
            "%.2f s at %dx%d needs %d chained segments, and chaining is off by default: "
            "%s. The longest single shot at this resolution is %.2f s (%d frames) -- "
            "shorten to that, %s. If you genuinely need more, %s. (Maintainers: "
            "VIDGEN_H3_ALLOW_CHAIN=1 restores the old auto-chaining.)"
            % (length / fps, width, height, segments, loses,
               seg_len / fps, seg_len, alt, instead),
            budget, why="would need chaining, which loses the anchor")

    ec = h3_easycache(req)
    seg_est = h3_time_estimate_s(width, height, seg_len, ref_bill)
    if ec > 0:
        seg_est = round(seg_est * H3_EASYCACHE_TIME, 0)
    total_est = round(seg_est * segments, 0)

    # The keyframe is stretched, never cropped, so a mismatched aspect ratio is
    # a visible defect in the output rather than a lost border. The planner is
    # the only place that holds both numbers, so it is the only place that can
    # say so before several minutes of GPU time have been spent.
    stretch = None
    if keyframed:
        dims = _image_dims(str(req.get("image") or ""))
        if dims and dims[0] > 0 and dims[1] > 0:
            src, dst = dims[0] / dims[1], width / height
            if abs(src - dst) / dst > 0.12:
                stretch = ("start frame is %dx%d (%.2f:1) but the canvas is %dx%d "
                           "(%.2f:1) -- H3 STRETCHES the keyframe onto the canvas "
                           "instead of cropping it, so crop the image to that aspect "
                           "first if the distortion matters"
                           % (dims[0], dims[1], src, width, height, dst))

    budget["activation_gb"] = act
    budget["committed_ram_gb"] = SSD_INCR_RAM
    budget["h3_canvas_from"] = ("start frame aspect" if follow_image
                                else "16:9 portrait" if portrait else "16:9 landscape")
    # The pinned frame is worth nothing if the prompt lets H3 cut away at frame 0.
    if keyframed and not h3_camera_pinned(req.get("prompt", "")):
        drift = ("you pinned a first frame but the prompt does not pin the CAMERA, so "
                 "H3 may cut to its own framing at frame 0 and never show your image "
                 "(measured: 54-77/255 away from the pinned frame, versus 7.4 with the "
                 "camera pinned). Add something like \"the camera stays wide and static "
                 "throughout, no cuts\" -- the clip still moves, it just stops "
                 "re-framing.")
        stretch = (stretch + " | " + drift) if stretch else drift
        budget["h3_camera_unpinned"] = True
    budget["h3_keyframe_stretch"] = stretch
    budget["h3_latent_tokens"] = h3_tokens(width, height, length)
    budget["h3_ref_tokens"] = ref_tokens
    budget["h3_ref_tokens_billed"] = ref_bill
    budget["h3_ref_image_size"] = req.get("ref_image_size", "match")
    budget["h3_time_estimate_s"] = total_est
    return {
        "ok": True, "tier": "h3", "model": m, "template": m["template"],
        "template_chain": m["template_chain"],
        "template_refkf": m["template_refkf"],
        "h3_route": route,
        "warning": stretch,
        "mode": req.get("mode", "r2v"), "width": width, "height": height,
        "length": length, "fps": fps, "steps": int(req.get("steps", 20)),
        "easycache": ec,
        "cfg": None, "split": None, "fast": False,
        "text_encoder": H3_TEXT_ENCODER, "cache_none": True,
        "audio": True, "segments": segments, "segment_length": seg_len,
        "segment_estimate_s": seg_est, "total_estimate_s": total_est,
        "shift_video": m["shift_video"], "shift_audio": m["shift_audio"],
        "budget": budget,
        "offload": {"weights": "int8 convrot safetensors -> DynamicVRAM stream (never commits RAM)",
                    "activations": "VRAM (cannot offload)", "cache_none": True,
                    "protects": "holds %.1f GB VRAM for the LLM + %.1f GB margin"
                                % (budget.get("llm_vram_held_gb", 0), VRAM_MARGIN)},
        "shrunk": shrinks,
        "ref_image_size": req.get("ref_image_size", "match"),
        "ref_tokens": ref_tokens,
        "plan": ("%s [%s] | %dx%d x%df @%dfps + audio | %d steps%s | %s | ~%.0f min | "
                 "act %.1f GB, spendable %.1f GB%s%s"
                 % (m["label"],
                    {"fl2va": "fl2va: --image is frame 0",
                     "ref2va": "ref2va: --ref images are the cast",
                     "ref2va+kf": "ref2va+kf: --ref images are the cast, "
                                  "--image is frame 0"}[route],
                    width, height, length, fps, int(req.get("steps", 20)),
                    _easycache_label(ec),
                    ("%d chained segments of %df" % (segments, seg_len)) if segments > 1
                    else "single shot",
                    total_est / 60.0, act, vram_budget,
                    ((" | %d keyframe(s) + %d ref(s) %s: %d tokens (%d billed)"
                      % (h3_keyframes(req), len(h3_refs(req)),
                         req.get("ref_image_size", "match"), ref_tokens, ref_bill))
                     if route == "ref2va+kf" else
                     (" | keyframes: %d tokens (%d billed)" % (ref_tokens, ref_bill))
                     if keyframed else
                     (" | refs %s: %d tokens (%d billed)"
                      % (req.get("ref_image_size", "match"), ref_tokens, ref_bill)))
                    if ref_tokens else "",
                    (" | WARNING: %s" % stretch) if stretch else "")),
    }


def llmsleep_is_lent() -> bool:
    import llmsleep
    return llmsleep.is_lent()


def _lend_plan() -> dict:
    """{} when the LLM stays resident (hold VRAM back for it, as always), else
    how much VRAM the render gets back by lending it (see llmsleep)."""
    import llmsleep
    if llmsleep.is_lent():
        return {"gb": 0.0, "why": "LLM lent to the render (asleep/unloaded until it ends)"}
    st = astapi.status()
    if not llmsleep.lendable(st):
        return {}
    if st.get("sleeping"):
        return {"gb": 0.0, "why": "LLM asleep; it is woken only after the render"}
    gb = llmsleep.llm_vram_gb() or 0.0
    return {"gb": gb, "why": "LLM (%s, %.1f GB VRAM) is lent to the render: put to sleep "
                             "or unloaded before it starts and given back after"
                             % (st.get("engine"), gb)}


def choose(req: dict) -> dict:
    # Planning MEASURES the LLM's reclaimable DRAM; it never trims it. The trim
    # would otherwise land before ComfyUI is up -- and choose() also runs for
    # `plan` dry-runs and for the preflight recheck, i.e. on paths that may end
    # in a refusal with no render at all. See release_llm_dram().
    llm = llm_dram()

    snap = sysmon.snapshot()
    fv = snap["gpu"].get("free_mb", 0) / GB
    total_vram = snap["gpu"].get("total_mb", 0) / GB
    fr = snap["ram"].get("avail_mb", 0) / GB
    total_ram = snap["ram"].get("total_mb", 0) / GB
    RAM_MARGIN = _ram_margin(total_ram, fr)   # adaptive: measured right now
    llm_hold, llm_hold_why = _llm_vram_reserved_gb(fv, total_vram)
    lend = _lend_plan()
    if lend:
        # An LLM that fills the card on purpose (Strata: 14.3 GB of 16) is put
        # to sleep or unloaded for the render instead of being protected; what
        # it holds now is ours for the job, and nothing is held back for it.
        fv += lend["gb"]
        llm_hold, llm_hold_why = 0.0, lend["why"]
        # Its RAM comes back too when it has to be unloaded (Strata page-locks
        # 13.5 GB of experts; a plan measured 0.8 GB free with it resident and
        # refused). When it is only put to sleep instead, that sleep itself
        # required 2 GiB+ free on top of its backup, so RAM was not short.
        if not llmsleep_is_lent():
            fr += llm.get("ws_gb") or 0.0
            llm = dict(llm, reclaimable_gb=0.0)     # already counted in full
    reclaim = llm.get("reclaimable_gb", 0.0)
    budget = {"free_vram_gb": round(fv, 1), "total_vram_gb": round(total_vram, 1),
              "free_ram_gb": round(fr, 1),
              "vram_margin_gb": VRAM_MARGIN, "ram_margin_gb": RAM_MARGIN,
              "llm_vram_budget_gb": LLM_VRAM_BUDGET,
              "llm_vram_held_gb": round(llm_hold, 1), "llm_vram_basis": llm_hold_why,
              "ram_margin_adaptive": os.environ.get("VIDGEN_RAM_MARGIN") is None,
              "llm_dram": llm,
              "llm_reclaimable_gb": reclaim}

    # spendable budgets after protecting everyone else. The LLM hold-back keeps
    # room for the 35B to come back even if it is unloaded right now.
    vram_budget = fv - VRAM_MARGIN - llm_hold   # for activations
    ram_budget = fr - RAM_MARGIN                # for our committed RAM
    # Clean, file-backed LLM pages are spendable too (see module docstring); when
    # we count them, the thing left to protect is the OS, not the LLM's residency.
    if reclaim > 0:
        ram_budget = max(ram_budget, fr + reclaim - OS_FLOOR)
        budget["ram_budget_basis"] = ("free %.1f + LLM reclaimable %.1f - OS floor %.1f"
                                      % (fr, reclaim, OS_FLOOR))
    budget["vram_budget_gb"] = round(vram_budget, 1)
    budget["ram_budget_gb"] = round(ram_budget, 1)
    shared, shared_why = shares_system_ram()
    if shared:
        budget["shared_memory"] = shared_why

    if vram_budget <= 0.6:
        return _refuse("only %.1f GB VRAM free and %.1f GB is held back for the LLM "
                       "(%s) plus a %.1f GB margin, leaving %.1f GB. Free some GPU memory, "
                       "unload the LLM, or lower VIDGEN_LLM_VRAM_BUDGET."
                       % (fv, llm_hold, llm_hold_why, VRAM_MARGIN, vram_budget), budget)

    # ---- tier ----
    tier = req.get("tier", "auto")
    if tier == "auto":
        tier = "14b"                          # always the SSD-backed path

    # Backstop for the rule gen.py enforces. The only 14B workflow here is
    # wan_i2v_14b_gguf.json -- an I2V model whose Lightning LoRA was distilled
    # for I2V too. A text-only request gives it no start frame, so it samples
    # from nothing and writes PURE NOISE; the render still succeeds and returns
    # an mp4, which is why this reads as success until somebody opens the file.
    # Refuse instead, and name the route that does work.
    if req.get("mode") == "t2v" and tier == "14b":
        return _refuse(
            "t2v has no working model on this box. The 14B tier is Wan 2.2 I2V-A14B "
            "(I2V-only Lightning LoRA), so a text-only prompt renders pure noise while "
            "still reporting success. Use the two-stage route instead: generate a still "
            "with the imggen skill, then animate it with `i2v --image <that png>`. "
            "Tier '5b' (Wan TI2V) is the only genuine text-to-video model here, but it "
            "is fp16 and is normally refused to protect the LLM.",
            budget, why="no usable model for this mode")

    m = MODELS[tier]
    if shared:
        # One pool. What the device holds -- this tier's weights, and later its
        # activations -- comes out of the same RAM the budget above already counts,
        # so the activation budget is what RAM has left once the weights are in.
        resident = DEVICE_RESIDENT_GB.get(tier, 0.0)
        pool = ram_budget - resident
        budget["shared_pool_gb"] = round(pool, 1)
        budget["shared_pool_basis"] = ("RAM spendable %.1f - %s weights on the GPU %.1f"
                                       % (ram_budget, tier, resident))
        if pool <= 0.6:
            return _refuse("the GPU shares system RAM (%s), and tier '%s' keeps ~%.0f GB "
                           "of weights on it, leaving %.1f GB of the %.1f GB that is safely "
                           "spendable. Free RAM, or use a smaller tier."
                           % (shared_why, tier, resident, pool, ram_budget), budget)
        vram_budget = min(vram_budget, pool)
        budget["vram_budget_gb"] = round(vram_budget, 1)
    if tier == "h3":
        return _with_speed_note(_plan_h3(req, m, vram_budget, fr, reclaim, budget))
    if not m["ssd_backed"]:
        # fp16 opt-in: only if RAM can genuinely hold the weights + margin
        need = m["weights_ram_gb"] + RAM_OVERHEAD
        if ram_budget < need:
            why = ("margin %.1f GB" % RAM_MARGIN if reclaim <= 0 else
                   "free %.1f GB + %.1f GB reclaimable from the LLM, minus a %.1f GB OS floor"
                   % (fr, reclaim, OS_FLOOR))
            hint = ("" if reclaim > 0 else
                    " The LLM's weights look like dirty private pages (--no-mmap?), so none of its "
                    "footprint can be counted -- evicting it would mean pagefile writes.")
            return _refuse("tier '5b' is fp16 and would commit ~%.0f GB RAM, but only %.1f GB is "
                           "safely spendable (%s). It is not SSD-pageable, so this is refused to "
                           "protect the LLM.%s Use tier 'auto'/'14b' (GGUF, SSD-mmap) instead."
                           % (need, ram_budget, why, hint), budget)

    # ---- fit resolution/length to the VRAM activation budget ----
    portrait = bool(req.get("portrait", False))
    fps = int(req.get("fps", 16))
    seconds = float(req.get("seconds", 3.0))
    short = int(req.get("short_side", 480))
    length = max(17, min(int(round(seconds * fps / 4) * 4) + 1, 121))

    shrinks = []
    while activation_gb(short, length) > vram_budget and (short > 480 or length > 17):
        if length > 17:
            length = max(17, length - 16)
        elif short > 480:
            short = {720: 576, 576: 480}.get(short, 480)
        shrinks.append((short, length))
    act = activation_gb(short, length)
    if act > vram_budget:
        return _refuse("even the floor job (480p, %d frames) needs %.1f GB VRAM for "
                       "activations but only %.1f GB is safely spendable. Free some GPU "
                       "memory (or unload the LLM) and retry." % (length, act, vram_budget), budget)

    # RAM check. SSD-backed (GGUF) weights are mmap'd and page to SSD under
    # pressure, and ComfyUI/torch is already resident -- so the incremental
    # hard-commit is just working buffers, and any excess falls on our OWN
    # weight pages (slow SSD paging), not the LLM. We only guard a small floor
    # so the box doesn't hard-lock. Non-SSD fp16 was already checked above.
    if m["ssd_backed"]:
        ram_need = SSD_INCR_RAM
        floor = _protected("VIDGEN_SSD_MIN_FREE_RAM", 0.6)
        # Same reasoning as the fp16 branch: pages the OS can drop for free are
        # not a constraint. Without this, the *default* GGUF path kept refusing
        # at "free RAM 0.8 GB" while the LLM sat on 10 GB of discardable mmap
        # weights -- and a refused agent starts improvising.
        effective = fr + reclaim
        if effective < ram_need + floor:
            return _refuse("free RAM %.1f GB%s is below ~%.1f GB working buffers + %.1f GB anti-lock "
                           "floor. SSD offload pages the weights, not the buffers. Free ~%.1f GB RAM, "
                           "or rebalance the LLM (keep a bit more on GPU so its RAM footprint drops)."
                           % (fr, (" (+%.1f GB reclaimable from the LLM)" % reclaim) if reclaim else "",
                              ram_need, floor, ram_need + floor - effective), budget)
    else:
        ram_need = RAM_OVERHEAD + act * 0.5
        if ram_need > ram_budget:
            return _refuse("job needs ~%.1f GB committed RAM but only %.1f GB is safely "
                           "spendable (keeping %.1f GB for the LLM/other apps)."
                           % (ram_need, ram_budget, RAM_MARGIN), budget)
    if shared and act + ram_need > budget["shared_pool_gb"]:
        # On a shared GPU the activations are RAM as well, so both bills are paid
        # from the one pool the weights left behind.
        return _refuse("the GPU shares system RAM (%s): activations %.1f GB plus %.1f GB "
                       "of working RAM exceed the %.1f GB left after the weights."
                       % (shared_why, act, ram_need, budget["shared_pool_gb"]), budget)

    width, height = _dims_for(short, portrait)
    fast = req.get("fast")
    if fast is None:
        fast = True                            # be quick/light by default when tight
    if tier == "14b":
        # 14B has matching Lightning I2V LoRAs -> 4+4 steps works.
        steps = 8 if fast else 20
        cfg = 1.0 if fast else 3.5
        split = steps // 2
    else:
        # 5B has NO distillation LoRA on disk, so few-step "fast" produces mush.
        # Always use full sampling for the 5B regardless of the fast flag.
        fast = False
        steps = 30
        cfg = 5.0
        split = None

    budget["activation_gb"] = act
    budget["committed_ram_gb"] = round(ram_need, 1)
    weight_path = ("SSD-mmap (GGUF) -> OS page-cache -> VRAM (pages, never commits RAM)"
                   if m["ssd_backed"] else "fp16 committed to RAM + VRAM")
    return _with_speed_note({
        "ok": True, "tier": tier, "model": m, "template": m["template"],
        "mode": req.get("mode", "i2v"), "width": width, "height": height,
        "length": length, "fps": fps, "steps": steps, "cfg": cfg, "split": split,
        "fast": bool(fast), "text_encoder": TEXT_ENCODER, "cache_none": True,
        "budget": budget,
        "offload": {"weights": weight_path, "activations": "VRAM (cannot offload)",
                    "cache_none": True, "protects": "keeps %.1f GB VRAM + %.1f GB RAM free"
                    % (VRAM_MARGIN, RAM_MARGIN)},
        "shrunk": shrinks,
        "plan": ("%s | %dx%d x%df @%dfps | %d steps cfg %.1f%s | %s | keep-free VRAM %.1f/RAM %.1f GB"
                 % (m["label"], width, height, length, fps, steps, cfg,
                    " (Lightning 4+4)" if fast else "", weight_path, VRAM_MARGIN, RAM_MARGIN)),
        "warning": None,
    })


if __name__ == "__main__":
    import json, sys
    req = {"mode": "i2v", "tier": "auto", "short_side": 480, "seconds": 3.0, "fps": 16}
    if len(sys.argv) > 1:
        req.update(json.loads(sys.argv[1]))
    print(json.dumps(choose(req), indent=2, ensure_ascii=False))
