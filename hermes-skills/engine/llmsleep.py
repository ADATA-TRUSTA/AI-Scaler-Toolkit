"""Lend the co-resident LLM's VRAM to a render, and give it back afterwards.

Why this module exists
----------------------
The rest of the pack protects the LLM by holding VRAM back for it. That works
for a 35B that holds ~7 GB, but not for an engine that fills the card on
purpose: Strata (TRUSTA-AST's local engine) sizes its expert cache to the free
VRAM and measured 14.3 GB of a 16 GB card. Holding that back leaves nothing to
render with, so every H3 job was refused.

The agent IS the LLM, and while a render runs it is waiting on that render and
sends nothing. So the GPU can be lent outright and returned afterwards. Two
ways, tried in this order, both through the backend (AST) so the backend's own
bookkeeping stays right:

  sleep   POST /inference/sleep. Strata copies its non-cache VRAM (5.6 GB) to
          page-locked host RAM, drops the expert cache and keeps the process and
          the conversation. Wake ~1 s. On Windows the backup must fit in free
          COMMIT plus 2 GiB, and Strata's resident mode had already page-locked
          13.5 GB of experts on a 32 GB box -- measured: "the backup needs 5615
          MiB of host RAM and only 3241 MiB are available". Then:
  unload  POST /inference/unload_model. AST saves the conversation to Strata's
          slot file first and restores it on the next load, so the agent's next
          turn does not re-read its context. Frees VRAM AND all of the engine's
          RAM -- which the render wants anyway: H3 alone filled 31.1 of 31.3 GB.
          Reload measured 34 s.

Either way the LLM is only touched once it is idle: the agent's own last
message ("rendering started, please wait") may still be streaming when the job
is launched, and an unload would cut it off.

A lent LLM is recorded in jobs/LLM_LENT.json until it is given back, so a
crashed worker, a later job or `gen.py llm-return` can still return it.
VIDGEN_LLM_RETURN=defer keeps it lent after a job (recast sets it for the many
short jobs of one run, then returns it once at the end).

Env:
  VIDGEN_LLM_SLEEP=off      never lend; hold VRAM back as before.
  VIDGEN_LLM_RETURN=defer   do not give the LLM back after this job.
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.error
import urllib.request

import paths          # resolves hermes-comfy.conf into os.environ -- import first
import osmem
import sysmon

# The installed engine talks to the backend through llmapi (any of five
# backends); the pack still ships astapi (AST only). Lending only ever works
# through AST, so either is enough -- one file serves both trees.
try:
    import llmapi
    _provider, _backend_port = llmapi.provider, llmapi.backend_port
except ImportError:                       # pragma: no cover - pack tree
    import astapi as llmapi
    _provider = lambda: "ast" if llmapi.reachable() else None   # noqa: E731
    _backend_port = llmapi.ast_port

MODE = os.environ.get("VIDGEN_LLM_SLEEP", "auto").lower()
LENT_FILE = os.path.join(paths.JOBS_DIR, "LLM_LENT.json")
GB = 1024.0

# Which LLM is lent is decided by what it measures, never by its engine's name
# (the backend hosts whatever it was configured with): one that holds more VRAM
# than the planner's hold-back budget cannot be protected without starving the
# render, so it is lent instead. A 35B holding ~7 GB stays protected as before.
HOLD_GB = float(os.environ.get("VIDGEN_LLM_VRAM_BUDGET") or 7.0)
_vram_cache: tuple[float, float | None] = (0.0, None)

# ComfyUI's /free answers before its models are off the card (~1 s later). Waking
# into that gap fails on Linux (the engine refuses: not enough free VRAM) and on
# Windows succeeds into an overcommitted card: WDDM parks 3-4 GB of the LLM in
# system RAM and prefill crawls until the next sleep. So the LLM goes back only
# once the card is down to what it held while the LLM slept, plus this margin.
WAKE_MARGIN_GB = float(os.environ.get("VIDGEN_WAKE_VRAM_MARGIN_GB") or 1.0)
WAKE_WAIT_S = float(os.environ.get("VIDGEN_WAKE_WAIT_S") or 20.0)

# /inference/status fields that describe the running model rather than the load
# request. Only a fallback: the load body is normally built from the backend's
# own schema, and anything still extra is dropped on the 422 that names it.
_STATUS_ONLY = {"loaded", "is_loading", "sleeping", "loading_error", "error_type", "is_oom",
                "device", "process_alive", "device_allocation", "served_context_length",
                "prefill_strategy", "loaded_at", "memory_usage"}


def _say(*a) -> None:
    print("[llm-lend]", *a, file=sys.stderr, flush=True)


def _post(path: str, body: dict | None = None, timeout: float = 900):
    url = llmapi.base_url().rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    key = (os.environ.get("VIDGEN_LLM_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def _err(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            return "HTTP %s %s" % (e.code, e.read().decode(errors="replace")[:300])
        except Exception:
            return "HTTP %s" % e.code
    return repr(e)


def _used_gb() -> float | None:
    g = sysmon.gpu_vram()
    return None if "error" in g else round(g["used_mb"] / GB, 1)


def _tree():
    """(backend pid, its descendants, {port: pid} of what they listen on)."""
    port = _backend_port()
    listeners = osmem.listening_pids()
    root = listeners.get(port) if port else None
    if not root:
        return None, [], {}
    desc = list(osmem.descendants(root))
    mine = {p: pid for p, pid in listeners.items() if pid in desc and p != port}
    return root, desc, mine


def lendable(st: dict | None = None) -> bool:
    """Is there a loaded model, too big to protect, that the backend can sleep or
    unload for us? An asleep one counts: it was lent before and will be again."""
    if MODE == "off" or _provider() != "ast":
        return False
    st = st if st is not None else llmapi.status()
    if not st or not (st.get("loaded") or st.get("sleeping")):
        return False
    if not on_render_card():
        return False
    if st.get("sleeping"):
        return True
    gb = _llm_vram_cached()
    return gb is not None and gb > HOLD_GB


def on_render_card() -> bool:
    """Does the LLM share the card ComfyUI renders on?

    True unless measured otherwise: only an LLM seen on another card may be
    left alone, and sleeping it needlessly costs less than holding nothing
    back on a card it does share.
    """
    root, desc, _ = _tree()
    cards = osmem.proc_vram_cards([root] + desc) if root else None
    if not cards:
        return True
    if paths.GPU_VENDOR and paths.GPU_VENDOR != "nvidia":
        return False                    # the LLM is on NVIDIA, the render is not
    return not paths.GPU_INDEX or paths.GPU_INDEX in cards


def _llm_vram_cached(ttl: float = 20.0) -> float | None:
    """llm_vram_gb(), reused for a few seconds: one planning pass asks several
    times, and each reading of the GPU counters takes ~1.5 s on Windows."""
    global _vram_cache
    at, gb = _vram_cache
    if time.time() - at > ttl:
        gb = llm_vram_gb()
        _vram_cache = (time.time(), gb)
    return gb


def is_lent() -> bool:
    return os.path.isfile(LENT_FILE)


def llm_vram_gb() -> float | None:
    """What the LLM's processes hold on the card right now (measured, not reported)."""
    root, desc, _ = _tree()
    if not root:
        return None
    return osmem.proc_vram_gb([root] + desc)


def _engine_busy() -> bool | None:
    """True/False from the engine's own live state; None if it has no such endpoint."""
    _, _, ports = _tree()
    for p in ports:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % p, timeout=3) as r:
                live = (json.loads(r.read().decode()) or {}).get("live")
        except Exception:
            continue
        if isinstance(live, dict):
            return live.get("state") not in (None, "idle") or bool(live.get("queued"))
    return None


QUIET_S = float(os.environ.get("VIDGEN_LLM_LEND_QUIET_S") or 45)


def _wait_idle(timeout: float = 900) -> None:
    """Wait until the agent's turn is over: the LLM idle for QUIET_S on end.

    Idle for a moment is not enough. Between two tool calls the LLM is idle
    too, and an agent that launched the render and then waited 20 s on a
    background process was unloaded mid-turn: its next request failed, the
    backend's error read like a context overflow, and Hermes compressed the
    session from 65 messages to 15 before giving up. A quiet window longer
    than a tool call's gap is the turn ending.
    """
    t0 = calm_since = time.time()
    said = False
    while time.time() - t0 < timeout:
        busy = _engine_busy()
        if busy is None:
            time.sleep(min(QUIET_S, 15))     # cannot see it: give the turn time to end
            return
        now = time.time()
        if busy:
            calm_since = now
        elif now - calm_since >= QUIET_S:
            return
        if not said and now - t0 > 5:
            _say("waiting until the agent's turn is over (the LLM quiet for %ds) before "
                 "taking its GPU" % QUIET_S)
            said = True
        time.sleep(3)
    _say("the LLM was still in use after %ds; lending it anyway" % timeout)


def _load_body(st: dict) -> dict:
    """The /inference/load_model body that brings this same model back.

    /inference/status carries the load config plus runtime fields, and the load
    endpoint rejects any extra key (422 extra_forbidden -- the first reload
    failed on `slot_restore_summary`). So keep only what AST's own schema says
    the config has; without a schema, fall back to dropping the known extras.
    """
    try:
        with urllib.request.urlopen(llmapi.base_url().rstrip("/") + "/openapi.json",
                                    timeout=5) as r:
            props = json.loads(r.read().decode())["components"]["schemas"][
                "InferenceConfig"]["properties"]
        return {k: v for k, v in st.items() if k in props}
    except Exception:
        return {k: v for k, v in st.items() if k not in _STATUS_ONLY}


def _post_load(body: dict) -> None:
    """POST load_model; on 422 drop the fields it names as extra and try once more."""
    try:
        _post("/inference/load_model", body, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code != 422:
            raise
        detail = json.loads(e.read().decode() or "{}").get("detail") or []
        extra = {d["loc"][-1] for d in detail
                 if isinstance(d, dict) and d.get("type") == "extra_forbidden" and d.get("loc")}
        if not extra:
            raise
        _post("/inference/load_model", {k: v for k, v in body.items() if k not in extra},
              timeout=60)


def _write_lent(rec: dict) -> None:
    os.makedirs(os.path.dirname(LENT_FILE), exist_ok=True)
    with open(LENT_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)


def lend(reason: str = "") -> dict:
    """Take the LLM's GPU for a render. Never raises; {} = nothing was done."""
    if is_lent():
        return {"how": "already-lent"}
    st = llmapi.status()
    if not lendable(st) or st.get("sleeping"):
        return {}
    config = _load_body(st)
    before, t0 = _used_gb(), time.time()
    _wait_idle()
    how = None
    try:
        _post("/inference/sleep", {"mode": "wait"})
        how = "sleep"
    except Exception as e:
        _say("sleep refused (%s); unloading instead -- AST keeps the conversation "
             "in the slot file" % _err(e))
    if how is None:
        try:
            _post("/inference/unload_model", timeout=300)
            for _ in range(60):
                s = llmapi.status() or {}
                if not s.get("loaded"):
                    break
                time.sleep(1)
            how = "unload"
        except Exception as e:
            _say("could not unload the LLM either (%s); planning around it" % _err(e))
            return {"error": _err(e)}
    time.sleep(1)
    after = _used_gb()
    rec = {"how": how, "config": config, "reason": reason,
           "at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "freed_gb": round(before - after, 1) if before is not None and after is not None else None,
           # what the card held with the LLM away: the level to wait for before giving it back
           "asleep_gb": after,
           "seconds": round(time.time() - t0, 1)}
    _write_lent(rec)
    _say("LLM %s for %s: %s GB of VRAM lent (%.1fs)"
         % ("asleep" if how == "sleep" else "unloaded", reason or "the render",
            rec["freed_gb"], rec["seconds"]))
    return rec


def _wait_card_freed(rec: dict) -> dict:
    """Wait until the render's VRAM is really off the card (see WAKE_MARGIN_GB)."""
    target = rec.get("asleep_gb")
    used, t0 = _used_gb(), time.time()
    if target is None or used is None:
        # nothing to compare against: a pause covers /free's ~1 s lag
        time.sleep(2.0)
        return {"waited_s": 2.0, "measured": False}
    target += WAKE_MARGIN_GB
    while used is not None and used > target and time.time() - t0 < WAKE_WAIT_S:
        time.sleep(0.25)
        used = _used_gb()
    waited = round(time.time() - t0, 1)
    if used is not None and used > target:
        _say("the card still holds %.1f GB after %.0fs (%.1f GB expected); giving the LLM back anyway"
             % (used, waited, target))
    return {"waited_s": waited, "measured": True, "used_gb": used, "target_gb": round(target, 1)}


def restore(tries: int = 6, wait_s: float = 5.0) -> dict:
    """Give a lent LLM back (wake or reload). Never raises."""
    if not is_lent():
        return {"returned": False, "why": "nothing lent"}
    with open(LENT_FILE, encoding="utf-8") as f:
        rec = json.load(f)
    try:
        freed = _wait_card_freed(rec)
    except Exception as e:                     # a failed reading must not keep the LLM away
        freed = {"error": _err(e)}
    t0, last = time.time(), None
    for _ in range(tries):
        try:
            if rec.get("how") == "sleep":
                _post("/inference/wake", timeout=300)
            else:
                st = llmapi.status() or {}
                if not (st.get("loaded") or st.get("is_loading")):
                    _post_load(rec["config"])
                for _ in range(600):
                    st = llmapi.status() or {}
                    if st.get("loaded"):
                        break
                    if st.get("loading_error"):
                        raise RuntimeError(st["loading_error"])
                    time.sleep(1)
                else:
                    raise RuntimeError("model did not finish loading in 600 s")
            os.remove(LENT_FILE)
            _say("LLM back (%s, %.1fs, after waiting %ss for the card to empty)"
                 % ("woken" if rec.get("how") == "sleep" else "reloaded", time.time() - t0,
                    freed.get("waited_s", "?")))
            return {"returned": True, "how": rec.get("how"), "seconds": round(time.time() - t0, 1),
                    "card": freed}
        except Exception as e:
            last = _err(e)
            time.sleep(wait_s)     # /free returns before the driver has every page back
    _say("could not give the LLM back (%s); run `gen.py llm-return` to retry" % last)
    return {"returned": False, "error": last}


def give_back(lent: dict | None) -> dict:
    """End of a job: return the LLM unless the caller asked to keep it lent."""
    if not is_lent():
        return {"returned": False}
    if os.environ.get("VIDGEN_LLM_RETURN", "").lower() == "defer":
        return {"returned": False, "deferred": True}
    return restore()
