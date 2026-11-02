"""Is the ComfyUI we are about to use actually protecting the co-resident LLM?

Both skills reach the same server, and the guarantee they rely on is set once,
at ComfyUI's launch: `--reserve-vram N` carves N GB out of what ComfyUI
believes it may allocate. Without it ComfyUI treats VRAM the LLM is already
holding as free space and will happily allocate into it.

The gap this closes: `ensure_comfy()` only runs the protected launcher when
nothing is listening. If a ComfyUI is ALREADY up -- started by hand, by another
tool, or by an earlier session before the LLM was loaded -- both skills just
connect to it and silently inherit no protection at all. Nothing anywhere else
notices, because every render still succeeds; the damage shows up as the LLM
being evicted.

ComfyUI reports its own argv through /system_stats, so the reserve can simply
be read back rather than assumed. That is the whole trick here.

The reserve is fixed at launch, so the only remedy is a restart -- which is why
this reports loudly and hands over the exact command instead of trying to
repair anything itself.
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from urllib.parse import urlparse


def reserve_status(url: str, timeout: float = 6.0) -> dict:
    """Read the running server's reserve.

    Returns {"up": bool, "reserve_gb": float|None, "argv": [...], "error": str}.
    `reserve_gb` is None when the server is up but was launched without the
    flag -- the case worth shouting about.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats",
                                    timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        return {"up": False, "reserve_gb": None, "argv": [], "error": str(e)}

    argv = [str(a) for a in (data.get("system", {}).get("argv") or [])]
    reserve = None
    for i, a in enumerate(argv):
        if a == "--reserve-vram" and i + 1 < len(argv):
            try:
                reserve = float(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--reserve-vram="):
            try:
                reserve = float(a.split("=", 1)[1])
            except ValueError:
                pass
    return {"up": True, "reserve_gb": reserve, "argv": argv, "error": ""}


def check(url: str, llm_present: bool, engine_dir: str = "<engine>") -> str | None:
    """Return a warning to show the user, or None when all is well.

    Silent when no co-resident LLM is visible: with nothing to protect, an
    unreserved ComfyUI is the correct configuration, and crying wolf there
    would train everyone to ignore the message that matters.
    """
    st = reserve_status(url)
    if not st["up"] or not llm_present or st["reserve_gb"] is not None:
        return None
    return (
        "⚠️ ComfyUI 正在跑,但**啟動時沒有帶 `--reserve-vram`**,而這台機器上有常駐 LLM。\n"
        "   這代表 ComfyUI 會把 LLM 已經佔用的顯存當成可以自由使用的空間,算圖時可能把它擠掉。\n"
        "   reserve 是啟動時決定的,改不了,只能重開 ComfyUI:\n"
        "       python %s/start_comfy.py --force\n"
        "   (它會先量測目前外部佔用量,再據此決定要保留多少。)" % engine_dir
    )


# --------------------------------------------------------------------------
# Running on the card that was picked
# --------------------------------------------------------------------------
# AI Scaler Toolkit's settings only record which card ComfyUI should use
# (selected.json); starting it is left to these skills, so the --reserve-vram
# above is always theirs. That leaves one case to handle here: a ComfyUI already
# up on the card that was picked BEFORE the user changed it. Connecting to it
# would quietly keep rendering on the old card.

def running_device(url: str, timeout: float = 6.0) -> str | None:
    """The running server's device as the toolkit names it: cuda:N, xpu:N, cpu.

    torch numbers the cards it can see, and --cuda-device hides the others, so
    /system_stats says cuda:0 whichever card that is. The index comes from the
    server's own argv instead.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats",
                                    timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    argv = [str(a) for a in (data.get("system", {}).get("argv") or [])]
    devs = data.get("devices") or []
    kind = str(devs[0].get("type", "")) if devs else ""
    if "--cpu" in argv or kind == "cpu":
        return "cpu"
    index = "0"
    if "--cuda-device" in argv:
        i = argv.index("--cuda-device")
        if i + 1 < len(argv):
            index = argv[i + 1]
    return "%s:%s" % (kind, index) if kind else None


def _same_device(running: str, wanted: str) -> bool:
    # An Intel card is narrowed with ONEAPI_DEVICE_SELECTOR, which leaves no trace
    # in argv and makes it xpu:0 to torch, so for xpu only the backend can be told.
    if running.startswith("xpu") and wanted.startswith("xpu"):
        return True
    return running == wanted


def _busy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/queue", timeout=6) as r:
            q = json.loads(r.read().decode())
    except Exception:
        return False
    return bool(q.get("queue_running") or q.get("queue_pending"))


def _stop_listener(url: str, timeout: float = 30.0) -> bool:
    """Stop whatever listens on the server's port; True once the port is free.

    Only the listener: a venv's python.exe is a small launcher that waits on the
    real interpreter, and exits when it does.
    """
    import osmem
    port = urlparse(url).port or 8188
    pid = osmem.listening_pids().get(port)
    if not pid:
        return True
    try:
        os.kill(pid, 15)
    except OSError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port not in osmem.listening_pids():
            return True
        time.sleep(0.5)
    return False


def realign(url: str, wanted: str | None) -> tuple[bool, str]:
    """Stop a running server that is on another card than `wanted`, if it is idle.

    Returns (stopped, message). stopped means the caller should launch the
    protected server afresh; message is for the log either way, empty when
    there was nothing to say. A server that is still rendering is left alone:
    switching would throw that work away, so this render uses it and the switch
    happens on a later call.
    """
    if not wanted:
        return False, ""
    running = running_device(url)
    if not running or _same_device(running, wanted):
        return False, ""
    if _busy(url):
        return False, ("ComfyUI is on %s but %s was picked; it is still working, so this "
                       "job runs on %s and the switch waits until it is idle."
                       % (running, wanted, running))
    if not _stop_listener(url):
        return False, ("ComfyUI is on %s but %s was picked, and it could not be stopped; "
                       "this job runs on %s." % (running, wanted, running))
    return True, "ComfyUI was on %s; %s was picked, so it is restarted there." % (running, wanted)

