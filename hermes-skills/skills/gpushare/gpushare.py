#!/usr/bin/env python3
"""Render on the card the LLM runs on, as well as on the card ComfyUI normally has.

On a box with two cards the LLM keeps one and ComfyUI the other, and the skills need no help for
that. This is for also using the LLM's card: it starts a second ComfyUI there, inside one sleep
window of the LLM.

    gpushare.py run imggen <gen.py args...>     sleep the LLM -> generate -> wake it, all in one call
    gpushare.py run vidgen <gen.py args...>     same for video (always waits for the clip)
    gpushare.py status | sleep | wake           manual control

Unlike engine/llmsleep.py, which lends the GPU behind one render once the agent's turn has ended,
the agent waits on this call: it sleeps at once and keeps the card for several shots, which is why
it returns well inside the agent's terminal limit.

Sleep and wake go through AST (POST /inference/sleep, /inference/wake; GET /inference/status
"sleeping"), not to the engine itself, so any engine AST can put to sleep works. AST also knows the
model is asleep, so a chat request that arrives meanwhile is auto-woken (or refused with 503)
instead of hanging. An engine that cannot sleep gets HTTP 409 with AST's reason.

Whatever happens in between, the LLM is woken before this returns: while it sleeps the agent that
called this cannot think.
"""
import json, os, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

# Children start from this, not from os.environ: importing paths below fills in the variables of the
# card ComfyUI normally has, and a child inheriting them would measure that card instead of its own.
BASE_ENV = os.environ.copy()
HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent


def find_engine():
    """As in imggen's gen.py: the conf or the environment, what install.py wrote, the pack."""
    cands = [os.environ.get("HERMES_COMFY_ENGINE", "").strip()]
    marker = HERE / "engine_dir.txt"
    if marker.is_file():
        cands.append(marker.read_text(encoding="utf-8").strip())
    cands += [str(HERE.parents[1] / "engine"),
              str(Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / "comfy-engine")]
    for c in cands:
        if c and (Path(c).expanduser() / "paths.py").is_file():
            return Path(c).expanduser()
    raise SystemExit("gpushare: the comfy engine was not found; run install.py")


ENGINE = find_engine()
sys.path.insert(0, str(ENGINE))
import paths  # noqa: E402  (resolves hermes-comfy.conf into os.environ)

AST = os.environ.get("GPUSHARE_AST_URL", "http://127.0.0.1:8571")
COMFY_PORT = os.environ.get("GPUSHARE_COMFY_PORT", "8189")
COMFY = f"http://127.0.0.1:{COMFY_PORT}"
# Beside the engine's state, not the skill: a reinstall replaces the skill's directory.
STATE = Path(paths.STATE_DIR) / "gpushare"
SELECTION = os.environ.get("GPUSHARE_SELECTION", str(STATE / "selected.json"))
PY = sys.executable
# Hermes' terminal tool moves anything past 600 s to the background and goes back to the model --
# which is asleep. Stop generating well before that and wake the LLM.
TIMEOUT = int(os.environ.get("GPUSHARE_TIMEOUT", "480"))


def http(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {"raw": raw.decode(errors="replace")}


def gpu_used_mb():
    """(used, total) MiB of the LLM's card, measured in a child that has that card selected."""
    code = f"import json,sys;sys.path.insert(0,{str(ENGINE)!r});import paths,sysmon;print(json.dumps(sysmon.gpu_vram()))"
    try:
        g = json.loads(subprocess.run([PY, "-c", code], env=comfy_env(), capture_output=True, text=True,
                                      timeout=30).stdout)
        return int(g["used_mb"]), int(g["total_mb"])
    except Exception:
        return None, None


def can_sleep():
    """Does this AST have sleep at all? False only when its API says so."""
    try:
        return "/inference/sleep" in http("GET", f"{AST}/openapi.json", timeout=10).get("paths", {})
    except Exception:
        return True


def need_selection():
    """Before anything is put to sleep: the LLM's card has to be described, and AST able to sleep it."""
    if not can_sleep():
        raise SystemExit(json.dumps({"status": "error", "error": "this AI Scaler Toolkit has no /inference/sleep, so "
                                     "the LLM's card cannot be freed. Nothing was changed and the LLM is awake. "
                                     "Render on ComfyUI's own card with imggen / vidgen, or update the toolkit."},
                                    ensure_ascii=False))
    if not Path(SELECTION).is_file():
        raise SystemExit(json.dumps({"status": "error", "error": f"no selection for the LLM's card at {SELECTION}: "
                                     "copy <ComfyUI home>/selected.json there and set device, backend, vendor, "
                                     "index, python and env to that card (see INSTALL.md)"}, ensure_ascii=False))


def ast(method, path, body=None, timeout=30):
    """An AST call; a refusal raises with AST's own reason (409: this load cannot sleep)."""
    try:
        return http(method, f"{AST}{path}", body, timeout)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"{}").get("detail")
        except ValueError:
            detail = None
        raise RuntimeError(f"AST {path}: HTTP {e.code} {detail or e.reason}") from None


def llm_status():
    return ast("GET", "/inference/status")


def llm_engine(st=None):
    """The engine really serving the model (AST's "engine" may name only how it was started)."""
    st = st or llm_status()
    return st.get("served_by") or st.get("engine")


def is_sleeping():
    return bool(llm_status().get("sleeping"))


def _term(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def guard():
    """Before putting the LLM to sleep: SIGTERM (how Hermes stops a command) unwinds through the
    finally that wakes it, and a detached watchdog -- outside this process tree and group, so the
    SIGKILL that follows cannot reach it -- wakes the LLM if this process dies while it sleeps."""
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    subprocess.Popen([PY, str(Path(__file__).resolve()), "_guard", str(os.getpid())], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()


def _guard(owner):
    if os.fork():                       # daemonize: the parent returns at once, we get reparented
        os._exit(0)
    def alive():
        try:
            os.kill(owner, 0)
            return not open(f"/proc/{owner}/stat").read().split()[2] == "Z"
        except OSError:
            return False
    while alive():
        time.sleep(1)
    try:
        if is_sleeping():
            stop_comfy_jobs(); free_comfy(); w, f, ok = wake_llm()
            msg = {"guard": "woke the LLM after its owner died", "owner": owner, "wake_s": w, "llm_ok": ok}
        else:
            return
    except Exception as e:
        msg = {"guard": "wake failed", "owner": owner, "error": str(e)}
    msg["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(msg, ensure_ascii=False) + "\n")


def sleep_llm():
    t0 = time.time()
    # "wait": a generation still running finishes first (AST returns once the engine sleeps)
    ast("POST", "/inference/sleep", {"mode": "wait"}, timeout=300)
    return round(time.time() - t0, 1)


def last_json(text):
    """The last JSON object printed (gen.py prints pretty, multi-line JSON)."""
    dec, found, i = json.JSONDecoder(), None, 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return found
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict) and ("status" in obj or "outputs" in obj):
                found = obj
            i = end
        except ValueError:
            i += 1


def model_id():
    return http("GET", f"{AST}/v1/models")["data"][0]["id"]


def wake_llm():
    t0 = time.time()
    for attempt in range(6):
        try:
            ast("POST", "/inference/wake", timeout=600)
            break
        except RuntimeError as e:
            # the driver's free-VRAM figure lags the release a moment
            if "free VRAM" not in str(e) or attempt == 5:
                raise
            time.sleep(5)
    woke = round(time.time() - t0, 1)
    # prove it answers, not just that the flag flipped
    t1 = time.time()
    r = http("POST", f"{AST}/v1/chat/completions", {
        "model": model_id(), "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}],
        "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    ok = bool(r.get("choices"))
    return woke, round(time.time() - t1, 1), ok


def comfy_env():
    env = dict(BASE_ENV, COMFY_URL=COMFY, COMFY_SELECTION=SELECTION,
               # start_comfy does not take the port from COMFY_URL
               COMFY_EXTRA_ARGS=f"--port {COMFY_PORT}",
               # own logs / job records, apart from the ComfyUI on the other card
               HERMES_COMFY_STATE=str(STATE),
               # the LLM is asleep on this same card: hold nothing back for it
               VIDGEN_LLM_VRAM_BUDGET="0")
    return env


def comfy_up():
    try:
        http("GET", f"{COMFY}/system_stats", timeout=5)
        return True
    except Exception:
        return False


def ensure_comfy():
    """ComfyUI on the shared card. Started only while the LLM sleeps, with --no-wake: start_comfy
    would otherwise wake the LLM to size --reserve-vram against it and leave ComfyUI ~3 GB."""
    if comfy_up():
        return 0.0
    t0 = time.time()
    r = subprocess.run([PY, str(ENGINE / "start_comfy.py"), "--no-wake", "--reserve", "0.6"],
                       env=comfy_env(), capture_output=True, text=True, timeout=600)
    if not comfy_up():
        raise RuntimeError("ComfyUI did not start on the shared GPU: " + (r.stdout + r.stderr)[-800:])
    return round(time.time() - t0, 1)


def stop_comfy_jobs():
    """Cancel whatever ComfyUI is still computing, so it does not hold the card while the LLM wakes."""
    for path, body in (("/queue", {"clear": True}), ("/interrupt", {})):
        try:
            http("POST", f"{COMFY}{path}", body, timeout=10)
        except Exception:
            pass


def comfy_pids():
    """The ComfyUI on the shared card (main.py --port COMFY_PORT)."""
    pids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if any(c.endswith(b"main.py") for c in cmd) and b"--port" in cmd \
                    and cmd[cmd.index(b"--port") + 1:][:1] == [COMFY_PORT.encode()]:
                pids.append(int(d.name))
    return pids


def free_comfy():
    """Unload ComfyUI's models, then end the process: an idle ComfyUI still holds a CUDA context
    (~350 MiB), and an engine may not wake until the VRAM it gave up is free again. ensure_comfy
    starts it afresh in the next sleep window."""
    try:
        http("POST", f"{COMFY}/free", {"unload_models": True, "free_memory": True}, timeout=60)
    except Exception:
        pass
    pids = comfy_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(30):
        if not [p for p in pids if Path(f"/proc/{p}").exists()]:
            break
        time.sleep(0.5)
    else:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    time.sleep(2)


def run(tool, args):
    if tool not in ("imggen", "vidgen"):
        raise SystemExit("usage: gpushare.py run imggen|vidgen <gen.py args...>")
    need_selection()
    if tool == "vidgen" and args and args[0] in ("i2v", "t2v", "r2v") and "--wait" not in args:
        args = args + ["--wait"]            # the LLM must not wake up to an unfinished job
    report = {"tool": tool, "args": args, "llm_engine": llm_engine()}
    before, total = gpu_used_mb()
    report["gpu_mb_llm_awake"] = before
    t_all = time.time()
    gen = None
    try:
        guard()
        report["sleep_s"] = sleep_llm()
        report["gpu_mb_llm_asleep"] = gpu_used_mb()[0]
        report["comfy_start_s"] = ensure_comfy()
        t0 = time.time()
        budget = max(30, TIMEOUT - (time.time() - t_all))
        try:
            p = subprocess.run([PY, str(SKILLS / tool / "gen.py"), *args], env=comfy_env(),
                               capture_output=True, text=True, timeout=budget)
        except subprocess.TimeoutExpired:
            stop_comfy_jobs()
            raise RuntimeError(f"generation did not finish within {int(budget)} s (GPUSHARE_TIMEOUT={TIMEOUT}); "
                               "cancelled it and woke the LLM. Use a smaller size (--short 480), fewer "
                               "seconds or a faster model.")
        report["generate_s"] = round(time.time() - t0, 1)
        gen = last_json(p.stdout)
        if gen is not None and gen.get("status") == "done" and gen.get("file"):
            gen = dict(gen, status="ok", outputs=[gen["file"]])   # vidgen says "done"
        if gen is None:
            gen = {"status": "error", "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-1500:]}
        report["gpu_mb_peak_note"] = "see nvidia-smi during generate"
    except (Exception, KeyboardInterrupt) as e:
        gen = {"status": "error", "error": str(e) or type(e).__name__}
    finally:
        if gen is None or gen.get("status") != "ok":
            stop_comfy_jobs()
        free_comfy()
        report["gpu_mb_after_free"] = gpu_used_mb()[0]
        try:
            report["wake_s"], report["first_token_s"], report["llm_ok"] = wake_llm()
        except Exception as e:
            report["wake_error"] = str(e)
            report["llm_ok"] = False
        report["gpu_mb_llm_awake_again"] = gpu_used_mb()[0]
    report["total_s"] = round(time.time() - t_all, 1)
    report["result"] = gen
    report["status"] = "ok" if (gen or {}).get("status") == "ok" and report.get("llm_ok") else "error"
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        with open(STATE / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
    except OSError:
        pass
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report["status"] == "ok" else 1


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); return 2
    if a[0] == "status":
        st = llm_status()
        print(json.dumps({"llm_engine": llm_engine(st), "llm_sleeping": bool(st.get("sleeping")), "can_sleep": can_sleep(),
                          "comfy_on_shared_gpu": comfy_up(),
                          "gpu_used_mb": gpu_used_mb()[0], "gpu_total_mb": gpu_used_mb()[1]})); return 0
    if a[0] == "sleep":
        print(json.dumps({"sleep_s": sleep_llm(), "gpu_used_mb": gpu_used_mb()[0]})); return 0
    if a[0] == "wake":
        free_comfy(); w, f, ok = wake_llm()
        print(json.dumps({"wake_s": w, "first_token_s": f, "llm_ok": ok, "gpu_used_mb": gpu_used_mb()[0]})); return 0
    if a[0] == "_guard" and len(a) == 2:
        _guard(int(a[1])); return 0
    if a[0] == "run" and len(a) >= 2:
        return run(a[1], a[2:])
    if a[0] in ("dispatch", "jobs", "wait", "assemble") and len(a) == 2:
        sys.path.insert(0, str(HERE))
        import dispatch as d
        return {"dispatch": d.dispatch, "jobs": d.poll, "wait": d.wait, "assemble": d.assemble}[a[0]](a[1])
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main())
