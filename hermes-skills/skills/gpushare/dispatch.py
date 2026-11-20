"""Two cards at once: the shots of one plan go to both, and come back as one film.

    gpushare.py dispatch <plan.json>     start the "render" shots in the background, then sleep the
                                         LLM, render the "llm" shots on its card and wake it -- one
                                         call, well under the agent's terminal limit
    gpushare.py wait <plan.json>         wait up to 500 s; joins the clips when every shot is done.
                                         On status "running", run it again
    gpushare.py jobs <plan.json>         where the shots are, without waiting
    gpushare.py assemble <plan.json>     join the clips (wait does this)

plan.json is written by the agent:

{"title": "...", "size": "1280:704",
 "shots": [{"id": "s1", "gpu": "render", "kind": "video", "args": ["r2v", "<prompt>", "--ref", "a.png",
                                                                  "--short", "720", "--seconds", "5"]},
           {"id": "s2", "gpu": "llm", "kind": "image", "args": ["qwen21", "<prompt>", "--ref", "a.png"]}]}

"gpu" is "render" (the card ComfyUI normally has) or "llm" (the LLM's card); "args" is what gen.py of
imggen (kind image) or vidgen (kind video) takes. --wait and --out are added here. "render" shots run
one after another in a detached runner: two at once do not fit on a card, and vidgen refuses the
second. "llm" shots run meanwhile, inside one sleep window. Keep them small: the window ends at
GPUSHARE_TIMEOUT. "size" is the film's frame (default 1280:704).

GPUSHARE_PRESET names a Python file for a production with its own vocabulary, used for the shots that
have no "args". It may define ALIASES ({its card name: "render" | "llm"}), validate(shot) -> list of
errors, args(shot, index, out_dir) -> (tool, gen.py args), and video_filter(shot) -> filters to run
on a clip, ending in a comma, and finish(shot, file, out_dir) -> fields to merge into a finished shot's record.
"""
import importlib.util, json, os, subprocess, time
from pathlib import Path

import gpushare as gs

RENDER_ENV = {k: v for k, v in gs.BASE_ENV.items() if not k.startswith(("COMFY_", "HERMES_COMFY_", "VIDGEN_"))}
TOOLS = {"video": "vidgen", "image": "imggen"}


def preset():
    path = os.environ.get("GPUSHARE_PRESET", "").strip()
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("gpushare_preset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRESET = preset()
ALIASES = getattr(PRESET, "ALIASES", {})


def role(s):
    return ALIASES.get(s.get("gpu"), s.get("gpu"))


def preset_hook(s, name):
    """The preset's function for this shot; a shot that spells out its args never uses one."""
    return None if "args" in s else getattr(PRESET, name, None)


def shot_errors(s):
    custom = preset_hook(s, "validate")
    if custom:
        return custom(s)
    args = s.get("args")
    if not (isinstance(args, list) and args and all(isinstance(a, str) for a in args)):
        return ["args must be a non-empty list of strings: what gen.py takes"]
    return []


def load(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    shots = plan.get("shots") or []
    errs = []
    if not shots:
        errs.append("plan has no shots")
    for i, s in enumerate(shots):
        tag = s.get("id") or f"#{i + 1}"
        if role(s) not in ("render", "llm"): errs.append(f"{tag}: gpu must be one of {sorted({'render', 'llm', *ALIASES})}")
        if s.get("kind") not in TOOLS: errs.append(f"{tag}: kind must be video or image")
        errs += [f"{tag}: {e}" for e in shot_errors(s)]
        s.setdefault("id", f"s{i + 1}")
    if errs:
        raise SystemExit(json.dumps({"status": "error", "plan_errors": errs}, ensure_ascii=False, indent=1))
    return plan


def shot_args(s, idx, out_dir):
    custom = preset_hook(s, "args")
    tool, args = custom(s, idx, out_dir) if custom else (TOOLS[s["kind"]], list(s["args"]))
    if tool == "vidgen" and "--wait" not in args:
        args = args + ["--wait"]
    if tool == "imggen" and "--out" not in args:
        args = args + ["--out", str(out_dir)]
    return tool, args


def finish(s, rec, out_dir):
    """Let the preset turn a finished shot's output into what the plan asked for, e.g. a
    poster taken from a short clip. The preset's finish(shot, file, out_dir) returns fields
    to merge into the record (normally a new "file")."""
    hook = preset_hook(s, "finish")
    if hook and rec.get("status") == "done":
        try:
            rec.update(hook(s, rec["file"], out_dir) or {})
        except Exception as e:
            rec.update(status="error", error=f"finish: {e}")
    return rec


def state_path(plan_path):
    return Path(plan_path).with_suffix(".state.json")


def render_path(plan_path):
    return Path(plan_path).with_suffix(".render.json")


def result(p, t0):
    j = gs.last_json(p.stdout) or {}
    f = j.get("file") or (j.get("outputs") or [None])[0]
    return {"status": "done" if f and j.get("status") in ("ok", "done") else "error", "file": f,
            "seconds": round(time.time() - t0, 1), "error": None if f else (p.stdout + p.stderr)[-600:]}


def dispatch(plan_path):
    plan = load(plan_path)
    out_dir = Path(plan_path).resolve().parent
    st = {"plan": str(plan_path), "title": plan.get("title"), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "shots": {}}
    t_all = time.time()
    local = [(i, s) for i, s in enumerate(plan["shots"]) if role(s) == "llm"]
    if local:
        gs.need_selection()
    # a re-run of the same plan path must not pick up the last run's results or film
    for stale in (render_path(plan_path), out_dir / "final.mp4"):
        stale.unlink(missing_ok=True)
    # 1) the shots on ComfyUI's own card
    for s in plan["shots"]:
        if role(s) == "render":
            st["shots"][s["id"]] = {"gpu": s["gpu"], "kind": s["kind"], "status": "queued"}
    state_path(plan_path).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    if any(role(s) == "render" for s in plan["shots"]):
        subprocess.Popen([gs.PY, str(Path(__file__).resolve()), "_render", str(plan_path)], env=RENDER_ENV,
                         stdout=open(Path(plan_path).with_suffix(".render.log"), "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    # 2) the shots on the LLM's card, all inside one sleep window
    win = {"gpu_mb_llm_awake": gs.gpu_used_mb()[0], "llm_engine": gs.llm_engine()}
    if local:
        try:
            gs.guard()
            win["sleep_s"] = gs.sleep_llm()
            win["gpu_mb_llm_asleep"] = gs.gpu_used_mb()[0]
            gs.ensure_comfy()
            for i, s in local:
                tool, args = shot_args(s, i, out_dir)
                budget = max(30, gs.TIMEOUT - (time.time() - t_all))
                t0 = time.time()
                rec = {"gpu": s["gpu"], "kind": s["kind"], "tool": tool}
                try:
                    p = subprocess.run([gs.PY, str(gs.SKILLS / tool / "gen.py"), *args], env=gs.comfy_env(),
                                       capture_output=True, text=True, timeout=budget)
                    rec.update(result(p, t0))
                    finish(s, rec, out_dir)
                except subprocess.TimeoutExpired:
                    gs.stop_comfy_jobs()
                    rec.update(status="error", error=f"did not finish inside the sleep window ({int(budget)} s)")
                st["shots"][s["id"]] = rec
                if rec["status"] != "done":
                    break
        finally:
            # anything not finished (error, timeout, or interrupted mid-shot) must stop before waking
            if len([1 for _, s in local if st["shots"].get(s["id"], {}).get("status") == "done"]) < len(local):
                gs.stop_comfy_jobs()
            gs.free_comfy()
            try:
                win["wake_s"], win["first_token_s"], win["llm_ok"] = gs.wake_llm()
            except Exception as e:
                win.update(llm_ok=False, wake_error=str(e))
            win["gpu_mb_llm_awake_again"] = gs.gpu_used_mb()[0]
    st["sleep_window"] = win
    st["dispatch_s"] = round(time.time() - t_all, 1)
    state_path(plan_path).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return report(st, plan_path)


def render_runner(plan_path):
    plan = load(plan_path)
    out_dir = Path(plan_path).resolve().parent
    res = {}
    for i, s in enumerate(plan["shots"]):
        if role(s) != "render":
            continue
        res[s["id"]] = {"status": "running", "started": time.time()}
        render_path(plan_path).write_text(json.dumps(res, ensure_ascii=False))
        tool, args = shot_args(s, i, out_dir)
        t0 = time.time()
        p = subprocess.run([gs.PY, str(gs.SKILLS / tool / "gen.py"), *args], env=RENDER_ENV, capture_output=True, text=True)
        res[s["id"]] = finish(s, result(p, t0), out_dir)
        render_path(plan_path).write_text(json.dumps(res, ensure_ascii=False))


def refresh(plan_path):
    """The state file with what the detached runner has finished folded in."""
    st = json.loads(state_path(plan_path).read_text())
    try:
        res = json.loads(render_path(plan_path).read_text())
    except (OSError, ValueError):
        res = {}
    for sid, r in res.items():
        if sid in st["shots"]:
            st["shots"][sid].update({k: v for k, v in r.items() if k != "started"})
    state_path(plan_path).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return st


def poll(plan_path):
    return report(refresh(plan_path), plan_path)


def wait(plan_path, limit=500):
    """Poll quietly for up to `limit` s (under the agent's 600 s per call); when every shot is done,
    join the clips too. One answer at the end: ok (with final) or running (run wait again)."""
    t0 = time.time()
    while True:
        st = refresh(plan_path)
        states = [r.get("status") for r in st["shots"].values()]
        if "error" in states:
            return report(st, plan_path)
        if all(x == "done" for x in states):
            return assemble(plan_path)
        if time.time() - t0 > limit:
            return report(st, plan_path, note=f"still rendering after {limit} s; run wait again")
        time.sleep(10)


def has_audio(path):
    return bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip())


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                                capture_output=True, text=True).stdout)


def assemble(plan_path):
    st = json.loads(state_path(plan_path).read_text())
    plan = load(plan_path)
    shots = [s for s in plan["shots"] if s["kind"] == "video"]
    if not shots and all(r.get("status") == "done" for r in st["shots"].values()):
        return report(st, plan_path)            # images only: there is no film to join
    vids = [st["shots"][s["id"]].get("file") for s in shots]
    if not vids or not all(vids) or any(st["shots"][s["id"]].get("status") != "done" for s in plan["shots"]):
        return report(st, plan_path, note="not every shot is done yet; run jobs first")
    out = Path(plan_path).resolve().parent / "final.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for v in vids:
        cmd += ["-i", v]
    size = plan.get("size", "1280:704")
    fl = ""
    for k, (s, v) in enumerate(zip(shots, vids)):
        prefilter = preset_hook(s, "video_filter")
        fl += f"[{k}:v]{prefilter(s) if prefilter else ''}scale={size},setsar=1,fps=24[v{k}];"
        # concat needs a sound track from every clip, and not every model makes one
        fl += (f"[{k}:a]aresample=48000[a{k}];" if has_audio(v) else
               f"anullsrc=r=48000:cl=stereo,atrim=duration={duration(v)}[a{k}];")
    fl += "".join(f"[v{k}][a{k}]" for k in range(len(vids))) + f"concat=n={len(vids)}:v=1:a=1[v][a]"
    subprocess.run(cmd + ["-filter_complex", fl, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20",
                          "-c:a", "aac", str(out)], check=True)
    st["final"] = str(out)
    state_path(plan_path).write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return report(st, plan_path)


def report(st, plan_path, note=None):
    shots = st["shots"]
    out = {"title": st.get("title"), "status": "ok" if all(r.get("status") == "done" for r in shots.values()) else
                     ("error" if any(r.get("status") == "error" for r in shots.values()) else "running"),
           "shots": {k: {x: r.get(x) for x in ("gpu", "kind", "status", "file", "job_id", "seconds", "error") if r.get(x)}
                     for k, r in shots.items()},
           "sleep_window": st.get("sleep_window"), "dispatch_s": st.get("dispatch_s"), "final": st.get("final")}
    film = any(r.get("kind") == "video" for r in shots.values())
    if out["status"] == "running" or (out["status"] == "ok" and film and not st.get("final")):
        out["next"] = f"{gs.PY} {gs.HERE / 'gpushare.py'} wait {plan_path}"
    if note:
        out["note"] = note
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if out["status"] != "error" else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "_render":
        render_runner(sys.argv[2])
