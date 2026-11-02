#!/usr/bin/env python3
"""recast -- reference video + character photos -> the same video with the people replaced.

    recast.py plan     <video> [--name N] [--project DIR]   shot list + contact sheet + spec.json
    recast.py render   <project> [--shots 1,3] [--tries N] [--fresh]
    recast.py assemble <project>                            (render runs this itself)
    recast.py status   <project>

Built on what a 13.7 s, 4-shot recast taught (2026-10-28):

* ONE vidgen job PER SHOT, each with its own wording, beats one job for the
  whole clip. A single generic swap prompt left the back-view man unswapped,
  re-framed close-ups and flattened every expression; per-shot wording fixed
  all three. Shots nobody is swapped in are copied from the source, not rendered.
* Two people in one shot = two passes, the second one following the first
  pass's output as its reference video.
* The failures that remain are SEED LUCK, and they are measurable, so this
  checks every render and re-rolls the seed:
    leak       H3 pastes the reference PHOTO's scenery (Okinawa sea, a
               restaurant) over the shot. Background colours closer to the
               photo than to the source -> leak. Calibrated on 15 renders
               judged by eye: every leak < 1.08, every good one >= 1.33.
    lag        the first frames after a cut still show the original person
               (seen: 9 frames, 2 frames). A frame closer to the source than
               to its own render 12 frames later has not swapped yet.
    unchanged  nothing was replaced at all.
  Up to 3 lag frames are bridged at assembly by holding the first swapped
  frame; more than that, or any leak, is re-rolled.
* Do NOT give H3 a tight face crop as a second reference, and do NOT pad the
  shot with a frozen lead-in: both made it copy the photo's scene and zoom in.

recast never talks to ComfyUI: every render goes through vidgen/gen.py, so
vidgen's memory protection, LLM hand-over and refusals apply unchanged.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent
PY = sys.executable or "python"
VIDGEN = Path(os.environ.get("RECAST_VIDGEN") or (SKILLS / "vidgen" / "gen.py"))

FPS = 24                      # H3's fixed rate; every frame index here is on this grid
REF_VIDEO_SIZE = 512          # 0.51 motion agreement vs 0.28 at vidgen's default 384
STEPS = 8
SEEDS = (20261028, 4242, 7, 1234, 99)
MAX_HOLD = 3                  # lag frames bridged by a hold; more -> re-roll
LEAK_RATIO = 1.3              # top-band d(photo) / d(source) below this = the photo's scenery
                              # leaked. 24 renders judged by eye: leaks <= 1.06, good >= 1.48.
                              # (A full ring including the side strips misread close-ups: a
                              # black T-shirt filling the frame edges looked like the photo.)
UNCHANGED = 4.0               # mean centre difference below this = nothing was swapped
REFRAME_KEEP = 0.12           # share of the frame still near the source, AND
REFRAME_LAYOUT = 0.30         # coarse-layout correlation: both below = re-composed shot.
                              # 4 videos: the re-composed render 0.07 / 0.14; good renders
                              # >= 0.21 keep (close-ups) or >= 0.45 layout
JUMP = 20.0                   # frame-to-frame jump of the render beyond the source's own:
                              # 4 videos, 20 renders: good <= 12.2; H3's invented cuts
                              # (orbit -> jump cut, push-in -> talking head, chunk seam) 35-52
SHORT_EDGE = 720
MIN_SHOT = 39                 # frames; vidgen's H3_MIN_SEGMENT -- shorter shots are merged
JOB_RE = re.compile(r"job_id=([0-9a-fA-F]{6,})")
# vidgen errors that say "the engine is unavailable", not "this seed failed".
INFRA_RE = re.compile(r"VRAM|\bRAM\b|refus|start_comfy|did not become ready|timed out|"
                      r"Connection refused|ComfyUI (is )?(down|not running)|URLError|"
                      r"WinError 10061", re.I)
INFRA_HINT = ("rendering stopped because ComfyUI is unavailable (out of memory, crashed or "
              "not starting) -- not a bad seed. Check `vidgen gen.py status`; if another "
              "program holds the GPU or ComfyUI has crashed, tell the user what holds it "
              "(do not kill processes yourself), then re-run render: finished shots are kept "
              "and no seed was used up.")


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    try:
        print(line, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # The terminal that launched us went away (a headless agent session
        # closed its pipes on exit). The render must carry on: one such write
        # killed a run between two passes. The log file still gets the line.
        pass
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def out(obj):
    try:
        print(json.dumps(obj, ensure_ascii=False, indent=1))
    except (OSError, ValueError):
        pass                      # nobody is reading stdout any more; state.json has it


def agent_home():
    """Same decision dramagen makes: the engine's paths.AGENT_HOME, else a local default."""
    env = os.environ.get("HERMES_COMFY_ENGINE", "").strip()
    cands = [Path(env)] if env else []
    marker = HERE / "engine_dir.txt"
    if marker.is_file():
        cands.append(Path(os.path.expanduser(marker.read_text(encoding="utf-8").strip())))
    # A linked install (install.py --link) runs this file straight from the pack,
    # whose engine/ is two levels up; nothing is written next to the script.
    cands.append(HERE.parents[1] / "engine")
    for c in cands:
        if (c / "paths.py").is_file():
            try:
                sys.path.insert(0, str(c))
                import paths  # noqa: E402
                return Path(paths.AGENT_HOME)
            except Exception:
                pass
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(os.environ.get("HERMES_HOME") or Path(local) / "hermes")
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def need(tool):
    p = shutil.which(tool)
    if not p:
        raise SystemExit(json.dumps({"ok": False, "error":
                                     "%s not found on PATH; recast needs ffmpeg and ffprobe" % tool}))
    return p


def ff(*args):
    p = subprocess.run([need("ffmpeg"), "-v", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("ffmpeg failed: " + p.stderr[-400:])


def probe(path):
    p = subprocess.run([need("ffprobe"), "-v", "error", "-show_entries",
                        "stream=codec_type,width,height:stream_tags=rotate:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(p.stdout)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    if str((v.get("tags") or {}).get("rotate", "0")) in ("90", "270", "-90"):
        w, h = h, w
    return {"w": w, "h": h, "duration": float(d["format"]["duration"]),
            "audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def run_json(cmd, log_path, on_line=None):
    """Run vidgen/gen.py: JSON on stdout, human text on stderr (drained live,
    so a job_id survives this process being killed mid-render)."""
    log("run: " + " ".join('"%s"' % c if " " in str(c) else str(c) for c in cmd), log_path)
    t0 = time.time()
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def pump():
        for raw in iter(p.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                log("  | " + line, log_path)
                if on_line:
                    try:
                        on_line(line)
                    except Exception:
                        pass

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    text = p.stdout.read().decode("utf-8", "replace").strip()
    p.wait()
    th.join(timeout=5)
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {"ok": False, "error": text[:500] or "no output"}
    data["_seconds"] = round(time.time() - t0, 1)
    return data


def wait_job(job_id, log_path, poll=30):
    """Re-attach to a vidgen job a killed run had already submitted.

    vidgen's worker is detached, so the render outlives recast. Returns the
    finished job's JSON, or None when it is gone or failed (render again).
    """
    log("re-attaching to vidgen job %s (still rendering, not re-submitting)" % job_id, log_path)
    while True:
        res = run_json([PY, VIDGEN, "status", job_id], log_path)
        st = res.get("status")
        if st == "done" and res.get("file"):
            return res
        if st != "running":
            log("job %s is %s; rendering that pass again" % (job_id, st), log_path)
            return None
        time.sleep(poll)


# ------------------------------------------------------------------ project


class Project:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.spec_path = self.root / "spec.json"
        self.state_path = self.root / "state.json"
        self.log = self.root / "recast.log"
        self.src = self.root / "src.mp4"
        self.src24 = self.root / "src24.mp4"

    def spec(self):
        if not self.spec_path.is_file():
            raise SystemExit(json.dumps({"ok": False, "error": "no spec.json in %s -- run plan first"
                                         % self.root}, ensure_ascii=False))
        return json.loads(self.spec_path.read_text(encoding="utf-8"))

    def state(self):
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"shots": {}}

    def save_state(self, st):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)


def frame_range(shot):
    return int(round(shot["start"] * FPS)), int(round(shot["end"] * FPS))


# ------------------------------------------------------------------- frames

def read_frames(path, w, h, resample=True):
    import numpy as np
    vf = ("fps=%d," % FPS if resample else "") + "scale=%d:%d:flags=area,format=rgb24" % (w, h)
    raw = subprocess.run([need("ffmpeg"), "-v", "error", "-i", str(path), "-vf", vf,
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32)


def _masks(w, h):
    import numpy as np
    ring = np.ones((h, w), bool)                   # top band + side strips: background
    ring[int(h * .15):, int(w * .2):int(w * .8)] = False
    top = np.zeros((h, w), bool)                   # above the heads: the scenery
    top[:int(h * .18), :] = True
    centre = np.zeros((h, w), bool)                # where the subject usually is
    centre[int(h * .1):int(h * .6), int(w * .25):int(w * .75)] = True
    return ring, centre, top


def _hist(px):
    import numpy as np
    q = (px.reshape(-1, 3) // 64).astype(int)
    hgm = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(float)
    return hgm / max(hgm.sum(), 1.0)


def check_render(render, src_frames, dims, a, b, photos, faces=True):
    """Score one render of shot [a, b) against the source and the character photos.

    faces=False (every swap in the shot is a back view or a tiny figure): the
    "still close to the original" half of the lag test is skipped. A dark
    back-of-head silhouette changes the centre so little that a back view
    replaced from frame 0 measured as 11 frames of lag.
    """
    import numpy as np
    from PIL import Image
    w, h = dims
    ring, centre, top = _masks(w, h)
    r = read_frames(render, w, h, resample=False)
    b = min(b, len(r), len(src_frames))
    if b - a < 2:
        return {"ok": False, "why": "render too short to check"}
    s = src_frames
    refs = [_hist(np.asarray(Image.open(p).convert("RGB").resize((w, h)), np.float32)[top])
            for p in photos]

    d_src, d_ref, cdiff, bdiff = [], [], [], []
    for t in range(a, b):
        cdiff.append(float(np.abs(r[t] - s[t]).mean(-1)[centre].mean()))
        bdiff.append(float(np.abs(r[t] - s[t]).mean(-1)[ring].mean()))
        if (t - a) % 3 == 0:
            hr = _hist(r[t][top])
            d_src.append(float(np.abs(hr - _hist(s[t][top])).sum()))
            d_ref.append(min(float(np.abs(hr - x).sum()) for x in refs) if refs else 9.0)
    leak_ratio = float(np.median(d_ref)) / max(float(np.median(d_src)), 1e-3)
    changed = float(np.median(cdiff))

    # Re-framed: H3 composed its own shot (a close-up of the new person, the
    # other people gone). Then almost no part of the frame is still near the
    # source AND the coarse layout no longer matches. Either alone misreads a
    # close-up, where the new person legitimately fills most of the frame.
    def coarse(x):
        g = x.mean(-1)
        hh, ww = (g.shape[0] // 6) * 6, (g.shape[1] // 6) * 6
        return g[:hh, :ww].reshape(hh // 6, 6, ww // 6, 6).mean((1, 3)).ravel()
    keep, corr = [], []
    for t in range(a, b, 3):
        keep.append(float((np.abs(r[t] - s[t]).mean(-1) < 15).mean()))
        cc = np.corrcoef(coarse(r[t]), coarse(s[t]))[0, 1]
        corr.append(float(cc) if np.isfinite(cc) else 0.0)
    keep, corr = float(np.median(keep)), float(np.median(corr))

    # Invented cut: the render jumps between two frames where the source moves
    # smoothly. Seen three ways in one 16 s video: a smooth orbit turned into a
    # jump cut, a push-in past the presenter ending on a talking head that is
    # not in the source, and a re-composed second half where vidgen split a long
    # shot into two renders. The source's own cuts cancel out (r - s).
    jump = max((float(np.abs(r[t] - r[t - 1]).mean()) - float(np.abs(s[t] - s[t - 1]).mean())
                for t in range(a + 1, b)), default=0.0)

    # A lag frame is either still close to the original (well under the shot's
    # typical change, AND nearer the source than to its own render a moment
    # later) or part of the jump right after the cut (far more change to its
    # own future than the shot ever shows again). Measured: catches the 9- and
    # 2-frame lags and a 5-frame background lag. The 0.6 is what keeps a crane
    # shot over a back view honest: its frame-to-future difference is large
    # all the way through, which alone flagged 107/122 frames.
    # A third kind: the person is already new but the BACKGROUND is still the
    # previous shot's for a few frames (seen across a 1-frame dissolve: 5 dark
    # frames before a lit stage). The centre barely moves there, the ring does.
    k = 12                                          # "its own render a moment later"
    fut = [float(np.abs(r[t] - r[t + k]).mean(-1)[centre].mean()) for t in range(a, b - k)]
    lag = 0
    if fut:
        med_fut = float(np.median(fut))
        med_bg = float(np.median(bdiff))
        for i, t in enumerate(range(a, b - k)):
            still_original = faces and cdiff[i] < fut[i] and cdiff[i] < 0.6 * changed
            settling = fut[i] > 1.8 * max(med_fut, 1.0)
            old_scene = bdiff[i] > 2.0 * max(med_bg, 8.0)
            if still_original or settling or old_scene:
                lag += 1
            else:
                break
    res = {"leak_ratio": round(leak_ratio, 2), "changed": round(changed, 1), "lag": lag,
           "keep": round(keep, 2), "layout": round(corr, 2), "jump": round(jump, 1)}
    problems = []
    if changed < UNCHANGED:
        problems.append("unchanged: nobody was replaced")
    if keep < REFRAME_KEEP and corr < REFRAME_LAYOUT:
        problems.append("reframed: the shot was re-composed (zoomed in, people missing)")
    if jump > JUMP:
        problems.append("jump: a cut or sudden jump that is not in the source video")
    if leak_ratio < LEAK_RATIO:
        problems.append("leak: background looks like the character photo, not the video")
    if lag > MAX_HOLD:
        problems.append("lag: first %d frames still show the original person" % lag)
    res["ok"] = not problems
    res["problems"] = problems
    return res


# ------------------------------------------------------------------ prompts

PRON = {"he": ("He", "his"), "she": ("She", "her"), "they": ("They", "their")}


def build_prompt(swap, char, shot_scene="", shot_camera=""):
    """The text appended to vidgen's own swap sentence for one pass.

    vidgen already says "Replace <who> in <Video 1> with the person from
    <Picture 1>. Keep everyone else ... same motion, positions, poses, timing
    and camera". Everything here is what that sentence was measured to miss.
    """
    looks = (char.get("looks") or "").strip()
    he, his = PRON.get((char.get("pronoun") or "they").lower(), PRON["they"])
    view = (swap.get("view") or "front").lower()
    clothes = (swap.get("clothes") or "own").strip()
    # Name the clothes. "the clothes from <Picture 1>" was too abstract: in the
    # first end-to-end run H3 kept the dancer's apron in two shots, while the
    # hand-written "black T-shirt" of the earlier run replaced it every time.
    outfit = (char.get("outfit") or "").strip()
    if clothes.lower() == "own":
        wear = ("wearing %s instead of the original clothes" % outfit if outfit
                else "wearing the same clothes as in <Picture 1>")
    elif clothes.lower() == "keep":
        wear = "still wearing the clothes from <Video 1>"
    else:
        wear = "wearing %s instead of the original clothes" % clothes
    # A concrete description of the video's place (the shot's `scene`) is
    # what held the stage in the hand-written prompt that worked; it wins over
    # the photo's scenery better than a generic "same background".
    # Positive wording only. The first version added "not the place in
    # <Picture 1>" and that very run pasted the photo's restaurant behind the
    # man; the same seed without the sentence kept the venue. Naming the photo's
    # scenery, even to forbid it, puts it in the prompt.
    where = (shot_scene or "").strip().rstrip(".")
    if where.startswith("<"):               # the plan template's placeholder, never filled
        where = ""
    scene = (("%s stays %s, with the same background and lighting as <Video 1>. " % (he, where))
             if where else
             "The background, lighting and location stay exactly as in <Video 1>. ")
    # "do not zoom in" stops H3 from cropping a static shot into a close-up,
    # but it contradicts a source that pushes in or orbits; there H3 replaced
    # the move with a jump cut (or a talking head once the person had left the
    # frame). A moving shot gets its move spelled out instead.
    camera = (shot_camera or "").strip().rstrip(".")
    if camera.startswith("<"):
        camera = ""
    if camera:
        scene += ("Same framing as <Video 1>, and the camera is exactly as in <Video 1>: %s, "
                  "one continuous shot." % camera)
    else:
        scene += "Same framing and camera movement as <Video 1>, do not zoom in."
    parts = []
    if view == "back":
        # Face words are useless from behind and only invite H3 to turn the
        # head; hair and build are what read as the person here.
        parts.append("%s has the hair and build of the person in <Picture 1>, not the "
                     "original hair. %s keeps %s back to the camera the whole time, %s."
                     % (swap["who"][0].upper() + swap["who"][1:], he, his, wear))
    elif view == "small":
        parts.append("%s becomes the person from <Picture 1>, %s. Everyone else "
                     "stays exactly as they are in <Video 1>." % (swap["who"].capitalize(), wear))
    else:
        # "becomes", not "From the very first frame X is the person from
        # <Picture 1>": that opening pulled frame 0 toward the photo itself and
        # all three seeds of the close-up shot came back in the photo's
        # restaurant; the hand-written "He becomes the person from <Picture 1>"
        # kept the venue on the same seed. Lag at the cut is the checks' job.
        parts.append("%s becomes the person from <Picture 1>%s, %s."
                     % (swap["who"][0].upper() + swap["who"][1:],
                        (": " + looks) if looks else "", wear))
        expr = (swap.get("expression") or "").strip()
        parts.append(("Facial expressions follow <Video 1>: %s." % expr.rstrip(".")) if expr
                     else "Facial expressions follow <Video 1> moment by moment.")
    if swap.get("extra"):
        parts.append(swap["extra"].strip())
    parts.append(scene)
    return " ".join(parts)


# ---------------------------------------------------------------- commands


def cmd_plan(a):
    need("ffmpeg")
    video = Path(a.video).expanduser().resolve()
    if not video.is_file():
        return out({"ok": False, "error": "video not found: %s" % video})
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", a.name or video.stem).strip("_") or "recast"
    root = Path(a.project) if a.project else agent_home() / "recast" / (
        "%s_%s" % (name, time.strftime("%m%d_%H%M")))
    pj = Project(root)
    pj.root.mkdir(parents=True, exist_ok=True)
    (pj.root / "sheets").mkdir(exist_ok=True)
    shutil.copy2(video, pj.src)
    meta = probe(pj.src)
    ff("-i", pj.src, "-vf", "fps=%d" % FPS, "-c:v", "libx264", "-crf", "14",
       "-pix_fmt", "yuv420p", "-c:a", "aac", pj.src24)
    probe_png = pj.root / "_probe.png"
    ff("-i", pj.src, "-frames:v", "1", probe_png)

    # Shots come from vidgen's cut detection, NOT from its chunk list: chunk
    # length follows the VRAM free at that moment, and a plan made while
    # another render held the GPU sliced a 4-shot clip into 12 x 1.3 s pieces.
    # At render time vidgen splits a shot longer than its current chunk limit
    # by itself.
    res = run_json([PY, VIDGEN, "plan", "--ref-video", pj.src, "--swap",
                    "the person=%s" % probe_png, "--ref-video-size", REF_VIDEO_SIZE,
                    "--steps", STEPS], pj.log)
    chunks = res.get("chunks") or []
    if not chunks:
        return out({"ok": False, "error": "vidgen plan returned no shots: %s"
                    % (res.get("error") or res.get("plan") or res)})
    shots = _shots_from_cuts(res.get("cuts") or [], chunks)

    sheet = pj.root / "sheets" / "shots.jpg"
    _shot_sheet(pj.src24, shots, meta, sheet)
    spec = {
        "video": str(pj.src),
        "characters": {"<name>": {"image": "<head-and-shoulders photo>",
                                  "pronoun": "he|she|they",
                                  "looks": "<hair, glasses, face shape, build -- what makes them THEM>",
                                  "outfit": "<the clothes in the photo, e.g. a black T-shirt with a white print>"}},
        "shots": [{"id": s["id"], "start": s["start"], "end": s["end"],
                   "scene": "<where this shot happens, e.g. on a dark concert stage under a spotlight>",
                   "camera": "<only if the camera moves: e.g. the camera pushes in past him onto the screen; leave empty for a static shot>",
                   "swaps": []}
                  for s in shots],
        "_swap_template": {"character": "<name>", "who": "<how to spot the person in THIS shot>",
                           "view": "front|back|small", "clothes": "own|keep|<text>",
                           "expression": "<optional: the expression arc, in time order>"},
    }
    if not pj.spec_path.exists():
        pj.spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    probe_png.unlink(missing_ok=True)
    out({"ok": True, "project": str(pj.root), "spec": str(pj.spec_path),
         "sheet": str(sheet), "duration_s": round(meta["duration"], 2),
         "size": "%dx%d" % (meta["w"], meta["h"]), "audio": meta["audio"],
         "shots": [{"id": s["id"], "start": s["start"], "end": s["end"],
                    "est_min_per_pass": round((s["est_s_per_pass"] or 0) / 60, 1)} for s in shots],
         "next": "Look at the sheet, fill characters + each shot's swaps in spec.json, "
                 "then run render (in the background)."})


def _shots_from_cuts(cuts, chunks):
    """Shot list from the cut times, short shots merged the way vidgen merges them.

    A shot under MIN_SHOT frames rides with the previous one (vidgen does the
    same: H3 copies a cut inside a chunk). The per-pass estimate is the sum of
    the estimates of vidgen's chunks inside the shot.
    """
    end = chunks[-1]["end"]
    bounds = [0.0] + sorted(c for c in cuts if 0 < c < end) + [end]
    merged = []
    for a, b in zip(bounds, bounds[1:]):
        if merged and ((b - a) * FPS < MIN_SHOT or (merged[-1][1] - merged[-1][0]) * FPS < MIN_SHOT):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    shots = []
    for i, (a, b) in enumerate(merged):
        est = sum(c.get("estimate_s") or 0 for c in chunks
                  if c["start"] < b - 1e-3 and c["end"] > a + 1e-3)
        shots.append({"id": i + 1, "start": round(a, 3), "end": round(b, 3),
                      "est_s_per_pass": est, "swaps": []})
    return shots


def _shot_sheet(src24, shots, meta, dst, label_rows=None):
    """One row per shot: start / middle / end frames, labelled. Pure PIL."""
    from PIL import Image, ImageDraw
    tw = 180
    th = int(round(tw * meta["h"] / meta["w"]))
    rows = []
    for s in shots:
        a, b = frame_range(s)
        idx = [min(a + 2, b - 1), (a + b) // 2, max(b - 3, a)]
        row = Image.new("RGB", (tw * 3 + 170, th), (24, 24, 24))
        for j, t in enumerate(idx):
            tmp = dst.parent / ("_f%d.png" % j)
            ff("-i", src24, "-vf", "select=eq(n\\,%d),scale=%d:%d" % (t, tw, th),
               "-frames:v", "1", tmp)
            row.paste(Image.open(tmp).convert("RGB"), (170 + j * tw, 0))
            tmp.unlink(missing_ok=True)
        d = ImageDraw.Draw(row)
        d.text((10, 10), "shot %d" % s["id"], fill=(255, 255, 255))
        d.text((10, 30), "%.2f - %.2f s" % (s["start"], s["end"]), fill=(200, 200, 200))
        if label_rows:
            for k, line in enumerate(label_rows.get(s["id"], [])):
                d.text((10, 54 + 18 * k), line[:24], fill=(255, 210, 120))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, th * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * th))
    sheet.save(dst, quality=88)


def _validate(spec, root):
    errs = []
    chars = spec.get("characters") or {}
    for name, c in chars.items():
        p = Path(c.get("image") or "")
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            errs.append("character %r: image not found: %s" % (name, c.get("image")))
        c["_image"] = str(p)
    todo = 0
    for s in spec.get("shots", []):
        for sw in s.get("swaps", []):
            todo += 1
            if sw.get("character") not in chars:
                errs.append("shot %s: unknown character %r" % (s["id"], sw.get("character")))
            who = (sw.get("who") or "").strip()
            if not who or "=" in who or "@" in who:
                errs.append("shot %s: 'who' must describe the person in the video and "
                            "may not contain '=' or '@'" % s["id"])
            if (sw.get("view") or "front") not in ("front", "back", "small"):
                errs.append("shot %s: view must be front, back or small" % s["id"])
    if not todo:
        errs.append("no shot has any swaps -- fill spec.json first")
    return errs


def cmd_render(a):
    """Every pass keeps a local LLM lent (vidgen puts it to sleep or unloads it
    when it fills the GPU, e.g. Strata) and it is given back once, here, at the
    end -- however the run ends. Giving it back after every pass would reload
    it between passes for nothing (~40 s each on Strata)."""
    os.environ["VIDGEN_LLM_RETURN"] = "defer"
    try:
        return _render(a)
    finally:
        os.environ.pop("VIDGEN_LLM_RETURN", None)
        try:
            subprocess.run([str(PY), str(VIDGEN), "llm-return"], capture_output=True,
                           timeout=900)
        except Exception:
            pass


def _render(a):
    pj = Project(a.project)
    spec = pj.spec()
    errs = _validate(spec, pj.root)
    if errs:
        return out({"ok": False, "errors": errs})
    st = pj.state()
    want = {int(x) for x in a.shots.split(",")} if a.shots else None
    meta = probe(pj.src)
    w = 48
    dims = (w, int(round(w * meta["h"] / meta["w"])))
    src_frames = read_frames(pj.src24, *dims, resample=False)
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else list(SEEDS)
    tries = max(1, min(a.tries, len(seeds)))

    for shot in spec["shots"]:
        if not shot.get("swaps") or (want and shot["id"] not in want):
            continue
        key = str(shot["id"])
        rec = st["shots"].get(key) or {"attempts": []}
        if a.fresh:
            rec = {"attempts": []}
        fa, fb = frame_range(shot)
        rng = "%.3f-%.3f" % (shot["start"], shot["end"])
        photos = [spec["characters"][sw["character"]]["_image"] for sw in shot["swaps"]]
        faces = any((sw.get("view") or "front") == "front" for sw in shot["swaps"])
        acc = rec.get("accepted")
        if acc and not acc.get("needs_review"):
            # Accepted under the checks of its day; a check added since must
            # still be able to send it back (an invented cut accepted before
            # the jump check existed was otherwise kept for good).
            if Path(acc["file"]).is_file():
                acc["check"] = check_render(acc["file"], src_frames, dims, fa, fb, photos, faces)
            if acc.get("check", {}).get("ok", True):
                log("shot %s: already accepted (%s), skipping" % (key, acc["file"]), pj.log)
                continue
            log("shot %s: accepted render no longer passes the current checks: %s"
                % (key, "; ".join(acc["check"].get("problems") or [])), pj.log)
        rec.pop("accepted", None)       # a needs_review pick is re-chosen below

        # Re-score what is already on disk with the current checks first: a
        # check that changed (or a run killed after rendering) must not cost
        # another render.
        for at in rec["attempts"]:
            if at.get("file") and Path(at["file"]).is_file():
                at["check"] = check_render(at["file"], src_frames, dims, fa, fb, photos, faces)
                if at["check"].get("ok") and not rec.get("accepted"):
                    rec["accepted"] = at
                    log("shot %s: earlier seed %s passes the current checks"
                        % (key, at["seed"]), pj.log)
        if rec.get("accepted"):
            rec.pop("inflight", None)
            st["shots"][key] = rec
            pj.save_state(st)
            continue

        # A job submitted by a run that was killed is still rendering in
        # vidgen's detached worker: re-attach instead of paying for it twice.
        resume = rec.pop("inflight", None)
        # An infrastructure error from an older run must not use up its seed.
        rec["attempts"] = [at for at in rec["attempts"]
                           if not (at.get("error") and INFRA_RE.search(at["error"]))]
        used = {at["seed"] for at in rec["attempts"]}
        order = [s for s in seeds if s not in used]
        if resume and resume.get("seed") in order:
            order.remove(resume["seed"])
            order.insert(0, resume["seed"])
        for seed in order[:max(0, tries - len(used))]:
            ref_video, ok_run, files = pj.src, True, []
            for pi, sw in enumerate(shot["swaps"]):
                char = spec["characters"][sw["character"]]
                prompt = build_prompt(sw, char, shot.get("scene"), shot.get("camera"))
                st["shots"][key] = rec
                # A finished earlier pass of this seed (run killed after it) is
                # reused, not rendered again: its output is the next pass's
                # reference video, and recording it is what makes that possible.
                done = (resume or {}).get("files") or []
                if (resume and resume.get("seed") == seed and pi < len(done)
                        and Path(done[pi]).is_file()):
                    log("shot %s seed %s pass %d: reusing %s from the interrupted run"
                        % (key, seed, pi + 1, done[pi]), pj.log)
                    files.append(done[pi])
                    ref_video = done[pi]
                    continue
                rec["inflight"] = {"seed": seed, "pass": pi + 1, "files": list(files)}
                pj.save_state(st)

                def grab(line, rec=rec):
                    m = JOB_RE.search(line)
                    if m:
                        rec["inflight"]["job_id"] = m.group(1)
                        pj.save_state(st)

                res = None
                if (resume and resume.get("seed") == seed and resume.get("pass") == pi + 1
                        and resume.get("job_id")):
                    res = wait_job(resume["job_id"], pj.log)    # None: it died, render again
                if res is None:
                    res = run_json([PY, VIDGEN, "r2v", prompt, "--ref-video", ref_video,
                                    "--swap", "%s=%s@%s" % (sw["who"], char["_image"], rng),
                                    "--ref-video-size", REF_VIDEO_SIZE, "--steps", STEPS,
                                    "--seed", seed, "--wait"], pj.log, on_line=grab)
                rec.pop("inflight", None)
                if not res.get("file"):
                    err = str(res.get("error") or res)
                    if INFRA_RE.search(err):
                        # Not this seed's fault: ComfyUI is down, hung or out of
                        # memory, and every further seed fails the same way. A
                        # crashed ComfyUI once burned all 12 seeds of a 4-shot
                        # clip in seconds. Stop, record nothing, keep the seed.
                        pj.save_state(st)
                        return out({"ok": False, "shot": shot["id"], "error": err[:600],
                                    "hint": INFRA_HINT})
                    rec["attempts"].append({"seed": seed, "error": err[:300]})
                    pj.save_state(st)
                    ok_run = False
                    break
                files.append(res["file"])
                ref_video = res["file"]
            if not ok_run:
                continue
            chk = check_render(files[-1], src_frames, dims, fa, fb, photos, faces)
            attempt = {"seed": seed, "file": files[-1], "passes": files, "check": chk}
            rec["attempts"].append(attempt)
            log("shot %s seed %s: %s" % (key, seed, json.dumps(chk, ensure_ascii=False)), pj.log)
            if chk.get("ok"):
                rec["accepted"] = attempt
                pj.save_state(st)
                break
            pj.save_state(st)
        if not rec.get("accepted"):
            good = [x for x in rec["attempts"] if x.get("check")]
            if good:
                def badness(x):
                    c = x["check"]
                    probs = " ".join(c.get("problems") or [])
                    return ("unchanged" in probs, "reframed" in probs, "jump" in probs,
                            c.get("leak_ratio", 9) < LEAK_RATIO, c.get("lag", 99))
                best = min(good, key=badness)
                rec["accepted"] = dict(best, needs_review=True)
                log("shot %s: no seed passed every check; keeping the best one (%s) for review"
                    % (key, best["check"]["problems"]), pj.log)
        st["shots"][key] = rec
        pj.save_state(st)
    return cmd_assemble(a, pj=pj, spec=spec, st=st)


def cmd_assemble(a, pj=None, spec=None, st=None):
    pj = pj or Project(a.project)
    spec = spec or pj.spec()
    st = st or pj.state()
    meta = probe(pj.src)
    if meta["w"] <= meta["h"]:
        ow, oh = SHORT_EDGE, int(round(SHORT_EDGE * meta["h"] / meta["w"] / 2)) * 2
    else:
        oh, ow = SHORT_EDGE, int(round(SHORT_EDGE * meta["w"] / meta["h"] / 2)) * 2
    tmp = pj.root / "_parts"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    lst, notes, review = [], [], {}
    for i, shot in enumerate(spec["shots"]):
        a_, b_ = frame_range(shot)
        rec = st["shots"].get(str(shot["id"])) or {}
        acc = rec.get("accepted") if shot.get("swaps") else None
        part = tmp / ("p%02d.mp4" % i)
        if acc:
            lag = min(acc["check"].get("lag", 0), MAX_HOLD) if acc.get("check") else 0
            # Hold the first swapped frame over the lag frames: the original
            # person never shows, and the shot keeps its exact length.
            vf = ("trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,"
                  "tpad=start_mode=clone:start=%d,tpad=stop_mode=clone:stop=%d,"
                  "trim=end_frame=%d,scale=%d:%d:flags=lanczos,setsar=1"
                  % (a_ + lag, b_, lag, b_ - a_, b_ - a_, ow, oh))
            ff("-i", acc["file"], "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
               "-pix_fmt", "yuv420p", "-r", FPS, part)
            review[shot["id"]] = ["seed %s" % acc["seed"],
                                  "REVIEW" if acc.get("needs_review") else "checks ok"]
            if acc.get("needs_review"):
                notes.append("shot %d: %s" % (shot["id"], "; ".join(acc["check"]["problems"])))
        else:
            vf = ("trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,"
                  "tpad=stop_mode=clone:stop=%d,trim=end_frame=%d,scale=%d:%d:flags=lanczos,setsar=1"
                  % (a_, b_, b_ - a_, b_ - a_, ow, oh))
            ff("-i", pj.src24, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
               "-pix_fmt", "yuv420p", "-r", FPS, part)
            if shot.get("swaps"):
                notes.append("shot %d: not rendered yet -- original frames used" % shot["id"])
            review[shot["id"]] = ["original"]
        lst.append("file '%s'" % part.as_posix())
    (tmp / "list.txt").write_text("\n".join(lst) + "\n", encoding="utf-8")
    final = pj.root / "final.mp4"
    args = ["-f", "concat", "-safe", "0", "-i", tmp / "list.txt"]
    if meta["audio"]:
        args += ["-i", pj.src, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "apad",
                 "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        args += ["-c:v", "copy"]
    ff(*args, final)
    shutil.rmtree(tmp, ignore_errors=True)
    fmeta = {"w": ow, "h": oh}
    _shot_sheet(final, spec["shots"], fmeta, pj.root / "sheets" / "final.jpg", review)
    out({"ok": True, "final": str(final), "review_sheet": str(pj.root / "sheets" / "final.jpg"),
         "source_sheet": str(pj.root / "sheets" / "shots.jpg"),
         "needs_review": notes,
         "shots": {k: ({"seed": v["accepted"]["seed"], "check": v["accepted"].get("check"),
                        "tries": len(v.get("attempts", []))} if v.get("accepted") else
                       {"tries": len(v.get("attempts", []))})
                   for k, v in st["shots"].items()}})


def cmd_status(a):
    pj = Project(a.project)
    st = pj.state()
    spec = pj.spec()
    rows = []
    for s in spec["shots"]:
        r = st["shots"].get(str(s["id"])) or {}
        rows.append({"shot": s["id"], "swaps": len(s.get("swaps", [])),
                     "tries": len(r.get("attempts", [])),
                     "inflight": r.get("inflight"),
                     "accepted": bool(r.get("accepted")),
                     "needs_review": bool((r.get("accepted") or {}).get("needs_review"))})
    final = pj.root / "final.mp4"
    out({"ok": True, "project": str(pj.root), "shots": rows,
         "final": str(final) if final.exists() else None})


def main():
    ap = argparse.ArgumentParser(description="Replace the people in a reference video.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("video")
    p.add_argument("--name")
    p.add_argument("--project")
    r = sub.add_parser("render")
    r.add_argument("project")
    r.add_argument("--shots", help="comma list of shot ids (default: every shot with swaps)")
    r.add_argument("--tries", type=int, default=3, help="seeds to try per shot (default 3)")
    r.add_argument("--seeds", help="comma list overriding the seed order")
    r.add_argument("--fresh", action="store_true", help="forget accepted renders of these shots")
    s = sub.add_parser("assemble")
    s.add_argument("project")
    t = sub.add_parser("status")
    t.add_argument("project")
    a = ap.parse_args()
    {"plan": cmd_plan, "render": cmd_render, "assemble": cmd_assemble,
     "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
