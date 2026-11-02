#!/usr/bin/env python
"""dramagen/produce.py -- render a drama.json into keyframes, clips and a cut.

This is the bridge between the screenwriting side (story.py, or a template the
agent wrote by hand) and the two rendering skills already installed here. It
never talks to ComfyUI: every image goes through `imggen/gen.py` and every clip
through `vidgen/gen.py`, exactly as those skills demand, so the memory
protection around the co-resident LLM stays intact.

Stages, in order (one at a time -- there is a single ComfyUI and a single GPU):

  refs       one head-and-shoulders portrait per character -- the reference
             sheet r2v runs on. MiniMax-H3 scales every reference down to the
             output's pixel area, so a portrait that is already head-and-
             shoulders keeps ~3x more face than a full-body shot at the same
             cost.
  keyframes  one still per i2v shot (Wan 2.2 is image-to-video: no start frame
             means pure noise, which is why t2v is blocked on this box).
             r2v shots need none, which is part of why they are the default.
  clips      r2v (MiniMax-H3) wherever the shot has a character -- it carries
             audio, holds the face across shots, and measured *faster* than Wan
             per second of video (255 s vs 288 s for a 5 s 480p portrait clip).
             i2v (Wan, silent) for shots with nobody in them, where H3 cannot
             run at all: it requires at least one reference image.
  stitch     concatenate in script order, and mux the dialogue as a subtitle
             track so a silent Wan cut is still watchable as a drama.

Everything is resumable: each step writes its result into state.json and a
re-run skips what is already done. A render that dies at shot 7 of 12 costs
you shot 7, not the whole episode.

Usage:
  produce.py <drama.json> --dry-run            # plan + time estimate, no GPU
  produce.py <drama.json>                      # full run
  produce.py <drama.json> --only keyframes     # one stage
  produce.py <drama.json> --max-shots 2        # cheap smoke test
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent
PY = sys.executable or "python"
IMGGEN = Path(os.environ.get("DRAMAGEN_IMGGEN") or (SKILLS / "imggen" / "gen.py"))
VIDGEN = Path(os.environ.get("DRAMAGEN_VIDGEN") or (SKILLS / "vidgen" / "gen.py"))

STAGES = ("refs", "keyframes", "clips", "stitch")
JOB_RE = re.compile(r"job_id=([0-9a-fA-F]{6,})")
CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
CARTOON = re.compile(r"cartoon|anime|illustration|flat 2d|vector|comic|cel[- ]shad",
                     re.I)

# Straight from vidgen's "faces look like the photo" notes: H3 reads words like
# `warrior` or `fencing` as *costume* and buries the face in a helmet, and a
# face under ~80 px carries no identity at all. Both are cheap to pre-empt.
FACE_HINT = ("medium close shot at eye level, faces large and sharp in frame, "
             "heads bare and uncovered, no helmet, no mask, no face guard")
# H3's own default is 20 steps. Measured here at 480p portrait, 5 s, same seed:
# 20 steps 256 s, 8 steps 129 s -- and the 8-step render is indistinguishable
# (same three-beat cut, same sharpness, both characters still matching their
# reference portraits, and faster-whisper reads both spoken lines back
# correctly from either). The sampler is res_multistep, a higher-order solver
# that converges early, which is why nothing breaks. So 8 is the default and
# 20 is available via --steps for anyone who wants to check a hard shot.
H3_STEPS = 8

PORTRAIT_HINT = ("head and shoulders portrait, face fills a third of the frame, "
                 "plain neutral background, soft even lighting, sharp focus, "
                 "looking at camera, single person only")


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, file=sys.stderr, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def hermes_home():
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(os.environ.get("HERMES_HOME") or Path(local) / "hermes")
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def slug(s, fallback="drama"):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or fallback


def has_cjk(s):
    return bool(CJK.search(s or ""))


# --------------------------------------------------------------- state file


class State:
    """Per-step ledger, so a re-run resumes instead of re-rendering.

    Written after every step, not at the end: the whole point is to survive a
    process that gets killed halfway (a terminal timeout, a reboot, Ctrl-C).
    """

    def __init__(self, path):
        self.path = Path(path)
        self.data = {"steps": {}}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        self.data.setdefault("steps", {})

    def done(self, key):
        st = self.data["steps"].get(key)
        return bool(st and st.get("status") == "done"
                    and st.get("file") and Path(st["file"]).exists())

    def get(self, key):
        return self.data["steps"].get(key) or {}

    def put(self, key, **kw):
        self.data["steps"][key] = kw
        self.data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# ------------------------------------------------------------- subprocesses


def run_json(cmd, log_path, timeout=None, on_line=None):
    """Run a skill entry point and return its stdout JSON.

    Both gen.py scripts print JSON on stdout and human text on stderr, so the
    contract is: parse stdout, echo stderr into the project log.

    stderr is drained LIVE, in a thread, rather than collected at exit: vidgen
    prints `job_id=...` there the moment it submits, and if this process is
    killed while `--wait` blocks (a terminal timeout, the user stopping a
    background task) a buffered stderr takes that id to the grave. The render
    itself survives -- it is a detached worker -- so losing the id is the
    difference between resuming and paying for the whole clip twice.
    """
    log("run: " + " ".join('"%s"' % c if " " in str(c) else str(c) for c in cmd),
        log_path)
    t0 = time.time()
    p = subprocess.Popen([str(c) for c in cmd],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def pump():
        for raw in iter(p.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            log("  | " + line, log_path)
            if on_line:
                try:
                    on_line(line)
                except Exception:
                    pass            # a bookkeeping hook must never kill a render

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    out = p.stdout.read().decode("utf-8", "replace").strip()
    p.wait(timeout=timeout)
    th.join(timeout=5)
    dt = time.time() - t0
    try:
        data = json.loads(out) if out else {}
    except json.JSONDecodeError:
        # A crash before the JSON print, or a stray banner in front of it.
        m = re.search(r"\{.*\}", out, re.S)
        data = json.loads(m.group(0)) if m else {"status": "error", "error":
                                                 out[:500] or "no output"}
    data["_seconds"] = round(dt, 1)
    data["_rc"] = p.returncode
    return data


def wait_for_headroom(refs, seconds, short, portrait, tier, log_path,
                      tries=8, pause=20):
    """Block until the GPU has room for the next clip.

    ComfyUI frees the previous render's weights asynchronously, so a clip
    submitted one second after the last one lands can measure the GPU as still
    full. Measured: six clips in a row succeeded, then shot 7 was refused with
    "only 1.2 GB VRAM free ... 14.8 GB resident" -- and by the time anyone
    looked, usage was back to 5.4 GB. Nothing was wrong except the timing.

    The gate is vidgen's own planner rather than a fixed sleep or a private
    nvidia-smi call: it is the exact code that would refuse the render, so it
    cannot disagree with it, and it is instant.
    """
    cmd = [PY, VIDGEN, "plan", "--seconds", "%.2f" % seconds,
           "--short", str(short), "--tier", tier]
    if portrait:
        cmd += ["--portrait"]
    for r in refs:
        cmd += ["--ref", str(Path(r).resolve())]
    for i in range(tries):
        res = run_json(cmd, log_path, timeout=300)
        if res.get("ok"):
            return True
        err = str(res.get("error") or "")
        if "VRAM" not in err and "RAM" not in err:
            return True     # a refusal for some other reason: let the real
                            # call make it, with its own error message
        log("GPU still busy (%d/%d), waiting %ds: %s"
            % (i + 1, tries, pause, err[:90]), log_path)
        time.sleep(pause)
    return False


def wait_job(job_id, dest, log_path, poll=20):
    """Re-attach to a render this project already submitted.

    vidgen's worker is detached on purpose, so a clip can outlive the run that
    started it. Re-submitting instead of re-attaching is the documented way to
    waste a render -- vidgen's own notes record a resubmit that landed four
    seconds after the original finished.
    """
    log("re-attaching to job %s (already rendering, not re-submitting)"
        % job_id, log_path)
    while True:
        res = run_json([PY, VIDGEN, "status", job_id], log_path, timeout=300)
        st = res.get("status")
        if st == "done" and res.get("file"):
            src = Path(res["file"])
            dest = Path(dest).with_suffix(src.suffix or ".mp4")
            shutil.copy2(src, dest)
            return str(dest), res.get("seconds") or 0
        if st != "running":
            log("job %s is %s, will render again: %s"
                % (job_id, st, res.get("error") or ""), log_path)
            return None
        time.sleep(poll)


def gen_image(model, prompt, dest, raw_dir, log_path, seed=None):
    cmd = [PY, IMGGEN, model, prompt, "--out", raw_dir]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    res = run_json(cmd, log_path)
    if res.get("status") != "ok" or not res.get("outputs"):
        raise RuntimeError(res.get("error") or "imggen failed (rc=%s)" % res.get("_rc"))
    src = Path(res["outputs"][0])
    dest = Path(dest).with_suffix(src.suffix)
    shutil.copy2(src, dest)
    return str(dest), res["_seconds"]


def gen_clip(mode, prompt, dest, log_path, image=None, refs=(), seconds=4.0,
             short=480, portrait=False, fast=True, steps=None, extra=(),
             on_job=None):
    # Check the inputs here rather than letting the engine discover it: a
    # submitted job has already started ComfyUI and handed it the co-resident
    # LLM's DRAM, so a job that was doomed from the first line costs the whole
    # machine a weight reload for nothing.
    for p in ([image] if image else []) + list(refs):
        if not Path(p).is_file():
            raise RuntimeError("input image missing: %s" % p)
    cmd = [PY, VIDGEN, mode, prompt, "--seconds", "%.2f" % seconds,
           "--short", str(short), "--wait"]
    if image:
        cmd += ["--image", str(Path(image).resolve())]
    for r in refs:
        cmd += ["--ref", str(Path(r).resolve())]
    if portrait:
        cmd += ["--portrait"]
    cmd += ["--fast"] if fast else ["--quality"]
    if steps:
        cmd += ["--steps", str(steps)]
    cmd += list(extra)

    def sniff(line):
        m = JOB_RE.search(line)
        if m and on_job:
            on_job(m.group(1))

    res = run_json(cmd, log_path, on_line=sniff)
    if not res.get("ok") or not res.get("file"):
        raise RuntimeError(res.get("error") or "vidgen failed (rc=%s)" % res.get("_rc"))
    src = Path(res["file"])
    dest = Path(dest).with_suffix(src.suffix or ".mp4")
    shutil.copy2(src, dest)
    return str(dest), res["_seconds"]


# ------------------------------------------------------------ prompt makers


def pick_image_model(requested, prompt, style):
    """`auto` picks the model that can actually read this prompt.

    realvis and flux are CJK-blind and fail silently on Chinese -- a clean
    picture of something else entirely -- so a prompt that still has Chinese in
    it (story.py translates, but a hand-written template might not) has to go
    to a CJK-capable model rather than to the prettiest one.
    """
    if requested != "auto":
        return requested
    if has_cjk(prompt):
        return "qwen"
    if CARTOON.search(style or ""):
        return "zimage"
    return "realvis"


def joined(*parts):
    return ". ".join(p.strip().strip(".") for p in parts if p and p.strip())


def ref_prompt(char, style):
    return joined(char.get("appearance") or char.get("en_name"), PORTRAIT_HINT, style)


def keyframe_prompt(drama, shot, cast, portrait):
    who = [cast[c] for c in shot.get("characters", []) if c in cast]
    apps = "; ".join(c["appearance"] for c in who if c.get("appearance"))
    frame = ("vertical composition, subject centred"
             if portrait else "wide composition, subject centred")
    return joined(shot["image_prompt"],
                  ("Characters: " + apps) if apps else "",
                  drama.get("style", ""), frame)


STAGE_DIR = re.compile(r"[（(][^）)]*[）)]")

# What language the cast speaks has to be stated, and stated in one voice. The
# prompt around the lines is English while the lines themselves are Chinese, so
# left to itself H3 picks a language per line and a clip comes back part
# Mandarin, part English. Naming the language once as a rule AND again on every
# line is what makes it stick.
SPEECH_LANGS = {
    "zh-hant": "Mandarin Chinese with a Taiwanese accent",
    "zh-tw": "Mandarin Chinese with a Taiwanese accent",
    "zh-hans": "Mandarin Chinese",
    "zh-cn": "Mandarin Chinese",
    "zh": "Mandarin Chinese",
    "ja": "Japanese", "ja-jp": "Japanese",
    "ko": "Korean", "ko-kr": "Korean",
    "en": "English", "en-us": "English",
}


def speech_language(drama, override=None):
    """--speech-lang beats drama.json's own phrase beats its language code.

    story.py writes `speech_language` as prose ("Japanese", "Cantonese")
    because that is what goes into the video prompt; the bare `language` code
    is the fallback for a template written by hand.
    """
    if override:
        return override
    d = drama or {}
    said = str(d.get("speech_language") or "").strip()
    if said:
        return said
    code = str(d.get("language") or "").strip().lower()
    return SPEECH_LANGS.get(code, SPEECH_LANGS["zh-hant"])


def spoken_lines(shot, who, lang):
    """Turn the script's dialogue into speech instructions for H3.

    H3's workflow has exactly one text input, so the only way the cast can say
    anything is to put the lines in the prompt. Two things have to happen first:

      * the speaker is named by reference slot, not by name -- the rest of the
        prompt already calls them Picture N, and a second naming scheme just
        gives the model someone new to invent;
      * parenthetical stage directions are cut. "（轉頭，眼神空洞）這味道…" is a
        direction plus a line, and feeding the direction in as speech asks the
        model to read the blocking out loud. A line that is nothing but a
        direction ("…(沉默)") yields no speech at all, which is correct.

    The text encoder here is Qwen3-VL, so Chinese in the prompt is read
    properly -- unlike the image side, where realvis and flux are CJK-blind.
    """
    # The script may credit a line to the character's native name, their ascii
    # en_name, or their id -- a Japanese brief happily writes `Aya` where the
    # cast list says 店員. Miss the match and the line is demoted to an
    # off-screen voice, so accept all three spellings.
    out, idx = [], {}
    for i, c in enumerate(who, 1):
        for key in (c.get("name"), c.get("en_name"), c.get("id")):
            if key:
                idx[str(key).strip().lower()] = i
    for d in shot.get("dialogue") or []:
        line = STAGE_DIR.sub("", d.get("line") or "").strip(" .。…·、,，")
        if not line:
            continue
        n = idx.get(str(d.get("character") or "").strip().lower())
        speaker = ("Picture %d" % n) if n else "An off-screen voice"
        out.append('%s says in %s: "%s"' % (speaker, lang, line))
    if not out:
        return ""
    return ("All spoken dialogue in this shot is in %s. Every character speaks "
            "only %s -- no English, no other language, no subtitles on screen. "
            % (lang, lang)) + " ".join(out)


def r2v_prompt(shot, who, speak=True, lang=None):
    """Name each reference as Picture N -- H3 ignores refs it cannot map.

    vidgen refuses a multi-ref call whose prompt never says "Picture N", and it
    is right to: without the mapping H3 invents generic people and the result
    looks like a working render of the wrong cast.
    """
    text = shot["motion_prompt"]
    for i, c in enumerate(who, 1):
        if c.get("en_name"):
            text = re.sub(re.escape(c["en_name"]), "Picture %d" % i, text, flags=re.I)
    # The mapping clause alone already satisfies vidgen's "the prompt must say
    # Picture N" guard, and it is what tells H3 which face belongs to whom.
    mapping = "; ".join(
        "Picture %d is %s" % (i, c.get("appearance") or c.get("en_name"))
        for i, c in enumerate(who, 1))
    speech = spoken_lines(shot, who, lang or SPEECH_LANGS["zh-hant"]) if speak else ""
    return joined(mapping, text, speech, FACE_HINT)


# ------------------------------------------------------------------ stitch


def ffprobe(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, timeout=60).stdout.decode("utf-8", "replace")
        d = json.loads(out)
        streams = d.get("streams") or []
        vid = next((s for s in streams if s.get("codec_type") == "video"), {})
        return {"duration": float((d.get("format") or {}).get("duration") or 0),
                "width": int(vid.get("width") or 0),
                "height": int(vid.get("height") or 0),
                "audio": any(s.get("codec_type") == "audio" for s in streams)}
    except Exception:
        return {"duration": 0.0, "width": 0, "height": 0, "audio": False}


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def build_srt(shots, clips, path):
    """Dialogue as a subtitle track, timed off the clips that actually rendered.

    Wan clips are silent, so without this the cut is a mood reel and the script
    is invisible. Timings come from ffprobe rather than the requested seconds:
    both engines snap length to a frame grid (H3 to 17k+5 at 24 fps), so the
    asked-for number is never quite what landed.
    """
    entries, t, n = [], 0.0, 0
    for sh in shots:
        clip = clips.get(sh["id"])
        if not clip:
            continue
        dur = ffprobe(clip)["duration"] or float(sh.get("seconds") or 4.0)
        lines = [d for d in (sh.get("dialogue") or []) if d.get("line")]
        if lines:
            slot = dur / len(lines)
            for i, d in enumerate(lines):
                n += 1
                a, b = t + i * slot, t + (i + 1) * slot
                who = d.get("character") or ""
                entries.append("%d\n%s --> %s\n%s\n" % (
                    n, srt_time(a), srt_time(b),
                    ("%s：%s" % (who, d["line"])) if who else d["line"]))
        t += dur
    Path(path).write_text("\n".join(entries), encoding="utf-8")
    return n


def stitch(clips_in_order, out_path, srt_path, log_path):
    """Concatenate with one re-encode.

    The concat *demuxer* would be free but needs identical streams, and this
    pipeline deliberately mixes engines: Wan i2v is 16 fps and silent, H3 r2v
    is 24 fps with stereo audio. So normalise through the concat filter, and
    when any clip has sound, synthesise a silent track for the ones that do not
    -- otherwise ffmpeg drops audio for the whole cut. The silence comes from
    an `anullsrc` *inside* the filter graph rather than an extra `-i`, which
    keeps every real input at a stable index for `-map`.
    """
    infos = [ffprobe(c) for c in clips_in_order]
    w = max((i["width"] for i in infos if i["width"]), default=832)
    h = max((i["height"] for i in infos if i["height"]), default=480)
    w, h = w - (w % 2), h - (h % 2)
    want_audio = any(i["audio"] for i in infos)
    use_srt = bool(srt_path and Path(srt_path).is_file()
                   and Path(srt_path).stat().st_size)

    # Every -i first: ffmpeg binds the options that follow an input to that
    # input, so an -i placed after -map would silently re-target the maps.
    cmd = ["ffmpeg", "-y"]
    for c in clips_in_order:
        cmd += ["-i", str(c)]
    if use_srt:
        cmd += ["-i", str(srt_path)]
    srt_index = len(clips_in_order)

    parts, streams = [], []
    for i, info in enumerate(infos):
        parts.append(
            "[%d:v]scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24[v%d]" % (i, w, h, w, h, i))
        streams.append("[v%d]" % i)
        if want_audio:
            if info["audio"]:
                parts.append("[%d:a]aresample=48000,asetpts=N/SR/TB[a%d]" % (i, i))
            else:
                parts.append("anullsrc=r=48000:cl=stereo:d=%.3f[a%d]"
                             % (max(info["duration"], 0.1), i))
            streams.append("[a%d]" % i)
    parts.append("%sconcat=n=%d:v=1:a=%d[vout]%s"
                 % ("".join(streams), len(clips_in_order),
                    1 if want_audio else 0, "[aout]" if want_audio else ""))
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if want_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "128k"]
    if use_srt:
        cmd += ["-map", "%d:s" % srt_index, "-c:s", "mov_text",
                "-metadata:s:s:0", "language=chi"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", str(out_path)]

    log("stitching %d clips -> %s" % (len(clips_in_order), out_path), log_path)
    p = subprocess.run(cmd, capture_output=True, timeout=1800)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-8:]
        raise RuntimeError("ffmpeg failed: " + " / ".join(tail))
    return str(out_path)


# -------------------------------------------------------------------- main


def load_drama(arg):
    p = Path(arg)
    if p.is_dir():
        p = p / "drama.json"
    if not p.is_file():
        raise SystemExit(json.dumps({"status": "error",
            "error": "no drama.json at %s -- run story.py first" % p},
            ensure_ascii=False))
    return json.loads(p.read_text(encoding="utf-8")), p


def all_shots(drama, max_shots=0):
    out = []
    for ep in drama.get("episodes", []):
        for sh in ep.get("shots", []):
            out.append(sh)
    return out[:max_shots] if max_shots else out


def plan_line(seconds, short, tier, portrait, log_path):
    cmd = [PY, VIDGEN, "plan", "--seconds", "%.2f" % seconds, "--short", str(short),
           "--tier", tier]
    if portrait:
        cmd += ["--portrait"]
    res = run_json(cmd, log_path, timeout=300)
    return res.get("plan") or res.get("error") or json.dumps(res, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(
        description="Render a drama.json with the imggen + vidgen skills.")
    ap.add_argument("drama", help="path to drama.json (or a project directory)")
    ap.add_argument("--project", help="output directory (default: <hermes>/drama/<title>)")
    ap.add_argument("--only", help="comma list: refs,keyframes,clips,stitch")
    ap.add_argument("--skip", help="comma list of stages to skip")
    ap.add_argument("--img-model", default="auto", dest="img_model",
                    choices=["auto", "qwen", "flux", "zimage", "realvis"])
    ap.add_argument("--short", type=int, default=480,
                    choices=[480, 576, 608, 720, 768])
    ap.add_argument("--seconds", type=float, help="override every shot's length")
    ap.add_argument("--mode", default="auto", choices=["auto", "i2v", "r2v"],
                    help="auto = whatever each shot asks for")
    ap.add_argument("--max-shots", type=int, default=0, dest="max_shots",
                    help="render only the first N shots (smoke test)")
    ap.add_argument("--quality", action="store_true", help="full steps (~3x slower)")
    ap.add_argument("--steps", type=int, default=None,
                    help="sampling steps for r2v/H3 (default %d). 20 is H3's own "
                         "default and takes twice as long for no measured gain."
                         % H3_STEPS)
    ap.add_argument("--speech-lang", dest="speech_lang", default=None,
                    help="language the cast speaks, written the way you would "
                         "say it to a model (\"Mandarin Chinese with a Taiwanese "
                         "accent\", \"Japanese\"). Default: from drama.json's "
                         "`language`, falling back to Taiwanese Mandarin.")
    ap.add_argument("--no-speak", dest="speak", action="store_false", default=True,
                    help="do not put the dialogue in the r2v prompt (H3 then "
                         "generates ambience only, and the lines live in the "
                         "subtitles alone)")
    ap.add_argument("--force", action="store_true", help="redo completed steps")
    ap.add_argument("--strict", action="store_true", help="stop at the first failure")
    ap.add_argument("--give-up-after", type=int, default=2, dest="give_up",
                    help="abort a stage after N consecutive failures (0 = never). "
                         "One bad shot is bad luck; two in a row is a broken setup, "
                         "and each attempt still costs a ComfyUI start.")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()

    drama, src = load_drama(a.drama)
    portrait = drama.get("aspect") == "portrait"
    cast = {c["id"]: c for c in drama.get("characters", [])}
    shots = all_shots(drama, a.max_shots)
    if a.seconds:
        for sh in shots:
            sh["seconds"] = a.seconds
    if a.mode != "auto":
        for sh in shots:
            sh["mode"] = a.mode
    for sh in shots:
        if sh.get("mode") == "r2v" and not [c for c in sh.get("characters", []) if c in cast]:
            sh["mode"] = "i2v"          # H3 cannot run without a reference

    # Absolute, always. vidgen submits to a DETACHED worker with its own cwd,
    # so a relative --image resolves to nothing there: measured as three clips
    # in a row failing with "start image not found" while the keyframes sat
    # happily on disk next to produce.py.
    lang = speech_language(drama, a.speech_lang)
    proj = (Path(a.project) if a.project else
            hermes_home() / "drama" / slug(drama.get("title"), src.parent.name)).resolve()
    for sub in ("refs", "keyframes", "clips", "_raw"):
        (proj / sub).mkdir(parents=True, exist_ok=True)
    log_path = proj / "produce.log"
    if src.resolve() != (proj / "drama.json").resolve():
        shutil.copy2(src, proj / "drama.json")
    state = State(proj / "state.json")

    stages = list(STAGES)
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        stages = [s for s in stages if s in want]
    if a.skip:
        drop = {s.strip() for s in a.skip.split(",")}
        stages = [s for s in stages if s not in drop]
    needs_refs = any(sh.get("mode") == "r2v" for sh in shots)
    if "refs" in stages and not needs_refs and not a.only:
        # Portraits are only *required* by r2v. For an all-i2v drama they are a
        # nice character sheet but two extra minutes of GPU for nothing.
        stages.remove("refs")

    if a.dry_run:
        modes = {}
        for sh in shots:
            modes.setdefault(sh.get("mode", "i2v"), []).append(float(sh.get("seconds") or 4))
        plans = {}
        for m, secs in modes.items():
            plans[m] = plan_line(max(secs), a.short, "h3" if m == "r2v" else "auto",
                                 portrait, log_path)
        print(json.dumps({
            "status": "ok", "dry_run": True, "project": str(proj),
            "title": drama.get("title"), "shots": len(shots),
            "stages": stages,
            "images": sum(1 for sh in shots if sh.get("mode") == "i2v")
                      + (len(cast) if needs_refs else 0),
            "by_mode": {m: len(v) for m, v in modes.items()},
            "plan": plans,
            "note": "clips dominate the wall clock. Measured at 480p portrait on a "
                    "16 GB laptop GPU with a co-resident LLM: r2v 129 s for a 5 s "
                    "shot at the default %d steps (256 s at H3's own 20), i2v 288 s "
                    "for 5 s / 231 s for 4 s. Cost is roughly 44 s fixed + 10.6 s "
                    "per step, so --short is what really gets expensive. "
                    "Run the real thing in the background." % H3_STEPS,
        }, ensure_ascii=False, indent=2))
        return

    summary = {"refs": {}, "keyframes": {}, "clips": {}, "errors": []}
    t_start = time.time()
    aborted = []

    def failed(kind, ident, exc, run):
        """Record a failure and say whether the stage should keep going."""
        summary["errors"].append("%s:%s %s" % (kind, ident, exc))
        log("%s %s FAILED: %s" % (kind, ident, exc), log_path)
        if a.strict:
            return True
        run[0] += 1
        if a.give_up and run[0] >= a.give_up:
            msg = ("%s stage aborted after %d consecutive failures -- this is a "
                   "setup problem, not bad luck. Fix it and re-run; finished "
                   "steps are already in state.json and will be skipped."
                   % (kind, run[0]))
            aborted.append(msg)
            log(msg, log_path)
            return True
        return False

    if "refs" in stages:
        run = [0]
        for cid, c in cast.items():
            key = "ref:" + cid
            if state.done(key) and not a.force:
                summary["refs"][cid] = state.get(key)["file"]
                continue
            prompt = ref_prompt(c, drama.get("style", ""))
            model = pick_image_model(a.img_model, prompt, drama.get("style", ""))
            log("ref %s (%s)" % (cid, model), log_path)
            try:
                f, dt = gen_image(model, prompt, proj / "refs" / cid,
                                  proj / "_raw", log_path)
                state.put(key, status="done", file=f, seconds=dt, model=model)
                summary["refs"][cid] = f
                run[0] = 0
            except Exception as e:
                state.put(key, status="error", error=str(e))
                if failed("ref", cid, e, run):
                    break

    if "keyframes" in stages:
        run = [0]
        for sh in shots:
            if sh.get("mode") != "i2v":
                continue
            key = "kf:" + sh["id"]
            if state.done(key) and not a.force:
                summary["keyframes"][sh["id"]] = state.get(key)["file"]
                continue
            prompt = keyframe_prompt(drama, sh, cast, portrait)
            model = pick_image_model(a.img_model, prompt, drama.get("style", ""))
            log("keyframe %s (%s)" % (sh["id"], model), log_path)
            try:
                f, dt = gen_image(model, prompt, proj / "keyframes" / sh["id"],
                                  proj / "_raw", log_path)
                state.put(key, status="done", file=f, seconds=dt, model=model,
                          prompt=prompt)
                summary["keyframes"][sh["id"]] = f
                run[0] = 0
            except Exception as e:
                state.put(key, status="error", error=str(e))
                if failed("keyframe", sh["id"], e, run):
                    break

    if "clips" in stages:
        log("clips stage: the GPU is busy from here on. On a box with a "
            "co-resident LLM its DRAM is handed to the render once ComfyUI "
            "starts -- do not send prompts until this finishes.", log_path)
        run = [0]
        for sh in shots:
            key = "clip:" + sh["id"]
            if state.done(key) and not a.force:
                summary["clips"][sh["id"]] = state.get(key)["file"]
                continue
            mode = sh.get("mode", "i2v")
            secs = float(sh.get("seconds") or 4.0)
            dest = proj / "clips" / sh["id"]

            # A previous run may have been killed mid-render. Its worker kept
            # going, so pick that clip up instead of paying for it again.
            prev = state.get(key)
            if prev.get("job_id") and prev.get("status") == "running" and not a.force:
                got = wait_job(prev["job_id"], dest, log_path)
                if got:
                    state.put(key, status="done", file=got[0], seconds=got[1],
                              mode=prev.get("mode", mode), job_id=prev["job_id"],
                              prompt=prev.get("prompt"))
                    summary["clips"][sh["id"]] = got[0]
                    run[0] = 0
                    log("clip %s adopted -> %s" % (sh["id"], got[0]), log_path)
                    continue

            def remember(jid, _k=key, _m=mode):
                state.put(_k, status="running", job_id=jid, mode=_m)

            try:
                if not wait_for_headroom(
                        [state.get("ref:" + c).get("file") for c in sh.get("characters", [])
                         if state.get("ref:" + c).get("file")] if mode == "r2v" else [],
                        secs, a.short, portrait,
                        "h3" if mode == "r2v" else "auto", log_path):
                    raise RuntimeError(
                        "the GPU never freed enough memory for this clip. Something "
                        "else is holding VRAM -- check with nvidia-smi.")
                if mode == "r2v":
                    who = [cast[c] for c in sh.get("characters", []) if c in cast][:9]
                    refs = []
                    for c in who:
                        rf = state.get("ref:" + c["id"]).get("file")
                        if rf and Path(rf).exists():
                            refs.append(rf)
                    if not refs:
                        raise RuntimeError("no character reference images -- "
                                           "run the refs stage first")
                    prompt = r2v_prompt(sh, who[:len(refs)], speak=a.speak,
                                        lang=lang)
                    log("clip %s r2v %.1fs, %d refs" % (sh["id"], secs, len(refs)), log_path)
                    f, dt = gen_clip("r2v", prompt, dest, log_path,
                                     refs=refs, seconds=secs, short=a.short,
                                     portrait=portrait, fast=not a.quality,
                                     steps=a.steps or H3_STEPS, on_job=remember)
                else:
                    kf = state.get("kf:" + sh["id"]).get("file")
                    if not (kf and Path(kf).exists()):
                        raise RuntimeError("no keyframe -- run the keyframes stage first")
                    prompt = joined(sh["motion_prompt"], drama.get("style", ""))
                    log("clip %s i2v %.1fs" % (sh["id"], secs), log_path)
                    f, dt = gen_clip("i2v", prompt, dest, log_path,
                                     image=kf, seconds=secs, short=a.short,
                                     portrait=portrait, fast=not a.quality,
                                     on_job=remember)
                state.put(key, status="done", file=f, seconds=dt, mode=mode,
                          prompt=prompt)
                summary["clips"][sh["id"]] = f
                run[0] = 0
                log("clip %s done in %.0fs -> %s" % (sh["id"], dt, f), log_path)
            except Exception as e:
                state.put(key, status="error", error=str(e))
                if failed("clip", sh["id"], e, run):
                    break

    final = None
    if "stitch" in stages:
        ordered = [(sh, summary["clips"].get(sh["id"])
                    or state.get("clip:" + sh["id"]).get("file")) for sh in shots]
        ready = [(sh, c) for sh, c in ordered if c and Path(c).exists()]
        if not ready:
            log("nothing to stitch", log_path)
        else:
            srt = proj / "dialogue.srt"
            n = build_srt([sh for sh, _ in ready],
                          {sh["id"]: c for sh, c in ready}, srt)
            log("subtitles: %d cues -> %s" % (n, srt), log_path)
            try:
                final = stitch([c for _, c in ready], proj / "final.mp4", srt, log_path)
                state.put("stitch", status="done", file=final)
            except Exception as e:
                summary["errors"].append("stitch: %s" % e)
                log("stitch FAILED: %s" % e, log_path)

    result = {
        "status": "error" if summary["errors"] and not summary["clips"] else "ok",
        "project": str(proj),
        "final": final,
        "clips": summary["clips"],
        "keyframes": summary["keyframes"],
        "refs": summary["refs"],
        "errors": summary["errors"],
        "wall_seconds": round(time.time() - t_start, 1),
        "log": str(log_path),
    }
    if aborted:
        result["aborted"] = aborted
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
