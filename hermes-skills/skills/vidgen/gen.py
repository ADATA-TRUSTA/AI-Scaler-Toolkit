#!/usr/bin/env python
"""vidgen skill entry point — the ONLY supported way to generate video here.

Thin front-end. It forwards to the memory-aware engine (video_gen.py, in the
comfy-engine directory resolved by _find_engine below), which measures live
VRAM/RAM, picks a model, chooses the VRAM->DRAM->SSD offload path, launches the
protected ComfyUI if needed, renders, and frees memory back to the LLM
afterward. Do NOT hand-write ComfyUI calls, and do not run video_gen.py itself.

The co-resident LLM's DRAM is handed to the render by the engine itself, and
only AFTER the prompt is confirmed executing -- never by the caller, and never
before ComfyUI is up. See release_llm_dram() in memplan.py; the agent driving
this script is that same LLM, so an early trim just stalls the agent.

A render takes several minutes. Prefer `--wait`: the job is submitted as usual
(so `status` and the job file still work) and this process blocks until it
lands, without the agent -- and therefore the co-resident 35B -- being woken
once per check. Without `--wait` you get the job_id back immediately.

Two engines are wired in. Wan 2.2 (i2v) is fast and silent. MiniMax-H3 is the
only one that also generates native stereo AUDIO, and it takes a still image
two different ways -- pick by what the image IS:

  r2v --ref IMG               the image is a REFERENCE: keep this character,
                              but compose a new shot around them
  i2v --image IMG --tier h3   the image IS frame 0: animate exactly this
                              picture, with sound (--last-frame pins the end)
  r2v --ref A --ref B --image START
                              BOTH: A and B are the cast, START is frame 0. The
                              opening composition is yours instead of H3's.
  r2v --ref-video V --swap "the man=A.png@0-8.6" --swap "the dancer=B.png@8.6-"
                              RECAST a video: keep V's motion, camera, timing and
                              soundtrack, put the pictured people in. Rendered shot
                              by shot (cut at V's own edits), 720p by default. H3
                              replaces ONE person per render, so give each --swap
                              the time range its person is on screen.

Long H3 clips are split into chained segments automatically -- its cost is
quadratic in length, so N short segments are far cheaper than one long shot.

Usage:
  gen.py plan                                   # what fits right now? (instant, no render)
  gen.py i2v "slow dolly-in" --image C:/x.png --wait   # blocks -> {"status":"done","file":...}
  gen.py t2v "a red fox in snow" --wait                # same, text-to-video
  gen.py r2v "she waves" --ref C:/girl.png --wait      # H3: with audio, keeps the character
  gen.py i2v "she waves" --image C:/girl.png --tier h3 --wait  # H3: animates THAT frame, with audio
  gen.py i2v "..." --image C:/x.png             # fire-and-forget -> {"job_id": "..."}
  gen.py status                                 # newest job -> running | done(+file) | error
  gen.py status <job_id>                        # a specific one
  gen.py stop                                   # interrupt the render, clear the queue, free VRAM
  gen.py crop C:/girl.png --rect 0.1,0.0,0.4,0.35    # head-and-shoulders ref -> better identity

Options: --tier auto|5b|14b|h3  --seconds N  --fps N  --short 480|576|608|720|768
         --portrait  --fast/--quality  --seed N  --steps N  --wait
         --easycache T / --no-easycache (H3: step cache, default 0.2 = ~20% faster)
         --ref PATH (H3 r2v, repeat up to 9; they become <Picture 1>, <Picture 2>, ...)
         --last-frame PATH (H3 i2v: pin the final frame too; implies --tier h3)
         --ref-image-size match|max (H3; see "making faces look like the photo" in SKILL.md)
         --segment-prompt TEXT (H3, repeat: one per chained segment)
         --ref-video PATH  --swap "WHO=IMG[|IMG...][@START-END]" (repeat, one per person)
         --ref-video-size N (default 384)  --ref-video-start S
         --ref-video-audio original|generate|reference (default original)
"""
import argparse, json, os, re, sys, time

# We emit Chinese on both streams (the standing notice on stderr, ensure_ascii=False
# JSON on stdout). A cp1252 console would make that a UnicodeEncodeError and kill
# the submit outright, so pin UTF-8 and degrade instead of dying.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path


def _find_engine() -> str:
    """Locate the comfy engine directory without hardcoding an absolute path.

    A skill is copied into the Hermes skills directory, which differs by OS and
    by profile, so it cannot assume where the engine landed. In order:
      1. HERMES_COMFY_ENGINE                    (set by the installer / conf)
      2. an `engine_dir.txt` written next to this file at install time
      3. the platform-native default under the Hermes home

    A fourth candidate, ~/.hermes/comfy-launcher, was dropped on 2026-09-10
    along with the directory itself: it had been stale for months and agents
    kept finding it by filename search and running it.
    """
    env = os.environ.get("HERMES_COMFY_ENGINE", "").strip()
    if env and (Path(env) / "paths.py").is_file():
        return str(Path(env))
    marker = Path(__file__).resolve().parent / "engine_dir.txt"
    if marker.is_file():
        p = Path(os.path.expanduser(marker.read_text(encoding="utf-8").strip()))
        if (p / "paths.py").is_file():
            return str(p)
    # A linked install (install.py --link) runs this file straight from the pack,
    # whose engine/ is two levels up; nothing is written next to the script.
    packed = Path(__file__).resolve().parents[2] / "engine"
    if (packed / "paths.py").is_file():
        return str(packed)
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        hermes_home = Path(os.environ.get("HERMES_HOME") or Path(local) / "hermes")
    else:
        hermes_home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    for cand in (hermes_home / "comfy-engine",
                 Path.home() / ".hermes-comfy" / "engine"):
        if (cand / "paths.py").is_file():
            return str(cand)
    raise SystemExit(json.dumps({"ok": False, "error":
        "comfy engine not found. Set HERMES_COMFY_ENGINE to the directory "
        "containing paths.py, or write that path into "
        + str(marker)}, ensure_ascii=False))


ENGINE_DIR = _find_engine()
sys.path.insert(0, ENGINE_DIR)
import paths      # noqa: E402  -- folds hermes-comfy.conf into os.environ first
import memplan    # noqa: E402  (h3_camera_pinned: the prompt check below)
import video_gen  # noqa: E402  (engine + planner live in the protected dir)

SKILLS_DIR = str(Path(__file__).resolve().parent.parent)

PY = sys.executable or "python"     # SKILL.md gets @@PY@@; gen.py resolves its own


def main():
    ap = argparse.ArgumentParser(
        description="Generate video via the protected local ComfyUI (Wan 2.2 or MiniMax-H3).")
    ap.add_argument("mode", choices=["i2v", "t2v", "r2v", "plan", "status", "stop",
                                     "crop", "llm-return"],
                    help="i2v | t2v | r2v (H3 reference-to-video, WITH AUDIO) | "
                         "plan (dry-run) | status [job_id] | stop [job_id] | "
                         "llm-return (wake/reload a local LLM a render borrowed the GPU from)")
    ap.add_argument("prompt", nargs="?", default="",
                    help="text prompt (motion/scene); for `status`/`stop`, the job_id; "
                         "for `crop`, the source image path")
    ap.add_argument("--rect", metavar="x,y,w,h",
                    help="crop box as FRACTIONS of the source image, e.g. "
                         "0.05,0.02,0.40,0.35. Look at the picture first and put the "
                         "box round the head and shoulders.")
    ap.add_argument("--out", metavar="PATH", help="where `crop` writes its result")
    ap.add_argument("--allow-recut", dest="allow_recut", action="store_true",
                    help="submit even though a first frame is pinned and the prompt "
                         "does not pin the camera. H3 will probably ignore the frame; "
                         "use this only when you want it to recut freely.")
    ap.add_argument("--image", help="start frame path (required for i2v). With "
                    "--tier h3 this becomes frame 0 of the clip (fl2va), not a "
                    "reference: the video starts on exactly this picture, and it "
                    "has audio.")
    ap.add_argument("--last-frame", dest="last_frame", metavar="PATH",
                    help="H3 i2v only: also pin the FINAL frame, so the clip "
                         "interpolates --image -> --last-frame. Implies --tier h3 "
                         "(Wan's graph has no end-frame input).")
    ap.add_argument("--ref", action="append", default=[], metavar="PATH",
                    help="H3 reference image (r2v); combine with --image to also "
                         "pin the opening frame. Repeat up to 9 times "
                         "(order matters: they become <Picture 1>, <Picture 2>, ... "
                         "and the prompt may refer to them by those names)")
    ap.add_argument("--ref-image-size", choices=["match", "max"], default="match",
                    dest="ref_image_size",
                    help="H3: how much of each --ref the model actually sees. "
                         "match (default) scales every reference DOWN to the output's "
                         "pixel area, so a 3213x5712 full-body phone photo arrives as "
                         "768x1344 with a ~78 px face and the identity is already gone. "
                         "max keeps a 2048 short edge for best identity fidelity, but "
                         "reference tokens ride through every sampling step, so it costs "
                         "real VRAM and time (that same photo becomes 29k latent tokens). "
                         "Cropping the reference to head-and-shoulders is cheaper than "
                         "max and works better -- try that first.")
    ap.add_argument("--ref-video", dest="ref_video", metavar="PATH",
                    help="H3 r2v: a video to RECAST. Its motion, camera, timing and "
                         "soundtrack carry over (the prompt calls it Video 1) and the "
                         "--swap / --ref pictures say who appears in it. Rendered shot "
                         "by shot at the video's own cuts, so length is not limited by "
                         "VRAM; defaults to the whole video at --short 720.")
    ap.add_argument("--swap", action="append", default=[],
                    metavar="WHO=IMG[|IMG...][@START-END]",
                    help="with --ref-video: replace one person, e.g. "
                         "\"the man on the left=C:/bob.png@0-8.6\". Several images of the "
                         "SAME person are joined with |. @START-END (seconds in the video; "
                         "either end may be left open, e.g. @8.6-) limits it to the shots "
                         "where that person is on screen -- do this whenever people appear "
                         "in different shots: H3 swaps one person per render, and shots "
                         "nobody is swapped in keep their original frames for free.")
    ap.add_argument("--ref-video-size", dest="ref_video_size", type=int, default=384,
                    metavar="N",
                    help="short edge the reference video is scaled to (default 384). "
                         "Every reference frame rides through every sampling step: 384 "
                         "roughly doubles render time at 480p, the node's own 768 would "
                         "not fit next to the clip at all.")
    ap.add_argument("--ref-video-start", dest="ref_video_start", type=float, default=0.0,
                    metavar="S", help="skip this many seconds of the reference video")
    ap.add_argument("--ref-video-audio", dest="ref_video_audio",
                    choices=["original", "generate", "reference"], default="original",
                    help="original (default): the result carries the reference video's own "
                         "soundtrack, frame-locked. generate: H3's sound per shot (it "
                         "changes at every chunk boundary). reference: also hand H3 the "
                         "soundtrack as a condition, then keep H3's sound.")
    ap.add_argument("--segment-prompt", action="append", default=[], metavar="TEXT",
                    dest="segment_prompts",
                    help="H3 only: prompt for one chained segment, in order. "
                         "Use when a long clip has distinct beats.")
    ap.add_argument("--negative", help="negative prompt (ignored by H3)")
    ap.add_argument("--tier", choices=["auto", "5b", "14b", "h3"], default="auto",
                    help="h3 = MiniMax-H3, the only tier that also generates AUDIO. "
                         "Implied by r2v; combine it with i2v to animate a still WITH "
                         "sound instead of silently on Wan.")
    ap.add_argument("--seconds", type=float, default=None,
                    help="clip length (default 3; with --ref-video, that video's length)")
    ap.add_argument("--fps", type=int, default=16, help="ignored by H3 (fixed 24)")
    ap.add_argument("--short", type=int, choices=[480, 576, 608, 720, 768],
                    default=None, dest="short_side",
                    help="short side (default 480; 720 with --ref-video); H3 is trained "
                         "at 768 (1344x768)")
    ap.add_argument("--steps", type=int, help="sampling steps (H3 default 20)")
    ec = ap.add_mutually_exclusive_group()
    ec.add_argument("--easycache", type=float, default=None, metavar="T",
                    help="H3: EasyCache reuse threshold (default 0.2, ~20%% faster, "
                         "near-identical output)")
    ec.add_argument("--no-easycache", dest="easycache", action="store_const", const=0.0,
                    help="H3: render every step (use to rule the cache out)")
    ap.add_argument("--portrait", action="store_true")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--fast", dest="fast", action="store_true", default=None,
                     help="Lightning 4+4 steps (default: auto by budget)")
    grp.add_argument("--quality", dest="fast", action="store_false",
                     help="full steps, no Lightning LoRA")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--anchor-retries", type=int, default=0, metavar="N",
                    help="h3 + --image only: if frame 0 does not come out as the "
                         "pinned image, re-roll the seed and render again, up to N "
                         "extra times. Whether H3 honours a pinned first frame is "
                         "decided by the seed (measured: the same request anchored "
                         "2 of 5 seeds), so a re-roll is the cheapest fix -- but "
                         "each retry is another full render.")
    ap.add_argument("--force", action="store_true",
                    help="submit even though an identical render is already in "
                         "flight. Almost never right: a terminal timeout kills "
                         "the waiting process, not the render, so the first one "
                         "is still going. Reconnect with `status <job_id>`.")
    ap.add_argument("--wait", action="store_true",
                    help="run synchronously (block until done); default is async job")
    a = ap.parse_args()

    # A linked install (install.py --link) re-renders its SKILL.md after a git
    # pull through a hook; this catches an update the hook did not see.
    import linkcheck
    linkcheck.refresh_if_stale()

    # crop: the head-and-shoulders cut the H3 docs keep asking for
    if a.mode == "crop":
        if not a.prompt or not a.rect:
            print(json.dumps({"ok": False, "error":
                              "usage: gen.py crop <image> --rect x,y,w,h [--out PATH]  "
                              "(fractions of the source, e.g. 0.05,0.02,0.40,0.35)"},
                             ensure_ascii=False)); sys.exit(1)
        res = video_gen.crop_image(a.prompt, a.rect, a.out)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res.get("ok") else 1)

    # llm-return: give back a local LLM that a render put to sleep or unloaded
    # (renders do this themselves; this is for a deferred or interrupted one)
    if a.mode == "llm-return":
        import llmsleep
        res = llmsleep.restore()
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res.get("returned") or res.get("why") == "nothing lent" else 1)

    # stop: interrupt the render, clear the queue, hand the VRAM back
    if a.mode == "stop":
        res = video_gen.stop_render(a.prompt or None)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res.get("ok") else 1)

    # status: just report a job's state. No argument means the newest job --
    # the id is on disk, so this still works when a terminal timeout threw away
    # everything gen.py printed.
    if a.mode == "status":
        job_id = a.prompt or video_gen.latest_job_id()
        if not job_id:
            print(json.dumps({"ok": False, "error":
                              "no job id given and none recorded yet. Usage: "
                              "gen.py status [job_id]"}, ensure_ascii=False))
            sys.exit(1)
        res = video_gen.job_status(job_id)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res.get("ok") else 1)

    dry = a.mode == "plan"
    # r2v is H3-only; asking for it implies the tier so the model cannot get a
    # silent Wan render with no audio. --last-frame is H3-only for the same
    # reason: Wan's graph has no end-frame input at all, so the flag would be
    # accepted and then quietly dropped.
    tier = a.tier
    if tier == "auto" and (a.mode == "r2v" or a.last_frame or a.ref_video):
        tier = "h3"
    # `plan` is a dry run but still has to price something, and on H3 the two
    # routes cost differently -- fl2va keyframes are one canvas each, r2v
    # references can be tens of thousands of tokens. Price whichever route the
    # same flags would actually render: --ref means r2v even alongside --image.
    plan_mode = ("r2v" if a.ref_video else
                 "i2v" if (tier != "h3" or (a.image and not a.ref)) else "r2v")
    # A recast is rendered shot by shot, so 720p costs chunk length rather than
    # clip length -- it is the default there; everything else keeps 480.
    short = a.short_side or (720 if a.ref_video else 480)
    req = {"mode": (plan_mode if dry else a.mode), "tier": tier,
           "fps": a.fps, "short_side": short}
    # A reference video brings its own length; everything else keeps the 3 s default.
    if a.seconds is not None or not a.ref_video:
        req["seconds"] = a.seconds if a.seconds is not None else 3.0
    # Send `portrait` only when it was actually asked for: H3 i2v takes the
    # orientation from the start frame when the caller stayed silent, and a
    # hard False there would read as "landscape, deliberately".
    if a.portrait: req["portrait"] = True
    if a.prompt:   req["prompt"] = a.prompt
    if a.image:    req["image"] = a.image
    if a.last_frame: req["last_frame"] = a.last_frame
    if a.ref:      req["refs"] = a.ref
    req["ref_image_size"] = a.ref_image_size
    if a.negative: req["negative"] = a.negative
    if a.steps is not None: req["steps"] = a.steps
    if a.easycache is not None:
        if not 0.0 <= a.easycache <= 3.0:
            ap.error("--easycache must be between 0 and 3 (0 = off)")
        req["easycache"] = a.easycache
    if a.segment_prompts: req["segment_prompts"] = a.segment_prompts
    if a.fast is not None: req["fast"] = a.fast
    if a.seed is not None: req["seed"] = a.seed
    if a.anchor_retries: req["anchor_retries"] = a.anchor_retries

    # --ref-video / --swap: recast a video. Checked first and as a block, because
    # every other H3 rule below is about pinned frames, which this route has none of.
    def fail(msg):
        print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False)); sys.exit(1)

    if a.swap and not a.ref_video:
        fail("--swap replaces a person IN a video, so it needs --ref-video <path>. "
             "To put a character into a new scene, use `r2v --ref <image>`.")
    if a.ref_video:
        if a.mode not in ("r2v", "plan"):
            fail("--ref-video works with r2v only: `r2v --ref-video V --swap \"WHO=IMG\"`.")
        if tier != "h3":
            fail("--ref-video is MiniMax-H3 only (drop --tier %s)." % tier)
        if a.image or a.last_frame:
            fail("--ref-video already decides the framing from frame 0 on, so it cannot "
                 "be combined with --image / --last-frame. Drop those.")
        if a.segment_prompts:
            fail("--ref-video is split at the video's own shots; --segment-prompt does not "
                 "apply. Use --swap ...@START-END to say who is replaced where.")
        if a.swap and a.ref:
            fail("give the pictures INSIDE each --swap (\"WHO=IMG|IMG\"), not as --ref: "
                 "with --swap every pass carries only its own person's pictures.")
        if not a.swap and not a.ref:
            fail("say who goes into the video: --swap \"the woman on the right=C:/her.png\" "
                 "(repeat once per person), or --ref <image> with a prompt that says what "
                 "to do with Video 1.")
        if not 128 <= a.ref_video_size <= 768:
            fail("--ref-video-size must be 128-768 (H3 never looks at more than 768).")
        swaps = []
        for s in a.swap:
            who, sep, imgs = s.partition("=")
            t0 = t1 = None
            body, at, rng = imgs.rpartition("@")
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)?\s*-\s*(\d+(?:\.\d+)?)?\s*", rng) if at else None
            if m:
                imgs = body
                t0 = float(m.group(1)) if m.group(1) else None
                t1 = float(m.group(2)) if m.group(2) else None
                if t0 is not None and t1 is not None and t1 <= t0:
                    fail("--swap %r: the range ends before it starts." % s)
            paths = [p.strip() for p in imgs.split("|") if p.strip()]
            if not sep or not who.strip() or not paths:
                fail("--swap %r is not WHO=IMG[@START-END]. Example: --swap \"the man on the "
                     "left=C:/bob.png@0-8.6\" (several images of him: C:/a.png|C:/b.png)." % s)
            if len(paths) > 9:
                fail("--swap %r has %d images; H3 takes at most 9." % (who, len(paths)))
            swaps.append({"who": who.strip(), "refs": paths, "start": t0, "end": t1})
        if len(swaps) > 6:
            fail("%d --swap given; 6 is the limit (a shot several of them cover is "
                 "rendered once per person, each pass following the last)." % len(swaps))
        for p in a.ref + [p for s in swaps for p in s["refs"]]:
            if not os.path.isfile(os.path.expanduser(p)):
                fail("image not found: %s" % p)
        # Without --swap the prompt itself has to say what to do with the video.
        if not swaps and not dry and not re.search(r"<?\bvideo\s*1\b", a.prompt or "", re.I):
            fail("with --ref-video and --ref, the prompt must say what to do with Video 1, "
                 "e.g. \"Replace the woman in Video 1 with the woman from Picture 1, keeping "
                 "Video 1's motion and camera\". Or use --swap, which writes that for you "
                 "(and is the only way to replace more than one person).")
        req["ref_video"] = a.ref_video
        req["ref_video_size"] = a.ref_video_size
        req["ref_video_start"] = a.ref_video_start
        req["ref_video_audio"] = a.ref_video_audio
        if swaps:
            req["swaps"] = swaps
        try:
            video_gen.probe_ref_video(req)          # fail now, not minutes into a job
        except Exception as e:
            fail(str(e))

    if a.mode == "i2v" and not a.image:
        print(json.dumps({"ok": False, "error": "i2v requires --image <start frame path>"},
                         ensure_ascii=False)); sys.exit(1)
    if a.last_frame and not (tier == "h3" and a.mode in ("i2v", "r2v", "plan")):
        print(json.dumps({"ok": False, "error":
                          "--last-frame needs MiniMax-H3: `i2v --tier h3 --image X "
                          "--last-frame Y`, or `r2v --ref ... --image X --last-frame Y`. "
                          "Wan's graph has no end-frame input at all."},
                         ensure_ascii=False)); sys.exit(1)
    if a.last_frame and a.mode == "r2v" and not a.image:
        print(json.dumps({"ok": False, "error":
                          "--last-frame pins the END of a clip whose START is pinned too, "
                          "so r2v also needs --image <first frame>. Without it there is no "
                          "keyframe track to attach an end frame to."},
                         ensure_ascii=False)); sys.exit(1)
    # A lone --image on r2v is the one genuinely ambiguous spelling: it used to
    # mean "use this as the reference", and now that r2v can pin a first frame
    # it just as plausibly means that. Guessing either way silently produces a
    # video the caller did not ask for, so make them say which.
    if a.mode == "r2v" and a.image and not a.ref:
        print(json.dumps({"ok": False, "error":
                          "`r2v --image X` with no --ref is ambiguous. Say which you mean: "
                          "`r2v --ref X` (X is a reference -- keep this character, H3 "
                          "composes the shot), `i2v --tier h3 --image X` (X is frame 0 -- "
                          "animate exactly this picture), or add --ref to get both, e.g. "
                          "`r2v --ref <person> --image X`."},
                         ensure_ascii=False)); sys.exit(1)
    # H3's fl2va graph has no ref_images input, so --ref here would be dropped
    # without a word and the caller would get a video missing their character.
    if a.mode == "i2v" and tier == "h3" and a.ref:
        print(json.dumps({"ok": False, "error":
                          "`i2v --tier h3` drives the clip from --image (it becomes "
                          "frame 0) and cannot take --ref. Pick one: `i2v --tier h3 "
                          "--image X` animates exactly that picture, with audio; "
                          "`r2v --ref X` keeps that character in a newly composed "
                          "shot, also with audio."},
                         ensure_ascii=False)); sys.exit(1)
    # A pinned first frame is only a condition, and without camera language H3
    # re-frames at frame 0 and never shows it. Measured 2026-09-09 across 15
    # renders on this box: 9 of the 15 without a camera cue opened 54-105/255
    # away from the pinned image; both runs WITH one landed at 7.4 (VAE+h264
    # noise, i.e. pixel-identical). Passing --image and getting the frame 40% of
    # the time is not a working feature, and the remedy is one sentence -- so
    # ask for it rather than hand back a clip that ignored the input.
    # A prompt that pins the camera AND asks for a close-up is not pinned: the
    # close-up wins and the frame is lost anyway. Measured 2026-09-10 in WSL --
    # round 1 ("camera stays wide and static") anchored at 7.06, round 2 kept
    # that clause but added "medium close shot at eye level, faces large and
    # sharp in frame" to fix a face, and frame 0 came back 47/255 away. Worth
    # its own message because the close-up advice comes from this skill's own
    # "face does not look like the reference" section, so the contradiction is
    # the documented path, not a slip.
    if (tier == "h3" and a.image and a.mode in ("i2v", "r2v") and not a.allow_recut
            and memplan.h3_camera_pinned(a.prompt or "")
            and memplan.h3_reframe_requested(a.prompt or "")):
        print(json.dumps({"ok": False, "error":
                          "the prompt pins the camera AND asks it to re-frame (a "
                          "close-up / push-in / cut). Those cannot both hold, and the "
                          "re-framing wins: measured 47-58/255 away from the pinned "
                          "frame, versus 7.1 for the same shot with the pin alone.\n"
                          "If you pinned --image, drop the close-up language. To make a "
                          "face read better WITHOUT re-framing, crop the reference "
                          "instead -- `gen.py crop <ref> --rect x,y,w,h` -- which is "
                          "free and is the bigger lever anyway; or pick a start frame "
                          "that is already the close shot you want.\n"
                          "If you actually want H3 to compose its own shots, drop "
                          "--image and use plain `r2v`, or pass --allow-recut."},
                         ensure_ascii=False)); sys.exit(1)

    if (tier == "h3" and a.image and a.mode in ("i2v", "r2v")
            and not a.allow_recut and not memplan.h3_camera_pinned(a.prompt or "")):
        print(json.dumps({"ok": False, "error":
                          "you pinned a first frame with --image, but the prompt does "
                          "not pin the CAMERA. H3 edits inside a single shot, so it "
                          "will very likely cut to its own framing at frame 0 and never "
                          "show your image (measured on this box: 9 of 15 renders "
                          "without a camera cue opened 54-105/255 away from the pinned "
                          "frame; both runs with one landed at 7.4, i.e. pixel-"
                          "identical). Add a clause such as:\n"
                          '  "The camera stays wide and static throughout, no cuts."\n'
                          "The clip still moves -- it just stops re-framing. If you "
                          "actually want H3 to recut freely, either drop --image and "
                          "use plain `r2v`, or pass --allow-recut."},
                         ensure_ascii=False)); sys.exit(1)

    if a.mode == "t2v" and not a.prompt:
        print(json.dumps({"ok": False, "error": "t2v requires a text prompt"},
                         ensure_ascii=False)); sys.exit(1)

    # t2v silently produces garbage here, which is worse than an error: the only
    # 14B workflow on this box is I2V (with an I2V-distilled Lightning LoRA), so
    # a text-only prompt gets no start frame and renders PURE NOISE -- and still
    # returns ok:true with an mp4 path, so it reads as a successful generation.
    # Block it and hand back the two-stage recipe that does work. `--tier 5b`
    # (Wan TI2V) stays reachable for anyone who explicitly wants real t2v.
    if a.mode == "t2v" and tier in ("auto", "14b"):
        print(json.dumps({"ok": False, "error":
                          "t2v is NOT usable on this machine and will render pure noise. "
                          "The 14B tier is Wan 2.2 I2V-A14B, an image-to-video model -- with "
                          "no start frame it samples from nothing. Use the two-stage route:\n"
                          "  1) \"%s\" \"%s\" zimage \"<the scene as a still image>\"\n"
                          "  2) \"%s\" \"%s\" i2v \"<how it should move>\" "
                          "--image \"<png from step 1>\" --seconds 3 --wait\n"
                          % (PY, os.path.join(SKILLS_DIR, "imggen", "gen.py"),
                             PY, os.path.join(SKILLS_DIR, "vidgen", "gen.py")) +
                          "This is the documented best practice for every text-only request, "
                          "not a workaround. (`--tier 5b` is the only real text-to-video model "
                          "here, but it is fp16 and is usually refused to protect the LLM.)"},
                         ensure_ascii=False)); sys.exit(1)
    if a.mode == "r2v" and not a.ref and not a.swap:
        print(json.dumps({"ok": False, "error":
                          "r2v requires at least one --ref <image path> "
                          "(the character/scene reference). Up to 9 allowed. Add "
                          "--image <first frame> as well to pin the opening shot."},
                         ensure_ascii=False)); sys.exit(1)
    if tier == "h3" and a.mode == "t2v":
        print(json.dumps({"ok": False, "error":
                          "H3 needs a picture to work from: `r2v --ref <path>` to keep "
                          "a character, or `i2v --tier h3 --image <path>` to animate a "
                          "still. With neither, make one with the imggen skill first."},
                         ensure_ascii=False)); sys.exit(1)

    # With several references, H3 has no way to know which person is which
    # unless the prompt says so. Silently accepting a generic prompt produces a
    # video that ignores the references entirely, which looks like the model
    # working when it is not. Fail loudly instead.
    # Not on a dry run: `plan` carries no prompt to inspect, and the real call
    # enforces this anyway. Applying it here made every multi-character
    # `plan --tier h3 --ref A --ref B` fail, which silently disabled dramagen's
    # headroom gate (it treats a non-VRAM refusal as "go ahead").
    if len(a.ref) > 1 and not dry:
        texts = " ".join([a.prompt or ""] + list(a.segment_prompts))
        # Accept "Picture 1" as readily as "<Picture 1>": small models reliably
        # write the words but not the angle brackets, and rejecting the bare
        # form just puts them in a retry loop. The engine adds the brackets.
        if not re.search(r"<?\bpicture\s*\d", texts, re.I):
            names = ", ".join("Picture %d = %s" % (i + 1, p)
                              for i, p in enumerate(a.ref))
            print(json.dumps({"ok": False, "error":
                              "You passed %d reference images, so the prompt MUST say which "
                              "is which by writing Picture 1 / Picture 2 (in --ref order: %s). "
                              "Without it H3 ignores the references and invents generic "
                              "characters. Example: \"The man from Picture 1 duels the man "
                              "from Picture 2 in a courtyard\"." % (len(a.ref), names)},
                             ensure_ascii=False)); sys.exit(1)

    if dry:
        res = video_gen.run(req, dry=True)
    elif a.wait:
        # Submit as a job (so `status` and the job file still work), then block
        # HERE in Python until it lands. The point is that nothing wakes the
        # local LLM while the render runs -- polling from the agent side costs a
        # full 35B inference per check, and it re-reads the weights from SSD.
        res = video_gen.submit_async(req, force=a.force)
        if res.get("ok"):
            job_id = res["job_id"]
            # Hand out the job handle BEFORE blocking. The caller's terminal
            # kills this process at its own timeout (600 s here), but the render
            # is a detached worker talking to a separate ComfyUI server -- it
            # keeps going and still writes its mp4. Measured: a --quality 3 s
            # 480p clip takes ~630 s, so it trips that limit while succeeding.
            # Without the id in the captured output the caller has no way back
            # in and re-submits the entire render, so print it up front. Goes to
            # stderr (which terminals still capture) to keep stdout pure JSON.
            print("[vidgen] job_id=%s -- rendering now.\n"
                  "[vidgen] If this command times out, the render is STILL RUNNING "
                  "and will finish on its own. Do NOT re-submit. Reconnect with:\n"
                  "[vidgen]   \"%s\" \"%s\" status %s\n"
                  "[vidgen] %s"
                  % (job_id, PY, os.path.abspath(__file__), job_id,
                     video_gen.busy_notice()),
                  file=sys.stderr, flush=True)
            while True:
                time.sleep(3)
                st = video_gen.job_status(job_id)
                if st.get("status") in ("done", "error"):
                    res = st
                    break
    else:
        res = video_gen.submit_async(req, force=a.force)   # default: job_id at once
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0 if res.get("ok") else 1)


if __name__ == "__main__":
    main()
