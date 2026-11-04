"""Hermes video-generation wrapper (thick) for local ComfyUI + Wan 2.2.

The LLM side is deliberately THIN: it fills a few named fields and calls this.
Everything hard -- reading the live VRAM/RAM budget, picking a model tier,
choosing the VRAM->DRAM->SSD offload path, patching the ComfyUI graph, polling,
and locating the mp4 -- happens here. Output is a single JSON object on stdout
so the model just reads fields; human status goes to stderr.

Requires the protected ComfyUI to be up (start_comfy.py keeps --reserve-vram
sized so the hosted LLM is never evicted). This wrapper never changes the reserve.

  # what can I make right now? (safe, no render)
  python video_gen.py plan '{"mode":"i2v","seconds":3}'

  # generate
  python video_gen.py run '{"mode":"i2v","image":"C:/path/first.png",
                            "prompt":"slow dolly-in, neon rain","seconds":3}'
  python video_gen.py run '{"mode":"t2v","tier":"5b","prompt":"a fox runs"}'

req fields (all optional except prompt for t2v / image for i2v):
  mode   't2v'|'i2v'         tier 'auto'|'5b'|'14b'
  prompt / negative          image (start frame, i2v)
  seconds  fps  short_side(480|576|720)  portrait(bool)
  fast  (bool: Lightning 4+4 steps; default auto by budget)
  seed
"""
from __future__ import annotations
import os, re, sys, json, time, uuid, shutil, subprocess, threading, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths          # resolves hermes-comfy.conf into os.environ -- import first
import astapi         # the only thing that talks to the TRUSTA-AST backend
import osmem          # per-OS process primitives (detach flags, liveness)
import memplan
import llmsleep       # lends the GPU of an LLM that fills it (Strata) to a render

# Every path below is resolved by paths.py, which knows that a Windows portable
# build nests ComfyUI one level deeper than a Linux clone. Nothing here may
# assume either shape.
COMFY_URL = paths.COMFY_URL
COMFY_HOME = paths.COMFY_HOME
LAUNCHER = os.path.join(HERE, "start_comfy.py")
WF_DIR = paths.WF_DIR
COMFY_INPUT = paths.COMFY_INPUT
COMFY_OUTPUT = paths.COMFY_OUTPUT


# Handed back on every submit so the agent can close its turn with it verbatim.
# Each extra prompt during a render is a full 35B inference that faults the
# weights we just handed to ComfyUI straight back off the SSD -- it fights the
# render for RAM and disk, and the answer itself is slow. Saying so is the only
# way the user knows that waiting is the correct action.
BUSY_NOTICE = (
    "⚠️ ComfyUI 正在算圖,本機 LLM 佔用的 DRAM 已交還給它。"
    "在這次算完之前,請不要再下任何 prompt 或指令 —— 每一次呼叫都會把 LLM 的權重"
    "從 SSD 重新讀回記憶體,不但那次回答會很慢,還會跟 ComfyUI 搶記憶體與磁碟,"
    "拖慢甚至弄壞這次算圖。算完我會回報輸出路徑。"
)

# The LLM gave the whole GPU to the render (llmsleep: asleep or unloaded). Any
# message now cannot be answered at all until the render ends and it is back.
_BUSY_NOTICE_LENT = (
    "⚠️ ComfyUI 正在算圖,本機 LLM 已暫時休眠/卸載,把顯示卡整個讓給算圖。"
    "在這次算完之前請不要再下任何 prompt 或指令 —— LLM 不在線上,任何對話都會失敗。"
    "算完會自動把 LLM 載回(約 40 秒),然後我會回報輸出路徑。"
)

_BUSY_NOTICE_NO_LLM = (
    "⚠️ ComfyUI 正在算圖。在這次算完之前,請不要再下任何會用到 GPU 的指令 —— "
    "會跟 ComfyUI 搶顯示記憶體與磁碟,拖慢甚至弄壞這次算圖。算完我會回報輸出路徑。"
)


def busy_notice() -> str:
    """The standing 'do not disturb' notice, matched to what actually happened.

    The DRAM sentence is only true when a co-resident LLM was visible and its
    working set was trimmed. On a split install (client in WSL, the LLM on
    Windows) /proc cannot see that process and nothing was handed back. Stating
    it anyway would put a claim the user can neither verify nor act on into the
    one message whose entire job is to explain why things are slow -- so the
    reason given always matches the mechanism that is really in play.
    """
    try:
        if llmsleep.is_lent() or llmsleep.lendable():
            return _BUSY_NOTICE_LENT
    except Exception:
        pass
    try:
        visible = bool(memplan.llm_dram().get("found"))
    except Exception:
        visible = False
    return BUSY_NOTICE if visible else _BUSY_NOTICE_NO_LLM

# After the render: the trim is why the next turn feels sluggish. Without this
# the slowness reads as a hang and invites exactly the retry we want to avoid.
RESUME_NOTE = (
    "算圖已結束,記憶體已還給 LLM。下一次回答會慢一點(權重要從 SSD 重新載入),"
    "這是正常現象,不是當機。"
)

_RESUME_NOTE_NO_LLM = "算圖已結束,GPU 已釋放。"


def resume_note() -> str:
    """Post-render note. Conditional for the same reason as busy_notice():
    'your weights are being re-read from SSD' is only true if we actually
    trimmed a local LLM."""
    try:
        visible = bool(memplan.llm_dram().get("found"))
    except Exception:
        visible = False
    return RESUME_NOTE if visible else _RESUME_NOTE_NO_LLM


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


_URL_RESOLVED = False


def _resolve_comfy_url():
    """Pick the first candidate URL that answers, and remember it.

    Exists for WSL: in NAT mode a Windows ComfyUI is not on the distro's
    loopback, it is on the default gateway. Probing beats pinning that address
    in the config, which goes stale every time WSL restarts. memplan keeps its
    own copy for the /queue ordering check, so it is rebound here too -- a
    mismatch there would silently disable the LLM trim.
    """
    global COMFY_URL, _URL_RESOLVED
    if _URL_RESOLVED:
        return COMFY_URL
    cands = paths.comfy_url_candidates()
    for u in cands:
        try:
            with urllib.request.urlopen(f"{u}/system_stats", timeout=5):
                pass
        except Exception:
            continue
        if u != COMFY_URL:
            _log(f"[boot] ComfyUI reached at {u} (configured {COMFY_URL})")
        COMFY_URL = u
        memplan.COMFY_URL = u
        _URL_RESOLVED = True
        return u
    return COMFY_URL


def _api(path, data=None):
    """One ComfyUI API call. An empty 200 is success, not a parse error.

    `/free`, `/interrupt` and `/queue {"clear": true}` all answer 200 with a
    zero-byte body, so an unconditional json.loads("") raised on every one of
    them -- the models WERE unloaded and the queue WAS cleared, while the log
    said the call failed. Wrong in the one place this engine most needs to be
    believed: an operator reading that log concludes the handback is broken
    when it is working, and `gen.py stop` reported ok:false after doing its job.
    """
    url = f"{COMFY_URL}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode().strip()
    return json.loads(raw) if raw else {}


def _require_comfy():
    """Ensure the memory-protected ComfyUI is up; launch it if not (idempotent).

    start_comfy.py sizes --reserve-vram from measured external usage, so the
    LLM the backend is hosting is never evicted when video weights load.
    When ComfyUI belongs to another OS (paths.COMFY_REMOTE -- the WSL split),
    we cannot launch or protect it, so we say exactly what the user must run
    over there instead of failing with a connection error.
    """
    _resolve_comfy_url()
    try:
        _api("/system_stats")
        # On another card than the one picked in AI Scaler Toolkit's settings?
        # Stopped if idle, and launched again below on the right one.
        import comfyguard
        stopped, note = comfyguard.realign(COMFY_URL, paths.SELECTION.get("device"))
        if note:
            _log("[boot] " + note)
        if stopped:
            raise RuntimeError("restarting on the selected card")
        # Already up: we did not launch it, so its --reserve-vram is whatever
        # someone else chose. Say so rather than assume the guarantee holds.
        try:
            import comfyguard
            # `is not False` on purpose: llm_loaded() returns None when the
            # backend cannot be reached, and an unreachable backend must not
            # silence the warning -- assume something is there to protect.
            msg = comfyguard.check(
                COMFY_URL, astapi.llm_loaded() is not False, HERE)
            if msg:
                _log(msg)
        except Exception:
            pass
        return
    except Exception:
        pass
    if paths.COMFY_REMOTE:
        raise RuntimeError(
            "ComfyUI is not reachable at any of %s, and it is not ours to "
            "start (COMFY_REMOTE): it lives on the host, not in this "
            "environment. Start it over there with the protected launcher so "
            "--reserve-vram still shields the local LLM:\n"
            "    python <engine>/start_comfy.py\n"
            "It listens on 127.0.0.1 only. From a NAT-mode WSL distro that is "
            "out of reach: turn on WSL mirrored networking "
            "(%%USERPROFILE%%\\.wslconfig: [wsl2] networkingMode=mirrored, then "
            "`wsl --shutdown`) and loopback reaches it."
            % ([u for u in paths.comfy_url_candidates()],))
    _log("[boot] ComfyUI down -> launching protected server (reserve-vram)...")
    import subprocess
    subprocess.run([sys.executable, LAUNCHER], check=True)
    for _ in range(150):
        try:
            _api("/system_stats"); return
        except Exception:
            time.sleep(2)
    raise RuntimeError("ComfyUI failed to become ready")


# ComfyUI's LoadImage goes through PIL, which cannot open these as shipped.
# Staging converts them instead of copying, so a user can hand us an iPhone
# photo path directly.
_NEEDS_CONVERT = {".heic", ".heif", ".avif", ".webp", ".bmp", ".tif", ".tiff"}


def _stage_image(path: str) -> str:
    """Put the user's image into ComfyUI/input and return the bare name.

    Converts formats PIL cannot open unaided (HEIC from iPhones is the common
    one) and downscales oversized phone photos on the way in.
    """
    path = os.path.expanduser(path or "")          # small models often pass ~/...
    if not path or not os.path.exists(path):
        raise FileNotFoundError("start image not found: %r" % path)
    os.makedirs(COMFY_INPUT, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in _NEEDS_CONVERT:
        name = "hermes_%s.png" % uuid.uuid4().hex[:8]
        _media("convert", path, os.path.join(COMFY_INPUT, name))
        return name
    name = "hermes_%s%s" % (uuid.uuid4().hex[:8], ext)
    shutil.copyfile(path, os.path.join(COMFY_INPUT, name))
    return name


def crop_image(src: str, rect: str, out: str | None = None) -> dict:
    """Crop `src` to `rect` = "x,y,w,h" as fractions of the source (0-1).

    Runs under ComfyUI's interpreter via h3_media, so it inherits the same
    format support as the staging path -- a HEIC reference crops as readily as
    a jpg, which a hand-rolled PIL snippet in the caller's environment would
    not.
    """
    src = os.path.expanduser(src or "")
    if not os.path.isfile(src):
        return {"ok": False, "error": "image not found: %s" % src}
    try:
        parts = [float(v) for v in str(rect).replace(" ", "").split(",")]
        if len(parts) != 4:
            raise ValueError
        x, y, w, h = parts
    except Exception:
        return {"ok": False, "error": "rect must be 'x,y,w,h' as fractions of the "
                                      "image, e.g. 0.05,0.02,0.40,0.35 (got %r)" % rect}
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
            and x + w <= 1.0001 and y + h <= 1.0001):
        return {"ok": False, "error": "rect out of range: x,y must be in [0,1) and "
                                      "x+w, y+h must not exceed 1 (got %s)" % rect}
    if not out:
        base, ext = os.path.splitext(os.path.basename(src))
        out = os.path.join(paths.OUT_DIR if hasattr(paths, "OUT_DIR")
                           else os.path.join(HERE, "out"), "%s_crop%s" % (base, ext or ".png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    try:
        res = _media("crop", src, out, "%g,%g,%g,%g" % (x, y, w, h))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    # h3_media prints "<path> WxH (from WxH)"; hand those back as numbers so the
    # caller never has to shell out to measure what it just asked us to make.
    size = src_size = None
    try:
        parts = (res or "").split()
        size = [int(v) for v in parts[-3].split("x")]
        src_size = [int(v) for v in parts[-1].rstrip(")").split("x")]
    except Exception:
        pass
    return {"ok": True, "file": os.path.abspath(out), "rect": [x, y, w, h],
            "size": size, "source_size": src_size,
            "note": res or "cropped",
            "next": "pass it with --ref; cropping to head-and-shoulders beats "
                    "--ref-image-size max for identity and costs far fewer tokens"}


def _load_wf(name: str) -> dict:
    with open(os.path.join(WF_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# H3's tokenizer labels each reference image with the literal token "<Picture N>:"
# (see comfy/text_encoders/minimax.py), so the prompt lines up best when it uses
# exactly that form. Small models write "Picture 1" and drop the brackets, so we
# add them here rather than bouncing the request back at them.
_PIC_RE = re.compile(r"<?\bpicture\s*(\d+)>?", re.I)


def _normalise_picture_refs(text: str) -> str:
    return _PIC_RE.sub(lambda m: "<Picture %s>" % m.group(1), text or "")


# Same for reference videos. Only applied when a --ref-video is wired, so an
# ordinary prompt that happens to say "video 2" is left alone.
_VID_RE = re.compile(r"<?\bvideo\s*(\d+)>?", re.I)


def _normalise_video_refs(text: str) -> str:
    return _VID_RE.sub(lambda m: "<Video %s>" % m.group(1), text or "")


def build_h3_graph(plan: dict, req: dict, seg_idx: int = 0,
                   first_frame: str | None = None, prefix: str = "video/h3") -> dict:
    """Patch a MiniMax-H3 graph.

    Which of H3's two graphs is used depends on the MODE, not only on the
    segment index:

      ref2va     r2v. The --ref images establish the cast and the prompt
                 addresses them as <Picture N>. H3 keeps the likeness but
                 re-invents the framing, so the clip does NOT start on the
                 pixels you handed it.
      fl2va      i2v. --image IS frame 0, so the shot starts on exactly that
                 picture; --last-frame optionally pins the final frame.
      ref2va+kf  r2v carrying an --image: both at once, on the ref2va weights.
                 Needs ComfyUI/custom_nodes/hermes_minimax_h3_refkf, which is
                 also where the reason core cannot do this is written down.
      seg >0     fl2va chained on the previous segment's last frame, whatever
                 the route -- H3's native continuity mechanism.

    memplan.h3_route() decides; this only loads what it chose.

    fl2va still decodes an audio latent (the AV latent is built by the node, and
    node 14 decodes it either way), so H3 i2v has sound exactly like r2v.

    NOTE ON REFERENCE IMAGES: the API name is the dotted autogrow path
    `ref_images.ref_image_N`, NOT `ref_image_N`. ComfyUI regroups the dotted
    keys back into the node's `ref_images` dict; the bare form raises
    "unexpected keyword argument 'ref_image_0'".
    """
    chained = seg_idx > 0 and first_frame is not None
    # A pinned frame means "start on THESE pixels". Handing that file to ref2va
    # instead (which is what happened before this was wired) demotes it to a
    # *reference*: the face survives, the composition does not, and the result
    # reads as a different shot of the same person rather than the picture
    # moving. Chained segments are always fl2va regardless of the route.
    route = plan.get("h3_route") or ("fl2va" if plan.get("mode") == "i2v" else "ref2va")
    keyframed = not chained and route in ("fl2va", "ref2va+kf")
    g = _load_wf(plan["template_chain"] if (chained or route == "fl2va")
                 else plan["template_refkf"] if route == "ref2va+kf"
                 else plan["template"])

    seed = int(req.get("seed", int(time.time()) % 2_000_000_000)) + seg_idx
    prompts = req.get("segment_prompts") or []
    prompt = (prompts[seg_idx] if seg_idx < len(prompts)
              else req.get("prompt", "a cinematic shot"))
    prompt = _normalise_picture_refs(prompt)
    ref_video = req.get("_ref_video_file")
    if ref_video:
        prompt = _normalise_video_refs(prompt)

    g["2"]["inputs"].update(shift_video=plan["shift_video"],
                            shift_audio=plan["shift_audio"])
    g["10"]["inputs"]["steps"] = plan["steps"]
    g["11"]["inputs"]["noise_seed"] = seed
    g["15"]["inputs"]["fps"] = float(plan["fps"])
    g["16"]["inputs"]["filename_prefix"] = prefix

    n7 = g["7"]["inputs"]
    n7.update(prompt=prompt, width=plan["width"], height=plan["height"],
              length=plan["segment_length"])

    if chained:
        g["6"]["inputs"]["image"] = first_frame          # already staged
    elif route == "fl2va":
        # fl2va has no ref_images input at all, so silently dropping --ref here
        # would hand back a video with none of the requested characters in it.
        if req.get("refs"):
            raise ValueError(
                "H3 i2v drives the shot from --image (frame 0) and its graph has "
                "no reference-image input, so --ref would be ignored. Use "
                "`r2v --ref ... --image ...` to have both, or `r2v --ref ...` to "
                "carry a character into a scene H3 composes itself.")
        start = req.get("image")
        if not start:
            raise ValueError("H3 i2v needs --image <start frame>")
        g["6"]["inputs"]["image"] = _stage_image(start)
    else:
        if route == "ref2va+kf":
            # node 6 is the first_frame slot on this template, not a ref loader
            g["6"]["inputs"]["image"] = _stage_image(req["image"])
        refs = memplan.h3_refs(req)
        if not refs:
            raise ValueError("H3 needs at least one reference image (--ref/--image)")
        if len(refs) > 9:
            raise ValueError("H3 accepts at most 9 reference images, got %d" % len(refs))
        # Backstop for the same rule gen.py enforces: with more than one
        # reference the prompt has to bind them, or H3 quietly ignores them.
        if len(refs) > 1 and not _PIC_RE.search(prompt):
            raise ValueError(
                "%d reference images were given but the prompt never mentions "
                "Picture 1 / Picture 2, so H3 cannot tell them apart and will ignore "
                "them. Name them in --ref order, e.g. 'the man from Picture 1 duels "
                "the man from Picture 2'." % len(refs))
        n7.pop("ref_images.ref_image_0", None)
        node_id = 100
        for i, path in enumerate(refs):
            g[str(node_id)] = {"class_type": "LoadImage",
                               "inputs": {"image": _stage_image(path)}}
            n7["ref_images.ref_image_%d" % i] = [str(node_id), 0]
            node_id += 1
        n7["ref_image_size"] = req.get("ref_image_size", "match")

        # --ref-video: the upstream node's own <Video 1> input, already
        # normalised by _stage_ref_video. Same dotted autogrow naming as above.
        if ref_video:
            if route != "ref2va":
                raise ValueError("a reference video only works on plain r2v (ref2va); "
                                 "it cannot be combined with --image")
            if not re.search(r"<Video 1>", prompt):
                raise ValueError(
                    "a reference video was given but the prompt never mentions Video 1, "
                    "so H3 has no instruction to follow it. Say what to do with it, e.g. "
                    "'Replace the woman in Video 1 with the woman from Picture 1, keeping "
                    "Video 1's motion and camera' -- or use --swap, which writes that for you.")
            g["200"] = {"class_type": "LoadVideo", "inputs": {"file": ref_video}}
            g["201"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["200", 0]}}
            n7["ref_videos.ref_video_0"] = ["201", 0]
            if req.get("_ref_audio_file"):
                g["202"] = {"class_type": "LoadAudio",
                            "inputs": {"audio": req["_ref_audio_file"]}}
                n7["ref_video_audios.ref_video_audio_0"] = ["202", 0]

    # --last-frame pins the END of the clip, so it goes on whichever segment is
    # genuinely last -- and only on the fl2va graph, the only one with the
    # input. Keeping it out here rather than inside the keyframe branch is what
    # stops a chained i2v (VIDGEN_H3_ALLOW_CHAIN=1) from dropping it silently:
    # that branch only ever runs for segment 0, which is not the end of a chain.
    last = req.get("last_frame")
    if last and (chained or keyframed) and seg_idx == max(0, int(plan.get("segments", 1)) - 1):
        # the chained template's last_frame slot is on the same node 7
        g["150"] = {"class_type": "LoadImage",
                    "inputs": {"image": _stage_image(last)}}
        n7["last_frame"] = ["150", 0]

    # EasyCache sits between the patched model and the guider, so it wraps
    # whatever the template applied (sigma shift, the refkf patch). The planner
    # chose the threshold (memplan.h3_easycache); 0 leaves the graph untouched.
    ec = float(plan.get("easycache") or 0)
    if ec > 0:
        gid = next(k for k, n in g.items()
                   if isinstance(n, dict) and n.get("class_type") == "BasicGuider")
        g["300"] = {"class_type": "EasyCache", "inputs": {
            "model": g[gid]["inputs"]["model"], "reuse_threshold": ec,
            "start_percent": 0.15, "end_percent": 0.95, "verbose": False}}
        g[gid]["inputs"]["model"] = ["300", 0]
    return g


def build_graph(plan: dict, req: dict) -> dict:
    """Load the chosen template and patch it from the plan + user request."""
    if plan["tier"] == "h3":
        return build_h3_graph(plan, req)
    with open(os.path.join(WF_DIR, plan["template"]), encoding="utf-8") as f:
        g = json.load(f)

    prompt = req.get("prompt", "a cinematic shot")
    negative = req.get("negative",
                       "blurry, low quality, distorted, watermark, static, flicker")
    seed = int(req.get("seed", int(time.time()) % 2_000_000_000))
    mode = plan["mode"]

    if plan["tier"] == "5b":
        g["11"]["inputs"]["shift"] = plan["model"]["shift"]
        g["13"]["inputs"]["text"] = prompt
        g["14"]["inputs"]["text"] = negative
        wtv, ks = g["17"]["inputs"], g["18"]["inputs"]
        wtv.update(width=plan["width"], height=plan["height"], length=plan["length"])
        ks.update(seed=seed, steps=plan["steps"], cfg=plan["cfg"])
        g["20"]["inputs"]["fps"] = float(plan["fps"])
        if mode == "t2v":
            wtv.pop("start_image", None); g.pop("16", None)
        else:
            g["16"]["inputs"]["image"] = _stage_image(req.get("image", ""))

    else:  # 14b dual-expert GGUF
        g["12"]["inputs"]["shift"] = plan["model"]["shift"]
        g["13"]["inputs"]["shift"] = plan["model"]["shift"]
        g["17"]["inputs"]["text"] = prompt
        g["18"]["inputs"]["text"] = negative
        wtv = g["21"]["inputs"]
        wtv.update(width=plan["width"], height=plan["height"], length=plan["length"])
        hi, lo = g["22"]["inputs"], g["23"]["inputs"]
        total, split = plan["steps"], plan["split"]
        hi.update(noise_seed=seed, steps=total, cfg=plan["cfg"], end_at_step=split)
        lo.update(noise_seed=seed, steps=total, cfg=plan["cfg"],
                  start_at_step=split, end_at_step=total)
        g["25"]["inputs"]["fps"] = float(plan["fps"])
        if not plan["fast"]:
            # quality mode: bypass Lightning LoRAs (samplers read ModelSampling directly)
            hi["model"] = ["12", 0]; lo["model"] = ["13", 0]
            g.pop("14", None); g.pop("15", None)
        if mode == "t2v":
            wtv.pop("start_image", None); g.pop("20", None)
        else:
            g["20"]["inputs"]["image"] = _stage_image(req.get("image", ""))
    return g


def _in_queue(prompt_id: str) -> bool | None:
    """Is this prompt still queued or running? None if we cannot tell."""
    try:
        q = _api("/queue")
    except Exception:
        return None
    for key in ("queue_running", "queue_pending"):
        for entry in q.get(key, []):
            # entries look like [index, prompt_id, prompt, extra, outputs]
            if len(entry) > 1 and entry[1] == prompt_id:
                return True
    return False


def _is_executing(prompt_id: str) -> bool:
    """Strictly: is ComfyUI executing THIS prompt right now (queue_running)?"""
    try:
        q = _api("/queue")
    except Exception:
        return False
    return any(len(e) > 1 and e[1] == prompt_id for e in q.get("queue_running", []))


_LLM_TRIMMED = False          # one-shot per worker process (H3 chains re-enter)


def _release_llm_when_running(prompt_id: str, timeout_s: float = 120.0) -> None:
    """Hand the LLM's DRAM back -- only after THIS prompt is confirmed running.

    Ordering rule, and the reason it is code rather than a doc note: the agent
    driving this pipeline IS the co-resident LLM. Trimming its working set makes
    its next token fault ~10 GB of GGUF back off the SSD. Do that before the
    render is really under way and we pay that cost for nothing every time the
    launch fails or the preflight refuses -- the agent just appears to hang.

    So we wait for the prompt to appear in queue_running, then ask memplan
    (which re-verifies independently) to trim. Best-effort throughout: a
    render must never fail because the trim did not happen.
    """
    global _LLM_TRIMMED
    if _LLM_TRIMMED:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _is_executing(prompt_id):
            break
        time.sleep(2)
    else:
        _log("[llm] not trimming: %s never reached queue_running in %.0fs"
             % (prompt_id, timeout_s))
        return
    try:
        res = memplan.release_llm_dram(reason="render %s is executing" % prompt_id)
    except Exception as e:
        _log("[llm] trim skipped: %r" % (e,)); return
    _LLM_TRIMMED = True
    if res.get("released"):
        _log("[llm] DRAM handed back: %.1f GB (working set %.1f -> %.1f GB); "
             "LLM still alive, VRAM untouched"
             % (res.get("freed_gb") or 0.0, res.get("ws_before_gb") or 0.0,
                res.get("ws_after_gb") or 0.0))
    else:
        _log("[llm] not trimmed:", res.get("why", "?"))


def _poll(prompt_id: str, timeout_s: int = 5400) -> dict:
    """Wait for /history[prompt_id]; return the SaveVideo output entry.

    A restarted ComfyUI loses both its queue and its history, so a prompt we
    submitted simply ceases to exist. Polling history alone cannot tell that
    apart from "still working" and spins for the whole timeout -- which cost a
    90-minute stall once, while the finished segments sat on disk. So we also
    watch the queue: once the prompt is neither in history nor queued for
    several consecutive checks, we declare it gone instead of waiting.
    """
    t0 = time.time()
    seen = False              # have we ever observed it queued/running?
    missing_since = None
    GRACE_NEW = 90.0          # allow this long for it to appear at all
    GRACE_GONE = 20.0         # ...and this long to disappear before we give up
    while time.time() - t0 < timeout_s:
        try:
            h = _api(f"/history/{prompt_id}")
        except Exception:
            h = {}
        if prompt_id not in h:
            q = _in_queue(prompt_id)
            if q is True:
                seen, missing_since = True, None
            elif q is False:
                if missing_since is None:
                    missing_since = time.time()
                gone_for = time.time() - missing_since
                if seen and gone_for > GRACE_GONE:
                    return {"error": "prompt %s vanished from ComfyUI (server restarted "
                                     "mid-render?); finished segments are still on disk"
                                     % prompt_id}
                if not seen and time.time() - t0 > GRACE_NEW:
                    return {"error": "prompt %s never appeared in ComfyUI's queue or "
                                     "history within %.0fs" % (prompt_id, GRACE_NEW)}
            # q is None -> server unreachable; keep waiting, it may be restarting
        if prompt_id in h:
            rec = h[prompt_id]
            status = rec.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False and status.get("messages"):
                # surface the first error message if present
                for m in status.get("messages", []):
                    if m and m[0] in ("execution_error", "execution_interrupted"):
                        return {"error": m[1]}
            for _nid, out in rec.get("outputs", {}).items():
                for key in ("videos", "gifs", "images"):
                    for v in out.get(key) or []:
                        # ComfyUI 0.37 also lists the LoadVideo preview of recast's
                        # reference clip here, as type "input" and ahead of the
                        # SaveVideo node; taking it failed every recast assemble.
                        if v.get("type", "output") != "output":
                            continue
                        return {"file": v.get("filename"), "subfolder": v.get("subfolder", ""),
                                "type": "output"}
            if status.get("completed"):
                return {"error": "completed but no video output found"}
        time.sleep(2)
    return {"error": "timed out after %ds" % timeout_s}


def _free_before_planning():
    """Unload whatever ComfyUI still holds before we measure.

    Length is decided from free VRAM at plan time, so a still-resident image
    model from a previous job silently shortens the clip -- a 5-second request
    right after generating a start frame came back as 17 frames. Freeing costs
    nothing here because every render already ends with cache_none=True, so the
    next job was going to be a cold load either way.

    Resolves the endpoint first: this runs BEFORE _require_comfy(), so on a
    split install (WSL client, Windows ComfyUI) the URL is still the unusable
    loopback default. The /free would then fail silently and the planner would
    size the clip against VRAM a stale model was still holding.
    """
    _resolve_comfy_url()
    try:
        _api("/free", {"unload_models": True, "free_memory": True})
        time.sleep(5)          # let the driver actually hand the VRAM back
    except Exception:
        pass                   # ComfyUI may not be up yet; planning still works


COMFY_PY = paths.COMFY_PYTHON
H3_MEDIA = os.path.join(HERE, "h3_media.py")
PREWARM_TOOL = os.path.join(HERE, "prewarm_llm.py")
PREWARM_TIMEOUT = float(os.environ.get("VIDGEN_PREWARM_TIMEOUT", "40"))


def _prewarm_llm():
    """Refill the page cache with the LLM's GGUF before we hand control back.

    This render trimmed the LLM's working set. Its next turn would fault those
    weights back as scattered small reads at QD1 -- the worst regime for an
    NVMe drive (measured on the reference box: ~477 MB/s, a 41.9 s turn). A
    sequential read moves the same bytes at ~1,785 MB/s and leaves them in the
    OS page cache, so those faults become RAM hits (16.1 s turn).

    Best-effort throughout -- never raises, never fails a render that already
    succeeded, and a missing tool is just a log line.
    """
    if os.environ.get("VIDGEN_PREWARM", "1") != "1":
        _log("[prewarm] skipped (VIDGEN_PREWARM=0)")
        return
    if not os.path.isfile(PREWARM_TOOL):
        _log("[prewarm] tool missing:", PREWARM_TOOL)
        return
    t0 = time.time()
    try:
        p = subprocess.run([COMFY_PY, PREWARM_TOOL], capture_output=True,
                           text=True, timeout=PREWARM_TIMEOUT)
        _log("[prewarm]", (p.stdout or p.stderr or "").strip()[:200],
             "(%.1fs)" % (time.time() - t0))
    except subprocess.TimeoutExpired:
        _log("[prewarm] timed out after %.0fs -- partial warm still helps"
             % PREWARM_TIMEOUT)
    except Exception as e:
        _log("[warn] prewarm failed:", e)


def _media(*args) -> str:
    """Run h3_media.py under ComfyUI's own python (it is the one with PyAV)."""
    if not os.path.exists(COMFY_PY):
        raise RuntimeError(
            "ComfyUI's python not found at %s. Set COMFY_PYTHON in %s "
            "(portable: <COMFY_HOME>/python_embeded/python.exe; "
            "Linux: <COMFY_HOME>/venv/bin/python)."
            % (COMFY_PY, paths.CONF_FILE or paths.CONF_NAME))
    p = subprocess.run([COMFY_PY, H3_MEDIA, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("h3_media %s failed: %s" % (args[0], (p.stderr or "")[-500:]))
    return p.stdout.strip()


# Windows error 1450 (ERROR_NO_SYSTEM_RESOURCES) from comfy_aimdo while it
# streams a model's weights into pinned memory. Transient: seen in the logs on
# 08-28, 09-09 and twice on 10-27, and the identical graph resubmitted at once
# went through every time. A multi-pass --swap job multiplies the exposure, so
# one quiet retry is worth more than failing a 10-minute job on pass 2.
_TRANSIENT_ERRORS = ("HostBuffer.read_file_slice failed",)


def _submit_and_wait(graph: dict, label: str) -> str:
    """Submit one graph, wait for it, return the absolute output path."""
    try:
        return _submit_and_wait_once(graph, label)
    except RuntimeError as e:
        if not any(t in str(e) for t in _TRANSIENT_ERRORS):
            raise
        _log("[%s] transient ComfyUI I/O error (Windows 1450), resubmitting once" % label)
        return _submit_and_wait_once(graph, label + " retry")


def _submit_and_wait_once(graph: dict, label: str) -> str:
    resp = _api("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    pid = resp.get("prompt_id")
    if not pid:
        raise RuntimeError("no prompt_id for %s: %r" % (label, resp))
    _log("[%s] submitted %s" % (label, pid))
    # Off the critical path: watch for this prompt to actually start executing,
    # and only then hand the LLM's DRAM over. In a thread so waiting for that
    # confirmation cannot delay _poll (whose own vanish-detection timers start
    # from here).
    threading.Thread(target=_release_llm_when_running, args=(pid,),
                     daemon=True).start()
    res = _poll(pid)
    if "error" in res:
        raise RuntimeError("%s failed: %s" % (label, res["error"]))
    return os.path.join(COMFY_OUTPUT, res["subfolder"], res["file"])


def _run_h3_chain(plan: dict, req: dict) -> dict:
    """Render H3 as N chained segments and join them.

    Segment 0 is ref2va (reference images) for r2v, or fl2va on the caller's own
    start frame for i2v. Each later segment is fl2va seeded
    with the previous segment's LAST FRAME, which keeps the cast, lighting and
    framing continuous. Splitting is not just a memory trick: because H3's cost
    is ~quadratic in latent tokens, N short segments are far cheaper than one
    long shot, and short submissions also stay clear of the driver resets that
    long single shots triggered on this box.
    """
    n = plan["segments"]
    stamp = uuid.uuid4().hex[:8]
    paths = []
    first_frame = None
    for i in range(n):
        graph = build_h3_graph(plan, req, seg_idx=i, first_frame=first_frame,
                               prefix="video/h3_%s_seg%d" % (stamp, i + 1))
        out = _submit_and_wait(graph, "h3 seg%d/%d" % (i + 1, n))
        paths.append(out)
        if i + 1 < n:
            name = "h3chain_%s_%d.png" % (stamp, i + 1)
            _media("lastframe", out, os.path.join(COMFY_INPUT, name))
            first_frame = name
            _log("[h3] chained on last frame ->", name)

    if n == 1:
        return {"file": paths[0], "segments": paths}
    dst = os.path.join(COMFY_OUTPUT, "video", "h3_%s_joined.mp4" % stamp)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    _media("concat", dst, *paths)
    _log("[h3] joined %d segments ->" % n, dst)
    return {"file": dst, "segments": paths}


def probe_ref_video(req: dict) -> dict:
    """Read --ref-video's geometry into req["ref_video_meta"] (the planner prices it).

    Raises on anything that would only fail minutes later inside ComfyUI: a
    missing file, a start past the end, or too little left to follow.
    """
    path = os.path.expanduser(req.get("ref_video") or "")
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("reference video not found: %r" % req.get("ref_video"))
    meta = json.loads(_media("probe", path))
    if not meta.get("duration"):
        raise ValueError("cannot read the duration of %s" % path)
    left = memplan.h3_ref_video_seconds(dict(req, ref_video_meta=meta))
    if left < memplan.H3_REF_VIDEO_MIN_S:
        raise ValueError(
            "only %.2f s of %s is left after --ref-video-start %.2f (the video is %.2f s); "
            "H3 needs at least %.0f s of reference to follow"
            % (left, path, float(req.get("ref_video_start") or 0), meta["duration"],
               memplan.H3_REF_VIDEO_MIN_S))
    req["ref_video_meta"] = meta
    return meta


def _stage_ref_video(src: str, short: int, start: float, frames: int,
                     wav: bool, end: float | None = None) -> tuple[str, str | None]:
    """Normalise a reference video into ComfyUI/input; return (video, wav) names.

    24 fps, short edge `short`, exactly `frames` frames from `start` s, no audio
    track (the soundtrack is a separate wav, and only when asked for). With
    `end`, nothing past it is read and the last frame is held to fill `frames`
    -- how one shot is padded up to H3's 17k+5 grid. See h3_media.refvideo and
    memplan.h3_ref_video_grid, which prices this.
    """
    os.makedirs(COMFY_INPUT, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    vname = "hermes_refvid_%s.mp4" % tag
    args = ["refvideo", os.path.expanduser(src), os.path.join(COMFY_INPUT, vname),
            str(short), "%.4f" % start, str(frames),
            os.path.join(COMFY_INPUT, "hermes_refvid_%s.wav" % tag) if wav else "-"]
    if end is not None:
        args.append("%.4f" % end)
    info = json.loads(_media(*args).splitlines()[-1])
    _log("[refvideo] %s -> %s %dx%d x%df%s" % (os.path.basename(src), vname, info["w"],
                                                info["h"], info["frames"],
                                                " + wav" if info.get("wav") else ""))
    return vname, (os.path.basename(info["wav"]) if info.get("wav") else None)


def _swap_prompt(who: str, n_refs: int, extra: str) -> str:
    """The prompt for one --swap pass.

    Worded after the 2026-10-27 spike, where this phrasing replaced exactly one
    person and left the rest of the shot -- motion, camera, background -- as it
    was in <Video 1>. "Keep everyone else" matters on the later passes, whose
    reference video already carries the people swapped before.
    """
    pics = ("<Picture 1>" if n_refs == 1 else
            " and ".join("<Picture %d>" % (i + 1) for i in range(n_refs))
            + " (all the same person)")
    text = ("Replace %s in <Video 1> with the person from %s. Keep everyone else exactly "
            "as they are in <Video 1>. Keep exactly the same motion, positions, poses, "
            "timing and camera as <Video 1>." % (who.strip(), pics))
    return (text + " " + extra.strip()) if (extra or "").strip() else text


def _run_h3_refvideo(plan: dict, req: dict) -> dict:
    """Recast --ref-video shot by shot (memplan._plan_h3_refvideo decides the chunks).

    Every chunk is its own ref2va render on its own slice of the ORIGINAL video,
    padded to H3's grid and trimmed back when the clip is assembled -- so the
    result stays frame-locked to the reference and its soundtrack. Within a
    chunk H3 replaces ONE person per render (two characters' pictures: it uses
    one and ignores the other, measured 2026-10-27), so a chunk with several
    --swap passes runs them in turn, each following the previous pass's output.
    A chunk no --swap covers keeps its original frames and costs nothing.
    """
    stamp = uuid.uuid4().hex[:8]
    short = int(req.get("ref_video_size") or memplan.H3_REF_VIDEO_SHORT)
    audio = req.get("ref_video_audio") or "original"
    swaps = req.get("swaps") or []
    base_seed = int(req.get("seed", int(time.time()) % 2_000_000_000))
    src, start = req["ref_video"], float(req.get("ref_video_start") or 0)
    fps = plan["fps"]

    segs, chunk_log = [], []
    for ci, ch in enumerate(plan["chunks"]):
        n, rl = ch["frame_b"] - ch["frame_a"], ch["render_frames"]
        t0, t1 = start + ch["frame_a"] / fps, start + ch["frame_b"] / fps
        tag = "chunk %d/%d %.2f-%.2fs" % (ci + 1, len(plan["chunks"]), t0, t1)
        if not ch["swaps"]:
            vname, _ = _stage_ref_video(src, 4096, t0, n, False, end=t1)
            segs.append((os.path.join(COMFY_INPUT, vname), n))
            chunk_log.append({"start": t0, "end": t1, "who": None, "file": None})
            _log("[%s] no --swap here: original frames kept" % tag)
            continue
        cur, cur0, cur1, wav, out = src, t0, t1, None, None
        for pi, idx in enumerate(ch["swaps"]):
            vname, w = _stage_ref_video(cur, short, cur0, rl,
                                        audio == "reference" and pi == 0, end=cur1)
            wav = wav or w                    # the original soundtrack, every pass
            preq = dict(req, seed=base_seed + ci * 10 + pi, _ref_video_file=vname,
                        _ref_audio_file=wav, segment_prompts=None)
            if idx is not None:
                sw = swaps[idx]
                preq["refs"] = sw["refs"]
                preq["prompt"] = _swap_prompt(sw["who"], len(sw["refs"]), req.get("prompt", ""))
            graph = build_h3_graph(dict(plan, segment_length=rl), preq,
                                   prefix="video/h3_%s_c%dp%d" % (stamp, ci + 1, pi + 1))
            label = "%s %s" % (tag, swaps[idx]["who"] if idx is not None else "prompt")
            out = _submit_and_wait(graph, label)
            # a second pass on this shot follows what the first one made
            cur, cur0, cur1 = out, 0.0, n / float(fps)
            chunk_log.append({"start": t0, "end": t1,
                              "who": swaps[idx]["who"] if idx is not None else None,
                              "file": out})
        segs.append((out, n))

    dst = os.path.join(COMFY_OUTPUT, "video", "h3_%s_recast.mp4" % stamp)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    audio_src = (os.path.expanduser(src) if audio == "original"
                 and (req.get("ref_video_meta") or {}).get("has_audio") else "-")
    _media("assemble", dst, str(plan.get("out_width") or plan["width"]),
           str(plan.get("out_height") or plan["height"]), audio_src, "%.4f" % start,
           *["%s:%d" % (p, k) for p, k in segs])
    _log("[h3] recast assembled from %d chunk(s) -> %s" % (len(segs), dst))
    return {"file": dst, "segments": [p for p, _ in segs], "passes": chunk_log}


def _measure_anchor(req: dict, plan: dict, chain: dict) -> dict:
    """Did frame 0 actually come out as the image the caller pinned?

    Anchoring a first frame is a CONDITION, not a guarantee: H3 sometimes
    re-composes the opening anyway, and predicting that from the prompt proved
    unreliable. So measure the result instead of only guarding the input --
    ~5-8/255 is the VAE+h264 round trip, i.e. frame 0 really is the image;
    35+ means H3 opened on something of its own.
    """
    if not req.get("image"):
        # Say it out loud. Dropping --image on a follow-up turn is the one
        # mistake that does not raise: the render just quietly becomes a
        # different product (H3 picks its own opening, and the canvas goes
        # back to 16:9). A silently absent field is easy to not notice, so
        # the field is always present and says which of the two this was.
        return {"pinned": False,
                "note": "no start frame was pinned on this render -- H3 chose "
                        "its own opening. If this conversation pinned a first "
                        "frame earlier, you have to keep passing --image on "
                        "every re-render; it is never remembered for you."}
    if plan.get("h3_route") not in ("fl2va", "ref2va+kf"):
        return None
    try:
        out = _media("frame0diff", chain["file"], os.path.expanduser(req["image"]))
        d = float(out.split()[0])
    except Exception as e:
        return {"pinned": True, "error": "frame-0 check failed: %s" % str(e)[:120]}
    return {"pinned": True, "frame0_diff": round(d, 2), "anchored": d < 12,
            "note": ("frame 0 is the image you pinned" if d < 12 else
                     "frame 0 is NOT the image you pinned -- H3 composed its own "
                     "opening. The seed decides this, so RE-ROLL THE SEED first "
                     "(--seed N, or --anchor-retries N to do it automatically); "
                     "rewriting the prompt is the slow way round. Also check that "
                     "the prompt's opening state agrees with the pinned image -- "
                     "describing a state the image does not show ('sitting on a "
                     "chair' for a photo of someone standing) makes the miss much "
                     "more likely, and no amount of camera language rescues it.")}


def run(req: dict, dry: bool) -> dict:
    """Plan (dry) or render. A render first borrows the GPU from an LLM that can
    be put to sleep or unloaded (llmsleep), and always gives it back -- refused,
    failed or finished -- unless VIDGEN_LLM_RETURN=defer keeps it lent."""
    if dry:
        return _run(req, dry)
    lent = llmsleep.lend("video job")
    try:
        return _run(req, dry)
    finally:
        llmsleep.give_back(lent)


def _run(req: dict, dry: bool) -> dict:
    if req.get("ref_video") and not req.get("ref_video_meta"):
        try:
            probe_ref_video(req)
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if not dry:
        _free_before_planning()
    plan = memplan.choose(req)
    if not plan.get("ok"):
        # planner refused: unsafe for co-running programs. Never submit.
        return {"ok": False, "refused": True, "error": plan["refuse_reason"],
                "plan": plan["plan"], "budget": plan.get("budget")}
    if dry:
        return {"ok": True, "dry_run": True, "tier": plan["tier"], "plan": plan["plan"],
                "budget": plan["budget"], "offload": plan["offload"],
                "shrunk": plan.get("shrunk"),
                # H3 only: which graph this would take (fl2va = --image is frame 0,
                # ref2va = --ref images are the cast), and the aspect-ratio warning,
                # as fields rather than only as prose inside `plan`.
                "h3_route": plan.get("h3_route"), "warning": plan.get("warning"),
                # --ref-video: the shots it will be cut into, and who is swapped
                # in each -- what --swap ...@START-END ranges are chosen from.
                "chunks": plan.get("chunks"),
                "cuts": (req.get("ref_video_meta") or {}).get("cuts")}

    try:
        _require_comfy()
    except Exception as e:
        return {"ok": False, "error": str(e), "plan": plan["plan"]}
    # Preflight: re-measure right before submitting -- headroom may have changed
    # (LLM could have grown). If it's no longer safe, abort instead of pushing in.
    recheck = memplan.choose(req)
    if not recheck.get("ok"):
        return {"ok": False, "refused": True, "error": "aborted at preflight: " + recheck["refuse_reason"],
                "plan": recheck["plan"], "budget": recheck.get("budget")}
    plan = recheck
    _log("[plan]", plan["plan"])

    if plan["tier"] == "h3":
        t0 = time.time()
        # Anchoring is a coin flip on some requests: on a fixed-seed sweep the
        # same prompt anchored 2 times out of 5 when its opening state
        # contradicted the pinned photo, and 4 out of 4 when it agreed. The
        # seed is what differs, so the cure for a miss is a re-roll -- do it
        # here rather than spend a turn and a guess on the agent's side.
        tries = max(0, int(req.get("anchor_retries") or 0)) + 1
        attempts = []
        try:
            if req.get("ref_video"):
                # frame 0 comes from <Video 1>, so there is no pin to check
                chain = _run_h3_refvideo(plan, req)
                anchor = {"pinned": False,
                          "note": "reference-video render: framing and motion follow "
                                  "<Video 1>, not a pinned start frame"}
                tries = 0
            for i in range(tries):
                areq = req if i == 0 else dict(
                    req, seed=int(time.time() * 1000) % 2_000_000_000 + i * 7919)
                chain = _run_h3_chain(plan, areq)
                anchor = _measure_anchor(areq, plan, chain)
                if tries > 1:
                    attempts.append({"seed": areq.get("seed"),
                                     "file": os.path.basename(chain["file"]),
                                     "frame0_diff": (anchor or {}).get("frame0_diff")})
                if (anchor or {}).get("anchored") is not False:
                    break
        except Exception as e:
            return {"ok": False, "error": str(e), "plan": plan["plan"]}
        finally:
            try:
                _api("/free", {"unload_models": True, "free_memory": True})
            except Exception:
                pass
            _prewarm_llm()
        return {"ok": True, "file": chain["file"],
                "filename": os.path.basename(chain["file"]),
                "segment_files": chain["segments"], "segments": plan["segments"],
                "tier": "h3", "audio": True, "seconds": round(time.time() - t0, 1),
                "duration_s": round(plan["length"] / plan["fps"], 2),
                "h3_route": plan.get("h3_route"), "warning": plan.get("warning"),
                "start_frame_check": anchor,
                "anchor_attempts": attempts or None,
                "swap_passes": chain.get("passes"),
                "plan": plan["plan"], "budget": plan["budget"],
                "offload": plan["offload"], "note": resume_note()}

    try:
        graph = build_graph(plan, req)          # stages the start image (i2v)
    except Exception as e:
        return {"ok": False, "error": str(e), "plan": plan["plan"]}
    client_id = uuid.uuid4().hex
    t0 = time.time()
    try:
        resp = _api("/prompt", {"prompt": graph, "client_id": client_id})
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": "submit rejected: %s" % e.read().decode()[:800],
                "plan": plan["plan"]}
    pid = resp.get("prompt_id")
    if not pid:
        return {"ok": False, "error": "no prompt_id in response: %r" % resp}
    _log("[submitted]", pid, "-- polling (weights stream from SSD/DRAM; first run is slow)")
    # Same ordering rule as the H3 path: trim the LLM only once this prompt is
    # confirmed executing (see _release_llm_when_running).
    threading.Thread(target=_release_llm_when_running, args=(pid,),
                     daemon=True).start()
    res = _poll(pid)
    dt = round(time.time() - t0, 1)
    if plan["cache_none"]:
        # Hand VRAM/RAM back to the LLM: unload models instead of caching.
        try:
            _api("/free", {"unload_models": True, "free_memory": True})
            _log("[freed] models unloaded (RAM returned to LLM)")
        except Exception as e:
            _log("[warn] /free failed:", e)
        _prewarm_llm()
    if "error" in res:
        return {"ok": False, "error": res["error"], "prompt_id": pid,
                "plan": plan["plan"], "seconds": dt}
    out_path = os.path.join(COMFY_OUTPUT, res["subfolder"], res["file"])
    return {"ok": True, "file": out_path, "filename": res["file"],
            "tier": plan["tier"], "seconds": dt, "plan": plan["plan"],
            "budget": plan["budget"], "offload": plan["offload"],
            "note": resume_note()}


# ---- async jobs (so Hermes never blocks on a multi-minute render) ---------
JOBS_DIR = paths.JOBS_DIR


def _job_path(job_id):
    return os.path.join(JOBS_DIR, job_id + ".json")


LATEST_FILE = "LATEST"        # holds the most recently submitted job_id


def _note_latest(job_id: str) -> None:
    """Remember the newest job outside the tool's stdout.

    A foreground call killed at the caller's timeout loses everything gen.py
    printed, job_id included, so "reconnect instead of resubmitting" is only
    actionable if the id is also on disk.
    """
    try:
        os.makedirs(JOBS_DIR, exist_ok=True)
        with open(os.path.join(JOBS_DIR, LATEST_FILE), "w", encoding="utf-8") as f:
            f.write(job_id + "\n")
    except Exception:
        pass                      # a missing pointer must never fail a render


def latest_job_id() -> str | None:
    try:
        with open(os.path.join(JOBS_DIR, LATEST_FILE), encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def running_jobs() -> list:
    """Job ids whose state resolves to 'running' right now, newest first."""
    out = []
    try:
        names = sorted(os.listdir(JOBS_DIR), reverse=True)
    except OSError:
        return out
    for name in names:
        if not name.endswith(".json"):
            continue
        jid = name[:-5]
        try:
            with open(os.path.join(JOBS_DIR, name), encoding="utf-8") as f:
                if json.load(f).get("status") != "running":
                    continue
        except Exception:
            continue
        if job_status(jid).get("status") == "running":
            out.append(jid)
    return out


def stop_render(job_id: str | None = None) -> dict:
    """Interrupt what ComfyUI is doing, clear the queue, hand the VRAM back.

    Deliberately does NOT kill the detached worker: once the prompt is
    interrupted the worker's own poll sees it fail and records the error, which
    leaves a truthful job file. Killing it instead would strand the job at
    "running" forever -- the state that sends agents off resubmitting.
    """
    _resolve_comfy_url()
    report = {"ok": True, "interrupted": [], "jobs": [], "queue_cleared": False,
              "vram_freed": False}
    try:
        q = _api("/queue")
        report["interrupted"] = [e[1] for e in q.get("queue_running", []) if len(e) > 1]
        report["pending_cleared"] = len(q.get("queue_pending", []))
    except Exception as e:
        return {"ok": False, "error": "ComfyUI is not reachable at %s (%s). Nothing to "
                                      "stop there; if a job file still says running, its "
                                      "worker will notice on its next poll."
                                      % (COMFY_URL, e)}
    try:
        _api("/interrupt", {})
        _api("/queue", {"clear": True})
        report["queue_cleared"] = True
    except Exception as e:
        report["ok"] = False
        report["error"] = "interrupt/clear failed: %s" % e
    time.sleep(3)
    try:
        _api("/free", {"unload_models": True, "free_memory": True})
        report["vram_freed"] = True
    except Exception:
        pass
    # Give the worker a moment to notice, then record whatever did not resolve.
    time.sleep(2)
    targets = [job_id] if job_id else running_jobs()
    for jid in targets:
        st = job_status(jid)
        if st.get("status") == "running":
            try:
                with open(_job_path(jid), encoding="utf-8") as f:
                    d = json.load(f)
                d.update(status="error", error="stopped by `gen.py stop`",
                         finished=time.strftime("%Y-%m-%d %H:%M:%S"))
                _write_job(d)
                st = {"job_id": jid, "status": "stopped"}
            except Exception as e:
                st = {"job_id": jid, "status": "running", "error": str(e)}
        report["jobs"].append({k: st.get(k) for k in ("job_id", "status", "file")})
    _prewarm_llm()                # the render had already been handed the LLM's DRAM
    report["note"] = resume_note()
    return report


def _write_job(data):
    os.makedirs(JOBS_DIR, exist_ok=True)
    p = _job_path(data["job_id"])
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)  # atomic: status readers never see a half-written file


def _pid_alive(pid):
    try:
        return osmem.pid_alive(int(pid))
    except Exception:
        return True  # unsure -> assume alive (don't falsely report failure)


# Fields that make two requests "the same render". Deliberately includes seed:
# an explicit different seed is a deliberate second take, not a resubmit.
_DUP_KEYS = ("mode", "tier", "prompt", "negative", "image", "last_frame", "refs",
             "seconds", "short_side", "portrait", "steps", "fps",
             "ref_image_size", "segment_prompts", "seed",
             "ref_video", "ref_video_size", "ref_video_start", "ref_video_audio", "swaps",
             "easycache")


def _active_duplicate(req: dict) -> dict | None:
    """An identical render already in flight, or None.

    The job files are the only state both attempts can see -- the second one is
    usually a brand-new process that knows nothing about the first -- so the
    check belongs here rather than in any single front-end.
    """
    sig = {k: req.get(k) for k in _DUP_KEYS}
    try:
        names = sorted(os.listdir(JOBS_DIR))
    except OSError:
        return None
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(JOBS_DIR, name), encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue                      # a half-written job file is not a match
        if d.get("status") != "running":
            continue
        if {k: (d.get("req") or {}).get(k) for k in _DUP_KEYS} != sig:
            continue
        # "running" in the file only means nobody wrote the ending. Ask
        # job_status, which reconciles a dead worker against ComfyUI's history;
        # without that, one crashed job would block this request forever.
        live = job_status(d["job_id"])
        if live.get("status") == "running":
            return live
    return None


def submit_async(req: dict, force: bool = False) -> dict:
    """Plan now; if safe, spawn a detached worker and return a job_id at once."""
    if not force:
        dup = _active_duplicate(req)
        if dup:
            return {"ok": False, "duplicate": True, "job_id": dup["job_id"],
                    "status": "running", "plan": dup.get("plan"),
                    "error": "an identical render is ALREADY RUNNING as job %s "
                             "(started %s). A terminal timeout kills only the "
                             "process that was waiting -- never the render, which "
                             "runs in a detached worker against a separate ComfyUI "
                             "server. Reconnect with `status %s` instead of "
                             "resubmitting; two H3 renders on one GPU make both "
                             "slower. Use --force only if you genuinely want a "
                             "second copy."
                             % (dup["job_id"], dup.get("started"), dup["job_id"]),
                    "notice": busy_notice()}
    _free_before_planning()
    plan = memplan.choose(req)
    if not plan.get("ok"):
        return {"ok": False, "refused": True, "error": plan["refuse_reason"], "plan": plan["plan"]}
    job_id = uuid.uuid4().hex[:12]
    data = {"job_id": job_id, "status": "running", "req": req, "plan": plan["plan"],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": None,
            "file": None, "seconds": None, "error": None}
    _write_job(data)
    _note_latest(job_id)
    log = os.path.join(JOBS_DIR, job_id + ".log")
    # Detached on purpose: the caller's terminal kills gen.py at its own timeout
    # (600 s here) while a --quality render needs ~630 s. The worker must outlive
    # that. osmem picks DETACHED_PROCESS on Windows, setsid() on POSIX.
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_worker", job_id],
                            stdout=open(log, "w", encoding="utf-8", errors="replace"),
                            stderr=subprocess.STDOUT,
                            cwd=HERE, **osmem.detach_kwargs())
    data["pid"] = proc.pid
    _write_job(data)
    out = {"ok": True, "job_id": job_id, "status": "running", "plan": plan["plan"],
           "message": "render started in background; poll with:  gen.py status %s" % job_id,
           "notice": busy_notice(),
           "notice_instruction": "Relay `notice` to the user verbatim as the last "
                                 "line of your reply, then stop. Do not issue "
                                 "further commands until the render finishes."}
    # The planner's warning, which the --wait path already returned and this one
    # dropped: H3's unpinned-camera note, and on a GPU other than NVIDIA that every
    # time estimate is off -- the one thing the user should hear before waiting.
    if plan.get("warning"):
        out["warning"] = plan["warning"]
        out["notice_instruction"] = ("Tell the user `warning` in your own words first, "
                                     "then relay `notice` verbatim as the last line of "
                                     "your reply, then stop. Do not issue further "
                                     "commands until the render finishes.")
    return out


def _worker(job_id: str):
    """Detached: run the render and record the outcome in the job file."""
    with open(_job_path(job_id), encoding="utf-8") as f:
        data = json.load(f)
    try:
        res = run(data["req"], dry=False)
        if res.get("ok"):
            # start_frame_check has to be persisted, not just returned: every
            # caller reaches a render through the job file (both --wait and the
            # async path poll job_status), so anything left only in run()'s
            # return value is computed and then thrown away.
            data.update(status="done", file=res.get("file"),
                        seconds=res.get("seconds"), tier=res.get("tier"),
                        start_frame_check=res.get("start_frame_check"),
                        anchor_attempts=res.get("anchor_attempts"),
                        swap_passes=res.get("swap_passes"))
        else:
            data.update(status="error", error=res.get("error"))
    except BaseException as e:   # incl. SystemExit: must never leave job stuck "running"
        data.update(status="error", error=repr(e))
    data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_job(data)


def _log_tail_error(job_id: str) -> str:
    """Best-effort: pull the real failure from the worker log (last JSON error line)."""
    try:
        with open(os.path.join(JOBS_DIR, job_id + ".log"), encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return ""
    for ln in reversed(lines):
        if ln.startswith("{"):
            try:
                err = json.loads(ln).get("error")
                if err:
                    return err
            except Exception:
                pass
    return lines[-1] if lines else ""


# Two log shapes reach here: "[submitted] <uuid>" from the Wan path and
# "[h3 seg1/3] submitted <uuid>" from the chained H3 path.
_SUBMITTED_RE = re.compile(r"submitted\]?\s+([0-9a-fA-F-]{36})")


def _job_prompt_id(job_id: str) -> str:
    """The last ComfyUI prompt_id this job submitted, read back from its log.

    The worker only ever logs it ("[submitted] <uuid> -- polling"), never into
    the job record, so this is the one way a caller that lost the worker can
    find the render again.
    """
    try:
        with open(os.path.join(JOBS_DIR, job_id + ".log"), encoding="utf-8", errors="replace") as f:
            ids = _SUBMITTED_RE.findall(f.read())
    except Exception:
        return ""
    return ids[-1] if ids else ""


def _history_output(prompt_id: str):
    """The finished video entry for a prompt, or None if it is not done yet."""
    try:
        h = _api(f"/history/{prompt_id}")
    except Exception:
        return None
    rec = h.get(prompt_id)
    if not rec:
        return None
    for _nid, out in rec.get("outputs", {}).items():
        for key in ("videos", "gifs", "images"):
            for v in out.get(key) or []:
                if v.get("type", "output") != "output":   # a loader's preview, see _poll
                    continue
                return {"file": v.get("filename"), "subfolder": v.get("subfolder", "")}
    return None


def job_status(job_id: str) -> dict:
    # Orphan recovery below asks ComfyUI whether the prompt is still queued, so
    # the endpoint has to be resolved even though nothing was submitted from
    # this process -- `gen.py status <id>` is usually a fresh interpreter.
    _resolve_comfy_url()
    p = _job_path(job_id)
    if not os.path.exists(p):
        # Hermes hands back its own `proc_*` session id when a command is
        # started with background=true, and that is NOT a vidgen job id. One
        # session spent two turns passing it here and reading "no such job" as
        # "the render never started". Say which id is wanted instead.
        if str(job_id).startswith("proc_"):
            hint = ("that is the terminal tool's background session id, not a vidgen "
                    "job id. The job id is the 12-hex-character one gen.py printed as "
                    "`job_id=...`, also saved in %s. Run `status` with no argument to "
                    "use it." % os.path.join(JOBS_DIR, LATEST_FILE))
        else:
            latest = latest_job_id()
            hint = ("the most recent job is %s -- run `status` with no argument to use "
                    "it automatically." % latest) if latest else \
                   "no jobs have been submitted yet."
        return {"ok": False, "error": "no such job: %s. %s" % (job_id, hint)}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    st = data.get("status")
    if st == "running" and data.get("pid") and not _pid_alive(data["pid"]):
        # A dead worker is NOT the same as a failed render. ComfyUI is a separate
        # server and the prompt outlives whoever submitted it, so a terminal that
        # hit its timeout and killed the waiting gen.py leaves a job that is
        # still rendering happily. Calling that "error" throws away a finished
        # mp4 and pushes the caller into re-submitting the whole thing -- which
        # is exactly what happened once, 4 seconds before the original landed.
        # So ask ComfyUI before believing the worst.
        cpid = _job_prompt_id(job_id)
        finished = _history_output(cpid) if cpid else None
        if finished:
            st = data["status"] = "done"
            data["file"] = os.path.join(COMFY_OUTPUT, finished["subfolder"], finished["file"])
            data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data["recovered"] = ("worker died, but the render had finished -- output "
                                 "reclaimed from ComfyUI history")
            _write_job(data)
        elif cpid and _in_queue(cpid) is True:
            data["orphaned"] = ("the waiting process is gone but ComfyUI is STILL "
                                "rendering prompt %s -- keep polling this job_id, "
                                "do not re-submit" % cpid)
            _write_job(data)
        else:
            st = "error"
            data["status"] = "error"
            data["error"] = (_log_tail_error(job_id)
                             or "worker exited without finishing (see jobs/%s.log)" % job_id)
            _write_job(data)   # persist so later polls don't keep re-deriving a dead job
    out = {"ok": st in ("running", "done"), "job_id": job_id, "status": st,
           "plan": data.get("plan"), "started": data.get("started")}
    if st == "done":
        out.update(file=data.get("file"), seconds=data.get("seconds"), tier=data.get("tier"))
        if data.get("start_frame_check") is not None:
            out["start_frame_check"] = data["start_frame_check"]
        if data.get("anchor_attempts"):
            out["anchor_attempts"] = data["anchor_attempts"]
        if data.get("swap_passes"):
            out["swap_passes"] = data["swap_passes"]
        if data.get("recovered"):
            out["note"] = data["recovered"]
    elif st == "error":
        out["ok"] = False
        out["error"] = data.get("error")
    elif st == "running":
        out["message"] = (data.get("orphaned")
                          or "still rendering (SSD-paged, a few minutes); poll again shortly")
    return out


def _refuse_direct(cmd: str):
    """Block rendering straight from this file.

    This is the engine, not the entry point. An agent that cannot get what it
    wants from gen.py will read this source and call it directly -- one run did
    exactly that, skipping the skill entirely and then lowering the LLM VRAM
    guard to push a job through. gen.py imports us in-process, and the detached
    worker re-enters via _worker, so neither path is affected.
    """
    raise SystemExit(json.dumps({
        "ok": False,
        "error": ("video_gen.py is an internal engine and will not render when called "
                  "directly. Use the vidgen skill's entry point instead:\n"
                  '  "%s" "%s" r2v "<prompt>" --ref <image> --wait\n'
                  "It performs the same render with the memory guards that protect the "
                  "co-resident LLM. (Maintainers: VIDGEN_ALLOW_DIRECT=1 re-enables '%s'.)"
                  % (sys.executable or "python",
                     os.path.join(paths.SKILLS_DIR, "vidgen", "gen.py"),
                     cmd))},
        ensure_ascii=False))


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "plan"
    direct_ok = os.environ.get("VIDGEN_ALLOW_DIRECT") == "1"
    if cmd == "_worker":
        _worker(argv[1]); return          # detached worker we spawned ourselves
    if cmd == "status":
        print(json.dumps(job_status(argv[1]), ensure_ascii=False, indent=2)); return
    if cmd in ("run", "async") and not direct_ok:
        _refuse_direct(cmd)
    if cmd == "async":
        req = json.loads(argv[1]) if len(argv) > 1 else {}
        print(json.dumps(submit_async(req), ensure_ascii=False, indent=2)); return
    req = json.loads(argv[1]) if len(argv) > 1 else {}
    if cmd not in ("plan", "run"):
        raise SystemExit(json.dumps({"ok": False,
            "error": "usage: video_gen.py plan|run|async '<json>' | status <job_id>"}))
    print(json.dumps(run(req, dry=(cmd == "plan")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
