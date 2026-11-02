#!/usr/bin/env python
"""imggen skill entry point — the ONLY supported way to generate images here.

Ensures the memory-protected ComfyUI is running, injects your prompt into a
pre-built, verified workflow, submits it, and returns the output path.
Do NOT hand-write ComfyUI API calls; small models produce worse results and
skip the safety/memory protections. Just call this script.

Usage:
  gen.py zimage  "a cyberpunk cat"                       # accurate, fast (default)
  gen.py realvis "rainy Taipei street at night, 35mm"    # photorealistic
  gen.py img2img "make it 3D glossy toy" --image C:/x.png --denoise 0.85
  gen.py status <prompt_id>                              # pick up a render that outran the wait

A render that is still going when the foreground wait runs out is not an error:
ComfyUI keeps rendering on its own, so this prints {"status": "running",
"prompt_id": ...} and `gen.py status <prompt_id>` collects the image later.
"""
import argparse, json, os, re, sys, time, uuid, random, io, mimetypes, subprocess, struct
import urllib.request, urllib.parse
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
    raise SystemExit(json.dumps({"status": "error", "error":
        "comfy engine not found. Set HERMES_COMFY_ENGINE to the directory "
        "containing paths.py, or write that path into "
        + str(marker)}, ensure_ascii=False))


ENGINE_DIR = _find_engine()
sys.path.insert(0, ENGINE_DIR)
import paths  # noqa: E402  -- also folds hermes-comfy.conf into os.environ

COMFY_URL   = paths.COMFY_URL
LAUNCHER    = os.path.join(ENGINE_DIR, "start_comfy.py")
WF_DIR      = paths.WF_DIR
DEFAULT_OUT = paths.OUT_DIR

# How long this process waits in the foreground, counted from its own start, before
# handing back a prompt_id instead of an image. Under the agent's terminal timeout
# (Hermes: terminal.timeout, 600 s), so a slow render is reported as running rather
# than killed mid-wait and reported as nothing at all.
WAIT_SECONDS = float(os.environ.get("IMGGEN_WAIT_SECONDS", "240"))
# How long a running render may go without a sign of progress before it is called
# stuck. A sign is the server's log growing -- its sampler writes a line per step.
# Generous because one step on a slow card has been measured at two minutes.
STALL_SECONDS = float(os.environ.get("IMGGEN_STALL_SECONDS", "600"))
STARTED = time.time()

# model keyword -> workflow file + where to inject each param (node id, field)
MODELS = {
    "qwen":    {"wf": "qwen_image_txt2img.json",
                "prompt": ("13", "text"), "seed": ("3", "seed")},
    # "refs": (encode node, Autogrow input name). "ref_image": optional
    # LoadImage node that already exists in the workflow and becomes image_1.
    "qwen21":  {"wf": "qwen_image_2_1_txt2img.json",
                "prompt": ("13", "prompt"), "seed": ("3", "seed"),
                "refs": ("13", "images"), "turbo": True},
    # Mirrors the official ComfyUI template (image_qwen_image_2_1_image_edit):
    # no prompt enhancer, cfg 1, canvas taken from image_1.
    "qwen21_edit": {"wf": "qwen_image_2_1_image_edit.json",
                    "prompt": ("13", "prompt"), "seed": ("3", "seed"),
                    "refs": ("13", "images"),
                    "ref_image": "20", "needs_ref": True, "turbo": True},
    # Reposition the subject: shift the pixels, outpaint the strip that opens
    # up, paste the untouched pixels back. The edit model alone will not move
    # a subject (it keeps <image1>'s layout), measured three times.
    "qwen21_move": {"wf": "qwen_image_2_1_image_edit.json",
                    "prompt": ("13", "prompt"), "seed": ("3", "seed"),
                    "refs": ("13", "images"),
                    "ref_image": "20", "needs_ref": True, "turbo": True, "move": True},
    "flux":    {"wf": "flux_schnell_txt2img.json",
                "prompt": ("13", "text"), "seed": ("3", "seed")},
    "zimage":  {"wf": "z_image_turbo_txt2img.json",
                "prompt": ("27", "text"), "seed": ("3", "seed")},
    "realvis": {"wf": "realvisxl_txt2img.json",
                "prompt": ("6", "text"),  "seed": ("3", "seed")},
    "img2img": {"wf": "z_image_turbo_img2img.json",
                "prompt": ("27", "text"), "seed": ("3", "seed"),
                "denoise": ("3", "denoise"), "image": ("10", "image"),
                "needs_image": True},
}


def ensure_comfy():
    """Start the memory-protected ComfyUI if it isn't already up (idempotent).

    Under WSL the candidate list also holds the Windows host's address: a
    ComfyUI on the host is not on this distro's loopback. And when ComfyUI
    belongs to the host we must not try to launch it -- say what to run there.
    """
    global COMFY_URL
    for u in paths.comfy_url_candidates():
        try:
            urllib.request.urlopen(f"{u}/system_stats", timeout=5)
            COMFY_URL = u
            # On another card than the one picked in AI Scaler Toolkit's settings?
            # Stopped if idle, then launched again below on the right one.
            import comfyguard
            stopped, note = comfyguard.realign(u, paths.SELECTION.get("device"))
            if note:
                print(note, file=sys.stderr, flush=True)
            if stopped:
                break
            # Connecting to a server someone else started means inheriting
            # whatever protection it was launched with -- possibly none. Images
            # load multi-GB models too, so this matters here as much as it does
            # for video. Warn on stderr so stdout stays pure JSON.
            try:
                import osmem, comfyguard
                msg = comfyguard.check(
                    u, bool(osmem.find_pid(paths.LLM_PROCESS)), ENGINE_DIR)
                if msg:
                    print(msg, file=sys.stderr, flush=True)
            except Exception:
                pass          # a advisory check must never block generation
            return
        except Exception:
            continue
    if paths.COMFY_REMOTE:
        raise RuntimeError(
            "ComfyUI is not reachable at any of %s, and it is not ours to "
            "start (COMFY_REMOTE): it lives on the host, not in this "
            "environment. Start it there with:\n"
            "    python <engine>/start_comfy.py\n"
            "It listens on 127.0.0.1 only. From a NAT-mode WSL distro that is out "
            "of reach: turn on WSL mirrored networking "
            "(%%USERPROFILE%%\\.wslconfig: [wsl2] networkingMode=mirrored, then "
            "`wsl --shutdown`) and loopback reaches it."
            % (paths.comfy_url_candidates(),))
    # launcher chatter goes to stderr so stdout stays pure JSON
    subprocess.run([sys.executable, LAUNCHER], check=True, stdout=sys.stderr)


def upload_image(path):
    path = os.path.expanduser(path)          # small models often pass ~/...
    if not os.path.exists(path):
        raise FileNotFoundError(f"input image not found: {path}")
    fn = os.path.basename(path)
    b = uuid.uuid4().hex
    buf = io.BytesIO()
    def part(name, val):
        buf.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode())
    part("subfolder", "imggen_in"); part("overwrite", "true")
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    buf.write(f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fn}\"\r\n".encode())
    buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
    with open(path, "rb") as f:
        buf.write(f.read())
    buf.write(f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(f"{COMFY_URL}/upload/image", data=buf.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req) as r:
        info = json.load(r)
    sub = info.get("subfolder", "")
    return f"{sub}/{info['name']}" if sub else info["name"]


# Models whose text encoder cannot read CJK. Not a quality difference -- the
# prompt simply does not reach the image, and the failure is SILENT: you get a
# clean, plausible picture of something else entirely. Measured: the same
# Chinese prompt gave RealVisXL a woman and a dog on a bed, and FLUX a night
# townscape with invented calligraphy, where the request was a kitten on a
# windowsill. Same prompt in English -> correct both times.
_CJK_BLIND = {
    "realvis": "RealVisXL (SDXL) 用 CLIP 文字編碼器",
    "flux":    "FLUX.1-schnell 用 T5-XXL + CLIP-L",
}


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" or "぀" <= c <= "ヿ"
               or "가" <= c <= "힯" for c in s or "")


def check_prompt_language(model: str, prompt: str) -> None:
    """Refuse a prompt the chosen model cannot read, the way t2v is refused.

    A warning would not do: the render succeeds, the JSON says ok, and the only
    way to notice is to open the file and recognise that it shows the wrong
    subject. Blocking is what makes the mistake visible.
    """
    if model in _CJK_BLIND and _has_cjk(prompt):
        raise ValueError(
            "%s,讀不懂中文/日文/韓文 —— 這個 prompt 送進去會產出「畫面乾淨但內容完全不對」"
            "的圖,而且不會有任何錯誤訊息(實測過)。兩個正確做法,擇一:\n"
            "  1) 把 prompt 翻成英文再用 %s(要保留這個模型的風格時選這個)\n"
            "  2) 改用吃得下中文的模型:qwen(遵循度最高)或 zimage(快)\n"
            "     例:gen.py qwen \"%s\"" % (_CJK_BLIND[model], model, prompt[:40]))


def wire_refs(wf, spec, refs):
    """Link each reference image into the encoder's Autogrow input.

    In the API graph an Autogrow input is a set of FLAT keys ("images.image_1",
    ...), each a real link [node_id, slot] to a LoadImage node. A nested dict,
    or a filename in the node-id slot, is dropped without any error: the
    encoder then sees no reference at all and the job quietly becomes
    text-to-image (measured: an outfit edit drew a different person).
    """
    n, f = spec["refs"]
    inputs = wf[n]["inputs"]
    for k in [k for k in inputs if k == f or k.startswith(f + ".")]:
        del inputs[k]
    for i, p in enumerate(refs, 1):
        if i == 1 and spec.get("ref_image"):
            node = spec["ref_image"]
        else:
            node = f"ref_{i}"
            wf[node] = {"class_type": "LoadImage", "inputs": {}}
        wf[node]["inputs"]["image"] = upload_image(p)
        inputs[f"{f}.image_{i}"] = [node, 0]


TURBO_LORA = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
TURBO_SIGMAS = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"   # v0.2.1 raw student nodes, 6 steps


def turbo_missing():
    """Why the Viggle Turbo path cannot run on this ComfyUI, or None if it can.

    Neither piece ships with the pack: the node file and the LoRA come from
    Viggle/Qwen-Image-2.1-viggle-turbo on Hugging Face, under the Qwen
    research (non-commercial) licence. See MODELS.md.
    """
    try:
        info = _api_json("/object_info/ViggleTurboLora").get("ViggleTurboLora")
    except Exception as e:
        return f"cannot ask ComfyUI for the ViggleTurboLora node ({type(e).__name__})"
    if not info:
        return "custom node viggle_turbo.py is not installed in ComfyUI"
    loras = (info.get("input", {}).get("required", {}).get("lora_name") or [[]])[0]
    if TURBO_LORA not in loras:
        return f"LoRA {TURBO_LORA} is not in models/loras"
    return None


def apply_turbo(wf):
    """Swap the KSampler (node "3") for the Viggle Turbo path, in place.

    Follows the model card (Viggle/Qwen-Image-2.1-viggle-turbo v0.2.1): LoRA at
    strength 1.0 applied UNMERGED (merging drops ~30% of it on bf16 and adds
    noise on int8), its own resolution-shifted 6-step sigmas, euler, cfg 1.
    Node id "3" is kept so VAEDecode's link to output 0 still holds.
    """
    ks = wf["3"]["inputs"]
    wf["turbo_lora"] = {"class_type": "ViggleTurboLora",
                        "inputs": {"model": ks["model"], "lora_name": TURBO_LORA, "strength": 1.0}}
    wf["turbo_sigmas"] = {"class_type": "ViggleTurboSigmas",
                          "inputs": {"latent": ks["latent_image"], "nodes": TURBO_SIGMAS}}
    wf["turbo_sampler"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["3"] = {"class_type": "SamplerCustom",
               "inputs": {"model": ["turbo_lora", 0], "add_noise": True,
                          "noise_seed": ks["seed"], "cfg": 1.0,
                          "positive": ks["positive"], "negative": ks["negative"],
                          "sampler": ["turbo_sampler", 0], "sigmas": ["turbo_sigmas", 0],
                          "latent_image": ks["latent_image"]}}


def image_size(path):
    """(width, height) from the file header: PNG, JPEG or WebP.

    The Hermes venv that runs this script has no PIL, so read the header.
    """
    with open(os.path.expanduser(path), "rb") as f:
        head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            kind = head[12:16]
            if kind == b"VP8X":
                return (1 + int.from_bytes(head[24:27], "little"),
                        1 + int.from_bytes(head[27:30], "little"))
            if kind == b"VP8 ":
                f.seek(26)
                w, h = struct.unpack("<HH", f.read(4))
                return (w & 0x3FFF, h & 0x3FFF)
            if kind == b"VP8L":
                bits = int.from_bytes(head[21:25], "little")
                return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
        if head[:2] == b"\xff\xd8":
            f.seek(2)
            sof = {0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3, 0xFFC5, 0xFFC6, 0xFFC7,
                   0xFFC9, 0xFFCA, 0xFFCB, 0xFFCD, 0xFFCE, 0xFFCF}
            while True:
                marker, seglen = struct.unpack(">HH", f.read(4))
                if marker in sof:
                    h, w = struct.unpack(">xHH", f.read(5))
                    return (w, h)
                f.seek(seglen - 2, 1)
    raise ValueError(f"cannot read the image size (PNG/JPEG/WebP only): {path}")


MOVE_PROMPT = ("Fill the plain gray area on the {side} of <image1> by seamlessly extending the "
               "existing background, with the same scenery, lighting and art style. Do not add "
               "any person, hand or object there. Keep everything else in <image1> unchanged.")


def apply_move(wf, path, move_x, move_y):
    """Shift <image1> by (move_x, move_y) of its size and outpaint the gap, in place.

    Positive x moves the subject right, positive y down. The side it moves
    toward is cropped off, the side it leaves is padded gray; the sampler only
    has to invent that strip. Outside a feathered band the result is the
    original pixels, pasted back, so the subject cannot drift.
    Returns (side filled, report). The report carries the shift actually
    applied (rounded to 16 px) so the caller can state it instead of guessing:
    an agent once described a 15% shift as "a third of the frame".
    """
    if not (move_x or move_y):
        raise ValueError("qwen21_move needs --move-x and/or --move-y (fraction of the image, e.g. -0.15)")
    if max(abs(move_x), abs(move_y)) > 0.4:
        raise ValueError("--move-x / --move-y above 0.4 crops away too much of the image")
    w, h = image_size(path)
    dx = int(round(w * move_x / 16)) * 16
    dy = int(round(h * move_y / 16)) * 16
    feather = max(32, min(w, h) // 12)
    wf["move_crop"] = {"class_type": "ImageCrop",
                       "inputs": {"image": ["20", 0], "width": w - abs(dx), "height": h - abs(dy),
                                  "x": -dx if dx < 0 else 0, "y": -dy if dy < 0 else 0}}
    wf["move_pad"] = {"class_type": "ImagePadForOutpaint",
                      "inputs": {"image": ["move_crop", 0],
                                 "left": max(dx, 0), "right": max(-dx, 0),
                                 "top": max(dy, 0), "bottom": max(-dy, 0),
                                 "feathering": feather}}
    wf["13"]["inputs"]["images.image_1"] = ["move_pad", 0]
    # the sampler works on a /32-rounded canvas; bring it back to the source size
    wf["move_scale"] = {"class_type": "ImageScale",
                        "inputs": {"image": ["8", 0], "upscale_method": "lanczos",
                                   "width": w, "height": h, "crop": "disabled"}}
    wf["move_keep"] = {"class_type": "InvertMask", "inputs": {"mask": ["move_pad", 1]}}
    wf["move_paste"] = {"class_type": "ImageCompositeMasked",
                        "inputs": {"destination": ["move_scale", 0], "source": ["move_pad", 0],
                                   "x": 0, "y": 0, "resize_source": False, "mask": ["move_keep", 0]}}
    wf["9"]["inputs"]["images"] = ["move_paste", 0]
    wf["9"]["inputs"]["filename_prefix"] = "qwen21_move"
    sides = [s for s, d in (("left side", dx > 0), ("right side", dx < 0),
                            ("top", dy > 0), ("bottom", dy < 0)) if d]
    moved = [f"{'right' if dx > 0 else 'left'} {abs(dx)} px ({abs(dx) / w:.0%} of the width)"] if dx else []
    moved += [f"{'down' if dy > 0 else 'up'} {abs(dy)} px ({abs(dy) / h:.0%} of the height)"] if dy else []
    report = {"moved": ", ".join(moved), "dx_px": dx, "dy_px": dy,
              "dx_frac": round(dx / w, 3), "dy_frac": round(dy / h, 3),
              "new_area": ", ".join(sides) + " (outpainted)",
              "cropped": ", ".join(e for e, d in (("left edge", dx < 0), ("right edge", dx > 0),
                                                   ("top edge", dy < 0), ("bottom edge", dy > 0)) if d),
              "image_size": [w, h]}
    return " and the ".join(sides), report


def run(model, prompt, image=None, denoise=None, seed=-1, steps=None, out_dir=DEFAULT_OUT, refs=None,
        turbo=None, move_x=0.0, move_y=0.0, info=None):
    """Render one job and return the saved paths; `info`, if a dict, gets extra facts to report."""
    spec = MODELS[model]
    check_prompt_language(model, prompt)
    wf = json.load(open(os.path.join(WF_DIR, spec["wf"]), encoding="utf-8"))

    n, f = spec["prompt"]; wf[n]["inputs"][f] = prompt
    n, f = spec["seed"];   wf[n]["inputs"][f] = random.randint(0, 2**32 - 1) if seed in (-1, None) else int(seed)
    if denoise is not None and "denoise" in spec:
        n, f = spec["denoise"]; wf[n]["inputs"][f] = float(denoise)
    if steps is not None:
        wf["3"]["inputs"]["steps"] = int(steps)
    if spec.get("needs_image"):
        if not image:
            raise ValueError("img2img requires --image <path>")
        n, f = spec["image"]; wf[n]["inputs"][f] = upload_image(image)
    refs = list(refs or [])
    if spec.get("needs_ref"):
        # an edit target given as --image is image_1, ahead of any --ref
        if image:
            refs.insert(0, image)
        if not refs:
            raise ValueError(f"{model} requires the image to edit: --ref <path> (or --image)")
    if refs:
        if "refs" not in spec:
            raise ValueError(f"--ref is only supported by models with a refs slot ({model} has none)")
        if len(refs) > 16:
            raise ValueError("at most 16 reference images (TextEncodeQwenImage21 Autogrow cap)")
        wire_refs(wf, spec, refs)
    if spec.get("move"):
        if len(refs) != 1:
            raise ValueError("qwen21_move takes exactly one image (--ref <path>)")
        side, report = apply_move(wf, refs[0], move_x, move_y)
        if info is not None:
            info["move"] = report
        hint = prompt.strip()
        wf["13"]["inputs"]["prompt"] = MOVE_PROMPT.format(side=side) + (
            f" The extended area shows: {hint}" if hint else "")
    elif move_x or move_y:
        raise ValueError("--move-x / --move-y belong to qwen21_move")
    # turbo is the default for the qwen21 family; an explicit --steps means
    # the caller wants the base sampler. A default that cannot run falls back
    # to base and says so; an explicit --turbo that cannot run is an error.
    if turbo and not spec.get("turbo"):
        raise ValueError(f"--turbo is only supported by the qwen21 family (not {model})")
    if turbo and steps is not None:
        raise ValueError("turbo uses a fixed 6-step schedule; add --no-turbo to set --steps")
    if spec.get("turbo"):
        missing = turbo_missing() if turbo is not False and steps is None else None
        if turbo and missing:
            raise ValueError(f"--turbo unavailable: {missing}. See MODELS.md, or use --no-turbo.")
        if turbo is None:
            turbo = steps is None and not missing
        if info is not None:
            info["turbo"] = ("on" if turbo else
                             f"off (base sampler): {missing}. See MODELS.md." if missing else
                             "off (base sampler)")
    if turbo:
        apply_turbo(wf)

    absent = _missing_model_files(wf)
    if absent:
        # Said before anything is queued. Submitting anyway made ComfyUI answer
        # a bare HTTP 400, three times in a row, before an agent worked out that
        # the Qwen-Image-2.1 files were simply not on this machine.
        raise RuntimeError(
            "model '%s' cannot run here: ComfyUI has no %s. Download it (MODELS.md) or "
            "use another model -- installed ones: %s"
            % (model, ", ".join("%s (%s)" % (f, k) for k, f in absent),
               ", ".join(m for m in MODELS if not _missing_model_files(
                   json.load(open(os.path.join(WF_DIR, MODELS[m]["wf"]), encoding="utf-8"))))))

    duplicate = _queued_duplicate(wf)
    if duplicate:
        # The same picture is already being made. Submitting it again only queues a
        # second copy behind the first -- which is what an agent that lost patience
        # did, three times, on a slow card.
        pid = duplicate
    else:
        body = json.dumps({"prompt": wf, "client_id": uuid.uuid4().hex}).encode()
        req = urllib.request.Request(f"{COMFY_URL}/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            res = json.load(r)
        if res.get("node_errors"):
            raise RuntimeError(f"node_errors: {res['node_errors']}")
        pid = res["prompt_id"]
    print(json.dumps({"prompt_id": pid, "resumed": bool(duplicate)}), file=sys.stderr,
          flush=True)
    return wait_and_collect(pid, out_dir)


_OBJECT_INFO: dict = {}


def _missing_model_files(wf) -> list:
    """[(input name, file)] the workflow names that this ComfyUI cannot load.

    Asked of ComfyUI itself (/object_info lists, per loader node, the files it
    can see -- extra model paths included), so it never disagrees with the
    server. Unanswerable (old server, network) means "assume present".
    """
    absent = []
    for node in wf.values():
        cls, inputs = node.get("class_type"), node.get("inputs") or {}
        for key, val in inputs.items():
            if not (key.endswith("_name") and isinstance(val, str)
                    and val.lower().endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin"))):
                continue
            if cls not in _OBJECT_INFO:
                try:
                    _OBJECT_INFO[cls] = _api_json("/object_info/" + urllib.parse.quote(cls))
                except Exception:
                    _OBJECT_INFO[cls] = None
            info = _OBJECT_INFO[cls]
            try:
                spec = info[cls]["input"]
                choices = (spec.get("required", {}).get(key) or spec.get("optional", {}).get(key))[0]
            except Exception:
                continue
            if isinstance(choices, list) and val not in choices:
                absent.append((key, val))
    return absent


def _api_json(path, timeout=10):
    with urllib.request.urlopen(f"{COMFY_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _queue_state(pid):
    """'running', 'pending', 'absent', or None when ComfyUI cannot be asked."""
    try:
        q = _api_json("/queue")
    except Exception:
        return None
    for key, state in (("queue_running", "running"), ("queue_pending", "pending")):
        if any(len(e) > 1 and e[1] == pid for e in q.get(key, [])):
            return state
    return "absent"


def _queued_duplicate(wf):
    """The prompt_id of an identical graph already queued or running, if any."""
    try:
        q = _api_json("/queue")
    except Exception:
        return None
    for key in ("queue_running", "queue_pending"):
        for e in q.get(key, []):
            if len(e) > 2 and isinstance(e[2], dict) and _same_graph(e[2], wf):
                return e[1]
    return None


def _same_graph(a, b):
    """Same nodes and inputs, ignoring the seed: a resubmission draws a new one."""
    def strip(g):
        return {n: {"class_type": v.get("class_type"),
                    "inputs": {k: x for k, x in (v.get("inputs") or {}).items()
                               if k not in ("seed", "noise_seed")}}
                for n, v in g.items() if isinstance(v, dict)}
    return strip(a) == strip(b)


def _server_log():
    """The log the running server writes, if start_comfy.py started it."""
    try:
        with open(paths.CURRENT_LOG_POINTER, encoding="utf-8") as f:
            path = f.read().strip()
        return path if path and os.path.isfile(path) else None
    except OSError:
        return None


_STEP = re.compile(rb"(\d+)/(\d+) \[")


def _progress(log_path):
    """(log size, "step N/M" or None). Both are read from the tail of the log."""
    if not log_path:
        return None, None
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read()
    except OSError:
        return None, None
    steps = _STEP.findall(tail)
    return size, ("step %s/%s" % (steps[-1][0].decode(), steps[-1][1].decode())
                  if steps else None)


class StillRunning(Exception):
    """The foreground wait ran out while ComfyUI kept rendering."""

    def __init__(self, pid, state, progress):
        super().__init__(pid)
        self.pid, self.state, self.progress = pid, state, progress


def wait_and_collect(pid, out_dir, wait_seconds=None):
    """Wait for `pid` and save its images, or raise StillRunning when time is up.

    Gives up early only for a reason: the prompt vanished (the server restarted),
    or it has run with no sign of progress for STALL_SECONDS. Slow is not a reason.
    """
    deadline = STARTED + (WAIT_SECONDS if wait_seconds is None else wait_seconds)
    log_path = _server_log()
    last_size, last_change, last_state = None, time.time(), None
    absent_since = None
    while True:
        try:
            hist = _api_json(f"/history/{pid}")
        except Exception:
            hist = {}
        if pid in hist:
            break
        state = _queue_state(pid)
        size, step = _progress(log_path)
        now = time.time()
        if size != last_size or state != last_state:
            last_size, last_state, last_change = size, state, now
        if state == "absent":
            absent_since = absent_since or now
            if now - absent_since > 20:
                raise RuntimeError("prompt %s is not in ComfyUI's queue or history any more "
                                   "-- the server was probably restarted. Generate again." % pid)
        else:
            absent_since = None
        if state == "running" and size is not None and now - last_change > STALL_SECONDS:
            raise RuntimeError("prompt %s has made no progress for %d s (last: %s); "
                               "ComfyUI looks stuck. Its log: %s"
                               % (pid, now - last_change, step or "no step yet", log_path))
        if now >= deadline:
            if state == "absent":
                raise RuntimeError("prompt %s is not queued, running or in ComfyUI's "
                                   "history. If the server was restarted it is gone; "
                                   "generate again." % pid)
            raise StillRunning(pid, state or "unknown", step)
        time.sleep(2)

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for node_out in hist[pid]["outputs"].values():
        for it in node_out.get("images", []):
            q = urllib.parse.urlencode({"filename": it["filename"],
                "subfolder": it.get("subfolder", ""), "type": it.get("type", "output")})
            dst = os.path.join(out_dir, it["filename"])
            with urllib.request.urlopen(f"{COMFY_URL}/view?{q}") as r, open(dst, "wb") as f:
                f.write(r.read())
            saved.append(os.path.abspath(dst))
    if not saved:
        status = hist[pid].get("status", {})
        for m in status.get("messages", []):
            if m and m[0] in ("execution_error", "execution_interrupted"):
                detail = m[1].get("exception_message") if isinstance(m[1], dict) else m[1]
                raise RuntimeError("ComfyUI reported %s: %s" % (m[0], detail))
        raise RuntimeError("prompt %s finished without an image" % pid)
    return saved


def free_models() -> dict:
    """Hand ComfyUI's VRAM back once the image is saved.

    Without this, imggen leaves its model resident and the next job pays for it.
    vidgen sizes a clip from *live* free VRAM, so a still-loaded image model was
    measured to cost 3.1 GB: 10.2 GB free became 7.1 GB, and the planner's
    spendable budget dropped 7.4 -> 5.6 GB. vidgen's render path frees
    defensively, but `vidgen plan` deliberately does not (it has to stay
    side-effect free), so the *advice* it gave was pessimistic and could pick an
    unnecessarily downgraded resolution or frame count.

    It matters for the co-resident LLM too, which is waiting on the same VRAM.

    Never raises: a failed unload must not turn a generated image into an error.
    Use --keep-loaded to skip it when generating several images in a row, where
    the per-image reload outweighs the tidiness.
    """
    body = json.dumps({"unload_models": True, "free_memory": True}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/free", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return {"freed": True}
    except Exception as e:
        return {"freed": False, "why": f"{type(e).__name__}: {e}"}


def _still_running(e: "StillRunning") -> dict:
    cmd = '"%s" "%s" status %s' % (Path(sys.executable).as_posix(),
                                    Path(__file__).resolve().as_posix(), e.pid)
    return {"status": "running", "prompt_id": e.pid, "state": e.state,
            "progress": e.progress,
            "note": "ComfyUI is still rendering this and will finish on its own. Do not "
                    "submit it again. Collect the image with the command below.",
            "next": cmd}


def status_main(argv) -> None:
    """gen.py status <prompt_id> [--out DIR] [--keep-loaded] [--wait SECONDS]"""
    ap = argparse.ArgumentParser(prog="gen.py status")
    ap.add_argument("prompt_id")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--keep-loaded", action="store_true")
    ap.add_argument("--wait", type=float, default=0,
                    help="seconds to wait for it to finish (default: just look)")
    a = ap.parse_args(argv)
    try:
        files = wait_and_collect(a.prompt_id, a.out,
                                 wait_seconds=(time.time() - STARTED) + a.wait)
    except StillRunning as e:
        print(json.dumps(_still_running(e), ensure_ascii=False))
        return
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        sys.exit(1)
    out = {"status": "ok", "outputs": files}
    out["vram"] = ("kept loaded (--keep-loaded)" if a.keep_loaded
                   else ("freed" if free_models()["freed"] else "NOT freed"))
    print(json.dumps(out, ensure_ascii=False))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        return status_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description="Generate images via protected local ComfyUI.")
    ap.add_argument("model", choices=list(MODELS),
                    help="qwen | qwen21 | qwen21_edit | qwen21_move | flux | zimage | realvis | img2img")
    ap.add_argument("prompt", help="text prompt (qwen21_move: optional hint for the new area, may be \"\")")
    ap.add_argument("--image", help="input image path (required for img2img; for qwen21_edit it is the edit target, same as the first --ref)")
    ap.add_argument("--ref", action="append", default=[], metavar="PATH",
                    help="reference image for qwen21 (repeatable, up to 16). "
                         "The model sees each ref via the Qwen3-VL vision tower and "
                         "splices its VAE latent into the sequence, so the output "
                         "can match style/subject/character of the refs.")
    ap.add_argument("--denoise", type=float, help="img2img strength (0.85 = change style, keep composition)")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random (default)")
    ap.add_argument("--steps", type=int, help="override step count")
    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--turbo", dest="turbo", action="store_true", default=None,
                    help="qwen21 family: Viggle Turbo LoRA, 6 steps (the default for these models)")
    tg.add_argument("--no-turbo", dest="turbo", action="store_false",
                    help="qwen21 family: base 25-step sampler. About 2.5-3x slower; try it for "
                         "small text, or when a multi-reference edit ghosts or duplicates figures.")
    ap.add_argument("--move-x", type=float, default=0.0,
                    help="qwen21_move: shift as a fraction of the width, + right / - left (e.g. -0.15)")
    ap.add_argument("--move-y", type=float, default=0.0,
                    help="qwen21_move: shift as a fraction of the height, + down / - up")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--keep-loaded", action="store_true",
                    help="leave the model in VRAM afterwards. Use when generating "
                         "several images in a row -- it saves a reload each time, "
                         "but the held VRAM shrinks the next video's budget and "
                         "starves the co-resident LLM. Free it yourself when done.")
    a = ap.parse_args()

    # A linked install (install.py --link) re-renders its SKILL.md after a git
    # pull through a hook; this catches an update the hook did not see.
    import linkcheck
    linkcheck.refresh_if_stale()
    # A local LLM that fills the GPU (Strata) is put to sleep or unloaded for
    # the image and given back afterwards -- see the engine's llmsleep.py.
    import llmsleep  # noqa: E402
    lent = llmsleep.lend("image")
    try:
        ensure_comfy()
        info = {}
        files = run(a.model, a.prompt, image=a.image, denoise=a.denoise,
                    seed=a.seed, steps=a.steps, out_dir=a.out, refs=a.ref,
                    turbo=a.turbo, move_x=a.move_x, move_y=a.move_y, info=info)
        out = {"status": "ok", "outputs": files, **info}
        if a.keep_loaded:
            # Say so: the next job's VRAM budget is measured live, so whoever
            # reads this needs to know the GPU was left occupied on purpose.
            out["vram"] = "kept loaded (--keep-loaded); run without it, or POST " \
                          "/free, before sizing a video job"
            if llmsleep.is_lent():
                out["llm"] = ("the local LLM stays asleep/unloaded while the model is "
                              "kept loaded; the next job without --keep-loaded gives it "
                              "back, or run vidgen's `gen.py llm-return`")
        else:
            r = free_models()
            out["vram"] = "freed" if r["freed"] else f"NOT freed -- {r['why']}"
            llmsleep.give_back(lent)
        print(json.dumps(out, ensure_ascii=False))
    except StillRunning as e:
        # Not freed: the model is in use by the render that is still going.
        print(json.dumps({**_still_running(e), **info}, ensure_ascii=False))
    except Exception as e:
        # Free on the failure path too: a run that died after the model loaded
        # would otherwise leave the GPU occupied with nothing to show for it.
        if not a.keep_loaded:
            free_models()
            llmsleep.give_back(lent)
        print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
