"""MiniMax-H3: reference images AND a pinned first/last frame in one pass.

Upstream ships H3 as two task families -- fl2va (keyframes) and ref2va
(references) -- and ComfyUI mirrors that with two mutually exclusive nodes. So
"keep these 3 people AND open on exactly this frame" has no route: ref2va
re-invents the framing, fl2va has no reference input.

It turns out the split is a packaging decision, not an architectural one:

  * The DiT takes keyframe rows and reference rows through the SAME mechanism --
    condition latents pinned near t=1 and re-injected every step (see
    `_cond_video_rows` / `PackedLayout` in comfy/ldm/minimax/model.py). The
    layout already has independent branches for both and packs them happily.
  * Measured on this box 2026-09-09: the ref2va checkpoint driven by the fl2va
    node honours a first frame exactly as well as the fl2va checkpoint does
    (frame-0 error 4.56 vs 4.89 / 255 against the source image). ref2va
    understands keyframe conditioning; nothing about it is fl2va-only.

Two things in core stop the combination from working when both are supplied, and
both are bugs of omission in a path nothing upstream exercises:

  1. comfy/model_base.py MiniMaxH3.extra_conds -- `cond_video_latents` is
     written by the keyframe branch and then OVERWRITTEN by the ref branch, so
     the keyframe's latent never reaches the DiT while its rows still exist.
  2. comfy/ldm/minimax/model.py PackedLayout -- a keyframe at pixel 0 is placed
     at temporal position `text_len`, which is the target's origin ONLY when
     there are no refs. Refs push the target forward by one position each, so
     with refs present the "first frame" lands on top of reference #1 and reads
     as just another reference. That is exactly the "it scrambles the opening"
     symptom.

Neither is patched in place here. The node below emits both conditionings, and
the companion patch node installs a DIFFUSION_MODEL wrapper -- a supported
extension point -- that repairs the payload on the way into the model. Both
repairs are strict no-ops unless keyframes AND refs are present together, so
plain fl2va and plain ref2va renders are bit-identical to stock ComfyUI.

Remove the whole directory to uninstall; nothing outside it is modified.
"""

import math

import torch

import comfy.model_management
import comfy.patcher_extension
import node_helpers
from comfy_api.latest import ComfyExtension, io

# Reused rather than copied: these are the exact sizing/packing rules the core
# nodes use, and a divergent copy would be a silent quality bug the first time
# upstream changed one of them.
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _resize,
)
from comfy.ldm.minimax.model import FRAME_RESCALE, _video_t_spans


def _ref_cursor_span(refs):
    """How far the reference blocks advance PackedLayout's temporal cursor.

    Mirrors the `cursor +=` arithmetic in PackedLayout.__init__ exactly; this is
    the amount the target video (and therefore a frame-0 keyframe) is pushed
    forward by the references sitting in front of it.
    """
    span = 0.0
    for blk in refs:
        kind = blk.get("kind")
        if kind == "image":
            span += 1.0
        elif kind == "audio":
            span += float(blk.get("ref_audio_t", 0))
        else:  # video / video_audio
            span += max(float(blk.get("ref_audio_t", 0)),
                        sum(_video_t_spans(blk["latent_t"])))
    return span


def _refkf_wrapper(executor, *args, **kwargs):
    """Repair the H3 payload when keyframes and refs are used together.

    Runs once per sampling run, not once per step: the payload dict is built in
    extra_conds and reused, so the guard key both prevents re-concatenating the
    latents and keeps the position fix from compounding.
    """
    payload = kwargs.get("minimax_payload")
    if not isinstance(payload, dict):
        return executor(*args, **kwargs)
    keyframes, refs = payload.get("keyframes"), payload.get("refs")
    if not (keyframes and refs) or payload.get("_hermes_refkf_done"):
        return executor(*args, **kwargs)
    payload["_hermes_refkf_done"] = True

    # (1) the ref branch clobbered the keyframe latents. Order matters: cond rows
    # are scattered into layout.img_pos where img_update is False, and that mask
    # is built keyframes-first, then refs.
    payload["cond_video_latents"] = (
        [kf["latent"] for kf in keyframes]
        + [r["latent"] for r in refs if "latent" in r])

    # (2) move the keyframe rows to the target's temporal origin, past the refs.
    layout = payload.get("layout")
    span = _ref_cursor_span(refs)
    if layout is not None and span:
        moved = 0
        for start, stop, kind in layout.segments:
            if kind == "cond":            # "cond" segments are keyframes only
                layout.position_ids[start:stop, 0] += span
                moved += stop - start
        if not moved:
            raise RuntimeError(
                "MiniMax-H3 ref+keyframe: the packed layout has no 'cond' rows to "
                "realign, so this ComfyUI's PackedLayout no longer matches what "
                "this plugin was written against. Refusing to render rather than "
                "producing a silently mis-positioned first frame.")
    return executor(*args, **kwargs)


class MiniMaxH3RefKeyframeToVideo(io.ComfyNode):
    """ref2va + fl2va in one conditioning: <Picture N> references AND a pinned frame."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefKeyframeToVideo",
            display_name="MiniMax H3 Reference + Keyframe to Video",
            description=("Reference images (<Picture i>) together with a pinned "
                         "first and/or last frame. Requires the companion "
                         "'MiniMax H3 Ref+Keyframe Patch' node on the model."),
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Applies to ref_images only. Keyframes are always "
                                       "stretched to the canvas, as in fl2va."),
                io.Combo.Input(
                    "keyframe_presentation", options=["picture", "hidden"], default="picture",
                    tooltip="'picture' also shows each keyframe to the text encoder as the "
                            "next <Picture N>, so the prompt can name it; the reference "
                            "numbering is unchanged because keyframes are appended last. "
                            "'hidden' conditions the DiT on the keyframe without telling "
                            "Qwen about it."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"), prefix="ref_image_",
                        min=0, max=9)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                ref_image_size="match", keyframe_presentation="picture",
                first_frame=None, last_frame=None, ref_images=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        # ---- references: identical sizing to MiniMaxH3ReferenceToVideo --------
        ref_items, ref_blocks = [], []
        for img in (ref_images or {}).values():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": vae.encode(resized)})

        # ---- keyframes: identical sizing to MiniMaxH3ImageToVideo -------------
        # first_frame is the geometry anchor (plain stretch); last_frame follows
        # it with a cover-crop. Presented AFTER the references so <Picture 1..N>
        # keeps meaning "the Nth --ref", which is what every prompt assumes.
        keyframes = []
        if first_frame is not None:
            img = _resize(first_frame[:1], width, height, "disabled")
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            img = _resize(last_frame[:1], width, height, "center")
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
        if keyframe_presentation == "picture":
            for kf in keyframes:
                ref_items.append({"type": "image", "data": kf["image"]})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


class MiniMaxH3RefKeyframePatch(io.ComfyNode):
    """Install the payload repair. Harmless on graphs that use only one of the two."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3RefKeyframePatch",
            display_name="MiniMax H3 Ref+Keyframe Patch",
            description=("Lets reference images and pinned keyframes coexist. Without "
                         "it the keyframe's latent is dropped and its rows sit on top "
                         "of reference #1. No effect unless both are present."),
            category="model/patch/minimax",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        m = model.clone()
        to = m.model_options["transformer_options"] = \
            m.model_options.get("transformer_options", {}).copy()
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "hermes_minimax_h3_refkf", _refkf_wrapper, to)
        return io.NodeOutput(m)


class HermesMiniMaxH3RefKFExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3RefKeyframeToVideo, MiniMaxH3RefKeyframePatch]


async def comfy_entrypoint() -> HermesMiniMaxH3RefKFExtension:
    return HermesMiniMaxH3RefKFExtension()
