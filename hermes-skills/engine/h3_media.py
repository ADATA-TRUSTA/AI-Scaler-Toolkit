"""Media helpers for the H3 chained path.

Runs under ComfyUI's embedded python (which already ships PyAV), so video_gen.py
itself can stay standard-library only and work under any interpreter Hermes uses.

  python h3_media.py lastframe <video.mp4> <out.png>
  python h3_media.py concat <out.mp4> <seg1.mp4> <seg2.mp4> ...
  python h3_media.py convert <in.heic> <out.png>
  python h3_media.py crop <in> <out> <x,y,w,h as fractions of the source>
  python h3_media.py frame0diff <video> <image>   -> mean abs diff, 0-255
  python h3_media.py probe <video>                -> JSON: w, h, duration, fps, has_audio, cuts
  python h3_media.py refvideo <in> <out.mp4> <short> <start_s> <frames> [<out.wav>|-] [<end_s>]
  python h3_media.py assemble <out> <w> <h> <audio_src|-> <audio_start> <seg.mp4>:<frames> ...
"""
import json
import math
import os
import sys

import av
from PIL import Image

try:                      # iPhone photos arrive as HEIC; ComfyUI's PIL cannot
    import pillow_heif    # open them unless the opener is registered here.
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def last_frame(src: str, dst: str) -> None:
    c = av.open(src)
    frame = None
    for frame in c.decode(video=0):
        pass
    if frame is None:
        raise SystemExit("no video frames in %s" % src)
    frame.to_image().save(dst)
    print(dst)


def concat(dst: str, segs: list[str]) -> None:
    first = av.open(segs[0])
    vs0 = [s for s in first.streams if s.type == "video"][0]
    W, H = vs0.width, vs0.height
    fps = float(vs0.average_rate or 24)

    out = av.open(dst, "w")
    vout = out.add_stream("libx264", rate=int(round(fps)))
    vout.width, vout.height, vout.pix_fmt = W, H, "yuv420p"
    # CRF 16: the segments are already h264, so this second pass must be well
    # above ComfyUI's default 23 or the joins visibly soften.
    vout.options = {"crf": "16", "preset": "slow"}
    aout = out.add_stream("aac", rate=32000)
    aout.layout = "stereo"

    nv = 0
    for s in segs:
        c = av.open(s)
        for f in c.decode(video=0):
            img = f.to_image()
            if img.size != (W, H):
                img = img.resize((W, H), Image.LANCZOS)
            for pkt in vout.encode(av.VideoFrame.from_image(img)):
                out.mux(pkt)
            nv += 1
        try:
            c2 = av.open(s)
            rs = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=32000)
            for fr in c2.decode(audio=0):
                for rf in rs.resample(fr):
                    rf.pts = None
                    for pkt in aout.encode(rf):
                        out.mux(pkt)
        except Exception:
            pass          # a segment without audio must not sink the join

    for pkt in vout.encode():
        out.mux(pkt)
    for pkt in aout.encode():
        out.mux(pkt)
    out.close()
    print("%s %d %.2f" % (dst, nv, nv / fps))


def convert(src: str, dst: str) -> None:
    """Re-encode any PIL-readable still (HEIC included) as PNG for ComfyUI.

    Phone photos are also huge (12 MP portrait); H3 downscales references to a
    2048 short edge anyway, so shrink here to keep staging cheap.
    """
    im = Image.open(src)
    im = im.convert("RGB")
    if min(im.size) > 2048:
        s = 2048 / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dst)
    print("%s %dx%d" % (dst, im.width, im.height))


def crop(src: str, dst: str, rect: str) -> None:
    """Cut a fractional box out of a still.

    Fractions, not pixels: the caller that picked the box looked at the picture
    rather than at its resolution, and the same numbers then work for the 447px
    thumbnail and the 5712px original.
    """
    x, y, w, h = (float(v) for v in rect.split(","))
    im = Image.open(src).convert("RGB")
    W, H = im.size
    box = (max(0, int(x * W)), max(0, int(y * H)),
           min(W, int((x + w) * W)), min(H, int((y + h) * H)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise SystemExit("crop box is degenerate: %s of %dx%d -> %s" % (rect, W, H, box))
    im = im.crop(box)
    if min(im.size) > 2048:                 # same cap staging applies
        s = 2048 / min(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(dst)
    print("%s %dx%d (from %dx%d)" % (dst, im.width, im.height, W, H))


def frame0diff(video: str, image: str) -> None:
    """Mean absolute difference between the video's first frame and `image`.

    Exists because anchoring a first frame is a CONDITION, not a guarantee --
    H3 sometimes re-composes anyway. Predicting that from the prompt turned out
    to be unreliable, so measure the result instead: ~5-8/255 is a VAE+h264
    round trip (i.e. the frame really is the image), 40+ means H3 opened on
    something else.

    The image is stretched to the video's canvas, matching how the fl2va node
    places a first frame (`_resize(..., "disabled")`) -- a cover-crop
    comparison would report a difference the model never introduced.
    """
    c = av.open(video)
    try:
        frame = next(c.decode(video=0))
    finally:
        c.close()
    a = frame.to_image().convert("RGB")
    b = Image.open(image).convert("RGB").resize(a.size, Image.LANCZOS)
    pa, pb = a.load(), b.load()
    w, h = a.size
    step = max(1, min(w, h) // 256)          # sample: exact enough, much faster
    tot = n = 0
    for y in range(0, h, step):
        for x in range(0, w, step):
            r1, g1, b1 = pa[x, y]
            r2, g2, b2 = pb[x, y]
            tot += abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
            n += 3
    print("%.2f %dx%d" % (tot / n if n else -1.0, w, h))


def _rotation(stream) -> int:
    """Clockwise degrees a player would rotate this stream by (phone videos)."""
    try:
        r = int(float((stream.metadata or {}).get("rotate", 0)))
    except (TypeError, ValueError):
        r = 0
    return r % 360


def _upright(img, rot: int):
    # PIL rotates counter-clockwise; the container tag is clockwise
    return img.rotate(-rot, expand=True) if rot in (90, 180, 270) else img


def probe(path: str) -> None:
    """Geometry of a reference video as H3 will see it (rotation applied)."""
    c = av.open(path)
    try:
        vs = c.streams.video[0]
        rot = _rotation(vs)
        w, h = vs.codec_context.width, vs.codec_context.height
        if rot in (90, 270):
            w, h = h, w
        dur = None
        if vs.duration is not None and vs.time_base is not None:
            dur = float(vs.duration * vs.time_base)
        elif c.duration is not None:
            dur = c.duration / 1e6
        fps = float(vs.average_rate or 0) or None
        has_audio = bool(c.streams.audio)
    finally:
        c.close()
    print(json.dumps({"w": w, "h": h, "duration": dur, "fps": fps, "rotate": rot,
                      "has_audio": has_audio, "cuts": _cuts(path)}))


def _cuts(path: str) -> list:
    """Hard cuts (and short dissolves) in a video, as the time of the new shot's first frame.

    A cut is a frame-to-frame jump far above the clip's own motion level: mean
    abs diff of a 48 px-wide grey thumbnail above max(12, 6 x median), and the
    biggest jump within +-3 frames so a two-frame dissolve counts once.
    Calibrated on a 13.9 s multi-shot clip: the three real cuts scored 27-58
    against a median of 1.9, the next-highest non-cut 5.6.
    """
    import numpy as np
    c = av.open(path)
    try:
        vs = c.streams.video[0]
        prev, ds, ts = None, [], []
        for f in c.decode(vs):
            img = f.to_image().convert("L")
            g = np.asarray(img.resize((48, max(1, round(48 * img.height / img.width)))),
                           dtype=np.float32)
            if prev is not None and g.shape == prev.shape:
                ds.append(float(np.abs(g - prev).mean()))
                ts.append(float(f.time or 0.0))
            prev = g
    finally:
        c.close()
    if len(ds) < 3:
        return []
    thr = max(12.0, 6.0 * float(np.median(ds)))
    cuts = []
    for i, d in enumerate(ds):
        if d > thr and d >= max(ds[max(0, i - 3):i + 4]):
            cuts.append(round(ts[i], 3))
    return cuts


def refvideo(src: str, dst: str, short: int, start: float, frames: int,
             wav: str | None = None, end: float | None = None) -> None:
    """Normalise a reference video for MiniMaxH3ReferenceToVideo.

    The node assumes 24 fps and only ever scales DOWN to H3's 768 canvas, so
    every reference token it builds comes straight from what we hand it.
    Here: resample to 24 fps by picking the latest source frame at each output
    tick, scale so the short edge is `short` (both sides on the /32 grid the node
    snaps to anyway), keep exactly `frames` frames starting at `start` seconds,
    and drop the soundtrack -- it goes to `wav` only when asked for.

    `end` stops reading the source there and holds its last frame for whatever
    is left of `frames`: one chunk of a shot-split render needs exactly its own
    shot, padded up to H3's 17k+5 grid, never the first frames of the next one.
    """
    end = float("inf") if end is None else end
    c = av.open(src)
    vs = c.streams.video[0]
    rot = _rotation(vs)
    out = vo = None
    n = 0
    prev = None                 # latest decoded frame whose time is <= the next tick
    cached = (None, None)       # (frame, encoded VideoFrame): 16->24 fps repeats frames

    def emit(frame):
        nonlocal out, vo, cached, n
        if out is None:
            out, vo = _open_encoder(dst, frame, rot, short)
        if cached[0] is not frame:
            img = _upright(frame.to_image(), rot).resize((vo.width, vo.height), Image.LANCZOS)
            cached = (frame, img)
        for pkt in vo.encode(av.VideoFrame.from_image(cached[1])):
            out.mux(pkt)
        n += 1

    try:
        for f in c.decode(vs):
            t = f.time if f.time is not None else 0.0
            # every tick strictly before this frame belongs to the previous one
            while n < frames and start + n / 24.0 < min(t - 1e-6, end - 1e-6):
                emit(prev if prev is not None else f)
            if n >= frames or start + n / 24.0 >= end - 1e-6:
                break
            prev = f
        while n < frames and prev is not None:      # source/chunk ended: hold
            emit(prev)
    finally:
        c.close()
    if out is None:
        raise SystemExit("no video frames in %s" % src)
    for pkt in vo.encode():
        out.mux(pkt)
    w, h = vo.width, vo.height
    out.close()

    has_audio = False
    if wav:
        has_audio = _extract_wav(src, wav, start, min(frames / 24.0, end - start))
    print(json.dumps({"file": dst, "w": w, "h": h, "frames": n,
                      "wav": wav if has_audio else None}))


def _open_encoder(dst, frame, rot, short):
    """h264 writer sized from the first frame: short edge -> `short`, /32 grid.

    Never upscales. memplan.h3_ref_video_dims() prices the same arithmetic, so
    keep the two in step.
    """
    w, h = frame.width, frame.height
    if rot in (90, 270):
        w, h = h, w
    s = min(1.0, short / float(min(w, h)))
    tw = max(32, int(round(w * s / 32.0)) * 32)
    th = max(32, int(round(h * s / 32.0)) * 32)
    out = av.open(dst, "w")
    vo = out.add_stream("libx264", rate=24)
    vo.width, vo.height, vo.pix_fmt = tw, th, "yuv420p"
    vo.options = {"crf": "14", "preset": "medium"}
    return out, vo


def _audio_planar(src: str, start: float, seconds: float, rate: int = 48000):
    """`seconds` of src's first audio stream from `start`, as float32 (2, n); zeros if none."""
    import numpy as np
    need = int(round(seconds * rate))
    out = np.zeros((2, need), dtype=np.float32)
    c = av.open(src)
    try:
        if not c.streams.audio:
            return out
        rs = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=rate)
        skip, got = int(round(start * rate)), 0
        for fr in c.decode(audio=0):
            for rf in rs.resample(fr):
                arr = rf.to_ndarray()
                lo = min(arr.shape[1], skip)
                skip -= lo
                take = min(arr.shape[1] - lo, need - got)
                if take > 0:
                    out[:, got:got + take] = arr[:, lo:lo + take]
                    got += take
            if got >= need:
                break
    finally:
        c.close()
    return out


def assemble(dst: str, width: int, height: int, audio_src: str | None,
             audio_start: float, segs: list) -> None:
    """Join shot-split chunks into one clip: exact frame counts, one canvas, one soundtrack.

    Each chunk was rendered on H3's 17k+5 grid and is LONGER than its shot by
    up to 16 frames of padding, so it is cut back to its own `n` frames here --
    that is what keeps the joined clip frame-locked to the reference video and
    its soundtrack. Audio comes from `audio_src` (the reference's own track)
    when given, else from each chunk's H3 audio, trimmed the same way.
    """
    import numpy as np
    out = av.open(dst, "w")
    vo = out.add_stream("libx264", rate=24)
    vo.width, vo.height, vo.pix_fmt = width, height, "yuv420p"
    vo.options = {"crf": "16", "preset": "slow"}
    rate = 48000
    ao = out.add_stream("aac", rate=rate)
    ao.layout = "stereo"

    total = 0
    for path, n in segs:
        c = av.open(path)
        last, k = None, 0
        try:
            for f in c.decode(video=0):
                if k >= n:
                    break
                last = f.to_image().convert("RGB").resize((width, height), Image.LANCZOS)
                for pkt in vo.encode(av.VideoFrame.from_image(last)):
                    out.mux(pkt)
                k += 1
        finally:
            c.close()
        while k < n and last is not None:           # chunk came back short: hold
            for pkt in vo.encode(av.VideoFrame.from_image(last)):
                out.mux(pkt)
            k += 1
        total += k
    for pkt in vo.encode():
        out.mux(pkt)

    if audio_src:
        pcm = _audio_planar(audio_src, audio_start, total / 24.0, rate)
    else:
        pcm = np.concatenate([_audio_planar(p, 0.0, n / 24.0, rate) for p, n in segs], axis=1)
    step = ao.codec_context.frame_size or 1024
    for i in range(0, pcm.shape[1], step):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, i:i + step]),
                                        format="fltp", layout="stereo")
        fr.sample_rate = rate
        for pkt in ao.encode(fr):
            out.mux(pkt)
    for pkt in ao.encode():
        out.mux(pkt)
    out.close()
    print(json.dumps({"file": dst, "frames": total, "seconds": round(total / 24.0, 3)}))


def _extract_wav(src: str, wav: str, start: float, seconds: float) -> bool:
    c = av.open(src)
    try:
        if not c.streams.audio:
            return False
        out = av.open(wav, "w")
        ao = out.add_stream("pcm_s16le", rate=48000)
        ao.layout = "stereo"
        rs = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=48000)
        need = int(math.ceil(seconds * 48000))
        skip = int(start * 48000)
        got = 0
        for fr in c.decode(audio=0):
            for rf in rs.resample(fr):
                arr = rf.to_ndarray()                    # (1, samples*2) packed s16
                total = arr.shape[1] // 2
                lo = min(total, max(0, skip))
                skip -= lo
                hi = min(total, lo + (need - got))
                if hi <= lo:
                    continue
                piece = av.AudioFrame.from_ndarray(
                    arr[:, lo * 2:hi * 2].copy(), format="s16", layout="stereo")
                piece.sample_rate = 48000
                for pkt in ao.encode(piece):
                    out.mux(pkt)
                got += hi - lo
            if got >= need:
                break
        for pkt in ao.encode():
            out.mux(pkt)
        out.close()
        return got > 0
    finally:
        c.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "lastframe":
        last_frame(sys.argv[2], sys.argv[3])
    elif cmd == "concat":
        concat(sys.argv[2], sys.argv[3:])
    elif cmd == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif cmd == "crop":
        crop(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "frame0diff":
        frame0diff(sys.argv[2], sys.argv[3])
    elif cmd == "probe":
        probe(sys.argv[2])
    elif cmd == "refvideo":
        wav = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else None
        refvideo(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]),
                 int(sys.argv[6]), wav, float(sys.argv[8]) if len(sys.argv) > 8 else None)
    elif cmd == "assemble":
        # assemble <out> <w> <h> <audio_src|-> <audio_start> <seg.mp4>:<frames> ...
        segs = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in sys.argv[7:]]
        assemble(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                 None if sys.argv[5] == "-" else sys.argv[5], float(sys.argv[6]), segs)
    else:
        raise SystemExit("unknown command %r" % cmd)
