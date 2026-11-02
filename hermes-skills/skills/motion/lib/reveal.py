"""reveal: hide content with its true background color, then show it back.

All rects in ORIGINAL-image coordinates; operate on orig-size layers
(draw first, render later — see stage.py).
"""
import numpy as np


def _ring_median(arr, rect):
    """Background color around rect = median of the MAJORITY luminance cluster
    in a 3-px ring outside it. The 2-means split discards thin border lines /
    neighboring glyph strokes so white inputs -> white, gray windows -> gray."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in rect]
    m = 3
    rx1, ry1 = max(0, x1 - m), max(0, y1 - m)
    rx2, ry2 = min(w, x2 + m), min(h, y2 + m)
    ring = []
    if ry1 < y1:
        ring.append(arr[ry1:y1, rx1:rx2].reshape(-1, 3))
    if y2 < ry2 and y2 < h:
        ring.append(arr[y2:ry2, rx1:rx2].reshape(-1, 3))
    if rx1 < x1:
        ring.append(arr[ry1:ry2, rx1:x1].reshape(-1, 3))
    if x2 < rx2 and x2 < w:
        ring.append(arr[ry1:ry2, x2:rx2].reshape(-1, 3))
    if not ring:
        return tuple(int(c) for c in np.median(arr.reshape(-1, 3), axis=0))
    pix = np.concatenate(ring).astype(np.float64)
    lum = pix.mean(axis=1)
    # 1-D 2-means (few Lloyd iterations) -> keep the larger cluster's median
    lo_c, hi_c = float(lum.min()), float(lum.max())
    for _ in range(8):
        mid = (lo_c + hi_c) / 2.0
        lo = lum < mid
        hi = ~lo
        if lo.sum() == 0 or hi.sum() == 0:
            break
        new_lo, new_hi = float(lum[lo].mean()), float(lum[hi].mean())
        if abs(new_lo - lo_c) < 1 and abs(new_hi - hi_c) < 1:
            lo_c, hi_c = new_lo, new_hi
            break
        lo_c, hi_c = new_lo, new_hi
    mid = (lo_c + hi_c) / 2.0
    sel = pix[lum < mid] if (lum < mid).sum() >= (lum >= mid).sum() else pix[lum >= mid]
    if len(sel) == 0:
        sel = pix
    return tuple(int(c) for c in np.median(sel, axis=0))


def _clip(rect, clip):
    if clip is None:
        return rect
    return (max(rect[0], clip[0]), max(rect[1], clip[1]),
            min(rect[2], clip[2]), min(rect[3], clip[3]))


def hide(img, rects, color='auto', clip_to=None):
    """Paint each rect over with its surrounding median color (white input ->
    white, gray window -> gray). Returns a NEW image; original untouched.

    clip_to: optional (x1,y1,x2,y2) hard limit so paint never escapes the
             enclosing input box / table cell.
    """
    out = np.array(img).copy()
    if isinstance(rects, tuple) or (isinstance(rects, list) and rects and not hasattr(rects[0], '__len__')):
        rects = [rects]
    for rect in rects:
        r = _clip([int(v) for v in rect], clip_to)
        x1, y1, x2, y2 = r
        if x2 <= x1 or y2 <= y1:
            continue
        col = color if not isinstance(color, str) else _ring_median(out, r)
        out[y1:y2, x1:x2] = col
    return out


def reveal_ltr(hidden, orig, rect, progress, steps=None):
    """Reveal original pixels inside `rect` from left to right.

    progress: 0..1 fraction of the width shown.
    steps: if given (e.g. character count), progress is quantised into that
           many discrete jumps (typewriter / per-character reveal).
    Returns a NEW image.
    """
    out = np.array(hidden).copy()
    x1, y1, x2, y2 = [int(v) for v in rect]
    if steps and steps > 0:
        k = int(round(min(1.0, max(0.0, progress)) * steps))
        prog = k / float(steps)
    else:
        prog = min(1.0, max(0.0, progress))
    cut = x1 + int(round((x2 - x1) * prog))
    if cut > x1 and y2 > y1:
        out[y1:y2, x1:min(cut, out.shape[1])] = np.array(orig)[y1:y2, x1:min(cut, out.shape[1])]
    return out


def reveal_rows(hidden, orig, row_rects, t, t0, per_row, fade=0.12):
    """Fade table rows in one by one, top to bottom.

    Row i starts at t0 + i*per_row and fades over `fade` seconds.
    Returns a NEW image (rows not yet faded show the hidden version).
    """
    hid = np.asarray(hidden)
    org = np.asarray(orig)
    out = hid.copy()
    for i, rect in enumerate(row_rects):
        start = t0 + i * per_row
        if t <= start:
            continue
        a = min(1.0, (t - start) / max(fade, 1e-6))
        x1, y1, x2, y2 = [int(v) for v in rect]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = out[y1:y2, x1:x2].astype(np.float64) * (1 - a) + \
                 org[y1:y2, x1:x2].astype(np.float64) * a
        out[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def assert_hidden(frame_img, rect, tol=6):
    """Assert region is the uniform hide color. Returns stats dict."""
    reg = np.asarray(frame_img)[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
    if reg.size == 0:
        raise AssertionError(f"assert_hidden: empty rect {rect}")
    spread = float(np.max(reg.max(axis=(0, 1)) - reg.min(axis=(0, 1))))
    if spread > tol:
        raise AssertionError(f"assert_hidden failed: rect={rect} color spread={spread:.1f}>{tol}")
    stats = {'spread': spread, 'color': tuple(int(c) for c in np.median(reg.reshape(-1, 3), axis=0))}
    return stats


def assert_revealed(frame_img, orig, rect, tol=4.0):
    """Assert region of frame matches the original pixels (mean |diff| < tol).

    If sizes differ, `orig` is resized to frame size first (so you can check a
    rendered 16:9 frame against the same view-box crop logic used by Stage).
    Returns stats dict.
    """
    from PIL import Image
    f = np.asarray(frame_img)
    o = np.asarray(orig)
    if f.shape[:2] != o.shape[:2]:
        o = np.array(Image.fromarray(o).resize((f.shape[1], f.shape[0]), Image.LANCZOS))
    rect = [int(v) for v in rect]
    rf = f[rect[1]:rect[3], rect[0]:rect[2]]
    ro = o[rect[1]:rect[3], rect[0]:rect[2]]
    if rf.size == 0:
        raise AssertionError(f"assert_revealed: empty rect {rect}")
    mad = float(np.mean(np.abs(rf.astype(np.float64) - ro.astype(np.float64))))
    if mad > tol:
        raise AssertionError(f"assert_revealed failed: rect={rect} mean|diff|={mad:.2f}>{tol}")
    return {'mean_abs_diff': mad}
