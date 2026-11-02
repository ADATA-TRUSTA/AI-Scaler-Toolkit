"""compose: AI-clip fitting and photo repair helpers (TOOLKIT_V2 §6).

fit_clip / restore_outside / skin_mask / retime work on frame-space pixels;
cutout / clean_plate / portrait_in_16x9 work in original-image coordinates.
"""
import numpy as np
from PIL import Image, ImageFilter


def _to_list(frames):
    return frames if isinstance(frames, (list, tuple)) else [frames]


def fit_clip(frames, size=(1280, 720)):
    """Stretch AI clips (e.g. 1280x704) to `size` with no letterboxing."""
    out = []
    for f in _to_list(frames):
        a = np.asarray(f)
        if (a.shape[1], a.shape[0]) == tuple(size):
            out.append(a.copy())
        else:
            out.append(np.array(Image.fromarray(a).resize(size, Image.LANCZOS)))
    return out


def restore_outside(ai_frame, orig_frame, keep_mask):
    """Return ai_frame with every pixel OUTSIDE keep_mask replaced by orig_frame."""
    ai = np.asarray(ai_frame).copy()
    og = np.asarray(orig_frame)
    m = np.asarray(keep_mask).astype(bool)
    if m.shape != ai.shape[:2]:
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
            (ai.shape[1], ai.shape[0]), Image.NEAREST)) > 127
    ai[~m] = og[~m]
    return ai


def skin_mask(ai_frame, orig_frame, grow_from='bottom', protect_rects=(),
              diff_min=36, sat_min=18, sat_max=90, v_min=120, r_g_min=8):
    """Mask of pixels that are BOTH unlike the original AND skin-toned, kept
    only in the component connected to a frame edge (`grow_from`).

    Inside protect_rects (e.g. an AI-redrawn button) only *strict* skin counts:
    high saturation band — prevents flat button colors from leaking into the
    arm/skin region. Returns bool ndarray (H, W).
    """
    import cv2
    ai = np.asarray(ai_frame).astype(np.int16)[:, :, :3]
    og = np.asarray(orig_frame).astype(np.int16)[:, :, :3]
    if ai.shape != og.shape:
        og = np.array(Image.fromarray(np.asarray(orig_frame)).resize(
            (ai.shape[1], ai.shape[0]), Image.LANCZOS)).astype(np.int16)

    diff = np.abs(ai - og).mean(axis=2) > diff_min

    r, g, b = ai[:, :, 0].astype(np.float64), ai[:, :, 1].astype(np.float64), ai[:, :, 2].astype(np.float64)
    mx = ai.max(axis=2).astype(np.float64)
    mn = ai.min(axis=2).astype(np.float64)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1) * 255.0, 0)
    skin_loose = (r > g) & (r > b) & ((r - g) >= r_g_min) & (mx >= v_min) & (sat >= 6)
    skin_strict = skin_loose & (sat >= sat_min) & (sat <= sat_max * 2.2)

    cand = diff & skin_loose
    h, w = cand.shape
    prot = np.zeros((h, w), dtype=bool)
    for (x1, y1, x2, y2) in protect_rects or ():
        prot[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = True
    cand[prot] &= skin_strict[prot]

    # connected components; keep those touching the requested edge(s)
    num, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=4)
    edge_labels = set()
    if grow_from in ('bottom', 'both'):
        edge_labels.update(labels[h - 1, :][labels[h - 1, :] > 0].tolist())
    if grow_from in ('top', 'both'):
        edge_labels.update(labels[0, :][labels[0, :] > 0].tolist())
    if grow_from in ('left', 'both'):
        edge_labels.update(labels[:, 0][labels[:, 0] > 0].tolist())
    if grow_from in ('right', 'both'):
        edge_labels.update(labels[:, -1][labels[:, -1] > 0].tolist())
    if not edge_labels:  # nothing touches the edge — fall back to largest comp
        sizes = np.bincount(labels.ravel())
        if num > 1:
            edge_labels = {int(np.argmax(sizes[1:]) + 1)}
    return np.isin(labels, list(edge_labels))


def retime(frames, keep_ranges):
    """Keep only frame index ranges [(s0,e0), (s1,e1)...] (e exclusive)."""
    frames = _to_list(frames)
    out = []
    for s, e in keep_ranges:
        out.extend(frames[int(s):int(e)])
    return out


def cutout(img, polygon, feather=3):
    """RGBA sticker of `img` inside `polygon` [(x,y),...] with feathered edge."""
    a = np.asarray(img)[:, :, :3]
    h, w = a.shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask_img).polygon([tuple(map(int, p)) for p in polygon], fill=255)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))
    m = np.array(mask_img)
    rgba = np.dstack([a, m])
    return rgba


def clean_plate(img, polygon, shift=(0, -40)):
    """Fill `polygon`'s area with the image content shifted by `shift`
    (typically one repeat period upward), removing an object that was taken
    out of a repeating pattern. Returns a new RGB image."""
    import cv2
    a = np.array(np.asarray(img)[:, :, :3])
    h, w = a.shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask_img).polygon([tuple(map(int, p)) for p in polygon], fill=255)
    m = np.array(mask_img) > 127
    dx, dy = int(shift[0]), int(shift[1])
    src = np.roll(np.roll(a, -dy, axis=0), -dx, axis=1)
    a[m] = src[m]
    return a


def fill_by_shift(img, hole_mask, shifts=((0, -828), (0, -1656), (0, -2484)),
                  avoid=None):
    """Fill `hole_mask` pixels with the image's own content sampled at
    (x+dx, y+dy) for each (dx,dy) in `shifts`, in order — repeating-pattern
    repair. A hole pixel takes that source only when the source pixel is NOT
    itself inside the hole (or inside `avoid`, e.g. an object elsewhere in the
    frame that must never be copied); leftovers try later shifts; anything
    still unfilled gets the median color of the hole's outer ring. New RGB img.

    Use instead of cv2.inpaint on structured backgrounds (inpaint smears)."""
    import cv2 as _cv2
    a = np.array(np.asarray(img)[:, :, :3])
    h, w = a.shape[:2]
    hole = np.asarray(hole_mask) > 0
    bad = hole if avoid is None else (hole | (np.asarray(avoid) > 0))
    out = a.copy()
    remaining = hole.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    for dx, dy in shifts:
        sy = np.clip(yy + dy, 0, h - 1)
        sx = np.clip(xx + dx, 0, w - 1)
        src_hole = bad[sy, sx]
        ok = remaining & ~src_hole
        out[ok] = a[sy[ok], sx[ok]]
        remaining &= ~ok
        if not remaining.any():
            break
    if remaining.any():
        ring = (_cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~hole
        med = np.median(a[ring], axis=0) if ring.any() else [0, 0, 0]
        out[remaining] = np.array(med)
    return out


_REMBG_SESSIONS = {}  # model name -> rembg session (module-level cache)


def matte(img, model="u2net", keep="bottom_right", scale=3):
    """Background-removal cutout of the object anchored in one corner.

    Runs rembg (`only_mask=True`) on a 1/`scale` copy for speed, upscales the
    mask back, thresholds + morphologically closes it, then keeps ONLY the
    largest connected component whose bounding box reaches the chosen `keep`
    corner (bottom_right / bottom / right) — so rows of parts elsewhere in the
    frame (e.g. an SSD column on the left) never leak into the foreground.
    Final mask is feathered 3px. Returns RGBA PIL Image at original size.
    """
    import cv2
    from rembg import new_session, remove
    if isinstance(img, Image.Image):
        rgb = img.convert("RGB")
    else:
        rgb = Image.fromarray(np.asarray(img)[:, :, :3])
    w, h = rgb.size

    sess = _REMBG_SESSIONS.get(model)
    if sess is None:
        sess = _REMBG_SESSIONS[model] = new_session(model)

    small = rgb.resize((max(1, w // scale), max(1, h // scale)), Image.LANCZOS)
    m_small = remove(small, session=sess, only_mask=True)
    m = np.array(m_small.resize((w, h), Image.LANCZOS))

    binm = (m > 64).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=4)
    best, best_area = -1, 0
    for i in range(1, num):
        x, y, cw, chh, area = stats[i]
        reach_b = (y + chh) > h * 0.7
        reach_r = (x + cw) > w * 0.7
        hit = {"bottom_right": reach_b and reach_r,
               "bottom": reach_b, "right": reach_r}[keep]
        if hit and area > best_area:
            best, best_area = i, area
    kept = (labels == best).astype(np.uint8) * 255 if best > 0 else np.zeros((h, w), np.uint8)

    mask_img = Image.fromarray(kept).filter(ImageFilter.GaussianBlur(radius=3))
    return Image.fromarray(np.dstack([np.array(rgb), np.array(mask_img)]), "RGBA")


def portrait_in_16x9(img, view_box=None):
    """Portrait content centered in 16:9 with blurred side extension.

    view_box (orig coords) may be wider than the image; areas outside the
    photo are filled with a blurred cover-scale copy of the whole image.
    Returns frame ndarray 1280x720.
    """
    try:
        from .stage import Stage
    except ImportError:
        from stage import Stage
    st = Stage(np.asarray(img), trim_top_white=False)
    box = st.full_box() if view_box is None else tuple(view_box)
    return st.render(None, box)[0]
