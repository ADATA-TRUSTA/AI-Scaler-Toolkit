"""overlay: additive highlight effects drawn on ORIGINAL-size layers.

Every function mutates `layer` (RGB ndarray, orig size) in place and returns it;
effects are outline/translucent only — never solid fills over text (METHOD.md).
Render the layer through Stage.render() afterwards so effects follow zooms.
"""
import math
import numpy as np


def _clamp_rect(layer, rect):
    h, w = layer.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in rect]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _blend(layer, mask, color, alpha):
    if alpha <= 0 or not mask.any():
        return layer
    c = np.array(color, dtype=np.float64)
    reg = layer[mask].astype(np.float64)
    a = (alpha * mask.astype(np.float64))[mask][:, None] if mask.ndim == 2 else alpha
    layer[mask] = np.clip(reg * (1 - a) + c * a, 0, 255).astype(np.uint8)
    return layer


def _box_mask(layer, x1, y1, x2, y2):
    """Boolean mask of the box; `layer` may be the image array or a shape."""
    sh = layer.shape if hasattr(layer, "shape") else layer
    m = np.zeros(sh[:2], dtype=bool)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def outline(layer, rect, color, width=3, glow=True, alpha=1.0):
    """Outline a rectangle (border only — NEVER fills the interior).

    glow=True adds a soft halo ring around the border at ~35% strength.
    """
    x1, y1, x2, y2 = _clamp_rect(layer, rect)
    if x2 <= x1 or y2 <= y1:
        return layer
    h, w = layer.shape[:2]

    def ring(ox1, oy1, ox2, oy2, ix1, iy1, ix2, iy2):
        outer = _box_mask(layer, ox1, oy1, ox2, oy2)
        inner = _box_mask(layer, max(0, ix1), max(0, iy1), min(w, ix2), min(h, iy2))
        return outer & ~inner

    if glow:
        g = width * 2
        m_glow = ring(x1 - g, y1 - g, x2 + g, y2 + g, x1, y1, x2, y2)
        _blend(layer, m_glow, color, 0.35 * alpha)
    m_core = ring(x1, y1, x2, y2, x1 + width, y1 + width, x2 - width, y2 - width)
    return _blend(layer, m_core, color, alpha)


def column_band(layer, x_range, y_range, color, alpha):
    """Translucent vertical band highlighting a table column."""
    h, w = layer.shape[:2]
    x1 = max(0, int(min(x_range))); x2 = min(w, int(max(x_range)))
    y1 = max(0, int(min(y_range))); y2 = min(h, int(max(y_range)))
    return _blend(layer, _box_mask(layer, x1, y1, x2, y2), color, alpha)


def scan_band(layer, rect, progress, color):
    """A horizontal light band sweeping top->bottom across `rect`.

    progress 0..1; band = bright core with soft edges, clipped to rect.
    """
    h, w = layer.shape[:2]
    x1, y1, x2, y2 = _clamp_rect(layer, rect)
    if x2 <= x1 or y2 <= y1:
        return layer
    p = min(1.0, max(0.0, float(progress)))
    bh = max(6, int((y2 - y1) * 0.14))
    cy = int(y1 + (y2 - y1) * p)
    c = np.array(color, dtype=np.float64)
    for off in range(-bh, bh + 1):
        y = cy + off
        if not (y1 <= y < y2):
            continue
        a = max(0.0, 1.0 - abs(off) / bh) ** 2 * 0.35
        reg = layer[y, x1:x2].astype(np.float64)
        layer[y, x1:x2] = np.clip(reg * (1 - a) + c * a, 0, 255).astype(np.uint8)
    return layer


def aim(layer, rect, state, t, origin=None):
    """Scanning-aim effect: corner brackets + thin scan line inside the box +
    optional faint light cone from `origin` (orig coords) to the box center.

    state='search' -> red; 'ok' -> green with a brief flash at t=0..0.25s.
    """
    h, w = layer.shape[:2]
    x1, y1, x2, y2 = _clamp_rect(layer, rect)
    if x2 <= x1 or y2 <= y1:
        return layer
    color = (60, 235, 90) if state == 'ok' else (255, 70, 60)

    if origin is not None:
        ox, oy = int(origin[0]), int(origin[1])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        steps = max(abs(cy - oy), abs(cx - ox))
        if steps > 4:
            c = np.array(color, dtype=np.float64)
            n = min(steps, 400)
            for i in range(n):
                u = i / float(n)
                y = int(round(oy + (cy - oy) * u))
                x = int(round(ox + (cx - ox) * u))
                spread = int(1 + 7 * u)
                yy = max(0, min(h - 1, y)); xx = max(0, min(w - 1, x))
                seg = layer[yy, max(0, xx - spread):min(w, xx + spread + 1)].astype(np.float64)
                a = 0.05 * (0.3 + 0.7 * u)
                layer[yy, max(0, xx - spread):min(w, xx + spread + 1)] = \
                    np.clip(seg * (1 - a) + c * a, 0, 255).astype(np.uint8)

    bw = max(3, int(min(x2 - x1, y2 - y1)) // 14)          # bracket thickness
    bl = max(6, int(min(x2 - x1, y2 - y1)) // 5)           # bracket length
    for (hx1, hy1, hx2, hy2) in (
            (x1, y1, x1 + bl, y1 + bw), (x1, y1, x1 + bw, y1 + bl),      # top-left
            (x2 - bl, y1, x2, y1 + bw), (x2 - bw, y1, x2, y1 + bl),      # top-right
            (x1, y2 - bw, x1 + bl, y2), (x1, y2 - bl, x1 + bw, y2),      # bottom-left
            (x2 - bl, y2 - bw, x2, y2), (x2 - bw, y2 - bl, x2, y2)):     # bottom-right
        _blend(layer, _box_mask(layer, hx1, hy1, hx2, hy2), color, 0.95)

    p = (t % 1.6) / 1.6
    sy = int(y1 + (y2 - y1) * p)
    _blend(layer, _box_mask(layer, x1 + 2, max(y1, sy), x2 - 2, min(y2, sy + 2)), color, 0.85)

    if state == 'ok':
        tt = t % 1.6
        if tt < 0.3:
            a = (1 - tt / 0.3) * 0.28
            m = _box_mask(layer, x1 + bw, y1 + bw, x2 - bw, y2 - bw)
            _blend(layer, m, color, a)
    return layer


def lower_third(frame, tag, text, color, alpha, text_color=None):
    """Subtitle in the lower third: capsule tag + Chinese main line.

    Font C:/Windows/Fonts/msjhbd.ttc. ONLY for subtitles the user explicitly
    asked for (TOOLKIT_V2 §4). Operates on a FRAME-size image.
    text_color: color of the main line (default = `color`, same as the tag).
    """
    from PIL import Image, ImageDraw, ImageFont
    FONT = r"C:/Windows/Fonts/msjhbd.ttc"
    h, w = frame.shape[:2]
    img = Image.fromarray(np.asarray(frame)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    base = int(h * 0.052)
    f_tag = ImageFont.truetype(FONT, int(base * 0.72))
    f_txt = ImageFont.truetype(FONT, base)
    x = int(w * 0.045)
    y_base = int(h * 0.86)

    tw = dr.textlength(tag, font=f_tag) if tag else 0
    pad_x, cap_h = int(base * 0.5), int(base * 1.25)
    capsule_w = int(tw + pad_x * 2)
    a = int(235 * min(1.0, max(0.0, alpha)))

    if tag:
        dr.rounded_rectangle([x, y_base - cap_h + 4, x + capsule_w, y_base + 8],
                             radius=cap_h // 2, fill=(30, 30, 36, a))
        dr.text((x + pad_x, y_base - cap_h + (cap_h - base * 0.72) / 2 + 4), tag,
                font=f_tag, fill=tuple(color) + (a,))
    tx = x + capsule_w + int(base * 0.5) if tag else x
    # soft shadow for legibility over any background
    dr.text((tx + 2, y_base + 2), text, font=f_txt, fill=(0, 0, 0, int(140 * alpha)))
    dr.text((tx, y_base), text, font=f_txt, fill=tuple(color) + (a,))

    out = Image.alpha_composite(img, ov).convert("RGB")
    arr = np.array(out)
    frame[:] = arr
    return frame
