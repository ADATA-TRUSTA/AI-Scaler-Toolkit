"""panel: side panels assembled from ORIGINAL-image table cells.

Rows are always taken with the SAME y range per row (整列對齊, METHOD.md §1.4):
each row slice is exactly its row_rect's y span; columns are cut by x ranges.
Panels are RGBA cards with rounded corners and a soft shadow. Only original
pixels appear in cells — no text is ever re-typed.
"""
import numpy as np
from PIL import Image, ImageDraw


def _rounded_alpha(size, radius):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return np.array(m)


class _Panel(np.ndarray):
    """ndarray subclass so the card can carry .row_bounds as an attribute."""
    pass


def make_panel(orig, col_ranges, header_rect, row_rects, scale=1.0, pad=10):
    """Stitch the given columns of each row into one white rounded card (RGBA).

    orig        : RGB ndarray (original screenshot)
    col_ranges  : [(x1,x2), ...] column x-spans to keep, left-to-right
    header_rect : (x1,y1,x2,y2) table header band → panel's first row
    row_rects   : data rows; each row slice uses exactly this rect's y span,
                  so a clipped last half-row stays half-height.

    The returned array carries `.row_bounds` = [y0, ...] card-pixel y of every
    row start (used by paste_panel(rows_visible=N)).
    """
    orig = np.asarray(orig)[:, :, :3]
    rects = list([header_rect]) + list(row_rects)
    col_w = [max(1, int(round((c[1] - c[0]) * scale))) for c in col_ranges]
    row_h = [max(1, int(round((int(r[3]) - int(r[1])) * scale))) for r in rects]

    gap_x = max(2, int(pad))
    body_w = sum(col_w) + gap_x * (len(col_ranges) - 1) + pad * 2
    body_h = sum(row_h) + pad * 2
    shadow = max(6, int(pad))
    W, H = body_w + shadow, body_h + shadow

    card = np.zeros((H, W, 4), dtype=np.uint8)
    a_body = np.zeros((H, W), dtype=np.uint8)
    m1 = _rounded_alpha((body_w, body_h), radius=min(12, max(4, body_h // 6)))

    # soft shadow: two offset copies of the rounded mask, low alpha
    for off, level in ((shadow, 45), (max(2, shadow // 2), 30)):
        reg = card[off:off + body_h, off:off + body_w, 3]
        np.maximum(reg, m1 // (255 // level), out=reg)

    # white rounded body under the content
    a_body[:body_h, :body_w] = m1
    card[:, :, :3] = np.where(a_body[..., None] > 0, 255, card[:, :, :3])
    card[:, :, 3] = np.maximum(card[:, :, 3], a_body)

    # paste original pixels row by row
    y = pad
    row_bounds = [pad]
    for rr, rh in zip(rects, row_h):
        x1, y1, x2, y2 = [int(v) for v in rr]
        x = pad
        for (cx1, cx2), cw in zip(col_ranges, col_w):
            cell = orig[max(0, y1):max(y1 + 1, y2), max(0, int(cx1)):max(int(cx1) + 1, int(cx2))]
            im = np.array(Image.fromarray(cell).resize((cw, rh), Image.LANCZOS)).astype(np.float64)
            aa = (a_body[y:y + rh, x:x + cw] / 255.0)[..., None]
            reg = card[y:y + rh, x:x + cw, :3].astype(np.float64)
            card[y:y + rh, x:x + cw, :3] = np.clip(reg * (1 - aa) + im * aa, 0, 255).astype(np.uint8)
            np.maximum(card[y:y + rh, x:x + cw, 3], a_body[y:y + rh, x:x + cw],
                       out=card[y:y + rh, x:x + cw, 3])
            x += cw + gap_x
        y += rh
        row_bounds.append(y)

    # subtle separator line between header and data rows (visual only)
    d = ImageDraw.Draw(Image.fromarray(card))
    sep_y = pad + row_h[0]
    d.line([pad // 2, sep_y, body_w - pad // 2, sep_y], fill=(214, 215, 220, 200), width=1)

    card = card.view(_Panel)   # ndarray subclass -> attributes survive slicing
    card.row_bounds = row_bounds
    return card


def paste_panel(frame, panel, xy, alpha=1.0, slide=(0, 0), rows_visible=None):
    """Paste an RGBA panel onto RGB frame at xy (frame coords).

    alpha        : 0..1 opacity (fade-in); also drives the slide interpolation.
    slide        : (dx, dy) start offset; position = xy + slide*(1-alpha), so
                   animating alpha 0→1 slides the card into place.
    rows_visible : show only header + first N data rows (cropped via the
                   panel's row_bounds). None = whole panel.
    Returns frame modified in place.
    """
    if alpha <= 0:
        return frame
    a = min(1.0, max(0.0, float(alpha)))
    fh, fw = frame.shape[:2]

    if rows_visible is not None:
        bounds = getattr(panel, "row_bounds", None)
        if bounds and len(bounds) >= 2:
            cut_idx = min(int(rows_visible) + 1, len(bounds) - 1)
            panel = panel[:bounds[cut_idx]]

    ph, pw = panel.shape[:2]
    px = int(round(xy[0] + slide[0] * (1.0 - a)))
    py = int(round(xy[1] + slide[1] * (1.0 - a)))

    fx1, fy1 = max(0, px), max(0, py)
    fx2, fy2 = min(fw, px + pw), min(fh, py + ph)
    if fx2 <= fx1 or fy2 <= fy1:
        return frame
    sub = panel[fy1 - py:fy2 - py, fx1 - px:fx2 - px]
    aa = (sub[:, :, 3:4].astype(np.float64) / 255.0) * a
    reg = frame[fy1:fy2, fx1:fx2].astype(np.float64)
    frame[fy1:fy2, fx1:fx2] = np.clip(reg * (1 - aa) + sub[:, :, :3].astype(np.float64) * a,
                                      0, 255).astype(np.uint8)
    return frame
