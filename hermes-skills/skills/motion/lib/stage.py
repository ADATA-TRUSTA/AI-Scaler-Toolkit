"""stage: a 16:9 camera window over screenshots/photos.

Sole coordinate system: every input rect/point is in ORIGINAL IMAGE pixels;
conversion to frame pixels happens only inside render()/Mapper.

Rule (METHOD 4.6): highlights/outlines are always drawn on an original-size
layer first, then the layer goes through Stage.render() so zooms follow content.
"""
import numpy as np
from PIL import Image, ImageFilter


def _smoothstep(t):
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def ease_box(box_a, box_b, t):
    """Smooth (smoothstep) interpolation between two view boxes (x1,y1,x2,y2).

    t=0 -> box_a, t=1 -> box_b. Use for push-in / pull-out moves.
    """
    e = _smoothstep(t)
    return tuple(float(a) + (float(b) - float(a)) * e for a, b in zip(box_a, box_b))


class Mapper:
    """Maps ORIGINAL-image coordinates to FRAME coordinates for one view box."""

    def __init__(self, box, fw=1280, fh=720):
        self.box = tuple(float(v) for v in box)
        self.fw, self.fh = fw, fh
        self.sx = fw / (self.box[2] - self.box[0])
        self.sy = fh / (self.box[3] - self.box[1])

    def pt(self, x, y):
        """(orig_x, orig_y) -> (frame_x, frame_y) as ints."""
        return (int(round((x - self.box[0]) * self.sx)),
                int(round((y - self.box[1]) * self.sy)))

    def rect(self, r):
        """(x1,y1,x2,y2) in orig coords -> same tuple in frame coords."""
        a = self.pt(r[0], r[1])
        b = self.pt(r[2], r[3])
        return (a[0], a[1], b[0], b[1])


class Stage:
    """A 16:9 window over one screenshot/photo.

    img_path_or_array : path or RGB ndarray (H, W, 3)
    trim_top_white    : auto-detect and ignore the pure-white strip on top of
                        screenshots when computing full_box() content bounds
                        (input coordinates stay in ORIGINAL image space).
    bg                : background color for letterbox bands; default = median
                        color of the window edge ring ("視窗邊緣底色").
    """

    FRAME_W, FRAME_H = 1280, 720

    def __init__(self, img_path_or_array, trim_top_white=True, bg=None):
        if isinstance(img_path_or_array, str):
            self.img = np.array(Image.open(img_path_or_array).convert("RGB"))
        else:
            arr = np.asarray(img_path_or_array)
            self.img = arr[:, :, :3] if arr.ndim == 3 and arr.shape[2] >= 3 else \
                np.repeat(arr[:, :, None], 3, axis=2)
        self.h, self.w = self.img.shape[:2]

        # content box (after optional top-white trim), still in orig coords
        y0 = 0
        if trim_top_white:
            for i in range(min(400, self.h)):
                if not np.all(self.img[i] > 250):
                    y0 = i
                    break
        # also trim bottom white symmetrically (only used for content bounds)
        self.content_box = (0.0, float(y0), float(self.w), float(self.h))

        if bg is None:
            ring = np.concatenate([
                self.img[:4].reshape(-1, 3), self.img[-4:].reshape(-1, 3),
                self.img[:, :4].reshape(-1, 3), self.img[:, -4:].reshape(-1, 3)])
            # ignore pure white in the ring estimate (screenshot margins)
            nz = ring[np.mean(ring, axis=1) < 249]
            bg = tuple(int(c) for c in np.median(nz if len(nz) else ring, axis=0))
        self.bg = tuple(bg)

    # ---------------- view boxes (orig coords) ----------------

    def full_box(self):
        """Whole image scaled to equal width, vertically centered.

        Wider than 16:9 -> box taller than the image; top/bottom filled with bg.
        Narrower than 16:9 (portrait) -> box wider than the image; left/right
        extended with a blurred background (done in render()).
        """
        cx0, cy0, cx1, cy1 = self.content_box
        cw, chh = cx1 - cx0, cy1 - cy0
        cxc, cyc = (cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0
        if cw / chh >= 16.0 / 9.0:      # wide: fit width, pad top/bottom
            bw = cw
            bh = bw * 9.0 / 16.0
        else:                            # portrait: fit height, pad sides
            bh = chh
            bw = bh * 16.0 / 9.0
        return (cxc - bw / 2.0, cyc - bh / 2.0, cxc + bw / 2.0, cyc + bh / 2.0)

    def box_around(self, rect, margin=0.1):
        """Smallest 16:9 box that contains `rect` padded by margin (fraction of
        the rect's own size), clamped inside the image where possible."""
        x1, y1, x2, y2 = [float(v) for v in rect]
        w, h = max(x2 - x1, 4.0), max(y2 - y1, 4.0)
        x1 -= margin * w; x2 += margin * w
        y1 -= margin * h; y2 += margin * h
        pw, ph = x2 - x1, y2 - y1
        if pw / ph >= 16.0 / 9.0:
            bw, bh = pw, pw * 9.0 / 16.0
        else:
            bh, bw = ph, ph * 16.0 / 9.0
        cxc, cyc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # clamp inside image when the box fits; otherwise center on content
        bx1, by1 = cxc - bw / 2.0, cyc - bh / 2.0
        if bw <= self.w:
            bx1 = min(max(bx1, 0.0), self.w - bw)
        else:
            bx1 = (self.w - bw) / 2.0
        if bh <= self.h:
            by1 = min(max(by1, 0.0), self.h - bh)
        else:
            by1 = (self.h - bh) / 2.0
        return (bx1, by1, bx1 + bw, by1 + bh)

    # ---------------- render ----------------

    def render(self, base_img=None, view_box=None, dim_sides=0.0):
        """Render `base_img` (orig-size; may be a hidden/annotated version)
        through `view_box` -> (frame 1280x720 uint8 RGB, Mapper).

        dim_sides: 0..1 — darken the blurred side-extension by this fraction
        (LESSONS B: portrait photos get blurred sides pressed ~45%)."""
        img = self.img if base_img is None else np.asarray(base_img)[:, :, :3]
        box = self.full_box() if view_box is None else tuple(float(v) for v in view_box)
        fw, fh = self.FRAME_W, self.FRAME_H
        bw, bh = box[2] - box[0], box[3] - box[1]

        # background canvas: flat bg color by default
        frame = np.empty((fh, fw, 3), dtype=np.uint8)
        frame[:, :] = self.bg

        out_left = box[0] < -0.5 or box[2] > self.w + 0.5
        if out_left:  # portrait side extension -> blurred fill of whole image
            scale_cover = max(fw / self.w, fh / self.h)
            cw, chh = int(round(self.w * scale_cover)), int(round(self.h * scale_cover))
            bgp = Image.fromarray(img).resize((cw, chh), Image.LANCZOS)
            blur_arr = np.array(bgp.filter(ImageFilter.GaussianBlur(radius=30)))
            ox, oy = max(0, (cw - fw) // 2), max(0, (chh - fh) // 2)
            tile = blur_arr[oy:oy + fh, ox:ox + fw]
            if dim_sides > 0:
                tile = (tile.astype(np.float64) * (1.0 - float(dim_sides))).astype(np.uint8)
            frame[:tile.shape[0], :tile.shape[1]] = tile

        # sharp content region = view box ∩ image ∩ content (trimmed) bounds
        cb = self.content_box
        ix1, iy1 = max(box[0], 0.0, cb[0]), max(box[1], 0.0, cb[1])
        ix2, iy2 = min(box[2], float(self.w), cb[2]), min(box[3], float(self.h), cb[3])
        if ix2 > ix1 and iy2 > iy1:
            sx, sy = fw / bw, fh / bh
            px1, py1 = int(round((ix1 - box[0]) * sx)), int(round((iy1 - box[1]) * sy))
            px2, py2 = int(round((ix2 - box[0]) * sx)), int(round((iy2 - box[1]) * sy))
            px1, py1 = max(px1, 0), max(py1, 0)
            px2, py2 = min(px2, fw), min(py2, fh)
            if px2 > px1 and py2 > py1:
                crop = img[int(iy1):int(np.ceil(iy2)), int(ix1):int(np.ceil(ix2))]
                rs = Image.fromarray(crop).resize((px2 - px1, py2 - py1), Image.LANCZOS)
                frame[py1:py2, px1:px2] = np.array(rs)

        return frame, Mapper(box, fw, fh)
