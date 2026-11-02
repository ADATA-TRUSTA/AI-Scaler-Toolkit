"""fx: visual effects — zoom, fade in/out, blur transition, hide-then-show, borders/glow."""
import numpy as np
from PIL import Image, ImageFilter
import math


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * np.clip(t, 0.0, 1.0)


def zoom_effect(frame, zoom_cx, zoom_cy, zoom_scale, target_w=1280, target_h=720):
    """Zoom into a region of the frame.
    
    Parameters:
        frame: numpy array (H, W, 3) RGB
        zoom_cx, zoom_cy: center of zoom in frame coordinates
        zoom_scale: scale factor (> 1 means zoom in)
        target_w, target_h: output dimensions
    
    Returns:
        numpy array (target_h, target_w, 3)
    """
    h, w = frame.shape[:2]
    
    hw = int((target_w / 2) / zoom_scale)
    hh = int((target_h / 2) / zoom_scale)
    
    sx1 = max(0, int(zoom_cx - hw))
    sy1 = max(0, int(zoom_cy - hh))
    sx2 = min(w, int(zoom_cx + hw))
    sy2 = min(h, int(zoom_cy + hh))
    
    region = frame[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return frame.copy()
    
    scaled = np.array(Image.fromarray(region).resize((target_w, target_h), Image.LANCZOS))
    return scaled


def fade_in(frame, alpha, target_h, target_w):
    """Fade in: blend frame with black based on alpha (0=black, 1=full)."""
    black = np.zeros((target_h, target_w, 3), dtype=np.float64)
    result = black * (1 - alpha) + frame.astype(np.float64) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def fade_out(frame, alpha):
    """Fade out: blend frame with color based on alpha (0=full, 1=black)."""
    result = frame.astype(np.float64) * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def fade_to_color(frame, target_color, alpha):
    """Fade frame toward a target color.
    
    Parameters:
        frame: numpy array (H, W, 3)
        target_color: RGB tuple or numpy array
        alpha: 0 = original, 1 = target color
    """
    tc = np.array(target_color, dtype=np.float64).reshape(1, 1, 3)
    result = frame.astype(np.float64) * (1 - alpha) + tc * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def blur_transition(frame, blur_radius):
    """Apply Gaussian blur to frame."""
    pil_img = Image.fromarray(frame)
    if blur_radius > 0:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(pil_img)


def hide_show(canvas, text_bbox, field_bbox, hide_color=(255, 255, 255), inset=2):
    """Hide content by filling with background color, clipped to field interior.
    
    This implements the "hide-then-show" technique from METHOD.md:
    Use the actual background color of the field to cover content,
    clipped to the intersection of text bbox and field interior.
    
    Parameters:
        canvas: numpy array (H, W, 3) to modify in-place
        text_bbox: [x1, y1, x2, y2] text region in canvas coordinates
        field_bbox: [x1, y1, x2, y2] input field bounds
        hide_color: RGB color to use for hiding (default white)
        inset: pixels to inset from field border (default 2, to avoid covering border lines)
    
    Returns:
        The modified canvas (same object, modified in-place)
    """
    # Field interior: inset pixels from each edge
    fi_x1 = field_bbox[0] + inset
    fi_y1 = field_bbox[1] + inset
    fi_x2 = field_bbox[2] - inset
    fi_y2 = field_bbox[3] - inset
    
    # Intersection of text bbox and field interior
    ix1 = max(text_bbox[0], fi_x1)
    iy1 = max(text_bbox[1], fi_y1)
    ix2 = min(text_bbox[2], fi_x2)
    iy2 = min(text_bbox[3], fi_y2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return canvas
    
    canvas[iy1:iy2, ix1:ix2] = hide_color
    return canvas


def draw_outer_border(frame, x1, y1, x2, y2, color=(30, 144, 255), thickness=3):
    """Draw a colored border rectangle on frame."""
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    
    # Top edge
    for y in range(y1, min(y1 + thickness, y2)):
        result[y, x1:x2, :] = c
    # Bottom edge
    for y in range(max(y2 - thickness, y1), y2):
        result[y, x1:x2, :] = c
    # Left edge
    for x in range(x1, min(x1 + thickness, x2)):
        result[y1:y2, x, :] = c
    # Right edge
    for x in range(max(x2 - thickness, x1), x2):
        result[y1:y2, x, :] = c
    
    return np.clip(result, 0, 255).astype(np.uint8)


def blend_rect(frame, rect, color, alpha):
    """Semi-transparent fill of a rectangle ('highlight' style, does not obscure text).

    Parameters:
        frame: numpy array (H, W, 3) RGB, modified in-place
        rect: (x1, y1, x2, y2) in frame coordinates
        color: RGB tuple to blend in
        alpha: 0..1 blend strength

    Returns the modified frame.
    """
    x1, y1, x2, y2 = rect
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1 or alpha <= 0:
        return frame
    c = np.array(color, dtype=np.float64)
    region = frame[y1:y2, x1:x2].astype(np.float64)
    frame[y1:y2, x1:x2] = np.clip(region * (1 - alpha) + c * alpha, 0, 255).astype(np.uint8)
    return frame


def draw_outer_border_alpha(frame, x1, y1, x2, y2, color=(30, 144, 255), thickness=3, alpha=1.0):
    """Draw a colored border rectangle with alpha blending (outline only, never fills)."""
    if alpha <= 0:
        return frame
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)

    def blend(y0, y1_, x0, x1_):
        result[y0:y1_, x0:x1_] = result[y0:y1_, x0:x1_] * (1 - alpha) + c * alpha

    blend(y1, min(y1 + thickness, y2), x1, x2)          # top
    blend(max(y2 - thickness, y1), y2, x1, x2)          # bottom
    blend(y1, y2, x1, min(x1 + thickness, x2))          # left
    blend(y1, y2, max(x2 - thickness, x1), x2)          # right

    frame[:] = np.clip(result, 0, 255).astype(np.uint8)
    return frame


def draw_light_strip(frame, x1, y1, x2, y2, color=(255, 255, 255), alpha=0.3, thickness=5):
    """Draw a semi-transparent light strip/ribbon over a region."""
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    
    # Draw horizontal strip (top and bottom edges of the region)
    for y in range(max(0, y1), min(result.shape[0], y2)):
        if y >= max(0, y2 - thickness) or y < min(result.shape[0], y1 + thickness):
            for x in range(x1, x2):
                if 0 <= x < result.shape[1]:
                    result[y, x] = result[y, x] * (1 - alpha) + c * alpha
    
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_corner_brackets(frame, cx, cy, size, color=(255, 0, 0), thickness=3):
    """Draw corner brackets (L-shapes) around a center point.
    
    Useful for targeting/aiming effects on barcodes or specific regions.
    """
    h, w = frame.shape[:2]
    half = size // 2
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    
    # Top-left corner
    for y in range(cy - half, cy - half + thickness):
        for x in range(cx - half, cx - half + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    for x in range(cx - half, cx - half + thickness):
        for y in range(cy - half, cy - half + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    
    # Top-right corner
    for y in range(cy - half, cy - half + thickness):
        for x in range(cx + half - half, cx + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    for x in range(cx + half - thickness, cx + half):
        for y in range(cy - half, cy - half + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    
    # Bottom-left corner
    for y in range(cy + half - thickness, cy + half):
        for x in range(cx - half, cx - half + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    for x in range(cx - half, cx - half + thickness):
        for y in range(cy + half - half, cy + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    
    # Bottom-right corner
    for y in range(cy + half - thickness, cy + half):
        for x in range(cx + half - half, cx + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    for x in range(cx + half - thickness, cx + half):
        for y in range(cy + half - half, cy + half):
            if 0 <= y < h and 0 <= x < w:
                result[y, x] = c
    
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_cone(frame, sy, ey, cx, width_base=10, width_top=30, color=(255, 0, 0), alpha=0.08):
    """Draw a light cone from sy to ey (vertical), centered at cx.
    
    Narrows at sy (source) and widens at ey (target).
    Low alpha so it doesn't obscure content.
    """
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    
    if ey <= sy:
        return np.clip(result, 0, 255).astype(np.uint8)
    
    for y in range(sy, min(ey, result.shape[0])):
        p = (y - sy) / (ey - sy)
        w = int(width_base + (width_top - width_base) * p)
        for x in range(max(0, cx - w), min(result.shape[1], cx + w)):
            result[y, x] = result[y, x] * (1 - alpha) + c * alpha
    
    return np.clip(result, 0, 255).astype(np.uint8)


def scan_line(frame, y, x1, x2, color=(255, 0, 0), thickness=1):
    """Draw a thin scan line across the frame.
    
    thickness=1 means exactly 1 pixel at y.
    """
    result = frame.astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    
    if 0 <= y < result.shape[0]:
        result[y, x1:x2, :] = c
    
    return np.clip(result, 0, 255).astype(np.uint8)


def animate_typing(frame, char_images, start_x, start_y, char_count, max_zoom=1.0):
    """Animate typing: paste character images one by one.
    
    Parameters:
        frame: numpy array (H, W, 3) to modify
        char_images: list of numpy arrays (each is a character image)
        start_x, start_y: starting position for first character
        char_count: number of characters to show (0 to len(char_images))
        max_zoom: zoom factor for character images
    
    Returns:
        Modified frame
    """
    result = frame.astype(np.float64).copy()
    cx = start_x
    
    for i in range(char_count):
        if i >= len(char_images):
            break
        char = char_images[i]
        ch, cw = char.shape[:2]
        if ch == 0 or cw == 0:
            continue
        if start_y + ch <= result.shape[0] and cx + cw <= result.shape[1]:
            result[start_y:start_y + ch, cx:cx + cw] = char
        cx += cw
    
    return np.clip(result, 0, 255).astype(np.uint8)
