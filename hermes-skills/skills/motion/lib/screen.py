"""screen: screenshot processing — remove top white border, pad to 16:9, coordinate transforms."""
import numpy as np
from PIL import Image, ImageDraw
import json


def detect_white_border_top(arr, threshold=250):
    """Detect how many top rows are pure white (or near-white).
    
    Returns the y-coordinate where non-white content starts.
    """
    check_rows = min(200, arr.shape[0])
    for i in range(check_rows):
        row = arr[i, :]
        # A row is "white" if all pixels have all channels > threshold
        if not np.all(row > threshold):
            return i
    return arr.shape[0]  # All white


def remove_top_white_border(arr, threshold=250):
    """Remove top white border from screenshot. Returns cropped array."""
    crop_y = detect_white_border_top(arr, threshold)
    return arr[crop_y:, :, :], crop_y


def pad_to_16_9(arr, window_bg_color=(177, 177, 177)):
    """Scale image to 16:9 (1280x720) by width, then pad top/bottom with window background color.
    
    The input image is typically wider than 16:9 (e.g., tool window).
    This scales it to target width, then fills top/bottom to reach 720 height.
    
    Returns: (frame, scale, offset_y)
        frame: numpy array (720, 1280, 3)
        scale: width ratio (target_w / orig_w)
        offset_y: vertical offset of the scaled image
    """
    target_w, target_h = 1280, 720
    orig_h, orig_w = arr.shape[:2]
    scale = target_w / orig_w
    new_h = int(orig_h * scale)
    
    src = Image.fromarray(arr).resize((target_w, new_h), Image.LANCZOS)
    frame = np.array(src, dtype=np.uint8)
    
    offset_y = (target_h - new_h) // 2
    full_frame = np.full((target_h, target_w, 3), window_bg_color, dtype=np.uint8)
    full_frame[offset_y:offset_y + new_h, :] = frame
    
    return full_frame, scale, offset_y


def extract_window_bg_color(arr, region=None):
    """Estimate window background color from edges or specified region.
    
    If region is None, samples from the bottom 10% of the image.
    Returns RGB tuple.
    """
    if region is not None:
        r = arr[region[1]:region[3], region[0]:region[2]]
    else:
        # Use bottom rows (usually empty space = window bg)
        h = arr.shape[0]
        r = arr[int(h * 0.85):, :]
    
    if r.size == 0:
        return (177, 177, 177)
    
    # Use median color
    median = np.median(r.reshape(-1, 3), axis=0)
    return tuple(int(c) for c in median)


def screen_coords_to_screen(sx, sy, scale, offset_y, sx1=0, sy1=0, sx2=None, sy2=None):
    """Convert original screenshot coordinates to 16:9 screen coordinates.
    
    Parameters:
        sx, sy: original screenshot coordinates
        scale: width scale factor (from pad_to_16_9)
        offset_y: vertical offset (from pad_to_16_9)
        sx1, sy1: top-left of zoom region in original coords (for zoomed-in view)
        sx2, sy2: bottom-right of zoom region in original coords
    
    Returns:
        (tx, ty) in 1280x720 screen coordinates
    """
    tx = int(sx * scale)
    if sx2 is not None:
        # In zoom mode: relative to zoom region
        tx = int((sx - sx1) / (sx2 - sx1) * 1280)
    return tx


def screen_coords_to_screen_rect(x1, y1, x2, y2, scale, offset_y, zoom_params=None):
    """Convert a rectangle from original screenshot coords to screen coords.
    
    Parameters:
        x1, y1, x2, y2: rectangle in original screenshot coordinates
        scale: width scale factor
        offset_y: vertical offset from padding
        zoom_params: dict with keys 'sx1', 'sy1', 'sx2', 'sy2' for zoom region
    
    Returns:
        (tx1, ty1, tx2, ty2) in 1280x720 screen coordinates
    """
    if zoom_params is not None:
        zp = zoom_params
        tx1 = int((x1 - zp['sx1']) / (zp['sx2'] - zp['sx1']) * 1280)
        ty1 = int((y1 - zp['sy1']) / (zp['sy2'] - zp['sy1']) * 720)
        tx2 = int((x2 - zp['sx1']) / (zp['sx2'] - zp['sx1']) * 1280)
        ty2 = int((y2 - zp['sy1']) / (zp['sy2'] - zp['sy1']) * 720)
    else:
        tx1 = int(x1 * scale)
        ty1 = int(y1 * scale) + offset_y
        tx2 = int(x2 * scale)
        ty2 = int(y2 * scale) + offset_y
    
    return tx1, ty1, tx2, ty2


def zoom_transform(orig_arr, zoom_cx, zoom_cy, zoom_scale, target_w=1280, target_h=720):
    """Extract and scale a zoomed region from original screenshot.
    
    Returns numpy array of shape (target_h, target_w, 3).
    """
    orig_h, orig_w = orig_arr.shape[:2]
    
    hw = int((target_w / 2) / zoom_scale)
    hh = int((target_h / 2) / zoom_scale)
    
    sx1 = max(0, int(zoom_cx - hw))
    sy1 = max(0, int(zoom_cy - hh))
    sx2 = min(orig_w, int(zoom_cx + hw))
    sy2 = min(orig_h, int(zoom_cy + hh))
    
    region = orig_arr[sy1:sy2, sx1:sx2]
    scaled = np.array(Image.fromarray(region).resize((target_w, target_h), Image.LANCZOS))
    
    return scaled, sx1, sy1, sx2, sy2


def zoom_to_screen_coords(orig_x, orig_y, sx1, sy1, sx2, sy2, target_w=1280, target_h=720):
    """Convert original screenshot coords to screen coords after zoom.
    
    Parameters:
        orig_x, orig_y: coordinates in original screenshot
        sx1, sy1, sx2, sy2: zoom region in original screenshot coords
        target_w, target_h: output dimensions
    
    Returns:
        (tx, ty) in screen coordinates
    """
    tx = int((orig_x - sx1) / (sx2 - sx1) * target_w)
    ty = int((orig_y - sy1) / (sy2 - sy1) * target_h)
    return tx, ty


def draw_border_on_screen(frame, x1, y1, x2, y2, color=(30, 144, 255), thickness=3):
    """Draw a colored border rectangle on a screen frame."""
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
