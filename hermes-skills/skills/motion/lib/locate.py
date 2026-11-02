"""locate: OCR text box finding, input field dark character detection, 2D barcode detection, table row/column detection."""
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import zxingcpp
import json

_OCR = None


def _get_ocr():
    """Shared RapidOCR instance (model load is expensive; keep one)."""
    global _OCR
    if _OCR is None:
        _OCR = RapidOCR()
    return _OCR


def ocr_find_text(arr, target_text, similarity_threshold=0.9,
                  crop_offset=(0, 0)):
    """Run OCR and find text box whose content matches target_text.

    Parameters:
        arr: numpy array (H, W, 3) RGB image
        target_text: string to find
        similarity_threshold: minimum RapidOCR confidence score
        crop_offset: (offset_x, offset_y) — adds this to every returned
            coordinate so that results are always in the *original screenshot*
            coordinate system.

    Returns:
        List of dicts: [{'text': str, 'score': float, 'bbox': [x1,y1,x2,y2], 'points': [[x,y],...]}]
        All coordinates are in the original screenshot coordinate system.
    """
    ocr = _get_ocr()
    results, _ = ocr(arr)

    found = []
    for item in results:
        box, text, score = item
        if score < similarity_threshold:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        # Add crop offset so bbox is always in original screenshot coords
        x0 += crop_offset[0]
        y0 += crop_offset[1]
        x1 += crop_offset[0]
        y1 += crop_offset[1]
        found.append({
            'text': text,
            'score': float(score),
            'bbox': [x0, y0, x1, y1],
            'points': [[int(p[0]) + crop_offset[0], int(p[1]) + crop_offset[1]] for p in box]
        })

    # Filter by exact match or contains
    if target_text:
        found = [f for f in found if target_text in f['text'] or f['text'] in target_text]

    return found


def ocr_find_all(arr, min_score=0.5):
    """Run OCR and return all text detections.
    
    Returns list of dicts with text, score, bbox, points.
    """
    ocr = _get_ocr()
    results, _ = ocr(arr)
    
    found = []
    for item in results:
        box, text, score = item
        if score < min_score:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        found.append({
            'text': text,
            'score': float(score),
            'bbox': [x0, y0, x1, y1],
            'points': [[int(p[0]), int(p[1])] for p in box]
        })
    
    return found


def find_dark_chars_in_field(arr, field_bbox, min_score=0.5):
    """Find dark characters inside an input field region.

    Parameters:
        arr: numpy array (H, W, 3) RGB image
        field_bbox: [x1, y1, x2, y2] bounding box of input field
        min_score: minimum confidence for dark pixel detection

    Returns:
        Dict with 'bbox': [x1,y1,x2,y2] of the dark character(s), or None if no dark pixels found.
        Also returns 'dark_mask' for debugging.
        The returned bbox is always in the *original screenshot* coordinate system.
        The height will be >= 8 px for real characters (anti-aliased gray strokes included).
    """
    x1, y1, x2, y2 = field_bbox
    # Ensure valid region
    if x2 <= x1 or y2 <= y1:
        return None

    field_region = arr[y1:y2, x1:x2]

    # Find dark pixels.  Use a lower threshold to catch anti-aliased
    # (gray) strokes — characters on white inputs are usually ~gray, not pure black.
    gray = np.mean(field_region, axis=2)
    dark_mask = gray < 150

    if np.sum(dark_mask) == 0:
        return None

    # Find bounding box of dark pixels
    dark_rows = np.where(np.any(dark_mask, axis=1))[0]
    dark_cols = np.where(np.any(dark_mask, axis=0))[0]

    if len(dark_rows) == 0 or len(dark_cols) == 0:
        return None

    local_x1, local_y1 = int(dark_cols[0]), int(dark_rows[0])
    local_x2, local_y2 = int(dark_cols[-1]), int(dark_rows[-1])

    # Return in original image coordinates
    return {
        'bbox': [x1 + local_x1, y1 + local_y1, x1 + local_x2, y1 + local_y2],
        'dark_mask': dark_mask,
        'pixel_count': int(np.sum(dark_mask))
    }


def detect_2d_barcode(arr, scale_factor=1.0):
    """Detect 2D barcodes (DataMatrix, QR) in image.
    
    Parameters:
        arr: numpy array (H, W, 3) RGB image
        scale_factor: scale down before detection (e.g., 0.5 for half-size)
    
    Returns:
        List of dicts: [{'type': str, 'content': str, 'bbox': [x1,y1,x2,y2]}]
    """
    if scale_factor < 1.0:
        h, w = arr.shape[:2]
        nh, nw = int(h * scale_factor), int(w * scale_factor)
        pil_img = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    else:
        pil_img = Image.fromarray(arr)
    
    # Convert to BGR for zxing-cpp (it expects numpy BGR array)
    bgr = np.array(pil_img.convert('RGB'))[:, :, ::-1]
    
    # Try QR first, then DataMatrix
    results = []
    
    # QR code detection
    qr_result = zxingcpp.read_barcodes(bgr, zxingcpp.BarcodeFormat.QRCode)
    for barcode in qr_result:
        if not barcode.valid:
            continue
        pos = barcode.position
        content = barcode.text or ''
        # Compute bounding box from 4 corners (handles rotation)
        pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
        xs = [int(p.x) for p in pts]
        ys = [int(p.y) for p in pts]
        results.append({
            'type': 'QR_CODE',
            'content': content,
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'points': [
                [int(pos.top_left.x), int(pos.top_left.y)],
                [int(pos.top_right.x), int(pos.top_right.y)],
                [int(pos.bottom_right.x), int(pos.bottom_right.y)],
                [int(pos.bottom_left.x), int(pos.bottom_left.y)]
            ]
        })
    
    # DataMatrix detection
    dm_result = zxingcpp.read_barcodes(bgr, zxingcpp.BarcodeFormat.DataMatrix)
    for barcode in dm_result:
        if not barcode.valid:
            continue
        pos = barcode.position
        content = barcode.text or ''
        # Compute bounding box from 4 corners (handles rotation)
        pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
        xs = [int(p.x) for p in pts]
        ys = [int(p.y) for p in pts]
        results.append({
            'type': 'DATA_MATRIX',
            'content': content,
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'points': [
                [int(pos.top_left.x), int(pos.top_left.y)],
                [int(pos.top_right.x), int(pos.top_right.y)],
                [int(pos.bottom_right.x), int(pos.bottom_right.y)],
                [int(pos.bottom_left.x), int(pos.bottom_left.y)]
            ]
        })
    
    # Scale coordinates back if we scaled down
    if scale_factor < 1.0:
        for r in results:
            r['bbox'] = [int(r['bbox'][0] / scale_factor), int(r['bbox'][1] / scale_factor),
                         int(r['bbox'][2] / scale_factor), int(r['bbox'][3] / scale_factor)]
            if 'points' in r:
                r['points'] = [[int(p[0] / scale_factor), int(p[1] / scale_factor)] for p in r['points']]
    
    return results


def detect_table_rows(arr, y_range=None, min_text_height=10):
    """Detect table rows by finding horizontal text lines.
    
    Parameters:
        arr: numpy array (H, W, 3) RGB image
        y_range: optional (y_start, y_end) to restrict search area
        min_text_height: minimum height for a text line to be considered a row
    
    Returns:
        List of dicts: [{'row_idx': int, 'y1': int, 'y2': int, 'text': str}]
    """
    search_area = arr
    y_offset = 0
    if y_range:
        y_offset = y_range[0]
        search_area = arr[y_range[0]:y_range[1], :]
    
    ocr = _get_ocr()
    results, _ = ocr(search_area)
    
    # Group text lines by y position
    lines = []
    for item in results:
        box, text, score = item
        if score < 0.5:
            continue
        ys = [p[1] for p in box]
        text_h = max(ys) - min(ys)
        if text_h < min_text_height:
            continue
        lines.append({
            'y1': int(min(ys)) + y_offset,
            'y2': int(max(ys)) + y_offset,
            'y_center': int(np.mean(ys)) + y_offset,
            'text': text,
            'score': float(score)
        })
    
    if not lines:
        return []
    
    # Group into rows by y proximity
    lines.sort(key=lambda x: x['y1'])
    rows = []
    current_row = [lines[0]]
    row_y_threshold = 15  # pixels
    
    for line in lines[1:]:
        if abs(line['y1'] - current_row[0]['y1']) < row_y_threshold:
            current_row.append(line)
        else:
            rows.append({
                'row_idx': len(rows),
                'y1': min(l['y1'] for l in current_row),
                'y2': max(l['y2'] for l in current_row),
                'texts': [l['text'] for l in current_row]
            })
            current_row = [line]
    
    if current_row:
        rows.append({
            'row_idx': len(rows),
            'y1': min(l['y1'] for l in current_row),
            'y2': max(l['y2'] for l in current_row),
            'texts': [l['text'] for l in current_row]
        })
    
    return rows


def detect_table_columns(arr, row_y_range, col_threshold=5):
    """Detect table columns by finding vertical separators.
    
    Parameters:
        arr: numpy array (H, W, 3) RGB image
        row_y_range: (y1, y2) to restrict search area
        col_threshold: pixel threshold for vertical line detection
    
    Returns:
        List of column x-positions.
    """
    y1, y2 = row_y_range
    row_area = arr[y1:y2, :]
    
    # Convert to grayscale
    gray = np.mean(row_area, axis=2)
    
    # Find dark vertical lines (columns boundaries)
    col_darkness = np.mean(gray, axis=0)
    
    # Find columns where darkness changes significantly
    thresholds = []
    for x in range(1, len(col_darkness) - 1):
        if col_darkness[x] < 150 and col_darkness[x-1] > 200:
            thresholds.append(x)
    
    # Group nearby thresholds
    cols = []
    if thresholds:
        current_group = [thresholds[0]]
        for t in thresholds[1:]:
            if t - current_group[-1] < col_threshold:
                current_group.append(t)
            else:
                cols.append(int(np.mean(current_group)))
                current_group = [t]
        cols.append(int(np.mean(current_group)))

    return cols


# =====================================================================
# v2 additions (TOOLKIT_V2.md §3) — content-driven component finding.
# All coordinates are ORIGINAL IMAGE pixels.
# =====================================================================
import unicodedata


def _gray(arr):
    a = np.asarray(arr)[:, :, :3].astype(np.float64)
    return a.mean(axis=2)


def _norm_text(s):
    """Normalize full/half-width differences and remove all whitespace."""
    s = unicodedata.normalize('NFKC', str(s))
    return ''.join(s.split())


def find_text_rect(arr, text, min_score=0.4):
    """Find a text occurrence allowing full/half-width + spacing differences.

    Returns {'text','bbox'(ocr line box),'rect'(tight ink bbox of just the
    matched prefix, orig coords)} or None.
    """
    target = _norm_text(text)
    for r in ocr_find_all(arr, min_score=min_score):
        got = _norm_text(r['text'])
        if target not in got:
            continue
        x1, y1, x2, y2 = r['bbox']
        g = _gray(arr)
        ink = g[max(0, y1 - 2):y2 + 3, max(0, x1 - 2):x2 + 3] < 165
        rows = np.where(np.any(ink, axis=1))[0]
        ys_lo, ys_hi = (int(rows[0]) + y1 - 2, int(rows[-1]) + y1 - 1) if len(rows) else (y1, y2)
        # Horizontal span of the match: the line's measured ink extent, cut in
        # proportion to where the match sits in the line (CJK glyphs ~1 em,
        # NFKC half-width ones ~0.55 em). Estimating the width from the font
        # height alone, as before, fit CJK UIs but cut "Total 12 items" off at
        # "ite" in a bold Latin font.
        cols = np.where(np.any(ink, axis=0))[0]
        ix1, ix2 = ((int(cols[0]) + x1 - 2, int(cols[-1]) + x1 - 1) if len(cols)
                    else (x1, x2))
        def _w(s):
            return sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in s)
        start = got.index(target)
        total = _w(got) or 1.0
        rx1 = ix1 + (ix2 - ix1) * _w(got[:start]) / total
        rx2 = ix1 + (ix2 - ix1) * _w(got[:start + len(target)]) / total
        return {'text': r['text'], 'bbox': [x1, y1, x2, y2],
                'rect': (int(rx1), ys_lo, int(np.ceil(rx2)) + 1, ys_hi),
                'score': r['score']}
    return None


def find_input_boxes(arr, min_w=40, min_h=8):
    """Find white input boxes: near-white rectangle with a continuous border.

    Returns list of {'bbox': outer (incl. border), 'interior', 'bordered': bool}
    sorted top-to-bottom. Uses connected components on the white mask
    (cv2 if available, pure-numpy BFS fallback).
    """
    g = _gray(arr)
    mask = (g > 247).astype(np.uint8)
    h, w = mask.shape
    boxes = []

    try:
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        comps = [(stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]) for i in range(1, n)]
    except Exception:  # pragma: no cover - fallback labeling
        lab = np.zeros((h, w), dtype=np.int32)
        cur = 0
        from collections import deque
        for yy in range(h):
            for xx in range(w):
                if mask[yy, xx] and lab[yy, xx] == 0:
                    cur += 1
                    q = deque([(yy, xx)])
                    lab[yy, xx] = cur
                    while q:
                        cy, cx = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                q.append((ny, nx))
        comps = []
        for i in range(1, cur + 1):
            ys, xs = np.where(lab == i)
            if len(ys):
                comps.append((int(xs.min()), int(ys.min()),
                              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))

    # merge horizontally adjacent components with the same vertical span:
    # text glyphs inside an input box split its white interior into pieces.
    comps.sort(key=lambda c: (c[1], c[0]))
    merged = []
    for c in comps:
        placed = False
        for i, m in enumerate(merged):
            ov = min(m[1] + m[3], c[1] + c[3]) - max(m[1], c[1])
            if ov >= 0.8 * min(m[3], c[3]):
                gap = c[0] - (m[0] + m[2])
                if -min(m[2], c[2]) <= gap <= 14:
                    x1, y1 = min(m[0], c[0]), min(m[1], c[1])
                    x2, y2 = max(m[0] + m[2], c[0] + c[2]), max(m[1] + m[3], c[1] + c[3])
                    merged[i] = (x1, y1, x2 - x1, y2 - y1)
                    placed = True
                    break
        if not placed:
            merged.append(c)
    comps = merged

    for cx0, cy0, cw, chh in comps:
        if cw < min_w or chh < min_h or cw * chh > 0.5 * w * h:
            continue
        # grow outward to include the border (dark/gray ring), max 4 px/side
        bx1, by1, bx2, by2 = cx0, cy0, cx0 + cw, cy0 + chh
        for _ in range(4):
            if by1 == 0 or np.mean(g[by1 - 1, cx0:cx0 + cw]) >= 247:
                break
            by1 -= 1
        for _ in range(4):
            if by2 >= h or np.mean(g[by2, cx0:cx0 + cw]) >= 247:
                break
            by2 += 1
        for _ in range(4):
            if bx1 == 0 or np.mean(g[cy0:cy0 + chh, bx1 - 1]) >= 247:
                break
            bx1 -= 1
        for _ in range(4):
            if bx2 >= w or np.mean(g[cy0:cy0 + chh, bx2]) >= 247:
                break
            bx2 += 1
        # border continuity: perimeter ring pixels that are not white
        ring_top = g[max(0, by1), bx1:bx2] if bx2 > bx1 else np.array([255.0])
        ring_bot = g[min(h - 1, by2 - 1), bx1:bx2] if bx2 > bx1 else np.array([255.0])
        ring_l = g[by1:by2, max(0, bx1)] if by2 > by1 else np.array([255.0])
        ring_r = g[by1:by2, min(w - 1, bx2 - 1)] if by2 > by1 else np.array([255.0])
        per = np.concatenate([ring_top, ring_bot, ring_l, ring_r])
        bordered = bool(np.mean(per < 247) > 0.85 and (per.max() - per.min()) >= 0)
        boxes.append({'bbox': [bx1, by1, bx2, by2],
                      'interior': [cx0, cy0, cx0 + cw, cy0 + chh],
                      'bordered': bordered})
    boxes.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
    return boxes


def find_field_right_of(arr, label_text):
    """OCR the label, then return the nearest input box on the same row to its
    right (white rectangle with continuous border). Orig-image coords.

    Returns {'label': {...ocr}, 'box': [x1,y1,x2,y2]} or None.
    """
    target = _norm_text(label_text)
    labels = [r for r in ocr_find_all(arr, min_score=0.4) if target and target in _norm_text(r['text'])]
    if not labels:
        return None
    boxes = [b for b in find_input_boxes(arr) if b['bordered']]
    best = None
    for lab in labels:
        lx1, ly1, lx2, ly2 = lab['bbox']
        lcy = (ly1 + ly2) / 2.0
        cands = [b for b in boxes
                 if b['bbox'][0] >= lx2 - 8 and b['bbox'][1] - 6 <= lcy <= b['bbox'][3] + 6]
        if not cands:  # allow label whose row-center is within box height range
            cands = [b for b in boxes if b['bbox'][0] >= lx2 - 8 and
                     not (b['bbox'][3] < ly1 or b['bbox'][1] > ly2)]
        if cands:
            pick = min(cands, key=lambda b: b['bbox'][0])
            if best is None or pick['bbox'][0] < best['box'][0]:
                best = {'label': lab, 'box': pick['bbox']}
    return best


def _strong_hlines(g, x1, x2, y1, y2, thr=205, frac_min=0.9):
    """Rows in [y1,y2) that are dark across >=frac_min of [x1,x2); grouped."""
    seg = g[y1:y2, int(x1):int(x2)]
    frac = np.mean(seg < thr, axis=1)
    ys = [y1 + i for i in range(len(frac)) if frac[i] > frac_min]
    groups = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 3:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [(gp[0], gp[-1]) for gp in groups]


def find_table(arr, min_rows=4):
    """Locate the main data table.

    Returns {'header': (x1,y1,x2,y2), 'rows': [rect per data row incl. a
    clipped last half-row], 'col_x': [column separator x list],
    'white_box': (x1,y1,x2,y2) white background extent, 'pitch': float} —
    all orig-image coords.

    Geometry rules (validated on two real admin-table screenshots):
      * candidate body = between two strong horizontal border lines, taller
        than 200 px, interior mostly white, gray window bg right below the
        bottom line, and a 15–45 px gray header band directly above the top
        line; best candidate = highest white fraction;
      * rows from clustering OCR text-line centers inside the body; pitch =
        median spacing; each row rect is one full pitch clipped to the body;
      * below-the-border clusters count as a half-row only if they span most
        of the table width (stray page-number glyphs do not);
      * col_x = midpoints between consecutive header cell text spans.
    """
    g = _gray(arr)
    h, w = g.shape
    xm1, xm2 = int(w * 0.05), int(w * 0.95)
    lines = _strong_hlines(g, xm1, xm2, 60, h - 4)

    def band_lum(ya, yb):
        ya, yb = max(0, ya), min(h, yb)
        if yb <= ya:
            return None
        rows = g[ya:yb, xm1:xm2].mean(axis=1)
        return float(np.mean((rows >= 195) & (rows <= 242)))

    def white_frac(ya, yb):
        ya, yb = max(0, ya), min(h, yb)
        if yb <= ya:
            return 0.0
        return float(np.mean(g[ya:yb, xm1:xm2] > 248))

    best = None  # (white_frac, top_y, bot_y, header_top)
    for i, (a0, a1) in enumerate(lines):
        top = a1 + 1
        for b0, b1 in lines[i + 1:]:
            bot = b0 - 1
            if bot - top < 200:
                continue
            # gray window bg right below the bottom border line
            y_after = b1 + 3
            if not (0 <= y_after < h) or not (195 <= g[y_after, xm1:xm2].mean() <= 242):
                continue
            # header band directly above top line: previous strong line at
            # least 8 px away (skip double-border/adjacent lines)
            prev = [ln for ln in lines if ln[1] <= a0 - 8]
            hdr_bottom = max(ln[1] for ln in prev) + 1 if prev else 0
            hdr_h = (top - 2) - hdr_bottom
            if not (15 <= hdr_h <= 45) or band_lum(hdr_bottom, top - 2) < 0.6:
                continue
            wf = white_frac(top, bot)
            if wf < 0.7:
                continue
            if best is None or wf > best[0]:
                best = (wf, top, b0, hdr_bottom)
    if best is None:
        return None
    _, top_y, bot_y, header_top = best
    header = (xm1, header_top, xm2, top_y - 1)

    # white background horizontal extent (sample body rows, skip dark bands)
    xs_l, xs_r = w, 0
    for yy in range(top_y + 5, bot_y - 2, max(3, (bot_y - top_y) // 12)):
        roww = g[yy, :] > 248
        idx = np.where(roww)[0]
        if len(idx) and np.mean(g[yy, xm1:xm2]) > 200:
            xs_l, xs_r = min(xs_l, int(idx.min())), max(xs_r, int(idx.max()))
    white_box = (xs_l, top_y - 1, xs_r + 1, bot_y)

    # rows from OCR text-line centers inside the body
    all_texts = ocr_find_all(arr, min_score=0.4)
    in_body = [r for r in all_texts if xs_l - 5 <= r['bbox'][0] <= xs_r + 5]
    texts = [r for r in in_body if top_y < (r['bbox'][1] + r['bbox'][3]) / 2.0 < bot_y]

    centers = sorted((r['bbox'][1] + r['bbox'][3]) / 2.0 for r in texts)
    clusters = []
    for c in centers:
        if clusters and c - clusters[-1][-1] < 14:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    rc = [float(np.mean(cl)) for cl in clusters]

    # last row is a clipped half-row when cut-off glyph ink exists BELOW the
    # bottom border (rows bot_y+3..bot_y+9; the border's own anti-aliasing
    # stays within 2 px, and gray window bg is ~216 > 150 threshold)
    partial_last = False
    if bot_y + 9 < h:
        sub = g[bot_y + 3:bot_y + 10, xs_l + 2:xs_r - 2]
        frac_ink = np.mean(sub < 150, axis=1)
        partial_last = bool(frac_ink.max() > 0.02 and frac_ink.max() < 0.9)

    if len(rc) < min_rows:
        return None
    diffs = sorted(rc[i + 1] - rc[i] for i in range(len(rc) - 1))
    pitch = float(np.median(diffs)) if diffs else 20.0

    rows = []
    for c in rc:
        y1 = max(top_y, int(round(c - pitch / 2)))
        y2 = min(bot_y + 1, int(round(c + pitch / 2)))
        rows.append((xs_l, y1, xs_r, y2))

    # column separators from header cell text spans (midpoints)
    col_x = []
    hdr_texts = [r for r in all_texts
                 if header[1] - 4 <= (r['bbox'][1] + r['bbox'][3]) / 2.0 <= top_y + 2]
    hdr_texts.sort(key=lambda r: r['bbox'][0])
    seen, uniq = set(), []
    for cell in hdr_texts:
        key = _norm_text(cell['text'])
        if key not in seen:
            seen.add(key)
            uniq.append(cell)
    for a, b in zip(uniq, uniq[1:]):
        col_x.append((a['bbox'][2] + b['bbox'][0]) // 2)

    return {'header': header, 'rows': rows, 'col_x': col_x,
            'white_box': white_box, 'pitch': pitch, 'partial_last': partial_last}


def find_codes(arr, auto_scale=True):
    """Detect 2D barcodes (QR + DataMatrix) with contents.

    Large images are downscaled automatically for detection and coordinates
    converted back to original-image pixels. Returns [{'type','content','bbox'}].
    """
    a = np.asarray(arr)[:, :, :3]
    h, w = a.shape[:2]
    scales = [1.0]
    if auto_scale and max(h, w) > 2048:
        base = min(1.0, 2048.0 / max(h, w))
        scales = [round(base * k, 3) for k in (1.0, 0.5)] if base < 1 else [0.5]
    found = []
    for s in scales:
        img = Image.fromarray(a) if s >= 1.0 else \
            Image.fromarray(a).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        bgr = np.array(img)[:, :, ::-1].copy()
        for fmt_name in ('QRCode', 'DataMatrix'):
            fmt = getattr(zxingcpp.BarcodeFormat, fmt_name)
            for bc in zxingcpp.read_barcodes(bgr, fmt):
                if not bc.valid:
                    continue
                pos = bc.position
                pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
                xs = [int(p.x) for p in pts if p.x >= 0] or [0]
                ys = [int(p.y) for p in pts if p.y >= 0] or [0]
                bbox = [int(min(xs) / s), int(min(ys) / s),
                        int(max(xs) / s), int(max(ys) / s)]
                dup = any(f['content'] == (bc.text or '') and
                          abs((f['bbox'][0] + f['bbox'][2]) - (bbox[0] + bbox[2])) < 40 and
                          abs((f['bbox'][1] + f['bbox'][3]) - (bbox[1] + bbox[3])) < 40
                          for f in found)
                if not dup:
                    found.append({'type': fmt_name.upper(), 'content': bc.text or '',
                                  'bbox': bbox})
    return found
