"""verify: frame-by-frame difference verification, solid color check, frame collage."""
import numpy as np
from PIL import Image, ImageDraw
import json
import os
import subprocess


def frame_difference(frames_a, frames_b, exclude_regions=None, exclude_color_mask=None):
    """Compare two sets of frames frame-by-frame.
    
    Parameters:
        frames_a: list of numpy arrays (H, W, 3)
        frames_b: list of numpy arrays (H, W, 3)
        exclude_regions: list of (x1, y1, x2, y2) to exclude from comparison
        exclude_color_mask: numpy boolean mask (H, W) for pixels to exclude
    
    Returns:
        List of dicts with per-frame stats.
    """
    results = []
    
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        if fa.shape != fb.shape:
            results.append({
                'frame': i,
                'error': f'Shape mismatch: {fa.shape} vs {fb.shape}',
                'diff_max': float('inf'),
                'diff_mean': float('inf'),
                'diff_mean_excluded': float('inf')
            })
            continue
        
        diff = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
        diff_max = float(diff.max())
        diff_mean = float(diff.mean())
        
        # Apply exclusion mask
        mask = np.zeros((fa.shape[0], fa.shape[1]), dtype=bool)
        
        # Exclude regions
        if exclude_regions:
            for x1, y1, x2, y2 in exclude_regions:
                mask[y1:y2, x1:x2] = True
        
        # Exclude color mask
        if exclude_color_mask is not None:
            mask = mask | exclude_color_mask
        
        if mask.any():
            diff_excluded = diff.copy()
            diff_excluded[mask] = 0
            diff_mean_excluded = float(diff_excluded.mean())
            diff_max_excluded = float(diff_excluded.max())
        else:
            diff_mean_excluded = diff_mean
            diff_max_excluded = diff_max
        
        results.append({
            'frame': i,
            'diff_max': diff_max,
            'diff_mean': diff_mean,
            'diff_max_excluded': diff_max_excluded,
            'diff_mean_excluded': diff_mean_excluded
        })
    
    return results


def check_solid_color(frame, x1, y1, x2, y2, tolerance=5):
    """Check if a region is solid color (all pixels within tolerance of each other).
    
    Parameters:
        frame: numpy array (H, W, 3)
        x1, y1, x2, y2: region coordinates
        tolerance: maximum allowed difference between any two pixels
    
    Returns:
        Dict with 'is_solid', 'mean_color', 'max_diff', 'min_color', 'max_color'
    """
    region = frame[y1:y2, x1:x2]
    
    if region.size == 0:
        return {'is_solid': False, 'error': 'Empty region'}
    
    mean_color = np.mean(region, axis=(0, 1))
    min_color = np.min(region, axis=(0, 1))
    max_color = np.max(region, axis=(0, 1))
    max_diff = float(np.max(max_color - min_color))
    
    return {
        'is_solid': max_diff <= tolerance,
        'mean_color': mean_color.tolist(),
        'max_diff': max_diff,
        'min_color': min_color.tolist(),
        'max_color': max_color.tolist()
    }


def create_frame_collage(frames, cols=4, max_frames=None):
    """Create a collage image from a list of frames.
    
    Parameters:
        frames: list of numpy arrays (H, W, 3)
        cols: number of columns in the collage
        max_frames: maximum number of frames to include
    
    Returns:
        PIL Image
    """
    if max_frames:
        frames = frames[:max_frames]
    
    if not frames:
        return Image.new('RGB', (100, 100), (0, 0, 0))
    
    # All frames should have same size
    h, w = frames[0].shape[:2]
    n = len(frames)
    rows = (n + cols - 1) // cols
    
    collage_w = w * cols
    collage_h = h * rows
    
    collage = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        collage[row * h:(row + 1) * h, col * w:(col + 1) * w] = frame
    
    return Image.fromarray(collage)


def save_worst_frame_diff(frames_a, frames_b, exclude_regions=None, output_path=None):
    """Find the frame with the largest difference and save it as a diff image.
    
    Returns:
        Dict with stats and path to diff image.
    """
    results = frame_difference(frames_a, frames_b, exclude_regions)
    
    if not results:
        return {'error': 'No frames to compare'}
    
    # Find worst frame by max difference (excluding regions)
    worst = max(results, key=lambda r: r.get('diff_max_excluded', r.get('diff_max', 0)))
    worst_idx = worst['frame']
    
    # Create diff image
    fa = frames_a[worst_idx]
    fb = frames_b[worst_idx]
    diff = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
    
    # Normalize for visualization
    max_val = diff.max()
    if max_val > 0:
        diff_norm = (diff / max_val * 255).astype(np.uint8)
    else:
        diff_norm = diff.astype(np.uint8)
    
    diff_img = Image.fromarray(diff_norm)
    
    if output_path:
        diff_img.save(output_path)
        print(f"  Saved diff image: {output_path}")
    
    return {
        'worst_frame': worst_idx,
        'diff_max': worst['diff_max'],
        'diff_max_excluded': worst['diff_max_excluded'],
        'diff_mean_excluded': worst['diff_mean_excluded'],
        'diff_image_path': output_path
    }


def verify_output_specs(video_path, expected_w=1280, expected_h=720, expected_fps=24, expected_duration=5.0):
    """Verify video output specs using ffprobe.
    
    Returns dict with actual specs.
    """
    import json
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,pix_fmt,r_frame_rate',
        '-show_entries', 'format=duration,nb_frames',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    d = json.loads(result.stdout)
    
    s = d['streams'][0]
    fmt = d.get('format', {})
    
    fr = s.get('r_frame_rate', '24/1')
    if '/' in fr:
        fps = int(fr.split('/')[0]) / int(fr.split('/')[1])
    else:
        fps = float(fr)
    
    return {
        'width': s['width'],
        'height': s['height'],
        'pix_fmt': s.get('pix_fmt', ''),
        'fps': fps,
        'duration': float(fmt.get('duration', 0)),
        'nb_frames': int(fmt.get('nb_frames', 0)),
        'width_ok': s['width'] == expected_w,
        'height_ok': s['height'] == expected_h,
        'fps_ok': abs(fps - expected_fps) < 0.5,
        'duration_ok': abs(float(fmt.get('duration', 0)) - expected_duration) < 0.1
    }
