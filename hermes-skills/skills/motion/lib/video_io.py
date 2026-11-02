"""video_io: read/write video, generate audio."""
import numpy as np
import subprocess
import wave
import struct
import os

def read_video_to_frames(video_path, output_format='rgb24'):
    """Read a video file and return list of numpy arrays (RGB if output_format='rgb24').
    
    Returns list of (frame_array, width, height) tuples.
    """
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-f', 'rawvideo', '-pix_fmt', output_format, '-vf', 'scale=iw:ih',
        '-'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
    
    # Get info via ffprobe
    info = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,pix_fmt,r_frame_rate,nb_frames',
         '-of', 'json', video_path], capture_output=True, text=True
    )
    import json
    d = json.loads(info.stdout)
    s = d['streams'][0]
    w, h = s['width'], s['height']
    # Parse frame rate
    fr = s.get('r_frame_rate', '24/1')
    if '/' in fr:
        fps = int(fr.split('/')[0]) / int(fr.split('/')[1])
    else:
        fps = float(fr)
    nb_frames = int(s.get('nb_frames', -1))
    
    total_bytes = w * h * (3 if output_format == 'rgb24' else 1)
    data = proc.stdout
    
    frames = []
    offset = 0
    frame_idx = 0
    while offset + total_bytes <= len(data):
        frame = np.frombuffer(data[offset:offset + total_bytes], dtype=np.uint8)
        if output_format == 'rgb24':
            frame = frame.reshape(h, w, 3)
        else:
            frame = frame.reshape(h, w)
        frames.append(frame)
        offset += total_bytes
        frame_idx += 1
        if nb_frames > 0 and frame_idx >= nb_frames:
            break
    
    return frames, w, h, fps

def read_video_first_n_frames(video_path, n, output_format='rgb24'):
    """Read first n frames from video."""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-f', 'rawvideo', '-pix_fmt', output_format, '-vf', 'scale=iw:ih',
        '-frames:v', str(n), '-'
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:200]}")
    
    import json
    info = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,pix_fmt',
         '-of', 'json', video_path], capture_output=True, text=True
    )
    d = json.loads(info.stdout)
    s = d['streams'][0]
    w, h = s['width'], s['height']
    
    total_bytes = w * h * (3 if output_format == 'rgb24' else 1)
    data = proc.stdout
    
    frames = []
    offset = 0
    while offset + total_bytes <= len(data) and len(frames) < n:
        frame = np.frombuffer(data[offset:offset + total_bytes], dtype=np.uint8)
        if output_format == 'rgb24':
            frame = frame.reshape(h, w, 3)
        else:
            frame = frame.reshape(h, w)
        frames.append(frame)
        offset += total_bytes
    
    return frames, w, h

def write_frames_to_video(frames, output_path, width, height, fps=24, pix_fmt='yuv420p',
                          crf=18, preset='fast'):
    """Write list of RGB frames to MP4 via ffmpeg stdin."""
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'libx264', '-pix_fmt', pix_fmt,
        '-preset', preset, '-crf', str(crf),
        '-movflags', '+faststart', output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed with code {proc.returncode}")
    return output_path

def generate_silence_wav(output_path, duration=5.0, sample_rate=48000, channels=2):
    """Generate silence WAV file."""
    ns = int(sample_rate * duration)
    audio = np.zeros(ns, dtype=np.float32)
    _write_wav(audio, output_path, sample_rate, channels)
    return output_path

def generate_beep(audio, start_sec, freq, duration, volume=0.3, sample_rate=48000):
    """Add a beep sound to audio array (in-place)."""
    ss = int(start_sec * sample_rate)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    beep = volume * np.sin(2 * np.pi * freq * t) * np.exp(-t * 30) * (1 - np.exp(-t * 200))
    end = min(ss + n, len(audio))
    audio[ss:end] += beep[:end - ss]

def generate_swoosh(audio, start_sec, duration=0.5, freq_start=200, freq_end=2000,
                    volume=0.15, sample_rate=48000):
    """Add a swoosh/chirp sound to audio array (in-place)."""
    ss = int(start_sec * sample_rate)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t * 5) * (1 - np.exp(-t * 20))
    chirp = np.sin(2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * n / sample_rate)))
    end = min(ss + n, len(audio))
    audio[ss:end] += envelope[:end - ss] * chirp[:end - ss] * volume

def generate_tick(audio, start_sec, freq=1200, duration=0.03, volume=0.1,
                  sample_rate=48000):
    """Add a short tick sound to audio array (in-place)."""
    ss = int(start_sec * sample_rate)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t * 80)
    tick = volume * np.sin(2 * np.pi * freq * t) * envelope
    end = min(ss + n, len(audio))
    audio[ss:end] += tick[:end - ss]

def normalize_audio(audio, peak=0.8):
    """Normalize audio to given peak level (in-place)."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio *= peak / max_val

def _write_wav(audio, output_path, sample_rate=48000, channels=2):
    """Write numpy audio array to WAV file."""
    samples = (audio * 32767).astype(np.int16)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        if channels == 2:
            left = samples
            right = samples
            wf.writeframes(np.stack([left, right], axis=1).flatten().tobytes())
        else:
            wf.writeframes(samples.tobytes())
