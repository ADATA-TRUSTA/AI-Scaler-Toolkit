"""segment: per-segment render + audio mux, and final assembly (TOOLKIT_V2 §7).

Every segment is 1280x720 @ 24fps; render_segment defaults to exactly 5.0 s
(120 frames) with a matching 5.0 s / 48 kHz stereo WAV, H.264 yuv420p.
"""
import os
import subprocess
import numpy as np

try:  # package-relative first (import toolkit.segment)
    from .video_io import write_frames_to_video, _write_wav
    from .verify import verify_output_specs
except ImportError:  # direct execution / run from inside toolkit/
    from video_io import write_frames_to_video, _write_wav
    from verify import verify_output_specs

SR = 48000


def synth_sound(audio, t, kind, volume=1.0, sample_rate=SR, **kw):
    """Add one UI sound at time t into float mono `audio` (in place)."""
    try:
        from .video_io import generate_beep, generate_swoosh, generate_tick
    except ImportError:
        from video_io import generate_beep, generate_swoosh, generate_tick
    if kind == 'beep':
        generate_beep(audio, t, kw.get('freq', 880), kw.get('duration', 0.12),
                      volume=kw.get('volume', 0.3) * volume, sample_rate=sample_rate)
    elif kind == 'tick':
        generate_tick(audio, t, kw.get('freq', 1200), kw.get('duration', 0.03),
                      volume=kw.get('volume', 0.12) * volume, sample_rate=sample_rate)
    elif kind == 'swoosh':
        generate_swoosh(audio, t, duration=kw.get('duration', 0.45),
                        freq_start=kw.get('freq_start', 300), freq_end=kw.get('freq_end', 2200),
                        volume=kw.get('volume', 0.16) * volume, sample_rate=sample_rate)
    elif kind == 'chime':
        ss = int(t * sample_rate)
        n = int(0.5 * sample_rate)
        tt = np.arange(n) / sample_rate
        sig = (np.sin(2 * np.pi * kw.get('freq', 1320) * tt) * 0.6 +
               np.sin(2 * np.pi * kw.get('freq', 1320) * 1.5 * tt) * 0.3)
        env = np.exp(-tt * 7) * (1 - np.exp(-tt * 120))
        end = min(ss + n, len(audio))
        audio[ss:end] += sig[:end - ss] * env[:end - ss] * kw.get('volume', 0.25) * volume
    else:
        raise ValueError(f"unknown sound kind: {kind}")
    return audio


def render_segment(frame_fn, out_stem, seconds=5.0, fps=24, sounds=None):
    """Render frames via frame_fn(t)->RGB 1280x720 and write mp4+wav+muxed mp4.

    Produces:
      out_stem.mp4 — H.264 yuv420p, exactly round(seconds*fps) frames, WITH
                     the audio track muxed in (spec-verified on completion).
      out_stem.wav — exactly `seconds` of 48 kHz stereo silence+sounds.
    sounds: [(t, kind, kwargs), ...] with kind in beep/tick/swoosh/chime.
    Returns verify_output_specs() dict extended with frame/wav facts; raises on
    any spec violation.
    """
    n = int(round(seconds * fps))
    frames = []
    for i in range(n):
        f = np.asarray(frame_fn(i / float(fps)))
        if f.shape[0] != 720 or f.shape[1] != 1280:
            raise ValueError(f"frame {i} is {f.shape}, expected (720,1280,3)")
        frames.append(np.ascontiguousarray(f[:, :, :3], dtype=np.uint8))

    # audio first (exact length)
    nsamp = int(round(seconds * SR))
    mono = np.zeros(nsamp, dtype=np.float64)
    for snd in sounds or []:
        t, kind = snd[0], snd[1]
        kw = snd[2] if len(snd) > 2 else {}
        synth_sound(mono, float(t), kind, **kw)
    peak = np.abs(mono).max()
    if peak > 0.9:
        mono *= 0.9 / peak
    wav_path = out_stem + ".wav"
    _write_wav(mono.astype(np.float32), wav_path, SR, 2)

    tmp_mp4 = out_stem + ".silent.mp4"
    write_frames_to_video(frames, tmp_mp4, 1280, 720, fps=fps)
    mp4_path = out_stem + ".mp4"
    cmd = ['ffmpeg', '-y', '-i', tmp_mp4, '-i', wav_path,
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k',
           '-shortest', '-movflags', '+faststart', mp4_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("mux failed: " + r.stderr.decode(errors='replace')[-300:])
    os.remove(tmp_mp4)

    res = verify_output_specs(mp4_path, expected_w=1280, expected_h=720,
                              expected_fps=fps, expected_duration=seconds)
    # exact frame count via ffprobe -count_frames (nb_frames in container can be
    # missing when audio is muxed)
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=nb_read_frames',
                        '-of', 'default=nw=1:nk=1', mp4_path], capture_output=True, text=True)
    counted = int(p.stdout.strip() or 0)
    res['frames_counted'] = counted
    res['frames_ok'] = (counted == n)

    # audio stream present + wav exact length
    p2 = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                         '-show_entries', 'stream=codec_name,sample_rate,channels',
                         '-of', 'default=nw=1:nk=1', mp4_path], capture_output=True, text=True)
    ainfo = p2.stdout.split()
    res['audio'] = {'codec': ainfo[0] if ainfo else None,
                    'sample_rate': int(ainfo[1]) if len(ainfo) > 1 else 0,
                    'channels': int(ainfo[2]) if len(ainfo) > 2 else 0}

    problems = []
    if not (res['width_ok'] and res['height_ok']):
        problems.append(f"size {res['width']}x{res['height']}")
    if not res['fps_ok']:
        problems.append(f"fps {res['fps']}")
    if 'yuv420p' not in str(res.get('pix_fmt', '')):
        problems.append(f"pix_fmt {res.get('pix_fmt')}")
    if not res['frames_ok']:
        problems.append(f"frames {counted} != {n}")
    if abs(res['duration'] - seconds) > 0.15:
        problems.append(f"duration {res['duration']}")
    import wave as _w
    with _w.open(wav_path) as wf:
        wav_dur = wf.getnframes() / float(wf.getframerate())
        res['wav'] = {'seconds': wav_dur, 'rate': wf.getframerate(),
                      'channels': wf.getnchannels()}
        if abs(wav_dur - seconds) > 1.0 / SR or wf.getframerate() != SR or wf.getnchannels() != 2:
            problems.append(f"wav {wf.getnframes()}/{wf.getframerate()}/{wf.getnchannels()}")
    if not (res['audio']['codec'] == 'aac' and res['audio']['sample_rate'] == SR
            and res['audio']['channels'] == 2):
        problems.append(f"audio stream {res['audio']}")
    res['problems'] = problems
    if problems:
        raise RuntimeError("render_segment spec violation: " + "; ".join(problems))
    return res


def _probe_frames(path):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=nb_read_frames',
                        '-of', 'default=nw=1:nk=1', path], capture_output=True, text=True)
    return int((p.stdout or '0').strip() or 0)


def assemble(segments, ending, out_path, expected_frames):
    """Concatenate segment mp4s then append the untouched ending clip.

    Tries a lossless stream-copy concat first (keeps the ending bitstream
    byte-for-byte); if that cannot produce the exact frame count it re-encodes
    to identical params with an output cap of sum(decoded frames per input) —
    without the cap, CFR + a slightly-longer audio tail would pad one extra
    duplicated frame. Verifies total decoded frames == `expected_frames`,
    else raises."""
    import tempfile
    files = list(segments) + [ending]
    for p in files:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    n_inputs = sum(_probe_frames(p) for p in files)

    def _run(copy):
        with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False,
                                         encoding='utf-8') as f:
            for p in files:
                f.write(f"file '{os.path.abspath(p).replace(os.sep, '/')}'\n")
            lst = f.name
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst]
        if copy:
            cmd += ['-c', 'copy', '-movflags', '+faststart']
        else:
            cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast',
                    '-crf', '18', '-r', '24', '-fps_mode', 'cfr',
                    '-frames:v', str(n_inputs),
                    '-af', 'aresample=48000', '-c:a', 'aac', '-b:a', '160k',
                    '-movflags', '+faststart']
        cmd.append(out_path)
        try:
            r = subprocess.run(cmd, capture_output=True)
        finally:
            os.remove(lst)
        return r

    mode = 'copy'
    r = _run(copy=True)
    got = _probe_frames(out_path) if r.returncode == 0 else -1
    if got != expected_frames:
        mode = 're-encode'
        r = _run(copy=False)
        if r.returncode != 0:
            raise RuntimeError("assemble failed: " + r.stderr.decode(errors='replace')[-400:])

    got = _probe_frames(out_path)
    end_frames = _probe_frames(ending)
    seg_total = sum(_probe_frames(s) for s in segments)
    if got != expected_frames:
        raise RuntimeError(f"assemble frame mismatch: {got} != expected {expected_frames} "
                           f"(segments={seg_total} + ending={end_frames})")
    return {'path': out_path, 'frames': got, 'segment_frames': seg_total,
            'ending_frames': end_frames, 'mode': mode}
