"""motion's self-test, on a synthetic screen only (no real screenshots ship here).

    <motion venv python> skills/motion/tests/test_motion.py

Checks the claims the skill rests on: things are found by content (table rows,
a label, a QR code with its content), hidden regions are solid, revealed ones
are the original pixels, and a segment comes out at spec with sound.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "lib"))

import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402
import motion                            # noqa: E402
import locate, reveal                    # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (("  " + str(detail)) if detail else ""))
    if not cond:
        FAIL.append(name)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    info = motion.sample_ui(tmp / "ui.png")
    img = np.array(Image.open(info["image"]).convert("RGB"))

    print("locate")
    tbl = locate.find_table(img)
    check("find_table finds the 12 rows", tbl is not None and len(tbl["rows"]) == 12,
          tbl and len(tbl["rows"]))
    hit = locate.find_text_rect(img, "Total 12 items")
    check("find_text_rect finds the total", hit is not None)
    if hit:
        x1, _, x2, _ = hit["rect"]
        bx1, _, bx2, _ = hit["bbox"]
        check("its rect spans the whole label (not cut at 'ite')", x2 - x1 >= 0.9 * (bx2 - bx1),
              (hit["rect"], hit["bbox"]))
    part = locate.find_text_rect(img, "12 items")
    check("a partial match starts inside the line", part is not None
          and part["rect"][0] > hit["rect"][0] + 20, part and part["rect"])
    codes = locate.find_codes(img)
    check("find_codes reads the QR content", any(c.get("content") == info["qr"] for c in codes),
          codes)

    print("reveal")
    rows = [tuple(r) for r in tbl["rows"]]
    hidden = reveal.hide(img, rows)
    r0 = rows[0]
    patch = hidden[r0[1] + 3:r0[3] - 3, r0[0] + 3:r0[2] - 3]
    check("a hidden row is one solid colour", float(patch.std()) < 2.0, round(float(patch.std()), 2))
    full = reveal.reveal_ltr(hidden, img, r0, 1.0)
    check("a fully revealed row is the original pixels",
          np.array_equal(full[r0[1]:r0[3], r0[0]:r0[2]], img[r0[1]:r0[3], r0[0]:r0[2]]))

    print("segment")
    proj = tmp / "proj"
    subprocess.run([sys.executable, str(HERE.parent / "motion.py"), "new", str(proj),
                    "--image", info["image"]], check=True, capture_output=True)
    r = subprocess.run([sys.executable, str(proj / "segment_01.py")], capture_output=True, text=True)
    check("the template segment renders", r.returncode == 0, r.stderr.strip()[-300:])
    mp4 = proj / "out" / "segment_01.mp4"
    check("1280x720, 120 frames, AAC audio, no problems",
          mp4.is_file() and "'problems': []" in r.stdout and "'frames_counted': 120" in r.stdout
          and "aac" in r.stdout, r.stdout.strip()[-200:])

print("\n%s" % ("all motion checks passed" if not FAIL else "FAILED: " + "; ".join(FAIL)))
sys.exit(1 if FAIL else 0)
