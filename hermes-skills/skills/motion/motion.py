#!/usr/bin/env python
"""motion -- turn screenshots and photos into explainer animation, by code.

The agent writes a short timeline script per 5-second segment on the library in
lib/ (pan and zoom over the original image, hide-then-reveal, outlines and light
bands, panels cut from the image, sound effects). Every letter on screen stays
the original image's pixels: nothing is re-typed in a font and nothing is drawn
by a model, which is what makes the result usable for a product's real UI.

    motion.py check                          dependencies and ffmpeg
    motion.py sample-ui <out.png>            a synthetic admin screen (tests, demos)
    motion.py new <dir> --image <png>        a project with a working 5 s segment script
    motion.py sheet <video.mp4> [out.jpg]    contact sheet of a video, for checking

No GPU and no ComfyUI: it runs beside a render without competing for the card.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
sys.path.insert(0, str(LIB))


def out(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------- check
def cmd_check(_a) -> int:
    res, ok = {"python": sys.executable, "lib": str(LIB)}, True
    for mod in ("numpy", "PIL", "cv2", "rapidocr_onnxruntime", "zxingcpp"):
        try:
            m = __import__(mod)
            res[mod] = getattr(m, "__version__", "ok")
        except Exception as e:                       # noqa: BLE001
            res[mod] = "MISSING: %s" % e
            ok = False
    for tool in ("ffmpeg", "ffprobe"):
        res[tool] = shutil.which(tool) or "MISSING"
        ok = ok and res[tool] != "MISSING"
    res["ok"] = ok
    out(res)
    return 0 if ok else 1


# --------------------------------------------------------------- sample-ui
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
            Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts")]
    for d in dirs:
        for n in names:
            p = d / n
            if p.is_file():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


ROWS = [("PN-10231", "SSD 2TB NVMe", 4, 12, "Alex"), ("PN-10457", "DRAM 32GB DDR5", 8, 3, "Sam"),
        ("PN-10502", "Thermal pad 1mm", 20, 27, "Lee"), ("PN-10618", "USB4 enclosure", 2, 41, "Kim"),
        ("PN-10733", "Power cable 1.5m", 6, 5, "Jordan"), ("PN-10812", "SSD 1TB SATA", 3, 18, "Alex"),
        ("PN-10944", "Screwdriver set", 1, 66, "Robin"), ("PN-11020", "Heatsink M.2", 12, 9, "Sam"),
        ("PN-11135", "Test fixture A", 1, 33, "Lee"), ("PN-11281", "DRAM 16GB DDR4", 10, 2, "Kim"),
        ("PN-11307", "Label printer", 1, 54, "Jordan"), ("PN-11466", "Anti-static mat", 5, 21, "Robin")]


def _qr(text: str, size: int):
    import numpy as np
    import zxingcpp
    from PIL import Image
    if hasattr(zxingcpp, "create_barcode"):          # zxing-cpp >= 2.3
        img = zxingcpp.write_barcode_to_image(
            zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.QRCode), scale=8)
    else:
        img = zxingcpp.write_barcode(zxingcpp.BarcodeFormat.QRCode, text, size, size)
    return Image.fromarray(np.asarray(img)).convert("RGB").resize((size, size), Image.NEAREST)


def sample_ui(path: Path) -> dict:
    """A 1600x1000 admin screen in the layout locate.find_table() is built for:
    a full-width table with a grey header band, white body and the grey window
    background below it; then a total and a QR asset tag. Every string is drawn
    once here; afterwards it is pixels, which is all the library ever moves."""
    from PIL import Image, ImageDraw
    W, H = 1600, 1000
    img = Image.new("RGB", (W, H), (226, 229, 234))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 56), fill=(32, 58, 96))
    d.text((24, 14), "Parts Loan Manager", font=_font(26, True), fill=(255, 255, 255))
    d.text((44, 80), "Search", font=_font(20), fill=(90, 96, 110))
    d.rectangle((140, 74, 560, 106), fill=(255, 255, 255), outline=(170, 176, 188))
    d.text((152, 79), "PN-1", font=_font(20), fill=(40, 40, 40))
    x0, y0, x1, rh = 16, 130, W - 16, 44
    cols = [x0, x0 + 230, x0 + 640, x0 + 800, x0 + 1010, x1]
    heads = ["Part No", "Description", "Qty", "Days Out", "Borrower"]
    bottom = y0 + rh * (len(ROWS) + 1)
    d.rectangle((x0, y0, x1, bottom), fill=(255, 255, 255))
    d.rectangle((x0, y0 - 40, x1, y0), fill=(214, 220, 230))       # grey header band
    f, fb = _font(20), _font(20, True)
    for i, h in enumerate(heads):
        d.text((cols[i] + 12, y0 - 30), h, font=fb, fill=(30, 40, 60))
    for yy in (y0 - 42, y0, bottom):                                 # strong border lines
        d.rectangle((x0, yy, x1, yy + 2), fill=(120, 126, 138))
    for r, row in enumerate(ROWS):
        y = y0 + rh * r
        if r:
            d.line((x0, y, x1, y), fill=(232, 234, 238), width=1)
        for i, v in enumerate(row):
            d.text((cols[i] + 12, y + 11), str(v), font=f,
                   fill=(200, 40, 40) if i == 3 and int(v) > 30 else (40, 40, 40))
    ty = bottom + 24
    d.text((x0 + 8, ty), "Total 12 items", font=_font(22, True), fill=(30, 40, 60))
    cx0, cy0 = W - 420, bottom + 14
    d.rectangle((cx0, cy0, W - 16, H - 14), fill=(255, 255, 255), outline=(205, 210, 218))
    d.text((cx0 + 18, cy0 + 16), "Asset tag", font=_font(22, True), fill=(30, 40, 60))
    d.text((cx0 + 18, cy0 + 52), "SN-0042", font=_font(22), fill=(40, 40, 40))
    q = min(150, H - 14 - cy0 - 20)
    img.paste(_qr("PN-10944/SN-0042", q), (W - 16 - q - 18, cy0 + 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {"image": str(path), "size": [W, H], "rows": len(ROWS),
            "texts": ["Total 12 items", "Days Out", "Asset tag", "SN-0042"],
            "qr": "PN-10944/SN-0042"}


def cmd_sample_ui(a) -> int:
    out(sample_ui(Path(a.out)))
    return 0


# --------------------------------------------------------------------- new
TEMPLATE = '''"""Segment 01 -- {image_name}. Edit the TIMELINE, then run this file.

Rules that keep it correct (see the skill's SKILL.md):
  * every coordinate is a pixel of the ORIGINAL image; Stage.render() maps to the frame;
  * draw outlines/bands on the original-size layer, then render -- they follow zoom;
  * find things by content (locate.*), never by guessing pixels;
  * never type text in a font: hide with the background colour, reveal original pixels.
"""
import sys
sys.path.insert(0, r"{lib}")
import numpy as np
from PIL import Image
import locate, reveal, overlay
from stage import Stage, ease_box
from segment import render_segment

IMAGE = r"{image}"
OUT = r"{out}"
FIND_TEXT = "{find_text}"          # a label to outline (exact text on the image)

img = np.array(Image.open(IMAGE).convert("RGB"))
stg = Stage(img)
full = stg.full_box()
tbl = locate.find_table(img)       # header / rows / column x / white box
rows = [tuple(r) for r in tbl["rows"]]
hidden = reveal.hide(img, rows)    # every row blanked with its own background
target = locate.find_text_rect(img, FIND_TEXT)["rect"]
zoom = stg.box_around(target, margin=0.35)

def frame_fn(t):
    # 0.0-2.4 s  rows appear one by one
    lay = reveal.reveal_rows(hidden, img, rows, t, t0=0.4, per_row=0.16, fade=0.2)
    # 2.6 s      outline the target text
    if t >= 2.6:
        overlay.outline(lay, target, (255, 140, 0), width=3)
    # 3.4-4.4 s  push in on it
    box = ease_box(full, zoom, float(np.clip((t - 3.4) / 1.0, 0, 1)))
    return stg.render(lay, box)[0]

res = render_segment(frame_fn, OUT, seconds=5.0, fps=24, sounds=[
    (0.3, "swoosh", {{}}), (2.6, "beep", {{"freq": 990}}), (3.4, "chime", {{}})])
print({{k: res[k] for k in ("width", "height", "fps", "frames_counted", "duration", "audio", "problems")}})
'''


def cmd_new(a) -> int:
    d = Path(a.dir).resolve()
    (d / "out").mkdir(parents=True, exist_ok=True)
    image = Path(a.image).resolve()
    if not image.is_file():
        out({"ok": False, "error": "no image at %s" % image})
        return 1
    script = d / "segment_01.py"
    if script.exists() and not a.force:
        out({"ok": False, "error": "%s exists; edit it, or pass --force" % script})
        return 1
    script.write_text(TEMPLATE.format(image_name=image.name, lib=str(LIB), image=str(image),
                                      out=str(d / "out" / "segment_01"),
                                      find_text=a.find_text.replace('"', '\\"')),
                      encoding="utf-8")
    out({"ok": True, "script": str(script), "output": str(d / "out" / "segment_01.mp4"),
         "run": '"%s" "%s"' % (sys.executable, script)})
    return 0


# ------------------------------------------------------------------- sheet
def cmd_sheet(a) -> int:
    video = Path(a.video)
    dst = Path(a.out) if a.out else video.with_suffix(".sheet.jpg")
    vf = "fps=2,scale=320:-1,tile=5x2"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", vf,
                        "-frames:v", "1", str(dst)], capture_output=True, text=True)
    out({"ok": r.returncode == 0, "sheet": str(dst), "error": r.stderr.strip()[:300] or None,
         "note": "10 frames, 0.5 s apart, left to right then top to bottom"})
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("sample-ui"); p.add_argument("out"); p.set_defaults(fn=cmd_sample_ui)
    p = sub.add_parser("new"); p.add_argument("dir"); p.add_argument("--image", required=True)
    p.add_argument("--find-text", default="Total 12 items", dest="find_text")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("sheet"); p.add_argument("video"); p.add_argument("out", nargs="?")
    p.set_defaults(fn=cmd_sheet)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
