"""Build the application icon from the brand artwork.

Windows draws this at nine different sizes and picks the nearest one it finds, so an
icon that carries only a couple of entries is upscaled at every other size. The
shortcut on the desktop is drawn at 48, Explorer's largest view at 256; an icon with
only 16x16 in it -- which is what shipped until this existed -- is a three- to
sixteen-fold upscale in both.

Rendering down from one large square source is the whole trick: every entry is
resampled from the original rather than from another entry.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Icon artwork is a packaging input, so it lives with the rest of packaging. The
# repository's own assets/ holds the images the READMEs display -- cd-flow is explicit
# that it "只服務 README 標頭圖示，不是 app icon" -- and those are searched only as a
# fallback, because one brand currently has nothing better.
ASSET_DIRS = (REPO_ROOT / "deploy" / "assets", REPO_ROOT / "assets")

# 16/32/48/256 is the set Microsoft asks for; the rest cover the DPI scalings that
# would otherwise be interpolated from a neighbour.
SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64, 128, 256)

# Windows composites the icon against a selection highlight and against neighbouring
# icons, so artwork that runs to the edge looks cramped and collides. Both brand files
# do run to the edge, so the margin is added here rather than assumed.
PADDING = 0.06

# Preferred source per brand, largest first: the more pixels to resample from, the
# cleaner every entry below 256.
SOURCES: dict[str, tuple[str, ...]] = {
    "trusta": ("trusta-icon-1024.png", "trusta-icon--512.png", "trusta-icon-256.png",
               "trusta-icon.png"),
    "adata": ("adata-icon-1024.png", "adata-icon-512.png", "adata-icon.png"),
}


def find_source(brand: str) -> Path:
    for name in SOURCES[brand]:
        for directory in ASSET_DIRS:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    searched = " or ".join(str(d) for d in ASSET_DIRS)
    raise SystemExit(
        f"no artwork for {brand} in {searched}. Expected one of: {', '.join(SOURCES[brand])}"
    )


def square(image, padding: float = PADDING):
    """Centre the artwork on a square canvas with a little room around it."""
    from PIL import Image

    image = image.convert("RGBA")
    mask = image.getchannel("A").point(lambda v: 255 if v > 8 else 0)
    box = mask.getbbox() or (0, 0, *image.size)
    ink = image.crop(box)
    side = round(max(ink.size) / (1 - 2 * padding))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(ink, ((side - ink.width) // 2, (side - ink.height) // 2), ink)
    return canvas


def ico_sizes(path: Path) -> list[int]:
    """Widths inside an .ico, so a build can check what it is about to ship."""
    data = path.read_bytes()
    if len(data) < 6:
        return []
    count = struct.unpack_from("<H", data, 4)[0]
    out = []
    for index in range(count):
        width = struct.unpack_from("<B", data, 6 + index * 16)[0]
        out.append(width or 256)
    return sorted(out)


def build(brand: str, destination: Path) -> Path:
    from PIL import Image

    source = find_source(brand)
    with Image.open(source) as raw:
        canvas = square(raw)
        if canvas.width < max(SIZES):
            print(f"[icon] warning: {source.name} gives a {canvas.width}px square, so the "
                  f"{max(SIZES)}px entry is an upscale")
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="ICO", sizes=[(s, s) for s in SIZES])

    print(f"[icon] {source.name} ({raw.size[0]}x{raw.size[1]}) -> {destination.name} "
          f"[{', '.join(str(s) for s in ico_sizes(destination))}]")
    return destination


# The mark the setup window shows beside its heading. A PNG rather than the .ico
# because the page is HTML: it is displayed at 38px and this covers the highest DPI
# scaling without carrying the other eight sizes along with it.
UI_MARK_SIZE = 128


def build_png(brand: str, destination: Path, size: int = UI_MARK_SIZE) -> Path:
    from PIL import Image

    source = find_source(brand)
    with Image.open(source) as raw:
        canvas = square(raw).resize((size, size), Image.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG", optimize=True)

    print(f"[icon] {source.name} -> {destination.name} ({size}px, "
          f"{destination.stat().st_size} bytes)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brand", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", required=True, help="path to write")
    parser.add_argument("--png", action="store_true",
                        help="write a single PNG for the window, not a multi-size .ico")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    build_png(args.brand, out) if args.png else build(args.brand, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
