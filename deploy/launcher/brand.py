"""Read the shipped brand's page background out of the frontend bundle.

The setup window and the settings window should look like the product they belong to,
and the product exists in two brands chosen at packaging time. Keeping a second copy
of the palette here would mean two places to change and one of them eventually being
wrong, so the value is read from the stylesheet that actually shipped in this payload.

The background is what carries the brand -- a very pale wash the interface sits on.
The action colour is deliberately *not* taken: it is a saturated magenta in one brand,
sized for buttons, and on a progress bar it both shouts and reads as a warning. Bars
stay the neutral blue they were designed as.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import applog, config

log = applog.get("brand")

# The token the frontend paints its page with.
BACKGROUND_TOKEN = "--brand-page-bg"

# A hex colour, an rgb()/oklch() function, or a reference to another custom property.
_VALUE = re.compile(r"--(?P<name>[\w-]+)\s*:\s*(?P<value>[^;}]+)")
_VAR_REF = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,[^)]*)?\)")


def _stylesheets() -> list[str]:
    directory = config.payload_dir() / "frontend" / "dist" / "assets"
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.css")):
        try:
            out.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return out


def _declarations(css: str) -> dict[str, str]:
    return {m.group("name"): m.group("value").strip() for m in _VALUE.finditer(css)}


def _resolve(name: str, declarations: dict[str, str], depth: int = 4) -> str | None:
    """Follow var() references. One brand states a literal, the other points at a
    palette entry defined further down the same file."""
    value = declarations.get(name)
    if value is None:
        return None
    while depth > 0:
        reference = _VAR_REF.fullmatch(value.strip())
        if not reference:
            return value.strip()
        value = declarations.get(reference.group(1).lstrip("-"), "")
        if not value:
            return None
        depth -= 1
    return None


@lru_cache(maxsize=1)
def page_background() -> str:
    """The shipped brand's page background, or "" when the bundle cannot be read.

    One brand states a gradient, the other a near-white; both are meant for a light
    page, so the caller applies this only in the light theme.
    """
    for css in _stylesheets():
        resolved = _resolve(BACKGROUND_TOKEN.lstrip("-"), _declarations(css))
        if resolved:
            log.info("brand background: %s", resolved)
            return resolved
    log.info("no brand stylesheet found; keeping the default background")
    return ""
