"""Frozen-build entry point.

The launcher is a package and uses relative imports, so it cannot be frozen through
launcher/__main__.py directly -- PyInstaller would run that file as __main__, outside
the package, and every `from . import ...` would fail. This wrapper keeps the package
intact and gives the build a plain script to start from.

From a source checkout `python -m launcher` is still the way in; this file exists only
for packaging.
"""

from __future__ import annotations

from launcher.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
