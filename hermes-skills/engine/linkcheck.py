"""Keep a linked install's skill documents in step with the pack.

`install.py --link` runs the engine and the skill scripts straight from the pack
and renders only SKILL.md (with this machine's paths) into the Hermes home. A
git hook re-renders them after every pull; this is the second line, for an
update the hook did not see (hooks not installed on this clone, a `git reset`,
files copied in by hand). A skill calls refresh_if_stale() on start: if any
document in the pack differs from what was rendered, the install is refreshed
before the next time the model reads it. The run in progress already read the
old text; the cost of that is one turn, and it is said on stderr.

Never raises and never blocks a render: a refresh that fails is reported and
the skill carries on.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import paths

RECORD = Path(paths.STATE_DIR) / "comfyui-skill-link.json"


def _sha(p: Path) -> str | None:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()
    except OSError:
        return None


def stale() -> bool:
    """True when the pack's documents no longer match what was rendered."""
    try:
        rec = json.loads(RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False                     # not a linked install
    pack = Path(rec["pack"])
    seen = rec.get("templates") or {}
    for rel, sha in seen.items():
        if _sha(pack / rel) != sha:
            return True
    # A document added to the pack since the last render.
    for f in (pack / "skills").rglob("*"):
        if (f.is_file() and f.suffix not in (".py", ".pyc", ".log")
                and f.name != "engine_dir.txt"
                and f.relative_to(pack).as_posix() not in seen):
            return True
    return False


def refresh_if_stale() -> None:
    try:
        if not stale():
            return
        rec = json.loads(RECORD.read_text(encoding="utf-8"))
        r = subprocess.run([sys.executable, str(Path(rec["pack"]) / "install.py"),
                            "--refresh", "--hermes-home", rec["home"]],
                           capture_output=True, text=True, timeout=120)
        print("[linkcheck] the pack changed since its skill documents were rendered; "
              + ("refreshed them for the next turn." if r.returncode == 0 else
                 "refresh FAILED (%s) -- run install.py --refresh" % r.stderr.strip()[:200]),
              file=sys.stderr)
    except Exception as e:               # never let this break a render
        print("[linkcheck] could not check the skill documents: %r" % (e,), file=sys.stderr)
