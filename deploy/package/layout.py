"""What lands in the public layout, and what must never land in it.

Two rules from the spec are load-bearing here:

1.  Copy by *target list*, never by preserve list. "Default to leaving alone, name
    what moves" fails safe; "delete everything except..." silently deletes whatever
    someone forgot to list.
2.  Exclusion patterns carry their path. `configs/current_*.json` as a bare pattern
    would also strip `console/configs/`, which is public example config that must
    ship. Everything scoped lives in PER_SOURCE_EXCLUDES under its own key.
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

# The three source repositories, and where each lands in the public layout.
SOURCE_KEYS: tuple[str, ...] = ("backend", "console", "frontend")

# Frontend ships build output, not source, so it arrives as these two directories
# rather than as a repository checkout.
FRONTEND_PACKAGES: tuple[str, ...] = ("dist", "dist_client")

# Where the application icon comes from, most preferred first.
#
# Deliberately not the same choice the payload makes. A Windows payload carries the
# dist_client (ADATA) frontend, but the application itself is asked to show the TRUSTA
# mark, so the icon is taken from dist. Keeping it here rather than in either script
# means the executable, the shortcut and the tray cannot end up showing two brands.
ICON_PACKAGE_ORDER: tuple[str, ...] = ("dist", "dist_client")

# Fixed name the icon is copied to inside the payload, so the running application does
# not have to know which package it came from.
PAYLOAD_ICON = "app.ico"


def find_icon(source_root):
    """First .ico from the most preferred frontend package present under source_root."""
    from pathlib import Path as _Path

    root = _Path(source_root)
    for package in ICON_PACKAGE_ORDER:
        directory = root / "frontend" / package
        if not directory.is_dir():
            continue
        for icon in sorted(directory.glob("*.ico")):
            return icon
    return None

# Top-level entries copied out of each source repo. Globs are allowed.
INCLUDE: dict[str, tuple[str, ...]] = {
    "backend": (
        "service",
        "tests",
        "scripts",
        "dataset",  # deliberately NOT excluded: the sample dataset ships
        "docs",  # stays at backend/docs/, never hoisted to the root
        "pytest.ini",
        ".env.example",
        "README.md",
        # Without the uv project files provisioning has no project to sync.
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".pre-commit-config.yaml",
    ),
    "console": (
        "ai_scaler_console",
        "examples",
        "configs",
        "docs",
        "tests",
        "pyproject.toml",  # uv sync installs the package editable so examples/ runs anywhere
        "uv.lock",
        ".python-version",
        "pytest.ini",
        ".env.example",
        "README*",
    ),
}

# Directory names dropped wherever they appear.
EXCLUDE_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".vscode",
        ".claude",
        "logs",
        "output",
        "runtime_data",
    }
)

# Directory names dropped when they end in this suffix (offload dumps are named per model).
EXCLUDE_DIR_SUFFIXES = ("offload",)

# File globs dropped wherever they appear. `.env.example` ships; the real `.env` does not.
EXCLUDE_FILE_GLOBS = ("*.log", ".env", "CLAUDE.md")

# Paths relative to the source repo root. Scoped, so console/configs/ is untouched.
PER_SOURCE_EXCLUDES: dict[str, tuple[str, ...]] = {
    "backend": (
        ".github/copilot-instructions.md",
        "service/configs/current_inference_config.json",
        "service/configs/current_training_config.json",
        "service/configs/last_finetuned_model.json",
        "service/configs/models_registry.json",
        "service/rag_store/rag.db",
        "service/rag_store/docs",
        "tests/results",
        "tests/.cache",
    ),
    "console": (
        ".github/copilot-instructions.md",
        # internal/ is the whole reason console needs a filter rather than a plain copy.
        "internal",
        "tests/results",
        "tests/.cache",
    ),
}

# Assembly must leave these alone. Checked after every run: a copy that reaches up out
# of its target directory shows up here as a missing root file.
ROOT_MUST_EXIST: tuple[str, ...] = (
    "deploy/windows/installer.iss",
    ".github",
    "LICENSE",
    "README.md",
    "docs",
    "wiki",
    "assets",
)

# The negative assert. These are artifacts of the OLD flattened console layout. If they
# survive assembly, the target directory was merged into rather than replaced (a `cp -r`
# where a wipe was meant, or an rsync missing --delete). The result is a tree where the
# new package and the old modules coexist -- which every positive check still passes.
MUST_NOT_EXIST: tuple[str, ...] = (
    "console/ai_client.py",
    "console/app_settings",
    "console/requirements.txt",
)


def is_excluded(rel: str, source_key: str) -> bool:
    """rel is a POSIX path relative to the source repo root."""
    path = PurePosixPath(rel)
    parts = path.parts

    for name in parts[:-1] if path.suffix else parts:
        if name in EXCLUDE_DIR_NAMES or name.endswith(EXCLUDE_DIR_SUFFIXES):
            return True
    # A directory entry itself (no suffix heuristics): check the final part too.
    if parts and (parts[-1] in EXCLUDE_DIR_NAMES or parts[-1].endswith(EXCLUDE_DIR_SUFFIXES)):
        return True

    if any(fnmatch.fnmatch(parts[-1], glob) for glob in EXCLUDE_FILE_GLOBS):
        return True

    for scoped in PER_SOURCE_EXCLUDES.get(source_key, ()):
        if rel == scoped or rel.startswith(scoped + "/"):
            return True

    return False
