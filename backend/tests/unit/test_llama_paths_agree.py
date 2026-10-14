"""
The llama install location and the pinned version live in three places that cannot import
each other: service/settings.py, scripts/windows/setup_env.ps1 and scripts/linux/setup_env.sh.

The convert-tooling revision used to be a fourth: each script carried a hardcoded fallback sha
for when the tag could not be resolved offline, and a test here checked the annotation still
named the pinned tag. Both scripts now fetch an archive of the tag itself, so there is no sha
to keep in step and nothing left to assert.

Changing one of them alone is silent - setup_env installs into A, the service looks in B, and
the only symptom is "binary not found" long after the fact. These tests read the literals back
out of the shell scripts and assert they still agree with the Python side.
"""

import re
from pathlib import Path

from service.settings import LLAMA_BIN_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
PS1 = REPO_ROOT / "scripts" / "windows" / "setup_env.ps1"
SH = REPO_ROOT / "scripts" / "linux" / "setup_env.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_windows_setup_env_installs_where_settings_looks() -> None:
    """setup_env.ps1 must unpack the release where settings.LLAMA_BIN_DIR points."""
    # $LlamaBinDir = Join-Path $ServiceDir "utils\llama-bin"
    m = re.search(r'\$LlamaBinDir\s*=\s*Join-Path\s+\$ServiceDir\s+"([^"]+)"', _read(PS1))
    assert m, "could not find $LlamaBinDir in setup_env.ps1"
    relative = m.group(1).replace("\\", "/")
    expected = LLAMA_BIN_DIR.relative_to(LLAMA_BIN_DIR.parents[1]).as_posix()
    assert relative == expected, (
        f"setup_env.ps1 installs into service/{relative} but settings.LLAMA_BIN_DIR is "
        f"service/{expected}"
    )


def test_linux_setup_env_installs_where_settings_looks() -> None:
    """setup_env.sh must build into where settings.LLAMA_BIN_DIR points."""
    # LLAMA_BIN_DIR="$SERVICE_DIR/utils/llama-bin"
    m = re.search(r'^LLAMA_BIN_DIR="\$SERVICE_DIR/([^"]+)"', _read(SH), re.MULTILINE)
    assert m, "could not find LLAMA_BIN_DIR in setup_env.sh"
    relative = m.group(1)
    expected = LLAMA_BIN_DIR.relative_to(LLAMA_BIN_DIR.parents[1]).as_posix()
    assert relative == expected, (
        f"setup_env.sh installs into service/{relative} but settings.LLAMA_BIN_DIR is "
        f"service/{expected}"
    )


def _pinned_versions(text: str) -> set[str]:
    """Every b<digits> build tag that appears as a default value, not in prose."""
    return (
        set(re.findall(r'["\']?(b\d{4,})["\']?\s*$', text, re.MULTILINE))
        | set(re.findall(r'(?:LlamaVersion|ver)\s*=\s*["\'](b\d{4,})["\']', text))
        | set(re.findall(r"TRUSTA_LLAMA_VERSION:-(b\d{4,})", text))
    )


def test_both_scripts_pin_the_same_llama_version() -> None:
    """Both platforms have to install the same llama build, or results diverge by OS."""
    win = _pinned_versions(_read(PS1))
    lin = _pinned_versions(_read(SH))
    assert win, "no pinned llama version found in setup_env.ps1"
    assert lin, "no pinned llama version found in setup_env.sh"
    assert len(win) == 1, f"setup_env.ps1 pins more than one version: {sorted(win)}"
    assert len(lin) == 1, f"setup_env.sh pins more than one version: {sorted(lin)}"
    assert win == lin, (
        f"setup_env.ps1 pins {win.pop()} but setup_env.sh pins {lin.pop()}; the two platforms "
        "must install the same llama build"
    )
