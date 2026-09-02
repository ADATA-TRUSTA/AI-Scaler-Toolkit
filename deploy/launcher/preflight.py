"""Check the machine can run what we are about to install, and fix what we can.

The split is deliberate: prerequisites *we* need are detected and installed here;
optional third-party tools are not this application's business at all.

git is the one that surprises people. It looks like something a user would already
have, but provisioning needs it directly -- setup_env fetches the GGUF conversion
tooling with a sparse checkout -- so a machine without it fails partway through setup
rather than at first use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import applog, config

log = applog.get("preflight")


@dataclass
class Check:
    key: str
    label: str
    ok: bool
    # A missing requirement stops provisioning; a missing optional degrades a feature.
    required: bool = True
    detail: str = ""
    # Shown when we cannot fix it ourselves.
    remedy: str = ""
    fixer: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    @property
    def ok(self) -> bool:
        return not self.blocking


def _on_path(name: str) -> bool:
    return shutil.which(name) is not None


def _reg_value_exists(root: str, key: str, value: str) -> bool:
    if not config.IS_WINDOWS:
        return False
    try:
        proc = subprocess.run(
            ["reg", "query", rf"{root}\{key}", "/v", value],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=config.subprocess_flags(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def check_git() -> Check:
    return Check(
        key="git",
        label="Git",
        ok=_on_path("git"),
        required=True,
        detail="Needed by setup during installation, not just by developers.",
        remedy="https://git-scm.com/download/win" if config.IS_WINDOWS
        else "install git with your package manager",
        fixer="winget" if config.IS_WINDOWS else "",
    )


def check_webview2() -> Check:
    """pywebview renders the window with Edge WebView2 on Windows.

    Present on most up-to-date Windows 11 installs, absent often enough on managed
    machines to be worth checking -- without it the app starts and shows nothing.
    """
    if not config.IS_WINDOWS:
        return Check("webview2", "WebView2 runtime", ok=True, required=False,
                     detail="Not used on this platform.")
    key = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    found = _reg_value_exists("HKLM", key, "pv") or _reg_value_exists(
        "HKCU", key.replace("WOW6432Node\\", ""), "pv"
    )
    return Check("webview2", "WebView2 runtime", ok=found, required=True,
                 detail="Renders the application window.",
                 remedy="https://developer.microsoft.com/microsoft-edge/webview2/",
                 fixer="webview2")


def check_vcredist() -> Check:
    """Native extensions in torch and llama link against the MSVC runtime."""
    if not config.IS_WINDOWS:
        return Check("vcredist", "Visual C++ runtime", ok=True, required=False,
                     detail="Not used on this platform.")
    found = _reg_value_exists("HKLM", r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64", "Installed")
    return Check("vcredist", "Visual C++ runtime", ok=found, required=True,
                 detail="Required by the GPU runtime and llama.",
                 remedy="https://aka.ms/vs/17/release/vc_redist.x64.exe",
                 fixer="vcredist")


def check_nvidia() -> Check:
    """Presence decides which accelerator setup_env picks, so report it rather than
    require it -- a machine with no NVIDIA card is a valid configuration."""
    return Check("nvidia", "NVIDIA driver", ok=_on_path("nvidia-smi"), required=False,
                 detail="Selects the CUDA build. Intel GPUs use the XPU build instead.")


def check_vulkan() -> Check:
    """The Intel path uses the Vulkan build of llama; the runtime normally arrives with
    the Intel driver."""
    if config.IS_WINDOWS:
        found = (Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32" / "vulkan-1.dll").is_file()
    else:
        found = _on_path("vulkaninfo")
    return Check("vulkan", "Vulkan runtime", ok=found, required=False,
                 detail="Used by the llama build for Intel and AMD cards.",
                 remedy="usually installed with the graphics driver")


def check_winget() -> Check:
    return Check("winget", "winget", ok=_on_path("winget") if config.IS_WINDOWS else True,
                 required=False,
                 detail="Used by one of the optional third-party components' own installer.")


def check_webkitgtk() -> Check:
    """Linux only. Not bundled in the AppImage on purpose: it is large and conflicts
    with the system copy often enough that shipping it causes more breakage than the
    dependency does."""
    if config.IS_WINDOWS:
        return Check("webkitgtk", "WebKitGTK", ok=True, required=False,
                     detail="Not used on this platform.")
    candidates = ("/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu")
    found = any(Path(d).glob("libwebkit2gtk-*.so*") for d in candidates if Path(d).is_dir())
    return Check("webkitgtk", "WebKitGTK", ok=found, required=True,
                 detail="Renders the application window on Linux.",
                 remedy="install your distribution's webkit2gtk package")


def check_disk() -> Check:
    from .provision import disk_free_gb  # local import: provision imports config, not this

    free = disk_free_gb(config.data_dir())
    return Check("disk", "Free disk space", ok=free >= config.MIN_FREE_DISK_GB, required=True,
                 detail=f"{free:.1f} GB available, {config.MIN_FREE_DISK_GB} GB needed for setup.",
                 remedy="free up space and start again")


CHECKS = (
    check_disk,
    check_git,
    check_webview2,
    check_vcredist,
    check_webkitgtk,
    check_nvidia,
    check_vulkan,
    check_winget,
)


def run() -> Report:
    log.info("running %d checks", len(CHECKS))
    report = Report()
    for check in CHECKS:
        try:
            result = check()
            log.info("%-14s %-9s %s", result.key,
                     "ok" if result.ok else ("BLOCKING" if result.required else "missing"),
                     result.detail)
            report.checks.append(result)
        except Exception as exc:  # a probe failing is not the same as the thing missing
            log.warning("%s could not be checked: %s", check.__name__, exc)
            report.checks.append(
                Check(check.__name__, check.__name__, ok=True, required=False,
                      detail=f"could not be checked: {exc}")
            )
    log.info("preflight %s", "passed" if report.ok else
             "BLOCKED by " + ", ".join(c.key for c in report.blocking))
    return report


def describe(checks: Iterable[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok " if c.ok else ("!! " if c.required else " ? ")
        lines.append(f"{mark} {c.label}: {c.detail}".rstrip())
        if not c.ok and c.remedy:
            lines.append(f"     -> {c.remedy}")
    return "\n".join(lines)


if __name__ == "__main__":
    report = run()
    print(describe(report.checks))
    print()
    print("ready" if report.ok else f"blocked by: {', '.join(c.label for c in report.blocking)}")
