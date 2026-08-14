"""
Shared guard for paths the service must never walk, wipe, or delete.

Three places need this: model deletion (`DELETE /config/models/{label}`), the recursive
size walk (`GET /system/resources?calc_size=true`), and the DeepSpeed NVMe offload cleanup,
which empties the directory it is pointed at. Each of them used to carry its own inline
POSIX prefix list, which made them wrong in two different ways:

* On Windows a POSIX prefix list matches nothing. ``os.path.realpath("/etc")`` is ``C:\\etc``
  there and the resolved form uses backslashes, so every check silently passed — including
  for ``C:\\Windows``. Windows needs its own set, compared case-insensitively.
* On Linux, listing ``/home`` as a *prefix* blocked ``/home/<user>/...``, which is exactly
  where the documented install keeps models and outputs, so the feature could never run.

Hence the split below: prefixes cover trees that hold nothing but OS files, while the
directories whose children are legitimate data locations are matched exactly.
"""

import os
from pathlib import Path

# Nothing legitimate ever lives underneath these, so the whole subtree is off limits.
_PROTECTED_POSIX_PREFIXES = (
    "/etc",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/usr/lib",
    "/lib",
    "/lib64",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/run",
)

# Removing these is catastrophic, but their children are ordinary data locations.
_PROTECTED_POSIX_EXACT = ("/", "/home", "/root", "/usr", "/var", "/opt")

# Windows counterparts, resolved from the environment so a non-C: install still works.
_PROTECTED_WINDOWS_PREFIX_VARS = ("SystemRoot", "ProgramFiles", "ProgramFiles(x86)", "ProgramData")


def is_protected_system_path(raw_path: str | os.PathLike[str]) -> bool:
    """
    Whether `raw_path` resolves into an OS directory that must not be walked or deleted.

    Unparseable input counts as protected: refusing is always safer than deleting a path
    the caller could not even resolve.
    """
    try:
        resolved = Path(raw_path).resolve()
    except (OSError, ValueError):
        return True

    if os.name != "nt":
        text = str(resolved)
        if text in _PROTECTED_POSIX_EXACT:
            return True
        return any(text == p or text.startswith(p + "/") for p in _PROTECTED_POSIX_PREFIXES)

    # A bare drive root (C:\) is never a data directory.
    if resolved == Path(resolved.anchor):
        return True

    prefixes = [
        Path(value)
        for value in (os.environ.get(var) for var in _PROTECTED_WINDOWS_PREFIX_VARS)
        if value
    ]
    profile = os.environ.get("USERPROFILE")
    # C:\Users itself, but not a user's own subtree, which holds models and caches.
    exact = [Path(profile).parent] if profile else []

    text = str(resolved).lower()
    if any(text == str(candidate).lower() for candidate in exact):
        return True
    return any(
        text == str(prefix).lower() or text.startswith(str(prefix).lower().rstrip(os.sep) + os.sep)
        for prefix in prefixes
    )
