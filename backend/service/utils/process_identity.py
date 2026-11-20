"""Process identity that survives wall-clock steps."""

import psutil


def process_identity(pid: int) -> float | None:
    """
    Return a value that stays the same for one process and differs after PID reuse.

    On Linux this is the start time in clock ticks from ``/proc/<pid>/stat``. psutil's
    ``create_time()`` is derived from ``time.time()`` minus uptime, so a wall-clock step (seen on
    WSL2, 2-4 s) changes it for a live process and the watchdog killed healthy workers.
    Elsewhere psutil's value is used. ``None`` means the process cannot be inspected.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as f:
            stat = f.read()
        # comm (field 2) may contain spaces and parentheses; split after the last ")".
        # starttime is field 22, which is index 19 of what follows the ")".
        return float(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        pass
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None
