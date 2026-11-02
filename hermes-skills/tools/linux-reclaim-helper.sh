#!/usr/bin/env bash
# Release a co-resident LLM's clean, file-backed pages on Linux.
#
# Wire it in as the engine's escape hatch (hermes-comfy.conf):
#     VIDGEN_LLM_RECLAIM_CMD=/path/to/linux-reclaim-helper.sh
# The engine appends the pid, so this is called as:  helper.sh <pid>
#
# You only need this when the engine's built-in call is refused. That call is
# process_madvise(MADV_PAGEOUT), which needs CAP_SYS_NICE or ptrace access to
# the target -- and most distros ship /proc/sys/kernel/yama/ptrace_scope = 1,
# which denies it between sibling processes.
#
# Two mechanisms, tried in order. Both measured on WSL2 / kernel 6.18 against a
# process holding 512 MB of mmap'd file pages:
#
#   cgroup v2 memory.reclaim   522 MB -> 124 MB   (asked for 400M, got it)
#   process_madvise + sudo     522 MB ->   8 MB   (pages out everything)
#
# ---------------------------------------------------------------------------
# The cgroup caveat that wastes an afternoon
# ---------------------------------------------------------------------------
# memory.reclaim only acts on memory CHARGED to that cgroup, and page cache is
# charged to whoever first faulted it in -- not to whoever is mapping it now.
# Moving an already-running inference server into a fresh cgroup therefore leaves
# memory.current at ~0 and every reclaim returns EIO, which reads like the
# feature is broken. It is not: the process has to have been started inside the
# cgroup. A systemd-managed service already is (system.slice/<unit>), so
# this works out of the box for the normal deployment.
set -u

PID="${1:-}"
[ -n "$PID" ] || { echo "usage: $0 <pid>" >&2; exit 2; }
[ -d "/proc/$PID" ] || { echo "no such process: $PID" >&2; exit 1; }

rss_kb(){ awk '/^VmRSS/{print $2; exit}' "/proc/$PID/status" 2>/dev/null || echo 0; }
BEFORE=$(rss_kb)

# --- 1. cgroup v2 -----------------------------------------------------------
# The pid's own cgroup, read from /proc/PID/cgroup ("0::/system.slice/foo").
CGREL=$(awk -F: '$1=="0"{print $3; exit}' "/proc/$PID/cgroup" 2>/dev/null || true)
CGDIR="/sys/fs/cgroup${CGREL:-}"

# Only take this path when the cgroup holds NOTHING BUT the target. Reclaim is
# a cgroup-wide operation: pointed at a shared cgroup it evicts every process
# in it, which is precisely the collateral damage this whole stack exists to
# avoid. Measured: an unmodified WSL distro puts everything in /init.scope, so
# without this guard the helper would page out the entire session. A
# systemd-managed service has its own unit cgroup and passes.
if [ -n "${CGREL:-}" ] && [ -w "$CGDIR/memory.reclaim" ] && [ "$CGREL" != "/" ]; then
    OCCUPANTS=$(wc -l < "$CGDIR/cgroup.procs" 2>/dev/null || echo 99)
    CHARGED=$(cat "$CGDIR/memory.current" 2>/dev/null || echo 0)
    if [ "$OCCUPANTS" -eq 1 ] && [ "$CHARGED" -gt 0 ]; then
        # Ask for what is actually CHARGED here, not the process's RSS: the two
        # differ (page cache is charged to whoever first faulted it in), and
        # over-asking makes the write return EIO even when the kernel did
        # reclaim. So judge by the measured drop, not by the exit code.
        echo "$CHARGED" > "$CGDIR/memory.reclaim" 2>/dev/null || true
        sleep 1
        AFTER=$(rss_kb)
        if [ "$AFTER" -lt "$BEFORE" ]; then
            echo "cgroup memory.reclaim on $CGREL: VmRSS ${BEFORE}kB -> ${AFTER}kB"
            exit 0
        fi
    fi
fi

# --- 2. process_madvise(MADV_PAGEOUT) --------------------------------------
# Same syscall the engine uses; run this helper with the privilege the engine
# lacks (sudoers entry, or setcap cap_sys_nice on the interpreter).
python3 - "$PID" <<'PY'
import ctypes, os, sys

pid = int(sys.argv[1])
NR_PIDFD_OPEN, NR_PROCESS_MADVISE, MADV_PAGEOUT, IOV_MAX = 434, 440, 21, 1024

class IOVEC(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

def ranges(pid):
    out = []
    with open("/proc/%d/maps" % pid, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split(None, 5)
            if len(p) < 6:
                continue
            path = p[5].strip()
            if not path or path.startswith("[") or "/dev/" in path or "(deleted)" in path:
                continue
            lo, _, hi = p[0].partition("-")
            a, b = int(lo, 16), int(hi, 16)
            if b > a:
                out.append((a, b - a))
    return out

def rss():
    with open("/proc/%d/status" % pid) as f:
        for l in f:
            if l.startswith("VmRSS"):
                return int(l.split()[1])
    return 0

libc = ctypes.CDLL(None, use_errno=True)
before = rss()
fd = libc.syscall(NR_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
if fd < 0:
    sys.exit("pidfd_open failed: %s" % os.strerror(ctypes.get_errno()))
try:
    rs, done = ranges(pid), 0
    for i in range(0, len(rs), IOV_MAX):
        chunk = rs[i:i + IOV_MAX]
        arr = (IOVEC * len(chunk))()
        for j, (base, ln) in enumerate(chunk):
            arr[j].iov_base = ctypes.c_void_p(base)
            arr[j].iov_len = ctypes.c_size_t(ln)
        n = libc.syscall(NR_PROCESS_MADVISE, ctypes.c_int(fd), ctypes.byref(arr),
                         ctypes.c_size_t(len(chunk)), ctypes.c_int(MADV_PAGEOUT),
                         ctypes.c_uint(0))
        if n < 0:
            err = ctypes.get_errno()
            hint = ""
            if err == 1:
                hint = ("  (EPERM: needs CAP_SYS_NICE, or run this helper via sudo; "
                        "see /proc/sys/kernel/yama/ptrace_scope)")
            sys.exit("process_madvise: %s%s" % (os.strerror(err), hint))
        done += n
    print("process_madvise: VmRSS %dkB -> %dkB (%.1f GB paged out)"
          % (before, rss(), done / 1024.0 ** 3))
finally:
    os.close(fd)
PY
