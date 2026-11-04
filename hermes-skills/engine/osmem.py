"""Process + memory primitives, one implementation per OS, one API for callers.

These are OS primitives only. *Which* process is the co-resident LLM is not
decided here -- astapi.py asks the backend that owns it -- so nothing in this
module names an inference runtime. What it provides is the measurement and the
trim: how much of a given process's DRAM is genuinely reclaimable, and handing
those pages back once a render is confirmed running.

The measurement contract (identical on both platforms)
------------------------------------------------------
    proc_dram(pid) -> (working_set_gb, private_gb)
    reclaimable    =  working_set - private          (clamped at 0)

That subtraction is the whole trick, and it is meaningful on both systems for
the same reason: it isolates *clean, file-backed* pages — an mmap'd GGUF the
kernel can drop for free and re-read from disk on the LLM's next token — from
dirty private pages, which can only be freed by writing them to swap. Any
runtime that loads its weights into dirty private pages instead of mapping them
-- a loader run with mmap disabled, or a framework that just reads the file --
yields 0 on both platforms without any flag being set, which is the correct
answer.

    Windows   WorkingSetSize      - PrivateUsage        (PSAPI)
    Linux     VmRSS               - RssAnon             (/proc/PID/status)

On Linux we additionally subtract VmLck: mlocked pages are still counted in
RssFile but the kernel physically may not evict them.

The trim, and why the two platforms differ in how much it matters
-----------------------------------------------------------------
Windows needs the explicit call. Its working-set manager will not trim another
process's clean pages just because *we* are about to allocate, so without
EmptyWorkingSet the memory stays parked and the render is refused a budget it
could actually have had.

Linux mostly does this on its own — clean file-backed pages sit on the inactive
LRU list and are dropped under allocation pressure, promptly and for free. So
process_madvise(MADV_PAGEOUT) here is an optimisation that front-loads work the
kernel would do anyway, and if it fails (Yama ptrace_scope, no CAP_SYS_NICE,
kernel < 5.10) the budgeting stays valid and the render still works. That is
why a failed trim is reported, never raised.
"""
from __future__ import annotations
import ctypes
import json
import os
import subprocess
import sys

WINDOWS = sys.platform == "win32"
GB = 1024.0 ** 3


# ==========================================================================
# Windows
# ==========================================================================
if WINDOWS:

    class _PROCENTRY(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                    ("th32ProcessID", ctypes.c_ulong),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
                    ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_char * 260)]

    class _MIB_TCPROW_OWNER_PID(ctypes.Structure):
        _fields_ = [("dwState", ctypes.c_ulong), ("dwLocalAddr", ctypes.c_ulong),
                    ("dwLocalPort", ctypes.c_ulong), ("dwRemoteAddr", ctypes.c_ulong),
                    ("dwRemotePort", ctypes.c_ulong), ("dwOwningPid", ctypes.c_ulong)]

    class _PMCEX(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    def system_ram():
        """Physical RAM total/available in MiB via GlobalMemoryStatusEx."""
        try:
            m = _MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return {"total_mb": m.ullTotalPhys // 2 ** 20,
                    "avail_mb": m.ullAvailPhys // 2 ** 20}
        except Exception as e:
            return {"error": str(e)}

    def find_pid(name: str):
        """Toolhelp32 snapshot — no subprocess, cheap enough for every plan()."""
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
        if snap == -1:
            return None
        try:
            e = _PROCENTRY()
            e.dwSize = ctypes.sizeof(_PROCENTRY)
            if not k32.Process32First(snap, ctypes.byref(e)):
                return None
            target = name.lower().encode()
            while True:
                if e.szExeFile.lower() == target:
                    return int(e.th32ProcessID)
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    return None
        finally:
            k32.CloseHandle(snap)

    def listening_pids() -> dict:
        """{port: pid} for every IPv4 TCP socket in LISTEN.

        Straight to iphlpapi, no subprocess: this runs on the plan() path, and
        shelling out to netstat there would cost more than the rest of the plan.
        """
        try:
            ip = ctypes.windll.iphlpapi
            size = ctypes.c_ulong(0)
            # AF_INET(2), TCP_TABLE_OWNER_PID_LISTENER(3). The first call is the
            # documented way to learn the buffer size and always "fails".
            ip.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)
            buf = ctypes.create_string_buffer(size.value)
            if ip.GetExtendedTcpTable(buf, ctypes.byref(size), False, 2, 3, 0) != 0:
                return {}
            n = ctypes.c_ulong.from_buffer_copy(buf, 0).value
            width = ctypes.sizeof(_MIB_TCPROW_OWNER_PID)
            out = {}
            for i in range(n):
                r = _MIB_TCPROW_OWNER_PID.from_buffer_copy(buf, 4 + i * width)
                lo = r.dwLocalPort & 0xFFFF          # network byte order
                out.setdefault(((lo & 0xFF) << 8) | (lo >> 8), int(r.dwOwningPid))
            return out
        except Exception:
            return {}

    def proc_tree() -> dict:
        """{pid: ppid} from one Toolhelp32 snapshot -- the same call find_pid makes."""
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snap == -1:
            return {}
        out = {}
        try:
            e = _PROCENTRY()
            e.dwSize = ctypes.sizeof(e)
            if not k32.Process32First(snap, ctypes.byref(e)):
                return out
            while True:
                out[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    return out
        finally:
            k32.CloseHandle(snap)

    def proc_start(pid: int):
        """Creation time as a FILETIME (100 ns ticks), or None if unreadable.

        Only ever compared against another pid's on the same box, so the unit
        does not have to agree with Linux's -- see descendants(), which is the
        one caller and the reason this exists at all.
        """
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return None
        try:
            created = ctypes.c_ulonglong()
            rest = (ctypes.c_ulonglong * 3)()
            if not ctypes.windll.kernel32.GetProcessTimes(
                    h, ctypes.byref(created), ctypes.byref(rest, 0),
                    ctypes.byref(rest, 8), ctypes.byref(rest, 16)):
                return None
            return int(created.value)
        except Exception:
            return None
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    def proc_name(pid: int) -> str:
        """The executable name for a pid, for messages only -- never to match on."""
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snap == -1:
            return ""
        try:
            e = _PROCENTRY()
            e.dwSize = ctypes.sizeof(e)
            if not k32.Process32First(snap, ctypes.byref(e)):
                return ""
            while True:
                if int(e.th32ProcessID) == int(pid):
                    return e.szExeFile.decode(errors="replace")
                if not k32.Process32Next(snap, ctypes.byref(e)):
                    return ""
        finally:
            k32.CloseHandle(snap)

    def _proc_dram_noh(pid: int):
        """(WS, private) without a process handle. Strata's engine denies even
        QUERY_LIMITED_INFORMATION (error 5), so OpenProcess cannot see the one
        process that holds 18 GB; the system process list (what Get-Process
        reads) still reports it."""
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "$p = Get-Process -Id %d -ErrorAction Stop; "
                 "'' + $p.WorkingSet64 + ' ' + $p.PrivateMemorySize64" % int(pid)],
                capture_output=True, text=True, timeout=15).stdout.split()
            return int(out[0]) / GB, int(out[1]) / GB
        except Exception:
            return None

    def proc_dram(pid: int):
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        if not h:
            return _proc_dram_noh(pid)
        try:
            c = _PMCEX()
            c.cb = ctypes.sizeof(_PMCEX)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                return None
            return c.WorkingSetSize / GB, c.PrivateUsage / GB
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    def release_dram(pid: int):
        """EmptyWorkingSet: DRAM working-set trim only.

        Measured on the reference box: the hosted model's VRAM went 6484 -> 6491
        MiB (i.e. untouched) and it kept serving. This is not a shutdown.
        """
        h = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0100, False, pid)  # QUERY|SET_QUOTA
        if not h:
            return False, "OpenProcess failed (err %d)" % ctypes.get_last_error()
        try:
            ok = bool(ctypes.windll.psapi.EmptyWorkingSet(h))
            return ok, ("EmptyWorkingSet ok" if ok else "EmptyWorkingSet returned false")
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    def pid_alive(pid: int) -> bool:
        try:
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not h:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == 259  # STILL_ACTIVE
        except Exception:
            return True  # unsure -> assume alive, never falsely report failure

    def proc_cmdline(name: str) -> str:
        try:
            return subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Process -Filter \"Name='%s'\").CommandLine" % name],
                text=True, timeout=30)
        except Exception:
            return ""

    def detach_kwargs() -> dict:
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: survives the parent's
        # terminal being killed at its timeout, which is the entire point.
        return {"creationflags": 0x00000008 | 0x00000200, "close_fds": True}


# ==========================================================================
# Linux / POSIX
# ==========================================================================
else:

    _PROC = "/proc"

    def _meminfo() -> dict:
        out = {}
        with open(f"{_PROC}/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        out[k] = int(parts[0])  # kB
                    except ValueError:
                        pass
        return out

    def system_ram():
        """Physical RAM total/available in MiB via /proc/meminfo.

        MemAvailable, not MemFree: it is the kernel's own estimate of what a new
        allocation can get without swapping, and already includes reclaimable
        page cache. MemFree alone would refuse every render on a warm box.
        """
        try:
            mi = _meminfo()
            total = mi.get("MemTotal", 0) // 1024
            avail = mi.get("MemAvailable")
            if avail is None:  # kernels < 3.14
                avail = mi.get("MemFree", 0) + mi.get("Cached", 0) + mi.get("Buffers", 0)
            return {"total_mb": total, "avail_mb": avail // 1024}
        except Exception as e:
            return {"error": str(e)}

    def _comm(pid: str) -> str:
        try:
            with open(f"{_PROC}/{pid}/comm", encoding="utf-8", errors="replace") as f:
                return f.read().strip()
        except OSError:
            return ""

    def find_pid(name: str):
        """Match /proc/PID/comm, then the cmdline's argv[0] basename.

        comm is truncated to 15 chars by the kernel, so a long binary name only
        ever matches on the argv[0] pass — both are tried before giving up.
        `.exe` is stripped so a Windows-shaped config value still works here.
        """
        want = name.lower()
        if want.endswith(".exe"):
            want = want[:-4]
        fallback = None
        for entry in os.listdir(_PROC):
            if not entry.isdigit():
                continue
            c = _comm(entry).lower()
            if c and (c == want or c == want[:15]):
                return int(entry)
            if fallback is None:
                try:
                    with open(f"{_PROC}/{entry}/cmdline", "rb") as f:
                        argv0 = f.read().split(b"\0", 1)[0].decode("utf-8", "replace")
                except OSError:
                    continue
                if argv0 and os.path.basename(argv0).lower() == want:
                    fallback = int(entry)
        return fallback

    def _stat_fields(pid) -> list:
        """/proc/PID/stat from the state field onward.

        Everything up to the LAST ')' is dropped rather than split off: comm is
        the process name verbatim, so it can contain both spaces and
        parentheses, and splitting on whitespace from the left mis-numbers every
        field after it for exactly the processes whose names are worth reading.
        """
        with open(f"{_PROC}/{pid}/stat", encoding="utf-8", errors="replace") as f:
            return f.read().rpartition(")")[2].split()

    def proc_tree() -> dict:
        """{pid: ppid} for every process in /proc."""
        out = {}
        for entry in os.listdir(_PROC):
            if not entry.isdigit():
                continue
            try:
                fields = _stat_fields(entry)
                out[int(entry)] = int(fields[1])       # stat field 4: ppid
            except (OSError, IndexError, ValueError):
                continue
        return out

    def proc_start(pid: int):
        """starttime (stat field 22) in clock ticks since boot, or None.

        Linux reparents an orphan to init instead of leaving a stale ppid
        behind, so the recycled-parent problem this guards against is really a
        Windows one. It is implemented on both so descendants() does not have to
        care which OS it is on.
        """
        try:
            return int(_stat_fields(pid)[19])
        except (OSError, IndexError, ValueError):
            return None

    def listening_pids() -> dict:
        """{port: pid} for listening TCP sockets, via /proc/net/tcp{,6}.

        Two hops, because the kernel splits the answer: the net tables map a
        port to a socket inode, and only a process's own fd table says who holds
        that inode. Sockets whose owner we cannot see (another user, a
        different netns) simply do not appear -- an absent port is reported the
        same as a port nobody listens on, which is what the caller wants.
        """
        want = {}                                     # inode -> port
        for name in ("tcp", "tcp6"):
            try:
                with open(f"{_PROC}/net/{name}", encoding="ascii",
                          errors="replace") as f:
                    next(f, None)                     # header row
                    for line in f:
                        parts = line.split()
                        if len(parts) < 10 or parts[3] != "0A":   # TCP_LISTEN
                            continue
                        try:
                            want[parts[9]] = int(parts[1].rpartition(":")[2], 16)
                        except ValueError:
                            continue
            except OSError:
                continue
        if not want:
            return {}
        out = {}
        for entry in os.listdir(_PROC):
            if not entry.isdigit():
                continue
            fd_dir = f"{_PROC}/{entry}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue                              # not ours to inspect
            for fd in fds:
                try:
                    link = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if not link.startswith("socket:["):
                    continue
                port = want.get(link[8:-1])
                if port is not None:
                    out.setdefault(port, int(entry))
        return out

    def proc_name(pid: int) -> str:
        """The executable name for a pid, for messages only -- never to match on."""
        return _comm(str(pid))

    def _status(pid: int) -> dict:
        out = {}
        with open(f"{_PROC}/{pid}/status", encoding="utf-8", errors="replace") as f:
            for line in f:
                k, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        out[k] = int(parts[0])  # kB for Vm*/Rss* fields
                    except ValueError:
                        out[k] = rest.strip()
        return out

    def proc_dram(pid: int):
        """(VmRSS, RssAnon + RssShmem + VmLck) in GB — see the module docstring.

        Folding VmLck into the "private" term is deliberate: mlocked pages are
        file-backed as far as RssFile is concerned but the kernel may not evict
        them, so counting them as reclaimable would overstate the budget by
        exactly the amount an mlocking runtime has pinned.
        """
        try:
            st = _status(pid)
        except OSError:
            return None
        rss = st.get("VmRSS")
        if rss is None:
            return None
        anon = st.get("RssAnon")
        if anon is None:
            # Pre-4.5 kernels have no RssAnon. Without it we cannot tell clean
            # from dirty, and guessing would risk swapping the LLM out.
            return rss / 1024.0 / 1024.0, rss / 1024.0 / 1024.0
        locked = st.get("VmLck", 0)
        # RssShmem is not reclaimable either: CUDA pinned host memory (an engine's
        # offloaded experts and KV DRAM tier) is mapped shared and shows up here,
        # not in RssAnon. Counting it as clean file pages inflated the RAM budget
        # by the whole pinned pool.
        shmem = st.get("RssShmem", 0)
        return rss / 1024.0 / 1024.0, (anon + shmem + locked) / 1024.0 / 1024.0

    # syscall numbers: identical on x86_64 and aarch64 (both use the generic
    # table for syscalls added this late).
    _NR_PIDFD_OPEN = 434
    _NR_PROCESS_MADVISE = 440
    _MADV_PAGEOUT = 21
    _IOV_MAX = 1024

    class _IOVEC(ctypes.Structure):
        _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

    def _file_backed_ranges(pid: int):
        """Address ranges of file-backed, non-locked mappings, from /proc/PID/maps.

        Only these are worth paging out: anonymous ranges would go to swap,
        which is the exact cost we are trying to avoid paying on the LLM's behalf.
        """
        ranges = []
        with open(f"{_PROC}/{pid}/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5].strip()
                if not path or path.startswith("["):
                    continue                      # [heap] [stack] [vvar] ... and anon
                if "/dev/" in path or "(deleted)" in path:
                    continue
                lo, _, hi = parts[0].partition("-")
                try:
                    a, b = int(lo, 16), int(hi, 16)
                except ValueError:
                    continue
                if b > a:
                    ranges.append((a, b - a))
        return ranges

    def release_dram(pid: int):
        """process_madvise(MADV_PAGEOUT) over the target's file-backed mappings.

        An operator escape hatch comes first: VIDGEN_LLM_RECLAIM_CMD is run with
        the pid appended, for setups that prefer cgroup memory.reclaim (v2) or a
        privileged helper. Returns (ok, human-readable reason) and never raises —
        on Linux a failed trim only forfeits an optimisation (see module docstring).
        """
        cmd = os.environ.get("VIDGEN_LLM_RECLAIM_CMD", "").strip()
        if cmd:
            try:
                r = subprocess.run(cmd.split() + [str(pid)], capture_output=True,
                                   text=True, timeout=60)
                return r.returncode == 0, "VIDGEN_LLM_RECLAIM_CMD rc=%d %s" % (
                    r.returncode, (r.stderr or "").strip()[:200])
            except Exception as e:
                return False, "VIDGEN_LLM_RECLAIM_CMD failed: %r" % (e,)

        libc = ctypes.CDLL(None, use_errno=True)
        pidfd = libc.syscall(_NR_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
        if pidfd < 0:
            err = ctypes.get_errno()
            if err == 38:  # ENOSYS
                return False, "pidfd_open unavailable (kernel < 5.3)"
            return False, "pidfd_open failed: %s" % os.strerror(err)
        try:
            ranges = _file_backed_ranges(pid)
            if not ranges:
                return False, "no file-backed mappings to page out"
            done = 0
            for i in range(0, len(ranges), _IOV_MAX):
                chunk = ranges[i:i + _IOV_MAX]
                arr = (_IOVEC * len(chunk))()
                for j, (base, ln) in enumerate(chunk):
                    arr[j].iov_base = ctypes.c_void_p(base)
                    arr[j].iov_len = ctypes.c_size_t(ln)
                n = libc.syscall(_NR_PROCESS_MADVISE, ctypes.c_int(pidfd),
                                 ctypes.byref(arr), ctypes.c_size_t(len(chunk)),
                                 ctypes.c_int(_MADV_PAGEOUT), ctypes.c_uint(0))
                if n < 0:
                    err = ctypes.get_errno()
                    hint = ""
                    if err == 1:    # EPERM
                        hint = (" -- MADV_PAGEOUT against another process needs "
                                "CAP_SYS_NICE, and being its parent does not "
                                "earn it: measured refused for our own child at "
                                "ptrace_scope=1, and accepted for the same child "
                                "with the capability. Grant it, or set "
                                "VIDGEN_LLM_RECLAIM_CMD. Harmless either way -- "
                                "Linux reclaims these clean pages under pressure "
                                "on its own.")
                    elif err == 38:  # ENOSYS
                        hint = " -- process_madvise needs kernel >= 5.10"
                    return (done > 0), "process_madvise: %s%s" % (os.strerror(err), hint)
                done += n
            return True, "process_madvise paged out %.1f GB of file-backed pages" % (done / GB)
        finally:
            os.close(pidfd)

    def pid_alive(pid: int) -> bool:
        try:
            st = _status(int(pid))
        except (OSError, ValueError):
            return False
        return not str(st.get("State", "")).startswith("Z")  # zombie != alive

    def proc_cmdline(name: str) -> str:
        pid = find_pid(name)
        if not pid:
            return ""
        try:
            with open(f"{_PROC}/{pid}/cmdline", "rb") as f:
                return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            return ""

    def detach_kwargs() -> dict:
        # setsid: the worker outlives the terminal that launched it, matching
        # what DETACHED_PROCESS buys us on Windows.
        return {"start_new_session": True, "close_fds": True}


# ==========================================================================
# shared
# ==========================================================================
def descendants(root: int) -> list:
    """Every pid below `root`, nearest first. Both OSes, one implementation.

    Recursive, not direct children only. A backend that spawns its engine
    through a worker process puts the interesting process TWO levels down, and
    a children-only walk finds the worker and misses the engine entirely --
    which is the shape actually measured on the reference box.

    Parentage is confirmed by age. Windows does not clear a dead parent's pid
    from an orphan's process entry, so a process whose parent exited keeps
    pointing at a pid the OS may since have handed to something else -- and if
    that something else is the backend we are walking down from, an unrelated
    process joins the list. A real child cannot be older than its parent, so
    that check throws them out. When a start time cannot be read the edge is
    KEPT: failing to find the engine turns the handover off silently, which is
    worse than trimming one working set we did not have to.
    """
    tree = proc_tree()
    kids: dict = {}
    for pid, ppid in tree.items():
        if pid != ppid:                  # pid 0/1 and System parent themselves
            kids.setdefault(ppid, []).append(pid)

    cache: dict = {}

    def younger_than(child: int, parent: int) -> bool:
        for pid in (child, parent):
            if pid not in cache:
                cache[pid] = proc_start(pid)
        c, p = cache[child], cache[parent]
        return True if c is None or p is None else c >= p

    out, seen, queue = [], {root}, [root]
    while queue:                          # breadth-first: nearest kin first
        parent = queue.pop(0)
        for child in sorted(kids.get(parent, ())):
            if child in seen or not younger_than(child, parent):
                continue                  # `seen` also breaks a recycled cycle
            seen.add(child)
            out.append(child)
            queue.append(child)
    return out


# ==========================================================================
# GPU memory — vendor-agnostic
# ==========================================================================
# The planner only ever needs one thing from the GPU: how much is already spoken
# for, and how much is left. That question has nothing to do with the vendor, so
# the measurement is a list of interchangeable sources rather than one hardcoded
# tool. The first that answers wins and is remembered for the process lifetime.
#
# Order is by trustworthiness, not convenience: vendor tooling first, service
# APIs only once none of it is installed.
#
# A vendor probe reads the driver and therefore sees every process on the card,
# including ones we know nothing about — that is the number we must budget
# against. Within Intel, `intel-ze` (Level Zero sysman, the runtime the driver
# itself ships) comes before `xpu-smi`, because the loader is present wherever
# an Intel GPU is usable while the CLI is a separate optional install.
#
# The last two are service endpoints, used only as a safety net. ComfyUI's
# /system_stats reports whatever backend torch is on (cuda, xpu, ...) and so
# works on hardware this pack has never run on; its weakness is that on
# Windows/WDDM the free figure can read high. The TRUSTA-AST backend's
# /system/resources is the final answer when no tooling exists at all — it is
# asked over its own API, never by inspecting the inference process it manages.

_gpu_source = None          # remembered after the first success


def _gpu_from_cmd(cmd):
    """Operator-supplied probe. Must print `total,used,free` in MiB.

    The guaranteed path on any accelerator we have not built in: point it at
    whatever your platform's tool is and reshape the output, e.g.
        VIDGEN_GPU_QUERY_CMD=/usr/local/bin/my-gpu-mib.sh
    """
    out = subprocess.check_output(cmd, shell=True, text=True, timeout=15)
    t, u, f = [int(float(x)) for x in out.strip().splitlines()[0].split(",")[:3]]
    return {"total_mb": t, "used_mb": u, "free_mb": f, "source": "VIDGEN_GPU_QUERY_CMD"}


def _gpu_nvidia():
    """The card VIDGEN_GPU_INDEX names (nvidia-smi's numbering), else the first.

    Without the index this read GPU 0 whatever ComfyUI was given, so on a box with
    two NVIDIA cards the reserve was sized from the wrong one.
    """
    cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
           "--format=csv,noheader,nounits"]
    index = os.environ.get("VIDGEN_GPU_INDEX", "").strip()
    if index.isdigit():
        cmd += ["-i", index]
    out = subprocess.check_output(cmd, text=True, timeout=10)
    t, u, f = [int(x) for x in out.strip().splitlines()[0].split(",")]
    return {"total_mb": t, "used_mb": u, "free_mb": f,
            "source": "nvidia-smi" + (":" + index if index.isdigit() else "")}


def _ze_probe():
    """The raw Level Zero sysman read. Runs in a child process -- see below.

    `libze_loader` ships with any usable Intel GPU install, so this answers on
    boxes where the optional `xpu-smi` package was never added. Only the entry
    points needed for a memory read are bound.

    zesMemoryGetState reports one figure per memory module, so a card with
    several stacks has to be summed rather than sampled.
    """
    # Read by zeInit on the legacy path, so it must be set before we call it.
    os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")

    names = ["ze_loader.dll"] if WINDOWS else ["libze_loader.so.1",
                                               "libze_loader.so"]
    lib = None
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError("Level Zero loader not found (%s)" % ", ".join(names))
    if not hasattr(lib, "zesDeviceEnumMemoryModules"):
        raise RuntimeError("loader has no sysman entry points")

    class _MemState(ctypes.Structure):
        _fields_ = [("stype", ctypes.c_int), ("pNext", ctypes.c_void_p),
                    ("health", ctypes.c_int), ("free", ctypes.c_uint64),
                    ("size", ctypes.c_uint64)]

    ZES_STRUCTURE_TYPE_MEM_STATE = 0x1e

    # Every signature is declared. Without argtypes ctypes narrows pointer
    # arguments to 32 bits on win64, and the first out-parameter write lands on
    # a truncated address -- an access violation inside the driver rather than
    # the clean "no Intel GPU here" error this is supposed to produce.
    _u32p = ctypes.POINTER(ctypes.c_uint32)
    _handlep = ctypes.POINTER(ctypes.c_void_p)

    def bind(name, argtypes):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    def call(fn, *args):
        rc = fn(*args)
        if rc != 0:
            raise RuntimeError("%s returned 0x%x" % (fn.__name__, rc))

    # zesInit is the modern entry point; older loaders expose only zeInit and
    # promote the core handles to sysman ones via ZES_ENABLE_SYSMAN above.
    if hasattr(lib, "zesInit"):
        call(bind("zesInit", [ctypes.c_uint32]), 0)
        driver_get = bind("zesDriverGet", [_u32p, _handlep])
        device_get = bind("zesDeviceGet", [ctypes.c_void_p, _u32p, _handlep])
    else:
        call(bind("zeInit", [ctypes.c_uint32]), 0)
        driver_get = bind("zeDriverGet", [_u32p, _handlep])
        device_get = bind("zeDeviceGet", [ctypes.c_void_p, _u32p, _handlep])

    enum_mem = bind("zesDeviceEnumMemoryModules",
                    [ctypes.c_void_p, _u32p, _handlep])
    mem_state = bind("zesMemoryGetState",
                     [ctypes.c_void_p, ctypes.POINTER(_MemState)])

    n_drv = ctypes.c_uint32(0)
    call(driver_get, ctypes.byref(n_drv), None)
    if not n_drv.value:
        raise RuntimeError("no Level Zero drivers")
    drivers = (ctypes.c_void_p * n_drv.value)()
    call(driver_get, ctypes.byref(n_drv), drivers)

    total = free = 0
    for drv in drivers:
        n_dev = ctypes.c_uint32(0)
        if device_get(drv, ctypes.byref(n_dev), None) or not n_dev.value:
            continue
        devices = (ctypes.c_void_p * n_dev.value)()
        if device_get(drv, ctypes.byref(n_dev), devices):
            continue
        for dev in devices:
            n_mod = ctypes.c_uint32(0)
            if enum_mem(dev, ctypes.byref(n_mod), None):
                continue
            if not n_mod.value:
                continue
            mods = (ctypes.c_void_p * n_mod.value)()
            if enum_mem(dev, ctypes.byref(n_mod), mods):
                continue
            for m in mods:
                st = _MemState()
                st.stype = ZES_STRUCTURE_TYPE_MEM_STATE
                if mem_state(m, ctypes.byref(st)):
                    continue
                total += int(st.size)
                free += int(st.free)
            if total:
                break          # first device that answered is the one we budget against
        if total:
            break

    if total <= 0:
        raise RuntimeError("Level Zero reported no memory modules")
    total_mb = total // 2 ** 20
    free_mb = free // 2 ** 20
    return {"total_mb": int(total_mb), "used_mb": int(total_mb - free_mb),
            "free_mb": int(free_mb), "source": "intel-ze"}


def _gpu_intel_ze():
    """Intel via Level Zero sysman, run in a child process on purpose.

    A hybrid-graphics box can have `ze_loader.dll` from the Intel display driver
    with no usable Level Zero driver behind it, and on the reference laptop
    `zesInit` faults inside the loader instead of returning an error. ctypes
    turns that into an OSError so we do survive it, but an access violation
    inside a driver is undefined behaviour and this process goes on to make
    memory decisions afterwards -- so the fault is kept out of it entirely.

    The cost is one interpreter spawn, and only on a box where nvidia-smi did
    not already answer. `--intel-ze` below is the child half.
    """
    out = subprocess.check_output([sys.executable, os.path.abspath(__file__),
                                   "--intel-ze"],
                                  text=True, timeout=30,
                                  stderr=subprocess.DEVNULL)
    res = json.loads(out)
    if "error" in res:
        raise RuntimeError(res["error"])
    return res


def _gpu_intel():
    """Intel via xpu-smi. Key names have moved between releases, so this walks
    the JSON for memory-shaped fields instead of trusting one schema."""
    out = subprocess.check_output(["xpu-smi", "stats", "-d", "0", "-j"],
                                  text=True, timeout=15)
    data = json.loads(out)
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if isinstance(v, (int, float)):
                    if "memory" in kl and "used" in kl:
                        found.setdefault("used", float(v))
                    elif "memory" in kl and ("total" in kl or "physical" in kl
                                             or "size" in kl):
                        found.setdefault("total", float(v))
                    elif "memory" in kl and "free" in kl:
                        found.setdefault("free", float(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    if "total" not in found or ("used" not in found and "free" not in found):
        raise RuntimeError("xpu-smi output had no usable memory fields")

    def mib(v):
        """Normalise one field. Units are decided per value, not once for the
        payload: xpu-smi has been seen reporting the physical size in bytes
        alongside a used figure already in MiB, and a shared scale factor turns
        the smaller of the two into zero."""
        return v / 1024.0 ** 2 if v > 10 ** 9 else v

    total = mib(found["total"])
    used = mib(found["used"]) if "used" in found else total - mib(found["free"])
    free = mib(found["free"]) if "free" in found else total - used
    return {"total_mb": int(total), "used_mb": int(used), "free_mb": int(free),
            "source": "xpu-smi"}


def _gpu_amd():
    out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"],
                                  text=True, timeout=15)
    data = json.loads(out)
    card = next(iter(data.values()))
    total = used = None
    for k, v in card.items():
        kl = k.lower()
        if "total" in kl and "vram" in kl:
            total = float(v)
        elif "used" in kl and "vram" in kl:
            used = float(v)
    if total is None or used is None:
        raise RuntimeError("rocm-smi output had no usable memory fields")
    total /= 1024.0 ** 2
    used /= 1024.0 ** 2
    return {"total_mb": int(total), "used_mb": int(used),
            "free_mb": int(total - used), "source": "rocm-smi"}


def _gpu_comfy():
    """ComfyUI's own /system_stats — whatever backend torch is running on.

    This is what makes an unfamiliar accelerator work with no configuration:
    on an Intel box it reports type 'xpu' with real numbers, on CUDA 'cuda'.
    Read from the environment rather than importing paths.py, so this module
    stays dependency-free (paths.py folds the conf into os.environ on import).
    """
    url = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
    import urllib.request
    with urllib.request.urlopen(url + "/system_stats", timeout=6) as r:
        data = json.loads(r.read().decode())
    devs = data.get("devices") or []
    if not devs:
        raise RuntimeError("no devices in /system_stats")
    d = devs[0]
    total = int(d["vram_total"]) // 2 ** 20
    free = int(d["vram_free"]) // 2 ** 20
    return {"total_mb": total, "used_mb": total - free, "free_mb": free,
            "source": "comfyui:%s" % d.get("type", "?")}


def _gpu_ast():
    """The TRUSTA-AST backend's /system/resources -- last resort, via its API.

    The point of going through the service rather than the inference process it
    spawns: the backend already owns a vendor-neutral probe (NVML, then
    nvidia-smi, then torch.xpu), and it keeps answering when the model it hosts
    is swapped, reloaded, or replaced by an entirely different runtime. Reading
    the worker process directly would bind this pack to whichever engine happens
    to be loaded today.

    mode=usage is required: in spec mode the backend fills in total_gb only.
    """
    url = os.environ.get("VIDGEN_AST_URL", "http://127.0.0.1:8571").rstrip("/")
    import urllib.request
    with urllib.request.urlopen(url + "/system/resources?mode=usage",
                                timeout=6) as r:
        data = json.loads(r.read().decode())
    gpus = (data.get("gpu") or {}).get("gpus") or []
    if not gpus:
        raise RuntimeError("no GPUs in /system/resources")
    # The backend lists every card. Take the one of the vendor being measured,
    # rather than whichever it happened to list first.
    vendor = os.environ.get("VIDGEN_GPU_VENDOR", "").strip().lower()
    matching = [g for g in gpus if str(g.get("vendor", "")).lower() == vendor] if vendor else []
    if vendor and not matching:
        raise RuntimeError("no %s GPU in /system/resources" % vendor)
    g = (matching or gpus)[0]
    total = float(g.get("total_gb") or 0) * 1024.0
    if total <= 0:
        raise RuntimeError("/system/resources reported no VRAM total")
    used = g.get("used_gb")
    free = g.get("free_gb")
    if used is None and free is None:
        raise RuntimeError("/system/resources returned neither used nor free")
    used_mb = float(used) * 1024.0 if used is not None else total - float(free) * 1024.0
    free_mb = float(free) * 1024.0 if free is not None else total - used_mb
    return {"total_mb": int(total), "used_mb": int(used_mb),
            "free_mb": int(free_mb), "source": "trusta-ast"}


def gpu_vram():
    """Total/used/free VRAM in MiB, plus which source answered.

    `used` is the whole point: the planner spends only what is left after every
    other process on the card, so a source that can see other processes is
    preferred over one that can only see our own allocations.
    """
    global _gpu_source
    cmd = os.environ.get("VIDGEN_GPU_QUERY_CMD", "").strip()
    sources = []
    if cmd:
        sources.append(("cmd", lambda: _gpu_from_cmd(cmd)))
    vendor_tools = {"nvidia": [("nvidia", _gpu_nvidia)],
                    "intel": [("intel-ze", _gpu_intel_ze), ("intel", _gpu_intel)],
                    "amd": [("amd", _gpu_amd)]}
    # VIDGEN_GPU_VENDOR names the card ComfyUI runs on (paths.py fills it in from
    # AI Scaler Toolkit's selected.json). Then only that vendor's tools are asked:
    # on a machine with an NVIDIA card and an Intel GPU, nvidia-smi answers first
    # and reports the NVIDIA card while ComfyUI is on the Intel one -- which is how
    # a reserve came to be sized from the LLM's card instead of ComfyUI's.
    vendor = os.environ.get("VIDGEN_GPU_VENDOR", "").strip().lower()
    if vendor in vendor_tools:
        sources += vendor_tools[vendor]
    else:
        sources += [t for tools in vendor_tools.values() for t in tools]
    sources += [("comfy", _gpu_comfy), ("ast", _gpu_ast)]
    if _gpu_source:                       # remembered winner goes first
        sources.sort(key=lambda s: s[0] != _gpu_source)

    errors = []
    for name, fn in sources:
        try:
            res = fn()
            if res.get("total_mb", 0) > 0:
                _gpu_source = name
                return res
            errors.append("%s: reported 0 total" % name)
        except Exception as e:
            errors.append("%s: %s" % (name, e))
    return {"error": "no GPU memory source worked -- " + "; ".join(errors)}


def proc_vram_gb(pids) -> float | None:
    """Dedicated VRAM held by these processes, in GB; None if it cannot be read.

    Needed for an LLM that will be put to sleep for a render: the backend may
    report nothing (AST reports 0 for Strata, which measured 14.3 GB) and the
    planner has to know how much of the card it is about to get back.

    Windows/WDDM: nvidia-smi shows N/A per process, so this reads the
    "GPU Process Memory" performance counters, which do see it. Elsewhere,
    nvidia-smi's compute-apps list has the figure directly.
    """
    want = {int(p) for p in pids if p}
    if not want:
        return None
    try:
        if WINDOWS:
            ps = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction "
                  "SilentlyContinue).CounterSamples | ForEach-Object { "
                  "$_.InstanceName + ' ' + [int64]$_.CookedValue }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=20).stdout
            total = 0
            for line in out.splitlines():
                name, _, val = line.strip().rpartition(" ")
                if name.startswith("pid_"):
                    pid = int(name.split("_")[1])
                    if pid in want:
                        total += int(val)
            return round(total / 1024.0 ** 3, 2)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        total = sum(float(v) for p, v in (l.split(",") for l in out.splitlines() if "," in l)
                    if int(p) in want)
        return round(total / 1024.0, 2)
    except Exception:
        return None


def proc_vram_cards(pids) -> dict | None:
    """{nvidia-smi index: GB} these processes hold on each card; None if unknown.

    Tells a two-card box apart from a one-card box: an LLM on one card is no
    reason to sleep it, or hold anything back for it, when ComfyUI renders on the
    other. Only nvidia-smi names the card per process, so None -- Windows/WDDM,
    which shows N/A here, or no NVIDIA tooling -- means "cannot tell", and the
    caller keeps assuming one shared card.
    """
    want = {int(p) for p in pids if p}
    if not want or WINDOWS:
        return None
    try:
        run = lambda q: subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout
        index = {u.strip(): i.strip() for i, u in
                 (l.split(",") for l in run("--query-gpu=index,uuid").splitlines() if "," in l)}
        cards: dict = {}
        for line in run("--query-compute-apps=gpu_uuid,pid,used_memory").splitlines():
            uuid, pid, mib = [f.strip() for f in line.split(",")]
            if int(pid) in want:
                card = index[uuid]
                cards[card] = round(cards.get(card, 0.0) + float(mib) / 1024.0, 2)
        return cards
    except Exception:
        return None


def snapshot() -> dict:
    return {"gpu": gpu_vram(), "ram": system_ram()}


if __name__ == "__main__":
    if "--intel-ze" in sys.argv:
        # Child half of _gpu_intel_ze(). Any fault dies here, in a process that
        # holds no planning state, and the parent sees a failed source.
        try:
            print(json.dumps(_ze_probe()))
        except Exception as e:
            print(json.dumps({"error": "%s: %s" % (type(e).__name__, e)}))
        sys.exit(0)
    print(json.dumps(snapshot(), indent=2))
