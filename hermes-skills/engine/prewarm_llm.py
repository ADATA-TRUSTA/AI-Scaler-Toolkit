"""Refill the page cache with the LLM's GGUF after a render.

Why this exists
---------------
vidgen trims the LLM server's working set so ComfyUI can have the DRAM. The LLM's
weights are mmap'd, so its next inference faults them back one scattered page at
a time. Measured on this box (2026-09-07, Qwen3.5-35B-A3B-Q4_K_M, 20.5 GB):

    page-fault refill (what happens today) :   ~477 MB/s   -> 39.7 s cold turn
    sequential refill, 2 threads, buffered : ~1,785 MB/s   -> 11.8 s, then a
                                                              16.1 s cold turn

Same drive, same bytes -- only the access pattern differs. The scattered faults
run at QD1 in 4-64 KB pieces, which is this SSD's worst regime; a sequential
stream is its best. Reading the file with ordinary buffered I/O repopulates the
OS page cache, so those faults become RAM hits instead of disk reads.

Two measured details worth keeping:
  * 2 threads saturate the drive (1 thread only reaches ~1,047 MB/s). More
    threads do not help.
  * Read the WHOLE file. Prewarming only the tail is strictly worse -- the
    experts a turn touches are spread across the file, not clustered:
    20.5 GB -> 16.1 s ask,  16 GB -> 20.3 s,  13 GB -> 23.8 s,  10 GB -> 27.4 s.
  * Reads must be BUFFERED. Unbuffered I/O bypasses the cache and warms nothing.

Run it right after the post-render `/free`. It is pure read-only I/O: it never
touches the LLM server, never allocates the model, and cannot fail a render.

    python prewarm_llm.py            # blocking, ~12 s
    python prewarm_llm.py --detach   # fire and forget
"""
import argparse, os, subprocess, sys, threading, time

# The optimum is DRIVE-DEPENDENT -- re-measure after any drive change with
#   ssdbench.py --phases llm   (see prewarm_by_thread)
#   Gen3 ADATA IM2P33F8: 1 thr 1,097 | 2 thr 1,780 | 4 thr 1,761 MB/s -> use 2
#   Gen4 ADATA LEGEND 850: 1 thr 1,723 | 2 thr 3,108 | 4 thr 4,051 MB/s -> use 4
DEFAULT_THREADS = 4
CHUNK = 16 << 20


def find_gguf() -> str | None:
    """The GGUF the LLM is serving, as AST reports it. No hardcoded paths.

    Asked of AST rather than read off a process's command line: which program
    hosts the model, and how it spells its arguments, is AST's business and
    changes with the engine it runs (see astapi.py). The weight file is the one
    thing every engine that mmaps a GGUF has in common, and AST names it.

    Only a file that exists here counts. A backend on another OS (agent in WSL,
    LLM on the Windows host) reports a path this process cannot open -- and
    could not warm that host's page cache from in here anyway.
    """
    try:
        import paths  # noqa: F401 -- folds hermes-comfy.conf (VIDGEN_AST_URL) into the env
    except Exception:
        pass
    import astapi
    for value in astapi._walk_strings(astapi.status() or {}):
        value = value.strip()
        if value.lower().endswith(".gguf") and os.path.isfile(value):
            return value
    return None


def prewarm(path: str, threads: int = DEFAULT_THREADS) -> dict:
    """Buffered sequential read -> repopulates the page cache. Never raises."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"ok": False, "why": "cannot stat %s: %r" % (path, e)}

    per = size // threads
    read = [0] * threads

    def worker(i):
        try:
            with open(path, "rb") as f:          # buffered on purpose
                f.seek(i * per)
                want = per if i < threads - 1 else size - per * i
                got = 0
                while got < want:
                    b = f.read(min(CHUNK, want - got))
                    if not b:
                        break
                    got += len(b)
                read[i] = got
        except Exception:
            pass                                  # a partial warm is still a win

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker, args=(i,), daemon=True)
          for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    el = time.perf_counter() - t0
    tot = sum(read)
    return {"ok": True, "gb": round(tot / 2**30, 2), "seconds": round(el, 1),
            "mb_s": int(tot / 2**20 / el) if el > 0 else 0, "file": path,
            "threads": threads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="GGUF to warm (default: the one AST reports loaded)")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--detach", action="store_true",
                    help="spawn and return immediately")
    a = ap.parse_args()

    path = a.file or find_gguf()
    if not path:
        # A split install (agent in WSL, LLM on the Windows host) has no local
        # GGUF to open -- and could not warm the host's page cache
        # from in here even if it found one. That is not a fault, so say which
        # of the two situations this is instead of asking a question the caller
        # cannot act on.
        print("AST reports no GGUF that is on this machine, so there is no page "
              "cache here to warm; nothing to do. (Expected on a split install where the LLM "
              "runs on the host: prewarming only helps when the LLM and this "
              "process share an OS page cache. Pass --file to warm a specific "
              "GGUF anyway.)")
        return

    if a.detach:
        # creationflags is a Windows-only Popen kwarg; passing it on Linux is a
        # TypeError, not a no-op. setsid is the POSIX equivalent of detaching
        # from the parent's console/session.
        if sys.platform == "win32":
            detach = {"creationflags": 0x00000008 | 0x08000000}  # DETACHED|NO_WINDOW
        else:
            detach = {"start_new_session": True}
        subprocess.Popen([sys.executable, os.path.abspath(__file__),
                          "--file", path, "--threads", str(a.threads)],
                         close_fds=True, **detach)
        print("prewarm detached for", os.path.basename(path))
        return

    r = prewarm(path, a.threads)
    if r.get("ok"):
        print("prewarmed %s: %.2f GB in %.1fs (%s MB/s, %d threads)"
              % (os.path.basename(r["file"]), r["gb"], r["seconds"],
                 f"{r['mb_s']:,}", r["threads"]))
    else:
        print("prewarm skipped:", r.get("why"))


if __name__ == "__main__":
    main()
