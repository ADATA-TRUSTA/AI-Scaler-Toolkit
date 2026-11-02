"""Verify safetensors files are complete.

Downloads can silently truncate (curl may exit 0 on a cleanly-closed partial
transfer). The safetensors header declares every tensor's byte range, so the
minimum valid file size is computable — anything shorter is truncated.

Usage: python verify_safetensors.py <file> [<file> ...]
"""
import struct, json, os, sys


def check(path):
    if not os.path.exists(path):
        return {"file": path, "status": "MISSING"}
    size = os.path.getsize(path)
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 200_000_000:
                return {"file": path, "status": "BAD_HEADER", "size": size}
            hdr = json.loads(f.read(n))
    except Exception as e:
        return {"file": path, "status": "UNREADABLE", "error": str(e), "size": size}

    end = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        off = v.get("data_offsets")
        if off:
            end = max(end, off[1])
    need = 8 + n + end
    return {
        "file": path,
        "status": "COMPLETE" if size >= need else "TRUNCATED",
        "size": size,
        "need": need,
        "missing_gb": round((need - size) / 1e9, 2) if size < need else 0,
        "tensors": len(hdr),
    }


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        r = check(p)
        if r["status"] != "COMPLETE":
            bad += 1
        print("%-46s %-10s %s" % (
            os.path.basename(r["file"]), r["status"],
            ("missing %.2f GB" % r["missing_gb"]) if r.get("missing_gb") else
            ("%d tensors" % r["tensors"] if "tensors" in r else "")))
    sys.exit(1 if bad else 0)
