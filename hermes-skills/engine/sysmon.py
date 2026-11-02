"""Real GPU VRAM and system RAM, measured from outside the process.

ComfyUI over-reports free VRAM on Windows (WDDM + cudaMallocAsync), and no
in-process API can see what *other* processes hold, so both numbers come from
the OS/driver instead. The platform-specific work lives in osmem.py; this
module is the stable name the rest of the engine imports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import osmem  # noqa: E402

gpu_vram = osmem.gpu_vram
system_ram = osmem.system_ram


def snapshot():
    return {"gpu": gpu_vram(), "ram": system_ram()}


if __name__ == "__main__":
    import json
    print(json.dumps(snapshot(), indent=2))
