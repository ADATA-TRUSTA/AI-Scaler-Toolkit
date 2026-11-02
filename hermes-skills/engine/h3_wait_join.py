"""Wait for the in-flight segment to land, then join a chained H3 run.

Used when the driving script died but ComfyUI kept rendering: the submission
outlives the client, so all that is missing is someone to notice and concat.

  python h3_wait_join.py <stamp> <expected_segments> [job_id]
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import video_gen as vg  # noqa: E402

stamp = sys.argv[1]
n = int(sys.argv[2])
job_id = sys.argv[3] if len(sys.argv) > 3 else None
VID = os.path.join(vg.COMFY_OUTPUT, "video")

if stamp == "__auto__":
    # The stamp is minted inside the worker, so it is unknowable until the run
    # writes its first chain frame. Wait for the worker's log to name it.
    import re
    log = os.path.join(HERE, "jobs", "%s.log" % job_id)
    for _ in range(180):                       # up to ~30 min for segment 1
        if os.path.exists(log):
            m = re.search(r"h3(?:chain|resume)_([0-9a-f]{8})_\d+\.png",
                          open(log, encoding="utf-8", errors="replace").read())
            if m:
                stamp = m.group(1)
                break
        print("[wait] stamp not known yet", flush=True)
        time.sleep(10)
    if stamp == "__auto__":
        raise SystemExit("could not determine the run's stamp from %s" % log)
    print("[wait] stamp = %s" % stamp, flush=True)


def segs_present():
    out = []
    for i in range(1, n + 1):
        g = sorted(glob.glob(os.path.join(VID, "h3_%s_seg%d_*.mp4" % (stamp, i))))
        if not g:
            break
        out.append(g[-1])
    return out


deadline = time.time() + 3600
while time.time() < deadline:
    have = segs_present()
    if len(have) >= n:
        break
    print("[wait] %d/%d segments" % (len(have), n), flush=True)
    time.sleep(20)

have = segs_present()
if len(have) < n:
    raise SystemExit("timed out with only %d/%d segments" % (len(have), n))

# The file can appear a moment before it is fully flushed.
time.sleep(3)
dst = os.path.join(VID, "h3_%s_joined.mp4" % stamp)
print("[join]", ", ".join(os.path.basename(p) for p in have), flush=True)
vg._media("concat", dst, *have)
print("[done]", dst, flush=True)

if job_id:
    jp = os.path.join(HERE, "jobs", job_id + ".json")
    if os.path.exists(jp):
        job = json.load(open(jp, encoding="utf-8"))
        job.update(status="done", file=dst, error=None, resumed=True)
        json.dump(job, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("[job] %s marked done" % job_id, flush=True)
