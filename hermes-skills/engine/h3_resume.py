"""Resume an interrupted H3 chained job.

A chained render can lose its place if ComfyUI restarts mid-run: the worker is
polling a prompt_id that no longer exists in the fresh server's history, so it
spins until its timeout while the finished segments sit safely on disk.

This picks up from whatever segments already exist, renders the rest chained on
the last available frame, and joins everything.

  python h3_resume.py <job_id>
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memplan          # noqa: E402
import video_gen as vg  # noqa: E402

job_id = sys.argv[1]
job_path = os.path.join(HERE, "jobs", job_id + ".json")
job = json.load(open(job_path, encoding="utf-8"))
req = job["req"]

plan = memplan.choose(req)
if not plan.get("ok"):
    raise SystemExit("planner refuses now: " + plan["refuse_reason"])
n = plan["segments"]

# Identify the stamp this run used. The worker log names it explicitly, which
# is far more reliable than comparing mtimes (the job file is rewritten on every
# status change, so it can easily be newer than the segments it describes).
stamp = sys.argv[2] if len(sys.argv) > 2 else None
if stamp is None:
    log_path = os.path.join(HERE, "jobs", job_id + ".log")
    if os.path.exists(log_path):
        txt = open(log_path, encoding="utf-8", errors="replace").read()
        m = re.search(r"h3(?:chain|resume)_([0-9a-f]{8})_\d+\.png", txt)
        if m:
            stamp = m.group(1)
if stamp is None:
    # last resort: newest seg1 on disk
    segs = sorted(glob.glob(os.path.join(vg.COMFY_OUTPUT, "video", "h3_*_seg1_*.mp4")),
                  key=os.path.getmtime)
    if segs:
        stamp = re.search(r"h3_([0-9a-f]{8})_seg1_", os.path.basename(segs[-1])).group(1)
if stamp is None:
    raise SystemExit("could not identify the run's stamp; pass it as argv[2]")

have = []
for i in range(1, n + 1):
    g = sorted(glob.glob(os.path.join(vg.COMFY_OUTPUT, "video",
                                      "h3_%s_seg%d_*.mp4" % (stamp, i))))
    if g:
        have.append(g[-1])
    else:
        break
print("stamp %s: %d/%d segments already on disk" % (stamp, len(have), n), flush=True)
if len(have) == n:
    print("nothing to render; joining", flush=True)

first_frame = None
for i in range(len(have), n):
    if first_frame is None:
        name = "h3chain_%s_%d.png" % (stamp, i)          # produced by the original run
        if not os.path.exists(os.path.join(vg.COMFY_INPUT, name)):
            name = "h3resume_%s_%d.png" % (stamp, i)
            vg._media("lastframe", have[-1], os.path.join(vg.COMFY_INPUT, name))
        first_frame = name
        print("chaining segment %d on %s" % (i + 1, first_frame), flush=True)
    graph = vg.build_h3_graph(plan, req, seg_idx=i, first_frame=first_frame,
                              prefix="video/h3_%s_seg%d" % (stamp, i + 1))
    out = vg._submit_and_wait(graph, "resume seg%d/%d" % (i + 1, n))
    have.append(out)
    if i + 1 < n:
        nm = "h3resume_%s_%d.png" % (stamp, i + 1)
        vg._media("lastframe", out, os.path.join(vg.COMFY_INPUT, nm))
        first_frame = nm

dst = os.path.join(vg.COMFY_OUTPUT, "video", "h3_%s_joined.mp4" % stamp)
vg._media("concat", dst, *have)
job.update(status="done", file=dst, error=None, resumed=True)
json.dump(job, open(job_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"ok": True, "file": dst, "segments": have}, ensure_ascii=False, indent=2))
