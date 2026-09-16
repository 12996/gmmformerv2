#!/usr/bin/env python3
"""Print GPU occupancy on this host. Free = no compute process AND used < 2048 MiB."""
import subprocess

q = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
try:
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
except subprocess.CalledProcessError:
    apps = ""

gpus = []
for line in q.splitlines():
    idx, uid, used, total, util = [x.strip() for x in line.split(",")]
    gpus.append(
        {
            "idx": idx,
            "uid": uid,
            "used": float(used),
            "total": float(total),
            "util": float(util),
            "procs": [],
        }
    )
gmap = {g["uid"]: g for g in gpus}
for line in apps.splitlines():
    if not line.strip():
        continue
    uid, pid, name, mem = [x.strip() for x in line.split(",", 3)]
    user = "?"
    cmd = "?"
    try:
        out = subprocess.check_output(["ps", "-p", pid, "-o", "user=,cmd="], text=True).strip()
        if out:
            user, cmd = out.split(None, 1)
    except Exception:
        pass
    if uid in gmap:
        gmap[uid]["procs"].append((pid, user, mem, name, cmd[:160]))

print("FREE_RULE: no compute process AND used < 2048 MiB; never GPU2 if fanhaipeng asr")
free = []
for g in gpus:
    procs = g["procs"]
    is_free = len(procs) == 0 and g["used"] < 2048
    asr = any(("fanhaipeng" in p[1]) or ("/asr/" in p[4]) for p in procs)
    skip2 = g["idx"] == "2" and asr
    print(
        "GPU%s used=%.0fMiB/%.0f util=%.0f%% nproc=%d free=%s asr=%s skip2=%s"
        % (g["idx"], g["used"], g["total"], g["util"], len(procs), is_free, asr, skip2)
    )
    for pid, user, mem, name, cmd in procs:
        print("  pid=%s user=%s mem=%s name=%s cmd=%s" % (pid, user, mem, name, cmd))
    if is_free and not skip2:
        free.append(g["idx"])
print("CANDIDATES", ",".join(free) if free else "none")
