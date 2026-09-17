#!/usr/bin/env python3
"""Pick leftover GPUs for ActivityNet. Never GPU2. Never GPU6/7 (live TVR)."""
import subprocess

q = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
cands = []
for line in q.splitlines():
    idx, used, total, util = [x.strip() for x in line.split(",")]
    leftover = float(total) - float(used)
    if idx in ("2", "6", "7"):
        print("SKIP_GPU%s leftover=%.0f" % (idx, leftover))
        continue
    free = leftover >= 30000 and float(used) < 2048
    print("LEFTOVER", idx, leftover, "free", free)
    if leftover >= 10000:
        cands.append((1 if free else 0, leftover, -float(util), idx))
cands.sort(reverse=True)
print("PICKS", ",".join(x[3] for x in cands) if cands else "none")
