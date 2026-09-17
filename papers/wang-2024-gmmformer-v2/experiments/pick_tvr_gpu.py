#!/usr/bin/env python3
"""Pick leftover GPUs. Never GPU2. Prefer fully free, then GPU6."""
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
    if idx == "2":
        print("SKIP_GPU2 leftover=%.0f" % leftover)
        continue
    free = leftover >= 30000 and float(used) < 2048
    prefer6 = 1 if idx == "6" else 0
    print("LEFTOVER", idx, leftover, "free", free, "prefer6", prefer6)
    if leftover >= 8000:
        cands.append((1 if free else 0, prefer6, leftover, -float(util), idx))
cands.sort(reverse=True)
print("PICKS", ",".join(x[4] for x in cands) if cands else "none")
