#!/usr/bin/env python3
"""Hold leftover VRAM on one GPU. Does not kill other processes."""
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("HOLD_GPU", "6")
import torch

free, total = torch.cuda.mem_get_info()
frac = float(os.environ.get("HOLD_FRAC", "0.80"))
nbytes = int(free * frac)
n = max(1, nbytes // 4)
x = torch.empty(n, device="cuda", dtype=torch.float32)
x.fill_(0)
held = n * 4
print(
    "HOLD_OK gpu=%s free_before=%s held_bytes=%s held_gib=%.2f pid=%s"
    % (os.environ["CUDA_VISIBLE_DEVICES"], free, held, held / 1024**3, os.getpid()),
    flush=True,
)
pid_path = os.environ.get(
    "HOLD_PIDFILE", "/data/zhaopu/wang-2024-gmmformer-v2/logs/e2_hold_gpu6.pid"
)
open(pid_path, "w").write(str(os.getpid()))
while True:
    time.sleep(3600)
    _ = float(x[0])
