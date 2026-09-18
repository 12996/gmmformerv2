#!/usr/bin/env python3
"""Poll 5090_1 sft06 log until first Best. Do not launch or kill."""
from __future__ import annotations

import subprocess
import sys
import time

HOST = "5090_1"
PID = "2019836"
REMOTE = r"""
set -e
LP=/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_e4_sft06_seed9527/results/tvr/gmmformer_v2/log.txt
echo ====GREP====
if [ -f "$LP" ]; then
  grep -E 'epoch:|Rsum:|Best:|Early Stop|New Best|CUDA out of memory|Traceback' "$LP" | tail -n 30 || true
else
  echo NO_LOG
fi
echo ====PS====
ps -p 2019836 -o pid,etime,stat,cmd || echo PID_DEAD
"""


def main() -> int:
    t0 = time.time()
    while time.time() - t0 < 900:
        r = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                HOST,
                "bash",
                "-lc",
                REMOTE,
            ],
            capture_output=True,
            text=True,
            timeout=40,
        )
        out = (r.stdout or "") + (r.stderr or "")
        print(time.strftime("%H:%M:%S"), "rc", r.returncode, flush=True)
        print(out[-2500:], flush=True)
        if "CUDA out of memory" in out:
            print("OOM", flush=True)
            return 2
        if "Traceback" in out and "Best: R@1:" not in out:
            print("TRACEBACK", flush=True)
            return 2
        if "Best: R@1:" in out:
            print("FIRST_BEST_OK", flush=True)
            return 0
        if "PID_DEAD" in out:
            print("PID_DEAD", flush=True)
            return 3
        time.sleep(25)
    print("TIMEOUT_NO_BEST", flush=True)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
