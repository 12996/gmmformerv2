#!/usr/bin/env python3
"""Poll 5090_1 for a free GPU and launch E1 seed 9527 once."""
from __future__ import annotations

import subprocess
import sys
import time

HOST = "5090_1"
POLL_S = 150
MAX_WAIT_S = 30 * 60
SEED = "9527"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"


def ssh(cmd: str, timeout: int = 40) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def occupancy() -> str:
    r = ssh("python3 /tmp/e1_gpu_occupancy.py")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("occupancy rc=%s\n%s" % (r.returncode, out))
    return out


def parse_candidates(text: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith("CANDIDATES "):
            rest = line.split(" ", 1)[1].strip()
            if rest == "none" or not rest:
                return []
            return [x for x in rest.split(",") if x and x != "2"]
    return []


def launch(gpu: str) -> str:
    remote = (
        "set -euo pipefail; "
        "ROOT=%s; "
        "GPU=%s; "
        "SEED=%s; "
        "mkdir -p \"$ROOT/logs/runs/tc_seed${SEED}_gpu${GPU}\" \"$ROOT/logs\"; "
        "nohup \"$ROOT/run_cha_tc_gpu.sh\" \"$GPU\" \"$SEED\" "
        "> \"$ROOT/logs/runs/tc_seed${SEED}_gpu${GPU}/launcher.out\" 2>&1 & "
        "echo LAUNCHER_PID=$!; "
        "sleep 4; "
        "echo '==== pid file ===='; "
        "cat \"$ROOT/logs/runs/tc_seed${SEED}_gpu${GPU}/pid\"; "
        "echo '==== run.json ===='; "
        "cat \"$ROOT/logs/runs/tc_seed${SEED}_gpu${GPU}/run.json\" || true; "
        "echo '==== ps ===='; "
        "ps -u zhaopu -o pid,ppid,cmd | grep -E 'main.py -d cha|run_cha_tc_gpu' | grep -v grep || true; "
        "echo '==== PRVR_ROOT exists ===='; "
        "ls -ld \"$ROOT/tmp/i3d_tc_seed${SEED}\" \"$ROOT/tmp/i3d_seed${SEED}\"; "
        "echo '==== occupancy after launch ===='; "
        "python3 /tmp/e1_gpu_occupancy.py"
    ) % (ROOT, gpu, SEED)
    r = ssh(remote, timeout=60)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("launch rc=%s\n%s" % (r.returncode, out))
    return out


def main() -> int:
    t0 = time.time()
    n = 0
    while True:
        n += 1
        elapsed = time.time() - t0
        try:
            text = occupancy()
        except Exception as e:
            print("POLL %s FAIL elapsed=%.0fs %s" % (n, elapsed, e), flush=True)
            text = ""
        print("POLL %s elapsed=%.0fs" % (n, elapsed), flush=True)
        if text:
            print(text, flush=True)
        cands = parse_candidates(text) if text else []
        if cands:
            gpu = cands[0]
            print("LAUNCH gpu=%s seed=%s" % (gpu, SEED), flush=True)
            print(launch(gpu), flush=True)
            return 0
        if elapsed >= MAX_WAIT_S:
            print("BLOCKED no free GPU after %.0fs" % elapsed, flush=True)
            return 2
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
