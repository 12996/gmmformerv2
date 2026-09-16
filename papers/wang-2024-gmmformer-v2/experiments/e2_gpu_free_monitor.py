#!/usr/bin/env python3
"""Wake only when 5090_1 has a free GPU or E2 already launched. Silent otherwise."""
from __future__ import annotations

import subprocess
import time

HOST = "5090_1"
POLL_S = 30
LOG = r"F:\论文\papers\wang-2024-gmmformer-v2\experiments\e2_gpu_free_monitor.log"


def ssh(cmd: str, timeout: int = 40) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        raise RuntimeError("rc=%s %s" % (r.returncode, out[:500]))
    return out


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")


def main() -> int:
    fails = 0
    while True:
        try:
            occ = ssh("python3 /data/zhaopu/wang-2024-gmmformer-v2/experiments/e2_gpu_occupancy.py")
            ps = ssh(
                "ps -u zhaopu -o pid,cmd | grep -E 'main.py -d cha|run_cha_e2_gpu' | grep -v grep || true"
            )
            fails = 0
        except Exception as e:
            fails += 1
            log("ssh_fail %s %s" % (fails, e))
            if fails >= 5:
                print("FAILED ssh occupancy")
                return 1
            time.sleep(POLL_S)
            continue
        log(occ.replace("\n", " | ")[:1000])
        cands = []
        for line in occ.splitlines():
            if line.startswith("CANDIDATES "):
                rest = line.split(" ", 1)[1].strip()
                if rest not in ("", "none"):
                    cands = [x for x in rest.split(",") if x and x != "2"]
        if ps:
            print("DONE e2 process already running")
            return 0
        if cands:
            print("DONE free GPU " + ",".join(cands))
            return 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
