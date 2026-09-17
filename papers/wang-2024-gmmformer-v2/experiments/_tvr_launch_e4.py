#!/usr/bin/env python3
"""Fix E4 ntrain check and launch E4 on a GPU that is not T1's GPU6."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tvr_after_scp import launch, log, scp_file, ssh_ok, pick_gpus

ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
T1_GPU = "6"


def main() -> int:
    scp_file("tvr_e4_ntrain_check.py")
    log("NTRAIN_E4")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/tvr_e4_ntrain_check.py" % (ROOT, ROOT),
        timeout=1800,
    )
    if "E4_NTRAIN_CHECK_OK" not in out:
        raise RuntimeError("E4_NTRAIN_FAIL")
    gpus = pick_gpus(4)
    log("GPUS %s" % gpus)
    e4_gpu = None
    for g in gpus:
        if g not in (T1_GPU, "2"):
            e4_gpu = g
            break
    if e4_gpu is None:
        print("E4_BLOCKED no leftover gpu after T1", flush=True)
        return 1
    log("LAUNCH_E4 gpu=%s" % e4_gpu)
    launch("tvr_e4", e4_gpu)
    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
