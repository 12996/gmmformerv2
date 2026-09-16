"""Watch Charades CLIP trains on 5090_1. Print DONE/FAILED/ACTION_REQUIRED only."""
import subprocess
import sys
import time

REMOTE = r"""
bash /data/zhaopu/wang-2024-gmmformer-v2/check_train.sh
"""


def ssh():
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5090_1", REMOTE],
        capture_output=True,
        text=True,
        timeout=40,
    )
    return r.returncode, (r.stdout or "") + (r.stderr or "")


seen_step = False
n_runs = 0
while True:
    try:
        code, out = ssh()
    except subprocess.TimeoutExpired:
        time.sleep(30)
        continue
    lines = out.splitlines()
    runs = [ln for ln in lines if ln.startswith("RUN ")]
    procs = [ln for ln in lines if ln.startswith("PROC ")]
    n_runs = max(n_runs, len(runs))
    any_fail = any("fail=True" in ln for ln in runs)
    any_alive = any("alive=True" in ln for ln in runs) or bool(procs)
    any_loss = any("loss=(" in ln and "loss=None" not in ln for ln in runs)
    all_exited_ok = (
        n_runs > 0
        and all("alive=False" in ln for ln in runs)
        and all("fail=True" not in ln for ln in runs)
        and all(("exit=0" in ln) or ("Early Stop" in out) for ln in runs)
    )
    if any_fail or (n_runs > 0 and not any_alive and not any_loss):
        print("FAILED: train died: " + " ".join(out.split())[:800])
        sys.exit(1)
    if any_loss and not seen_step:
        seen_step = True
        print("ACTION_REQUIRED: first train step has a loss")
    if n_runs > 0 and not any_alive and seen_step:
        if all_exited_ok or "Early Stop" in out:
            print("DONE: train jobs finished")
            sys.exit(0)
        print("FAILED: processes gone: " + " ".join(out.split())[:800])
        sys.exit(1)
    time.sleep(30)
