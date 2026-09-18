#!/usr/bin/env python3
"""SCP E4 lr=2e-4 files, fast-check, launch on leftover GPU 6 or 7. Never GPU2."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HOST = "5090_1"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
HERE = Path(__file__).resolve().parent
SEED = "9527"
FILES = [
    "tvr_e4_lr2e-4_5090_1.py",
    "run_tvr_e4_lr2e-4_gpu.sh",
    "tvr_e4_lr2e-4_ntrain_check.py",
    "pick_tvr_gpu.py",
]


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S "), msg, sep="", flush=True)


def run(cmd, timeout=60):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def ssh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    return run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            HOST,
            cmd,
        ],
        timeout=timeout,
    )


def ssh_ok(cmd: str, timeout: int = 120) -> str:
    rc, out = ssh(cmd, timeout=timeout)
    print(out, flush=True)
    if rc != 0:
        raise RuntimeError("ssh fail rc=%s\n%s" % (rc, out[-2000:]))
    return out


def scp_file(name: str) -> None:
    src = HERE / name
    dst = "%s:%s/experiments/%s" % (HOST, ROOT, name)
    log("SCP " + name)
    rc, out = run(
        ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", str(src), dst],
        timeout=90,
    )
    if rc != 0:
        raise RuntimeError("scp fail %s %s" % (name, out[-400:]))


def pick_gpu() -> str:
    out = ssh_ok("python3 %s/experiments/pick_tvr_gpu.py" % ROOT, timeout=40)
    picks = []
    for line in out.splitlines():
        if line.startswith("PICKS "):
            rest = line.split(" ", 1)[1].strip()
            if rest == "none" or not rest:
                break
            picks = [x for x in rest.split(",") if x and x != "2"]
    for prefer in ("6", "7", "5"):
        if prefer in picks:
            return prefer
    if picks:
        return picks[0]
    raise RuntimeError("NO_GPU")


def main() -> int:
    log("SCP")
    ssh_ok("mkdir -p %s/experiments %s/logs" % (ROOT, ROOT), timeout=30)
    for name in FILES:
        scp_file(name)
    log("DEPLOY_CFG")
    out = ssh_ok(
        r"""
set -euo pipefail
ROOT=%s
CFG="$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"
SRC="$ROOT/experiments/tvr_e4_lr2e-4_5090_1.py"
chmod +x "$ROOT/experiments/run_tvr_e4_lr2e-4_gpu.sh"
if [ ! -f "$CFG.bak_before_e4_lr2e-4" ]; then
  cp -a "$CFG" "$CFG.bak_before_e4_lr2e-4"
  echo BAK_OK
else
  echo BAK_KEEP
fi
cp -a "$SRC" "$CFG"
grep -n 'lr\|n_epoch\|q_feat_size\|visual_feature\|sft_factor\|hidden_size\|root' "$CFG"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_seed9527"
test ! -e "$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed9527/results/tvr/gmmformer_v2/log.txt"
test -f "$ROOT/data/prvr/tvr/TextData/tvr_clip_B32_proj.h5"
echo DEPLOY_OK
"""
        % ROOT,
        timeout=40,
    )
    if "DEPLOY_OK" not in out:
        raise RuntimeError("DEPLOY_FAIL")
    log("FAST_CHECK")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/tvr_e4_lr2e-4_ntrain_check.py --fast"
        % (ROOT, ROOT),
        timeout=180,
    )
    if "E4_LR2E4_FAST_CHECK_OK" not in out:
        raise RuntimeError("FAST_CHECK_FAIL")
    gpu = pick_gpu()
    if gpu == "2":
        raise RuntimeError("REFUSE GPU2")
    log("LAUNCH gpu=%s" % gpu)
    rundir = "tvr_e4_lr2e-4_seed%s_gpu%s" % (SEED, gpu)
    remote = r"""
set -euo pipefail
ROOT=%s
GPU=%s
SEED=%s
SH="$ROOT/experiments/run_tvr_e4_lr2e-4_gpu.sh"
RUN="$ROOT/logs/runs/%s"
mkdir -p "$RUN" "$ROOT/logs"
chmod +x "$SH"
nohup "$SH" "$GPU" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER_PID=$!
sleep 12
echo ====pid====
cat "$RUN/pid" || echo NO_PID
echo ====runjson====
cat "$RUN/run.json" || echo NO_RUNJSON
echo ====ps====
ps -u zhaopu -o pid,etime,cmd | grep -E 'main.py -d tvr|run_tvr_e4_lr2e-4' | grep -v grep || echo NO_PS
echo ====nvidia====
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo ====orig_e4_lr====
python3 - <<'PY'
import yaml
p="/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_e4_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
print(yaml.safe_load(open(p))["lr"])
PY
""" % (
        ROOT,
        gpu,
        SEED,
        rundir,
    )
    out = ssh_ok(remote, timeout=40)
    if "NO_PID" in out or "NO_PS" in out:
        raise RuntimeError("LAUNCH_FAIL gpu=%s" % gpu)
    log("LAUNCH_OK gpu=%s" % gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
