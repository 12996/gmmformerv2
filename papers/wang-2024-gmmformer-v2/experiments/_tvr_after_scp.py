#!/usr/bin/env python3
"""After complete tvr.zip is on 5090_1: unzip, setup, ntrain, launch T1 then E4."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HOST = "5090_1"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
HERE = Path(__file__).resolve().parent
SEED = "9527"
FILES = [
    "tvr_5090_1.py",
    "tvr_e4_5090_1.py",
    "patch_tvr_jsonl.py",
    "patch_tvr_e4_text.py",
    "unzip_ms_sl.py",
    "run_tvr_gpu.sh",
    "run_tvr_e4_gpu.sh",
    "tvr_do_setup.sh",
    "tvr_ntrain_check.py",
    "tvr_e4_ntrain_check.py",
    "tvr_extract_clip_text.sh",
    "extract_clip_text_proj.py",
    "e2_gpu_occupancy.py",
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


def pick_gpus(n: int) -> list[str]:
    out = ssh_ok("python3 %s/experiments/pick_tvr_gpu.py" % ROOT, timeout=40)
    picks = []
    for line in out.splitlines():
        if line.startswith("PICKS "):
            rest = line.split(" ", 1)[1].strip()
            if rest == "none" or not rest:
                return []
            picks = [x for x in rest.split(",") if x and x != "2"]
    return picks[:n]


def launch(kind: str, gpu: str) -> None:
    scripts = {
        "tvr_i3d": ("run_tvr_gpu.sh", "tvr_i3d_seed%s_gpu%s" % (SEED, gpu)),
        "tvr_e4": ("run_tvr_e4_gpu.sh", "tvr_e4_seed%s_gpu%s" % (SEED, gpu)),
    }
    sh, rundir = scripts[kind]
    remote = r"""
set -euo pipefail
ROOT=%s
GPU=%s
SEED=%s
SH="$ROOT/experiments/%s"
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
ps -u zhaopu -o pid,etime,cmd | grep -E 'main.py -d tvr|run_tvr' | grep -v grep || echo NO_PS
echo ====logtxt====
ls -l "$ROOT/tmp/tvr_i3d_seed${SEED}/results/tvr/gmmformer_v2/log.txt" 2>/dev/null || true
ls -l "$ROOT/tmp/tvr_i3d_cliptext_e4_seed${SEED}/results/tvr/gmmformer_v2/log.txt" 2>/dev/null || true
""" % (ROOT, gpu, SEED, sh, rundir)
    out = ssh_ok(remote, timeout=40)
    if "NO_PID" in out or "NO_PS" in out:
        raise RuntimeError("LAUNCH_FAIL %s gpu=%s" % (kind, gpu))
    log("LAUNCH_OK %s gpu=%s" % (kind, gpu))


def launch_train() -> int:
    gpus = pick_gpus(2)
    log("GPUS %s" % gpus)
    if not gpus:
        raise RuntimeError("NO_GPU")
    log("LAUNCH_T1 gpu=%s" % gpus[0])
    launch("tvr_i3d", gpus[0])
    extract_gpu = gpus[1] if len(gpus) > 1 else ""
    log("EXTRACT_CLIP gpu=%s" % (extract_gpu or "cpu"))
    extract_cmd = "bash %s/experiments/tvr_extract_clip_text.sh" % ROOT
    if extract_gpu:
        extract_cmd += " " + extract_gpu
    out = ssh_ok(extract_cmd, timeout=7200)
    if "EXTRACT_OK" not in out and "CLIP_ALREADY" not in out:
        raise RuntimeError("EXTRACT_FAIL")
    log("NTRAIN_E4")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/tvr_e4_ntrain_check.py" % (ROOT, ROOT),
        timeout=1800,
    )
    if "E4_NTRAIN_CHECK_OK" not in out:
        raise RuntimeError("E4_NTRAIN_FAIL")
    gpus2 = pick_gpus(2)
    e4_gpu = None
    for g in gpus2:
        if g != gpus[0]:
            e4_gpu = g
            break
    if e4_gpu is None:
        log("E4_NO_SECOND_GPU leftover after T1; skip E4")
        print("E4_BLOCKED no leftover gpu after T1", flush=True)
        return 0
    log("LAUNCH_E4 gpu=%s" % e4_gpu)
    launch("tvr_e4", e4_gpu)
    log("DONE")
    return 0


def main() -> int:
    log("SCP_SCRIPTS")
    ssh_ok("mkdir -p %s/experiments %s/logs" % (ROOT, ROOT), timeout=30)
    for name in FILES + ["pick_tvr_gpu.py"]:
        scp_file(name)
    log("UNZIP")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/unzip_ms_sl.py tvr" % (ROOT, ROOT),
        timeout=7200,
    )
    if "UNZIP_OK" not in out:
        raise RuntimeError("UNZIP_FAIL")
    log("SETUP")
    out = ssh_ok("bash %s/experiments/tvr_do_setup.sh" % ROOT, timeout=180)
    if "TVR_SETUP_OK" not in out:
        raise RuntimeError("SETUP_FAIL")
    log("NTRAIN_T1")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/tvr_ntrain_check.py" % (ROOT, ROOT),
        timeout=1800,
    )
    if "NTRAIN_CHECK_OK" not in out:
        raise RuntimeError("NTRAIN_FAIL")
    return launch_train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "continue":
        scp_file("pick_tvr_gpu.py")
        raise SystemExit(launch_train())
    raise SystemExit(main())
