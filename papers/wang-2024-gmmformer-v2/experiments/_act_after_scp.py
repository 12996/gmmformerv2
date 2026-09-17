#!/usr/bin/env python3
"""After complete activitynet.tar is on 5090_1: extract, setup, ntrain, launch T1 then E4."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HOST = "5090_1"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
HERE = Path(__file__).resolve().parent
SEED = "9527"
EXPECT_SIZE = 14895718400
TVR_PIDS = ("564493", "613385")
FILES = [
    "act_5090_1.py",
    "act_e4_5090_1.py",
    "patch_act_i3d_loader.py",
    "extract_act_tar.py",
    "extract_act_clip_text.sh",
    "extract_clip_text_proj.py",
    "pick_act_gpu.py",
    "run_act_gpu.sh",
    "run_act_e4_gpu.sh",
    "t1_act_do_setup.sh",
    "t1_act_ntrain_check.py",
    "act_e4_ntrain_check.py",
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


def tvr_alive() -> None:
    out = ssh_ok(
        "ps -p %s -o pid= || true" % ",".join(TVR_PIDS),
        timeout=30,
    )
    have = {x.strip() for x in out.split() if x.strip().isdigit()}
    missing = [p for p in TVR_PIDS if p not in have]
    if missing:
        log("WARN TVR pid missing %s (do not restart)" % missing)


def pick_gpus(n: int) -> list[str]:
    out = ssh_ok("python3 %s/experiments/pick_act_gpu.py" % ROOT, timeout=40)
    picks = []
    for line in out.splitlines():
        if line.startswith("PICKS "):
            rest = line.split(" ", 1)[1].strip()
            if rest == "none" or not rest:
                return []
            picks = [x for x in rest.split(",") if x and x not in ("2", "6", "7")]
    return picks[:n]


def wait_gpus(n: int, timeout_s: int = 6 * 3600) -> list[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        tvr_alive()
        gpus = pick_gpus(n)
        if gpus:
            return gpus
        log("NO_GPU leftover; wait 60s")
        time.sleep(60)
    return []


def launch(kind: str, gpu: str) -> None:
    scripts = {
        "act_i3d": ("run_act_gpu.sh", "act_i3d_seed%s_gpu%s" % (SEED, gpu)),
        "act_e4": ("run_act_e4_gpu.sh", "act_e4_seed%s_gpu%s" % (SEED, gpu)),
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
chmod +x "$SH" "$ROOT/run_act_gpu.sh" "$ROOT/run_act_e4_gpu.sh"
if [ "$GPU" = "2" ] || [ "$GPU" = "6" ] || [ "$GPU" = "7" ]; then
  echo REFUSE_GPU"$GPU"
  exit 2
fi
nohup "$SH" "$GPU" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER_PID=$!
sleep 12
echo ====pid====
cat "$RUN/pid" || echo NO_PID
echo ====runjson====
cat "$RUN/run.json" || echo NO_RUNJSON
echo ====ps====
ps -u zhaopu -o pid,etime,cmd | grep -E '[m]ain.py -d act|[r]un_act_' || echo NO_PS
echo ====tvr====
ps -p 564493,613385 -o pid,etime,cmd || echo TVR_PIDS_CHANGED
echo ====logtxt====
ls -l "$ROOT/tmp/act_i3d_seed${SEED}/results/activitynet/gmmformer_v2/log.txt" 2>/dev/null || true
ls -l "$ROOT/tmp/act_i3d_cliptext_e4_seed${SEED}/results/activitynet/gmmformer_v2/log.txt" 2>/dev/null || true
""" % (ROOT, gpu, SEED, sh, rundir)
    out = ssh_ok(remote, timeout=40)
    if "REFUSE_GPU" in out:
        raise RuntimeError("LAUNCH_REFUSED %s gpu=%s" % (kind, gpu))
    pid_ok = False
    in_pid = False
    for line in out.splitlines():
        s = line.strip()
        if s == "====pid====":
            in_pid = True
            continue
        if s.startswith("===="):
            in_pid = False
            continue
        if in_pid and s.isdigit():
            pid_ok = True
    if not pid_ok:
        raise RuntimeError("LAUNCH_FAIL %s gpu=%s" % (kind, gpu))
    log("LAUNCH_OK %s gpu=%s" % (kind, gpu))


def wait_tar() -> None:
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        rc, out = ssh(
            "stat -c %%s %s/data/prvr/activitynet.tar 2>/dev/null || echo 0" % ROOT,
            timeout=30,
        )
        try:
            sz = int((out or "0").strip().splitlines()[-1])
        except ValueError:
            sz = 0
        log("TAR_SIZE %s / %s" % (sz, EXPECT_SIZE))
        if sz == EXPECT_SIZE:
            time.sleep(5)
            rc2, out2 = ssh(
                "stat -c %%s %s/data/prvr/activitynet.tar 2>/dev/null || echo 0" % ROOT,
                timeout=30,
            )
            try:
                sz2 = int((out2 or "0").strip().splitlines()[-1])
            except ValueError:
                sz2 = 0
            if sz2 == EXPECT_SIZE:
                return
            log("TAR_SIZE_FLAP %s" % sz2)
        time.sleep(20)
    raise RuntimeError("TAR_TIMEOUT")


def main() -> int:
    log("SCP_SCRIPTS")
    ssh_ok("mkdir -p %s/experiments %s/logs" % (ROOT, ROOT), timeout=30)
    for name in FILES:
        scp_file(name)
    log("WAIT_TAR")
    wait_tar()
    log("EXTRACT")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/extract_act_tar.py" % (ROOT, ROOT),
        timeout=7200,
    )
    if "EXTRACT_OK" not in out:
        raise RuntimeError("EXTRACT_FAIL")
    log("SETUP")
    out = ssh_ok("bash %s/experiments/t1_act_do_setup.sh" % ROOT, timeout=1800)
    if "SETUP_OK" not in out:
        raise RuntimeError("SETUP_FAIL")
    if "MEAN_STILL_PRESENT" in out:
        raise RuntimeError("TC_MEAN")
    if "NTRAIN_CHECK_OK" not in out:
        log("NTRAIN_T1")
        out = ssh_ok(
            "%s/conda/bin/python -u %s/experiments/t1_act_ntrain_check.py" % (ROOT, ROOT),
            timeout=1800,
        )
        if "NTRAIN_CHECK_OK" not in out:
            raise RuntimeError("NTRAIN_FAIL")
    else:
        log("NTRAIN_T1_IN_SETUP")
    gpus = wait_gpus(1)
    if not gpus:
        print("blocked: no leftover GPU (>=10GiB, never 2/6/7)", flush=True)
        return 2
    log("LAUNCH_T1 gpu=%s" % gpus[0])
    launch("act_i3d", gpus[0])
    t1_gpu = gpus[0]
    gpus2 = pick_gpus(2)
    extract_gpu = ""
    for g in gpus2:
        if g != t1_gpu:
            extract_gpu = g
            break
    log("EXTRACT_CLIP gpu=%s" % (extract_gpu or "cpu"))
    extract_cmd = "bash %s/experiments/extract_act_clip_text.sh" % ROOT
    if extract_gpu:
        extract_cmd += " " + extract_gpu
    out = ssh_ok(extract_cmd, timeout=7200)
    if "EXTRACT_OK" not in out and "CLIP_ALREADY" not in out:
        log("E4_EXTRACT_FAIL skip E4")
        print("E4_BLOCKED clip extract failed", flush=True)
        return 0
    log("NTRAIN_E4")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/act_e4_ntrain_check.py" % (ROOT, ROOT),
        timeout=1800,
    )
    if "E4_NTRAIN_CHECK_OK" not in out:
        log("E4_NTRAIN_FAIL skip E4")
        print("E4_BLOCKED ntrain failed", flush=True)
        return 0
    gpus3 = pick_gpus(2)
    e4_gpu = None
    for g in gpus3:
        if g != t1_gpu:
            e4_gpu = g
            break
    if e4_gpu is None:
        log("E4_NO_SECOND_GPU leftover after T1; skip E4")
        print("E4_BLOCKED no leftover gpu after T1", flush=True)
        return 0
    log("LAUNCH_E4 gpu=%s" % e4_gpu)
    launch("act_e4", e4_gpu)
    log("DONE")
    return 0


def continue_after_extract() -> int:
    log("SCP_SCRIPTS_CONTINUE")
    ssh_ok("mkdir -p %s/experiments %s/logs" % (ROOT, ROOT), timeout=30)
    for name in FILES:
        scp_file(name)
    log("SETUP")
    out = ssh_ok("bash %s/experiments/t1_act_do_setup.sh" % ROOT, timeout=1800)
    if "SETUP_OK" not in out:
        raise RuntimeError("SETUP_FAIL")
    if "MEAN_STILL_PRESENT" in out:
        raise RuntimeError("TC_MEAN")
    if "NTRAIN_CHECK_OK" not in out:
        log("NTRAIN_T1")
        out = ssh_ok(
            "%s/conda/bin/python -u %s/experiments/t1_act_ntrain_check.py" % (ROOT, ROOT),
            timeout=1800,
        )
        if "NTRAIN_CHECK_OK" not in out:
            raise RuntimeError("NTRAIN_FAIL")
    else:
        log("NTRAIN_T1_IN_SETUP")
    gpus = wait_gpus(1)
    if not gpus:
        print("blocked: no leftover GPU (>=10GiB, never 2/6/7)", flush=True)
        return 2
    log("LAUNCH_T1 gpu=%s" % gpus[0])
    launch("act_i3d", gpus[0])
    t1_gpu = gpus[0]
    gpus2 = pick_gpus(2)
    extract_gpu = ""
    for g in gpus2:
        if g != t1_gpu:
            extract_gpu = g
            break
    log("EXTRACT_CLIP gpu=%s" % (extract_gpu or "cpu"))
    extract_cmd = "bash %s/experiments/extract_act_clip_text.sh" % ROOT
    if extract_gpu:
        extract_cmd += " " + extract_gpu
    out = ssh_ok(extract_cmd, timeout=7200)
    if "EXTRACT_OK" not in out and "CLIP_ALREADY" not in out:
        log("E4_EXTRACT_FAIL skip E4")
        print("E4_BLOCKED clip extract failed", flush=True)
        return 0
    log("NTRAIN_E4")
    out = ssh_ok(
        "%s/conda/bin/python -u %s/experiments/act_e4_ntrain_check.py" % (ROOT, ROOT),
        timeout=1800,
    )
    if "E4_NTRAIN_CHECK_OK" not in out:
        log("E4_NTRAIN_FAIL skip E4")
        print("E4_BLOCKED ntrain failed", flush=True)
        return 0
    gpus3 = pick_gpus(2)
    e4_gpu = None
    for g in gpus3:
        if g != t1_gpu:
            e4_gpu = g
            break
    if e4_gpu is None:
        log("E4_NO_SECOND_GPU leftover after T1; skip E4")
        print("E4_BLOCKED no leftover gpu after T1", flush=True)
        return 0
    log("LAUNCH_E4 gpu=%s" % e4_gpu)
    launch("act_e4", e4_gpu)
    log("DONE")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "continue":
        raise SystemExit(continue_after_extract())
    raise SystemExit(main())

