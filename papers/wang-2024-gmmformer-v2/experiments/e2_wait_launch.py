#!/usr/bin/env python3
"""Poll 5090_1 for extract-done + a free GPU, then launch E2 seed 9527 once."""
from __future__ import annotations

import os
import subprocess
import sys
import time

HOST = "5090_1"
POLL_S = 60
MAX_WAIT_S = int(os.environ.get("E2_GPU_WAIT_S", str(40 * 60)))
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
    r = ssh("python3 %s/experiments/e2_gpu_occupancy.py" % ROOT)
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


def extract_status() -> str:
    r = ssh(
        "python3 - <<'PY'\n"
        "import os\n"
        "p='/data/zhaopu/wang-2024-gmmformer-v2/logs/e2_extract_proj.log'\n"
        "print('EXISTS', os.path.exists(p), 'SIZE', os.path.getsize(p) if os.path.exists(p) else 0)\n"
        "txt=open(p, errors='replace').read() if os.path.exists(p) else ''\n"
        "print('EXTRACT_OK', 'EXTRACT_OK' in txt)\n"
        "print('TRACEBACK', 'Traceback' in txt)\n"
        "lines=[ln for ln in txt.splitlines() if ln.strip()]\n"
        "print('TAIL', ' | '.join(lines[-6:])[:500])\n"
        "PY"
    )
    return ((r.stdout or "") + (r.stderr or "")).strip()


def install_proj_h5() -> str:
    r = ssh("bash %s/experiments/e2_install_proj_h5.sh" % ROOT, timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("install_proj_h5 rc=%s\n%s" % (r.returncode, out))
    return out


def clip_cfg_ok() -> str:
    r = ssh(
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "p=Path('/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Configs/cha.py')\n"
        "t=p.read_text()\n"
        "print('CHA', p)\n"
        "print('HAS_CLIP', 'visual_feature\"] = \"clip\"' in t or \"visual_feature'] = 'clip'\" in t)\n"
        "print('HAS_Q512', 'q_feat_size\"] = 512' in t or \"q_feat_size'] = 512\" in t)\n"
        "print('HAS_V512', 'visual_feat_dim\"] = 512' in t or \"visual_feat_dim'] = 512\" in t)\n"
        "print('HAS_I3D', 'i3d_rgb_lgi' in t)\n"
        "mc=Path('/data/zhaopu/wang-2024-gmmformer-v2/vendor/GMMFormer_v2/src/Models/gmmformerV2/model_components.py').read_text()\n"
        "print('TC_SUM', 'out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)' in mc)\n"
        "print('TC_MEAN_OVERWRITE', 'out = torch.mean(oo, dim = -1).squeeze()' in mc or 'out = torch.mean(oo, dim=-1).squeeze()' in mc)\n"
        "PY"
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        raise RuntimeError("clip_cfg rc=%s\n%s" % (r.returncode, out))
    if "HAS_CLIP True" not in out or "HAS_Q512 True" not in out or "HAS_V512 True" not in out:
        raise RuntimeError("cha.py is not CLIP 512-d\n%s" % out)
    if "HAS_I3D True" in out:
        raise RuntimeError("cha.py still mentions i3d_rgb_lgi\n%s" % out)
    if "TC_SUM True" not in out or "TC_MEAN_OVERWRITE True" in out:
        raise RuntimeError("TC patch missing\n%s" % out)
    return out


def already_running() -> str:
    r = ssh(
        "ps -u zhaopu -o pid,cmd | grep -E 'main.py -d cha|run_cha_e2_gpu' | grep -v grep || true; "
        "ls -ld /data/zhaopu/wang-2024-gmmformer-v2/tmp/clip_e2_seed9527 2>/dev/null || true; "
        "ls /data/zhaopu/wang-2024-gmmformer-v2/logs/runs/e2_seed9527_gpu*/pid 2>/dev/null || true"
    )
    return ((r.stdout or "") + (r.stderr or "")).strip()


def launch(gpu: str) -> str:
    remote = (
        "set -euo pipefail; "
        "ROOT=%s; "
        "GPU=%s; "
        "SEED=%s; "
        "mkdir -p \"$ROOT/logs/runs/e2_seed${SEED}_gpu${GPU}\" \"$ROOT/logs\"; "
        "nohup \"$ROOT/run_cha_e2_gpu.sh\" \"$GPU\" \"$SEED\" "
        "> \"$ROOT/logs/runs/e2_seed${SEED}_gpu${GPU}/launcher.out\" 2>&1 & "
        "echo LAUNCHER_PID=$!; "
        "sleep 5; "
        "echo '==== pid file ===='; "
        "cat \"$ROOT/logs/runs/e2_seed${SEED}_gpu${GPU}/pid\"; "
        "echo '==== run.json ===='; "
        "cat \"$ROOT/logs/runs/e2_seed${SEED}_gpu${GPU}/run.json\" || true; "
        "echo '==== ps ===='; "
        "ps -u zhaopu -o pid,ppid,cmd | grep -E 'main.py -d cha|run_cha_e2_gpu' | grep -v grep || true; "
        "echo '==== PRVR_ROOT exists ===='; "
        "ls -ld \"$ROOT/tmp/clip_e2_seed${SEED}\" \"$ROOT/tmp/i3d_tc_seed${SEED}\" \"$ROOT/tmp/i3d_seed${SEED}\"; "
        "echo '==== occupancy after launch ===='; "
        "python3 \"$ROOT/experiments/e2_gpu_occupancy.py\""
    ) % (ROOT, gpu, SEED)
    r = ssh(remote, timeout=60)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("launch rc=%s\n%s" % (r.returncode, out))
    return out


def main() -> int:
    t0 = time.time()
    gpu_t0 = None
    n = 0
    extract_ok = False
    h5_installed = False
    while True:
        n += 1
        elapsed = time.time() - t0
        try:
            est = extract_status()
        except Exception as e:
            print("POLL %s EXTRACT_FAIL elapsed=%.0fs %s" % (n, elapsed, e), flush=True)
            est = ""
        print("POLL %s elapsed=%.0fs" % (n, elapsed), flush=True)
        if est:
            print(est, flush=True)
        if "TRACEBACK True" in est:
            print("BLOCKED extract traceback", flush=True)
            return 1
        if "EXTRACT_OK True" in est:
            extract_ok = True
        if extract_ok and not h5_installed:
            print("INSTALL_PROJ_H5", flush=True)
            try:
                print(install_proj_h5(), flush=True)
            except Exception as e:
                print("INSTALL_FAIL", e, flush=True)
                return 1
            h5_installed = True
            gpu_t0 = time.time()
            print("GPU_WAIT_START", flush=True)
        try:
            text = occupancy()
        except Exception as e:
            print("OCCUPANCY_FAIL", e, flush=True)
            text = ""
        if text:
            print(text, flush=True)
        cands = parse_candidates(text) if text else []
        if extract_ok and h5_installed and cands:
            try:
                print("CFG_CHECK", flush=True)
                print(clip_cfg_ok(), flush=True)
            except Exception as e:
                print("CFG_FAIL", e, flush=True)
                return 1
            ar = already_running()
            if ar and "main.py -d cha" in ar:
                print("SKIP_ALREADY", flush=True)
                print(ar, flush=True)
                return 0
            gpu = cands[0]
            print("LAUNCH gpu=%s seed=%s" % (gpu, SEED), flush=True)
            print(launch(gpu), flush=True)
            return 0
        gpu_wait = (time.time() - gpu_t0) if gpu_t0 is not None else 0.0
        if gpu_t0 is not None and gpu_wait >= MAX_WAIT_S:
            print("BLOCKED no free GPU after %.0fs extract_ok=%s" % (gpu_wait, extract_ok), flush=True)
            return 2
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
