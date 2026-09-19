#!/usr/bin/env python3
"""SCP query-Bert bypass files and start the 5090_1 pipeline. Never GPU2."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

HOST = "5090_1"
ROOT = "/data/zhaopu/wang-2024-gmmformer-v2"
HERE = Path(__file__).resolve().parent
SEED = "9527"
FILES = [
    "patch_tvr_qbert_bypass.py",
    "tvr_e4_qbert_bypass_5090_1.py",
    "tvr_e4_qbert_bypass_ntrain_check.py",
    "run_tvr_e4_qbert_bypass_gpu.sh",
    "run_tvr_e4_qbert_bypass_pipeline.sh",
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


def main() -> int:
    for name in FILES:
        if not (HERE / name).is_file():
            raise SystemExit("missing " + name)
        scp_file(name)
    log("STRIP_CRLF_AND_START")
    remote = r"""
set -euo pipefail
ROOT=%s
python3 - <<'PY'
from pathlib import Path
root = Path("/data/zhaopu/wang-2024-gmmformer-v2/experiments")
names = [
    "patch_tvr_qbert_bypass.py",
    "tvr_e4_qbert_bypass_5090_1.py",
    "tvr_e4_qbert_bypass_ntrain_check.py",
    "run_tvr_e4_qbert_bypass_gpu.sh",
    "run_tvr_e4_qbert_bypass_pipeline.sh",
]
for n in names:
    p = root / n
    t = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    p.write_bytes(t)
    print("LF", n, len(t))
PY
chmod +x "$ROOT/experiments/run_tvr_e4_qbert_bypass_gpu.sh" \
  "$ROOT/experiments/run_tvr_e4_qbert_bypass_pipeline.sh"
mkdir -p "$ROOT/logs" "$ROOT/logs/runs"
PIPE="$ROOT/logs/tvr_e4_qbert_bypass_pipeline.log"
: > "$PIPE"
nohup bash "$ROOT/experiments/run_tvr_e4_qbert_bypass_pipeline.sh" 6 >> "$PIPE" 2>&1 &
echo PIPE_PID=$!
sleep 2
ps -p $! -o pid,etime,cmd || true
""" % ROOT
    out = ssh_ok(remote, timeout=40)
    if "PIPE_PID=" not in out:
        raise RuntimeError("no pipeline pid")
    log("PIPELINE_STARTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
