"""Poll 5090 nvidia-smi until a GPU is actually free, then notify.

A GPU counts as free only when used memory is below --max-used-mib
(default 2048). Idle vLLM at 0% util with 28GB held is NOT free.

Usage:
  python watch_5090_gpu.py              # print status each poll
  python watch_5090_gpu.py --monitor    # stdout only ACTION_REQUIRED/FAILED
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


HOST = "5090"
DEFAULT_MAX_USED_MIB = 2048
DEFAULT_POLL_S = 30
SSH_FAIL_LIMIT = 8


def log_path(monitor: bool) -> str:
    if monitor:
        base = os.path.join(os.path.expanduser("~"), ".grok", "long-running-background-tasks")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "watch_5090_gpu.log")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "watch_5090_gpu.log")


def log(path: str, msg: str) -> None:
    line = "%s %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def query_gpus() -> list[dict]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        HOST,
        "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(err or "ssh nvidia-smi exit %s" % r.returncode)
    gpus = []
    for raw in r.stdout.splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "used": float(parts[1]),
                "free": float(parts[2]),
                "util": float(parts[3]),
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return gpus


def free_ids(gpus: list[dict], max_used: float) -> list[int]:
    return [g["index"] for g in gpus if g["used"] < max_used]


def fmt_gpus(gpus: list[dict]) -> str:
    bits = []
    for g in gpus:
        bits.append("gpu%s used=%.0fMiB free=%.0fMiB util=%.0f%%" % (g["index"], g["used"], g["free"], g["util"]))
    return "; ".join(bits)


def balloon(text: str, logf: str) -> None:
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$n = New-Object System.Windows.Forms.NotifyIcon; "
        "$n.Icon = [System.Drawing.SystemIcons]::Information; "
        "$n.Visible = $true; "
        "$n.BalloonTipTitle = '5090 GPU'; "
        "$n.BalloonTipText = '%s'; "
        "$n.ShowBalloonTip(15000); "
        "Start-Sleep -Seconds 16; "
        "$n.Dispose()"
    ) % text.replace("'", "''")
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log(logf, "balloon failed: %s" % e)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--max-used-mib", type=float, default=DEFAULT_MAX_USED_MIB)
    p.add_argument("--poll", type=int, default=DEFAULT_POLL_S)
    p.add_argument("--once", action="store_true", help="one query then exit")
    args = p.parse_args()

    logf = log_path(args.monitor)
    announced: set[int] = set()
    fails = 0

    while True:
        try:
            gpus = query_gpus()
            fails = 0
        except Exception as e:
            fails += 1
            log(logf, "query fail %s/%s: %s" % (fails, SSH_FAIL_LIMIT, e))
            if not args.monitor:
                print("query fail %s/%s: %s" % (fails, SSH_FAIL_LIMIT, e), flush=True)
            if fails >= SSH_FAIL_LIMIT:
                if args.monitor:
                    print("FAILED: ssh/nvidia-smi failed %s times: %s" % (fails, e), flush=True)
                else:
                    print("FAILED after %s ssh errors" % fails, flush=True)
                return 1
            if args.once:
                return 1
            time.sleep(args.poll)
            continue

        free = free_ids(gpus, args.max_used_mib)
        log(logf, "free=%s | %s" % (free, fmt_gpus(gpus)))
        if not args.monitor:
            print(fmt_gpus(gpus), flush=True)
            print("free (used < %.0f MiB): %s" % (args.max_used_mib, free if free else "none"), flush=True)

        new = [i for i in free if i not in announced]
        if new:
            announced.update(new)
            msg = "5090 free GPU: %s (used < %.0f MiB)" % (",".join(str(i) for i in new), args.max_used_mib)
            balloon(msg, logf)
            if args.monitor:
                print("ACTION_REQUIRED: " + msg, flush=True)
            else:
                print(msg, flush=True)
            if args.once:
                return 0

        if args.once:
            return 0 if free else 2
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
