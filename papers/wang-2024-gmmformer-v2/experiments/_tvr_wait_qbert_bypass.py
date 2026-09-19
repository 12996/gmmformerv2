#!/usr/bin/env python3
"""Poll 5090_1 qbert-bypass pipeline until launch or block."""
from __future__ import annotations

import subprocess
import time

HOST = "5090_1"
LOG = "/data/zhaopu/wang-2024-gmmformer-v2/logs/tvr_e4_qbert_bypass_pipeline.log"
REMOTE = r"""
python3 - <<'PY'
from pathlib import Path
p = Path("/data/zhaopu/wang-2024-gmmformer-v2/logs/tvr_e4_qbert_bypass_pipeline.log")
t = p.read_text(errors="replace") if p.exists() else ""
print("SIZE", p.stat().st_size if p.exists() else 0)
print("HAS_NTRAIN", "QBERT_BYPASS_NTRAIN_CHECK_OK" in t)
print("HAS_LAUNCH", "PIPE_LAUNCHED" in t)
print("HAS_BLOCKED", "BLOCKED" in t)
print("HAS_FAIL", "FAIL " in t or "SystemExit" in t or "Traceback" in t)
print("----TAIL----")
print("\n".join(t.splitlines()[-40:]))
PY
"""


def ssh() -> str:
    r = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            HOST,
            REMOTE,
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out, flush=True)
    if r.returncode != 0:
        raise RuntimeError("ssh fail rc=%s" % r.returncode)
    return out


def main() -> int:
    deadline = time.time() + 1200
    n = 0
    while time.time() < deadline:
        n += 1
        print("POLL", n, time.strftime("%H:%M:%S"), flush=True)
        out = ssh()
        if "HAS_LAUNCH True" in out:
            print("FOUND_LAUNCH", flush=True)
            return 0
        if "HAS_BLOCKED True" in out:
            print("FOUND_BLOCKED", flush=True)
            return 3
        if "HAS_FAIL True" in out and "HAS_NTRAIN False" in out:
            print("FOUND_FAIL", flush=True)
            return 4
        time.sleep(20)
    print("TIMEOUT", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
