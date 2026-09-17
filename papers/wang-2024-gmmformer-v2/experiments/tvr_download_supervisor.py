#!/usr/bin/env python3
"""If tvr_parallel_download.py exits, restart it. Resume is in .parts.json."""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(ROOT, "tvr_parallel_download.py")
OUT = r"F:\论文\data\tvr\tvr.zip" if os.name == "nt" else "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/tvr.zip"
LOG = os.path.join(os.path.dirname(OUT), "download.log")
EXPECT_CHUNKS = 2121


def done_n() -> int:
    import json

    p = OUT + ".parts.json"
    if not os.path.isfile(p):
        return 0
    try:
        return len(json.load(open(p, encoding="utf-8")).get("done", []))
    except Exception:
        return 0


def main() -> int:
    env = os.environ.copy()
    if os.name == "nt":
        env["HTTPS_PROXY"] = env.get("HTTPS_PROXY") or "http://127.0.0.1:7897"
        env["HTTP_PROXY"] = env["HTTPS_PROXY"]
        env["TVR_NWORK"] = env.get("TVR_NWORK") or "4"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    round_n = 0
    while done_n() < EXPECT_CHUNKS:
        round_n += 1
        print("SUPERVISOR start round=%s done=%s/%s" % (round_n, done_n(), EXPECT_CHUNKS), flush=True)
        with open(LOG, "a", encoding="utf-8") as log:
            log.write("\nSUPERVISOR round=%s done=%s\n" % (round_n, done_n()))
            p = subprocess.Popen(
                [sys.executable, "-u", DL, OUT],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            rc = p.wait()
        print("SUPERVISOR child_rc=%s done=%s/%s" % (rc, done_n(), EXPECT_CHUNKS), flush=True)
        if done_n() >= EXPECT_CHUNKS:
            print("SUPERVISOR_OK", flush=True)
            return 0
        time.sleep(15)
    print("SUPERVISOR_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
