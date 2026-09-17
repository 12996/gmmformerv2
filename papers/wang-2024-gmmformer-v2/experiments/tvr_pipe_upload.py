#!/usr/bin/env python3
"""Upload completed TVR zip chunks to 5090_1 while local 7897 download continues."""
from __future__ import annotations

import json
import os
import struct
import subprocess
import time

LOCAL = r"F:\论文\data\tvr\tvr.zip"
STATE = LOCAL + ".parts.json"
SENT = LOCAL + ".sent.json"
REMOTE = "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/tvr.zip"
EXPECT = 17787514489
CHUNK = 8 * 1024 * 1024
N_CHUNKS = (EXPECT + CHUNK - 1) // CHUNK
HOST = "5090_1"
RECV = "/data/zhaopu/wang-2024-gmmformer-v2/experiments/e5_recv_chunks.py"


def load_done(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    try:
        return set(json.load(open(path, encoding="utf-8")).get("done", []))
    except Exception:
        return set()


def save_sent(done: set) -> None:
    tmp = SENT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "expect": EXPECT, "chunk": CHUNK}, f)
    os.replace(tmp, SENT)


def ssh(cmd: str, timeout: int = 40) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError("ssh fail %s %s" % (r.returncode, out[-500:]))
    return out


def main() -> int:
    print("UPLOAD_START proxy=local-7897", flush=True)
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST,
         "pkill -f tvr_parallel_download.py; mkdir -p /data/zhaopu/wang-2024-gmmformer-v2/data/prvr /data/zhaopu/wang-2024-gmmformer-v2/logs"],
        capture_output=True,
        text=True,
        timeout=40,
    )
    proc = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=30",
            HOST,
            "python3 -u %s %s %s >> /data/zhaopu/wang-2024-gmmformer-v2/logs/tvr_download.log 2>&1"
            % (RECV, REMOTE, EXPECT),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    sent = load_done(SENT)
    t0 = time.time()
    nbytes = 0
    idle = 0
    f = open(LOCAL, "rb")
    try:
        while True:
            local = load_done(STATE)
            todo = sorted(local - sent)
            if not todo:
                if len(sent) >= N_CHUNKS and os.path.isfile(LOCAL) and os.path.getsize(LOCAL) == EXPECT:
                    proc.stdin.write(struct.pack("!QQ", (1 << 64) - 1, 0))
                    proc.stdin.flush()
                    print("UPLOAD_OK n=%s bytes=%s" % (len(sent), nbytes), flush=True)
                    break
                idle += 1
                if idle % 6 == 0:
                    print("WAIT local=%s sent=%s" % (len(local), len(sent)), flush=True)
                time.sleep(5)
                if proc.poll() is not None:
                    err = (proc.stderr.read() if proc.stderr else b"") or b""
                    raise RuntimeError("recv died %s %s" % (proc.returncode, err[-400:]))
                continue
            idle = 0
            for i in todo:
                start = i * CHUNK
                ln = min(CHUNK, EXPECT - start)
                f.seek(start)
                payload = f.read(ln)
                if len(payload) != ln:
                    print("SHORT_LOCAL chunk=%s" % i, flush=True)
                    break
                proc.stdin.write(struct.pack("!QQ", start, ln))
                proc.stdin.write(payload)
                sent.add(i)
                nbytes += ln
                if len(sent) % 10 == 0 or len(sent) == N_CHUNKS:
                    proc.stdin.flush()
                    save_sent(sent)
                    dt = max(time.time() - t0, 1e-6)
                    print(
                        "SENT %s/%s MBps=%.2f" % (len(sent), N_CHUNKS, nbytes / dt / 1024 / 1024),
                        flush=True,
                    )
        proc.stdin.close()
        rc = proc.wait(timeout=120)
        save_sent(sent)
        print("RECV_RC", rc, flush=True)
        return 0 if rc == 0 else rc
    finally:
        f.close()


if __name__ == "__main__":
    raise SystemExit(main())
