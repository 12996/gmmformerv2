#!/usr/bin/env python3
"""Watch E2 Charades CLIP train on 5090_1 until Early Stop / 100 epoch / crash."""
from __future__ import annotations

import re
import subprocess
import sys
import time

HOST = "5090_1"
POLL_S = 60
FAIL_RE = re.compile(
    r"CUDA out of memory|RuntimeError|Traceback \(most recent call last\)|No module named"
)


def ssh(cmd: str, timeout: int = 40) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ((r.stdout or "") + (r.stderr or "")).strip()


def dump() -> str:
    cmd = r"""
python3 - <<'PY'
import json, os, re, glob, time
root = "/data/zhaopu/wang-2024-gmmformer-v2"
print("TS", time.strftime("%Y-%m-%dT%H:%M:%S"))
dirs = sorted(glob.glob(root + "/logs/runs/e2_seed9527_gpu*"))
print("RUNDIRS", dirs)
fail_re = re.compile(r"CUDA out of memory|RuntimeError|Traceback \(most recent call last\)|No module named")
for d in dirs:
    meta = {}
    rp = os.path.join(d, "run.json")
    if os.path.exists(rp):
        try:
            meta = json.loads(open(rp).read().split("exit=")[0])
        except Exception as e:
            print("JSON_FAIL", d, e)
            continue
    pid = meta.get("pid")
    gpu = meta.get("gpu")
    stdout = meta.get("stdout") or os.path.join(d, "stdout.log")
    log_txt = meta.get("log_txt") or ""
    stxt = open(stdout, errors="replace").read() if os.path.exists(stdout) else ""
    ltxt = open(log_txt, errors="replace").read() if log_txt and os.path.exists(log_txt) else ""
    blob = stxt + "\n" + ltxt
    failed = bool(fail_re.search(blob))
    loss = re.findall(r"epoch:\s*(\d+)\s+iter:\s*(\d+)\s+loss:([0-9.]+)", blob.replace("\r","\n"))
    rsum = re.findall(r"Rsum: ([0-9.]+)", ltxt)
    best = re.findall(r"Best: R@1: ([0-9.]+) R@5: ([0-9.]+) R@10: ([0-9.]+) R@100: ([0-9.]+) Rsum: ([0-9.]+)", ltxt)
    ep = re.findall(r"Epoch: +(\d+)", ltxt)
    ntrain = re.findall(r"ntrain[=: ]+(\d+)", blob, flags=re.I)
    early = "Early Stop" in ltxt
    alive = False
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            alive = True
        except Exception:
            alive = False
    ec = None
    ecp = os.path.join(d, "exit_code")
    if os.path.exists(ecp):
        ec = open(ecp).read().strip()
    print("RUN gpu=%s pid=%s alive=%s exit=%s fail=%s ntrain=%s loss=%s epoch=%s last_rsum=%s best=%s early=%s" % (
        gpu, pid, alive, ec, failed,
        ntrain[-1] if ntrain else None,
        loss[-1] if loss else None,
        ep[-1] if ep else None,
        rsum[-1] if rsum else None,
        best[-1] if best else None,
        early,
    ))
    if failed:
        tail = [ln for ln in blob.replace("\r","\n").splitlines() if ln.strip()][-16:]
        print("TAIL")
        print("\n".join(tail))
    if early:
        print("EARLY_STOP")
    if ep:
        print("N_EPOCH_LOGGED", len(set(ep)))
PY
"""
    return ssh(cmd, timeout=50)


def classify(text: str) -> str:
    if "Traceback" in text or "CUDA out of memory" in text or "RuntimeError" in text:
        return "crash"
    if "EARLY_STOP" in text:
        return "early_stop"
    m = re.search(r"last_rsum=([0-9.]+)", text)
    if m:
        val = float(m.group(1))
        if val <= 10.2:
            return "random_like"
    if re.search(r"alive=False", text) and re.search(r"exit=([1-9]|[1-9][0-9]+)", text):
        return "crash"
    if re.search(r"alive=False", text) and re.search(r"exit=0", text):
        return "done"
    if re.search(r"epoch=99\b", text) or re.search(r"N_EPOCH_LOGGED 100", text):
        return "hundred"
    return "running"


def main() -> int:
    t0 = time.time()
    first_ok = False
    while time.time() - t0 < 25 * 60:
        text = dump()
        print(text, flush=True)
        st = classify(text)
        if st == "crash":
            print("FAILED crash", flush=True)
            return 1
        if st == "random_like":
            print("FAILED first-epoch Rsum looks random", flush=True)
            return 1
        if "last_rsum=" in text and "last_rsum=None" not in text:
            print("FIRST_EPOCH_OK", flush=True)
            first_ok = True
            break
        time.sleep(30)
    if not first_ok:
        print("FAILED no first-epoch val within 25min", flush=True)
        return 1

    while True:
        time.sleep(POLL_S)
        text = dump()
        print(text, flush=True)
        st = classify(text)
        if st in ("early_stop", "done", "hundred"):
            print("DONE", st, flush=True)
            return 0
        if st == "crash":
            print("FAILED crash", flush=True)
            return 1
        if st == "random_like":
            print("FAILED random-like later", flush=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
