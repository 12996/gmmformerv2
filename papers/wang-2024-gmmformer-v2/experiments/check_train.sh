#!/bin/bash
# Compact training health dump for 5090_1 Charades CLIP jobs.
python3 - <<'PY'
import json, os, re, glob, subprocess, time
root = "/data/zhaopu/wang-2024-gmmformer-v2"
run_dirs = sorted(glob.glob(root + "/logs/runs/seed*_gpu*"))
fail_re = re.compile(r"CUDA out of memory|RuntimeError|Traceback \(most recent call last\)|No module named")
print("TS", time.strftime("%Y-%m-%dT%H:%M:%S"))
ps = subprocess.check_output(["ps", "-u", "zhaopu", "-o", "pid=,ppid=,cmd="], text=True)
alive_pids = set()
for line in ps.splitlines():
    parts = line.split(None, 2)
    if len(parts) < 3 or "main.py -d cha" not in parts[2]:
        continue
    pid = int(parts[0])
    cmd = parts[2]
    # parent train procs only; skip DataLoader workers
    if "conda/bin/python -u main.py" in cmd:
        alive_pids.add(pid)
        print("PROC", pid, cmd)
if not alive_pids:
    print("ALIVE 0")

def last_loss(text):
    text = text.replace("\r", "\n")
    hits = re.findall(r"epoch:\s*(\d+)\s+iter:\s*(\d+)\s+loss:([0-9.]+)", text)
    return hits[-1] if hits else None

def last_val(text):
    rsum = re.findall(r"Rsum: ([0-9.]+)", text)
    r1 = re.findall(r"R@1: ([0-9.]+)", text)
    best = re.findall(r"Best: R@1: ([0-9.]+) R@5: ([0-9.]+) R@10: ([0-9.]+) R@100: ([0-9.]+) Rsum: ([0-9.]+)", text)
    ep = re.findall(r"Epoch: +(\d+)", text)
    return {
        "epoch": ep[-1] if ep else None,
        "r1": r1[-1] if r1 else None,
        "rsum": rsum[-1] if rsum else None,
        "best": best[-1] if best else None,
    }

smi = ""
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        text=True,
    )
except Exception:
    pass
uuid_to_idx = {}
try:
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).splitlines():
        idx, uid = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[uid] = idx
except Exception:
    pass

if not run_dirs:
    print("NO_RUN_DIRS")
for d in run_dirs:
    meta = {}
    rp = os.path.join(d, "run.json")
    if os.path.exists(rp):
        try:
            meta = json.loads(open(rp).read().split("exit=")[0])
        except Exception as e:
            print("RUN", d, "JSON_FAIL", e)
            continue
    pid = meta.get("pid")
    seed = meta.get("seed")
    gpu = meta.get("gpu")
    alive = pid in alive_pids if isinstance(pid, int) else False
    stdout = meta.get("stdout") or os.path.join(d, "stdout.log")
    log_txt = meta.get("log_txt") or ""
    stxt = open(stdout, errors="replace").read() if os.path.exists(stdout) else ""
    ltxt = open(log_txt, errors="replace").read() if log_txt and os.path.exists(log_txt) else ""
    blob = stxt + "\n" + ltxt
    failed = bool(fail_re.search(blob))
    loss = last_loss(blob)
    val = last_val(ltxt)
    mem = None
    gidx = None
    if pid:
        for line in smi.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if parts and parts[0] == str(pid):
                mem = parts[-1]
                gidx = uuid_to_idx.get(parts[1], gpu)
    ec = None
    ecp = os.path.join(d, "exit_code")
    if os.path.exists(ecp):
        ec = open(ecp).read().strip()
    print(
        "RUN seed={seed} gpu={gpu} pid={pid} alive={alive} exit={ec} mem={mem} on={gidx} loss={loss} val={val} fail={failed}".format(
            seed=seed, gpu=gpu, pid=pid, alive=alive, ec=ec, mem=mem, gidx=gidx,
            loss=loss, val=val, failed=failed,
        )
    )
    if failed:
        tail = [ln for ln in blob.replace("\r", "\n").splitlines() if ln.strip()][-6:]
        print("  TAIL", " | ".join(tail)[:400])
PY
