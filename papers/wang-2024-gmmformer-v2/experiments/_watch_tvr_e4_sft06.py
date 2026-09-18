#!/usr/bin/env python3
"""Watch 5090_1 TVR E4 sft=0.6 until Early Stop or n_epoch=100.

Do not launch. Do not kill anyone. Ban GPU2.
Stdout: only DONE or FAILED (plus one JSON line after the tag).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HOST = "5090_1"
POLL_S = 90
PID = 2019836
LAUNCHER = 2019823
BAN_GPU = "2"
E4_BASE = 178.6
I3D_BASE = 185.3
HERE = Path(__file__).resolve().parent
RESULT = HERE / "tvr_e4_sft06_watch_result.json"
LOG = HERE / "tvr_e4_sft06_watch.log"

REMOTE = r"""
import json, os, re, subprocess, time
root = "/data/zhaopu/wang-2024-gmmformer-v2"
pid, launcher = 2019836, 2019823
lp = root + "/tmp/tvr_i3d_cliptext_e4_sft06_seed9527/results/tvr/gmmformer_v2/log.txt"
hp = root + "/tmp/tvr_i3d_cliptext_e4_sft06_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
sp = root + "/logs/runs/tvr_e4_sft06_seed9527_gpu5/stdout.log"
ecp = root + "/logs/runs/tvr_e4_sft06_seed9527_gpu5/exit_code"
pidf = root + "/logs/runs/tvr_e4_sft06_seed9527_gpu5/pid"
jp = root + "/logs/runs/tvr_e4_sft06_seed9527_gpu5/run.json"
old_hp = root + "/tmp/tvr_i3d_cliptext_e4_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
old_log = root + "/tmp/tvr_i3d_cliptext_e4_seed9527/results/tvr/gmmformer_v2/log.txt"

def alive(p):
    if not p:
        return False
    r = subprocess.run(["ps", "-p", str(p), "-o", "pid="], capture_output=True, text=True, errors="replace")
    return bool((r.stdout or "").strip())

def subprocess_run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return (r.stdout or "") + (r.stderr or "")

out = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "pid_alive": alive(pid),
    "launcher_alive": alive(launcher),
    "file_pid": open(pidf).read().strip() if os.path.exists(pidf) else "",
    "exit_code": open(ecp).read().strip() if os.path.exists(ecp) else None,
    "log_mtime": os.path.getmtime(lp) if os.path.exists(lp) else None,
    "log_size": os.path.getsize(lp) if os.path.exists(lp) else 0,
    "stdout_size": os.path.getsize(sp) if os.path.exists(sp) else 0,
}
ps = subprocess_run(["ps", "-p", "%s,%s" % (pid, launcher), "-o", "pid,etime,stat,cmd"])
out["ps"] = ps.strip()
smi = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader,nounits"], text=True)
uuid_to_idx = {}
gpus = []
for line in smi.splitlines():
    idx, uuid, used, total, util = [x.strip() for x in line.split(",")]
    uuid_to_idx[uuid] = idx
    gpus.append({"idx": idx, "used": used, "total": total, "util": util})
out["gpus"] = gpus
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
     "--format=csv,noheader,nounits"], text=True, errors="replace")
gpu_of = {}
gpu5 = []
gpu2 = []
for line in apps.splitlines():
    if not line.strip():
        continue
    parts = [x.strip() for x in line.split(",")]
    if len(parts) < 4:
        continue
    uuid, p, name, mem = parts[:4]
    idx = uuid_to_idx.get(uuid, "?")
    gpu_of[p] = idx
    if idx == "5":
        gpu5.append({"pid": p, "mem": mem, "name": name[-80:]})
    if idx == "2":
        gpu2.append({"pid": p, "mem": mem, "name": name[-80:]})
out["gpu5_apps"] = gpu5
out["gpu2_apps"] = gpu2
out["pid_gpu"] = gpu_of.get(str(pid))
out["lr"] = None
out["sft_factor"] = None
out["n_epoch"] = None
if os.path.exists(hp):
    for ln in open(hp, errors="replace"):
        if ln.startswith("lr:"):
            out["lr"] = ln.split(":", 1)[1].strip()
        if ln.startswith("sft_factor:"):
            out["sft_factor"] = ln.split(":", 1)[1].strip()
        if ln.startswith("n_epoch:"):
            out["n_epoch"] = ln.split(":", 1)[1].strip()
        if ln.startswith("visual_feature:"):
            out["visual_feature"] = ln.split(":", 1)[1].strip()
        if ln.startswith("q_feat_size:"):
            out["q_feat_size"] = ln.split(":", 1)[1].strip()
        if ln.startswith("hidden_size:"):
            out["hidden_size"] = ln.split(":", 1)[1].strip()
out["old_e4_lr"] = None
out["old_e4_sft"] = None
if os.path.exists(old_hp):
    for ln in open(old_hp, errors="replace"):
        if ln.startswith("lr:"):
            out["old_e4_lr"] = ln.split(":", 1)[1].strip()
        if ln.startswith("sft_factor:"):
            out["old_e4_sft"] = ln.split(":", 1)[1].strip()
old_txt = open(old_log, errors="replace").read() if os.path.exists(old_log) else ""
old_best = re.findall(
    r"Best: R@1: ([0-9.]+) R@5: ([0-9.]+) R@10: ([0-9.]+) R@100: ([0-9.]+) Rsum: ([0-9.]+)",
    old_txt,
)
out["old_e4_best_sumr"] = float(old_best[-1][-1]) if old_best else None
ltxt = open(lp, errors="replace").read() if os.path.exists(lp) else ""
stxt = ""
if os.path.exists(sp):
    with open(sp, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 250000), os.SEEK_SET)
        stxt = f.read().decode("utf-8", "replace")
blob = (stxt + "\n" + ltxt).replace("\r", "\n")
fail_re = re.compile(r"CUDA out of memory|RuntimeError|Traceback \(most recent call last\)")
out["oom"] = "CUDA out of memory" in blob
out["fail"] = bool(fail_re.search(blob))
out["early"] = "Early Stop" in ltxt
loss = re.findall(r"epoch:\s*(\d+)\s+iter:\s*(\d+)\s+loss:([0-9.]+)", blob)
out["loss"] = list(loss[-1]) if loss else None
newbest = re.findall(r"Epoch:\s+(\d+)\s+New Best Model", ltxt)
fail_ep = re.findall(r"Epoch:\s+(\d+)\s+A Relative Failure Epoch", ltxt)
ep = re.findall(r"Epoch:\s+(\d+)", ltxt)
out["new_best_epochs"] = [int(x) for x in newbest]
out["fail_epochs"] = [int(x) for x in fail_ep]
out["last_epoch"] = int(ep[-1]) if ep else None
out["best_epoch"] = int(newbest[-1]) if newbest else None
rsum = re.findall(r"^.*INFO - Rsum: ([0-9.]+)\s*$", ltxt, re.M)
out["last_rsum"] = float(rsum[-1]) if rsum else None
best = re.findall(
    r"Best: R@1: ([0-9.]+) R@5: ([0-9.]+) R@10: ([0-9.]+) R@100: ([0-9.]+) Rsum: ([0-9.]+)",
    ltxt,
)
out["best"] = list(best[-1]) if best else None
first_best = re.findall(
    r"Epoch:\s+(\d+)\s+New Best Model !!!\n.*\n.*\n.*INFO - R@1: ([0-9.]+)\n.*INFO - R@5: ([0-9.]+)\n.*INFO - R@10: ([0-9.]+)\n.*INFO - R@100: ([0-9.]+)\n.*INFO - Rsum: ([0-9.]+)\n.*INFO - Best:.*Rsum: ([0-9.]+)",
    ltxt,
)
out["first_best"] = list(first_best[0]) if first_best else None
early_ln = [ln for ln in ltxt.splitlines() if "Early Stop" in ln]
out["early_line"] = early_ln[-1] if early_ln else None
if os.path.exists(jp):
    out["runjson"] = open(jp).read().replace("\n", " ")[:500]
if out["fail"]:
    lines = [ln for ln in blob.splitlines() if ln.strip()][-20:]
    out["fail_tail"] = lines
print(json.dumps(out))
"""


def slog(msg: str) -> None:
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + msg
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def probe() -> dict:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "python3", "-"],
        input=REMOTE,
        capture_output=True,
        text=True,
        timeout=60,
    )
    text = (r.stdout or "").strip()
    if not text:
        raise RuntimeError("empty ssh stdout rc=%s err=%s" % (r.returncode, (r.stderr or "")[:400]))
    last = text.splitlines()[-1]
    return json.loads(last)


def bucket(s: float | None) -> str:
    if s is None:
        return "unknown"
    if s <= E4_BASE:
        return "no_gain_le_178.6"
    if s <= I3D_BASE:
        return "gain_not_over_i3d_178.6_to_185.3"
    return "over_local_i3d_gt_185.3"


def finished(info: dict, dead_streak: int) -> str | None:
    early = bool(info.get("early"))
    last_ep = info.get("last_epoch")
    hit_max = isinstance(last_ep, int) and last_ep >= 99
    alive = bool(info.get("pid_alive"))
    launcher = bool(info.get("launcher_alive"))
    ec = info.get("exit_code")
    if info.get("oom") and (not alive) and dead_streak >= 2:
        return "FAILED"
    if info.get("fail") and (not alive) and dead_streak >= 2 and not early and not hit_max:
        return "FAILED"
    if not alive and dead_streak >= 2 and (ec is not None or not launcher):
        if (ec not in (None, "0", 0)) and (not early) and (not hit_max):
            return "FAILED"
        return "DONE"
    if not alive and dead_streak >= 2 and (early or hit_max):
        return "DONE"
    return None


def pull_log() -> None:
    src = "5090_1:/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_e4_sft06_seed9527/results/tvr/gmmformer_v2/log.txt"
    dst = HERE / "tvr_e4_sft06_log.txt"
    try:
        subprocess.run(
            ["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", src, str(dst)],
            check=False,
            timeout=40,
        )
    except Exception as e:
        slog("SCP_FAIL %s" % e)


def main() -> int:
    dead_streak = 0
    last = None
    slog("start pid=%s launcher=%s no-relaunch no-kill ban-gpu2" % (PID, LAUNCHER))
    while True:
        try:
            info = probe()
        except Exception as e:
            slog("probe_err %s" % e)
            time.sleep(POLL_S)
            continue
        last = info
        if info.get("pid_alive"):
            dead_streak = 0
        else:
            dead_streak += 1
        best = info.get("best") or []
        slog(
            "alive=%s launcher=%s gpu=%s ep=%s loss=%s rsum=%s best=%s best_ep=%s early=%s exit=%s sft=%s lr=%s dead=%s"
            % (
                info.get("pid_alive"),
                info.get("launcher_alive"),
                info.get("pid_gpu"),
                info.get("last_epoch"),
                info.get("loss"),
                info.get("last_rsum"),
                best,
                info.get("best_epoch"),
                info.get("early"),
                info.get("exit_code"),
                info.get("sft_factor"),
                info.get("lr"),
                dead_streak,
            )
        )
        tag = finished(info, dead_streak)
        if tag:
            best_sum = float(best[-1]) if best else None
            payload = {
                "status": tag,
                "host": HOST,
                "pid": PID,
                "launcher": LAUNCHER,
                "gpu": 5,
                "lr": info.get("lr"),
                "sft_factor": info.get("sft_factor"),
                "n_epoch": info.get("n_epoch"),
                "visual_feature": info.get("visual_feature"),
                "q_feat_size": info.get("q_feat_size"),
                "hidden_size": info.get("hidden_size"),
                "early_stop": bool(info.get("early")),
                "early_line": info.get("early_line"),
                "last_epoch": info.get("last_epoch"),
                "last_rsum": info.get("last_rsum"),
                "best": best,
                "best_epoch": info.get("best_epoch"),
                "best_sumr": best_sum,
                "first_best": info.get("first_best"),
                "bucket": bucket(best_sum),
                "vs_e4_178_6": None if best_sum is None else round(best_sum - E4_BASE, 3),
                "vs_i3d_185_3": None if best_sum is None else round(best_sum - I3D_BASE, 3),
                "new_best_epochs": info.get("new_best_epochs"),
                "fail_epochs": info.get("fail_epochs"),
                "exit_code": info.get("exit_code"),
                "oom": info.get("oom"),
                "fail": info.get("fail"),
                "fail_tail": info.get("fail_tail"),
                "pid_alive": info.get("pid_alive"),
                "launcher_alive": info.get("launcher_alive"),
                "pid_gpu": info.get("pid_gpu"),
                "gpu2_apps": info.get("gpu2_apps"),
                "old_e4_lr": info.get("old_e4_lr"),
                "old_e4_sft": info.get("old_e4_sft"),
                "old_e4_best_sumr": info.get("old_e4_best_sumr"),
                "ts": info.get("ts"),
            }
            RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            pull_log()
            slog("END %s %s" % (tag, json.dumps(payload)))
            print(tag, flush=True)
            return 0 if tag == "DONE" else 1
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
