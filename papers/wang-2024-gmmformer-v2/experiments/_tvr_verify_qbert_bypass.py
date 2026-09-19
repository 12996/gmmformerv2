#!/usr/bin/env python3
import subprocess
import sys

REMOTE = r"""
python3 - <<'PY'
import os, subprocess, time, yaml
root="/data/zhaopu/wang-2024-gmmformer-v2"
pid, launcher = 2918538, 2918515
lp=root+"/tmp/tvr_i3d_cliptext_qbert_bypass_seed9527/results/tvr/gmmformer_v2/log.txt"
hp=root+"/tmp/tvr_i3d_cliptext_qbert_bypass_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
old=root+"/tmp/tvr_i3d_cliptext_e4_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
plog=root+"/logs/tvr_e4_qbert_bypass_pipeline.log"
run=root+"/logs/runs/tvr_e4_qbert_bypass_seed9527_gpu5/run.json"
print("pid_alive", bool(subprocess.run(["ps","-p",str(pid),"-o","pid="],capture_output=True,text=True).stdout.strip()))
print("launcher_alive", bool(subprocess.run(["ps","-p",str(launcher),"-o","pid="],capture_output=True,text=True).stdout.strip()))
print("PS")
print(subprocess.check_output(["ps","-p","%s,%s"%(pid,launcher),"-o","pid,etime,cmd"], text=True, errors="replace"))
print("log_exists", os.path.isfile(lp), "size", os.path.getsize(lp) if os.path.isfile(lp) else 0)
if os.path.isfile(lp):
    t=open(lp,errors="replace").read()
    print("log_has_epoch", ("epoch:" in t) or ("Epoch:" in t))
    print("LOGTAIL")
    print("\n".join(t.splitlines()[-40:]))
print("RUNJSON")
print(open(run).read() if os.path.isfile(run) else "NO")
d=yaml.safe_load(open(hp)) if os.path.isfile(hp) else {}
o=yaml.safe_load(open(old))
keys=["lr","sft_factor","n_epoch","max_es_cnt","q_feat_size","visual_feature","root","hard_negative_start_epoch","batchsize","bypass_query_encoder","seed","hidden_size"]
print("NEW", {k:d.get(k) for k in keys})
print("OLD_E4", {k:o.get(k) for k in ["lr","sft_factor","root","q_feat_size"]})
print("OLD_E4_mtime", time.ctime(os.path.getmtime(old)))
print("PIPE_NTRAIN")
for line in open(plog, errors="replace"):
    if any(x in line for x in ["text_feat","CANDS","USE_GPU","LAUNCH_GPU","QBERT_BYPASS","one_shape","encoder_calls","video_query","ntrain ","nctx"]):
        print(line.rstrip())
print("SMI_APPS")
print(subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader"], text=True, errors="replace"))
print("SMI_GPU")
print(subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"], text=True))
PY
"""

r = subprocess.run(
    [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "5090_1",
        REMOTE,
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
sys.stdout.write(r.stdout or "")
sys.stderr.write(r.stderr or "")
raise SystemExit(r.returncode)
