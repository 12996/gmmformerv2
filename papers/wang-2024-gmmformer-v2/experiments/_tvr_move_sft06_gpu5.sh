#!/bin/bash
# Move our aborted GPU6 sft06 job to empty GPU5. Kill only our pids. Never GPU2.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
OLD_PID=1994870
OLD_LAUNCH=1994857
ps -p "$OLD_PID" -o pid=,user=,cmd= || true
ps -p "$OLD_LAUNCH" -o pid=,user=,cmd= || true
CMD=$(ps -p "$OLD_PID" -o cmd= || true)
LCMD=$(ps -p "$OLD_LAUNCH" -o cmd= || true)
echo "$CMD" | grep -q "main.py -d tvr --gpu 6" || { echo "REFUSE_KILL unexpected pid"; exit 1; }
echo "$CMD" | grep -q "wang-2024-gmmformer-v2/conda/bin/python" || { echo "REFUSE_KILL not our python"; exit 1; }
echo "$LCMD" | grep -q "run_tvr_e4_sft06_gpu.sh" || { echo "REFUSE_KILL unexpected launcher"; exit 1; }
kill "$OLD_PID" "$OLD_LAUNCH" || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if ! ps -p "$OLD_PID" >/dev/null 2>&1 && ! ps -p "$OLD_LAUNCH" >/dev/null 2>&1; then
    echo KILLED_OK
    break
  fi
  sleep 1
done
if ps -p "$OLD_PID" >/dev/null 2>&1; then
  echo STILL_ALIVE
  exit 1
fi
rm -rf "$ROOT/logs/runs/tvr_e4_sft06_seed9527.lock"
SFT="$ROOT/tmp/tvr_i3d_cliptext_e4_sft06_seed9527"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_seed9527"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed9527"
test -d "$ROOT/tmp/tvr_i3d_seed9527"
rm -rf "$SFT/results"
mkdir -p "$SFT"
USED=$(nvidia-smi -i 5 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo GPU5_USED="$USED"
python3 - <<'PY'
import subprocess
q = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
    text=True,
    errors="replace",
)
smi = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
)
uuid = None
for line in smi.splitlines():
    idx, u = [x.strip() for x in line.split(",")]
    if idx == "5":
        uuid = u
n = 0
for line in q.splitlines():
    if not line.strip():
        continue
    u, p = [x.strip() for x in line.split(",")[:2]]
    if u == uuid:
        n += 1
        print("GPU5_PROC", p)
if n:
    raise SystemExit("GPU5_TAKEN")
print("GPU5_EMPTY")
PY
GPU=5
SEED=9527
SH="$ROOT/experiments/run_tvr_e4_sft06_gpu.sh"
RUN="$ROOT/logs/runs/tvr_e4_sft06_seed${SEED}_gpu${GPU}"
mkdir -p "$RUN" "$ROOT/logs"
chmod +x "$SH"
nohup "$SH" "$GPU" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER_PID=$!
sleep 12
echo ====pid====
cat "$RUN/pid" || echo NO_PID_FILE
echo ====runjson====
cat "$RUN/run.json" || echo NO_RUNJSON_FILE
echo ====ps====
ps -u zhaopu -o pid,etime,cmd | grep -E 'main.py -d tvr|run_tvr_e4_sft06' | grep -v grep || echo NO_TVR_PS
echo ====hp====
python3 - <<'PY'
import os
import yaml
p = "/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_e4_sft06_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
print("exists", os.path.isfile(p))
if os.path.isfile(p):
    d = yaml.safe_load(open(p))
    print({k: d[k] for k in ["lr", "sft_factor", "n_epoch", "q_feat_size", "visual_feature", "root", "seed"]})
PY
echo ====smi56====
nvidia-smi -i 5,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo MOVE_OK
