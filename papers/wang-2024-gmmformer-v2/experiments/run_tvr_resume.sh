#!/bin/bash
# Resume T1-tvr I3D+RoBERTa from best.ckpt. n_epoch must already be 200 in Configs/tvr.py.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/tvr_i3d_seed${SEED}"
export PYTHONUNBUFFERED=1
if [ "$GPU" = "2" ]; then
  echo "REFUSE GPU2"
  exit 2
fi
CKPT="${3:-$PRVR_ROOT/results/tvr/gmmformer_v2/best.ckpt}"
if [ ! -f "$CKPT" ]; then
  echo "NO_CKPT $CKPT"
  exit 1
fi
if grep -q 'cfg\["n_epoch"\] = 100' "$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"; then
  echo "n_epoch still 100; resume loop would be empty"
  exit 1
fi
if ! grep -q 'cfg\["n_epoch"\] = 200' "$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"; then
  echo "n_epoch is not 200"
  exit 1
fi
RUNDIR="$ROOT/logs/runs/tvr_i3d_seed${SEED}_gpu${GPU}_resume200"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/tvr_i3d_seed${SEED}_gpu${GPU}_resume200.log"
ln -sfn "$STDOUT" "$LN"
PY="$ROOT/conda/bin/python"
unset PRVR_EVAL_SPLIT || true
"$PY" -u main.py -d tvr --gpu "$GPU" --resume "$CKPT" >> "$STDOUT" 2>&1 &
PID=$!
echo "$PID" > "$RUNDIR/pid"
python3 - <<PY
import json, os, time
run = {
    "seed": int("$SEED"),
    "gpu": "$GPU",
    "pid": int("$PID"),
    "mode": "resume",
    "cmd": "python main.py -d tvr --gpu $GPU --resume $CKPT",
    "prvr_root": os.environ["PRVR_ROOT"],
    "ckpt": "$CKPT",
    "stdout": "$STDOUT",
    "log_txt": os.path.join(os.environ["PRVR_ROOT"], "results", "tvr", "gmmformer_v2", "log.txt"),
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "T1-tvr resume best.ckpt n_epoch=200 seed $SEED",
}
open("$RUNDIR/run.json", "w").write(json.dumps(run, indent=2) + "\n")
PY
set +e
wait "$PID"
EC=$?
echo "$EC" > "$RUNDIR/exit_code"
echo "exit=$EC ts=$(date -Is)" >> "$RUNDIR/run.json"
exit "$EC"
