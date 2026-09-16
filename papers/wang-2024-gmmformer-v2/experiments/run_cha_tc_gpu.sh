#!/bin/bash
# E1: Charades I3D+RoBERTa train after removing DyGMMBlock mean overwrite.
# Separate PRVR_ROOT so it cannot clobber tmp/i3d_seed${SEED}.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/i3d_tc_seed${SEED}"
export PYTHONUNBUFFERED=1
RUNDIR="$ROOT/logs/runs/tc_seed${SEED}_gpu${GPU}"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/cha_tc_seed${SEED}_gpu${GPU}.log"
ln -sfn "$STDOUT" "$LN"
"$ROOT/conda/bin/python" -u main.py -d cha --gpu "$GPU" >> "$STDOUT" 2>&1 &
PID=$!
echo "$PID" > "$RUNDIR/pid"
python3 - <<PY
import json, os, time
run = {
    "seed": int("$SEED"),
    "gpu": "$GPU",
    "pid": int("$PID"),
    "cmd": "python main.py -d cha --gpu $GPU",
    "prvr_root": os.environ["PRVR_ROOT"],
    "stdout": "$STDOUT",
    "log_txt": os.path.join(os.environ["PRVR_ROOT"], "results", "charades", "gmmformer_v2", "log.txt"),
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "E1 delete DyGMMBlock mean overwrite only",
}
open("$RUNDIR/run.json", "w").write(json.dumps(run, indent=2) + "\n")
PY
set +e
wait "$PID"
EC=$?
echo "$EC" > "$RUNDIR/exit_code"
echo "exit=$EC ts=$(date -Is)" >> "$RUNDIR/run.json"
exit "$EC"
