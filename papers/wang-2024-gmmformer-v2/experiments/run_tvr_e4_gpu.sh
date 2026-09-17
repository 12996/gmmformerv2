#!/bin/bash
# E4 TVR: I3D+ResNet visual + CLIP projected text. Separate PRVR_ROOT. Never GPU2.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
if [ "$GPU" = "2" ]; then
  echo "NEVER_GPU2 asr" >&2
  exit 1
fi
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/tvr_i3d_cliptext_e4_seed${SEED}"
export PYTHONUNBUFFERED=1
RUNDIR="$ROOT/logs/runs/tvr_e4_seed${SEED}_gpu${GPU}"
LOCKDIR="$ROOT/logs/runs/tvr_e4_seed${SEED}.lock"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
else
  if pgrep -u zhaopu -f "PRVR_ROOT=$PRVR_ROOT" >/dev/null; then
    echo "TVR_E4_ALREADY_RUNNING" >&2
    exit 1
  fi
  echo "STALE_LOCK $LOCKDIR" >&2
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR"
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
fi
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/tvr_e4_seed${SEED}_gpu${GPU}.log"
ln -sfn "$STDOUT" "$LN"
LOGTXT="$PRVR_ROOT/results/tvr/gmmformer_v2/log.txt"
"$ROOT/conda/bin/python" -u main.py -d tvr --gpu "$GPU" >> "$STDOUT" 2>&1 &
PID=$!
echo "$PID" > "$RUNDIR/pid"
python3 - <<PY
import json, os, time
run = {
    "seed": int("$SEED"),
    "gpu": "$GPU",
    "pid": int("$PID"),
    "cmd": "python main.py -d tvr --gpu $GPU",
    "prvr_root": os.environ["PRVR_ROOT"],
    "stdout": "$STDOUT",
    "log_txt": "$LOGTXT",
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "E4 TVR I3D vis + CLIP proj text, not Table 1",
}
open("$RUNDIR/run.json", "w").write(json.dumps(run, indent=2) + "\n")
PY
set +e
wait "$PID"
EC=$?
echo "$EC" > "$RUNDIR/exit_code"
echo "exit=$EC ts=$(date -Is)" >> "$RUNDIR/run.json"
rm -rf "$LOCKDIR"
exit "$EC"
