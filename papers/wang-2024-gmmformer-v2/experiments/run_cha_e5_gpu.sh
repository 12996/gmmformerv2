#!/bin/bash
# E5: CLIP ViT-L/14 visual + E2 CLIP-B/32 projected text.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
if [ "$GPU" = "2" ]; then
  echo "NEVER_GPU2 asr" >&2
  exit 1
fi
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/clip_l14_e5_seed${SEED}"
export PYTHONUNBUFFERED=1
RUNDIR="$ROOT/logs/runs/e5_seed${SEED}_gpu${GPU}"
LOCKDIR="$ROOT/logs/runs/e5_seed${SEED}.lock"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
else
  if pgrep -u zhaopu -f "main.py -d cha" >/dev/null; then
    echo "E5_ALREADY_RUNNING" >&2
    exit 1
  fi
  echo "STALE_LOCK $LOCKDIR" >&2
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR"
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
fi
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/cha_e5_seed${SEED}_gpu${GPU}.log"
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
    "note": "E5 ViT-L/14 visual + E2 CLIP-B/32 projected text, TC-fixed",
}
open("$RUNDIR/run.json", "w").write(json.dumps(run, indent=2) + "\n")
PY
set +e
wait "$PID"
EC=$?
echo "$EC" > "$RUNDIR/exit_code"
echo "exit=$EC ts=$(date -Is)" >> "$RUNDIR/run.json"
exit "$EC"
