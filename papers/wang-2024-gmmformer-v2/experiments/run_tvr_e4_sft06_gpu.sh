#!/bin/bash
# E4 TVR sft_factor=0.6 only. Parent E4 lr=3e-4. Do not overwrite original/lr2e-4/I3D tmp. Never GPU2.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
if [ "$GPU" = "2" ]; then
  echo "NEVER_GPU2 asr" >&2
  exit 1
fi
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/tvr_i3d_cliptext_e4_sft06_seed${SEED}"
export PYTHONUNBUFFERED=1
OLD_E4="$ROOT/tmp/tvr_i3d_cliptext_e4_seed${SEED}"
OLD_LR="$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed${SEED}"
OLD_I3D="$ROOT/tmp/tvr_i3d_seed${SEED}"
if [ ! -d "$OLD_E4" ]; then
  echo "MISSING_ORIG_E4 $OLD_E4" >&2
  exit 1
fi
if [ ! -d "$OLD_LR" ]; then
  echo "MISSING_LR2E4 $OLD_LR" >&2
  exit 1
fi
if [ ! -d "$OLD_I3D" ]; then
  echo "MISSING_I3D $OLD_I3D" >&2
  exit 1
fi
if [ "$PRVR_ROOT" = "$OLD_E4" ] || [ "$PRVR_ROOT" = "$OLD_LR" ] || [ "$PRVR_ROOT" = "$OLD_I3D" ]; then
  echo "REFUSE_OVERWRITE $PRVR_ROOT" >&2
  exit 1
fi
CFG="$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"
SRC_CFG="$ROOT/experiments/tvr_e4_sft06_5090_1.py"
if [ ! -f "$SRC_CFG" ]; then
  echo "MISSING_SRC_CFG $SRC_CFG" >&2
  exit 1
fi
if [ ! -f "$CFG.bak_before_e4_sft06" ]; then
  cp -a "$CFG" "$CFG.bak_before_e4_sft06"
fi
cp -a "$SRC_CFG" "$CFG"
if ! grep -q 'cfg\["sft_factor"\] = 0.6' "$CFG"; then
  echo "SFT_NOT_0.6" >&2
  exit 1
fi
if grep -q 'cfg\["sft_factor"\] = 0.09' "$CFG"; then
  echo "SFT_STILL_0.09" >&2
  exit 1
fi
if ! grep -q 'cfg\["lr"\] = 0.0003' "$CFG"; then
  echo "LR_NOT_3e-4" >&2
  exit 1
fi
if grep -q 'cfg\["lr"\] = 0.0002' "$CFG"; then
  echo "LR_STILL_2e-4" >&2
  exit 1
fi
if ! grep -q 'cfg\["n_epoch"\] = 100' "$CFG"; then
  echo "N_EPOCH_NOT_100" >&2
  exit 1
fi
if grep -q -- '--resume' "$CFG"; then
  echo "RESUME_IN_CFG" >&2
  exit 1
fi
RUNDIR="$ROOT/logs/runs/tvr_e4_sft06_seed${SEED}_gpu${GPU}"
LOCKDIR="$ROOT/logs/runs/tvr_e4_sft06_seed${SEED}.lock"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
else
  if pgrep -u zhaopu -f "PRVR_ROOT=$PRVR_ROOT" >/dev/null; then
    echo "TVR_E4_SFT06_ALREADY_RUNNING" >&2
    exit 1
  fi
  echo "STALE_LOCK $LOCKDIR" >&2
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR"
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
fi
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/tvr_e4_sft06_seed${SEED}_gpu${GPU}.log"
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
    "lr": 0.0003,
    "sft_factor": 0.6,
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "E4 TVR I3D vis + CLIP text, sft 0.6 only, lr 3e-4, not Table 1, no resume",
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
