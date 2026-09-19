#!/bin/bash
# TVR E4 query-Bert bypass. Parent E4 lr=3e-4 sft=0.09 CLIP token seq. Never GPU2.
# Do not overwrite orig E4 / Charades E4 / Act E4 / lr2e-4 / sft06 / EOT / I3D tmp.
set -euo pipefail
GPU="${1:?gpu id}"
SEED="${2:?seed}"
ROOT="/data/zhaopu/wang-2024-gmmformer-v2"
if [ "$GPU" = "2" ]; then
  echo "NEVER_GPU2 asr" >&2
  exit 1
fi
export PRVR_SEED="$SEED"
export PRVR_ROOT="$ROOT/tmp/tvr_i3d_cliptext_qbert_bypass_seed${SEED}"
export PYTHONUNBUFFERED=1
OLD_E4="$ROOT/tmp/tvr_i3d_cliptext_e4_seed${SEED}"
OLD_LR="$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed${SEED}"
OLD_SFT="$ROOT/tmp/tvr_i3d_cliptext_e4_sft06_seed${SEED}"
OLD_EOT="$ROOT/tmp/tvr_i3d_cliptext_e4_eot_seed${SEED}"
OLD_I3D="$ROOT/tmp/tvr_i3d_seed${SEED}"
OLD_CHA="$ROOT/tmp/i3d_cliptext_e4_seed${SEED}"
OLD_ACT="$ROOT/tmp/act_i3d_cliptext_e4_seed${SEED}"
for d in "$OLD_E4" "$OLD_LR" "$OLD_SFT" "$OLD_EOT" "$OLD_I3D" "$OLD_CHA" "$OLD_ACT"; do
  if [ ! -d "$d" ]; then
    echo "MISSING_FREEZE $d" >&2
    exit 1
  fi
done
if [ "$PRVR_ROOT" = "$OLD_E4" ] || [ "$PRVR_ROOT" = "$OLD_LR" ] || [ "$PRVR_ROOT" = "$OLD_SFT" ] || [ "$PRVR_ROOT" = "$OLD_EOT" ] || [ "$PRVR_ROOT" = "$OLD_I3D" ] || [ "$PRVR_ROOT" = "$OLD_CHA" ] || [ "$PRVR_ROOT" = "$OLD_ACT" ]; then
  echo "REFUSE_OVERWRITE $PRVR_ROOT" >&2
  exit 1
fi
if [ ! -f "$ROOT/data/prvr/tvr/TextData/tvr_clip_B32_proj.h5" ]; then
  echo "MISSING_PROJ_H5" >&2
  exit 1
fi
CFG="$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"
SRC_CFG="$ROOT/experiments/tvr_e4_qbert_bypass_5090_1.py"
if [ ! -f "$SRC_CFG" ]; then
  echo "MISSING_SRC_CFG $SRC_CFG" >&2
  exit 1
fi
if [ ! -f "$CFG.bak_before_qbert_bypass" ]; then
  cp -a "$CFG" "$CFG.bak_before_qbert_bypass"
fi
cp -a "$SRC_CFG" "$CFG"
if ! grep -q 'cfg\["sft_factor"\] = 0.09' "$CFG"; then
  echo "SFT_NOT_0.09" >&2
  exit 1
fi
if grep -q 'cfg\["sft_factor"\] = 0.6' "$CFG"; then
  echo "SFT_STILL_0.6" >&2
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
if ! grep -q 'cfg\["q_feat_size"\] = 512' "$CFG"; then
  echo "Q_FEAT_NOT_512" >&2
  exit 1
fi
if ! grep -q 'cfg\["bypass_query_encoder"\] = True' "$CFG"; then
  echo "BYPASS_NOT_TRUE" >&2
  exit 1
fi
if ! grep -q 'tvr_i3d_cliptext_qbert_bypass_seed' "$CFG"; then
  echo "ROOT_NOT_QBERT_BYPASS" >&2
  exit 1
fi
if grep -q -- '--resume' "$CFG"; then
  echo "RESUME_IN_CFG" >&2
  exit 1
fi
BD="$ROOT/vendor/GMMFormer_v2/src/Datasets/builder.py"
if ! grep -q 'qbert_bypass' "$BD"; then
  echo "BUILDER_NO_QBERT" >&2
  exit 1
fi
if ! grep -q 'clip_B32_proj.h5' "$BD"; then
  echo "BUILDER_NO_PROJ_H5" >&2
  exit 1
fi
if grep -q 'clip_ViT_B_32_tvr_query_feat.hdf5' "$BD"; then
  echo "BUILDER_ZIP_EOT" >&2
  exit 1
fi
MP="$ROOT/vendor/GMMFormer_v2/src/Models/gmmformerV2/model.py"
if ! grep -q 'bypass_query_encoder' "$MP"; then
  echo "MODEL_NO_BYPASS" >&2
  exit 1
fi
if ! grep -q 'self.query_encoder = BertAttention' "$MP"; then
  echo "MODEL_QUERY_ENCODER_INIT_GONE" >&2
  exit 1
fi
MB="$ROOT/vendor/GMMFormer_v2/src/Models/builder.py"
if ! grep -q 'bypass_query_encoder' "$MB"; then
  echo "MODELS_BUILDER_NO_BYPASS" >&2
  exit 1
fi
RUNDIR="$ROOT/logs/runs/tvr_e4_qbert_bypass_seed${SEED}_gpu${GPU}"
LOCKDIR="$ROOT/logs/runs/tvr_e4_qbert_bypass_seed${SEED}.lock"
mkdir -p "$PRVR_ROOT" "$RUNDIR" "$ROOT/logs"
if mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
else
  if pgrep -u zhaopu -f "PRVR_ROOT=$PRVR_ROOT" >/dev/null; then
    echo "TVR_QBERT_BYPASS_ALREADY_RUNNING" >&2
    exit 1
  fi
  echo "STALE_LOCK $LOCKDIR" >&2
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR"
  echo "$$ $GPU $(date -Is)" > "$LOCKDIR/info"
fi
cd "$ROOT/vendor/GMMFormer_v2/src"
STDOUT="$RUNDIR/stdout.log"
LN="$ROOT/logs/tvr_e4_qbert_bypass_seed${SEED}_gpu${GPU}.log"
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
    "sft_factor": 0.09,
    "q_feat_size": 512,
    "text": "tvr_clip_B32_proj.h5",
    "bypass_query_encoder": True,
    "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "E4 TVR I3D vis + CLIP token text, bypass query BertAttention, lr 3e-4 sft 0.09, not Table 1, no resume",
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
