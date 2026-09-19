#!/bin/bash
# Patch query-Bert bypass, ntrain sequence check, launch TVR E4 bypass train.
# Prefer GPU6. Never GPU2. Do not overwrite E4 freeze tmp.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
SEED=9527
WANT_GPU="${1:-6}"
PIPELOG="$ROOT/logs/tvr_e4_qbert_bypass_pipeline.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$PIPELOG") 2>&1
echo "PIPE_START $(date -Is) want_gpu=$WANT_GPU"

pick_gpu() {
  python3 "$ROOT/experiments/e2_gpu_occupancy.py" | tee /dev/stderr | awk '
    /^CANDIDATES / {
      n=split($2, a, ",")
      for (i=1;i<=n;i++) if (a[i] != "none" && a[i] != "2") print a[i]
    }'
}

if [ "$WANT_GPU" = "2" ]; then
  echo REFUSE_GPU2
  exit 2
fi

CANDS=$(pick_gpu || true)
echo "CANDS $CANDS"
GPU=""
if echo " $CANDS " | grep -q " $WANT_GPU "; then
  GPU="$WANT_GPU"
else
  for g in $CANDS; do
    GPU="$g"
    break
  done
fi
if [ -z "$GPU" ] || [ "$GPU" = "2" ]; then
  echo "BLOCKED no free gpu (never 2)"
  python3 "$ROOT/experiments/e2_gpu_occupancy.py" || true
  exit 3
fi
echo "USE_GPU $GPU"

chmod +x "$ROOT/experiments/run_tvr_e4_qbert_bypass_gpu.sh"

echo "====PATCH===="
"$ROOT/conda/bin/python" -u "$ROOT/experiments/patch_tvr_qbert_bypass.py"

CFG="$ROOT/vendor/GMMFormer_v2/src/Configs/tvr.py"
SRC="$ROOT/experiments/tvr_e4_qbert_bypass_5090_1.py"
if [ ! -f "$CFG.bak_before_qbert_bypass" ]; then
  cp -a "$CFG" "$CFG.bak_before_qbert_bypass"
  echo BAK_OK
else
  echo BAK_KEEP
fi
cp -a "$SRC" "$CFG"
grep -n 'sft_factor\|lr\|n_epoch\|q_feat_size\|visual_feature\|hidden_size\|root\|bypass_query_encoder' "$CFG"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_seed9527"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed9527"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_sft06_seed9527"
test -d "$ROOT/tmp/tvr_i3d_cliptext_e4_eot_seed9527"
test -d "$ROOT/tmp/tvr_i3d_seed9527"
test -d "$ROOT/tmp/i3d_cliptext_e4_seed9527"
test -d "$ROOT/tmp/act_i3d_cliptext_e4_seed9527"
test -f "$ROOT/data/prvr/tvr/TextData/tvr_clip_B32_proj.h5"
grep -q 'cfg\["sft_factor"\] = 0.09' "$CFG"
grep -q 'cfg\["lr"\] = 0.0003' "$CFG"
grep -q 'cfg\["bypass_query_encoder"\] = True' "$CFG"
grep -q 'tvr_i3d_cliptext_qbert_bypass_seed' "$CFG"
! grep -q -- '--resume' "$CFG"
echo DEPLOY_OK

echo "====FAST_CHECK===="
"$ROOT/conda/bin/python" -u "$ROOT/experiments/tvr_e4_qbert_bypass_ntrain_check.py" --fast
echo "====NTRAIN===="
"$ROOT/conda/bin/python" -u "$ROOT/experiments/tvr_e4_qbert_bypass_ntrain_check.py"

CANDS2=$(pick_gpu || true)
echo "CANDS_AFTER $CANDS2"
GPU2=""
if echo " $CANDS2 " | grep -q " $GPU "; then
  GPU2="$GPU"
else
  for g in $CANDS2; do
    GPU2="$g"
    break
  done
fi
if [ -z "$GPU2" ] || [ "$GPU2" = "2" ]; then
  echo "BLOCKED no free gpu after ntrain"
  python3 "$ROOT/experiments/e2_gpu_occupancy.py" || true
  exit 3
fi
echo "LAUNCH_GPU $GPU2"

if pgrep -u zhaopu -f "main.py -d tvr" >/dev/null; then
  echo "BLOCKED zhaopu main.py already running"
  ps -u zhaopu -o pid,etime,cmd | grep -E 'main.py -d tvr' | grep -v grep || true
  exit 5
fi

RUN="$ROOT/logs/runs/tvr_e4_qbert_bypass_seed${SEED}_gpu${GPU2}"
mkdir -p "$RUN"
nohup "$ROOT/experiments/run_tvr_e4_qbert_bypass_gpu.sh" "$GPU2" "$SEED" > "$RUN/launcher.out" 2>&1 &
echo LAUNCHER_PID=$!
sleep 15
echo ====pid====
if [ ! -f "$RUN/pid" ]; then
  echo NO_PID
  echo "BLOCKED no train pid"
  tail -n 80 "$RUN/launcher.out" || true
  exit 7
fi
cat "$RUN/pid"
echo ====runjson====
cat "$RUN/run.json" || echo NO_RUNJSON
echo ====ps====
ps -u zhaopu -o pid,etime,cmd | grep -E 'main.py -d tvr|run_tvr_e4_qbert_bypass' | grep -v grep || echo NO_PS
if ! pgrep -u zhaopu -f "main.py -d tvr --gpu" >/dev/null; then
  echo "BLOCKED main.py not running"
  tail -n 80 "$RUN/launcher.out" || true
  exit 7
fi
echo ====cmd_resume====
if ps -u zhaopu -o cmd | grep 'main.py -d tvr' | grep -- '--resume' >/dev/null; then
  echo HAS_RESUME
  exit 6
fi
echo NO_RESUME_FLAG
echo ====hp====
python3 - <<'PY'
import os, yaml
p="/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_qbert_bypass_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
old="/data/zhaopu/wang-2024-gmmformer-v2/tmp/tvr_i3d_cliptext_e4_seed9527/results/tvr/gmmformer_v2/hyperparams.yaml"
d=yaml.safe_load(open(p)) if os.path.isfile(p) else {}
o=yaml.safe_load(open(old))
keys=["lr","sft_factor","n_epoch","max_es_cnt","q_feat_size","visual_feature","root","hard_negative_start_epoch","batchsize","bypass_query_encoder"]
print("NEW", {k:d.get(k) for k in keys})
print("OLD_E4", {k:o.get(k) for k in ["lr","sft_factor","root"]})
PY
echo ====protected====
ls -ld "$ROOT/tmp/tvr_i3d_cliptext_e4_seed9527" \
  "$ROOT/tmp/tvr_i3d_cliptext_e4_lr2e-4_seed9527" \
  "$ROOT/tmp/tvr_i3d_cliptext_e4_sft06_seed9527" \
  "$ROOT/tmp/tvr_i3d_cliptext_e4_eot_seed9527" \
  "$ROOT/tmp/tvr_i3d_seed9527" \
  "$ROOT/tmp/i3d_cliptext_e4_seed9527" \
  "$ROOT/tmp/act_i3d_cliptext_e4_seed9527"
echo "PIPE_LAUNCHED $(date -Is) gpu=$GPU2"
