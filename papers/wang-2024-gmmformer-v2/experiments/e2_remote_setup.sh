#!/bin/bash
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
CFG="$ROOT/vendor/GMMFormer_v2/src/Configs"
echo "START_SETUP $(date -Is)"
if [ ! -f "$CFG/cha_i3d.py" ]; then
  cp -a "$CFG/cha.py" "$CFG/cha_i3d.py"
  echo "WROTE $CFG/cha_i3d.py"
else
  echo "KEEP $CFG/cha_i3d.py"
fi
cp -a "$CFG/cha.py" "$CFG/cha.py.i3d.bak_e2_20260916"
cp -a "$ROOT/experiments/cha_5090_1.py" "$CFG/cha.py"
cp -a "$ROOT/experiments/run_cha_e2_gpu.sh" "$ROOT/run_cha_e2_gpu.sh"
chmod +x "$ROOT/run_cha_e2_gpu.sh" "$ROOT/experiments/e2_extract_run.sh"
echo "==== cha.py visual_feature ===="
grep -n 'visual_feature\|q_feat_size\|visual_feat_dim\|batchsize\|seed\|max_ctx_l' "$CFG/cha.py"
echo "==== cha_i3d visual_feature ===="
grep -n 'visual_feature\|q_feat_size\|visual_feat_dim' "$CFG/cha_i3d.py"
echo "==== launcher ===="
ls -l "$ROOT/run_cha_e2_gpu.sh"
echo "END_SETUP $(date -Is)"
