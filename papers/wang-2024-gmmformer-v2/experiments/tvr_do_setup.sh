#!/bin/bash
# TVR Table 1 + E4 setup. Do not touch Charades tmp or cha.py.
set -euo pipefail
ROOT=/data/zhaopu/wang-2024-gmmformer-v2
SRC="$ROOT/vendor/GMMFormer_v2/src"
CFG="$SRC/Configs"
echo "START_TVR_SETUP $(date -Is)"
if [ ! -f "$CFG/tvr.py.bak_official_20260917" ]; then
  cp -a "$CFG/tvr.py" "$CFG/tvr.py.bak_official_20260917"
  echo BACKUP_TVR
else
  echo KEEP_TVR_BAK
fi
cp -a "$ROOT/experiments/tvr_5090_1.py" "$CFG/tvr.py"
chmod +x "$ROOT/experiments/run_tvr_gpu.sh" "$ROOT/experiments/run_tvr_e4_gpu.sh"
"$ROOT/conda/bin/python" "$ROOT/experiments/patch_tvr_jsonl.py"
"$ROOT/conda/bin/python" "$ROOT/experiments/patch_tvr_e4_text.py"
echo "==== tvr.py keys ===="
grep -n "visual_feature\|q_feat_size\|batchsize\|lr\|root\|data_root\|seed" "$CFG/tvr.py"
echo "==== cha.py visual (must stay) ===="
grep -n visual_feature "$CFG/cha.py" || true
echo "==== TC ===="
if grep -n "out = torch.mean(oo, dim = -1).squeeze()" "$SRC/Models/gmmformerV2/model_components.py"; then
  echo MEAN_STILL
  exit 1
fi
grep -n "out = torch.sum(oo \* weight" "$SRC/Models/gmmformerV2/model_components.py"
echo "==== charades tmp keep ===="
ls -ld "$ROOT/tmp/i3d_seed9527" "$ROOT/tmp/clip_e2_seed9527" "$ROOT/tmp/clip_roberta_e3_seed9527" "$ROOT/tmp/i3d_cliptext_e4_seed9527"
echo "==== charades i3d h5 ===="
ls -lh "$ROOT/data/prvr/charades/FeatureData/charades_i3d.h5"
echo TVR_SETUP_OK
